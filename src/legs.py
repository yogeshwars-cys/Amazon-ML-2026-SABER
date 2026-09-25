"""Candidate-generation legs (GPU). Every leg returns (idx int32 (nq, k), score float32 (nq, k)), rows sorted by score.

vault_search  - BQBOOST path emulated on the GPU, bit-exact in ranking semantics with bqboost:
                centred sign codes (bit = x - mu > 0), Hamming shortlist (= 192 - 2*hamming via +-1 matmul),
                then symmetric per-row INT8 rescoring (scale = max|x|/127, round half to even, int32 dot * scales).
dense_search  - exact fp16 cosine (the ceiling of the vault path).
tfidf_search  - char-trigram TF-IDF cosine via GPU sparse x dense products, keys chunked to fit 6 GB."""
import numpy as np, torch

DEV = "cuda"


def _merge_topk(vals, idxs, k):
    v = torch.cat(vals, 1); i = torch.cat(idxs, 1)
    k = min(k, v.shape[1]); t = v.topk(k, dim=1)
    return t.values, i.gather(1, t.indices)


def quantize_i8(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    s = x.abs().amax(1).clamp_min(1e-12) / 127.0
    return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8), s


def vault_search(Q, K, k, shortlist=None, mu=None, qblock=256, kchunk=1_500_000):
    """Q (nq,d), K (nk,d) float16/32 numpy, L2-normalised. Returns top-k by INT8 rescored dot."""
    shortlist = shortlist or 4 * k
    mu = np.zeros(Q.shape[1], np.float32) if mu is None else mu.astype(np.float32)
    mu_t = torch.from_numpy(mu).to(DEV)
    Ks = []                                             # +-1 sign codes of keys, fp16 on GPU
    for s in range(0, len(K), kchunk):
        x = torch.from_numpy(np.asarray(K[s:s + kchunk], np.float32)).to(DEV) - mu_t
        Ks.append(torch.where(x > 0, 1.0, -1.0).half()); del x
    Kc = torch.empty((len(K), K.shape[1]), dtype=torch.int8); Kscale = torch.empty(len(K))
    for s in range(0, len(K), 1_000_000):                  # INT8 rows on CPU (gathered per block), chunked for RAM
        Kc[s:s + 1_000_000], Kscale[s:s + 1_000_000] = quantize_i8(np.asarray(K[s:s + 1_000_000], np.float32))
    Kc = Kc.pin_memory(); Kscale = Kscale.to(DEV)
    nq = len(Q); out_i = np.full((nq, k), -1, np.int32); out_s = np.full((nq, k), -1.0, np.float32)
    for s in range(0, nq, qblock):
        q = torch.from_numpy(np.asarray(Q[s:s + qblock], np.float32)).to(DEV)
        qs = torch.where(q - mu_t > 0, 1.0, -1.0).half()
        vals, idxs, off = [], [], 0
        for kc in Ks:
            sc = qs @ kc.T; t = sc.topk(min(shortlist, kc.shape[0]), dim=1)
            vals.append(t.values); idxs.append(t.indices + off); off += kc.shape[0]
        _, cand = _merge_topk(vals, idxs, shortlist)                        # (b, S) Hamming shortlist
        cand_cpu = cand.cpu()
        rows = Kc.index_select(0, cand_cpu.reshape(-1)).to(DEV, non_blocking=True).view(*cand.shape, -1).float()
        qc, qsc = quantize_i8(q.cpu()); qc = qc.to(DEV).float()
        dot = torch.bmm(rows, qc[:, :, None]).squeeze(2)                   # exact int32 range in fp32
        sc = dot * Kscale[cand] * qsc.to(DEV)[:, None]
        t = sc.topk(min(k, sc.shape[1]), dim=1)
        kk = t.values.shape[1]
        out_s[s:s + len(q), :kk] = t.values.cpu().numpy(); out_i[s:s + len(q), :kk] = cand.gather(1, t.indices).cpu().numpy()
    return out_i, out_s


def dense_search(Q, K, k, qblock=512, kchunk=2_000_000):
    Kt = [torch.from_numpy(np.asarray(K[s:s + kchunk], np.float16)).to(DEV) for s in range(0, len(K), kchunk)]
    nq = len(Q); out_i = np.empty((nq, k), np.int32); out_s = np.empty((nq, k), np.float32)
    for s in range(0, nq, qblock):
        q = torch.from_numpy(np.asarray(Q[s:s + qblock], np.float16)).to(DEV)
        vals, idxs, off = [], [], 0
        for kc in Kt:
            t = (q @ kc.T).topk(min(k, kc.shape[0]), dim=1); vals.append(t.values.float()); idxs.append(t.indices + off); off += kc.shape[0]
        v, i = _merge_topk(vals, idxs, k)
        out_s[s:s + len(q)] = v.cpu().numpy(); out_i[s:s + len(q)] = i.cpu().numpy()
    return out_i, out_s


def _csr_gpu(M):
    M = M.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int32)), torch.from_numpy(M.indices.astype(np.int32)),
                                   torch.from_numpy(M.data.astype(np.float32)), size=M.shape, device=DEV)


def tfidf_search(Qs, Ks, k, qblock=512, nnz_chunk=40_000_000, col_k=0):
    """Qs (nq,V), Ks (nk,V) scipy CSR, rows L2-normalised. Keys chunked by nnz so a chunk + a score block fit.
    col_k > 0 also returns, from the same score blocks, each key's top-col_k queries (the reverse direction),
    so a bidirectional leg costs one sparse product instead of two."""
    Ks = Ks.tocsr(); Qs = Qs.tocsr()
    bounds, s = [], 0
    while s < Ks.shape[0]:                               # row ranges holding <= nnz_chunk non-zeros and <= 600k rows
        e = int(np.searchsorted(Ks.indptr, Ks.indptr[s] + nnz_chunk, side="right")) - 1
        e = max(s + 1, min(e, s + 600_000, Ks.shape[0])); bounds.append((s, e)); s = e
    nq, nk = Qs.shape[0], Ks.shape[0]
    best_v = torch.full((nq, k), -1.0); best_i = torch.full((nq, k), -1, dtype=torch.int32)
    ck = min(col_k, nq)
    col_v = torch.full((nk, ck), -1.0); col_i = torch.full((nk, ck), -1, dtype=torch.int32)
    for (a, b) in bounds:                                # outer loop over key chunks: each chunk uploaded once
        Kg = _csr_gpu(Ks[a:b])
        if ck: cv = col_v[a:b].to(DEV); ci = col_i[a:b].to(DEV).long()
        for s in range(0, nq, qblock):
            qdT = _csr_gpu(Qs[s:s + qblock]).to_dense().T.contiguous()          # (V, b) contiguous, densified on GPU:
            sc = torch.sparse.mm(Kg, qdT).T.contiguous()                      # a transposed view is 35x slower
            t = sc.topk(min(k, sc.shape[1]), dim=1)
            v = torch.cat([best_v[s:s + qblock].to(DEV), t.values], 1)
            i = torch.cat([best_i[s:s + qblock].to(DEV).long(), t.indices + a], 1)
            tt = v.topk(k, dim=1)
            best_v[s:s + qblock] = tt.values.cpu(); best_i[s:s + qblock] = i.gather(1, tt.indices).int().cpu()
            if ck:                                                            # reverse: best queries per key
                tc = sc.topk(min(ck, sc.shape[0]), dim=0)
                v2 = torch.cat([cv, tc.values.T], 1); i2 = torch.cat([ci, tc.indices.T + s], 1)
                t2 = v2.topk(ck, dim=1); cv = t2.values; ci = i2.gather(1, t2.indices)
        if ck: col_v[a:b] = cv.cpu(); col_i[a:b] = ci.int().cpu()
        del Kg; torch.cuda.empty_cache()
    if ck: return best_i.numpy(), best_v.numpy(), col_i.numpy(), col_v.numpy()
    return best_i.numpy(), best_v.numpy()


def _merge_rows(best_i, best_s, rows, new_i, new_s):
    """Merge new (len(rows), k') candidates into running per-row top-k, dropping duplicate keys."""
    bi, bs = best_i[rows], best_s[rows]
    dup = (new_i[:, :, None] == bi[:, None, :]).any(2) | (new_i < 0)
    ns = np.where(dup, -np.inf, new_s)
    ci = np.concatenate([bi, new_i], 1); cs = np.concatenate([bs, ns], 1)
    o = np.argsort(-cs, axis=1, kind="stable")[:, :best_i.shape[1]]
    best_i[rows] = np.take_along_axis(ci, o, 1); best_s[rows] = np.take_along_axis(cs, o, 1)


def tfidf_partitioned(Qs, Ks, gq, gk, k, min_block=1):
    """Trigram search restricted to shared regions. gq/gk: list of frozensets (empty = unknown -> search everything).
    Query q is compared with key j if gq[q] & gk[j], or either set is empty."""
    from collections import defaultdict
    nq = Qs.shape[0]; best_i = np.full((nq, k), -1, np.int32); best_s = np.full((nq, k), -np.inf, np.float32)
    qg, kg = defaultdict(list), defaultdict(list)
    for i, g in enumerate(gq):
        for x in g: qg[x].append(i)
    for j, g in enumerate(gk):
        for x in g: kg[x].append(j)
    k_unk = np.array([j for j, g in enumerate(gk) if not g], np.int64)
    q_unk = np.array([i for i, g in enumerate(gq) if not g], np.int64)
    jobs = [(np.array(qg[x], np.int64), np.union1d(np.array(kg.get(x, []), np.int64), k_unk)) for x in qg]
    if len(q_unk): jobs.append((q_unk, np.arange(Ks.shape[0])))
    for qi, kj in sorted(jobs, key=lambda t: -len(t[0]) * len(t[1])):
        if len(kj) == 0: continue
        ii, ss = tfidf_search(Qs[qi], Ks[kj], min(k, len(kj)))
        ii = np.where(ii >= 0, kj[np.clip(ii, 0, None)], -1).astype(np.int32)
        if ii.shape[1] < k:
            ii = np.pad(ii, ((0, 0), (0, k - ii.shape[1])), constant_values=-1); ss = np.pad(ss, ((0, 0), (0, k - ss.shape[1])), constant_values=-np.inf)
        _merge_rows(best_i, best_s, qi, ii, ss)
    best_s[~np.isfinite(best_s)] = -1.0
    return best_i, best_s


def tfidf_partitioned_both(Qs, Ks, gq, gk, kq, kr):
    """Both trigram directions in one pass over the region-partitioned pairs. The pair set is symmetric
    (q vs j iff gq[q] & gk[j], or either is empty), so each key's top-kr queries come from the same score blocks.
    Returns (q->k idx, score, k->q idx, score), matching tfidf_partitioned(Qs, Ks, ...) and (Ks, Qs, ...)."""
    from collections import defaultdict
    nq, nk = Qs.shape[0], Ks.shape[0]
    bq_i = np.full((nq, kq), -1, np.int32); bq_s = np.full((nq, kq), -np.inf, np.float32)
    bk_i = np.full((nk, kr), -1, np.int32); bk_s = np.full((nk, kr), -np.inf, np.float32)
    qg, kg = defaultdict(list), defaultdict(list)
    for i, g in enumerate(gq):
        for x in g: qg[x].append(i)
    for j, g in enumerate(gk):
        for x in g: kg[x].append(j)
    k_unk = np.array([j for j, g in enumerate(gk) if not g], np.int64)
    q_unk = np.array([i for i, g in enumerate(gq) if not g], np.int64)
    jobs = [(np.array(qg[x], np.int64), np.union1d(np.array(kg.get(x, []), np.int64), k_unk)) for x in qg]
    if len(q_unk): jobs.append((q_unk, np.arange(nk)))
    for qi, kj in sorted(jobs, key=lambda t: -len(t[0]) * len(t[1])):
        if len(kj) == 0: continue
        ii, ss, ci, cs = tfidf_search(Qs[qi], Ks[kj], min(kq, len(kj)), col_k=kr)
        ii = np.where(ii >= 0, kj[np.clip(ii, 0, None)], -1).astype(np.int32)
        ci = np.where(ci >= 0, qi[np.clip(ci, 0, None)], -1).astype(np.int32)
        pad = lambda a, b, w: (np.pad(a, ((0, 0), (0, w - a.shape[1])), constant_values=-1),
                               np.pad(b, ((0, 0), (0, w - b.shape[1])), constant_values=-np.inf)) if a.shape[1] < w else (a, b)
        ii, ss = pad(ii, ss, kq); ci, cs = pad(ci, cs, kr)
        cs = np.where(ci >= 0, cs, -np.inf).astype(np.float32)
        _merge_rows(bq_i, bq_s, qi, ii, ss); _merge_rows(bk_i, bk_s, kj, ci, cs)
    bq_s[~np.isfinite(bq_s)] = -1.0; bk_s[~np.isfinite(bk_s)] = -1.0
    return bq_i, bq_s, bk_i, bk_s
