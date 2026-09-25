"""Hybrid blocker: per country (open set), union of
  V  vault leg     tailored encoder -> centred sign codes -> Hamming shortlist -> INT8 rescore   (S1->R and R->S1)
  T  trigram leg   char_wb 3-gram TF-IDF cosine                                                (S1->R and R->S1)
  A  address key   exact normalised address (R-side bucket size capped)
  N  name key      exact core name (legal suffixes stripped) + last two address tokens (capped)
Raw legs are searched once at a generous K and cached (work/legs_{split}_{country}.npz); pruning is adaptive
(keep rank < kmin, or score >= top1 - gap, up to kmax) and applied at merge time, so it can be tuned on the holdout.

  python blocker.py search --split train --emb TAG      # run legs
  python blocker.py merge  --split train --emb TAG      # prune + union + pair features, recall report
"""
import argparse, json, os, re, sys, time, gc, numpy as np, polars as pl, torch
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, os.path.dirname(__file__))
from legs import vault_search, tfidf_search, tfidf_partitioned
from region import regions_for_split
from pair_features import string_features
from partitions import load as partition_ids
from record_features import CountryRecords

from config import WORK as W
KQ, KR = 50, 5                 # raw depth: S1 -> R and R -> S1
LEGAL = re.compile(r"\b(private|pvt|limited|ltd|llc|l l c|inc|incorporated|corp|corporation|co|company|llp|plc|"
                   r"the|and|pc|pllc|lp|sarl|sas|sa|eurl|sci|gmbh)\b")


def load(split):
    cols = ["entity_id", "country", "name", "addr", "business_name", "business_address"]
    s1 = pl.read_parquet(W + f"{split}_source1.parquet", columns=cols)
    r = pl.concat([pl.read_parquet(W + f"{split}_{s}.parquet", columns=cols)
                   for s in ("source2", "source3")])
    return s1, r


def load_emb(tag, split):
    e1 = np.load(W + f"emb_{tag}_{split}_source1.npy", mmap_mode="r")
    er = np.concatenate([np.load(W + f"emb_{tag}_{split}_{s}.npy", mmap_mode="r") for s in ("source2", "source3")])
    return e1, er


def key_pairs(k1, kr, cap):
    """All (i, j) with k1[i] == kr[j] != '' where the R bucket holds <= cap records."""
    a = pl.DataFrame({"k": k1, "i": np.arange(len(k1), dtype=np.int32)}).filter(pl.col("k") != "")
    b = pl.DataFrame({"k": kr, "j": np.arange(len(kr), dtype=np.int32)}).filter(pl.col("k") != "")
    b = b.filter(pl.len().over("k") <= cap)
    p = a.join(b, on="k")
    return p["i"].to_numpy(), p["j"].to_numpy()


def name_key(names, addrs):
    core = pl.Series(names).str.replace_all(LEGAL.pattern, " ").str.replace_all(r"[^a-z0-9 ]", "").str.replace_all(r"\s+", " ").str.strip_chars()
    loc = pl.Series(addrs).str.split(" ").list.slice(-2, 2).list.join(" ")
    return pl.select(pl.when(core.str.len_chars() >= 3).then(core + "|" + loc).otherwise(pl.lit(""))).to_series().to_list()


def addr_key(addrs):
    k = pl.Series(addrs).str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    return pl.select(pl.when(k.str.len_chars() >= 12).then(k).otherwise(pl.lit(""))).to_series().to_list()


def build_tfidf(tq, tr, max_df=1.0, fit_n=1_500_000, chunk=400_000):
    """char_wb 3-gram TF-IDF. Vocabulary + idf fitted on a fixed-seed sample of the R side (RAM-bounded at
    full scale); both sides transformed in chunks. Deterministic, so search and merge see identical vectors."""
    import scipy.sparse as sp
    idx = np.sort(np.random.default_rng(0).choice(len(tr), fit_n, replace=False)) if len(tr) > fit_n else np.arange(len(tr))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, max_df=max_df, dtype=np.float32)
    vec.fit([tr[i] for i in idx])
    tf = lambda xs: sp.vstack([vec.transform(xs[s:s + chunk]) for s in range(0, len(xs), chunk)]).tocsr()
    return tf(tq), tf(tr), vec


def search(a):
    s1, r = load(a.split); e1, er = load_emb(a.emb, a.split)
    g1, gr = regions_for_split(a.split, s1, r) if a.partition else (None, None)
    for c in sorted(set(s1["country"].unique().to_list()) | set(r["country"].unique().to_list())):
        out = W + f"legs_{a.emb}_{a.split}_{c}.npz"
        if os.path.exists(out) and not a.force: print("skip", c); continue
        qi = np.where((s1["country"] == c).to_numpy())[0]; ri = np.where((r["country"] == c).to_numpy())[0]
        print(f"== {c}: S1 {len(qi)}  R {len(ri)}", flush=True)
        if len(qi) == 0 or len(ri) == 0: np.savez(out, qi=qi, ri=ri); continue
        res = {"qi": qi, "ri": ri}; t = time.time()
        Q = np.asarray(e1[qi]); R = np.asarray(er[ri])
        mu = (Q.astype(np.float32).sum(0) + R.astype(np.float32).sum(0)) / (len(Q) + len(R))
        res["vq_i"], res["vq_s"] = vault_search(Q, R, min(KQ, len(R)), mu=mu); print(" vault S1->R", f"{time.time()-t:.0f}s", flush=True)
        res["vr_i"], res["vr_s"] = vault_search(R, Q, min(KR, len(Q)), mu=mu); print(" vault R->S1", f"{time.time()-t:.0f}s", flush=True)
        del Q, R; gc.collect(); torch.cuda.empty_cache()
        tq = (s1["name"][qi] + ", " + s1["addr"][qi]).to_list(); tr = (r["name"][ri] + ", " + r["addr"][ri]).to_list()
        Tq, Tr, vec = build_tfidf(tq, tr, a.max_df); print(f" tfidf fit V={len(vec.vocabulary_)} nnz={Tr.nnz}", f"{time.time()-t:.0f}s", flush=True)
        if g1 is not None:                         # region-partitioned trigram leg (unknown regions -> whole country)
            gq = [g1[i] for i in qi]; gk = [gr[j] for j in ri]
            res["tq_i"], res["tq_s"] = tfidf_partitioned(Tq, Tr, gq, gk, min(KQ, len(ri))); print(" tfidf S1->R (partitioned)", f"{time.time()-t:.0f}s", flush=True)
            res["tr_i"], res["tr_s"] = tfidf_partitioned(Tr, Tq, gk, gq, min(KR, len(qi))); print(" tfidf R->S1 (partitioned)", f"{time.time()-t:.0f}s", flush=True)
        else:
            res["tq_i"], res["tq_s"] = tfidf_search(Tq, Tr, min(KQ, len(ri))); print(" tfidf S1->R", f"{time.time()-t:.0f}s", flush=True)
            res["tr_i"], res["tr_s"] = tfidf_search(Tr, Tq, min(KR, len(qi))); print(" tfidf R->S1", f"{time.time()-t:.0f}s", flush=True)
        del Tr, Tq, vec; gc.collect()
        res["a_i"], res["a_j"] = key_pairs(addr_key(s1["addr"][qi].to_list()), addr_key(r["addr"][ri].to_list()), 30)
        res["n_i"], res["n_j"] = key_pairs(name_key(s1["name"][qi].to_list(), s1["addr"][qi].to_list()),
                                           name_key(r["name"][ri].to_list(), r["addr"][ri].to_list()), 30)
        print(f" keys addr {len(res['a_i'])} name {len(res['n_i'])}", f"{time.time()-t:.0f}s", flush=True)
        np.savez(out, **res)


# ---------------------------------------------------------------- merge
def prune(idx, sc, kmin, kmax, gap):
    """Row-wise adaptive K. Returns a boolean keep mask over (n, K)."""
    k = idx.shape[1]; r = np.arange(k)[None, :]
    keep = (idx >= 0) & (r < kmax) & ((r < kmin) | (sc >= sc[:, :1] - gap))
    return keep


def leg_pairs(L, p):
    """-> dict leg -> (i local S1 idx, j local R idx) after pruning with params p."""
    nq = len(L["qi"]); out = {}
    for leg in ("v", "t"):
        if f"{leg}q_i" not in L: continue
        m = prune(L[f"{leg}q_i"], L[f"{leg}q_s"], p[f"{leg}q_kmin"], p[f"{leg}q_kmax"], p[f"{leg}q_gap"])
        qq = np.broadcast_to(np.arange(nq)[:, None], m.shape)
        i1, j1 = qq[m], L[f"{leg}q_i"][m]
        m = prune(L[f"{leg}r_i"], L[f"{leg}r_s"], p[f"{leg}r_kmin"], p[f"{leg}r_kmax"], p[f"{leg}r_gap"])
        rr = np.broadcast_to(np.arange(len(L["ri"]))[:, None], m.shape)
        out[leg + "q"] = (i1.astype(np.int64), j1.astype(np.int64))
        out[leg + "r"] = (L[f"{leg}r_i"][m].astype(np.int64), rr[m].astype(np.int64))
    if "a_i" in L: out["a"] = (L["a_i"].astype(np.int64), L["a_j"].astype(np.int64))
    if "n_i" in L: out["n"] = (L["n_i"].astype(np.int64), L["n_j"].astype(np.int64))
    return out


DEFAULT = dict(vq_kmin=5, vq_kmax=30, vq_gap=0.10, vr_kmin=2, vr_kmax=5, vr_gap=0.05,
               tq_kmin=5, tq_kmax=30, tq_gap=0.15, tr_kmin=2, tr_kmax=5, tr_gap=0.10)


def truth_local(split, s1c, rc):
    """True pairs as local (i, j) for this country, + holdout flag on i. Every split except test is train-derived."""
    if split == "test": return None
    pairs = pl.read_parquet(W + "train_pairs.parquet")
    hq = set(pl.read_parquet(W + "q.parquet")["entity_id"].to_list())
    qpos = pl.DataFrame({"s1": s1c, "i": np.arange(len(s1c))}); rpos = pl.DataFrame({"m": rc, "j": np.arange(len(rc))})
    t = pairs.join(qpos, on="s1").join(rpos, on="m")
    return t["i"].to_numpy(), t["j"].to_numpy(), t["s1"].is_in(list(hq)).to_numpy()


def evaluate_country(L, truth, p, nq):
    legs = leg_pairs(L, p); nr = len(L["ri"])
    enc = lambda i, j: i * nr + j
    allk = np.unique(np.concatenate([enc(*v) for v in legs.values()]))
    ti, tj, th = truth; tk = enc(ti, tj)
    rep = {"pairs": int(len(allk)), "per_s1": round(len(allk) / nq, 2)}
    for name, mask in (("holdout", th), ("all", np.ones_like(th))):
        k = tk[mask]
        rep[f"recall_{name}"] = round(float(np.isin(k, allk).mean()), 5)
        for leg, v in legs.items(): rep[f"{name}_{leg}"] = round(float(np.isin(k, np.unique(enc(*v))).mean()), 4)
    return rep


CHUNK = 2_000_000


def pair_features(i, j, L, e1c, erc, tq, tr, leg_keys, nr):
    """Per-pair leg evidence: trigram cosine, leg membership, rank in every forward / reverse list (99 = absent)."""
    f = {"cos_t": np.asarray(tq[i].multiply(tr[j]).sum(1)).ravel().astype(np.float32)}
    key = i * nr + j
    for leg, lk in leg_keys.items():
        f["in_" + leg] = np.isin(key, lk)
    for leg in ("v", "t"):
        for d, (rows, cols, K) in {"q": (i, j, L[f"{leg}q_i"]), "r": (j, i, L[f"{leg}r_i"])}.items():
            hit = K[rows] == cols[:, None]
            f[f"rank_{leg}{d}"] = np.where(hit.any(1), hit.argmax(1), 99).astype(np.int8)
    return f


def _cos_chunked(i, j, E1, ER, ch=4_000_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), ch):
        out[s:s + ch] = np.einsum("ij,ij->i", E1[i[s:s + ch]].astype(np.float32), ER[j[s:s + ch]].astype(np.float32))
    return out


def _rank_within(group, score):
    """Ordinal rank (1 = best) of score within each group value."""
    o = np.lexsort((-score, group)); g = group[o]
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    run = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    rk = np.empty(len(g), np.int32); rk[o] = run + 1
    return rk


def merge(a):
    p = dict(DEFAULT); p.update(json.loads(a.params) if a.params else {})
    s1, r = load(a.split); e1, er = load_emb(a.emb, a.split); reps = {}; wrote = False; nrows = {}; missed = []
    parts = None if a.split == "test" else partition_ids().rename({"part": "s1_part"})
    g1, gr = (None, None) if (a.report_only or a.tune) else regions_for_split(a.split, s1, r)
    for f in sorted(os.listdir(W)):
        if not (f.startswith(f"legs_{a.emb}_{a.split}_") and f.endswith(".npz")): continue
        c = f[len(f"legs_{a.emb}_{a.split}_"):-4]; L = dict(np.load(W + f)); qi, ri = L["qi"], L["ri"]
        s1c = s1["entity_id"].to_numpy()[qi]; rc = r["entity_id"].to_numpy()[ri]
        if len(qi) == 0 or len(ri) == 0: continue
        truth = truth_local(a.split, s1c, rc)
        if truth is not None and a.tune:
            reps[c] = tune(L, truth, len(qi), p); print(c, json.dumps(reps[c]), flush=True); continue
        if truth is not None: reps[c] = evaluate_country(L, truth, p, len(qi)); print(c, json.dumps(reps[c]), flush=True)
        if a.report_only: continue
        t = time.time(); nr = len(ri)
        legs = leg_pairs(L, p); leg_keys = {leg: np.unique(i * nr + j) for leg, (i, j) in legs.items()}; del legs
        key = np.unique(np.concatenate(list(leg_keys.values()))); I, J = key // nr, key % nr
        tkeys = np.unique(truth[0] * nr + truth[1]) if truth is not None else None
        E1 = np.asarray(e1[qi]); ER = np.asarray(er[ri])
        cos_v = _cos_chunked(I, J, E1, ER)                       # global competition context (spans chunks)
        n_s1 = np.bincount(I, minlength=len(qi))[I].astype(np.int32); n_r = np.bincount(J, minlength=nr)[J].astype(np.int32)
        rk_s1 = _rank_within(I, cos_v); rk_r = _rank_within(J, cos_v)
        tq_ = (s1["name"][qi] + ", " + s1["addr"][qi]).to_list(); tr_ = (r["name"][ri] + ", " + r["addr"][ri]).to_list()
        Tq, Tr, _ = build_tfidf(tq_, tr_, a.max_df); del tq_, tr_
        rec = CountryRecords(s1, r, qi, ri)                     # IDF / frequency caches for this country
        print(f"{c}: {len(key)} pairs, global features {time.time()-t:.0f}s", flush=True)
        if tkeys is not None:                                    # true pairs the blocker lost, with how close each leg came
            mk = tkeys[~np.isin(tkeys, key)]; mi, mj = mk // nr, mk % nr
            if len(mk):
                mf = {"cos_v": _cos_chunked(mi, mj, E1, ER), **pair_features(mi, mj, L, E1, ER, Tq, Tr, leg_keys, nr)}
                mf = {k: v for k, v in mf.items() if not k.startswith("in_")}
                gi, gj = qi[mi], ri[mj]
                mf.update(string_features(s1["name"][gi], s1["addr"][gi], r["name"][gj], r["addr"][gj],
                                          [g1[x] for x in gi], [gr[x] for x in gj]))
                mf.update(rec.pair(mi, mj))
                missed.append(pl.DataFrame({"source1_entity_id": s1c[mi], "candidate_entity_id": rc[mj], "country": c,
                                            "n_cand_s1": np.bincount(I, minlength=len(qi))[mi].astype(np.int32), **mf}))
        for old in [x for x in os.listdir(W) if x.startswith(f"cand_{a.emb}_{a.split}_{c}_")]: os.remove(W + old)
        bounds = np.r_[np.searchsorted(I, np.arange(0, len(qi), max(1, int(CHUNK * len(qi) / max(len(key), 1))))), len(key)]
        bounds = np.unique(bounds)
        for part, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):     # chunks end on S1 boundaries
            i, j = I[s:e], J[s:e]
            feats = {"cos_v": cos_v[s:e], **pair_features(i, j, L, E1, ER, Tq, Tr, leg_keys, nr),
                     "n_cand_s1": n_s1[s:e], "n_cand_r": n_r[s:e], "cos_v_rank_in_s1": rk_s1[s:e], "cos_v_rank_in_r": rk_r[s:e]}
            gi, gj = qi[i], ri[j]
            feats.update(string_features(s1["name"][gi], s1["addr"][gi], r["name"][gj], r["addr"][gj],
                                         [g1[x] for x in gi], [gr[x] for x in gj]))
            feats.update(rec.pair(i, j))
            df = pl.DataFrame({"source1_entity_id": s1c[i], "candidate_entity_id": rc[j], "country": c, **feats})
            df = df.with_columns(pl.col("candidate_entity_id").str.slice(0, 2).alias("cand_source"),
                                 pl.Series("s1_idx", gi.astype(np.int32)), pl.Series("cand_idx", gj.astype(np.int32)))
            if tkeys is not None:
                df = df.with_columns(pl.Series("label", np.isin(key[s:e], tkeys)))
                df = df.join(parts, left_on="source1_entity_id", right_on="entity_id", how="left", maintain_order="left") \
                       .with_columns(pl.col("s1_part").fill_null("-"))
            nrows[c] = nrows.get(c, 0) + df.height
            df.write_parquet(W + f"cand_{a.emb}_{a.split}_{c}_{part:03d}.parquet")
        wrote = True
        print(c, "pairs", len(key), "parts", len(bounds) - 1, f"{time.time()-t:.0f}s", flush=True)
        del E1, ER, Tq, Tr, L, rec; gc.collect()
    json.dump(reps, open(W + f"block_report_{a.emb}_{a.split}.json", "w"), indent=1)
    if missed:
        pl.concat(missed).join(partition_ids().rename({"entity_id": "source1_entity_id", "part": "s1_part"}),
                               on="source1_entity_id", how="left").write_parquet(W + f"missed_{a.emb}_{a.split}.parquet")
    if wrote:
        write_tsv(a, s1); write_manifest(a, p, reps, nrows, s1)


def write_manifest(a, p, reps, nrows, s1):
    """What produced this candidate set, so the selector can refuse a stale or mismatched one."""
    from region import ALIAS_PARTS
    import hashlib, datetime
    embf = sorted(f for f in os.listdir(W) if f.startswith(f"emb_{a.emb}_{a.split}_"))
    m = {"schema_version": 3, "split": a.split, "emb": a.emb, "created": datetime.datetime.now().isoformat(timespec="seconds"),
         "prune_params": p, "raw_depth": {"KQ": KQ, "KR": KR}, "tfidf_max_df": a.max_df, "region_alias_parts": ALIAS_PARTS,
         "partitioned_trigram": a.partition, "rows_by_country": nrows, "n_s1": s1.height,
         "s1_order_sha1": hashlib.sha1("\n".join(s1["entity_id"].to_list()).encode()).hexdigest(),
         "embedding_files": {f: [os.path.getsize(W + f), int(os.path.getmtime(W + f))] for f in embf},
         "recall_report": reps}
    json.dump(m, open(W + f"cand_{a.emb}_{a.split}_manifest.json", "w"), indent=1)


def write_tsv(a, s1):
    files = sorted(W + f for f in os.listdir(W) if f.startswith(f"cand_{a.emb}_{a.split}_") and f.endswith(".parquet"))
    cand = pl.concat([pl.read_parquet(f, columns=["source1_entity_id", "candidate_entity_id"]) for f in files])
    g = cand.group_by("source1_entity_id").agg(pl.col("candidate_entity_id").str.join(",").alias("candidate_entity_ids"))
    out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(g, on="source1_entity_id", how="left").fill_null("")
    out.write_csv(W + f"candidate_pairs_{a.split}.tsv", separator="	", quote_style="never")
    print("wrote", W + f"candidate_pairs_{a.split}.tsv", out.height, "S1 rows,", cand.height, "pairs")


def tune(L, truth, nq, p0):
    """Coordinate search over pruning params on the holdout: best recall at <= budget candidates per S1."""
    ti, tj, th = truth; hold = (ti[th], tj[th], np.ones(th.sum(), bool))
    hq = np.unique(ti[th]); sub = dict(L)
    # restrict S1->R lists to holdout S1 rows (candidate count measured on them); R->S1 lists stay full
    grid = {"vq_gap": [0.02, 0.05, 0.1, 0.2], "vq_kmax": [10, 20, 30, 50], "vq_kmin": [3, 5, 10],
            "tq_gap": [0.05, 0.1, 0.2, 0.3], "tq_kmax": [10, 20, 30, 50], "tq_kmin": [3, 5, 10],
            "vr_kmax": [1, 2, 3, 5], "tr_kmax": [1, 2, 3, 5]}
    res = []
    for leg, vals in grid.items():
        for v in vals:
            p = dict(p0); p[leg] = v
            if leg.endswith("kmax"): p[leg.replace("kmax", "kmin")] = min(p[leg.replace("kmax", "kmin")], v)
            r = evaluate_holdout(sub, hold, hq, p); res.append({"param": leg, "value": v, **r})
    base = evaluate_holdout(sub, hold, hq, p0)
    return {"base": base, "sweep": res}


def evaluate_holdout(L, hold, hq, p):
    legs = leg_pairs(L, p); nr = len(L["ri"])
    k = np.unique(np.concatenate([i * nr + j for i, j in legs.values()]))
    k = k[np.isin(k // nr, hq)]
    tk = hold[0] * nr + hold[1]
    return {"recall": round(float(np.isin(tk, k).mean()), 5), "per_s1": round(len(k) / len(hq), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "merge"]); ap.add_argument("--split", default="train")
    ap.add_argument("--emb", required=True); ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-df", dest="max_df", type=float, default=1.0)
    ap.add_argument("--no-partition", dest="partition", action="store_false")
    ap.add_argument("--params", default=None); ap.add_argument("--tune", action="store_true")
    ap.add_argument("--report-only", dest="report_only", action="store_true")
    a = ap.parse_args()
    search(a) if a.cmd == "search" else merge(a)
