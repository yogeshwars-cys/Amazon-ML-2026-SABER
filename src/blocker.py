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

from config import WORK as W
KQ, KR = 50, 5                 # raw depth: S1 -> R and R -> S1
LEGAL = re.compile(r"\b(private|pvt|limited|ltd|llc|l l c|inc|incorporated|corp|corporation|co|company|llp|plc|"
                   r"the|and|pc|pllc|lp|sarl|sas|sa|eurl|sci|gmbh)\b")


def load(split):
    cols = ["entity_id", "country", "name", "addr", "business_address"]
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
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, max_df=a.max_df, dtype=np.float32)
        Tr = vec.fit_transform(tr); Tq = vec.transform(tq); print(f" tfidf fit V={len(vec.vocabulary_)} nnz={Tr.nnz}", f"{time.time()-t:.0f}s", flush=True)
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
    """True pairs as local (i, j) for this country, + holdout flag on i."""
    if split not in ("train", "mini"): return None
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


def pair_features(i, j, L, e1c, erc, tq, tr, legs, nr):
    """Per-pair features for the matcher: exact cosine of both legs + leg membership / reciprocal ranks."""
    f = {}
    f["cos_v"] = np.einsum("ij,ij->i", e1c[i].astype(np.float32), erc[j].astype(np.float32))
    f["cos_t"] = np.asarray(tq[i].multiply(tr[j]).sum(1)).ravel().astype(np.float32)
    key = i * nr + j
    for leg, (a, b) in legs.items():
        f["in_" + leg] = np.isin(key, a * nr + b)
    for leg in ("v", "t"):                       # rank of j in i's S1->R list and of i in j's R->S1 list (99 = absent)
        for d, (rows, cols, K) in {"q": (i, j, L[f"{leg}q_i"]), "r": (j, i, L[f"{leg}r_i"])}.items():
            hit = K[rows] == cols[:, None]
            f[f"rank_{leg}{d}"] = np.where(hit.any(1), hit.argmax(1), 99).astype(np.int8)
    return f


def merge(a):
    p = dict(DEFAULT); p.update(json.loads(a.params) if a.params else {})
    s1, r = load(a.split); e1, er = load_emb(a.emb, a.split); reps = {}; parts = []
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
        legs = leg_pairs(L, p); nr = len(ri)
        key = np.unique(np.concatenate([i * nr + j for i, j in legs.values()]))
        i, j = key // nr, key % nr
        tq_ = (s1["name"][qi] + ", " + s1["addr"][qi]).to_list(); tr_ = (r["name"][ri] + ", " + r["addr"][ri]).to_list()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2, dtype=np.float32)
        Tr = vec.fit_transform(tr_).tocsr(); Tq = vec.transform(tq_).tocsr()
        feats = pair_features(i, j, L, np.asarray(e1[qi]), np.asarray(er[ri]), Tq, Tr, legs, nr)
        gi, gj = qi[i], ri[j]
        feats.update(string_features(s1["name"][gi], s1["addr"][gi], r["name"][gj], r["addr"][gj],
                                     [g1[x] for x in gi], [gr[x] for x in gj]))
        df = pl.DataFrame({"source1_entity_id": s1c[i], "candidate_entity_id": rc[j], "country": c, **feats})
        df = df.with_columns(pl.col("candidate_entity_id").str.slice(0, 2).alias("cand_source"),
                             pl.len().over("source1_entity_id").alias("n_cand_s1"),
                             pl.len().over("candidate_entity_id").alias("n_cand_r"),
                             pl.col("cos_v").rank("ordinal", descending=True).over("source1_entity_id").alias("cos_v_rank_in_s1"),
                             pl.col("cos_v").rank("ordinal", descending=True).over("candidate_entity_id").alias("cos_v_rank_in_r"))
        if truth is not None:
            df = df.with_columns(pl.Series("label", np.isin(key, truth[0] * nr + truth[1])))
        df.write_parquet(W + f"cand_{a.emb}_{a.split}_{c}.parquet"); parts.append(c)
        print(c, "pairs", df.height, flush=True)
    json.dump(reps, open(W + f"block_report_{a.emb}_{a.split}.json", "w"), indent=1)
    if parts and not a.report_only and not a.tune: write_tsv(a, s1)


def write_tsv(a, s1):
    cand = pl.concat([pl.read_parquet(W + f"cand_{a.emb}_{a.split}_{c}.parquet", columns=["source1_entity_id", "candidate_entity_id"])
                      for c in sorted({f[len(f"cand_{a.emb}_{a.split}_"):-8] for f in os.listdir(W)
                                       if f.startswith(f"cand_{a.emb}_{a.split}_") and f.endswith(".parquet")})])
    g = cand.group_by("source1_entity_id").agg(pl.col("candidate_entity_id").str.join(",").alias("candidate_entity_ids"))
    out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(g, on="source1_entity_id", how="left").fill_null("")
    out.write_csv(W + f"candidate_pairs_{a.split}.tsv", separator="\t", quote_style="never")
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
