"""Selector evaluation sets from real blocker output (companion to tests/fixtures/selector_v1).

For each labelled split with candidates (mini, minid, train) writes work/selector_eval/:
  roster_{emb}_{split}.parquet   independent S1 roster (every S1 of the split, zero-candidate ones included):
                                 entity_id, country, s1_part, n_cand, n_truth, n_truth_in_cands, singleton, zero_cand
  eval_sets_{emb}_{split}.json   candidates/S1 p50/p95/p99 per country, blocker ceiling (oracle macro F0.5 given the
                                 candidates, singletons and misses included), orphan-distractor share, and the
                                 cross-country variants (fit on one country's anchors, evaluate on the other's)

  python eval_sets.py --emb xsmE --split mini minid
"""
import argparse, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from config import WORK as W

OUT = W + "selector_eval/"


def roster(emb, split):
    s1 = pl.read_parquet(W + f"{split}_source1.parquet", columns=["entity_id", "country"])
    r_ids = pl.concat([pl.read_parquet(W + f"{split}_{s}.parquet", columns=["entity_id"]) for s in ("source2", "source3")])["entity_id"]
    c = pl.scan_parquet(W + f"cand_{emb}_{split}_*.parquet").select("source1_entity_id", "candidate_entity_id", "label").collect()
    n = c.group_by("source1_entity_id").agg(pl.len().alias("n_cand"), pl.col("label").sum().alias("n_truth_in_cands"))
    t = pl.read_parquet(W + "train_pairs.parquet").filter(pl.col("s1").is_in(s1["entity_id"].implode()) &
                                                          pl.col("m").is_in(r_ids.implode()))
    nt = t.group_by("s1").len("n_truth")
    parts = pl.read_parquet(W + "partitions.parquet").rename({"part": "s1_part"})
    out = s1.join(parts, on="entity_id", how="left").join(n, left_on="entity_id", right_on="source1_entity_id", how="left") \
        .join(nt, left_on="entity_id", right_on="s1", how="left").with_columns(
            pl.col("s1_part").fill_null("-"), pl.col("n_cand", "n_truth_in_cands", "n_truth").fill_null(0).cast(pl.Int32)) \
        .with_columns((pl.col("n_truth") == 0).alias("singleton"), (pl.col("n_cand") == 0).alias("zero_cand"))
    # R records whose true S1 is not in this split (orphans: pure distractors)
    t_all = pl.read_parquet(W + "train_pairs.parquet").filter(pl.col("m").is_in(r_ids.implode()))
    orphan = int(t_all.filter(~pl.col("s1").is_in(s1["entity_id"].implode())).height)
    return out, orphan, len(r_ids)


def oracle_f05(ro):
    """Macro F0.5 if the selector picked exactly the true pairs among the candidates."""
    tp, T = ro["n_truth_in_cands"].to_numpy(), ro["n_truth"].to_numpy()
    f = np.where(T == 0, 1.0, np.where(tp == 0, 0.0, 1.25 * tp / np.maximum(0.25 * T + tp, 1e-9)))   # precision = 1
    return round(float(f.mean()), 5)


def summarise(ro, orphan, n_r, emb, split):
    rep = {"emb": emb, "split": split, "s1": ro.height, "r": n_r, "r_per_s1": round(n_r / ro.height, 3),
           "orphan_distractors": orphan, "orphan_share_of_r": round(orphan / n_r, 4), "countries": {}}
    for (c,), g in ro.group_by(["country"], maintain_order=True):
        k = g.filter(~pl.col("zero_cand"))["n_cand"]
        rep["countries"][c] = {
            "s1": g.height, "zero_cand": int(g["zero_cand"].sum()), "singleton_share": round(float(g["singleton"].mean()), 4),
            "cand_per_s1": {"mean": round(float(k.mean()), 2), "p50": float(k.quantile(0.5)), "p95": float(k.quantile(0.95)),
                            "p99": float(k.quantile(0.99)), "max": int(k.max())},
            "pair_recall": round(float(g["n_truth_in_cands"].sum() / max(g["n_truth"].sum(), 1)), 5),
            "oracle_macro_f05": oracle_f05(g)}
    rep["oracle_macro_f05"] = oracle_f05(ro)
    cs = sorted(rep["countries"])
    rep["variants"] = {f"{a}_to_{b}": {"fit_filter": {"country": a}, "eval_filter": {"country": b},
                                        "note": "competition universe = the eval country's full candidate set"}
                       for a in cs for b in cs if a != b}
    rep["variants"]["france_like_no_alias"] = {"status": "pending: needs a blocker run with region aliases disabled "
                                                         "(test France has no train labels, so no aliases)"}
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--emb", required=True); ap.add_argument("--split", nargs="+", default=["mini", "minid"])
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    for sp in a.split:
        ro, orphan, n_r = roster(a.emb, sp)
        ro.write_parquet(OUT + f"roster_{a.emb}_{sp}.parquet")
        rep = summarise(ro, orphan, n_r, a.emb, sp)
        with open(OUT + f"eval_sets_{a.emb}_{sp}.json", "w") as f: json.dump(rep, f, indent=1)
        print(json.dumps({k: rep[k] for k in ("split", "s1", "r_per_s1", "orphan_share_of_r", "oracle_macro_f05")}),
              {c: (v["cand_per_s1"], v["oracle_macro_f05"]) for c, v in rep["countries"].items()}, flush=True)
