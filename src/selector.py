"""Base selector: blocker candidates -> per-S1 match sets for macro F0.5.

stage 1  LightGBM pair scorer on blocker features (out-of-fold by S1 group)
stage 2  LightGBM with competition / consistency features built from stage-1 probabilities:
         margin vs the best other S1 wanting the same record (exclusivity), rank / gap inside the S1's list,
         similarity to the S1's other likely matches (weighted embedding centroid), expected S2 / S3 counts
select   exclusivity (each S2/S3 record -> at most its best S1), then per S1 the top-k prefix maximising
         E[F0.5] ~ 1.25 * sum_{i<=k} p_i / (0.25 * E|T| + k). The empty set is scored by an explicit
         anchor-level P(no match), never by multiplying dependent edge probabilities.
score    exact macro F0.5 per S1 (singletons included; |T| from ground truth, so blocker misses count)

  python selector.py dev   --emb stock --split mini       # 50/50 S1 split of one candidate set, for development
  python selector.py fit   --emb TAG --split train         # train on non-holdout S1s, report on the holdout
  python selector.py apply --emb TAG --split test          # write matching_results.tsv
"""
import argparse, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from config import WORK as W

EDGE_ID = ("source1_entity_id", "candidate_entity_id")
REQUIRED_COLUMNS = {
    "source1_entity_id", "candidate_entity_id", "country", "cand_source",
    "cos_v", "cos_t", "in_vq", "in_vr", "in_tq", "in_tr", "in_a", "in_n",
    "rank_vq", "rank_vr", "rank_tq", "rank_tr", "n_cand_s1", "n_cand_r",
    "cos_v_rank_in_s1", "cos_v_rank_in_r",
}
RETRIEVAL_RANKS = ("rank_vq", "rank_vr", "rank_tq", "rank_tr")
COMPETITION_RANKS = ("cos_v_rank_in_s1", "cos_v_rank_in_r")
DROP = {"source1_entity_id", "candidate_entity_id", "country", "label", "cand_source", "fold"}
P1 = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10)


class CandidateContractError(ValueError):
    """Raised when blocker output cannot be consumed safely by the selector."""


@dataclass(frozen=True)
class CandidateManifest:
    """Validated description of one frozen blocker candidate set."""

    paths: tuple[str, ...]
    rows: int
    anchors_with_candidates: int
    roster_anchors: int | None
    zero_candidate_anchors: int | None
    countries: tuple[str, ...]
    has_label: bool


def candidate_parts(emb, split, work=W):
    """Return the ordered immutable Parquet parts for a blocker run."""
    root = Path(work)
    paths = tuple(str(p) for p in sorted(root.glob(f"cand_{emb}_{split}_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no candidate parts for emb={emb!r}, split={split!r} in {root}")
    return paths


def validate_candidate_frame(df, require_label=False):
    """Validate the blocker/selector edge contract on a materialized frame.

    Retrieval ranks are zero based with 99 meaning absent. Competition ranks
    are one based. These conventions are deliberately checked separately.
    """
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if require_label and "label" not in df.columns:
        missing.append("label")
    if missing:
        raise CandidateContractError(f"missing candidate columns: {', '.join(missing)}")
    if df.select(pl.struct(EDGE_ID).is_duplicated().any()).item():
        raise CandidateContractError("duplicate (source1_entity_id, candidate_entity_id) edge")
    non_null = (*EDGE_ID, "country", "cand_source", *RETRIEVAL_RANKS, *COMPETITION_RANKS)
    if df.select(pl.any_horizontal([pl.col(c).is_null() for c in non_null]).any()).item():
        raise CandidateContractError("null value in required candidate identity/rank columns")
    sources = set(df["cand_source"].unique().to_list())
    if not sources <= {"S2", "S3"}:
        raise CandidateContractError(f"cand_source must be S2/S3, got {sorted(sources)}")
    for c in RETRIEVAL_RANKS:
        bad = df.filter((pl.col(c) < 0) | (pl.col(c) > 99))
        if bad.height:
            raise CandidateContractError(f"{c} must be zero based with 99 as the absent sentinel")
        flag = "in_" + c.removeprefix("rank_")
        inconsistent = df.filter(
            (pl.col(flag) & (pl.col(c) == 99)) |
            (~pl.col(flag) & (pl.col(c) != 99))
        )
        if inconsistent.height:
            raise CandidateContractError(f"{c} sentinel disagrees with {flag}")
    for c in COMPETITION_RANKS:
        if df.filter(pl.col(c) < 1).height:
            raise CandidateContractError(f"{c} must be one based")
    return df


def validate_candidate_parts(paths, require_label=False, anchor_roster=None):
    """Validate schemas and global edge uniqueness across candidate shards."""
    scans = []
    has_label = None
    for path in paths:
        schema = pl.scan_parquet(path).collect_schema()
        missing = sorted(REQUIRED_COLUMNS - set(schema.names()))
        if require_label and "label" not in schema:
            missing.append("label")
        if missing:
            raise CandidateContractError(f"{path}: missing columns: {', '.join(missing)}")
        labelled = "label" in schema
        if has_label is not None and labelled != has_label:
            raise CandidateContractError("candidate shards disagree on label presence")
        has_label = labelled
        scans.append(pl.scan_parquet(path).select(*EDGE_ID, "country"))
    edges = pl.concat(scans, how="vertical_relaxed")
    summary = edges.select(
        pl.len().alias("rows"),
        pl.col("source1_entity_id").n_unique().alias("anchors"),
        pl.col("country").drop_nulls().unique().sort().alias("countries"),
        pl.struct(EDGE_ID).is_duplicated().any().alias("duplicated"),
    ).collect()
    if summary["duplicated"][0]:
        raise CandidateContractError("duplicate candidate edge across blocker shards")
    covered = int(summary["anchors"][0])
    roster_count = zero_count = None
    if anchor_roster is not None:
        roster = anchor_roster.to_list() if isinstance(anchor_roster, pl.Series) else list(anchor_roster)
        if len(roster) != len(set(roster)):
            raise CandidateContractError("anchor roster contains duplicate S1 ids")
        edge_ids = set(edges.select("source1_entity_id").unique().collect()["source1_entity_id"].to_list())
        unknown = edge_ids - set(roster)
        if unknown:
            raise CandidateContractError(f"candidate set contains {len(unknown)} anchors absent from roster")
        roster_count = len(roster)
        zero_count = roster_count - len(edge_ids)
    return CandidateManifest(
        paths=tuple(map(str, paths)), rows=int(summary["rows"][0]),
        anchors_with_candidates=covered, roster_anchors=roster_count,
        zero_candidate_anchors=zero_count,
        countries=tuple(summary["countries"][0]), has_label=bool(has_label),
    )


def load_cands(emb, split, s1_keep=None, validate=False):
    """Lazy scan of the candidate parts, optionally restricted to a set of S1 ids (RAM-bounded at full scale)."""
    fs = candidate_parts(emb, split)
    q = pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")
    if s1_keep is not None: q = q.filter(pl.col("source1_entity_id").is_in(s1_keep.implode()))
    df = q.collect()
    if validate:
        validate_candidate_frame(df, require_label=split != "test")
    if "label" not in df.columns: df = df.with_columns(pl.lit(False).alias("label"))
    rank_missing = [
        (pl.col(c) == 99).cast(pl.Int8).alias(f"{c}_missing")
        for c in RETRIEVAL_RANKS
    ]
    return df.with_columns(
        (pl.col("cand_source") == "S3").cast(pl.Int8).alias("is_s3"),
        pl.col(pl.Boolean).exclude("label").cast(pl.Int8),
        *rank_missing,
    )


def s1_folds(ids, k, seed=0):
    """Stable, balanced entity-grouped folds."""
    u = ids.unique().sort()
    if k < 2 or len(u) < k:
        raise ValueError(f"need at least {k} unique S1 ids for {k} folds")
    f = np.arange(len(u), dtype=np.int16) % k
    np.random.default_rng(seed).shuffle(f)
    return pl.DataFrame({"source1_entity_id": u, "fold": f})


def feats(df):
    return [c for c in df.columns if c not in DROP and not c.startswith("_")]


def oof_predict(df, cols, params, rounds, k=4):
    """Out-of-fold probabilities (by S1 group) + a model trained on all rows."""
    X = df.select(cols).to_numpy().astype(np.float32); y = df["label"].to_numpy(); fold = df["fold"].to_numpy()
    if np.unique(y).size < 2:
        raise ValueError("pair model needs both positive and negative training edges")
    oof = np.zeros(len(df), np.float32)
    for f in range(k):
        tr, te = fold != f, fold == f
        if not te.any():
            raise ValueError(f"fold {f} has no validation rows")
        if np.unique(y[tr]).size < 2:
            raise ValueError(f"fold {f} training rows do not contain both classes")
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), rounds)
        oof[te] = m.predict(X[te])
    full = lgb.train(params, lgb.Dataset(X, y), rounds)
    return oof, full


def ctx_features(df, p, emb_r=None):
    """Stage-2 context from scores over a complete competition universe.

    Callers must pass every candidate edge for the relevant countries. Filtering
    to evaluation S1s before this reduction makes child ambiguity optimistic.
    """
    if len(df) != len(p):
        raise ValueError("probability array must be row-aligned with candidates")
    d = df.select("source1_entity_id", "candidate_entity_id", "country", "is_s3").with_columns(pl.Series("p", p))
    d = d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("source1_entity_id").alias("p_rank_s1"),
        (pl.col("p").max().over("source1_entity_id") - pl.col("p")).alias("p_gap_best_s1"),
        pl.col("p").sum().over("source1_entity_id").alias("p_sum_s1"),
        (pl.col("p") > 0.5).sum().over("source1_entity_id").alias("n_hi_s1"),
        (pl.col("p") * pl.col("is_s3")).sum().over("source1_entity_id").alias("e_s3_s1"),
        (pl.col("p") * (1 - pl.col("is_s3"))).sum().over("source1_entity_id").alias("e_s2_s1"),
        pl.col("p").rank("ordinal", descending=True).over("country", "candidate_entity_id").alias("p_rank_r"),
        pl.len().over("country", "candidate_entity_id").alias("n_s1_r"))
    # best competing S1 for the same record (exclusivity margin)
    top2 = d.group_by("country", "candidate_entity_id").agg(pl.col("p").top_k(2).alias("t"))
    top2 = top2.with_columns(pl.col("t").list.get(0).alias("b1"), pl.col("t").list.get(1, null_on_oob=True).fill_null(0.0).alias("b2")).drop("t")
    d = d.join(top2, on=["country", "candidate_entity_id"], how="left", maintain_order="left")
    d = d.with_columns(pl.when(pl.col("p") >= pl.col("b1")).then(pl.col("b2")).otherwise(pl.col("b1")).alias("p_best_other_s1"))
    d = d.with_columns((pl.col("p") - pl.col("p_best_other_s1")).alias("p_margin_r"))
    out = d.select("p", "p_rank_s1", "p_gap_best_s1", "p_sum_s1", "n_hi_s1", "e_s3_s1", "e_s2_s1",
                   "p_rank_r", "n_s1_r", "p_best_other_s1", "p_margin_r").rename({"p": "p1"})
    if emb_r is not None:                                   # similarity to the S1's other likely matches
        E = emb_r.astype(np.float32); s1 = df["source1_entity_id"].to_numpy()
        _, gi = np.unique(s1, return_inverse=True)
        w = p.astype(np.float32)
        C = np.zeros((gi.max() + 1, E.shape[1]), np.float32); np.add.at(C, gi, E * w[:, None])
        Wt = np.bincount(gi, weights=w)
        Cx = C[gi] - E * w[:, None]; wx = Wt[gi] - w
        cen = Cx / np.maximum(np.linalg.norm(Cx, axis=1, keepdims=True), 1e-6)
        out = out.with_columns(pl.Series("sim_peers", np.where(wx > 0.3, np.einsum("ij,ij->i", E, cen), -1.0).astype(np.float32)),
                               pl.Series("w_peers", wx.astype(np.float32)))
    return out


def select_sets(df, p, beta2=0.25, p_min=0.05, p_has_match=None, anchor_ids=None):
    """Resolve child ownership and choose each anchor's expected-F0.5 prefix.

    ``df`` and ``p`` define the complete competition universe. ``anchor_ids``
    is the independent output roster and may include zero-candidate anchors.
    ``p_has_match`` is a mapping from S1 id to an anchor-model probability. If
    it is omitted, ``max(edge probability)`` is used as an explicit baseline
    proxy; dependent edge probabilities are never multiplied.
    """
    if len(df) != len(p):
        raise ValueError("probability array must be row-aligned with candidates")
    prob = np.asarray(p, dtype=np.float64)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("edge probabilities must be finite values in [0, 1]")
    roster = (anchor_ids.to_list() if isinstance(anchor_ids, pl.Series) else list(anchor_ids)) if anchor_ids is not None else df["source1_entity_id"].unique().to_list()
    if len(roster) != len(set(roster)):
        raise ValueError("anchor roster contains duplicate S1 ids")
    out = {s: [] for s in roster}
    if df.is_empty():
        return out

    d = df.select("source1_entity_id", "candidate_entity_id", "country").with_columns(pl.Series("p", prob))
    anchor = d.group_by("source1_entity_id").agg(
        pl.col("p").sum().alias("expected_truth"),
        pl.col("p").max().alias("max_p"),
    )
    d = d.sort(
        ["country", "candidate_entity_id", "p", "source1_entity_id"],
        descending=[False, False, True, False],
    ).with_columns(
        pl.int_range(0, pl.len()).over("country", "candidate_entity_id").alias("child_order")
    )
    owned = d.filter((pl.col("child_order") == 0) & (pl.col("p") >= p_min)).join(
        anchor, on="source1_entity_id", how="left"
    ).sort(["source1_entity_id", "p", "candidate_entity_id"], descending=[False, True, False])
    if owned.is_empty():
        return out
    owned = owned.with_columns(
        pl.col("p").cum_sum().over("source1_entity_id").alias("expected_tp"),
        pl.int_range(1, pl.len() + 1).over("source1_entity_id").alias("k"),
    ).with_columns(
        ((1 + beta2) * pl.col("expected_tp") /
         (beta2 * pl.col("expected_truth") + pl.col("k"))).alias("expected_f")
    )
    best = owned.group_by("source1_entity_id").agg(
        pl.col("expected_f").max().alias("best_nonempty"),
        pl.col("expected_f").arg_max().alias("best_index"),
        pl.col("max_p").first(),
    ).filter(pl.col("source1_entity_id").is_in(roster))
    if p_has_match is None:
        best = best.with_columns((1 - pl.col("max_p")).alias("empty_score"))
    else:
        unknown = set(best["source1_entity_id"].to_list()) - set(p_has_match)
        if unknown:
            raise ValueError(f"p_has_match missing {len(unknown)} anchors")
        vals = np.asarray([p_has_match[s] for s in best["source1_entity_id"].to_list()], dtype=np.float64)
        if not np.isfinite(vals).all() or ((vals < 0) | (vals > 1)).any():
            raise ValueError("p_has_match values must be finite values in [0, 1]")
        best = best.with_columns(pl.Series("empty_score", 1 - vals))
    keep = owned.join(best, on="source1_entity_id", how="left").filter(
        (pl.col("best_nonempty") > pl.col("empty_score")) &
        (pl.col("k") <= pl.col("best_index") + 1)
    )
    sets = keep.filter(pl.col("source1_entity_id").is_in(roster)).group_by(
        "source1_entity_id", maintain_order=True
    ).agg("candidate_entity_id")
    out.update(dict(zip(sets["source1_entity_id"].to_list(), sets["candidate_entity_id"].to_list())))
    return out


def macro_f05(pred, truth):
    """pred/truth: {s1: list}. Exact challenge metric over the S1s in pred."""
    f = []
    for s, P in pred.items():
        T = truth.get(s, set()); P = set(P)
        if not T and not P: f.append(1.0); continue
        tp = len(P & T)
        if tp == 0: f.append(0.0); continue
        pr, rc = tp / len(P), tp / len(T); f.append(1.25 * pr * rc / (0.25 * pr + rc))
    return float(np.mean(f))


def truth_sets(s1_ids):
    t = pl.read_parquet(W + "train_pairs.parquet").filter(pl.col("s1").is_in(s1_ids.implode()))
    g = t.group_by("s1").agg("m")
    d = {s: set(m) for s, m in zip(g["s1"].to_list(), g["m"].to_list())}
    return {s: d.get(s, set()) for s in s1_ids.to_list()}


def emb_for(df, emb, split):
    """Embedding rows for each candidate (R side), aligned with df."""
    r = pl.concat([pl.read_parquet(W + f"{split}_{s}.parquet", columns=["entity_id"]) for s in ("source2", "source3")])
    er = np.concatenate([np.load(W + f"emb_{emb}_{split}_{s}.npy", mmap_mode="r") for s in ("source2", "source3")])
    row = df.select("candidate_entity_id").join(
        r.with_row_index("row"), left_on="candidate_entity_id", right_on="entity_id",
        how="left", maintain_order="left"
    )["row"]
    if row.null_count():
        raise CandidateContractError("candidate id missing from normalized child records/embedding map")
    idx = row.to_numpy().astype(np.int64, copy=False)
    if (idx < 0).any() or (idx >= len(er)).any():
        raise CandidateContractError("candidate embedding row index is out of range")
    return np.asarray(er[idx])


def fit_two_stage(df, emb_r=None, rounds=(300, 300)):
    """Fit both LightGBM stages from cross-fitted training predictions."""
    cols1 = feats(df.drop("is_s3")) + ["is_s3"]
    t = time.time()
    oof1, m1 = oof_predict(df, cols1, P1, rounds[0])
    context = ctx_features(df, oof1, emb_r)
    train2 = pl.concat([df, context], how="horizontal")
    cols2 = cols1 + context.columns
    _, m2 = oof_predict(train2, cols2, P1, rounds[1])
    print(f"trained two stages on {df.height} rows in {time.time()-t:.0f}s", flush=True)
    return m1, m2, cols1, cols2


def predict_two_stage(df, models, emb_r=None):
    """Score one complete inference universe with global context."""
    m1, m2, cols1, cols2 = models
    p1 = m1.predict(df.select(cols1).to_numpy().astype(np.float32)).astype(np.float32)
    context = ctx_features(df, p1, emb_r)
    scored = pl.concat([df, context], how="horizontal")
    p2 = m2.predict(scored.select(cols2).to_numpy().astype(np.float32)).astype(np.float32)
    return p1, p2


def run(df, train_mask, emb_r, rounds=(300, 300), report=None):
    """Two-stage fit with context reduced over the full supplied universe.

    Training rows receive out-of-fold scores and evaluation rows receive scores
    from the model fitted on all training rows. Context is computed only after
    those arrays are reassembled, so non-evaluation parents remain competitors.
    """
    cols1 = feats(df.drop("is_s3")) + ["is_s3"]
    tr = df.filter(pl.Series(train_mask)); te = df.filter(pl.Series(~train_mask))
    t = time.time()
    oof1, m1 = oof_predict(tr, cols1, P1, rounds[0])
    p1_te = m1.predict(te.select(cols1).to_numpy().astype(np.float32))
    p1 = np.empty(df.height, np.float32); p1[train_mask] = oof1; p1[~train_mask] = p1_te
    print(f"stage 1: {len(tr)} train rows, {time.time()-t:.0f}s", flush=True)
    context = ctx_features(df, p1, emb_r)
    c_tr = context.filter(pl.Series(train_mask)); c_te = context.filter(pl.Series(~train_mask))
    tr2, te2 = pl.concat([tr, c_tr], how="horizontal"), pl.concat([te, c_te], how="horizontal")
    cols2 = cols1 + c_tr.columns
    oof2, m2 = oof_predict(tr2, cols2, P1, rounds[1])
    p2_te = m2.predict(te2.select(cols2).to_numpy().astype(np.float32))
    p2 = np.empty(df.height, np.float32); p2[train_mask] = oof2; p2[~train_mask] = p2_te
    print(f"stage 2 done {time.time()-t:.0f}s", flush=True)
    imp = sorted(zip(cols2, m2.feature_importance("gain")), key=lambda x: -x[1])[:15]
    print("top stage-2 features:", [(c, round(g / 1e3)) for c, g in imp], flush=True)
    return p1, p2, (m1, m2, cols1, cols2)


def sample_s1(emb, split, frac, exclude=None, seed=0):
    fs = sorted(W + f for f in os.listdir(W) if f.startswith(f"cand_{emb}_{split}_") and f.endswith(".parquet"))
    ids = pl.concat([pl.scan_parquet(f).select("source1_entity_id") for f in fs]).unique().collect()["source1_entity_id"]
    if exclude is not None: ids = ids.filter(~ids.is_in(exclude.implode()))
    return ids.sample(fraction=frac, seed=seed) if frac < 1 else ids


def evaluate(universe, p1, p2, truth):
    """Evaluate a scoped S1 roster while resolving ownership globally."""
    s1_te = pl.Series(list(truth))
    rep = {}
    for name, p in (("stage1", p1), ("stage2", p2)):
        rep[name] = round(macro_f05(select_sets(universe, p, anchor_ids=s1_te), truth), 5)
    lab = universe.filter(pl.col("label") == 1).group_by("source1_entity_id").agg("candidate_entity_id")
    orc = {s: [] for s in s1_te.to_list()}; orc.update(dict(zip(lab["source1_entity_id"].to_list(), lab["candidate_entity_id"].to_list())))
    orc = {s: orc[s] for s in truth}
    rep["oracle_given_candidates"] = round(macro_f05(orc, truth), 5)
    rep["by_country"] = {}
    for c in universe["country"].unique().to_list():
        m = (universe["country"] == c).to_numpy()
        ids = [s for s in universe.filter(pl.Series(m))["source1_entity_id"].unique().to_list() if s in truth]
        if ids:
            rep["by_country"][c] = round(macro_f05(
                select_sets(universe.filter(pl.Series(m)), p2[m], anchor_ids=ids),
                {s: truth[s] for s in ids}), 5)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "fit", "apply"]); ap.add_argument("--emb", required=True)
    ap.add_argument("--split", default="train"); ap.add_argument("--no-emb", dest="use_emb", action="store_false")
    ap.add_argument("--train-frac", type=float, default=1.0, help="fraction of training S1s used (RAM)")
    ap.add_argument("--rounds", type=int, default=300, help="boosting rounds per stage")
    a = ap.parse_args(); t0 = time.time()
    hq = pl.read_parquet(W + "q.parquet")["entity_id"]
    if a.cmd == "dev":                                        # 50/50 by S1 of one candidate set
        df = load_cands(a.emb, a.split, validate=True)
        half = s1_folds(df["source1_entity_id"], 2, seed=7).rename({"fold": "_half"})
        df = df.join(half, on="source1_entity_id"); train_mask = (df["_half"] == 0).to_numpy(); df = df.drop("_half")
        emb_r = emb_for(df, a.emb, a.split) if a.use_emb else None
    elif a.cmd == "fit":                                      # train on sampled non-holdout S1s, evaluate on the holdout
        keep = pl.concat([sample_s1(a.emb, "train", a.train_frac, exclude=hq), hq])
        df = load_cands(a.emb, "train", keep, validate=True)
        train_mask = (~df["source1_entity_id"].is_in(hq.implode())).to_numpy()
        emb_r = emb_for(df, a.emb, "train") if a.use_emb else None
    else:                                                     # train, then score test as a separate competition universe
        tr = load_cands(a.emb, "train", sample_s1(a.emb, "train", a.train_frac), validate=True)
        tr = tr.join(s1_folds(tr["source1_entity_id"], 4), on="source1_entity_id", maintain_order="left")
        emb_tr = emb_for(tr, a.emb, "train") if a.use_emb else None
        models = fit_two_stage(tr, emb_tr, rounds=(a.rounds, a.rounds))
        del tr, emb_tr
        te = load_cands(a.emb, a.split, validate=True)
        te = te.join(s1_folds(te["source1_entity_id"], 4), on="source1_entity_id", maintain_order="left")
        emb_te = emb_for(te, a.emb, a.split) if a.use_emb else None
        _, p2 = predict_two_stage(te, models, emb_te)
        s1_all = pl.read_parquet(W + f"{a.split}_source1.parquet", columns=["entity_id"])["entity_id"]
        pred = select_sets(te, p2, anchor_ids=s1_all)
        out = pl.DataFrame({"source1_entity_id": s1_all}).with_columns(
            pl.col("source1_entity_id").replace_strict(
                {k: ",".join(v) for k, v in pred.items()}, default=""
            ).alias("matched_entity_ids"))
        out.write_csv(W + f"matching_results_{a.split}.tsv", separator="\t", quote_style="never")
        print("wrote", W + f"matching_results_{a.split}.tsv", out.height, "rows,",
              sum(len(v) for v in pred.values()), "matches", flush=True)
        return
    df = df.join(s1_folds(df["source1_entity_id"], 4), on="source1_entity_id", maintain_order="left")
    print(f"{a.cmd}: {df.height} pairs, {df['source1_entity_id'].n_unique()} S1, train rows {train_mask.sum()}, {time.time()-t0:.0f}s", flush=True)
    p1, p2, _ = run(df, train_mask, emb_r, rounds=(a.rounds, a.rounds))
    s1_te = df.filter(pl.Series(~train_mask))["source1_entity_id"].unique() if a.cmd == "dev" else hq
    rep = evaluate(df, p1, p2, truth_sets(s1_te))
    print(json.dumps(rep), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(rep, open(W + f"selector_{a.cmd}_{a.emb}_{a.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
