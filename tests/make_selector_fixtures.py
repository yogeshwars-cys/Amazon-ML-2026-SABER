"""Selector contract fixtures (blocker -> selector), requested in comms/ASTRA_TO_CLAUDE.md "Test data needed".

Writes tests/fixtures/selector_v1/:
  valid/cand_fx_valid_{country}_{part:03d}.parquet   immutable shards, full schema v3 (same columns/dtypes as the blocker)
  valid/universe_roster.parquet every S1 of the split (entity_id, country): the roster for validate_candidate_parts
  valid/eval_roster.parquet     output / evaluation roster for select_sets(anchor_ids=...); includes zero-candidate
                                anchors and excludes competition-only parents (a stronger parent can sit outside it)
  valid/edge_scores.parquet     source1_entity_id, candidate_entity_id, country, p   (row-aligned with the shards, in order)
  valid/p_has_match.parquet     source1_entity_id, p_has_match                        (every roster anchor with candidates)
  valid/truth.parquet           s1, m   (full truth; includes pairs the blocker missed)
  invalid/<case>/...            shards (and roster) that must be rejected, with the expected error substring
  expected.json                 accepted edges per anchor for both empty-set modes, per-anchor F0.5, macro F0.5,
                                invalid-case errors, exact metric cases

Expected sets come from `reference_decode` below, an independent restatement of the decoding spec (not selector.py):
  ownership   per (country, child): highest p wins, ties -> lexically smallest S1 id; an owner needs p >= p_min.
              Ownership is resolved over the whole universe, so non-roster parents still compete.
  prefix      per anchor, owned children by p desc (ties -> candidate id); E|T| = sum of p over ALL the anchor's edges;
              E[F](k) = 1.25 * sum_{i<=k} p_i / (0.25 * E|T| + k); take the first argmax k.
  empty set   chosen unless best E[F] > 1 - p_has_match  (fallback mode: p_has_match = max edge p of the anchor).
The key cases are also hand-checked with asserts, so a wrong reference cannot silently produce wrong fixtures.

  python tests/make_selector_fixtures.py
"""
import json, shutil
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent / "fixtures" / "selector_v1"
P_MIN, BETA2 = 0.05, 0.25

SCHEMA = {  # blocker candidate schema v3, identical names / dtypes / order to cand_{emb}_{split}_{country}_{part}.parquet
    "source1_entity_id": pl.String, "candidate_entity_id": pl.String, "country": pl.String,
    "cos_v": pl.Float32, "cos_t": pl.Float32,
    "in_vq": pl.Boolean, "in_vr": pl.Boolean, "in_tq": pl.Boolean, "in_tr": pl.Boolean, "in_a": pl.Boolean, "in_n": pl.Boolean,
    "rank_vq": pl.Int8, "rank_vr": pl.Int8, "rank_tq": pl.Int8, "rank_tr": pl.Int8,
    "n_cand_s1": pl.Int32, "n_cand_r": pl.Int32, "cos_v_rank_in_s1": pl.Int32, "cos_v_rank_in_r": pl.Int32,
    "name_ratio": pl.Float32, "name_tset": pl.Float32, "name_partial": pl.Float32, "core_jw": pl.Float32,
    "core_equal": pl.Boolean, "addr_tset": pl.Float32, "addr_tsort": pl.Float32, "addr_partial": pl.Float32,
    "name_len_diff": pl.Int16, "num_jacc": pl.Float32, "house_eq": pl.Int8, "r_addr_empty": pl.Boolean,
    "region_overlap": pl.Int8, "name_wjacc": pl.Float32, "name_rare_miss_s1": pl.Float32, "name_rare_miss_r": pl.Float32,
    "name_idf_match": pl.Float32, "addr_wjacc": pl.Float32, "addr_rare_miss_s1": pl.Float32, "addr_rare_miss_r": pl.Float32,
    "nfreq_s1": pl.Int32, "nfreq_r": pl.Int32, "afreq_s1": pl.Int32, "afreq_r": pl.Int32, "r_nonlatin": pl.Boolean,
    "cand_source": pl.String, "s1_idx": pl.Int32, "cand_idx": pl.Int32, "label": pl.Boolean, "s1_part": pl.String,
}

# ------------------------------------------------------------------ cases
# anchor -> (country, shard part, in roster, p_has_match or None, [(child, p)], truth set)
C = "S1-FX-"
CASES = {
    # a child whose stronger parent is outside the roster and in another shard
    "outside_roster_parent": {
        C + "A01": ("US", 0, True, 0.95, [("S3-FX-AX", 0.60), ("S2-FX-A02", 0.80)], {"S2-FX-A02"}),
        C + "A99": ("US", 1, False, None, [("S3-FX-AX", 0.90)], {"S3-FX-AX"}),
    },
    # zero-candidate anchors: one with a blocker-missed match, one true singleton
    "zero_candidate": {
        C + "B01": ("US", 0, True, None, [], {"S2-FX-B01"}),
        C + "B02": ("India", 0, True, None, [], set()),
    },
    # true singleton with several moderate distractors
    "singleton_distractors": {
        C + "C01": ("India", 0, True, 0.10, [("S2-FX-C1", 0.30), ("S3-FX-C2", 0.25), ("S2-FX-C3", 0.20)], set()),
    },
    # two parents tie exactly on one child: lexical S1 id wins
    "tied_parents": {
        C + "D01": ("US", 0, True, 0.90, [("S2-FX-DT", 0.70)], {"S2-FX-DT"}),
        C + "D02": ("US", 0, True, 0.85, [("S2-FX-DT", 0.70), ("S2-FX-D2B", 0.65)], {"S2-FX-D2B"}),
    },
    # several accepted children for one S1 (S2 and S3)
    "multi_accept": {
        C + "E01": ("India", 1, True, 0.99, [("S2-FX-E1", 0.95), ("S3-FX-E2", 0.90), ("S2-FX-E3", 0.85)],
                    {"S2-FX-E1", "S3-FX-E2", "S2-FX-E3"}),
    },
    # children below p_min are never owned, even when they are the only candidate or the top parent is below p_min
    "below_p_min": {
        C + "F01": ("US", 1, True, 0.90, [("S2-FX-F1", 0.04)], {"S2-FX-F1"}),
        C + "F02": ("US", 1, True, 0.80, [("S2-FX-F2A", 0.04), ("S3-FX-F2B", 0.60)], {"S3-FX-F2B"}),
        C + "F03": ("US", 1, True, 0.50, [("S2-FX-FL", 0.045)], set()),
        C + "F04": ("US", 1, True, 0.50, [("S2-FX-FL", 0.030)], set()),
    },
    # the same child id in two countries: competition is country-local, so both anchors own it
    "same_child_two_countries": {
        C + "G01": ("US", 0, True, 0.90, [("S2-FX-GG", 0.80)], {"S2-FX-GG"}),
        C + "G02": ("India", 0, True, 0.90, [("S2-FX-GG", 0.90)], {"S2-FX-GG"}),
    },
    # expected-F0.5 prefix stops after the first child (partial recall)
    "prefix_cut": {
        C + "H01": ("India", 1, True, 0.95, [("S2-FX-H1", 0.90), ("S3-FX-H2", 0.50), ("S2-FX-H3", 0.10)],
                    {"S2-FX-H1", "S3-FX-H2"}),
    },
    # explicit p_has_match and the max-edge fallback disagree
    "p_has_match_decides": {
        C + "I01": ("US", 0, True, 0.20, [("S2-FX-I1", 0.60)], set()),
    },
    # competition between two roster anchors
    "roster_competition": {
        C + "J01": ("India", 1, True, 0.95, [("S2-FX-JX", 0.70), ("S2-FX-J1", 0.75)], {"S2-FX-J1"}),
        C + "J02": ("India", 1, True, 0.95, [("S2-FX-JX", 0.80)], {"S2-FX-JX"}),
    },
    # equal-p children inside one anchor: order by candidate id, both accepted
    "tied_children": {
        C + "K01": ("US", 1, True, 0.99, [("S3-FX-KB", 0.85), ("S2-FX-KA", 0.85)], {"S2-FX-KA", "S3-FX-KB"}),
    },
}
ANCHORS = {a: v for case in CASES.values() for a, v in case.items()}
CASE_OF = {a: name for name, case in CASES.items() for a in case}


# ------------------------------------------------------------------ independent reference decoder + metric
def reference_decode(edges, roster, p_has_match=None):
    """edges: list of (s1, child, country, p). Returns {roster s1: sorted accepted children}."""
    by_child = {}
    for s, c, ctry, p in edges:
        by_child.setdefault((ctry, c), []).append((s, p))
    owner = {}
    for key, parents in by_child.items():
        s, p = min(parents, key=lambda x: (-x[1], x[0]))
        if p >= P_MIN: owner[key] = s
    out = {}
    for s in roster:
        mine = [(c, ctry, p) for (s_, c, ctry, p) in edges if s_ == s]
        if not mine: out[s] = []; continue
        e_t = sum(p for _, _, p in mine)
        owned = sorted([(c, p) for c, ctry, p in mine if owner.get((ctry, c)) == s], key=lambda x: (-x[1], x[0]))
        if not owned: out[s] = []; continue
        best_f, best_k, tp = -1.0, 0, 0.0
        for k, (_, p) in enumerate(owned, 1):
            tp += p; f = (1 + BETA2) * tp / (BETA2 * e_t + k)
            if f > best_f: best_f, best_k = f, k
        phm = max(p for _, _, p in mine) if p_has_match is None else p_has_match[s]
        out[s] = sorted(c for c, _ in owned[:best_k]) if best_f > 1 - phm else []
    return out


def f05(P, T):
    P, T = set(P), set(T)
    if not P and not T: return 1.0
    tp = len(P & T)
    if tp == 0: return 0.0
    pr, rc = tp / len(P), tp / len(T)
    return (1 + BETA2) * pr * rc / (BETA2 * pr + rc)


# ------------------------------------------------------------------ edge frame with consistent blocker features
def build_edges():
    rows = []
    for a, (ctry, part, _, _, kids, truth) in ANCHORS.items():
        for c, p in kids:
            rows.append({"source1_entity_id": a, "candidate_entity_id": c, "country": ctry, "_part": part, "p": p,
                         "label": c in truth})
    d = pl.DataFrame(rows).with_columns(
        (0.5 + 0.5 * pl.col("p")).cast(pl.Float32).alias("cos_v"),
        (0.9 * pl.col("p")).cast(pl.Float32).alias("cos_t"))
    s1_ids = sorted(ANCHORS); r_ids = sorted(set(d["candidate_entity_id"]))
    # ranks and counts over the whole universe per country, exactly as the blocker computes them across shards
    d = d.with_columns(
        pl.col("cos_v").rank("ordinal", descending=True).over("source1_entity_id").cast(pl.Int32).alias("cos_v_rank_in_s1"),
        pl.col("cos_v").rank("ordinal", descending=True).over("country", "candidate_entity_id").cast(pl.Int32).alias("cos_v_rank_in_r"),
        pl.len().over("source1_entity_id").cast(pl.Int32).alias("n_cand_s1"),
        pl.len().over("country", "candidate_entity_id").cast(pl.Int32).alias("n_cand_r"))
    rq = (pl.col("cos_v_rank_in_s1") - 1).cast(pl.Int8); rr = (pl.col("cos_v_rank_in_r") - 1).cast(pl.Int8)
    d = d.with_columns(
        pl.lit(True).alias("in_vq"), rq.alias("rank_vq"), pl.lit(True).alias("in_tq"), rq.alias("rank_tq"),
        (rr < 5).alias("in_vr"), pl.when(rr < 5).then(rr).otherwise(99).cast(pl.Int8).alias("rank_vr"),
        (rr < 5).alias("in_tr"), pl.when(rr < 5).then(rr).otherwise(99).cast(pl.Int8).alias("rank_tr"),
        pl.lit(False).alias("in_a"), pl.lit(False).alias("in_n"))
    q = pl.col("p")
    d = d.with_columns(
        *[(q * w).cast(pl.Float32).alias(c) for c, w in [
            ("name_ratio", 1.0), ("name_tset", 1.0), ("name_partial", 1.0), ("core_jw", 1.0), ("addr_tset", 0.9),
            ("addr_tsort", 0.9), ("addr_partial", 0.95), ("num_jacc", 0.8), ("name_wjacc", 0.9), ("addr_wjacc", 0.85),
            ("name_idf_match", 10.0)]],
        *[((1 - q) * 5).cast(pl.Float32).alias(c) for c in
          ("name_rare_miss_s1", "name_rare_miss_r", "addr_rare_miss_s1", "addr_rare_miss_r")],
        (q > 0.8).alias("core_equal"), pl.lit(0, pl.Int16).alias("name_len_diff"),
        pl.when(q > 0.5).then(1).otherwise(0).cast(pl.Int8).alias("house_eq"), pl.lit(False).alias("r_addr_empty"),
        pl.lit(1, pl.Int8).alias("region_overlap"),
        pl.lit(1, pl.Int32).alias("nfreq_s1"), pl.lit(1, pl.Int32).alias("nfreq_r"),
        pl.lit(1, pl.Int32).alias("afreq_s1"), pl.lit(1, pl.Int32).alias("afreq_r"), pl.lit(False).alias("r_nonlatin"),
        pl.col("candidate_entity_id").str.slice(0, 2).alias("cand_source"),
        pl.col("source1_entity_id").replace_strict({s: i for i, s in enumerate(s1_ids)}, return_dtype=pl.Int32).alias("s1_idx"),
        pl.col("candidate_entity_id").replace_strict({s: i for i, s in enumerate(r_ids)}, return_dtype=pl.Int32).alias("cand_idx"),
        pl.lit("H").alias("s1_part"))
    # shard order = country, part, S1 id, then p desc (shards end on S1 boundaries like the blocker's)
    return d.sort(["country", "_part", "source1_entity_id", "p", "candidate_entity_id"], descending=[False, False, False, True, False])


def shard_frames(d):
    return {(c, p): g.select(list(SCHEMA)).cast(SCHEMA)
            for (c, p), g in d.group_by(["country", "_part"], maintain_order=True)}


# ------------------------------------------------------------------ invalid cases
def invalid_cases(d):
    base = d.select(list(SCHEMA)).cast(SCHEMA)
    one = base.filter(pl.col("source1_entity_id") == C + "E01")
    mod = lambda df, **kw: df.with_columns(**{k: pl.lit(v, SCHEMA[k]) if not isinstance(v, pl.Expr) else v.cast(SCHEMA[k]) for k, v in kw.items()})
    first = pl.int_range(0, pl.len()) == 0
    cases = {
        "dup_within_shard": ({"000": pl.concat([one, one.head(1)])}, None, "frame", "duplicate"),
        "dup_across_shards": ({"000": one, "001": one.head(1)}, None, "parts", "duplicate candidate edge across"),
        "null_candidate_id": ({"000": mod(one, candidate_entity_id=pl.when(first).then(None).otherwise(pl.col("candidate_entity_id")))}, None, "frame", "null value"),
        "null_cand_source": ({"000": mod(one, cand_source=pl.when(first).then(None).otherwise(pl.col("cand_source")))}, None, "frame", "null value"),
        "invalid_cand_source": ({"000": mod(one, cand_source=pl.when(first).then(pl.lit("S4")).otherwise(pl.col("cand_source")))}, None, "frame", "cand_source must be S2/S3"),
        "retrieval_rank_negative": ({"000": mod(one, rank_vq=pl.when(first).then(-1).otherwise(pl.col("rank_vq")))}, None, "frame", "zero based"),
        "retrieval_rank_over_99": ({"000": mod(one, rank_tq=pl.when(first).then(100).otherwise(pl.col("rank_tq")))}, None, "frame", "zero based"),
        "sentinel_flag_mismatch": ({"000": mod(one, in_vr=pl.when(first).then(False).otherwise(pl.col("in_vr")))}, None, "frame", "sentinel disagrees"),
        "competition_rank_zero_based": ({"000": mod(one, cos_v_rank_in_s1=pl.col("cos_v_rank_in_s1") - 1)}, None, "frame", "one based"),
        "missing_required_column": ({"000": one.drop("cos_t")}, None, "frame", "missing candidate columns"),
        "roster_missing_anchor": ({"000": one}, [C + "D01"], "parts", "absent from roster"),
        "roster_duplicate_anchor": ({"000": one}, [C + "E01", C + "E01"], "parts", "duplicate S1 ids"),
        "label_presence_disagrees": ({"000": one, "001": base.filter(pl.col("source1_entity_id") == C + "H01").drop("label")},
                                     None, "parts", "disagree on label presence"),
    }
    return cases


# ------------------------------------------------------------------ main
def main():
    if OUT.exists(): shutil.rmtree(OUT)
    (OUT / "valid").mkdir(parents=True); (OUT / "invalid").mkdir()
    d = build_edges()
    for (c, p), g in shard_frames(d).items():
        g.write_parquet(OUT / "valid" / f"cand_fx_valid_{c}_{p:03d}.parquet")
    edges_in_order = pl.concat([g.select("source1_entity_id", "candidate_entity_id", "country")
                                for _, g in sorted(shard_frames(d).items())])
    scores = edges_in_order.join(d.select("source1_entity_id", "candidate_entity_id", "country", "p"),
                                 on=["source1_entity_id", "candidate_entity_id", "country"], how="left", maintain_order="left")
    scores.write_parquet(OUT / "valid" / "edge_scores.parquet")
    roster = sorted(a for a, v in ANCHORS.items() if v[2])
    pl.DataFrame({"entity_id": roster, "country": [ANCHORS[a][0] for a in roster]}).write_parquet(OUT / "valid" / "eval_roster.parquet")
    uni = sorted(ANCHORS)
    pl.DataFrame({"entity_id": uni, "country": [ANCHORS[a][0] for a in uni]}).write_parquet(OUT / "valid" / "universe_roster.parquet")
    phm = {a: ANCHORS[a][3] for a in roster if ANCHORS[a][4]}
    pl.DataFrame({"source1_entity_id": list(phm), "p_has_match": list(phm.values())}).write_parquet(OUT / "valid" / "p_has_match.parquet")
    truth = {a: v[5] for a, v in ANCHORS.items()}
    pl.DataFrame([{"s1": a, "m": m} for a, ms in sorted(truth.items()) for m in sorted(ms)]).write_parquet(OUT / "valid" / "truth.parquet")

    edges = [(r["source1_entity_id"], r["candidate_entity_id"], r["country"], r["p"]) for r in scores.iter_rows(named=True)]
    modes = {"explicit_p_has_match": reference_decode(edges, roster, phm), "max_edge_fallback": reference_decode(edges, roster)}
    # hand checks of the key cases (independent of reference_decode)
    ex, fb = modes["explicit_p_has_match"], modes["max_edge_fallback"]
    assert ex[C + "A01"] == ["S2-FX-A02"]                         # AX owned by the non-roster parent A99
    assert ex[C + "B01"] == [] and ex[C + "B02"] == []
    assert ex[C + "C01"] == [] and fb[C + "C01"] == []            # best E[F] 0.316 < 0.9 (explicit) / 0.7 (fallback)
    assert ex[C + "D01"] == ["S2-FX-DT"] and ex[C + "D02"] == ["S2-FX-D2B"]
    assert ex[C + "E01"] == ["S2-FX-E1", "S2-FX-E3", "S3-FX-E2"]
    assert ex[C + "F01"] == [] and ex[C + "F02"] == ["S3-FX-F2B"] and ex[C + "F03"] == [] and ex[C + "F04"] == []
    assert ex[C + "G01"] == ["S2-FX-GG"] and ex[C + "G02"] == ["S2-FX-GG"]
    assert ex[C + "H01"] == ["S2-FX-H1"]                          # E[F]: k1 0.818 > k2 0.737 > k3 0.556
    assert ex[C + "I01"] == [] and fb[C + "I01"] == ["S2-FX-I1"]  # E[F] 0.652 vs empty 0.8 / 0.4
    assert ex[C + "J01"] == ["S2-FX-J1"] and ex[C + "J02"] == ["S2-FX-JX"]
    assert ex[C + "K01"] == ["S2-FX-KA", "S3-FX-KB"]

    expected = {"fixture_version": 1, "schema_version": 3, "p_min": P_MIN, "beta2": BETA2,
                "shards": sorted(p.name for p in (OUT / "valid").glob("cand_fx_valid_*.parquet")),
                "rows": d.height, "universe_anchors": len(ANCHORS), "roster_anchors": len(roster),
                "zero_candidate_anchors": sorted(a for a in roster if not ANCHORS[a][4]),
                "competition_only_anchors": sorted(a for a, v in ANCHORS.items() if not v[2]),
                "case_of_anchor": {a: CASE_OF[a] for a in sorted(ANCHORS)}, "modes": {}}
    for m, pred in modes.items():
        per = {a: round(f05(pred[a], truth[a]), 6) for a in roster}
        expected["modes"][m] = {"accepted": pred, "f05": per, "macro_f05": round(float(np.mean(list(per.values()))), 6)}
    inv = {}
    for name, (parts, rost, level, err) in invalid_cases(d).items():
        dd = OUT / "invalid" / name; dd.mkdir()
        for k, df in parts.items(): df.write_parquet(dd / f"cand_fx_{name}_US_{k}.parquet")
        if rost is not None: pl.DataFrame({"entity_id": rost}).write_parquet(dd / "roster.parquet")
        inv[name] = {"validate": level, "error_contains": err}
    expected["invalid"] = inv
    expected["metric_cases"] = [
        {"name": "empty_empty", "pred": [], "truth": [], "f05": 1.0},
        {"name": "pred_empty_truth_nonempty", "pred": [], "truth": ["a"], "f05": 0.0},
        {"name": "pred_nonempty_truth_empty", "pred": ["a"], "truth": [], "f05": 0.0},
        {"name": "false_positive_only", "pred": ["b"], "truth": ["a"], "f05": 0.0},
        {"name": "exact", "pred": ["a", "b"], "truth": ["a", "b"], "f05": 1.0},
        {"name": "partial_p05_r033", "pred": ["a", "b"], "truth": ["a", "c", "d"], "f05": round(f05(["a", "b"], ["a", "c", "d"]), 6)},
        {"name": "partial_p1_r05", "pred": ["a"], "truth": ["a", "b"], "f05": round(1.25 * 0.5 / 0.75, 6)},
        {"name": "macro_mix", "pred": {"x": [], "y": ["a"], "z": ["a", "b"]}, "truth": {"x": [], "y": ["b"], "z": ["a"]},
         "macro_f05": round((1.0 + 0.0 + f05(["a", "b"], ["a"])) / 3, 6)},
    ]
    with open(OUT / "expected.json", "w") as f: json.dump(expected, f, indent=1)
    print(f"wrote {OUT}: {d.height} edges, {len(roster)} roster anchors, {len(inv)} invalid cases")
    for m in modes: print(m, "macro F0.5", expected["modes"][m]["macro_f05"])


if __name__ == "__main__":
    main()
