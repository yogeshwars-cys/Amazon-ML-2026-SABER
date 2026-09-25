"""selector.py against the frozen blocker fixtures (tests/fixtures/selector_v1, built by make_selector_fixtures.py)."""
import json
import sys
import unittest
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
FX = HERE / "fixtures" / "selector_v1"

import selector


def _expected():
    with open(FX / "expected.json") as f:
        return json.load(f)


def _load(name):
    return pl.read_parquet(FX / "valid" / name)


class FixtureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _expected()
        cls.paths = [str(FX / "valid" / s) for s in cls.exp["shards"]]
        cls.universe = _load("universe_roster.parquet")["entity_id"]

    def test_valid_shards_pass_frame_and_parts_validation(self):
        df = pl.concat([pl.read_parquet(p) for p in self.paths])
        selector.validate_candidate_frame(df, require_label=True)
        man = selector.validate_candidate_parts(self.paths, require_label=True, anchor_roster=self.universe)
        self.assertEqual(man.rows, self.exp["rows"])
        self.assertEqual(man.roster_anchors, self.exp["universe_anchors"])
        self.assertEqual(man.zero_candidate_anchors, len(self.exp["zero_candidate_anchors"]))

    def test_invalid_cases_are_rejected(self):
        for name, spec in self.exp["invalid"].items():
            with self.subTest(case=name):
                d = FX / "invalid" / name
                paths = sorted(str(p) for p in d.glob("cand_fx_*.parquet"))
                roster = pl.read_parquet(d / "roster.parquet")["entity_id"] if (d / "roster.parquet").exists() else None
                with self.assertRaises(selector.CandidateContractError) as cm:
                    if spec["validate"] == "frame":
                        selector.validate_candidate_frame(pl.concat([pl.read_parquet(p) for p in paths]), require_label=True)
                    else:
                        selector.validate_candidate_parts(paths, anchor_roster=roster)
                self.assertIn(spec["error_contains"], str(cm.exception))


class FixtureDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _expected()
        cls.df = pl.concat([pl.read_parquet(FX / "valid" / s) for s in cls.exp["shards"]])
        sc = _load("edge_scores.parquet")
        assert sc.select("source1_entity_id", "candidate_entity_id", "country").equals(
            cls.df.select("source1_entity_id", "candidate_entity_id", "country")), "edge_scores not row-aligned"
        cls.p = sc["p"].to_numpy()
        cls.roster = _load("eval_roster.parquet")["entity_id"]
        ph = _load("p_has_match.parquet")
        cls.phm = dict(zip(ph["source1_entity_id"].to_list(), ph["p_has_match"].to_list()))
        t = _load("truth.parquet").group_by("s1").agg("m")
        truth = {s: set(m) for s, m in zip(t["s1"].to_list(), t["m"].to_list())}
        cls.truth = {s: truth.get(s, set()) for s in cls.roster.to_list()}

    def _check(self, mode, **kw):
        exp = self.exp["modes"][mode]
        got = selector.select_sets(self.df, self.p, p_min=self.exp["p_min"], beta2=self.exp["beta2"],
                                   anchor_ids=self.roster, **kw)
        self.assertEqual(set(got), set(self.roster.to_list()), "output must cover exactly the roster")
        for s, want in exp["accepted"].items():
            with self.subTest(mode=mode, anchor=s, case=self.exp["case_of_anchor"][s]):
                self.assertEqual(sorted(got[s]), want)
        self.assertAlmostEqual(selector.macro_f05(got, self.truth), exp["macro_f05"], places=6)

    def test_explicit_p_has_match(self):
        self._check("explicit_p_has_match", p_has_match=self.phm)

    def test_max_edge_fallback(self):
        self._check("max_edge_fallback")

    def test_row_order_invariance(self):
        perm = list(reversed(range(self.df.height)))
        a = selector.select_sets(self.df, self.p, anchor_ids=self.roster, p_has_match=self.phm)
        b = selector.select_sets(self.df[perm], self.p[perm], anchor_ids=self.roster, p_has_match=self.phm)
        self.assertEqual({k: sorted(v) for k, v in a.items()}, {k: sorted(v) for k, v in b.items()})

    def test_metric_cases(self):
        for c in self.exp["metric_cases"]:
            with self.subTest(case=c["name"]):
                if "macro_f05" in c:
                    got = selector.macro_f05(c["pred"], {k: set(v) for k, v in c["truth"].items()})
                    self.assertAlmostEqual(got, c["macro_f05"], places=6)
                else:
                    self.assertAlmostEqual(selector.macro_f05({"s": c["pred"]}, {"s": set(c["truth"])}), c["f05"], places=6)


if __name__ == "__main__":
    unittest.main()
