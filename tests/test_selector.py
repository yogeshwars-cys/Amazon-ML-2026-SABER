import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import selector


def candidate_frame(rows):
    return pl.DataFrame(rows).with_columns(
        pl.lit("US").alias("country"),
        pl.lit(0).cast(pl.Int8).alias("is_s3"),
    )


def contract_frame():
    data = {
        "source1_entity_id": ["a"], "candidate_entity_id": ["x"],
        "country": ["US"], "cand_source": ["S2"],
        "cos_v": [0.8], "cos_t": [0.7],
        "in_vq": [True], "in_vr": [False], "in_tq": [True],
        "in_tr": [False], "in_a": [False], "in_n": [False],
        "rank_vq": [0], "rank_vr": [99], "rank_tq": [3], "rank_tr": [99],
        "n_cand_s1": [1], "n_cand_r": [1],
        "cos_v_rank_in_s1": [1], "cos_v_rank_in_r": [1],
        "label": [True],
    }
    return pl.DataFrame(data)


class CandidateContractTests(unittest.TestCase):
    def test_rank_conventions_are_distinct(self):
        self.assertEqual(selector.validate_candidate_frame(contract_frame(), require_label=True).height, 1)

        retrieval_bad = contract_frame().with_columns(pl.lit(-1).alias("rank_vq"))
        with self.assertRaisesRegex(selector.CandidateContractError, "zero based"):
            selector.validate_candidate_frame(retrieval_bad)

        competition_bad = contract_frame().with_columns(pl.lit(0).alias("cos_v_rank_in_r"))
        with self.assertRaisesRegex(selector.CandidateContractError, "one based"):
            selector.validate_candidate_frame(competition_bad)

    def test_duplicate_edges_are_rejected(self):
        with self.assertRaisesRegex(selector.CandidateContractError, "duplicate"):
            selector.validate_candidate_frame(pl.concat([contract_frame(), contract_frame()]))

    def test_null_required_values_are_rejected(self):
        bad = contract_frame().with_columns(pl.lit(None).cast(pl.String).alias("cand_source"))
        with self.assertRaisesRegex(selector.CandidateContractError, "null value"):
            selector.validate_candidate_frame(bad)

    def test_duplicate_edges_across_shards_are_rejected(self):
        with patch.object(selector.pl, "scan_parquet", return_value=contract_frame().lazy()):
            with self.assertRaisesRegex(selector.CandidateContractError, "across blocker shards"):
                selector.validate_candidate_parts(["part_0.parquet", "part_1.parquet"], require_label=True)


class CompetitionFeatureTests(unittest.TestCase):
    def test_competitor_outside_evaluation_roster_is_visible(self):
        df = candidate_frame({
            "source1_entity_id": ["eval", "train"],
            "candidate_entity_id": ["child", "child"],
        })
        ctx = selector.ctx_features(df, np.array([0.8, 0.7], dtype=np.float32))
        self.assertAlmostEqual(ctx["p_best_other_s1"][0], 0.7, places=6)
        self.assertAlmostEqual(ctx["p_best_other_s1"][1], 0.8, places=6)
        self.assertEqual(ctx["n_s1_r"].to_list(), [2, 2])

    def test_same_child_id_does_not_compete_across_countries(self):
        df = pl.DataFrame({
            "source1_entity_id": ["us", "in"],
            "candidate_entity_id": ["child", "child"],
            "country": ["US", "India"],
            "is_s3": [0, 0],
        })
        ctx = selector.ctx_features(df, np.array([0.8, 0.9], dtype=np.float32))
        self.assertEqual(ctx["n_s1_r"].to_list(), [1, 1])
        self.assertEqual(ctx["p_best_other_s1"].to_list(), [0.0, 0.0])


class DecoderTests(unittest.TestCase):
    def test_global_owner_can_be_outside_output_roster(self):
        df = candidate_frame({
            "source1_entity_id": ["eval", "other"],
            "candidate_entity_id": ["child", "child"],
        })
        pred = selector.select_sets(
            df, [0.8, 0.9], anchor_ids=["eval"], p_has_match={"eval": 0.99}
        )
        self.assertEqual(pred, {"eval": []})

    def test_zero_candidate_anchor_is_preserved(self):
        df = candidate_frame({
            "source1_entity_id": ["a"], "candidate_entity_id": ["x"],
        })
        pred = selector.select_sets(df, [0.9], anchor_ids=["a", "singleton"])
        self.assertEqual(pred["singleton"], [])

    def test_anchor_probability_controls_empty_set(self):
        df = candidate_frame({
            "source1_entity_id": ["a", "a"],
            "candidate_entity_id": ["x", "y"],
        })
        pred = selector.select_sets(df, [0.55, 0.54], p_has_match={"a": 0.1})
        self.assertEqual(pred["a"], [])
        pred = selector.select_sets(df, [0.55, 0.54], p_has_match={"a": 0.99})
        self.assertEqual(pred["a"], ["x", "y"])

    def test_empty_candidate_universe_preserves_roster(self):
        empty = pl.DataFrame(schema={
            "source1_entity_id": pl.String,
            "candidate_entity_id": pl.String,
            "country": pl.String,
        })
        self.assertEqual(selector.select_sets(empty, [], anchor_ids=["a", "b"]), {"a": [], "b": []})

    def test_tied_child_owner_is_deterministic(self):
        df = candidate_frame({
            "source1_entity_id": ["b", "a"],
            "candidate_entity_id": ["x", "x"],
        })
        pred = selector.select_sets(
            df, [0.9, 0.9], anchor_ids=["a", "b"],
            p_has_match={"a": 0.99, "b": 0.99},
        )
        self.assertEqual(pred["a"], ["x"])
        self.assertEqual(pred["b"], [])

    def test_macro_f05_includes_singletons(self):
        pred = {"a": ["x"], "b": []}
        truth = {"a": {"x", "y"}, "b": set()}
        expected_a = 1.25 / (0.25 * 2 + 1)
        self.assertAlmostEqual(selector.macro_f05(pred, truth), (expected_a + 1) / 2)


if __name__ == "__main__":
    unittest.main()
