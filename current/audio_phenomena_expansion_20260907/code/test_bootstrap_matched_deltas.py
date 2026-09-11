#!/usr/bin/env python3
"""Deterministic arithmetic tests for paired global-group uncertainty."""

import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import bootstrap_matched_deltas as subject


class WeightedMetricTests(unittest.TestCase):
    def test_known_weighted_auc(self):
        # Positive score 0.8 beats negative weights 1+2; positive 0.4 beats only
        # negative weight 1. Numerator = 3 + 1, denominator = 2 * 3.
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.6, 0.4, 0.8])
        weights = np.array([1.0, 2.0, 1.0, 1.0])
        self.assertAlmostEqual(subject.weighted_auc(labels, scores, weights), 4.0 / 6.0)

    def test_auc_ties_receive_half_credit(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.7, 0.7, 0.9])
        weights = np.array([2.0, 3.0, 5.0, 7.0])
        # Weighted numerator: 5*(2 + .5*3) + 7*(2+3) = 52.5;
        # denominator: (5+7)*(2+3) = 60.
        self.assertAlmostEqual(subject.weighted_auc(labels, scores, weights), 52.5 / 60.0)

    def test_integer_weights_equal_explicit_row_duplication(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.1, 0.8, 0.9])
        weights = np.array([3, 2, 1, 4])
        expanded_labels = np.repeat(labels, weights)
        expanded_scores = np.repeat(scores, weights)
        expected = subject.weighted_auc(
            expanded_labels, expanded_scores, np.ones(len(expanded_labels))
        )
        self.assertAlmostEqual(subject.weighted_auc(labels, scores, weights), expected)

    def test_zero_class_weight_is_explicitly_undefined(self):
        metric = subject.weighted_metrics(
            np.array([0, 1]), np.array([0.1, 0.9]), np.array([1.0, 0.0])
        )
        self.assertTrue(math.isnan(metric["roc_auc"]))
        self.assertTrue(math.isnan(metric["balanced_accuracy"]))
        self.assertEqual(metric["human_weight"], 1.0)
        self.assertEqual(metric["ai_weight"], 0.0)


class GlobalGroupBootstrapTests(unittest.TestCase):
    def test_draw_is_deterministic_and_preserves_draw_count(self):
        first = subject.draw_group_multiplicities(7, 20, 20260907)
        second = subject.draw_group_multiplicities(7, 20, 20260907)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first.sum(axis=1), np.full(20, 7))

    def test_cross_class_group_receives_one_shared_multiplicity(self):
        # group 0 occurs once in each class. With only group 0 resampled, both
        # classes remain represented; independently resampling within class
        # would violate this invariant.
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.8, 0.2])
        group_indices = np.array([0, 0, 1, 2])
        multiplicities = np.array([[3, 0, 0], [1, 2, 4]])
        metric = subject._replicate_metrics(
            labels, scores, group_indices, multiplicities
        )
        self.assertAlmostEqual(metric["roc_auc"][0], 1.0)
        self.assertAlmostEqual(metric["balanced_accuracy"][0], 1.0)
        self.assertEqual(metric["human_weight"][0], 3)
        self.assertEqual(metric["ai_weight"][0], 3)
        for replicate in range(2):
            explicit_weights = multiplicities[replicate, group_indices]
            expected = subject.weighted_metrics(labels, scores, explicit_weights)
            self.assertAlmostEqual(metric["roc_auc"][replicate], expected["roc_auc"])
            self.assertAlmostEqual(
                metric["balanced_accuracy"][replicate], expected["balanced_accuracy"]
            )

    def test_vectorized_metrics_equal_scalar_with_ties_and_group_duplicates(self):
        rng = np.random.default_rng(17)
        for case in range(20):
            labels = np.array([0, 1] + rng.integers(0, 2, 28).tolist())
            scores = np.round(rng.normal(size=30), 1)  # deliberately create ties
            group_indices = rng.integers(0, 11, size=30)
            multiplicities = subject.draw_group_multiplicities(11, 7, 1000 + case)
            observed = subject._replicate_metrics(
                labels, scores, group_indices, multiplicities
            )
            for replicate in range(len(multiplicities)):
                expected = subject.weighted_metrics(
                    labels, scores, multiplicities[replicate, group_indices]
                )
                for metric in (
                    "roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity"
                ):
                    left, right = observed[metric][replicate], expected[metric]
                    if math.isnan(left) and math.isnan(right):
                        continue
                    self.assertAlmostEqual(left, right, places=12)

    def test_macro_order_matches_pair_then_source_then_direction(self):
        cells = [
            {"row": {"fold_type": "human_source_holdout", "heldout_source": "h1"}},
            {"row": {"fold_type": "human_source_holdout", "heldout_source": "h1"}},
            {"row": {"fold_type": "human_source_holdout", "heldout_source": "h2"}},
            {"row": {"fold_type": "generator_holdout", "heldout_source": "a1"}},
            {"row": {"fold_type": "generator_holdout", "heldout_source": "a2"}},
        ]
        values = np.array([
            [0.2, 0.6, 0.9, 0.1, 0.5],
            [0.4, np.nan, 0.8, np.nan, 0.6],
        ])
        macro = subject.aggregate_macro(values, cells)
        # Replicate 0: human = mean(mean(.2,.6), .9)=.65;
        # generator=mean(.1,.5)=.3; equal=.475.
        self.assertAlmostEqual(macro["human_source_holdout"][0], 0.65)
        self.assertAlmostEqual(macro["generator_holdout"][0], 0.3)
        self.assertAlmostEqual(macro["equal_mean"][0], 0.475)
        # NaNs skip exactly as pandas groupby mean does in the evaluator.
        self.assertAlmostEqual(macro["human_source_holdout"][1], 0.6)
        self.assertAlmostEqual(macro["generator_holdout"][1], 0.6)
        self.assertEqual(macro["defined_pair_cells"][1], 3)

    def test_target_delta_reader_uses_added_combination_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deltas.csv"
            path.write_text(
                "comparison_id,baseline_combination,added_combination,quantity\n"
                "wanted,S+D+R,S+D+R+F,all\n"
                "wrong_quantity,S+D+R,S+D+R+H,25\n"
                "wrong_combo,S,S+F,all\n",
                encoding="utf-8",
            )
            rows = subject._read_target_deltas(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["comparison_id"], "wanted")


class AuthorizationIntegrationTests(unittest.TestCase):
    def make_manifest(self, root: Path, *, receipt_status: str = "frozen"):
        contract = {
            "schema_version": 2,
            "authorized_stage": "dev",
            "evaluator_sha256": "a" * 64,
            "preserved_base_evaluator_sha256": "b" * 64,
            "runtime": {"python": "3.11.15", "numpy": "1.26.4", "pandas": "2.3.3"},
            "families_json": {"path": "/frozen/families.json", "sha256": "c" * 64},
            "metadata": {"path": "/frozen/metadata.csv", "sha256": "d" * 64},
            "features": [{"path": "/frozen/features.csv", "sha256": "e" * 64}],
            "source_contract": {
                "source_column": "source_group", "global_group_column": "group_id"
            },
            "parameters": {
                "seed": 20260907, "quantities": [25, 50, 100, 200, "all"],
                "group_folds": 5, "model": "ridge", "threshold_policy": "fixed_0.5",
                "development_role": "development", "historical_roles": ["locked"],
                "min_train_groups_per_class": 5, "min_test_groups_per_class": 2,
            },
        }
        contract_hash = subject._evaluator_contract_hash(contract)
        receipt_path = root / "preregistration_draft.json"
        receipt_path.write_text(json.dumps({
            "schema_version": 2,
            "status": receipt_status,
            "authorized_stage": "dev",
            "contract_sha256": contract_hash,
            "contract": contract,
        }), encoding="utf-8")
        receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        return {
            "schema_version": 2,
            "stage": "dev",
            "authorization": {
                "status": "frozen_verified",
                "contract_sha256": contract_hash,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha256,
            },
            "contract": contract,
        }

    def test_frozen_verified_manifest_and_exact_frozen_receipt_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(Path(directory))
            verified = subject.verify_completed_evaluation_authorization(manifest)
            self.assertEqual(
                verified["contract_sha256"], manifest["authorization"]["contract_sha256"]
            )
            self.assertEqual(
                verified["receipt_sha256"], manifest["authorization"]["receipt_sha256"]
            )

    def test_bare_frozen_manifest_authorization_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(Path(directory))
            manifest["authorization"]["status"] = "frozen"
            with self.assertRaisesRegex(ValueError, "must be frozen_verified"):
                subject.verify_completed_evaluation_authorization(manifest)

    def test_draft_receipt_is_rejected_even_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(Path(directory), receipt_status="draft")
            with self.assertRaisesRegex(ValueError, "receipt itself is not frozen"):
                subject.verify_completed_evaluation_authorization(manifest)

    def test_receipt_sha_mismatch_is_rejected_before_status_trust(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(Path(directory))
            manifest["authorization"]["receipt_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "receipt bytes differ"):
                subject.verify_completed_evaluation_authorization(manifest)


if __name__ == "__main__":
    unittest.main()
