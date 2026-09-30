"""Boundary tests and independent replay of archived research predictions."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

from music_detector.scoring import (
    FAMILIES, SCOPE_KEYS, canonical_hash, load_model_receipt, score_features,
)


def synthetic_receipt(families="D", mode="values_plus_missing"):
    """Deliberately artificial arithmetic fixture, never a trained detector."""
    columns = [name for family in families.split("+") for name in FAMILIES[family]]
    count = len(columns)
    dimensions = count * (2 if mode == "values_plus_missing" else 1)
    model = dict(columns=columns, medians=[2.0] * count, mean=[1.0] * dimensions,
        scale=[2.0] * dimensions, coefficients_with_intercept=[0.5] + [1.0] * dimensions,
        ridge=10.0, threshold=0.5, feature_mode=mode, training_rows=2,
        training_source_counts={"0:synthetic_human": 1, "1:synthetic_ai": 1},
        weighting="classes equal; sources equal within class; groups equal within source; samples equal within group",
        missing_value_policy="training-only median imputation plus missingness indicators",
        observed_fraction_by_column={name: 0.5 for name in columns})
    payload = dict(status="fold_only_model_and_predictions_not_selected", contract_sha256="a" * 64,
        model=model, task={"combination": families, "feature_mode": mode}, predictions=[],
        metrics={}, training_numerical_audit={}, **{key: False for key in SCOPE_KEYS})
    return resign(payload)


def resign(payload):
    for key in ("model", "task", "predictions"):
        payload[key + "_sha256"] = canonical_hash(payload[key])
    return {"payload": payload, "receipt_sha256": canonical_hash(payload)}


class ScoringTests(unittest.TestCase):
    def test_hand_computed_imputation_missingness_and_decomposition(self):
        receipt = synthetic_receipt()
        names = FAMILIES["D"]
        # Values [3,2,5] => [1,.5,2]; masks [0,1,0] => [-.5,0,-.5].
        result = score_features(receipt, dict(zip(names, [3.0, None, 5.0])))
        self.assertEqual(result["raw_score"], 3.0)
        self.assertEqual(result["predicted_label"], 1)
        self.assertEqual(result["missing_features"], [names[1]])
        self.assertEqual(len(result["contributions"]), 6)
        self.assertEqual(result["intercept"] + result["family_contributions"]["D"], 3.0)
        self.assertIsNone(result["ai_probability"])
        self.assertFalse(result["research_only"])

    def test_unbounded_score_is_not_clipped_or_called_probability(self):
        result = score_features(synthetic_receipt(), {name: -100.0 for name in FAMILIES["D"]})
        self.assertLess(result["raw_score"], 0)
        self.assertEqual(result["predicted_label"], 0)
        self.assertIsNone(result["ai_probability"])

    def test_threshold_equality_is_positive(self):
        receipt = synthetic_receipt()
        receipt["payload"]["model"]["coefficients_with_intercept"] = [0.5] + [0.0] * 6
        result = score_features(resign(receipt["payload"]), {name: 1.0 for name in FAMILIES["D"]})
        self.assertEqual((result["raw_score"], result["predicted_label"]), (0.5, 1))

    def test_null_and_nan_share_historical_missingness(self):
        receipt = synthetic_receipt()
        null = score_features(receipt, {name: None for name in FAMILIES["D"]})
        nan = score_features(receipt, {name: math.nan for name in FAMILIES["D"]})
        self.assertEqual(null, nan)

    def test_invalid_measurements_and_column_failures(self):
        receipt = synthetic_receipt()
        baseline = {name: 1.0 for name in FAMILIES["D"]}
        for value in (math.inf, -math.inf, True, "1", [1]):
            features = dict(baseline)
            features[FAMILIES["D"][0]] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                score_features(receipt, features)
        for features in ({}, {**baseline, "id": "not_a_feature"}):
            with self.assertRaises(ValueError):
                score_features(receipt, features)

    def test_cannot_select_subset_by_discarding_coefficients(self):
        receipt = synthetic_receipt("D+R")
        features = {name: 1.0 for name in receipt["payload"]["model"]["columns"]}
        with self.assertRaisesRegex(ValueError, "do not drop coefficients"):
            score_features(receipt, features, families="D")

    def test_receipt_tampering_and_noncanonical_parameters(self):
        receipt = synthetic_receipt()
        receipt["payload"]["model"]["threshold"] = 0.2
        features = {name: 1.0 for name in FAMILIES["D"]}
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            score_features(receipt, features)
        with self.assertRaisesRegex(ValueError, "threshold changed"):
            score_features(resign(receipt["payload"]), features)
        for change in (lambda m: m["scale"].__setitem__(0, 0),
                       lambda m: m["columns"].reverse(),
                       lambda m: m.__setitem__("unexpected_field", 1)):
            changed = synthetic_receipt()
            change(changed["payload"]["model"])
            with self.assertRaises(ValueError):
                score_features(resign(changed["payload"]), features)

    def test_bc_and_missingness_only_remain_research(self):
        for families, mode in (("BC", "values_plus_missing"), ("D", "missingness_only"), ("D", "median_only")):
            receipt = synthetic_receipt(families, mode)
            features = {name: 3.0 for name in receipt["payload"]["model"]["columns"]}
            with self.assertRaisesRegex(ValueError, "research opt-in"):
                score_features(receipt, features)
            result = score_features(receipt, features, allow_research=True)
            self.assertTrue(result["research_only"])
            self.assertIsNone(result["ai_probability"])

    def test_calibration_requires_separate_trust_and_validation(self):
        receipt = synthetic_receipt()
        features = {name: 1.0 for name in FAMILIES["D"]}
        calibration = dict(schema_version="music-detector-calibration-v1", method="platt",
            model_sha256=receipt["payload"]["model_sha256"], families=["D"],
            feature_mode="values_plus_missing", slope=1.0, intercept=0.0,
            validation=dict(status="accepted", report_sha256="b" * 64, fit_role="development",
                validation_role="development", group_disjoint=True, both_classes=True,
                locked_labels_used=False, brier_score=0.2, ece=0.1, rows=20))
        # This is an artificial schema fixture, not real calibration evidence.
        with self.assertRaisesRegex(ValueError, "trusted release hash"):
            score_features(receipt, features, calibration=calibration)
        result = score_features(receipt, features, calibration=calibration,
            expected_calibration_sha256=canonical_hash(calibration))
        self.assertAlmostEqual(result["ai_probability"], 1 / (1 + math.exp(-result["raw_score"])))
        for key, value in (("locked_labels_used", True), ("group_disjoint", False), ("both_classes", False)):
            changed = copy.deepcopy(calibration)
            changed["validation"][key] = value
            with self.assertRaisesRegex(ValueError, "development-only"):
                score_features(receipt, features, calibration=changed,
                    expected_calibration_sha256=canonical_hash(changed))
        changed = {**calibration, "model_sha256": "c" * 64}
        with self.assertRaisesRegex(ValueError, "different model"):
            score_features(receipt, features, calibration=changed,
                expected_calibration_sha256=canonical_hash(changed))


class ArchivedReplayTests(unittest.TestCase):
    def test_persisted_historical_predictions(self):
        root = Path(__file__).parent / "fixtures"
        fixture = json.loads((root / "native30_replay.json").read_text())
        for archived in fixture["receipts"]:
            receipt = load_model_receipt(root / archived["file"], expected_sha256=archived["sha256"], allow_research=True)
            predictions = {row["id"]: row for row in receipt["payload"]["predictions"]}
            columns = receipt["payload"]["model"]["columns"]
            for row in archived["rows"]:
                with self.subTest(receipt=archived["file"], id=row["id"]):
                    result = score_features(receipt, {name: row["features"][name] for name in columns}, allow_research=True)
                    self.assertLessEqual(abs(result["raw_score"] - predictions[row["id"]]["score"]), 1e-12)
                    self.assertEqual(result["predicted_label"], predictions[row["id"]]["predicted_label"])
                    self.assertIsNone(result["ai_probability"])
                    self.assertAlmostEqual(math.fsum([result["intercept"], *[v["contribution"] for v in result["contributions"]]]), result["raw_score"], places=14)


if __name__ == "__main__":
    unittest.main()
