"""Leakage, weighting, numerical parity, and probability semantics checks."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from music_detector.calibration import (
    SCHEMA, SCOPE, binary_metrics, fit_platt, fit_ridge, load_dependencies,
    partition_metadata, predict_deployment, probabilities, ridge_scores, sample_weights,
)
from music_detector.scoring import FAMILIES, canonical_hash

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "current" / "audio_phenomena_expansion_20260907" / "code"


def fixture():
    rows, components = [], []
    for label in (0, 1):
        for index in range(30):
            uid = f"sample_{label}_{index}"
            row = dict(id=uid, label=str(label), source_group=f"source_{label}", role="development",
                group_id=f"group_{label}_{index}", component_id=uid)
            rows.append(row)
            components.append(dict(component_id=uid, members=[uid], link_tokens=[], protected_relationships=[]))
    # Different group IDs share a conditioning token through the full graph.
    components[0]["link_tokens"] = ["conditioning:shared_external_anchor"]
    components[31]["link_tokens"] = ["conditioning:shared_external_anchor"]
    # Non-development relationship protects an otherwise development sample.
    outside = dict(id="outside_locked", label="1", source_group="other", role="locked_test",
                   group_id="group_0_2", component_id="outside")
    components.append(dict(component_id="outside", members=["outside_locked"], link_tokens=[], protected_relationships=[]))
    return rows, dict(rows=rows + [outside], components=components)


class CalibrationTests(unittest.TestCase):
    def test_full_graph_closure_and_protected_dependencies(self):
        rows, screen = fixture()
        dependencies = load_dependencies(HISTORICAL / "plan_native30_evaluation_schedule_v1.py")
        split, excluded, graph = partition_metadata(rows, screen, dependencies)
        locations = {row["id"]: part for part, current in split.items() for row in current}
        self.assertEqual(locations["sample_0_0"], locations["sample_1_1"])
        self.assertEqual([r["id"] for r in excluded], ["sample_0_2"])
        self.assertEqual(sum(len(v) for v in split.values()), len(rows) - 1)
        again, _, _ = partition_metadata(list(reversed(rows)), screen, dependencies)
        self.assertEqual(split, again)
        for part, current in split.items():
            self.assertEqual({row["label"] for row in current}, {"0", "1"})

    def test_weights_preserve_class_source_group_balance(self):
        rows = [dict(label="0", source_group="human_a", group_id="shared"),
                dict(label="0", source_group="human_a", group_id="shared"),
                dict(label="0", source_group="human_a", group_id="solo"),
                dict(label="0", source_group="human_b", group_id="third"),
                dict(label="1", source_group="ai", group_id="shared")]
        weights = sample_weights(rows)
        self.assertAlmostEqual(weights.sum(), 5)
        self.assertAlmostEqual(weights[:4].sum(), weights[4])
        self.assertAlmostEqual(weights[:3].sum(), weights[3])
        self.assertAlmostEqual(weights[:2].sum(), weights[2])

    def test_new_ridge_reuses_historical_train_only_numerics(self):
        spec = importlib.util.spec_from_file_location("archived_ridge_core", HISTORICAL / "frozen_evaluate_expanded_20260905.py")
        historical = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(historical)
        rows, _ = fixture()
        columns = FAMILIES["D"]
        rng = np.random.default_rng(13)
        matrix = rng.normal(size=(len(rows), len(columns)))
        matrix[::3, 1] = np.nan
        table = pd.DataFrame(rows).rename(columns={"label": "__label", "source_group": "__source", "group_id": "__group"})
        table["__label"] = table["__label"].astype(int)
        for index, name in enumerate(columns):
            table[name] = matrix[:, index]
        old = historical.fit_model(table, columns)
        new = fit_ridge(rows, matrix, columns)
        for name in ("medians", "mean", "scale", "coefficients_with_intercept"):
            np.testing.assert_array_equal(new[name], old[name])
        np.testing.assert_allclose(ridge_scores(matrix, new), historical.predict(table, old), rtol=0, atol=1e-14)

    def test_ties_and_perfect_metrics(self):
        flat = binary_metrics([0, 1, 0, 1], [0.5] * 4)
        self.assertAlmostEqual(flat["brier_score"], 0.25)
        self.assertAlmostEqual(flat["logloss"], math.log(2))
        self.assertEqual(flat["ece"], 0)
        self.assertEqual(flat["auc"], 0.5)
        self.assertEqual(flat["balanced_accuracy"], 0.5)
        perfect = binary_metrics([0, 1, 0, 1], [0, 1, 0, 1])
        self.assertEqual(perfect["brier_score"], 0)
        self.assertEqual(perfect["auc"], 1)
        self.assertEqual(perfect["balanced_accuracy"], 1)

    def test_separate_deployment_schema_and_decision_scale(self):
        rows, _ = fixture()
        scores = np.linspace(-1, 2, len(rows))
        calibration = fit_platt(scores, rows)
        self.assertTrue(calibration["optimizer_success"])
        self.assertGreaterEqual(calibration["slope"], 0)
        self.assertTrue(np.all(np.diff(probabilities(scores, calibration)) >= 0))
        columns = FAMILIES["D"]
        matrix = np.column_stack([scores, scores**2, scores**3])
        model = fit_ridge(rows, matrix, columns)
        payload = dict(families=["D"], scope=SCOPE, protocol_sha256="a" * 64,
            schema_version="music-detector-deployment-model-v1", status="supported", external_validation=False,
            ridge_model=model, calibration=calibration,
            calibration_status="synthetic_schema_test_not_release", internal_test_metrics={},
            probability_interpretation="synthetic fixture")
        bundle = dict(schema_version=SCHEMA, scope=SCOPE, protocol_sha256="a" * 64,
                      models={"D": {"payload": payload, "sha256": canonical_hash(payload)}})
        result = predict_deployment(bundle, dict(zip(columns, matrix[0].tolist())), ["D"])
        self.assertEqual(result["decision_scale"], "new_deployment_calibrated_probability")
        self.assertEqual(result["historical_style_raw_decision"], int(result["raw_score"] >= 0.5))
        self.assertAlmostEqual(result["intercept"] + sum(result["family_contributions"].values()), result["raw_score"])
        with self.assertRaisesRegex(ValueError, "BC"):
            predict_deployment(bundle, {}, ["BC"])
        payload["ridge_model"]["scale"][0] = 0
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            predict_deployment(bundle, dict(zip(columns, matrix[0].tolist())), ["D"])


if __name__ == "__main__":
    unittest.main()
