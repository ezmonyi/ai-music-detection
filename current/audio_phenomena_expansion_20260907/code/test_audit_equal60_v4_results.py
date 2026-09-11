#!/usr/bin/env python3
"""Isolated synthetic result audit and adversarial output regression tests."""
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import audit_equal60_v4_results as A
import evaluate_new_phenomena_v4 as E
import prepare_evaluation_inputs_v4 as P
from test_evaluate_new_phenomena_v4 import fixture, dump


class ResultAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        fixture(cls.root / "evidence")
        cls.package = cls.root / "package"
        P.prepare(cls.root / "evidence", cls.package, True)
        cls.original = cls.root / "original"
        with contextlib.redirect_stdout(io.StringIO()):
            E.main(["--stage", "dev", "--package-dir", str(cls.package), "--output-dir", str(cls.original), "--synthetic-test-only"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.run = self.root / self._testMethodName
        shutil.copytree(self.original, self.run)

    def rehash(self, name):
        manifest = A.strict_json(self.run / "run_manifest.json")
        manifest["files_sha256"][name] = P.sha(self.run / name)
        dump(self.run / "run_manifest.json", manifest)

    def change_csv(self, name, mutate):
        rows = A.read_csv(self.run / name)
        changed = mutate(rows)
        (rows if changed is None else changed).to_csv(self.run / name, index=False)
        self.rehash(name)

    def run_audit(self):
        return A.audit(self.run, self.package, synthetic=True)

    def test_00_complete_run_independent_no_metric_reuse_or_fit(self):
        with patch.object(E.V2, "metrics_at_threshold", side_effect=AssertionError("evaluator metric reused")), \
             patch.object(E.V2, "matched_metric_deltas", side_effect=AssertionError("evaluator deltas reused")), \
             patch.object(E.V2, "fit_candidate", side_effect=AssertionError("auditor fit")), \
             patch.object(np.linalg, "solve", side_effect=AssertionError("auditor solve")):
            report = self.run_audit()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["primary_predictions"], 15875)
        self.assertEqual(report["diagnostic_predictions"], 6350)
        self.assertEqual(report["matched_delta_rows"], 5250)
        self.assertEqual(report["source_pair_metrics"], 6350)
        self.assertEqual(report["pooled_metrics"], 3175)
        self.assertIsNone(report["source_transfer_J"])

    def test_metrics_ties_direction_threshold_and_single_class(self):
        m = A.metrics([0, 1, 0, 1], [.5, .5, 0, 1])
        self.assertEqual(m["roc_auc"], .875)
        self.assertEqual(m["balanced_accuracy"], .75)
        self.assertEqual((m["tp"], m["tn"], m["fp"], m["fn"]), (2, 1, 1, 0))
        self.assertEqual(A.metrics([0, 1], [1, 0])["roc_auc"], 0)
        self.assertTrue(np.isnan(A.metrics([0], [.5])["roc_auc"]))
        with self.assertRaises(ValueError):
            A.metrics([0, 1], [0, np.inf])

    def test_missing_file_and_unhashed_extra_forbidden(self):
        path = self.run / "development_group_cv_pooled_fold_metrics.csv"
        path.rename(self.run / "historical_predictions.csv")
        with self.assertRaisesRegex(ValueError, "Missing/extra"):
            self.run_audit()

    def test_unhashed_tamper_rejected(self):
        with (self.run / "fold_models.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "Result hash mismatch"):
            self.run_audit()

    def test_rehashed_duplicate_prediction_rejected(self):
        self.change_csv("development_group_cv_predictions.csv", lambda frame: pd.concat([frame.iloc[:-1], frame.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            self.run_audit()

    def test_rehashed_missing_diagnostic_rejected(self):
        self.change_csv("development_group_cv_diagnostic_predictions.csv", lambda frame: frame.iloc[:-1])
        # The complete baseline test exercises every model; this corruption targets grid completeness.
        with patch.object(A, "check_model"), self.assertRaisesRegex(ValueError, "Incomplete prediction grid"):
            self.run_audit()

    def test_rehashed_label_threshold_score_and_id_corruptions(self):
        name = "development_group_cv_predictions.csv"
        original = (self.run / name).read_bytes()
        for column, value, message in [("label", "9", "identity mismatch"), ("threshold", "0.6", "score/threshold"),
                                        ("score", "nan", "score/threshold"), ("predicted_label", "9", "threshold mismatch"),
                                        ("row_id", "synthetic_v4_row_25", "test IDs mismatch"), ("score", "-123.5", "threshold mismatch|replay mismatch")]:
            with self.subTest(column=column, value=value):
                (self.run / name).write_bytes(original)
                self.change_csv(name, lambda f: f.__setitem__(column, f[column].mask(f.index == 0, value)))
                with self.assertRaisesRegex(ValueError, message):
                    self.run_audit()

    def test_rehashed_metrics_and_deltas_corruptions(self):
        cases = [("development_group_cv_pooled_fold_metrics.csv", "roc_auc", "pooled metrics"),
                 ("development_group_cv_metrics_by_source_pair.csv", "balanced_accuracy", "source-pair metrics"),
                 ("development_group_cv_diagnostic_metrics.csv", "tp", "diagnostic metrics"),
                 ("development_group_cv_matched_deltas.csv", "delta_roc_auc__added_minus_baseline", "matched deltas")]
        for name, column, message in cases:
            with self.subTest(name=name):
                original = (self.run / name).read_bytes()
                self.change_csv(name, lambda f: f.__setitem__(column, f[column].mask(f.index == 0, "123.5")))
                with patch.object(A, "check_model"), self.assertRaisesRegex(ValueError, message):
                    self.run_audit()
                (self.run / name).write_bytes(original)
                self.rehash(name)

    def test_rehashed_fold_cap_and_cross_role_corruptions(self):
        path = self.run / "fold_registry.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["train_ids"].append("synthetic_v4_row_25")
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        self.rehash(path.name)
        with self.assertRaisesRegex(ValueError, "Fold/cap schedule mismatch"):
            self.run_audit()

    def test_model_hash_and_predictor_restrictions(self):
        models = A.strict_json(self.run / "fold_models.json")
        first = next(iter(models))
        models[first]["ridge"] = 0.0
        dump(self.run / "fold_models.json", models)
        self.rehash("fold_models.json")
        with self.assertRaisesRegex(ValueError, "Model hash mismatch"):
            self.run_audit()
        meta = A.read_csv(self.package / "metadata_60s.csv").set_index("id")
        meta["label"] = meta.label.astype(int)
        features = A.read_csv(self.package / "features_60s.csv").set_index("id").apply(pd.to_numeric, errors="coerce")
        _, train, _ = A.schedule(meta)[0]
        original = A.strict_json(self.original / "fold_models.json")[first]
        for key, value in [("ridge", 0), ("columns", ["status"]), ("threshold", .7),
                           ("medians", [1e8] * len(original["medians"])),
                           ("coefficients_with_intercept", [1e8] * len(original["coefficients_with_intercept"]))]:
            with self.subTest(key=key):
                modified = copy.deepcopy(original)
                modified[key] = value
                with self.assertRaises(ValueError):
                    A.check_model(modified, train, features, original["columns"], original["feature_mode"])

    def test_stale_contract_and_nonnull_j(self):
        path = self.run / "run_manifest.json"
        original = A.strict_json(path)
        for mutate in [lambda m: m.__setitem__("source_transfer_J", 0.0),
                       lambda m: m["contract"]["code_sha256"].__setitem__("evaluate_new_phenomena_v4.py", "0" * 64),
                       lambda m: m["contract"]["parameters"].__setitem__("threshold", .7),
                       lambda m: m["authorization"].__setitem__("status", "draft")]:
            current = copy.deepcopy(original)
            mutate(current)
            dump(path, current)
            with self.assertRaises(ValueError):
                self.run_audit()

    def test_unfrozen_stale_receipt_real_gate(self):
        # Mock only evidence admission to exercise real authorization with synthetic bytes.
        path = self.run / "run_manifest.json"
        manifest = A.strict_json(path)
        manifest["contract"]["synthetic_test_only"] = False
        receipt = dict(schema_version=4, status="draft", authorized_stage="dev", contract=manifest["contract"], contract_sha256=A.digest(manifest["contract"]))
        receipt_path = self.root / "receipt.json"
        dump(receipt_path, receipt)
        manifest["authorization"] = dict(status="frozen_verified", contract_sha256=A.digest(manifest["contract"]), receipt_sha256=P.sha(receipt_path))
        dump(path, manifest)
        proof = P.validate_package(self.package, True)
        with patch.object(P, "validate_package", return_value=proof):
            with self.assertRaisesRegex(ValueError, "Frozen receipt required"):
                A.verify_contract(self.run, self.package, False, None)
            with self.assertRaisesRegex(ValueError, "not frozen"):
                A.verify_contract(self.run, self.package, False, receipt_path)
            receipt["status"] = "frozen"
            receipt["contract"] = copy.deepcopy(receipt["contract"])
            receipt["contract"]["parameters"]["ridge"] = 11
            dump(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "Stale frozen receipt"):
                A.verify_contract(self.run, self.package, False, receipt_path)

    def test_csv_and_json_duplicate_malformed_rejected(self):
        path = self.root / "malformed.csv"
        for text in ["a,a\n1,2\n", "a,b\n1\n", "a,b\n1,2,3\n"]:
            path.write_text(text)
            with self.assertRaises(ValueError):
                A.read_csv(path)
        path = self.root / "malformed.json"
        for text in ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}']:
            path.write_text(text)
            with self.assertRaises(ValueError):
                A.strict_json(path)

    def test_cap_truncation_and_shared_global_fold(self):
        rows = [dict(id=f"synthetic_v4_cap_{label}_{i}", label=label, source_group="Suno" if label else "human",
                     group_id=f"synthetic_v4_cap_group_{i}", role="development") for label in (0, 1) for i in range(150)]
        meta = pd.DataFrame(rows).set_index("id")
        for record, train, test in A.schedule(meta):
            self.assertFalse(set(train.group_id) & set(test.group_id))
            for _, source in train.groupby("source_group"):
                self.assertEqual(source.group_id.nunique(), min(record["quantity"], 150 - test.group_id.nunique()) if record["quantity"] != "all" else 150 - test.group_id.nunique())
            # Both labels of a shared group have the same global fold and cap ranking.
            self.assertTrue(train.groupby("group_id").size().eq(2).all())
            self.assertTrue(test.groupby("group_id").size().eq(2).all())

    def test_all_missing_zero_fallback_and_unequal_group_weights(self):
        meta = A.read_csv(self.package / "metadata_60s.csv").set_index("id")
        meta["label"] = meta.label.astype(int)
        _, train, _ = A.schedule(meta)[0]
        features = pd.DataFrame({"synthetic_feature": np.nan}, index=meta.index)
        table = train.rename(columns={"label": "__label", "source_group": "__source", "group_id": "__group"}).copy()
        table["synthetic_feature"] = np.nan
        # A synthetic fit supplies a valid saved model; audit itself never fits.
        for mode in A.MODES:
            model = E.V2.fit_candidate(table, ["synthetic_feature"], "ridge", feature_mode=mode)
            model["missing_value_policy"] = A.POLICIES[mode]
            self.assertEqual(model["medians"], [0.0])
            A.check_model(model, train, features, ["synthetic_feature"], mode)
        np.testing.assert_allclose(A.training_weights(train), E.BASE.sample_weights(table), rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
