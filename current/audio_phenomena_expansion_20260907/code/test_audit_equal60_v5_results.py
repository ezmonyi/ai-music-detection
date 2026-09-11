#!/usr/bin/env python3
"""Synthetic-only tests for the independent exploratory-v5 auditor."""
from __future__ import annotations

from collections import Counter
import copy
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import audit_equal60_v5_results as A


def group_for_bucket(bucket: int, tag: str) -> str:
    index = 0
    while True:
        value = f"{tag}_{index}"
        if A.hash_fold(value) == bucket:
            return value
        index += 1


def scheduling_table() -> pd.DataFrame:
    rows = []
    for label, sources in ((0, ("h1", "h2")), (1, ("a1", "a2"))):
        for source in sources:
            for bucket in range(A.FOLDS):
                for repeat in range(2):
                    group = group_for_bucket(bucket, f"{source}_{bucket}_{repeat}")
                    rows.append({"__id": f"id_{source}_{bucket}_{repeat}", "__label": label,
                                 "__source": source, "__group": group, "__role": "development"})
    return pd.DataFrame(rows)


def training_table() -> pd.DataFrame:
    rows = []
    sources = ((0, "h1"), (0, "h2"), (1, "a1"), (1, "a2"))
    for number, (label, source) in enumerate(sources * 3):
        row = {"__id": f"tr{number}", "__label": label, "__source": source,
               "__group": f"g{number // 2}", "__role": "development"}
        for index, column in enumerate(A.DESCRIPTORS):
            row[column] = float(number + index / 100.0)
        rows.append(row)
    rows[1][A.DESCRIPTORS[0]] = math.nan
    rows[3][A.DESCRIPTORS[1]] = math.nan
    return pd.DataFrame(rows)


def synthetic_model(cache: A.TrainingCache, columns: list[str], mode: str) -> dict:
    indices, x, mean, scale, standardized = cache.arrays(columns, mode)
    design = np.column_stack((np.ones(len(cache.train)), standardized))
    penalty = np.eye(design.shape[1]) * 10.0; penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ (design * cache.weights[:, None]) + penalty,
                           design.T @ (cache.weights * cache.y))
    return {
        "columns": columns, "medians": cache.medians[indices].tolist(), "mean": mean.tolist(),
        "scale": scale.tolist(), "coefficients_with_intercept": coef.tolist(), "ridge": 10.0,
        "threshold": 0.5, "feature_mode": mode, "training_rows": len(cache.train),
        "training_source_counts": dict(sorted(cache.source_counts.items())),
        "weighting": "classes equal; sources equal within class; groups equal within source; samples equal within group",
        "missing_value_policy": A.POLICIES[mode],
        "observed_fraction_by_column": {name: float((~cache.missing[:, cache.positions[name]]).mean()) for name in columns},
        "model_type": "weighted_ridge_linear_probability", "prediction_link": "identity",
    }


def write_csv(path: Path, fields: list[str], rows: list[list[str]], final_newline: bool = True) -> tuple[str, int]:
    text = ",".join(fields) + "\n" + "".join(",".join(row) + "\n" for row in rows)
    if not final_newline:
        text = text[:-1]
    path.write_bytes(text.encode())
    return A.sha256_file(path), path.stat().st_size


class GridAndScheduleTests(unittest.TestCase):
    def test_fixed_family_and_complete_grid(self):
        self.assertEqual([len(A.FAMILIES[key]) for key in A.FAMILY_ORDER], [15, 3, 3, 6, 15, 6, 6])
        self.assertEqual(len(A.DESCRIPTORS), 54)
        self.assertEqual(len(A.COMBINATIONS), 127)
        self.assertEqual(A.COMBINATIONS[:7], list(A.FAMILY_ORDER))
        self.assertEqual(A.COMBINATIONS[-1], "S+D+R+P+F+H+M")

    def test_schedule_membership_caps_and_tamper(self):
        schedule, omitted, accounting = A.build_schedule(scheduling_table())
        self.assertEqual(len(schedule), 25 * 5)
        self.assertEqual(omitted, [])
        self.assertEqual(accounting["valid_folds"], 25)
        expected = [record for record, *_ in schedule]
        A.validate_schedule_records(copy.deepcopy(expected), expected)
        tampered = copy.deepcopy(expected)
        tampered[0]["train"]["ids"][0] = "not_a_training_id"
        with self.assertRaisesRegex(ValueError, "membership/order"):
            A.validate_schedule_records(tampered, expected)
        with self.assertRaisesRegex(ValueError, "row count"):
            A.validate_schedule_records(expected[:-1], expected)

    def test_train_test_and_source_holdout_contract(self):
        schedule, _, _ = A.build_schedule(scheduling_table())
        for record, train, test, _ in schedule:
            self.assertFalse(set(train["__id"]) & set(test["__id"]))
            self.assertFalse(set(train["__group"]) & set(test["__group"]))
            if record["heldout_source"] != "__all_sources__":
                self.assertNotIn(record["heldout_source"], set(train["__source"]))
        first = copy.deepcopy(schedule[0][0])
        first["test_id_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "membership/order"):
            A.validate_schedule_records([first], [schedule[0][0]])


class ModelReplayTests(unittest.TestCase):
    def setUp(self):
        self.train = training_table()
        self.cache = A.TrainingCache(self.train)
        self.columns = A.DESCRIPTORS[:3]

    def test_valid_transforms_stationarity_and_replay(self):
        for mode in A.MODES:
            model = synthetic_model(self.cache, self.columns, mode)
            _, residual = A.validate_model(model, self.cache, self.columns, mode)
            self.assertLess(residual, 1e-8)
            scores = A.replay_scores(self.train, self.columns, mode, model)
            self.assertTrue(np.isfinite(scores).all())

    def test_tampered_training_transforms(self):
        base = synthetic_model(self.cache, self.columns, A.MODES[0])
        for field in ("medians", "mean", "scale"):
            current = copy.deepcopy(base); current[field][0] += 0.01
            with self.subTest(field=field), self.assertRaises(ValueError):
                A.validate_model(current, self.cache, self.columns, A.MODES[0])
        current = copy.deepcopy(base); current["training_source_counts"]["0:h1"] += 1
        with self.assertRaisesRegex(ValueError, "source counts"):
            A.validate_model(current, self.cache, self.columns, A.MODES[0])

    def test_tampered_coefficients_and_model_hash(self):
        base = synthetic_model(self.cache, self.columns, A.MODES[0])
        current = copy.deepcopy(base); current["coefficients_with_intercept"][1] += 0.01
        with self.assertRaisesRegex(ValueError, "normal-equation"):
            A.validate_model(current, self.cache, self.columns, A.MODES[0])
        record = {"quantity": "all", "fold_uid": "f", "fold_index": 0,
                  "fold_type": "ordinary_group_holdout_descriptive", "heldout_source": "__all_sources__",
                  "opposite_group_fold": 0, "train_id_set_sha256": "a" * 64, "test_id_set_sha256": "b" * 64,
                  "train": {"rows": len(self.train)}, "test": {"rows": 2}}
        common = A.expected_common(record, "S", A.MODES[0], 1, base)
        self.assertEqual(common["model_sha256"], A.digest({"model": base, "threshold": 0.5}))
        tampered = dict(common); tampered["model_sha256"] = "0" * 64
        self.assertNotEqual(tampered, A.expected_common(record, "S", A.MODES[0], 1, base))

    def test_prediction_label_score_threshold_and_decision_tamper(self):
        identity = {"model_uid": "u", "row_id": "r", "label": 1, "source_group": "a",
                    "group_id": "g", "role": "development"}
        valid = {**{key: str(value) for key, value in identity.items()}, "score": "0.75",
                 "threshold": "0.5", "predicted_label": "1"}
        self.assertEqual(A.validate_prediction_row(valid, identity, 0.75, "valid"), (0.75, 1))
        changes = {"label": "0", "score": "0.70", "threshold": "0.4", "predicted_label": "0"}
        for field, value in changes.items():
            row = dict(valid); row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                A.validate_prediction_row(row, identity, 0.75, field)

    def test_near_threshold_exact_decision_regression(self):
        identity = {"model_uid": "u", "row_id": "r", "label": 1, "source_group": "a",
                    "group_id": "g", "role": "development"}
        row = {**{key: str(value) for key, value in identity.items()}, "score": "0.5",
               "threshold": "0.5", "predicted_label": "1"}
        with self.assertRaisesRegex(ValueError, "decision"):
            A.validate_prediction_row(row, identity, float(np.nextafter(0.5, -math.inf)), "near threshold")


class StreamTests(unittest.TestCase):
    def test_order_duplicate_extra_and_truncated_streams(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fields = ["id", "value"]
            path = root / "valid.csv"; sha, size = write_csv(path, fields, [["a", "1"], ["b", "2"]])
            stream = A.StrictCSVStream(path, fields, sha, size)
            A.compare_record(stream.next("first"), {"id": "a", "value": 1}, integer={"value"}, context="first")
            A.compare_record(stream.next("second"), {"id": "b", "value": 2}, integer={"value"}, context="second")
            self.assertEqual(stream.finish(), sha)

            duplicate = root / "duplicate.csv"; sha, size = write_csv(duplicate, fields, [["a", "1"], ["a", "1"]])
            stream = A.StrictCSVStream(duplicate, fields, sha, size)
            stream.next("first")
            with self.assertRaisesRegex(ValueError, "Identity/order"):
                A.compare_record(stream.next("expected b"), {"id": "b", "value": 2}, integer={"value"}, context="expected b")
            stream.close()

            extra = root / "extra.csv"; sha, size = write_csv(extra, fields, [["a", "1"], ["b", "2"]])
            stream = A.StrictCSVStream(extra, fields, sha, size); stream.next("only expected")
            with self.assertRaisesRegex(ValueError, "extra"):
                stream.finish()
            stream.close()

            truncated = root / "truncated.csv"; sha, size = write_csv(truncated, fields, [["a", "1"]], final_newline=False)
            stream = A.StrictCSVStream(truncated, fields, sha, size)
            with self.assertRaisesRegex(ValueError, "Truncated"):
                stream.next("row")
            stream.close()

    def test_constructor_schema_failure_closes_handle(self):
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            path = Path(folder) / "bad.csv"; sha, size = write_csv(path, ["wrong"], [["x"]])
            with self.assertRaisesRegex(ValueError, "schema"):
                A.StrictCSVStream(path, ["right"], sha, size)
            gc.collect()

    def test_jsonl_duplicate_truncation_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "items.jsonl"
            path.write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")
            stream = A.StrictJSONLStream(path, A.sha256_file(path), path.stat().st_size)
            self.assertEqual(stream.next("one"), {"x": 1})
            with self.assertRaisesRegex(ValueError, "Unexpected/duplicate"):
                stream.finish()
            stream.close()
            path.write_text('{"x":1}', encoding="utf-8")
            stream = A.StrictJSONLStream(path, A.sha256_file(path), path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "Truncated"):
                stream.next("one")
            stream.close()


class MetricAndOOFTests(unittest.TestCase):
    def test_auc_counts_and_per_source_component_denominators(self):
        metrics = A.independent_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (2, 1, 1, 0))
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        test = pd.DataFrame({"__id": ["a", "b", "c", "d"], "__label": [0] * 4,
                             "__source": ["h"] * 4, "__group": ["g1", "g1", "g1", "g2"]})
        row = A.expected_source_rows(test, np.asarray([0.6, 0.4, 0.4, 0.6]), "u")[0]
        self.assertAlmostEqual(row["recording_positive_rate"], 0.5)
        self.assertAlmostEqual(row["equal_component_positive_rate"], (1 / 3 + 1) / 2)
        self.assertNotAlmostEqual(row["recording_positive_rate"], row["equal_component_positive_rate"])

    def test_oof_global_component_not_fold_component_mean(self):
        def frame(ids, groups):
            return pd.DataFrame({"__id": ids, "__label": [0] * len(ids), "__source": ["h"] * len(ids),
                                 "__group": groups, "__role": ["development"] * len(ids)})
        first = frame(["a", "b", "c"], ["g1", "g1", "g2"])
        second = frame(["d", "e", "f"], ["g3", "g3", "g3"])
        records = []
        for index, test in enumerate((first, second)):
            record = {"quantity": "all", "fold_type": "human_source_holdout", "heldout_source": "h",
                      "fold_uid": f"f{index}"}
            records.append((record, pd.DataFrame(), test, {}))
        oof = A.OOFAccumulator(records)
        oof.add("primary", "S", A.MODES[0], "all", records[0][0], first, np.asarray([0, 1, 0]))
        oof.add("primary", "S", A.MODES[0], "all", records[1][0], second, np.asarray([1, 1, 1]))
        endpoint = list(oof.iter_records())[0]
        self.assertEqual(endpoint["rows"], 6); self.assertEqual(endpoint["components"], 3)
        self.assertAlmostEqual(endpoint["recording_correct_rate"], 2 / 6)
        self.assertAlmostEqual(endpoint["equal_global_component_correct_rate"], (0.5 + 1 + 0) / 3)
        # Fold component means would be ((.5+1)/2 + 0)/2=.375, and is forbidden.
        self.assertNotAlmostEqual(endpoint["equal_global_component_correct_rate"], 0.375)

    def test_oof_duplicate_membership_refused(self):
        test = pd.DataFrame({"__id": ["a"], "__label": [0], "__source": ["h"], "__group": ["g"],
                             "__role": ["development"]})
        record = {"quantity": "all", "fold_type": "human_source_holdout", "heldout_source": "h"}
        with self.assertRaisesRegex(ValueError, "Duplicate OOF"):
            A.OOFAccumulator([(record, pd.DataFrame(), test, {}), (record, pd.DataFrame(), test, {})])


class ReceiptAndPinTests(unittest.TestCase):
    def test_frozen_receipt_and_review_tamper(self):
        contract = {"schema_version": 5, "value": 1}
        receipt = {"status": "frozen", "authorized_stage": A.STAGE, "contract": contract,
                   "contract_sha256": A.digest(contract),
                   "independent_review": {"approved": True, "reviewer": "root", "reviewed_utc": "2026-09-07T00:00:00Z"}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"; path.write_bytes(A.canonical(receipt) + b"\n"); value = A.sha256_file(path)
            manifest = {"authorization": {"status": "frozen_verified", "contract_sha256": A.digest(contract),
                                           "receipt_sha256": value}}
            A.validate_authorization(path, value, copy.deepcopy(receipt), manifest, contract)
            bad = copy.deepcopy(receipt); bad["independent_review"]["approved"] = False
            path.write_bytes(A.canonical(bad) + b"\n"); bad_sha = A.sha256_file(path)
            bad_manifest = {"authorization": {"status": "frozen_verified", "contract_sha256": A.digest(contract),
                                               "receipt_sha256": bad_sha}}
            with self.assertRaisesRegex(ValueError, "review"):
                A.validate_authorization(path, bad_sha, bad, bad_manifest, contract)

    def test_original_core_pin_tamper(self):
        valid = {f"/f/{name}": value for name, value in A.FIXED_CORE.items()}
        A.validate_fixed_core_bindings(valid)
        bad = dict(valid); bad["/f/evaluate_new_phenomena_v2.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "core pin"):
            A.validate_fixed_core_bindings(bad)
        duplicate = dict(valid); duplicate["/other/evaluate_new_phenomena_v2.py"] = A.FIXED_CORE["evaluate_new_phenomena_v2.py"]
        with self.assertRaisesRegex(ValueError, "core pin"):
            A.validate_fixed_core_bindings(duplicate)

    def test_exclusive_receipt_and_no_orphan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / "audit.json"; receipt = {"status": "passed", "x": 1}
            A.publish_receipt(path, receipt)
            self.assertEqual(A.read_json(path), receipt)
            self.assertEqual(list(root.glob(".*.pending.*")), [])
            with self.assertRaisesRegex(ValueError, "New canonical"):
                A.publish_receipt(path, receipt)
            self.assertEqual(A.read_json(path), receipt)


if __name__ == "__main__":
    unittest.main()
