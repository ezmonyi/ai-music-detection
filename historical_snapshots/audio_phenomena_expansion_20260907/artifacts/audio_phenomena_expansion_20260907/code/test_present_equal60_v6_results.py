#!/usr/bin/env python3
"""Synthetic-only regression tests for exploratory-v6 presentation aggregation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import present_equal60_v6_results as P


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def identities(heldout, fold):
    prefix = "synthetic_v6_" + heldout.replace("_", "")
    if heldout not in ("H1", "H2"):
        base = identities("H1", fold)
        return [base[0], next(row for row in base if row[1] == "1")]
    if heldout == "H1" and fold == 0:
        return [(f"{prefix}_h0a", "0", "H1", "H1_big"), (f"{prefix}_h0b", "0", "H1", "H1_big"),
                (f"{prefix}_h0c", "0", "H1", "H1_small"), (f"{prefix}_a0a", "1", "A1", "A1_0"),
                (f"{prefix}_a0b", "1", "A1", "A1_0")]
    if heldout == "H1" and fold == 1:
        return [(f"{prefix}_h1", "0", "H1", "H1_third"), (f"{prefix}_a1a", "1", "A1", "A1_1"),
                (f"{prefix}_a1b", "1", "A1", "A1_1")]
    return [(f"{prefix}_{fold}_h", "0", "H2" if heldout == "H2" else "H1", f"{prefix}_{fold}_hg"),
            (f"{prefix}_{fold}_a", "1", "A1", f"{prefix}_{fold}_ag")]


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    package = root / "package"
    package.mkdir()
    put(package / "COMMIT.json", {"synthetic_test_only": True})
    results = root / "scored"
    results.mkdir(parents=True)
    arm_folds = [("human_source_holdout", "H1", 0), ("human_source_holdout", "H1", 1),
                 ("human_source_holdout", "H2", 0), ("generator_holdout", "A1", 0),
                 ("generator_holdout", "A1", 1), ("ordinary_group_holdout_descriptive", "__all_sources__", 0),
                 ("ordinary_group_holdout_descriptive", "__all_sources__", 1)]
    schedule = []
    for fold_type, heldout, fold in arm_folds:
        test = identities(heldout, fold)
        for cap in P.CAPS:
            uid = f"synthetic_{fold_type}_{heldout}_{fold}_{cap}"
            train_rows = 8 + P.CAPS.index(cap)
            schedule.append({"fold_index": len(schedule), "quantity": cap, "fold_type": fold_type,
                             "heldout_source": heldout, "opposite_group_fold": fold,
                             "fold_uid": uid, "train_id_set_sha256": P.digest(["train", uid]),
                             "test_id_set_sha256": P.digest([r[0] for r in test]),
                             "train": {"rows": train_rows, "ids": [], "groups": [f"g{i}" for i in range(4)],
                                       "label_counts": {"0": 4, "1": 4},
                                       "source_rows": {"H1": train_rows // 2, "A1": train_rows - train_rows // 2},
                                       "source_groups": {"H1": 2, "A1": 2}},
                             "test": {"rows": len(test), "ids": sorted(r[0] for r in test),
                                      "groups": sorted({r[3] for r in test}), "label_counts": {},
                                      "source_rows": {}, "source_groups": {}}, "uncapped_train": {}})
    primary_index, diagnostic_index = [], []
    primary_predictions, diagnostic_predictions = [], []
    primary_pairs, diagnostic_pairs, primary_pooled, diagnostic_pooled = [], [], [], []

    def add(category, record, combo, mode, serial):
        uid = P.digest([category, record["fold_uid"], combo, mode])[:24]
        index = {"model_uid": uid, "model_sha256": P.digest(["model", uid]), "model_jsonl_line": str(serial),
                 "combination": combo, "feature_mode": mode, "quantity": str(record["quantity"]),
                 "fold_uid": record["fold_uid"], "fold_index": str(record["fold_index"]),
                 "fold_type": record["fold_type"], "heldout_source": record["heldout_source"],
                 "opposite_group_fold": str(record["opposite_group_fold"]),
                 "train_id_set_sha256": record["train_id_set_sha256"],
                 "test_id_set_sha256": record["test_id_set_sha256"],
                 "training_rows": str(record["train"]["rows"]), "test_rows": str(record["test"]["rows"])}
        target_index = primary_index if category == "primary" else diagnostic_index
        target_pred = primary_predictions if category == "primary" else diagnostic_predictions
        target_pair = primary_pairs if category == "primary" else diagnostic_pairs
        target_pooled = primary_pooled if category == "primary" else diagnostic_pooled
        target_index.append(index)
        test = identities(record["heldout_source"], record["opposite_group_fold"])
        has_f = "SC" in combo.split("+")
        for position, (row_id, label, source, group) in enumerate(test):
            correct = not (record["heldout_source"] == "H1" and
                           ((record["opposite_group_fold"] == 0 and group == "H1_big") or
                            (record["opposite_group_fold"] == 1 and label == "1" and position == len(test) - 1)))
            if has_f and record["heldout_source"] == "H1" and group == "H1_big":
                correct = True
            predicted = int(label) if correct else 1 - int(label)
            target_pred.append({"model_uid": uid, "row_id": row_id, "label": label, "source_group": source,
                                "group_id": group, "role": "development", "score": "0.8" if predicted else "0.2",
                                "threshold": "0.5", "predicted_label": str(predicted)})
        arm_auc = {"H1": (0.6, 0.8), "H2": (0.2,), "A1": (0.55, 0.65),
                   "__all_sources__": (0.7, 0.7)}[record["heldout_source"]][record["opposite_group_fold"]]
        auc = arm_auc + .001 * len(combo.split("+")) + (.0001 if mode == "missingness_only" else 0)
        common = {"model_uid": uid, "threshold": "0.5", "roc_auc": str(auc),
                  "balanced_accuracy": str(auc - .05), "ai_sensitivity": "0.5", "human_specificity": "0.5",
                  "tp": "1", "tn": "1", "fp": "1", "fn": "1"}
        target_pair.append({"model_uid": uid, "human_source": "H1", "ai_source": "A1",
                            "test_human": "2", "test_ai": "2", "test_human_groups": "1",
                            "test_ai_groups": "1", **{k: common[k] for k in P.PAIR if k in common and k != "model_uid"}})
        # Dict construction above keeps model_uid from the explicit field.
        target_pooled.append({"model_uid": uid, "test_rows": str(len(test)), "test_groups": str(len({r[3] for r in test})),
                              **{k: common[k] for k in P.POOLED if k in common and k != "model_uid"}})

    serial = 0
    for record in schedule:
        for combo in P.COMBINATIONS:
            serial += 1
            add("primary", record, combo, P.PRIMARY_MODE, serial)
    for record in schedule:
        if str(record["quantity"]) != "all":
            continue
        for combo in P.COMBINATIONS:
            for mode in P.DIAGNOSTIC_MODES:
                serial += 1
                add("diagnostic", record, combo, mode, serial)
    table(results / "primary_model_index.csv", primary_index, P.INDEX)
    table(results / "diagnostic_model_index.csv", diagnostic_index, P.INDEX)
    table(results / "primary_predictions.csv", primary_predictions, P.PRED)
    table(results / "diagnostic_predictions.csv", diagnostic_predictions, P.PRED)
    table(results / "primary_pair_metrics.csv", primary_pairs, P.PAIR)
    table(results / "diagnostic_pair_metrics.csv", diagnostic_pairs, P.PAIR)
    table(results / "primary_pooled_metrics.csv", primary_pooled, P.POOLED)
    table(results / "diagnostic_pooled_metrics.csv", diagnostic_pooled, P.POOLED)
    # The independent audit owns fold-local endpoint validation. Presentation recomputes cross-fold endpoints from predictions.
    placeholder = {name: "0" for name in P.SOURCE}
    placeholder.update(model_uid=primary_index[0]["model_uid"], source_group="synthetic", label="0", rows="1",
                       components="1", threshold="0.5")
    table(results / "primary_per_source_endpoints.csv", [placeholder], P.SOURCE)
    placeholder = dict(placeholder, model_uid=diagnostic_index[0]["model_uid"])
    table(results / "diagnostic_per_source_endpoints.csv", [placeholder], P.SOURCE)
    (results / "fold_models.jsonl").write_text("{}\n")
    with (results / "fold_registry.jsonl").open("w") as stream:
        for record in schedule:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    put(results / "omitted_fold_cells.json", [{"synthetic_test_only": True}])
    accounting = {"valid_folds": len(arm_folds), "primary_fits": len(primary_index),
                  "primary_prediction_rows": len(primary_predictions), "diagnostic_fits": len(diagnostic_index),
                  "diagnostic_prediction_rows": len(diagnostic_predictions)}
    unique = {r["row_id"]: r for r in primary_predictions}
    population = {"ids": sorted(unique), "label_counts": {label: sum(r["label"] == label for r in unique.values()) for label in ("0", "1")},
                  "source_rows": {source: sum(r["source_group"] == source for r in unique.values()) for source in {r["source_group"] for r in unique.values()}}}
    contract = {"schema_version": 6, "authorized_stage": P.STAGE, "synthetic_test_only": True,
                "rows": len(unique), "population": population, "input_package": str(package),
                "families": {x: [x + str(i) for i in range(P.FAMILY_COUNTS[x])] for x in P.FAMILIES},
                "combinations": list(P.COMBINATIONS), "quantities": [25, 50, 100, 200, "all"],
                "primary_feature_mode": P.PRIMARY_MODE, "all_cap_diagnostics": list(P.DIAGNOSTIC_MODES),
                "fixed_threshold": .5, "selection": False, "threshold_tuning": False,
                "full_cohort_refit": False, "schedule": schedule, "schedule_sha256": P.digest(schedule),
                "accounting": accounting}
    frozen = {"status": "frozen", "authorized_stage": P.STAGE, "contract": contract,
              "contract_sha256": P.digest(contract),
              "independent_review": {"approved": True, "reviewer": "root", "reviewed_utc": "synthetic"}}
    original_receipt_sha = P.digest(["original_pretty_frozen_receipt_bytes"])
    put(results / "frozen_authorization.json", frozen)
    stream_counts = {
        "primary": {"model_index": len(primary_index), "predictions": len(primary_predictions),
                    "pair_metrics": len(primary_pairs), "pooled_metrics": len(primary_pooled), "per_source_endpoints": 1},
        "diagnostic": {"model_index": len(diagnostic_index), "predictions": len(diagnostic_predictions),
                       "pair_metrics": len(diagnostic_pairs), "pooled_metrics": len(diagnostic_pooled), "per_source_endpoints": 1}}
    put(results / "run_manifest.json", {"schema_version": 6, "stage": P.STAGE,
                                        "status": "completed_exploratory_cv", "synthetic_test_only": True,
                                        "contract": contract, "stream_counts": stream_counts,
                                        "authorization": {"status": "frozen_verified",
                                                          "contract_sha256": P.digest(contract),
                                                          "receipt_sha256": original_receipt_sha}})
    commit = {"status": "committed", "publication": "exclusive directory reservation; hardlink COMMIT last",
              "files": {name: {"sha256": P.sha(results / name), "bytes": (results / name).stat().st_size}
                        for name in P.RESULT_FILES}}
    put(results / "COMMIT.json", commit)
    result_hashes = {name: P.sha(results / name) for name in (*P.RESULT_FILES, "COMMIT.json")}
    total_models = len(primary_index) + len(diagnostic_index)
    total_predictions = len(primary_predictions) + len(diagnostic_predictions)
    metrics = {"primary_pair_rows": len(primary_pairs), "primary_pooled_rows": len(primary_pooled),
               "primary_per_source_rows": 1, "diagnostic_pair_rows": len(diagnostic_pairs),
               "diagnostic_pooled_rows": len(diagnostic_pooled), "diagnostic_per_source_rows": 1}
    audit = {"status": "passed", "schema_version": 6, "stage": P.STAGE, "synthetic_test_only": True,
             "rows": len(unique), "label_counts": population["label_counts"], "source_counts": population["source_rows"],
             "families": P.FAMILY_COUNTS, "candidates": 255, "caps": [25, 50, 100, 200, "all"],
             "valid_folds": accounting["valid_folds"], "omitted_fold_cells": 1,
             "contract_sha256": P.digest(contract), "schedule_sha256": contract["schedule_sha256"],
             "result_commit_sha256": P.sha(results / "COMMIT.json"),
             "run_manifest_sha256": P.sha(results / "run_manifest.json"),
             "frozen_receipt_sha256": original_receipt_sha,
             "package_commit_sha256": P.sha(package / "COMMIT.json"),
             "result_files_sha256": result_hashes, "checked_binding_count": 3,
             "models": {"total": total_models, "primary": len(primary_index), "diagnostic": len(diagnostic_index),
                        "training_transforms_checked": total_models,
                        "normal_equation_residual_checked": total_models, "scores_replayed": total_predictions,
                        "maximum_normal_equation_absolute_residual": 1e-12,
                        "maximum_absolute_score_replay_error": 2e-15,
                        "minimum_absolute_saved_score_distance_from_threshold": .1},
             "streams": stream_counts, "metrics": metrics,
             "oof_source_endpoints": {"endpoint_cells": 10, "endpoint_rows_summed_across_cells": 100,
                                      "endpoint_components_summed_across_cells": 50,
                                      "duplicate_oof_ids_within_arm_source": 0, "denominator_checks_passed": 10,
                                      "canonical_endpoints_sha256": P.digest(["endpoints"])},
             "aggregation_contract": {"unique_oof_recording_rates_within_arm_source": True,
                                      "equal_global_component_rates": True, "equal_fold_component_mean": False,
                                      "fold_pair_macro_hierarchy": True, "pool_across_held_source_arms": False,
                                      "pooled_out_of_fold_auc": False},
             "model_fitting_performed": False, "scorer_predict_called": False,
             "protected_metadata_and_feature_lineage_independently_reconstructed": False,
             "accounting": accounting, "immutable_helper_sha256": P.AUDITOR_HELPER_SHA, "training_stationarity_checked": True,
             "metrics_independently_recomputed": True,
             "auditor_sha256": P.sha(P.AUDITOR),
             }
    audit_path = root / "audit.json"
    put(audit_path, audit)
    return results, audit_path


class PresentationV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.results, cls.audit = fixture(cls.root)
        cls.audit_value = P.strict_json(cls.audit)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_complete_grids_sc_pairs_estimands_and_diagnostics(self):
        out = self.root / "complete"
        receipt = P.export(self.results, self.audit, P.sha(self.audit), out, synthetic=True)
        self.assertEqual(receipt["matched_sc_pairs"], 127)
        self.assertEqual(receipt["primary_family_cap_cells_per_reporting_arm"], 1275)
        def load(name):
            with (out / name).open() as f:
                return list(csv.DictReader(f))
        source = load("primary_source_endpoints_all_arms.csv")
        row = next(r for r in source if r["heldout_source"] == "H1" and r["combination"] == "S"
                   and r["quantity"] == "25" and r["source_group"] == "H1")
        self.assertAlmostEqual(float(row["recording_rate"]), .5)
        self.assertAlmostEqual(float(row["equal_component_rate"]), 2 / 3)
        pair = load("primary_pair_macro_all_arms.csv")
        macro = next(r for r in pair if r["summary_level"] == "fold_type_macro"
                     and r["fold_type"] == "human_source_holdout" and r["combination"] == "S"
                     and r["quantity"] == "25")
        self.assertAlmostEqual(float(macro["roc_auc"]), (.701 + .201) / 2)
        self.assertNotAlmostEqual(float(macro["roc_auc"]), (.601 + .801 + .201) / 3)
        deltas = load("primary_matched_sc_deltas.csv")
        matching = [r for r in deltas if r["baseline_combination"] == "S"
                    and r["heldout_source"] == "H1" and r["source_group"] == "H1"
                    and r["quantity"] == "25"]
        self.assertEqual(len(matching), 2)
        self.assertEqual({r["added_combination"] for r in matching}, {"S+SC"})
        by_endpoint = {r["endpoint"]: float(r["added_minus_baseline_pp"]) for r in matching}
        self.assertAlmostEqual(by_endpoint["human_specificity_recording_rate"], 50.)
        self.assertAlmostEqual(by_endpoint["human_specificity_equal_component_rate"], 100 / 3)
        scopes = {}
        for r in deltas:
            key = (r["scope"], r["summary_level"], r["fold_type"], r["heldout_source"],
                   r["source_group"], r["endpoint"], r["feature_mode"])
            scopes.setdefault(key, []).append(r)
        for values in scopes.values():
            self.assertEqual(len(values), 127 * 5)
            self.assertEqual({r["baseline_combination"] for r in values}, set(P.BASELINES))
        diagnostic = load("diagnostic_all_cap_matched_sc_deltas.csv")
        self.assertEqual({r["quantity"] for r in diagnostic}, {"all"})
        self.assertEqual({r["feature_mode"] for r in diagnostic}, set(P.DIAGNOSTIC_MODES))
        self.assertTrue((out / "COMMIT.json").is_file())
        self.assertIn("Synthetic fixture", (out / "EQUAL60_V6_PRESENTATION_TABLES_EN.md").read_text())
        self.assertIn("SYNTHETIC FIXTURE", (out / "equal60_v6_presentation_tables_en.tex").read_text())

    def test_02_real_mode_rejects_synthetic_before_output(self):
        out = self.root / "forbidden_real"
        with self.assertRaisesRegex(ValueError, "scope changed"):
            P.export(self.results, self.audit, P.sha(self.audit), out, synthetic=False)
        self.assertFalse(out.exists())

    def test_03_unpassed_stale_or_incomplete_audit_rejected(self):
        for key, value in (("status", "failed"), ("result_commit_sha256", "0" * 64),
                           ("auditor_sha256", "0" * 64), ("immutable_helper_sha256", "0" * 64),
                           ("training_stationarity_checked", False),
                           ("protected_metadata_and_feature_lineage_independently_reconstructed", True)):
            with self.subTest(key=key):
                altered = dict(self.audit_value, **{key: value})
                path = self.root / ("bad_" + key + ".json")
                put(path, altered)
                with self.assertRaises(ValueError):
                    P.validate_gate(self.results, path, P.sha(path), True)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            P.validate_gate(self.results, self.audit, "0" * 64, True)

    def test_04_identity_or_denominator_mismatch_rejected(self):
        c = P.strict_json(self.results / "run_manifest.json")["contract"]
        _, by_uid = P.validate_schedule(c)
        indexes = P.validate_indexes(self.results, c, by_uid)
        primary = indexes["primary"][0]
        bad = [dict(r) for r in primary]
        next(r for r in bad if r["combination"] == "S+SC")["train_id_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identities differ"):
            P.sc_increments([], [], bad)
        aggregate = dict(summary_level="held_source_arm", fold_type="human_source_holdout",
                         heldout_source="H1", feature_mode=P.PRIMARY_MODE, quantity="25",
                         combination="S", held_source_arms=1, valid_folds=2, metric_cells=2,
                         roc_auc=.6, balanced_accuracy=.6)
        mismatched = dict(aggregate, combination="S+SC", valid_folds=1)
        with self.assertRaisesRegex(ValueError, "denominators differ"):
            P.sc_increments([aggregate, mismatched], [], primary)
        # Explicit diagnostic-mode identity must be independently matched as well.
        bad = [dict(r) for r in indexes["diagnostic"][0]]
        next(r for r in bad if r["combination"] == "S+SC" and r["feature_mode"] == "median_only")["test_id_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identities differ"):
            P.sc_increments([], [], bad)

    def test_05_publication_orphan_and_symlink_refused(self):
        orphan = self.results / "unexpected.csv"
        orphan.write_text("synthetic orphan")
        try:
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                P.validate_gate(self.results, self.audit, P.sha(self.audit), True)
        finally:
            orphan.unlink()
        link = self.root / "audit_symlink.json"
        link.symlink_to(self.audit)
        with self.assertRaisesRegex(ValueError, "canonical"):
            P.validate_gate(self.results, link, P.sha(link), True)

    def test_06_no_numerical_imports_or_fit_calls(self):
        import ast
        source = Path(P.__file__).read_text()
        tree = ast.parse(source)
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("evaluate" in ast.dump(n) or "numpy" in ast.dump(n) or "pandas" in ast.dump(n) for n in imports))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr in ("fit", "predict", "fit_candidate", "predict_candidate")
                             for n in ast.walk(tree)))
        self.assertEqual(len(P.COMBINATIONS), 255)
        self.assertEqual(len(P.INCREMENTS), 127)

    def test_07_committed_stream_byte_change_rejected(self):
        path = self.results / "primary_predictions.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        try:
            with self.assertRaisesRegex(ValueError, "hash/size changed"):
                P.validate_gate(self.results, self.audit, P.sha(self.audit), True)
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
