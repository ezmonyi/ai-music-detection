#!/usr/bin/env python3
"""Synthetic-only regression tests for exploratory-v5 presentation aggregation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import present_equal60_v5_results as P


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
    prefix = heldout.replace("_", "")
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
    if not P.AUDITOR.exists():
        P.AUDITOR.write_text("# synthetic independent auditor fixture\n")
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
        has_f = "F" in combo.split("+")
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
    contract = {"schema_version": 5, "authorized_stage": P.STAGE, "synthetic_test_only": True,
                "rows": 8, "label_counts": {"0": 4, "1": 4}, "source_counts": {"H1": 4, "A1": 4},
                "families": {x: [x] for x in P.FAMILIES},
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
    put(results / "run_manifest.json", {"schema_version": 5, "stage": P.STAGE,
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
    audit = {"status": "passed", "schema_version": 5, "stage": P.STAGE, "synthetic_test_only": True,
             "rows": 8, "label_counts": contract["label_counts"], "source_counts": contract["source_counts"],
             "families": P.FAMILY_COUNTS, "candidates": 127, "caps": [25, 50, 100, 200, "all"],
             "valid_folds": accounting["valid_folds"], "omitted_fold_cells": 1,
             "contract_sha256": P.digest(contract), "schedule_sha256": contract["schedule_sha256"],
             "result_commit_sha256": P.sha(results / "COMMIT.json"),
             "run_manifest_sha256": P.sha(results / "run_manifest.json"),
             "frozen_receipt_sha256": original_receipt_sha,
             "package_commit_sha256": P.digest(["synthetic_package"]),
             "result_files_sha256": result_hashes, "checked_binding_count": 3,
             "models": {"total": total_models, "primary": len(primary_index), "diagnostic": len(diagnostic_index),
                        "model_hashes_checked": total_models, "training_transforms_checked": total_models,
                        "normal_equation_residual_checked": total_models, "scores_replayed": total_predictions,
                        "score_threshold_decisions_checked": total_predictions,
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
             "optimizer_solution_independently_reproduced": False, "training_stationarity_checked": True,
             "metrics_independently_recomputed": True,
             "auditor_sha256": P.sha(P.AUDITOR),
             }
    audit_path = root / "audit.json"
    put(audit_path, audit)
    return results, audit_path


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        P.AUDITOR = cls.root / "audit_equal60_v5_results.py"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.case_root = self.root / self._testMethodName
        self.results, self.audit = fixture(self.case_root)

    def export(self, name):
        output = self.case_root / name
        return output, P.export(self.results, self.audit, P.sha(self.audit), output, synthetic=True)

    def refresh_commit_and_audit(self):
        commit = P.strict_json(self.results / "COMMIT.json")
        for name in P.RESULT_FILES:
            path = self.results / name
            commit["files"][name] = {"sha256": P.sha(path), "bytes": path.stat().st_size}
        put(self.results / "COMMIT.json", commit)
        audit = P.strict_json(self.audit)
        audit["result_files_sha256"] = {name: P.sha(self.results / name)
                                        for name in (*P.RESULT_FILES, "COMMIT.json")}
        audit["result_commit_sha256"] = P.sha(self.results / "COMMIT.json")
        audit["run_manifest_sha256"] = P.sha(self.results / "run_manifest.json")
        put(self.audit, audit)

    def test_complete_grids_estimands_overviews_and_commit(self):
        output, receipt = self.export("complete")
        with (output / "primary_source_endpoints_all_arms.csv").open() as stream:
            source = list(csv.DictReader(stream))
        row = next(r for r in source if r["heldout_source"] == "H1" and r["combination"] == "S+D+R+P"
                   and r["quantity"] == "25" and r["source_group"] == "H1")
        self.assertAlmostEqual(float(row["recording_rate"]), .5)
        self.assertAlmostEqual(float(row["equal_component_rate"]), 2 / 3)
        self.assertNotAlmostEqual(float(row["equal_component_rate"]), .75)  # Mean of fold component means is forbidden.
        with (output / "primary_pair_macro_all_arms.csv").open() as stream:
            pair = list(csv.DictReader(stream))
        macro = next(r for r in pair if r["summary_level"] == "fold_type_macro"
                     and r["fold_type"] == "human_source_holdout" and r["combination"] == "S+D+R+P"
                     and r["quantity"] == "25")
        self.assertAlmostEqual(float(macro["roc_auc"]), (.704 + .204) / 2)
        self.assertNotAlmostEqual(float(macro["roc_auc"]), (.604 + .804 + .204) / 3)
        with (output / "primary_fixed12_overview.csv").open() as stream:
            overview = list(csv.DictReader(stream))
        self.assertEqual({r["combination"] for r in overview}, set(P.FIXED12))
        with (output / "diagnostic_all_cap_pair_macro_all_arms.csv").open() as stream:
            diagnostics = list(csv.DictReader(stream))
        self.assertEqual({r["quantity"] for r in diagnostics}, {"all"})
        self.assertEqual({r["feature_mode"] for r in diagnostics}, set(P.DIAGNOSTIC_MODES))
        self.assertEqual(receipt["primary_family_cap_cells_per_reporting_arm"], 635)
        self.assertNotEqual(P.strict_json(self.audit)["frozen_receipt_sha256"],
                            P.sha(self.results / "frozen_authorization.json"))
        self.assertTrue((output / "COMMIT.json").is_file())
        latex = (output / "equal60_v5_presentation_tables_en.tex").read_text().splitlines()
        self.assertTrue(any(line.startswith("H-source &") and line.endswith(r"\\") for line in latex))

    def test_missing_primary_grid_rejected(self):
        rows = P.read_csv(self.results / "primary_model_index.csv", P.INDEX)
        table(self.results / "primary_model_index.csv", rows[:-1], P.INDEX)
        self.refresh_commit_and_audit()
        with self.assertRaisesRegex(ValueError, "Incomplete primary"):
            self.export("missing")
        table(self.results / "primary_model_index.csv", rows, P.INDEX)
        self.refresh_commit_and_audit()

    def test_duplicate_oof_identity_rejected_after_rehash(self):
        rows = P.read_csv(self.results / "primary_predictions.csv", P.PRED)
        # Duplicate one H1 identity into the second H1 fold for every matching model while retaining per-model size.
        first_by_combo_cap = {}
        index = {r["model_uid"]: r for r in P.read_csv(self.results / "primary_model_index.csv", P.INDEX)}
        for row in rows:
            model = index[row["model_uid"]]
            if model["heldout_source"] == "H1" and model["opposite_group_fold"] == "0" and row["source_group"] == "H1":
                first_by_combo_cap.setdefault((model["combination"], model["quantity"]), row["row_id"])
            if model["heldout_source"] == "H1" and model["opposite_group_fold"] == "1" and row["source_group"] == "H1":
                row["row_id"] = first_by_combo_cap[(model["combination"], model["quantity"])]
        table(self.results / "primary_predictions.csv", rows, P.PRED)
        self.refresh_commit_and_audit()
        with self.assertRaisesRegex(ValueError, "repeats"):
            self.export("duplicate")

    def test_duplicate_global_component_across_folds_rejected(self):
        rows = P.read_csv(self.results / "primary_predictions.csv", P.PRED)
        index = {r["model_uid"]: r for r in P.read_csv(self.results / "primary_model_index.csv", P.INDEX)}
        for row in rows:
            model = index[row["model_uid"]]
            if model["heldout_source"] == "H1" and model["opposite_group_fold"] == "1" and row["source_group"] == "H1":
                row["group_id"] = "H1_big"
        table(self.results / "primary_predictions.csv", rows, P.PRED)
        self.refresh_commit_and_audit()
        with self.assertRaisesRegex(ValueError, "component repeats"):
            self.export("duplicate_component")

    def test_existing_output_and_real_mode_rejected(self):
        # Restore fixture because the preceding mutation is class-scoped.
        output, _ = self.export("one")
        with self.assertRaisesRegex(ValueError, "New canonical"):
            P.export(self.results, self.audit, P.sha(self.audit), output, synthetic=True)
        with self.assertRaisesRegex(ValueError, "contract scope|Real v5"):
            P.export(self.results, self.audit, P.sha(self.audit), self.case_root / "real", synthetic=False)

    def test_stale_audit_and_missing_commit_rejected(self):
        (self.results / "primary_pair_metrics.csv").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "Committed result"):
            self.export("stale")
        self.results, self.audit = fixture(self.case_root / "fresh3")
        (self.results / "COMMIT.json").unlink()
        with self.assertRaisesRegex(ValueError, "COMMIT"):
            self.export("uncommitted")

    def test_explicit_audit_sha_is_required(self):
        with self.assertRaisesRegex(ValueError, "audit SHA"):
            P.export(self.results, self.audit, "0" * 64, self.case_root / "bad_audit_sha", synthetic=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
