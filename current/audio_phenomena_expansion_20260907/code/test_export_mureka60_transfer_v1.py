#!/usr/bin/env python3
"""Synthetic-only tests for the Mureka transfer table exporter."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import export_mureka60_transfer_v1 as E


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def table(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(root):
    scored = root / "verified_mirror"
    scored.mkdir()
    identities = [{"id": f"synthetic_mureka_v4_{i}", "label": "1", "source_group": "Mureka_v9",
                   "group_id": f"synthetic_group_{i // 2}", "role": "external_generator_unscored",
                   "acquisition_role": "reserved_unscored"} for i in range(3)]
    table(scored / "identity_roles.csv", identities, E.META)
    index, summaries = [], []
    for combo in E.COMBINATIONS:
        for cap in E.CAPS:
            for fold in E.FOLDS:
                common = {"combination": combo, "quantity": cap, "fold_index": fold,
                          "fold_uid": f"synthetic_{cap}_{fold}", "model_sha256": h(combo + cap + fold),
                          "candidate_key": "synthetic_" + combo,
                          "train_id_set_sha256": h("train" + cap + fold),
                          "test_id_set_sha256": h("test" + cap + fold)}
                index.append(common)
                tp = int(fold) % 3 + 1
                summaries.append(dict(common, synthetic_test_only="True", rows="3", unique_ids="3",
                                      unique_groups="2", tp=str(tp), fn=str(3 - tp), threshold="0.5",
                                      ai_sensitivity=str(tp / 3), false_negative_rate=str((3 - tp) / 3)))
    table(scored / "model_index.csv", index, E.INDEX)
    table(scored / "per_model_sensitivity.csv", summaries, E.SUMMARY)
    (scored / "predictions.csv").write_text("synthetic_test_only\nTrue\n")
    put(scored / "scoring_receipt.json", {"synthetic_test_only": True})
    files = {}
    for path in scored.iterdir():
        files[path.name] = {"sha256": E.sha(path), "bytes": path.stat().st_size}
    manifest = {"status": "scored", "synthetic_test_only": True, "contract_sha256": h("contract"),
                "classifier_fitted": False, "original_v4_development_admission": False,
                "unique_new_ids": 3, "model_instances": 3175, "prediction_rows": 9525,
                "summary_rows": 3175, "files": files}
    put(scored / "publication_manifest.json", manifest)
    original = root / "original_remote_style_path"
    checked = {str(original / path.name): E.sha(path) for path in scored.iterdir()}
    checked[str(E.AUDITOR.resolve())] = E.sha(E.AUDITOR)
    audit = {"schema_version": 1, "status": "passed", "synthetic_test_only": True,
             "rows": 3, "unique_ids": 3, "positive_class_count": 3, "unique_groups": 2,
             "model_instances": 3175, "combinations": 127, "caps": list(E.CAPS),
             "fold_models_per_cap": 5, "prediction_rows": 9525,
             "sensitivity_summary_rows": 3175, "denominator_per_model": 3,
             "all_ids_in_every_model": True,
             "all_decisions_match_published_and_independent_scores": True,
             "model_fitting_performed": False, "new_source_transform_learning_performed": False,
             "source_media_rehashed": False, "source_media_decoded": False,
             "endpoints": ["ai_sensitivity", "false_negative_rate"],
             "publication_manifest_sha256": E.sha(scored / "publication_manifest.json"),
             "auditor_sha256": E.sha(E.AUDITOR), "checked_files_sha256": checked}
    audit_path = root / "audit.json"
    put(audit_path, audit)
    return scored, audit_path


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.scored, self.audit = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_export(self, name="output"):
        output = self.root / name
        receipt = E.export(self.scored, self.audit, output, synthetic=True, verified_mirror=True)
        return output, receipt

    def refresh_manifest_and_audit(self):
        manifest_path = self.scored / "publication_manifest.json"
        manifest = E.strict_json(manifest_path)
        for name in manifest["files"]:
            path = self.scored / name
            manifest["files"][name] = {"sha256": E.sha(path), "bytes": path.stat().st_size}
        put(manifest_path, manifest)
        audit = E.strict_json(self.audit)
        for name in E.SCORED_FILES:
            path = self.scored / name
            original = next(key for key in audit["checked_files_sha256"] if Path(key).name == name)
            audit["checked_files_sha256"][original] = E.sha(path)
        audit["publication_manifest_sha256"] = E.sha(manifest_path)
        put(self.audit, audit)

    def test_means_counts_and_complete_outputs(self):
        output, receipt = self.run_export()
        rows = E.read_csv(output / "mureka60_transfer_all_635.csv",
                          ("combination", "quantity", "stored_fold_models", "songs_per_model",
                           "model_song_applications", "mean_sensitivity", "min_sensitivity",
                           "max_sensitivity", "mean_fnr", "mean_tp", "mean_fn"))
        self.assertEqual(len(rows), 635)
        first = rows[0]
        self.assertAlmostEqual(float(first["mean_sensitivity"]), (1 + 2 + 3 + 1 + 2) / 15)
        self.assertEqual(first["songs_per_model"], "3")
        self.assertEqual(first["model_song_applications"], "15")
        self.assertEqual(receipt["exported_cells"], 635)
        self.assertIn("not confidence intervals", (output / "MUREKA60_TRANSFER_TABLES_EN.md").read_text())
        self.assertEqual(len(E.read_csv(output / "mureka60_transfer_overview.csv",
                                        ("combination",) + E.CAPS)), 12)

    def test_missing_grid_rejected(self):
        rows = E.read_csv(self.scored / "per_model_sensitivity.csv", E.SUMMARY)
        table(self.scored / "per_model_sensitivity.csv", rows[:-1], E.SUMMARY)
        self.refresh_manifest_and_audit()
        with self.assertRaisesRegex(ValueError, "3,175"):
            self.run_export()

    def test_stale_audit_hash_rejected(self):
        with (self.scored / "per_model_sensitivity.csv").open("a") as handle:
            handle.write("tamper\n")
        with self.assertRaisesRegex(ValueError, "manifest record mismatch|audit hash mismatch|no unique audited"):
            self.run_export()

    def test_verified_compact_mirror_may_omit_predictions(self):
        (self.scored / "predictions.csv").unlink()
        output, receipt = self.run_export("compact")
        self.assertTrue((output / "mureka60_transfer_all_635.csv").is_file())
        self.assertEqual(receipt["locally_omitted_audit_bound_scored_files"], ["predictions.csv"])

    def test_endpoint_tamper_rejected_even_when_rehashed(self):
        rows = E.read_csv(self.scored / "per_model_sensitivity.csv", E.SUMMARY)
        rows[0]["ai_sensitivity"] = "0.9"
        table(self.scored / "per_model_sensitivity.csv", rows, E.SUMMARY)
        self.refresh_manifest_and_audit()
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            self.run_export()

    def test_existing_output_rejected(self):
        output, _ = self.run_export()
        with self.assertRaisesRegex(ValueError, "existing"):
            E.export(self.scored, self.audit, output, synthetic=True, verified_mirror=True)

    def test_small_fixture_cannot_enter_real_mode(self):
        with self.assertRaisesRegex(ValueError, "500 rows"):
            E.export(self.scored, self.audit, self.root / "real", synthetic=False, verified_mirror=True)


if __name__ == "__main__":
    unittest.main()
