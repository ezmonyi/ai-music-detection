#!/usr/bin/env python3
"""Synthetic-only context, leakage, receipt and full127 schema-v4 regression tests."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import prepare_evaluation_inputs_v4 as P
import evaluate_new_phenomena_v4 as E


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(root):
    """Complete audit graph using no actual media or real corpus rows."""
    root.mkdir()
    native = []
    used = set()

    def group(fold, prefix):
        n = 0
        while True:
            value = f"synthetic_v4_{prefix}_{n}"
            n += 1
            if value not in used and E.BASE.hash_fold(value, 5, E.SEED) == fold:
                used.add(value)
                return value

    def add(label, role, source, g):
        i = "synthetic_v4_row_" + str(len(native))
        native.append({"id": i, "label": str(label), "source_group": source, "group_id": g,
                       "role": role, "duration_view": "60s", "duration_sec": "60",
                       "audio_path": "/synthetic/" + i + ".wav", "audio_offset_s": "1",
                       "crop_start_frame": "44100", "crop_frames": "2646000", "crop_end_frame_exclusive": "2690100",
                       "physical_sample_rate_hz": "44100", "native_sample_rate_hz": "44100", "physical_channels": "2", "physical_frames": "3000000",
                       "registered_raw_sha256": P.digest(i)})

    for fold in range(5):
        for label, source in ((0, "human_a"), (0, "human_b"), (1, "Suno")):
            add(label, "development", source, group(fold, f"{source}{fold}"))
        shared = group(fold, "shared" + str(fold))
        add(0, "development", "human_a", shared)
        add(1, "development", "Suno", shared)
    add(0, "locked", "human_a", "synthetic_v4_locked")
    add(1, "pilot", "ai_diffrhythm_pilot", "synthetic_v4_pilot")
    P.write_csv(root / "native.csv", native)
    fc = {"duration": 60, "preflight_only": False, "feature_names": P.NEW_COLUMNS,
          "metadata_sha256": P.sha(root / "native.csv"), "selected_ids": [r["id"] for r in native]}
    fh = P.digest(fc)
    dump(root / "fhm_contract.json", {**fc, "contract_hash": fh})
    fhm = []
    rng = np.random.default_rng(20260907)
    for n in native:
        row = {**n, "source_audio_path": n["audio_path"], "source_audio_sha256": n["registered_raw_sha256"],
               "source_sample_rate": "44100", "source_channels": "2", "source_total_frames": "3000000",
               "checked_crop_start_frame": "44100", "checked_crop_frames": "2646000",
               "extraction_status": "ok", "extraction_contract_hash": fh, "requested_duration_sec": "60",
               "input_row_hash": P.digest(n)}
        for columns in P.NEW_COLUMNS.values():
            row.update({c: str(rng.normal(float(n["label"]), 0.5)) for c in columns})
        fhm.append(row)
    # One entire new family unavailable on a dev row; keep it in every arm.
    for c in P.NEW_COLUMNS["M"]:
        fhm[0][c] = ""
    P.write_csv(root / "fhm.csv", fhm)
    dev = [r for r in native if r["role"] == "development"]
    mc = {"selection": "all_development", "manifest_sha256": P.sha(root / "native.csv"), "reference_features_sha256": P.sha(root / "fhm.csv"), "item_ids": [n["id"] for n in dev],
          "configuration": {"duration_sec": 60, "sample_rate_hz": 44100, "channels": 2, "short_input_padding": False, "normalization": False, "limiting": False, "output_subtype": "FLOAT", "coordinate_system": "exact_frozen_native_crop_then_resample"}}
    mh = P.digest(mc)
    dump(root / "materialization_contract.json", {**mc, "contract_sha256": mh})
    prepared, items, old = [], [], []
    run = {"duration": 60, "extractor_sha256": P.OLD_CODE["extract_expanded_four_family.py"], "core_sha256": P.OLD_CODE["expanded_feature_definitions.py"], "bias_sha256": P.BIAS_SHA}
    for n in dev:
        i = n["id"]
        p = {"item_id": i, "label": n["label"], "source_id": n["source_group"], "group_id": n["group_id"], "role": n["role"],
             "source_audio_path": n["audio_path"], "source_audio_sha256": n["registered_raw_sha256"], "source_sample_rate": "44100", "source_channels": "2", "source_total_frames": "3000000", "crop_start_frame": "44100", "crop_frames": "2646000",
             "input_row_sha256": P.digest(n), "contract_sha256": mh, "status": "verified", "duration": "60", "audio_offset_s": "0", "requires_crop": "0", "standardized_sr": "44100", "standardized_channels": "2", "standardized_frames": "2646000", "native_crop_shared_with_fhm": "True", "standardized_path": "/synthetic/std/" + i + ".wav", "standardized_file_sha256": P.digest(i + "std")}
        p["native_sr"] = "44100"
        prepared.append(p)
        audio = {"frames": 2646000, "sample_rate": 44100, "channels": 2, "finite": True, "subtype": "FLOAT", "path": p["standardized_path"], "sha256": p["standardized_file_sha256"]}
        stems = {s: {**audio, "path": "/synthetic/" + i + s, "sha256": P.digest(i + s)} for s in ("bass", "drums", "other", "vocals")}
        item = {"item_id": i, "input": audio, "stems": stems,
                "beats": {"sha256": P.digest(i + "beats"), "status": "nonempty", "beat_count": 120}, "structure": {"sha256": P.digest(i + "struct"), "input_path": p["standardized_path"]}, "spectrogram": {"shape": [4, 6000, 81], "sha256": P.digest(i + "spec")}}
        items.append(item)
        hashes = {"source_audio_sha256": audio["sha256"], **{s + "_sha256": v["sha256"] for s, v in stems.items()}, "beats_sha256": item["beats"]["sha256"], "structure_sha256": item["structure"]["sha256"]}
        o = {"item_id": i, "label": n["label"], "source_id": n["source_group"], "group_id": n["group_id"], "duration_sec": "60", "run_fingerprint": P.digest(run), "bias_sha256": P.BIAS_SHA, "status": "complete", "input_hashes": json.dumps(hashes)}
        o["native_sample_rate_hz"] = "44100"
        for columns in P.OLD_COLUMNS.values():
            o.update({c: str(rng.normal(float(n["label"]), 0.5)) for c in columns})
        old.append(o)
    for c in P.OLD_COLUMNS["P"]:
        old[0][c] = ""
    P.write_csv(root / "prepared.csv", prepared)
    P.write_csv(root / "old_features.csv", old)
    dump(root / "materialization_summary.json", {"status": "verified", "rows": len(dev), "contract_sha256": mh, "manifest_sha256": P.sha(root / "prepared.csv")})
    dump(root / "strict_inference_audit.json", {"status": "passed", "rows": len(dev), "manifest_sha256": P.sha(root / "prepared.csv"), "materialization_contract_sha256": mh, "items": items})
    dump(root / "extraction_receipt.json", {"status": "passed", "rows": len(dev), "no_short_padding_possible": True, "classifier_fitted": False, "features_sha256": P.sha(root / "old_features.csv"), "strict_inference_audit_sha256": P.sha(root / "strict_inference_audit.json")})
    om = {"rows": len(dev), "duration_sec": 60, "sample_rate_hz": 44100, "run_payload": run, "run_fingerprint": P.digest(run)}
    for c, prefix, name in (("S", "s8__", "s8_features"), ("D", "d__", "d_features"), ("R", "r__", "r_features"), ("P", "p__", "p_features")):
        om[name] = [v[len(prefix):] for v in P.OLD_COLUMNS[c]]
    dump(root / "old_metadata.json", om)


class SchemaV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence"
        fixture(self.evidence)

    def tearDown(self):
        self.temp.cleanup()

    def package(self):
        path = self.root / "package"
        P.prepare(self.evidence, path, True)
        return path

    def test_package_retains_missing_rows_and_exact127_plan(self):
        package = self.package()
        P.validate_package(package, True)
        table = E.load_table(package)
        self.assertEqual(len(table), 25)
        self.assertTrue(table.loc[table["__id"] == "synthetic_v4_row_0", P.NEW_COLUMNS["M"]].isna().all().all())
        plan, cohorts, _ = E.build_plan(table, E.load_family_config(package / "families_60s_v4.json"))
        self.assertEqual(plan["candidate_key"].nunique(), 127)
        self.assertEqual(len(plan), 232)
        self.assertEqual(plan["eligible_id_set_sha256"].nunique(), 1)
        self.assertEqual(len(cohorts), 1)
        self.assertEqual(len(E.make_schedule(table)), 25)

    def test_context_mutations_fail_closed(self):
        original = {n: (self.evidence / n).read_bytes() for n in P.EVIDENCE_NAMES}
        mutations = (("native.csv", "id", "duration_sec", "30"), ("native.csv", "id", "group_id", "wrong"), ("native.csv", "id", "role", "locked"), ("native.csv", "id", "source_group", "wrong"), ("fhm.csv", "id", "source_audio_sha256", "0" * 64), ("fhm.csv", "id", "checked_crop_start_frame", "0"), ("prepared.csv", "item_id", "standardized_frames", "1323000"), ("old_features.csv", "item_id", "input_hashes", "{}"))
        for name, key, field, value in mutations:
            with self.subTest(field=field):
                rows, _ = P.read_rows(self.evidence / name, key)
                rows[0][field] = value
                (self.evidence / name).unlink()
                P.write_csv(self.evidence / name, rows)
                with self.assertRaises(ValueError):
                    P.validate_evidence(self.evidence, True)
                (self.evidence / name).write_bytes(original[name])

    def test_group_cross_role_is_rejected(self):
        rows, _ = P.read_rows(self.evidence / "native.csv", "id")
        rows[-1]["group_id"] = rows[0]["group_id"]
        (self.evidence / "native.csv").unlink()
        P.write_csv(self.evidence / "native.csv", rows)
        with self.assertRaisesRegex(ValueError, "Global group crosses"):
            P.validate_evidence(self.evidence, True)

    def test_predictor_or_package_tamper_rejected(self):
        package = self.package()
        config_path = package / "families_60s_v4.json"
        config = P.read_json(config_path)
        config["old_families"]["S"]["columns"].append("s8_computed")
        dump(config_path, config)
        with self.assertRaises(ValueError):
            E.load_family_config(config_path)
        with self.assertRaisesRegex(ValueError, "Package hash mismatch"):
            P.validate_package(package, True)

    def test_audited_wrong_frames_and_hashes_rejected_after_receipt_rehash(self):
        path = self.evidence / "strict_inference_audit.json"
        audit = P.read_json(path)
        receipt_path = self.evidence / "extraction_receipt.json"
        receipt = P.read_json(receipt_path)
        for mutation, expected in (("frames", "audio context mismatch"), ("sha256", "different input hashes")):
            changed = copy.deepcopy(audit)
            changed["items"][0]["stems"]["vocals"][mutation] = 1323000 if mutation == "frames" else "0" * 64
            dump(path, changed)
            dump(receipt_path, {**receipt, "strict_inference_audit_sha256": P.sha(path)})
            with self.assertRaisesRegex(ValueError, expected):
                P.validate_evidence(self.evidence, True)

    def test_schedule_rejects_non_development_and_keeps_cross_label_group_together(self):
        package = self.package()
        table = E.load_table(package)
        cross = table.groupby("__group")["__label"].nunique()
        shared = set(cross[cross > 1].index)
        self.assertEqual(len(shared), 5)
        for _, train, test, _ in E.make_schedule(table):
            for g in shared:
                ids = set(table.loc[table["__group"] == g, "__id"])
                self.assertTrue(ids <= set(train["__id"]) or ids <= set(test["__id"]))
        table.loc[0, "__role"] = "locked"
        with self.assertRaisesRegex(ValueError, "Role leakage"):
            E.make_schedule(table)

    def test_authorization_real_requires_exact_frozen_receipt(self):
        contract = E.build_contract(self.package(), True)
        contract["synthetic_test_only"] = False
        with self.assertRaisesRegex(ValueError, "matching frozen receipt"):
            E.verify_authorization(contract)
        path = self.root / "receipt.json"
        receipt = {"status": "draft", "authorized_stage": "dev", "contract": contract, "contract_sha256": P.digest(contract)}
        dump(path, receipt)
        with self.assertRaises(ValueError):
            E.verify_authorization(contract, path)

        receipt["status"] = "frozen"
        dump(path, receipt)
        self.assertEqual(E.verify_authorization(contract, path)["status"], "frozen_verified")
        contract["parameters"]["threshold"] = 0.6
        with self.assertRaises(ValueError):
            E.verify_authorization(contract, path)

    def test_runner_cannot_fit_without_verified_authorization(self):
        with patch.object(E.V2, "fit_candidate", side_effect=AssertionError("unauthorized fit")):
            with self.assertRaisesRegex(ValueError, "Missing verified scoring authorization"):
                E.run_group_cv(self.root / "no_result", None, None, {"synthetic_test_only": False}, None)

    def test_synthetic_mode_cannot_admit_real_ids_or_real_package(self):
        with self.assertRaisesRegex(ValueError, "Frozen native.csv SHA"):
            P.validate_evidence(self.evidence, False)
        package = self.package()
        with self.assertRaisesRegex(ValueError, "Synthetic/real package mismatch"):
            P.validate_package(package, False)
        rows, _ = P.read_rows(self.evidence / "native.csv", "id")
        rows[-1]["id"] = "a_real_id"
        (self.evidence / "native.csv").unlink()
        P.write_csv(self.evidence / "native.csv", rows)
        with self.assertRaises(ValueError):
            P.validate_evidence(self.evidence, True)

    def test_full_synthetic127_run_shared_ids_no_transfer_and_diagnostics(self):
        package = self.package()
        output = self.root / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            E.main(["--stage", "dev", "--package-dir", str(package), "--output-dir", str(output), "--synthetic-test-only"])
        predictions = pd.read_csv(output / "development_group_cv_predictions.csv")
        self.assertEqual(predictions["combination"].nunique(), 127)
        self.assertEqual(len(predictions), 127 * 5 * 25)
        self.assertEqual(set(predictions["quantity"].astype(str)), {"25", "50", "100", "200", "all"})
        for _, frame in predictions.groupby("fold_uid"):
            self.assertEqual(frame["test_id_set_sha256"].nunique(), 1)
            self.assertEqual(frame["train_id_set_sha256"].nunique(), 1)
            expected = set(frame["row_id"])
            for _, candidate in frame.groupby("combination"):
                self.assertEqual(set(candidate["row_id"]), expected)
        for line in (output / "fold_registry.jsonl").read_text().splitlines():
            fold = json.loads(line)
            self.assertFalse(set(fold["train_group_ids"]) & set(fold["test_group_ids"]))
            self.assertFalse({"synthetic_v4_row_25", "synthetic_v4_row_26"} & set(fold["train_ids"] + fold["test_ids"]))
        diag = pd.read_csv(output / "development_group_cv_diagnostic_predictions.csv")
        self.assertEqual(set(diag["feature_mode"]), {"median_only", "missingness_only"})
        self.assertEqual(set(diag["quantity"]), {"all"})
        self.assertEqual(len(diag), 127 * 25 * 2)
        deltas = pd.read_csv(output / "development_group_cv_matched_deltas.csv")
        self.assertEqual(deltas["comparison_id"].nunique(), 105)
        self.assertFalse(any("source_holdout" in p.name for p in output.iterdir()))
        manifest = P.read_json(output / "run_manifest.json")
        self.assertIsNone(manifest["source_transfer_J"])
        self.assertFalse(manifest["winner_selected"])
        self.assertTrue((predictions["threshold"] == 0.5).all())

    def test_train_only_imputation_not_test_or_status(self):
        package = self.package()
        table = E.load_table(package)
        _, train, test, _ = E.make_schedule(table)[0]
        col = P.NEW_COLUMNS["M"][0]
        model = E.V2.fit_candidate(train, [col], "ridge")
        expected = np.nanmedian(train[col])
        self.assertEqual(model["medians"], [expected])
        model_before = copy.deepcopy(model)
        test[col] = 1e10
        E.V2.predict_candidate(test, model)
        self.assertEqual(model, model_before)
        self.assertEqual(model["columns"], [col])

    def test_draft_does_not_fit_and_existing_output_refused(self):
        package = self.package()
        output = self.root / "draft"
        with patch.object(E.V2, "fit_candidate", side_effect=AssertionError("draft fitted")):
            E.main(["--stage", "preregistration-draft", "--package-dir", str(package), "--output-dir", str(output), "--synthetic-test-only"])
        receipt = P.read_json(output / "preregistration_draft.json")
        self.assertEqual(receipt["status"], "draft")
        self.assertEqual(receipt["contract_sha256"], P.digest(receipt["contract"]))
        with self.assertRaises(ValueError):
            P.prepare(self.evidence, package, True)


if __name__ == "__main__":
    unittest.main()
