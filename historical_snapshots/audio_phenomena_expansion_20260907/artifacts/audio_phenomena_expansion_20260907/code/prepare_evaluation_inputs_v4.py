#!/usr/bin/env python3
"""Prepare matched-native-context exact60 S/D/R/P/F/H/M; never fit a model.

Evidence is copied into the package and revalidated by the evaluator. Physical
audio verification belongs to the strict inference auditor, whose receipt and
per-input hashes are required here; this program never downloads or decodes it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

SCHEMA_VERSION = 4
MANIFEST_SHA = "00ff8671c589ba280f2fc2a5bf3019f2d907fdf7a8933afa573f433112a8ee1b"
FHM_SHA = "46ea2a81d4653fa8d830e7f31a858ad75999105e8b6ce813032862743c7e51fa"
FHM_CONTRACT = "7051466edec749af6b2563821eb78cad77b2f444aa8f2b97bfa9dd659e92af1e"
MATERIALIZATION_CONTRACT = "6b51d502a68f3030592f2ddabda0596aa18f27471b79bfaf328fa08d6e6ed29c"
PREPARED_SHA = "f52b93bf4ff820bd35845749b91a2c7139b901588ebde2ebf62f4442f583d3bf"
BIAS_SHA = "bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e"
OLD_CODE = {
    "extract_expanded_four_family.py": "98fc6caa8b54ed1370269db13fe8d49559d0b877b4b0d5a3a069c8409906c5fe",
    "expanded_feature_definitions.py": "8b9745085c7518e84613b2ba499dbae775a57e4dcf95670c5e86a05ab524ff00",
}
OLD_COLUMNS = {
    "S": ["s8__" + n for n in ("tilt_1_5k_db_oct", "hf_tilt_5_7p5k_db_oct", "hf_ratio_5_7p5_db", "sibilance_ratio_5_7p5_db", "hf_flatness_5_7p5", "hf_entropy_5_7p5", "hf_crest_5_7p5_db", "fakeprint_peak_density_5_7p5_per_khz", "fakeprint_periodicity_5_7p5", "hf_flux_5_7p5", "hf_frame_similarity_5_7p5", "hf_power_sd_5_7p5_db", "hf_mod_4_12_share_5_7p5", "sibilance_contrast_5_7p5_db", "sibilance_burst_rate_5_7p5_hz")],
    "D": ["d__" + n for n in ("dynamics_span", "dynamics_iqr", "dynamics_adjacent_change")],
    "R": ["r__" + n for n in ("ibi_cv", "tempo_tv", "tempo_entropy")],
    "P": ["p__" + n for n in ("section_duration_cv", "section_duration_entropy", "section_bars_cv", "section_bars_offmode_fraction", "section_duration_median", "section_bars_median")],
}
NEW_COLUMNS = {
    "F": ["F_" + n for n in ("phase_residual_cvar_all", "phase_residual_cvar_attack", "phase_residual_cvar_sustain", "phase_residual_cvar_decay", "group_delay_iqr_ms_all", "group_delay_iqr_ms_attack", "group_delay_iqr_ms_sustain", "group_delay_iqr_ms_decay", "group_delay_cross_band_iqr_ms_all", "group_delay_cross_band_iqr_ms_attack", "group_delay_cross_band_iqr_ms_sustain", "group_delay_cross_band_iqr_ms_decay", "phase_residual_cvar_decay_minus_sustain", "group_delay_iqr_ms_decay_minus_sustain", "group_delay_cross_band_iqr_ms_decay_minus_sustain")],
    "H": ["H_" + n for n in ("pitch_class_entropy_norm", "pc_token_entropy_norm", "chroma_path_change_median", "chroma_path_change_iqr", "pc_path_step_median", "pc_path_large_step_rate")],
    "M": ["M_" + n for n in ("recurrence_peak_similarity", "recurrence_density", "recurrence_lag_contrast", "best_lag_sec", "best_transposition_semitones", "returning_pattern_count")],
}
COLUMNS = OLD_COLUMNS | NEW_COLUMNS
EVIDENCE_NAMES = ("native.csv", "fhm.csv", "fhm_contract.json", "materialization_contract.json", "materialization_summary.json", "prepared.csv", "strict_inference_audit.json", "old_features.csv", "old_metadata.json", "extraction_receipt.json")
META_COLUMNS = ["id", "label", "source_group", "group_id", "role", "duration_view"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_rows(path, id_column):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate/missing CSV header")
        rows = list(reader)
    ids = [r.get(id_column, "") for r in rows]
    require(bool(ids) and all(ids) and len(ids) == len(set(ids)), f"Blank/duplicate IDs: {path}")
    require(all(None not in r for r in rows), f"Malformed CSV: {path}")
    return rows, {r[id_column]: r for r in rows}


def check_hash(value, context):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"Invalid SHA256: {context}")


def assert_fields(left, right, fields, context):
    for lkey, rkey in fields.items():
        require(lkey in left and rkey in right and str(left[lkey]) == str(right[rkey]), f"{context}: mismatch {lkey}/{rkey}")


def number(row, key):
    value = float(row[key])
    require(math.isfinite(value), f"Nonfinite context field: {key}")
    return value


def build_config():
    def spec(code, columns):
        return {"state": "available" if columns else "planned", "columns": columns,
                "eligibility": {}, "status_columns": [], "minimum_observed_fraction": 0.05,
                "phenomenon": {"F": "phase evolution", "H": "pitch-class path", "M": "long-range recurrence proxy"}.get(code, code)}
    return {"schema_version": 4, "cohort": "all exact60 development native intervals",
            "eligibility": {}, "status_columns": [],
            "old_families": {c: spec(c, cols) for c, cols in OLD_COLUMNS.items()},
            "new_families": {c: spec(c, NEW_COLUMNS.get(c, [])) for c in ("F", "H", "M", "V", "B", "A", "T")}}


def validate_evidence(root, synthetic=False):
    """Validate audit graph and return deterministic dev-only model input rows."""
    root = Path(root)
    for name in EVIDENCE_NAMES:
        require((root / name).is_file(), f"Missing required evidence: {name}")
    native_rows, native = read_rows(root / "native.csv", "id")
    _, fhm = read_rows(root / "fhm.csv", "id")
    prepared_rows, prepared = read_rows(root / "prepared.csv", "item_id")
    _, old = read_rows(root / "old_features.csv", "item_id")
    require(set(native) == set(fhm), "Native/FHM ID sets differ")
    roles = {r["role"] for r in native_rows}
    require(roles <= {"development", "locked", "pilot"}, "Unexpected native role")
    dev = {i: r for i, r in native.items() if r["role"] == "development"}
    require(bool(dev) and set(prepared) == set(dev) == set(old), "Prepared/old IDs must equal ALL development IDs")
    if synthetic:
        require(all(i.startswith("synthetic_v4_") for i in native), "Synthetic bypass requires exclusively synthetic_v4_ IDs")
        require(len(native) < 500, "Synthetic corpus is too large")
    else:
        for name, expected in (("native.csv", MANIFEST_SHA), ("fhm.csv", FHM_SHA), ("prepared.csv", PREPARED_SHA)):
            require(sha(root / name) == expected, f"Frozen {name} SHA mismatch")
        require(len(native) == 2092 and len(dev) == 1604, "Frozen cohort size mismatch")
        require(sum(r["role"] == "locked" for r in native_rows) == 438 and sum(r["role"] == "pilot" for r in native_rows) == 50, "Historical/pilot role counts mismatch")
        require(sum(r["label"] == "0" for r in dev.values()) == 1208 and sum(r["label"] == "1" for r in dev.values()) == 396, "Development label count mismatch")
        require(len({r["source_group"] for r in dev.values() if r["label"] == "0"}) == 5, "Human source count mismatch")
    require({r["source_group"] for r in dev.values() if r["label"] == "1"} == {"Suno"}, "Sole development AI source must be Suno")
    require(all(r["label"] in {"0", "1"} and r["group_id"].strip() and r["source_group"].strip() for r in native_rows), "Invalid label/group/source")
    # Development groups may cross source/label, but cannot straddle frozen roles.
    group_roles = {}
    for r in native_rows:
        group_roles.setdefault(r["group_id"], set()).add(r["role"])
    require(all(len(v) == 1 for v in group_roles.values()), "Global group crosses development/locked/pilot roles")

    fc = read_json(root / "fhm_contract.json")
    fh = fc.pop("contract_hash")
    require(digest(fc) == fh and fc["duration"] == 60 and not fc.get("preflight_only", False), "FHM contract mismatch")
    require(fc["feature_names"] == NEW_COLUMNS, "FHM predictor definitions changed")
    require(fc["metadata_sha256"] == sha(root / "native.csv") and set(fc["selected_ids"]) == set(native) and len(fc["selected_ids"]) == len(native), "FHM contract input mismatch")
    mc = read_json(root / "materialization_contract.json")
    mh = mc.pop("contract_sha256")
    require(digest(mc) == mh and mc["selection"] == "all_development", "Materialization contract/probe mismatch")
    require(mc["manifest_sha256"] == sha(root / "native.csv") and mc["reference_features_sha256"] == sha(root / "fhm.csv"), "Materialization native/FHM hashes mismatch")
    require(mc["item_ids"] == [r["item_id"] for r in prepared_rows], "Materialization item order mismatch")
    config = mc["configuration"]
    for key, value in {"duration_sec": 60, "sample_rate_hz": 44100, "channels": 2, "short_input_padding": False, "normalization": False, "limiting": False, "output_subtype": "FLOAT", "coordinate_system": "exact_frozen_native_crop_then_resample"}.items():
        require(config.get(key) == value, f"Materialization setting changed: {key}")
    if not synthetic:
        require(fh == FHM_CONTRACT and mh == MATERIALIZATION_CONTRACT, "Frozen context contract mismatch")
    summary = read_json(root / "materialization_summary.json")
    require(summary["status"] == "verified" and summary["rows"] == len(dev) and summary["contract_sha256"] == mh and summary["manifest_sha256"] == sha(root / "prepared.csv"), "Materialization summary mismatch")
    audit = read_json(root / "strict_inference_audit.json")
    require(audit["status"] == "passed" and audit["rows"] == len(dev) and audit["manifest_sha256"] == sha(root / "prepared.csv") and audit["materialization_contract_sha256"] == mh, "Strict inference audit mismatch")
    items = {r["item_id"]: r for r in audit["items"]}
    require(len(items) == len(audit["items"]) and set(items) == set(dev), "Strict inference item IDs mismatch")
    if not synthetic:
        require(audit["code_sha256"] == sha(Path(__file__).with_name("verify_extract_equal60.py")), "Strict auditor code differs from reviewed version")
        checks = audit["runtime_code_checkpoint_bias_checks"]
        require(all(any(Path(p).name == name and value == expected for p, value in checks.items()) for name, expected in OLD_CODE.items()), "Old extractor code audit mismatch")
        require(BIAS_SHA in checks.values(), "Frozen external bias absent from audit")
    receipt = read_json(root / "extraction_receipt.json")
    require(receipt["status"] == "passed" and receipt["rows"] == len(dev) and receipt["no_short_padding_possible"] is True and receipt["classifier_fitted"] is False, "Old extraction receipt incomplete")
    require(receipt["features_sha256"] == sha(root / "old_features.csv") and receipt["strict_inference_audit_sha256"] == sha(root / "strict_inference_audit.json"), "Old receipt hashes mismatch")
    om = read_json(root / "old_metadata.json")
    require(om["duration_sec"] == 60 and om["sample_rate_hz"] == 44100 and om["rows"] == len(dev), "Old feature metadata context mismatch")
    payload = om["run_payload"]
    require(digest(payload) == om["run_fingerprint"] and payload["duration"] == 60, "Old run fingerprint mismatch")
    for key, expected in (("extractor_sha256", OLD_CODE["extract_expanded_four_family.py"]), ("core_sha256", OLD_CODE["expanded_feature_definitions.py"]), ("bias_sha256", BIAS_SHA)):
        require(payload[key] == expected, f"Old feature definition changed: {key}")
    for code, prefix, key in (("S", "s8__", "s8_features"), ("D", "d__", "d_features"), ("R", "r__", "r_features"), ("P", "p__", "p_features")):
        require([prefix + n for n in om[key]] == OLD_COLUMNS[code], f"Old family definition changed: {code}")
    metadata, features = [], []
    for i in sorted(dev):
        n, f, p, o, a = dev[i], fhm[i], prepared[i], old[i], items[i]
        assert_fields(n, f, {k: k for k in META_COLUMNS + ["audio_path", "audio_offset_s", "crop_start_frame", "crop_frames", "crop_end_frame_exclusive", "registered_raw_sha256"]}, i + " native/FHM")
        require(n["duration_view"] == "60s" and number(n, "duration_sec") == number(f, "requested_duration_sec") == 60, "Wrong native/FHM duration")
        require(f["extraction_contract_hash"] == fh and f["extraction_status"] == "ok" and f["input_row_hash"] == digest(n), "FHM row contract/hash mismatch")
        rate, start, count = number(n, "physical_sample_rate_hz"), number(n, "crop_start_frame"), number(n, "crop_frames")
        require(start >= 0 and start.is_integer() and count == 60 * rate and number(n, "crop_end_frame_exclusive") == start + count and start == round(number(n, "audio_offset_s") * rate), "Wrong exact native crop")
        require(start + count <= number(n, "physical_frames"), "Native crop exceeds source")
        assert_fields(n, f, {"audio_path": "source_audio_path", "registered_raw_sha256": "source_audio_sha256", "physical_sample_rate_hz": "source_sample_rate", "physical_channels": "source_channels", "physical_frames": "source_total_frames", "crop_start_frame": "checked_crop_start_frame", "crop_frames": "checked_crop_frames"}, i + " FHM source audit")
        assert_fields(n, p, {"label": "label", "role": "role", "group_id": "group_id", "source_group": "source_id", "audio_path": "source_audio_path", "registered_raw_sha256": "source_audio_sha256", "crop_start_frame": "crop_start_frame", "crop_frames": "crop_frames", "physical_sample_rate_hz": "source_sample_rate", "physical_channels": "source_channels", "physical_frames": "source_total_frames"}, i + " standardized/native")
        require(p["contract_sha256"] == mh and p["input_row_sha256"] == digest(n) and p["status"] == "verified", "Prepared row receipt mismatch")
        require(number(p, "duration") == 60 and number(p, "audio_offset_s") == 0 and number(p, "requires_crop") == 0 and number(p, "standardized_sr") == 44100 and number(p, "standardized_channels") == 2 and number(p, "standardized_frames") == 2646000, "Standardized context mismatch")
        require(number(p, "native_sr") == number(n, "native_sample_rate_hz") == rate, "Prepared native sample rate mismatch")
        require(p["native_crop_shared_with_fhm"] in (True, "True", "true", "1"), "Missing same-native-crop assertion")
        for media in [a["input"], *a["stems"].values()]:
            require(media["frames"] == 2646000 and media["sample_rate"] == 44100 and media["channels"] == 2 and media["finite"] is True, "Strict audit audio context mismatch")
            check_hash(media["sha256"], i)
        require(set(a["stems"]) == {"bass", "drums", "other", "vocals"}, "Wrong stem set")
        require(a["input"]["subtype"] == "FLOAT" and a["input"]["path"] == p["standardized_path"] and a["input"]["sha256"] == p["standardized_file_sha256"], "Audited standardized input mismatch")
        require(a["structure"]["input_path"] == p["standardized_path"] and a["spectrogram"]["shape"] == [4, 6000, 81], "Structure/spectrogram context mismatch")
        expected = {"source_audio_sha256": p["standardized_file_sha256"], **{s + "_sha256": a["stems"][s]["sha256"] for s in a["stems"]}, "beats_sha256": a["beats"]["sha256"], "structure_sha256": a["structure"]["sha256"]}
        for h in expected.values():
            check_hash(h, i)
        require(json.loads(o["input_hashes"]) == expected, "Old features opened different input hashes")
        assert_fields(n, o, {"label": "label", "source_group": "source_id", "group_id": "group_id"}, i + " old identity")
        require(number(o, "native_sample_rate_hz") == rate, "Old feature native sample rate mismatch")
        require(number(o, "duration_sec") == 60 and o["run_fingerprint"] == om["run_fingerprint"] and o["bias_sha256"] == BIAS_SHA and o["status"] == "complete", "Old feature run context mismatch")
        metadata.append({k: n[k] for k in META_COLUMNS})
        row = {"id": i}
        for code, columns in COLUMNS.items():
            source = o if code in OLD_COLUMNS else f
            for col in columns:
                value = source[col]
                require(value.strip().lower() in {"", "nan", "na", "null"} or math.isfinite(float(value)), f"Invalid numeric feature: {col}")
                row[col] = value
        features.append(row)
    return metadata, features


def write_csv(path, rows):
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prepare(evidence, output, synthetic=False):
    output = Path(output).resolve()
    require(not output.exists(), f"Refusing existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".tmp.", dir=output.parent))
    try:
        copied = temp / "evidence"
        copied.mkdir()
        for name in EVIDENCE_NAMES:
            shutil.copyfile(Path(evidence) / name, copied / name)
        metadata, features = validate_evidence(copied, synthetic)
        write_csv(temp / "metadata_60s.csv", metadata)
        write_csv(temp / "features_60s.csv", features)
        (temp / "families_60s_v4.json").write_text(json.dumps(build_config(), indent=2) + "\n")
        proof = {"schema_version": 4, "status": "prepared_not_authorized_for_scoring", "synthetic_test_only": synthetic,
                 "rows": len(metadata), "scoring_authorized": False,
                 "preparer_sha256": sha(Path(__file__)),
                 "files_sha256": {str(p.relative_to(temp)): sha(p) for p in sorted(temp.rglob("*")) if p.is_file()}}
        (temp / "preparation_audit.json").write_text(json.dumps(proof, indent=2) + "\n")
        require(not output.exists(), f"Output appeared during preparation: {output}")
        os.rename(temp, output)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return proof


def validate_package(root, synthetic=False):
    root = Path(root)
    proof = read_json(root / "preparation_audit.json")
    require(proof.get("schema_version") == 4 and proof.get("status") == "prepared_not_authorized_for_scoring" and proof.get("scoring_authorized") is False, "Invalid preparation proof")
    require(proof["synthetic_test_only"] is synthetic, "Synthetic/real package mismatch")
    require(proof["preparer_sha256"] == sha(Path(__file__)), "Preparer code changed")
    expected_names = {"evidence/" + n for n in EVIDENCE_NAMES} | {"metadata_60s.csv", "features_60s.csv", "families_60s_v4.json"}
    require(set(proof["files_sha256"]) == expected_names, "Incomplete package hash map")
    for name, expected in proof["files_sha256"].items():
        require(sha(root / name) == expected, f"Package hash mismatch: {name}")
    metadata, features = validate_evidence(root / "evidence", synthetic)
    require(read_rows(root / "metadata_60s.csv", "id")[0] == metadata and read_rows(root / "features_60s.csv", "id")[0] == features, "Prepared model input differs from validated evidence")
    require(read_json(root / "families_60s_v4.json") == build_config() and proof["rows"] == len(metadata), "Prepared family/row contract mismatch")
    return proof


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--synthetic-test-only", action="store_true")
    a = p.parse_args(argv)
    print(json.dumps(prepare(a.evidence_dir, a.output_dir, a.synthetic_test_only), indent=2))


if __name__ == "__main__":
    main()
