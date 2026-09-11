#!/usr/bin/env python3
"""Freeze-bound NSynth DEVELOPMENT pilot. Draft never extracts features.

The only run admission flag is an explicitly supplied hash of the reviewed
draft. No fitting, classifier, codec processing, or external-gate decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.io import wavfile

import modulation_candidate_v1 as candidate

VERSION = "modulation_nsynth_development_pilot_v1_20260907"
CANDIDATE_SHA256 = "8742a44746d638024832db6b2d8933a5925a811d481785cb1d741b1c694984fb"
SOURCE_COMMIT_SHA256 = "85f70e63a033e1509d3c2fbad6c4561c3f1fce2511117756af64f78b9c39e330"
SOURCE_ARCHIVE_SHA256 = "0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c"
SOURCE_ACCEPTANCE_SHA256 = "df5cc7a0ab882e581ff044ded6aa8396a25afae0380d07c583906f286e93df81"
CANDIDATE_ACCEPTANCE_SHA256 = "28e7a5126a4fd4d11183defd780abd0ca16b912c402fefd980cd47ca673b1028"
NOTE_COUNT = 4096
INSTRUMENT_COUNT = 53
PILOT_INSTRUMENT_COUNT = 27
NOTES_PER_INSTRUMENT = 2
CONDITIONS = ("baseline", "common_gain", "polarity", "am8_depth02", "am8_depth06",
              "am48_depth02", "am48_depth06", "chirp16to64_depth06")
POLICY = {
    "status": "pilot_only_not_external_gate_passed",
    "instrument_sort": "sha256('Q-pilot-20260907|' + instrument_str), instrument_str",
    "pilot_instruments": 27, "reserved_instruments": 26,
    "notes_per_instrument": 2, "note_sort": "abs(pitch-60), sha256('Q-note-20260907|'+id), id",
    "require_distinct_pcm_sha256_within_instrument": True,
    "conditions": list(CONDITIONS), "sample_rate": 16000, "samples": 64000,
    "derivative_dtype": "float64", "known_peak_interval_hz": [2, 128],
    "known_peak_tolerance_hz": 1.0, "local_power_halfwidth_hz": 1.0,
    "aggregation": "mean within instrument, then equal-weight mean across covered instruments",
    "no_numeric_admission_threshold": True, "codecs": False,
    "ai_human_labels": False, "autofit": False,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical_bytes(value))


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def product_fingerprint(path):
    record = fingerprint(path)
    return {key: record[key] for key in ("bytes", "sha256")}


def safe_child(root, name):
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe product path: {name}")
    target = root / rel
    # Reject symlinks in any path component, not just escapes from the root.
    if any((root / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
        raise ValueError(f"symlink product: {name}")
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"not a contained regular file: {name}")
    return resolved


def verify_source(source_dir):
    """Byte validation and existing decode-inventory validation, never DSP."""
    root = Path(source_dir).resolve(strict=True)
    commit_path = safe_child(root, "COMMIT.json")
    if sha256_file(commit_path) != SOURCE_COMMIT_SHA256:
        raise ValueError("source COMMIT differs from the parent-verified official acquisition")
    commit = read_json(commit_path)
    if commit.get("status") != "committed" or not isinstance(commit.get("products"), dict):
        raise ValueError("source has no valid committed product inventory")
    products = commit["products"]
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual_files != set(products) | {"COMMIT.json"}:
        raise ValueError("source file set differs from COMMIT")
    hashes = {}
    for name, expected in sorted(products.items()):
        path = safe_child(root, name)
        actual = product_fingerprint(path)
        if actual != expected:
            raise ValueError(f"source product mismatch: {name}")
        hashes[name] = actual
    hashes["COMMIT.json"] = product_fingerprint(commit_path)
    if hashes.get("nsynth-test.jsonwav.tar.gz", {}).get("sha256") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive differs from the verified official release")
    receipt = read_json(root / "acquisition_receipt.json")
    if (receipt.get("decoded_all") is not True or receipt.get("notes") != NOTE_COUNT
            or receipt.get("instruments") != INSTRUMENT_COUNT
            or receipt.get("ai_human_labels_assigned") is not False
            or receipt.get("features_extracted") is not False):
        raise ValueError("source decode receipt or source role mismatch")
    metadata = read_json(root / "nsynth-test/examples.json")
    with (root / "manifest.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != NOTE_COUNT or len(metadata) != NOTE_COUNT:
        raise ValueError("source must contain all 4096 note metadata records")
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != NOTE_COUNT or set(by_id) != set(metadata):
        raise ValueError("source note ID inventory mismatch or duplicate")
    expected_wave_paths = set()
    for note_id, row in sorted(by_id.items()):
        name = f"nsynth-test/audio/{note_id}.wav"
        expected_wave_paths.add(name)
        path = safe_child(root, name)
        if (row.get("path") != str(path) or row.get("metadata") != metadata[note_id]
                or row.get("file_sha256") != hashes[name]["sha256"]
                or row.get("native_channels") != 1 or row.get("native_sample_rate") != 16000
                or row.get("frames") != 64000 or row.get("duration_s") != 4
                or row.get("ai_human_label") is not None
                or row.get("role") != "external_measurement_control_only"
                or row["metadata"].get("sample_rate") != 16000
                or row["metadata"].get("note_str") != note_id):
            raise ValueError(f"decoded inventory mismatch: {note_id}")
        pcm_hash = row.get("pcm_sha256", "")
        if len(pcm_hash) != 64 or any(c not in "0123456789abcdef" for c in pcm_hash):
            raise ValueError(f"invalid decoded PCM digest: {note_id}")
    if {name for name in products if name.endswith(".wav")} != expected_wave_paths:
        raise ValueError("wave inventory does not match note IDs")
    instrument_names = {row["metadata"]["instrument_str"] for row in rows}
    if len(instrument_names) != INSTRUMENT_COUNT:
        raise ValueError("source must contain exactly 53 instruments")
    return {"source_dir": str(root), "all_source_byte_hashes": hashes,
            "all_4096_manifest_sha256": hashes["manifest.jsonl"]["sha256"],
            "all_4096_metadata_file_sha256": hashes["nsynth-test/examples.json"]["sha256"],
            "all_4096_manifest_metadata_sha256": hashlib.sha256(canonical_bytes(
                [{"id": key, "metadata": by_id[key]["metadata"]} for key in sorted(by_id)])).hexdigest(),
            "acquisition_code_sha256": receipt["code_sha256"],
            "notes": len(rows), "instruments": len(instrument_names)}, by_id


def selected_inventory(by_id):
    grouped = defaultdict(list)
    for note_id, row in by_id.items():
        grouped[row["metadata"]["instrument_str"]].append(row)
    ordered = sorted(grouped, key=lambda s: (hashlib.sha256(
        ("Q-pilot-20260907|" + s).encode()).hexdigest(), s))
    if len(ordered) != INSTRUMENT_COUNT:
        raise ValueError("selection requires 53 instruments")
    pilot, reserved = ordered[:PILOT_INSTRUMENT_COUNT], ordered[PILOT_INSTRUMENT_COUNT:]
    pilot_pcm = {row["pcm_sha256"] for name in pilot for row in grouped[name]}
    reserved_pcm = {row["pcm_sha256"] for name in reserved for row in grouped[name]}
    if pilot_pcm & reserved_pcm:
        raise ValueError("pilot and reserved instruments share decoded PCM")
    selected = []
    for instrument in pilot:
        candidates = sorted(grouped[instrument], key=lambda row: (
            abs(row["metadata"]["pitch"] - 60),
            hashlib.sha256(("Q-note-20260907|" + row["id"]).encode()).hexdigest(), row["id"]))
        seen = set()
        for row in candidates:
            if row["pcm_sha256"] in seen:
                continue
            selected.append(row)
            seen.add(row["pcm_sha256"])
            if len(seen) == NOTES_PER_INSTRUMENT:
                break
        if len(seen) != NOTES_PER_INSTRUMENT:
            raise ValueError(f"cannot select two distinct PCM notes for {instrument}; no substitution")
    if len(selected) != 54:
        raise ValueError("selection must contain exactly 54 notes")
    return {"ordered_instrument_ids": ordered, "pilot_instrument_ids": pilot,
            "reserved_instrument_ids": reserved, "selected_notes": selected,
            "pilot_reserved_pcm_sets_disjoint": True}


def runtime_bindings():
    return {"python_version": platform.python_version(), "python_executable": fingerprint(sys.executable),
            "platform_system": platform.system(), "platform_machine": platform.machine(),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "module_files": {name: fingerprint(module.__file__) for name, module in
                             (("numpy", np), ("scipy", scipy), ("scipy.io.wavfile", wavfile))}}


def code_bindings(protocol_path):
    code = Path(__file__).resolve().parent
    result = {"runner": fingerprint(__file__), "candidate": fingerprint(candidate.__file__),
              "runner_tests": fingerprint(code / "test_modulation_nsynth_pilot_v1.py"),
              "candidate_tests": fingerprint(code / "test_modulation_candidate_v1.py"),
              "acquisition_code": fingerprint(code / "acquire_nsynth_test_controls_v1.py"),
              "protocol": fingerprint(protocol_path), "runtime": runtime_bindings(),
              "source_parent_acceptance": fingerprint(code.parent / "audit/nsynth_source_parent_acceptance_v1.json"),
              "candidate_parent_acceptance": fingerprint(code.parent / "audit/modulation_candidate_parent_acceptance_v1.json")}
    if result["candidate"]["sha256"] != CANDIDATE_SHA256:
        raise ValueError("candidate immutable SHA256 mismatch")
    if (result["source_parent_acceptance"]["sha256"] != SOURCE_ACCEPTANCE_SHA256
            or result["candidate_parent_acceptance"]["sha256"] != CANDIDATE_ACCEPTANCE_SHA256):
        raise ValueError("parent source or candidate acceptance receipt differs from reviewed hash")
    source_acceptance = read_json(result["source_parent_acceptance"]["path"])
    candidate_acceptance = read_json(result["candidate_parent_acceptance"]["path"])
    if (source_acceptance.get("passed") is not True
            or candidate_acceptance.get("status") != "accepted_for_external_development_pilot_only"
            or candidate_acceptance.get("code_sha256") != CANDIDATE_SHA256
            or candidate_acceptance.get("classifier_admitted") is not False):
        raise ValueError("parent acceptance scope mismatch")
    return result


def new_output_path(path, source_dir):
    output = Path(path).resolve()
    if output.exists() or output.is_relative_to(Path(source_dir).resolve()):
        raise ValueError("output must be a new directory outside the read-only source")
    return output


def commit_directory(directory, extra):
    products = {str(p.relative_to(directory)): product_fingerprint(p)
                for p in sorted(directory.rglob("*")) if p.is_file()}
    write_json_new(directory / "COMMIT.json", {"status": "committed", **extra, "products": products})


def draft(source_dir, protocol_path, draft_dir, run_output_dir):
    draft_path = new_output_path(draft_dir, source_dir)
    run_path = new_output_path(run_output_dir, source_dir)
    if draft_path == run_path or draft_path.is_relative_to(run_path) or run_path.is_relative_to(draft_path):
        raise ValueError("draft and run output directories must be separate")
    sources, by_id = verify_source(source_dir)
    bindings = code_bindings(protocol_path)
    if sources["acquisition_code_sha256"] != bindings["acquisition_code"]["sha256"]:
        raise ValueError("acquisition code differs from source receipt")
    document = {"version": VERSION, "stage": "draft_requires_explicit_reviewed_sha256",
                "policy": POLICY, "planned_run_output_dir": str(run_path),
                "sources": sources, "selection": selected_inventory(by_id), "bindings": bindings,
                "features_extracted": False, "reserved_waveforms_decoded_or_extracted": False}
    if code_bindings(protocol_path) != bindings:
        raise ValueError("code or runtime bindings changed during draft")
    draft_path.mkdir(parents=True, exist_ok=False)
    write_json_new(draft_path / "draft.json", document)
    digest = sha256_file(draft_path / "draft.json")
    commit_directory(draft_path, {"kind": "development_pilot_draft", "draft_sha256": digest})
    return {"draft": str(draft_path / "draft.json"), "freeze_sha256_for_review": digest,
            "selected_notes": 54, "reserved_instruments": 26}


def derivatives(x):
    t = np.arange(64000, dtype=np.float64) / 16000.0
    result = {"baseline": .25 * x, "common_gain": .1 * x, "polarity": -.25 * x}
    for frequency in (8, 48):
        for depth, tag in ((.2, "02"), (.6, "06")):
            result[f"am{frequency}_depth{tag}"] = .25 * x * (
                1 + depth * np.cos(2 * np.pi * frequency * t))
    phase = 2 * np.pi * (16 * t + .5 * (48 / 4) * t ** 2)
    result["chirp16to64_depth06"] = .25 * x * (1 + .6 * np.cos(phase))
    if tuple(result) != CONDITIONS:
        raise ValueError("intervention set differs from the exact eight frozen conditions")
    return result


def finite_difference(a, b):
    return float(a - b) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) else None


def invariant_comparison(reference, other):
    common = [(reference["features"][k], other["features"][k]) for k in candidate.FEATURE_NAMES]
    errors = [abs(a - b) for a, b in common if a is not None and b is not None
              and np.isfinite(a) and np.isfinite(b)]
    signatures = lambda r: (r["status"], [(b["status"], [w["status"] for w in b["windows"]]) for b in r["bands"]])
    return {"max_abs_finite_feature_error": float(max(errors)) if errors else None,
            "comparable_features": len(errors), "feature_denominator": 6,
            "feature_missingness_match": [a is None for a, _ in common] == [b is None for _, b in common],
            "status_match": signatures(reference) == signatures(other)}


def known_am_peak(result, band_index, target):
    """Dominant analysis peak; a nearest-bin lookup would be tautological."""
    band = result["bands"][band_index]
    windows = band["windows"]
    if len(windows) != 1:
        raise ValueError("four-second notes must have exactly one Q analysis window")
    window = windows[0]
    if window["status"] != "ok":
        return {"dominant_peak_hz": None, "absolute_error_hz": None, "within_1hz": None,
                "status": window["status"]}
    f, p = np.asarray(result["frequency_hz"]), np.asarray(window["modulation_power"])
    mask = (f >= 2) & (f < 128)
    peak = float(f[mask][np.argmax(p[mask])])
    return {"dominant_peak_hz": peak, "absolute_error_hz": abs(peak - target),
            "within_1hz": abs(peak - target) <= 1, "status": "ok"}


def near_power(result, band_index, target):
    w = result["bands"][band_index]["windows"][0]
    if w["status"] != "ok":
        return None
    frequency = np.asarray(result["frequency_hz"])
    return float(np.sum(np.asarray(w["modulation_power"])[np.abs(frequency - target) <= 1]))


def summarize_note(row, results):
    bands = []
    for index, name in enumerate(candidate.BAND_NAMES):
        b = {"name": name, "condition_status": {c: r["bands"][index]["windows"][0]["status"]
                                                for c, r in results.items()}}
        for target in (8, 48):
            b[f"am{target}_depth06_peak"] = known_am_peak(results[f"am{target}_depth06"], index, target)
            low = near_power(results[f"am{target}_depth02"], index, target)
            high = near_power(results[f"am{target}_depth06"], index, target)
            b[f"am{target}_near_power_depth02"] = low
            b[f"am{target}_near_power_depth06"] = high
            b[f"am{target}_near_power_ratio_depth06_over_depth02"] = (
                float(high / low) if low is not None and high is not None and low > 0 else None)
        feature = lambda condition, metric: results[condition]["features"][f"Q_{name}_{metric}_median"]
        b["fast_fraction_delta_am48_minus_am8_depth06"] = finite_difference(
            feature("am48_depth06", "fast_fraction"), feature("am8_depth06", "fast_fraction"))
        b["entropy_delta_chirp_minus_am48_depth06"] = finite_difference(
            feature("chirp16to64_depth06", "entropy"), feature("am48_depth06", "entropy"))
        bands.append(b)
    return {"id": row["id"], "instrument_str": row["metadata"]["instrument_str"],
            "metadata": row["metadata"],
            "bands": bands, "common_gain": invariant_comparison(results["baseline"], results["common_gain"]),
            "polarity": invariant_comparison(results["baseline"], results["polarity"])}


def balanced_summary(values_by_instrument):
    instrument_means = {}
    valid_notes = 0
    for instrument, values in values_by_instrument.items():
        finite = [float(v) for v in values if v is not None and np.isfinite(v)]
        valid_notes += len(finite)
        instrument_means[instrument] = {"mean": float(np.mean(finite)) if finite else None,
                                        "valid_notes": len(finite), "note_denominator": 2}
    means = [r["mean"] for r in instrument_means.values() if r["mean"] is not None]
    return {"instrument_balanced_mean": float(np.mean(means)) if means else None,
            "covered_instruments": len(means), "instrument_denominator": len(values_by_instrument),
            "valid_notes": valid_notes, "note_denominator": 2 * len(values_by_instrument),
            "per_instrument": instrument_means}


def aggregate_notes(notes, pilot_instruments):
    result = {"bands": {}, "invariance": {}}
    def aggregate(getter):
        groups = {name: [] for name in pilot_instruments}
        for note in notes:
            groups[note["instrument_str"]].append(getter(note))
        return balanced_summary(groups)
    for i, name in enumerate(candidate.BAND_NAMES):
        metrics = ("am8_near_power_ratio_depth06_over_depth02", "am48_near_power_ratio_depth06_over_depth02",
                   "fast_fraction_delta_am48_minus_am8_depth06", "entropy_delta_chirp_minus_am48_depth06")
        entries = {metric: aggregate(lambda n, m=metric: n["bands"][i][m]) for metric in metrics}
        for target in (8, 48):
            entries[f"am{target}_depth06_peak_within_1hz_rate"] = aggregate(
                lambda n, t=target: n["bands"][i][f"am{t}_depth06_peak"]["within_1hz"])
        for condition in CONDITIONS:
            entries[f"{condition}_quality_valid_rate"] = aggregate(
                lambda n, c=condition: n["bands"][i]["condition_status"][c] == "ok")
        result["bands"][name] = entries
    for condition in ("common_gain", "polarity"):
        result["invariance"][condition] = {metric: aggregate(lambda n, m=metric: n[condition][m])
            for metric in ("max_abs_finite_feature_error", "status_match", "feature_missingness_match")}
    return result


def verify_draft_publication(draft_file, freeze_sha256):
    if not freeze_sha256 or sha256_file(draft_file) != freeze_sha256:
        raise ValueError("explicit reviewed freeze SHA256 is missing or mismatched")
    commit_path = draft_file.parent / "COMMIT.json"
    if not commit_path.is_file():
        raise ValueError("draft publication has no COMMIT")
    commit = read_json(commit_path)
    if (draft_file.name != "draft.json" or commit.get("status") != "committed"
            or commit.get("kind") != "development_pilot_draft"
            or commit.get("draft_sha256") != freeze_sha256
            or commit.get("products") != {"draft.json": product_fingerprint(draft_file)}
            or {p.name for p in draft_file.parent.iterdir()} != {"draft.json", "COMMIT.json"}):
        raise ValueError("draft COMMIT inventory or freeze hash mismatch")
    return product_fingerprint(commit_path)


def run(draft_file, freeze_sha256, output_dir):
    draft_file = Path(draft_file).resolve(strict=True)
    draft_commit_start = verify_draft_publication(draft_file, freeze_sha256)
    frozen = read_json(draft_file)
    if frozen.get("version") != VERSION or frozen.get("policy") != POLICY or frozen.get("features_extracted") is not False:
        raise ValueError("freeze policy or version mismatch")
    output = new_output_path(output_dir, frozen["sources"]["source_dir"])
    if str(output) != frozen["planned_run_output_dir"]:
        raise ValueError("output must exactly match the frozen new output directory")
    protocol = frozen["bindings"]["protocol"]["path"]
    binding_start = code_bindings(protocol)
    if binding_start != frozen["bindings"]:
        raise ValueError("code, candidate, protocol or runtime differs from freeze")
    sources, by_id = verify_source(frozen["sources"]["source_dir"])
    if sources != frozen["sources"] or selected_inventory(by_id) != frozen["selection"]:
        raise ValueError("source bytes or deterministic selection differs from freeze")
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "freeze.json", frozen)
    (output / "derivatives").mkdir()
    (output / "candidate_json").mkdir()
    note_summaries = []
    source_reads = []
    for row in frozen["selection"]["selected_notes"]:
        # Only selected pilot waveforms are decoded. Reserved notes never enter Q.
        sample_rate, pcm = wavfile.read(row["path"])
        if sample_rate != 16000 or pcm.shape != (64000,) or pcm.dtype != np.int16:
            raise ValueError(f"source decode differs from fixed PCM16 contract: {row['id']}")
        # Acquisition hashes little-endian signed PCM16, before float conversion.
        pcm_digest = hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest()
        if pcm_digest != row["pcm_sha256"]:
            raise ValueError(f"decoded PCM hash differs from source inventory: {row['id']}")
        x = pcm.astype(np.float64) / 32768.0
        results = {}
        for condition, y in derivatives(x).items():
            basename = f"{row['id']}__{condition}"
            wav_path = output / "derivatives" / (basename + ".wav")
            wavfile.write(wav_path, 16000, y.astype(np.float64, copy=False))
            decoded_rate, decoded = wavfile.read(wav_path)
            if decoded_rate != 16000 or decoded.dtype != np.float64 or not np.array_equal(decoded, y):
                raise ValueError(f"float64 derivative roundtrip mismatch: {basename}")
            measurement = candidate.extract_modulation_candidate(decoded, 16000)
            if measurement["window_count"] != 1:
                raise ValueError("full four-second source must produce one Q window")
            results[condition] = measurement
            write_json_new(output / "candidate_json" / (basename + ".json"), {
                "note_id": row["id"], "instrument_str": row["metadata"]["instrument_str"],
                "condition": condition, "source_file_sha256": row["file_sha256"],
                "derivative": product_fingerprint(wav_path), "measurement": measurement})
        note_summaries.append(summarize_note(row, results))
        source_reads.append(row["id"])
    summary = {"status": POLICY["status"], "freeze_sha256": freeze_sha256,
               "notes": note_summaries,
               "instrument_balanced_descriptive_aggregates": aggregate_notes(
                   note_summaries, frozen["selection"]["pilot_instrument_ids"]),
               "no_numeric_admission_threshold": True, "external_gate_passed": False,
               "interpretation": "Development pilot only; natural envelopes and filter sidebands can change known-AM responses."}
    summary["instrument_balanced_source_and_family_strata"] = {}
    for field in ("instrument_source_str", "instrument_family_str"):
        strata = {}
        for value in sorted({note["metadata"][field] for note in note_summaries}):
            subset = [note for note in note_summaries if note["metadata"][field] == value]
            instruments = sorted({note["instrument_str"] for note in subset})
            strata[value] = aggregate_notes(subset, instruments)
        summary["instrument_balanced_source_and_family_strata"][field] = strata
    write_json_new(output / "paired_summaries.json", summary)
    sources_end, _ = verify_source(frozen["sources"]["source_dir"])
    bindings_end = code_bindings(protocol)
    draft_commit_end = verify_draft_publication(draft_file, freeze_sha256)
    if sources_end != sources or bindings_end != binding_start or draft_commit_end != draft_commit_start:
        raise ValueError("source, code, runtime, protocol or freeze changed during execution; no COMMIT")
    write_json_new(output / "run_receipt.json", {"status": POLICY["status"],
        "freeze_sha256": freeze_sha256, "code_bindings_start": binding_start, "code_bindings_end": bindings_end,
        "draft_commit_start": draft_commit_start, "draft_commit_end": draft_commit_end,
        "source_bindings_verified_start_and_end": True, "decoded_source_note_ids": source_reads,
        "reserved_instrument_ids": frozen["selection"]["reserved_instrument_ids"],
        "reserved_waveforms_decoded_or_extracted": False, "notes": len(source_reads),
        "conditions_per_note": 8, "derivative_wavs": len(source_reads) * 8,
        "candidate_json_files": len(source_reads) * 8, "ai_human_labels_assigned": False,
        "classifier_fits": 0, "codecs": False, "external_gate_passed": False})
    commit_directory(output, {"kind": "development_pilot_result", "pilot_status": POLICY["status"],
                              "freeze_sha256": freeze_sha256})
    return {"output_dir": str(output), "status": POLICY["status"], "notes": len(source_reads)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("draft")
    for flag in ("source-dir", "protocol", "draft-dir", "run-output-dir"):
        prepare.add_argument("--" + flag, required=True)
    execute = sub.add_parser("run")
    for flag in ("draft", "freeze-sha256", "output-dir"):
        execute.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    result = (draft(args.source_dir, args.protocol, args.draft_dir, args.run_output_dir)
              if args.command == "draft" else run(args.draft, args.freeze_sha256, args.output_dir))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
