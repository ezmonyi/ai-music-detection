#!/usr/bin/env python3
"""Held-out NSynth Q measurement gate; independent of classifier admission.

Definitions are frozen in MODULATION_NSYNTH_GATE_PROTOCOL_EN.md. No fitting or
classification is performed. Scientific failures are published, not retuned.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import modulation_nsynth_pilot_v1 as pilot

VERSION = "modulation_nsynth_heldout_measurement_gate_v1_20260907"
PILOT_RUNNER_SHA256 = "eef6c3d81d6ba8e057da6e51aebe0e27522b62900de8d948430f9ddbaa091c33"
PILOT_FREEZE_SHA256 = "020c806a5785176cfbffd26a0d0c6ec6fa1f28bc8a0a1c1bb004ce7a0ec7f222"
PILOT_COMMIT_SHA256 = "8ea44272d4edcef94ef3dc1e4a24e8c6b6aad8fd54e08e57c8e8c1b6a8c095d5"
PILOT_ACCEPTANCE_SHA256 = "ebfe8b50cb258bcd901b51da8f7fbdbc97353466d7286f66aa08ac82a573c8a1"
PILOT_AUDIT_SHA256 = "a8d9146352dd0effe679543fe65bdf3db712d390939e1d6333cbbd775575a380"
PILOT_DSP_AUDIT_SHA256 = "69c823d227be9cbe1bffacd18b1cc8055d2ece887ef7948cffdf8b5c0775d259"
CONDITIONS = pilot.CONDITIONS
CODECS = {
    "mp3_128k": {"encoder": "libmp3lame", "requested_bitrate": "128k", "extension": ".mp3", "codec_name": "mp3"},
    "aac_lc_64k": {"encoder": "aac", "requested_bitrate": "64k", "extension": ".m4a", "codec_name": "aac"},
}
VIEWS = ("original", *CODECS)
POLICY = {
    "samples": 64000, "sample_rate": 16000, "notes": 52, "instruments": 26,
    "conditions": list(CONDITIONS), "views": list(VIEWS), "codecs": CODECS,
    "minimum_note_coverage": .80, "minimum_instrument_coverage": .80,
    "minimum_success_rate": .90, "peak_tolerance_hz": 1.,
    "fast_fraction_target_minimum": .25, "entropy_target_minimum": .10,
    "depth_ratio_strict_minimum": 1., "invariance_maximum_error": 1e-9,
    "codec_nuisance_usual_maximum_ratio": .10, "codec_nuisance_any_maximum_ratio": 1.,
    "eligibility_agreement_minimum": .90,
    "analysis_slice": "first_64000_no_padding_no_lag_optimization",
    "all_bands_required": True, "classifier_fits": 0,
}


def verified_prior_pilot(pilot_dir):
    """Verify accepted pilot bytes and recorded scope without extracting Q."""
    root = Path(pilot_dir).resolve(strict=True)
    if pilot.sha256_file(pilot.__file__) != PILOT_RUNNER_SHA256:
        raise ValueError("immutable pilot runner changed")
    if pilot.sha256_file(root / "COMMIT.json") != PILOT_COMMIT_SHA256:
        raise ValueError("pilot result COMMIT differs from parent acceptance")
    commit = pilot.read_json(root / "COMMIT.json")
    if (commit.get("status") != "committed" or commit.get("kind") != "development_pilot_result"
            or commit.get("freeze_sha256") != PILOT_FREEZE_SHA256):
        raise ValueError("pilot publication scope mismatch")
    actual_paths = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual_paths != set(commit["products"]) | {"COMMIT.json"}:
        raise ValueError("pilot publication file set mismatch")
    for name, expected in commit["products"].items():
        if pilot.product_fingerprint(pilot.safe_child(root, name)) != expected:
            raise ValueError(f"pilot product changed: {name}")
    if pilot.sha256_file(root / "freeze.json") != PILOT_FREEZE_SHA256:
        raise ValueError("pilot freeze bytes mismatch")
    frozen = pilot.read_json(root / "freeze.json")
    receipt = pilot.read_json(root / "run_receipt.json")
    if (receipt.get("reserved_waveforms_decoded_or_extracted") is not False
            or receipt.get("external_gate_passed") is not False
            or receipt.get("classifier_fits") != 0 or receipt.get("notes") != 54):
        raise ValueError("pilot receipt does not preserve the held-out scope")
    audit_dir = Path(__file__).resolve().parent.parent / "audit"
    audit_pins = {"modulation_nsynth_pilot_parent_acceptance_v1.json": PILOT_ACCEPTANCE_SHA256,
                  "modulation_nsynth_pilot_independent_v1.json": PILOT_AUDIT_SHA256,
                  "modulation_nsynth_pilot_dsp_replay_v1.json": PILOT_DSP_AUDIT_SHA256}
    bound_audits = {}
    for name, expected in audit_pins.items():
        record = pilot.fingerprint(audit_dir / name)
        if record["sha256"] != expected:
            raise ValueError(f"pilot acceptance/audit changed: {name}")
        bound_audits[name] = record
    return frozen, {"pilot_dir": str(root), "commit": pilot.fingerprint(root / "COMMIT.json"),
                    "freeze": pilot.fingerprint(root / "freeze.json"), "audits": bound_audits}


def reserved_roster(by_id, prior_selection):
    """Exactly two unique PCM notes from each of the original 26 reserves."""
    if pilot.selected_inventory(by_id) != prior_selection:
        raise ValueError("source inventory no longer reproduces the accepted pilot selection")
    reserved = prior_selection["reserved_instrument_ids"]
    if len(reserved) != 26 or len(set(reserved)) != 26:
        raise ValueError("gate must retain exactly the original 26 reserved instruments")
    selected = []
    for instrument in reserved:
        candidates = sorted((r for r in by_id.values() if r["metadata"]["instrument_str"] == instrument),
            key=lambda r: (abs(r["metadata"]["pitch"] - 60),
                           hashlib.sha256(("Q-note-20260907|" + r["id"]).encode()).hexdigest(), r["id"]))
        seen_pcm = set()
        for row in candidates:
            if row["pcm_sha256"] in seen_pcm:
                continue
            selected.append(row)
            seen_pcm.add(row["pcm_sha256"])
            if len(seen_pcm) == 2:
                break
        if len(seen_pcm) != 2:
            raise ValueError(f"insufficient distinct-PCM gate notes for {instrument}; no replacement")
    if len(selected) != 52:
        raise ValueError("gate roster must contain exactly 52 notes")
    return {"gate_instrument_ids": reserved, "selected_notes": selected,
            "excluded_development_instrument_ids": prior_selection["pilot_instrument_ids"],
            "source_partition": "original_26_reserved_instruments"}


def command_report(command, *, log_path=None, timeout=120):
    """Preserve full outputs, including failures; no shell invocation."""
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        report = {"command": command, "returncode": completed.returncode,
                  "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired as exc:
        decode = lambda value: value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        report = {"command": command, "returncode": None, "timeout_seconds": timeout,
                  "stdout": decode(exc.stdout), "stderr": decode(exc.stderr)}
    if log_path is not None:
        report["elapsed_seconds"] = time.monotonic() - started
        pilot.write_json_new(log_path, report)
    if report["returncode"] != 0:
        raise RuntimeError(f"codec/tool command failed, full log={log_path}: {command}")
    return report


def toolchain_bindings(ffmpeg, ffprobe):
    ffmpeg = str(Path(ffmpeg).resolve(strict=True)); ffprobe = str(Path(ffprobe).resolve(strict=True))
    return {"ffmpeg": pilot.fingerprint(ffmpeg), "ffprobe": pilot.fingerprint(ffprobe),
            "reports": {"ffmpeg_version": command_report([ffmpeg, "-hide_banner", "-version"]),
                        "ffprobe_version": command_report([ffprobe, "-hide_banner", "-version"]),
                        "mp3_encoder": command_report([ffmpeg, "-hide_banner", "-h", "encoder=libmp3lame"]),
                        "aac_encoder": command_report([ffmpeg, "-hide_banner", "-h", "encoder=aac"])}}


def gate_bindings(protocol, prior_frozen, ffmpeg, ffprobe):
    code = Path(__file__).resolve().parent
    accepted_helpers = pilot.code_bindings(prior_frozen["bindings"]["protocol"]["path"])
    if accepted_helpers != prior_frozen["bindings"]:
        raise ValueError("accepted pilot helpers, candidate or numerical runtime changed")
    return {"runner": pilot.fingerprint(__file__), "tests": pilot.fingerprint(code / "test_modulation_nsynth_gate_v1.py"),
            "protocol": pilot.fingerprint(protocol), "accepted_pilot_helpers": accepted_helpers,
            "codec_toolchain": toolchain_bindings(ffmpeg, ffprobe)}


def gate_inputs(source_dir, pilot_dir):
    prior_frozen, prior_bindings = verified_prior_pilot(pilot_dir)
    sources, by_id = pilot.verify_source(source_dir)
    if sources != prior_frozen["sources"]:
        raise ValueError("source publication differs from the accepted pilot source")
    return prior_frozen, {"source": sources, "prior_pilot": prior_bindings,
                          "selection": reserved_roster(by_id, prior_frozen["selection"])}


def draft(source_dir, pilot_dir, protocol, ffmpeg, ffprobe, draft_dir, run_output_dir):
    draft_path = pilot.new_output_path(draft_dir, source_dir)
    output = pilot.new_output_path(run_output_dir, source_dir)
    if draft_path == output or draft_path.is_relative_to(output) or output.is_relative_to(draft_path):
        raise ValueError("draft and run output directories must be separate")
    prior, inputs = gate_inputs(source_dir, pilot_dir)
    bindings = gate_bindings(protocol, prior, ffmpeg, ffprobe)
    frozen = {"version": VERSION, "stage": "draft_requires_explicit_reviewed_sha256", "policy": POLICY,
              "inputs": inputs, "bindings": bindings, "planned_run_output_dir": str(output),
              "features_extracted": False, "reserved_note_codec_processing_performed": False}
    # Tool reports contain no source-note processing. Verify all source and
    # prior publications again before publishing the no-measurement draft.
    prior_end, inputs_end = gate_inputs(source_dir, pilot_dir)
    if inputs_end != inputs or gate_bindings(protocol, prior_end, ffmpeg, ffprobe) != bindings:
        raise ValueError("source, code or toolchain changed during draft")
    draft_path.mkdir(parents=True, exist_ok=False)
    pilot.write_json_new(draft_path / "draft.json", frozen)
    freeze = pilot.sha256_file(draft_path / "draft.json")
    pilot.commit_directory(draft_path, {"kind": "heldout_measurement_gate_draft", "draft_sha256": freeze})
    return {"draft": str(draft_path / "draft.json"), "freeze_sha256_for_review": freeze, "notes": 52,
            "planned_measurements": 1248, "features_extracted": False}


def verify_gate_draft(path, freeze_sha256):
    path = Path(path).resolve(strict=True)
    if path.name != "draft.json" or not freeze_sha256 or pilot.sha256_file(path) != freeze_sha256:
        raise ValueError("explicit reviewed gate freeze SHA256 missing or mismatched")
    commit_path = path.parent / "COMMIT.json"
    if not commit_path.is_file():
        raise ValueError("gate draft has no COMMIT")
    commit = pilot.read_json(commit_path)
    if (commit.get("status") != "committed" or commit.get("kind") != "heldout_measurement_gate_draft"
            or commit.get("draft_sha256") != freeze_sha256
            or commit.get("products") != {"draft.json": pilot.product_fingerprint(path)}
            or {p.name for p in path.parent.iterdir()} != {"draft.json", "COMMIT.json"}):
        raise ValueError("gate draft COMMIT inventory mismatch")
    return pilot.product_fingerprint(commit_path)


def zero_lag_correlation(original, decoded):
    a, b = original - np.mean(original), decoded - np.mean(decoded)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(np.dot(a, b) / norm, -1., 1.)) if norm > 0 else None


def validated_analysis_decode(path):
    rate, decoded = wavfile.read(path)
    if rate != 16000 or decoded.dtype != np.float64 or decoded.ndim != 1 or not np.all(np.isfinite(decoded)):
        raise ValueError("codec full decode must be finite native 16k mono float64")
    if decoded.size < 64000:
        raise ValueError(f"codec decode too short ({decoded.size}); padding is forbidden")
    return decoded, decoded[:64000].copy()


def codec_roundtrip(raw_path, output, stem, view, toolchain, original):
    spec = CODECS[view]
    ffmpeg, ffprobe = toolchain["ffmpeg"]["path"], toolchain["ffprobe"]["path"]
    encoded = output / "encoded" / (stem + "__" + view + spec["extension"])
    full_path = output / "decoded_full" / (stem + "__" + view + ".wav")
    analysis_path = output / "analysis" / (stem + "__" + view + ".wav")
    prefix = [ffmpeg, "-hide_banner", "-nostdin", "-n"]
    encode = prefix + ["-i", str(raw_path), "-map", "0:a:0", "-vn", "-map_metadata", "-1",
                      "-c:a", spec["encoder"], "-b:a", spec["requested_bitrate"], "-ar", "16000", "-ac", "1",
                      "-threads", "1", "-fflags", "+bitexact"]
    if view == "aac_lc_64k":
        encode += ["-profile:a", "aac_low"]
    encode += [str(encoded)]
    log_stem = output / "command_logs" / (stem + "__" + view)
    command_report(encode, log_path=Path(str(log_stem) + "__encode.json"))
    probe = command_report([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_streams",
                            "-show_format", "-of", "json", str(encoded)],
                           log_path=Path(str(log_stem) + "__probe.json"))
    metadata = json.loads(probe["stdout"])
    streams = metadata.get("streams", [])
    if (len(streams) != 1 or streams[0].get("codec_name") != spec["codec_name"]
            or int(streams[0].get("sample_rate", 0)) != 16000 or streams[0].get("channels") != 1
            or (view == "aac_lc_64k" and streams[0].get("profile") != "LC")):
        raise ValueError("encoded stream differs from frozen native codec/rate/channel/profile contract")
    # No -ar/-ac conversion at decode: a wrong native stream must fail above.
    decode = prefix + ["-i", str(encoded), "-map", "0:a:0", "-vn", "-map_metadata", "-1",
                       "-c:a", "pcm_f64le", "-threads", "1", "-fflags", "+bitexact", str(full_path)]
    command_report(decode, log_path=Path(str(log_stem) + "__decode.json"))
    full, analysis = validated_analysis_decode(full_path)
    wavfile.write(analysis_path, 16000, analysis)
    return analysis_path, {"view": view, "requested_bitrate": spec["requested_bitrate"],
        "actual_stream_bit_rate": streams[0].get("bit_rate"), "stream_metadata": metadata,
        "encoded": {"path": str(encoded.relative_to(output)), **pilot.product_fingerprint(encoded)},
        "full_decode": {"path": str(full_path.relative_to(output)), **pilot.product_fingerprint(full_path)},
        "analysis": {"path": str(analysis_path.relative_to(output)), **pilot.product_fingerprint(analysis_path)},
        "full_decoded_frames": int(full.size), "analysis_frames": 64000, "discarded_tail_frames": int(full.size-64000),
        "analysis_start_frame": 0, "zero_lag_correlation": zero_lag_correlation(original, analysis),
        "lag_optimization_performed": False, "padding_performed": False}


def instrument_endpoint(notes, instruments, getter):
    grouped = {instrument: [] for instrument in instruments}
    for note in notes:
        grouped[note["instrument_str"]].append(getter(note))
    if any(len(values) != 2 for values in grouped.values()):
        raise ValueError("each gate instrument must retain exactly two notes")
    aggregate = pilot.balanced_summary(grouped)
    aggregate["note_coverage"] = aggregate["valid_notes"] / aggregate["note_denominator"]
    aggregate["instrument_coverage"] = aggregate["covered_instruments"] / aggregate["instrument_denominator"]
    return aggregate


def coverage_pass(endpoint):
    return endpoint["note_coverage"] >= .80 and endpoint["instrument_coverage"] >= .80


def codec_nuisance(note_views, codec):
    original, changed = note_views["original"], note_views[codec]
    result = {"features": {}, "band_eligibility_signature_match": {}}
    for i, name in enumerate(pilot.candidate.BAND_NAMES):
        a, b = original["bands"][i], changed["bands"][i]
        result["band_eligibility_signature_match"][name] = all(
            (a["condition_status"][c] == "ok") == (b["condition_status"][c] == "ok") for c in CONDITIONS)
        for metric, target in (("fast_fraction", "fast_fraction_delta_am48_minus_am8_depth06"),
                               ("entropy", "entropy_delta_chirp_minus_am48_depth06")):
            key = f"Q_{name}_{metric}_median"
            pairs = [(original["condition_features"][c][key], changed["condition_features"][c][key]) for c in CONDITIONS]
            errors = [abs(x-y) for x, y in pairs if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
            numerator = float(max(errors)) if len(errors) == 8 else None
            denominator = a[target]
            reason = ("incomplete_eight_condition_pairs" if numerator is None else
                      "missing_target_response" if denominator is None else
                      "nonpositive_target_response" if denominator <= 0 else "ok")
            result["features"][key] = {"max_eight_condition_absolute_error": numerator,
                "original_target_response": denominator, "comparable_conditions": len(errors),
                "condition_denominator": 8, "ratio": float(numerator/denominator) if reason == "ok" else None,
                "status": reason}
    return result


def decisions(notes, instruments):
    if len(notes) != 52 or len(instruments) != 26:
        raise ValueError("gate decision requires all 52 notes and 26 instruments")
    criteria = []
    def add(identifier, passed, **evidence):
        criteria.append({"id": identifier, "passed": bool(passed), **evidence})
    for view in VIEWS:
        for i, band in enumerate(pilot.candidate.BAND_NAMES):
            for condition in CONDITIONS:
                endpoint = instrument_endpoint(notes, instruments, lambda n, c=condition:
                    True if n["views"][view]["bands"][i]["condition_status"][c] == "ok" else None)
                add(f"A/{view}/{band}/{condition}/eligibility", coverage_pass(endpoint),
                    endpoint=endpoint, required_note_coverage=.80, required_instrument_coverage=.80)
            getters = {
                "am8_peak": lambda n: n["views"][view]["bands"][i]["am8_depth06_peak"]["within_1hz"],
                "am48_peak": lambda n: n["views"][view]["bands"][i]["am48_depth06_peak"]["within_1hz"],
                "fast_fraction_change": lambda n: n["views"][view]["bands"][i]["fast_fraction_delta_am48_minus_am8_depth06"],
                "entropy_change": lambda n: n["views"][view]["bands"][i]["entropy_delta_chirp_minus_am48_depth06"],
                "am8_depth_increase": lambda n: n["views"][view]["bands"][i]["am8_near_power_ratio_depth06_over_depth02"],
                "am48_depth_increase": lambda n: n["views"][view]["bands"][i]["am48_near_power_ratio_depth06_over_depth02"],
            }
            for name, getter in getters.items():
                predicate = (lambda v: v >= .25) if name == "fast_fraction_change" else (
                    (lambda v: v >= .10) if name == "entropy_change" else (
                    (lambda v: v > 1.) if "depth" in name else (lambda v: bool(v))))
                raw = instrument_endpoint(notes, instruments, getter)
                success = instrument_endpoint(notes, instruments, lambda n, g=getter, p=predicate:
                    None if g(n) is None else p(g(n)))
                rate = success["instrument_balanced_mean"]
                add(f"A/{view}/{band}/{name}", coverage_pass(success) and rate is not None and rate >= .90,
                    endpoint=raw, success=success, required_success_rate=.90, coverage_required=True)
    for condition in ("common_gain", "polarity"):
        comparisons = [n["views"]["original"][condition] for n in notes]
        finite_errors = [r["max_abs_finite_feature_error"] for r in comparisons if r["max_abs_finite_feature_error"] is not None]
        signatures = all(r["status_match"] and r["feature_missingness_match"] for r in comparisons)
        maximum = max(finite_errors) if finite_errors else None
        add(f"B/original/{condition}", signatures and all(error <= 1e-9 for error in finite_errors),
            all_52_status_and_missingness_signatures_match=signatures, maximum_finite_feature_error=maximum,
            comparable_notes=len(finite_errors), note_denominator=52, maximum_allowed_error=1e-9)
    for codec in CODECS:
        for feature in pilot.candidate.FEATURE_NAMES:
            getter = lambda n: n["codec_nuisance"][codec]["features"][feature]["ratio"]
            endpoint = instrument_endpoint(notes, instruments, getter)
            success = instrument_endpoint(notes, instruments, lambda n: None if getter(n) is None else getter(n) <= .10)
            available = [getter(n) for n in notes if getter(n) is not None]
            worst = max(available) if available else None
            rate = success["instrument_balanced_mean"]
            add(f"C/{codec}/{feature}/nuisance", coverage_pass(endpoint) and rate is not None and rate >= .90
                and all(value <= 1. for value in available), endpoint=endpoint, success=success,
                maximum_available_ratio=worst, usual_maximum_ratio=.10, any_maximum_ratio=1.,
                required_success_rate=.90, coverage_required=True)
        for name in pilot.candidate.BAND_NAMES:
            endpoint = instrument_endpoint(notes, instruments,
                lambda n: n["codec_nuisance"][codec]["band_eligibility_signature_match"][name])
            add(f"C/{codec}/{name}/eligibility_signature", endpoint["instrument_balanced_mean"] >= .90,
                endpoint=endpoint, minimum_agreement=.90)
    passed = all(item["passed"] for item in criteria)
    return {"criteria_passed": passed, "status": "criteria_passed_pending_independent_audit" if passed else "criteria_failed",
            "criteria": criteria, "failed_criterion_ids": [r["id"] for r in criteria if not r["passed"]],
            "all_bands_required": True, "independently_audited": False, "external_gate_passed": False,
            "classifier_admitted": False, "classification_fits": 0,
            "limitations": ["Codec ratio bounds apply only to notes with all eight finite condition pairs and a positive original target response; partial observations have no ratio nuisance bound.",
                            "Reserved instruments come from the reused NSynth sample-library corpus, not a new recording domain or performed-song set.",
                            "A criterion pass requires independent audit before any separately frozen classification extension."]}


def descriptive_strata(notes, instruments):
    result = {"overall": {}, "strata": {}}
    for view in VIEWS:
        result["overall"][view] = pilot.aggregate_notes([n["views"][view] for n in notes], instruments)
    for field in ("instrument_source_str", "instrument_family_str"):
        result["strata"][field] = {}
        for label in sorted({n["metadata"][field] for n in notes}):
            subset = [n for n in notes if n["metadata"][field] == label]
            ids = sorted({n["instrument_str"] for n in subset})
            result["strata"][field][label] = {view: pilot.aggregate_notes([n["views"][view] for n in subset], ids) for view in VIEWS}
    return result


def run(draft_file, freeze_sha256, output_dir):
    draft_file = Path(draft_file).resolve(strict=True)
    draft_commit = verify_gate_draft(draft_file, freeze_sha256)
    frozen = pilot.read_json(draft_file)
    if (frozen.get("version") != VERSION or frozen.get("policy") != POLICY
            or frozen.get("features_extracted") is not False
            or frozen.get("reserved_note_codec_processing_performed") is not False):
        raise ValueError("gate freeze definitions differ from implementation")
    source_dir = frozen["inputs"]["source"]["source_dir"]
    pilot_dir = frozen["inputs"]["prior_pilot"]["pilot_dir"]
    output = pilot.new_output_path(output_dir, source_dir)
    if str(output) != frozen["planned_run_output_dir"]:
        raise ValueError("output must equal the exact frozen new output path")
    prior, inputs = gate_inputs(source_dir, pilot_dir)
    protocol = frozen["bindings"]["protocol"]["path"]
    tools = frozen["bindings"]["codec_toolchain"]
    bindings = gate_bindings(protocol, prior, tools["ffmpeg"]["path"], tools["ffprobe"]["path"])
    if inputs != frozen["inputs"] or bindings != frozen["bindings"]:
        raise ValueError("gate source/code/toolchain bindings changed before run")
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json_new(output / "freeze.json", frozen)
    for directory in ("original", "encoded", "decoded_full", "analysis", "candidate_json", "command_logs"):
        (output / directory).mkdir()
    notes = []
    try:
        for row in inputs["selection"]["selected_notes"]:
            sr, pcm = wavfile.read(row["path"])
            if (sr != 16000 or pcm.dtype != np.int16 or pcm.shape != (64000,)
                    or hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest() != row["pcm_sha256"]):
                raise ValueError(f"held-out source PCM decode changed: {row['id']}")
            x = pcm.astype(np.float64) / 32768.
            measurements = {view: {} for view in VIEWS}
            codec_records = {}
            for condition, original in pilot.derivatives(x).items():
                stem = row["id"] + "__" + condition
                raw_path = output / "original" / (stem + ".wav")
                wavfile.write(raw_path, 16000, original)
                paths = {"original": raw_path}
                codec_records[condition] = {}
                for view in CODECS:
                    paths[view], codec_records[condition][view] = codec_roundtrip(raw_path, output, stem, view, tools, original)
                for view, path in paths.items():
                    rate, analyzed = wavfile.read(path)
                    if rate != 16000 or analyzed.shape != (64000,) or analyzed.dtype != np.float64 or not np.all(np.isfinite(analyzed)):
                        raise ValueError("analysis WAV contract violation")
                    if view == "original" and not np.array_equal(analyzed, original):
                        raise ValueError("original derivative float64 roundtrip mismatch")
                    measurement = pilot.candidate.extract_modulation_candidate(analyzed, 16000)
                    if measurement["window_count"] != 1:
                        raise ValueError("gate analysis must retain exactly one Q window")
                    measurements[view][condition] = measurement
                    pilot.write_json_new(output / "candidate_json" / (stem + "__" + view + ".json"), {
                        "note_id": row["id"], "instrument_str": row["metadata"]["instrument_str"],
                        "condition": condition, "view": view, "source_file_sha256": row["file_sha256"],
                        "analysis_waveform": {"path": str(path.relative_to(output)), **pilot.product_fingerprint(path)},
                        "measurement": measurement})
            note = {"id": row["id"], "instrument_str": row["metadata"]["instrument_str"], "metadata": row["metadata"],
                    "views": {}, "codec_records": codec_records, "codec_nuisance": {}}
            for view in VIEWS:
                note["views"][view] = pilot.summarize_note(row, measurements[view])
                note["views"][view]["condition_features"] = {c: measurements[view][c]["features"] for c in CONDITIONS}
            for codec in CODECS:
                note["codec_nuisance"][codec] = codec_nuisance(note["views"], codec)
            notes.append(note)
        instruments = inputs["selection"]["gate_instrument_ids"]
        summary = {"version": VERSION, "freeze_sha256": freeze_sha256, "notes": notes,
                   "decision": decisions(notes, instruments), "descriptive": descriptive_strata(notes, instruments)}
        pilot.write_json_new(output / "gate_results.json", summary)
        prior_end, inputs_end = gate_inputs(source_dir, pilot_dir)
        bindings_end = gate_bindings(protocol, prior_end, tools["ffmpeg"]["path"], tools["ffprobe"]["path"])
        if (inputs_end != inputs or bindings_end != bindings
                or verify_gate_draft(draft_file, freeze_sha256) != draft_commit):
            raise ValueError("source, pilot, code, runtime, codec toolchain or freeze changed during run")
        counts = {directory: len(list((output / directory).iterdir())) for directory in
                  ("original", "encoded", "decoded_full", "analysis", "candidate_json", "command_logs")}
        expected_counts = {"original": 416, "encoded": 832, "decoded_full": 832,
                           "analysis": 832, "candidate_json": 1248, "command_logs": 2496}
        if counts != expected_counts:
            raise ValueError(f"complete gate grid artifact count mismatch: {counts}")
        pilot.write_json_new(output / "run_receipt.json", {"status": "execution_completed",
            "freeze_sha256": freeze_sha256, "draft_commit": draft_commit,
            "bindings_start": bindings, "bindings_end": bindings_end,
            "input_bindings_rechecked_start_and_end": True, "decoded_source_note_ids": [n["id"] for n in notes],
            "gate_instrument_ids": instruments, "excluded_development_instrument_ids": inputs["selection"]["excluded_development_instrument_ids"],
            "notes": 52, "instruments": 26, "counts": counts, "decision_status": summary["decision"]["status"],
            "ai_human_labels_assigned": False, "classifier_fits": 0, "candidate_definition_changed": False,
            "padding_performed": False, "lag_optimization_performed": False, "external_gate_passed": False})
        pilot.commit_directory(output, {"kind": "heldout_measurement_gate_result", "freeze_sha256": freeze_sha256,
            "criteria_passed": summary["decision"]["criteria_passed"], "decision_status": summary["decision"]["status"]})
        return {"output_dir": str(output), "notes": 52, "measurements": 1248,
                "criteria_passed": summary["decision"]["criteria_passed"], "status": summary["decision"]["status"]}
    except Exception as exc:
        pilot.write_json_new(output / "EXECUTION_FAILURE.json", {"status": "execution_failed_no_commit",
            "freeze_sha256": freeze_sha256, "exception_type": type(exc).__name__, "message": str(exc),
            "completed_notes": len(notes), "scientific_gate_decision_available": False,
            "settings_changed_or_retried": False, "partial_artifacts_and_command_logs_retained": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("draft")
    for flag in ("source-dir", "pilot-dir", "protocol", "ffmpeg", "ffprobe", "draft-dir", "run-output-dir"):
        prepare.add_argument("--" + flag, required=True)
    execute = sub.add_parser("run")
    for flag in ("draft", "freeze-sha256", "output-dir"):
        execute.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    result = (draft(args.source_dir, args.pilot_dir, args.protocol, args.ffmpeg, args.ffprobe, args.draft_dir, args.run_output_dir)
              if args.command == "draft" else run(args.draft, args.freeze_sha256, args.output_dir))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
