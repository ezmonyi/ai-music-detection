#!/usr/bin/env python3
"""Independent gate audit: formula, codec linkage, stored-power and decision replay.

Does not import the producer or candidate. This is not an independent full-DSP
replay or a historical proof of non-access to held-out notes. No encoding occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.io import wavfile

HELPER_SHA = "414635980ceec5be09b74b9ff576065d5021193b1ca01fae9741771dbdee4ac0"
_helper_path = Path(__file__).with_name("audit_modulation_nsynth_pilot_v1.py")
if hashlib.sha256(_helper_path.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError("independent pilot audit helper pin changed")
_spec = importlib.util.spec_from_file_location("_immutable_q_audit_helper", _helper_path)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

VERSION = "independent_modulation_nsynth_gate_audit_v1_20260907"
GATE_VERSION = "modulation_nsynth_heldout_measurement_gate_v1_20260907"
PILOT_FREEZE = "020c806a5785176cfbffd26a0d0c6ec6fa1f28bc8a0a1c1bb004ce7a0ec7f222"
PILOT_COMMIT = "8ea44272d4edcef94ef3dc1e4a24e8c6b6aad8fd54e08e57c8e8c1b6a8c095d5"
AUDIT_PINS = {
    "modulation_nsynth_pilot_parent_acceptance_v1.json": "ebfe8b50cb258bcd901b51da8f7fbdbc97353466d7286f66aa08ac82a573c8a1",
    "modulation_nsynth_pilot_independent_v1.json": "a8d9146352dd0effe679543fe65bdf3db712d390939e1d6333cbbd775575a380",
    "modulation_nsynth_pilot_dsp_replay_v1.json": "69c823d227be9cbe1bffacd18b1cc8055d2ece887ef7948cffdf8b5c0775d259",
}
GATE_PINS = {
    "runner": "e2df986d19c6339078ddb16ebd47c7d88fefac60f951845d23b388e891eb2e56",
    "tests": "466e6680195e279f255db4c20a570ddd32a47e9de95ee2364bedc7f3d23e4abc",
    "protocol": "73fc93cfeb293a6da18d2d071eb2d45e8fcb312b8a3be2a2e8917e9ac678802c",
}
CODECS = {
    "mp3_128k": {"encoder": "libmp3lame", "requested_bitrate": "128k", "extension": ".mp3", "codec_name": "mp3"},
    "aac_lc_64k": {"encoder": "aac", "requested_bitrate": "64k", "extension": ".m4a", "codec_name": "aac"},
}
VIEWS = ("original", *CODECS)
POLICY = {
    "samples": 64000, "sample_rate": 16000, "notes": 52, "instruments": 26,
    "conditions": list(h.CONDITIONS), "views": list(VIEWS), "codecs": CODECS,
    "minimum_note_coverage": .80, "minimum_instrument_coverage": .80,
    "minimum_success_rate": .90, "peak_tolerance_hz": 1.,
    "fast_fraction_target_minimum": .25, "entropy_target_minimum": .10,
    "depth_ratio_strict_minimum": 1., "invariance_maximum_error": 1e-9,
    "codec_nuisance_usual_maximum_ratio": .10, "codec_nuisance_any_maximum_ratio": 1.,
    "eligibility_agreement_minimum": .90,
    "analysis_slice": "first_64000_no_padding_no_lag_optimization",
    "all_bands_required": True, "classifier_fits": 0,
}
COUNTS = {"original": 416, "encoded": 832, "decoded_full": 832,
          "analysis": 832, "candidate_json": 1248, "command_logs": 2496}


def bound_file(record):
    h.compare(h.file_record(record["path"]), {k: record[k] for k in ("bytes", "sha256")}, "bound file")


def call(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    h.require(result.returncode == 0, "independent read-only tool probe failed")
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def verify_tools(tools):
    h.require(set(tools) == {"ffmpeg", "ffprobe", "reports"}, "tool binding structure")
    for role in ("ffmpeg", "ffprobe"):
        bound_file(tools[role])
    ffmpeg, ffprobe = (tools[k]["path"] for k in ("ffmpeg", "ffprobe"))
    commands = {"ffmpeg_version": [ffmpeg, "-hide_banner", "-version"],
                "ffprobe_version": [ffprobe, "-hide_banner", "-version"],
                "mp3_encoder": [ffmpeg, "-hide_banner", "-h", "encoder=libmp3lame"],
                "aac_encoder": [ffmpeg, "-hide_banner", "-h", "encoder=aac"]}
    h.compare(tools["reports"], {k: call(v) for k, v in commands.items()}, "live tool reports")


def verify_bindings(bindings):
    h.require(set(bindings) == {"runner", "tests", "protocol", "accepted_pilot_helpers", "codec_toolchain"},
              "gate binding structure")
    for role in ("runner", "tests", "protocol"):
        h.require(bindings[role]["sha256"] == GATE_PINS[role], "unreviewed gate binding:" + role)
        bound_file(bindings[role])
    helpers = bindings["accepted_pilot_helpers"]
    h.verify_bindings(helpers)
    runtime = helpers["runtime"]
    for key, value in {"numpy_version": np.__version__, "scipy_version": scipy.__version__,
                       "python_version": platform.python_version(), "platform_system": platform.system(),
                       "platform_machine": platform.machine()}.items():
        h.compare(runtime[key], value, "active runtime:" + key)
    h.require(Path(sys.executable).resolve() == Path(runtime["python_executable"]["path"]).resolve(),
              "active Python differs from frozen runtime")
    for key, module in (("numpy", np), ("scipy", scipy), ("scipy.io.wavfile", wavfile)):
        h.require(Path(module.__file__).resolve() == Path(runtime["module_files"][key]["path"]).resolve(),
                  "active module differs from frozen runtime")
    verify_tools(bindings["codec_toolchain"])


def verify_inputs(frozen):
    inputs = frozen["inputs"]
    prior = inputs["prior_pilot"]
    root = Path(prior["pilot_dir"]).resolve(strict=True)
    publication = h.verify_products(root, PILOT_COMMIT)
    h.require(publication["kind"] == "development_pilot_result" and publication["freeze_sha256"] == PILOT_FREEZE,
              "accepted pilot scope")
    h.require(h.digest(root / "freeze.json") == PILOT_FREEZE, "pilot frozen bytes")
    for role, path in (("commit", root / "COMMIT.json"), ("freeze", root / "freeze.json")):
        h.compare(prior[role], {"path": str(path), **h.file_record(path)}, "prior pilot linkage")
    h.require(set(prior["audits"]) == set(AUDIT_PINS), "prior audit inventory")
    for name, pin in AUDIT_PINS.items():
        record = prior["audits"][name]
        h.require(record["sha256"] == pin, "prior audit pin")
        bound_file(record)
    pilot = h.load(root / "freeze.json")
    h.compare(frozen["bindings"]["accepted_pilot_helpers"], pilot["bindings"], "accepted helpers")
    h.compare(inputs["source"], pilot["sources"], "accepted source")
    h.verify_roster(pilot)  # Source hashes/metadata only; no note decoding here.
    receipt = h.load(root / "run_receipt.json")
    for key, value in {"reserved_waveforms_decoded_or_extracted": False, "external_gate_passed": False,
                       "classifier_fits": 0, "notes": 54}.items():
        h.compare(receipt[key], value, "prior scope:" + key)
    source = Path(inputs["source"]["source_dir"])
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines() if line]
    reserved = pilot["selection"]["reserved_instrument_ids"]
    h.require(len(reserved) == len(set(reserved)) == 26, "26 distinct original reserves required")
    selected = []
    for instrument in reserved:
        ordered = sorted((row for row in rows if row["metadata"]["instrument_str"] == instrument),
                         key=lambda r: (abs(r["metadata"]["pitch"]-60),
                             hashlib.sha256(("Q-note-20260907|" + r["id"]).encode()).hexdigest(), r["id"]))
        seen = set()
        for row in ordered:
            if row["pcm_sha256"] not in seen:
                selected.append(row)
                seen.add(row["pcm_sha256"])
            if len(seen) == 2:
                break
        h.require(len(seen) == 2, "insufficient held-out distinct PCM support")
    expected = {"gate_instrument_ids": reserved, "selected_notes": selected,
                "excluded_development_instrument_ids": pilot["selection"]["pilot_instrument_ids"],
                "source_partition": "original_26_reserved_instruments"}
    h.compare(inputs["selection"], expected, "independent gate roster")
    h.require(len(selected) == 52, "52 selected notes required")
    return selected


def endpoint(notes, instruments, getter):
    result = h.weighted(notes, instruments, getter)
    result["note_coverage"] = result["valid_notes"] / result["note_denominator"]
    result["instrument_coverage"] = result["covered_instruments"] / result["instrument_denominator"]
    return result


def sufficient(value):
    return value["note_coverage"] >= .8 and value["instrument_coverage"] >= .8


def nuisance(views, codec):
    before, after = views["original"], views[codec]
    result = {"features": {}, "band_eligibility_signature_match": {}}
    for index, band in enumerate(h.BAND_NAMES):
        a, b = before["bands"][index], after["bands"][index]
        result["band_eligibility_signature_match"][band] = all(
            (a["condition_status"][c] == "ok") == (b["condition_status"][c] == "ok") for c in h.CONDITIONS)
        for metric, effect in (("fast_fraction", "fast_fraction_delta_am48_minus_am8_depth06"),
                               ("entropy", "entropy_delta_chirp_minus_am48_depth06")):
            feature = f"Q_{band}_{metric}_median"
            errors = []
            for condition in h.CONDITIONS:
                x, y = [v["condition_features"][condition][feature] for v in (before, after)]
                if x is not None and y is not None:
                    h.require(math.isfinite(x) and math.isfinite(y), "nonfinite descriptor")
                    errors.append(abs(x-y))
            numerator = max(errors) if len(errors) == 8 else None
            denominator = a[effect]
            status = ("incomplete_eight_condition_pairs" if numerator is None else
                      "missing_target_response" if denominator is None else
                      "nonpositive_target_response" if denominator <= 0 else "ok")
            result["features"][feature] = {"max_eight_condition_absolute_error": numerator,
                "original_target_response": denominator, "comparable_conditions": len(errors),
                "condition_denominator": 8, "ratio": numerator/denominator if status == "ok" else None,
                "status": status}
    return result


def decision(notes, instruments):
    h.require(len(notes) == 52 and len(instruments) == len(set(instruments)) == 26, "complete gate decision roster")
    criteria = []
    def add(key, passed, **evidence):
        criteria.append({"id": key, "passed": bool(passed), **evidence})
    for view in VIEWS:
        for index, band in enumerate(h.BAND_NAMES):
            def band_value(note):
                return note["views"][view]["bands"][index]
            for condition in h.CONDITIONS:
                value = endpoint(notes, instruments, lambda n: 1. if band_value(n)["condition_status"][condition] == "ok" else None)
                add(f"A/{view}/{band}/{condition}/eligibility", sufficient(value), endpoint=value,
                    required_note_coverage=.8, required_instrument_coverage=.8)
            specs = [("am8_peak", "am8_depth06_peak", lambda v: v["within_1hz"], bool),
                     ("am48_peak", "am48_depth06_peak", lambda v: v["within_1hz"], bool),
                     ("fast_fraction_change", "fast_fraction_delta_am48_minus_am8_depth06", lambda v: v, lambda v: v >= .25),
                     ("entropy_change", "entropy_delta_chirp_minus_am48_depth06", lambda v: v, lambda v: v >= .1),
                     ("am8_depth_increase", "am8_near_power_ratio_depth06_over_depth02", lambda v: v, lambda v: v > 1.),
                     ("am48_depth_increase", "am48_near_power_ratio_depth06_over_depth02", lambda v: v, lambda v: v > 1.)]
            for name, field, extract, predicate in specs:
                def getter(note):
                    return extract(band_value(note)[field])
                raw = endpoint(notes, instruments, getter)
                success = endpoint(notes, instruments, lambda n: None if getter(n) is None else predicate(getter(n)))
                rate = success["instrument_balanced_mean"]
                add(f"A/{view}/{band}/{name}", sufficient(success) and rate is not None and rate >= .9,
                    endpoint=raw, success=success, required_success_rate=.9, coverage_required=True)
    for condition in ("common_gain", "polarity"):
        values = [note["views"]["original"][condition] for note in notes]
        errors = [v["max_abs_finite_feature_error"] for v in values if v["max_abs_finite_feature_error"] is not None]
        signatures = all(v["status_match"] and v["feature_missingness_match"] for v in values)
        add(f"B/original/{condition}", signatures and all(e <= 1e-9 for e in errors),
            all_52_status_and_missingness_signatures_match=signatures,
            maximum_finite_feature_error=max(errors) if errors else None, comparable_notes=len(errors),
            note_denominator=52, maximum_allowed_error=1e-9)
    for codec in CODECS:
        for feature in h.FEATURES:
            def getter(note):
                return note["codec_nuisance"][codec]["features"][feature]["ratio"]
            raw = endpoint(notes, instruments, getter)
            success = endpoint(notes, instruments, lambda n: None if getter(n) is None else getter(n) <= .1)
            ratios = [getter(n) for n in notes if getter(n) is not None]
            rate = success["instrument_balanced_mean"]
            add(f"C/{codec}/{feature}/nuisance", sufficient(raw) and rate is not None and rate >= .9 and all(r <= 1 for r in ratios),
                endpoint=raw, success=success, maximum_available_ratio=max(ratios) if ratios else None,
                usual_maximum_ratio=.1, any_maximum_ratio=1., required_success_rate=.9, coverage_required=True)
        for band in h.BAND_NAMES:
            raw = endpoint(notes, instruments, lambda n: n["codec_nuisance"][codec]["band_eligibility_signature_match"][band])
            add(f"C/{codec}/{band}/eligibility_signature", raw["instrument_balanced_mean"] >= .9,
                endpoint=raw, minimum_agreement=.9)
    failed = [c["id"] for c in criteria if not c["passed"]]
    return {"criteria_passed": not failed, "status": "criteria_failed" if failed else "criteria_passed_pending_independent_audit",
            "criteria": criteria, "failed_criterion_ids": failed, "all_bands_required": True,
            "independently_audited": False, "external_gate_passed": False, "classifier_admitted": False,
            "classification_fits": 0,
            "limitations": ["Codec ratio bounds apply only to notes with all eight finite condition pairs and a positive original target response; partial observations have no ratio nuisance bound.",
                            "Reserved instruments come from the reused NSynth sample-library corpus, not a new recording domain or performed-song set.",
                            "A criterion pass requires independent audit before any separately frozen classification extension."]}


def strata(notes, instruments):
    result = {"overall": {v: h.aggregates([n["views"][v] for n in notes], instruments) for v in VIEWS}, "strata": {}}
    for field in ("instrument_source_str", "instrument_family_str"):
        result["strata"][field] = {}
        for label in sorted({n["metadata"][field] for n in notes}):
            subset = [n for n in notes if n["metadata"][field] == label]
            ids = sorted({n["instrument_str"] for n in subset})
            result["strata"][field][label] = {v: h.aggregates([n["views"][v] for n in subset], ids) for v in VIEWS}
    return result


def expected_paths(selected):
    paths = {"freeze.json", "run_receipt.json", "gate_results.json"}
    for row in selected:
        for condition in h.CONDITIONS:
            stem = row["id"] + "__" + condition
            paths.add(f"original/{stem}.wav")
            paths.update(f"candidate_json/{stem}__{v}.json" for v in VIEWS)
            for codec, spec in CODECS.items():
                tag = stem + "__" + codec
                paths.add(f"encoded/{tag}{spec['extension']}")
                paths.update(f"{d}/{tag}.wav" for d in ("decoded_full", "analysis"))
                paths.update(f"command_logs/{tag}__{step}.json" for step in ("encode", "probe", "decode"))
    return paths


def wave(path, full=False):
    rate, data = wavfile.read(path)
    h.require(rate == 16000 and data.dtype == np.float64 and data.ndim == 1 and np.all(np.isfinite(data)), "float64 wave contract")
    h.require(data.size >= 64000 if full else data.size == 64000, "wave frame count")
    return data


def codec_record(root, stem, codec, tools, original, live_probe=True):
    spec = CODECS[codec]
    encoded = root / "encoded" / (stem + "__" + codec + spec["extension"])
    full_path = root / "decoded_full" / (stem + "__" + codec + ".wav")
    analysis_path = root / "analysis" / (stem + "__" + codec + ".wav")
    ffmpeg, ffprobe = (tools[k]["path"] for k in ("ffmpeg", "ffprobe"))
    prefix = [ffmpeg, "-hide_banner", "-nostdin", "-n"]
    encode = prefix + ["-i", str(root / "original" / (stem + ".wav")), "-map", "0:a:0", "-vn", "-map_metadata", "-1",
        "-c:a", spec["encoder"], "-b:a", spec["requested_bitrate"], "-ar", "16000", "-ac", "1", "-threads", "1", "-fflags", "+bitexact"]
    if codec == "aac_lc_64k":
        encode.extend(["-profile:a", "aac_low"])
    encode.append(str(encoded))
    probe = [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_streams", "-show_format", "-of", "json", str(encoded)]
    decode = prefix + ["-i", str(encoded), "-map", "0:a:0", "-vn", "-map_metadata", "-1",
                      "-c:a", "pcm_f64le", "-threads", "1", "-fflags", "+bitexact", str(full_path)]
    logs = {}
    for step, command in (("encode", encode), ("probe", probe), ("decode", decode)):
        report = h.load(root / "command_logs" / (stem + "__" + codec + "__" + step + ".json"))
        h.require(set(report) == {"command", "returncode", "stdout", "stderr", "elapsed_seconds"}, "complete command log keys")
        h.compare(report["command"], command, "codec command")
        h.require(type(report["returncode"]) is int and report["returncode"] == 0, "codec command failed")
        h.require(isinstance(report["stdout"], str) and isinstance(report["stderr"], str)
                  and math.isfinite(report["elapsed_seconds"]) and report["elapsed_seconds"] >= 0, "command log outputs")
        logs[step] = report
    metadata = json.loads(logs["probe"]["stdout"])
    if live_probe:
        actual = call(probe)
        h.compare(json.loads(actual["stdout"]), metadata, "encoded bytes independent ffprobe")
        h.compare(actual["stderr"], logs["probe"]["stderr"], "probe stderr")
    streams = metadata.get("streams", [])
    h.require(len(streams) == 1, "encoded stream count")
    stream = streams[0]
    h.require(stream.get("codec_name") == spec["codec_name"] and stream.get("sample_rate") == "16000"
              and stream.get("channels") == 1 and (codec != "aac_lc_64k" or stream.get("profile") == "LC"), "native codec contract")
    h.require(metadata.get("format", {}).get("filename") == str(encoded), "encoded metadata path")
    formats = metadata["format"].get("format_name", "").split(",")
    h.require("mp3" in formats if codec == "mp3_128k" else "m4a" in formats, "encoded container")
    h.require(int(metadata["format"].get("size", -1)) == encoded.stat().st_size, "encoded metadata size")
    full, analysis = wave(full_path, full=True), wave(analysis_path)
    h.require(np.array_equal(full[:64000], analysis), "analysis is not fixed first-64000 full-decode slice")
    a, b = original-np.mean(original), analysis-np.mean(analysis)
    norm = float(np.linalg.norm(a)*np.linalg.norm(b))
    correlation = float(np.clip(np.dot(a, b)/norm, -1., 1.)) if norm > 0 else None
    return analysis, {"view": codec, "requested_bitrate": spec["requested_bitrate"],
        "actual_stream_bit_rate": stream.get("bit_rate"), "stream_metadata": metadata,
        "encoded": {"path": str(encoded.relative_to(root)), **h.file_record(encoded)},
        "full_decode": {"path": str(full_path.relative_to(root)), **h.file_record(full_path)},
        "analysis": {"path": str(analysis_path.relative_to(root)), **h.file_record(analysis_path)},
        "full_decoded_frames": int(full.size), "analysis_frames": 64000,
        "discarded_tail_frames": int(full.size-64000), "analysis_start_frame": 0,
        "zero_lag_correlation": correlation, "lag_optimization_performed": False, "padding_performed": False}


def audit(result_dir, draft_file, freeze_sha256):
    root, draft = Path(result_dir).resolve(strict=True), Path(draft_file).resolve(strict=True)
    h.require(draft.name == "draft.json" and len(freeze_sha256) == 64 and h.digest(draft) == freeze_sha256, "explicit reviewed freeze SHA mismatch")
    draft_commit = h.verify_products(draft.parent)
    h.require(draft_commit.get("kind") == "heldout_measurement_gate_draft" and draft_commit.get("draft_sha256") == freeze_sha256
              and set(draft_commit["products"]) == {"draft.json"}, "gate draft publication linkage")
    commit = h.verify_products(root)
    h.require(commit.get("kind") == "heldout_measurement_gate_result" and commit.get("freeze_sha256") == freeze_sha256,
              "gate result publication linkage")
    h.require(h.digest(root / "freeze.json") == freeze_sha256, "copied freeze bytes")
    frozen = h.load(draft)
    h.require(frozen["version"] == GATE_VERSION and frozen["planned_run_output_dir"] == str(root)
              and frozen["features_extracted"] is False and frozen["reserved_note_codec_processing_performed"] is False,
              "gate draft scope")
    h.compare(frozen["policy"], POLICY, "independent frozen policy")
    verify_bindings(frozen["bindings"])
    selected = verify_inputs(frozen)
    expected = expected_paths(selected)
    h.require(set(commit["products"]) == expected and len(expected) == 6659, "complete gate product grid")
    notes = []
    for row in selected:
        rate, pcm = wavfile.read(row["path"])
        h.require(rate == 16000 and pcm.dtype == np.int16 and pcm.shape == (64000,), "source PCM contract")
        h.require(hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest() == row["pcm_sha256"], "source PCM digest")
        x = pcm.astype(np.float64)/32768
        replay = {view: {} for view in VIEWS}
        records = {}
        for condition in h.CONDITIONS:
            stem = row["id"] + "__" + condition
            raw_path = root / "original" / (stem + ".wav")
            original = wave(raw_path)
            h.require(np.array_equal(original, h.expected_audio(x, condition)), "original derivative formula mismatch")
            audio = {"original": original}
            records[condition] = {}
            for codec in CODECS:
                audio[codec], records[condition][codec] = codec_record(root, stem, codec, frozen["bindings"]["codec_toolchain"], original)
            for view in VIEWS:
                relative = f"original/{stem}.wav" if view == "original" else f"analysis/{stem}__{view}.wav"
                product = h.load(root / "candidate_json" / (stem + "__" + view + ".json"))
                linkage = {"note_id": row["id"], "instrument_str": row["metadata"]["instrument_str"],
                           "condition": condition, "view": view, "source_file_sha256": row["file_sha256"],
                           "analysis_waveform": {"path": relative, **h.file_record(root / relative)}}
                h.compare({k: v for k, v in product.items() if k != "measurement"}, linkage, "measurement linkage")
                replay[view][condition] = h.replay_measurement(product["measurement"], audio[view])
        views = {}
        for view in VIEWS:
            views[view] = h.paired(row, replay[view])
            views[view]["condition_features"] = {c: replay[view][c]["features"] for c in h.CONDITIONS}
        notes.append({"id": row["id"], "instrument_str": row["metadata"]["instrument_str"], "metadata": row["metadata"],
                      "views": views, "codec_records": records, "codec_nuisance": {c: nuisance(views, c) for c in CODECS}})
    instruments = frozen["inputs"]["selection"]["gate_instrument_ids"]
    outcome = decision(notes, instruments)
    h.compare(h.load(root / "gate_results.json"), {"version": GATE_VERSION, "freeze_sha256": freeze_sha256,
              "notes": notes, "decision": outcome, "descriptive": strata(notes, instruments)}, "complete gate result")
    h.compare(commit["criteria_passed"], outcome["criteria_passed"])
    h.compare(commit["decision_status"], outcome["status"])
    h.compare(h.load(root / "run_receipt.json"), {"status": "execution_completed", "freeze_sha256": freeze_sha256,
        "draft_commit": h.file_record(draft.parent / "COMMIT.json"), "bindings_start": frozen["bindings"],
        "bindings_end": frozen["bindings"], "input_bindings_rechecked_start_and_end": True,
        "decoded_source_note_ids": [n["id"] for n in notes], "gate_instrument_ids": instruments,
        "excluded_development_instrument_ids": frozen["inputs"]["selection"]["excluded_development_instrument_ids"],
        "notes": 52, "instruments": 26, "counts": COUNTS, "decision_status": outcome["status"],
        "ai_human_labels_assigned": False, "classifier_fits": 0, "candidate_definition_changed": False,
        "padding_performed": False, "lag_optimization_performed": False, "external_gate_passed": False}, "gate execution receipt")
    h.compare(h.verify_products(root), commit, "result unchanged during audit")
    h.compare(h.verify_products(draft.parent), draft_commit, "draft unchanged during audit")
    verify_bindings(frozen["bindings"])
    h.compare(verify_inputs(frozen), selected, "source/prior/roster unchanged during audit")
    return {"version": VERSION, "passed": True, "auditor_sha256": h.digest(__file__), "helper_sha256": HELPER_SHA,
        "auditor_tests_sha256": h.digest(Path(__file__).with_name("test_audit_modulation_nsynth_gate_v1.py")),
        "freeze_sha256": freeze_sha256, "result_commit_sha256": h.digest(root / "COMMIT.json"), "result_dir": str(root),
        "gate_criteria_passed": outcome["criteria_passed"], "failed_criterion_ids": outcome["failed_criterion_ids"],
        "selected_notes_replayed": 52, "reserved_instruments": 26, "float64_originals_recreated": 416,
        "codec_analysis_slices_checked": 832, "encoded_streams_independently_probed": 832,
        "measurement_records_replayed": 1248, "stored_band_spectra_replayed": 3744,
        "criteria_replayed": len(outcome["criteria"]), "products_hashed": len(expected),
        "external_gate_passed": False, "classifier_admitted": False,
        "limitations": ["Stored-power replay is not independent full filter/Hilbert DSP recomputation.",
                        "Encoded files are independently probed; waveform decoding is checked by stored command logs and slices, not a second decode.",
                        "Recorded scope is not OS-level proof of historical held-out non-access.",
                        "A passed consistency audit can accompany scientifically failed criteria; it does not itself admit classification."]}


def receipt_path(output, result_dir, draft):
    output = Path(output).resolve()
    h.require(not output.exists(), "new audit receipt required")
    frozen = h.load(draft)
    protected = (result_dir, Path(draft).parent, frozen["inputs"]["source"]["source_dir"],
                 frozen["inputs"]["prior_pilot"]["pilot_dir"])
    for root in protected:
        h.require(not output.is_relative_to(Path(root).resolve()), "receipt must be outside protected publications")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("result-dir", "draft", "freeze-sha256", "output"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    output = receipt_path(args.output, args.result_dir, args.draft)
    result = audit(args.result_dir, args.draft, args.freeze_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(h.canonical(result))
    print(json.dumps({"passed": True, "gate_criteria_passed": result["gate_criteria_passed"],
                      "receipt": str(output), "sha256": h.digest(output)}))


if __name__ == "__main__":
    main()
