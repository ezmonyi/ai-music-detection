#!/usr/bin/env python3
"""Freeze-bound NSynth BC development pilot. Draft performs no BC extraction.

Only a reviewed draft SHA authorizes run. All conditions/cells remain visible;
this is sensitivity to constructed injections, never a causal baseline label,
AI probability, fitted admission threshold, or independent-song sample count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.io import wavfile

import bicoherence_audio_v1 as extractor
import bicoherence_primitive_v2 as primitive

VERSION = "bicoherence_nsynth_development_pilot_v1"
PROTOCOL_SHA = "5c2cb09dbc8f79cb44e42ca89fbf1d625f7ccd3e29a16703d8ed11c814f15368"
SOURCE_COMMIT_SHA = "85f70e63a033e1509d3c2fbad6c4561c3f1fce2511117756af64f78b9c39e330"
SOURCE_ARCHIVE_SHA = "0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c"
ROSTER_SHA = "020c806a5785176cfbffd26a0d0c6ec6fa1f28bc8a0a1c1bb004ce7a0ec7f222"
EXTRACTOR_SHA = "e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1"
PRIMITIVE_SHA = "9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5"
NOTE_COUNT, INSTRUMENT_COUNT, PILOT_COUNT = 4096, 53, 27
CONDITIONS = ("baseline", "common_gain", "polarity", "closed_minus6db",
              "independent_minus6db", "closed_0db", "independent_0db")
TARGET = [32, 48, 80]
POLICY = {"conditions": list(CONDITIONS), "target_bins": TARGET,
          "seed_prefix": "BC-injection-20260907|", "phase_knots": 33,
          "sample_rate": 16000, "samples": 64000, "background_gain": 0.25,
          "relative_injection_db": [-6, 0], "prototype_tone_amplitude": 0.2,
          "aggregation": "mean within instrument then equal weight covered instruments",
          "framewise_power_matched": False, "numeric_admission_threshold": None,
          "external_gate_passed": False, "classifier_admitted": False,
          "baseline_causal_label": None, "reserved_waveforms_measured": False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def product(path):
    return {k: v for k, v in fingerprint(path).items() if k != "path"}


def safe_child(root, name):
    root, rel = Path(root).resolve(strict=True), Path(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("unsafe source-relative path")
    if any((root.joinpath(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
        raise ValueError("source symlink is not accepted")
    target = (root / rel).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("source product is not a contained regular file")
    return target


def verify_source(source_dir):
    """Hash all source products; validate inventory without decoding reserved PCM."""
    root = Path(source_dir).resolve(strict=True)
    commit_path = safe_child(root, "COMMIT.json")
    if sha(commit_path) != SOURCE_COMMIT_SHA:
        raise ValueError("source COMMIT immutable hash mismatch")
    commit = read_json(commit_path)
    if commit.get("status") != "committed" or not isinstance(commit.get("products"), dict):
        raise ValueError("invalid committed source inventory")
    expected = commit["products"]
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("source symlinks are not accepted")
    actual_names = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual_names != set(expected) | {"COMMIT.json"}:
        raise ValueError("source inventory file set mismatch")
    hashes = {}
    for name, binding in sorted(expected.items()):
        hashes[name] = product(safe_child(root, name))
        if hashes[name] != binding:
            raise ValueError(f"source product hash mismatch: {name}")
    hashes["COMMIT.json"] = product(commit_path)
    if hashes.get("nsynth-test.jsonwav.tar.gz", {}).get("sha256") != SOURCE_ARCHIVE_SHA:
        raise ValueError("official source archive hash mismatch")
    receipt = read_json(root / "acquisition_receipt.json")
    if (receipt.get("decoded_all") is not True or receipt.get("notes") != NOTE_COUNT
            or receipt.get("instruments") != INSTRUMENT_COUNT
            or receipt.get("features_extracted") is not False
            or receipt.get("ai_human_labels_assigned") is not False):
        raise ValueError("source acquisition receipt mismatch")
    metadata = read_json(root / "nsynth-test/examples.json")
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    by_id = {r["id"]: r for r in rows}
    if len(rows) != NOTE_COUNT or len(by_id) != NOTE_COUNT or set(by_id) != set(metadata):
        raise ValueError("source metadata note inventory mismatch")
    wave_paths = set()
    for note_id, row in by_id.items():
        name = f"nsynth-test/audio/{note_id}.wav"
        wave_paths.add(name)
        path = safe_child(root, name)
        if (row.get("path") != str(path) or row.get("metadata") != metadata[note_id]
                or row.get("file_sha256") != hashes[name]["sha256"]
                or row.get("native_channels") != 1 or row.get("native_sample_rate") != 16000
                or row.get("frames") != 64000 or row.get("duration_s") != 4
                or row.get("role") != "external_measurement_control_only"
                or row.get("ai_human_label") is not None
                or metadata[note_id].get("note_str") != note_id
                or metadata[note_id].get("sample_rate") != 16000):
            raise ValueError(f"source decoded manifest mismatch: {note_id}")
        pcm_sha = row.get("pcm_sha256", "")
        if len(pcm_sha) != 64 or any(c not in "0123456789abcdef" for c in pcm_sha):
            raise ValueError("invalid manifest PCM digest")
    if {name for name in expected if name.endswith(".wav")} != wave_paths:
        raise ValueError("wave products do not match note roster")
    if len({r["metadata"]["instrument_str"] for r in rows}) != INSTRUMENT_COUNT:
        raise ValueError("source instrument count mismatch")
    return {"source_dir": str(root), "products": hashes}, by_id


def select_roster(by_id):
    """Reimplement the old deterministic roster from metadata, without Q features."""
    grouped = defaultdict(list)
    for row in by_id.values():
        grouped[row["metadata"]["instrument_str"]].append(row)
    ordered = sorted(grouped, key=lambda name: (hashlib.sha256(
        ("Q-pilot-20260907|" + name).encode()).hexdigest(), name))
    if len(ordered) != INSTRUMENT_COUNT:
        raise ValueError("roster requires the full instrument inventory")
    pilot, reserved = ordered[:PILOT_COUNT], ordered[PILOT_COUNT:]
    pilot_pcm = {r["pcm_sha256"] for name in pilot for r in grouped[name]}
    reserved_pcm = {r["pcm_sha256"] for name in reserved for r in grouped[name]}
    if pilot_pcm & reserved_pcm:
        raise ValueError("pilot/reserved PCM collision")
    selected = []
    for instrument in pilot:
        rows = sorted(grouped[instrument], key=lambda r: (abs(r["metadata"]["pitch"] - 60),
                      hashlib.sha256(("Q-note-20260907|" + r["id"]).encode()).hexdigest(), r["id"]))
        seen = set()
        for row in rows:
            if row["pcm_sha256"] in seen:
                continue
            seen.add(row["pcm_sha256"])
            selected.append(row)
            if len(seen) == 2:
                break
        if len(seen) != 2:
            raise ValueError("two distinct PCM notes required; no substitution")
    return {"ordered_instrument_ids": ordered, "pilot_instrument_ids": pilot,
            "reserved_instrument_ids": reserved, "selected_notes": selected,
            "pilot_reserved_pcm_sets_disjoint": True}


def read_pcm(row, source_dir):
    path = safe_child(source_dir, f"nsynth-test/audio/{row['id']}.wav")
    if str(path) != row["path"] or sha(path) != row["file_sha256"]:
        raise ValueError("selected WAV binding mismatch")
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                stream.getnframes(), stream.getcomptype()) != (1, 2, 16000, 64000, "NONE"):
            raise ValueError("selected WAV must be mono 16kHz 16-bit PCM, 64000 frames")
        pcm = stream.readframes(64000)
        if len(pcm) != 128000 or stream.readframes(1):
            raise ValueError("selected PCM length mismatch")
    if hashlib.sha256(pcm).hexdigest() != row["pcm_sha256"]:
        raise ValueError("selected decoded PCM digest mismatch")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0


def runtime_bindings():
    if sys.version_info[:2] != (3, 12) or np.__version__ != "2.4.4" or scipy.__version__ != "1.17.1":
        raise ValueError("requires Python 3.12, NumPy 2.4.4, SciPy 1.17.1")
    modules = {"numpy": np, "scipy": scipy, "scipy.io.wavfile": wavfile, "wave": wave,
               "numpy.fft._pocketfft": np.fft._pocketfft,
               "numpy.fft._pocketfft_umath": np.fft._pocketfft.pfu,
               "numpy._core._multiarray_umath": np._core._multiarray_umath}
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "executable": fingerprint(sys.executable),
            "modules": {name: fingerprint(module.__file__) for name, module in modules.items()}}


def code_bindings(protocol, roster):
    code = Path(__file__).resolve().parent
    bindings = {"runner": fingerprint(__file__),
                "runner_tests": fingerprint(code / "test_bicoherence_nsynth_pilot_v1.py"),
                "extractor": fingerprint(extractor.__file__), "primitive": fingerprint(primitive.__file__),
                "extractor_tests": fingerprint(code / "test_bicoherence_audio_v1.py"),
                "primitive_tests": fingerprint(code / "test_bicoherence_primitive_v2.py"),
                "protocol": fingerprint(protocol), "roster": fingerprint(roster),
                "runtime": runtime_bindings()}
    for key, expected in (("protocol", PROTOCOL_SHA), ("roster", ROSTER_SHA),
                          ("extractor", EXTRACTOR_SHA), ("primitive", PRIMITIVE_SHA)):
        if bindings[key]["sha256"] != expected:
            raise ValueError(f"immutable {key} hash mismatch")
    return bindings


def snapshot(source, protocol, roster):
    bindings = code_bindings(protocol, roster)
    source_binding, by_id = verify_source(source)
    selection = select_roster(by_id)
    if selection != read_json(roster).get("selection"):
        raise ValueError("deterministic source roster differs from exact reused Q roster")
    # Byte/PCM verification is permitted during drafting; no BC call here.
    for row in selection["selected_notes"]:
        read_pcm(row, source)
    return {"bindings": bindings, "source": source_binding, "selection": selection}


def new_output(path, source):
    output = Path(path).resolve()
    if output.exists() or output.is_relative_to(Path(source).resolve()):
        raise ValueError("output must be a new exclusive directory outside source")
    return output


def commit_directory(output, fields):
    files = sorted(p for p in output.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("symlink in output inventory")
    write_json(output / "COMMIT.json", {"status": "committed", **fields,
               "products": {str(p.relative_to(output)): product(p) for p in files}})


def draft(source, protocol, roster, draft_dir, run_output):
    out, future = new_output(draft_dir, source), new_output(run_output, source)
    if out == future or out.is_relative_to(future) or future.is_relative_to(out):
        raise ValueError("draft and run output must be separate directories")
    frozen = snapshot(source, protocol, roster)
    document = {"version": VERSION, "stage": "draft_requires_reviewed_sha256", "policy": POLICY,
                "planned_run_output_dir": str(future), **frozen, "features_extracted": False,
                "selected_pcm_verified": True, "reserved_pcm_decoded": False}
    if snapshot(source, protocol, roster) != frozen:
        raise ValueError("source/code/runtime changed during draft")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "draft.json", document)
    digest = sha(out / "draft.json")
    commit_directory(out, {"kind": "BC_development_draft", "draft_sha256": digest})
    return {"draft": str(out / "draft.json"), "freeze_sha256_for_review": digest,
            "selected_notes": len(frozen["selection"]["selected_notes"]),
            "reserved_instruments": len(frozen["selection"]["reserved_instrument_ids"])}


def constructions(x, note_id):
    if not isinstance(x, np.ndarray) or x.dtype != np.float64 or x.shape != (64000,) or not np.isfinite(x).all():
        raise ValueError("construction input requires finite float64 mono 64000 samples")
    seed = int.from_bytes(hashlib.sha256(("BC-injection-20260907|" + note_id).encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    knots = rng.uniform(-np.pi, np.pi, size=(3, 33))
    unwrapped = np.unwrap(knots, axis=1)
    knot_times = np.arange(33, dtype=np.float64) * 0.125
    t = np.arange(64000, dtype=np.float64) / 16000
    phases = np.asarray([np.interp(t, knot_times, row) for row in unwrapped])
    closed_phases = np.stack((phases[0], phases[1], phases[0] + phases[1]))
    carriers = 2 * np.pi * np.asarray([500., 750., 1250.])[:, None] * t
    closed = 0.2 * np.cos(carriers + closed_phases).sum(axis=0)
    independent = 0.2 * np.cos(carriers + phases).sum(axis=0)
    cr, ir = float(np.sqrt(np.mean(closed**2))), float(np.sqrt(np.mean(independent**2)))
    if not cr > 0 or not ir > 0:
        raise FloatingPointError("unsupported zero prototype RMS")
    b = 0.25 * x
    rms = float(np.sqrt(np.mean(b**2)))
    if not np.isfinite(rms) or (rms == 0 and np.any(b)):
        raise FloatingPointError("unsupported background RMS arithmetic")
    unit_closed, unit_independent = closed / cr, independent / ir
    derivatives = {"baseline": b, "common_gain": 0.1 * b, "polarity": -b}
    arrays = {"source_audio": x, "background": b, "sample_times_seconds": t,
              "phase_knot_times_seconds": knot_times, "phase_knots_wrapped": knots,
              "phase_knots_unwrapped": unwrapped, "independent_phase_trajectories": phases,
              "closed_phase_trajectories": closed_phases, "closed_prototype": closed,
              "independent_prototype": independent, "closed_unit_rms": unit_closed,
              "independent_unit_rms": unit_independent,
              "prototype_rms": np.asarray([cr, ir]), "background_rms": np.asarray(rms)}
    for label, db in (("minus6db", -6), ("0db", 0)):
        scale = rms * 10**(db / 20)
        for kind, unit in (("closed", unit_closed), ("independent", unit_independent)):
            injection = scale * unit
            arrays[f"{kind}_{label}_injection"] = injection
            derivatives[f"{kind}_{label}"] = b + injection
    if tuple(derivatives) != CONDITIONS or not all(np.isfinite(a).all() for a in derivatives.values()):
        raise FloatingPointError("invalid derivative construction")
    return {"waveforms": derivatives, "arrays": arrays,
            "metadata": {"note_id": note_id, "seed": seed, "bit_generator": "PCG64",
                         "status": "ok" if rms > 0 else "unsupported_zero_background_rms",
                         "background_rms": rms, "prototype_rms": {"closed": cr, "independent": ir},
                         "prototype_normalization_factors": {"closed": 1 / cr, "independent": 1 / ir},
                         "injected_full_record_rms_matched": rms > 0,
                         "framewise_power_matched": False,
                         "zero_rms_interpretation": "zero waveforms retained; relative-level intervention unsupported"}}


def save_arrays(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def save_waveform(path, samples):
    if samples.dtype != np.float64 or samples.shape != (64000,) or not np.isfinite(samples).all():
        raise ValueError("derivative WAV must be finite float64 mono 64000 samples")
    with Path(path).open("xb") as stream:
        wavfile.write(stream, 16000, samples)
    rate, decoded = wavfile.read(path)
    if rate != 16000 or decoded.dtype != np.float64 or decoded.shape != samples.shape or not np.array_equal(decoded, samples):
        raise ValueError("derivative float64 WAV did not roundtrip exactly")


def descriptor(metadata):
    if metadata["pool_count"] != 1:
        raise ValueError("NSynth requires exactly one 4-second pool")
    cells = metadata["pools"][0]["cells"]
    if [c["frequency_bins"] for c in cells] != extractor.pair_grid().tolist():
        raise ValueError("extractor grid differs from frozen complete grid")
    target = next(c for c in cells if c["frequency_bins"] == TARGET)
    return {"target": target, "grid_cell_count": len(cells),
            "eligible_cell_count": sum(c["eligible"] for c in cells),
            "grid_eligibility": [c["eligible"] for c in cells],
            "grid_squared_bicoherence": [c["squared_bicoherence"] for c in cells]}


def nuisance(reference, other):
    a, b = reference["grid_eligibility"], other["grid_eligibility"]
    differences = [abs(x - y) for x, y in zip(reference["grid_squared_bicoherence"],
                   other["grid_squared_bicoherence"], strict=True) if x is not None and y is not None]
    ta, tb = reference["target"], other["target"]
    return {"grid_denominator": len(a), "grid_eligibility_agreement_count": sum(x == y for x, y in zip(a, b, strict=True)),
            "missing_to_finite_count": sum(not x and y for x, y in zip(a, b, strict=True)),
            "finite_to_missing_count": sum(x and not y for x, y in zip(a, b, strict=True)),
            "common_finite_cell_count": len(differences),
            "maximum_finite_b2_difference": max(differences) if differences else None,
            "target_eligibility_agreement": ta["eligible"] == tb["eligible"],
            "target_difference": (tb["squared_bicoherence"] - ta["squared_bicoherence"]
                                  if ta["eligible"] and tb["eligible"] else None)}


def weighted_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instrument"]].append(row)
    per_instrument = []
    for instrument, notes in sorted(grouped.items()):
        values = [n["difference"] for n in notes if n["difference"] is not None]
        per_instrument.append({"instrument": instrument, "note_denominator": len(notes),
                               "covered_notes": len(values),
                               "mean_difference": float(np.mean(values)) if values else None})
    means = [r["mean_difference"] for r in per_instrument if r["mean_difference"] is not None]
    values = [r["difference"] for r in rows if r["difference"] is not None]
    return {"note_denominator": len(rows), "covered_notes": len(values),
            "instrument_denominator": len(grouped), "covered_instruments": len(means),
            "equal_instrument_mean_difference": float(np.mean(means)) if means else None,
            "positive": sum(v > 0 for v in values), "zero": sum(v == 0 for v in values),
            "negative": sum(v < 0 for v in values), "per_instrument": per_instrument}


def aggregate(notes):
    result = {"note_denominator": len(notes), "policy": POLICY, "levels": {},
              "independent_replay_status": "required_before_accepting_values",
              "baseline": {"target_covered_notes": sum(n["conditions"]["baseline"]["target"]["eligible"] for n in notes),
                           "grid_cell_denominator": sum(n["conditions"]["baseline"]["grid_cell_count"] for n in notes),
                           "grid_eligible_cells": sum(n["conditions"]["baseline"]["eligible_cell_count"] for n in notes)},
              "per_note": notes}
    for level in ("minus6db", "0db"):
        rows = []
        for note in notes:
            a, b = (note["conditions"][f"{kind}_{level}"]["target"] for kind in ("closed", "independent"))
            rows.append({k: note[k] for k in ("note_id", "instrument", "source_category", "instrument_family")} |
                        {"difference": float(a["squared_bicoherence"] - b["squared_bicoherence"])
                         if a["eligible"] and b["eligible"] else None,
                         "closed_eligible": a["eligible"], "independent_eligible": b["eligible"]})
        strata = {}
        for key in ("source_category", "instrument_family"):
            strata[key] = {value: weighted_summary([r for r in rows if r[key] == value])
                           for value in sorted({r[key] for r in rows})}
        result["levels"][level] = {**weighted_summary(rows), "strata": strata, "per_note": rows}
    result["nuisance"] = {}
    for condition in ("common_gain", "polarity"):
        checks = [n["nuisance"][condition] for n in notes]
        maxima = [c["maximum_finite_b2_difference"] for c in checks if c["maximum_finite_b2_difference"] is not None]
        result["nuisance"][condition] = {
            key: sum(c[key] for c in checks) for key in ("grid_denominator", "grid_eligibility_agreement_count",
                "missing_to_finite_count", "finite_to_missing_count", "common_finite_cell_count")}
        result["nuisance"][condition]["maximum_finite_b2_difference"] = max(maxima) if maxima else None
        result["nuisance"][condition]["target_eligibility_agreement_notes"] = sum(c["target_eligibility_agreement"] for c in checks)
    return result


def run(draft_file, freeze_sha256, output_dir):
    if not isinstance(freeze_sha256, str) or len(freeze_sha256) != 64 or sha(draft_file) != freeze_sha256:
        raise ValueError("explicit reviewed draft SHA256 required and must match")
    document = read_json(draft_file)
    if (document.get("version") != VERSION or document.get("policy") != POLICY
            or document.get("stage") != "draft_requires_reviewed_sha256"
            or document.get("features_extracted") is not False):
        raise ValueError("draft schema/policy does not match this runner")
    source = document["source"]["source_dir"]
    protocol, roster = (document["bindings"][key]["path"] for key in ("protocol", "roster"))
    output = new_output(output_dir, source)
    if str(output) != document["planned_run_output_dir"]:
        raise ValueError("run output differs from frozen planned output")
    expected = {k: document[k] for k in ("bindings", "source", "selection")}
    if snapshot(source, protocol, roster) != expected:
        raise ValueError("frozen source/code/runtime/roster mismatch at run start")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen_draft.json", document)
    notes = []
    try:
        for row in document["selection"]["selected_notes"]:
            note_id, m = row["id"], row["metadata"]
            if m["instrument_str"] in document["selection"]["reserved_instrument_ids"]:
                raise ValueError("reserved note reached run; fail closed")
            note_dir = output / note_id
            note_dir.mkdir(exist_ok=False)
            construction = constructions(read_pcm(row, source), note_id)
            save_arrays(note_dir / "construction.npz", construction["arrays"])
            write_json(note_dir / "construction.json", construction["metadata"])
            record = {"note_id": note_id, "instrument": m["instrument_str"],
                      "source_category": m["instrument_source_str"], "instrument_family": m["instrument_family_str"],
                      "pitch": m["pitch"], "velocity": m["velocity"], "source_record": row,
                      "construction_status": construction["metadata"]["status"], "conditions": {}}
            for condition in CONDITIONS:
                samples = construction["waveforms"][condition]
                save_waveform(note_dir / f"{condition}.wav", samples)
                measured = extractor.extract(samples, 16000)
                # A zero background yields literal zero derivatives and missing
                # BC arrays, but does not become a supported relative-RMS trial.
                measured["metadata"]["construction_status"] = construction["metadata"]["status"]
                save_arrays(note_dir / f"{condition}.npz", measured["arrays"])
                write_json(note_dir / f"{condition}.json", measured["metadata"])
                record["conditions"][condition] = descriptor(measured["metadata"])
            record["nuisance"] = {name: nuisance(record["conditions"]["baseline"], record["conditions"][name])
                                  for name in ("common_gain", "polarity")}
            write_json(note_dir / "summary.json", record)
            notes.append(record)
            print(f"BC development note {len(notes)}/{len(document['selection']['selected_notes'])}: {note_id}",
                  file=sys.stderr, flush=True)
        write_json(output / "summary.json", aggregate(notes))
        if sha(draft_file) != freeze_sha256 or snapshot(source, protocol, roster) != expected:
            raise ValueError("frozen inputs changed during run; no COMMIT")
        write_json(output / "verification.json", {"frozen_start_end_equal": True,
                   "derivative_float64_wavs_exact_roundtrip": True, "notes": len(notes),
                   "conditions": len(notes) * len(CONDITIONS), "reserved_pcm_decoded": False,
                   "independent_replay_status": "required_before_accepting_values"})
        commit_directory(output, {"kind": "BC_external_note_development_only",
                         "draft_sha256": freeze_sha256, "classifier_admitted": False,
                         "external_gate_passed": False, "independent_replay_passed": False})
    except Exception as exc:
        write_json(output / "FAILED.json", {"status": "failed_no_commit", "exception": type(exc).__name__,
                   "message": str(exc), "completed_notes": len(notes), "outputs_preserved": True})
        raise
    return {"output": str(output), "commit_sha256": sha(output / "COMMIT.json"),
            "notes": len(notes), "conditions": len(notes) * len(CONDITIONS),
            "independent_replay_required": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    drafting = sub.add_parser("draft")
    for name in ("source", "protocol", "roster", "draft-dir", "run-output"):
        drafting.add_argument("--" + name, required=True)
    running = sub.add_parser("run")
    for name in ("draft", "freeze-sha256", "output"):
        running.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = (draft(args.source, args.protocol, args.roster, args.draft_dir, args.run_output)
              if args.command == "draft" else run(args.draft, args.freeze_sha256, args.output))
    print(canonical(result).decode(), end="")


if __name__ == "__main__":
    main()
