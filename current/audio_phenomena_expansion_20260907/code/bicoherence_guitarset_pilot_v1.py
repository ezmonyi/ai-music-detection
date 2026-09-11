#!/usr/bin/env python3
"""Parent-freeze-bound GuitarSet BC development producer; never creates a freeze.

CLI requires a separately reviewed receipt whose SHA is supplied by the parent.
See ``FREEZE_KEYS`` and ``verify_freeze`` for its exact schema. Only the crossed
development intersection is decoded. No reserved/unused BC, fitting, calibrated
null, significance, sensitivity gate, or causal baseline label is produced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.io import wavfile

import bicoherence_audio_v1 as extractor
import bicoherence_primitive_v2 as primitive
import draft_bicoherence_guitarset_pilot_v2 as drafting

VERSION = "bicoherence_guitarset_development_pilot_v1"
EXTRACTOR_SHA = "e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1"
PRIMITIVE_SHA = "9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5"
SAMPLES, RATE, POOLS, DEVELOPMENT_COUNT = 128000, 16000, 2, 90
CONDITIONS = ("baseline", "common_gain", "polarity", "closed_minus6db",
              "independent_minus6db", "closed_0db", "independent_0db")
TARGET = [32, 48, 80]
POLICY = {"conditions": list(CONDITIONS), "target_bins": TARGET,
          "seed_prefix": "BC-GuitarSet-injection-20260907|", "phase_knots": 65,
          "knot_spacing_seconds": .125, "sample_rate_hz": RATE, "samples": SAMPLES,
          "background_gain": .25, "common_gain_factor": .1,
          "relative_injection_db": [-6, 0], "prototype_tone_amplitude": .2,
          "framewise_power_matched": False, "mixture_normalization_or_clipping": False,
          "aggregation": "paired pools mean within recording; mean within score; equal covered scores",
          "secondary_aggregation": "mean within player; equal covered players",
          "performance_strata": ["comp", "solo"], "development_recordings": DEVELOPMENT_COUNT,
          "pools_per_recording_condition": POOLS, "cells_per_pool": 228,
          "numeric_admission_threshold": None, "null_calibrated": False,
          "significance_inferred": False, "external_gate_passed": False,
          "classifier_admitted": False, "classifier_fits": 0,
          "reserved_or_unused_waveforms_measured": False, "baseline_causal_label": None}
FREEZE_KEYS = {"schema", "status", "draft", "draft_commit", "producer_bindings",
               "policy", "planned_output_dir", "source_commit_sha256"}
require = drafting.require
binding = drafting.binding
fp = drafting.fp
read_json = drafting.read_json


def write_json(path, value):
    drafting.write_new(path, drafting.canonical(value))


def producer_bindings():
    """Read-only helper for the parent; calling it does not authorize a run."""
    code = Path(__file__).resolve().parent
    require(fp(extractor.__file__)["sha256"] == EXTRACTOR_SHA, "Frozen BC extractor changed")
    require(fp(primitive.__file__)["sha256"] == PRIMITIVE_SHA, "Frozen BC primitive changed")
    runtime = drafting.runtime_bindings()
    pocketfft = importlib.import_module("numpy.fft._pocketfft")
    # NumPy 1.26's executed FFT extension, in addition to its Python wrapper.
    fft_binary = importlib.import_module("numpy.fft._pocketfft_internal")
    runtime["producer_modules"] = {"scipy.io.wavfile": binding(wavfile.__file__),
        "numpy.fft._pocketfft": binding(pocketfft.__file__),
        "numpy.fft._pocketfft_internal": binding(fft_binary.__file__)}
    return {"producer": binding(__file__),
            "producer_tests": binding(code / "test_bicoherence_guitarset_pilot_v1.py"),
            "extractor": binding(extractor.__file__), "primitive": binding(primitive.__file__),
            "extractor_tests": binding(code / "test_bicoherence_audio_v1.py"),
            "primitive_tests": binding(code / "test_bicoherence_primitive_v2.py"),
            "runtime": runtime}


def verify_draft(draft_file):
    """Recheck full bindings/split, without decoding any source waveforms."""
    path = drafting.source_validator.regular_file(draft_file)
    document = read_json(path)
    require(document.get("version") == drafting.VERSION
            and document.get("stage") == "partition_preprocessing_draft_only_requires_parent_review"
            and document.get("bc_extracted") is False
            and document.get("injections_constructed") is False
            and document.get("derived_audio_saved") is False
            and document.get("reserved_or_unused_decoded_by_this_tool") is False
            and document.get("classifier_fits") == 0
            and document.get("external_gate_passed") is False,
            "Wrong prospective draft version/stage/scope")
    commit_path = path.parent / "COMMIT.json"
    commit = read_json(commit_path)
    require(set(commit) == {"status", "kind", "products", "bc_extracted", "source_materialization_commit_sha256"}
            and commit["status"] == "committed"
            and commit["kind"] == "guitarset_bc_partition_preprocessing_draft_only_v2"
            and commit["products"] == {"draft.json": fp(path)}
            and commit["bc_extracted"] is False, "Draft COMMIT mismatch")
    inventory = drafting.source_validator._source_inventory(path.parent)
    require(set(inventory) == {"draft.json", "COMMIT.json"}, "Draft product inventory differs")
    codes = document["bindings"]
    require(drafting.code_bindings(codes["design"]["path"], codes["source_validator"]["path"]) == codes,
            "Draft code/runtime bindings changed")
    source, split = drafting.verify_materialization(document["source"]["root"],
        document["source"]["commit"]["sha256"], codes["source_validator"]["path"])
    require(source == document["source"] and split == document["split"], "Draft source or deterministic split changed")
    require(commit["source_materialization_commit_sha256"] == source["commit"]["sha256"],
            "Draft source COMMIT binding mismatch")
    overlap = document["overlap"]["acceptance"]
    require(drafting.verify_overlap(overlap["path"], overlap["sha256"], source) == document["overlap"],
            "Accepted overlap evidence changed")
    selected = [r for r in split["rows"] if r["split_role"] == "development"]
    require(len(selected) == DEVELOPMENT_COUNT and split["role_counts"] == {
            "development": 90, "reserved": 90, "unused": 180}, "Wrong development intersection")
    preprocessing = document["development_preprocessing"]
    require(len(preprocessing) == DEVELOPMENT_COUNT
            and [r["item_id"] for r in preprocessing] == [r["item_id"] for r in selected],
            "Draft preprocessing rows differ from ordered development intersection")
    for row, prepared in zip(selected, preprocessing, strict=True):
        original = row["source_record"]
        require(prepared["original_audio"] == binding(drafting.child(Path(source["root"]), original["audio"]["materialized_path"]))
                and prepared["original_annotation"] == binding(drafting.child(Path(source["root"]), original["annotation"]["materialized_path"]))
                and prepared["source_pcm_sha256"] == original["audio"]["decoded"]["decoded_pcm_sha256"],
                "Development preprocessing source binding mismatch")
    return document


def exclusive_output(output, source, draft_directory, receipt_file):
    output = drafting.source_validator.safe_filesystem_path(output)
    require(not output.exists() and output.parent.is_dir(), "Output must be new with an existing parent")
    for protected in (Path(source).resolve(), Path(draft_directory).resolve()):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                "Output overlaps source or draft directory")
    require(not Path(receipt_file).resolve().is_relative_to(output), "Output contains freeze receipt")
    return output


def verify_freeze(draft_file, receipt_file, trusted_sha, output_dir):
    """Require explicit parent receipt and recheck every bound input at call time."""
    drafting.check_sha(trusted_sha, "parent freeze receipt")
    receipt_binding = binding(receipt_file)
    require(receipt_binding["sha256"] == trusted_sha, "Parent freeze receipt SHA mismatch")
    receipt = read_json(receipt_file)
    require(set(receipt) == FREEZE_KEYS
            and receipt["schema"] == "bc_guitarset_development_parent_freeze_v1"
            and receipt["status"] == "authorized_development_only"
            and receipt["policy"] == POLICY, "Parent freeze schema/status/policy mismatch")
    require(receipt["draft"] == binding(draft_file)
            and receipt["draft_commit"] == binding(Path(draft_file).parent / "COMMIT.json"),
            "Parent freeze draft binding mismatch")
    require(receipt["producer_bindings"] == producer_bindings(), "Parent-frozen producer/code/runtime changed")
    document = verify_draft(draft_file)
    require(receipt["source_commit_sha256"] == document["source"]["commit"]["sha256"],
            "Parent freeze source mismatch")
    output = drafting.source_validator.safe_filesystem_path(output_dir)
    require(str(output) == receipt["planned_output_dir"], "Output differs from parent-frozen planned directory")
    return document, receipt


def storage_estimate(recordings=DEVELOPMENT_COUNT):
    """Conservative uncompressed estimate plus explicit JSON/construction margins."""
    require(type(recordings) is int and recordings > 0, "Invalid storage record count")
    condition = POOLS * 247 * 513 * 16 + SAMPLES * 8 + 4 * 1024**2
    record = len(CONDITIONS) * condition + 32 * SAMPLES * 8 + 1024**2
    return {"recordings": recordings, "estimated_product_bytes": recordings * record,
            "reserve_bytes": 1024**3, "required_free_bytes": recordings * record + 1024**3,
            "estimation_basis": "uncompressed spectra/WAV plus 4MiB per-condition metadata margin and 32 crop arrays per record"}


def ensure_capacity(parent, estimate):
    free = shutil.disk_usage(parent).free
    require(free >= estimate["required_free_bytes"], "Insufficient capacity; preserve all products and stop")
    return {**estimate, "filesystem_free_bytes_before": free}


def constructions(x, item_id):
    require(isinstance(x, np.ndarray) and x.dtype == np.float64 and x.shape == (SAMPLES,)
            and np.isfinite(x).all(), "Construction requires finite mono float64 128000-sample crop")
    require(isinstance(item_id, str) and item_id, "Item ID required")
    seed = int.from_bytes(hashlib.sha256((POLICY["seed_prefix"] + item_id).encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    knots = rng.uniform(-np.pi, np.pi, (3, 65))
    unwrapped = np.unwrap(knots, axis=1)
    knot_times = np.arange(65, dtype=np.float64) * .125
    times = np.arange(SAMPLES, dtype=np.float64) / RATE
    phases = np.asarray([np.interp(times, knot_times, row) for row in unwrapped])
    closed_phases = np.stack((phases[0], phases[1], phases[0] + phases[1]))
    carriers = 2 * np.pi * np.array([500., 750., 1250.])[:, None] * times
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        closed = .2 * np.cos(carriers + closed_phases).sum(axis=0)
        independent = .2 * np.cos(carriers + phases).sum(axis=0)
        cr, ir = float(np.sqrt(np.mean(closed**2))), float(np.sqrt(np.mean(independent**2)))
        require(cr > 0 and ir > 0, "Unsupported zero prototype RMS")
        b = .25 * x
        if np.any((x != 0) & (b == 0)):
            raise FloatingPointError("Unsupported background-gain underflow")
        rms = float(np.sqrt(np.mean(b**2)))
        if not np.isfinite(rms) or (rms == 0 and np.any(b)):
            raise FloatingPointError("Unsupported background RMS arithmetic")
        uc, ui = closed / cr, independent / ir
        waves = {"baseline": b, "common_gain": .1 * b, "polarity": -b}
        arrays = {"standardized_crop": x, "background": b, "sample_times_seconds": times,
            "phase_knot_times_seconds": knot_times, "phase_knots_wrapped": knots,
            "phase_knots_unwrapped": unwrapped, "independent_phase_trajectories": phases,
            "closed_phase_trajectories": closed_phases, "closed_prototype": closed,
            "independent_prototype": independent, "closed_unit_rms": uc, "independent_unit_rms": ui,
            "prototype_rms": np.asarray([cr, ir]), "background_rms": np.asarray(rms)}
        for label, db in (("minus6db", -6), ("0db", 0)):
            for kind, unit in (("closed", uc), ("independent", ui)):
                injection = rms * 10**(db / 20) * unit
                arrays[f"{kind}_{label}_injection"] = injection
                waves[f"{kind}_{label}"] = b + injection
    require(tuple(waves) == CONDITIONS and all(np.isfinite(v).all() for v in waves.values()),
            "Invalid derivative construction")
    return {"waveforms": waves, "arrays": arrays,
        "metadata": {"item_id": item_id, "seed": seed, "bit_generator": "PCG64",
            "status": "ok" if rms > 0 else "unsupported_zero_background_rms",
            "background_rms": rms, "prototype_rms": {"closed": cr, "independent": ir},
            "prototype_normalization_factors": {"closed": 1 / cr, "independent": 1 / ir},
            "injected_full_record_rms_matched": rms > 0, "framewise_power_matched": False,
            "mixtures_clipped_or_normalized": False,
            "zero_rms_interpretation": "zero derivatives retained; relative-level intervention unsupported"}}


def save_arrays(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def save_waveform(path, samples):
    require(samples.dtype == np.float64 and samples.shape == (SAMPLES,) and np.isfinite(samples).all(),
            "Derivative must be finite mono float64 with 128000 samples")
    with Path(path).open("xb") as stream:
        wavfile.write(stream, RATE, samples)
    rate, decoded = wavfile.read(path)
    require(rate == RATE and decoded.dtype == np.float64 and np.array_equal(decoded, samples),
            "Float64 derivative WAV failed exact roundtrip")


def descriptor(metadata):
    require(metadata["pool_count"] == POOLS and metadata["input_samples"] == SAMPLES
            and metadata["discarded_tail_samples"] == 0, "Expected two complete four-second pools")
    pools = []
    for index, pool in enumerate(metadata["pools"]):
        cells = pool["cells"]
        require(pool["pool_index"] == index and pool["start_sample"] == index * 64000
                and pool["coefficient_rows"] == 247
                and [c["frequency_bins"] for c in cells] == extractor.pair_grid().tolist(),
                "Extracted pool/grid differs from frozen measurement")
        pools.append({"pool_index": index, "pool_status": pool["status"],
            "target": next(c for c in cells if c["frequency_bins"] == TARGET),
            "grid_cell_count": len(cells), "eligible_cell_count": sum(c["eligible"] for c in cells),
            "grid_eligibility": [c["eligible"] for c in cells],
            "grid_status": [c["status"] for c in cells],
            "grid_squared_bicoherence": [c["squared_bicoherence"] for c in cells],
            "grid_raw_squared_bicoherence": [c["primitive"]["squared_bicoherence"] for c in cells]})
    require(len(pools) == POOLS, "Pool record count mismatch")
    return {"pools": pools}


def nuisance(reference, other):
    require(len(reference["pools"]) == len(other["pools"]) == POOLS, "Nuisance pools mismatch")
    records = []
    for first, second in zip(reference["pools"], other["pools"], strict=True):
        require(first["pool_index"] == second["pool_index"], "Nuisance pool identity mismatch")
        a, b = first["grid_eligibility"], second["grid_eligibility"]
        differences = [abs(x - y) for x, y in zip(first["grid_squared_bicoherence"],
            second["grid_squared_bicoherence"], strict=True) if x is not None and y is not None]
        raw_pairs = list(zip(first["grid_raw_squared_bicoherence"], second["grid_raw_squared_bicoherence"], strict=True))
        raw_differences = [abs(x-y) for x, y in raw_pairs if x is not None and y is not None]
        ta, tb = first["target"], second["target"]
        records.append({"pool_index": first["pool_index"], "grid_denominator": len(a),
            "grid_eligibility_agreement_count": sum(x == y for x, y in zip(a, b, strict=True)),
            "missing_to_finite_count": sum(not x and y for x, y in zip(a, b, strict=True)),
            "finite_to_missing_count": sum(x and not y for x, y in zip(a, b, strict=True)),
            "common_finite_cell_count": len(differences),
            "maximum_finite_b2_difference": max(differences) if differences else None,
            "primary_difference_scope": "reported_energy_masked_b2_common_eligible_cells",
            "raw_common_finite_cell_count": len(raw_differences),
            "raw_maximum_finite_b2_difference": max(raw_differences) if raw_differences else None,
            "raw_missing_to_finite_count": sum(x is None and y is not None for x, y in raw_pairs),
            "raw_finite_to_missing_count": sum(x is not None and y is None for x, y in raw_pairs),
            "target_eligibility_agreement": ta["eligible"] == tb["eligible"],
            "target_missing_to_finite": not ta["eligible"] and tb["eligible"],
            "target_finite_to_missing": ta["eligible"] and not tb["eligible"],
            "target_difference": tb["squared_bicoherence"] - ta["squared_bicoherence"]
                if ta["eligible"] and tb["eligible"] else None})
    return records


def grouped_summary(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    groups = []
    for identity, records in sorted(grouped.items()):
        values = [r["difference"] for r in records if r["difference"] is not None]
        groups.append({key: identity, "recording_denominator": len(records), "covered_recordings": len(values),
            "pool_denominator": sum(r["pool_denominator"] for r in records),
            "paired_covered_pools": sum(r["covered_pools"] for r in records),
            "mean_difference": float(np.mean(values)) if values else None})
    means = [g["mean_difference"] for g in groups if g["mean_difference"] is not None]
    return {"group_denominator": len(groups), "covered_groups": len(means),
            "equal_group_mean_difference": float(np.mean(means)) if means else None, "rows": groups}


def paired_summary(rows):
    values = [r["difference"] for r in rows if r["difference"] is not None]
    return {"recording_denominator": len(rows), "covered_recordings": len(values),
        "pool_denominator": sum(r["pool_denominator"] for r in rows),
        "paired_covered_pools": sum(r["covered_pools"] for r in rows),
        "positive_recordings": sum(v > 0 for v in values), "zero_recordings": sum(v == 0 for v in values),
        "negative_recordings": sum(v < 0 for v in values),
        "equal_score": grouped_summary(rows, "score_id"),
        "equal_player_secondary": grouped_summary(rows, "player_id"), "per_recording": rows}


def aggregate(records):
    baseline = [p for r in records for p in r["conditions"]["baseline"]["pools"]]
    result = {"version": VERSION, "policy": POLICY, "recording_denominator": len(records),
        "independent_replay_status": "required_before_accepting_values",
        "baseline": {"target_pool_denominator": len(baseline),
            "target_covered_pools": sum(p["target"]["eligible"] for p in baseline),
            "grid_cell_denominator": sum(p["grid_cell_count"] for p in baseline),
            "grid_eligible_cells": sum(p["eligible_cell_count"] for p in baseline)},
        "levels": {}, "nuisance": {}, "per_recording": records}
    for level in ("minus6db", "0db"):
        rows = []
        for record in records:
            paired = []
            for closed_pool, independent_pool in zip(record["conditions"][f"closed_{level}"]["pools"],
                    record["conditions"][f"independent_{level}"]["pools"], strict=True):
                require(closed_pool["pool_index"] == independent_pool["pool_index"], "Paired pool identity mismatch")
                a, b = closed_pool["target"], independent_pool["target"]
                paired.append({"pool_index": closed_pool["pool_index"],
                    "closed_eligible": a["eligible"], "independent_eligible": b["eligible"],
                    "closed_status": a["status"], "independent_status": b["status"],
                    "closed_b2": a["squared_bicoherence"], "independent_b2": b["squared_bicoherence"],
                    "difference": a["squared_bicoherence"] - b["squared_bicoherence"]
                        if a["eligible"] and b["eligible"] else None})
            values = [p["difference"] for p in paired if p["difference"] is not None]
            rows.append({key: record[key] for key in ("item_id", "player_id", "score_id", "performance", "style_from_score_prefix")} |
                {"pool_denominator": len(paired), "covered_pools": len(values),
                 "difference": float(np.mean(values)) if values else None, "paired_pools": paired})
        result["levels"][level] = paired_summary(rows) | {"performance_strata": {
            label: paired_summary([r for r in rows if r["performance"] == label]) for label in ("comp", "solo")}}
    for condition in ("common_gain", "polarity"):
        checks = [p for r in records for p in r["nuisance"][condition]]
        maxima = [p["maximum_finite_b2_difference"] for p in checks if p["maximum_finite_b2_difference"] is not None]
        raw_maxima = [p["raw_maximum_finite_b2_difference"] for p in checks if p["raw_maximum_finite_b2_difference"] is not None]
        result["nuisance"][condition] = {key: sum(p[key] for p in checks) for key in (
            "grid_denominator", "grid_eligibility_agreement_count", "missing_to_finite_count",
            "finite_to_missing_count", "common_finite_cell_count", "target_eligibility_agreement",
            "target_missing_to_finite", "target_finite_to_missing", "raw_common_finite_cell_count",
            "raw_missing_to_finite_count", "raw_finite_to_missing_count")}
        result["nuisance"][condition].update(pool_denominator=len(checks),
            maximum_finite_b2_difference=max(maxima) if maxima else None,
            primary_difference_scope="reported_energy_masked_b2_common_eligible_cells",
            raw_maximum_finite_b2_difference=max(raw_maxima) if raw_maxima else None)
    return result


def commit_directory(output, fields):
    inventory = drafting.source_validator._source_inventory(output)
    require("COMMIT.json" not in inventory and "FAILED.json" not in inventory, "Cannot commit failed or committed output")
    write_json(output / "COMMIT.json", {"status": "committed", **fields,
        "products": {name: fp(path) for name, path in sorted(inventory.items())}})


def process_record(row, prepared, source, output):
    """Single development record; explicit guard precedes any path/decode call."""
    require(row["split_role"] == "development", "Reserved/unused recording reached producer")
    require(row["item_id"] == prepared["item_id"], "Preprocessing identity differs")
    original = row["source_record"]
    path = drafting.child(Path(source), original["audio"]["materialized_path"])
    native = drafting.decode_selected(path, original["audio"]["decoded"])
    crop, provenance = drafting.standardize(native, original["audio"]["decoded"]["sample_rate_hz"])
    require(provenance == prepared["preprocessing"], "Actual crop/resampling differs from frozen draft")
    destination = output / row["item_id"]
    require(destination.parent == output and destination.name == row["item_id"], "Unsafe item output path")
    destination.mkdir(exist_ok=False)
    built = constructions(crop, row["item_id"])
    save_arrays(destination / "construction.npz", built["arrays"])
    write_json(destination / "construction.json", built["metadata"] | {
        "frozen_preprocessing": prepared, "standardized_crop_saved": True,
        "source_pcm_sha256": original["audio"]["decoded"]["decoded_pcm_sha256"]})
    record = {key: row[key] for key in ("item_id", "player_id", "score_id", "performance", "style_from_score_prefix", "split_role")}
    record.update(source_record=original, construction_status=built["metadata"]["status"], conditions={})
    for condition in CONDITIONS:
        samples = built["waveforms"][condition]
        save_waveform(destination / f"{condition}.wav", samples)
        measured = extractor.extract(samples, RATE)
        measured["metadata"]["construction_status"] = built["metadata"]["status"]
        save_arrays(destination / f"{condition}.npz", measured["arrays"])
        write_json(destination / f"{condition}.json", measured["metadata"])
        record["conditions"][condition] = descriptor(measured["metadata"])
    record["nuisance"] = {condition: nuisance(record["conditions"]["baseline"], record["conditions"][condition])
                           for condition in ("common_gain", "polarity")}
    write_json(destination / "summary.json", record)
    return record


def run(draft_file, freeze_receipt, freeze_sha256, output_dir):
    document, receipt = verify_freeze(draft_file, freeze_receipt, freeze_sha256, output_dir)
    output = exclusive_output(output_dir, document["source"]["root"], Path(draft_file).parent, freeze_receipt)
    capacity = ensure_capacity(output.parent, storage_estimate())
    selected = [r for r in document["split"]["rows"] if r["split_role"] == "development"]
    require(len(selected) == DEVELOPMENT_COUNT, "Exactly 90 development recordings required")
    prepared = document["development_preprocessing"]
    require([r["item_id"] for r in selected] == [r["item_id"] for r in prepared], "Selected preprocessing order mismatch")
    output.mkdir(exist_ok=False)
    records = []
    try:
        write_json(output / "frozen_draft.json", document)
        write_json(output / "parent_freeze_receipt.json", receipt)
        write_json(output / "storage_estimate.json", capacity)
        for row, preparation in zip(selected, prepared, strict=True):
            ensure_capacity(output, storage_estimate(len(selected) - len(records)))
            records.append(process_record(row, preparation, document["source"]["root"], output))
            print(f"BC GuitarSet development {len(records)}/{len(selected)}: {row['item_id']}", file=sys.stderr, flush=True)
        summary = aggregate(records)
        require(summary["recording_denominator"] == DEVELOPMENT_COUNT
                and summary["baseline"]["target_pool_denominator"] == DEVELOPMENT_COUNT * POOLS
                and summary["baseline"]["grid_cell_denominator"] == DEVELOPMENT_COUNT * POOLS * 228,
                "Complete development denominators mismatch")
        write_json(output / "summary.json", summary)
        end_document, end_receipt = verify_freeze(draft_file, freeze_receipt, freeze_sha256, output_dir)
        require(end_document == document and end_receipt == receipt, "Frozen bindings changed during run; no COMMIT")
        write_json(output / "verification.json", {"frozen_start_end_equal": True,
            "recordings": len(records), "derivative_float64_wavs": len(records) * len(CONDITIONS),
            "condition_pools": len(records) * len(CONDITIONS) * POOLS,
            "cell_records": len(records) * len(CONDITIONS) * POOLS * 228,
            "derivative_float64_wavs_exact_roundtrip": True,
            "reserved_or_unused_decoded_or_measured": False, "classifier_fits": 0,
            "independent_replay_status": "required_before_accepting_values"})
        commit_directory(output, {"kind": "BC_GuitarSet_development_only_v1",
            "parent_freeze_receipt_sha256": freeze_sha256, "draft_sha256": receipt["draft"]["sha256"],
            "source_commit_sha256": document["source"]["commit"]["sha256"],
            "external_gate_passed": False, "classifier_admitted": False,
            "independent_replay_passed": False})
    except BaseException as exc:
        if not (output / "FAILED.json").exists():
            write_json(output / "FAILED.json", {"status": "failed_no_commit", "exception": type(exc).__name__,
                "message": str(exc), "completed_recordings": len(records), "outputs_preserved": True})
        raise
    return {"output": str(output), "commit_sha256": fp(output / "COMMIT.json")["sha256"],
            "recordings": len(records), "independent_replay_required": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("draft", "freeze-receipt", "freeze-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(drafting.canonical(run(args.draft, args.freeze_receipt, args.freeze_sha256, args.output)).decode(), end="")


if __name__ == "__main__":
    main()
