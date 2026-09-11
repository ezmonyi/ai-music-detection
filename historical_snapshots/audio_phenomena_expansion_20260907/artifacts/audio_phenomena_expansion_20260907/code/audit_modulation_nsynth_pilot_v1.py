#!/usr/bin/env python3
"""Independent read-only replay auditor; imports neither producer nor Q DSP.

Replays derivative samples and stored modulation powers, not the filter/Hilbert
DSP that created those powers. A successful audit is not an external gate pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

VERSION = "independent_modulation_nsynth_pilot_audit_v1_20260907"
SOURCE_COMMIT = "85f70e63a033e1509d3c2fbad6c4561c3f1fce2511117756af64f78b9c39e330"
SOURCE_ARCHIVE = "0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c"
CODE_PINS = {
    "runner": "eef6c3d81d6ba8e057da6e51aebe0e27522b62900de8d948430f9ddbaa091c33",
    "runner_tests": "d7b8a26562e6a98744fdcbd2a9a3f4dac3d04ed67db56080cc708670298390dd",
    "candidate": "8742a44746d638024832db6b2d8933a5925a811d481785cb1d741b1c694984fb",
    "candidate_tests": "89e24161a07af056bf1dc9af3b604b93f7da98bbfa4474e5eda40faab8d03377",
    "protocol": "d5f601274f8bb1091c3c97eda5cc3b613b8d3c06449d3fa95b8263d8cadc406c",
    "source_parent_acceptance": "df5cc7a0ab882e581ff044ded6aa8396a25afae0380d07c583906f286e93df81",
    "candidate_parent_acceptance": "28e7a5126a4fd4d11183defd780abd0ca16b912c402fefd980cd47ca673b1028",
    "acquisition_code": "8620edf65943f6be8557fa1328fa592d4eee2110ffd7dfdea38dabf41c035447",
}
CONDITIONS = ("baseline", "common_gain", "polarity", "am8_depth02", "am8_depth06",
              "am48_depth02", "am48_depth06", "chirp16to64_depth06")
BANDS = ((250, 1000), (1000, 3000), (3000, 7000))
BAND_NAMES = [f"{lo}_{hi}hz" for lo, hi in BANDS]
FEATURES = [f"Q_{name}_{metric}_median" for name in BAND_NAMES for metric in ("fast_fraction", "entropy")]
STATUS = "pilot_only_not_external_gate_passed"
SOURCE_NOTE_COUNT = 4096
Q_CONFIG = {
    "sample_rate_hz": 16000, "envelope_rate_hz": 500, "butterworth_prototype_order": 4,
    "filter_padtype": "odd", "filter_padlen_samples": 27, "edge_discard_seconds_each": .25,
    "window_seconds": 2., "hop_seconds": 2., "window": "periodic_hann",
    "antialias_taps": 1281, "antialias_cutoff_hz": 250., "antialias_kaiser_beta": 8.6,
    "relative_band_energy_floor": 1e-8, "normalized_total_modulation_power_floor": 1e-8,
    "normalized_analysis_modulation_power_floor": 1e-8, "maximum_input_absolute_value": 1e150,
    "minimum_summary_windows": 1, "fast_interval_hz": [16., 64.], "analysis_interval_hz": [2., 128.]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    return {"bytes": Path(path).stat().st_size, "sha256": digest(path)}


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"nonfinite JSON token: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def contained(root, relative):
    part = Path(relative)
    require(not part.is_absolute() and ".." not in part.parts, "unsafe inventory path")
    require(not any((root / Path(*part.parts[:i])).is_symlink() for i in range(1, len(part.parts) + 1)),
            "symlink inventory path")
    path = (root / part).resolve(strict=True)
    require(path.is_file() and path.is_relative_to(root), "inventory path escaped root")
    return path


def verify_products(root, expected_commit=None):
    root = Path(root).resolve(strict=True)
    commit_path = contained(root, "COMMIT.json")
    if expected_commit:
        require(digest(commit_path) == expected_commit, "COMMIT identity mismatch")
    receipt = load(commit_path)
    require(receipt.get("status") == "committed", "publication is not committed")
    products = receipt.get("products")
    require(isinstance(products, dict), "invalid product inventory")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual == set(products) | {"COMMIT.json"}, "exact product file set mismatch")
    for relative, expected in products.items():
        require(file_record(contained(root, relative)) == expected, f"product hash mismatch: {relative}")
    return receipt


def compare(actual, expected, path="root"):
    """Exact structures/None/bools; tight numerical replay tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"keys differ: {path}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(actual) == len(expected), f"length differs: {path}")
        for i, value in enumerate(expected):
            compare(actual[i], value, f"{path}[{i}]")
    elif expected is None or isinstance(expected, (str, bool)):
        require(type(actual) is type(expected) and actual == expected, f"value differs: {path}")
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f"number differs: {path}")


def verify_bindings(bindings):
    require(set(bindings) == set(CODE_PINS) | {"runtime"}, "binding inventory mismatch")
    for role, pinned in CODE_PINS.items():
        record = bindings[role]
        require(record["sha256"] == pinned, f"unaccepted code pin: {role}")
        require(file_record(record["path"]) == {k: record[k] for k in ("bytes", "sha256")},
                f"bound file changed: {role}")
    runtime = bindings["runtime"]
    for record in [runtime["python_executable"], *runtime["module_files"].values()]:
        require(file_record(record["path"]) == {k: record[k] for k in ("bytes", "sha256")}, "runtime file changed")


def verify_roster(frozen):
    source = Path(frozen["sources"]["source_dir"]).resolve(strict=True)
    publication = verify_products(source, SOURCE_COMMIT)
    require(publication["products"]["nsynth-test.jsonwav.tar.gz"]["sha256"] == SOURCE_ARCHIVE,
            "official archive identity mismatch")
    hashes = {**publication["products"], "COMMIT.json": file_record(source / "COMMIT.json")}
    compare(frozen["sources"]["all_source_byte_hashes"], hashes, "source_bytes")
    metadata = load(source / "nsynth-test/examples.json")
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines() if line]
    lookup = {row["id"]: row for row in rows}
    require(len(rows) == len(lookup) == len(metadata) == SOURCE_NOTE_COUNT, "full note inventory count mismatch")
    require(set(lookup) == set(metadata), "metadata note IDs differ")
    groups = defaultdict(list)
    for row in rows:
        note_id = row["id"]
        compare(row["metadata"], metadata[note_id], f"metadata:{note_id}")
        path = contained(source, f"nsynth-test/audio/{note_id}.wav")
        require(row["path"] == str(path), "source path mismatch")
        require(row["file_sha256"] == hashes[str(path.relative_to(source))]["sha256"], "source waveform hash mismatch")
        require(row["native_sample_rate"] == 16000 and row["native_channels"] == 1
                and row["frames"] == 64000 and row["duration_s"] == 4
                and row["ai_human_label"] is None, "source decode inventory contract mismatch")
        groups[row["metadata"]["instrument_str"]].append(row)
    require(len(groups) == 53, "instrument inventory count mismatch")
    instruments = sorted(groups, key=lambda s: (hashlib.sha256(("Q-pilot-20260907|" + s).encode()).hexdigest(), s))
    selected = []
    for instrument in instruments[:27]:
        ordered = sorted(groups[instrument], key=lambda row: (abs(row["metadata"]["pitch"] - 60),
            hashlib.sha256(("Q-note-20260907|" + row["id"]).encode()).hexdigest(), row["id"]))
        seen = set()
        for row in ordered:
            if row["pcm_sha256"] not in seen:
                selected.append(row); seen.add(row["pcm_sha256"])
            if len(seen) == 2:
                break
        require(len(seen) == 2, "cannot select two unique PCM notes")
    pilot_pcm = {row["pcm_sha256"] for name in instruments[:27] for row in groups[name]}
    reserved_pcm = {row["pcm_sha256"] for name in instruments[27:] for row in groups[name]}
    require(not pilot_pcm & reserved_pcm, "pilot/reserved PCM overlap")
    compare(frozen["selection"], {"ordered_instrument_ids": instruments,
        "pilot_instrument_ids": instruments[:27], "reserved_instrument_ids": instruments[27:],
        "selected_notes": selected, "pilot_reserved_pcm_sets_disjoint": True}, "selection")
    compare(frozen["sources"]["all_4096_manifest_sha256"], hashes["manifest.jsonl"]["sha256"])
    compare(frozen["sources"]["all_4096_metadata_file_sha256"], hashes["nsynth-test/examples.json"]["sha256"])
    combined = hashlib.sha256(canonical([{"id": key, "metadata": lookup[key]["metadata"]}
                                        for key in sorted(lookup)])).hexdigest()
    compare(frozen["sources"]["all_4096_manifest_metadata_sha256"], combined)
    return selected


def expected_audio(x, condition):
    if condition in ("baseline", "common_gain", "polarity"):
        return x * {"baseline": .25, "common_gain": .10, "polarity": -.25}[condition]
    time = np.arange(len(x), dtype=np.float64) / 16000
    if condition == "chirp16to64_depth06":
        phase = 2 * np.pi * (16 * time + 6 * time ** 2)
        depth = .6
    else:
        rate = 8 if condition.startswith("am8_") else 48
        depth = .2 if condition.endswith("02") else .6
        require(condition in CONDITIONS, "unknown condition")
        phase = 2 * np.pi * rate * time
    return .25 * x * (1 + depth * np.cos(phase))


def replay_measurement(measurement, audio):
    frequency = np.arange(501, dtype=float) * .5
    compare(measurement["config"], Q_CONFIG, "candidate configuration")
    compare(measurement["frequency_hz"], frequency.tolist(), "frequency bins")
    for key, value in {"version": "modulation_candidate_v1_20260907", "sample_count": 64000,
                       "sample_rate_hz": 16000, "window_count": 1,
                       "discarded_tail_envelope_samples": 750,
                       "covered_seconds": 2., "temporal_coverage_fraction": .5}.items():
        compare(measurement[key], value, key)
    require(len(measurement["bands"]) == 3, "carrier band count")
    features = dict.fromkeys(FEATURES)
    valid = 0
    diagnostics = []
    input_energy = float(np.mean(audio[4000:36000] ** 2))
    for index, (band, name) in enumerate(zip(measurement["bands"], BAND_NAMES)):
        compare(band["name"], name); compare(band["carrier_hz"], list(BANDS[index]))
        require(len(band["windows"]) == 1, "note window count")
        window = band["windows"][0]
        for key, value in {"index": 0, "start_seconds": .25, "end_seconds": 2.25,
                           "sample_count": 1000, "coverage": 1.}.items():
            compare(window[key], value, key)
        compare(window["input_mean_square"], input_energy, "input waveform energy")
        power = np.asarray(window["modulation_power"], dtype=float)
        require(power.shape == (501,) and np.all(np.isfinite(power)) and np.all(power >= 0), "invalid stored powers")
        total = float(np.sum(power)); analysis = float(np.sum(power[4:256]))
        compare(window["total_modulation_power"], total, "total power")
        compare(window["analysis_modulation_power"], analysis, "analysis power")
        require(math.isfinite(window["band_mean_square"]) and window["band_mean_square"] >= 0,
                "invalid carrier energy")
        relative = window["band_mean_square"] / input_energy if input_energy else 0.
        compare(window["relative_band_energy"], relative, "relative carrier energy")
        require(math.isfinite(window["envelope_mean"]) and window["envelope_mean"] >= 0, "invalid envelope mean")
        status = ("silence" if input_energy == 0 else "insufficient_band_energy" if relative < 1e-8 else
                  "no_modulation" if total <= 1e-8 else
                  "insufficient_analysis_modulation" if analysis <= 1e-8 else "ok")
        compare(window["status"], status, "window status")
        compare(band["status"], status, "band status")
        compare(band["missing_reasons"], [] if status == "ok" else [status])
        compare(band["valid_window_count"], int(status == "ok"))
        compare(band["valid_window_fraction"], float(status == "ok"))
        fast, entropy, peak = None, None, None
        if status == "ok":
            valid += 1
            fast = min(1., max(0., float(np.sum(power[32:128]) / analysis)))
            probabilities = power[4:256] / analysis
            nonzero = probabilities[probabilities > 0]
            entropy = min(1., max(0., float(-sum(nonzero * np.log(nonzero)) / math.log(252))))
            peak = float(frequency[4:256][np.argmax(power[4:256])])
        compare(window["fast_fraction"], fast); compare(window["entropy"], entropy)
        features[f"Q_{name}_fast_fraction_median"] = fast
        features[f"Q_{name}_entropy_median"] = entropy
        diagnostics.append({"status": status, "power": power, "peak": peak,
                            "fast_fraction": fast, "entropy": entropy})
    compare(measurement["features"], features, "six recomputed features")
    compare(measurement["valid_band_window_count"], valid)
    compare(measurement["status"], "ok" if valid == 3 else "partial" if valid else "no_valid_band_windows")
    return {"features": features, "bands": diagnostics, "status": measurement["status"]}


def invariant(reference, other):
    errors = [abs(reference["features"][key] - other["features"][key]) for key in FEATURES
              if reference["features"][key] is not None and other["features"][key] is not None]
    return {"max_abs_finite_feature_error": max(errors) if errors else None,
            "comparable_features": len(errors), "feature_denominator": 6,
            "feature_missingness_match": all((reference["features"][key] is None) ==
                                             (other["features"][key] is None) for key in FEATURES),
            "status_match": reference["status"] == other["status"] and
                            [b["status"] for b in reference["bands"]] == [b["status"] for b in other["bands"]]}


def paired(row, replay):
    summaries = []
    for index, name in enumerate(BAND_NAMES):
        bands = {condition: value["bands"][index] for condition, value in replay.items()}
        entry = {"name": name, "condition_status": {c: bands[c]["status"] for c in CONDITIONS}}
        for target in (8, 48):
            high, low = bands[f"am{target}_depth06"], bands[f"am{target}_depth02"]
            peak = high["peak"]
            entry[f"am{target}_depth06_peak"] = {"dominant_peak_hz": peak,
                "absolute_error_hz": abs(peak-target) if peak is not None else None,
                "within_1hz": abs(peak-target) <= 1 if peak is not None else None, "status": high["status"]}
            near = lambda b: float(sum(b["power"][int(2*target)-2:int(2*target)+3])) if b["status"] == "ok" else None
            hp, lp = near(high), near(low)
            entry[f"am{target}_near_power_depth02"] = lp
            entry[f"am{target}_near_power_depth06"] = hp
            entry[f"am{target}_near_power_ratio_depth06_over_depth02"] = hp/lp if hp is not None and lp is not None and lp > 0 else None
        for metric, left, right, key in (
            ("fast_fraction", "am48_depth06", "am8_depth06", "fast_fraction_delta_am48_minus_am8_depth06"),
            ("entropy", "chirp16to64_depth06", "am48_depth06", "entropy_delta_chirp_minus_am48_depth06")):
            a, b = bands[left][metric], bands[right][metric]
            entry[key] = a-b if a is not None and b is not None else None
        summaries.append(entry)
    return {"id": row["id"], "instrument_str": row["metadata"]["instrument_str"], "metadata": row["metadata"],
            "bands": summaries, "common_gain": invariant(replay["baseline"], replay["common_gain"]),
            "polarity": invariant(replay["baseline"], replay["polarity"])}


def weighted(notes, instruments, getter):
    per_instrument = {}
    for name in instruments:
        values = [getter(note) for note in notes if note["instrument_str"] == name]
        require(len(values) == 2, "instrument must retain exactly two notes")
        finite = [float(value) for value in values if value is not None]
        per_instrument[name] = {"mean": sum(finite)/len(finite) if finite else None,
                                "valid_notes": len(finite), "note_denominator": 2}
    means = [entry["mean"] for entry in per_instrument.values() if entry["mean"] is not None]
    return {"instrument_balanced_mean": sum(means)/len(means) if means else None,
            "covered_instruments": len(means), "instrument_denominator": len(instruments),
            "valid_notes": sum(entry["valid_notes"] for entry in per_instrument.values()),
            "note_denominator": 2*len(instruments), "per_instrument": per_instrument}


def aggregates(notes, instruments):
    output = {"bands": {}, "invariance": {}}
    for index, name in enumerate(BAND_NAMES):
        entries = {}
        for metric in ("am8_near_power_ratio_depth06_over_depth02", "am48_near_power_ratio_depth06_over_depth02",
                       "fast_fraction_delta_am48_minus_am8_depth06", "entropy_delta_chirp_minus_am48_depth06"):
            entries[metric] = weighted(notes, instruments, lambda n, m=metric: n["bands"][index][m])
        for rate in (8, 48):
            entries[f"am{rate}_depth06_peak_within_1hz_rate"] = weighted(notes, instruments,
                lambda n, r=rate: n["bands"][index][f"am{r}_depth06_peak"]["within_1hz"])
        for condition in CONDITIONS:
            entries[f"{condition}_quality_valid_rate"] = weighted(notes, instruments,
                lambda n, c=condition: n["bands"][index]["condition_status"][c] == "ok")
        output["bands"][name] = entries
    for condition in ("common_gain", "polarity"):
        output["invariance"][condition] = {metric: weighted(notes, instruments, lambda n, m=metric: n[condition][m])
            for metric in ("max_abs_finite_feature_error", "status_match", "feature_missingness_match")}
    return output


def audit(result_dir, draft_file, freeze_sha256):
    root = Path(result_dir).resolve(strict=True)
    draft_file = Path(draft_file).resolve(strict=True)
    require(digest(draft_file) == freeze_sha256, "reviewed freeze SHA mismatch")
    draft_commit = verify_products(draft_file.parent)
    require(draft_commit.get("kind") == "development_pilot_draft"
            and draft_commit.get("draft_sha256") == freeze_sha256
            and set(draft_commit["products"]) == {"draft.json"}, "draft linkage mismatch")
    result_commit = verify_products(root)
    require(result_commit.get("kind") == "development_pilot_result" and result_commit.get("pilot_status") == STATUS
            and result_commit.get("freeze_sha256") == freeze_sha256, "result linkage mismatch")
    require(digest(root / "freeze.json") == freeze_sha256, "result freeze byte mismatch")
    frozen = load(draft_file)
    require(frozen["planned_run_output_dir"] == str(root), "frozen output path mismatch")
    require(frozen["features_extracted"] is False and frozen["reserved_waveforms_decoded_or_extracted"] is False,
            "draft extraction state mismatch")
    require(frozen["policy"]["status"] == STATUS and frozen["policy"]["conditions"] == list(CONDITIONS)
            and frozen["policy"]["no_numeric_admission_threshold"] is True
            and frozen["policy"]["codecs"] is False and frozen["policy"]["autofit"] is False,
            "frozen policy scope mismatch")
    verify_bindings(frozen["bindings"])
    selected = verify_roster(frozen)
    expected_files = {"freeze.json", "run_receipt.json", "paired_summaries.json"}
    for row in selected:
        for condition in CONDITIONS:
            stem = row["id"] + "__" + condition
            expected_files.update({f"derivatives/{stem}.wav", f"candidate_json/{stem}.json"})
    require(set(result_commit["products"]) == expected_files and len(expected_files) == 867, "432-condition product inventory mismatch")
    notes = []
    for row in selected:
        rate, pcm = wavfile.read(row["path"])
        require(rate == 16000 and pcm.shape == (64000,) and pcm.dtype == np.int16, "selected PCM decode mismatch")
        require(hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest() == row["pcm_sha256"], "selected PCM digest mismatch")
        x = pcm.astype(np.float64) / 32768
        replay = {}
        for condition in CONDITIONS:
            stem = row["id"] + "__" + condition
            wav_path = root / "derivatives" / (stem + ".wav")
            rate, actual = wavfile.read(wav_path)
            expected = expected_audio(x, condition)
            require(rate == 16000 and actual.dtype == np.float64 and actual.shape == (64000,)
                    and np.array_equal(actual, expected), f"derivative samples mismatch: {stem}")
            product = load(root / "candidate_json" / (stem + ".json"))
            for key, value in {"note_id": row["id"], "instrument_str": row["metadata"]["instrument_str"],
                               "condition": condition, "source_file_sha256": row["file_sha256"],
                               "derivative": file_record(wav_path)}.items():
                compare(product[key], value, f"product linkage:{key}")
            replay[condition] = replay_measurement(product["measurement"], actual)
        notes.append(paired(row, replay))
    summary = load(root / "paired_summaries.json")
    compare(summary["notes"], notes, "per-note paired summaries")
    compare(summary["instrument_balanced_descriptive_aggregates"],
            aggregates(notes, frozen["selection"]["pilot_instrument_ids"]), "instrument aggregates")
    strata = {}
    for field in ("instrument_source_str", "instrument_family_str"):
        strata[field] = {}
        for label in sorted({note["metadata"][field] for note in notes}):
            subset = [note for note in notes if note["metadata"][field] == label]
            strata[field][label] = aggregates(subset, sorted({note["instrument_str"] for note in subset}))
    compare(summary["instrument_balanced_source_and_family_strata"], strata, "stratified aggregates")
    require(summary["status"] == STATUS and summary["freeze_sha256"] == freeze_sha256
            and summary["external_gate_passed"] is False and summary["no_numeric_admission_threshold"] is True,
            "summary incorrectly claims admission")
    receipt = load(root / "run_receipt.json")
    expected_receipt = {"status": STATUS, "freeze_sha256": freeze_sha256,
        "code_bindings_start": frozen["bindings"], "code_bindings_end": frozen["bindings"],
        "draft_commit_start": file_record(draft_file.parent / "COMMIT.json"),
        "draft_commit_end": file_record(draft_file.parent / "COMMIT.json"),
        "source_bindings_verified_start_and_end": True, "decoded_source_note_ids": [r["id"] for r in selected],
        "reserved_instrument_ids": frozen["selection"]["reserved_instrument_ids"],
        "reserved_waveforms_decoded_or_extracted": False, "notes": 54, "conditions_per_note": 8,
        "derivative_wavs": 432, "candidate_json_files": 432, "ai_human_labels_assigned": False,
        "classifier_fits": 0, "codecs": False, "external_gate_passed": False}
    compare(receipt, expected_receipt, "run receipt")
    # Rehash publication/bindings after replay so an audit cannot silently bless
    # a result that changed while it was read.
    compare(verify_products(root), result_commit, "result unchanged during audit")
    verify_bindings(frozen["bindings"])
    return {"version": VERSION, "passed": True, "scope": STATUS,
            "auditor_sha256": digest(__file__), "freeze_sha256": freeze_sha256,
            "result_commit_sha256": digest(root / "COMMIT.json"),
            "result_dir": str(root), "selected_notes_replayed": 54, "pilot_instruments": 27,
            "reserved_instruments": 26, "reserved_waveforms_decoded": 0,
            "float64_derivatives_recreated": 432, "stored_spectra_replayed": 1296,
            "products_hashed": 867, "external_gate_passed": False,
            "limitations": ["Stored-power replay is not independent filter/Hilbert/full-DSP recomputation.",
                            "Receipt and output rosters establish recorded scope; they are not OS-level proof of historical non-access.",
                            "No clinical, codec-robustness, physical-instrument-identity or AI/Human validity is established."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("result-dir", "draft", "freeze-sha256", "output"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(not output.exists(), "audit receipt output already exists")
    frozen = load(args.draft)
    for protected in (args.result_dir, Path(args.draft).parent, frozen["sources"]["source_dir"]):
        require(not output.is_relative_to(Path(protected).resolve()), "audit receipt must be outside input artifacts")
    receipt = audit(args.result_dir, args.draft, args.freeze_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical(receipt))
    print(json.dumps({"passed": True, "receipt": str(output), "sha256": digest(output)}))


if __name__ == "__main__":
    main()
