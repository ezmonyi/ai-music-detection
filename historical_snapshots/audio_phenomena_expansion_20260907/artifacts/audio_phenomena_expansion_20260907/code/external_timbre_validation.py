#!/usr/bin/env python3
"""Controlled external-real-audio validation for T continuity descriptors.

Only original isolated source audio is eligible.  Selected-corpus mixtures and
all Demucs outputs are audited but never used as source-identity truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import fftconvolve, resample_poly

from timbre_features import T_MEASURE_NAMES, extract_timbre_features, timbre_boundary_distance


SR = 16_000
CHUNK_SEC = 3.0
SEED = 20260907
MAX_REGISTER_MATCH_CENTS = 300.0
MIN_PITCH_COVERAGE = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    mono -= np.mean(mono, dtype=np.float64)
    if sr != SR:
        divisor = math.gcd(int(sr), SR)
        mono = resample_poly(mono, SR // divisor, int(sr) // divisor).astype(np.float32)
        sr = SR
    return mono, int(sr)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))


def pitch_summary(audio: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """Return approximate median F0 MIDI and voiced/active frame coverage.

    This autocorrelation estimate is a matching covariate, not ground-truth F0.
    """
    frame_length = max(512, int(round(0.064 * sr)))
    hop = max(128, int(round(0.032 * sr)))
    starts = range(0, max(0, len(audio) - frame_length + 1), hop)
    frames = [np.asarray(audio[start : start + frame_length], dtype=np.float64) for start in starts]
    if not frames:
        return float("nan"), 0.0
    levels = np.asarray([rms(frame) for frame in frames])
    active = levels >= max(float(np.max(levels)) * 0.05, 1e-5)
    pitches: list[float] = []
    min_lag = max(1, int(sr / 1000.0))
    max_lag = min(frame_length - 2, int(sr / 65.0))
    window = np.hanning(frame_length)
    for frame, is_active in zip(frames, active):
        if not is_active:
            continue
        centred = (frame - np.mean(frame)) * window
        correlation = fftconvolve(centred, centred[::-1], mode="full")[frame_length - 1 :]
        if correlation[0] <= 1e-10:
            continue
        search = correlation[min_lag : max_lag + 1] / correlation[0]
        lag_offset = int(np.argmax(search))
        periodicity = float(search[lag_offset])
        if periodicity < 0.35:
            continue
        lag = min_lag + lag_offset
        frequency = sr / lag
        pitches.append(69.0 + 12.0 * math.log2(frequency / 440.0))
    active_count = int(np.sum(active))
    coverage = float(len(pitches) / active_count) if active_count else 0.0
    return (float(np.median(pitches)) if pitches else float("nan"), coverage)


def fixed_musdb_chunks(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    count = int(round(CHUNK_SEC * SR))
    required = 2 * count
    if len(audio) < required:
        raise ValueError("source shorter than two non-overlapping chunks")
    spare = len(audio) - required
    margin = min(int(round(0.2 * SR)), spare // 2)
    first_start = margin
    second_start = len(audio) - margin - count
    if first_start + count > second_start:
        raise AssertionError("MUSDB chunks overlap")
    return (
        audio[first_start : first_start + count].copy(),
        audio[second_start : second_start + count].copy(),
        first_start / SR,
        second_start / SR,
    )


def activity_selected_chunks(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Select two non-overlapping MedleyDB windows using only RMS/F0 quality."""
    count = int(round(CHUNK_SEC * SR))
    starts = list(range(0, len(audio) - count + 1, count))
    candidates: list[tuple[int, float, float, float]] = []
    for start in starts:
        chunk = audio[start : start + count]
        level = rms(chunk)
        if level < 1e-5:
            continue
        midi, coverage = pitch_summary(chunk)
        candidates.append((start, level, midi, coverage))
    informative = [x for x in candidates if np.isfinite(x[2]) and x[3] >= MIN_PITCH_COVERAGE]
    if len(informative) < 2:
        raise ValueError("fewer than two pitch-informative active chunks")
    # Limit to the most active windows, then choose the closest-register pair.
    pool = sorted(informative, key=lambda x: (-x[1], x[0]))[:12]
    first, second = min(
        ((a, b) for i, a in enumerate(pool) for b in pool[i + 1 :]),
        key=lambda pair: (abs(pair[0][2] - pair[1][2]), -min(pair[0][1], pair[1][1]), pair[0][0]),
    )
    if first[0] > second[0]:
        first, second = second, first
    return (
        audio[first[0] : first[0] + count].copy(),
        audio[second[0] : second[0] + count].copy(),
        first[0] / SR,
        second[0] / SR,
    )


def load_source_records(musdb_manifest: Path, medley_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    inventory = [
        {"corpus": "MUSDB18", "candidate_rows": 144, "original_isolated_available": 144,
         "used_kind": "reference vocal stem", "status": "eligible_after_quality_screen"},
        {"corpus": "MedleyDB_sample", "candidate_rows": 9, "original_isolated_available": 9,
         "used_kind": "metadata-labelled original stem", "status": "eligible_groups_after_quality_screen"},
        {"corpus": "MoisesDB_selected_mirror", "candidate_rows": 239, "original_isolated_available": 0,
         "used_kind": "none", "status": "excluded_mixture_only"},
        {"corpus": "URMP_Reduced_selected", "candidate_rows": 44, "original_isolated_available": 0,
         "used_kind": "none", "status": "excluded_AuMix_only"},
        {"corpus": "Demucs_outputs", "candidate_rows": 0, "original_isolated_available": 0,
         "used_kind": "none", "status": "excluded_estimated_sources_not_truth"},
    ]
    for line in musdb_manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            {
                "corpus": "MUSDB18",
                "source_id": row["track_id"],
                "source_path": Path(row["reference_vocals"]),
                "track_name": row["track_name"],
                "instrument_group": "vocal",
                "identity_semantics": "track_vocal_stem_not_verified_singer",
                "chunk_selector": "fixed_nonoverlap",
            }
        )
    for metadata_path in sorted(medley_root.glob("Audio/*/*_METADATA.yaml")):
        metadata = yaml.safe_load(metadata_path.read_text())
        track_dir = metadata_path.parent
        for key, stem in sorted(metadata.get("stems", {}).items()):
            instrument = str(stem.get("instrument", "unknown"))
            path = track_dir / metadata["stem_dir"] / stem["filename"]
            normalized = instrument.lower().replace(" ", "_")
            if normalized == "female_singer":
                group = "vocal"
            elif normalized == "acoustic_guitar":
                group = "acoustic_guitar"
            else:
                group = normalized
            records.append(
                {
                    "corpus": "MedleyDB_sample",
                    "source_id": f"{track_dir.name}_{key}",
                    "source_path": path,
                    "track_name": track_dir.name,
                    "instrument_group": group,
                    "identity_semantics": "metadata_stem_channel_not_verified_performer",
                    "chunk_selector": "rms_f0_quality_only",
                }
            )
    return records, inventory


def prepare_chunks(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["source_path"])
        base = {k: str(v) if isinstance(v, Path) else v for k, v in record.items()}
        if not path.is_file():
            audit.append({**base, "status": "excluded_missing_file", "reason": "source path absent"})
            continue
        try:
            audio, sr = read_mono(path)
            if record["chunk_selector"] == "fixed_nonoverlap":
                left, right, left_start, right_start = fixed_musdb_chunks(audio)
            else:
                left, right, left_start, right_start = activity_selected_chunks(audio)
            left_pitch, left_coverage = pitch_summary(left, sr)
            right_pitch, right_coverage = pitch_summary(right, sr)
            reason = ""
            if rms(left) < 1e-5 or rms(right) < 1e-5:
                reason = "low_energy_chunk"
            elif not np.isfinite(left_pitch) or not np.isfinite(right_pitch):
                reason = "no_informative_pitch"
            elif min(left_coverage, right_coverage) < MIN_PITCH_COVERAGE:
                reason = "low_pitch_coverage"
            status = "eligible" if not reason else "excluded_quality"
            audit.append({
                **base, "status": status, "reason": reason,
                "source_sha256": sha256_file(path), "native_sr_hz": sf.info(path).samplerate,
                "native_duration_sec": sf.info(path).duration,
                "left_start_sec": left_start, "right_start_sec": right_start,
                "chunk_duration_sec": CHUNK_SEC, "chunks_overlap": 0,
                "left_rms": rms(left), "right_rms": rms(right),
                "left_pitch_midi_approx": left_pitch, "right_pitch_midi_approx": right_pitch,
                "left_pitch_coverage": left_coverage, "right_pitch_coverage": right_coverage,
            })
            if status == "eligible":
                chunks.append({**record, "left": left, "right": right,
                               "left_start_sec": left_start, "right_start_sec": right_start,
                               "left_pitch": left_pitch, "right_pitch": right_pitch,
                               "left_pitch_coverage": left_coverage, "right_pitch_coverage": right_coverage,
                               "source_sha256": sha256_file(path)})
        except Exception as error:
            audit.append({**base, "status": "excluded_error", "reason": f"{type(error).__name__}: {error}"})
    return chunks, audit


def choose_pairs(chunks: list[dict[str, Any]], max_musdb: int) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    exclusions: list[dict[str, Any]] = []
    ordered = sorted(chunks, key=lambda x: stable_key(x["source_id"]))
    musdb_used = 0
    for anchor in ordered:
        if anchor["corpus"] == "MUSDB18" and musdb_used >= max_musdb:
            continue
        candidates = [
            other for other in chunks
            if other["source_id"] != anchor["source_id"]
            and other["corpus"] == anchor["corpus"]
            and other["instrument_group"] == anchor["instrument_group"]
        ]
        if not candidates:
            exclusions.append({"source_id": anchor["source_id"], "reason": "no_same_instrument_peer"})
            continue
        peer = min(candidates, key=lambda x: (abs(x["right_pitch"] - anchor["right_pitch"]), stable_key(x["source_id"])))
        residual = abs(peer["right_pitch"] - anchor["right_pitch"]) * 100.0
        if residual > MAX_REGISTER_MATCH_CENTS:
            exclusions.append({"source_id": anchor["source_id"], "reason": "no_peer_within_300_cents",
                               "nearest_residual_cents": residual})
            continue
        pairs.append((anchor, peer))
        if anchor["corpus"] == "MUSDB18":
            musdb_used += 1
    return pairs, exclusions


def level_match(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray(target * (rms(reference) / max(rms(target), 1e-12)), dtype=np.float32)


def bootstrap_median_ci(values: list[float]) -> list[float] | None:
    array = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(array) < 2:
        return None
    rng = np.random.default_rng(SEED)
    samples = rng.choice(array, size=(5000, len(array)), replace=True)
    return [float(x) for x in np.percentile(np.median(samples, axis=1), [2.5, 97.5])]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    stimuli = args.output / "stimuli"
    stimuli.mkdir(exist_ok=True)
    records, inventory = load_source_records(args.musdb_manifest, args.medley_root)
    chunks, chunk_audit = prepare_chunks(records)
    pairs, pair_exclusions = choose_pairs(chunks, args.max_musdb)
    if not pairs:
        raise RuntimeError("no eligible source pairs")

    pair_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    paired_results: list[dict[str, Any]] = []
    for index, (anchor, peer) in enumerate(pairs):
        left = np.asarray(anchor["left"], dtype=np.float32)
        same_right = level_match(left, anchor["right"])
        switched_right = level_match(left, peer["right"])
        scale = 0.90 / max(float(np.max(np.abs(left))), float(np.max(np.abs(same_right))),
                           float(np.max(np.abs(switched_right))), 1e-12)
        left = left * scale
        same_right = same_right * scale
        switched_right = switched_right * scale
        variants = {
            "same_source": (left, same_right),
            "switched_source_register_matched": (left, switched_right),
            "same_source_gain_half": (left, same_right * 0.5),
            "same_source_polarity_flip": (left, -same_right),
        }
        pair_id = f"pair_{index:03d}_{anchor['source_id']}"
        pair_rows.append({
            "pair_id": pair_id, "corpus": anchor["corpus"],
            "instrument_group": anchor["instrument_group"],
            "anchor_source_id": anchor["source_id"], "peer_source_id": peer["source_id"],
            "anchor_identity_semantics": anchor["identity_semantics"],
            "peer_identity_semantics": peer["identity_semantics"],
            "anchor_source_path": str(anchor["source_path"]), "peer_source_path": str(peer["source_path"]),
            "anchor_source_sha256": anchor["source_sha256"], "peer_source_sha256": peer["source_sha256"],
            "anchor_left_start_sec": anchor["left_start_sec"],
            "anchor_right_start_sec": anchor["right_start_sec"],
            "peer_right_start_sec": peer["right_start_sec"], "chunk_duration_sec": CHUNK_SEC,
            "anchor_chunks_overlap": 0,
            "anchor_left_pitch_midi_approx": anchor["left_pitch"],
            "anchor_right_pitch_midi_approx": anchor["right_pitch"],
            "peer_right_pitch_midi_approx": peer["right_pitch"],
            "same_boundary_pitch_residual_cents": abs(anchor["left_pitch"] - anchor["right_pitch"]) * 100.0,
            "switched_boundary_pitch_residual_cents": abs(anchor["left_pitch"] - peer["right_pitch"]) * 100.0,
            "counterfactual_right_register_match_cents": abs(anchor["right_pitch"] - peer["right_pitch"]) * 100.0,
            "anchor_left_pitch_coverage": anchor["left_pitch_coverage"],
            "anchor_right_pitch_coverage": anchor["right_pitch_coverage"],
            "peer_right_pitch_coverage": peer["right_pitch_coverage"],
        })
        measured_by_variant: dict[str, dict[str, Any]] = {}
        for name, (variant_left, variant_right) in variants.items():
            waveform = np.concatenate([variant_left, variant_right]).astype(np.float32)
            audio_path = stimuli / f"{index:03d}_{name}.wav"
            sf.write(audio_path, waveform, SR, subtype="PCM_16")
            boundary = timbre_boundary_distance(variant_left, variant_right, SR)
            measured = {**extract_timbre_features(waveform, SR), **boundary}
            measured_by_variant[name] = measured
            recipe = {
                "pair_id": pair_id, "variant": name, "audio_path": str(audio_path),
                "audio_sha256": sha256_file(audio_path), "float_waveform_sha256": sha256_array(waveform),
                "duration_sec": len(waveform) / SR, "sample_rate_hz": SR,
                "left_source_id": anchor["source_id"],
                "right_source_id": peer["source_id"] if "switched" in name else anchor["source_id"],
                "right_transform": "gain_0.5" if "gain" in name else "polarity_-1" if "polarity" in name else "none",
                "rms_level_matching": int("gain" not in name), "hard_concat_no_crossfade": 1,
            }
            variant_rows.append(recipe)
            feature_rows.append({**recipe, **measured})
        same = measured_by_variant["same_source"]
        switched = measured_by_variant["switched_source_register_matched"]
        gain = measured_by_variant["same_source_gain_half"]
        polarity = measured_by_variant["same_source_polarity_flip"]
        paired_results.append({
            "pair_id": pair_id, "corpus": anchor["corpus"],
            "instrument_group": anchor["instrument_group"],
            "switch_minus_same_boundary_distance": float(switched["T_boundary_mfcc2_12_distance"]) - float(same["T_boundary_mfcc2_12_distance"]),
            "switch_minus_same_trajectory_p90": float(switched["T_mfcc2_12_step_p90"]) - float(same["T_mfcc2_12_step_p90"]),
            "switch_minus_same_trajectory_max": float(switched["T_mfcc2_12_step_max"]) - float(same["T_mfcc2_12_step_max"]),
            "gain_abs_delta_boundary": abs(float(gain["T_boundary_mfcc2_12_distance"]) - float(same["T_boundary_mfcc2_12_distance"])),
            "polarity_abs_delta_boundary": abs(float(polarity["T_boundary_mfcc2_12_distance"]) - float(same["T_boundary_mfcc2_12_distance"])),
            "gain_max_abs_delta_family": max(abs(float(gain[name]) - float(same[name])) for name in T_MEASURE_NAMES),
            "polarity_max_abs_delta_family": max(abs(float(polarity[name]) - float(same[name])) for name in T_MEASURE_NAMES),
        })

    write_csv(args.output / "source_inventory.csv", inventory)
    write_csv(args.output / "chunk_manifest.csv", chunk_audit)
    write_csv(args.output / "pair_manifest.csv", pair_rows)
    write_csv(args.output / "pair_exclusions.csv", pair_exclusions)
    write_csv(args.output / "variant_manifest.csv", variant_rows)
    write_csv(args.output / "feature_measurements.csv", feature_rows)
    write_csv(args.output / "paired_results.csv", paired_results)

    summary: dict[str, Any] = {
        "schema": "external-timbre-validation-v1",
        "source_records_audited": len(records),
        "eligible_source_records": len(chunks),
        "pair_count": len(pairs),
        "pair_counts_by_corpus": dict(Counter(row["corpus"] for row in pair_rows)),
        "pair_counts_by_instrument_group": dict(Counter(row["instrument_group"] for row in pair_rows)),
        "variants": len(variant_rows), "duration_sec_each": 6.0,
        "no_ai_human_labels_used": True, "fitted_classifier": False,
        "verified_performer_identity": False,
        "natural_identity_continuity_validated": False,
        "source_channel_signal_continuity_sensitivity_tested": True,
        "pitch_register_control": "nearest approximate autocorrelation-F0; residual recorded; <=300 cents on counterfactual right chunk",
        "source_inventory_sha256": sha256_file(args.output / "source_inventory.csv"),
        "pair_manifest_sha256": sha256_file(args.output / "pair_manifest.csv"),
        "variant_manifest_sha256": sha256_file(args.output / "variant_manifest.csv"),
        "extractor_sha256": sha256_file(Path(__file__).with_name("timbre_features.py")),
        "script_sha256": sha256_file(Path(__file__)),
        "runtime": {"numpy": np.__version__, "scipy": __import__("scipy").__version__,
                    "soundfile": sf.__version__},
    }
    for key in ("switch_minus_same_boundary_distance", "switch_minus_same_trajectory_p90",
                "switch_minus_same_trajectory_max"):
        values = [float(row[key]) for row in paired_results]
        summary[key] = {
            "valid_pairs": len(values), "positive_pairs": int(np.sum(np.asarray(values) > 0)),
            "positive_fraction": float(np.mean(np.asarray(values) > 0)),
            "median_delta": float(np.median(values)), "median_bootstrap_95ci": bootstrap_median_ci(values),
        }
    for key in ("gain_abs_delta_boundary", "polarity_abs_delta_boundary",
                "gain_max_abs_delta_family", "polarity_max_abs_delta_family"):
        values = [float(row[key]) for row in paired_results]
        summary[key] = {"median": float(np.median(values)), "maximum": float(np.max(values))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = f"""# T controlled external-real-source validation

This is a source-channel signal-continuity sensitivity test, not validation of
natural performer identity. MUSDB's label is a track-level vocal stem and may
contain multiple singers, doubles, or processing. MedleyDB identifies stem
channels and instrument labels, not performer identities. Each control uses two
non-overlapping 3-second chunks; no samples are shared between the anchor
halves. Switched right halves are from a different source channel in the same
corpus and coarse instrument group and are nearest-matched to the original
right half by approximate median autocorrelation F0 (at most 300 cents). Exact
residual cents and pitch coverage are in `pair_manifest.csv`.

All right halves except the explicit half-gain control are RMS-matched to the
left half. Polarity and gain controls test expected invariance. Hard concatenation
is identical across conditions; the primary boundary distance summarizes the
one-second context on each side and therefore excludes the cut-crossing STFT
frame. AI/human labels, model fitting, and identity classification are absent.

MoisesDB and URMP selected mirrors were audited as mixture-only (`.wav` full
tracks and `AuMix`, respectively), so they are excluded rather than presented
as isolated truth. All Demucs estimates are excluded. The descriptor overlaps
the existing S family because both use magnitude spectra; T specifically tracks
time-local MFCC2-12 envelope discontinuities on an isolated source, whereas S
summarizes spectral/high-frequency behavior and cannot establish source identity.

What this can validate: controlled sensitivity of the frozen signal descriptor
to a source-channel switch under approximate register and level controls.
What it cannot validate: singer/instrument-instance recognition, pitch-invariant
timbre, natural continuity semantics, causal AI-generation differences, or any
AI/human attribution claim.

```json
{json.dumps(summary, indent=2)}
```
"""
    (args.output / "REPORT_EN.md").write_text(report)
    print(json.dumps(summary), flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts_root = root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb-manifest", type=Path,
                        default=artifacts_root / "musdb18_oracle_vocal_eval_20260901" / "manifest.jsonl")
    parser.add_argument("--medley-root", type=Path,
                        default=artifacts_root / "demucs_bias_corrected_1000_20260901" / "oracle_medleydb" / "MedleyDB_sample")
    parser.add_argument("--output", type=Path, default=root / "results" / "external_timbre_validation_v1")
    parser.add_argument("--max-musdb", type=int, default=48)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
