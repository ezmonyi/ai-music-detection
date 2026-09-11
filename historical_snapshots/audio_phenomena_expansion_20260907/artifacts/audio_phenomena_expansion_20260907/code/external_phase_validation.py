#!/usr/bin/env python3
"""External real-audio intervention validation for frozen phase features.

This script never creates AI/human labels.  It checks whether the F phase
family responds to a phase-only intervention, remains numerically stable under
polarity/gain controls, quantifies crop-alignment sensitivity, and measures the
effect of replacing MUSDB18 oracle vocals with existing Demucs estimates.

Outputs are auditable CSV/JSONL/Markdown files under the assigned result
directory.  No audio, models, jobs, or network resources are created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags, resample_poly

from phase_features import CONFIG as PHASE_CONFIG
from phase_features import FEATURE_NAMES, VERSION as PHASE_VERSION
from phase_features import extract_phase_features


SCRIPT_VERSION = "external_phase_validation_v1_20260907"
TARGET_SR = 16_000
GLOBAL_SEED = 20_260_907
GAIN = 0.5
SHIFT_SUBHOP = 17
SHIFT_HOP = 256
CODEC_TRACKS = 24
CODEC_BITRATE = "128k"
INTERVENTIONS = (
    "phase_randomized_equal_global_magnitude",
    "polarity_inversion",
    "gain_x0p5",
    "circular_shift_17_samples",
    "circular_shift_256_samples",
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "artifacts/musdb18_oracle_vocal_eval_20260901/manifest.jsonl"
DEFAULT_RESULTS = ROOT / "artifacts/audio_phenomena_expansion_20260907/results/external_phase_validation"
DEMUCS_ROOT = ROOT / "artifacts/musdb18_oracle_vocal_eval_20260901/demucs_outputs/htdemucs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def waveform_sha256(audio: np.ndarray) -> str:
    canonical = np.ascontiguousarray(audio, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def stable_seed(*parts: str) -> int:
    message = ":".join((str(GLOBAL_SEED), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little")


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    required = {"track_id", "track_name", "split", "mixture", "reference_vocals"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"Manifest row {index} missing {sorted(missing)}")
    track_ids = [str(row["track_id"]) for row in rows]
    if len(set(track_ids)) != len(track_ids):
        raise RuntimeError("Manifest contains duplicate track_id values")
    # Hash ordering is deterministic and independent of the source manifest order.
    rows.sort(key=lambda row: hashlib.sha256(str(row["track_id"]).encode()).hexdigest())
    if limit:
        if limit < 24:
            raise ValueError("--limit must be 0 (all) or at least 24")
        rows = rows[:limit]
    return rows


def load_audio(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    source_sr, raw = wavfile.read(path)
    source_sr = int(source_sr)
    if np.issubdtype(raw.dtype, np.integer):
        if raw.dtype == np.uint8:
            audio = (raw.astype(np.float64) - 128.0) / 128.0
        else:
            scale = float(1 << (8 * raw.dtype.itemsize - 1))
            audio = raw.astype(np.float64) / scale
    elif np.issubdtype(raw.dtype, np.floating):
        audio = raw.astype(np.float64)
    else:
        raise TypeError(f"Unsupported WAV dtype {raw.dtype}: {path}")
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise RuntimeError(f"Unexpected WAV shape {audio.shape}: {path}")
    frames, channels = map(int, audio.shape)
    if audio.shape != (frames, channels) or not np.all(np.isfinite(audio)):
        raise RuntimeError(f"Incomplete or nonfinite decode: {path}")
    mono = audio.mean(axis=1, dtype=np.float64)
    mono -= float(mono.mean())
    if source_sr != TARGET_SR:
        divisor = math.gcd(source_sr, TARGET_SR)
        mono = resample_poly(
            mono,
            TARGET_SR // divisor,
            source_sr // divisor,
            window=("kaiser", 5.0),
            padtype="constant",
        )
    mono = np.ascontiguousarray(mono, dtype=np.float64)
    return mono, {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "source_sample_rate": source_sr,
        "source_channels": channels,
        "source_frames": frames,
        "analysis_sample_rate": TARGET_SR,
        "analysis_frames": int(mono.size),
        "analysis_duration_s": float(mono.size / TARGET_SR),
        "waveform_sha256": waveform_sha256(mono),
        "rms": float(np.sqrt(np.mean(np.square(mono)))),
    }


def phase_randomize_equal_magnitude(audio: np.ndarray, seed: int) -> np.ndarray:
    spectrum = np.fft.rfft(audio)
    magnitude = np.abs(spectrum)
    randomized = np.empty_like(spectrum)
    randomized[0] = spectrum[0]  # Preserve DC exactly.
    rng = np.random.default_rng(seed)
    stop = spectrum.size - 1 if audio.size % 2 == 0 else spectrum.size
    randomized[1:stop] = magnitude[1:stop] * np.exp(
        1j * rng.uniform(-np.pi, np.pi, stop - 1)
    )
    if audio.size % 2 == 0:
        randomized[-1] = spectrum[-1]  # Nyquist must remain real for irfft.
    result = np.fft.irfft(randomized, n=audio.size)
    return np.ascontiguousarray(result, dtype=np.float64)


def intervention(audio: np.ndarray, name: str, seed: int) -> tuple[np.ndarray, float]:
    if name == "phase_randomized_equal_global_magnitude":
        return phase_randomize_equal_magnitude(audio, seed), 1.0
    if name == "polarity_inversion":
        return -audio, 1.0
    if name == "gain_x0p5":
        return GAIN * audio, GAIN
    if name == "circular_shift_17_samples":
        return np.roll(audio, SHIFT_SUBHOP), 1.0
    if name == "circular_shift_256_samples":
        return np.roll(audio, SHIFT_HOP), 1.0
    raise KeyError(name)


def magnitude_errors(original: np.ndarray, changed: np.ndarray, expected_scale: float) -> tuple[float, float]:
    target = abs(expected_scale) * np.abs(np.fft.rfft(original))
    observed = np.abs(np.fft.rfft(changed))
    difference = observed - target
    relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(target), np.finfo(float).tiny))
    relative_max = float(np.max(np.abs(difference)) / max(float(np.max(target)), np.finfo(float).tiny))
    return relative_l2, relative_max


def mp3_roundtrip_aligned(audio: np.ndarray, ffmpeg: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Encode/decode MP3 in memory, then remove measured integer encoder delay."""
    raw = np.ascontiguousarray(audio, dtype="<f8").tobytes()
    encode = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "f64le", "-ar", str(TARGET_SR),
         "-ac", "1", "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", CODEC_BITRATE,
         "-id3v2_version", "0", "-f", "mp3", "pipe:1"],
        input=raw, capture_output=True, check=True,
    )
    decode = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "mp3", "-i", "pipe:0",
         "-f", "f64le", "-ar", str(TARGET_SR), "-ac", "1", "pipe:1"],
        input=encode.stdout, capture_output=True, check=True,
    )
    decoded = np.frombuffer(decode.stdout, dtype="<f8").copy()
    if decoded.size < 4_096 or not np.all(np.isfinite(decoded)):
        raise RuntimeError("MP3 round-trip returned short or nonfinite audio")
    correlations = correlate(decoded, audio, mode="full", method="fft")
    lags = correlation_lags(decoded.size, audio.size, mode="full")
    search = np.abs(lags) <= 4_096
    lag = int(lags[search][np.argmax(correlations[search])])
    if lag >= 0:
        changed = decoded[lag:]
        base = audio
    else:
        changed = decoded
        base = audio[-lag:]
    common = min(base.size, changed.size)
    if common < audio.size - 4_096:
        raise RuntimeError(f"MP3 aligned overlap unexpectedly short: {common}/{audio.size}")
    base = np.ascontiguousarray(base[:common], dtype=np.float64)
    changed = np.ascontiguousarray(changed[:common], dtype=np.float64)
    audit = {
        "codec_name": "libmp3lame",
        "codec_bitrate": CODEC_BITRATE,
        "codec_container_sha256": hashlib.sha256(encode.stdout).hexdigest(),
        "codec_container_bytes": len(encode.stdout),
        "codec_decoded_frames_before_alignment": int(decoded.size),
        "codec_alignment_lag_samples": lag,
        "codec_alignment_lag_ms": float(1_000.0 * lag / TARGET_SR),
        "codec_aligned_frames": int(common),
    }
    return base, changed, audit


def finite_float(value: Any) -> float:
    result = float(value)
    return result if np.isfinite(result) else float("nan")


def make_pair_row(
    manifest_row: dict[str, Any],
    source_kind: str,
    comparison_kind: str,
    comparison_name: str,
    base_audio: np.ndarray,
    changed_audio: np.ndarray,
    base_audit: dict[str, Any],
    changed_audit: dict[str, Any] | None,
    expected_magnitude_scale: float | None,
    transform_seed: int | None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if base_audio.shape != changed_audio.shape:
        common = min(base_audio.size, changed_audio.size)
        base_audio = base_audio[:common]
        changed_audio = changed_audio[:common]
    base_features = extract_phase_features(base_audio, TARGET_SR)
    changed_features = extract_phase_features(changed_audio, TARGET_SR)
    pair_payload = f"{manifest_row['track_id']}:{source_kind}:{comparison_name}:{SCRIPT_VERSION}"
    row: dict[str, Any] = {
        "pair_id": hashlib.sha256(pair_payload.encode()).hexdigest()[:20],
        "track_id": manifest_row["track_id"],
        "track_name": manifest_row["track_name"],
        "split": manifest_row["split"],
        "source_kind": source_kind,
        "comparison_kind": comparison_kind,
        "comparison_name": comparison_name,
        "base_path": base_audit["path"],
        "base_file_sha256": base_audit["file_sha256"],
        "base_waveform_sha256": waveform_sha256(base_audio),
        "changed_path": changed_audit["path"] if changed_audit else "",
        "changed_file_sha256": changed_audit["file_sha256"] if changed_audit else "",
        "changed_waveform_sha256": waveform_sha256(changed_audio),
        "analysis_sample_rate": TARGET_SR,
        "analysis_frames": int(base_audio.size),
        "transform_seed": transform_seed if transform_seed is not None else "",
        "expected_fft_magnitude_scale": expected_magnitude_scale if expected_magnitude_scale is not None else "",
        "fft_magnitude_relative_l2_error": "",
        "fft_magnitude_relative_max_error": "",
        "base_status": base_features["F_status"],
        "changed_status": changed_features["F_status"],
        "base_quality_eligible": base_features["F_quality_eligible"],
        "changed_quality_eligible": changed_features["F_quality_eligible"],
        "codec_name": "",
        "codec_bitrate": "",
        "codec_container_sha256": "",
        "codec_container_bytes": "",
        "codec_decoded_frames_before_alignment": "",
        "codec_alignment_lag_samples": "",
        "codec_alignment_lag_ms": "",
        "codec_aligned_frames": "",
    }
    if extra_metadata:
        row.update(extra_metadata)
    if expected_magnitude_scale is not None:
        l2_error, max_error = magnitude_errors(base_audio, changed_audio, expected_magnitude_scale)
        row["fft_magnitude_relative_l2_error"] = l2_error
        row["fft_magnitude_relative_max_error"] = max_error
    for feature in FEATURE_NAMES:
        base_value = finite_float(base_features[feature])
        changed_value = finite_float(changed_features[feature])
        row[f"base__{feature}"] = base_value
        row[f"changed__{feature}"] = changed_value
        row[f"delta__{feature}"] = changed_value - base_value if np.isfinite(base_value + changed_value) else float("nan")
    return row


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def summarize_pairs(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        for feature in FEATURE_NAMES:
            grouped[(str(row["source_kind"]), str(row["comparison_name"]), feature)].append(row)
    summaries: list[dict[str, Any]] = []
    for (source_kind, comparison_name, feature), rows in sorted(grouped.items()):
        base = np.asarray([finite_float(row[f"base__{feature}"]) for row in rows])
        changed = np.asarray([finite_float(row[f"changed__{feature}"]) for row in rows])
        keep = np.isfinite(base) & np.isfinite(changed)
        base, changed = base[keep], changed[keep]
        delta = changed - base
        nonzero = delta[np.abs(delta) > 1e-15]
        iqr = percentile(base, 75.0) - percentile(base, 25.0) if base.size else float("nan")
        normalized = float(np.median(np.abs(delta)) / max(abs(iqr), 1e-12)) if delta.size else float("nan")
        summaries.append({
            "source_kind": source_kind,
            "comparison_name": comparison_name,
            "feature": feature,
            "n_pairs_total": len(rows),
            "n_pairs_finite": int(delta.size),
            "base_median": float(np.median(base)) if base.size else float("nan"),
            "changed_median": float(np.median(changed)) if changed.size else float("nan"),
            "delta_median": float(np.median(delta)) if delta.size else float("nan"),
            "delta_q05": percentile(delta, 5.0),
            "delta_q95": percentile(delta, 95.0),
            "absolute_delta_median": float(np.median(np.abs(delta))) if delta.size else float("nan"),
            "absolute_delta_q95": percentile(np.abs(delta), 95.0),
            "base_iqr": iqr,
            "median_absolute_delta_over_base_iqr": normalized,
            "paired_sign_balance": float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / nonzero.size) if nonzero.size else 0.0,
        })
    return summaries


def summarize_quality(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["source_kind"]), str(row["comparison_name"]))].append(row)
    result = []
    for (source_kind, comparison_name), rows in sorted(grouped.items()):
        base_status = Counter(str(row["base_status"]) for row in rows)
        changed_status = Counter(str(row["changed_status"]) for row in rows)
        errors = np.asarray([
            finite_float(row["fft_magnitude_relative_l2_error"])
            for row in rows if row["fft_magnitude_relative_l2_error"] != ""
        ])
        result.append({
            "source_kind": source_kind,
            "comparison_name": comparison_name,
            "n_pairs": len(rows),
            "base_status_counts_json": json.dumps(base_status, sort_keys=True),
            "changed_status_counts_json": json.dumps(changed_status, sort_keys=True),
            "both_quality_eligible": sum(
                int(row["base_quality_eligible"]) == 1 and int(row["changed_quality_eligible"]) == 1
                for row in rows
            ),
            "fft_magnitude_relative_l2_error_max": float(np.max(errors)) if errors.size else float("nan"),
        })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def render_report(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]], quality: list[dict[str, Any]], metadata: dict[str, Any]
) -> str:
    by_comparison: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_comparison[(str(row["source_kind"]), str(row["comparison_name"]))].append(row)

    def top_lines(source: str, comparison: str, count: int = 4) -> list[str]:
        candidates = [row for row in by_comparison[(source, comparison)] if int(row["n_pairs_finite"]) >= 12]
        candidates.sort(key=lambda row: finite_float(row["median_absolute_delta_over_base_iqr"]), reverse=True)
        return [
            f"  - `{row['feature']}`: median Δ={float(row['delta_median']):.6g}, "
            f"median |Δ|={float(row['absolute_delta_median']):.6g}, "
            f"median |Δ|/base-IQR={float(row['median_absolute_delta_over_base_iqr']):.3g} "
            f"(finite {row['n_pairs_finite']}/{row['n_pairs_total']})."
            for row in candidates[:count]
        ]

    null_rows = [
        row for row in summaries
        if row["comparison_name"] in {"polarity_inversion", "gain_x0p5"} and int(row["n_pairs_finite"]) > 0
    ]
    null_max_median = max((float(row["absolute_delta_median"]) for row in null_rows), default=float("nan"))
    phase_quality = [row for row in quality if row["comparison_name"] == "phase_randomized_equal_global_magnitude"]
    phase_mag_error = max((finite_float(row["fft_magnitude_relative_l2_error_max"]) for row in phase_quality), default=float("nan"))
    null_pass = bool(np.isfinite(null_max_median) and null_max_median <= 1e-10)
    magnitude_pass = bool(np.isfinite(phase_mag_error) and phase_mag_error <= 1e-12)

    lines = [
        "# External real-audio validation of the frozen F phase family",
        "",
        f"Run timestamp: {metadata['created_at_utc']}. Extractor: `{PHASE_VERSION}`. Validator: `{SCRIPT_VERSION}`.",
        "",
        "## Scope",
        "",
        f"This is a deterministic measurement validation on {metadata['track_count']} unique MUSDB18 preview tracks, "
        f"not an AI-versus-human experiment. It produced {len(rows)} matched pairs. Each original mixture and oracle "
        "vocal was compared with a global-FFT phase-randomized signal, polarity inversion, 0.5 gain, a 17-sample "
        "circular shift, and a 256-sample circular shift. Oracle vocals were also paired with the already-generated "
        "Demucs `htdemucs` vocal estimate.",
        "",
        "All transformations operate on mono audio resampled to 16 kHz. The phase-randomized signal uses a real "
        "inverse FFT and preserves every global Fourier magnitude up to floating-point error. Circular shifts preserve "
        "global magnitudes but introduce a wrap seam; they test STFT-grid alignment, not perceptual equivalence.",
        "",
        f"A separate MP3 128 kbps encode/decode sensitivity check was "
        f"{metadata['codec_validation_status']} on {metadata['codec_track_count']} tracks "
        f"({metadata['codec_pair_count']} mix/vocal pairs), with measured integer alignment recorded per pair.",
        "",
        "## Prespecified control checks",
        "",
        f"- Equal-global-magnitude phase randomization: {'PASS' if magnitude_pass else 'FAIL'}; worst relative L2 "
        f"magnitude error={phase_mag_error:.3e}, threshold=1e-12.",
        f"- Polarity/gain numerical invariance: {'PASS' if null_pass else 'FAIL'}; largest across-metric median |Δ|="
        f"{null_max_median:.3e}, threshold=1e-10.",
        "- Time-shift controls are descriptive rather than pass/fail because finite-window STFT features are expected "
        "to depend on sub-hop alignment and circular boundary placement.",
        "- Separator replacement is descriptive: it changes both source content and phase and is not a controlled "
        "phase-only intervention.",
        "",
        "## Largest robust paired changes",
        "",
    ]
    for source, comparison in (
        ("mixture", "phase_randomized_equal_global_magnitude"),
        ("reference_vocals", "phase_randomized_equal_global_magnitude"),
        ("mixture", "circular_shift_17_samples"),
        ("reference_vocals", "circular_shift_17_samples"),
        ("mixture", "mp3_128kbps_aligned"),
        ("reference_vocals", "mp3_128kbps_aligned"),
        ("reference_vocals", "demucs_separator_replacement"),
    ):
        lines.append(f"### {source}: {comparison}")
        lines.append("")
        lines.extend(top_lines(source, comparison) or ["  - No feature had at least 12 finite pairs."])
        lines.append("")
    lines.extend([
        "## Interpretation and limitations",
        "",
        "- Phase-randomized counterparts are deliberately unnatural matched interventions. Detecting them establishes "
        "phase sensitivity, not validity for detecting generated music.",
        "- Equal global FFT magnitudes do not imply equal local STFT magnitudes, envelopes, transients, or perception. "
        "Any changed descriptor may respond jointly to local time-frequency redistribution.",
        "- Polarity and gain are narrow algebraic controls. Passing them does not establish invariance to codecs, EQ, "
        "reverberation, resampling kernels, channel folding, or arbitrary crop offsets.",
        "- A circular shift preserves the global magnitude spectrum but creates a finite-record boundary seam. The "
        "17-sample and one-hop results must be treated as alignment sensitivity, not causal phase evidence.",
        "- MP3 sensitivity uses a 24-track deterministic subset and an integer cross-correlation alignment. Lossy "
        "coding may also introduce frequency-dependent timing, padding, and local magnitude changes; this is a "
        "pipeline sensitivity check, not a phase-only intervention.",
        "- MUSDB18 preview excerpts are short and not representative of every genre, production chain, or full song. "
        "Region-specific metrics can be unavailable when attack/decay coverage is insufficient; missing values were "
        "retained rather than zero-filled.",
        "- Demucs estimates are not ground truth interventions. Their paired differences quantify separator sensitivity "
        "and warn against interpreting separated-stem phase statistics as if they were native recordings.",
        "- No thresholds or feature directions were tuned against AI/human labels. These results cannot support an "
        "authorship claim, accuracy, AUC, or causal explanation of any generator.",
        "",
        "## Audit artifacts",
        "",
        "- `pair_metrics.csv`: pair IDs, track IDs, file/waveform hashes, transform seeds, magnitude errors, statuses, "
        "and base/changed/delta values for every frozen F feature.",
        "- `effect_summary.csv`: robust paired effect summaries without classification labels.",
        "- `quality_summary.csv`: eligibility/status counts and magnitude-preservation evidence.",
        "- `selected_inputs.jsonl`: selected source paths and hashes.",
        "- `run_metadata.json`: exact runtime, configuration, versions, and artifact contracts.",
        "- `artifact_sha256.txt`: hashes of the validator, frozen extractor, manifest, and result evidence.",
        "",
    ])
    return "\n".join(lines)


def run(manifest: Path, output: Path, limit: int, codec_tracks: int) -> None:
    rows = load_rows(manifest, limit)
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    codec_track_count = min(codec_tracks, len(rows)) if ffmpeg and codec_tracks > 0 else 0
    pair_rows: list[dict[str, Any]] = []
    selected_inputs: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(rows, start=1):
        track_id = str(manifest_row["track_id"])
        paths = {
            "mixture": Path(manifest_row["mixture"]),
            "reference_vocals": Path(manifest_row["reference_vocals"]),
            "demucs_vocals": DEMUCS_ROOT / track_id / "vocals.wav",
        }
        loaded = {kind: load_audio(path) for kind, path in paths.items()}
        lengths = {kind: audio.size for kind, (audio, _) in loaded.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Analysis-length mismatch for {track_id}: {lengths}")
        selected_inputs.append({
            "track_id": track_id,
            "track_name": manifest_row["track_name"],
            "split": manifest_row["split"],
            "mixture": loaded["mixture"][1],
            "reference_vocals": loaded["reference_vocals"][1],
            "demucs_vocals": loaded["demucs_vocals"][1],
        })
        for source_kind in ("mixture", "reference_vocals"):
            base_audio, base_audit = loaded[source_kind]
            for name in INTERVENTIONS:
                seed = stable_seed(track_id, source_kind, name)
                changed, scale = intervention(base_audio, name, seed)
                pair_rows.append(make_pair_row(
                    manifest_row, source_kind, "controlled_intervention", name,
                    base_audio, changed, base_audit, None, scale, seed if name.startswith("phase_") else None,
                ))
            if index <= codec_track_count:
                aligned_base, codec_audio, codec_audit = mp3_roundtrip_aligned(base_audio, ffmpeg)
                pair_rows.append(make_pair_row(
                    manifest_row, source_kind, "codec_sensitivity", "mp3_128kbps_aligned",
                    aligned_base, codec_audio, base_audit, None, None, None, codec_audit,
                ))
        reference_audio, reference_audit = loaded["reference_vocals"]
        demucs_audio, demucs_audit = loaded["demucs_vocals"]
        pair_rows.append(make_pair_row(
            manifest_row, "reference_vocals", "separator_sensitivity", "demucs_separator_replacement",
            reference_audio, demucs_audio, reference_audit, demucs_audit, None, None,
        ))
        if index % 12 == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)} tracks", flush=True)

    summaries = summarize_pairs(pair_rows)
    quality = summarize_quality(pair_rows)
    metadata = {
        "validator_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "track_count": len(rows),
        "pair_count": len(pair_rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "selection_order": "ascending_sha256(track_id)",
        "selection_limit": limit,
        "target_sample_rate": TARGET_SR,
        "mono": "arithmetic channel mean then DC removal",
        "resampling": "scipy.signal.resample_poly Kaiser beta=5.0 constant padding",
        "global_seed": GLOBAL_SEED,
        "interventions": list(INTERVENTIONS),
        "gain_scale": GAIN,
        "integer_shifts_samples": [SHIFT_SUBHOP, SHIFT_HOP],
        "codec_validation_status": "completed" if codec_track_count else "pending_ffmpeg_unavailable_or_disabled",
        "codec_track_count": codec_track_count,
        "codec_pair_count": 2 * codec_track_count,
        "codec": "libmp3lame 128 kbps, decoded to mono 16 kHz float64, integer cross-correlation alignment",
        "ffmpeg_executable": ffmpeg,
        "phase_extractor_version": PHASE_VERSION,
        "phase_feature_names": list(FEATURE_NAMES),
        "phase_config": PHASE_CONFIG,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "wav_decoder": "scipy.io.wavfile.read",
        "no_ai_human_labels": True,
        "no_gpu_inference_or_download": True,
    }
    write_csv(output / "pair_metrics.csv", pair_rows)
    write_csv(output / "effect_summary.csv", summaries)
    write_csv(output / "quality_summary.csv", quality)
    with (output / "selected_inputs.jsonl").open("w") as handle:
        for row in selected_inputs:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
    (output / "run_metadata.json").write_text(json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(render_report(pair_rows, summaries, quality, metadata))

    hash_targets = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("phase_features.py"),
        manifest.resolve(),
        output / "pair_metrics.csv",
        output / "effect_summary.csv",
        output / "quality_summary.csv",
        output / "selected_inputs.jsonl",
        output / "run_metadata.json",
        output / "REPORT.md",
    ]
    checksum_lines = [f"{sha256_file(path)}  {path}" for path in hash_targets]
    (output / "artifact_sha256.txt").write_text("\n".join(checksum_lines) + "\n")
    print(f"wrote {len(pair_rows)} pairs to {output}", flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int, default=0, help="0 uses all rows; otherwise must be >=24")
    parser.add_argument("--codec-tracks", type=int, default=CODEC_TRACKS,
                        help="deterministic leading track subset for MP3 sensitivity; 0 disables")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.manifest.resolve(), arguments.output.resolve(), arguments.limit, arguments.codec_tracks)
