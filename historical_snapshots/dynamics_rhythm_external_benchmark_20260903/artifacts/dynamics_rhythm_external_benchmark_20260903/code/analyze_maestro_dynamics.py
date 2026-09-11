#!/usr/bin/env python3
"""Benchmark acoustic onset/dynamics estimators against MAESTRO MIDI velocity."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
from scipy.signal import resample_poly, stft
from scipy.stats import spearmanr


SEED = 20260903
N_BOOT = 10_000
TARGET_SR = 44_100
N_FFT = 4096
HOP = 512
EPS = 1e-12
ESTIMATORS = (
    "broadband_onset_delta_db",
    "harmonic_onset_delta_db",
    "harmonic_attack_level_db",
)
WINDOW_FEATURES = ("dynamics_span", "dynamics_iqr", "dynamics_adjacent_change")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def read_segment(path: Path, start_seconds: float, duration_seconds: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(path) as handle:
        start_frame = int(round(start_seconds * handle.samplerate))
        frames = int(round(duration_seconds * handle.samplerate))
        handle.seek(start_frame)
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)
        sr = int(handle.samplerate)
    mono = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    if sr != TARGET_SR:
        divisor = math.gcd(sr, TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // divisor, sr // divisor).astype(np.float32)
        sr = TARGET_SR
    expected = int(round(duration_seconds * TARGET_SR))
    if len(mono) < expected:
        mono = np.pad(mono, (0, expected - len(mono)))
    elif len(mono) > expected:
        mono = mono[:expected]
    return mono, sr


def rms_db(audio: np.ndarray, sr: int, begin: float, end: float) -> float:
    left = max(0, int(round(begin * sr)))
    right = min(len(audio), int(round(end * sr)))
    if right <= left:
        return float("nan")
    power = np.mean(np.square(audio[left:right], dtype=np.float64))
    return float(10.0 * np.log10(power + EPS))


def harmonic_bins(frequencies: np.ndarray, midi_pitch: int) -> tuple[np.ndarray, np.ndarray]:
    f0 = 440.0 * 2.0 ** ((midi_pitch - 69) / 12.0)
    bin_width = frequencies[1] - frequencies[0]
    indices: list[int] = []
    weights: list[float] = []
    half_semitone_sixth = 2.0 ** (1.0 / 72.0)
    for harmonic in range(1, 9):
        center = f0 * harmonic
        if center >= frequencies[-1]:
            break
        half_width = max(center * (half_semitone_sixth - 1.0), 0.75 * bin_width)
        chosen = np.flatnonzero(np.abs(frequencies - center) <= half_width)
        if not len(chosen):
            chosen = np.array([int(np.argmin(np.abs(frequencies - center)))])
        indices.extend(chosen.tolist())
        weights.extend([1.0 / harmonic / len(chosen)] * len(chosen))
    return np.asarray(indices, dtype=int), np.asarray(weights, dtype=float)


def interval_power_db(
    power: np.ndarray,
    frame_times: np.ndarray,
    frequency_indices: np.ndarray,
    weights: np.ndarray,
    begin: float,
    end: float,
) -> float:
    frame_keep = (frame_times >= begin) & (frame_times < end)
    if not frame_keep.any() or not len(frequency_indices):
        return float("nan")
    selected = power[np.ix_(frequency_indices, np.flatnonzero(frame_keep))]
    weighted_frame_power = np.sum(selected * weights[:, None], axis=0)
    return float(10.0 * np.log10(np.median(weighted_frame_power) + EPS))


def extract_window_events(
    audio: np.ndarray,
    sr: int,
    notes: list[pretty_midi.Note],
    start_seconds: float,
    duration_seconds: float,
) -> list[dict[str, float | int]]:
    frequencies, frame_times, spectrum = stft(
        audio,
        fs=sr,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    power = np.square(np.abs(spectrum), dtype=np.float64)
    bin_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, float | int]] = []
    for note in notes:
        onset = float(note.start - start_seconds)
        if onset < 0.15 or onset > duration_seconds - 0.15 or note.velocity <= 0:
            continue
        pre_broadband = rms_db(audio, sr, onset - 0.10, onset - 0.02)
        post_broadband = rms_db(audio, sr, onset + 0.02, onset + 0.12)
        if note.pitch not in bin_cache:
            bin_cache[note.pitch] = harmonic_bins(frequencies, note.pitch)
        freq_indices, weights = bin_cache[note.pitch]
        pre_harmonic = interval_power_db(
            power, frame_times, freq_indices, weights, onset - 0.10, onset - 0.02
        )
        post_harmonic = interval_power_db(
            power, frame_times, freq_indices, weights, onset + 0.02, onset + 0.12
        )
        rows.append(
            {
                "onset_seconds": note.start,
                "local_onset_seconds": onset,
                "pitch": int(note.pitch),
                "velocity": int(note.velocity),
                "broadband_onset_delta_db": post_broadband - pre_broadband,
                "harmonic_onset_delta_db": post_harmonic - pre_harmonic,
                "harmonic_attack_level_db": post_harmonic,
            }
        )
    return rows


def cluster_bootstrap_values(
    table: pd.DataFrame,
    value_column: str,
    aggregate: str,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    per_recording = {
        key: group[value_column].dropna().to_numpy(float)
        for key, group in table.groupby("recording_id")
    }
    per_recording = {key: values for key, values in per_recording.items() if len(values)}
    recording_ids = np.array(sorted(per_recording))
    all_values = np.concatenate([per_recording[key] for key in recording_ids])
    reducer = np.median if aggregate == "median" else np.mean
    draws = np.empty(N_BOOT, dtype=float)
    for index in range(N_BOOT):
        sampled = rng.choice(recording_ids, len(recording_ids), replace=True)
        values = np.concatenate([per_recording[key] for key in sampled])
        draws[index] = reducer(values)
    estimate = reducer(all_values)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "sign_stability": float(np.mean(draws > 0) if estimate >= 0 else np.mean(draws < 0)),
        "n_recordings": int(len(recording_ids)),
        "n_groups": int(len(all_values)),
    }


def clustered_correlation(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    clean = table[["recording_id", x_column, y_column]].dropna()
    rho, pvalue = safe_spearman(clean[x_column].to_numpy(), clean[y_column].to_numpy())
    groups = {key: group for key, group in clean.groupby("recording_id")}
    recording_ids = np.array(sorted(groups))
    draws = []
    for _ in range(N_BOOT):
        sampled = rng.choice(recording_ids, len(recording_ids), replace=True)
        x = np.concatenate([groups[key][x_column].to_numpy(float) for key in sampled])
        y = np.concatenate([groups[key][y_column].to_numpy(float) for key in sampled])
        value, _ = safe_spearman(x, y)
        if np.isfinite(value):
            draws.append(value)
    draws = np.asarray(draws, dtype=float)
    return {
        "estimate": rho,
        "pvalue": pvalue,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "sign_stability": float(np.mean(draws > 0) if rho >= 0 else np.mean(draws < 0)),
        "n_recordings": int(len(recording_ids)),
        "n_windows": int(len(clean)),
        "n_boot_valid": int(len(draws)),
    }


def pairwise_quartile_accuracy(group: pd.DataFrame, estimator: str) -> float:
    q25, q75 = group["velocity"].quantile([0.25, 0.75])
    low = group[group["velocity"] <= q25][estimator].dropna().to_numpy(float)
    high = group[group["velocity"] >= q75][estimator].dropna().to_numpy(float)
    if not len(low) or not len(high):
        return float("nan")
    comparison = high[:, None] - low[None, :]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def center_by_recording_pitch(events: pd.DataFrame, column: str) -> pd.Series:
    medians = events.groupby(["recording_id", "pitch"])[column].transform("median")
    return events[column] - medians


def summarize_window(group: pd.DataFrame, column: str) -> dict[str, float]:
    values = group[column].dropna().to_numpy(float)
    if len(values) < 8:
        return {name: float("nan") for name in WINDOW_FEATURES}
    ordered = group.dropna(subset=[column]).sort_values("onset_seconds")[column].to_numpy(float)
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    return {
        "dynamics_span": float(q90 - q10),
        "dynamics_iqr": float(q75 - q25),
        "dynamics_adjacent_change": float(np.median(np.abs(np.diff(ordered)))),
    }


def bh_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        result[order] = np.minimum(ranked, 1.0)
    return result.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.window_manifest.open(newline="", encoding="utf-8") as handle:
        windows = list(csv.DictReader(handle))
    if len(windows) != 200:
        raise RuntimeError(f"Expected 200 frozen windows, got {len(windows)}")

    midi_cache: dict[str, list[pretty_midi.Note]] = {}
    event_rows: list[dict[str, object]] = []
    processing_rows: list[dict[str, object]] = []
    for index, window in enumerate(windows, start=1):
        recording_id = window["recording_id"]
        start = float(window["start_seconds"])
        duration = float(window["duration_seconds"])
        audio_path = args.audio_root / window["audio_filename"]
        midi_path = args.midi_root / window["midi_filename"]
        if recording_id not in midi_cache:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            midi_cache[recording_id] = sorted(
                [note for instrument in midi.instruments for note in instrument.notes],
                key=lambda note: (note.start, note.pitch, note.end),
            )
        notes = midi_cache[recording_id]
        audio, sr = read_segment(audio_path, start, duration)
        extracted = extract_window_events(audio, sr, notes, start, duration)
        window_id = f"{recording_id}__w{int(window['window_index']):02d}"
        for event in extracted:
            event_rows.append(
                {
                    "recording_id": recording_id,
                    "window_id": window_id,
                    "window_index": int(window["window_index"]),
                    "window_start_seconds": start,
                    **event,
                }
            )
        processing_rows.append(
            {
                "recording_id": recording_id,
                "window_id": window_id,
                "audio_path": str(audio_path),
                "midi_path": str(midi_path),
                "sample_rate": sr,
                "audio_samples": len(audio),
                "usable_notes": len(extracted),
                "status": "ok",
            }
        )
        print(f"[{index:03d}/{len(windows):03d}] {window_id}: {len(extracted)} notes", flush=True)

    events = pd.DataFrame(event_rows)
    processing = pd.DataFrame(processing_rows)
    events.to_csv(args.output_dir / "event_measurements.csv", index=False)
    processing.to_csv(args.output_dir / "window_processing.csv", index=False)

    group_sizes = events.groupby(["recording_id", "pitch"]).size()
    valid_groups = group_sizes[group_sizes >= 4].index
    group_key = pd.MultiIndex.from_frame(events[["recording_id", "pitch"]])
    valid_events = events[group_key.isin(valid_groups)].copy()
    valid_events["velocity_centered"] = center_by_recording_pitch(valid_events, "velocity")
    for estimator in ESTIMATORS:
        valid_events[f"{estimator}_centered"] = center_by_recording_pitch(valid_events, estimator)

    rng = np.random.default_rng(SEED)
    group_rows: list[dict[str, object]] = []
    event_summary: dict[str, object] = {}
    for estimator in ESTIMATORS:
        for (recording_id, pitch), group in valid_events.groupby(["recording_id", "pitch"]):
            rho, pvalue = safe_spearman(
                group["velocity"].to_numpy(float), group[estimator].to_numpy(float)
            )
            group_rows.append(
                {
                    "recording_id": recording_id,
                    "pitch": pitch,
                    "estimator": estimator,
                    "n_notes": len(group),
                    "spearman_rho": rho,
                    "spearman_pvalue": pvalue,
                    "quartile_ranking_accuracy": pairwise_quartile_accuracy(group, estimator),
                }
            )
        centered_rho, centered_p = safe_spearman(
            valid_events["velocity_centered"].to_numpy(float),
            valid_events[f"{estimator}_centered"].to_numpy(float),
        )
        event_summary[estimator] = {
            "pooled_pitch_centered_rho": centered_rho,
            "pooled_pitch_centered_pvalue": centered_p,
        }

    group_table = pd.DataFrame(group_rows)
    group_table.to_csv(args.output_dir / "recording_pitch_metrics.csv", index=False)
    event_admission = {}
    for estimator in ESTIMATORS:
        subset = group_table[group_table["estimator"] == estimator]
        rho_stats = cluster_bootstrap_values(subset, "spearman_rho", "median", rng)
        ranking_stats = cluster_bootstrap_values(
            subset, "quartile_ranking_accuracy", "mean", rng
        )
        admitted = bool(
            rho_stats["estimate"] >= 0.40
            and rho_stats["ci_low"] > 0.25
            and ranking_stats["estimate"] >= 0.70
            and ranking_stats["ci_low"] > 0.65
            and rho_stats["sign_stability"] >= 0.90
        )
        event_summary[estimator].update(
            {
                "within_group_rho": rho_stats,
                "quartile_ranking_accuracy": ranking_stats,
                "admitted": admitted,
            }
        )
        event_admission[estimator] = admitted

    window_rows: list[dict[str, object]] = []
    for window_id, group in valid_events.groupby("window_id"):
        row: dict[str, object] = {
            "recording_id": group["recording_id"].iloc[0],
            "window_id": window_id,
            "n_notes": len(group),
        }
        midi_features = summarize_window(group, "velocity_centered")
        row.update({f"midi_{name}": value for name, value in midi_features.items()})
        for estimator in ESTIMATORS:
            column = f"{estimator}_centered"
            features = summarize_window(group, column)
            row.update({f"{estimator}__{name}": value for name, value in features.items()})
            # +6 dB changes absolute dB levels by a constant and leaves onset deltas
            # unchanged. Re-center exactly as in the primary path before summarizing.
            scaled = group.copy()
            if estimator == "harmonic_attack_level_db":
                scaled[f"{column}_gain6"] = scaled[column] + 6.0
            else:
                scaled[f"{column}_gain6"] = scaled[column]
            gain_features = summarize_window(scaled, f"{column}_gain6")
            row.update(
                {f"{estimator}__{name}__gain6": value for name, value in gain_features.items()}
            )
        window_rows.append(row)
    window_table = pd.DataFrame(window_rows).sort_values(["recording_id", "window_id"])
    window_table.to_csv(args.output_dir / "window_feature_pairs.csv", index=False)

    fidelity_rows: list[dict[str, object]] = []
    for estimator in ESTIMATORS:
        for feature in WINDOW_FEATURES:
            audio_column = f"{estimator}__{feature}"
            midi_column = f"midi_{feature}"
            stats = clustered_correlation(window_table, midi_column, audio_column, rng)
            gain_column = f"{estimator}__{feature}__gain6"
            gain_drift = np.abs(
                window_table[audio_column].to_numpy(float)
                - window_table[gain_column].to_numpy(float)
            )
            max_gain_drift = float(np.nanmax(gain_drift))
            admitted = bool(
                event_admission[estimator]
                and stats["estimate"] >= 0.50
                and stats["ci_low"] > 0.35
                and stats["sign_stability"] >= 0.90
                and max_gain_drift < 1e-3
            )
            fidelity_rows.append(
                {
                    "estimator": estimator,
                    "feature": feature,
                    **stats,
                    "max_abs_gain6_drift": max_gain_drift,
                    "admitted": admitted,
                }
            )
    qvalues = bh_adjust([float(row["pvalue"]) for row in fidelity_rows])
    for row, qvalue in zip(fidelity_rows, qvalues):
        row["qvalue"] = qvalue
    fidelity = pd.DataFrame(fidelity_rows)
    fidelity.to_csv(args.output_dir / "window_feature_fidelity.csv", index=False)

    summary = {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "target_sample_rate": TARGET_SR,
        "stft_n_fft": N_FFT,
        "stft_hop": HOP,
        "n_manifest_windows": len(windows),
        "n_processed_windows": len(processing),
        "n_event_measurements": len(events),
        "n_valid_group_events": len(valid_events),
        "n_valid_recording_pitch_groups": int(len(valid_groups)),
        "event_estimators": event_summary,
        "window_feature_fidelity": fidelity_rows,
        "admitted_window_features": [
            f"{row['estimator']}::{row['feature']}" for row in fidelity_rows if row["admitted"]
        ],
        "gain_invariance_note": (
            "The +6 dB branch applies the exact dB-domain effect of multiplying each "
            "window waveform by 10^(6/20), then applies the same centering and summaries."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

