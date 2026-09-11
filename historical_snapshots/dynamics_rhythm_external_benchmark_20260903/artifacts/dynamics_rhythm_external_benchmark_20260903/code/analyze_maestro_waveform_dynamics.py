#!/usr/bin/env python3
"""Validate a deployable waveform-only dynamics estimator on frozen MAESTRO windows."""

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
from scipy.signal import find_peaks, resample_poly, stft
from scipy.stats import spearmanr


SEED = 20260903
N_BOOT = 10_000
TARGET_SR = 44_100
N_FFT = 4096
HOP = 512
EPS = 1e-12
FEATURES = ("dynamics_span", "dynamics_iqr", "dynamics_adjacent_change")


def read_segment(path: Path, start_seconds: float, duration_seconds: float) -> np.ndarray:
    with sf.SoundFile(path) as handle:
        handle.seek(int(round(start_seconds * handle.samplerate)))
        frames = int(round(duration_seconds * handle.samplerate))
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)
        sr = int(handle.samplerate)
    mono = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    if sr != TARGET_SR:
        divisor = math.gcd(sr, TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // divisor, sr // divisor).astype(np.float32)
    expected = int(round(duration_seconds * TARGET_SR))
    if len(mono) < expected:
        mono = np.pad(mono, (0, expected - len(mono)))
    return mono[:expected]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return {name: float("nan") for name in FEATURES}
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    return {
        "dynamics_span": float(q90 - q10),
        "dynamics_iqr": float(q75 - q25),
        "dynamics_adjacent_change": float(np.median(np.abs(np.diff(values)))),
    }


def chord_events(notes: list[pretty_midi.Note], start: float, duration: float) -> list[tuple[float, float]]:
    events = sorted(
        (float(note.start - start), float(note.velocity))
        for note in notes
        if start <= note.start < start + duration and note.velocity > 0
    )
    clusters: list[list[tuple[float, float]]] = []
    for event in events:
        if not clusters or event[0] - clusters[-1][0][0] > 0.030:
            clusters.append([event])
        else:
            clusters[-1].append(event)
    return [
        (float(np.median([event[0] for event in cluster])), float(np.median([event[1] for event in cluster])))
        for cluster in clusters
    ]


def waveform_events(audio: np.ndarray) -> list[tuple[float, float]]:
    frequencies, times, spectrum = stft(
        audio,
        fs=TARGET_SR,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    power = np.square(np.abs(spectrum), dtype=np.float64)
    band = (frequencies >= 30.0) & (frequencies <= 8000.0)
    log_power = np.log1p(1e6 * power[band])
    flux = np.maximum(np.diff(log_power, axis=1, prepend=log_power[:, :1]), 0.0).mean(axis=0)
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    minimum_distance = max(1, int(round(0.080 * TARGET_SR / HOP)))
    peak_frames, _ = find_peaks(flux, height=median + mad, distance=minimum_distance)
    band_indices = np.flatnonzero(band)
    events: list[tuple[float, float]] = []
    for peak_frame in peak_frames:
        onset_time = float(times[peak_frame])
        post = np.flatnonzero((times >= onset_time + 0.020) & (times < onset_time + 0.120))
        if not len(post):
            continue
        profile = np.median(power[np.ix_(band_indices, post)], axis=1)
        local = np.flatnonzero((profile[1:-1] > profile[:-2]) & (profile[1:-1] >= profile[2:])) + 1
        if not len(local):
            continue
        strongest = local[np.argsort(profile[local])[-16:]]
        level_db = float(10.0 * np.log10(np.sum(profile[strongest]) + EPS))
        events.append((onset_time, level_db))
    return events


def match_events(
    reference: list[tuple[float, float]], prediction: list[tuple[float, float]], tolerance: float = 0.070
) -> list[tuple[int, int]]:
    candidates = []
    for i, (reference_time, _) in enumerate(reference):
        for j, (prediction_time, _) in enumerate(prediction):
            distance = abs(reference_time - prediction_time)
            if distance <= tolerance:
                candidates.append((distance, i, j))
    matches = []
    used_reference: set[int] = set()
    used_prediction: set[int] = set()
    for _, i, j in sorted(candidates):
        if i not in used_reference and j not in used_prediction:
            matches.append((i, j))
            used_reference.add(i)
            used_prediction.add(j)
    return sorted(matches)


def cluster_bootstrap_correlation(
    table: pd.DataFrame, x_column: str, y_column: str, rng: np.random.Generator
) -> dict[str, float | int]:
    clean = table[["recording_id", x_column, y_column]].dropna()
    point, pvalue = safe_spearman(clean[x_column].to_numpy(float), clean[y_column].to_numpy(float))
    groups = {key: group for key, group in clean.groupby("recording_id")}
    ids = np.array(sorted(groups))
    draws = []
    for _ in range(N_BOOT):
        sampled = rng.choice(ids, len(ids), replace=True)
        x = np.concatenate([groups[key][x_column].to_numpy(float) for key in sampled])
        y = np.concatenate([groups[key][y_column].to_numpy(float) for key in sampled])
        value, _ = safe_spearman(x, y)
        if np.isfinite(value):
            draws.append(value)
    draws = np.asarray(draws)
    return {
        "estimate": point,
        "pvalue": pvalue,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "sign_stability": float(np.mean(draws > 0) if point >= 0 else np.mean(draws < 0)),
        "n_recordings": int(len(ids)),
        "n_items": int(len(clean)),
        "n_boot_valid": int(len(draws)),
    }


def cluster_bootstrap_quartile(table: pd.DataFrame, rng: np.random.Generator) -> dict[str, float | int]:
    def per_recording(group: pd.DataFrame) -> float:
        q25, q75 = group["velocity"].quantile([0.25, 0.75])
        low = group[group["velocity"] <= q25]["attack_level_db"].to_numpy(float)
        high = group[group["velocity"] >= q75]["attack_level_db"].to_numpy(float)
        comparison = high[:, None] - low[None, :]
        return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))

    values = table.groupby("recording_id").apply(per_recording, include_groups=False).dropna()
    ids = values.index.to_numpy()
    point = float(values.mean())
    draws = np.empty(N_BOOT)
    for index in range(N_BOOT):
        draws[index] = values.loc[rng.choice(ids, len(ids), replace=True)].mean()
    return {
        "estimate": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "sign_stability": float(np.mean(draws > 0.5)),
        "n_recordings": int(len(ids)),
    }


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
    window_rows: list[dict[str, object]] = []
    for index, window in enumerate(windows, 1):
        recording_id = window["recording_id"]
        window_id = f"{recording_id}__w{window['window_index']}"
        start = float(window["start_seconds"])
        duration = float(window["duration_seconds"])
        audio = read_segment(args.audio_root / window["audio_filename"], start, duration)
        if recording_id not in midi_cache:
            midi = pretty_midi.PrettyMIDI(str(args.midi_root / window["midi_filename"]))
            midi_cache[recording_id] = [note for instrument in midi.instruments for note in instrument.notes]
        reference = chord_events(midi_cache[recording_id], start, duration)
        prediction = waveform_events(audio)
        matches = match_events(reference, prediction)
        for ref_index, pred_index in matches:
            event_rows.append(
                {
                    "recording_id": recording_id,
                    "window_id": window_id,
                    "reference_time": reference[ref_index][0],
                    "prediction_time": prediction[pred_index][0],
                    "absolute_timing_error": abs(reference[ref_index][0] - prediction[pred_index][0]),
                    "velocity": reference[ref_index][1],
                    "attack_level_db": prediction[pred_index][1],
                }
            )
        reference_features = summarize(np.asarray([value for _, value in reference]))
        prediction_features = summarize(np.asarray([value for _, value in prediction]))
        gain_features = summarize(np.asarray([value + 6.0 for _, value in prediction]))
        row: dict[str, object] = {
            "recording_id": recording_id,
            "window_id": window_id,
            "n_reference": len(reference),
            "n_prediction": len(prediction),
            "n_matches": len(matches),
        }
        for feature in FEATURES:
            row[f"ref_{feature}"] = reference_features[feature]
            row[f"pred_{feature}"] = prediction_features[feature]
            row[f"gain6_{feature}"] = gain_features[feature]
        window_rows.append(row)
        if index == 1 or index % 20 == 0 or index == len(windows):
            print(f"waveform dynamics {index}/{len(windows)}", flush=True)

    events = pd.DataFrame(event_rows)
    window_table = pd.DataFrame(window_rows)
    events["velocity_centered"] = events["velocity"] - events.groupby("recording_id")["velocity"].transform("median")
    events["attack_centered"] = events["attack_level_db"] - events.groupby("recording_id")["attack_level_db"].transform("median")
    events.to_csv(args.output_dir / "matched_events.csv", index=False)
    window_table.to_csv(args.output_dir / "window_features.csv", index=False)

    total_reference = int(window_table["n_reference"].sum())
    total_prediction = int(window_table["n_prediction"].sum())
    total_matches = int(window_table["n_matches"].sum())
    precision = total_matches / total_prediction
    recall = total_matches / total_reference
    onset_f1 = 2.0 * precision * recall / (precision + recall)
    rng = np.random.default_rng(SEED)
    event_rho = cluster_bootstrap_correlation(events, "velocity_centered", "attack_centered", rng)
    quartile = cluster_bootstrap_quartile(events, rng)
    event_admitted = bool(
        onset_f1 >= 0.65
        and event_rho["estimate"] >= 0.40
        and event_rho["ci_low"] > 0.25
        and quartile["estimate"] >= 0.70
        and quartile["ci_low"] > 0.65
    )
    feature_rows = []
    for feature in FEATURES:
        stats = cluster_bootstrap_correlation(window_table, f"ref_{feature}", f"pred_{feature}", rng)
        drift = float(np.nanmax(np.abs(window_table[f"gain6_{feature}"] - window_table[f"pred_{feature}"])))
        admitted = bool(
            event_admitted
            and stats["estimate"] >= 0.50
            and stats["ci_low"] > 0.35
            and stats["sign_stability"] >= 0.90
            and drift < 0.001
        )
        feature_rows.append({"feature": feature, **stats, "max_abs_gain6_drift": drift, "admitted": admitted})
    summary = {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_windows": len(window_table),
        "n_reference_onsets": total_reference,
        "n_predicted_onsets": total_prediction,
        "n_matched_onsets": total_matches,
        "onset_precision": precision,
        "onset_recall": recall,
        "onset_f1": onset_f1,
        "matched_event_rho": event_rho,
        "quartile_ranking_accuracy": quartile,
        "event_admitted": event_admitted,
        "feature_fidelity": feature_rows,
        "admitted_features": [row["feature"] for row in feature_rows if row["admitted"]],
        "protocol_status": "post-protocol deployment addendum frozen before execution",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
