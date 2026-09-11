#!/usr/bin/env python3
"""Frozen 10 s audio-to-feature functions for the external generator test."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, resample_poly, stft


SR = 44_100
DURATION = 10.0
N_SAMPLES = int(SR * DURATION)
N_FFT = 4096
HOP = 512
EPS = 1e-12

DYNAMICS_FEATURES = ("dynamics_span", "dynamics_iqr", "dynamics_adjacent_change")
RHYTHM_FEATURES = ("ibi_cv", "tempo_tv", "tempo_entropy")
STRUCTURE_FEATURES = (
    "section_duration_cv",
    "section_duration_entropy",
    "section_bars_cv",
    "section_bars_offmode_fraction",
)
DESCRIPTIVE_STRUCTURE_FEATURES = ("section_duration_median", "section_bars_median")


def resample(audio: np.ndarray, source_sr: int) -> np.ndarray:
    if source_sr == SR:
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(int(source_sr), SR)
    return resample_poly(audio, SR // divisor, int(source_sr) // divisor, axis=0).astype(np.float32)


def read_audio(path: Path, start: float = 0.0, duration: float = DURATION) -> np.ndarray:
    with sf.SoundFile(path) as handle:
        handle.seek(max(0, int(round(start * handle.samplerate))))
        frames = int(round(duration * handle.samplerate))
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)
        source_sr = int(handle.samplerate)
    audio = resample(audio, source_sr)
    expected = int(round(duration * SR))
    if len(audio) < expected:
        audio = np.pad(audio, ((0, expected - len(audio)), (0, 0)))
    return audio[:expected]


def peak_protect(audio: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return np.asarray(audio, dtype=np.float32)


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1, dtype=np.float64).astype(np.float32)


def waveform_events(audio: np.ndarray) -> list[tuple[float, float]]:
    frequencies, times, spectrum = stft(
        mono(audio), fs=SR, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - HOP, nfft=N_FFT, boundary=None, padded=False,
    )
    power = np.square(np.abs(spectrum), dtype=np.float64)
    band = (frequencies >= 30.0) & (frequencies <= 8000.0)
    log_power = np.log1p(1e6 * power[band])
    flux = np.maximum(np.diff(log_power, axis=1, prepend=log_power[:, :1]), 0.0).mean(axis=0)
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    minimum_distance = max(1, int(round(0.080 * SR / HOP)))
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


def dynamics_features(audio: np.ndarray) -> dict[str, float]:
    values = np.asarray([level for _, level in waveform_events(audio)], dtype=float)
    result = {feature: float("nan") for feature in DYNAMICS_FEATURES}
    if len(values) < 8:
        return result
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    result.update({
        "dynamics_span": float(q90 - q10),
        "dynamics_iqr": float(q75 - q25),
        "dynamics_adjacent_change": float(np.median(np.abs(np.diff(values)))),
    })
    return result


def load_beats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 2:
        return np.asarray(data[:, 0], float), np.ones(len(data), dtype=int)
    return np.asarray(data[:, 0], float), np.asarray(data[:, 1], int)


def rhythm_features(times: np.ndarray) -> dict[str, float]:
    result = {feature: float("nan") for feature in RHYTHM_FEATURES}
    times = np.asarray(times, dtype=float)
    if len(times) < 16:
        return result
    ibi = np.diff(times)
    ibi = ibi[(ibi > 0) & np.isfinite(ibi)]
    if len(ibi) < 15 or np.median(ibi) <= 0:
        return result
    q25, q75 = np.quantile(ibi, [0.25, 0.75])
    result["ibi_cv"] = float((q75 - q25) / np.median(ibi))
    bpm = 60.0 / ibi
    bpm = bpm[(bpm >= 30.0) & (bpm <= 300.0)]
    if len(bpm) >= 14 and np.median(bpm) > 0:
        result["tempo_tv"] = float(np.median(np.abs(np.diff(bpm))) / np.median(bpm))
        counts, _ = np.histogram(bpm, bins=np.arange(30.0, 305.0, 5.0))
        probabilities = counts[counts > 0] / counts.sum()
        result["tempo_entropy"] = float(
            -(probabilities * np.log(probabilities)).sum() / np.log(len(counts))
        )
    return result


def clean_boundaries(values: np.ndarray, duration: float = DURATION) -> np.ndarray:
    values = np.unique(np.clip(np.asarray(values, dtype=float), 0.0, duration))
    interior = values[(values >= 1.0) & (values <= duration - 1.0)]
    clean = [0.0]
    for value in interior:
        if value - clean[-1] >= 1.0:
            clean.append(float(value))
    if duration - clean[-1] < 1.0 and len(clean) > 1:
        clean.pop()
    clean.append(float(duration))
    return np.asarray(clean)


def allinone_boundaries(path: Path, duration: float = DURATION) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not segments:
        return np.asarray([0.0, duration])
    values = [float(segment["start"]) for segment in segments]
    values.append(float(segments[-1]["end"]))
    return clean_boundaries(np.asarray(values), duration)


def structure_features(boundaries: np.ndarray, downbeats: np.ndarray) -> dict[str, float]:
    names = STRUCTURE_FEATURES + DESCRIPTIVE_STRUCTURE_FEATURES
    result = {feature: float("nan") for feature in names}
    durations = np.diff(np.asarray(boundaries, dtype=float))
    durations = durations[durations > 0]
    if len(durations) >= 3:
        median = float(np.median(durations))
        q25, q75 = np.quantile(durations, [0.25, 0.75])
        bins = np.arange(0.0, max(32.0, np.ceil(durations.max() / 2.0) * 2.0 + 2.0), 2.0)
        counts, _ = np.histogram(durations, bins=bins)
        probabilities = counts[counts > 0] / counts.sum()
        result["section_duration_median"] = median
        result["section_duration_cv"] = float((q75 - q25) / median) if median > 0 else float("nan")
        result["section_duration_entropy"] = float(
            -(probabilities * np.log(probabilities)).sum() / np.log(max(len(counts), 2))
        )
    downbeats = np.asarray(downbeats, dtype=float)
    if len(downbeats) >= 4 and len(boundaries) >= 4:
        snapped = np.unique([int(np.argmin(np.abs(downbeats - boundary))) for boundary in boundaries])
        lengths = np.diff(snapped)
        lengths = lengths[lengths > 0]
        if len(lengths) >= 3:
            median = float(np.median(lengths))
            q25, q75 = np.quantile(lengths, [0.25, 0.75])
            values, counts = np.unique(lengths, return_counts=True)
            mode = values[np.flatnonzero(counts == counts.max())[0]]
            result["section_bars_median"] = median
            result["section_bars_cv"] = float((q75 - q25) / median) if median > 0 else float("nan")
            result["section_bars_offmode_fraction"] = float(np.mean(lengths != mode))
    return result
