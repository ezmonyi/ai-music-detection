#!/usr/bin/env python3
"""Frozen S16/S8/D3/R3/P6 definitions for the source-diversity expansion.

The S16 implementation deliberately matches the 2026-09-04 four-family
extractor.  S8 is a bandwidth-controlled analogue whose highest used bin is
strictly below 7.5 kHz.  Neither representation estimates or repairs missing
native bandwidth; eligibility is a separate output field.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, resample_poly, stft


SR = 44_100
N_FFT = 4_096
SPECTRAL_HOP = 1_024
EVENT_HOP = 512
EPS = 1e-14
WORKFLOW_EPS = 1e-12

S16_FEATURES = (
    "tilt_1_5k_db_oct", "hf_tilt_5_16k_db_oct", "hf_ratio_5_16_db",
    "air_ratio_12_20_db", "sibilance_ratio_5_10_db", "hf_flatness",
    "hf_entropy", "hf_crest_db", "fakeprint_peak_density",
    "fakeprint_periodicity", "hf_flux", "hf_frame_similarity",
    "hf_power_sd_db", "hf_mod_4_12_share", "sibilance_contrast_db",
    "sibilance_burst_rate_hz",
)
S8_FEATURES = (
    "tilt_1_5k_db_oct", "hf_tilt_5_7p5k_db_oct", "hf_ratio_5_7p5_db",
    "sibilance_ratio_5_7p5_db", "hf_flatness_5_7p5",
    "hf_entropy_5_7p5", "hf_crest_5_7p5_db",
    "fakeprint_peak_density_5_7p5_per_khz", "fakeprint_periodicity_5_7p5",
    "hf_flux_5_7p5", "hf_frame_similarity_5_7p5",
    "hf_power_sd_5_7p5_db", "hf_mod_4_12_share_5_7p5",
    "sibilance_contrast_5_7p5_db", "sibilance_burst_rate_5_7p5_hz",
)
D_FEATURES = ("dynamics_span", "dynamics_iqr", "dynamics_adjacent_change")
R_FEATURES = ("ibi_cv", "tempo_tv", "tempo_entropy")
P_FEATURES = (
    "section_duration_cv", "section_duration_entropy", "section_bars_cv",
    "section_bars_offmode_fraction", "section_duration_median",
    "section_bars_median",
)


def read_audio(path: Path, duration: float, *, strict: bool = False) -> np.ndarray:
    with sf.SoundFile(path) as handle:
        frames = int(round(duration * handle.samplerate))
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)
        source_sr = int(handle.samplerate)
    if strict and len(audio) < int(frames * 0.99):
        raise RuntimeError(f"Short audio for {duration:.3f}s view: {path}")
    if source_sr != SR:
        divisor = math.gcd(source_sr, SR)
        audio = resample_poly(audio, SR // divisor, source_sr // divisor, axis=0).astype(np.float32)
    expected = int(round(duration * SR))
    if len(audio) < expected:
        audio = np.pad(audio, ((0, expected - len(audio)), (0, 0)))
    return audio[:expected]


def read_spectral_audio(path: Path, duration: float) -> np.ndarray:
    """Match the prior S16 decoder: float64, 44.1 kHz, mono then DC removal."""
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if sample_rate != SR:
        raise RuntimeError(f"Unexpected spectral stem sample rate {sample_rate}: {path}")
    expected = int(round(duration * SR))
    if audio.shape[0] < int(expected * 0.99):
        raise RuntimeError(f"Short spectral stem ({audio.shape[0]} samples): {path}")
    if audio.shape[0] < expected:
        audio = np.pad(audio, ((0, expected-audio.shape[0]), (0, 0)))
    value = audio[:expected].mean(axis=1)
    value -= value.mean()
    return value


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1, dtype=np.float64).astype(np.float32)


def framed_power(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float64) if audio.ndim == 1 else audio.mean(axis=1, dtype=np.float64)
    audio = audio - audio.mean()
    if audio.size < N_FFT:
        audio = np.pad(audio, (0, N_FFT - audio.size))
    frame_count = 1 + (audio.size - N_FFT) // SPECTRAL_HOP
    stride = audio.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(frame_count, N_FFT),
        strides=(SPECTRAL_HOP * stride, stride), writeable=False,
    )
    spectrum = np.fft.rfft(frames * np.hanning(N_FFT), axis=1)
    power = np.abs(spectrum) ** 2 + EPS
    rms = np.sqrt(np.mean(frames * frames, axis=1) + EPS)
    return power, rms, np.fft.rfftfreq(N_FFT, 1.0 / SR)


def vocal_activity(mix: np.ndarray, vocals: np.ndarray) -> dict[str, float | int]:
    mix_mono = mix.mean(axis=1, dtype=np.float64) if mix.ndim > 1 else np.asarray(mix, dtype=np.float64)
    vocal_mono = vocals.mean(axis=1, dtype=np.float64) if vocals.ndim > 1 else np.asarray(vocals, dtype=np.float64)
    mix_rms = float(np.sqrt(np.mean(mix_mono * mix_mono) + EPS))
    vocal_rms = float(np.sqrt(np.mean(vocal_mono * vocal_mono) + EPS))
    raw_power, frame_rms, _ = framed_power(vocal_mono)
    del raw_power
    rms_db = 20.0 * np.log10(frame_rms + EPS)
    active = rms_db >= max(float(np.percentile(rms_db, 30)), float(rms_db.max() - 35.0))
    if int(active.sum()) < 16:
        active[np.argsort(frame_rms)[-min(16, frame_rms.size):]] = True
    ratio = 20.0 * math.log10((vocal_rms + EPS) / (mix_rms + EPS))
    rms_dbfs = 20.0 * math.log10(vocal_rms + EPS)
    return {
        "vocal_to_mix_rms_db": ratio,
        "vocal_rms_dbfs": rms_dbfs,
        "vocal_active_frame_count": int(active.sum()),
        "vocal_active_frame_ratio": float(active.mean()),
        "vocal_analysis_eligible": int(ratio >= -18.0 and rms_dbfs >= -60.0 and int(active.sum()) >= 16),
    }


def _band(power: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    return power[:, (frequencies >= low) & (frequencies < high)]


def _db_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10((numerator + EPS) / (denominator + EPS))


def _modulation_share(series: np.ndarray, low: float = 4.0, high: float = 12.0) -> float:
    frame_rate = SR / SPECTRAL_HOP
    if series.size < 16 or float(np.std(series)) < 1e-9:
        return 0.0
    size = max(3, int(frame_rate) | 1)
    centered = series - median_filter(series, size=size, mode="nearest")
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size))) ** 2
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / frame_rate)
    analysis = (frequencies >= 0.5) & (frequencies <= 20.0)
    target = (frequencies >= low) & (frequencies <= high)
    return float(spectrum[target].sum() / (spectrum[analysis].sum() + EPS))


def _lower_envelope_residual(values: np.ndarray, area: int = 10) -> np.ndarray:
    if values.size <= area:
        area = max(2, values.size // 2)
    patches = np.lib.stride_tricks.sliding_window_view(values, area)
    minima = np.argmin(patches, axis=1) + np.arange(patches.shape[0])
    indices = np.unique(np.concatenate(([0], minima, [values.size - 1])))
    envelope = np.interp(np.arange(values.size), indices, values[indices])
    residual = np.clip(values - np.maximum(envelope, -80.0), 0.0, 8.0)
    maximum = float(residual.max())
    return residual / maximum if maximum > 1e-9 else residual


def _active_corrected_power(vocals: np.ndarray, bias_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_power, rms, frequencies = framed_power(vocals)
    rms_db = 20.0 * np.log10(rms + EPS)
    active = rms_db >= max(float(np.percentile(rms_db, 30)), float(rms_db.max() - 35.0))
    if int(active.sum()) < 16:
        active[np.argsort(rms)[-min(16, rms.size):]] = True
    if bias_db.shape != (raw_power.shape[1],):
        raise RuntimeError(f"Bias has {bias_db.shape}, expected {(raw_power.shape[1],)}")
    corrected = raw_power * np.power(10.0, -bias_db / 10.0)[None, :]
    return corrected[active], frequencies


def _spectral_metrics(
    p: np.ndarray, frequencies: np.ndarray, *, hf_high: float, legacy_s16: bool,
) -> dict[str, float]:
    low = _band(p, frequencies, 300, 5_000).sum(axis=1)
    body = _band(p, frequencies, 1_000, 5_000).sum(axis=1)
    sib = _band(p, frequencies, 5_000, 10_000 if legacy_s16 else 7_500).sum(axis=1)
    hf = _band(p, frequencies, 5_000, hf_high)
    hf_power = hf.sum(axis=1)
    sib_ratio = _db_ratio(sib, low)
    hf_ratio = _db_ratio(hf_power, body)
    flatness = np.exp(np.mean(np.log(hf + EPS), axis=1)) / (np.mean(hf, axis=1) + EPS)
    normalized = hf / (hf.sum(axis=1, keepdims=True) + EPS)
    entropy = -np.sum(normalized * np.log(normalized + EPS), axis=1) / math.log(hf.shape[1])
    crest = 10.0 * np.log10((hf.max(axis=1) + EPS) / (hf.mean(axis=1) + EPS))
    log_hf = np.log(hf + EPS)
    centered = log_hf - log_hf.mean(axis=1, keepdims=True)
    unit = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + EPS)
    similarity = np.sum(unit[1:] * unit[:-1], axis=1)
    flux = 1.0 - similarity
    hf_db = 10.0 * np.log10(hf_power + EPS)
    middle = float(np.median(sib_ratio))
    threshold = middle + 1.5 * float(np.median(np.abs(sib_ratio - middle)))
    bursts = sib_ratio > threshold
    active_duration = max(1e-6, p.shape[0] * SPECTRAL_HOP / SR)
    burst_rate = float(np.sum(bursts[1:] & ~bursts[:-1]) / active_duration)
    mean_db = 10.0 * np.log10(np.mean(p, axis=0) + EPS)
    hf_mask = (frequencies >= 5_000) & (frequencies < hf_high)
    residual = _lower_envelope_residual(mean_db[hf_mask])
    peaks, _ = find_peaks(residual, height=0.35, prominence=0.08, distance=2)
    residual_power = np.abs(np.fft.rfft(residual - residual.mean())) ** 2
    periodicity = float(residual_power[4:].max() / (residual_power[1:].sum() + EPS))

    def slope(low_hz: float, high_hz: float) -> float:
        mask = (frequencies >= low_hz) & (frequencies <= high_hz)
        return float(np.polyfit(np.log2(frequencies[mask]), mean_db[mask], 1)[0])

    common = {
        "tilt_1_5k_db_oct": slope(1_000, 5_000),
        "hf_ratio": float(np.median(hf_ratio)),
        "sib_ratio": float(np.median(sib_ratio)),
        "flatness": float(np.median(flatness)),
        "entropy": float(np.median(entropy)),
        "crest": float(np.median(crest)),
        "periodicity": periodicity,
        "flux": float(np.mean(flux)),
        "similarity": float(np.mean(similarity)),
        "power_sd": float(np.std(hf_db)),
        "modulation": _modulation_share(hf_db),
        "sib_contrast": float(np.percentile(sib_ratio, 90) - np.median(sib_ratio)),
        "sib_burst_rate": burst_rate,
    }
    if legacy_s16:
        air = _band(p, frequencies, 12_000, 20_000).sum(axis=1)
        result = {
            "tilt_1_5k_db_oct": common["tilt_1_5k_db_oct"],
            "hf_tilt_5_16k_db_oct": slope(5_000, 16_000),
            "hf_ratio_5_16_db": common["hf_ratio"],
            "air_ratio_12_20_db": float(np.median(_db_ratio(air, body))),
            "sibilance_ratio_5_10_db": common["sib_ratio"],
            "hf_flatness": common["flatness"], "hf_entropy": common["entropy"],
            "hf_crest_db": common["crest"],
            "fakeprint_peak_density": float(peaks.size / 11.0),
            "fakeprint_periodicity": common["periodicity"], "hf_flux": common["flux"],
            "hf_frame_similarity": common["similarity"], "hf_power_sd_db": common["power_sd"],
            "hf_mod_4_12_share": common["modulation"],
            "sibilance_contrast_db": common["sib_contrast"],
            "sibilance_burst_rate_hz": common["sib_burst_rate"],
        }
    else:
        result = {
            "tilt_1_5k_db_oct": common["tilt_1_5k_db_oct"],
            "hf_tilt_5_7p5k_db_oct": slope(5_000, 7_500),
            "hf_ratio_5_7p5_db": common["hf_ratio"],
            "sibilance_ratio_5_7p5_db": common["sib_ratio"],
            "hf_flatness_5_7p5": common["flatness"], "hf_entropy_5_7p5": common["entropy"],
            "hf_crest_5_7p5_db": common["crest"],
            "fakeprint_peak_density_5_7p5_per_khz": float(peaks.size / 2.5),
            "fakeprint_periodicity_5_7p5": common["periodicity"],
            "hf_flux_5_7p5": common["flux"],
            "hf_frame_similarity_5_7p5": common["similarity"],
            "hf_power_sd_5_7p5_db": common["power_sd"],
            "hf_mod_4_12_share_5_7p5": common["modulation"],
            "sibilance_contrast_5_7p5_db": common["sib_contrast"],
            "sibilance_burst_rate_5_7p5_hz": common["sib_burst_rate"],
        }
    if not all(np.isfinite(list(result.values()))):
        raise RuntimeError("Non-finite spectral metric")
    return result


def spectral_families(vocals: np.ndarray, bias_db: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    power, frequencies = _active_corrected_power(vocals, bias_db)
    return (
        _spectral_metrics(power, frequencies, hf_high=16_000, legacy_s16=True),
        _spectral_metrics(power, frequencies, hf_high=7_500, legacy_s16=False),
    )


def waveform_events(audio: np.ndarray) -> list[float]:
    frequencies, times, spectrum = stft(
        mono(audio), fs=SR, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - EVENT_HOP, nfft=N_FFT, boundary=None, padded=False,
    )
    power = np.square(np.abs(spectrum), dtype=np.float64)
    mask = (frequencies >= 30.0) & (frequencies <= 8_000.0)
    log_power = np.log1p(1e6 * power[mask])
    flux = np.maximum(np.diff(log_power, axis=1, prepend=log_power[:, :1]), 0.0).mean(axis=0)
    median = float(np.median(flux)); mad = float(np.median(np.abs(flux - median)))
    peaks, _ = find_peaks(flux, height=median + mad, distance=max(1, round(0.080 * SR / EVENT_HOP)))
    bins = np.flatnonzero(mask)
    values: list[float] = []
    for peak in peaks:
        onset = float(times[peak])
        post = np.flatnonzero((times >= onset + 0.020) & (times < onset + 0.120))
        if not len(post):
            continue
        profile = np.median(power[np.ix_(bins, post)], axis=1)
        local = np.flatnonzero((profile[1:-1] > profile[:-2]) & (profile[1:-1] >= profile[2:])) + 1
        if len(local):
            strongest = local[np.argsort(profile[local])[-16:]]
            values.append(float(10.0 * np.log10(np.sum(profile[strongest]) + WORKFLOW_EPS)))
    return values


def dynamics_features(audio: np.ndarray) -> dict[str, float]:
    values = np.asarray(waveform_events(audio), dtype=float)
    result = {name: float("nan") for name in D_FEATURES}
    if len(values) >= 8:
        q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
        result.update(dynamics_span=float(q90-q10), dynamics_iqr=float(q75-q25),
                      dynamics_adjacent_change=float(np.median(np.abs(np.diff(values)))))
    return result


def load_beats(path: Path, duration: float) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists() or not path.stat().st_size:
        return np.asarray([]), np.asarray([], dtype=int)
    data = np.loadtxt(path, ndmin=2)
    times = np.asarray(data[:, 0], float)
    numbers = np.asarray(data[:, 1], int) if data.shape[1] >= 2 else np.ones(len(data), dtype=int)
    keep = times <= duration + 1e-6
    return times[keep], numbers[keep]


def rhythm_features(times: np.ndarray) -> dict[str, float]:
    result = {name: float("nan") for name in R_FEATURES}
    if len(times) < 16:
        return result
    ibi = np.diff(np.asarray(times, dtype=float)); ibi = ibi[(ibi > 0) & np.isfinite(ibi)]
    if len(ibi) < 15 or np.median(ibi) <= 0:
        return result
    q25, q75 = np.quantile(ibi, [0.25, 0.75])
    result["ibi_cv"] = float((q75-q25)/np.median(ibi))
    bpm = 60.0 / ibi; bpm = bpm[(bpm >= 30.0) & (bpm <= 300.0)]
    if len(bpm) >= 14 and np.median(bpm) > 0:
        result["tempo_tv"] = float(np.median(np.abs(np.diff(bpm))) / np.median(bpm))
        counts, _ = np.histogram(bpm, bins=np.arange(30.0, 305.0, 5.0))
        probabilities = counts[counts > 0] / counts.sum()
        result["tempo_entropy"] = float(-(probabilities*np.log(probabilities)).sum()/np.log(len(counts)))
    return result


def clean_boundaries(values: np.ndarray, duration: float) -> np.ndarray:
    values = np.unique(np.clip(np.asarray(values, dtype=float), 0.0, duration))
    interior = values[(values >= 1.0) & (values <= duration - 1.0)]
    clean = [0.0]
    for value in interior:
        if value-clean[-1] >= 1.0:
            clean.append(float(value))
    if duration-clean[-1] < 1.0 and len(clean) > 1:
        clean.pop()
    return np.asarray([*clean, float(duration)])


def allinone_boundaries(path: Path, duration: float) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not segments:
        return np.asarray([0.0, duration])
    values = [float(x["start"]) for x in segments] + [float(segments[-1]["end"])]
    return clean_boundaries(np.asarray(values), duration)


def phrase_features(boundaries: np.ndarray, downbeats: np.ndarray) -> dict[str, float]:
    result = {name: float("nan") for name in P_FEATURES}
    durations = np.diff(np.asarray(boundaries, dtype=float)); durations = durations[durations > 0]
    if len(durations) >= 3:
        median = float(np.median(durations)); q25, q75 = np.quantile(durations, [0.25, 0.75])
        bins = np.arange(0.0, max(32.0, np.ceil(durations.max()/2.0)*2.0+2.0), 2.0)
        counts, _ = np.histogram(durations, bins=bins); probs = counts[counts > 0]/counts.sum()
        result.update(section_duration_median=median,
                      section_duration_cv=float((q75-q25)/median) if median > 0 else float("nan"),
                      section_duration_entropy=float(-(probs*np.log(probs)).sum()/np.log(max(len(counts), 2))))
    downbeats = np.asarray(downbeats, dtype=float)
    if len(downbeats) >= 4 and len(boundaries) >= 4:
        snapped = np.unique([int(np.argmin(np.abs(downbeats-b))) for b in boundaries])
        lengths = np.diff(snapped); lengths = lengths[lengths > 0]
        if len(lengths) >= 3:
            median = float(np.median(lengths)); q25, q75 = np.quantile(lengths, [0.25, 0.75])
            values, counts = np.unique(lengths, return_counts=True); mode = values[np.flatnonzero(counts == counts.max())[0]]
            result.update(section_bars_median=median,
                          section_bars_cv=float((q75-q25)/median) if median > 0 else float("nan"),
                          section_bars_offmode_fraction=float(np.mean(lengths != mode)))
    return result
