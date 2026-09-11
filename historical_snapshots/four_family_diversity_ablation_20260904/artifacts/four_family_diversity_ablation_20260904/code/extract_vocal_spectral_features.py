#!/usr/bin/env python3
"""Extract the frozen high-frequency Demucs-vocal feature family for 10 s tracks.

The MUSDB-derived average Demucs response is removed directly from STFT power.
Frame selection is frozen from the raw vocal RMS so the correction cannot alter
which frames are analysed.  The JSONL state file makes the run resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


SR = 44_100
N_SAMPLES = 441_000
N_FFT = 4_096
HOP = 1_024
FEATURES = (
    "tilt_1_5k_db_oct",
    "hf_tilt_5_16k_db_oct",
    "hf_ratio_5_16_db",
    "air_ratio_12_20_db",
    "sibilance_ratio_5_10_db",
    "hf_flatness",
    "hf_entropy",
    "hf_crest_db",
    "fakeprint_peak_density",
    "fakeprint_periodicity",
    "hf_flux",
    "hf_frame_similarity",
    "hf_power_sd_db",
    "hf_mod_4_12_share",
    "sibilance_contrast_db",
    "sibilance_burst_rate_hz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--stems", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_cohort(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tracks = [row["track"] for row in rows]
    if len(tracks) != len(set(tracks)):
        raise RuntimeError("Cohort track ids are not unique")
    return rows


def load_bias(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        frequencies = np.asarray(payload["frequencies_hz"], dtype=np.float64)
        bias_db = np.asarray(payload["selected_bias_db"], dtype=np.float64)
    expected = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    if frequencies.shape != expected.shape or not np.allclose(frequencies, expected):
        raise RuntimeError("Demucs bias frequency grid does not match the frozen STFT")
    if not np.all(np.isfinite(bias_db)):
        raise RuntimeError("Demucs bias contains non-finite values")
    return bias_db


def decode(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if sample_rate != SR:
        raise RuntimeError(f"Unexpected sample rate {sample_rate}: {path}")
    mono = audio.mean(axis=1)
    if mono.size < int(N_SAMPLES * 0.99):
        raise RuntimeError(f"Short vocal stem ({mono.size} samples): {path}")
    if mono.size < N_SAMPLES:
        mono = np.pad(mono, (0, N_SAMPLES - mono.size))
    mono = mono[:N_SAMPLES]
    mono -= mono.mean()
    return mono


def framed_power(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = 1 + (audio.size - N_FFT) // HOP
    stride = audio.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_count, N_FFT),
        strides=(HOP * stride, stride),
        writeable=False,
    )
    spectrum = np.fft.rfft(frames * np.hanning(N_FFT), axis=1)
    power = np.abs(spectrum) ** 2 + 1e-14
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-14)
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    return power, rms, frequencies


def band(power: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    return power[:, (frequencies >= low) & (frequencies < high)]


def db_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10((numerator + 1e-14) / (denominator + 1e-14))


def modulation_share(series: np.ndarray, low: float, high: float) -> float:
    frame_rate = SR / HOP
    if series.size < 16 or float(np.std(series)) < 1e-9:
        return 0.0
    size = max(3, int(frame_rate) | 1)
    centered = series - median_filter(series, size=size, mode="nearest")
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size))) ** 2
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / frame_rate)
    analysis = (frequencies >= 0.5) & (frequencies <= 20.0)
    target = (frequencies >= low) & (frequencies <= high)
    return float(spectrum[target].sum() / (spectrum[analysis].sum() + 1e-14))


def lower_envelope_residual(values: np.ndarray, area: int = 10) -> np.ndarray:
    patches = np.lib.stride_tricks.sliding_window_view(values, area)
    minima = np.argmin(patches, axis=1) + np.arange(patches.shape[0])
    indices = np.unique(np.concatenate(([0], minima, [values.size - 1])))
    envelope = np.interp(np.arange(values.size), indices, values[indices])
    residual = np.clip(values - np.maximum(envelope, -80.0), 0.0, 8.0)
    maximum = float(residual.max())
    return residual / maximum if maximum > 1e-9 else residual


def extract_metrics(audio: np.ndarray, bias_db: np.ndarray) -> dict[str, float]:
    raw_power, rms, frequencies = framed_power(audio)
    rms_db = 20.0 * np.log10(rms + 1e-14)
    active = rms_db >= max(float(np.percentile(rms_db, 30)), float(rms_db.max() - 35.0))
    if int(active.sum()) < 16:
        active[np.argsort(rms)[-16:]] = True

    # Magnitude multiplication by 10**(-bias/20) is exactly power
    # multiplication by 10**(-bias/10).  No phase-dependent metric is used.
    power = raw_power * np.power(10.0, -bias_db / 10.0)[None, :]
    p = power[active]
    low = band(p, frequencies, 300, 5_000).sum(axis=1)
    body = band(p, frequencies, 1_000, 5_000).sum(axis=1)
    sib = band(p, frequencies, 5_000, 10_000).sum(axis=1)
    hf = band(p, frequencies, 5_000, 16_000)
    hf_power = hf.sum(axis=1)
    air = band(p, frequencies, 12_000, 20_000).sum(axis=1)

    hf_ratio = db_ratio(hf_power, body)
    air_ratio = db_ratio(air, body)
    sib_ratio = db_ratio(sib, low)
    flatness = np.exp(np.mean(np.log(hf + 1e-14), axis=1)) / (np.mean(hf, axis=1) + 1e-14)
    normalized_hf = hf / (hf.sum(axis=1, keepdims=True) + 1e-14)
    entropy = -np.sum(normalized_hf * np.log(normalized_hf + 1e-14), axis=1) / math.log(hf.shape[1])
    crest = 10.0 * np.log10((hf.max(axis=1) + 1e-14) / (hf.mean(axis=1) + 1e-14))

    log_hf = np.log(hf + 1e-14)
    centered = log_hf - log_hf.mean(axis=1, keepdims=True)
    normalized = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-14)
    similarity = np.sum(normalized[1:] * normalized[:-1], axis=1)
    flux = 1.0 - similarity
    hf_db = 10.0 * np.log10(hf_power + 1e-14)

    middle = float(np.median(sib_ratio))
    threshold = middle + 1.5 * float(np.median(np.abs(sib_ratio - middle)))
    bursts = sib_ratio > threshold
    burst_count = int(np.sum(bursts[1:] & ~bursts[:-1]))
    active_duration = max(1e-6, active.sum() * HOP / SR)

    mean_db = 10.0 * np.log10(np.mean(p, axis=0) + 1e-14)
    hf_mask = (frequencies >= 5_000) & (frequencies < 16_000)
    residual = lower_envelope_residual(mean_db[hf_mask])
    peaks, _ = find_peaks(residual, height=0.35, prominence=0.08, distance=2)
    residual_power = np.abs(np.fft.rfft(residual - residual.mean())) ** 2
    periodicity = float(residual_power[4:].max() / (residual_power[1:].sum() + 1e-14))

    def slope(low_hz: float, high_hz: float) -> float:
        mask = (frequencies >= low_hz) & (frequencies <= high_hz)
        return float(np.polyfit(np.log2(frequencies[mask]), mean_db[mask], 1)[0])

    metrics = {
        "tilt_1_5k_db_oct": slope(1_000, 5_000),
        "hf_tilt_5_16k_db_oct": slope(5_000, 16_000),
        "hf_ratio_5_16_db": float(np.median(hf_ratio)),
        "air_ratio_12_20_db": float(np.median(air_ratio)),
        "sibilance_ratio_5_10_db": float(np.median(sib_ratio)),
        "hf_flatness": float(np.median(flatness)),
        "hf_entropy": float(np.median(entropy)),
        "hf_crest_db": float(np.median(crest)),
        "fakeprint_peak_density": float(peaks.size / 11.0),
        "fakeprint_periodicity": periodicity,
        "hf_flux": float(np.mean(flux)),
        "hf_frame_similarity": float(np.mean(similarity)),
        "hf_power_sd_db": float(np.std(hf_db)),
        "hf_mod_4_12_share": modulation_share(hf_db, 4.0, 12.0),
        "sibilance_contrast_db": float(np.percentile(sib_ratio, 90) - np.median(sib_ratio)),
        "sibilance_burst_rate_hz": float(burst_count / active_duration),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Non-finite high-frequency metric")
    return {
        **{f"spectral__{name}": value for name, value in metrics.items()},
        "spectral_vocal_rms_dbfs": float(20.0 * np.log10(np.sqrt(np.mean(audio * audio)) + 1e-14)),
        "spectral_active_frame_ratio": float(active.mean()),
        "spectral_active_frame_count": int(active.sum()),
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_cohort(args.cohort)
    if args.limit is not None:
        rows = rows[: args.limit]
    bias_db = load_bias(args.bias)
    state_path = args.output_dir / "spectral_features.jsonl"
    existing: dict[str, dict[str, object]] = {}
    if state_path.exists():
        with state_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    existing[str(item["track"])] = item
    pending = [row for row in rows if row["track"] not in existing]

    def process(row: dict[str, str]) -> dict[str, object]:
        path = args.stems / "htdemucs" / row["track"] / "vocals.wav"
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        return {"track": row["track"], **extract_metrics(decode(path), bias_db)}

    started = time.time()
    with state_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, item in enumerate(executor.map(process, pending), 1):
                existing[str(item["track"])] = item
                output.write(json.dumps(item, sort_keys=True) + "\n")
                output.flush()
                if index == 1 or index % 100 == 0 or index == len(pending):
                    print(
                        f"spectral {len(existing)}/{len(rows)} "
                        f"new={index}/{len(pending)} elapsed={time.time()-started:.1f}s",
                        flush=True,
                    )

    ordered = [existing[row["track"]] for row in rows]
    csv_path = args.output_dir / "spectral_features.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    metadata = {
        "tracks": len(ordered),
        "sample_rate_hz": SR,
        "duration_seconds": 10.0,
        "n_fft": N_FFT,
        "hop": HOP,
        "feature_names": list(FEATURES),
        "feature_prefix": "spectral__",
        "correction": "power *= 10**(-MUSDB_oracle_Demucs_bias_db/10)",
        "frame_selection": "raw vocal RMS: >= max(P30, max-35 dB), minimum 16 frames",
        "phase_dependent_metrics": False,
        "state_file": str(state_path.resolve()),
        "pid": os.getpid(),
    }
    (args.output_dir / "spectral_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
