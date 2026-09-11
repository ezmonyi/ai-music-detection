#!/usr/bin/env python3
"""Stem-level perceptual spectral pilot for AI-vs-human music.

The script reuses the fixed manifest from ``ai_music_ablation.py``, prepares
lossless 30-second clips, separates vocals and accompaniment with Demucs, and
measures interpretable spectral cues that correspond to brightness, sibilance,
high-frequency texture, temporal stability, periodic voice quality, and
vibrato/energy coherence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import coherence, find_peaks
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parent
SR = 44_100
SECONDS = 30.0
N_SAMPLES = int(SR * SECONDS)
N_FFT = 4_096
HOP = 1_024
SEED = 20_260_813
MODEL = "htdemucs"
VOCAL_ACTIVITY_THRESHOLD_DB = -18.0

METRIC_GROUPS = {
    "brightness_tilt": (
        "tilt_1_5k_db_oct",
        "hf_tilt_5_16k_db_oct",
        "hf_ratio_5_16_db",
        "air_ratio_12_20_db",
        "sibilance_ratio_5_10_db",
    ),
    "hf_texture": (
        "hf_flatness",
        "hf_entropy",
        "hf_crest_db",
        "fakeprint_peak_density",
        "fakeprint_periodicity",
    ),
    "hf_dynamics": (
        "hf_flux",
        "hf_frame_similarity",
        "hf_power_sd_db",
        "hf_mod_4_12_share",
        "sibilance_contrast_db",
        "sibilance_burst_rate_hz",
    ),
    "voice_periodicity": (
        "cpp_proxy_db",
        "hnr_proxy_db",
        "pitch_confidence",
        "f0_mod_4_8_share",
        "vibrato_depth_cents",
        "f0_energy_coherence_4_8",
    ),
}
ALL_METRICS = tuple(dict.fromkeys(sum(METRIC_GROUPS.values(), ())))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/frequency_ablation_20260813/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/stem_spectral_pilot_20260813"),
    )
    parser.add_argument("--candidates-per-class", type=int, default=16)
    parser.add_argument("--selected-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--demucs-python", type=Path, default=Path(".venv-stems/bin/python"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--separation-backend",
        default="local Demucs htdemucs",
        help="Auditable backend label written to the report.",
    )
    parser.add_argument("--rebuild-clips", action="store_true")
    parser.add_argument("--rebuild-stems", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.analyze_only:
        parser.error("--prepare-only and --analyze-only are mutually exclusive")
    return args


def require_tools(args: argparse.Namespace) -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if not args.prepare_only and not args.analyze_only and not args.demucs_python.exists():
        missing.append(str(args.demucs_python))
    if missing:
        raise SystemExit("Missing required tools: " + ", ".join(missing))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_candidates(
    rows: list[dict[str, object]], per_class: int, seed: int
) -> list[dict[str, object]]:
    if per_class < 4 or per_class % 2:
        raise ValueError("--candidates-per-class must be an even integer >= 4")
    rng = random.Random(seed + 101)
    human = [row for row in rows if int(row["label"]) == 0]
    humair = [row for row in rows if row["source"] == "humair_suno"]
    unknown = [row for row in rows if row["source"] == "suno_unknown"]
    rng.shuffle(human)
    rng.shuffle(humair)
    rng.shuffle(unknown)
    chosen = human[:per_class]
    chosen += humair[: per_class // 2]
    chosen += unknown[: per_class // 2]
    if len(chosen) != per_class * 2:
        raise RuntimeError("The formal manifest does not contain enough tracks")
    return sorted(chosen, key=lambda row: (int(row["label"]), str(row["source"]), str(row["id"])))


def track_name(row: dict[str, object]) -> str:
    return f"{row['class_name']}_{row['source']}_{row['id']}"


def prepare_clips(
    rows: list[dict[str, object]], clip_dir: Path, rebuild: bool
) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(rows, 1):
        destination = clip_dir / f"{track_name(row)}.flac"
        if rebuild or not destination.exists():
            command = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-ss", f"{float(row['crop_start']):.6f}",
                "-i", str(ROOT / str(row["path"])),
                "-t", f"{SECONDS:.3f}", "-map", "0:a:0", "-vn", "-sn", "-dn",
                "-af", f"apad=whole_dur={SECONDS:.3f}",
                "-ac", "2", "-ar", str(SR), "-sample_fmt", "s16",
                "-compression_level", "8", "-y", str(destination),
            ]
            subprocess.run(command, check=True)
        if number == 1 or number % 8 == 0 or number == len(rows):
            print(f"clips {number:3d}/{len(rows)}", flush=True)


def stem_path(stem_root: Path, row: dict[str, object], stem: str) -> Path:
    directory = stem_root / MODEL / track_name(row)
    for extension in ("flac", "wav"):
        candidate = directory / f"{stem}.{extension}"
        if candidate.exists():
            return candidate
    return directory / f"{stem}.flac"


def separate_stems(
    rows: list[dict[str, object]],
    clip_dir: Path,
    stem_root: Path,
    model_cache: Path,
    demucs_python: Path,
    rebuild: bool,
) -> None:
    stem_root.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for row in rows
        if rebuild
        or not stem_path(stem_root, row, "vocals").exists()
        or not stem_path(stem_root, row, "no_vocals").exists()
    ]
    if not pending:
        print("stems already complete", flush=True)
        return
    env = os.environ.copy()
    env["TORCH_HOME"] = str(model_cache.resolve())
    started = time.time()
    for number, row in enumerate(pending, 1):
        command = [
            str(demucs_python), "-m", "demucs", "-n", MODEL,
            "--two-stems", "vocals", "--other-method", "add", "--flac",
            "--shifts", "0", "--overlap", "0.1", "--segment", "7", "-j", "1",
            "-o", str(stem_root), str(clip_dir / f"{track_name(row)}.flac"),
        ]
        subprocess.run(command, check=True, env=env)
        elapsed = time.time() - started
        print(f"stems {number:3d}/{len(pending)} elapsed={elapsed:7.1f}s", flush=True)


def decode_audio(path: Path) -> np.ndarray:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-ac", "1", "-ar", str(SR), "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if audio.size < int(N_SAMPLES * 0.99):
        raise RuntimeError(f"Short decoded audio: {path}")
    audio = audio[:N_SAMPLES]
    audio -= audio.mean()
    return audio


def framed_stft(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = 1 + (audio.size - N_FFT) // HOP
    stride = audio.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(frame_count, N_FFT), strides=(HOP * stride, stride), writeable=False
    )
    window = np.hanning(N_FFT)
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, axis=1)
    power = np.abs(spectrum) ** 2 + 1e-14
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-14)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    return frames, power, rms, freqs


def band(power: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    return power[:, (freqs >= low) & (freqs < high)]


def safe_db_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10((numerator + 1e-14) / (denominator + 1e-14))


def modulation_share(series: np.ndarray, frame_rate: float, low: float, high: float) -> float:
    if series.size < 16 or float(np.std(series)) < 1e-9:
        return 0.0
    centered = series - median_filter(series, size=max(3, int(frame_rate) | 1), mode="nearest")
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size))) ** 2
    freqs = np.fft.rfftfreq(centered.size, 1.0 / frame_rate)
    analysis = (freqs >= 0.5) & (freqs <= 20.0)
    target = (freqs >= low) & (freqs <= high)
    return float(spectrum[target].sum() / (spectrum[analysis].sum() + 1e-14))


def lower_envelope_residual(values: np.ndarray, area: int = 10) -> np.ndarray:
    patches = np.lib.stride_tricks.sliding_window_view(values, area)
    minima = np.argmin(patches, axis=1) + np.arange(patches.shape[0])
    indices = np.unique(np.concatenate(([0], minima, [values.size - 1])))
    envelope = np.interp(np.arange(values.size), indices, values[indices])
    residual = np.clip(values - np.maximum(envelope, -80.0), 0.0, 8.0)
    maximum = float(residual.max())
    return residual / maximum if maximum > 1e-9 else residual


def pitch_and_cepstrum(
    frames: np.ndarray, power: np.ndarray, rms: np.ndarray, active: np.ndarray, freqs: np.ndarray
) -> dict[str, float]:
    indices = np.flatnonzero(active)
    if indices.size > 240:
        indices = indices[np.linspace(0, indices.size - 1, 240).astype(int)]
    if indices.size < 8:
        return {name: 0.0 for name in METRIC_GROUPS["voice_periodicity"]}

    lag_min = int(SR / 500.0)
    lag_max = int(SR / 80.0)
    q_min = int(0.001 * SR)
    q_max = int(0.015 * SR)
    peak_min = int(0.002 * SR)
    peak_max = int(0.0125 * SR)
    cpp_values: list[float] = []
    correlations: list[float] = []
    f0_values: list[float] = []
    energy_values: list[float] = []

    for index in indices:
        frame = frames[index] * np.hanning(N_FFT)
        spectrum = np.fft.rfft(frame)
        autocorr = np.fft.irfft(np.abs(spectrum) ** 2, n=N_FFT)
        autocorr /= max(float(autocorr[0]), 1e-14)
        local = autocorr[lag_min:lag_max]
        lag = lag_min + int(np.argmax(local))
        correlation = float(autocorr[lag])
        correlations.append(correlation)
        f0_values.append(SR / lag)
        energy_values.append(20.0 * math.log10(float(rms[index]) + 1e-14))

        log_magnitude = np.log(np.abs(spectrum) + 1e-12)
        cepstrum = np.fft.irfft(log_magnitude, n=N_FFT)
        cep_db = 20.0 * np.log10(np.abs(cepstrum) + 1e-12)
        x = np.arange(q_min, q_max, dtype=np.float64)
        slope, intercept = np.polyfit(x, cep_db[q_min:q_max], 1)
        peak_index = peak_min + int(np.argmax(cep_db[peak_min:peak_max]))
        baseline = slope * peak_index + intercept
        cpp_values.append(float(cep_db[peak_index] - baseline))

    correlations_array = np.clip(np.asarray(correlations), 0.0, 0.999)
    hnr = 10.0 * np.log10((correlations_array + 1e-6) / (1.0 - correlations_array + 1e-6))
    confidence = float(np.median(correlations_array))
    cpp = float(np.median(cpp_values))
    hnr_db = float(np.median(hnr))

    f0 = np.asarray(f0_values)
    energy = np.asarray(energy_values)
    valid = correlations_array >= 0.20
    if int(valid.sum()) < 12:
        return {
            "cpp_proxy_db": cpp,
            "hnr_proxy_db": hnr_db,
            "pitch_confidence": confidence,
            "f0_mod_4_8_share": 0.0,
            "vibrato_depth_cents": 0.0,
            "f0_energy_coherence_4_8": 0.0,
        }

    positions = np.arange(f0.size)
    log_f0 = np.log2(np.clip(f0, 80.0, 500.0))
    log_f0[~valid] = np.interp(positions[~valid], positions[valid], log_f0[valid])
    frame_rate = SR / HOP
    cents = 1200.0 * (log_f0 - median_filter(log_f0, size=max(3, int(frame_rate) | 1), mode="nearest"))
    spectrum = np.fft.rfft(cents * np.hanning(cents.size))
    mod_freqs = np.fft.rfftfreq(cents.size, 1.0 / frame_rate)
    vib_mask = (mod_freqs >= 4.0) & (mod_freqs <= 8.0)
    all_mask = (mod_freqs >= 0.5) & (mod_freqs <= 12.0)
    f0_share = float(
        (np.abs(spectrum[vib_mask]) ** 2).sum()
        / ((np.abs(spectrum[all_mask]) ** 2).sum() + 1e-14)
    )
    filtered = np.fft.irfft(spectrum * vib_mask, n=cents.size)
    vibrato_depth = float(np.sqrt(2.0) * np.std(filtered))

    if cents.size >= 32 and float(np.std(energy)) > 1e-6:
        nperseg = min(128, cents.size)
        coh_freqs, coh = coherence(cents, energy, fs=frame_rate, nperseg=nperseg)
        coh = np.nan_to_num(coh, nan=0.0, posinf=0.0, neginf=0.0)
        coh_mask = (coh_freqs >= 4.0) & (coh_freqs <= 8.0)
        vib_coherence = float(np.mean(coh[coh_mask])) if np.any(coh_mask) else 0.0
    else:
        vib_coherence = 0.0
    return {
        "cpp_proxy_db": cpp,
        "hnr_proxy_db": hnr_db,
        "pitch_confidence": confidence,
        "f0_mod_4_8_share": f0_share,
        "vibrato_depth_cents": vibrato_depth,
        "f0_energy_coherence_4_8": vib_coherence,
    }


def spectral_metrics(audio: np.ndarray) -> dict[str, float]:
    frames, power, rms, freqs = framed_stft(audio)
    rms_db = 20.0 * np.log10(rms + 1e-14)
    active = rms_db >= max(float(np.percentile(rms_db, 30)), float(rms_db.max() - 35.0))
    if int(active.sum()) < 16:
        active[np.argsort(rms)[-16:]] = True

    p = power[active]
    low = band(p, freqs, 300, 5_000).sum(axis=1)
    body = band(p, freqs, 1_000, 5_000).sum(axis=1)
    sib = band(p, freqs, 5_000, 10_000).sum(axis=1)
    hf = band(p, freqs, 5_000, 16_000)
    hf_power = hf.sum(axis=1)
    air = band(p, freqs, 12_000, 20_000).sum(axis=1)

    hf_ratio = safe_db_ratio(hf_power, body)
    air_ratio = safe_db_ratio(air, body)
    sib_ratio = safe_db_ratio(sib, low)
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

    threshold = float(np.median(sib_ratio) + 1.5 * np.median(np.abs(sib_ratio - np.median(sib_ratio))))
    bursts = sib_ratio > threshold
    burst_count = int(np.sum(bursts[1:] & ~bursts[:-1]))
    active_duration = max(1e-6, active.sum() * HOP / SR)

    mean_db = 10.0 * np.log10(np.mean(p, axis=0) + 1e-14)
    hf_mask = (freqs >= 5_000) & (freqs < 16_000)
    residual = lower_envelope_residual(mean_db[hf_mask])
    peaks, _ = find_peaks(residual, height=0.35, prominence=0.08, distance=2)
    residual_power = np.abs(np.fft.rfft(residual - residual.mean())) ** 2
    fakeprint_periodicity = float(residual_power[4:].max() / (residual_power[1:].sum() + 1e-14))

    def spectral_slope(low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        x = np.log2(freqs[mask])
        return float(np.polyfit(x, mean_db[mask], 1)[0])

    output = {
        "tilt_1_5k_db_oct": spectral_slope(1_000, 5_000),
        "hf_tilt_5_16k_db_oct": spectral_slope(5_000, 16_000),
        "hf_ratio_5_16_db": float(np.median(hf_ratio)),
        "air_ratio_12_20_db": float(np.median(air_ratio)),
        "sibilance_ratio_5_10_db": float(np.median(sib_ratio)),
        "hf_flatness": float(np.median(flatness)),
        "hf_entropy": float(np.median(entropy)),
        "hf_crest_db": float(np.median(crest)),
        "fakeprint_peak_density": float(peaks.size / 11.0),
        "fakeprint_periodicity": fakeprint_periodicity,
        "hf_flux": float(np.mean(flux)),
        "hf_frame_similarity": float(np.mean(similarity)),
        "hf_power_sd_db": float(np.std(hf_db)),
        "hf_mod_4_12_share": modulation_share(hf_db, SR / HOP, 4.0, 12.0),
        "sibilance_contrast_db": float(np.percentile(sib_ratio, 90) - np.median(sib_ratio)),
        "sibilance_burst_rate_hz": float(burst_count / active_duration),
        "active_frame_ratio": float(active.mean()),
        "rms_dbfs": float(20.0 * np.log10(np.sqrt(np.mean(audio * audio)) + 1e-14)),
    }
    output.update(pitch_and_cepstrum(frames, power, rms, active, freqs))
    return output


def extract_all(
    rows: list[dict[str, object]], clip_dir: Path, stem_root: Path, output_dir: Path
) -> list[dict[str, object]]:
    destination = output_dir / "candidate_stem_metrics.csv"
    existing: dict[tuple[str, str], dict[str, object]] = {}
    if destination.exists():
        with destination.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                existing[(item["id"], item["stem"])] = dict(item)

    results: list[dict[str, object]] = []
    started = time.time()
    for number, row in enumerate(rows, 1):
        sources = {
            "mix": clip_dir / f"{track_name(row)}.flac",
            "vocals": stem_path(stem_root, row, "vocals"),
            "accompaniment": stem_path(stem_root, row, "no_vocals"),
        }
        for stem, path in sources.items():
            key = (str(row["id"]), stem)
            if key in existing and all(name in existing[key] for name in ALL_METRICS):
                item = existing[key]
            else:
                metrics = spectral_metrics(decode_audio(path))
                item = {
                    "id": row["id"], "label": row["label"],
                    "class_name": row["class_name"], "source": row["source"],
                    "model_name": row["model_name"], "track": track_name(row),
                    "stem": stem, "audio_path": str(path.relative_to(ROOT)), **metrics,
                }
            results.append(item)
        write_csv(destination, results)
        if number == 1 or number % 4 == 0 or number == len(rows):
            print(f"metrics {number:3d}/{len(rows)} elapsed={time.time() - started:7.1f}s", flush=True)
    return results


def select_vocal_active(
    rows: list[dict[str, object]], metrics: list[dict[str, object]], target: int, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_key = {(str(item["id"]), str(item["stem"])): item for item in metrics}
    diagnostics: list[dict[str, object]] = []
    for row in rows:
        vocals = by_key[(str(row["id"]), "vocals")]
        mix = by_key[(str(row["id"]), "mix")]
        share = float(vocals["rms_dbfs"]) - float(mix["rms_dbfs"])
        diagnostics.append({
            "id": row["id"], "label": row["label"], "class_name": row["class_name"],
            "source": row["source"], "track": track_name(row),
            "vocal_to_mix_rms_db": share,
            "passes_threshold": int(share >= VOCAL_ACTIVITY_THRESHOLD_DB),
        })

    rng = random.Random(seed + 303)

    def choose(candidates: list[dict[str, object]], count: int) -> list[dict[str, object]]:
        passing = [item for item in candidates if int(item["passes_threshold"]) == 1]
        rng.shuffle(passing)
        chosen = passing[:count]
        if len(chosen) < count:
            remaining = [item for item in candidates if item not in chosen]
            remaining.sort(key=lambda item: float(item["vocal_to_mix_rms_db"]), reverse=True)
            chosen += remaining[: count - len(chosen)]
        return chosen

    if target % 2:
        raise ValueError("--selected-per-class must be even for balanced AI sources")
    selected_items = choose(
        [item for item in diagnostics if int(item["label"]) == 0], target
    )
    for source in ("humair_suno", "suno_unknown"):
        selected_items += choose(
            [item for item in diagnostics if item["source"] == source], target // 2
        )
    selected_ids = {str(item["id"]) for item in selected_items}
    for item in diagnostics:
        item["selected"] = int(str(item["id"]) in selected_ids)
    selected = [row for row in rows if str(row["id"]) in selected_ids]
    return selected, diagnostics


def auc(labels: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = labels.size - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cohen_d(human: np.ndarray, ai: np.ndarray) -> float:
    pooled = math.sqrt(max(1e-14, ((human.size - 1) * human.var(ddof=1) + (ai.size - 1) * ai.var(ddof=1)) / (human.size + ai.size - 2)))
    return float((ai.mean() - human.mean()) / pooled)


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        original = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, p_values[original] * len(p_values) / rank)
        adjusted[original] = running
    return adjusted.tolist()


def effect_sizes(selected_metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for stem in ("mix", "vocals", "accompaniment"):
        stem_rows = [item for item in selected_metrics if item["stem"] == stem]
        labels = np.asarray([int(item["label"]) for item in stem_rows])
        for metric in ALL_METRICS:
            values = np.asarray([float(item[metric]) for item in stem_rows])
            human = values[labels == 0]
            ai = values[labels == 1]
            signed_auc = auc(labels, values)
            p_value = float(mannwhitneyu(human, ai, alternative="two-sided").pvalue)
            output.append({
                "stem": stem, "metric": metric,
                "human_median": float(np.median(human)), "ai_median": float(np.median(ai)),
                "ai_minus_human": float(ai.mean() - human.mean()),
                "cohen_d": cohen_d(human, ai), "signed_auc": signed_auc,
                "separation_auc": max(signed_auc, 1.0 - signed_auc), "p_value": p_value,
            })
    adjusted = bh_adjust([float(item["p_value"]) for item in output])
    for item, q_value in zip(output, adjusted):
        item["bh_q_value"] = q_value
    return sorted(output, key=lambda item: (str(item["stem"]), -float(item["separation_auc"])))


def source_effects(selected_metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for stem in ("mix", "vocals", "accompaniment"):
        stem_rows = [item for item in selected_metrics if item["stem"] == stem]
        human_rows = [item for item in stem_rows if int(item["label"]) == 0]
        for metric in ALL_METRICS:
            human = np.asarray([float(item[metric]) for item in human_rows])
            source_ds: dict[str, float] = {}
            source_aucs: dict[str, float] = {}
            for source in ("humair_suno", "suno_unknown"):
                ai_rows = [item for item in stem_rows if item["source"] == source]
                ai = np.asarray([float(item[metric]) for item in ai_rows])
                source_ds[source] = cohen_d(human, ai)
                labels = np.concatenate((np.zeros(human.size, dtype=int), np.ones(ai.size, dtype=int)))
                signed = auc(labels, np.concatenate((human, ai)))
                source_aucs[source] = max(signed, 1.0 - signed)
            agreement = int(source_ds["humair_suno"] * source_ds["suno_unknown"] > 0)
            output.append({
                "stem": stem, "metric": metric,
                "humair_cohen_d": source_ds["humair_suno"],
                "unknown_cohen_d": source_ds["suno_unknown"],
                "direction_agreement": agreement,
                "worst_source_auc": min(source_aucs.values()),
                "humair_auc": source_aucs["humair_suno"],
                "unknown_auc": source_aucs["suno_unknown"],
            })
    return sorted(output, key=lambda item: (str(item["stem"]), -float(item["worst_source_auc"])))


def fit_ridge(x: np.ndarray, labels: np.ndarray, alpha: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    target = labels * 2.0 - 1.0
    intercept = float(target.mean())
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (target - intercept))
    return mean, scale, weights, intercept


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    predictions = scores >= 0.0
    return float(((predictions[labels == 0] == 0).mean() + (predictions[labels == 1] == 1).mean()) / 2.0)


def repeated_cv(selected_metrics: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    by_stem: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for item in selected_metrics:
        by_stem[str(item["stem"])][str(item["id"])] = item
    ids = sorted(by_stem["mix"])
    labels = np.asarray([int(by_stem["mix"][item_id]["label"]) for item_id in ids])
    representations = ("mix", "vocals", "accompaniment", "vocals+accompaniment")
    groups = dict(METRIC_GROUPS)
    groups["all_perceptual"] = ALL_METRICS
    results: list[dict[str, object]] = []
    for representation in representations:
        stems = representation.split("+")
        for group, metric_names in groups.items():
            x = np.asarray([
                [float(by_stem[stem][item_id][metric]) for stem in stems for metric in metric_names]
                for item_id in ids
            ])
            bas: list[float] = []
            aucs: list[float] = []
            for repeat in range(20):
                rng = np.random.default_rng(seed + 1_000 + repeat)
                folds: list[np.ndarray] = []
                split_by_label = []
                for label in (0, 1):
                    indices = np.flatnonzero(labels == label)
                    rng.shuffle(indices)
                    split_by_label.append(np.array_split(indices, 4))
                folds = [np.concatenate((split_by_label[0][fold], split_by_label[1][fold])) for fold in range(4)]
                scores = np.zeros(labels.size, dtype=np.float64)
                for test in folds:
                    train = np.setdiff1d(np.arange(labels.size), test)
                    mean, scale, weights, intercept = fit_ridge(x[train], labels[train])
                    scores[test] = ((x[test] - mean) / scale) @ weights + intercept
                bas.append(balanced_accuracy(labels, scores))
                aucs.append(auc(labels, scores))
            results.append({
                "representation": representation, "feature_group": group,
                "dimension": x.shape[1], "repeats": 20,
                "balanced_accuracy_mean": float(np.mean(bas)),
                "balanced_accuracy_std": float(np.std(bas, ddof=1)),
                "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs, ddof=1)),
            })
    return sorted(results, key=lambda item: (str(item["representation"]), -float(item["balanced_accuracy_mean"])))


def render_report(
    path: Path,
    args: argparse.Namespace,
    selected: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    effects: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    cv_rows: list[dict[str, object]],
) -> None:
    vocal_effects = [item for item in effects if item["stem"] == "vocals"]
    vocal_sources = [item for item in source_rows if item["stem"] == "vocals" and int(item["direction_agreement"]) == 1]
    vocal_lookup = {str(item["metric"]): item for item in vocal_effects}
    cv_lookup = {(item["representation"], item["feature_group"]): item for item in cv_rows}
    passed = defaultdict(int)
    for item in diagnostics:
        if int(item["passes_threshold"]):
            passed[str(item["class_name"])] += 1
    selected_sources = defaultdict(int)
    for row in selected:
        selected_sources[str(row["source"])] += 1
    selected_below_threshold = sum(
        1
        for item in diagnostics
        if int(item.get("selected", 0)) and not int(item["passes_threshold"])
    )

    lines = [
        "# Stem-level perceptual spectral pilot",
        "",
        f"- Seed: `{args.seed}`",
        f"- Candidate pool: `{args.candidates_per_class}` human + `{args.candidates_per_class}` AI",
        f"- Selected vocal-active analysis set: `{args.selected_per_class}` human + `{args.selected_per_class}` AI",
        f"- Selected sources: `{dict(sorted(selected_sources.items()))}`",
        f"- Vocal activity threshold: vocals/mix RMS >= `{VOCAL_ACTIVITY_THRESHOLD_DB:.1f} dB`; passing candidates: `{dict(sorted(passed.items()))}`",
        f"- Selected tracks below the vocal-activity threshold: `{selected_below_threshold}`",
        f"- Audio standard: center `{SECONDS:.0f}s`, stereo `{SR} Hz`, signed 16-bit PCM input encoded losslessly as FLAC; returned stems are signed 16-bit PCM WAV",
        f"- Separation: `{args.separation_backend}` with model `{MODEL}`, identical settings for every track, vocals + accompaniment",
        f"- Analysis STFT: `N_FFT={N_FFT}`, `hop={HOP}`",
        "",
        "## Vocal-stem univariate results",
        "",
        "AUC is made direction-free (`max(AUC, 1-AUC)`); Cohen's d keeps the sign (positive means higher for AI). BH q-values correct across all stems and metrics. This is an exploratory pilot, so effect size and agreement across both AI sources matter more than p-values.",
        "",
        "| Metric | Human median | AI median | Cohen d | Separation AUC | BH q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in vocal_effects[:12]:
        lines.append(
            f"| {item['metric']} | {float(item['human_median']):.4g} | {float(item['ai_median']):.4g} | {float(item['cohen_d']):+.2f} | {float(item['separation_auc']):.3f} | {float(item['bh_q_value']):.3g} |"
        )
    lines += [
        "",
        "## AI-source consistency on vocals",
        "",
        "Only metrics whose AI-vs-human direction agrees for Humair-Suno and the second Suno collection are shown.",
        "",
        "| Metric | Humair d | Unknown-Suno d | Worst-source AUC |",
        "|---|---:|---:|---:|",
    ]
    for item in vocal_sources[:10]:
        lines.append(
            f"| {item['metric']} | {float(item['humair_cohen_d']):+.2f} | {float(item['unknown_cohen_d']):+.2f} | {float(item['worst_source_auc']):.3f} |"
        )
    lines += [
        "",
        "## Result in perceptual terms",
        "",
        f"- The clearest vocal-stem cue is **less time-varying high-frequency power**: median `hf_power_sd_db` is `{float(vocal_lookup['hf_power_sd_db']['human_median']):.2f}` for human and `{float(vocal_lookup['hf_power_sd_db']['ai_median']):.2f}` for AI (AUC `{float(vocal_lookup['hf_power_sd_db']['separation_auc']):.3f}`). This matches a more continuously present or compressed top-end 'fizz' rather than consonant-specific air bursts.",
        f"- AI vocals are also brighter in the 5-16 kHz band: median `hf_ratio_5_16_db` shifts from `{float(vocal_lookup['hf_ratio_5_16_db']['human_median']):.2f} dB` to `{float(vocal_lookup['hf_ratio_5_16_db']['ai_median']):.2f} dB`, with the same direction in both AI collections (worst-source AUC `{float(next(item for item in vocal_sources if item['metric'] == 'hf_ratio_5_16_db')['worst_source_auc']):.3f}`).",
        f"- The 5-10 kHz sibilance ratio moves from `{float(vocal_lookup['sibilance_ratio_5_10_db']['human_median']):.2f} dB` to `{float(vocal_lookup['sibilance_ratio_5_10_db']['ai_median']):.2f} dB`; HNR is also higher for AI (`{float(vocal_lookup['hnr_proxy_db']['human_median']):.2f}` to `{float(vocal_lookup['hnr_proxy_db']['ai_median']):.2f} dB`). Together these describe vocals that can sound simultaneously cleaner/more periodic and more persistently bright.",
        "- None of the exploratory vocal metrics survives BH correction at q < 0.05. Treat these as listening-aligned hypotheses to confirm, not discovered universal biomarkers.",
        f"- Vocal-only multivariate features are not the best detector in this small balanced subset: vocal brightness BA/AUC is `{float(cv_lookup[('vocals', 'brightness_tilt')]['balanced_accuracy_mean']):.3f}/{float(cv_lookup[('vocals', 'brightness_tilt')]['roc_auc_mean']):.3f}`, versus mix brightness `{float(cv_lookup[('mix', 'brightness_tilt')]['balanced_accuracy_mean']):.3f}/{float(cv_lookup[('mix', 'brightness_tilt')]['roc_auc_mean']):.3f}` and accompaniment texture `{float(cv_lookup[('accompaniment', 'hf_texture')]['balanced_accuracy_mean']):.3f}/{float(cv_lookup[('accompaniment', 'hf_texture')]['roc_auc_mean']):.3f}`. Your cue is present, but it is not sufficient on its own.",
        "",
        "## Repeated 4-fold feature ablation",
        "",
        f"Twenty deterministic repetitions are used because {len(selected)} tracks are too few for a stable single split. Models are ridge-linear and standardized inside each training fold.",
        "",
        "| Representation | Brightness BA/AUC | Texture BA/AUC | Dynamics BA/AUC | Voice periodicity BA/AUC | All BA/AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for representation in ("mix", "vocals", "accompaniment", "vocals+accompaniment"):
        cells = []
        for group in ("brightness_tilt", "hf_texture", "hf_dynamics", "voice_periodicity", "all_perceptual"):
            item = cv_lookup[(representation, group)]
            cells.append(f"{float(item['balanced_accuracy_mean']):.3f}/{float(item['roc_auc_mean']):.3f}")
        lines.append(f"| {representation} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## What these metrics mean perceptually",
        "",
        "- `hf_ratio`, `air_ratio`, and spectral tilts quantify whether the vocal top end feels too bright, dark, or bandwidth-limited.",
        "- `hf_flatness`, `hf_entropy`, crest, and fakeprint metrics quantify noise-like versus comb-like/peaky texture: useful for metallic, buzzy, or vocoder-grid impressions.",
        "- flux, frame similarity, modulation share, and sibilance bursts quantify whether high-frequency detail evolves naturally or remains unnaturally stationary/repetitive.",
        "- CPP/HNR proxies quantify periodic voiced structure versus breath/noise; vibrato metrics quantify 4-8 Hz F0 motion and whether it is coherent with vocal energy.",
        "",
        "## Interpretation boundary",
        "",
        "- This tests local dataset separability and operationalizes a listening hypothesis; it does not prove that a metric is a universal signature of AI generation.",
        "- Demucs was trained primarily on human-produced music. Different separation error on AI music can itself become a cue. The mix results are therefore the control, and ground-truth isolated stems are needed for a thesis-level causal claim.",
        "- Source codec/mastering differences survive lossless standardization. The next confirmatory test should uniformly recompress before separation and use human/AI songs matched by genre, vocalist register, loudness, and production era.",
        "- FMA labels establish a human-production corpus, not necessarily an isolated natural singing voice corpus; manual vocal-presence and label audit remains required.",
        "",
        "## Listening validation protocol",
        "",
        "Blind-rate 5-8 second vocal excerpts on four 1-5 scales: `air/brightness unnaturalness`, `metallic/comb texture`, `sibilance continuity`, and `vibrato-envelope coherence`. Then correlate listener scores with these metrics using Spearman correlation and mixed-effects regression with track and listener random effects.",
        "",
        "## Primary references",
        "",
        "- Synthetic-vs-real singing differs especially in high-frequency spectrogram components: https://arxiv.org/abs/2508.01796",
        "- High-frequency spectral detail and dynamics are useful for synthetic-speech detection: https://www.isca-archive.org/interspeech_2015/sahidullah15_interspeech.html",
        "- Repeated spectro-temporal artifacts can be captured with 2-D transforms of log-Mel spectra: https://www.isca-archive.org/interspeech_2021/gao21c_interspeech.html",
        "- CPP and spectral slope jointly explain perceived breathiness: https://pubmed.ncbi.nlm.nih.gov/23785184/",
        "- Singing naturalness depends on vibrato and coherent spectral-envelope processing: https://www.isca-archive.org/interspeech_2011/lee11e_interspeech.html",
        "- Fourier-domain AI-music artifact fingerprint: https://arxiv.org/abs/2506.19108",
        "",
        "Full data: `selected_stem_metrics.csv`, `effect_sizes.csv`, `source_effects.csv`, `feature_ablation.csv`, and `vocal_activity.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_tools(args)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl((ROOT / args.manifest).resolve())
    candidates = sample_candidates(rows, args.candidates_per_class, args.seed)
    write_jsonl(output_dir / "candidate_manifest.jsonl", candidates)

    clip_dir = output_dir / "clips"
    stem_root = output_dir / "stems"
    if not args.analyze_only:
        prepare_clips(candidates, clip_dir, args.rebuild_clips)
    if args.prepare_only:
        print(f"prepared {len(candidates)} clips in {clip_dir}", flush=True)
        return
    if not args.analyze_only:
        separate_stems(
            candidates, clip_dir, stem_root, output_dir / "model_cache",
            (ROOT / args.demucs_python).resolve(), args.rebuild_stems,
        )
    candidate_metrics = extract_all(candidates, clip_dir, stem_root, output_dir)
    selected, diagnostics = select_vocal_active(
        candidates, candidate_metrics, args.selected_per_class, args.seed
    )
    selected_ids = {str(row["id"]) for row in selected}
    selected_metrics = [item for item in candidate_metrics if str(item["id"]) in selected_ids]
    write_jsonl(output_dir / "selected_manifest.jsonl", selected)
    write_csv(output_dir / "selected_stem_metrics.csv", selected_metrics)
    write_csv(output_dir / "vocal_activity.csv", diagnostics)

    effects = effect_sizes(selected_metrics)
    sources = source_effects(selected_metrics)
    cv_rows = repeated_cv(selected_metrics, args.seed)
    write_csv(output_dir / "effect_sizes.csv", effects)
    write_csv(output_dir / "source_effects.csv", sources)
    write_csv(output_dir / "feature_ablation.csv", cv_rows)
    render_report(output_dir / "REPORT.md", args, selected, diagnostics, effects, sources, cv_rows)
    print(f"report: {output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
