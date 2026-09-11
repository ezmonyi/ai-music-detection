"""Exact-crop, read-only audio loading and auditable input fingerprints."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

CONFIG = {"target_sr": 16000, "mono": "channel_arithmetic_mean", "dc": "global_mean_subtraction", "resampler": "scipy.signal.resample_poly", "window": ["kaiser", 5.0], "padtype": "constant", "padding_short_inputs": False}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()

def load_exact(row, duration=30.0, target_sr=16000):
    path = Path(row.get("audio_path") or row.get("standardized_path") or row.get("source_audio_path") or "")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    offset = float(row.get("audio_offset_s") or 0)
    if offset < 0 or not math.isfinite(offset) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid crop offset/duration")
    if not isinstance(target_sr, int) or target_sr <= 0:
        raise ValueError("target_sr must be a positive integer")
    with sf.SoundFile(path) as f:
        rate, channels, total = f.samplerate, f.channels, f.frames
        start, count = int(round(offset * rate)), int(round(duration * rate))
        if start + count > total:
            raise ValueError(f"Short input: {path}: {total} < {start}+{count}")
        f.seek(start)
        audio = f.read(count, dtype="float64", always_2d=True)
    if not np.isfinite(audio).all() or len(audio) != count:
        raise ValueError("Nonfinite/incomplete decoded audio")
    mono = audio.mean(axis=1)
    mono -= mono.mean()
    divisor = math.gcd(rate, target_sr)
    if rate != target_sr:
        mono = resample_poly(mono, target_sr // divisor, rate // divisor, window=("kaiser", 5.0), padtype="constant")
    expected = int(round(duration * target_sr))
    if len(mono) != expected:
        raise ValueError(f"Resampled length mismatch: {len(mono)} vs {expected}")
    # Fix byte order/dtype before hashing; values passed to extractors are identical.
    mono = np.ascontiguousarray(mono, dtype="<f8")
    audit = {"source_audio_path": str(path), "source_audio_sha256": sha256_file(path), "source_sample_rate": rate, "source_channels": channels, "source_total_frames": total, "crop_start_frame": start, "crop_frames": count, "analysis_sr": target_sr, "analysis_frames": expected, "analysis_waveform_sha256": hashlib.sha256(mono.tobytes()).hexdigest(), "analysis_rms": float(np.sqrt(np.mean(mono ** 2)))}
    return mono, audit
