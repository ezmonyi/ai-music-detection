"""Independent read-only numerical/integrity audit for the Native30 BC cohort.

The auditor is intentionally terminal: it accepts only a caller-pinned parent
freeze and a complete caller-pinned producer COMMIT.  It never invokes the
producer extractor, primitive, scalar reducer, or resampling implementation.
Scientific nulls are retained and checked; producer failures, incomplete
inventories, changed inputs, and changed authorities fail closed and do not
publish a success receipt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys

import numpy as np
import scipy
from scipy.signal import upfirdn
from scipy.special import i0
import soundfile as sf


VERSION = "audit_native30_bc_v1"
PRODUCER_VERSION = "run_native30_bc_v2"
FEATURE = "BC_b2_500_750_1250hz_center8s_median"
FREEZE_VERSION = "native30-bc-parent-freeze-v1"
EXPECTED_COUNT = 3830
SOURCE_COUNTS = {
    "ACE-Step": 400,
    "FMA": 354,
    "HeartMuLa": 366,
    "MTG-Jamendo": 500,
    "Mureka_v9": 500,
    "Suno": 400,
    "Udio": 500,
    "human_maestro_v3": 300,
    "human_medleydb": 168,
    "human_moisesdb": 239,
    "human_saraga_hindustani_v1": 103,
}
LABEL_COUNTS = {"0": 1664, "1": 2166}
ORIGIN_COUNTS = {"new": 1656, "prior": 2174}
INPUT_FORMAT = {
    "format": "WAV", "subtype": "FLOAT", "sample_rate_hz": 44100,
    "channels": 2, "frames": 1323000,
}
VIEW = {
    "channel_policy": "promote_float64_then_arithmetic_mean",
    "global_dc_subtraction": False,
    "source_resampled_samples": 480000,
    "resample_up": 160,
    "resample_down": 441,
    "resample_window": ["kaiser", 5.0],
    "resample_padtype": "constant",
    "resample_cval": 0.0,
    "resample_scope": "whole_native30_before_crop",
    "analysis_sample_rate_hz": 16000,
    "crop_start_sample": 176000,
    "crop_stop_sample_exclusive": 304000,
    "condition_local_start_sample": 0,
    "condition_local_stop_sample_exclusive": 128000,
    "baseline_gain": 0.25,
    "pool_samples": 64000,
    "pool_count": 2,
    "discarded_tail_samples": 0,
    "analysis_duration_s": 8,
    "cohort_duration_s": 30,
    "normalization": False,
    "clipping": False,
    "padding_short_inputs": False,
    "channel_fallback": False,
    "intermediate_pcm_encoding": "contiguous_little_endian_float64_mono",
}
CROP = {"start_sample": 176000, "stop_sample_exclusive": 304000,
        "source_resampled_samples": 480000}
DECISION_SCOPE = {
    "external_measurement_gate_accepted": True,
    "prospective_native30_implementation_and_synthetic_tests_authorized": True,
    "actual_cohort_measurement_authorized_by_this_file": False,
    "classifier_fits_authorized_by_this_file": False,
    "model_scoring_authorized_by_this_file": False,
    "threshold_changes_authorized": False,
    "separate_measurement_freeze_required": True,
    "separate_fit_freeze_required": True,
}
SCOPE = {
    "cpu_only": True,
    "classifier_fits": 0,
    "model_scoring": False,
    "classifier_admission": False,
    "neural_inference": False,
    "recovery_applied": False,
}
RUNNER_SHA = "822f51f3e961bb4edf16389283a91b898393084f34d1a87da0956d7a92cffe0d"
RUNNER_TESTS_SHA = "4d24a5e8689eb4d7599669e560ab504eb29a34b8409e9471b6a304c48ff5c7dd"
AUDIT_GATE_SHA = "637f07ec893ed64a85e4fcd4971da3b177ec4f9efa83315b76e4441cae5ec30e"
DECISION_SHA = "5f49573923d9ccbc1fcfc6e5b04a32b90b85c3f929e589d2f69d2969d020ca82"
RESERVED_COMMIT_SHA = "a96b7ecc93f4bd0c44954b6c64afc0a9b4c782482ec025291a248dd35862b2e4"
PREPARED_SHA = "3509a425dda56ec32c12bd0a481d22e8006f5892edc744ef1de8b350c00f58c9"
COHORT_SHA = "730b9bb25d18715c5ed0191ec9eb7a921d473b9534c9e5bc9196c25f5ce44510"
PLAN_SHA = "94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964"
SCREEN_SHA = "ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87"
PINS = {
    "materialize_native30_new1695_v1": "165f28d4e553ee679772aa1ac47b31f9a485b94d3a8ac409f25595577abe395a",
    "run_native30_fhsc_cohort_v1": "dc26465459df133b89d6023bb1c732e91ebab5b63ff02143280b50c383e7b3cc",
    "run_native30_inference_batches_v1": "84dce45426596888b3a4e3b20e4ad1c61dc976645506f269c7ec7990b12f3933",
    "validate_guitarset_archives_v2": "cda25ca63e382a1363ade813cd114838099bee55ee60f346888f2b1cd2ff6d6c",
    "draft_bicoherence_guitarset_pilot_v2": "c7368fb060c2583b107ecf3b01d0ad173160cd546c1ee6219933f5d934d9ce0e",
    "bicoherence_primitive_v2": "9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5",
    "bicoherence_audio_v1": "e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1",
    "bicoherence_scalar_v1": "b648653830e182dcbaaf1fbf3518f48810fd4b63beaa1136cba97ea538100f85",
    "summarize_bc_reserved_admission_v2": "7a20d24a9e78d951a7386cf316adf855f14b096459ffbf0e84d8d2a588d0518f",
}
PCM_ENCODING = (
    "SHA-256 of decoded samples in frame-major/channel-minor C order; each sample is "
    "IEEE-754 float64 encoded explicitly little-endian (<f8); no resampling, mixing, "
    "normalization, or feature extraction"
)
GRID = np.asarray([(a, b, a + b) for a in range(8, 193, 8)
                   for b in range(a, 193, 8) if a + b <= 256], dtype=np.int64)
TARGET = (32, 48, 80)
TARGET_INDEX = next(i for i, row in enumerate(GRID.tolist()) if row == list(TARGET))
N_FFT, HOP, FRAMES, POOL, RATE = 1024, 256, 247, 64000, 16000
ENERGY_FLOOR = 1e-6
ATOL, RTOL = 2e-12, 2e-11
HERE = Path(__file__).resolve().parent
AUDITOR_TESTS = HERE / "test_audit_native30_bc_v1.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False) + "\n").encode("utf-8")


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _no_symlink_components(path):
    path = Path(path)
    for current in (path, *path.parents):
        if current.is_symlink():
            raise ValueError("symlink path component is forbidden: " + str(current))


def safe_path(value):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts,
            "absolute path without parent traversal required: " + str(path))
    _no_symlink_components(path)
    return path


def file_binding(path):
    path = safe_path(path)
    require(path.is_file() and not path.is_symlink(),
            "bound regular file required: " + str(path))
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def binding(entry, *, hash_audio=True):
    require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"},
            "malformed file binding")
    require(hash_string(entry["sha256"]) and type(entry["bytes"]) is int
            and entry["bytes"] >= 0, "malformed file binding values")
    path = safe_path(entry["path"])
    require(path.is_file() and path.stat().st_size == entry["bytes"],
            "bound file missing/size changed: " + str(path))
    if hash_audio or path.suffix.lower() not in {
        ".wav", ".wave", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
        ".aif", ".aiff", ".wma", ".mp4", ".webm",
    }:
        require(digest(path) == entry["sha256"],
                "bound file hash changed: " + str(path))
    return entry


def read_json(path, expected=None):
    entry = file_binding(path)
    if expected is not None:
        require(entry == expected, "canonical JSON authority binding changed: " + str(path))
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    try:
        value = json.loads(Path(entry["path"]).read_text(encoding="utf-8"),
                           parse_constant=invalid)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON: " + str(path)) from exc
    return value, entry


def compare(actual, expected, label="value"):
    """Compare JSON values with exact schemas and the inherited tight margin."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + " keys")
        for key in expected:
            compare(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for index, value in enumerate(expected):
            compare(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool), label + " number")
        require(math.isclose(float(actual), expected, rel_tol=RTOL, abs_tol=ATOL), label + " numeric parity")
    else:
        require(actual == expected, label + " exact parity")


def array_value_hash(array):
    value = np.asarray(array)
    raw = np.ascontiguousarray(value).tobytes(order="C")
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest()}


def exact_array(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.dtype == expected.dtype and actual.shape == expected.shape,
            label + " shape/dtype")
    require(np.array_equal(actual, expected), label + " exact parity")


def close_array(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.dtype == expected.dtype and actual.shape == expected.shape,
            label + " shape/dtype")
    require(not np.isinf(actual).any() and not np.isinf(expected).any(),
            label + " infinity")
    require(np.allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True),
            label + " numeric parity")


def _array_specs():
    return {
        "window": ((1024,), np.dtype("float64")),
        "frequency_bins": ((228, 3), np.dtype("int64")),
        "frequency_hz": ((228, 3), np.dtype("float64")),
        "pool_start_samples": ((2,), np.dtype("int64")),
        "frame_offset_samples": ((247,), np.dtype("int64")),
        "frame_start_samples": ((2, 247), np.dtype("int64")),
        "frame_means": ((2, 247), np.dtype("float64")),
        "spectra": ((2, 247, 513), np.dtype("complex128")),
        "bin_coefficient_energy": ((2, 513), np.dtype("float64")),
        "total_non_dc_coefficient_energy": ((2,), np.dtype("float64")),
        "bin_energy_fraction": ((2, 513), np.dtype("float64")),
        "energy_floor_mask": ((2, 228), np.dtype("bool")),
        "primitive_defined_mask": ((2, 228), np.dtype("bool")),
        "eligible_mask": ((2, 228), np.dtype("bool")),
        "squared_bicoherence": ((2, 228), np.dtype("float64")),
        "biphase_radians": ((2, 228), np.dtype("float64")),
        "primitive_squared_bicoherence": ((2, 228), np.dtype("float64")),
        "primitive_biphase_radians": ((2, 228), np.dtype("float64")),
    }


def validate_array_schema(arrays):
    specs = _array_specs()
    require(set(arrays) == set(specs), "exact full-grid NPZ key set")
    nullable = {"bin_energy_fraction", "squared_bicoherence", "biphase_radians",
                "primitive_squared_bicoherence", "primitive_biphase_radians"}
    for name, (shape, dtype) in specs.items():
        value = arrays[name]
        require(value.shape == shape and value.dtype == dtype,
                "NPZ shape/dtype: " + name)
        require(not np.isinf(value).any(), "NPZ infinity: " + name)
        if name not in nullable:
            require(np.isfinite(value).all(), "NPZ nonfinite: " + name)
    exact_array(arrays["frequency_bins"], GRID, "fixed 228-cell grid")
    exact_array(arrays["frequency_hz"], GRID.astype(np.float64) * RATE / N_FFT,
                "grid frequencies")
    starts = np.arange(2, dtype=np.int64) * POOL
    offsets = np.arange(FRAMES, dtype=np.int64) * HOP
    exact_array(arrays["pool_start_samples"], starts, "pool starts")
    exact_array(arrays["frame_offset_samples"], offsets, "frame offsets")
    exact_array(arrays["frame_start_samples"], starts[:, None] + offsets[None, :],
                "frame starts")


def _binary_scaled(value, factors):
    """Independent finite binary mantissa/exponent representation."""
    mantissa, exponent = math.frexp(float(value))
    if mantissa == 0:
        return {"mantissa": 0.0, "exponent2": 0}
    for factor in factors:
        part, shift = math.frexp(float(factor))
        mantissa, renormalize = math.frexp(mantissa * part)
        exponent += shift + renormalize
    return {"mantissa": mantissa, "exponent2": exponent}


def independent_primitive(spectra, pair):
    """Replay one triad without importing bicoherence_primitive_v2.

    The normalized columns are deliberately F-contiguous.  This preserves the
    corrected independent-auditor layout lesson: adjacent complex columns are
    not accidentally treated as a C-order interleaved realization buffer.
    """
    source = np.asarray(spectra)
    require(source.dtype.kind in "iufc", "numeric spectra required")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        coefficients = np.asarray(source, dtype=np.complex128)
    require(not np.any((source != 0) & (coefficients == 0)),
            "unsupported coefficient dtype-conversion underflow")
    require(coefficients.ndim == 2 and coefficients.shape[0] >= 2
            and coefficients.shape[1] > 0 and np.isfinite(coefficients).all(),
            "finite [realization, frequency] spectra required")
    require(isinstance(pair, (tuple, list)) and len(pair) == 2,
            "two integer frequency bins required")
    require(all(isinstance(value, (int, np.integer)) and
                not isinstance(value, (bool, np.bool_)) for value in pair),
            "frequency bins must be integers")
    f1, f2 = map(int, pair)
    require(0 < f1 <= f2 and f1 + f2 < coefficients.shape[1], "invalid positive triad")
    columns = coefficients[:, [f1, f2, f1 + f2]]
    scales = np.maximum(np.max(np.abs(columns.real), axis=0),
                        np.max(np.abs(columns.imag), axis=0))
    result = {
        "version": "bicoherence_primitive_v2", "realizations": int(coefficients.shape[0]),
        "frequency_bins": [f1, f2, f1 + f2], "status": "ok",
        "squared_bicoherence": None, "biphase_radians": None,
        "biphase_status": "missing_triad_energy",
        "biphase_interpretation": "descriptive_only_no_significance",
        "coefficient_scales": scales.tolist(), "normalized_sums": None,
        "raw_sums": None, "raw_sum_encoding": "value = mantissa * 2**exponent2",
        "external_validation_passed": False, "classifier_admitted": False,
    }
    if np.any(scales == 0):
        result["status"] = "zero_energy" if not np.any(coefficients) else "missing_triad_energy"
        return result
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        normalized = np.empty(columns.shape, dtype=np.complex128, order="F")
        require(normalized.flags.f_contiguous and normalized.strides[0] == normalized.itemsize,
                "independent normalized columns are not F-contiguous")
        normalized.real = columns.real / scales
        normalized.imag = columns.imag / scales
        require(not np.any((columns != 0) & (normalized == 0)),
                "unsupported coefficient normalization underflow")
        u = normalized[:, 0] * normalized[:, 1]
        v = normalized[:, 2]
        if np.any((normalized[:, 0] != 0) & (normalized[:, 1] != 0) & (u == 0)):
            raise FloatingPointError("unsupported coefficient product underflow")
        u_magnitude = np.abs(u)
        v_magnitude = np.abs(v)
        u_power, v_power = u_magnitude * u_magnitude, v_magnitude * v_magnitude
        require(not np.any((u_magnitude != 0) & (u_power == 0)) and
                not np.any((v_magnitude != 0) & (v_power == 0)),
                "unsupported triad energy underflow")
        product_energy = float(np.sum(u_power))
        sum_energy = float(np.sum(v_power))
        terms = u * np.conj(v)
        if np.any((u != 0) & (np.conj(v) != 0) & (terms == 0)):
            raise FloatingPointError("unsupported triple-product underflow")
        triple = complex(np.sum(terms))
        require(math.isfinite(product_energy) and math.isfinite(sum_energy)
                and math.isfinite(triple.real) and math.isfinite(triple.imag),
                "nonfinite normalized primitive sums")
        sums = {"product_energy_sum": product_energy,
                "sum_frequency_energy_sum": sum_energy,
                "triple_sum_real": triple.real, "triple_sum_imag": triple.imag}
        result["normalized_sums"] = sums
        s1, s2, s3 = map(float, scales)
        result["raw_sums"] = {
            "product_energy_sum": _binary_scaled(product_energy, (s1, s1, s2, s2)),
            "sum_frequency_energy_sum": _binary_scaled(sum_energy, (s3, s3)),
            "triple_sum_real": _binary_scaled(triple.real, (s1, s2, s3)),
            "triple_sum_imag": _binary_scaled(triple.imag, (s1, s2, s3)),
        }
        if product_energy == 0 or sum_energy == 0:
            result["status"] = "missing_triad_product_energy"
            return result
        magnitude = abs(triple) / math.sqrt(product_energy) / math.sqrt(sum_energy)
        score = magnitude * magnitude
        rounding = 64 * np.finfo(np.float64).eps
        require(math.isfinite(score) and -rounding <= score <= 1 + rounding,
                "normalized primitive violates Cauchy-Schwarz bound")
        if score < 0:
            score = 0.0
        result["squared_bicoherence"] = min(1.0, score)
        if triple == 0:
            result["biphase_status"] = "undefined_zero_resultant"
        else:
            result["biphase_radians"] = math.atan2(triple.imag, triple.real)
            result["biphase_status"] = "descriptive_only_no_significance"
    return result


# Familiar name used by the earlier independent BC auditors.  This remains
# the standalone routine above and never dispatches to producer code.
primitive = independent_primitive


def independent_resample(samples):
    """Manually design the exact fixed Kaiser-5 FIR and call only upfirdn."""
    require(isinstance(samples, np.ndarray) and samples.dtype == np.dtype("float64")
            and samples.ndim == 1 and len(samples) > 0 and np.isfinite(samples).all(),
            "finite native float64 mono input required")
    half, up, down = 4410, 160, 441
    index = np.arange(-half, half + 1, dtype=np.float64)
    cutoff = 1.0 / down
    taps = cutoff * np.sinc(cutoff * index)
    taps *= i0(5.0 * np.sqrt(1.0 - (index / half) ** 2)) / i0(5.0)
    taps /= np.sum(taps)
    taps *= up
    prepad = down - half % down
    remove = (half + prepad) // down
    expected = (len(samples) * up + down - 1) // down
    padded = np.r_[np.zeros(prepad, dtype=np.float64), taps]
    result = upfirdn(padded, samples, up=up, down=down, mode="constant", cval=0.0)
    result = result[remove:remove + expected]
    require(len(result) == expected and np.isfinite(result).all(),
            "fixed whole-record resampling support")
    return result


# Familiar name used by the earlier independent BC auditors.
native_resample = independent_resample


def pcm_hash(values):
    values = np.asarray(values, dtype="<f8", order="C")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def reconstruct_view(stereo):
    """Promote stereo and perform whole30 resample -> center8 crop -> gain."""
    require(isinstance(stereo, np.ndarray) and stereo.dtype == np.dtype("float64")
            and stereo.shape == (INPUT_FORMAT["frames"], 2)
            and np.isfinite(stereo).all(), "exact finite float64 stereo required")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        mono = (stereo[:, 0] + stereo[:, 1]) / 2.0
        full = independent_resample(mono)
        require(len(full) == VIEW["source_resampled_samples"], "resampled frame count")
        start, stop = VIEW["crop_start_sample"], VIEW["crop_stop_sample_exclusive"]
        require(0 <= start < stop <= len(full) and stop - start == 128000,
                "fixed center crop support")
        crop = full[start:stop].copy()
        wave = VIEW["baseline_gain"] * crop
        require(not np.any((crop != 0) & (wave == 0)),
                "unsupported background-gain underflow")
        rms = float(np.sqrt(np.mean(wave ** 2)))
        require(math.isfinite(rms) and not (rms == 0 and np.any(wave)),
                "invalid baseline RMS")
    provenance = {
        "native_sample_rate_hz": 44100,
        "native_frames": INPUT_FORMAT["frames"],
        "target_sample_rate_hz": RATE,
        "resampled_frames": len(full),
        "resampling_applied": True,
        "up": 160, "down": 441,
        "window": ["kaiser", 5.0], "padtype": "constant", "cval": 0.0,
        "resample_scope": "entire_native_recording_before_crop",
        "native_16khz_policy": "unchanged_array",
        "crop_start": start, "crop_stop_exclusive": stop, "crop_frames": 128000,
        "crop_center_rule": "floor((resampled_frames - 128000)/2)",
        "crop_pcm_encoding": PCM_ENCODING,
        "crop_pcm_sha256": pcm_hash(crop),
        "crop_zero_amplitude": not np.any(crop),
        "measurement_support": "unsupported_zero_background_rms" if not np.any(crop) else "not_measured",
        "derived_audio_saved": False,
        "bc_extracted": False,
    }
    view = {**VIEW, "input_format": INPUT_FORMAT, "resampling": provenance,
            "downmix_float64_sha256": pcm_hash(mono),
            "crop_float64_sha256": pcm_hash(crop),
            "analysis_float64_sha256": pcm_hash(wave),
            "baseline_rms": rms,
            "construction_status": "ok" if rms > 0 else "unsupported_zero_background_rms"}
    return wave, view, {"stereo_float64_sha256": pcm_hash(stereo),
                        "resampled_float64_sha256": pcm_hash(full),
                        "crop_float64_sha256": pcm_hash(crop),
                        "analysis_float64_sha256": pcm_hash(wave)}


def read_native_source(row):
    """Read only the bound materialized Native30 WAV, as FLOAT32 then promote."""
    require(isinstance(row, dict) and isinstance(row.get("input"), dict),
            "row input binding required")
    path = safe_path(row["input"]["path"])
    require(binding(row["input"]) == row["input"], "source input binding changed before decode")
    chunks, frames = [], 0
    try:
        with sf.SoundFile(str(path), mode="r") as stream:
            require((str(stream.format), str(stream.subtype), int(stream.samplerate),
                     int(stream.channels), int(stream.frames)) ==
                    ("WAV", "FLOAT", 44100, 2, 1323000),
                    "exact Native30 FLOAT stereo header required")
            for block in stream.blocks(blocksize=65536, dtype="float32", always_2d=True):
                require(block.dtype == np.dtype("float32") and block.shape[1] == 2,
                        "FLOAT32 stereo block shape")
                require(np.isfinite(block).all(), "nonfinite source PCM")
                block = np.ascontiguousarray(block, dtype="<f4")
                chunks.append(block.copy())
                frames += len(block)
            tail = stream.read(1, dtype="float32", always_2d=True)
            require(len(tail) == 0, "source decoder has bytes beyond exact EOF")
    except (OSError, RuntimeError) as exc:
        raise ValueError("source SoundFile decode failed") from exc
    require(frames == 1323000, "source frame count changed")
    pcm32 = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), np.float32)
    require(hashlib.sha256(pcm32.astype("<f4", copy=False).tobytes(order="C")).hexdigest() ==
            row["waveform_float32_sha256"], "source FLOAT32 waveform hash mismatch")
    stereo = pcm32.astype(np.float64)
    require(stereo.shape == (1323000, 2) and np.isfinite(stereo).all(),
            "promoted float64 stereo shape/finite")
    require(binding(row["input"]) == row["input"], "source changed during decode")
    return stereo, {
        "input": row["input"],
        "waveform_float32_sha256": row["waveform_float32_sha256"],
        "waveform_float64_sha256": pcm_hash(stereo),
        "decoded_dtype": "float32_storage_promoted_to_float64",
        "frames": int(frames), "channels": 2, "sample_rate_hz": 44100,
    }


def replay_pool(wave, construction_status):
    """Independently replay both pools, all STFT arrays, and all grid cells."""
    require(isinstance(wave, np.ndarray) and wave.dtype == np.dtype("float64")
            and wave.shape == (128000,) and np.isfinite(wave).all(),
            "analysis waveform must be exact finite float64 center8")
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    pool_starts = np.arange(2, dtype=np.int64) * POOL
    offsets = np.arange(FRAMES, dtype=np.int64) * HOP
    frame_starts = pool_starts[:, None] + offsets[None, :]
    spectra = np.empty((2, FRAMES, N_FFT // 2 + 1), dtype=np.complex128)
    means = np.empty((2, FRAMES), dtype=np.float64)
    energies = np.empty((2, N_FFT // 2 + 1), dtype=np.float64)
    fractions = np.full_like(energies, np.nan)
    total = np.empty(2, dtype=np.float64)
    floor = np.zeros((2, len(GRID)), dtype=np.bool_)
    primitive_mask = np.zeros_like(floor)
    eligible = np.zeros_like(floor)
    scores = np.full((2, len(GRID)), np.nan, dtype=np.float64)
    phases = np.full_like(scores, np.nan)
    primitive_scores = np.full_like(scores, np.nan)
    primitive_phases = np.full_like(scores, np.nan)
    records = []
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for p, start in enumerate(pool_starts):
            pool = wave[int(start):int(start) + POOL]
            frames = np.asarray([pool[int(offset):int(offset) + N_FFT]
                                 for offset in offsets], dtype=np.float64)
            means[p] = frames.mean(axis=1)
            centered = frames - means[p, :, None]
            windowed = centered * window
            require(not np.any((centered != 0) & (window != 0) & (windowed == 0)),
                    "unsupported STFT window underflow")
            unnormalized = np.fft.rfft(windowed, axis=1)
            spectra[p] = unnormalized / window.sum()
            require(not np.any((unnormalized != 0) & (spectra[p] == 0)) and
                    np.isfinite(spectra[p]).all(), "unsupported/nonfinite STFT normalization")
            magnitudes = np.abs(spectra[p])
            powers = magnitudes * magnitudes
            require(not np.any((magnitudes != 0) & (powers == 0)),
                    "unsupported coefficient energy underflow")
            energies[p] = powers.sum(axis=0)
            total[p] = energies[p, 1:].sum()
            require(np.isfinite(energies[p]).all() and math.isfinite(float(total[p])),
                    "nonfinite pooled energy")
            if total[p] > 0:
                fractions[p] = energies[p] / total[p]
                require(not np.any((energies[p] != 0) & (fractions[p] == 0)),
                        "unsupported energy-fraction underflow")
                floor[p] = np.all(fractions[p, GRID] >= ENERGY_FLOOR, axis=1)
            zero_amplitude = not np.any(pool)
            pool_status = ("zero_amplitude" if zero_amplitude else
                           "zero_non_dc_energy" if total[p] == 0 else "ok")
            cells = []
            for c, bins in enumerate(GRID.tolist()):
                primitive = independent_primitive(spectra[p], (int(bins[0]), int(bins[1])))
                raw_score = primitive["squared_bicoherence"]
                raw_phase = primitive["biphase_radians"]
                primitive_mask[p, c] = primitive["status"] == "ok" and raw_score is not None
                eligible[p, c] = bool(floor[p, c] and primitive_mask[p, c])
                if raw_score is not None:
                    primitive_scores[p, c] = raw_score
                if raw_phase is not None:
                    primitive_phases[p, c] = raw_phase
                if eligible[p, c]:
                    scores[p, c] = raw_score
                    if raw_phase is not None:
                        phases[p, c] = raw_phase
                    status = "ok"
                elif not primitive_mask[p, c]:
                    status = primitive["status"]
                else:
                    status = "below_energy_fraction_floor"
                triad_fractions = ([float(v) for v in fractions[p, bins]]
                                   if total[p] > 0 else [None, None, None])
                cells.append({
                    "frequency_bins": [int(v) for v in bins],
                    "status": status,
                    "energy_fractions": triad_fractions,
                    "energy_floor_passed": bool(floor[p, c]),
                    "eligible": bool(eligible[p, c]),
                    "squared_bicoherence": raw_score if eligible[p, c] else None,
                    "biphase_radians": raw_phase if eligible[p, c] else None,
                    "primitive": primitive,
                })
            records.append({
                "pool_index": p, "start_sample": int(start),
                "stop_sample_exclusive": int(start + POOL), "status": pool_status,
                "zero_amplitude": bool(zero_amplitude), "coefficient_rows": FRAMES,
                "independent_realization_count": None,
                "total_non_dc_coefficient_energy": float(total[p]),
                "eligible_cell_count": int(eligible[p].sum()), "cells": cells,
            })
    metadata = {
        "version": "bicoherence_audio_v1", "primitive_version": "bicoherence_primitive_v2",
        "status": "ok", "input_samples": 128000, "sample_rate_hz": RATE,
        "input_zero_amplitude": not np.any(wave), "pool_samples": POOL,
        "pool_duration_seconds": 4.0, "pool_count": 2, "analyzed_samples": 128000,
        "discarded_tail_samples": 0, "tail_start_sample": 128000,
        "tail_policy": "discard_no_padding", "n_fft": N_FFT, "hop_samples": HOP,
        "frames_per_pool": FRAMES, "window": "periodic_hann",
        "window_sum": float(window.sum()),
        "transform": "rfft((frame - frame.mean()) * window) / window.sum()",
        "pool_boundary_policy": "no_frame_crosses_pool_boundary",
        "energy": "sum_frames(abs(coefficient)**2); no one-sided doubling",
        "fraction_denominator_bins_inclusive": [1, 512],
        "energy_fraction_min_inclusive": ENERGY_FLOOR,
        "grid_cell_count": 228,
        "grid": "positive parents 8..192 step 8; f1<=f2; f1+f2<=256",
        "biphase_convention": "atan2(imag,real) in [-pi,pi]; descriptive_only",
        "array_missing_encoding": "NaN; JSON metadata missing values are null",
        "independent_realization_count": None,
        "frames_overlap_and_are_not_asserted_independent": True,
        "amplitude_weighted_not_phase_only": True,
        "linear_fixed_phase_tones_can_have_high_bicoherence": True,
        "null_calibrated": False, "significance_inferred": False,
        "external_validation_passed": False, "classifier_admitted": False,
        "pools": records,
    }
    require(construction_status in ("ok", "unsupported_zero_background_rms"),
            "construction status")
    metadata["construction_status"] = construction_status
    arrays = {
        "window": window, "frequency_bins": GRID,
        "frequency_hz": GRID.astype(np.float64) * RATE / N_FFT,
        "pool_start_samples": pool_starts, "frame_offset_samples": offsets,
        "frame_start_samples": frame_starts, "frame_means": means,
        "spectra": spectra, "bin_coefficient_energy": energies,
        "total_non_dc_coefficient_energy": total, "bin_energy_fraction": fractions,
        "energy_floor_mask": floor, "primitive_defined_mask": primitive_mask,
        "eligible_mask": eligible, "squared_bicoherence": scores,
        "biphase_radians": phases, "primitive_squared_bicoherence": primitive_scores,
        "primitive_biphase_radians": primitive_phases,
    }
    validate_array_schema(arrays)
    return metadata, arrays


def reduce_scalar(metadata):
    """Independent fixed target reducer; no import of bicoherence_scalar_v1."""
    constants = {
        "version": "bicoherence_audio_v1", "primitive_version": "bicoherence_primitive_v2",
        "sample_rate_hz": RATE, "pool_samples": POOL, "pool_duration_seconds": 4.0,
        "n_fft": N_FFT, "hop_samples": HOP, "frames_per_pool": FRAMES,
        "grid_cell_count": 228, "energy_fraction_min_inclusive": ENERGY_FLOOR,
        "window": "periodic_hann", "tail_policy": "discard_no_padding",
        "pool_boundary_policy": "no_frame_crosses_pool_boundary",
    }
    require(isinstance(metadata, dict), "metadata dictionary required")
    for key, expected in constants.items():
        require(metadata.get(key) == expected, "extractor metadata mismatch: " + key)
    require(metadata.get("input_samples") == 128000 and metadata.get("pool_count") == 2
            and metadata.get("analyzed_samples") == 128000
            and metadata.get("discarded_tail_samples") == 0
            and metadata.get("tail_start_sample") == 128000
            and metadata.get("status") == "ok", "fixed two-pool support")
    construction = metadata.get("construction_status")
    require(construction in ("ok", "unsupported_zero_background_rms"),
            "invalid construction status")
    pools = metadata.get("pools")
    require(isinstance(pools, list) and len(pools) == 2, "metadata pools")
    values, records = [], []
    for index, pool in enumerate(pools):
        require(pool.get("pool_index") == index and pool.get("start_sample") == index * POOL
                and pool.get("stop_sample_exclusive") == (index + 1) * POOL
                and pool.get("coefficient_rows") == FRAMES, "pool order/support")
        status = pool.get("status")
        require(status in ("ok", "zero_amplitude", "zero_non_dc_energy"),
                "invalid pool status")
        energy = pool.get("total_non_dc_coefficient_energy")
        require(isinstance(energy, (int, float)) and not isinstance(energy, bool)
                and math.isfinite(float(energy)), "pool energy")
        require((float(energy) > 0) == (status == "ok"), "pool energy/status")
        require(type(pool.get("zero_amplitude")) is bool and
                pool["zero_amplitude"] == (status == "zero_amplitude"),
                "pool zero-amplitude status")
        cells = pool.get("cells")
        require(isinstance(cells, list) and [c.get("frequency_bins") for c in cells] == GRID.tolist(),
                "full grid order")
        cell = cells[TARGET_INDEX]
        primitive = cell.get("primitive")
        require(isinstance(primitive, dict) and primitive.get("version") == "bicoherence_primitive_v2"
                and primitive.get("realizations") == FRAMES and
                primitive.get("frequency_bins") == list(TARGET), "target primitive identity")
        primitive_status = primitive.get("status")
        require(primitive_status in ("ok", "zero_energy", "missing_triad_energy",
                                     "missing_triad_product_energy"), "target primitive status")
        raw = primitive.get("squared_bicoherence")
        if primitive_status == "ok":
            require(isinstance(raw, (int, float)) and not isinstance(raw, bool)
                    and math.isfinite(float(raw)) and 0 <= float(raw) <= 1,
                    "target primitive score")
        else:
            require(raw is None, "missing target primitive score")
        fractions = cell.get("energy_fractions")
        if status == "ok":
            require(isinstance(fractions, list) and len(fractions) == 3 and
                    all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in fractions),
                    "target energy fractions")
            floor_passed = all(float(v) >= ENERGY_FLOOR for v in fractions)
        else:
            require(fractions == [None, None, None], "zero-energy target fractions")
            require(primitive_status == "zero_energy", "zero-energy primitive status")
            floor_passed = False
        eligible = bool(floor_passed and primitive_status == "ok")
        expected_status = "ok" if eligible else primitive_status if primitive_status != "ok" else "below_energy_fraction_floor"
        require(cell.get("energy_floor_passed") == floor_passed and cell.get("eligible") == eligible
                and cell.get("status") == expected_status, "target masks/status")
        score = cell.get("squared_bicoherence")
        require(score == raw if eligible else score is None, "masked target score")
        values.append(float(score) if eligible else None)
        records.append({
            "pool_index": index, "start_sample": index * POOL,
            "stop_sample_exclusive": (index + 1) * POOL,
            "source_start_sample": CROP["start_sample"] + index * POOL,
            "source_stop_sample_exclusive": CROP["start_sample"] + (index + 1) * POOL,
            "pool_status": status, "target_status": cell["status"],
            "eligible": eligible, "squared_bicoherence": float(score) if eligible else None,
        })
    available = [v for v in values if v is not None]
    return {
        "version": "bicoherence_scalar_v1", "target_frequency_bins": list(TARGET),
        "target_frequency_hz": [500, 750, 1250], "crop": CROP,
        "input_samples": 128000, "pool_count": 2, "analyzed_samples": 128000,
        "discarded_tail_samples": 0, "eligible_pool_count": len(available),
        "missing_pool_count": 2 - len(available), "eligibility_mask": [v is not None for v in values],
        "status": "ok" if available else "no_eligible_target_pools",
        "median_squared_bicoherence": float(statistics.median(available)) if available else None,
        "pools": records, "construction_status": construction,
        "null_calibrated": False, "significance_inferred": False,
        "external_validation_passed": False, "classifier_admitted": False,
    }


def validate_view(view, expected):
    expected_keys = set(VIEW) | {"input_format", "resampling", "downmix_float64_sha256",
                                "crop_float64_sha256", "analysis_float64_sha256",
                                "baseline_rms", "construction_status"}
    require(isinstance(view, dict) and set(view) == expected_keys,
            "analysis view schema")
    require(view["input_format"] == INPUT_FORMAT, "analysis input format")
    for key, value in VIEW.items():
        require(view[key] == value, "fixed analysis view: " + key)
    compare(view, expected, "analysis view")
    provenance = view["resampling"]
    require(provenance["native_sample_rate_hz"] == 44100 and provenance["native_frames"] == 1323000
            and provenance["target_sample_rate_hz"] == RATE
            and provenance["resampled_frames"] == 480000
            and provenance["crop_start"] == 176000 and provenance["crop_stop_exclusive"] == 304000
            and provenance["crop_frames"] == 128000
            and provenance["crop_pcm_sha256"] == view["crop_float64_sha256"]
            and provenance["up"] == 160 and provenance["down"] == 441
            and provenance["window"] == ["kaiser", 5.0]
            and provenance["padtype"] == "constant" and provenance["cval"] == 0.0
            and provenance["resample_scope"] == "entire_native_recording_before_crop",
            "resampling view provenance")
    for key in ("downmix_float64_sha256", "crop_float64_sha256", "analysis_float64_sha256"):
        require(hash_string(view[key]), "view PCM hash")
    require(type(view["baseline_rms"]) is float and math.isfinite(view["baseline_rms"])
            and view["baseline_rms"] >= 0
            and view["construction_status"] ==
            ("ok" if view["baseline_rms"] > 0 else "unsupported_zero_background_rms"),
            "baseline construction status")


def _metadata_and_array_parity(metadata, arrays, expected_metadata, expected_arrays):
    compare(metadata, expected_metadata, "independent full metadata")
    validate_array_schema(arrays)
    validate_array_schema(expected_arrays)
    max_abs = 0.0
    for name in expected_arrays:
        if expected_arrays[name].dtype.kind in "fc":
            close_array(arrays[name], expected_arrays[name], "array " + name)
            finite = np.isfinite(arrays[name]) & np.isfinite(expected_arrays[name])
            if np.any(finite):
                max_abs = max(max_abs, float(np.max(np.abs(arrays[name][finite] - expected_arrays[name][finite]))))
        else:
            exact_array(arrays[name], expected_arrays[name], "array " + name)
    # Metadata and arrays must agree independently of the stored producer view.
    for p, pool in enumerate(metadata["pools"]):
        require(arrays["total_non_dc_coefficient_energy"][p] == pool["total_non_dc_coefficient_energy"],
                "metadata/array pooled energy")
        for c, cell in enumerate(pool["cells"]):
            require(np.array_equal(arrays["frequency_bins"][c], cell["frequency_bins"]),
                    "metadata/array grid")
            require(bool(arrays["energy_floor_mask"][p, c]) == cell["energy_floor_passed"]
                    and bool(arrays["eligible_mask"][p, c]) == cell["eligible"]
                    and bool(arrays["primitive_defined_mask"][p, c]) ==
                    (cell["primitive"]["status"] == "ok"), "metadata/array masks")
            for name, value in (
                ("squared_bicoherence", cell["squared_bicoherence"]),
                ("biphase_radians", cell["biphase_radians"]),
                ("primitive_squared_bicoherence", cell["primitive"]["squared_bicoherence"]),
                ("primitive_biphase_radians", cell["primitive"]["biphase_radians"]),
            ):
                observed = arrays[name][p, c]
                require(np.isnan(observed) if value is None else
                        math.isclose(float(observed), float(value), rel_tol=RTOL, abs_tol=ATOL),
                        "metadata/array scalar: " + name)
    return max_abs


def _validate_rows(rows):
    require(isinstance(rows, list) and len(rows) == EXPECTED_COUNT, "Native30 exact 3830-row roster")
    ids = [row.get("id") for row in rows]
    require(ids == sorted(ids) and len(ids) == len(set(ids)), "rows must be unique and ID-sorted")
    expected_keys = {"component_id", "component_sha256", "group_id", "id", "input", "label",
                     "origin_family", "origin_plan_row_sha256", "producer_receipt_sha256",
                     "role", "screen_row_sha256", "source_group", "waveform_float32_sha256"}
    for row in rows:
        require(isinstance(row, dict) and set(row) == expected_keys, "full producer row schema")
        require(isinstance(row["id"], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["id"]),
                "safe row ID")
        require(row["role"] == "development" and row["label"] in {"0", "1"}
                and row["origin_family"] in {"new", "prior"}
                and isinstance(row["source_group"], str) and row["source_group"],
                "row identity/scope")
        for key in ("component_sha256", "origin_plan_row_sha256", "screen_row_sha256",
                    "producer_receipt_sha256", "waveform_float32_sha256"):
            require(hash_string(row[key]), "row provenance hash: " + key)
        require(isinstance(row["component_id"], str) and row["component_id"]
                and isinstance(row["group_id"], str) and row["group_id"], "row group/component")
        require(set(row["input"]) == {"path", "bytes", "sha256"}
                and Path(row["input"]["path"]).name == row["id"] + ".wav",
                "canonical row input binding")
        binding(row["input"], hash_audio=False)
    require({key: sum(row["source_group"] == key for row in rows) for key in SOURCE_COUNTS} == SOURCE_COUNTS
            and len({row["source_group"] for row in rows} - set(SOURCE_COUNTS)) == 0,
            "source coverage")
    require({key: sum(row["label"] == key for row in rows) for key in LABEL_COUNTS} == LABEL_COUNTS,
            "label coverage")
    require({key: sum(row["origin_family"] == key for row in rows) for key in ORIGIN_COUNTS} == ORIGIN_COUNTS,
            "origin coverage")


def _validate_origins(origins, rows):
    require(isinstance(origins, dict) and set(origins) == {row["id"] for row in rows},
            "native-origin roster")
    for ident, origin in origins.items():
        require(isinstance(origin, dict) and origin.get("channels") == 2
                and origin.get("sample_rate_hz") in {44100, 48000}
                and origin.get("scope") in {
                    "full_original_file",
                    "accepted_native60_interval_from_full_original",
                    "accepted_native60_interval_from_archive_member_original",
                }
                and hash_string(origin.get("sha256")), "native-origin metadata")
        require(isinstance(origin.get("path"), str) and Path(origin["path"]).is_absolute(),
                "native-origin path")
        if "duration_s" in origin:
            require(isinstance(origin["duration_s"], (int, float))
                    and not isinstance(origin["duration_s"], bool)
                    and math.isfinite(float(origin["duration_s"]))
                    and float(origin["duration_s"]) > 0, "native-origin duration")
        if "provenance_receipt" in origin:
            receipt = origin["provenance_receipt"]
            require(isinstance(receipt, dict) and hash_string(receipt.get("sha256"))
                    and isinstance(receipt.get("path"), str)
                    and Path(receipt["path"]).is_absolute(),
                    "native-origin provenance receipt")
        if "original_receipt_path" in origin:
            require(isinstance(origin["original_receipt_path"], str)
                    and Path(origin["original_receipt_path"]).is_absolute()
                    and origin.get("path_relation") in {
                        "same_path", "byte_identity_by_sha_not_path",
                    },
                    "native-origin original receipt lineage")
        if "conditioning" in origin:
            require(isinstance(origin["conditioning"], dict), "native-origin conditioning lineage")
        if "archive_member" in origin:
            require(isinstance(origin["archive_member"], dict)
                    and isinstance(origin["archive_member"].get("path"), str)
                    and hash_string(origin["archive_member"].get("sha256")),
                    "native-origin archive lineage")


def _validate_runtime(runtime):
    """Rehash the frozen runtime and package graph without importing producer code."""
    require(isinstance(runtime, dict), "frozen runtime required")
    require(platform.python_version() == runtime.get("python")
            and np.__version__ == runtime.get("numpy") == "1.26.4"
            and scipy.__version__ == runtime.get("scipy") == "1.17.1"
            and sf.__version__ == runtime.get("soundfile") == "0.14.0",
            "pinned Python/NumPy/SciPy/SoundFile runtime")
    # The producer freezes the resolved interpreter path.  A venv launcher is
    # commonly a symlink, so bind the same resolved executable here.
    require(file_binding(Path(sys.executable).resolve()) == runtime.get("executable"),
            "Python executable pin")
    for name, path in runtime.get("import_origins", {}).items():
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == safe_path(path), "runtime import origin: " + name)
    # Frozen runtime manifests store bindings as dictionaries, not paths.
    # Re-validate both the path/size/hash tuple and the bytes on disk.
    for entry in runtime.get("module_files", {}).values():
        require(binding(entry, hash_audio=True) == entry, "runtime module binding")
    for entry in runtime.get("package_files", {}).values():
        require(binding(entry, hash_audio=True) == entry, "runtime package binding")
    if runtime.get("libsndfile_binary"):
        require(binding(runtime["libsndfile_binary"], hash_audio=True) == runtime["libsndfile_binary"],
                "libsndfile binary pin")
    if runtime.get("libsndfile_resolution", {}).get("mapped_path"):
        mapped = runtime["libsndfile_resolution"]["mapped_path"]
        require(file_binding(mapped)["sha256"] == runtime["libsndfile_binary"]["sha256"],
                "live libsndfile mapping pin")
    for key, value in runtime.get("thread_environment", {}).items():
        require(os.environ.get(key) == value, "thread environment pin: " + key)
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "soundfile": sf.__version__,
            "platform": platform.platform()}


def _validate_gate(request, gate):
    require(isinstance(request, dict) and set(request) ==
            {"audit", "cohort_contract", "decision", "plan", "producer_commit", "screen"},
            "six pinned producer authorities required")
    for name, entry in request.items():
        binding(entry, hash_audio=True)
    require(request["audit"]["sha256"] == AUDIT_GATE_SHA
            and request["decision"]["sha256"] == DECISION_SHA
            and request["producer_commit"]["sha256"] == RESERVED_COMMIT_SHA
            and request["cohort_contract"]["sha256"] == COHORT_SHA
            and request["plan"]["sha256"] == PLAN_SHA
            and request["screen"]["sha256"] == SCREEN_SHA,
            "fixed external authority pins")
    decision, _ = read_json(request["decision"]["path"], request["decision"])
    require(decision.get("version") == "bc_native30_transfer_decision_v1"
            and decision.get("status") ==
            "external_measurement_gate_accepted_prospective_transfer_implementation_only"
            and decision.get("scope") == DECISION_SCOPE
            and decision.get("prospective_cohort_rows") == EXPECTED_COUNT
            and decision.get("prospective_feature") == FEATURE
            and decision.get("producer_commit_sha256") == RESERVED_COMMIT_SHA
            and decision.get("tolerances_and_scientific_margins_unchanged") is True
            and decision.get("independent_audit") ==
            {key: request["audit"][key] for key in ("path", "sha256")},
            "external decision scope")
    audit, _ = read_json(request["audit"]["path"], request["audit"])
    require(audit.get("passed") is True and audit.get("BC_admitted") is False
            and audit.get("classifier_fits") == 0 and audit.get("model_scoring") is False
            and audit.get("thresholds_changed") is False
            and audit.get("status") ==
            "passed_independent_reserved_numerical_and_accountability_replay_not_admitted",
            "external corrected audit gate")
    reserved_commit, _ = read_json(request["producer_commit"]["path"], request["producer_commit"])
    require(isinstance(reserved_commit.get("status"), str)
            and reserved_commit["status"].startswith("committed_reserved_measurements_")
            and reserved_commit.get("BC_admitted") is False,
            "external reserved COMMIT gate")
    expected_gate = {
        "actual_execution_authorized": False,
        "audit": request["audit"], "decision": request["decision"],
        "producer_commit": request["producer_commit"],
        "scientific_gate_satisfied": True,
    }
    require(isinstance(gate, dict), "prepared gate")
    for key, value in expected_gate.items():
        require(gate.get(key) == value, "prepared gate binding: " + key)
    require(isinstance(gate.get("summary"), dict) and set(gate["summary"]) == {"path", "bytes", "sha256"}
            and binding(gate["summary"], hash_audio=True) == gate["summary"],
            "prepared gate summary binding")


def _validate_cohort(prepared):
    request = prepared["request"]
    cohort, cohort_entry = read_json(request["cohort_contract"]["path"], request["cohort_contract"])
    require(value_hash(cohort) == COHORT_SHA and cohort_entry == request["cohort_contract"],
            "frozen cohort contract canonical hash")
    require(cohort.get("version") == "run_native30_fhsc_cohort_v1"
            and cohort.get("status") == "frozen_before_any_measurement_audio_reads"
            and cohort.get("input_format") == INPUT_FORMAT
            and cohort.get("expected_count") == EXPECTED_COUNT
            and cohort.get("source_counts") == SOURCE_COUNTS
            and cohort.get("human") == LABEL_COUNTS["0"]
            and cohort.get("ai") == LABEL_COUNTS["1"], "cohort contract schema/accounting")
    require(cohort.get("rows") == prepared["rows"], "prepared/cohort roster join")
    bindings = cohort.get("bindings")
    require(isinstance(bindings, dict), "cohort binding graph")
    for path, entry in bindings.items():
        require(path == entry.get("path"), "cohort binding path index")
        binding(entry, hash_audio=False)
    for row in prepared["rows"]:
        require(bindings.get(row["input"]["path"]) == row["input"],
                "cohort omitted row input binding")
    inventories = cohort.get("upstream_inventories", {})
    require(isinstance(inventories, dict), "cohort upstream inventory")
    for root, declared in inventories.items():
        root_path = safe_path(root)
        require(root_path.is_dir() and not root_path.is_symlink(), "cohort inventory root")
        actual = []
        for path in sorted(root_path.rglob("*")):
            require(not path.is_symlink(), "upstream inventory symlink")
            if path.is_file() and path.name != "writer.lock":
                actual.append(path.relative_to(root_path).as_posix())
        require(actual == declared, "cohort upstream inventory changed: " + root)
    return cohort_entry


def _validate_prepared(prepared, freeze):
    keys = {"audio_reads_in_preflight", "classifier_admission", "classifier_fits", "code",
            "cpu_only", "expected_count", "family", "feature_names", "gate", "grid_role",
            "input_format", "missingness", "model_scoring", "native_origins", "neural_inference",
            "output_root", "recovery_applied", "request", "rows", "runtime", "source_counts", "status", "version", "view"}
    require(isinstance(prepared, dict) and set(prepared) == keys, "prepared contract exact schema")
    require(prepared.get("version") == PRODUCER_VERSION
            and prepared.get("status") == "prepared_metadata_only_not_execution_authority"
            and prepared.get("expected_count") == EXPECTED_COUNT
            and prepared.get("input_format") == INPUT_FORMAT
            and prepared.get("source_counts") == SOURCE_COUNTS
            and prepared.get("feature_names") == [FEATURE]
            and prepared.get("family") == "BC"
            and prepared.get("audio_reads_in_preflight") == 0
            and prepared.get("grid_role") == "diagnostic_only_not_predictors"
            and prepared.get("missingness") == "scalar_null_retains_row"
            and all(prepared.get(key) == value for key, value in SCOPE.items()),
            "prepared contract scope/view/accounting")
    require(prepared.get("view") == VIEW, "prepared fixed view")
    _validate_gate(prepared["request"], prepared["gate"])
    _validate_rows(prepared["rows"])
    _validate_origins(prepared["native_origins"], prepared["rows"])
    _validate_cohort(prepared)
    code = prepared.get("code")
    require(isinstance(code, dict) and set(code) == set(PINS) | {"runner", "tests"},
            "prepared producer code pins")
    for name, expected_sha in PINS.items():
        entry = code[name]
        require(entry["sha256"] == expected_sha and binding(entry, hash_audio=True) == entry,
                "prepared code pin: " + name)
    runner = HERE / (PRODUCER_VERSION + ".py")
    runner_tests = HERE / ("test_" + PRODUCER_VERSION + ".py")
    require(code["runner"]["sha256"] == RUNNER_SHA and file_binding(runner) == code["runner"],
            "producer runner pin")
    require(code["tests"]["sha256"] == RUNNER_TESTS_SHA and file_binding(runner_tests) == code["tests"],
            "producer test pin")
    _validate_runtime(prepared["runtime"])
    output_root = safe_path(prepared["output_root"])
    require(output_root.is_absolute() and ".." not in output_root.parts,
            "prepared output root")
    return prepared


def _validate_authorities(result_root, freeze_path, freeze_sha, commit_path, commit_sha):
    freeze_path = safe_path(freeze_path)
    commit_path = safe_path(commit_path)
    require(freeze_path.is_file() and not freeze_path.is_symlink(),
            "parent-freeze file required")
    require(commit_path.is_file() and not commit_path.is_symlink(),
            "complete COMMIT file required")
    require(hash_string(freeze_sha) and digest(freeze_path) == freeze_sha,
            "caller parent-freeze SHA mismatch")
    require(hash_string(commit_sha) and digest(commit_path) == commit_sha,
            "caller COMMIT SHA mismatch")
    freeze, freeze_entry = read_json(freeze_path, file_binding(freeze_path))
    require(freeze.get("version") == FREEZE_VERSION
            and freeze.get("status") == "parent_frozen_for_native30_BC_measurement"
            and freeze.get("measurement_authorized") is True
            and all(freeze.get(key) == value for key, value in SCOPE.items()),
            "separate parent measurement freeze required")
    prepared_entry = freeze.get("prepared_contract")
    require(isinstance(prepared_entry, dict) and prepared_entry["sha256"] == PREPARED_SHA,
            "prepared contract fixed pin")
    prepared, current_prepared_entry = read_json(prepared_entry["path"], prepared_entry)
    require(current_prepared_entry == prepared_entry and value_hash(prepared) == PREPARED_SHA,
            "prepared contract hash")
    require(freeze.get("runner") == prepared["code"]["runner"]
            and freeze.get("tests") == prepared["code"]["tests"]
            and freeze.get("decision") == prepared["request"]["decision"]
            and freeze.get("view") == VIEW and freeze.get("runtime") == prepared["runtime"],
            "freeze code/decision/view/runtime binding")
    _validate_prepared(prepared, freeze)
    result_root = safe_path(result_root)
    require(result_root == safe_path(prepared["output_root"])
            and result_root.is_dir() and not result_root.is_symlink(),
            "result root differs from frozen output root")
    for row in prepared["rows"]:
        source_parent = safe_path(Path(row["input"]["path"]).parent)
        require(not (result_root.is_relative_to(source_parent)
                     or source_parent.is_relative_to(result_root)),
                "result root overlaps bound source")
    require(commit_path == result_root / "COMMIT.json", "caller COMMIT path differs from result root")
    # Authorities must all remain outside the result tree.  Audio input bytes
    # are intentionally not opened until after this terminal product check.
    for entry in [freeze_entry, prepared_entry, *prepared["request"].values(),
                  *prepared["code"].values()]:
        path = safe_path(entry["path"])
        require(not path.is_relative_to(result_root), "authority overlaps result output")
    commit, commit_entry = read_json(commit_path, file_binding(commit_path))
    require(commit_entry["sha256"] == commit_sha, "caller COMMIT binding")
    return freeze, freeze_entry, prepared, prepared_entry, commit, commit_entry


def _expected_inventory(root, rows):
    expected = {"writer.lock", "contract.json", "manifest.json", "COMMIT.json"}
    expected.update("items/" + row["id"] + ".json" for row in rows)
    expected.update("metadata/" + row["id"] + ".json" for row in rows)
    expected.update("arrays/" + row["id"] + ".npz" for row in rows)
    actual = set()
    directories = set()
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "result symlink forbidden")
        if path.is_dir():
            directories.add(path.relative_to(root).as_posix())
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == expected, "complete result inventory required; no failures/orphans")
    require(directories == {"items", "metadata", "arrays", "failures"},
            "unexpected result directory")
    for folder in ("items", "metadata", "arrays", "failures"):
        path = root / folder
        require(path.is_dir() and not path.is_symlink(), "result product directory")
    lock = root / "writer.lock"
    require(lock.is_file() and not lock.is_symlink(), "result writer lock required")
    return actual


def _execution_contract(freeze_entry, prepared_entry, prepared):
    """Reconstruct the producer's parent-authorized run contract envelope."""
    return {
        "version": PRODUCER_VERSION,
        "status": "parent_authorized_BC_measurement_only",
        "prepared": prepared,
        "parent_freeze": freeze_entry,
        "prepared_contract": prepared_entry,
        **SCOPE,
    }


def _products(root, inventory):
    result = {}
    for name in sorted(inventory - {"writer.lock", "COMMIT.json"}):
        entry = file_binding(root / name)
        result[name] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
    return result


def _validate_commit_and_manifest(root, rows, prepared, freeze_entry, prepared_entry,
                                  commit, commit_entry, commit_sha):
    require(set(commit) == {"version", "status", "contract_sha256", "completed", "products",
                            "all_bound_inputs_and_products_end_rehashed", *SCOPE},
            "complete COMMIT schema")
    contract, contract_entry = read_json(root / "contract.json")
    expected_contract = _execution_contract(freeze_entry, prepared_entry, prepared)
    contract_hash = value_hash(expected_contract)
    require(contract == expected_contract and contract_entry["sha256"] == contract_hash
            and commit.get("contract_sha256") == contract_hash,
            "producer contract binding")
    inventory = _expected_inventory(root, rows)
    products = _products(root, inventory)
    require(commit.get("version") == PRODUCER_VERSION
            and commit.get("status") == "committed_native30_BC_measurements_not_admitted"
            and commit.get("completed") == EXPECTED_COUNT
            and commit.get("all_bound_inputs_and_products_end_rehashed") is True
            and all(commit.get(key) == value for key, value in SCOPE.items())
            and commit.get("products") == products,
            "complete producer COMMIT/inventory")
    require(commit_entry["sha256"] == commit_sha and digest(root / "COMMIT.json") == commit_sha,
            "caller COMMIT pin")
    return inventory, products, contract_hash


def _verify_item(root, row, prepared, contract_hash):
    ident = row["id"]
    item_path = root / "items" / (ident + ".json")
    metadata_path = root / "metadata" / (ident + ".json")
    arrays_path = root / "arrays" / (ident + ".npz")
    item, item_entry = read_json(item_path)
    require(set(item) == {"payload", "receipt_sha256"}
            and item["receipt_sha256"] == value_hash(item["payload"])
            and item_entry["sha256"] == value_hash(item), "canonical item receipt")
    payload = item["payload"]
    expected_payload_keys = {"version", "status", "id", "row", "row_sha256", "contract_sha256",
                             "native_origin", "analysis_view", "metadata", "arrays", "scalar",
                             "features", *SCOPE}
    require(set(payload) == expected_payload_keys
            and payload["version"] == PRODUCER_VERSION
            and payload["status"] == "measured_BC_not_classifier_admitted"
            and payload["id"] == ident and payload["row"] == row
            and payload["row_sha256"] == value_hash(row)
            and payload["contract_sha256"] == contract_hash
            and payload["native_origin"] == prepared["native_origins"][ident]
            and all(payload.get(key) == value for key, value in SCOPE.items()),
            "item lineage/scope")
    require(payload["metadata"] == file_binding(metadata_path)
            and payload["arrays"] == file_binding(arrays_path),
            "item product bindings")
    metadata, metadata_entry = read_json(metadata_path)
    require(metadata_entry["sha256"] == value_hash(metadata), "canonical BC metadata")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    stereo, source = read_native_source(row)
    wave, expected_view, reconstruction = reconstruct_view(stereo)
    validate_view(payload["analysis_view"], expected_view)
    expected_metadata, expected_arrays = replay_pool(wave, expected_view["construction_status"])
    max_abs = _metadata_and_array_parity(metadata, arrays, expected_metadata, expected_arrays)
    expected_scalar = reduce_scalar(expected_metadata)
    compare(payload["scalar"], expected_scalar, "independent scalar")
    compare(payload["features"], {FEATURE: expected_scalar["median_squared_bicoherence"]},
            "feature/null reducer")
    require(metadata.get("construction_status") == expected_view["construction_status"],
            "metadata/view construction status")
    return {
        "id": ident, "source_group": row["source_group"], "label": row["label"],
        "row_sha256": value_hash(row), "input": row["input"], "source": source,
        "reconstruction": reconstruction, "analysis_view": expected_view,
        "item": item_entry, "metadata": metadata_entry,
        "arrays": file_binding(arrays_path),
        "array_value_hashes": {key: array_value_hash(arrays[key]) for key in sorted(arrays)},
        "scalar": expected_scalar, "scientific_null": expected_scalar["median_squared_bicoherence"] is None,
        "full_grid_cells": 2 * len(GRID), "numeric_max_abs_error": max_abs,
    }


def _manifest(root, rows, item_evidence, contract_hash):
    by_source = {}
    for source in sorted(SOURCE_COUNTS):
        records = [entry for entry in item_evidence if entry["source_group"] == source]
        by_source[source] = {
            "rows": len(records),
            "available": sum(not entry["scientific_null"] for entry in records),
            "missing": sum(entry["scientific_null"] for entry in records),
            "eligible_pool_counts": dict(sorted(
                ((str(count), sum(entry["scalar"]["eligible_pool_count"] == count for entry in records))
                 for count in sorted({entry["scalar"]["eligible_pool_count"] for entry in records})),
                key=lambda pair: pair[0])),
            "pool_statuses": dict(sorted(
                ((status, sum(pool["target_status"] == status for entry in records
                              for pool in entry["scalar"]["pools"]))
                 for status in sorted({pool["target_status"] for entry in records for pool in entry["scalar"]["pools"]})),
                key=lambda pair: pair[0])),
        }
    expected = {
        "version": PRODUCER_VERSION,
        "status": "all_BC_rows_measured_nulls_retained_not_admitted",
        "contract_sha256": contract_hash, "count": EXPECTED_COUNT,
        "ids": [row["id"] for row in rows], "source_coverage": by_source,
        "feature_names": [FEATURE], "view": VIEW,
        "measurement_receipts": {entry["id"]: entry["item"] for entry in item_evidence},
        **SCOPE,
    }
    manifest, manifest_entry = read_json(root / "manifest.json")
    require(manifest == expected and manifest_entry["sha256"] == value_hash(expected),
            "manifest coverage/arithmetic replay")
    return manifest_entry


def _overlap(path, root):
    try:
        return path.is_relative_to(root) or root.is_relative_to(path)
    except ValueError:
        return False


@contextmanager
def shared_result_lock(root):
    lock = root / "writer.lock"
    require(lock.is_file() and not lock.is_symlink(), "regular result writer lock required")
    with lock.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("result writer lock is held") from None
        yield


@contextmanager
def shared_source_locks(prepared):
    """Take non-mutating locks offered by the frozen source inventories."""
    cohort, _ = read_json(prepared["request"]["cohort_contract"]["path"],
                          prepared["request"]["cohort_contract"])
    roots = set(cohort.get("upstream_inventories", {}))
    roots.update(str(Path(row["input"]["path"]).parent) for row in prepared["rows"])
    streams = []
    try:
        for root in sorted(roots):
            lock = safe_path(Path(root) / "writer.lock")
            if not lock.exists():
                continue
            require(lock.is_file() and not lock.is_symlink(), "unsafe source writer lock")
            stream = lock.open("rb")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                raise RuntimeError("source writer lock is held") from None
            streams.append(stream)
        yield
    finally:
        for stream in reversed(streams):
            stream.close()


def audit(result_dir, freeze_path, freeze_sha256, commit_path, commit_sha256, output_path):
    """Audit the complete caller-pinned Native30 BC result, then publish one receipt."""
    result_dir = safe_path(result_dir)
    output_path = safe_path(output_path)
    require(result_dir.is_dir() and not result_dir.is_symlink(), "result root required")
    require(not output_path.exists() and not output_path.is_symlink(),
            "audit output must be new and non-symlink")
    require(output_path.parent.is_dir() and not output_path.parent.is_symlink(),
            "audit output parent required")
    freeze, freeze_entry, prepared, prepared_entry, commit, commit_entry = _validate_authorities(
        result_dir, freeze_path, freeze_sha256, commit_path, commit_sha256)
    immutable = [HERE, safe_path(freeze_path), safe_path(prepared_entry["path"]),
                 safe_path(AUDITOR_TESTS)]
    immutable += [safe_path(entry["path"]) for entry in prepared["request"].values()]
    immutable += [safe_path(entry["path"]) for entry in prepared["code"].values()]
    cohort, _ = read_json(prepared["request"]["cohort_contract"]["path"],
                          prepared["request"]["cohort_contract"])
    immutable += [safe_path(root) for root in cohort.get("upstream_inventories", {})]
    if cohort.get("output_root"):
        immutable.append(safe_path(cohort["output_root"]))
    for row in prepared["rows"]:
        input_path = safe_path(row["input"]["path"])
        immutable.append(input_path.parent)
    require(not any(_overlap(output_path, root) for root in [result_dir, *immutable]),
            "audit output overlaps immutable input/result")
    contract_hash = value_hash(prepared)
    with shared_result_lock(result_dir), shared_source_locks(prepared):
        inventory, products, contract_hash = _validate_commit_and_manifest(
            result_dir, prepared["rows"], prepared, freeze_entry, prepared_entry,
            commit, commit_entry, commit_sha256)
        evidence = []
        for index, row in enumerate(prepared["rows"], 1):
            evidence.append(_verify_item(result_dir, row, prepared, contract_hash))
            if index % 25 == 0 or index == EXPECTED_COUNT:
                print(f"Native30 BC independent audit {index}/{EXPECTED_COUNT}", file=sys.stderr, flush=True)
        manifest_entry = _manifest(result_dir, prepared["rows"], evidence, contract_hash)
        # End rehashes close the read-only audit transaction.  In particular,
        # a source/product mutation after its item was read cannot be resealed.
        require(file_binding(safe_path(freeze_path)) == freeze_entry
                and file_binding(prepared_entry["path"]) == prepared_entry
                and file_binding(commit_path) == commit_entry,
                "freeze/prepared/COMMIT changed during audit")
        for entry in prepared["request"].values():
            require(binding(entry, hash_audio=True) == entry, "authority changed during audit")
        for entry in prepared["code"].values():
            require(binding(entry, hash_audio=True) == entry, "code binding changed during audit")
        for row in prepared["rows"]:
            require(binding(row["input"], hash_audio=True) == row["input"],
                    "source input changed during audit")
        require(_products(result_dir, inventory) == products,
                "result products changed during audit")
        receipt = {
            "version": VERSION,
            "status": "passed_native30_BC_independent_numeric_and_integrity_audit_not_admission",
            "passed": True,
            "result_root": str(result_dir),
            "result_COMMIT": commit_entry,
            "parent_freeze": freeze_entry,
            "prepared_contract": prepared_entry,
            "prepared_contract_value_sha256": contract_hash,
            "producer_version": PRODUCER_VERSION,
            "producer_runner": prepared["code"]["runner"],
            "audit_code": file_binding(Path(__file__).resolve()),
            "audit_tests": file_binding(AUDITOR_TESTS),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "soundfile": sf.__version__,
                        "platform": platform.platform()},
            "runtime_binding_sha256": value_hash(prepared["runtime"]),
            "scope": {
                "source_audio_read": True, "source_audio_modified": False,
                "result_products_modified": False, "classifier_fits": 0,
                "model_scoring": False, "classifier_admission": False,
                "BC_admission": False, "thresholds_changed": False,
            },
            "rows_expected": EXPECTED_COUNT, "rows_replayed": len(evidence),
            "ids_sha256": value_hash([row["id"] for row in prepared["rows"]]),
            "source_coverage": {source: SOURCE_COUNTS[source] for source in sorted(SOURCE_COUNTS)},
            "label_counts": LABEL_COUNTS, "origin_counts": ORIGIN_COUNTS,
            "scientific_null_count": sum(entry["scientific_null"] for entry in evidence),
            "scientific_null_ids": [entry["id"] for entry in evidence if entry["scientific_null"]],
            "full_grid_cells_per_item": 2 * len(GRID),
            "full_grid_cells_replayed": len(evidence) * 2 * len(GRID),
            "pools_replayed": len(evidence) * 2,
            "frames_replayed": len(evidence) * 2 * FRAMES,
            "numeric_tolerance": {"absolute": ATOL, "relative": RTOL,
                                  "masks_and_nulls": "exact"},
            "resampling_replay": "independent_manual_Kaiser5_FIR_plus_upfirdn_whole30_then_floor_center8",
            "channel_replay": "FLOAT32_storage_promoted_to_FLOAT64_then_explicit_(L+R)/2",
            "stft_replay": "explicit_frames_mean_subtraction_periodic_Hann_rfft_window_sum",
            "primitive_replay": "independent_F_contiguous_normalized_columns_binary_scaled_sums",
            "producer_extractor_primitive_scalar_imported": False,
            "all_grid_evidence_retained_and_checked": True,
            "nulls_retain_rows": True,
            "failures_or_incomplete_COMMIT": "audit fails; no success receipt",
            "product_inventory_sha256": value_hash(products),
            "product_count": len(products),
            "manifest": manifest_entry,
            "items": evidence,
        }
    # The only write performed by a successful audit is the caller-selected
    # receipt.  Publication is exclusive and canonical; failed runs write none.
    with output_path.open("xb") as stream:
        stream.write(canonical(receipt))
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--commit-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.result_dir, args.freeze, args.freeze_sha256,
                   args.commit, args.commit_sha256, args.output)
    print(canonical(result).decode().strip())


if __name__ == "__main__":
    main()
