#!/usr/bin/env python3
"""Prove one physical Saraga center-60-second interval; never write audio/files.

Public API: verify_center_interval(source_path, physical_receipt) ->
    (native_float64_stereo, standardized_float32_stereo, audit).
Accept a full physical receipt envelope or its exact `record` value. The caller
owns cohort/path/contract admission and publication; this module admits nothing.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf

VERSION = "saraga_interval_audio_v1"
BLOCK_FRAMES = 65536
SECONDS = 60
TARGET_RATE = 44100
TARGET_FRAMES = 2646000
PCM_ENCODING = "IEEE754 float64 little-endian; C order [frame, channel]; no header"
PHYSICAL_STATUS = "passed_physical_acquisition_not_audio_admission"
# This configuration is intentionally identical to materialize_equal60_inputs_v2.
# The additional Saraga input restriction is stereo only, at 44100 or 48000 Hz.
CONFIG = {
    "duration_sec": SECONDS, "sample_rate_hz": TARGET_RATE, "channels": 2,
    "mono_policy": "duplicate_channel", "multichannel_policy": "reject_above_2",
    "resampler": "scipy.signal.resample_poly", "window": ["kaiser", 5.0],
    "padtype": "constant", "short_input_padding": False,
    "dc_removal": False, "normalization": False, "limiting": False,
    "output_format": "WAV", "output_subtype": "FLOAT",
    "coordinate_system": "exact_frozen_native_crop_then_resample",
}
MEASUREMENT_KEYS = {
    "sample_rate", "channels", "decoder_format", "decoder_subtype", "header_frames",
    "actual_frames", "header_minus_actual_frames", "header_matches_actual_eof",
    "actual_duration_seconds", "duration_at_least_60_seconds", "read_calls_including_empty_eof",
    "real_empty_read_observed", "read_block_frames", "pcm_sha256", "pcm_encoding",
    "sample_count", "finite_sample_count", "nonfinite_sample_count", "sample_min", "sample_max",
    "peak_absolute", "rms_all_samples", "raw_hashes_before_decode", "raw_hashes_after_decode",
}
CODE_PATH = Path(__file__).absolute()
LOADED_CODE_SHA256 = hashlib.sha256(CODE_PATH.read_bytes()).hexdigest()


class IntervalVerificationError(ValueError):
    """Failure with partial proof, including the stage; no repaired output exists."""

    def __init__(self, message, audit=None):
        super().__init__(message)
        self.audit = audit if audit is not None else {}


def _require(condition, message):
    if not condition:
        raise IntervalVerificationError(message)


def _canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _object_sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def _stat_signature(info):
    return {"device": info.st_dev, "inode": info.st_ino, "mode": info.st_mode,
            "links": info.st_nlink, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def _path_without_symlinks(path):
    # Reject rather than normalize '..': normalization could hide a symlink.
    candidate = Path(path)
    _require(".." not in candidate.parts, "Parent traversal is forbidden")
    candidate = candidate.absolute()
    for part in [*reversed(candidate.parents), candidate]:
        _require(not part.is_symlink(), f"Symlink component is forbidden: {part}")
    return candidate


@contextmanager
def _open_source(path, expected_stat=None):
    """Anchor every open in directory descriptors, disallowing symlink traversal."""
    path = _path_without_symlinks(path)
    _require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
             "O_NOFOLLOW and O_DIRECTORY are required")
    directory_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    raw_fd = None
    try:
        for name in path.parts[1:-1]:
            next_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        raw_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        info = os.fstat(raw_fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                 "Source must be a regular, singly linked file")
        signature = _stat_signature(info)
        if expected_stat is not None:
            _require(signature == expected_stat, "Source stats changed before open")
        with os.fdopen(raw_fd, "rb") as handle:
            raw_fd = None
            yield handle, signature
            _require(_stat_signature(os.fstat(handle.fileno())) == signature,
                     "Source stats changed during read")
            _path_without_symlinks(path)
            _require(_stat_signature(path.stat()) == signature,
                     "Source path or stats changed during read")
    finally:
        if raw_fd is not None:
            os.close(raw_fd)
        os.close(directory_fd)


def _raw_hashes(path, expected_stat=None):
    md5, sha, count = hashlib.md5(), hashlib.sha256(), 0
    with _open_source(path, expected_stat) as (handle, signature):
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            count += len(block)
            md5.update(block)
            sha.update(block)
    _require(count == signature["bytes"], "Raw byte count differs from stat")
    return {"bytes": count, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}, signature


def _record(physical_receipt):
    _require(isinstance(physical_receipt, dict), "Physical receipt must be an object")
    envelope = "record" in physical_receipt or "record_sha256" in physical_receipt
    if envelope:
        _require(set(physical_receipt) == {"record", "record_sha256"}, "Invalid receipt envelope")
        record = physical_receipt["record"]
        _require(physical_receipt["record_sha256"] == _object_sha(record), "Receipt digest mismatch")
    else:
        record = physical_receipt
    _require(isinstance(record, dict) and set(record) == {
        "status", "contract_sha256", "item", "measurement", "classifier_admission", "cohort_selected"
    }, "Invalid physical record schema")
    _require(record["status"] == PHYSICAL_STATUS and record["classifier_admission"] is False
             and record["cohort_selected"] is False and _digest(record["contract_sha256"]),
             "Invalid physical status, admission flags, or contract digest")
    item, m = record["item"], record["measurement"]
    _require(isinstance(item, dict) and isinstance(item.get("archive_member"), dict),
             "Missing physical item/archive member")
    _require(isinstance(m, dict) and set(m) == MEASUREMENT_KEYS, "Invalid measurement schema")
    expected = {key: item["archive_member"].get(key) for key in ("bytes", "md5", "sha256")}
    _require(_integer(expected["bytes"], 1) and _digest(expected["md5"], 32)
             and _digest(expected["sha256"]), "Invalid physical raw hashes")
    _require(expected == m["raw_hashes_before_decode"] == m["raw_hashes_after_decode"],
             "Inconsistent physical raw hashes")
    for name in ("sample_rate", "channels", "actual_frames", "sample_count", "finite_sample_count"):
        _require(_integer(m[name], 1), f"Invalid measurement: {name}")
    _require(m["sample_rate"] in (44100, 48000), "Native rate must be 44100 or 48000 Hz")
    _require(m["channels"] == 2, "Native input must be stereo")
    _require(_integer(m["header_frames"]) and _integer(m["read_calls_including_empty_eof"], 2)
             and m["read_calls_including_empty_eof"] >= math.ceil(m["actual_frames"] / BLOCK_FRAMES) + 1
             and m["real_empty_read_observed"] is True and m["read_block_frames"] == BLOCK_FRAMES
             and m["pcm_encoding"] == PCM_ENCODING and _digest(m["pcm_sha256"])
             and isinstance(m["decoder_format"], str) and bool(m["decoder_format"])
             and isinstance(m["decoder_subtype"], str), "Invalid EOF/PCM evidence")
    _require(m["sample_count"] == m["finite_sample_count"] == m["actual_frames"] * m["channels"]
             and type(m["nonfinite_sample_count"]) is int and m["nonfinite_sample_count"] == 0
             and type(m["header_minus_actual_frames"]) is int
             and m["header_minus_actual_frames"] == m["header_frames"] - m["actual_frames"]
             and m["header_matches_actual_eof"] is (m["header_frames"] == m["actual_frames"])
             and m["actual_duration_seconds"] == m["actual_frames"] / m["sample_rate"]
             and m["duration_at_least_60_seconds"] is (m["actual_frames"] >= SECONDS * m["sample_rate"]),
             "Inconsistent physical measurements")
    for name in ("sample_min", "sample_max", "peak_absolute", "rms_all_samples"):
        _require(type(m[name]) in (int, float) and math.isfinite(m[name]), "Invalid amplitude statistics")
    _require(m["sample_min"] <= m["sample_max"] and 0 <= m["rms_all_samples"] <= m["peak_absolute"]
             and m["peak_absolute"] == max(abs(m["sample_min"]), abs(m["sample_max"])),
             "Inconsistent amplitude statistics")
    _require(m["actual_frames"] >= SECONDS * m["sample_rate"], "Short source: padding is forbidden")
    return record, expected, envelope


def _check_decoder(decoder, measurement):
    actual = {"sample_rate": int(decoder.samplerate), "channels": int(decoder.channels),
              "header_frames": int(decoder.frames), "decoder_format": decoder.format,
              "decoder_subtype": decoder.subtype}
    for key, value in actual.items():
        _require(value == measurement[key], f"Decoder {key} differs from physical receipt")
    return actual


def _check_block(block, maximum):
    _require(isinstance(block, np.ndarray) and block.dtype == np.dtype("float64")
             and block.ndim == 2 and block.shape[1] == 2 and block.shape[0] <= maximum,
             "Decoder must return float64 [frame, 2] within requested count")
    _require(np.isfinite(block).all(), "Nonfinite decoded PCM")


def _sample_sha(samples, dtype):
    return hashlib.sha256(np.asarray(samples, dtype=dtype, order="C").tobytes(order="C")).hexdigest()


def verify_center_interval(source_path, physical_receipt):
    """Return proven native float64 and exact-60s 44100-Hz stereo little-float32.

    Raises IntervalVerificationError (partial evidence in `.audit`) on any failed
    check/decoder error. Does not select a cohort, write a file, or admit a model
    input. A bare record's provenance must be pinned externally by its caller.
    """
    audit = {"schema_version": 1, "helper_version": VERSION,
             "status": "verification_in_progress", "stage": "physical_record",
             "classifier_admission": False, "cohort_selected": False,
             "files_written": 0}
    try:
        record, expected_raw, enveloped = _record(physical_receipt)
        m, item = record["measurement"], record["item"]
        path = _path_without_symlinks(source_path)
        audit.update({"source_audio_path": str(path), "mbid": item.get("mbid"),
                      "physical_contract_sha256": record["contract_sha256"],
                      "physical_record_sha256": _object_sha(record),
                      "physical_receipt_envelope_verified": enveloped,
                      "physical_raw_path": item.get("raw_path"),
                      "archive_member": item["archive_member"],
                      "catalog_advertised_length_ms": item.get("reconciled_metadata", {}).get("advertised_length_ms"),
                      "physical_header_frames": m["header_frames"],
                      "physical_actual_frames": m["actual_frames"],
                      "physical_actual_duration_seconds": m["actual_duration_seconds"],
                      "configuration": dict(CONFIG), "configuration_sha256": _object_sha(CONFIG),
                      "saraga_input_policy": {"channels": 2, "sample_rates": [44100, 48000]}})
        audit["stage"] = "raw_hashes_before"
        before_raw, signature = _raw_hashes(path)
        audit.update({"raw_hashes_before": before_raw, "source_stat_before": signature})
        _require(before_raw == expected_raw, "Raw source bytes/MD5/SHA256 differ from physical receipt")

        # The receipt provides provisional allocation/coordinates. No output is
        # accepted unless this fresh full-EOF scan independently reproduces them.
        rate, native_count = m["sample_rate"], SECONDS * m["sample_rate"]
        start = (m["actual_frames"] - native_count) // 2
        native = np.empty((native_count, 2), dtype=np.float64)
        retained = frames = calls = 0
        pcm = hashlib.sha256()
        audit["stage"] = "sequential_actual_eof"
        with _open_source(path, signature) as (handle, _):
            with sf.SoundFile(handle, mode="r") as decoder:
                header = _check_decoder(decoder, m)
                while True:
                    block = decoder.read(BLOCK_FRAMES, dtype="float64", always_2d=True)
                    calls += 1
                    _check_block(block, BLOCK_FRAMES)
                    if block.shape[0] == 0:
                        break
                    pcm.update(np.asarray(block, dtype="<f8", order="C").tobytes(order="C"))
                    left, right = max(frames, start), min(frames + len(block), start + native_count)
                    if left < right:
                        native[left - start:right - start] = block[left - frames:right - frames]
                        retained += right - left
                    frames += len(block)
        audit.update({"observed_header": header, "observed_actual_frames": frames,
                      "observed_actual_duration_seconds": frames / rate,
                      "observed_header_minus_actual_frames": header["header_frames"] - frames,
                      "header_matches_actual_eof": header["header_frames"] == frames,
                      "real_empty_read_observed": True, "read_block_frames": BLOCK_FRAMES,
                      "read_calls_including_empty_eof": calls, "whole_pcm_sha256": pcm.hexdigest(),
                      "whole_pcm_encoding": PCM_ENCODING,
                      "crop_start_frame": start, "crop_frames": native_count,
                      "crop_end_frame_exclusive": start + native_count,
                      "retained_crop_frames": retained, "finite_sample_count": frames * 2})
        _require(frames == m["actual_frames"], "Actual EOF frame count differs from physical receipt")
        _require(pcm.hexdigest() == m["pcm_sha256"], "Whole float64 PCM SHA256 differs from physical receipt")
        _require(retained == native_count and start == (frames - native_count) // 2,
                 "Incomplete or noncentral native crop")
        audit["native_float64_sha256"] = _sample_sha(native, "<f8")

        audit["stage"] = "fresh_seek_equality"
        with _open_source(path, signature) as (handle, _):
            with sf.SoundFile(handle, mode="r") as decoder:
                _check_decoder(decoder, m)
                seek_position = decoder.seek(start)
                _require(seek_position == start, "Fresh seek returned the wrong frame position")
                sought = decoder.read(native_count, dtype="float64", always_2d=True)
                _check_block(sought, native_count)
        seek_hash = _sample_sha(sought, "<f8")
        equal_shape = sought.shape == native.shape
        equal_values = equal_shape and np.array_equal(sought, native)
        # Byte equality additionally distinguishes positive and negative zero.
        equal_bytes = equal_shape and np.array_equal(
            np.ascontiguousarray(sought, dtype="<f8").view(np.uint8),
            np.ascontiguousarray(native, dtype="<f8").view(np.uint8))
        audit["seek_proof"] = {"fresh_decoder": True, "requested_start_frame": start,
                               "returned_start_frame": int(seek_position), "requested_frames": native_count,
                               "observed_frames": int(len(sought)), "float64_sha256": seek_hash,
                               "exact_array_equal": bool(equal_values), "exact_bytes_equal": bool(equal_bytes)}
        _require(equal_shape, "Fresh seek returned an incomplete crop")
        _require(equal_values and equal_bytes, "Fresh seek float64 crop differs from sequential crop")
        del sought

        audit["stage"] = "standardization"
        divisor = math.gcd(rate, TARGET_RATE)
        up, down = TARGET_RATE // divisor, rate // divisor
        if rate == TARGET_RATE:
            standard64 = native
        else:
            standard64 = resample_poly(native, up, down, axis=0,
                                       window=("kaiser", 5.0), padtype="constant")
        _require(standard64.shape == (TARGET_FRAMES, 2) and np.isfinite(standard64).all(),
                 "Standardized waveform is not finite exact 60-second stereo")
        standard32 = np.ascontiguousarray(standard64, dtype="<f4")
        _require(np.isfinite(standard32).all(), "Float32 conversion produced nonfinite samples")
        audit.update({"native_float32_sha256": _sample_sha(native, "<f4"),
                      "native_sample_rate_hz": rate, "native_channels": 2,
                      "native_float64_encoding": PCM_ENCODING,
                      "native_float32_encoding": "IEEE754 float32 little-endian; C order [frame, channel]; no header",
                      "native_peak_absolute": float(np.max(np.abs(native))),
                      "native_samples_abs_above_one": int(np.count_nonzero(np.abs(native) > 1)),
                      "resample_up": up, "resample_down": down,
                      "resample_applied": rate != TARGET_RATE,
                      "standardized_sample_rate_hz": TARGET_RATE, "standardized_frames": TARGET_FRAMES,
                      "standardized_channels": 2, "standardized_dtype": standard32.dtype.str,
                      "standardized_float32_sha256": _sample_sha(standard32, "<f4"),
                      "standardized_peak_absolute": float(np.max(np.abs(standard32))),
                      "standardized_samples_abs_above_one": int(np.count_nonzero(np.abs(standard32) > 1))})
        audit["stage"] = "raw_hashes_after"
        after_raw, after_signature = _raw_hashes(path, signature)
        audit.update({"raw_hashes_after": after_raw, "source_stat_after": after_signature})
        _require(after_raw == before_raw == expected_raw, "Raw source content changed during verification")
        helper_sha = hashlib.sha256(CODE_PATH.read_bytes()).hexdigest()
        _require(helper_sha == LOADED_CODE_SHA256, "Helper code changed after import")
        audit["runtime"] = {"python": sys.version, "python_executable": sys.executable,
                            "platform": platform.platform(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "soundfile": sf.__version__,
                            "libsndfile": sf.__libsndfile_version__,
                            "helper_code_path": str(CODE_PATH), "helper_code_sha256": helper_sha}
        audit.update({"stage": "complete", "status": "passed_interval_proof_not_classifier_admission"})
        return native, standard32, audit
    except Exception as error:
        audit.update({"status": "failed_interval_verification", "error_type": type(error).__name__,
                      "error": str(error)})
        if isinstance(error, IntervalVerificationError):
            error.audit = audit
            raise
        raise IntervalVerificationError(f"{audit['stage']}: {error}", audit) from error
