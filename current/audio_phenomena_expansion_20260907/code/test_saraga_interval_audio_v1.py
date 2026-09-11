#!/usr/bin/env python3
"""Synthetic-only interval contract tests. No Saraga file is decoded here."""
from __future__ import annotations

import copy
import hashlib
import importlib
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    import scipy
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

audio = importlib.import_module("saraga_interval_audio_v1") if np is not None and sf is not None else None
READY = audio is not None


def _wave(start, count, channels=2):
    # Nonconstant, exactly representable float64 samples. Values exceed one.
    t = (np.arange(start, start + count, dtype=np.int64) % 1009).astype(np.float64) / 512
    return np.column_stack([t if channel % 2 == 0 else -t / 2 for channel in range(channels)])


def _hashes(data):
    return {"bytes": len(data), "md5": hashlib.md5(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest()}


def _consistent(measurement):
    m = measurement
    m.update({"sample_count": m["actual_frames"] * m["channels"],
              "finite_sample_count": m["actual_frames"] * m["channels"],
              "header_minus_actual_frames": m["header_frames"] - m["actual_frames"],
              "header_matches_actual_eof": m["header_frames"] == m["actual_frames"],
              "actual_duration_seconds": m["actual_frames"] / m["sample_rate"],
              "duration_at_least_60_seconds": m["actual_frames"] >= 60 * m["sample_rate"],
              "read_calls_including_empty_eof": math.ceil(m["actual_frames"] / 65536) + 1})


def _physical(raw, rate, frames, channels=2, header=None, fmt="MP3", subtype="MPEG_LAYER_III", waveform=_wave):
    pcm = hashlib.sha256()
    minimum, maximum, squares = math.inf, -math.inf, 0.0
    for start in range(0, frames, 65536):
        block = waveform(start, min(65536, frames - start), channels)
        pcm.update(block.astype("<f8").tobytes(order="C"))
        minimum = min(minimum, float(np.min(block)))
        maximum = max(maximum, float(np.max(block)))
        squares += float(np.sum(block * block))
    raw_hashes = _hashes(raw)
    m = {"sample_rate": rate, "channels": channels, "decoder_format": fmt, "decoder_subtype": subtype,
         "actual_frames": frames, "header_frames": frames if header is None else header,
         "real_empty_read_observed": True, "read_block_frames": 65536,
         "pcm_sha256": pcm.hexdigest(), "pcm_encoding": audio.PCM_ENCODING,
         "nonfinite_sample_count": 0, "sample_min": minimum, "sample_max": maximum,
         "peak_absolute": max(abs(minimum), abs(maximum)), "rms_all_samples": math.sqrt(squares / (frames * channels)),
         "raw_hashes_before_decode": copy.deepcopy(raw_hashes), "raw_hashes_after_decode": copy.deepcopy(raw_hashes)}
    _consistent(m)
    return {"status": audio.PHYSICAL_STATUS, "contract_sha256": "1" * 64,
            "classifier_admission": False, "cohort_selected": False,
            "item": {"mbid": "synthetic-fixture", "raw_path": "raw/synthetic.mp3",
                     "archive_member": {**raw_hashes, "path": "synthetic.mp3"},
                     "reconciled_metadata": {"advertised_length_ms": 60001}}, "measurement": m}


class FakeDecoder:
    def __init__(self, factory):
        self.factory = factory
        self.instance_index = len(factory.instances)
        self.samplerate = factory.rate
        self.channels = factory.channels
        self.frames = factory.header
        self.format, self.subtype = "MP3", "MPEG_LAYER_III"
        self.position = 0
        self.read_sizes = []
        self.empty_reads = 0
        self.sought = False
        factory.instances.append(self)

    def __enter__(self):
        if self.factory.on_open is not None:
            self.factory.on_open(self)
        return self

    def __exit__(self, *args):
        return False

    def seek(self, frame):
        if self.factory.seek_error:
            raise RuntimeError("synthetic decoder seek failure")
        self.sought = True
        self.position = frame + self.factory.seek_offset
        return frame if self.factory.seek_return is None else self.factory.seek_return

    def read(self, count, dtype, always_2d):
        if dtype != "float64" or always_2d is not True:
            raise AssertionError("Expected explicit float64 and always_2d")
        self.read_sizes.append(count)
        length = min(count, max(0, self.factory.actual_frames - self.position))
        if self.sought and self.factory.short_seek:
            length -= 1
        block = _wave(self.position, length, self.channels)
        self.position += length
        if not length:
            self.empty_reads += 1
        if length and self.factory.nonfinite:
            block[0, 0] = np.nan
        if length and self.sought and self.factory.tiny_seek_delta:
            block[0, 0] += 2 ** -40
        if self.factory.return_float32:
            block = block.astype(np.float32)
        return block


class FakeFactory:
    def __init__(self, record):
        m = record["measurement"]
        self.rate, self.channels = m["sample_rate"], m["channels"]
        self.actual_frames, self.header = m["actual_frames"], m["header_frames"]
        self.instances = []
        self.seek_offset = 0
        self.seek_return = None
        self.short_seek = self.seek_error = self.tiny_seek_delta = False
        self.nonfinite = self.return_float32 = False
        self.on_open = None

    def __call__(self, handle, mode):
        if mode != "r" or not hasattr(handle, "read"):
            raise AssertionError("Expected protected source handle, mode=r")
        return FakeDecoder(self)


@unittest.skipUnless(READY, "requires existing NumPy, SciPy, and SoundFile; no installs performed")
class IntervalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = b"synthetic compressed source fixture\n"
        cls.base = _physical(cls.raw, 44100, 60 * 44100 + 12345)

    def setUp(self):
        # realpath removes the macOS /var -> /private/var alias from temp roots.
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "synthetic.mp3"
        self.source.write_bytes(self.raw)
        self.record = copy.deepcopy(self.base)
        self.factory = FakeFactory(self.record)

    def invoke(self, record=None, source=None):
        with mock.patch.object(audio.sf, "SoundFile", self.factory):
            return audio.verify_center_interval(self.source if source is None else source,
                                                self.record if record is None else record)

    def expect_failure(self, text, **kwargs):
        with self.assertRaisesRegex(audio.IntervalVerificationError, text) as caught:
            self.invoke(**kwargs)
        self.assertEqual(caught.exception.audit["status"], "failed_interval_verification")
        return caught.exception.audit

    def test_exact_44100_crop_real_empty_eof_and_peak_preserved(self):
        native, standard, audit = self.invoke()
        expected_start = (self.factory.actual_frames - 60 * 44100) // 2
        self.assertEqual(native.shape, (2646000, 2))
        self.assertEqual(native.dtype, np.dtype("float64"))
        self.assertEqual(standard.dtype.str, "<f4")
        self.assertTrue(standard.flags.c_contiguous)
        np.testing.assert_array_equal(native[:20], _wave(expected_start, 20))
        np.testing.assert_array_equal(standard, native.astype("<f4"))
        self.assertGreater(np.max(standard), 1)
        self.assertEqual(len(self.factory.instances), 2)
        self.assertEqual(self.factory.instances[0].empty_reads, 1)
        self.assertEqual(set(self.factory.instances[0].read_sizes), {65536})
        self.assertEqual(self.factory.instances[1].read_sizes, [2646000])
        self.assertFalse(audit["resample_applied"])
        self.assertFalse(audit["classifier_admission"])
        self.assertEqual(audit["files_written"], 0)
        self.assertEqual(audit["native_float32_sha256"], audit["standardized_float32_sha256"])
        self.assertTrue(audit["seek_proof"]["exact_bytes_equal"])
        self.assertEqual(audit["raw_hashes_before"], audit["raw_hashes_after"])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["synthetic.mp3"])

    def test_full_envelope_hash_and_status(self):
        envelope = {"record": self.record, "record_sha256": audio._object_sha(self.record)}
        _, _, audit = self.invoke(record=envelope)
        self.assertTrue(audit["physical_receipt_envelope_verified"])
        envelope["record_sha256"] = "0" * 64
        self.expect_failure("Receipt digest mismatch", record=envelope)
        self.record["status"] = "approved"
        self.expect_failure("Invalid physical status")

    def test_physical_admission_flags_rejected(self):
        self.record["classifier_admission"] = True
        self.expect_failure("Invalid physical status")

    def test_raw_hashes_all_required(self):
        for field in ("bytes", "md5", "sha256"):
            with self.subTest(field=field):
                record = copy.deepcopy(self.record)
                value = len(self.raw) + 1 if field == "bytes" else "0" * (32 if field == "md5" else 64)
                record["item"]["archive_member"][field] = value
                record["measurement"]["raw_hashes_before_decode"][field] = value
                record["measurement"]["raw_hashes_after_decode"][field] = value
                self.expect_failure("Raw source bytes/MD5/SHA256", record=record)
        self.assertEqual(len(self.factory.instances), 0)

    def test_upstream_raw_hash_disagreement_not_weakened(self):
        self.record["measurement"]["raw_hashes_after_decode"]["md5"] = "0" * 32
        self.expect_failure("Inconsistent physical raw hashes")

    def test_pcm_hash_mismatch(self):
        self.record["measurement"]["pcm_sha256"] = "0" * 64
        audit = self.expect_failure("Whole float64 PCM SHA256")
        self.assertTrue(audit["real_empty_read_observed"])
        self.assertEqual(len(self.factory.instances), 1)

    def test_actual_frames_not_header_authority(self):
        self.record["measurement"]["actual_frames"] += 1
        _consistent(self.record["measurement"])
        audit = self.expect_failure("Actual EOF frame count")
        self.assertEqual(audit["observed_actual_frames"], self.factory.actual_frames)

    def test_phantom_header_discrepancy_passes_only_with_actual_eof(self):
        self.factory.header += 20164
        self.record["measurement"]["header_frames"] = self.factory.header
        _consistent(self.record["measurement"])
        _, _, audit = self.invoke()
        self.assertFalse(audit["header_matches_actual_eof"])
        self.assertEqual(audit["observed_header_minus_actual_frames"], 20164)
        self.assertEqual(audit["crop_start_frame"], 12345 // 2)
        self.assertEqual(audit["observed_actual_frames"], self.factory.actual_frames)
        self.assertEqual(audit["retained_crop_frames"], 2646000)
        self.assertEqual(self.factory.instances[0].empty_reads, 1)

    def test_changed_decoder_header_rejected(self):
        self.factory.header += 1
        self.expect_failure("Decoder header_frames")

    def test_decoder_rate_mismatch(self):
        self.factory.rate = 48000
        self.expect_failure("Decoder sample_rate")

    def test_decoder_channel_mismatch(self):
        self.factory.channels = 1
        self.expect_failure("Decoder channels")

    def test_unsupported_receipt_rate_and_channels(self):
        for key, value, message in [("sample_rate", 32000, "Native rate"), ("channels", 1, "stereo")]:
            record = copy.deepcopy(self.record)
            record["measurement"][key] = value
            _consistent(record["measurement"])
            self.expect_failure(message, record=record)

    def test_nonfinite_decoder_pcm(self):
        self.factory.nonfinite = True
        self.expect_failure("Nonfinite decoded PCM")

    def test_nonfinite_physical_evidence(self):
        self.record["measurement"]["nonfinite_sample_count"] = 1
        self.expect_failure("Inconsistent physical measurements")

    def test_short_source_before_read_or_after_true_eof(self):
        short = copy.deepcopy(self.record)
        short["measurement"]["actual_frames"] = 44100 * 60 - 1
        _consistent(short["measurement"])
        self.expect_failure("Short source", record=short)
        self.assertEqual(len(self.factory.instances), 0)
        self.factory.actual_frames = 44100 * 60 - 1
        self.expect_failure("Actual EOF frame count")

    def test_seek_offset_mismatch(self):
        self.factory.seek_offset = 1
        audit = self.expect_failure("Fresh seek float64 crop differs")
        self.assertFalse(audit["seek_proof"]["exact_array_equal"])

    def test_seek_float64_difference_hidden_by_float32_rejected(self):
        start = 12345 // 2
        sample = _wave(start, 1)[0, 0]
        self.assertEqual(np.float32(sample), np.float32(sample + 2 ** -40))
        self.assertNotEqual(sample, sample + 2 ** -40)
        self.factory.tiny_seek_delta = True
        audit = self.expect_failure("Fresh seek float64 crop differs")
        self.assertFalse(audit["seek_proof"]["exact_bytes_equal"])

    def test_decoder_returning_float32_rejected(self):
        self.factory.return_float32 = True
        self.expect_failure("Decoder must return float64")

    def test_seek_short_read_not_padded(self):
        self.factory.short_seek = True
        audit = self.expect_failure("Fresh seek returned an incomplete crop")
        self.assertEqual(audit["seek_proof"]["observed_frames"], 2645999)

    def test_seek_error_and_wrong_position_preserved(self):
        self.factory.seek_error = True
        audit = self.expect_failure("synthetic decoder seek failure")
        self.assertEqual(audit["error_type"], "RuntimeError")
        self.factory.seek_error = False
        self.factory.seek_return = 0
        self.expect_failure("Fresh seek returned the wrong frame")

    def test_symlink_file_and_parent_and_parent_traversal_rejected(self):
        link = self.root / "linked.mp3"
        link.symlink_to(self.source)
        self.expect_failure("Symlink component", source=link)
        directory_link = self.root / "linked-dir"
        directory_link.symlink_to(self.root, target_is_directory=True)
        self.expect_failure("Symlink component", source=directory_link / self.source.name)
        self.expect_failure("Parent traversal", source=self.root / "unused" / ".." / self.source.name)

    def test_hardlink_source_rejected(self):
        os.link(self.source, self.root / "hardlink.mp3")
        self.expect_failure("singly linked")

    def test_changed_stats_rejected_even_if_bytes_unchanged(self):
        def touch(decoder):
            if decoder.instance_index == 0:
                info = self.source.stat()
                os.utime(self.source, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        self.factory.on_open = touch
        self.expect_failure("Source stats changed during read")

    def test_changed_content_rejected(self):
        def mutate(decoder):
            if decoder.instance_index == 0:
                with self.source.open("r+b") as handle:
                    handle.write(b"X")
        self.factory.on_open = mutate
        self.expect_failure("Source stats changed during read")

    def test_swapped_source_inode_rejected(self):
        def replace(decoder):
            if decoder.instance_index == 0:
                other = self.root / "replacement.mp3"
                other.write_bytes(self.raw)
                os.replace(other, self.source)
        self.factory.on_open = replace
        self.expect_failure("Source (stats|path or stats) changed during read")

    def test_missing_empty_eof_evidence_rejected(self):
        self.record["measurement"]["real_empty_read_observed"] = False
        self.expect_failure("Invalid EOF/PCM evidence")


@unittest.skipUnless(READY, "real synthetic WAV tests require existing NumPy, SciPy, and SoundFile")
class RealSoundFileTests(unittest.TestCase):
    def test_real_double_wav_44100_and_48000_exact_lengths_and_resampling(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                source = Path(directory).resolve() / "synthetic-double.wav"
                frames = rate * 60 + 25
                data = _wave(0, frames)
                # A true DOUBLE WAV proves dtype-sensitive decode, seek and crop.
                data[0, 0] += 2 ** -40
                sf.write(source, data, rate, format="WAV", subtype="DOUBLE")
                record = _physical(source.read_bytes(), rate, frames, fmt="WAV", subtype="DOUBLE",
                                   waveform=lambda start, count, channels: data[start:start + count])
                native, standard, audit = audio.verify_center_interval(source, record)
                start = 25 // 2
                np.testing.assert_array_equal(native, data[start:start + rate * 60])
                self.assertEqual(standard.shape, (2646000, 2))
                self.assertTrue(np.isfinite(standard).all())
                self.assertGreater(float(np.max(standard)), 1.0)
                if rate == 44100:
                    expected = native.astype("<f4")
                    self.assertFalse(audit["resample_applied"])
                    self.assertEqual((audit["resample_up"], audit["resample_down"]), (1, 1))
                else:
                    expected = scipy.signal.resample_poly(native, 147, 160, axis=0,
                                                         window=("kaiser", 5.0), padtype="constant").astype("<f4")
                    self.assertTrue(audit["resample_applied"])
                    self.assertEqual((audit["resample_up"], audit["resample_down"]), (147, 160))
                np.testing.assert_array_equal(standard, expected)
                self.assertEqual(audit["runtime"]["libsndfile"], sf.__libsndfile_version__)
                self.assertEqual(audit["standardized_float32_sha256"], hashlib.sha256(expected.tobytes()).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
