#!/usr/bin/env python3
"""CPU-only synthetic fixtures; never opens the real Saraga archive or audio."""
import copy
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave
import zipfile

import numpy as np

import materialize_saraga_hindustani_v1 as m


def synthetic_wav(values, rate=8000):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as handle:
        handle.setnchannels(values.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(values, dtype="<i2").tobytes())
    return stream.getvalue()


class SyntheticWaveDecoder:
    """A mock SoundFile API backed only by a tiny stdlib-created PCM WAV."""
    def __init__(self, path, mode="r"):
        assert mode == "r"
        self.handle = wave.open(path, "rb")
        self.samplerate = self.handle.getframerate()
        self.channels = self.handle.getnchannels()
        self.frames = self.handle.getnframes()
        self.format, self.subtype = "SYNTHETIC_WAV_MOCK", "PCM_16"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.handle.close()

    def read(self, frames, dtype, always_2d):
        assert frames == 65536 and dtype == "float64" and always_2d is True
        return np.frombuffer(self.handle.readframes(frames), dtype="<i2").reshape(
            -1, self.channels).astype(np.float64) / 32768.0


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="saraga_synthetic_test_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.archive = self.base / "synthetic.zip"
        self.recon_path = self.base / "reconciliation.json"
        self.audit_path = self.base / "audit.json"
        self.output = self.base / "physical"
        self.values = np.array([[0, 1000], [-3000, 2000], [32767, -32768]], dtype=np.int16)
        self.wav = synthetic_wav(self.values)
        self.names = [f"{m.ROOT}/synthetic_{i}.mp3.mp3" for i in range(2)]
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as source:
            for i, name in enumerate(self.names):
                source.writestr(name, synthetic_wav(self.values + i, rate=8000 + i))
                source.writestr("__MACOSX/._" + name.rsplit("/", 1)[-1], b"synthetic AppleDouble " + bytes([i]))
        records = []
        with zipfile.ZipFile(self.archive) as source:
            for info in source.infolist():
                data = source.read(info)
                records.append({"path": info.filename, "bytes": len(data),
                                "compressed_bytes": info.compress_size,
                                "crc32": f"{info.CRC:08x}", "crc_verified": True,
                                "md5": hashlib.md5(data).hexdigest(),
                                "sha256": hashlib.sha256(data).hexdigest()})
        ah = m.hashes(self.archive)
        self.audit = {"status": "passed_archive_integrity_not_audio_admission", "source_record": 4301737,
                      "archive_bytes": ah["bytes"], "archive_hashes": {k: ah[k] for k in ("md5", "sha256")},
                      "classifier_admission": False, "extracted_audio_files": 0,
                      "physically_decoded_audio_files": 0, "records": records,
                      "archive_members_verified": len(records), "mp3_members": 4}
        self.audit_path.write_bytes(m.canonical(self.audit))
        audit_sha = m.hashes(self.audit_path)["sha256"]
        self.recon = {"status": "passed_archive_catalog_checksum_path_reconciliation_v2_not_audio_admission",
                      "failure_reasons": [], "physical_audio_decoded": False, "classifier_admission": False,
                      "cohort_selected": False, "tracks": [], "matched_audio": [],
                      "archive_integrity_evidence": {"archive_bytes": ah["bytes"],
                                                     "archive_hashes": self.audit["archive_hashes"]},
                      "inputs": {"archive_audit": {"sha256": audit_sha}}}
        for i, name in enumerate(self.names):
            mbid = f"00000000-0000-0000-0000-{i:012d}"
            record = next(r for r in records if r["path"] == name)
            track = {"mbid": mbid, "archive_audio_path": name, "path_resolution": "exact_unique_mp3_md5",
                     "speech_title_review_flag": i == 1, "performer_credits": [] if i == 1 else [{"synthetic": True}]}
            match = {"mbid": mbid, "archive_audio_path": name, "path_resolution": "exact_unique_mp3_md5",
                     "catalog_md5": record["md5"], "archive_md5": record["md5"]}
            self.recon["tracks"].append(track)
            self.recon["matched_audio"].append(match)
        self.recon_path.write_bytes(m.canonical(self.recon))
        self.stack.enter_context(patch.multiple(m, COUNT=2, ARCHIVE_BYTES=ah["bytes"],
                                               ARCHIVE_MD5=ah["md5"], ARCHIVE_SHA256=ah["sha256"],
                                               AUDIT_SHA256=audit_sha,
                                               RECON_SHA256=m.hashes(self.recon_path)["sha256"]))
        self.stack.enter_context(patch.object(m, "runtime_info", return_value={"synthetic_mock_runtime": True}))
        self.stack.enter_context(patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(SoundFile=SyntheticWaveDecoder)}))

    def run_fixture(self, workers=1):
        return m.run(self.archive, self.recon_path, self.audit_path, self.output, workers)

    def test_synthetic_full_pipeline_and_resume_no_decode(self):
        summary = self.run_fixture(workers=2)
        self.assertEqual(summary["record_count"], 2)
        self.assertFalse(summary["classifier_admission"])
        self.assertEqual(summary["speech_title_review_flags"], 1)
        self.assertEqual(summary["missing_performer_credits"], 1)
        before = {str(p): m.hashes(p) for p in self.output.rglob("*") if p.is_file()}
        with patch.object(m, "decode_to_eof", side_effect=AssertionError("Resume must not decode")):
            self.assertEqual(self.run_fixture(), summary)
        self.assertEqual(before, {str(p): m.hashes(p) for p in self.output.rglob("*") if p.is_file()})
        first = summary["records"][0]["measurement"]
        pcm = self.values.astype(np.float64) / 32768
        self.assertEqual(first["pcm_sha256"], hashlib.sha256(pcm.astype("<f8").tobytes()).hexdigest())
        self.assertAlmostEqual(first["rms_all_samples"], float(np.sqrt(np.mean(pcm**2))))
        self.assertEqual(first["actual_frames"], 3)
        self.assertEqual(first["sample_count"], 6)
        self.assertEqual(first["sample_rate"], 8000)
        self.assertEqual(summary["records"][1]["measurement"]["sample_rate"], 8001)
        self.assertEqual({p.name for p in (self.output / "raw").iterdir()},
                         {t["mbid"] + ".mp3" for t in self.recon["tracks"]})

    def test_wrong_input_pin_and_whole_archive_hash_fail_before_output(self):
        with patch.object(m, "RECON_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Pinned input hash"):
                self.run_fixture()
        self.assertFalse(self.output.exists())
        data = bytearray(self.archive.read_bytes())
        data[50] ^= 1
        self.archive.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Whole archive size/hash"):
            self.run_fixture()
        self.assertFalse(self.output.exists())

    def test_duplicate_ids_and_checksums_rejected(self):
        bad = copy.deepcopy(self.recon)
        bad["tracks"][1]["mbid"] = bad["tracks"][0]["mbid"]
        with self.assertRaisesRegex(ValueError, "Duplicate/invalid metadata"):
            m.validate_map(bad, self.audit)
        audit = copy.deepcopy(self.audit)
        audio = [r for r in audit["records"] if r["path"] in self.names]
        audio[1]["md5"] = audio[0]["md5"]
        bad = copy.deepcopy(self.recon)
        bad["matched_audio"][1]["catalog_md5"] = bad["matched_audio"][1]["archive_md5"] = audio[0]["md5"]
        with self.assertRaisesRegex(ValueError, "Duplicate audio checksum"):
            m.validate_map(bad, audit)

    def test_path_traversal_and_unsafe_mbid_rejected(self):
        for name in ("../oops", "/absolute", "a/../../oops", "a\\b", "a//b", "a/./b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.safe_member(name)
        bad = copy.deepcopy(self.recon)
        for key in ("tracks", "matched_audio"):
            bad[key][0]["mbid"] = "../../escape"
        with self.assertRaisesRegex(ValueError, "Unsafe/noncanonical"):
            m.validate_map(bad, self.audit)

    def test_zip_crc_corruption_never_publishes_raw(self):
        items, _ = m.validate_map(self.recon, self.audit)
        with zipfile.ZipFile(self.archive) as source:
            info = source.getinfo(self.names[0])
            offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        data = bytearray(self.archive.read_bytes())
        data[offset + 45] ^= 1
        self.archive.write_bytes(data)
        raw = self.base / "corrupt.mp3"
        with self.assertRaises(zipfile.BadZipFile):
            m.extract_audio(self.archive, items[0], raw)
        self.assertFalse(raw.exists())

    def test_extraction_wrong_expected_hash_and_no_overwrite(self):
        items, _ = m.validate_map(self.recon, self.audit)
        bad = copy.deepcopy(items[0])
        bad["archive_member"]["sha256"] = "0" * 64
        raw = self.base / "wrong.mp3"
        with self.assertRaisesRegex(ValueError, "Extracted member hash"):
            m.extract_audio(self.archive, bad, raw)
        self.assertFalse(raw.exists())
        raw.write_bytes(b"user-owned")
        with self.assertRaisesRegex(ValueError, "Raw destination conflict"):
            m.extract_audio(self.archive, items[0], raw)
        self.assertEqual(raw.read_bytes(), b"user-owned")

    def test_false_header_short_reads_and_true_empty_eof(self):
        raw = self.base / "synthetic.wav"
        raw.write_bytes(self.wav)
        for fake_header in (1, 900000000):
            class FalseHeader(SyntheticWaveDecoder):
                calls = 0
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.frames = fake_header
                def read(self, frames, dtype, always_2d):
                    self.calls += 1
                    data = np.frombuffer(self.handle.readframes(1), dtype="<i2")
                    return data.reshape(-1, self.channels).astype(np.float64) / 32768
            with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(SoundFile=FalseHeader)}):
                measurement = m.decode_to_eof(raw)
            self.assertEqual(measurement["actual_frames"], 3)
            self.assertEqual(measurement["header_frames"], fake_header)
            self.assertEqual(measurement["read_calls_including_empty_eof"], 4)
            self.assertTrue(measurement["real_empty_read_observed"])
            self.assertFalse(measurement["header_matches_actual_eof"])

    def test_nonfinite_and_empty_decoder_fail_without_summary(self):
        for value in (float("nan"), float("inf")):
            class Nonfinite(SyntheticWaveDecoder):
                def read(self, *args, **kwargs):
                    return np.full((1, self.channels), value, dtype=np.float64)
            with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(SoundFile=Nonfinite)}):
                raw = self.base / "nonfinite.wav"
                raw.write_bytes(self.wav)
                with self.assertRaisesRegex(ValueError, "Nonfinite decoded PCM"):
                    m.decode_to_eof(raw)
        empty = self.base / "empty.wav"
        empty.write_bytes(synthetic_wav(np.empty((0, 1), dtype=np.int16)))
        with self.assertRaisesRegex(ValueError, "Empty decoded"):
            m.decode_to_eof(empty)

    def test_failed_decode_leaves_explicit_conflict_and_no_summary(self):
        with patch.object(m, "decode_to_eof", side_effect=ValueError("synthetic decode failure")):
            with self.assertRaisesRegex(ValueError, "synthetic decode failure"):
                self.run_fixture()
        self.assertFalse((self.output / "summary.json").exists())
        self.assertFalse((self.output / "writer.lock").exists())
        with self.assertRaisesRegex(ValueError, "Raw file without receipt"):
            self.run_fixture()

    def test_lock_and_unexpected_inventory_fail(self):
        self.output.mkdir()
        with m.writer_lock(self.output):
            with self.assertRaises(FileExistsError):
                with m.writer_lock(self.output):
                    self.fail("Acquired a second writer lock")
        (self.output / "unexpected.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "Unexpected output-root"):
            self.run_fixture()
        self.assertEqual((self.output / "unexpected.txt").read_text(), "keep")

    def test_symlink_parent_raw_and_archive_member_rejected(self):
        linked = self.base / "linked"
        linked.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink forbidden"):
            m.safe_path(linked / "child")
        items, records = m.validate_map(self.recon, self.audit)
        symlink_zip = self.base / "symlink.zip"
        with zipfile.ZipFile(symlink_zip, "w") as source:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            source.writestr(info, "target")
        with self.assertRaisesRegex(ValueError, "Symlink/special"):
            m.validate_zip(symlink_zip, records)
        self.run_fixture()
        raw = self.output / items[0]["raw_path"]
        raw.unlink()
        raw.symlink_to(self.archive)
        with self.assertRaisesRegex(ValueError, "Symlink forbidden"):
            self.run_fixture()

    def test_resume_detects_raw_receipt_and_code_changes(self):
        self.run_fixture()
        with patch.object(m, "LOADED_CODE_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Loaded code changed"):
                self.run_fixture()
        receipt_path = next((self.output / "receipts").iterdir())
        receipt = m.read_json(receipt_path)
        receipt["record"]["measurement"]["actual_frames"] += 1
        receipt_path.write_bytes(m.canonical(receipt))
        with self.assertRaisesRegex(ValueError, "Receipt digest mismatch"):
            self.run_fixture()

    def test_resume_rejects_raw_hash_and_runtime_contract_changes(self):
        self.run_fixture()
        with patch.object(m, "runtime_info", return_value={"different_runtime": True}):
            with self.assertRaisesRegex(ValueError, "Existing output contract conflict"):
                self.run_fixture()
        raw = next((self.output / "raw").iterdir())
        data = bytearray(raw.read_bytes())
        data[-1] ^= 1
        raw.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Receipt/raw content mismatch"):
            self.run_fixture()

    def test_json_publication_never_overwrites_and_staging_is_explicit_conflict(self):
        target = self.base / "existing.json"
        m.write_new_json(target, {"original": True})
        with self.assertRaises(FileExistsError):
            m.write_new_json(target, {"replacement": True})
        self.assertEqual(m.read_json(target), {"original": True})
        self.run_fixture()
        (self.output / "raw" / ".abandoned.pending").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "staging files require review"):
            self.run_fixture()

    def test_success_summary_is_not_published_if_source_changes_at_end(self):
        original = m.decode_to_eof
        def mutate_source(path):
            result = original(path)
            self.archive.write_bytes(self.archive.read_bytes() + b"synthetic mutation")
            return result
        with patch.object(m, "decode_to_eof", side_effect=mutate_source):
            with self.assertRaisesRegex(ValueError, "Source archive changed"):
                self.run_fixture()
        self.assertFalse((self.output / "summary.json").exists())

    def test_receipt_schema_and_json_safety(self):
        summary = self.run_fixture()
        item = summary["records"][0]["item"]
        receipt = {"record": summary["records"][0], "record_sha256": m.object_sha(summary["records"][0])}
        raw_hash = m.hashes(self.output / item["raw_path"])
        for name, value in (("sample_rate", True), ("actual_frames", 3.5),
                            ("real_empty_read_observed", False), ("pcm_sha256", "bad"),
                            ("rms_all_samples", float("inf")), ("sample_count", 1)):
            bad = copy.deepcopy(receipt)
            bad["record"]["measurement"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                bad["record_sha256"] = m.object_sha(bad["record"])
                m.validate_receipt(bad, item, summary["contract_sha256"], raw_hash)
        for text in ('{"x":1,"x":2}', '{"x":NaN}'):
            path = self.base / "bad.json"
            path.write_text(text)
            with self.assertRaises(ValueError):
                m.read_json(path)

    def test_member_inventory_crc_and_worker_limits(self):
        _, records = m.validate_map(self.recon, self.audit)
        bad = copy.deepcopy(records)
        bad[self.names[0]]["crc32"] = "00000000"
        with self.assertRaisesRegex(ValueError, "ZIP member directory mismatch"):
            m.validate_zip(self.archive, bad)
        bad.pop(self.names[0])
        with self.assertRaisesRegex(ValueError, "ZIP file inventory"):
            m.validate_zip(self.archive, bad)
        for workers in (0, 5, True):
            with self.assertRaisesRegex(ValueError, "Workers must"):
                self.run_fixture(workers)

    def test_source_mutation_during_decode_rejected(self):
        raw = self.base / "mutating.wav"
        raw.write_bytes(self.wav)
        class Mutating(SyntheticWaveDecoder):
            def __exit__(self, *args):
                super().__exit__(*args)
                raw.write_bytes(raw.read_bytes() + b"changed")
        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(SoundFile=Mutating)}):
            with self.assertRaisesRegex(ValueError, "Raw source changed"):
                m.decode_to_eof(raw)


class OptionalRealSoundFileSyntheticTest(unittest.TestCase):
    def test_real_soundfile_on_synthetic_wav_only(self):
        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("SoundFile absent; real-decoder synthetic test requires existing runtime")
        with tempfile.TemporaryDirectory(prefix="saraga_real_sf_synthetic_") as directory:
            raw = Path(directory).resolve() / "synthetic.wav"
            values = np.arange(70000, dtype=np.int16).reshape(-1, 1)
            raw.write_bytes(synthetic_wav(values, rate=11025))
            result = m.decode_to_eof(raw)
            self.assertEqual(result["actual_frames"], 70000)
            self.assertEqual(result["channels"], 1)
            self.assertEqual(result["sample_rate"], 11025)
            self.assertEqual(result["read_calls_including_empty_eof"], 3)
            self.assertTrue(result["real_empty_read_observed"])
            self.assertEqual(result["pcm_sha256"], hashlib.sha256(
                (values.astype(np.float64) / 32768).astype("<f8").tobytes()).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
