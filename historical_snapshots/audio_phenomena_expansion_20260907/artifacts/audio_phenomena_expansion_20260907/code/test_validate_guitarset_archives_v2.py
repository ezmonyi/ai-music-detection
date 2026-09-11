#!/usr/bin/env python3
"""Offline synthetic tests only; these do not accept the real GuitarSet archives."""

from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings
import wave
import zipfile

import numpy as np

import validate_guitarset_archives_v2 as target

try:
    import soundfile
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def wav_bytes(values, sample_rate=8000):
    values = np.asarray(values, dtype="<i2")
    if values.ndim == 1:
        values = values[:, None]
    stream = io.BytesIO()
    with wave.open(stream, "wb") as handle:
        handle.setnchannels(values.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(values.tobytes(order="C"))
    return stream.getvalue()


def jams_bytes(duration=1.0, namespaces=("note_midi",)):
    return target.canonical_json({
        "file_metadata": {"duration": duration},
        "annotations": [
            {"namespace": namespace,
             "data": [{"time": 0.0, "duration": duration, "value": 60, "confidence": 1.0}]}
            for namespace in namespaces
        ],
    })


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SyntheticPublication:
    PLAYERS = ("00", "01")
    SCORES = ("Tiny-120-C",)
    EXTRA_PRODUCTS = {
        "audio_mono-mic.zip.headers.txt", "audio_mono-mic.zip.command.json",
        "annotation.zip.headers.txt", "annotation.zip.command.json",
        "audio_mono-mic.zip.attempts/00.headers.txt",
    }

    def __init__(self, root):
        self.root = Path(root)
        self.source = self.root / "acquired"
        self.output = self.root / "validated"
        self.source.mkdir()
        self.identities = [
            f"{player}_{score}_{performance}"
            for player in self.PLAYERS
            for score in self.SCORES
            for performance in target.PERFORMANCES
        ]
        audio_path = self.source / "audio_mono-mic.zip"
        annotation_path = self.source / "annotation.zip"
        values = np.array([0, 1, -2, 32767, -32768], dtype=np.int16)
        with zipfile.ZipFile(audio_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("audio_mono-mic/", b"")
            archive.writestr("__MACOSX/._ignored.wav", b"AppleDouble")
            for item_id in self.identities:
                archive.writestr(f"audio_mono-mic/{item_id}_mic.wav", wav_bytes(values))
        with zipfile.ZipFile(annotation_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("annotation/README.txt", b"synthetic test fixture")
            for item_id in self.identities:
                archive.writestr(f"annotation/{item_id}.jams",
                                 jams_bytes(5 / 8000, ("note_midi", "beat")))

        self.archive_expected = {}
        metadata_files = []
        receipt_archives = []
        for name in target.EXPECTED_ARCHIVES:
            data = (self.source / name).read_bytes()
            record = {"bytes": len(data), "md5": hashlib.md5(data).hexdigest()}
            self.archive_expected[name] = record
            metadata_files.append({
                "key": name,
                "size": record["bytes"],
                "checksum": "md5:" + record["md5"],
                "links": {"self": f"https://zenodo.org/api/records/3371780/files/{name}/content"},
            })
            receipt_archives.append({"name": name, **record, "sha256": sha(data)})
        metadata = {
            "doi": "10.5281/zenodo.3371780",
            "metadata": {"license": {"id": "cc-by-4.0"}, "access_right": "open",
                         "version": "1.1.0"},
            "files": metadata_files,
        }
        metadata_data = target.canonical_json(metadata)
        (self.source / "upstream_metadata.json").write_bytes(metadata_data)
        self.metadata_sha = sha(metadata_data)
        receipt = {
            "status": "archives_verified_not_decoded",
            "metadata_sha256": self.metadata_sha,
            "doi": "10.5281/zenodo.3371780",
            "license": "CC-BY-4.0",
            "archives": receipt_archives,
            "code_sha256": "a" * 64,
            "zip_crc_checked": False,
            "audio_decoded": False,
            "bc_measured": False,
            "classifier_fits": 0,
            "external_gate_passed": False,
        }
        (self.source / "acquisition_receipt.json").write_bytes(target.canonical_json(receipt))
        for archive_name in target.EXPECTED_ARCHIVES:
            (self.source / f"{archive_name}.headers.txt").write_text("synthetic HTTP headers\n")
            (self.source / f"{archive_name}.command.json").write_bytes(
                target.canonical_json({"synthetic": True, "archive": archive_name}))
        attempts = self.source / "audio_mono-mic.zip.attempts"
        attempts.mkdir()
        (attempts / "00.headers.txt").write_text("synthetic resumable request headers\n")
        products = {}
        for name in sorted(target.REQUIRED_SOURCE_PRODUCTS | self.EXTRA_PRODUCTS):
            data = (self.source / name).read_bytes()
            products[name] = {"bytes": len(data), "sha256": sha(data)}
        commit_data = target.canonical_json({
            "status": "committed", "kind": "archive_acquisition_only", "products": products,
        })
        (self.source / "COMMIT.json").write_bytes(commit_data)
        self.commit_sha = sha(commit_data)

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(target, "EXPECTED_ARCHIVES", self.archive_expected))
        stack.enter_context(patch.object(target, "METADATA_SHA256", self.metadata_sha))
        stack.enter_context(patch.object(target, "EXPECTED_PLAYERS", self.PLAYERS))
        stack.enter_context(patch.object(target, "EXPECTED_SCORE_COUNT", len(self.SCORES)))
        return stack


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="guitarset_validator_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def minimal_rows(players=("00", "01"), score="Tiny-120-C"):
        identities = [f"{player}_{score}_{performance}"
                      for player in players for performance in target.PERFORMANCES]
        audio = [{"path": f"root/{item}_mic.wav", "basename": f"{item}_mic.wav"}
                 for item in identities]
        annotations = [{"path": f"root/{item}.jams", "basename": f"{item}.jams"}
                       for item in identities]
        return audio, annotations

    def test_safe_member_traversal_backslash_nul_absolute(self):
        for name in ("../escape", "a/../../escape", "/absolute", "C:/absolute", "a\\b", "bad\x00name",
                     "a//b", "a/./b"):
            with self.subTest(name=repr(name)), self.assertRaises(ValueError):
                target.safe_member_name(name)

    def test_duplicate_and_symlink_zip_members_rejected(self):
        duplicate = self.root / "duplicate.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate, "w") as archive:
                archive.writestr("same", b"one")
                archive.writestr("same", b"two")
        with self.assertRaisesRegex(ValueError, "Duplicate ZIP"):
            target.scan_zip(duplicate)

        symlink = self.root / "symlink.zip"
        with zipfile.ZipFile(symlink, "w") as archive:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "target")
        with self.assertRaisesRegex(ValueError, "Symlink or special"):
            target.scan_zip(symlink)

    def test_zip_crc_corruption_detected_by_full_member_read(self):
        archive_path = self.root / "corrupt.zip"
        member_name = "audio/test.wav"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(member_name, b"this payload must be fully CRC checked")
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(member_name)
        raw = bytearray(archive_path.read_bytes())
        offset = info.header_offset
        self.assertEqual(raw[offset:offset + 4], b"PK\x03\x04")
        filename_length, extra_length = struct.unpack_from("<HH", raw, offset + 26)
        data_offset = offset + 30 + filename_length + extra_length
        raw[data_offset + 3] ^= 0x01
        archive_path.write_bytes(raw)
        with self.assertRaises(zipfile.BadZipFile):
            target.scan_zip(archive_path)

    def test_zip_inventory_records_directories_and_apple_metadata(self):
        archive_path = self.root / "inventory.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("root/", b"")
            archive.writestr("__MACOSX/._x.wav", b"resource")
            archive.writestr("root/x.wav", b"not really audio")
        rows = target.scan_zip(archive_path)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["crc_verified_by_full_read"] for row in rows))
        self.assertTrue(next(row for row in rows if row["path"] == "root/")["is_directory"])
        self.assertTrue(next(row for row in rows if "._x" in row["path"])["apple_metadata"])

    def test_exact_identity_join_and_optional_audio_mic_suffix(self):
        audio, annotations = self.minimal_rows()
        pairs = target.validate_media_join(audio, annotations, ("00", "01"), 1)
        self.assertEqual(len(pairs), 4)
        self.assertEqual({pair["item_id"] for pair in pairs},
                         {row["path"].split("/")[-1][:-5] for row in annotations})
        audio[0]["path"] = audio[0]["path"].replace("_mic.wav", ".wav")
        self.assertEqual(len(target.validate_media_join(audio, annotations, ("00", "01"), 1)), 4)

    def test_missing_audio_or_jams_fails_closed(self):
        audio, annotations = self.minimal_rows()
        with self.assertRaisesRegex(ValueError, "microphone WAV"):
            target.validate_media_join(audio[:-1], annotations, ("00", "01"), 1)
        with self.assertRaisesRegex(ValueError, "JAMS"):
            target.validate_media_join(audio, annotations[:-1], ("00", "01"), 1)
        mismatched = [dict(row) for row in annotations]
        mismatched[0]["path"] = mismatched[0]["path"].replace("Tiny-120-C", "Other-90-D")
        with self.assertRaises(ValueError):
            target.validate_media_join(audio, mismatched, ("00", "01"), 1)

    @unittest.skipUnless(HAS_SOUNDFILE, "requires actual SoundFile decoder; no fake success")
    def test_decode_reads_all_frames_and_canonical_pcm_sha(self):
        audio = self.root / "long.wav"
        values = ((np.arange(target.DECODE_FRAMES + 17, dtype=np.int64) % 65536) - 32768).astype(np.int16)
        audio.write_bytes(wav_bytes(values, sample_rate=11025))
        result = target.decode_audio(audio)
        expected = (values.astype(np.float64) / 32768.0).reshape(-1, 1)
        expected_sha = hashlib.sha256(np.asarray(expected, dtype="<f8", order="C").tobytes(order="C")).hexdigest()
        self.assertEqual(result["decoded_frames"], len(values))
        self.assertEqual(result["read_calls_including_empty_eof"], 3)
        self.assertTrue(result["empty_eof_observed"])
        self.assertEqual(result["decoded_pcm_sha256"], expected_sha)
        self.assertEqual(result["decoded_pcm_canonical_encoding"], target.PCM_ENCODING)

    def test_nonfinite_decode_rejected(self):
        audio = self.root / "fake.wav"
        audio.write_bytes(b"synthetic decoder input")

        class NonfiniteDecoder:
            def __init__(self, *_args, **_kwargs):
                self.samplerate = 8000
                self.channels = 1
                self.frames = 1
                self.format = "SYNTHETIC"
                self.subtype = "FLOAT"

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, *_args, **_kwargs):
                return np.array([[np.nan]], dtype=np.float64)

        fake = types.SimpleNamespace(SoundFile=NonfiniteDecoder)
        with patch.dict(sys.modules, {"soundfile": fake}):
            with self.assertRaisesRegex(ValueError, "Nonfinite decoded PCM"):
                target.decode_audio(audio)

    @unittest.skipUnless(HAS_SOUNDFILE, "requires actual SoundFile decoder; no fake success")
    def test_stereo_microphone_rejected(self):
        audio = self.root / "stereo.wav"
        audio.write_bytes(wav_bytes(np.array([[1, 2], [3, 4]], dtype=np.int16)))
        with self.assertRaisesRegex(ValueError, "non-mono"):
            target.decode_audio(audio)

    def test_jams_namespace_counts_and_no_fixed_namespace_cardinality(self):
        path = self.root / "tiny.jams"
        path.write_bytes(jams_bytes(2.5, ("note_midi", "note_midi", "beat")))
        parsed = target.parse_jams(path)
        self.assertEqual(parsed["file_metadata_duration_seconds"], 2.5)
        self.assertEqual(parsed["namespace_annotation_counts"], {"beat": 1, "note_midi": 2})
        self.assertEqual(parsed["namespace_observation_counts"], {"beat": 1, "note_midi": 2})

    @unittest.skipUnless(HAS_SOUNDFILE, "requires actual SoundFile decoder; no fake success")
    def test_full_synthetic_publication_json_and_commit_products(self):
        fixture = SyntheticPublication(self.root)
        with fixture.patches():
            summary = target.run(fixture.source, fixture.commit_sha, fixture.output)
        self.assertEqual(summary["counts"]["exact_audio_annotation_pairs"], 4)
        self.assertEqual(summary["counts"]["players"], 2)
        self.assertEqual(summary["counts"]["scores"], 1)
        self.assertIsNone(summary["class_label"])
        self.assertEqual(summary["role"], "external_measurement_control_only")
        self.assertFalse(summary["claims"]["overlap_clean"])
        self.assertFalse(summary["claims"]["external_measurement_gate_passed"])
        self.assertTrue((fixture.output / "source" / "upstream_metadata.json").read_bytes()
                        == (fixture.source / "upstream_metadata.json").read_bytes())
        self.assertTrue((fixture.output / "source" / "acquisition_COMMIT.json").read_bytes()
                        == (fixture.source / "COMMIT.json").read_bytes())
        records = [target.strict_json_bytes(line)
                   for line in (fixture.output / "source_manifest.jsonl").read_bytes().splitlines()]
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record["class_label"] is None for record in records))
        self.assertTrue(all(record["annotation_is_ground_truth_for_nonlinear_coupling"] is False
                            for record in records))
        commit = target.strict_json_bytes((fixture.output / "COMMIT.json").read_bytes())
        actual = {path.relative_to(fixture.output).as_posix()
                  for path in fixture.output.rglob("*") if path.is_file()
                  and path.name != "COMMIT.json"}
        self.assertEqual(set(commit["products"]), actual)
        for name, expected in commit["products"].items():
            self.assertEqual(target.file_fingerprint(fixture.output / name, False), expected)

    def test_wrong_source_commit_leaves_failure_log_and_no_commit(self):
        fixture = SyntheticPublication(self.root)
        with fixture.patches(), self.assertRaisesRegex(ValueError, "Source COMMIT"):
            target.run(fixture.source, "0" * 64, fixture.output)
        self.assertTrue((fixture.output / "EXECUTION_FAILURE.json").is_file())
        self.assertFalse((fixture.output / "COMMIT.json").exists())
        failure = target.strict_json_bytes((fixture.output / "EXECUTION_FAILURE.json").read_bytes())
        self.assertFalse(failure["commit_published"])

    def test_source_product_mutation_rejected(self):
        fixture = SyntheticPublication(self.root)
        path = fixture.source / "annotation.zip.headers.txt"
        path.write_text("changed after acquisition commit")
        with fixture.patches(), self.assertRaisesRegex(ValueError, "product bytes/hash"):
            target.run(fixture.source, fixture.commit_sha, fixture.output)
        self.assertFalse((fixture.output / "COMMIT.json").exists())

    def test_uncommitted_source_file_rejected_by_exact_inventory(self):
        fixture = SyntheticPublication(self.root)
        (fixture.source / "not-in-commit.log").write_text("must not be ignored")
        with fixture.patches(), self.assertRaisesRegex(ValueError, "exact product inventory"):
            target.run(fixture.source, fixture.commit_sha, fixture.output)
        self.assertFalse((fixture.output / "COMMIT.json").exists())

    def test_existing_output_preserved(self):
        fixture = SyntheticPublication(self.root)
        fixture.output.mkdir()
        sentinel = fixture.output / "user-owned.txt"
        sentinel.write_text("preserve exactly")
        with fixture.patches(), self.assertRaisesRegex(ValueError, "new exclusive"):
            target.run(fixture.source, fixture.commit_sha, fixture.output)
        self.assertEqual(sentinel.read_text(), "preserve exactly")
        self.assertEqual({path.name for path in fixture.output.iterdir()}, {"user-owned.txt"})


class ColumnJamsRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="guitarset_v2_jams_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def columns():
        # Shape observed in official 00_BN1-129-Eb_comp.jams; tiny synthetic values.
        return {"time": [0.2, 0.3], "duration": [0.0, 0.0],
                "value": [{"voiced": True, "index": 0, "frequency": 106.492},
                          {"voiced": True, "index": 0, "frequency": 105.43}],
                "confidence": [None, None]}

    def parse(self, data, namespace="pitch_contour"):
        path = self.root / "case.jams"
        path.write_bytes(target.canonical_json({"file_metadata": {"duration": 1.0},
                        "annotations": [{"namespace": namespace, "data": data}]}))
        return target.parse_jams(path)

    def test_official_shape_column_count_is_observations_not_four_keys(self):
        result = self.parse(self.columns())
        self.assertEqual(result["namespace_observation_counts"], {"pitch_contour": 2})
        self.assertEqual(result["namespace_annotation_counts"], {"pitch_contour": 1})
        self.assertEqual(result["annotation_data_layout_counts"], {"column_arrays": 1})

    def test_mixed_list_and_column_namespaces_counts(self):
        path = self.root / "mixed.jams"
        payload = {"file_metadata": {"duration": 1.0}, "annotations": [
            {"namespace": "pitch_contour", "data": self.columns()},
            {"namespace": "pitch_contour", "data": self.columns()},
            {"namespace": "note_midi", "data": [
                {"time": 0.0, "duration": 0.5, "value": 60.1, "confidence": None}]}]}
        path.write_bytes(target.canonical_json(payload))
        result = target.parse_jams(path)
        self.assertEqual(result["namespace_observation_counts"], {"pitch_contour": 4, "note_midi": 1})
        self.assertEqual(result["annotation_data_layout_counts"], {"column_arrays": 2, "observation_list": 1})

    def test_each_column_length_mismatch_rejected(self):
        for field in ("time", "duration", "value", "confidence"):
            columns = self.columns()
            columns[field].pop()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unequal lengths"):
                self.parse(columns)

    def test_missing_extra_or_nonarray_columns_rejected(self):
        for field in ("time", "duration", "value", "confidence"):
            columns = self.columns()
            del columns[field]
            with self.assertRaises(ValueError):
                self.parse(columns)
            columns = self.columns()
            columns[field] = "not an array"
            with self.assertRaisesRegex(ValueError, "must be JSON arrays"):
                self.parse(columns)
        columns = self.columns()
        columns["unexpected"] = [1, 2]
        with self.assertRaises(ValueError):
            self.parse(columns)

    def test_empty_columns_and_empty_observation_list(self):
        for data in ({field: [] for field in self.columns()}, []):
            self.assertEqual(self.parse(data)["namespace_observation_counts"], {"pitch_contour": 0})

    def test_malformed_rows_rejected(self):
        for data in ([0], [None], [{}], [{"time": 0, "duration": 0, "value": 1}],
                     [{"time": 0, "duration": 0, "value": 1, "confidence": None, "extra": 1}],
                     None, "not observations", 2):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.parse(data)

    def test_numeric_types_confidence_and_duration_validation(self):
        for field, value in (("time", True), ("time", "0.2"), ("time", None),
                             ("duration", -0.1), ("duration", False), ("duration", None),
                             ("confidence", True), ("confidence", "1"),
                             ("confidence", -0.1), ("confidence", 1.1)):
            columns = self.columns()
            columns[field][0] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.parse(columns)

    def test_nonfinite_nested_values_and_duplicate_keys_rejected(self):
        path = self.root / "malformed.jams"
        payload = {"file_metadata": {"duration": 1.0}, "annotations": [
            {"namespace": "pitch_contour", "data": self.columns()}]}
        raw = json.dumps(payload).replace("106.492", "1e309")
        path.write_text(raw)
        with self.assertRaisesRegex(ValueError, "Nonfinite JAMS observation value"):
            target.parse_jams(path)
        raw = json.dumps(payload).replace('"confidence": [null, null]',
                                          '"confidence": [null, null], "confidence": [null, null]')
        path.write_text(raw)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            target.parse_jams(path)

    def test_signed_timestamp_and_structured_value_preserved_not_repaired(self):
        columns = self.columns()
        columns["time"][0] = -0.001
        columns["value"][0] = {"nested": [1, "chord", True, None, {"x": 0.2}]}
        self.assertEqual(self.parse(columns)["namespace_observation_counts"]["pitch_contour"], 2)

    def test_raw_jams_bytes_preserved_through_materialization_and_parse(self):
        document = {"file_metadata": {"duration": 1}, "annotations": [
            {"namespace": "pitch_contour", "data": self.columns()}]}
        raw = ("  " + json.dumps(document, indent=3) + "\n\n").encode()
        archive_path = self.root / "annotations.zip"
        name = "annotation/00_Test-120-C_comp.jams"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(name, raw)
        member = target.scan_zip(archive_path)[0]
        output = self.root / "preserved.jams"
        with zipfile.ZipFile(archive_path) as archive:
            materialized = target.materialize_member(archive, member, output)
        result = target.parse_jams(output)
        self.assertEqual(result["namespace_observation_counts"]["pitch_contour"], 2)
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(materialized["sha256"], sha(raw))
        self.assertEqual(target.file_fingerprint(output, False), materialized)

    def test_version_two_self_fingerprint_and_schema_compatibility(self):
        self.assertEqual(target.VERSION, "2.0")
        self.assertEqual(target.CODE_PATH.name, "validate_guitarset_archives_v2.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
