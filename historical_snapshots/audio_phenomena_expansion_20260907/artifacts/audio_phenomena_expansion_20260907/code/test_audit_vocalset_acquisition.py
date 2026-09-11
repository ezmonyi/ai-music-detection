import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zlib

from audit_vocalset_acquisition import (
    AuditPrerequisiteError,
    audit_acquisition,
    merge_intervals,
    sha256_file,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_wav(path, frames=1600, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frames)


class VocalSetAcquisitionAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.acquisition = self.root / "acquisition"
        self.labels = self.root / "labels_source"
        self.acquisition.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def make_fixture(self):
        specifications = [
            ("f1_caro_straight.wav", "festus", [(.01, .06, "high"), (.03, .08, "medium")]),
            ("m1_scales_straight_a.wav", "", []),
        ]
        manifest = []
        annotation_rows = []
        for name, labeler, events in specifications:
            label_path = self.labels / "labels" / "vocalset" / (Path(name).stem + ".breath.json")
            label = {
                "audio_file": name,
                "duration_sec": .1,
                "labeler": labeler,
                "review_time_sec": 10,
                "breath_events": [
                    {"start_sec": start, "end_sec": end, "confidence": confidence}
                    for start, end, confidence in events
                ],
                "silent_breaths": [],
                "uncertain": [],
                "hard_negatives": [],
                "exhales": [],
            }
            write_json(label_path, label)
            annotation_sha = sha256_file(label_path)
            audio_path = self.acquisition / "audio" / name
            make_wav(audio_path)
            audio_raw = audio_path.read_bytes()
            row = {
                "filename": name,
                "singer": name.split("_", 1)[0],
                "duration_sec": .1,
                "annotation_path": str(label_path.resolve()),
                "annotation_sha256": annotation_sha,
                "high_medium_breaths": len(events),
                "prefill": "fixture",
                "labeler": labeler,
                "audio_path": str(audio_path.resolve()),
                "audio_sha256": hashlib.sha256(audio_raw).hexdigest(),
                "archive_member": "FULL/" + name,
                "member_crc32": f"{zlib.crc32(audio_raw) & 0xffffffff:08x}",
                "member_bytes": len(audio_raw),
                "sample_rate": 16000,
                "frames": 1600,
                "channels": 1,
                "duration_verified": .1,
            }
            manifest.append(row)
            annotation_rows.append({
                "filename": name,
                "singer": row["singer"],
                "annotation_path": str(label_path.resolve()),
                "annotation_sha256": annotation_sha,
            })
            write_json(self.acquisition / "receipts" / (name + ".json"), row)
        annotation_audit = {
            "selected_labels": 2,
            "labels": annotation_rows,
            "whole_archive_hash_verified": False,
            "archive_reported_md5": "fixture",
        }
        write_json(self.acquisition / "annotation_audit.json", annotation_audit)
        summary = {
            "status": "complete_accounting",
            "selected": 2,
            "downloaded_verified": 2,
            "errors": [],
            "manifest": manifest,
            "whole_archive_hash_verified": False,
            "annotation_audit_sha256": sha256_file(self.acquisition / "annotation_audit.json"),
        }
        write_json(self.acquisition / "acquisition_summary.json", summary)

    def expected(self):
        return {
            "selected": 2,
            "singers": 2,
            "named_reviewer_primary_clips": 1,
            "primary_raw_high_medium_events": 2,
            "primary_merged_high_medium_events": 1,
            "primary_zero_event_clips": 0,
        }

    def test_complete_fixture_passes_and_merges_overlap(self):
        self.make_fixture()
        result = audit_acquisition(self.acquisition, self.labels, self.expected())
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["observed"]["overlapping_raw_primary_clips"], 1)
        self.assertEqual(result["observed"]["empty_reviewer_clips"], 1)
        self.assertEqual(result["reviewer_strata"], {"empty_positive_time": 1, "named_positive_time": 1})

    def test_audio_corruption_fails_hash_crc_and_pcm(self):
        self.make_fixture()
        path = self.acquisition / "audio" / "m1_scales_straight_a.wav"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        result = audit_acquisition(self.acquisition, self.labels, self.expected())
        self.assertFalse(result["gate_passed"])
        checks = {error["check"] for error in result["errors"]}
        self.assertIn("audio_sha256", checks)
        self.assertIn("member_crc32", checks)

    def test_missing_summary_produces_no_audit_result(self):
        with self.assertRaises(AuditPrerequisiteError):
            audit_acquisition(self.acquisition, self.labels, self.expected())

    def test_named_alternate_summary_is_supported(self):
        self.make_fixture()
        original = self.acquisition / "acquisition_summary.json"
        alternate = self.acquisition / "acquisition_summary_v2.json"
        original.replace(alternate)
        result = audit_acquisition(
            self.acquisition,
            self.labels,
            self.expected(),
            summary_name="acquisition_summary_v2.json",
        )
        self.assertTrue(result["gate_passed"])

    def test_merge_does_not_leave_overlap(self):
        self.assertEqual(merge_intervals([(1., 2.), (1.5, 3.), (4., 5.)]), [(1., 3.), (4., 5.)])


if __name__ == "__main__":
    unittest.main()
