import hashlib
from pathlib import Path
import unittest
import unicodedata

import reconcile_saraga_hindustani_v1 as v1
from reconcile_saraga_hindustani_v2 import (
    AUDIO_SUFFIX,
    V1_CODE_SHA256,
    V1_FAILED_RECEIPT_SHA256,
    V1_TEST_SHA256,
    archive_audio_candidates,
    reconcile_v2,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def md5(label):
    return hashlib.md5(label.encode()).hexdigest()


def record(path, checksum=None):
    return {
        "path": path,
        "bytes": 123,
        "md5": checksum or md5(path),
        "sha256": hashlib.sha256(path.encode()).hexdigest(),
        "crc_verified": True,
    }


def fixture(specs):
    catalog = {}
    metadata = {}
    declared = {}
    records = {}
    for index, spec in enumerate(specs):
        mbid = spec["mbid"]
        pinned = spec["pinned"]
        actual = spec.get("actual", pinned)
        checksum = spec.get("catalog_md5", md5(mbid))
        companion = spec.get("companions", ["json", "mp3.md5", "ctonic.txt"])
        catalog[mbid] = {
            "mbid": mbid,
            "metadata_path": f"dataset/hindustani/{spec.get('metadata', pinned)}.json",
            "audio_md5": checksum,
            "companion_files": companion,
            "speech_title_review_flag": False,
        }
        metadata[mbid] = {
            "mbid": mbid,
            "title": mbid,
            "length": 60_000 + index,
            "artists": spec.get("artists", []),
            "album_artists": spec.get("album_artists", []),
            "release": spec.get("release", []),
        }
        declared[pinned] = mbid
        audio_path = f"{v1.ARCHIVE_ROOT}/{actual}{AUDIO_SUFFIX}"
        records[audio_path] = record(audio_path, spec.get("archive_md5", checksum))
        metadata_path = f"{v1.ARCHIVE_ROOT}/{actual}.json"
        records[metadata_path] = record(metadata_path)
        if "ctonic.txt" in companion:
            annotation_path = f"{v1.ARCHIVE_ROOT}/{actual}.ctonic.txt"
            records[annotation_path] = record(annotation_path)
    return catalog, metadata, declared, records


class V2ReconciliationTests(unittest.TestCase):
    def test_unique_checksum_resolves_renamed_path_without_fuzzy_rewrite(self):
        values = fixture([{
            "mbid": "track-1",
            "pinned": "Album/Raag Dagori__Deepki/Raag Dagori__Deepki",
            "metadata": "Album/Raag Dagori_Deepki/Raag Dagori_Deepki",
            "actual": "Album/Raag Dagori_Deepki/Raag Dagori_Deepki",
        }])
        report = reconcile_v2(*values)
        self.assertTrue(report["status"].startswith("passed_"))
        aliases = report[
            "file_paths_csv_vs_checksum_resolved_archive_path_differences"]
        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0]["relation"], "exact_unique_mp3_md5")
        self.assertFalse(report["checksum_path_resolution"]["fuzzy_path_rewriting_used"])

    def test_companions_and_metadata_use_checksum_resolved_base(self):
        values = fixture([{
            "mbid": "track-1", "pinned": "Old/Track/Track",
            "actual": "New/Track/Track",
        }])
        report = reconcile_v2(*values)
        self.assertEqual(report["inventory"]["missing_annotation_members"], 0)
        self.assertEqual(report["inventory"]["missing_track_metadata_members"], 0)
        track = report["tracks"][0]
        self.assertEqual(track["checksum_resolved_archive_base_path"], "New/Track/Track")
        annotation = next(a for a in track["annotations"] if a["kind"] == "ctonic.txt")
        self.assertEqual(annotation["archive_path"],
                         f"{v1.ARCHIVE_ROOT}/New/Track/Track.ctonic.txt")

    def test_missing_checksum_is_a_hard_failure_with_inventory(self):
        values = fixture([{"mbid": "track-1", "pinned": "A/A/A",
                           "archive_md5": "f" * 32}])
        report = reconcile_v2(*values)
        self.assertEqual(report["status"],
                         "failed_archive_catalog_checksum_path_reconciliation_v2")
        self.assertIn("catalog_checksum_missing_from_archive", report["failure_reasons"])
        self.assertIn("archive_checksum_unmatched_to_catalog", report["failure_reasons"])
        resolution = report["checksum_path_resolution"]
        self.assertEqual(len(resolution["catalog_checksums_missing_from_archive"]), 1)
        self.assertEqual(len(resolution["archive_checksums_unmatched_to_catalog"]), 1)

    def test_duplicate_checksum_is_rejected_on_both_sides(self):
        duplicate = "a" * 32
        values = fixture([
            {"mbid": "track-1", "pinned": "A/A/A", "catalog_md5": duplicate},
            {"mbid": "track-2", "pinned": "B/B/B", "catalog_md5": duplicate},
        ])
        report = reconcile_v2(*values)
        self.assertIn("duplicate_catalog_mp3_checksum", report["failure_reasons"])
        self.assertIn("duplicate_archive_mp3_checksum", report["failure_reasons"])
        self.assertEqual(report["checksum_path_resolution"][
            "duplicate_catalog_checksums"][0]["md5"], duplicate)

    def test_normalized_archive_base_collision_is_rejected(self):
        composed = f"{v1.ARCHIVE_ROOT}/Album/Café/Track{AUDIO_SUFFIX}"
        decomposed = unicodedata.normalize("NFD", composed)
        records = {
            composed: record(composed),
            decomposed: record(decomposed),
        }
        with self.assertRaises(v1.ReconciliationError) as caught:
            archive_audio_candidates(records)
        self.assertEqual(caught.exception.report["status"],
                         "failed_ambiguous_path_normalization")

    def test_frozen_v1_code_and_receipt_are_unchanged(self):
        self.assertEqual(v1.sha256_file(HERE / "reconcile_saraga_hindustani_v1.py"),
                         V1_CODE_SHA256)
        self.assertEqual(v1.sha256_file(HERE / "test_reconcile_saraga_hindustani_v1.py"),
                         V1_TEST_SHA256)
        self.assertEqual(v1.sha256_file(ROOT / "audit" /
                                       "saraga_hindustani_reconciliation_v1.json"),
                         V1_FAILED_RECEIPT_SHA256)


if __name__ == "__main__":
    unittest.main()
