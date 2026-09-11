import hashlib
from pathlib import Path
import tempfile
import unittest
import unicodedata

from reconcile_saraga_hindustani_v1 import (
    ARCHIVE_ROOT,
    ReconciliationError,
    build_connected_groups,
    checked_normalized_map,
    reconcile,
    require_snapshot_unchanged,
    snapshot_files,
)


def digest(label):
    return hashlib.md5(label.encode()).hexdigest()


def archive_record(path, md5=None):
    return {
        "path": path,
        "bytes": 123,
        "md5": md5 or digest(path),
        "sha256": hashlib.sha256(path.encode()).hexdigest(),
        "crc_verified": True,
    }


def inputs(track_specs):
    catalog = {}
    metadata = {}
    declared = {}
    records = {}
    for number, spec in enumerate(track_specs):
        mbid = spec["mbid"]
        base = spec["base"]
        checksum = spec.get("md5", digest(mbid))
        catalog[mbid] = {
            "mbid": mbid,
            "tradition": "hindustani",
            "title": mbid,
            "metadata_path": f"dataset/hindustani/{base}.json",
            "metadata_sha256": "0" * 64,
            "advertised_length_ms": 60_000 + number,
            "audio_md5": checksum,
            "speech_title_review_flag": False,
            "companion_files": ["json", "mp3.md5", "ctonic.txt"],
        }
        metadata[mbid] = {
            "mbid": mbid,
            "title": mbid,
            "length": 60_000 + number,
            "artists": spec.get("artists", []),
            "album_artists": spec.get("album_artists", []),
            "release": spec.get("release", []),
        }
        declared[base] = mbid
        audio_path = f"{ARCHIVE_ROOT}/{base}.mp3.mp3"
        records[audio_path] = archive_record(audio_path, spec.get("archive_md5", checksum))
        annotation_path = f"{ARCHIVE_ROOT}/{base}.ctonic.txt"
        records[annotation_path] = archive_record(annotation_path)
        metadata_path = f"{ARCHIVE_ROOT}/{base}.json"
        records[metadata_path] = archive_record(metadata_path)
    return catalog, metadata, declared, records


def credit(mbid, name):
    return {"artist": {"mbid": mbid, "name": name},
            "instrument": {"mbid": "voice", "name": "Voice"},
            "lead": True, "attributes": "lead vocals"}


class ReconciliationTests(unittest.TestCase):
    def test_repeated_mp3_suffix_and_resource_fork_are_explicitly_classified(self):
        archive_base = "Album/Raag Dagori__Deepki/Raag Dagori__Deepki"
        values = inputs([{"mbid": "track-1", "base": archive_base}])
        # The pinned file_paths.csv is the explicit MBID bridge for a real kind
        # of catalog/archive spelling difference; no underscore rewrite is used.
        values[0]["track-1"]["metadata_path"] = (
            "dataset/hindustani/Album/Raag Dagori_Deepki/Raag Dagori_Deepki.json")
        records = values[3]
        fork = f"__MACOSX/{ARCHIVE_ROOT}/Album/._Raag Dagori__Deepki.mp3.mp3"
        records[fork] = archive_record(fork)
        report = reconcile(*values)
        self.assertEqual(report["status"],
                         "passed_archive_catalog_reconciliation_not_audio_admission")
        self.assertEqual(report["inventory"]["data_mp3_records"], 1)
        self.assertEqual(report["inventory"]["appledouble_resource_fork_mp3_records"], 1)
        self.assertEqual(report["inventory"]["archive_mp3_suffix_records"], 2)
        self.assertEqual(report["matched_audio"][0]["archive_audio_path"],
                         f"{ARCHIVE_ROOT}/{archive_base}.mp3.mp3")
        self.assertEqual(report["declared_path_aliases"][0]["mbid"], "track-1")
        self.assertFalse(report["physical_audio_decoded"])

    def test_unicode_normalization_collision_fails_with_both_raw_paths(self):
        composed = "Album/Café/Track"
        decomposed = unicodedata.normalize("NFD", composed)
        with self.assertRaises(ReconciliationError) as caught:
            checked_normalized_map([(composed, 1), (decomposed, 2)], "synthetic")
        self.assertEqual(caught.exception.report["status"],
                         "failed_ambiguous_path_normalization")
        self.assertEqual(set(caught.exception.report["collisions"][0]["raw_paths"]),
                         {composed, decomposed})

    def test_checksum_mismatch_is_preserved_and_fails_report(self):
        values = inputs([{"mbid": "track-1", "base": "Album/Track/Track",
                          "md5": "1" * 32, "archive_md5": "2" * 32}])
        report = reconcile(*values)
        self.assertEqual(report["status"], "failed_archive_catalog_reconciliation")
        self.assertIn("audio_md5_mismatch", report["failure_reasons"])
        self.assertEqual(report["mismatched_audio_md5"], [{
            "mbid": "track-1",
            "catalog_metadata_path": "dataset/hindustani/Album/Track/Track.json",
            "declared_archive_base_path": "Album/Track/Track",
            "archive_audio_path": f"{ARCHIVE_ROOT}/Album/Track/Track.mp3.mp3",
            "catalog_md5": "1" * 32,
            "archive_md5": "2" * 32,
        }])

    def test_missing_and_extra_audio_paths_are_both_preserved(self):
        values = inputs([{"mbid": "track-1", "base": "Album/Track/Track"}])
        records = values[3]
        expected = f"{ARCHIVE_ROOT}/Album/Track/Track.mp3.mp3"
        del records[expected]
        extra = f"{ARCHIVE_ROOT}/Unexpected/Track/Track.mp3.mp3"
        records[extra] = archive_record(extra)
        report = reconcile(*values)
        self.assertIn("missing_archive_audio", report["failure_reasons"])
        self.assertIn("extra_or_unexpected_archive_audio", report["failure_reasons"])
        self.assertEqual(report["missing_audio"][0]["expected_archive_path"], expected)
        self.assertEqual(report["extra_archive_audio"][0]["path"], extra)

    def test_artist_then_release_bridge_forms_one_conservative_component(self):
        track_specs = [
            {"mbid": "track-a", "base": "A/A/A", "artists": [credit("artist-x", "X")]},
            {"mbid": "track-b", "base": "B/B/B", "artists": [credit("artist-x", "X")],
             "release": [{"mbid": "release-y", "title": "Y"}]},
            {"mbid": "track-c", "base": "C/C/C", "artists": [credit("artist-z", "Z")],
             "release": [{"mbid": "release-y", "title": "Y"}]},
        ]
        report = reconcile(*inputs(track_specs))
        groups = report["performer_connected_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["track_mbids"], ["track-a", "track-b", "track-c"])
        bridge_kinds = {item["kind"] for item in groups[0]["shared_identity_bridges"]}
        self.assertEqual(bridge_kinds, {"artist", "release_or_concert"})

    def test_artist_mbid_bridges_performer_and_album_artist_roles(self):
        track_specs = [
            {"mbid": "track-a", "base": "A/A/A", "artists": [credit("artist-x", "X")]},
            {"mbid": "track-b", "base": "B/B/B",
             "album_artists": [{"mbid": "artist-x", "name": "X"}]},
        ]
        groups = reconcile(*inputs(track_specs))["performer_connected_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["track_mbids"], ["track-a", "track-b"])
        self.assertEqual(groups[0]["shared_identity_bridges"], [{
            "kind": "artist",
            "identity_mbid": "artist-x",
            "source_roles": ["album_artist", "performer_credit"],
            "track_mbids": ["track-a", "track-b"],
        }])

    def test_input_mutation_is_rejected_with_before_and_after_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text("before")
            snapshot = snapshot_files({"synthetic": path})
            path.write_text("after")
            with self.assertRaises(ReconciliationError) as caught:
                require_snapshot_unchanged(snapshot)
            report = caught.exception.report
            self.assertEqual(report["status"],
                             "failed_input_mutation_during_reconciliation")
            self.assertNotEqual(report["changed_files"][0]["before_sha256"],
                                report["changed_files"][0]["after_sha256"])


if __name__ == "__main__":
    unittest.main()
