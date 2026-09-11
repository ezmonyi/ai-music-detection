import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

import audit_saraga_physical_v1 as target


MBIDS = ["00000000-0000-0000-0000-000000000001",
         "00000000-0000-0000-0000-000000000002"]


def file_hashes(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "md5": hashlib.md5(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path, value):
    Path(path).write_bytes(target.canonical(value))


def write_pretty_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n")


class Fixture:
    """Synthetic receipt fixture; production CLI exposes no pin/count override."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.physical = self.root / "physical"
        self.raw = self.physical / "raw"
        self.receipts = self.physical / "receipts"
        self.raw.mkdir(parents=True)
        self.receipts.mkdir()
        self.materializer = self.root / "materializer.py"
        self.reconciliation = self.root / "reconciliation.json"
        self.archive_audit = self.root / "archive_audit.json"
        self.materializer.write_text("# synthetic materializer fixture\n")
        archive_records = []
        tracks = []
        matches = []
        for index, mbid in enumerate(MBIDS):
            member_path = f"saraga1.5_hindustani/performer/track{index}.mp3.mp3"
            payload = (f"synthetic raw {index}\n".encode()) * (index + 3)
            raw_path = self.raw / f"{mbid}.mp3"
            raw_path.write_bytes(payload)
            hashes = file_hashes(raw_path)
            member = {"path": member_path, "bytes": hashes["bytes"],
                      "compressed_bytes": hashes["bytes"], "crc32": f"{index + 1:08x}",
                      "crc_verified": True, "md5": hashes["md5"], "sha256": hashes["sha256"]}
            archive_records.append(member)
            tracks.append({"mbid": mbid, "title": f"Track {index}",
                           "advertised_length_ms": 70000,
                           "speech_title_review_flag": index == 0,
                           "catalog_metadata_path": f"metadata/{mbid}.json",
                           "archive_audio_path": member_path,
                           "performer_credits": [] if index == 1 else [{"artist_mbid": "artist"}],
                           "album_artists": [], "release_or_concert_groups": [],
                           "annotations": {}, "pinned_file_paths_csv_base_path": f"track{index}",
                           "checksum_resolved_archive_base_path": f"track{index}",
                           "path_resolution": "exact_unique_mp3_md5"})
            matches.append({"mbid": mbid, "catalog_metadata_path": f"metadata/{mbid}.json",
                            "archive_audio_path": member_path, "catalog_md5": hashes["md5"],
                            "archive_md5": hashes["md5"],
                            "pinned_file_paths_csv_base_path": f"track{index}",
                            "checksum_resolved_archive_base_path": f"track{index}",
                            "path_resolution": "exact_unique_mp3_md5"})
        archive_id = {"bytes": 12345, "md5": "a" * 32, "sha256": "b" * 64}
        audit = {"status": "passed_archive_integrity_not_audio_admission",
                 "source_record": 4301737, "archive_path": "/synthetic/archive.zip",
                 "archive_bytes": archive_id["bytes"],
                 "archive_hashes": {"md5": archive_id["md5"], "sha256": archive_id["sha256"]},
                 "metadata_sha256": "c" * 64, "code_sha256": "d" * 64,
                 "archive_members_verified": len(archive_records), "records": archive_records,
                 "mp3_members": len(archive_records), "extracted_audio_files": 0,
                 "physically_decoded_audio_files": 0, "classifier_admission": False}
        write_pretty_json(self.archive_audit, audit)
        recon = {"status": "passed_archive_catalog_checksum_path_reconciliation_v2_not_audio_admission",
                 "failure_reasons": [], "tracks": tracks, "matched_audio": matches,
                 "physical_audio_decoded": False, "classifier_admission": False,
                 "cohort_selected": False,
                 "archive_integrity_evidence": {"archive_bytes": archive_id["bytes"],
                                                "archive_hashes": {"md5": archive_id["md5"],
                                                                   "sha256": archive_id["sha256"]}}}
        write_pretty_json(self.reconciliation, recon)
        self.pins = target.Pins(len(MBIDS), archive_id["bytes"], archive_id["md5"],
                                archive_id["sha256"], file_hashes(self.reconciliation)["sha256"],
                                file_hashes(self.archive_audit)["sha256"],
                                file_hashes(self.materializer)["sha256"])
        items = target.build_expected_items(recon, audit, self.pins)
        named = {"code": self.materializer, "reconciliation": self.reconciliation,
                 "archive_audit": self.archive_audit}
        contract_inputs = {}
        for name, path in named.items():
            contract_inputs[name] = {"path": str(path.resolve()), **file_hashes(path)}
        contract = {"version": target.MATERIALIZER_VERSION, "expected_recordings": len(MBIDS),
                    "archive_path": "/synthetic/archive.zip", "archive_hashes": archive_id,
                    "inputs": contract_inputs,
                    "runtime": {"python": "fixture", "python_executable": "/fixture/python",
                                "platform": "fixture", "numpy": "fixture", "soundfile": "fixture",
                                "libsndfile": "fixture", "numpy_module_sha256": "e" * 64,
                                "soundfile_module_sha256": "f" * 64,
                                "decoder_claim": "SoundFile/libsndfile only; fixture"},
                    "decode_block_frames": target.BLOCK_FRAMES,
                    "pcm_encoding": target.PCM_ENCODING, "cohort_selected": False,
                    "classifier_admission": False, "items": items}
        self.contract = self.physical / "contract.json"
        write_json(self.contract, contract)
        contract_sha = file_hashes(self.contract)["sha256"]
        records = []
        for index, item in enumerate(items):
            raw_hash = {key: item["archive_member"][key] for key in ("bytes", "md5", "sha256")}
            frames = 70_000 + index
            header = frames if index == 0 else frames - 1
            measurement = {"sample_rate": 1000, "channels": 1,
                           "decoder_format": "MPEG-1/2 Audio", "decoder_subtype": "MPEG Layer III",
                           "header_frames": header, "actual_frames": frames,
                           "header_minus_actual_frames": header - frames,
                           "header_matches_actual_eof": header == frames,
                           "actual_duration_seconds": frames / 1000,
                           "duration_at_least_60_seconds": True,
                           "read_calls_including_empty_eof": 3,
                           "real_empty_read_observed": True,
                           "read_block_frames": target.BLOCK_FRAMES, "pcm_sha256": str(index) * 64,
                           "pcm_encoding": target.PCM_ENCODING, "sample_count": frames,
                           "finite_sample_count": frames, "nonfinite_sample_count": 0,
                           "sample_min": -0.5, "sample_max": 0.5, "peak_absolute": 0.5,
                           "rms_all_samples": 0.25, "raw_hashes_before_decode": raw_hash,
                           "raw_hashes_after_decode": raw_hash}
            record = {"status": "passed_physical_acquisition_not_audio_admission",
                      "contract_sha256": contract_sha, "item": item, "measurement": measurement,
                      "classifier_admission": False, "cohort_selected": False}
            records.append(record)
            write_json(self.receipts / f"{item['mbid']}.json",
                       {"record": record, "record_sha256": target.object_sha256(record)})
        summary = {"status": "passed_all_physical_acquisition_not_audio_admission",
                   "version": target.MATERIALIZER_VERSION, "contract_sha256": contract_sha,
                   "record_count": len(records), "source_archive_hashes_before": archive_id,
                   "source_archive_hashes_after": archive_id,
                   "input_code_contract_unchanged": True, "runtime_unchanged": True,
                   "expected_file_inventory_verified": True, "classifier_admission": False,
                   "cohort_selected": False, "annotation_alignment_verified": False,
                   "human_music_usability_claim": False, "speech_title_review_flags": 1,
                   "missing_performer_credits": 1,
                   "duration_at_least_60_seconds_count": len(records),
                   "header_frame_mismatch_count": 1, "records": records}
        self.summary = self.physical / "summary.json"
        write_json(self.summary, summary)

    def audit(self):
        return target._audit(self.physical, self.reconciliation, self.archive_audit,
                             self.materializer, self.pins)

    def update_record(self, index, mutator):
        path = self.receipts / f"{MBIDS[index]}.json"
        receipt = json.loads(path.read_text())
        mutator(receipt["record"])
        receipt["record_sha256"] = target.object_sha256(receipt["record"])
        write_json(path, receipt)
        summary = json.loads(self.summary.read_text())
        summary["records"][index] = receipt["record"]
        write_json(self.summary, summary)


class SaragaPhysicalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_fixture_passes_without_decode_or_admission(self):
        self.assertNotEqual(self.fixture.reconciliation.read_bytes(),
                            target.canonical(json.loads(self.fixture.reconciliation.read_text())))
        self.assertNotEqual(self.fixture.archive_audit.read_bytes(),
                            target.canonical(json.loads(self.fixture.archive_audit.read_text())))
        result = self.fixture.audit()
        self.assertEqual(result["record_count"], 2)
        self.assertTrue(result["all_raw_md5_sha256_rehashed_twice_and_matched"])
        self.assertFalse(result["second_full_decode_performed"])
        self.assertFalse(result["classifier_admission"])
        self.assertEqual(result["header_frame_mismatch_count"], 1)
        self.assertEqual(result["duration_total_seconds"], 140.001)
        self.assertEqual(result["neither_speech_nor_missing_performer_review_flag_count"], 0)

    def test_partial_output_rejected(self):
        self.fixture.summary.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.fixture.audit()

    def test_busy_writer_lock_rejected(self):
        (self.fixture.physical / "writer.lock").write_text("busy\n")
        with self.assertRaisesRegex(ValueError, "busy or stale-locked"):
            self.fixture.audit()

    def test_mutated_receipt_seal_rejected(self):
        path = self.fixture.receipts / f"{MBIDS[0]}.json"
        receipt = json.loads(path.read_text())
        receipt["record"]["measurement"]["actual_duration_seconds"] += 1
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "seal mismatch"):
            self.fixture.audit()

    def test_duplicate_summary_identity_rejected(self):
        summary = json.loads(self.fixture.summary.read_text())
        summary["records"][1] = summary["records"][0]
        write_json(self.fixture.summary, summary)
        with self.assertRaisesRegex(ValueError, "Duplicate/invalid summary MBID"):
            self.fixture.audit()

    def test_duplicate_hardlinked_raw_rejected(self):
        second = self.fixture.raw / f"{MBIDS[1]}.mp3"
        second.unlink()
        os.link(self.fixture.raw / f"{MBIDS[0]}.mp3", second)
        with self.assertRaisesRegex(ValueError, "singly linked"):
            self.fixture.audit()

    def test_missing_raw_rejected(self):
        (self.fixture.raw / f"{MBIDS[1]}.mp3").unlink()
        with self.assertRaisesRegex(ValueError, "raw file"):
            self.fixture.audit()

    def test_inconsistent_summary_aggregate_rejected(self):
        summary = json.loads(self.fixture.summary.read_text())
        summary["duration_at_least_60_seconds_count"] = 1
        write_json(self.fixture.summary, summary)
        with self.assertRaisesRegex(ValueError, "Summary aggregate mismatch"):
            self.fixture.audit()

    def test_nonfinite_sample_accounting_rejected_even_with_new_seal(self):
        def mutate(record):
            record["measurement"]["nonfinite_sample_count"] = 1
        self.fixture.update_record(0, mutate)
        with self.assertRaisesRegex(ValueError, "finite-sample"):
            self.fixture.audit()

    def test_raw_hash_mutation_rejected(self):
        path = self.fixture.raw / f"{MBIDS[0]}.mp3"
        path.write_bytes(path.read_bytes() + b"mutation")
        with self.assertRaisesRegex(ValueError, "Raw byte/hash mismatch"):
            self.fixture.audit()

    def test_final_rebind_detects_mutation_after_initial_hashes(self):
        original = target.verify_bound_snapshot
        raw = self.fixture.raw / f"{MBIDS[0]}.mp3"

        def mutate_then_verify(bound, physical, paths, raw_paths):
            raw.write_bytes(raw.read_bytes() + b"late mutation")
            return original(bound, physical, paths, raw_paths)

        with mock.patch.object(target, "verify_bound_snapshot", side_effect=mutate_then_verify):
            with self.assertRaisesRegex(ValueError, "Raw file changed during audit"):
                self.fixture.audit()

    def test_cli_has_no_fixture_or_pin_overrides(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit):
            target.main(["--help"])
        self.assertNotIn("synthetic", stdout.getvalue().casefold())
        self.assertNotIn("--pin", stdout.getvalue().casefold())
        self.assertNotIn("--count", stdout.getvalue().casefold())

    def test_report_write_is_exclusive(self):
        report = self.fixture.root / "report.json"
        target.write_exclusive(report, {"status": "first"})
        with self.assertRaises(FileExistsError):
            target.write_exclusive(report, {"status": "second"})

    def test_ancestor_symlink_input_path_rejected_before_resolution(self):
        link = self.fixture.root / "linked"
        link.symlink_to(self.fixture.root, target_is_directory=True)
        linked_reconciliation = link / self.fixture.reconciliation.name
        with self.assertRaisesRegex(ValueError, "Symlink path component forbidden"):
            target._audit(self.fixture.physical, linked_reconciliation,
                          self.fixture.archive_audit, self.fixture.materializer,
                          self.fixture.pins)

    def test_source_parser_rejects_duplicate_and_nonfinite_json(self):
        source = self.fixture.root / "bad_source.json"
        for raw in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n'):
            with self.subTest(raw=raw):
                source.write_bytes(raw)
                with self.assertRaises(ValueError):
                    target.read_json(source)

    def test_physical_contract_remains_canonical_only(self):
        contract = json.loads(self.fixture.contract.read_text())
        write_pretty_json(self.fixture.contract, contract)
        with self.assertRaisesRegex(ValueError, "Noncanonical JSON"):
            self.fixture.audit()


if __name__ == "__main__":
    unittest.main()
