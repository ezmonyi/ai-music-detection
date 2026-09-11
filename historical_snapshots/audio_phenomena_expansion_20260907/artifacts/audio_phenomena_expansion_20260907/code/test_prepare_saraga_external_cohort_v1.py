"""Synthetic-only tests: no real corpus, audio decoder, network, model or scoring."""
import ast
import copy
import csv
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import prepare_saraga_external_cohort_v1 as p


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def fixture(root, mutate=None):
    """At most five explicit synthetic recordings; no MusicBrainz identities."""
    items, measurements = [], []
    for index in range(5):
        mbid = f"synthetic_record_{index}"
        # Album artist in row 0 is performer in row 1: role-agnostic edge.
        performer = "synthetic_bridge" if index == 1 else f"synthetic_artist_{index}"
        album = "synthetic_bridge" if index == 0 else f"synthetic_album_{index}"
        raw = {"sha256": p.sha(f"synthetic_raw_{index}".encode()), "md5": f"{index:032x}", "bytes": 100 + index}
        sr = 48000 if index == 2 else 44100
        frames = 61 * sr + 7
        metadata = {"mbid": mbid, "title": "SYNTHETIC TEST ONLY", "speech_title_review_flag": index == 3,
                    "performer_credits": [] if index == 4 else [{"artist_mbid": performer}],
                    "album_artists": [{"mbid": album}],
                    "release_or_concert_groups": [{"kind": "release", "mbid": f"synthetic_release_{index}"}]}
        items.append({"mbid": mbid, "raw_path": f"raw/{mbid}.mp3", "reconciled_metadata": metadata, "archive_member": raw.copy()})
        measurements.append({"sample_rate": sr, "actual_frames": frames, "header_frames": frames + 19,
                             "channels": 2, "real_empty_read_observed": True,
                             "header_matches_actual_eof": False, "header_minus_actual_frames": 19,
                             "raw_hashes_before_decode": raw.copy(), "raw_hashes_after_decode": raw.copy()})
    if mutate:
        mutate(items, measurements)
    objects = {}
    def write_json(name, obj, marker=True):
        if marker:
            obj["synthetic_test_only"] = True
        raw = p.canonical(obj)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        objects[name] = obj
        return p.sha(raw)
    contract_hash = write_json(f"{p.PHYSICAL}/contract.json", {"expected_recordings": 5, "items": items})
    records = [{"item": item, "measurement": measure, "contract_sha256": contract_hash,
                "status": "passed_physical_acquisition_not_audio_admission", "classifier_admission": False,
                "cohort_selected": False} for item, measure in zip(items, measurements)]
    receipts = []
    for record in records:
        mbid = record["item"]["mbid"]
        digest = write_json(f"{p.PHYSICAL}/receipts/{mbid}.json", {"record": record, "record_sha256": p.seal(record)}, False)
        receipts.append({"mbid": mbid, "sha256": digest})
    summary_hash = write_json(f"{p.PHYSICAL}/summary.json", {"record_count": 5, "records": records,
                              "contract_sha256": contract_hash, "status": "passed_all_physical_acquisition_not_audio_admission"})
    recon_hash = write_json("audit/saraga_hindustani_reconciliation_v2.json", {"tracks": [i["reconciled_metadata"] for i in items]})
    write_json("audit/saraga_physical_independent_v1.json", {
        "status": "passed_independent_byte_accounting_audit_not_second_decode_not_admission", "record_count": 5,
        "summary_sha256": summary_hash, "contract_sha256": contract_hash, "reconciliation_sha256": recon_hash,
        "receipt_manifest_sha256": p.seal(receipts),
        "records": [{"mbid": i["mbid"], **m["raw_hashes_before_decode"]} for i, m in zip(items, measurements)]})
    write_json("audit/saraga_catalog_v1.json", {"status": "synthetic_catalog"})
    old = [{"id": "synthetic_old_0", "role": "development"}]
    native = [{**old[0], "registered_raw_sha256": p.sha(b"synthetic old source"), "physical_sha256": ""}]
    fhm = [{**old[0], "source_audio_sha256": p.sha(b"synthetic old source")}]
    package = root / p.PACKAGE
    (package / "evidence").mkdir(parents=True, exist_ok=True)
    package_pins = {}
    for name, raw in (("metadata_60s.csv", csv_bytes(old)), ("evidence/native.csv", csv_bytes(native)),
                      ("evidence/fhm.csv", csv_bytes(fhm)), ("evidence/extraction_receipt.json", p.canonical({"synthetic_test_only": True}))):
        (package / name).write_bytes(raw)
        package_pins[name] = p.sha(raw)
    old_contract = {"package_files_sha256": package_pins}
    write_json("preregistration/equal60_v4_with_recovery_frozen_v1.json", {"status": "frozen", "contract": old_contract})
    run_hash = write_json("results/equal60_results_v4_with_recovery_v1/run_manifest.json", {"contract": old_contract})
    write_json("audit/equal60_v4_with_recovery_results_audit_v1.json", {"status": "passed", "rows": 1, "result_manifest_sha256": run_hash})
    pins = {name: p.sha((root / name).read_bytes()) for name in p.PINS}
    return pins, records


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="saraga-synthetic-tests-")
        self.root = pathlib.Path(self.temp.name).resolve()
        self.output = self.root / "draft"
        self.pins, self.records = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_draft(self):
        return p.prepare(self.root, self.output, _synthetic_pins=self.pins)

    def alter_old_package(self, name, rows):
        raw = csv_bytes(rows)
        (self.root / p.PACKAGE / name).write_bytes(raw)
        frozen_name = "preregistration/equal60_v4_with_recovery_frozen_v1.json"
        run_name = "results/equal60_results_v4_with_recovery_v1/run_manifest.json"
        audit_name = "audit/equal60_v4_with_recovery_results_audit_v1.json"
        frozen = p.strict_json((self.root / frozen_name).read_bytes())
        frozen["contract"]["package_files_sha256"][name] = p.sha(raw)
        (self.root / frozen_name).write_bytes(p.canonical(frozen))
        run = p.strict_json((self.root / run_name).read_bytes())
        run["contract"] = frozen["contract"]
        (self.root / run_name).write_bytes(p.canonical(run))
        audit = p.strict_json((self.root / audit_name).read_bytes())
        audit["result_manifest_sha256"] = p.sha(p.canonical(run))
        (self.root / audit_name).write_bytes(p.canonical(audit))
        for relative in (frozen_name, run_name, audit_name):
            self.pins[relative] = p.sha((self.root / relative).read_bytes())

    def test_end_to_end_draft_and_integer_native_interval(self):
        result = self.run_draft()
        self.assertEqual((result["status"], result["eligible_count"], result["excluded_count"]), ("draft", 3, 2))
        self.assertEqual(result["group_sizes"], [2, 1])
        self.assertEqual(result["native_rate_counts"], {"44100": 2, "48000": 1})
        rows = p.csv_rows((self.output / "metadata.csv").read_bytes())
        for row in rows:
            sr = int(row["native_sample_rate_hz"])
            count = 60 * sr
            start = (int(row["actual_eof_frames"]) - count) // 2
            self.assertEqual(int(row["crop_start_frame"]), start)
            self.assertEqual(int(row["crop_frames"]), count)
            self.assertEqual(int(row["crop_end_frame_exclusive"]), start + count)
            self.assertEqual(round(float(row["source_offset_seconds"]) * sr), start)
            self.assertEqual(row["role"], p.ROLE)
            self.assertEqual(row["label"], "0")
            self.assertEqual(row["soundfile_header_frames"], str(int(row["actual_eof_frames"]) + 19))
            self.assertEqual(row["evaluation_allowed"], "False")
        excluded = p.csv_rows((self.output / "exclusions.csv").read_bytes())
        self.assertEqual({r["reasons"] for r in excluded}, {"speech_title_review_flag", "missing_performer_credit"})
        self.assertTrue(all("measurement" in json.loads(r["source_record_json"]) for r in excluded))
        receipt = p.strict_json((self.output / "evidence_receipt.json").read_bytes())
        for name, expected in receipt["outputs_sha256"].items():
            self.assertEqual(p.sha((self.output / name).read_bytes()), expected)
        self.assertFalse(result["frozen"] or result["scoring_performed"] or result["measurement_acceptance"])

    def test_deterministic_grouping_order_and_role_bridge(self):
        records = self.records[:3]
        a = p.group_records(records, True)
        b = p.group_records(list(reversed(records)), True)
        self.assertEqual(a, b)
        self.assertEqual(a[1]["synthetic_record_0"], a[1]["synthetic_record_1"])
        self.assertNotEqual(a[1]["synthetic_record_0"], a[1]["synthetic_record_2"])

    def test_group_digest_collision_rejected(self):
        with mock.patch.object(p, "seal", return_value="0" * 64), self.assertRaisesRegex(ValueError, "collision"):
            p.group_records(self.records[:3], True)

    def test_unknown_artist_and_release_ids_rejected(self):
        for field, key in (("performer_credits", "artist_mbid"), ("album_artists", "mbid"), ("release_or_concert_groups", "mbid")):
            records = copy.deepcopy(self.records[:3])
            records[0]["item"]["reconciled_metadata"][field][0][key] = ""
            with self.assertRaisesRegex(ValueError, "identity"):
                p.group_records(records, True)

    def test_missing_input(self):
        (self.root / "audit/saraga_catalog_v1.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_draft()
        self.assertFalse(self.output.exists())

    def test_stale_pin(self):
        (self.root / "audit/saraga_catalog_v1.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "Stale pin"):
            self.run_draft()

    def test_input_symlink(self):
        original = self.root / "audit/saraga_catalog_v1.json"
        target = self.root / "synthetic-target.json"
        original.rename(target)
        original.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.run_draft()

    def test_output_symlink_and_existing_directory(self):
        self.output.symlink_to(self.root / "nonexistent")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.run_draft()
        self.output.unlink()
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "Output conflict"):
            self.run_draft()

    def test_kernel_no_replace(self):
        source, dest = self.root / "stage", self.root / "exists"
        source.mkdir()
        dest.mkdir()
        with self.assertRaises(OSError):
            p.rename_exclusive(source, dest)
        self.assertTrue(source.exists() and dest.exists())

    def test_bad_native_rate_short_duration_or_header_flag(self):
        for field, value in (("sample_rate", 22050), ("actual_frames", 44100), ("actual_frames", 1.5),
                             ("channels", 1), ("header_matches_actual_eof", True), ("real_empty_read_observed", False)):
            records = copy.deepcopy(self.records[:3])
            records[0]["measurement"][field] = value
            groups, assigned = p.group_records(records, True)
            with self.assertRaises(ValueError):
                p.make_rows(records, assigned, True)

    def test_no_production_ids_in_synthetic_mode(self):
        self.pins, _ = fixture(self.root, lambda items, measurements: items[0].update(mbid=next(iter(p.SPEECH))))
        with self.assertRaises(ValueError):
            self.run_draft()

    def test_no_synthetic_fixture_under_production_pins(self):
        with self.assertRaisesRegex(ValueError, "Stale pin"):
            p.prepare(self.root, self.output)

    def test_old_role_drift_rejected(self):
        self.alter_old_package("metadata_60s.csv", [{"id": "synthetic_old_0", "role": "external_human_unscored"}])
        with self.assertRaisesRegex(ValueError, "role drift"):
            self.run_draft()

    def test_missing_complete_old_source_hash_rejected(self):
        self.alter_old_package("evidence/fhm.csv", [{"id": "synthetic_old_0", "role": "development", "source_audio_sha256": ""}])
        with self.assertRaisesRegex(ValueError, "Invalid digest"):
            self.run_draft()

    def test_old_source_identity_join_incomplete_rejected(self):
        self.alter_old_package("evidence/fhm.csv", [{"id": "synthetic_wrong_0", "role": "development", "source_audio_sha256": p.sha(b"synthetic old")}])
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_draft()

    def test_exact_recorded_source_overlap_rejected(self):
        raw_sha = self.records[0]["measurement"]["raw_hashes_before_decode"]["sha256"]
        self.alter_old_package("evidence/fhm.csv", [{"id": "synthetic_old_0", "role": "development", "source_audio_sha256": raw_sha}])
        with self.assertRaisesRegex(ValueError, "Exact raw hash overlap"):
            self.run_draft()

    def test_bad_receipt_seal(self):
        name = self.root / p.PHYSICAL / "receipts/synthetic_record_0.json"
        receipt = p.strict_json(name.read_bytes())
        receipt["record_sha256"] = "0" * 64
        name.write_bytes(p.canonical(receipt))
        with self.assertRaisesRegex(ValueError, "Receipt record seal"):
            self.run_draft()

    def test_marker_required(self):
        name = "audit/saraga_catalog_v1.json"
        (self.root / name).write_bytes(b"{}")
        self.pins[name] = p.sha(b"{}")
        with self.assertRaisesRegex(ValueError, "Synthetic marker missing"):
            self.run_draft()

    def test_duplicate_and_nonfinite_json(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'):
            with self.assertRaises(ValueError):
                p.strict_json(raw)

    def test_duplicate_ids_and_csv_headers(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            p.keyed([{"id": "synthetic_x"}] * 2, "id")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            p.csv_rows(b"id,id\nx,x\n")

    def test_no_audio_network_model_calls(self):
        seen = []
        original = p.read_bytes
        def guard(path):
            seen.append(str(path))
            self.assertNotIn(pathlib.Path(path).suffix.lower(), {".mp3", ".wav", ".flac", ".npz", ".pt", ".pth"})
            return original(path)
        with mock.patch.object(p, "read_bytes", side_effect=guard):
            self.run_draft()
        imports = {n.names[0].name for n in ast.walk(ast.parse(pathlib.Path(p.__file__).read_text())) if isinstance(n, ast.Import)}
        self.assertFalse(imports & {"subprocess", "socket", "requests", "soundfile", "numpy", "torch", "sklearn"})
        self.assertTrue(seen)

    def test_recheck_prevents_publication_after_input_changes(self):
        original = p.read_bytes
        name = self.root / "audit/saraga_catalog_v1.json"
        calls = {"n": 0}
        def changed(path):
            if pathlib.Path(path) == name:
                calls["n"] += 1
                if calls["n"] == 2:
                    return b"{}"
            return original(path)
        with mock.patch.object(p, "read_bytes", side_effect=changed), self.assertRaisesRegex(ValueError, "changed before publication"):
            self.run_draft()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
