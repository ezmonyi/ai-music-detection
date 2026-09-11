#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module("overlap_audit_v2", HERE / "audit_guitarset_prior_corpus_overlap_v2.py")
v1_tests = load_module("overlap_audit_v1_tests", HERE / "test_audit_guitarset_prior_corpus_overlap_v1.py")


# Schema-shaped from the committed validator-v2 first source-manifest row.  The
# hashes and dimensions are metadata only; this test does not read source audio.
ACTUAL_FIRST_ROW = {
    "item_id": "00_BN1-129-Eb_comp",
    "player_id": "00",
    "score_id": "BN1-129-Eb",
    "performance": "comp",
    "version": "2.0",
    "audio": {
        "archive": "audio_mono-mic.zip",
        "archive_member": "00_BN1-129-Eb_comp_mic.wav",
        "archive_member_sha256":
            "59e1e3c9bebdabd48afe28050a2c031e6cf36e5697268301cdf2e74bc66b71f1",
        "materialized_path": "audio/00_BN1-129-Eb_comp_mic.wav",
        "materialized": {
            "bytes": 1969056,
            "sha256":
                "59e1e3c9bebdabd48afe28050a2c031e6cf36e5697268301cdf2e74bc66b71f1",
        },
        "decoded": {
            "channels": 1,
            "decode_block_frames": 65536,
            "decoded_frames": 984506,
            "decoded_pcm_canonical_encoding": audit.GUITARSET_PCM_ENCODING_V2,
            "decoded_pcm_sha256":
                "322d33252427cc2eda8e823f07b85ecb1a467061d7b1c40bc52ad4e834ae6482",
            "duration_seconds": 22.32439909297052,
            "empty_eof_observed": True,
            "float64_samples_checked_finite": 984506,
            "format": "WAV",
            "header_frames": 984506,
            "materialized_file": {
                "bytes": 1969056,
                "sha256":
                    "59e1e3c9bebdabd48afe28050a2c031e6cf36e5697268301cdf2e74bc66b71f1",
            },
            "nonfinite_samples": 0,
            "read_calls_including_empty_eof": 17,
            "sample_rate_hz": 44100,
            "subtype": "PCM_16",
        },
    },
}


def upgrade_fixture_to_v2(fixture):
    manifest = fixture.guitarset / "source_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        decoded = row["audio"]["decoded"]
        decoded.update({
            "decoded_pcm_canonical_encoding": audit.GUITARSET_PCM_ENCODING_V2,
            "header_frames": decoded["decoded_frames"],
            "float64_samples_checked_finite": decoded["decoded_frames"] * decoded["channels"],
            "nonfinite_samples": 0,
            "empty_eof_observed": True,
            "materialized_file": row["audio"]["materialized"],
        })
    v1_tests.write_jsonl(manifest, rows)
    summary_path = fixture.guitarset / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["audio"] = {"canonical_pcm_encoding": audit.GUITARSET_PCM_ENCODING_V2}
    summary["validator_code"] = {"bytes": 41380,
                                 "sha256":
                                 "cda25ca63e382a1363ade813cd114838099bee55ee60f346888f2b1cd2ff6d6c"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    products = {}
    for path in sorted(fixture.guitarset.rglob("*")):
        if path.is_file() and path.name != "COMMIT.json":
            products[path.relative_to(fixture.guitarset).as_posix()] = v1_tests.file_fp(path)
    commit = {"status": "committed",
              "kind": "guitarset_archive_validation_materialization_v1",
              "products": products}
    commit_path = fixture.guitarset / "COMMIT.json"
    commit_path.write_text(json.dumps(commit, sort_keys=True, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    fixture.commit_sha = v1_tests.file_fp(commit_path)["sha256"]


class OverlapAuditV2Tests(unittest.TestCase):
    def test_actual_first_row_descriptor_maps_to_canonical_f64(self):
        evidence = audit.validate_guitarset_decoded_descriptor(
            ACTUAL_FIRST_ROW["audio"]["decoded"])
        self.assertEqual(evidence["representation"], audit.F64_REPRESENTATION)
        self.assertEqual(evidence["semantic"], "full_source")
        self.assertEqual(evidence["sample_rate_hz"], 44100)
        self.assertEqual(evidence["channels"], 1)
        self.assertEqual(evidence["frames"], 984506)
        self.assertEqual(evidence["sha256"],
                         ACTUAL_FIRST_ROW["audio"]["decoded"]["decoded_pcm_sha256"])

    def test_incompatible_encoding_description_is_rejected(self):
        decoded = dict(ACTUAL_FIRST_ROW["audio"]["decoded"])
        decoded["decoded_pcm_canonical_encoding"] = (
            "IEEE754 float64 little-endian; C order [frame, channel]; no header"
        )
        with self.assertRaisesRegex(audit.AuditError, "Unexpected.*PCM encoding"):
            audit.validate_guitarset_decoded_descriptor(decoded)

    def test_full_decode_invariant_mismatch_is_rejected(self):
        for key, value in (("header_frames", 984505),
                           ("float64_samples_checked_finite", 984505),
                           ("nonfinite_samples", 1),
                           ("empty_eof_observed", False)):
            with self.subTest(key=key):
                decoded = dict(ACTUAL_FIRST_ROW["audio"]["decoded"])
                decoded[key] = value
                with self.assertRaises(audit.AuditError):
                    audit.validate_guitarset_decoded_descriptor(decoded)

    def test_v2_end_to_end_offline_report_pins_both_descriptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = v1_tests.Fixture(tmp)
            upgrade_fixture_to_v2(fixture)
            output = Path(tmp) / "out"
            report = audit.run(fixture.guitarset, fixture.commit_sha, fixture.inputs, output,
                               expected_counts=fixture.counts,
                               _expected_guitarset_shape=(1, 1))
            self.assertEqual(report["version"], audit.VERSION)
            contracts = report["encoding_contracts"]
            self.assertEqual(contracts["guitarset_validator_v2"]["exact_description"],
                             audit.GUITARSET_PCM_ENCODING_V2)
            self.assertEqual(contracts["saraga_whole_pcm"]["exact_description"],
                             audit.SARAGA_WHOLE_PCM_ENCODING)
            self.assertNotEqual(audit.GUITARSET_PCM_ENCODING_V2,
                                audit.SARAGA_WHOLE_PCM_ENCODING)
            self.assertEqual(contracts["guitarset_validator_v2"]["semantic_namespace"],
                             contracts["saraga_whole_pcm"]["semantic_namespace"])
            self.assertTrue(report["guitarset"]["decoded_pcm_contract"]
                            ["all_rows_full_decode_invariants_verified"])
            self.assertFalse(report["claims"]["global_non_overlap_established"])
            commit = json.loads((output / "COMMIT.json").read_text(encoding="utf-8"))
            self.assertEqual(commit["kind"], audit.VERSION)
            self.assertFalse(commit["independent_reviewer_acceptance"])

    def test_tampered_row_encoding_fails_visibly_without_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = v1_tests.Fixture(tmp)
            upgrade_fixture_to_v2(fixture)
            manifest = fixture.guitarset / "source_manifest.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[0]["audio"]["decoded"]["decoded_pcm_canonical_encoding"] += " altered"
            v1_tests.write_jsonl(manifest, rows)
            # Recommit the tampered bytes so failure is specifically the encoding contract.
            commit_path = fixture.guitarset / "COMMIT.json"
            commit = json.loads(commit_path.read_text())
            commit["products"]["source_manifest.jsonl"] = v1_tests.file_fp(manifest)
            commit_path.write_text(json.dumps(commit, sort_keys=True, separators=(",", ":")) + "\n")
            fixture.commit_sha = v1_tests.file_fp(commit_path)["sha256"]
            output = Path(tmp) / "out"
            with self.assertRaisesRegex(audit.AuditError, "Unexpected.*PCM encoding"):
                audit.run(fixture.guitarset, fixture.commit_sha, fixture.inputs, output,
                          expected_counts=fixture.counts,
                          _expected_guitarset_shape=(1, 1))
            self.assertTrue((output / "EXECUTION_FAILURE.json").is_file())
            self.assertFalse((output / "COMMIT.json").exists())


if __name__ == "__main__":
    unittest.main()
