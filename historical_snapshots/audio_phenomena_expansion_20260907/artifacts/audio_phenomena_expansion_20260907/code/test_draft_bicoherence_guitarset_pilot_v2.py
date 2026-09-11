"""Offline synthetic provenance fixtures; no real GuitarSet access or BC calls."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.signal import resample_poly

import draft_bicoherence_guitarset_pilot_v2 as tool


def digest(data):
    return hashlib.sha256(data).hexdigest()


def synthetic_rows():
    rows = []
    for player in tool.EXPECTED_PLAYERS:
        for score_index in range(30):
            score = f"Style{score_index}-120-C"
            for performance in ("comp", "solo"):
                item_id = f"{player}_{score}_{performance}"
                audio_bytes = ("offline-audio-" + item_id).encode()
                annotation_bytes = ("offline-annotation-" + item_id).encode()
                audio_fp = {"bytes": len(audio_bytes), "sha256": digest(audio_bytes)}
                annotation_fp = {"bytes": len(annotation_bytes), "sha256": digest(annotation_bytes)}
                decoded = {"sample_rate_hz": 16000, "channels": 1, "subtype": "PCM_16", "format": "WAV",
                           "header_frames": 128009, "decoded_frames": 128009, "duration_seconds": 128009 / 16000,
                           "float64_samples_checked_finite": 128009, "nonfinite_samples": 0,
                           "empty_eof_observed": True, "decoded_pcm_sha256": digest(item_id.encode()),
                           "decoded_pcm_canonical_encoding": tool.PCM_ENCODING, "materialized_file": audio_fp}
                rows.append({"item_id": item_id, "player_id": player, "score_id": score,
                    "performance": performance, "class_label": None, "role": "external_measurement_control_only",
                    "annotation_is_ground_truth_for_nonlinear_coupling": False, "known_annotation_warnings": [],
                    "audio": {"archive": "audio_mono-mic.zip", "archive_member": item_id + "_mic.wav",
                              "archive_member_sha256": audio_fp["sha256"],
                              "materialized_path": f"audio/{item_id}_mic.wav", "materialized": audio_fp, "decoded": decoded},
                    "annotation": {"archive": "annotation.zip", "archive_member": item_id + ".jams",
                              "archive_member_sha256": annotation_fp["sha256"],
                              "materialized_path": f"annotations/{item_id}.jams", "materialized": annotation_fp}})
    return rows


class PureTests(unittest.TestCase):
    def test_v2_identity_and_reviewed_validator_hash_binding(self):
        self.assertEqual(tool.VERSION, "draft_bicoherence_guitarset_pilot_v2")
        self.assertEqual(Path(tool.source_validator.__file__).name, "validate_guitarset_archives_v2.py")
        self.assertEqual(tool.fp(tool.source_validator.__file__)["sha256"], tool.VALIDATOR_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            design = Path(tmp) / "design.md"
            design.write_text("offline design binding fixture\n")
            with mock.patch.object(tool, "DESIGN_SHA", tool.fp(design)["sha256"]), \
                    mock.patch.object(tool, "runtime_bindings", return_value={"runtime": "offline"}):
                bound = tool.code_bindings(design, tool.source_validator.__file__)
            self.assertEqual(bound["source_validator"]["sha256"], tool.VALIDATOR_SHA)
            self.assertEqual(Path(bound["draft_tool"]["path"]).name, "draft_bicoherence_guitarset_pilot_v2.py")
            self.assertEqual(Path(bound["draft_tests"]["path"]).name, "test_draft_bicoherence_guitarset_pilot_v2.py")

    def test_arbitrary_substituted_validator_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            design = Path(tmp) / "design.md"
            design.write_text("offline design binding fixture\n")
            arbitrary = Path(tmp) / "validate_guitarset_archives_v2.py"
            arbitrary.write_text("# unreviewed substituted validator\n")
            with mock.patch.object(tool, "DESIGN_SHA", tool.fp(design)["sha256"]), \
                    mock.patch.object(tool, "runtime_bindings", side_effect=AssertionError("must reject before runtime")):
                with self.assertRaisesRegex(ValueError, "Validator v2 immutable hash mismatch"):
                    tool.code_bindings(design, arbitrary)

    def test_runtime_executable_symlink_resolves_without_relaxing_source_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            binary = root / "actual-python"
            binary.write_bytes(b"offline executable placeholder")
            invocation = root / "venv-python"
            invocation.symlink_to(binary)
            recorded = tool.runtime_executable_binding(invocation)
            self.assertEqual(recorded["invocation_path"], str(invocation))
            self.assertTrue(recorded["invocation_is_symlink"])
            self.assertTrue(recorded["invocation_differs_from_resolved"])
            self.assertEqual(recorded["resolved_binary"], tool.binding(binary))
            with self.assertRaisesRegex(ValueError, "Symlink filesystem path forbidden"):
                tool.binding(invocation)

    def test_loaded_libsndfile_maps_binding_requires_unique_executable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp).resolve() / "libsndfile-wheelhash.so.1"
            library.write_bytes(b"offline ELF placeholder")
            maps = f"0000-1000 r--p 0000 00:01 1 {library}\n1000-2000 r-xp 1000 00:01 1 {library}\n"
            self.assertEqual(tool.libsndfile_from_maps(maps), library)
            with self.assertRaisesRegex(ValueError, "Cannot uniquely bind"):
                tool.libsndfile_from_maps(maps.replace("r-xp", "r--p"))
            second = library.with_name("libsndfile-other.so.1")
            second.write_bytes(b"second offline library")
            with self.assertRaisesRegex(ValueError, "Cannot uniquely bind"):
                tool.libsndfile_from_maps(maps + f"2000-3000 r-xp 0000 00:01 2 {second}\n")

    def test_deterministic_crossed_partition_all_roles(self):
        rows = synthetic_rows()
        a, b = tool.partition(rows), tool.partition(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual(a["role_counts"], {"development": 90, "reserved": 90, "unused": 180})
        self.assertEqual(len(a["rows"]), 360)
        self.assertFalse(set(a["development_player_ids"]) & set(a["reserved_player_ids"]))
        self.assertFalse(set(a["development_score_ids"]) & set(a["reserved_score_ids"]))
        pairs = {}
        for row in a["rows"]:
            pairs.setdefault((row["player_id"], row["score_id"]), set()).add(row["split_role"])
        self.assertTrue(all(len(roles) == 1 for roles in pairs.values()))
        expected_players = sorted(tool.EXPECTED_PLAYERS, key=lambda p: (digest(
            ("BC-GuitarSet-player-20260907|" + p).encode()), p))
        self.assertEqual(a["ordered_player_ids"], expected_players)

    def test_crossing_canonical_pcm_collision_rejected(self):
        rows = synthetic_rows()
        split = tool.partition(rows)
        dev = next(r for r in split["rows"] if r["split_role"] == "development")
        reserved = next(r for r in split["rows"] if r["split_role"] == "reserved")
        reserved["source_record"]["audio"]["decoded"]["decoded_pcm_sha256"] = dev["source_record"]["audio"]["decoded"]["decoded_pcm_sha256"]
        with self.assertRaisesRegex(ValueError, "PCM duplicate crosses"):
            tool.partition(rows)

    def test_bad_grid_metadata_short_and_empty_rejected(self):
        for change in (lambda rows: rows.pop(),
                       lambda rows: rows[0].update(player_id="09"),
                       lambda rows: rows[0]["audio"]["decoded"].update(channels=2),
                       lambda rows: rows[0]["audio"]["decoded"].update(decoded_frames=0)):
            rows = synthetic_rows()
            change(rows)
            with self.assertRaises(ValueError):
                tool.partition(rows)
        rows = synthetic_rows()
        d = rows[0]["audio"]["decoded"]
        d.update(decoded_frames=127999, header_frames=127999, float64_samples_checked_finite=127999,
                 duration_seconds=127999 / 16000)
        with self.assertRaisesRegex(ValueError, "Short source"):
            tool.partition(rows)

    def test_native_16k_floor_center_exact_no_resampler(self):
        x = np.arange(128009, dtype=np.float64)
        with mock.patch.object(tool, "resample_poly", side_effect=AssertionError("native 16k resampled")):
            crop, provenance = tool.standardize(x, 16000)
        np.testing.assert_array_equal(crop, x[4:128004])
        self.assertEqual(provenance["crop_start"], 4)
        self.assertFalse(provenance["resampling_applied"])
        self.assertEqual(provenance["crop_pcm_sha256"], digest(np.asarray(crop, dtype="<f8").tobytes()))

    def test_full_record_resample_before_floor_center(self):
        t = np.arange(360001, dtype=np.float64) / 44100
        x = np.sin(2 * np.pi * 713.2 * t) + np.linspace(-0.1, 0.3, len(t))
        crop, provenance = tool.standardize(x, 44100)
        full = resample_poly(x, 160, 441, window=("kaiser", 5.0), padtype="constant", cval=0.0)
        start = (len(full) - 128000) // 2
        np.testing.assert_array_equal(crop, full[start:start + 128000])
        self.assertEqual((provenance["up"], provenance["down"]), (160, 441))
        self.assertEqual(provenance["resampled_frames"], (360001 * 160 + 440) // 441)
        self.assertEqual(provenance["crop_start"], start)

    def test_zero_crop_preserved_unsupported_and_short_fails(self):
        crop, provenance = tool.standardize(np.zeros(128000), 16000)
        self.assertFalse(np.any(crop))
        self.assertEqual(provenance["measurement_support"], "unsupported_zero_background_rms")
        for signal, rate in ((np.empty(0), 16000), (np.zeros(127999), 16000),
                             (np.zeros(100), 44100), (np.array([np.nan]), 16000),
                             (np.zeros((128000, 1)), 16000), (np.zeros(128000), True)):
            with self.assertRaises(ValueError):
                tool.standardize(signal, rate)

    def test_selected_full_pcm_decode_eof_and_hash(self):
        samples = np.linspace(-0.25, 0.3, 128009)
        class Decoder:
            channels, samplerate, frames, format, subtype = 1, 16000, len(samples), "WAV", "DOUBLE"
            def __init__(self, *args, **kwargs): self.position = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, count, dtype, always_2d):
                self.assertion = (dtype, always_2d)
                block = samples[self.position:self.position + count, None]
                self.position += len(block)
                return block
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.wav"
            path.write_bytes(b"mock decoder content")
            metadata = {"sample_rate_hz": 16000, "decoded_frames": len(samples), "format": "WAV", "subtype": "DOUBLE",
                        "decoded_pcm_sha256": digest(np.asarray(samples, dtype="<f8").tobytes()),
                        "materialized_file": tool.fp(path)}
            result = tool.decode_selected(path, metadata, Decoder)
            np.testing.assert_array_equal(result, samples)
            metadata["decoded_pcm_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "PCM SHA mismatch"):
                tool.decode_selected(path, metadata, Decoder)


class ProvenanceFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "materialized"
        for folder in ("audio", "annotations", "source"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.rows = synthetic_rows()
        archives = {"audio_mono-mic.zip": [], "annotation.zip": []}
        for row in self.rows:
            for kind in ("audio", "annotation"):
                entry = row[kind]
                (self.root / entry["materialized_path"]).write_bytes((f"offline-{kind}-" + row["item_id"]).encode())
                archives[entry["archive"]].append({"path": entry["archive_member"], **entry["materialized"],
                    "is_directory": False, "apple_metadata": False, "crc_verified_by_full_read": True})
        self.write_json("archive_member_inventory.json", {"all_members_read_fully": True,
                    "all_members_crc_verified": True, "archives": archives})
        self.write_json("source/upstream_metadata.json", {"offline": True})
        patch = mock.patch.object(tool.source_validator, "METADATA_SHA256", tool.fp(self.root / "source/upstream_metadata.json")["sha256"])
        patch.start()
        self.addCleanup(patch.stop)
        archive_records = {name: {**expected, "sha256": digest(name.encode())}
                           for name, expected in tool.source_validator.EXPECTED_ARCHIVES.items()}
        self.write_json("source/acquisition_COMMIT.json", {"status": "committed", "kind": "archive_acquisition_only",
                        "products": {name: {k: rec[k] for k in ("bytes", "sha256")} for name, rec in archive_records.items()}})
        self.acquisition_sha = tool.fp(self.root / "source/acquisition_COMMIT.json")["sha256"]
        self.validator = self.base / "validator.py"
        self.validator.write_text("# offline validator identity\n")
        self.summary = {"status": "passed_archive_crc_materialization_and_physical_decode_not_measurement_admission",
            "role": "external_measurement_control_only", "class_label": None, "validator_code": tool.fp(self.validator),
            "source_manifest": "source_manifest.jsonl", "archive_member_inventory": "archive_member_inventory.json",
            "claims": {"overlap_clean": False, "overlap_audited": False, "bc_extracted": False,
                "partition_frozen": False, "classifier_fits": 0, "external_measurement_gate_passed": False,
                "audio_annotations_are_exact_identity_matched": True},
            "source": {"doi": "10.5281/zenodo.3371780", "version": "1.1.0", "license": "CC-BY-4.0",
                "official_metadata_sha256": tool.source_validator.METADATA_SHA256,
                "copied_official_metadata": tool.fp(self.root / "source/upstream_metadata.json"),
                "copied_acquisition_commit": tool.fp(self.root / "source/acquisition_COMMIT.json"),
                "acquisition_commit_sha256": self.acquisition_sha, "archives": archive_records},
            "counts": {"real_microphone_wav": 360, "real_jams": 360, "exact_audio_annotation_pairs": 360,
                "players": 6, "scores": 30, "files_per_player": dict(Counter(r["player_id"] for r in self.rows)),
                "files_per_score": dict(Counter(r["score_id"] for r in self.rows)), "performances": {"comp": 180, "solo": 180}},
            "audio": {"all_files_fully_decoded_float64_finite_to_empty_eof": True,
                "canonical_pcm_encoding": tool.PCM_ENCODING, "resampled": False, "mixed": False, "features_extracted": False,
                "channel_file_counts": {"1": 360}, "sample_rate_file_counts": {"16000": 360}, "total_decoded_frames": 360 * 128009},
            "annotations": {"silently_corrected_or_excluded_warning_items": 0, "ground_truth_for_nonlinear_coupling": False}}
        self.refresh()
        self.report = self.base / "overlap-report.md"
        self.report.write_text("Offline fixture accepted review; not real corpus evidence.\n")
        self.covered = self.base / "covered-manifest.json"
        self.covered.write_text('{"offline":true}\n')
        self.overlap_path = self.base / "overlap.json"
        self.overlap = {"schema": "bc_guitarset_overlap_acceptance_v1", "status": "accepted_for_development_split_only",
            "source_materialization_commit_sha256": self.commit_sha,
            "source_manifest_sha256": tool.fp(self.root / "source_manifest.jsonl")["sha256"],
            "report": tool.binding(self.report),
            "independent_review": {"accepted": True, "reviewer_id": "offline-reviewer",
                                   "reviewed_report_sha256": tool.fp(self.report)["sha256"]},
            "corpora": [{"corpus_id": "offline-corpus", "manifest": tool.binding(self.covered), "rows_checked": 1,
                "representations": ["native float64 PCM"], "checks": {"source_lineage": True, "identifiers": True,
                "exact_file_hashes": True, "canonical_pcm_hashes": True}, "limitations": ["No transformed content comparison"]}],
            "unresolved_matches": [], "global_non_overlap_proven": False}
        self.save_overlap()

    def write_json(self, name, value):
        (self.root / name).write_bytes(tool.canonical(value))

    def refresh(self):
        (self.root / "source_manifest.jsonl").write_bytes(b"".join(tool.canonical(r) for r in self.rows))
        self.write_json("validation_summary.json", self.summary)
        products = {p.relative_to(self.root).as_posix(): tool.fp(p) for p in self.root.rglob("*")
                    if p.is_file() and p.name != "COMMIT.json"}
        self.write_json("COMMIT.json", {"status": "committed", "kind": "guitarset_archive_validation_materialization_v1",
            "products": products, "source_commit_sha256": self.acquisition_sha, "classifier_fits": 0,
            "external_measurement_gate_passed": False})
        self.commit_sha = tool.fp(self.root / "COMMIT.json")["sha256"]

    def save_overlap(self):
        self.overlap_path.write_bytes(tool.canonical(self.overlap))
        self.overlap_sha = tool.fp(self.overlap_path)["sha256"]

    def verify(self):
        return tool.verify_materialization(self.root, self.commit_sha, self.validator)

    def test_full_exact_inventory_and_validation_scope(self):
        evidence, split = self.verify()
        self.assertEqual(len(evidence["products"]), 725)
        self.assertEqual(split["role_counts"]["development"], 90)

    def test_wrong_commit_inventory_and_bytes_fail(self):
        with self.assertRaisesRegex(ValueError, "COMMIT mismatch"):
            tool.verify_materialization(self.root, "0" * 64, self.validator)
        (self.root / "extra").write_text("extra")
        with self.assertRaisesRegex(ValueError, "exact product inventory"):
            self.verify()
        (self.root / "extra").unlink()
        path = self.root / self.rows[0]["audio"]["materialized_path"]
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "product hash mismatch"):
            self.verify()

    def test_wrong_summary_scope_fails_even_with_recommitted_fixture(self):
        self.summary["claims"]["overlap_clean"] = True
        self.refresh()
        with self.assertRaisesRegex(ValueError, "claims boundary"):
            self.verify()

    def test_wrong_native_metadata_fails(self):
        self.rows[0]["audio"]["decoded"]["channels"] = 2
        self.refresh()
        with self.assertRaisesRegex(ValueError, "must be mono"):
            self.verify()

    def test_wrong_filename_identity_fails(self):
        self.rows[0]["annotation"], self.rows[1]["annotation"] = self.rows[1]["annotation"], self.rows[0]["annotation"]
        self.refresh()
        with self.assertRaisesRegex(ValueError, "filename identity"):
            self.verify()

    def test_overlap_acceptance_and_referenced_manifest_binding(self):
        source, _ = self.verify()
        accepted = tool.verify_overlap(self.overlap_path, self.overlap_sha, source)
        self.assertFalse(accepted["evidence"]["global_non_overlap_proven"])
        self.covered.write_text("changed")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            tool.verify_overlap(self.overlap_path, self.overlap_sha, source)

    def test_overlap_missing_acceptance_unresolved_and_wrong_sha_fail(self):
        source, _ = self.verify()
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            tool.verify_overlap(self.overlap_path, "0" * 64, source)
        for change in (lambda e: e["independent_review"].update(accepted=False),
                       lambda e: e.update(unresolved_matches=["unresolved"]),
                       lambda e: e.update(global_non_overlap_proven=True),
                       lambda e: e.update(corpora=[])):
            original = deepcopy(self.overlap)
            change(self.overlap)
            self.save_overlap()
            with self.assertRaises(ValueError):
                tool.verify_overlap(self.overlap_path, self.overlap_sha, source)
            self.overlap = original

    def invoke_draft(self, output, decode=None, codes=None):
        return tool.draft(self.root, self.commit_sha, self.overlap_path, self.overlap_sha,
                          self.base / "design.md", self.validator, output)

    def test_draft_only_development_decode_no_bc_and_commit_last(self):
        selected = {r["item_id"] for r in tool.partition(self.rows)["rows"] if r["split_role"] == "development"}
        decoded = []
        def decode(path, meta):
            item = tool.source_validator.parse_media_identity(path.name, "audio")["item_id"]
            self.assertIn(item, selected)
            decoded.append(item)
            return np.zeros(meta["decoded_frames"], dtype=np.float64)
        with mock.patch.object(tool, "code_bindings", return_value={"runtime": "offline-mock"}), \
                mock.patch.object(tool, "decode_selected", side_effect=decode):
            result = self.invoke_draft(self.base / "draft")
        self.assertEqual(set(decoded), selected)
        self.assertEqual(len(decoded), 90)
        document = tool.read_json(result["draft"])
        self.assertEqual(document["version"], "draft_bicoherence_guitarset_pilot_v2")
        self.assertFalse(document["bc_extracted"])
        self.assertFalse(document["injections_constructed"])
        self.assertFalse(document["reserved_or_unused_decoded_by_this_tool"])
        self.assertEqual(len(document["development_preprocessing"]), 90)
        self.assertEqual(set(p.name for p in (self.base / "draft").iterdir()), {"draft.json", "COMMIT.json"})
        self.assertEqual(tool.read_json(self.base / "draft/COMMIT.json")["kind"],
                         "guitarset_bc_partition_preprocessing_draft_only_v2")
        self.assertEqual(tool.fp(result["draft"])["sha256"], result["draft_sha256_for_parent_review"])

    def test_output_conflict_and_end_binding_change_fail_closed(self):
        output = self.base / "draft"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.invoke_draft(output)
        other = self.base / "mutated-draft"
        with mock.patch.object(tool, "code_bindings", side_effect=[{"runtime": "one"}, {"runtime": "two"}]), \
                mock.patch.object(tool, "decode_selected", return_value=np.zeros(128009)):
            with self.assertRaisesRegex(ValueError, "Code/runtime changed"):
                self.invoke_draft(other)
        self.assertTrue((other / "FAILED.json").is_file())
        self.assertFalse((other / "COMMIT.json").exists())


if __name__ == "__main__":
    unittest.main()
