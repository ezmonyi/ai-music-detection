#!/usr/bin/env python3
"""Synthetic fixtures only: never open the official NSynth source in these tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

import modulation_nsynth_pilot_v1 as pilot


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="q_pilot_synthetic_")
        cls.root = Path(cls.temp.name).resolve()
        cls.source = cls.root / "synthetic_source"
        (cls.source / "nsynth-test/audio").mkdir(parents=True)
        cls.protocol = cls.root / "synthetic_protocol.md"
        cls.protocol.write_text("Synthetic test protocol only. No real NSynth measurement.\n")
        (cls.source / "nsynth-test.jsonwav.tar.gz").write_bytes(b"synthetic archive placeholder; no real source")
        cls.archive_digest = pilot.sha256_file(cls.source / "nsynth-test.jsonwav.tar.gz")
        metadata, cls.rows = {}, {}
        t = np.arange(64000) / 16000
        for instrument in range(53):
            for note in range(2):
                name = f"synthetic_test_{instrument:03d}-{60 + note:03d}-100"
                record = {"instrument_str": f"synthetic_test_{instrument:03d}",
                          "instrument": instrument, "pitch": 60 + note, "velocity": 100,
                          "note_str": name, "sample_rate": 16000,
                          "instrument_source_str": "synthetic_fixture",
                          "instrument_family_str": "test_carrier"}
                # Every synthetic instrument/note has distinct decoded PCM.
                pcm = np.rint(6000 * np.cos(2 * np.pi * (550 + instrument * 7 + note * 2) * t)).astype(np.int16)
                path = cls.source / "nsynth-test/audio" / (name + ".wav")
                wavfile.write(path, 16000, pcm)
                cls.rows[name] = {"id": name, "metadata": record, "path": str(path),
                                  "file_sha256": pilot.sha256_file(path),
                                  "pcm_sha256": hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest(),
                                  "frames": 64000, "duration_s": 4, "native_channels": 1,
                                  "native_sample_rate": 16000, "ai_human_label": None,
                                  "role": "external_measurement_control_only"}
                metadata[name] = record
        pilot.write_json_new(cls.source / "nsynth-test/examples.json", metadata)
        with (cls.source / "manifest.jsonl").open("w") as stream:
            for row in cls.rows.values():
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        acq = Path(pilot.__file__).parent / "acquire_nsynth_test_controls_v1.py"
        pilot.write_json_new(cls.source / "acquisition_receipt.json", {
            "decoded_all": True, "notes": 106, "instruments": 53,
            "ai_human_labels_assigned": False, "features_extracted": False,
            "code_sha256": pilot.sha256_file(acq)})
        pilot.commit_directory(cls.source, {})
        cls.source_digest = pilot.sha256_file(cls.source / "COMMIT.json")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        # Only synthetic tests relax acquisition identity/count. CLI has no
        # override for either constant and production remains pinned to 4096.
        self.addCleanup(patch.stopall)
        patch.object(pilot, "NOTE_COUNT", 106).start()
        patch.object(pilot, "SOURCE_COMMIT_SHA256", self.source_digest).start()
        patch.object(pilot, "SOURCE_ARCHIVE_SHA256", self.archive_digest).start()

    def paths(self, tag):
        base = self.root / tag
        return base / "draft", base / "run"

    def make_draft(self, tag):
        draft_dir, run_dir = self.paths(tag)
        receipt = pilot.draft(self.source, self.protocol, draft_dir, run_dir)
        return receipt, run_dir

    def test_full_source_validation_and_no_decode_in_draft(self):
        with patch.object(pilot.wavfile, "read", side_effect=AssertionError("draft decoded audio")), \
             patch.object(pilot.candidate, "extract_modulation_candidate", side_effect=AssertionError("draft extracted Q")):
            receipt, _ = self.make_draft("no_extract")
        document = pilot.read_json(receipt["draft"])
        self.assertFalse(document["features_extracted"])
        self.assertEqual(len(document["selection"]["selected_notes"]), 54)
        self.assertEqual(len(document["selection"]["reserved_instrument_ids"]), 26)
        self.assertEqual(document["sources"]["notes"], 106)
        self.assertEqual(len(document["sources"]["all_source_byte_hashes"]), 111)
        self.assertEqual(receipt["freeze_sha256_for_review"], pilot.sha256_file(receipt["draft"]))

    def test_selection_is_order_invariant_and_pcm_distinct(self):
        baseline = pilot.selected_inventory(self.rows)
        reversed_rows = dict(reversed(list(self.rows.items())))
        self.assertEqual(baseline, pilot.selected_inventory(reversed_rows))
        self.assertEqual(set(baseline["pilot_instrument_ids"]) & set(baseline["reserved_instrument_ids"]), set())
        self.assertTrue(baseline["pilot_reserved_pcm_sets_disjoint"])
        for instrument in baseline["pilot_instrument_ids"]:
            notes = [r for r in baseline["selected_notes"] if r["metadata"]["instrument_str"] == instrument]
            self.assertEqual(len({r["pcm_sha256"] for r in notes}), 2)
            self.assertEqual({r["metadata"]["pitch"] for r in notes}, {60, 61})

    def test_distinct_pcm_selection_skips_duplicate_and_fails_without_replacement(self):
        rows = copy.deepcopy(self.rows)
        selected = pilot.selected_inventory(rows)
        instrument = selected["pilot_instrument_ids"][0]
        notes = [r for r in rows.values() if r["metadata"]["instrument_str"] == instrument]
        notes[1]["pcm_sha256"] = notes[0]["pcm_sha256"]
        with self.assertRaisesRegex(ValueError, "no substitution"):
            pilot.selected_inventory(rows)
        extra = copy.deepcopy(notes[1]); extra["id"] += "-extra"
        extra["metadata"]["pitch"] = 62; extra["pcm_sha256"] = "e" * 64
        rows[extra["id"]] = extra
        recovered = pilot.selected_inventory(rows)
        self.assertIn(extra["id"], {r["id"] for r in recovered["selected_notes"]})

    def test_partition_pcm_overlap_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        selection = pilot.selected_inventory(rows)
        pilot_note = selection["selected_notes"][0]
        reserved = next(r for r in rows.values() if r["metadata"]["instrument_str"] in selection["reserved_instrument_ids"])
        reserved["pcm_sha256"] = pilot_note["pcm_sha256"]
        with self.assertRaisesRegex(ValueError, "share decoded PCM"):
            pilot.selected_inventory(rows)

    def test_exact_eight_interventions_and_chirp_window_rates(self):
        x = np.ones(64000)
        variants = pilot.derivatives(x)
        self.assertEqual(tuple(variants), pilot.CONDITIONS)
        self.assertEqual(len(variants), 8)
        np.testing.assert_array_equal(variants["baseline"], .25 * x)
        np.testing.assert_array_equal(variants["common_gain"], .1 * x)
        np.testing.assert_array_equal(variants["polarity"], -.25 * x)
        t = np.arange(64000) / 16000
        np.testing.assert_allclose(variants["chirp16to64_depth06"], .25 * (1 + .6 * np.cos(2 * np.pi * (16*t + 6*t*t))))
        self.assertEqual(16 + 12 * .25, 19)
        self.assertEqual(16 + 12 * 2.25, 43)

    def test_guards_before_output_creation(self):
        receipt, output = self.make_draft("guards")
        with self.assertRaisesRegex(ValueError, "freeze SHA256"):
            pilot.run(receipt["draft"], "0" * 64, output)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            pilot.run(receipt["draft"], receipt["freeze_sha256_for_review"], output.parent / "other")
        with patch.object(pilot, "code_bindings", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "differs from freeze"):
                pilot.run(receipt["draft"], receipt["freeze_sha256_for_review"], output)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "outside"):
            pilot.new_output_path(self.source / "forbidden", self.source)

    def test_source_byte_and_inventory_mismatch_are_rejected(self):
        with patch.object(pilot, "product_fingerprint", return_value={"sha256": "wrong", "bytes": 0}):
            with self.assertRaisesRegex(ValueError, "source product mismatch"):
                pilot.verify_source(self.source)
        with patch.object(pilot, "NOTE_COUNT", 4096):
            with self.assertRaisesRegex(ValueError, "receipt"):
                pilot.verify_source(self.source)
        with patch.object(pilot, "SOURCE_COMMIT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "parent-verified"):
                pilot.verify_source(self.source)

    def test_unsafe_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsafe"):
            pilot.safe_child(self.source, "../escape")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            pilot.safe_child(self.source, "/absolute")

    def test_incomplete_draft_publication_is_rejected(self):
        receipt, output = self.make_draft("incomplete_draft")
        commit_path = Path(receipt["draft"]).parent / "COMMIT.json"
        commit_path.rename(commit_path.with_suffix(".incomplete"))
        with self.assertRaisesRegex(ValueError, "no COMMIT"):
            pilot.run(receipt["draft"], receipt["freeze_sha256_for_review"], output)
        self.assertFalse(output.exists())

    def test_instrument_balancing_and_missing_coverage(self):
        result = pilot.balanced_summary({"a": [0, 0], "b": [1, None], "c": [None, None]})
        self.assertEqual(result["instrument_balanced_mean"], .5)
        self.assertEqual(result["covered_instruments"], 2)
        self.assertEqual(result["instrument_denominator"], 3)
        self.assertEqual(result["valid_notes"], 3)
        self.assertEqual(result["note_denominator"], 6)

    def test_missing_to_finite_change_is_reported_as_invariance_failure(self):
        blank = pilot.candidate.extract_modulation_candidate(np.zeros(64000), 16000)
        changed = copy.deepcopy(blank)
        changed["features"][pilot.candidate.FEATURE_NAMES[0]] = .2
        comparison = pilot.invariant_comparison(blank, changed)
        self.assertFalse(comparison["feature_missingness_match"])
        self.assertIsNone(comparison["max_abs_finite_feature_error"])
        self.assertEqual(comparison["comparable_features"], 0)

    def test_peak_means_dominant_peak_not_tautological_nearest_bin(self):
        t = np.arange(64000) / 16000
        x = (1 + .6 * np.cos(2*np.pi*48*t)) * np.cos(2*np.pi*2000*t)
        result = pilot.candidate.extract_modulation_candidate(x, 16000)
        correct = pilot.known_am_peak(result, 1, 48)
        incorrect = pilot.known_am_peak(result, 1, 8)
        self.assertTrue(correct["within_1hz"])
        self.assertFalse(incorrect["within_1hz"])
        self.assertEqual(incorrect["dominant_peak_hz"], 48)

    def test_complete_synthetic_run_commits_432_derivatives_and_never_reads_reserved(self):
        receipt, output = self.make_draft("complete")
        frozen = pilot.read_json(receipt["draft"])
        selected_paths = {r["path"] for r in frozen["selection"]["selected_notes"]}
        original_read = pilot.wavfile.read
        decoded_sources = []
        def guarded_read(path, *args, **kwargs):
            if Path(path).is_relative_to(self.source):
                self.assertIn(str(path), selected_paths)
                decoded_sources.append(str(path))
            return original_read(path, *args, **kwargs)
        with patch.object(pilot.wavfile, "read", side_effect=guarded_read):
            result = pilot.run(receipt["draft"], receipt["freeze_sha256_for_review"], output)
        self.assertEqual(result["notes"], 54)
        self.assertEqual(len(decoded_sources), 54)
        self.assertEqual(len(list((output / "derivatives").glob("*.wav"))), 432)
        self.assertEqual(len(list((output / "candidate_json").glob("*.json"))), 432)
        summary = pilot.read_json(output / "paired_summaries.json")
        self.assertEqual(len(summary["notes"]), 54)
        self.assertFalse(summary["external_gate_passed"])
        self.assertEqual(summary["status"], "pilot_only_not_external_gate_passed")
        commit = pilot.read_json(output / "COMMIT.json")
        self.assertEqual(len(commit["products"]), 867)
        for name, expected in commit["products"].items():
            self.assertEqual(pilot.product_fingerprint(output / name), expected)
        with self.assertRaisesRegex(ValueError, "new directory"):
            pilot.run(receipt["draft"], receipt["freeze_sha256_for_review"], output)

    def test_failure_leaves_no_commit(self):
        receipt, output = self.make_draft("fail_transaction")
        with patch.object(pilot.candidate, "extract_modulation_candidate", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                pilot.run(receipt["draft"], receipt["freeze_sha256_for_review"], output)
        self.assertTrue(output.exists())
        self.assertFalse((output / "COMMIT.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
