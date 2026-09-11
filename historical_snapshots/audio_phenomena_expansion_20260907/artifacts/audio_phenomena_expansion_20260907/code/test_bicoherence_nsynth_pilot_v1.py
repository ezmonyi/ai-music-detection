"""Offline synthetic fixtures only; never loads or measures the NSynth source."""
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.io import wavfile

import bicoherence_nsynth_pilot_v1 as pilot


def fake_measurement(samples, sample_rate):
    cells = []
    for bins in pilot.extractor.pair_grid().tolist():
        eligible = bins == pilot.TARGET and bool(np.any(samples))
        score = 0.25 if eligible else None
        cells.append({"frequency_bins": bins, "eligible": eligible,
                      "squared_bicoherence": score, "biphase_radians": None,
                      "energy_fractions": [0.1] * 3 if eligible else [None] * 3,
                      "primitive": {"squared_bicoherence": score}})
    return {"arrays": {"test_samples": samples},
            "metadata": {"pool_count": 1, "pools": [{"cells": cells}]}}


class OfflineFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / "source"
        (self.source / "nsynth-test/audio").mkdir(parents=True)
        self.rows = {}
        metadata = {}
        for inst_idx, instrument in enumerate(("instrument_a", "instrument_b")):
            for pitch in (60, 61):
                note_id = f"{instrument}-{pitch}"
                path = self.source / "nsynth-test/audio" / f"{note_id}.wav"
                integers = np.full(64000, 100 + inst_idx * 50 + pitch, dtype="<i2")
                pcm = integers.tobytes()
                with wave.open(str(path), "wb") as stream:
                    stream.setparams((1, 2, 16000, 64000, "NONE", "not compressed"))
                    stream.writeframes(pcm)
                m = {"note_str": note_id, "sample_rate": 16000, "instrument_str": instrument,
                     "instrument": inst_idx, "instrument_source_str": "acoustic",
                     "instrument_family_str": "test_family", "pitch": pitch, "velocity": 100}
                metadata[note_id] = m
                self.rows[note_id] = {"id": note_id, "path": str(path), "metadata": m,
                    "file_sha256": pilot.sha(path), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "native_channels": 1, "native_sample_rate": 16000, "frames": 64000,
                    "duration_s": 4, "ai_human_label": None, "role": "external_measurement_control_only"}
        pilot.write_json(self.source / "nsynth-test/examples.json", metadata)
        with (self.source / "manifest.jsonl").open("wb") as stream:
            for row in self.rows.values():
                stream.write((json.dumps(row) + "\n").encode())
        pilot.write_json(self.source / "acquisition_receipt.json", {
            "decoded_all": True, "notes": 4, "instruments": 2,
            "features_extracted": False, "ai_human_labels_assigned": False})
        (self.source / "nsynth-test.jsonwav.tar.gz").write_bytes(b"offline archive placeholder")
        pilot.commit_directory(self.source, {})
        for name, value in (("NOTE_COUNT", 4), ("INSTRUMENT_COUNT", 2), ("PILOT_COUNT", 1),
                            ("SOURCE_COMMIT_SHA", pilot.sha(self.source / "COMMIT.json")),
                            ("SOURCE_ARCHIVE_SHA", pilot.sha(self.source / "nsynth-test.jsonwav.tar.gz"))):
            patcher = mock.patch.object(pilot, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.roster = self.root / "roster.json"
        pilot.write_json(self.roster, {"selection": pilot.select_roster(self.rows)})
        self.protocol = self.root / "protocol.md"
        self.protocol.write_text("offline fixture protocol\n")
        self.bindings = {"protocol": pilot.fingerprint(self.protocol), "roster": pilot.fingerprint(self.roster),
                         "runner": {"sha256": "mock-runner"}, "runtime": {"version": "mock-runtime"}}
        patcher = mock.patch.object(pilot, "code_bindings", return_value=self.bindings)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_draft(self):
        return pilot.draft(self.source, self.protocol, self.roster, self.root / "draft", self.root / "run")

    def test_source_inventory_and_pcm_decode(self):
        binding, rows = pilot.verify_source(self.source)
        self.assertEqual(rows, self.rows)
        self.assertIn("manifest.jsonl", binding["products"])
        row = next(iter(rows.values()))
        x = pilot.read_pcm(row, self.source)
        self.assertEqual(x.dtype, np.float64)
        self.assertEqual(x.shape, (64000,))
        self.assertEqual(x[0], 160 / 32768)

    def test_source_mutation_fails_closed(self):
        (self.source / "manifest.jsonl").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            pilot.verify_source(self.source)

    def test_source_extra_file_fails_closed(self):
        (self.source / "extra").write_text("extra")
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            pilot.verify_source(self.source)

    def test_selected_pcm_digest_mismatch(self):
        row = dict(next(iter(self.rows.values())), pcm_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "PCM digest"):
            pilot.read_pcm(row, self.source)

    def test_draft_never_extracts_and_never_decodes_reserved(self):
        selected_ids = {r["id"] for r in pilot.select_roster(self.rows)["selected_notes"]}
        actual_read = pilot.read_pcm
        reads = []
        def checked_read(row, source):
            reads.append(row["id"])
            self.assertIn(row["id"], selected_ids)
            return actual_read(row, source)
        with mock.patch.object(pilot.extractor, "extract", side_effect=AssertionError("draft extracted")), \
                mock.patch.object(pilot, "read_pcm", side_effect=checked_read):
            made = self.make_draft()
        self.assertEqual(set(reads), selected_ids)
        self.assertEqual(len(reads), 4)  # Start/end recheck, two selected notes.
        document = pilot.read_json(made["draft"])
        self.assertFalse(document["features_extracted"])
        self.assertFalse(document["reserved_pcm_decoded"])
        self.assertEqual(pilot.sha(made["draft"]), made["freeze_sha256_for_review"])

    def test_roster_mismatch_rejected_without_extraction(self):
        roster = pilot.read_json(self.roster)
        roster["selection"]["selected_notes"].reverse()
        self.roster.write_bytes(pilot.canonical(roster))
        with mock.patch.object(pilot.extractor, "extract", side_effect=AssertionError("unexpected")):
            with self.assertRaisesRegex(ValueError, "exact reused Q roster"):
                self.make_draft()

    def test_wrong_freeze_rejected(self):
        made = self.make_draft()
        with mock.patch.object(pilot.extractor, "extract", side_effect=AssertionError("unexpected")):
            with self.assertRaisesRegex(ValueError, "reviewed draft SHA256"):
                pilot.run(made["draft"], "0" * 64, self.root / "run")
        self.assertFalse((self.root / "run").exists())

    def test_runtime_or_source_freeze_mutation_rejected_at_start(self):
        made = self.make_draft()
        self.bindings["runtime"]["version"] = "changed"
        with self.assertRaisesRegex(ValueError, "mismatch at run start"):
            pilot.run(made["draft"], made["freeze_sha256_for_review"], self.root / "run")

    def test_output_exclusive_and_frozen_path(self):
        made = self.make_draft()
        with self.assertRaisesRegex(ValueError, "frozen planned output"):
            pilot.run(made["draft"], made["freeze_sha256_for_review"], self.root / "other")
        (self.root / "run").mkdir()
        with self.assertRaisesRegex(ValueError, "new exclusive"):
            pilot.run(made["draft"], made["freeze_sha256_for_review"], self.root / "run")

    def test_mocked_run_preserves_all_conditions_and_commits_last(self):
        made = self.make_draft()
        with mock.patch.object(pilot.extractor, "extract", side_effect=fake_measurement) as mocked:
            done = pilot.run(made["draft"], made["freeze_sha256_for_review"], self.root / "run")
        self.assertEqual(mocked.call_count, 14)
        self.assertEqual(done["conditions"], 14)
        commit = pilot.read_json(self.root / "run/COMMIT.json")
        self.assertFalse(commit["independent_replay_passed"])
        self.assertFalse(commit["classifier_admitted"])
        for name, expected in commit["products"].items():
            self.assertEqual(pilot.product(self.root / "run" / name), expected)
        summary = pilot.read_json(self.root / "run/summary.json")
        self.assertEqual(summary["levels"]["0db"]["note_denominator"], 2)
        self.assertEqual(summary["levels"]["0db"]["instrument_denominator"], 1)

    def test_end_mutation_preserves_failure_without_commit(self):
        made = self.make_draft()
        actual_snapshot = pilot.snapshot
        calls = 0
        def mutate_end(*args):
            nonlocal calls
            calls += 1
            result = actual_snapshot(*args)
            if calls == 2:
                result = {**result, "source": {"changed": True}}
            return result
        with mock.patch.object(pilot, "snapshot", side_effect=mutate_end), \
                mock.patch.object(pilot.extractor, "extract", side_effect=fake_measurement):
            with self.assertRaisesRegex(ValueError, "changed during run"):
                pilot.run(made["draft"], made["freeze_sha256_for_review"], self.root / "run")
        self.assertTrue((self.root / "run/FAILED.json").is_file())
        self.assertTrue((self.root / "run/summary.json").is_file())
        self.assertFalse((self.root / "run/COMMIT.json").exists())


class ConstructionAndSummaryTests(unittest.TestCase):
    def test_immutable_code_protocol_roster_bindings_enforced(self):
        digests = {"protocol.md": pilot.PROTOCOL_SHA, "roster.json": pilot.ROSTER_SHA,
                   "bicoherence_audio_v1.py": pilot.EXTRACTOR_SHA,
                   "bicoherence_primitive_v2.py": pilot.PRIMITIVE_SHA}
        def fake_fingerprint(path):
            return {"path": str(path), "bytes": 1, "sha256": digests.get(Path(path).name, "other")}
        with mock.patch.object(pilot, "fingerprint", side_effect=fake_fingerprint), \
                mock.patch.object(pilot, "runtime_bindings", return_value={"runtime": "fixture"}):
            bindings = pilot.code_bindings("protocol.md", "roster.json")
            self.assertIn("runner_tests", bindings)
            self.assertIn("primitive_tests", bindings)
            for name in list(digests):
                original = digests[name]
                digests[name] = "mutated"
                with self.assertRaisesRegex(ValueError, "immutable .* hash mismatch"):
                    pilot.code_bindings("protocol.md", "roster.json")
                digests[name] = original

    def test_seed_phase_construction_and_rms_independent_replay(self):
        x = np.cos(2 * np.pi * 127 * np.arange(64000) / 16000)
        result = pilot.constructions(x, "offline-test-note")
        arrays = result["arrays"]
        expected_seed = int.from_bytes(hashlib.sha256(b"BC-injection-20260907|offline-test-note").digest()[:8], "big")
        self.assertEqual(result["metadata"]["seed"], expected_seed)
        knots = np.random.Generator(np.random.PCG64(expected_seed)).uniform(-np.pi, np.pi, (3, 33))
        np.testing.assert_array_equal(arrays["phase_knots_wrapped"], knots)
        phase = np.array([np.interp(np.arange(64000) / 16000, np.arange(33) / 8, row)
                          for row in np.unwrap(knots, axis=1)])
        np.testing.assert_array_equal(arrays["independent_phase_trajectories"], phase)
        np.testing.assert_array_equal(arrays["closed_phase_trajectories"][2], phase[0] + phase[1])
        t = np.arange(64000) / 16000
        closed = 0.2 * (np.cos(2 * np.pi * 500 * t + phase[0])
                        + np.cos(2 * np.pi * 750 * t + phase[1])
                        + np.cos(2 * np.pi * 1250 * t + phase[0] + phase[1]))
        np.testing.assert_allclose(arrays["closed_prototype"], closed, atol=4e-12, rtol=0)
        self.assertEqual(tuple(result["waveforms"]), pilot.CONDITIONS)
        b = 0.25 * x
        for level, db in (("minus6db", -6), ("0db", 0)):
            target_rms = np.sqrt(np.mean(b**2)) * 10**(db / 20)
            for kind in ("closed", "independent"):
                injected = result["waveforms"][f"{kind}_{level}"] - b
                self.assertAlmostEqual(np.sqrt(np.mean(injected**2)), target_rms, places=14)
                np.testing.assert_allclose(injected, arrays[f"{kind}_{level}_injection"], atol=3e-17)
        np.testing.assert_array_equal(result["waveforms"]["common_gain"], 0.1 * b)
        np.testing.assert_array_equal(result["waveforms"]["polarity"], -b)
        repeated = pilot.constructions(x, "offline-test-note")
        np.testing.assert_array_equal(result["waveforms"]["closed_0db"], repeated["waveforms"]["closed_0db"])

    def test_zero_rms_explicit_unsupported_and_missing(self):
        result = pilot.constructions(np.zeros(64000), "zero-note")
        self.assertEqual(result["metadata"]["status"], "unsupported_zero_background_rms")
        self.assertFalse(result["metadata"]["injected_full_record_rms_matched"])
        for samples in result["waveforms"].values():
            self.assertFalse(np.any(samples))
            desc = pilot.descriptor(fake_measurement(samples, 16000)["metadata"])
            self.assertFalse(desc["target"]["eligible"])
            self.assertIsNone(desc["target"]["squared_bicoherence"])

    def test_float64_wav_exact_roundtrip_no_clip_and_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.wav"
            samples = np.linspace(-3.5, 2.5, 64000, dtype=np.float64)
            pilot.save_waveform(path, samples)
            rate, restored = wavfile.read(path)
            self.assertEqual(rate, 16000)
            self.assertEqual(restored.dtype, np.float64)
            np.testing.assert_array_equal(restored, samples)
            with self.assertRaises(FileExistsError):
                pilot.save_waveform(path, samples)

    def test_equal_instrument_weight_and_missing_denominators(self):
        rows = [{"instrument": "a", "difference": 0.0}, {"instrument": "a", "difference": 0.2},
                {"instrument": "b", "difference": 0.9}, {"instrument": "b", "difference": None},
                {"instrument": "c", "difference": None}, {"instrument": "c", "difference": None}]
        summary = pilot.weighted_summary(rows)
        self.assertAlmostEqual(summary["equal_instrument_mean_difference"], 0.5)
        self.assertEqual(summary["note_denominator"], 6)
        self.assertEqual(summary["covered_notes"], 3)
        self.assertEqual(summary["instrument_denominator"], 3)
        self.assertEqual(summary["covered_instruments"], 2)
        self.assertEqual(summary["zero"], 1)

    def test_nuisance_missing_transitions_cannot_disappear(self):
        def desc(eligibility, values):
            return {"grid_eligibility": eligibility, "grid_squared_bicoherence": values,
                    "target": {"eligible": eligibility[0], "squared_bicoherence": values[0]}}
        comparison = pilot.nuisance(desc([True, False, True], [0.4, None, 0.2]),
                                   desc([False, True, True], [None, 0.7, 0.3]))
        self.assertEqual(comparison["grid_denominator"], 3)
        self.assertEqual(comparison["missing_to_finite_count"], 1)
        self.assertEqual(comparison["finite_to_missing_count"], 1)
        self.assertEqual(comparison["grid_eligibility_agreement_count"], 1)
        self.assertAlmostEqual(comparison["maximum_finite_b2_difference"], 0.1)
        self.assertFalse(comparison["target_eligibility_agreement"])

    def test_aggregate_strata_and_all_note_differences(self):
        notes = []
        for note_id, instrument, category, family, difference in [
                ("a1", "a", "acoustic", "strings", 0.1),
                ("a2", "a", "acoustic", "strings", None),
                ("b1", "b", "electronic", "keys", -0.2)]:
            baseline = pilot.descriptor(fake_measurement(np.ones(64000), 16000)["metadata"])
            conditions = {name: json.loads(json.dumps(baseline)) for name in pilot.CONDITIONS}
            for level in ("minus6db", "0db"):
                conditions[f"closed_{level}"]["target"]["eligible"] = difference is not None
                conditions[f"closed_{level}"]["target"]["squared_bicoherence"] = (
                    0.25 + difference if difference is not None else None)
            notes.append({"note_id": note_id, "instrument": instrument, "source_category": category,
                          "instrument_family": family, "conditions": conditions,
                          "nuisance": {name: pilot.nuisance(baseline, baseline)
                                       for name in ("common_gain", "polarity")}})
        summary = pilot.aggregate(notes)
        for level in summary["levels"].values():
            self.assertEqual(level["note_denominator"], 3)
            self.assertEqual(level["covered_notes"], 2)
            self.assertEqual(len(level["per_note"]), 3)
            self.assertAlmostEqual(level["equal_instrument_mean_difference"], -0.05)
            self.assertEqual(level["strata"]["source_category"]["acoustic"]["note_denominator"], 2)
            self.assertEqual(level["strata"]["instrument_family"]["strings"]["covered_notes"], 1)
            self.assertEqual(level["positive"], 1)
            self.assertEqual(level["negative"], 1)
        json.dumps(summary, allow_nan=False)

    def test_full_53_instrument_roster_deterministic_distinct_pcm_reserved_exclusion(self):
        rows = {}
        for i in range(53):
            for n, pitch in enumerate((60, 60, 61)):
                note_id = f"i{i}-n{n}"
                rows[note_id] = {"id": note_id,
                    "metadata": {"instrument_str": f"i{i}", "pitch": pitch},
                    "pcm_sha256": hashlib.sha256(f"{i}-{int(n == 2)}".encode()).hexdigest()}
        selection = pilot.select_roster(rows)
        self.assertEqual(len(selection["selected_notes"]), 54)
        self.assertEqual(len(selection["reserved_instrument_ids"]), 26)
        self.assertEqual(selection, pilot.select_roster(dict(reversed(list(rows.items())))))
        for instrument in selection["pilot_instrument_ids"]:
            picked = [r for r in selection["selected_notes"] if r["metadata"]["instrument_str"] == instrument]
            self.assertEqual(len({r["pcm_sha256"] for r in picked}), 2)
        self.assertFalse({r["metadata"]["instrument_str"] for r in selection["selected_notes"]}
                         & set(selection["reserved_instrument_ids"]))
        reserved = selection["reserved_instrument_ids"][0]
        pilot_name = selection["pilot_instrument_ids"][0]
        pilot_pcm = next(r["pcm_sha256"] for r in rows.values() if r["metadata"]["instrument_str"] == pilot_name)
        next(r for r in rows.values() if r["metadata"]["instrument_str"] == reserved)["pcm_sha256"] = pilot_pcm
        with self.assertRaisesRegex(ValueError, "PCM collision"):
            pilot.select_roster(rows)


if __name__ == "__main__":
    unittest.main()
