#!/usr/bin/env python3
"""Synthetic tests for the frozen held-out NSynth measurement gate."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

import modulation_nsynth_gate_v1 as gate


def synthetic_roster():
    rows = {}
    for instrument in range(53):
        for note in range(3):
            key = f"gate_test_{instrument:03d}-{60 + note}"
            # The middle note duplicates the first PCM; selection must skip it.
            identity = f"{instrument}|{int(note == 2)}"
            rows[key] = {"id": key, "metadata": {"instrument_str": f"gate_test_{instrument:03d}",
                         "pitch": 60 + note}, "pcm_sha256": hashlib.sha256(identity.encode()).hexdigest()}
    return rows


def passing_notes():
    instruments = [f"synthetic_gate_{i:02d}" for i in range(26)]
    notes = []
    for instrument in instruments:
        for j in range(2):
            views = {}
            for view in gate.VIEWS:
                bands = []
                features = {condition: {} for condition in gate.CONDITIONS}
                for name in gate.pilot.candidate.BAND_NAMES:
                    bands.append({"name": name, "condition_status": dict.fromkeys(gate.CONDITIONS, "ok"),
                        "am8_depth06_peak": {"within_1hz": True}, "am48_depth06_peak": {"within_1hz": True},
                        "fast_fraction_delta_am48_minus_am8_depth06": .4,
                        "entropy_delta_chirp_minus_am48_depth06": .4,
                        "am8_near_power_ratio_depth06_over_depth02": 9.,
                        "am48_near_power_ratio_depth06_over_depth02": 9.})
                    for condition in gate.CONDITIONS:
                        features[condition][f"Q_{name}_fast_fraction_median"] = .6 if condition == "am48_depth06" else .2
                        features[condition][f"Q_{name}_entropy_median"] = .6 if condition == "chirp16to64_depth06" else .2
                invariance = {"status_match": True, "feature_missingness_match": True,
                              "max_abs_finite_feature_error": 0.}
                views[view] = {"bands": bands, "condition_features": features,
                               "common_gain": dict(invariance), "polarity": dict(invariance)}
            notes.append({"id": f"{instrument}-{j}", "instrument_str": instrument, "views": views,
                          "codec_nuisance": {codec: gate.codec_nuisance(views, codec) for codec in gate.CODECS}})
    return notes, instruments


def synthetic_run_fixture(root):
    """Mock only provenance for transaction tests; real source is never read."""
    source = root / "source"; source.mkdir()
    pcm = np.rint(3000*np.cos(2*np.pi*2000*np.arange(64000)/16000)).astype(np.int16)
    audio = source / "synthetic.wav"; wavfile.write(audio, 16000, pcm)
    _, instruments = passing_notes()
    rows = [{"id": f"{instrument}-{j}", "path": str(audio), "file_sha256": gate.pilot.sha256_file(audio),
             "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
             "metadata": {"instrument_str": instrument, "instrument_source_str": "synthetic_fixture",
                          "instrument_family_str": "test_carrier"}}
            for instrument in instruments for j in range(2)]
    inputs = {"source": {"source_dir": str(source)}, "prior_pilot": {"pilot_dir": str(root / "prior")},
              "selection": {"selected_notes": rows, "gate_instrument_ids": instruments,
                            "excluded_development_instrument_ids": [f"excluded_{i}" for i in range(27)]}}
    bindings = {"protocol": {"path": str(root / "synthetic_protocol.md")},
                "codec_toolchain": {"ffmpeg": {"path": "/synthetic/ffmpeg"}, "ffprobe": {"path": "/synthetic/ffprobe"}}}
    draft_dir = root / "draft"; draft_dir.mkdir()
    output = root / "run"
    frozen = {"version": gate.VERSION, "policy": gate.POLICY, "inputs": inputs, "bindings": bindings,
              "planned_run_output_dir": str(output), "features_extracted": False,
              "reserved_note_codec_processing_performed": False}
    gate.pilot.write_json_new(draft_dir / "draft.json", frozen)
    digest = gate.pilot.sha256_file(draft_dir / "draft.json")
    gate.pilot.commit_directory(draft_dir, {"kind": "heldout_measurement_gate_draft", "draft_sha256": digest})
    return draft_dir / "draft.json", digest, output, inputs, bindings


class GateTests(unittest.TestCase):
    def test_exact_original_reserves_and_distinct_pcm_selection(self):
        rows = synthetic_roster()
        prior = gate.pilot.selected_inventory(rows)
        selected = gate.reserved_roster(rows, prior)
        self.assertEqual(selected["gate_instrument_ids"], prior["reserved_instrument_ids"])
        self.assertEqual(len(selected["selected_notes"]), 52)
        self.assertEqual(len(selected["gate_instrument_ids"]), 26)
        self.assertFalse(set(selected["gate_instrument_ids"]) & set(prior["pilot_instrument_ids"]))
        for instrument in selected["gate_instrument_ids"]:
            notes = [r for r in selected["selected_notes"] if r["metadata"]["instrument_str"] == instrument]
            self.assertEqual({r["metadata"]["pitch"] for r in notes}, {60, 62})
            self.assertEqual(len({r["pcm_sha256"] for r in notes}), 2)

    def test_gate_selection_is_independent_of_mapping_order(self):
        rows = synthetic_roster()
        prior = gate.pilot.selected_inventory(rows)
        self.assertEqual(gate.reserved_roster(rows, prior),
                         gate.reserved_roster(dict(reversed(list(rows.items()))), prior))

    def test_insufficient_reserved_pcm_fails_without_substitution(self):
        rows = synthetic_roster()
        prior = gate.pilot.selected_inventory(rows)
        target = prior["reserved_instrument_ids"][0]
        affected = [r for r in rows.values() if r["metadata"]["instrument_str"] == target]
        for row in affected:
            row["pcm_sha256"] = affected[0]["pcm_sha256"]
        with self.assertRaisesRegex(ValueError, "no replacement"):
            gate.reserved_roster(rows, prior)

    def test_changed_prior_partition_is_rejected(self):
        rows = synthetic_roster()
        prior = copy.deepcopy(gate.pilot.selected_inventory(rows))
        prior["reserved_instrument_ids"] = list(reversed(prior["reserved_instrument_ids"]))
        with self.assertRaisesRegex(ValueError, "accepted pilot selection"):
            gate.reserved_roster(rows, prior)

    def test_prior_pilot_hash_guard_rejects_unaccepted_source(self):
        with tempfile.TemporaryDirectory(prefix="q_gate_hash_test_") as temp:
            root = Path(temp)
            (root / "COMMIT.json").write_text('{"status":"committed","products":{}}')
            with self.assertRaisesRegex(ValueError, "parent acceptance"):
                gate.verified_prior_pilot(root)

    def test_all_required_criteria_pass_on_synthetic_positive_controls(self):
        notes, instruments = passing_notes()
        result = gate.decisions(notes, instruments)
        self.assertTrue(result["criteria_passed"])
        self.assertFalse(result["external_gate_passed"])
        self.assertFalse(result["classifier_admitted"])
        self.assertEqual(len(result["criteria"]), 146)
        self.assertEqual(result["failed_criterion_ids"], [])

    def test_any_band_view_failure_fails_complete_family(self):
        notes, instruments = passing_notes()
        for note in notes:
            note["views"]["aac_lc_64k"]["bands"][2]["entropy_delta_chirp_minus_am48_depth06"] = .099
        decision = gate.decisions(notes, instruments)
        self.assertFalse(decision["criteria_passed"])
        self.assertIn("A/aac_lc_64k/3000_7000hz/entropy_change", decision["failed_criterion_ids"])

    def test_sensitivity_threshold_equality_and_strict_depth_ratio(self):
        notes, instruments = passing_notes()
        for note in notes:
            for view in gate.VIEWS:
                for band in note["views"][view]["bands"]:
                    band["fast_fraction_delta_am48_minus_am8_depth06"] = .25
                    band["entropy_delta_chirp_minus_am48_depth06"] = .10
        self.assertTrue(gate.decisions(notes, instruments)["criteria_passed"])
        for note in notes:
            note["views"]["original"]["bands"][0]["am8_near_power_ratio_depth06_over_depth02"] = 1.
        self.assertIn("A/original/250_1000hz/am8_depth_increase",
                      gate.decisions(notes, instruments)["failed_criterion_ids"])

    def test_coverage_and_success_thresholds_use_all_notes(self):
        notes, instruments = passing_notes()
        for note in notes[:11]:
            note["views"]["original"]["bands"][0]["am8_depth06_peak"]["within_1hz"] = None
        failed = gate.decisions(notes, instruments)
        criterion = next(c for c in failed["criteria"] if c["id"] == "A/original/250_1000hz/am8_peak")
        self.assertFalse(criterion["passed"])
        self.assertEqual(criterion["success"]["valid_notes"], 41)
        self.assertEqual(criterion["success"]["note_denominator"], 52)
        notes, instruments = passing_notes()
        for note in notes[:5]: note["views"]["original"]["bands"][0]["am8_depth06_peak"]["within_1hz"] = False
        self.assertTrue(gate.decisions(notes, instruments)["criteria_passed"])
        notes[5]["views"]["original"]["bands"][0]["am8_depth06_peak"]["within_1hz"] = False
        self.assertFalse(gate.decisions(notes, instruments)["criteria_passed"])

    def test_raw_invariance_missing_transition_and_error_bound(self):
        notes, instruments = passing_notes()
        notes[0]["views"]["original"]["common_gain"]["max_abs_finite_feature_error"] = 1e-9
        self.assertTrue(gate.decisions(notes, instruments)["criteria_passed"])
        notes[0]["views"]["original"]["common_gain"]["max_abs_finite_feature_error"] = 1.01e-9
        self.assertIn("B/original/common_gain", gate.decisions(notes, instruments)["failed_criterion_ids"])
        notes[0]["views"]["original"]["polarity"]["feature_missingness_match"] = False
        self.assertIn("B/original/polarity", gate.decisions(notes, instruments)["failed_criterion_ids"])

    def test_codec_all_eight_pairs_and_positive_denominator_are_required(self):
        notes, _ = passing_notes()
        views = notes[0]["views"]
        key = gate.pilot.candidate.FEATURE_NAMES[0]
        views["mp3_128k"]["condition_features"]["baseline"][key] = None
        nuisance = gate.codec_nuisance(views, "mp3_128k")["features"][key]
        self.assertEqual(nuisance["comparable_conditions"], 7)
        self.assertIsNone(nuisance["ratio"])
        self.assertIsNone(nuisance["max_eight_condition_absolute_error"])
        views["mp3_128k"]["condition_features"]["baseline"][key] = .2
        views["original"]["bands"][0]["fast_fraction_delta_am48_minus_am8_depth06"] = 0.
        nuisance = gate.codec_nuisance(views, "mp3_128k")["features"][key]
        self.assertEqual(nuisance["status"], "nonpositive_target_response")
        self.assertIsNone(nuisance["ratio"])

    def test_codec_maximum_over_eight_not_mean_and_ratio_bounds(self):
        notes, instruments = passing_notes()
        key = gate.pilot.candidate.FEATURE_NAMES[0]
        views = notes[0]["views"]
        views["mp3_128k"]["condition_features"]["baseline"][key] += .04
        nuisance = gate.codec_nuisance(views, "mp3_128k")["features"][key]
        self.assertAlmostEqual(nuisance["ratio"], .1)
        for note in notes:
            note["codec_nuisance"]["mp3_128k"]["features"][key]["ratio"] = .1
        self.assertTrue(gate.decisions(notes, instruments)["criteria_passed"])
        notes[0]["codec_nuisance"]["mp3_128k"]["features"][key]["ratio"] = 1.
        self.assertTrue(gate.decisions(notes, instruments)["criteria_passed"])
        notes[0]["codec_nuisance"]["mp3_128k"]["features"][key]["ratio"] = 1.001
        self.assertFalse(gate.decisions(notes, instruments)["criteria_passed"])

    def test_codec_eight_condition_eligibility_agreement(self):
        notes, instruments = passing_notes()
        key = gate.pilot.candidate.BAND_NAMES[0]
        for note in notes[:5]: note["codec_nuisance"]["mp3_128k"]["band_eligibility_signature_match"][key] = False
        self.assertTrue(gate.decisions(notes, instruments)["criteria_passed"])
        notes[5]["codec_nuisance"]["mp3_128k"]["band_eligibility_signature_match"][key] = False
        self.assertFalse(gate.decisions(notes, instruments)["criteria_passed"])

    def test_instrument_averaging_preserves_missing_denominators(self):
        notes = [{"instrument_str": inst, "value": value} for inst, values in
                 (("a", [0, 0]), ("b", [1, None]), ("c", [None, None])) for value in values]
        endpoint = gate.instrument_endpoint(notes, ["a", "b", "c"], lambda n: n["value"])
        self.assertEqual(endpoint["instrument_balanced_mean"], .5)
        self.assertEqual(endpoint["valid_notes"], 3)
        self.assertEqual(endpoint["note_denominator"], 6)
        self.assertEqual(endpoint["covered_instruments"], 2)

    def test_decode_never_pads_and_rejects_wrong_native_rate(self):
        with tempfile.TemporaryDirectory(prefix="q_gate_decode_") as temp:
            path = Path(temp) / "short.wav"
            wavfile.write(path, 16000, np.zeros(63999, dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "padding is forbidden"):
                gate.validated_analysis_decode(path)
            wavfile.write(path, 48000, np.zeros(64000, dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "native 16k"):
                gate.validated_analysis_decode(path)
            full = np.arange(65000, dtype=np.float64) / 65000
            wavfile.write(path, 16000, full)
            decoded, analyzed = gate.validated_analysis_decode(path)
            self.assertEqual(decoded.size, 65000)
            np.testing.assert_array_equal(analyzed, full[:64000])

    def test_real_codec_roundtrips_on_synthetic_four_second_carrier(self):
        ffmpeg = Path("/opt/homebrew/bin/ffmpeg")
        ffprobe = Path("/opt/homebrew/bin/ffprobe")
        self.assertTrue(ffmpeg.exists() and ffprobe.exists(), "pinned local codec tools required for synthetic integration")
        with tempfile.TemporaryDirectory(prefix="q_gate_codec_synthetic_") as temp:
            root = Path(temp).resolve()
            for name in ("encoded", "decoded_full", "analysis", "command_logs"):
                (root / name).mkdir()
            t = np.arange(64000)/16000
            x = .1*(1+.6*np.cos(2*np.pi*48*t))*np.cos(2*np.pi*2000*t)
            raw = root / "synthetic.wav"; wavfile.write(raw, 16000, x)
            tools = gate.toolchain_bindings(ffmpeg, ffprobe)
            for codec in gate.CODECS:
                path, record = gate.codec_roundtrip(raw, root, "synthetic", codec, tools, x)
                rate, analyzed = wavfile.read(path)
                self.assertEqual(rate, 16000)
                self.assertEqual(analyzed.dtype, np.float64)
                self.assertEqual(analyzed.size, 64000)
                self.assertGreaterEqual(record["full_decoded_frames"], 64000)
                self.assertFalse(record["lag_optimization_performed"])
                self.assertFalse(record["padding_performed"])
                self.assertEqual(record["requested_bitrate"], "128k" if codec == "mp3_128k" else "64k")
                self.assertIn("actual_stream_bit_rate", record)
            self.assertEqual(len(list((root / "command_logs").glob("*.json"))), 6)
            for path in (root / "command_logs").glob("*.json"):
                log = gate.pilot.read_json(path)
                self.assertEqual(log["returncode"], 0)
                self.assertIn("stdout", log); self.assertIn("stderr", log); self.assertIn("command", log)

    def test_command_failure_retains_full_log_and_does_not_retry(self):
        with tempfile.TemporaryDirectory(prefix="q_gate_command_failure_") as temp:
            log = Path(temp) / "failed.json"
            with self.assertRaisesRegex(RuntimeError, "full log"):
                gate.command_report(["/opt/homebrew/bin/ffmpeg", "-definitely_not_a_valid_option"], log_path=log)
            report = gate.pilot.read_json(log)
            self.assertNotEqual(report["returncode"], 0)
            self.assertTrue(report["stderr"])

    def test_draft_has_no_note_decode_q_or_codec_processing(self):
        with tempfile.TemporaryDirectory(prefix="q_gate_draft_fixture_") as temp:
            root = Path(temp).resolve()
            source = root / "synthetic_source"; source.mkdir()
            inputs = {"source": {"source_dir": str(source)}, "prior_pilot": {"pilot_dir": str(root / "prior")},
                      "selection": {"selected_notes": [], "gate_instrument_ids": []}}
            with patch.object(gate, "gate_inputs", return_value=({}, inputs)), \
                 patch.object(gate, "gate_bindings", return_value={"synthetic": True}), \
                 patch.object(gate.wavfile, "read", side_effect=AssertionError("draft decoded note")), \
                 patch.object(gate.pilot.candidate, "extract_modulation_candidate", side_effect=AssertionError("draft extracted Q")), \
                 patch.object(gate, "codec_roundtrip", side_effect=AssertionError("draft encoded note")):
                result = gate.draft(source, root / "prior", root / "protocol", "/unused/ffmpeg", "/unused/ffprobe",
                                    root / "draft", root / "run")
            frozen = gate.pilot.read_json(result["draft"])
            self.assertFalse(frozen["features_extracted"])
            self.assertFalse(frozen["reserved_note_codec_processing_performed"])
            gate.verify_gate_draft(result["draft"], result["freeze_sha256_for_review"])
            with self.assertRaisesRegex(ValueError, "freeze SHA256"):
                gate.verify_gate_draft(result["draft"], "0"*64)

    def test_scientific_failure_publishes_complete_mocked_codec_grid(self):
        with tempfile.TemporaryDirectory(prefix="q_gate_transaction_synthetic_") as temp:
            root = Path(temp).resolve()
            draft, freeze, output, inputs, bindings = synthetic_run_fixture(root)
            # Candidate mathematics and both real codecs are tested separately.
            # Here missing Q observations test fixed-grid failure publication.
            missing = gate.pilot.candidate.extract_modulation_candidate(np.zeros(64000), 16000)
            def fixture_codec(raw, out, stem, view, tools, original):
                full = out / "decoded_full" / (stem+"__"+view+".wav")
                analysis = out / "analysis" / (stem+"__"+view+".wav")
                full.hardlink_to(raw); analysis.hardlink_to(raw)
                (out / "encoded" / (stem+"__"+view+gate.CODECS[view]["extension"])).write_bytes(b"synthetic codec fixture")
                for operation in ("encode", "probe", "decode"):
                    gate.pilot.write_json_new(out / "command_logs" / (stem+"__"+view+"__"+operation+".json"),
                                              {"synthetic": True, "returncode": 0})
                return analysis, {"synthetic_codec_fixture": True}
            with patch.object(gate, "gate_inputs", return_value=({}, inputs)), \
                 patch.object(gate, "gate_bindings", return_value=bindings), \
                 patch.object(gate, "codec_roundtrip", side_effect=fixture_codec), \
                 patch.object(gate.pilot.candidate, "extract_modulation_candidate", return_value=missing) as extract:
                result = gate.run(draft, freeze, output)
            self.assertEqual(extract.call_count, 1248)
            self.assertFalse(result["criteria_passed"])
            self.assertEqual(result["status"], "criteria_failed")
            self.assertTrue((output / "COMMIT.json").exists())
            self.assertFalse((output / "EXECUTION_FAILURE.json").exists())
            receipt = gate.pilot.read_json(output / "run_receipt.json")
            self.assertEqual(receipt["counts"], {"original": 416, "encoded": 832, "decoded_full": 832,
                                               "analysis": 832, "candidate_json": 1248, "command_logs": 2496})
            commit = gate.pilot.read_json(output / "COMMIT.json")
            self.assertEqual(len(commit["products"]), 6659)
            for name, expected in commit["products"].items():
                self.assertEqual(gate.pilot.product_fingerprint(output / name), expected)

    def test_execution_failure_retains_artifacts_without_commit(self):
        with tempfile.TemporaryDirectory(prefix="q_gate_execution_failure_") as temp:
            root = Path(temp).resolve()
            draft, freeze, output, inputs, bindings = synthetic_run_fixture(root)
            with patch.object(gate, "gate_inputs", return_value=({}, inputs)), \
                 patch.object(gate, "gate_bindings", return_value=bindings), \
                 patch.object(gate, "codec_roundtrip", side_effect=RuntimeError("synthetic codec failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic codec failure"):
                    gate.run(draft, freeze, output)
            self.assertFalse((output / "COMMIT.json").exists())
            failure = gate.pilot.read_json(output / "EXECUTION_FAILURE.json")
            self.assertEqual(failure["status"], "execution_failed_no_commit")
            self.assertFalse(failure["settings_changed_or_retried"])
            self.assertEqual(len(list((output / "original").glob("*.wav"))), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
