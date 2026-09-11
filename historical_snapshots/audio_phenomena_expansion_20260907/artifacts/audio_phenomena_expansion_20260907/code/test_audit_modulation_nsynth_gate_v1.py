#!/usr/bin/env python3
"""Synthetic independent auditor tests. No producer, candidate or real-note reads."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

import audit_modulation_nsynth_gate_v1 as a
from test_audit_modulation_nsynth_pilot_v1 import synthetic_measurement


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(a.h.canonical(value))


def fixture_notes():
    audio = np.ones(64000)*.2
    replay = {c: a.h.replay_measurement(synthetic_measurement(audio, c), audio) for c in a.h.CONDITIONS}
    instruments = [f"fixture_{i:02d}" for i in range(26)]
    notes = []
    for instrument in instruments:
        for pitch in (60, 61):
            row = {"id": instrument+str(pitch), "metadata": {"instrument_str": instrument,
                   "instrument_source_str": "synthetic_fixture", "instrument_family_str": "fixture_family"}}
            views = {}
            for view in a.VIEWS:
                views[view] = a.h.paired(row, replay)
                views[view]["condition_features"] = {c: copy.deepcopy(replay[c]["features"]) for c in a.h.CONDITIONS}
            notes.append({"id": row["id"], "instrument_str": instrument, "metadata": row["metadata"], "views": views,
                          "codec_nuisance": {c: a.nuisance(views, c) for c in a.CODECS}})
    return notes, instruments


class GateAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="q_gate_audit_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_complete_success_and_exact_criterion_grid(self):
        notes, instruments = fixture_notes()
        result = a.decision(notes, instruments)
        self.assertTrue(result["criteria_passed"])
        self.assertEqual(len(result["criteria"]), 146)
        self.assertEqual(len({r["id"] for r in result["criteria"]}), 146)
        self.assertEqual(result["failed_criterion_ids"], [])
        self.assertFalse(result["external_gate_passed"])
        self.assertFalse(result["classifier_admitted"])

    def test_known_effect_and_depth_boundaries_are_per_note(self):
        notes, instruments = fixture_notes()
        for note in notes:
            band = note["views"]["original"]["bands"][0]
            band["fast_fraction_delta_am48_minus_am8_depth06"] = .25
            band["entropy_delta_chirp_minus_am48_depth06"] = .1
            band["am8_near_power_ratio_depth06_over_depth02"] = 1.
        result = a.decision(notes, instruments)
        failures = result["failed_criterion_ids"]
        self.assertEqual(failures, ["A/original/250_1000hz/am8_depth_increase"])
        for note in notes[:6]:
            note["views"]["original"]["bands"][0]["fast_fraction_delta_am48_minus_am8_depth06"] = .249
        self.assertIn("A/original/250_1000hz/fast_fraction_change", a.decision(notes, instruments)["failed_criterion_ids"])

    def test_conditional_two_level_weighting_and_fixed_denominators(self):
        notes = [{"instrument_str": n, "value": v} for n, vals in
                 (("a", [0., 0.]), ("b", [1., None]), ("c", [None, None])) for v in vals]
        value = a.endpoint(notes, ["a", "b", "c"], lambda n: n["value"])
        self.assertEqual(value["instrument_balanced_mean"], .5)
        self.assertEqual(value["note_coverage"], .5)
        self.assertEqual(value["instrument_coverage"], 2/3)
        self.assertFalse(a.sufficient(value))
        self.assertTrue(a.sufficient({"note_coverage": .8, "instrument_coverage": .8}))

    def test_unavailable_ratios_never_become_zero(self):
        notes, _ = fixture_notes()
        views = notes[0]["views"]
        feature = a.h.FEATURES[0]
        views["mp3_128k"]["condition_features"]["baseline"][feature] = None
        result = a.nuisance(views, "mp3_128k")["features"][feature]
        self.assertEqual(result["status"], "incomplete_eight_condition_pairs")
        self.assertEqual(result["comparable_conditions"], 7)
        self.assertIsNone(result["ratio"])
        for value, status in ((None, "missing_target_response"), (0., "nonpositive_target_response"), (-1., "nonpositive_target_response")):
            views["mp3_128k"]["condition_features"]["baseline"][feature] = 1.
            views["original"]["bands"][0]["fast_fraction_delta_am48_minus_am8_depth06"] = value
            self.assertEqual(a.nuisance(views, "mp3_128k")["features"][feature]["status"], status)

    def test_codec_numerator_uses_all_eight_and_note_ratio(self):
        notes, _ = fixture_notes()
        views = notes[0]["views"]
        key = a.h.FEATURES[0]
        views["mp3_128k"]["condition_features"]["polarity"][key] -= .4
        views["original"]["bands"][0]["fast_fraction_delta_am48_minus_am8_depth06"] = .5
        result = a.nuisance(views, "mp3_128k")["features"][key]
        self.assertAlmostEqual(result["max_eight_condition_absolute_error"], .4)
        self.assertAlmostEqual(result["ratio"], .8)

    def test_rare_large_codec_ratio_fails_despite_high_usual_rate(self):
        notes, instruments = fixture_notes()
        feature = a.h.FEATURES[0]
        notes[0]["codec_nuisance"]["mp3_128k"]["features"][feature]["ratio"] = 1.01
        result = a.decision(notes, instruments)
        self.assertEqual(result["failed_criterion_ids"], [f"C/mp3_128k/{feature}/nuisance"])
        evidence = next(c for c in result["criteria"] if c["id"].endswith(feature+"/nuisance") and "mp3" in c["id"])
        self.assertGreater(evidence["success"]["instrument_balanced_mean"], .9)

    def test_ratio_coverage_failure_and_signature_all_note_denominator(self):
        notes, instruments = fixture_notes()
        feature = a.h.FEATURES[0]
        for note in notes[:11]:
            note["codec_nuisance"]["mp3_128k"]["features"][feature]["ratio"] = None
        result = a.decision(notes, instruments)
        self.assertIn(f"C/mp3_128k/{feature}/nuisance", result["failed_criterion_ids"])
        for note in notes[:6]:
            note["codec_nuisance"]["aac_lc_64k"]["band_eligibility_signature_match"][a.h.BAND_NAMES[2]] = False
        self.assertIn("C/aac_lc_64k/3000_7000hz/eligibility_signature", a.decision(notes, instruments)["failed_criterion_ids"])

    def test_invariance_status_and_missingness_are_mandatory(self):
        notes, instruments = fixture_notes()
        notes[0]["views"]["original"]["common_gain"]["feature_missingness_match"] = False
        notes[1]["views"]["original"]["polarity"]["status_match"] = False
        self.assertEqual(a.decision(notes, instruments)["failed_criterion_ids"], ["B/original/common_gain", "B/original/polarity"])

    def test_gate_shape_and_descriptive_strata(self):
        notes, instruments = fixture_notes()
        with self.assertRaises(ValueError):
            a.decision(notes[:-1], instruments)
        changed = copy.deepcopy(notes)
        changed[-1]["instrument_str"] = instruments[0]
        with self.assertRaises(ValueError):
            a.decision(changed, instruments)
        result = a.strata(notes, instruments)
        self.assertEqual(set(result["overall"]), set(a.VIEWS))
        self.assertEqual(result["overall"]["original"], result["strata"]["instrument_source_str"]["synthetic_fixture"]["original"])

    def test_product_grid_counts_and_freeze_rejection(self):
        notes, _ = fixture_notes()
        paths = a.expected_paths(notes)
        self.assertEqual(len(paths), 6659)
        for directory, count in a.COUNTS.items():
            self.assertEqual(sum(p.startswith(directory+"/") for p in paths), count)
        draft = self.root / "draft.json"
        save(draft, {})
        with self.assertRaisesRegex(ValueError, "freeze SHA"):
            a.audit(self.root, draft, "0"*64)

    def codec_fixture(self, codec):
        spec = a.CODECS[codec]
        stem = "fixture__baseline"
        for directory in ("original", "encoded", "decoded_full", "analysis", "command_logs"):
            (self.root / directory).mkdir(exist_ok=True)
        original = .2*np.cos(2*np.pi*600*np.arange(64000)/16000)
        tag = stem+"__"+codec
        raw = self.root / "original" / (stem+".wav")
        encoded = self.root / "encoded" / (tag+spec["extension"])
        full = self.root / "decoded_full" / (tag+".wav")
        analysis = self.root / "analysis" / (tag+".wav")
        wavfile.write(raw, 16000, original)
        encoded.write_bytes(b"synthetic encoded-byte fixture, not real codec content")
        wavfile.write(full, 16000, np.r_[original, np.zeros(128)])
        wavfile.write(analysis, 16000, original)
        tools = {"ffmpeg": {"path": "/synthetic/ffmpeg"}, "ffprobe": {"path": "/synthetic/ffprobe"}}
        prefix = ["/synthetic/ffmpeg", "-hide_banner", "-nostdin", "-n"]
        commands = {"encode": prefix+["-i", str(raw), "-map", "0:a:0", "-vn", "-map_metadata", "-1", "-c:a", spec["encoder"],
           "-b:a", spec["requested_bitrate"], "-ar", "16000", "-ac", "1", "-threads", "1", "-fflags", "+bitexact"]
           +(["-profile:a", "aac_low"] if codec == "aac_lc_64k" else [])+[str(encoded)],
           "probe": ["/synthetic/ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams", "-show_format", "-of", "json", str(encoded)],
           "decode": prefix+["-i", str(encoded), "-map", "0:a:0", "-vn", "-map_metadata", "-1", "-c:a", "pcm_f64le", "-threads", "1", "-fflags", "+bitexact", str(full)]}
        metadata = {"streams": [{"codec_name": spec["codec_name"], "sample_rate": "16000", "channels": 1,
                   "profile": "LC" if codec == "aac_lc_64k" else "unknown", "bit_rate": "61234"}],
                   "format": {"filename": str(encoded), "format_name": "mp3" if codec == "mp3_128k" else "mov,mp4,m4a,3gp,3g2,mj2",
                              "size": str(encoded.stat().st_size)}}
        for step, command in commands.items():
            save(self.root / "command_logs" / (tag+"__"+step+".json"), {"command": command, "returncode": 0,
                "stdout": json.dumps(metadata) if step == "probe" else "", "stderr": "", "elapsed_seconds": .01})
        return stem, original, tools, metadata, full, analysis

    def test_codec_metadata_logs_slice_and_nonrequested_actual_bitrate(self):
        for codec in a.CODECS:
            stem, original, tools, metadata, _, _ = self.codec_fixture(codec)
            with patch.object(a, "call", return_value={"stdout": json.dumps(metadata), "stderr": ""}) as probe:
                audio, record = a.codec_record(self.root, stem, codec, tools, original)
            self.assertEqual(probe.call_count, 1)
            np.testing.assert_array_equal(audio, original)
            self.assertEqual(record["actual_stream_bit_rate"], "61234")
            self.assertEqual(record["discarded_tail_frames"], 128)
            self.assertAlmostEqual(record["zero_lag_correlation"], 1)

    def test_codec_crop_short_decode_metadata_and_command_tampering(self):
        codec = "aac_lc_64k"
        stem, original, tools, metadata, full, analysis = self.codec_fixture(codec)
        def check():
            return a.codec_record(self.root, stem, codec, tools, original, live_probe=False)
        wavfile.write(analysis, 16000, np.roll(original, 1))
        with self.assertRaisesRegex(ValueError, "slice"): check()
        wavfile.write(analysis, 16000, original)
        wavfile.write(full, 16000, original[:-1])
        with self.assertRaisesRegex(ValueError, "frame"): check()
        wavfile.write(full, 16000, original)
        log = self.root / "command_logs" / (stem+"__"+codec+"__decode.json")
        report = a.h.load(log)
        report["command"][4:4] = ["-ar", "22050"]
        save(log, report)
        with self.assertRaisesRegex(ValueError, "codec command"): check()
        self.codec_fixture(codec)
        probe = self.root / "command_logs" / (stem+"__"+codec+"__probe.json")
        report = a.h.load(probe)
        metadata["streams"][0]["sample_rate"] = "22050"
        report["stdout"] = json.dumps(metadata)
        save(probe, report)
        with self.assertRaisesRegex(ValueError, "native codec"): check()

    def test_independent_probe_rejects_forged_stored_metadata(self):
        stem, original, tools, metadata, _, _ = self.codec_fixture("mp3_128k")
        metadata["streams"][0]["bit_rate"] = "9999"
        with patch.object(a, "call", return_value={"stdout": json.dumps(metadata), "stderr": ""}):
            with self.assertRaisesRegex(ValueError, "independent ffprobe"):
                a.codec_record(self.root, stem, "mp3_128k", tools, original)

    def test_new_receipt_outside_all_protected_publications(self):
        draft = self.root / "draft" / "draft.json"
        source, result, pilot = (self.root / name for name in ("source", "result", "pilot"))
        save(draft, {"inputs": {"source": {"source_dir": str(source)}, "prior_pilot": {"pilot_dir": str(pilot)}}})
        for protected in (source, result, pilot, draft.parent):
            with self.assertRaisesRegex(ValueError, "protected"):
                a.receipt_path(protected / "new.json", result, draft)
        output = self.root / "audit.json"
        self.assertEqual(a.receipt_path(output, result, draft), output)
        save(output, {})
        with self.assertRaisesRegex(ValueError, "new audit"):
            a.receipt_path(output, result, draft)

    def test_bound_file_change_and_exact_helper_pin(self):
        file = self.root / "bound"
        file.write_bytes(b"one")
        record = {"path": str(file), **a.h.file_record(file)}
        a.bound_file(record)
        file.write_bytes(b"two")
        with self.assertRaises(ValueError): a.bound_file(record)
        self.assertEqual(a.h.digest(a._helper_path), a.HELPER_SHA)
        self.assertNotIn("modulation_candidate_v1", a.__dict__)

    def test_complete_synthetic_transaction_replays_all_1248_records(self):
        """Synthetic in-memory I/O fixture; source/tool verification has separate tests.

        Powers are intentionally hand constructed, never represented as full DSP.
        No real archive, note, producer, encoding or reserved waveform is accessed.
        """
        notes, instruments = fixture_notes()
        source, result, draft_dir = [self.root/name for name in ("synthetic_source", "result", "draft")]
        result.mkdir(); draft_dir.mkdir()
        pcm = np.tile(np.array([-6000, 6000], dtype=np.int16), 32000)
        x = pcm.astype(np.float64)/32768
        byte_record = {"bytes": 1, "sha256": "1"*64}
        rows = [{"id": n["id"], "metadata": n["metadata"], "path": str(source/(n["id"]+".wav")),
                 "file_sha256": "2"*64, "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest()} for n in notes]
        lookup = {row["id"]: row for row in rows}
        for note in notes:
            note["codec_records"] = {c: {v: {"synthetic_codec_record": v} for v in a.CODECS} for c in a.h.CONDITIONS}
        frozen = {"version": a.GATE_VERSION, "planned_run_output_dir": str(result), "features_extracted": False,
                  "reserved_note_codec_processing_performed": False, "policy": a.POLICY,
                  "bindings": {"codec_toolchain": {}}, "inputs": {"selection": {"gate_instrument_ids": instruments,
                  "selected_notes": rows, "excluded_development_instrument_ids": ["synthetic_pilot"]}}}
        draft = draft_dir/"draft.json"
        save(draft, frozen); save(result/"freeze.json", frozen)
        freeze = a.h.digest(draft)
        outcome = a.decision(notes, instruments)
        summary = {"version": a.GATE_VERSION, "freeze_sha256": freeze, "notes": notes,
                   "decision": outcome, "descriptive": a.strata(notes, instruments)}
        draft_commit = {"kind": "heldout_measurement_gate_draft", "draft_sha256": freeze, "products": {"draft.json": byte_record}}
        result_commit = {"kind": "heldout_measurement_gate_result", "freeze_sha256": freeze,
                         "products": {name: byte_record for name in a.expected_paths(rows)},
                         "criteria_passed": True, "decision_status": outcome["status"]}
        save(result/"COMMIT.json", result_commit)
        receipt = {"status": "execution_completed", "freeze_sha256": freeze, "draft_commit": byte_record,
            "bindings_start": frozen["bindings"], "bindings_end": frozen["bindings"], "input_bindings_rechecked_start_and_end": True,
            "decoded_source_note_ids": [n["id"] for n in notes], "gate_instrument_ids": instruments,
            "excluded_development_instrument_ids": ["synthetic_pilot"], "notes": 52, "instruments": 26,
            "counts": a.COUNTS, "decision_status": outcome["status"], "ai_human_labels_assigned": False,
            "classifier_fits": 0, "candidate_definition_changed": False, "padding_performed": False,
            "lag_optimization_performed": False, "external_gate_passed": False}
        real_load = a.h.load
        def load(path):
            path = Path(path)
            if path.name == "gate_results.json": return summary
            if path.name == "run_receipt.json": return receipt
            if path.parent.name == "candidate_json":
                note_id, condition, view = path.stem.split("__")
                row = lookup[note_id]
                relative = f"original/{note_id}__{condition}.wav" if view == "original" else f"analysis/{note_id}__{condition}__{view}.wav"
                return {"note_id": note_id, "instrument_str": row["metadata"]["instrument_str"], "condition": condition,
                        "view": view, "source_file_sha256": row["file_sha256"],
                        "analysis_waveform": {"path": relative, **byte_record},
                        "measurement": synthetic_measurement(a.h.expected_audio(x, condition), condition)}
            return real_load(path)
        def source_read(path):
            self.assertIn(str(path), {r["path"] for r in rows})
            return 16000, pcm.copy()
        def codec(root, stem, view, tools, original):
            return original.copy(), {"synthetic_codec_record": view}
        with patch.object(a, "verify_bindings") as bindings, patch.object(a, "verify_inputs", return_value=rows) as inputs, \
             patch.object(a.h, "verify_products", side_effect=lambda p: draft_commit if Path(p) == draft_dir else result_commit), \
             patch.object(a.h, "file_record", return_value=byte_record), patch.object(a.h, "load", side_effect=load), \
             patch.object(a.wavfile, "read", side_effect=source_read) as reads, \
             patch.object(a, "wave", side_effect=lambda p: a.h.expected_audio(x, Path(p).stem.split("__")[1])), \
             patch.object(a, "codec_record", side_effect=codec) as codecs, \
             patch.object(a.h, "replay_measurement", wraps=a.h.replay_measurement) as replay:
            audited = a.audit(result, draft, freeze)
        self.assertTrue(audited["passed"])
        self.assertTrue(audited["gate_criteria_passed"])
        self.assertFalse(audited["external_gate_passed"])
        self.assertEqual(replay.call_count, 1248)
        self.assertEqual(codecs.call_count, 832)
        self.assertEqual(reads.call_count, 52)
        self.assertEqual(bindings.call_count, 2)
        self.assertEqual(inputs.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
