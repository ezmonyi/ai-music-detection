#!/usr/bin/env python3
"""Synthetic audit tests. No producer or candidate import or real dataset read."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

import audit_modulation_nsynth_pilot_v1 as auditor


def save(path, value):
    Path(path).write_bytes(auditor.canonical(value))


def publish(root, **extra):
    products = {str(path.relative_to(root)): auditor.file_record(path)
                for path in root.rglob("*") if path.is_file() and path.name != "COMMIT.json"}
    save(root / "COMMIT.json", {"status": "committed", "products": products, **extra})


def synthetic_measurement(audio, condition="baseline", missing=False):
    """Hand-constructed valid power records, not claimed to be computed DSP."""
    if condition.startswith("am"):
        target = 8 if condition.startswith("am8_") else 48
        depth = .2 if condition.endswith("02") else .6
    else:
        target, depth = 24, .2
    power = np.zeros(501)
    if condition.startswith("chirp"):
        power[38:87] = .18 / 49
    elif not missing:
        power[2*target] = depth * depth / 2
    total = float(power.sum())
    fast = float(power[32:128].sum() / total) if total else None
    p = power[4:256] / total if total else np.zeros(252)
    entropy = float(-np.sum(p[p > 0]*np.log(p[p > 0]))/np.log(252)) if total else None
    energy = float(np.mean(audio[4000:36000]**2))
    status = "ok" if total else "no_modulation"
    bands, features = [], {}
    for i, name in enumerate(auditor.BAND_NAMES):
        window = {"index": 0, "start_seconds": .25, "end_seconds": 2.25,
                  "sample_count": 1000, "coverage": 1., "input_mean_square": energy,
                  "band_mean_square": energy * .2, "relative_band_energy": .2,
                  "envelope_mean": .1, "modulation_power": power.tolist(),
                  "total_modulation_power": total, "analysis_modulation_power": total,
                  "fast_fraction": fast, "entropy": entropy, "status": status}
        bands.append({"name": name, "carrier_hz": list(auditor.BANDS[i]), "windows": [window],
                      "status": status, "missing_reasons": [] if total else [status],
                      "valid_window_count": int(bool(total)), "valid_window_fraction": float(bool(total))})
        features[f"Q_{name}_fast_fraction_median"] = fast
        features[f"Q_{name}_entropy_median"] = entropy
    return {"version": "modulation_candidate_v1_20260907", "config": auditor.Q_CONFIG,
            "sample_count": 64000, "sample_rate_hz": 16000, "window_count": 1,
            "discarded_tail_envelope_samples": 750, "covered_seconds": 2.,
            "temporal_coverage_fraction": .5, "frequency_hz": (np.arange(501)*.5).tolist(),
            "bands": bands, "features": features, "valid_band_window_count": 3 if total else 0,
            "status": "ok" if total else "no_valid_band_windows"}


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="q_independent_audit_synthetic_")
        cls.root = Path(cls.temp.name).resolve()
        cls.source, cls.result, cls.draft = [cls.root / name for name in ("source", "result", "draft")]
        (cls.source / "nsynth-test/audio").mkdir(parents=True)
        (cls.result / "derivatives").mkdir(parents=True)
        (cls.result / "candidate_json").mkdir()
        cls.draft.mkdir()
        rows, metadata = [], {}
        t = np.arange(64000)/16000
        for inst in range(53):
            for note in range(2):
                name = f"audit_test_{inst:03d}-{60+note}-100"
                pcm = np.rint(6000*np.cos(2*np.pi*(600+inst*7+note*2)*t)).astype(np.int16)
                path = cls.source / "nsynth-test/audio" / (name+".wav")
                wavfile.write(path, 16000, pcm)
                meta = {"instrument_str": f"audit_test_{inst:03d}", "pitch": 60+note,
                        "instrument_source_str": "synthetic_fixture", "instrument_family_str": "carrier_fixture"}
                row = {"id": name, "metadata": meta, "path": str(path),
                       "file_sha256": auditor.digest(path), "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                       "native_sample_rate": 16000, "native_channels": 1, "frames": 64000,
                       "duration_s": 4, "ai_human_label": None}
                rows.append(row); metadata[name] = meta
        (cls.source / "nsynth-test.jsonwav.tar.gz").write_bytes(b"Synthetic archive fixture, not NSynth.")
        save(cls.source / "nsynth-test/examples.json", metadata)
        (cls.source / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")
        publish(cls.source)
        cls.source_commit = auditor.digest(cls.source / "COMMIT.json")
        cls.source_archive = auditor.digest(cls.source / "nsynth-test.jsonwav.tar.gz")
        instrument_order = sorted({r["metadata"]["instrument_str"] for r in rows}, key=lambda s:
                                  hashlib.sha256(("Q-pilot-20260907|"+s).encode()).hexdigest())
        selected = []
        for instrument in instrument_order[:27]:
            selected += sorted([r for r in rows if r["metadata"]["instrument_str"] == instrument],
                               key=lambda r: (abs(r["metadata"]["pitch"]-60),
                               hashlib.sha256(("Q-note-20260907|"+r["id"]).encode()).hexdigest()))
        cls.selected = selected
        product_hashes = auditor.load(cls.source / "COMMIT.json")["products"]
        frozen = {"sources": {"source_dir": str(cls.source),
                  "all_source_byte_hashes": {**product_hashes, "COMMIT.json": auditor.file_record(cls.source / "COMMIT.json")},
                  "all_4096_manifest_sha256": auditor.digest(cls.source / "manifest.jsonl"),
                  "all_4096_metadata_file_sha256": auditor.digest(cls.source / "nsynth-test/examples.json"),
                  "all_4096_manifest_metadata_sha256": hashlib.sha256(auditor.canonical(
                      [{"id": key, "metadata": metadata[key]} for key in sorted(metadata)])).hexdigest()},
                  "selection": {"ordered_instrument_ids": instrument_order, "pilot_instrument_ids": instrument_order[:27],
                  "reserved_instrument_ids": instrument_order[27:], "selected_notes": selected,
                  "pilot_reserved_pcm_sets_disjoint": True}, "bindings": {},
                  "planned_run_output_dir": str(cls.result), "features_extracted": False,
                  "reserved_waveforms_decoded_or_extracted": False,
                  "policy": {"status": auditor.STATUS, "conditions": list(auditor.CONDITIONS),
                             "no_numeric_admission_threshold": True, "codecs": False, "autofit": False}}
        save(cls.draft / "draft.json", frozen)
        cls.freeze = auditor.digest(cls.draft / "draft.json")
        publish(cls.draft, kind="development_pilot_draft", draft_sha256=cls.freeze)
        save(cls.result / "freeze.json", frozen)
        notes = []
        for row in selected:
            _, pcm = wavfile.read(row["path"])
            x = pcm.astype(np.float64)/32768
            replay = {}
            for condition in auditor.CONDITIONS:
                y = auditor.expected_audio(x, condition)
                stem = row["id"]+"__"+condition
                path = cls.result / "derivatives" / (stem+".wav")
                wavfile.write(path, 16000, y)
                measurement = synthetic_measurement(y, condition)
                save(cls.result / "candidate_json" / (stem+".json"), {
                     "note_id": row["id"], "instrument_str": row["metadata"]["instrument_str"],
                     "condition": condition, "source_file_sha256": row["file_sha256"],
                     "derivative": auditor.file_record(path), "measurement": measurement})
                replay[condition] = auditor.replay_measurement(measurement, y)
            notes.append(auditor.paired(row, replay))
        summary = {"status": auditor.STATUS, "freeze_sha256": cls.freeze, "notes": notes,
                   "instrument_balanced_descriptive_aggregates": auditor.aggregates(notes, instrument_order[:27]),
                   "instrument_balanced_source_and_family_strata": {}, "external_gate_passed": False,
                   "no_numeric_admission_threshold": True}
        for field in ("instrument_source_str", "instrument_family_str"):
            value = notes[0]["metadata"][field]
            summary["instrument_balanced_source_and_family_strata"][field] = {
                value: auditor.aggregates(notes, instrument_order[:27])}
        save(cls.result / "paired_summaries.json", summary)
        commit_record = auditor.file_record(cls.draft / "COMMIT.json")
        save(cls.result / "run_receipt.json", {"status": auditor.STATUS, "freeze_sha256": cls.freeze,
             "code_bindings_start": {}, "code_bindings_end": {}, "draft_commit_start": commit_record,
             "draft_commit_end": commit_record, "source_bindings_verified_start_and_end": True,
             "decoded_source_note_ids": [r["id"] for r in selected], "reserved_instrument_ids": instrument_order[27:],
             "reserved_waveforms_decoded_or_extracted": False, "notes": 54, "conditions_per_note": 8,
             "derivative_wavs": 432, "candidate_json_files": 432, "ai_human_labels_assigned": False,
             "classifier_fits": 0, "codecs": False, "external_gate_passed": False})
        publish(cls.result, kind="development_pilot_result", pilot_status=auditor.STATUS, freeze_sha256=cls.freeze)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(auditor, "SOURCE_COMMIT", self.source_commit).start()
        patch.object(auditor, "SOURCE_ARCHIVE", self.source_archive).start()
        patch.object(auditor, "SOURCE_NOTE_COUNT", 106).start()

    def run_fixture_audit(self):
        with patch.object(auditor, "verify_bindings"):
            return auditor.audit(self.result, self.draft / "draft.json", self.freeze)

    def test_complete_synthetic_audit_and_reserved_decode_guard(self):
        original = auditor.wavfile.read
        allowed = {r["path"] for r in self.selected}
        seen = []
        def guarded(path, *args, **kwargs):
            if Path(path).is_relative_to(self.source):
                self.assertIn(str(path), allowed); seen.append(str(path))
            return original(path, *args, **kwargs)
        with patch.object(auditor.wavfile, "read", side_effect=guarded):
            receipt = self.run_fixture_audit()
        self.assertTrue(receipt["passed"])
        self.assertFalse(receipt["external_gate_passed"])
        self.assertEqual(receipt["float64_derivatives_recreated"], 432)
        self.assertEqual(receipt["stored_spectra_replayed"], 1296)
        self.assertEqual(len(seen), 54)

    def test_power_replay_known_values_and_missingness(self):
        audio = np.ones(64000)*.2
        for condition, expected in (("am8_depth06", 0.), ("am48_depth06", 1.)):
            measurement = synthetic_measurement(audio, condition)
            result = auditor.replay_measurement(measurement, audio)
            self.assertEqual(result["bands"][0]["fast_fraction"], expected)
            self.assertEqual(result["bands"][0]["entropy"], 0.)
        missing = auditor.replay_measurement(synthetic_measurement(audio, missing=True), audio)
        self.assertTrue(all(v is None for v in missing["features"].values()))

    def test_scalar_and_spectrum_tampering_are_rejected(self):
        audio = np.ones(64000)*.2
        for mutation in ("scalar", "negative", "frequency", "status", "config", "coverage"):
            with self.subTest(mutation=mutation):
                measurement = synthetic_measurement(audio)
                if mutation == "scalar": measurement["features"][auditor.FEATURES[0]] = .55
                if mutation == "negative": measurement["bands"][0]["windows"][0]["modulation_power"][0] = -1
                if mutation == "frequency": measurement["frequency_hz"][1] = .6
                if mutation == "status": measurement["bands"][0]["windows"][0]["status"] = "silence"
                if mutation == "config":
                    measurement["config"] = copy.deepcopy(measurement["config"])
                    measurement["config"]["window_seconds"] = 3
                if mutation == "coverage": measurement["temporal_coverage_fraction"] = 1
                with self.assertRaises(ValueError): auditor.replay_measurement(measurement, audio)

    def test_independent_paired_analytic_endpoints(self):
        audio = np.ones(64000)*.2
        replay = {c: auditor.replay_measurement(synthetic_measurement(audio, c), audio) for c in auditor.CONDITIONS}
        result = auditor.paired(self.selected[0], replay)
        for band in result["bands"]:
            self.assertTrue(band["am8_depth06_peak"]["within_1hz"])
            self.assertTrue(band["am48_depth06_peak"]["within_1hz"])
            self.assertAlmostEqual(band["am8_near_power_ratio_depth06_over_depth02"], 9)
            self.assertAlmostEqual(band["am48_near_power_ratio_depth06_over_depth02"], 9)
            self.assertEqual(band["fast_fraction_delta_am48_minus_am8_depth06"], 1.)
            self.assertGreater(band["entropy_delta_chirp_minus_am48_depth06"], 0)
        self.assertEqual(result["common_gain"]["max_abs_finite_feature_error"], 0.)

    def test_weighting_uses_instruments_and_retains_missing_denominators(self):
        notes = [{"instrument_str": name, "value": value} for name, values in
                 (("a", [0., 0.]), ("b", [1., None]), ("c", [None, None])) for value in values]
        result = auditor.weighted(notes, ["a", "b", "c"], lambda note: note["value"])
        self.assertEqual(result["instrument_balanced_mean"], .5)
        self.assertEqual(result["covered_instruments"], 2)
        self.assertEqual(result["instrument_denominator"], 3)
        self.assertEqual(result["note_denominator"], 6)

    def test_missing_transition_is_not_hidden_in_invariance(self):
        audio = np.ones(64000)*.2
        full = auditor.replay_measurement(synthetic_measurement(audio), audio)
        missing = auditor.replay_measurement(synthetic_measurement(audio, missing=True), audio)
        result = auditor.invariant(full, missing)
        self.assertFalse(result["status_match"])
        self.assertFalse(result["feature_missingness_match"])
        self.assertEqual(result["comparable_features"], 0)
        self.assertIsNone(result["max_abs_finite_feature_error"])

    def test_derivative_formulas_and_chirp(self):
        x = np.ones(64000)
        t = np.arange(64000)/16000
        np.testing.assert_array_equal(auditor.expected_audio(x, "baseline"), .25*x)
        np.testing.assert_array_equal(auditor.expected_audio(x, "common_gain"), .1*x)
        np.testing.assert_array_equal(auditor.expected_audio(x, "polarity"), -.25*x)
        np.testing.assert_array_equal(auditor.expected_audio(x, "am48_depth06"), .25*(1+.6*np.cos(2*np.pi*48*t)))
        # Keep the protocol's t**2 evaluation order for bitwise WAV replay.
        np.testing.assert_array_equal(auditor.expected_audio(x, "chirp16to64_depth06"), .25*(1+.6*np.cos(2*np.pi*(16*t+6*t**2))))

    def test_exact_product_set_and_hash_rejects_changes(self):
        folder = self.root / "isolated_commit"
        folder.mkdir(); (folder / "a.txt").write_text("one")
        publish(folder)
        auditor.verify_products(folder)
        (folder / "a.txt").write_text("two")
        with self.assertRaisesRegex(ValueError, "hash mismatch"): auditor.verify_products(folder)
        (folder / "a.txt").write_text("one"); (folder / "extra.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "file set mismatch"): auditor.verify_products(folder)

    def test_frozen_code_pin_and_runtime_file_validation(self):
        file = self.root / "binding.txt"; file.write_text("bound")
        record = {"path": str(file), **auditor.file_record(file)}
        bindings = {"fixture": record, "runtime": {"python_executable": record, "module_files": {}}}
        with patch.object(auditor, "CODE_PINS", {"fixture": record["sha256"]}):
            auditor.verify_bindings(bindings)
            file.write_text("changed")
            with self.assertRaisesRegex(ValueError, "bound file changed"): auditor.verify_bindings(bindings)

    def test_freeze_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "freeze SHA"):
            auditor.audit(self.result, self.draft / "draft.json", "0"*64)

    def test_json_duplicate_keys_nonfinite_and_unsafe_paths_rejected(self):
        file = self.root / "bad.json"
        for text in ('{"x":1,"x":2}', '{"x":NaN}'):
            file.write_text(text)
            with self.assertRaises(ValueError): auditor.load(file)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            auditor.contained(self.result, "../source")


if __name__ == "__main__":
    unittest.main(verbosity=2)
