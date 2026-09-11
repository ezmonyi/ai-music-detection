"""Synthetic producer tests only; no external source reads or real freezes."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

import bicoherence_guitarset_pilot_v1 as p


def fake_condition(values, width=228):
    pools = []
    for index, value in enumerate(values):
        eligible = value is not None
        pools.append({"pool_index": index, "pool_status": "ok", "grid_cell_count": width,
            "eligible_cell_count": width if eligible else 0, "grid_eligibility": [eligible] * width,
            "grid_status": ["ok" if eligible else "below_energy_fraction_floor"] * width,
            "grid_squared_bicoherence": [value] * width,
            "grid_raw_squared_bicoherence": [value] * width,
            "target": {"frequency_bins": [32, 48, 80], "eligible": eligible,
                "squared_bicoherence": value, "status": "ok" if eligible else "below_energy_fraction_floor"}})
    return {"pools": pools}


def fake_record(item="p_s_comp", player="p", score="s", performance="comp",
                closed=(.8, .8), independent=(.2, .2)):
    record = {"item_id": item, "player_id": player, "score_id": score,
        "performance": performance, "style_from_score_prefix": "s", "conditions": {}}
    for condition in p.CONDITIONS:
        values = closed if condition.startswith("closed") else independent
        record["conditions"][condition] = fake_condition(values)
    record["nuisance"] = {name: p.nuisance(record["conditions"]["baseline"], record["conditions"][name])
                          for name in ("common_gain", "polarity")}
    return record


class TestGuitarSetPilot(unittest.TestCase):
    def test_construction_seed_phase_and_rms(self):
        x = np.random.default_rng(5).normal(size=p.SAMPLES).astype(np.float64)
        built = p.constructions(x, "00_s_comp")
        expected_seed = int.from_bytes(hashlib.sha256(b"BC-GuitarSet-injection-20260907|00_s_comp").digest()[:8], "big")
        self.assertEqual(built["metadata"]["seed"], expected_seed)
        knots = np.random.Generator(np.random.PCG64(expected_seed)).uniform(-np.pi, np.pi, (3, 65))
        np.testing.assert_array_equal(built["arrays"]["phase_knots_wrapped"], knots)
        self.assertEqual(built["arrays"]["phase_knot_times_seconds"][-1], 8.)
        phases = built["arrays"]["closed_phase_trajectories"]
        np.testing.assert_array_equal(phases[2], phases[0] + phases[1])
        for label, db in (("minus6db", -6), ("0db", 0)):
            expected = built["metadata"]["background_rms"] * 10**(db / 20)
            for kind in ("closed", "independent"):
                injection = built["arrays"][f"{kind}_{label}_injection"]
                self.assertAlmostEqual(float(np.sqrt(np.mean(injection**2))), expected, places=14)
                np.testing.assert_array_equal(built["waveforms"][f"{kind}_{label}"], .25*x + injection)
        np.testing.assert_array_equal(built["waveforms"]["common_gain"], .1 * (.25*x))
        np.testing.assert_array_equal(built["waveforms"]["polarity"], -.25*x)
        self.assertFalse(built["metadata"]["framewise_power_matched"])

    def test_zero_background_is_retained_unsupported(self):
        built = p.constructions(np.zeros(p.SAMPLES, np.float64), "zero")
        self.assertEqual(built["metadata"]["status"], "unsupported_zero_background_rms")
        self.assertEqual(tuple(built["waveforms"]), p.CONDITIONS)
        self.assertTrue(all(not np.any(x) for x in built["waveforms"].values()))
        measured = p.extractor.extract(built["waveforms"]["baseline"], p.RATE)
        self.assertEqual(measured["metadata"]["pool_count"], 2)
        self.assertFalse(np.any(measured["arrays"]["eligible_mask"]))

    def test_no_clipping_and_exact_waveform_npz_roundtrip(self):
        built = p.constructions(np.full(p.SAMPLES, 8., np.float64), "large")
        self.assertGreater(np.max(built["waveforms"]["closed_0db"]), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p.save_waveform(root / "wave.wav", built["waveforms"]["closed_0db"])
            rate, actual = wavfile.read(root / "wave.wav")
            self.assertEqual(rate, p.RATE)
            np.testing.assert_array_equal(actual, built["waveforms"]["closed_0db"])
            p.save_arrays(root / "arrays.npz", built["arrays"])
            with np.load(root / "arrays.npz", allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(built["arrays"]))
                for name, original in built["arrays"].items():
                    np.testing.assert_array_equal(saved[name], original)
            with self.assertRaises(FileExistsError):
                p.save_waveform(root / "wave.wav", built["waveforms"]["baseline"])

    def test_invalid_construction_and_underflow(self):
        for x in (np.zeros(127999), np.zeros((p.SAMPLES, 1)), np.zeros(p.SAMPLES, np.float32),
                  np.full(p.SAMPLES, np.nan)):
            with self.assertRaises(ValueError):
                p.constructions(x, "bad")
        with self.assertRaises(FloatingPointError):
            p.constructions(np.full(p.SAMPLES, np.nextafter(0., 1.)), "underflow")

    def test_full_grid_gain_polarity_invariance_synthetic_audio(self):
        audio = np.random.default_rng(8).normal(size=p.SAMPLES).astype(np.float64)
        baseline = p.descriptor(p.extractor.extract(audio, p.RATE)["metadata"])
        for transformed in (.1 * audio, -audio):
            other = p.descriptor(p.extractor.extract(transformed, p.RATE)["metadata"])
            checks = p.nuisance(baseline, other)
            self.assertEqual(len(checks), 2)
            self.assertEqual(sum(c["grid_eligibility_agreement_count"] for c in checks), 456)
            self.assertEqual(sum(c["missing_to_finite_count"] + c["finite_to_missing_count"] for c in checks), 0)
            self.assertLess(max(c["maximum_finite_b2_difference"] for c in checks), 1e-12)
            self.assertLess(max(c["raw_maximum_finite_b2_difference"] for c in checks), 1e-12)

    def test_descriptor_rejects_wrong_pool_or_grid(self):
        metadata = p.extractor.extract(np.zeros(p.SAMPLES, np.float64), p.RATE)["metadata"]
        for field, value in (("pool_count", 1), ("discarded_tail_samples", 1), ("input_samples", 64000)):
            bad = deepcopy(metadata)
            bad[field] = value
            with self.assertRaises(ValueError):
                p.descriptor(bad)
        metadata["pools"][0]["cells"].pop()
        with self.assertRaises(ValueError):
            p.descriptor(metadata)

    def test_pair_before_averaging_and_unequal_group_weights(self):
        rows = [fake_record("a", "p1", "s1", closed=(.8, 1.), independent=(.2, None)),
                fake_record("b", "p2", "s1", "solo", closed=(.1, None), independent=(.5, .2)),
                fake_record("c", "p1", "s2", closed=(1., .8), independent=(0., 0.))]
        summary = p.aggregate(rows)["levels"]["0db"]
        # Record differences .6,-.4,.9; s1=.1,s2=.9 -> equal-score=.5.
        self.assertAlmostEqual(summary["equal_score"]["equal_group_mean_difference"], .5)
        self.assertAlmostEqual(summary["equal_player_secondary"]["equal_group_mean_difference"], .175)
        self.assertEqual(summary["paired_covered_pools"], 4)
        self.assertEqual(summary["pool_denominator"], 6)
        self.assertEqual(summary["negative_recordings"], 1)
        self.assertEqual([r["covered_pools"] for r in summary["per_recording"]], [1, 1, 2])
        self.assertEqual(summary["performance_strata"]["solo"]["negative_recordings"], 1)

    def test_no_same_pool_overlap_remains_missing(self):
        record = fake_record(closed=(.8, None), independent=(None, .2))
        summary = p.aggregate([record])["levels"]["minus6db"]
        self.assertEqual(summary["covered_recordings"], 0)
        self.assertEqual(summary["paired_covered_pools"], 0)
        self.assertIsNone(summary["equal_score"]["equal_group_mean_difference"])
        self.assertEqual(summary["equal_score"]["group_denominator"], 1)
        self.assertEqual(summary["equal_score"]["covered_groups"], 0)
        self.assertEqual(len(summary["per_recording"][0]["paired_pools"]), 2)

    def test_nuisance_transitions_both_directions_and_max(self):
        a, b = fake_condition((.2, None), 3), fake_condition((.3, .4), 3)
        b["pools"][0]["grid_eligibility"][1] = False
        b["pools"][0]["grid_squared_bicoherence"][1] = None
        checks = p.nuisance(a, b)
        self.assertEqual(sum(c["missing_to_finite_count"] for c in checks), 3)
        self.assertEqual(sum(c["finite_to_missing_count"] for c in checks), 1)
        self.assertEqual(sum(c["common_finite_cell_count"] for c in checks), 2)
        self.assertAlmostEqual(checks[0]["maximum_finite_b2_difference"], .1)

    def test_raw_invariance_audit_includes_below_floor_cells(self):
        a, b = fake_condition((None, None), 2), fake_condition((None, None), 2)
        a["pools"][0]["grid_raw_squared_bicoherence"] = [.1, .9]
        b["pools"][0]["grid_raw_squared_bicoherence"] = [.15, .6]
        checks = p.nuisance(a, b)
        self.assertEqual(checks[0]["common_finite_cell_count"], 0)
        self.assertEqual(checks[0]["raw_common_finite_cell_count"], 2)
        self.assertAlmostEqual(checks[0]["raw_maximum_finite_b2_difference"], .3)

    def test_reserved_guard_precedes_decode_or_measurement(self):
        for role in ("reserved", "unused"):
            with patch.object(p.drafting, "decode_selected") as decode, patch.object(p.extractor, "extract") as measure:
                with self.assertRaises(ValueError):
                    p.process_record({"split_role": role}, {}, "missing", Path("missing"))
                decode.assert_not_called()
                measure.assert_not_called()

    def test_storage_capacity_and_exclusive_output(self):
        estimate = p.storage_estimate()
        self.assertGreater(estimate["required_free_bytes"], 5 * 1024**3)
        with patch.object(p.shutil, "disk_usage") as usage:
            usage.return_value.free = estimate["required_free_bytes"] - 1
            with self.assertRaises(ValueError):
                p.ensure_capacity(Path("."), estimate)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, draft = root / "source", root / "draft"
            source.mkdir(); draft.mkdir()
            for output in (source / "output", draft / "output", root):
                with self.assertRaises(ValueError):
                    p.exclusive_output(output, source, draft, root / "freeze.json")
            self.assertEqual(p.exclusive_output(root / "new", source, draft, root / "freeze.json"), (root / "new").resolve())

    def test_untrusted_freeze_rejected_before_source_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "freeze.json"
            p.write_json(receipt, {"schema": "synthetic_wrong"})
            with patch.object(p.drafting, "decode_selected") as decode:
                with self.assertRaises(ValueError):
                    p.verify_freeze("absent", receipt, "0" * 64, "absent")
                with self.assertRaises(ValueError):
                    p.verify_freeze("absent", receipt, p.fp(receipt)["sha256"], "absent")
                decode.assert_not_called()

    def test_commit_inventory_and_failure_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p.write_json(root / "data.json", {"synthetic": True})
            p.commit_directory(root, {"kind": "synthetic_test_only"})
            receipt = p.read_json(root / "COMMIT.json")
            self.assertEqual(receipt["products"], {"data.json": p.fp(root / "data.json")})
            with self.assertRaises(ValueError):
                p.commit_directory(root, {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p.write_json(root / "FAILED.json", {})
            with self.assertRaises(ValueError):
                p.commit_directory(root, {})

    def test_parent_receipt_binds_code_draft_source_output_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            draft = root / "draft.json"
            p.write_json(draft, {"synthetic": True})
            p.write_json(root / "COMMIT.json", {"synthetic": True})
            receipt_path = root / "receipt.json"
            codes = {"synthetic_codes": "only_for_unit_test"}
            source_sha = "a" * 64
            document = {"source": {"commit": {"sha256": source_sha}}}
            receipt = {"schema": "bc_guitarset_development_parent_freeze_v1",
                "status": "authorized_development_only", "draft": p.binding(draft),
                "draft_commit": p.binding(root / "COMMIT.json"), "producer_bindings": codes,
                "policy": p.POLICY, "planned_output_dir": str(root / "output"),
                "source_commit_sha256": source_sha}
            with patch.object(p, "producer_bindings", return_value=codes), \
                 patch.object(p, "verify_draft", return_value=document):
                # Every variant gets its own synthetic receipt file; a matching
                # receipt hash alone must not bypass semantic input bindings.
                for index, (field, value) in enumerate(((None, None), ("producer_bindings", {}),
                        ("source_commit_sha256", "b"*64), ("policy", {}),
                        ("planned_output_dir", str(root / "wrong")), ("draft", {}))):
                    variant = deepcopy(receipt)
                    if field:
                        variant[field] = value
                    path = receipt_path.with_name(f"receipt_{index}.json")
                    p.write_json(path, variant)
                    args = (draft, path, p.fp(path)["sha256"], root / "output")
                    if field:
                        with self.assertRaises(ValueError):
                            p.verify_freeze(*args)
                    else:
                        self.assertEqual(p.verify_freeze(*args), (document, receipt))

    def test_one_synthetic_record_complete_waveform_and_stft_artifacts(self):
        native = np.random.default_rng(72).normal(scale=.1, size=p.SAMPLES+5).astype(np.float64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, output = root / "source", root / "output"
            source.mkdir(); output.mkdir()
            wavfile.write(source / "synthetic.wav", p.RATE, native)
            decoded = {"sample_rate_hz": p.RATE, "decoded_frames": len(native),
                "format": "WAV", "subtype": "DOUBLE", "materialized_file": p.fp(source / "synthetic.wav"),
                "decoded_pcm_sha256": hashlib.sha256(native.astype("<f8").tobytes()).hexdigest()}
            original = {"audio": {"materialized_path": "synthetic.wav", "decoded": decoded}}
            row = {"item_id": "synthetic", "player_id": "p", "score_id": "s", "performance": "comp",
                "style_from_score_prefix": "s", "split_role": "development", "source_record": original}
            crop, provenance = p.drafting.standardize(native, p.RATE)
            np.testing.assert_array_equal(crop, native[2:2+p.SAMPLES])
            class SyntheticDecoder:
                # Exercise the real selected-decode hash/EOF loop with the
                # draft's documented decoder injection seam. Local test runtime
                # has no soundfile; production remains bound to libsndfile.
                def __init__(self, path, mode):
                    self.samplerate, self.samples = wavfile.read(path)
                    self.channels, self.frames = 1, len(self.samples)
                    self.format, self.subtype, self.position = "WAV", "DOUBLE", 0
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
                def read(self, count, dtype, always_2d):
                    block = self.samples[self.position:self.position+count, None]
                    self.position += len(block)
                    return block
            actual_decode = p.drafting.decode_selected
            with patch.object(p.drafting, "decode_selected", side_effect=lambda path, meta:
                              actual_decode(path, meta, decoder_factory=SyntheticDecoder)):
                record = p.process_record(row, {"item_id": "synthetic", "preprocessing": provenance}, source, output)
            products = list((output / "synthetic").iterdir())
            self.assertEqual(len(products), 24)
            self.assertEqual(sum(path.suffix == ".wav" for path in products), 7)
            self.assertEqual(set(record["conditions"]), set(p.CONDITIONS))
            with np.load(output / "synthetic/construction.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["standardized_crop"], crop)
            with np.load(output / "synthetic/baseline.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["spectra"].shape, (2, 247, 513))
                frame = (.25*crop)[:1024]
                window = .5-.5*np.cos(2*np.pi*np.arange(1024)/1024)
                expected = np.fft.rfft((frame-frame.mean())*window)/window.sum()
                np.testing.assert_array_equal(saved["spectra"][0, 0], expected)
            metadata = p.read_json(output / "synthetic/baseline.json")
            self.assertEqual(sum(len(pool["cells"]) for pool in metadata["pools"]), 456)
            self.assertIn("raw_sums", metadata["pools"][0]["cells"][0]["primitive"])

    def test_synthetic_transaction_commit_last_and_end_binding_failure(self):
        # This transaction uses two synthetic records and a mocked trusted-input
        # boundary; it does not fabricate a real receipt or bypass production CLI.
        for fail_end in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "source").mkdir(); (root / "draft").mkdir()
                rows = [{"item_id": "synthetic_a", "split_role": "development"},
                        {"item_id": "synthetic_b", "split_role": "development"},
                        {"item_id": "never_decode", "split_role": "reserved"}]
                document = {"source": {"root": str(root / "source"), "commit": {"sha256": "0"*64}},
                    "split": {"rows": rows}, "development_preprocessing": rows[:2]}
                receipt = {"draft": {"sha256": "1"*64}}
                outputs = [(document, receipt), ValueError("changed bindings") if fail_end else (document, receipt)]
                def process(row, preparation, source, output):
                    self.assertEqual(row["split_role"], "development")
                    self.assertFalse((output / "COMMIT.json").exists())
                    p.write_json(output / (row["item_id"] + ".json"), {"synthetic": True})
                    return fake_record(item=row["item_id"])
                with patch.object(p, "DEVELOPMENT_COUNT", 2), patch.object(p, "verify_freeze", side_effect=outputs), \
                     patch.object(p, "ensure_capacity", return_value={"synthetic": True}), \
                     patch.object(p, "process_record", side_effect=process) as processing:
                    arguments = (root / "draft/draft.json", root / "freeze.json", "2"*64, root / "output")
                    if fail_end:
                        with self.assertRaisesRegex(ValueError, "changed bindings"):
                            p.run(*arguments)
                        self.assertFalse((root / "output/COMMIT.json").exists())
                        self.assertTrue((root / "output/FAILED.json").exists())
                    else:
                        result = p.run(*arguments)
                        self.assertEqual(result["recordings"], 2)
                        self.assertTrue((root / "output/COMMIT.json").exists())
                        self.assertFalse((root / "output/FAILED.json").exists())
                    self.assertEqual(processing.call_count, 2)


if __name__ == "__main__":
    unittest.main()
