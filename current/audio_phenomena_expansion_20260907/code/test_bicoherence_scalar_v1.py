"""Synthetic metadata fixtures only: no source audio or observed BC results."""
import copy
import json
import unittest

import bicoherence_scalar_v1 as scalar


def fixture(values, tail=0, start=123):
    grid = [[a, b, a + b] for a in range(8, 193, 8)
            for b in range(a, 193, 8) if a + b <= 256]
    n = len(values) * 64000 + tail
    meta = {"version": "bicoherence_audio_v1", "primitive_version": "bicoherence_primitive_v2",
            "status": "ok" if values else "insufficient_support", "sample_rate_hz": 16000,
            "pool_samples": 64000, "pool_duration_seconds": 4.0, "n_fft": 1024,
            "hop_samples": 256, "frames_per_pool": 247, "grid_cell_count": 228,
            "energy_fraction_min_inclusive": 1e-6, "window": "periodic_hann",
            "tail_policy": "discard_no_padding", "pool_boundary_policy": "no_frame_crosses_pool_boundary",
            "input_samples": n, "pool_count": len(values), "analyzed_samples": len(values) * 64000,
            "discarded_tail_samples": tail, "tail_start_sample": len(values) * 64000, "pools": []}
    for index, value in enumerate(values):
        cells = []
        for bins in grid:
            target = bins == [32, 48, 80]
            score = value if target and value is not None else 0.3
            eligible = value is not None if target else True
            cells.append({"frequency_bins": bins, "eligible": eligible,
                          "energy_floor_passed": eligible,
                          "energy_fractions": [0.01] * 3 if eligible else [1e-8, 0.01, 0.01],
                          "status": "ok" if eligible else "below_energy_fraction_floor",
                          "squared_bicoherence": score if eligible else None,
                          "primitive": {"version": "bicoherence_primitive_v2", "realizations": 247,
                                        "frequency_bins": bins, "status": "ok", "squared_bicoherence": score}})
        meta["pools"].append({"pool_index": index, "start_sample": index * 64000,
                              "stop_sample_exclusive": (index + 1) * 64000,
                              "coefficient_rows": 247, "status": "ok", "zero_amplitude": False,
                              "total_non_dc_coefficient_energy": 1.0, "cells": cells})
    crop = {"start_sample": start, "stop_sample_exclusive": start + n,
            "source_resampled_samples": start + n + 77}
    return meta, crop


def target(meta):
    return next(c for c in meta["pools"][0]["cells"] if c["frequency_bins"] == [32, 48, 80])


class ScalarTests(unittest.TestCase):
    def compare(self, a, b):
        first, crop = fixture(a)
        second, other_crop = fixture(b)
        return scalar.compare_conditions(first, second, first_crop=crop, second_crop=other_crop)

    def test_two_fully_covered_equivalence_even_median(self):
        result = self.compare([0.8, 0.6], [0.2, 0.3])
        self.assertAlmostEqual(result["first"]["median_squared_bicoherence"], 0.7)
        self.assertAlmostEqual(result["operational_median_difference"], 0.45)
        self.assertAlmostEqual(result["paired_pool_mean_difference"], 0.45)
        self.assertEqual(result["paired_pool_count"], 2)

    def test_unequal_masks_have_different_estimands(self):
        result = self.compare([0.8, 0.2], [None, 0.1])
        self.assertAlmostEqual(result["operational_median_difference"], 0.4)
        self.assertAlmostEqual(result["paired_pool_mean_difference"], 0.1)
        self.assertEqual(result["paired_pool_mask"], [False, True])
        self.assertEqual(result["paired_pool_differences"], [None, 0.1])
        self.assertEqual(result["second"]["eligible_pool_count"], 1)
        self.assertEqual(result["total_pool_count"], 2)

    def test_disjoint_masks_operational_still_defined(self):
        result = self.compare([0.8, None], [None, 0.1])
        self.assertAlmostEqual(result["operational_median_difference"], 0.7)
        self.assertIsNone(result["paired_pool_mean_difference"])
        self.assertEqual(result["paired_status"], "no_common_eligible_pools")

    def test_more_than_two_pools_median_not_mean(self):
        result = self.compare([0.1, 0.2, 0.9], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(result["operational_median_difference"], 0.2)
        self.assertAlmostEqual(result["paired_pool_mean_difference"], 0.4)

    def test_zero_is_legitimate(self):
        meta, crop = fixture([0.0])
        result = scalar.reduce_metadata(meta, crop=crop)
        self.assertEqual(result["median_squared_bicoherence"], 0.0)
        self.assertEqual(result["status"], "ok")

    def test_all_missing_and_no_complete_pool(self):
        for values, tail, status in [([None, None], 9, "no_eligible_target_pools"),
                                     ([], 63999, "no_complete_pools"), ([], 0, "no_complete_pools")]:
            meta, crop = fixture(values, tail)
            result = scalar.reduce_metadata(meta, crop=crop)
            self.assertIsNone(result["median_squared_bicoherence"])
            self.assertEqual(result["status"], status)
            self.assertEqual(result["discarded_tail_samples"], tail)
            self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_zero_energy_scientific_missing(self):
        meta, crop = fixture([None])
        pool = meta["pools"][0]
        pool.update(status="zero_amplitude", zero_amplitude=True, total_non_dc_coefficient_energy=0.0)
        cell = target(meta)
        cell.update(status="zero_energy", energy_fractions=[None] * 3)
        cell["primitive"].update(status="zero_energy", squared_bicoherence=None)
        meta["construction_status"] = "unsupported_zero_background_rms"
        result = scalar.reduce_metadata(meta, crop=crop)
        self.assertEqual(result["missing_pool_count"], 1)

    def test_nonfinite_and_failure_not_silently_missing(self):
        for bad in [float("nan"), float("inf"), float("-inf")]:
            meta, crop = fixture([bad])
            with self.assertRaises(FloatingPointError):
                scalar.reduce_metadata(meta, crop=crop)
        meta, crop = fixture([None])
        target(meta)["squared_bicoherence"] = float("nan")
        with self.assertRaises(ValueError):
            scalar.reduce_metadata(meta, crop=crop)
        for level in ["metadata", "pool", "primitive"]:
            meta, crop = fixture([None])
            obj = meta if level == "metadata" else meta["pools"][0] if level == "pool" else target(meta)["primitive"]
            obj["status"] = "arithmetic_failure"
            with self.assertRaises(ValueError):
                scalar.reduce_metadata(meta, crop=crop)

    def test_reordered_and_duplicate_pools_rejected(self):
        for duplicate in [False, True]:
            meta, crop = fixture([0.1, 0.2])
            meta["pools"] = [meta["pools"][1], meta["pools"][0]] if not duplicate else [meta["pools"][0]] * 2
            with self.assertRaises(ValueError):
                scalar.reduce_metadata(meta, crop=crop)

    def test_crop_tail_support_mismatches_rejected(self):
        mutations = [("input_samples", 128124), ("pool_count", 1), ("analyzed_samples", 128001),
                     ("discarded_tail_samples", 124), ("tail_start_sample", 128001)]
        for key, value in mutations:
            meta, crop = fixture([0.1, 0.2], 123)
            meta[key] = value
            with self.assertRaises(ValueError):
                scalar.reduce_metadata(meta, crop=crop)
        meta, crop = fixture([0.1])
        for key, value in [("start_sample", -1), ("stop_sample_exclusive", 1000000),
                           ("source_resampled_samples", 1)]:
            broken = dict(crop, **{key: value})
            with self.assertRaises(ValueError):
                scalar.reduce_metadata(meta, crop=broken)
        other = {key: value + 1 for key, value in crop.items()}
        with self.assertRaises(ValueError):
            scalar.compare_conditions(meta, meta, first_crop=crop, second_crop=other)

    def test_invalid_target_grid_and_mask_rejected(self):
        meta, crop = fixture([0.2])
        for mutation in [lambda m: target(m).update(eligible=False),
                         lambda m: target(m).update(squared_bicoherence=0.4),
                         lambda m: target(m)["primitive"].update(squared_bicoherence=1.01),
                         lambda m: m["pools"][0]["cells"].reverse()]:
            broken = copy.deepcopy(meta)
            mutation(broken)
            with self.assertRaises(ValueError):
                scalar.reduce_metadata(broken, crop=crop)

    def test_input_unchanged_and_json_comparison(self):
        meta, crop = fixture([0.2, None], 123)
        before = copy.deepcopy(meta)
        result = scalar.compare_conditions(meta, meta, first_crop=crop, second_crop=crop)
        self.assertEqual(meta, before)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual(result["first"]["pools"][1]["source_start_sample"], 64123)
        self.assertFalse(result["first"]["classifier_admitted"])


if __name__ == "__main__":
    unittest.main()
