"""Study safeguards: disjoint audit, frozen proposals, and rejected corruption."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from music_detector.rl.numerical_calibration import (
    audit_candidate, build_selection, digest, freeze_candidate, load_options,
    summarize, time_indices, verify_coverage, write_once,
)


class NumericalCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.options = load_options(Path(__file__).parents[1] / "configs/rl/numerical_calibration_v1.json")
        self.selection = build_selection(self.options)

    def rows(self, split, error=1e-5):
        return [{"key": g["key"], "split": split, "behavior": g["behavior"], "sample": sample,
                 "step": step, "recorded_logprob": .25, "replayed_logprob": .25 + error,
                 "repeated_logprob": .25 + error}
                for g in self.selection["groups"] if g["split"] == split
                for sample in range(4) for step in time_indices(range(10, 41), 8)]

    def test_pre_registered_groups_are_disjoint_and_exclude_discovery(self):
        self.assertEqual(build_selection(self.options), self.selection)
        self.assertEqual(len(self.selection["groups"]), 48)
        self.assertNotIn(0, self.selection["portable_blocks"])
        self.assertFalse(any(g["shard"] == "legacy24" and g["group"] in (0, 1) for g in self.selection["groups"]))
        for split in ("calibration", "audit"):
            self.assertEqual(sum(g["split"] == split for g in self.selection["groups"]), 24)

    def test_audit_outlier_cannot_refit_frozen_candidate(self):
        frozen = freeze_candidate(self.rows("calibration"), self.selection, self.options, controls_passed=True)
        original = copy.deepcopy(frozen)
        audit = self.rows("audit")
        audit[-1]["replayed_logprob"] = audit[-1]["repeated_logprob"] = .251
        result = audit_candidate(audit, frozen, self.selection, self.options, controls_passed=True)
        self.assertFalse(result["audit_passed"])
        self.assertEqual(frozen, original)
        self.assertFalse(result["production_protocol_activation"])

    def test_incomplete_or_duplicate_measurements_fail_closed(self):
        rows = self.rows("calibration")
        for broken in (rows[:-1], rows + [rows[0]], self.rows("audit")):
            with self.assertRaises(ValueError):
                verify_coverage(broken, self.selection, self.options, "calibration")

    def test_native_control_failure_forbids_budget(self):
        value = freeze_candidate(self.rows("calibration"), self.selection, self.options, controls_passed=False)
        self.assertIsNone(value["candidate_absolute_logprob_budget"])

    def test_no_candidate_above_precommitted_ratio_ceiling(self):
        value = freeze_candidate(self.rows("calibration", error=1e-3), self.selection, self.options, controls_passed=True)
        self.assertIsNone(value["candidate_absolute_logprob_budget"])
        self.assertLessEqual(math.expm1(max(self.options["candidate_absolute_logprob_budgets"])), .00010000000001)

    def test_cross_split_measurements_are_not_fit_data(self):
        with self.assertRaises(ValueError):
            freeze_candidate(self.rows("audit"), self.selection, self.options, controls_passed=True)

    def test_same_device_nondeterminism_is_an_independent_failure(self):
        rows = self.rows("calibration")
        rows[0]["repeated_logprob"] += .001
        value = freeze_candidate(rows, self.selection, self.options, controls_passed=True)
        self.assertIsNone(value["candidate_absolute_logprob_budget"])

    def test_missing_repeat_measurements_are_unknown_not_zero(self):
        rows = self.rows("calibration")
        rows[0]["repeated_logprob"] = None
        self.assertIsNone(summarize(rows)["same_gpu_repeat_logprob_error_maximum"])
        self.assertIsNone(freeze_candidate(rows, self.selection, self.options, controls_passed=True)["candidate_absolute_logprob_budget"])

    def test_nonfinite_likelihood_cannot_be_removed_as_outlier(self):
        for bad in (float('nan'), float('inf')):
            rows = self.rows("calibration"); rows[0]["replayed_logprob"] = bad
            with self.assertRaises(ValueError):
                freeze_candidate(rows, self.selection, self.options, controls_passed=True)

    def test_changed_plan_or_options_rejected_after_freeze(self):
        frozen = freeze_candidate(self.rows("calibration"), self.selection, self.options, controls_passed=True)
        options = copy.deepcopy(self.options); options["calibration_safety_multiplier"] = 1
        with self.assertRaises(ValueError):
            audit_candidate(self.rows("audit"), frozen, self.selection, options, controls_passed=True)

    def test_success_is_only_a_proposal_not_training_approval(self):
        frozen = freeze_candidate(self.rows("calibration"), self.selection, self.options, controls_passed=True)
        result = audit_candidate(self.rows("audit"), frozen, self.selection, self.options, controls_passed=True)
        self.assertTrue(result["audit_passed"])
        self.assertEqual(result["formal_optimizer_updates"], 0)
        self.assertFalse(result["production_protocol_activation"])

    def test_report_is_immutable_and_steps_cover_endpoints(self):
        self.assertEqual(time_indices(range(10, 41), 8)[::7], [10, 40])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"; write_once(path, {"status": "failed"})
            with self.assertRaises(FileExistsError):
                write_once(path, {"status": "passed"})
            self.assertEqual(json.loads(path.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
