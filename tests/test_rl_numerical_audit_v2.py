import copy
import json
from pathlib import Path
import unittest

from music_detector.rl.numerical_audit_v2 import evaluate_audit, validate_plan


class V2NumericalAuditTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parents[1] / 'reports/rl/numerical_calibration_20261031'
        self.plan = json.loads((root / 'PROTOCOL_V2_PROPOSAL.json').read_text())
        self.previous = json.loads((root / 'PRE_REGISTERED_SELECTION.json').read_text())

    def rows(self):
        return [{'key': g['key'], 'behavior': g['behavior'], 'split': 'audit', 'sample': s, 'step': t,
                 'recorded_logprob': .25, 'replayed_logprob': .2501, 'repeated_logprob': .2501}
                for g in self.plan['fresh_audit_groups'] for s in range(4) for t in range(10, 41)]

    def test_reviewed_plan_is_independent(self):
        validate_plan(self.plan, self.previous)
        for field, value in (('candidate_absolute_logprob_budget', .001), ('production_protocol_activation', True)):
            bad = copy.deepcopy(self.plan); bad[field] = value
            with self.assertRaises(ValueError):validate_plan(bad, self.previous)

    def test_previous_group_cannot_be_reused(self):
        bad = copy.deepcopy(self.plan)
        bad['fresh_audit_groups'][0] = {**self.previous['groups'][0], 'split': 'audit'}
        with self.assertRaises(ValueError):validate_plan(bad, self.previous)

    def test_failure_cannot_refit_or_activate_production(self):
        rows = self.rows(); rows[-1]['replayed_logprob'] = rows[-1]['repeated_logprob'] = .251
        value = evaluate_audit(rows, self.plan, controls_passed=True)
        self.assertFalse(value['audit_passed']); self.assertEqual(value['receiver_logprob_tolerance'], .0003)
        self.assertFalse(value['production_protocol_activation']); self.assertEqual(value['formal_optimizer_updates'], 0)

    def test_every_stochastic_step_and_repeat_is_required(self):
        rows = self.rows()
        with self.assertRaises(ValueError):evaluate_audit(rows[:-1], self.plan, controls_passed=True)
        rows[0]['repeated_logprob'] = None
        self.assertFalse(evaluate_audit(rows, self.plan, controls_passed=True)['audit_passed'])

    def test_pass_does_not_perform_training(self):
        value = evaluate_audit(self.rows(), self.plan, controls_passed=True)
        self.assertTrue(value['audit_passed']); self.assertFalse(value['production_protocol_activation'])


if __name__ == '__main__':
    unittest.main()
