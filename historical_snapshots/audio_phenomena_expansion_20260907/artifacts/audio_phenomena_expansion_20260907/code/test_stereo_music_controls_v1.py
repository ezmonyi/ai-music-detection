import unittest
import numpy as np
from stereo_music_controls_v1 import align, extract, frame_error, rms_match, scalar_pair


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.y = np.random.default_rng(20260907).normal(0, .05, (44100, 2))

    def test_alignment_both_directions(self):
        for shift in [0, 17, -31]:
            if shift >= 0:
                b = np.concatenate([np.zeros((shift, 2)), self.y, np.zeros((43, 2))])
            else:
                b = self.y[-shift:]
            a, b, receipt = align(self.y, b)
            self.assertEqual(receipt['lag_samples'], shift)
            np.testing.assert_array_equal(a, b)

    def test_iid_and_width_targets(self):
        base = extract(self.y)
        y = self.y.copy()
        y[:, 0] *= 10**(.3)
        modified = extract(rms_match(y, self.y))
        for i in range(5):
            check = frame_error(modified['per_frame']['iid_db'][i],
                                base['per_frame']['iid_db'][i]+6)
            self.assertTrue(check['passed'])
        mid, side = self.y.mean(axis=1), .5*(self.y[:, 0]-self.y[:, 1])
        for width in [.5, 2]:
            y = rms_match(np.column_stack([mid+width*side, mid-width*side]), self.y)
            modified = extract(y)
            q = base['per_frame']['side_energy_fraction']
            target = width**2*q/(1-q+width**2*q)
            for i in range(5):
                self.assertTrue(frame_error(modified['per_frame']['side_energy_fraction'][i],
                                            target[i])['passed'])

    def test_antiphase_alignment_and_silence_rejection(self):
        y = np.column_stack([self.y[:, 0], -self.y[:, 0]])
        a, b, receipt = align(y, y.copy())
        self.assertEqual(receipt['lag_samples'], 0)
        np.testing.assert_array_equal(a, b)
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            align(np.zeros_like(y), np.zeros_like(y))

    def test_unavailable_and_failed_coverage_are_not_pass(self):
        check = frame_error(np.full(100, np.nan), np.ones(100))
        self.assertFalse(check['passed'])
        check = frame_error(np.r_[np.ones(30), np.full(70, np.nan)], np.ones(100))
        self.assertTrue(check['arithmetic_pass'])
        self.assertFalse(check['coverage_pass'])
        self.assertFalse(check['passed'])

    def test_swap_and_gain_scalar_invariance(self):
        base = extract(self.y)
        for y in [self.y[:, ::-1], self.y*.5]:
            rows = scalar_pair(base, extract(y))
            self.assertTrue(all(not r['missingness_changed'] for r in rows.values()))
            finite = [r['absolute_delta'] for r in rows.values() if r['absolute_delta'] is not None]
            self.assertTrue(finite)
            self.assertTrue(all(d <= 1e-8 for d in finite))


if __name__ == '__main__':
    unittest.main()
