import unittest
import numpy as np
import bicoherence_math_probe_v1 as b


class TestBicoherenceMath(unittest.TestCase):
    def test_closed_random_and_deterministic_confound(self):
        f=b.fixtures()
        self.assertAlmostEqual(b.estimate(f['phase_closed_ensemble'])['squared_bicoherence'],1.)
        self.assertLess(b.estimate(f['independent_phase_ensemble'])['squared_bicoherence'],.1)
        self.assertAlmostEqual(b.estimate(f['fixed_three_oscillators_no_nonlinear_operation'])['squared_bicoherence'],1.)

    def test_polarity_changes_phase_not_magnitude(self):
        f=b.fixtures(); a=b.estimate(f['phase_closed_ensemble']); n=b.estimate(-f['phase_closed_ensemble'])
        self.assertAlmostEqual(a['squared_bicoherence'],n['squared_bicoherence'])
        self.assertAlmostEqual(abs(np.angle(np.exp(1j*(a['biphase_radians']-n['biphase_radians'])))),np.pi)

    def test_gain_and_delay_invariance(self):
        f=b.fixtures(); expected=b.estimate(f['phase_closed_ensemble'])['squared_bicoherence']
        for name in ['phase_closed_gain_0p1','phase_closed_circular_delay_13_samples']:
            self.assertAlmostEqual(b.estimate(f[name])['squared_bicoherence'],expected)

    def test_single_frame_invalid_and_missing_not_zero(self):
        with self.assertRaises(ValueError): b.estimate(np.ones((1,129)))
        self.assertIsNone(b.estimate(np.zeros((128,129)))['squared_bicoherence'])
        f=b.fixtures()['phase_closed_ensemble'].copy(); f[:,11]=0
        self.assertEqual(b.estimate(f)['status'],'missing_triad_energy')

    def test_nonfinite_and_invalid_pair(self):
        f=b.fixtures()['phase_closed_ensemble'].copy(); f[0,4]=np.nan
        with self.assertRaises(ValueError): b.estimate(f)
        for pair in [(0,1),(7,4),(64,65)]:
            with self.assertRaises(ValueError): b.estimate(np.ones((128,129)),pair)

    def test_saved_time_frames_preserve_coefficients(self):
        for f in b.fixtures().values():
            np.testing.assert_allclose(np.fft.rfft(np.fft.irfft(f,n=256,axis=1),axis=1),f,atol=1e-14)


if __name__=='__main__': unittest.main()
