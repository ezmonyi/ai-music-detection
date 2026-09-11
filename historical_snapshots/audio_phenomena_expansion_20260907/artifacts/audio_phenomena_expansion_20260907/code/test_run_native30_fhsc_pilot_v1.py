import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
import run_native30_fhsc_pilot_v1 as m


class PilotTests(unittest.TestCase):
    def test_missing_not_zero_and_infinite_is_error(self):
        self.assertEqual(m.clean({'x':np.float64(0),'y':float('nan')}),{'x':0.0,'y':None})
        with self.assertRaises(ValueError):m.clean({'x':float('inf')})

    def test_fixed_six_stereo_predictors(self):
        self.assertEqual(len(m.SC_KEYS),6)
        self.assertFalse(any('12000' in k or '20000' in k for k in m.SC_KEYS))

    def test_selected_six_not_gated_by_global_diagnostics(self):
        x={'features':{k:0.0 for k in m.SC_KEYS},'status':'partial','frame_count':1288,'bands':[]}
        result=m.selected_sc(x)
        self.assertEqual(result['selected_six_status'],'complete')
        self.assertEqual(result['selected_six_finite_count'],6)
        self.assertEqual(result['full_grid_diagnostic_status'],'partial')
        x['features'][m.SC_KEYS[0]]=float('nan')
        self.assertEqual(m.selected_sc(x)['selected_six_finite_count'],5)

    def test_real_synthetic_silence_30s(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'silence.wav';sf.write(p,np.zeros((1323000,2)),44100,subtype='FLOAT')
            r,sc=m.measure(p)
            self.assertEqual(r['F']['F_status'],'low_energy')
            self.assertEqual(r['H']['H_status'],'missing_low_energy')
            self.assertEqual(r['M_diagnostic_not_predictor']['M_status'],'missing_low_energy')
            self.assertTrue(all(x is None for x in r['SC']['features'].values()))
            self.assertEqual(r['analysis_view_audit']['analysis_frames'],480000)
            self.assertEqual(sc['frame_count'],1288)


if __name__=='__main__':unittest.main()
