"""Offline independent-auditor tests. No corpus reads or producer imports."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_bicoherence_nsynth_pilot_v1 as audit


class PrimitiveTests(unittest.TestCase):
    def test_known_direct_sums(self):
        spectra = np.zeros((3, 6), complex)
        spectra[:,1] = [1,2,3]
        spectra[:,2] = [2,3,4]
        spectra[:,3] = [1j,2j,3j]
        result = audit.independent_primitive(spectra, [1,2,3])
        direct = {'product_energy_sum':184., 'sum_frequency_energy_sum':14.,
                  'triple_sum_real':0., 'triple_sum_imag':-50.}
        for key,value in direct.items():
            encoded = result['raw_sums'][key]
            self.assertAlmostEqual(math.ldexp(encoded['mantissa'], encoded['exponent2']), value)
        self.assertAlmostEqual(result['squared_bicoherence'],2500/(184*14))
        self.assertAlmostEqual(result['biphase_radians'],-np.pi/2)

    def test_exact_cancellation_zero_score_missing_phase(self):
        spectra = np.ones((2,6), complex)
        spectra[:,3] = [1,-1]
        result = audit.independent_primitive(spectra,[1,2,3])
        self.assertEqual(result['squared_bicoherence'],0.)
        self.assertIsNone(result['biphase_radians'])
        self.assertEqual(result['biphase_status'],'undefined_zero_resultant')
        self.assertEqual(result['raw_sums']['triple_sum_real'], {'mantissa':0.,'exponent2':0})

    def test_silence_and_missing_column(self):
        x = np.zeros((2,6),complex)
        result = audit.independent_primitive(x,[1,2,3])
        self.assertEqual(result['status'],'zero_energy')
        self.assertIsNone(result['squared_bicoherence'])
        self.assertIsNone(result['normalized_sums'])
        x[:,4] = 1
        self.assertEqual(audit.independent_primitive(x,[1,2,3])['status'],'missing_triad_energy')

    def test_disjoint_parent_support_missing_product(self):
        x = np.zeros((2,6),complex)
        x[:,1], x[:,2], x[:,3] = [1,0], [0,1], [1,1]
        result = audit.independent_primitive(x,[1,2,3])
        self.assertEqual(result['status'],'missing_triad_product_energy')
        self.assertEqual(result['normalized_sums']['product_energy_sum'],0)
        self.assertIsNone(result['squared_bicoherence'])

    def test_gain_and_polarity(self):
        x = np.ones((3,6),complex)
        x[:,3] = np.exp(1j*np.array([.3,.5,.9]))
        a,b,c = [audit.independent_primitive(v,[1,2,3]) for v in (x,x*.1,-x)]
        self.assertAlmostEqual(a['squared_bicoherence'],b['squared_bicoherence'])
        self.assertAlmostEqual(a['squared_bicoherence'],c['squared_bicoherence'])
        self.assertAlmostEqual(abs(a['biphase_radians']-c['biphase_radians']),np.pi)


class ConstructionTests(unittest.TestCase):
    def test_seed_rms_and_all_derivatives(self):
        audio = np.full(64000,.125,dtype=np.float64)
        waves,arrays,meta = audit.reconstruct(audio,'offline-note')
        expected_seed = int.from_bytes(hashlib.sha256(b'BC-injection-20260907|offline-note').digest()[:8],'big')
        self.assertEqual(meta['seed'], expected_seed)
        rng = np.random.Generator(np.random.PCG64(expected_seed))
        np.testing.assert_array_equal(arrays['phase_knots_wrapped'],rng.uniform(-np.pi,np.pi,(3,33)))
        self.assertEqual(tuple(waves), audit.CONDITIONS)
        np.testing.assert_array_equal(waves['baseline'],audio*.25)
        np.testing.assert_array_equal(waves['polarity'],-waves['baseline'])
        self.assertEqual(meta['background_rms'],.03125)
        for level, ratio in [('minus6db',10**(-6/20)),('0db',1)]:
            for kind in ('closed','independent'):
                injection=arrays[kind+'_'+level+'_injection']
                self.assertAlmostEqual(float(np.sqrt(np.mean(injection**2))),.03125*ratio,places=15)
                np.testing.assert_array_equal(waves[kind+'_'+level],waves['baseline']+injection)
        self.assertFalse(meta['framewise_power_matched'])
        self.assertTrue(meta['injected_full_record_rms_matched'])

    def test_zero_rms_never_replaced(self):
        waves,arrays,meta = audit.reconstruct(np.zeros(64000), 'silent')
        self.assertEqual(meta['status'],'unsupported_zero_background_rms')
        self.assertFalse(meta['injected_full_record_rms_matched'])
        self.assertEqual(float(arrays['background_rms']),0)
        for values in waves.values():
            self.assertFalse(np.any(values))

    def test_full_silence_replay_has_complete_missing_cells(self):
        arrays,meta = audit.replay_features(np.zeros(64000),'unsupported_zero_background_rms')
        self.assertEqual(arrays['spectra'].shape,(1,247,513))
        self.assertEqual(arrays['frequency_bins'].shape,(228,3))
        self.assertFalse(np.any(arrays['eligible_mask']))
        self.assertTrue(np.isnan(arrays['squared_bicoherence']).all())
        self.assertTrue(np.isnan(arrays['primitive_biphase_radians']).all())
        self.assertTrue(np.isnan(arrays['bin_energy_fraction']).all())
        self.assertEqual(len(meta['pools'][0]['cells']),228)
        self.assertEqual(meta['pools'][0]['status'],'zero_amplitude')
        self.assertTrue(all(c['squared_bicoherence'] is None for c in meta['pools'][0]['cells']))

    def test_fixed_triad_offline_replay(self):
        t=np.arange(64000)/16000
        audio=.2*(np.cos(2*np.pi*500*t+.2)+np.cos(2*np.pi*750*t+.7)+np.cos(2*np.pi*1250*t+.9))
        arrays,meta=audit.replay_features(audio,'ok')
        target=audit.descriptor(meta)['target']
        self.assertTrue(target['eligible'])
        self.assertAlmostEqual(target['squared_bicoherence'],1,places=12)
        self.assertAlmostEqual(target['biphase_radians'],0,places=10)
        np.testing.assert_array_equal(arrays['frame_start_samples'][0],np.arange(247)*256)
        self.assertIsNone(meta['independent_realization_count'])
        self.assertFalse(meta['classifier_admitted'])


class SummaryTests(unittest.TestCase):
    def test_equal_instrument_weight_missing_denominators(self):
        rows=[{'instrument':k,'difference':v} for k,v in [('a',1.),('a',3.),('b',8.),('b',None),('c',None),('c',None)]]
        result=audit.balanced(rows)
        self.assertEqual(result['note_denominator'],6)
        self.assertEqual(result['covered_notes'],3)
        self.assertEqual(result['instrument_denominator'],3)
        self.assertEqual(result['covered_instruments'],2)
        self.assertEqual(result['equal_instrument_mean_difference'],5.)
        self.assertEqual(result['per_instrument'][2]['mean_difference'],None)

    def test_all_missing_and_signed_zero(self):
        self.assertIsNone(audit.balanced([{'instrument':'a','difference':None}])['equal_instrument_mean_difference'])
        r=audit.balanced([{'instrument':'a','difference':v} for v in [-1.,0.,-0.,2.,None]])
        self.assertEqual((r['positive'],r['zero'],r['negative']),(1,2,1))

    def test_nuisance_keeps_missing_transitions(self):
        def desc(values):
            return {'grid_eligibility':[v is not None for v in values], 'grid_squared_bicoherence':values,
                    'target':{'eligible':values[0] is not None,'squared_bicoherence':values[0]}}
        r=audit.nuisance(desc([None,.2,.3,None]),desc([.1,None,.4,None]))
        self.assertEqual(r['grid_denominator'],4)
        self.assertEqual(r['grid_eligibility_agreement_count'],2)
        self.assertEqual(r['missing_to_finite_count'],1)
        self.assertEqual(r['finite_to_missing_count'],1)
        self.assertEqual(r['common_finite_cell_count'],1)
        self.assertAlmostEqual(r['maximum_finite_b2_difference'],.1)
        self.assertIsNone(r['target_difference'])
        self.assertFalse(r['target_eligibility_agreement'])


class FailClosedTests(unittest.TestCase):
    def test_mutations_are_rejected(self):
        expected={'mask':np.array([[True,False]]), 'b2':np.array([[.5,np.nan]]),
                  'cell':{'score':None,'eligible':False}, 'sum':{'mantissa':.625,'exponent2':4}}
        mutants=[]
        for key in ('mask','b2','cell','sum'):
            value=copy.deepcopy(expected)
            if key=='mask': value[key][0,1]=True
            if key=='b2': value[key][0,1]=0
            if key=='cell': value[key]['score']=0
            if key=='sum': value[key]['exponent2']=5
            mutants.append(value)
        extra=copy.deepcopy(expected); extra['extra']=1; mutants.append(extra)
        for value in mutants:
            with self.assertRaises(ValueError): audit.compare(value,expected)
        with self.assertRaises(ValueError): audit.compare(True,1)
        with self.assertRaises(ValueError): audit.compare(np.zeros((2,)),np.zeros((1,2)))

    def test_inventory_bytes_extra_missing_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'a').write_bytes(b'abc')
            marker={'status':'committed','products':{'a':audit.product(root/'a')}}
            (root/'COMMIT.json').write_text(json.dumps(marker))
            digest=audit.sha(root/'COMMIT.json')
            audit.inventory(root,digest)
            (root/'a').write_bytes(b'xyz')
            with self.assertRaises(ValueError): audit.inventory(root,digest)
            (root/'a').write_bytes(b'abc')
            (root/'extra').write_text('x')
            with self.assertRaises(ValueError): audit.inventory(root,digest)
            (root/'extra').unlink()
            with self.assertRaises(ValueError): audit.child(root,'../escape')
            (root/'link').symlink_to(root/'a')
            with self.assertRaises(ValueError): audit.inventory(root,digest)
            (root/'link').unlink()
            (root/'a').unlink()
            with self.assertRaises(ValueError): audit.inventory(root,digest)

    def test_json_duplicate_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.json'
            for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
                path.write_text(text)
                with self.assertRaises(ValueError): audit.read_json(path)

    def test_npz_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'arrays.npz'
            np.savez(path,value=np.array([1.,2.]))
            audit.npz_check(path,{'value':np.array([1.,2.])},exact=True)
            with self.assertRaises(ValueError): audit.npz_check(path,{'value':np.array([1.,3.])})
            with self.assertRaises(ValueError): audit.npz_check(path,{'other':np.array([1.,2.])})

    def test_array_json_disagreement_fails(self):
        arrays,metadata=audit.replay_features(np.zeros(64000),'unsupported_zero_background_rms')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'arrays.npz'
            np.savez(path,**arrays)
            audit.array_json_agreement(path,metadata)
            metadata['pools'][0]['cells'][0]['squared_bicoherence']=0.
            with self.assertRaises(ValueError): audit.array_json_agreement(path,metadata)

    def test_no_producer_imports_or_dynamic_exec(self):
        tree=ast.parse(Path(audit.__file__).read_text())
        forbidden={'bicoherence_nsynth_pilot_v1','bicoherence_audio_v1','bicoherence_primitive_v2',
                   'audit_bicoherence_audio_development_v1'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                self.assertFalse({n.name for n in node.names}&forbidden)
            if isinstance(node,ast.ImportFrom): self.assertNotIn(node.module,forbidden)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id,{'exec','eval','__import__'})


if __name__=='__main__':
    unittest.main()
