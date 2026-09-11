"""Synthetic-only tests for the independent GuitarSet auditor. No corpus access."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

import audit_bicoherence_guitarset_pilot_v1 as audit


def source_fixture():
    rows=[]
    for player in range(6):
        for score in range(30):
            for performance in ('comp','solo'):
                p,s=f'{player:02d}',f'Style{score}-100-C'
                item=p+'_'+s+'_'+performance
                rows.append({'item_id':item,'player_id':p,'score_id':s,'performance':performance,
                    'audio':{'decoded':{'decoded_pcm_sha256':hashlib.sha256(item.encode()).hexdigest()}}})
    return rows


def descriptor_fixture(masked,raw=None):
    pools=[]
    for index,values in enumerate(masked):
        raw_values=values if raw is None else raw[index]
        pools.append({'pool_index':index,'pool_status':'ok','grid_cell_count':len(values),
            'eligible_cell_count':sum(v is not None for v in values),
            'grid_eligibility':[v is not None for v in values],
            'grid_status':['ok' if v is not None else 'below_energy_fraction_floor' for v in values],
            'grid_squared_bicoherence':values,'grid_raw_squared_bicoherence':raw_values,
            'target':{'eligible':values[0] is not None,'squared_bicoherence':values[0],
                'status':'ok' if values[0] is not None else 'below_energy_fraction_floor'}})
    return {'pools':pools}


def record_fixture(item,player,score,performance,closed,independent):
    base=descriptor_fixture([[.2,None],[None,.3]])
    conditions={name:copy.deepcopy(base) for name in audit.CONDITIONS}
    for level in ('minus6db','0db'):
        conditions['closed_'+level]=descriptor_fixture([[v] for v in closed])
        conditions['independent_'+level]=descriptor_fixture([[v] for v in independent])
    return {'item_id':item,'player_id':player,'score_id':score,'performance':performance,
            'style_from_score_prefix':'Style','conditions':conditions,
            'nuisance':{name:audit.nuisance(base,base) for name in ('common_gain','polarity')}}


class RuntimeTests(unittest.TestCase):
    def test_pinned_versions_and_executed_library_bindings(self):
        result=audit.runtime_snapshot(True)
        self.assertEqual((result['numpy'],result['scipy'],result['soundfile']),('1.26.4','1.17.1','0.14.0'))
        self.assertIn('numpy.fft._pocketfft_internal',result['producer_modules'])
        audit.check_bound_tree(result)

    def test_no_producer_extractor_or_draft_import(self):
        tree=ast.parse(Path(audit.__file__).read_text())
        forbidden={'bicoherence_guitarset_pilot_v1','bicoherence_nsynth_pilot_v1','bicoherence_audio_v1',
            'bicoherence_primitive_v2','draft_bicoherence_guitarset_pilot_v2','audit_bicoherence_nsynth_pilot_v1'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertFalse({alias.name for alias in node.names}&forbidden)
            if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,forbidden)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):self.assertNotIn(node.func.id,{'exec','eval','__import__'})


class SplitTests(unittest.TestCase):
    def test_complete_crossed_selection_and_performance_pairs(self):
        rows=source_fixture()
        split=audit.crossed_split(rows)
        self.assertEqual(split['role_counts'],{'development':90,'reserved':90,'unused':180})
        order=sorted({r['player_id'] for r in rows},key=lambda p:(hashlib.sha256(('BC-GuitarSet-player-20260907|'+p).encode()).hexdigest(),p))
        self.assertEqual(split['development_player_ids'],order[:3])
        for row in split['rows']:
            if row['split_role']=='development':
                self.assertIn(row['player_id'],split['development_player_ids'])
                self.assertIn(row['score_id'],split['development_score_ids'])
        for player in split['development_player_ids']:
            for score in split['development_score_ids']:
                self.assertEqual({r['performance'] for r in split['rows'] if r['player_id']==player and r['score_id']==score},{'comp','solo'})
        self.assertEqual(split['style_recording_counts_by_role']['development'],{'Style':90})
        self.assertFalse(split['independent_performers_claimed'])

    def test_duplicate_missing_component_and_pcm_collision_fail(self):
        rows=source_fixture()
        with self.assertRaises(ValueError):audit.crossed_split(rows[:-1])
        broken=copy.deepcopy(rows);broken[-1]=broken[0]
        with self.assertRaises(ValueError):audit.crossed_split(broken)
        broken=copy.deepcopy(rows);broken[0]['item_id']='wrong'
        with self.assertRaises(ValueError):audit.crossed_split(broken)
        split=audit.crossed_split(rows)
        d=next(r for r in split['rows'] if r['split_role']=='development')['item_id']
        r=next(r for r in split['rows'] if r['split_role']=='reserved')['item_id']
        selected={x['item_id']:x for x in rows}
        selected[r]['audio']['decoded']['decoded_pcm_sha256']=selected[d]['audio']['decoded']['decoded_pcm_sha256']
        with self.assertRaises(ValueError):audit.crossed_split(rows)


class AudioTests(unittest.TestCase):
    def test_native_16k_identity_center_crop_and_short_failure(self):
        x=np.arange(128003,dtype=np.float64)/128003
        crop,p=audit.standardize(x,16000)
        np.testing.assert_array_equal(crop,x[1:128001])
        self.assertEqual((p['crop_start'],p['crop_stop_exclusive']),(1,128001))
        self.assertFalse(p['resampling_applied'])
        self.assertEqual(p['crop_pcm_sha256'],hashlib.sha256(crop.astype('<f8').tobytes()).hexdigest())
        with self.assertRaises(ValueError):audit.standardize(x[:127999],16000)
        with self.assertRaises(ValueError):audit.standardize(x.astype(np.float32),16000)

    def test_entire_44100_resample_then_crop_length_and_zero(self):
        x=np.zeros(441001,dtype=np.float64)
        crop,p=audit.standardize(x,44100)
        self.assertEqual((p['up'],p['down']),(160,441))
        self.assertEqual(p['resampled_frames'],160001)
        self.assertEqual((p['crop_start'],p['crop_stop_exclusive']),(16000,144000))
        self.assertEqual(p['window'],['kaiser',5.])
        self.assertTrue(p['crop_zero_amplitude'])
        self.assertFalse(np.any(crop))

    def test_full_native_decode_and_pcm_hash_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            samples=np.sin(np.arange(160010,dtype=np.float64)/53)*.1
            sf.write(root/'audio.wav',samples,16000,subtype='PCM_16')
            (root/'annotation.jams').write_text('{}')
            native,rate=sf.read(root/'audio.wav',dtype='float64')
            decoded={'channels':1,'decode_block_frames':65536,'decoded_frames':len(native),
                'decoded_pcm_canonical_encoding':audit.ENCODING,'decoded_pcm_sha256':audit.canonical_pcm(native),
                'duration_seconds':len(native)/rate,'empty_eof_observed':True,'float64_samples_checked_finite':len(native),
                'format':'WAV','header_frames':len(native),'materialized_file':audit.product(root/'audio.wav'),
                'nonfinite_samples':0,'read_calls_including_empty_eof':4,'sample_rate_hz':rate,'subtype':'PCM_16'}
            row={'item_id':'offline','split_role':'development','source_record':{
                'audio':{'materialized_path':'audio.wav','decoded':decoded},'annotation':{'materialized_path':'annotation.jams'}}}
            crop,provenance=audit.standardize(native,rate)
            prepared={'item_id':'offline','original_audio':audit.full_binding(root/'audio.wav'),
                'original_annotation':audit.full_binding(root/'annotation.jams'),'source_pcm_sha256':decoded['decoded_pcm_sha256'],
                'preprocessing':provenance}
            np.testing.assert_array_equal(audit.decode_development(root,row,prepared),crop)
            mutated=copy.deepcopy(prepared);mutated['preprocessing']['crop_start']+=1
            with self.assertRaises(ValueError):audit.decode_development(root,row,mutated)
            decoded['decoded_pcm_sha256']='0'*64
            with self.assertRaises(ValueError):audit.decode_development(root,row,prepared)
            row['split_role']='reserved'
            with self.assertRaisesRegex(ValueError,'forbidden'):audit.decode_development(root,row,prepared)

    def test_deterministic_65_knots_all_seven_rms_and_zero(self):
        x=np.full(128000,.125)
        waves,arrays,meta=audit.reconstruct(x,'offline')
        seed=int.from_bytes(hashlib.sha256(b'BC-GuitarSet-injection-20260907|offline').digest()[:8],'big')
        knots=np.random.Generator(np.random.PCG64(seed)).uniform(-np.pi,np.pi,(3,65))
        self.assertEqual(meta['seed'],seed)
        np.testing.assert_array_equal(arrays['phase_knots_wrapped'],knots)
        self.assertEqual(arrays['phase_knot_times_seconds'][-1],8.)
        self.assertEqual(tuple(waves),audit.CONDITIONS)
        self.assertEqual(meta['background_rms'],.03125)
        for level,ratio in [('minus6db',10**(-6/20)),('0db',1.)]:
            for kind in ('closed','independent'):
                injection=arrays[kind+'_'+level+'_injection']
                self.assertAlmostEqual(float(np.sqrt(np.mean(injection**2))),.03125*ratio,places=15)
                np.testing.assert_array_equal(waves[kind+'_'+level],x*.25+injection)
        zero,_,meta=audit.reconstruct(np.zeros(128000),'zero')
        self.assertEqual(meta['status'],'unsupported_zero_background_rms')
        self.assertTrue(all(not np.any(v) for v in zero.values()))


class PrimitiveTests(unittest.TestCase):
    def test_direct_sums_and_phase(self):
        x=np.zeros((3,6),complex);x[:,1]=[1,2,3];x[:,2]=[2,3,4];x[:,3]=[1j,2j,3j]
        result=audit.independent_primitive(x,[1,2,3])
        for key,value in {'product_energy_sum':184.,'sum_frequency_energy_sum':14.,'triple_sum_real':0.,'triple_sum_imag':-50.}.items():
            r=result['raw_sums'][key]
            self.assertAlmostEqual(math.ldexp(r['mantissa'],r['exponent2']),value)
        self.assertAlmostEqual(result['squared_bicoherence'],2500/(184*14))
        self.assertAlmostEqual(result['biphase_radians'],-np.pi/2)

    def test_cancellation_zero_energy_and_disjoint_product_missing(self):
        x=np.ones((2,6),complex);x[:,3]=[1,-1]
        r=audit.independent_primitive(x,[1,2,3])
        self.assertEqual(r['squared_bicoherence'],0.)
        self.assertIsNone(r['biphase_radians'])
        self.assertEqual(r['biphase_status'],'undefined_zero_resultant')
        x[:]=0
        self.assertEqual(audit.independent_primitive(x,[1,2,3])['status'],'zero_energy')
        x[:,1],x[:,2],x[:,3]=[1,0],[0,1],[1,1]
        r=audit.independent_primitive(x,[1,2,3])
        self.assertEqual(r['status'],'missing_triad_product_energy')
        self.assertIsNone(r['squared_bicoherence'])

    def test_two_pools_analytic_target_and_boundary(self):
        t=np.arange(64000)/16000
        tone=.2*(np.cos(2*np.pi*500*t+.2)+np.cos(2*np.pi*750*t+.7)+np.cos(2*np.pi*1250*t+.9))
        arrays,meta=audit.replay_features(np.concatenate([tone,np.zeros(64000)]),'ok')
        self.assertEqual(arrays['spectra'].shape,(2,247,513))
        self.assertEqual(arrays['frequency_bins'].shape,(228,3))
        np.testing.assert_array_equal(arrays['frame_start_samples'][1],64000+np.arange(247)*256)
        targets=[p['target'] for p in audit.descriptor(meta)['pools']]
        self.assertTrue(targets[0]['eligible'])
        self.assertAlmostEqual(targets[0]['squared_bicoherence'],1.,places=12)
        self.assertAlmostEqual(targets[0]['biphase_radians'],0.,places=10)
        self.assertIsNone(targets[1]['squared_bicoherence'])
        self.assertFalse(np.any(arrays['spectra'][1]))
        self.assertTrue(np.isnan(arrays['squared_bicoherence'][1]).all())
        self.assertTrue(np.isnan(arrays['bin_energy_fraction'][1]).all())
        self.assertEqual(meta['pools'][1]['status'],'zero_amplitude')
        self.assertIsNone(meta['independent_realization_count'])


class SummaryTests(unittest.TestCase):
    def test_paired_pool_record_then_equal_score_and_secondary_player(self):
        records=[record_fixture('a','p1','s1','comp',[.9,.7],[.1,None]),
                 record_fixture('b','p2','s1','solo',[.5,.5],[.3,.3]),
                 record_fixture('c','p1','s2','comp',[.2,None],[.4,None]),
                 record_fixture('d','p2','s3','solo',[None,None],[None,None])]
        result=audit.summarize(records)
        level=result['levels']['0db']
        self.assertEqual((level['recording_denominator'],level['covered_recordings']),(4,3))
        self.assertEqual((level['pool_denominator'],level['paired_covered_pools']),(8,4))
        self.assertEqual(level['equal_score']['group_denominator'],3)
        self.assertEqual(level['equal_score']['covered_groups'],2)
        self.assertAlmostEqual(level['equal_score']['equal_group_mean_difference'],.15)
        self.assertAlmostEqual(level['equal_player_secondary']['equal_group_mean_difference'],.25)
        self.assertEqual((level['positive_recordings'],level['negative_recordings'],level['zero_recordings']),(2,1,0))
        self.assertEqual(level['performance_strata']['solo']['recording_denominator'],2)
        self.assertEqual(level['performance_strata']['solo']['covered_recordings'],1)
        self.assertIsNone(level['per_recording'][0]['paired_pools'][1]['difference'])
        self.assertIsNone(level['equal_score']['rows'][2]['mean_difference'])
        self.assertEqual(result['baseline']['target_pool_denominator'],8)

    def test_raw_and_masked_nuisance_missing_transitions(self):
        a=descriptor_fixture([[None,.2,.3,None],[None,None,None,None]],[[.1,.2,.3,None],[None]*4])
        b=descriptor_fixture([[.1,None,.4,None],[None,None,None,None]],[[.1,.3,None,.4],[None]*4])
        r=audit.nuisance(a,b)[0]
        self.assertEqual((r['grid_denominator'],r['grid_eligibility_agreement_count']),(4,2))
        self.assertEqual((r['missing_to_finite_count'],r['finite_to_missing_count']),(1,1))
        self.assertEqual((r['raw_missing_to_finite_count'],r['raw_finite_to_missing_count']),(1,1))
        self.assertEqual(r['raw_common_finite_cell_count'],2)
        self.assertAlmostEqual(r['maximum_finite_b2_difference'],.1)
        self.assertAlmostEqual(r['raw_maximum_finite_b2_difference'],.1)
        self.assertTrue(r['target_missing_to_finite'])
        self.assertIsNone(r['target_difference'])

    def test_all_missing_kept_in_group_denominators(self):
        rows=[{'score_id':'s','player_id':'p','difference':None,'pool_denominator':2,'covered_pools':0}]
        r=audit.paired(rows)
        self.assertEqual(r['recording_denominator'],1)
        self.assertEqual(r['pool_denominator'],2)
        self.assertEqual(r['equal_score']['group_denominator'],1)
        self.assertEqual(r['equal_score']['covered_groups'],0)
        self.assertIsNone(r['equal_score']['equal_group_mean_difference'])


class FailClosedTests(unittest.TestCase):
    def test_signed_zero_is_not_bit_exact(self):
        with self.assertRaises(ValueError):audit.compare(np.array([-0.]),np.array([0.]),rtol=0,atol=0)
        audit.compare(np.array([-0.]),np.array([0.]))

    def test_json_duplicate_nan_and_numeric_schema_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'x.json'
            for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
                p.write_text(text)
                with self.assertRaises(ValueError):audit.read_json(p)
        for a,e in [(True,1),(0.,None),({'x':1,'extra':0},{'x':1}),(np.zeros(2),np.zeros((1,2))),
                    ({'mantissa':.5,'exponent2':2},{'mantissa':.5,'exponent2':3})]:
            with self.assertRaises(ValueError):audit.compare(a,e)

    def test_exact_inventory_extra_missing_mutation_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'a').write_bytes(b'abc')
            (root/'COMMIT.json').write_text(json.dumps({'status':'committed','products':{'a':audit.product(root/'a')}}))
            digest=audit.sha(root/'COMMIT.json')
            audit.inventory(root,digest)
            (root/'a').write_bytes(b'xyz')
            with self.assertRaises(ValueError):audit.inventory(root,digest)
            (root/'a').write_bytes(b'abc');(root/'extra').write_text('x')
            with self.assertRaises(ValueError):audit.inventory(root,digest)
            (root/'extra').unlink();(root/'link').symlink_to(root/'a')
            with self.assertRaises(ValueError):audit.inventory(root,digest)
            with self.assertRaises(ValueError):audit.child(root,'../escape')
            (root/'link').unlink();(root/'a').unlink()
            with self.assertRaises(ValueError):audit.inventory(root,digest)

    def test_array_json_and_npz_mutations_fail(self):
        arrays,metadata=audit.replay_features(np.zeros(128000),'unsupported_zero_background_rms')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'x.npz';np.savez(path,**arrays)
            audit.npz_check(path,arrays,exact=True)
            audit.array_json_agreement(path,metadata)
            broken=copy.deepcopy(metadata);broken['pools'][1]['cells'][1]['squared_bicoherence']=0.
            with self.assertRaises(ValueError):audit.array_json_agreement(path,broken)
            mutated={**arrays,'spectra':arrays['spectra'].copy()};mutated['spectra'][1,246,512]=1.
            with self.assertRaises(ValueError):audit.npz_check(path,mutated)

    def test_exact_product_count_and_unsafe_item(self):
        rows=[r for r in audit.crossed_split(source_fixture())['rows'] if r['split_role']=='development']
        self.assertEqual(len(audit.expected_product_names(rows)),2165)
        rows[0]['item_id']='../bad'
        with self.assertRaises(ValueError):audit.expected_product_names(rows)


if __name__=='__main__':
    unittest.main()
