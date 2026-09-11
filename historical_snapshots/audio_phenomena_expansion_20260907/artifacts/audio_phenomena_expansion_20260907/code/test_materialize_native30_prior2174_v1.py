"""Synthetic tests; no real source audio, feature extraction or corpus runs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import materialize_native30_prior2174_v1 as m


def fixture():
    rows=[];screens=[];components=[]
    for index,(source,label) in enumerate([('Suno','1'),('Mureka_v9','1'),('human_saraga_hindustani_v1','0')]):
        ident=f'item{index}';special=source in m.SPECIAL
        row={'id':ident,'group_id':'group'+ident,'component_id':'component'+ident,'label':label,
            'source_group':source,'role':'development','input_occurrences':['additional60'],
            'selected_metadata_input':'additional60','origin_set':'prior60',
            'source_origin':{'path':'/mnt/native/'+ident+'.wav','sha256':'a'*64,'channels':2,'sample_rate_hz':44100,
                'scope':'accepted_native60_interval_from_full_original' if source=='Mureka_v9' else
                'accepted_native60_interval_from_archive_member_original' if source=='human_saraga_hindustani_v1' else 'full_original_file'},
            'native_region':{'start_frame':7,'frames':60*44100,'source_sha256':'a'*64,
                'evidence_strength':m.HASH_VERIFIED if special else m.BOUNDS_ONLY,
                'float64_sha256':'b'*64 if special else None,'decoder_limit':'retained explicit limitation'},
            'planned_native30':{'coordinate_space':'native_source_frames','derived_from_native_region':True,
                'frames':30*44100,'must_reproduce_native_region_before_subcrop':special,
                'policy':'center_exact30_within_approved_native60_region','start_frame':7+15*44100},
            'prior_standardized60_evidence_not_origin':{'path':'/mnt/standardized/'+ident+'.wav','sha256':'c'*64}}
        if special:row['native_region']['seek_equals_sequential_float64']=True
        rows.append(row)
        screens.append({k:row[k] for k in ('id','group_id','component_id','label','source_group','role','input_occurrences','selected_metadata_input')} |
            {'duration_exposure_candidate':True,'exclusion_reasons':[],'recorded_native_channels':2,'recorded_native_channel_status':'stereo'})
        components.append({'component_id':row['component_id'],'members':[ident],'candidate_screen_members':[ident],
                           'protected_relationships':[]})
    plan={'schema_version':'native30-origin-plan-v2','status':'draft_not_admitted_not_frozen_for_execution',
        'counts':{'audio_files_opened':0,'classifier_fits':0,'planned_rows':3,'prior60_rows':3,'new_native30_stereo_rows':0},
        'rows':rows,'prior60_source_counts':{'Suno':1,'Mureka_v9':1,'human_saraga_hindustani_v1':1}}
    review={'status':'metadata_consistency_accepted_not_execution_frozen_or_audio_admitted',
            'plan_sha256':m.PLAN_SHA,'physical_source_audio_opened_by_review':0,'classifier_fits':0}
    screen={'rows':screens,'components':components,'classifier_fits':0,'feature_extraction_authorized':False}
    expected={'plan':3,'prior':3,'new':0,'human':1,'ai':2,'ordinary':1,'mandatory_hash':2,'sources':plan['prior60_source_counts']}
    return plan,review,screen,expected


def audit_fixture(row,configuration):
    n,r=row['native_evidence'],row['native_region'];rate=n['sample_rate_hz'];start=r['start_frame'];frames=r['frames']
    crop=start+15*rate;special=r['float64_sha256'] is not None;observed=r['float64_sha256'] or 'd'*64
    p={'decoded_through_frame_exclusive':start+frames,'prefix_frames_before_region':start,
       'retained_native_region_frames':frames,'observed_region_float64_sha256':observed,
       'observation':'bounded_sequential_prefix_through_region_end','empty_eof_probe_performed':False,
       'full_stream_completeness_established':False}
    return {'status':'verified_bounded_native60_to30_DSP_only','configuration':configuration,
        'region_kind':'frozen_native60_interval','coordinates':{'region_start_frame':start,'region_frames':frames,
            'region_end_frame_exclusive':start+frames,'crop_start_frame':crop,'crop_frames':30*rate,
            'crop_end_frame_exclusive':crop+30*rate,'crop_offset_within_native_region':15*rate},
        'source_path':row['execution_native_path'],'source_sha256_before':n['sha256'],'source_sha256_after':n['sha256'],
        'native_rate_hz':rate,'native_channels':2,'source_byte_hash_scope':'whole_file_bytes_not_full_waveform_decode',
        'full_stream_completeness_established':False,'independent_decoder_agreement_established':False,
        'evidence_strength':r['evidence_strength'],'historical_expected_region_float64_sha256':r['float64_sha256'],
        'historical_region_waveform_hash_available':special,'historical_region_hash_match':True if special else None,
        'observed_current_region_float64_sha256':observed,'sequential_passes':[copy.deepcopy(p),copy.deepcopy(p)],
        'two_sequential_observations_equal':True,'decoded_through_frame_exclusive':start+frames,
        'output_frames':1323000,'output_rate_hz':44100,'output_channels':2,'sample_count':2646000,
        'native_crop_float64_sha256':'e'*64,'output_waveform_float32_sha256':'f'*64,
        'classifier_admitted':False,'cohort_admitted':False,'feature_extraction_authorized':False}


class MetadataTests(unittest.TestCase):
    def test_exact_rows_native_original_and_historical_null(self):
        plan,review,screen,expected=fixture()
        with mock.patch.object(Path,'open',side_effect=AssertionError('metadata reconcile must not read audio')):
            rows=m._reconcile(plan,review,screen,expected)
        self.assertEqual(len(rows),3)
        self.assertIsNone(rows[0]['native_region']['float64_sha256'])
        self.assertEqual(rows[0]['native_evidence'],rows[0]['source_origin'])
        self.assertEqual(rows[1]['execution_native_path'],'/mnt/native/item1.wav')
        self.assertNotEqual(rows[1]['execution_native_path'],rows[1]['prior_standardized60_evidence_not_origin']['path'])

    def test_no_special_hash_downgrade_or_ordinary_invention(self):
        for index,value in [(0,'b'*64),(1,None),(2,None)]:
            plan,review,screen,expected=fixture();plan['rows'][index]['native_region']['float64_sha256']=value
            with self.assertRaises(ValueError):m._reconcile(plan,review,screen,expected)
        plan,review,screen,expected=fixture();plan['rows'][1]['native_region']['evidence_strength']=m.BOUNDS_ONLY
        with self.assertRaises(ValueError):m._reconcile(plan,review,screen,expected)

    def test_fullscreen_protected_identity_and_coordinate_mutations(self):
        mutations=[lambda p,s:s['components'][0]['protected_relationships'].append('reserved'),
            lambda p,s:s['rows'][0].update(role='reserved'),lambda p,s:s['rows'][0].update(group_id='wrong'),
            lambda p,s:p['rows'][0]['native_region'].update(start_frame=True),
            lambda p,s:p['rows'][0]['planned_native30'].update(start_frame=0),
            lambda p,s:p['rows'][0]['source_origin'].update(path='/Users/local/audio.wav'),
            lambda p,s:p['rows'][0]['source_origin'].update(path=p['rows'][0]['prior_standardized60_evidence_not_origin']['path'])]
        for mutate in mutations:
            p,r,s,e=fixture();mutate(p,s)
            with self.assertRaises(ValueError):m._reconcile(p,r,s,e)

    def test_production_counts_are_not_cli_adjustable(self):
        self.assertEqual(m.EXPECTED['prior'],2174)
        self.assertEqual((m.EXPECTED['human'],m.EXPECTED['ai']),(1278,896))
        code=Path(m.__file__).read_text()
        self.assertNotIn("add_argument('--limit'",code)
        self.assertNotIn("add_argument('--expected",code)
        self.assertIn("default='preflight'",code)
        self.assertIn("default=4",code)


class AuditTests(unittest.TestCase):
    def test_region_native_bounds_and_two_bounded_passes(self):
        rows=m._reconcile(*fixture());configuration={'test':True}
        for row in rows:
            audit=audit_fixture(row,configuration)
            m.verify_region_audit(audit,row,configuration)
            self.assertFalse(audit['full_stream_completeness_established'])
            self.assertEqual(audit['coordinates']['crop_start_frame'],row['planned_native30']['start_frame'])

    def test_region_audit_claim_hash_and_coordinates_mutations_fail(self):
        row=m._reconcile(*fixture())[1];configuration={'test':True}
        mutations=[lambda a:a.update(full_stream_completeness_established=True),
            lambda a:a.update(historical_region_hash_match=None),
            lambda a:a.update(historical_expected_region_float64_sha256=None),
            lambda a:a['coordinates'].update(crop_start_frame=0),
            lambda a:a['sequential_passes'][1].update(empty_eof_probe_performed=True),
            lambda a:a.update(observed_current_region_float64_sha256='0'*64)]
        for mutate in mutations:
            audit=audit_fixture(row,configuration);mutate(audit)
            with self.assertRaises(ValueError):m.verify_region_audit(audit,row,configuration)


class MaterializationTests(unittest.TestCase):
    def test_real_synthetic_native_region_to_FLOAT_roundtrip_and_resume(self):
        # This integration fixture is newly generated, never a corpus waveform.
        region=m.pinned_module('standardize_native30_frozen_region_v1',m.REGION_DSP_SHA)
        dsp=region.dsp
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'synthetic_native.wav';rate=32000;start=7;frames=60*rate
            t=dsp.np.arange(start+frames+100,dtype=dsp.np.float64)/rate
            samples=dsp.np.column_stack((.02*dsp.np.sin(2*dsp.np.pi*440*t),.03*dsp.np.cos(2*dsp.np.pi*330*t)))
            dsp.sf.write(source,samples,rate,format='WAV',subtype='DOUBLE')
            historical=dsp.pcm_hash(samples[start:start+frames])
            row=m._reconcile(*fixture())[1]
            native={**row['native_evidence'],'path':str(source),'sha256':m.base.digest(source),'sample_rate_hz':rate}
            row={**row,'source_origin':native,'native_evidence':native,'execution_native_path':str(source),
                'original_native_path':str(source),'native_region':{**row['native_region'],'start_frame':start,
                    'frames':frames,'source_sha256':native['sha256'],'float64_sha256':historical},
                'planned_native30':{**row['planned_native30'],'start_frame':start+15*rate,'frames':30*rate}}
            contract={'version':m.VERSION,'expected_count':1,'rows':[row],'output_root':str(root/'output'),
                'configuration':region.CONFIG,'bindings':{},'runtime':None}
            result=m.run_contract(contract,region,1)
            self.assertEqual(result['status'],'complete_DSP_only')
            wav=root/'output/audio/item1.wav'
            values,output_rate=dsp.sf.read(wav,dtype='float32',always_2d=True)
            expected=dsp.resample_crop(samples[start+15*rate:start+45*rate],rate)
            self.assertEqual(output_rate,44100)
            self.assertEqual(values.shape,(1323000,2))
            self.assertEqual(values.tobytes(),expected.tobytes())
            proof=m.base.read_json(root/'output/items/item1.json')['payload']['audit']
            self.assertTrue(proof['historical_region_hash_match'])
            self.assertEqual(proof['decoded_through_frame_exclusive'],start+frames)
            self.assertFalse(proof['full_stream_completeness_established'])
            self.assertFalse(proof['sequential_passes'][0]['empty_eof_probe_performed'])
            self.assertEqual(m.run_contract(contract,region,1)['status'],'verified_existing_COMMIT')

    def prepare_fake(self,directory):
        root=Path(directory);source=root/'source.wav';source.write_bytes(b'synthetic_source_only')
        row=m._reconcile(*fixture())[0]
        native={**row['native_evidence'],'path':str(source),'sha256':m.base.digest(source)}
        row={**row,'source_origin':native,'native_evidence':native,'execution_native_path':str(source),
             'original_native_path':str(source),'native_region':{**row['native_region'],'source_sha256':native['sha256']}}
        config={'synthetic_fixture':True}
        contract={'version':m.VERSION,'expected_count':1,'rows':[row],'output_root':str(root/'output'),
                  'configuration':config,'bindings':{},'runtime':None}
        fake=types.SimpleNamespace(dsp=object())
        fake.standardize=mock.Mock(return_value=(b'synthetic_values',audit_fixture(row,config)))
        return contract,fake,source

    @staticmethod
    def publish_fake(dsp,path,values):
        with path.open('xb') as stream:stream.write(b'synthetic_FLOAT_WAV_mock')

    def test_resume_commit_and_source_output_receipt_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            contract,dsp,source=self.prepare_fake(directory)
            with mock.patch.object(m.base,'publish_waveform',side_effect=self.publish_fake):
                first=m.run_contract(contract,dsp,workers=1)
                self.assertEqual(first['status'],'complete_DSP_only')
                self.assertEqual(dsp.standardize.call_count,1)
                args,kwargs=dsp.standardize.call_args
                self.assertIsNone(kwargs['expected_region_float64_sha256'])
                self.assertEqual(kwargs['region_frames'],60*44100)
                second=m.run_contract(contract,dsp,workers=1)
                self.assertEqual(second['status'],'verified_existing_COMMIT')
                self.assertEqual(dsp.standardize.call_count,1)
            output=Path(contract['output_root']);before=m.base.digest(output/'COMMIT.json')
            source.write_bytes(b'changed_source')
            with self.assertRaises(ValueError):m.run_contract(contract,dsp,workers=1)
            self.assertEqual(m.base.digest(output/'COMMIT.json'),before)
            self.assertTrue((output/'audio/item0.wav').is_file())

    def test_output_and_receipt_hash_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            contract,dsp,_=self.prepare_fake(directory)
            with mock.patch.object(m.base,'publish_waveform',side_effect=self.publish_fake):m.run_contract(contract,dsp,1)
            root=Path(contract['output_root']);row=contract['rows'][0];sha=m.base.value_hash(contract)
            wav=root/'audio/item0.wav';original=wav.read_bytes();wav.write_bytes(b'wrong')
            with self.assertRaises(ValueError):m.verify_receipt(root,row,sha,contract['configuration'])
            wav.write_bytes(original)
            path=root/'items/item0.json';envelope=m.base.read_json(path);envelope['payload']['row']['role']='reserved'
            path.write_text(json.dumps(envelope))
            with self.assertRaises(ValueError):m.verify_receipt(root,row,sha,contract['configuration'])

    def test_retained_failure_then_successful_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            contract,dsp,_=self.prepare_fake(directory)
            dsp.standardize.side_effect=ValueError('synthetic decoder failure')
            result=m.run_contract(contract,dsp,1);root=Path(contract['output_root'])
            self.assertEqual(result['status'],'partial_no_COMMIT')
            self.assertFalse((root/'COMMIT.json').exists())
            retained=list((root/'failures').iterdir());self.assertEqual(len(retained),1)
            digest=m.base.digest(retained[0]);dsp.standardize.side_effect=None
            with mock.patch.object(m.base,'publish_waveform',side_effect=self.publish_fake):result=m.run_contract(contract,dsp,1)
            self.assertEqual(result['status'],'complete_DSP_only')
            self.assertEqual(m.base.digest(retained[0]),digest)
            self.assertIn('failures/'+retained[0].name,m.base.read_json(root/'COMMIT.json')['products'])

    def test_unreceipted_audio_retained_and_no_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            contract,dsp,_=self.prepare_fake(directory);root=Path(contract['output_root'])
            dsp.standardize.side_effect=ValueError('create initialized failure first')
            m.run_contract(contract,dsp,1);dsp.standardize.side_effect=None
            (root/'audio/item0.wav').write_bytes(b'unreceipted')
            result=m.run_contract(contract,dsp,1)
            self.assertEqual(result['status'],'partial_no_COMMIT')
            self.assertEqual((root/'audio/item0.wav').read_bytes(),b'unreceipted')
            self.assertFalse((root/'COMMIT.json').exists())

    def test_contract_conflict_writer_lock_and_extra_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            contract,dsp,_=self.prepare_fake(directory);root=Path(contract['output_root'])
            with m.base.writer_lock(root):
                with self.assertRaises(RuntimeError):m.run_contract(contract,dsp,1)
            with mock.patch.object(m.base,'publish_waveform',side_effect=self.publish_fake):m.run_contract(contract,dsp,1)
            changed={**contract,'extra_contract_field':True}
            with self.assertRaises(ValueError):m.run_contract(changed,dsp,1)
            (root/'unplanned.txt').write_text('unplanned')
            with self.assertRaises(ValueError):m.run_contract(contract,dsp,1)


if __name__=='__main__':
    unittest.main()
