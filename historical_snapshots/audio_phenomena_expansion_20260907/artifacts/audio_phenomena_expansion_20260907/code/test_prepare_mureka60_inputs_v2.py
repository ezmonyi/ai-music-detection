"""Synthetic-only v2 interval and provenance tests; no acquired audio access."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import prepare_mureka60_inputs_v2 as M


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(M.json_bytes(value))


class FakeSoundFile:
    actual_override = None
    header_override = None
    shift_seek = False
    short_sequential_reads = False
    calls = 0
    def __init__(self,path):
        self.ident=Path(path).stem
        self.frames=9581607 if self.ident=='97136' else 12
        if self.header_override is not None:self.frames=self.header_override
        self.actual=11 if self.ident=='97136' else 12
        if self.actual_override is not None:self.actual=self.actual_override
        self.samplerate,self.channels,self.pos=M.SR,2,0
        self.seeked=False
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def blocks(self,*args,**kwargs):raise AssertionError('blocks() must never be used')
    def seek(self,pos):self.pos=pos;self.seeked=True;return pos
    def read(self,frames,dtype='float64',always_2d=True):
        FakeSoundFile.calls+=1
        count=max(0,min(frames,self.actual-self.pos))
        if self.short_sequential_reads and not self.seeked:count=min(count,3)
        result=(np.arange(self.pos*2,(self.pos+count)*2,dtype=np.float64).reshape(-1,2)/10)+1e-12
        self.pos+=count
        if self.shift_seek and self.seeked:result+=1e-10
        return result.astype(dtype)


def fixture(root,audit):
    identifiers=[str(i) for i in range(649)]+['97136']
    ranked=[dict(id=i,path=f'mureka_v9/{i}.mp3',bytes=1,sha256=hashlib.sha256(b'x').hexdigest(),
                 role='reserved_unscored',source='Mureka v9',reference_group_id='music8k_reference_'+i,
                 rank=hashlib.sha256((M.SEED+'|'+i).encode()).hexdigest()) for i in identifiers]
    ranked.sort(key=lambda r:r['rank'])
    c=dict(repo='homura23/MUSIC8K',revision=M.REVISION,seed=M.SEED,status='frozen_for_acquisition_only',
           classifier_authorized=False,selected_count=500,ranked_candidates=ranked,selected=ranked[:500],
           excluded_probe_ids=[str(i) for i in range(1000,1012)],intended_bytes=500)
    assert any(r['id']=='97136' for r in c['selected'])
    put(root/'contract.json',c);ch=M.sha_file(root/'contract.json')
    receipts,physical,surveyrows={ },[],[]
    for row in c['selected']:
        path=root/'raw'/row['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x')
        receipt=dict(row,contract_sha256=ch,status='passed',eligible_native60=True,all_samples_finite=True,
                     sample_rate=M.SR,channels=2,decoded_frames=12,decoded_duration_s=12/M.SR,
                     decoded_native_float32_sha256='f'*64)
        name=row['id']+'.json';put(root/'items'/name,receipt);receipts[name]=M.sha_file(root/'items'/name)
        physical.append(dict(id=row['id'],sha256=row['sha256'],bytes=1,sample_rate=M.SR,channels=2,frames=12,
                             eligible_native60=True,decoded_native_float32_sha256='f'*64))
        surveyrows.append(dict(id=row['id'],source_expected_sha256=row['sha256'],ffmpeg_decoded_frames=12,
                               sf_header_frames=9581607 if row['id']=='97136' else 12,
                               sf_sample_rate=M.SR,ffmpeg_sample_rate=M.SR,sf_channels=2,ffmpeg_channels=2))
    summary=dict(status='completed',selected=500,passed=500,eligible_native60=500,contract_sha256=ch,
                 bytes=500,classifiers_fitted=0,neural_inference=False,receipts_sha256=receipts)
    put(root/'summary.json',summary)
    local=dict(status='passed',selected=500,eligible_native60=500,all_original_files_rehashed=True,
               selection_independently_reconstructed=True,classifier_admission_authorized=False,
               source_revision=M.REVISION,contract_sha256=ch,summary_sha256=M.sha_file(root/'summary.json'),bytes=500,rows=physical)
    put(root/'local_physical_audit_v1.json',local)
    put(root/'remote_copy_audit_v1.json',dict(status='passed',rows=500,bytes=500,
        local_audit_sha256=M.sha_file(root/'local_physical_audit_v1.json'),remote_root=str(root),
        all_raw_files_rehashed=True,all_item_receipts_rehashed=True,classifier_fitted=False))
    put(audit/'mureka500_decoder_frames_v1.json',dict(count=500,source_contract_sha256=ch,changed_admission=False,
                                                 records=surveyrows,header_mismatch_count=1))
    put(audit/'mureka97136_decoder_diagnosis_v3.json',dict(status='read_only_diagnosis',changed_admission=False,
        source_and_receipt_unchanged=True,source_sha256=hashlib.sha256(b'x').hexdigest(),acquisition_frames=12,
        sf_header=dict(frames=9581607)))
    return c,ch,{p.name:M.sha_file(p) for p in audit.iterdir()}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'source';self.out=Path(self.tmp.name)/'output';self.audit=Path(self.tmp.name)/'audit'
        self.c,ch,evidence=fixture(self.root,self.audit)
        for name,value in [('ACQUISITION_SHA',ch),('AUDIT_ROOT',self.audit),('EVIDENCE',evidence),('FRAMES',8)]:
            p=patch.object(M,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(M.sf,'SoundFile',FakeSoundFile);p.start();self.addCleanup(p.stop)
        FakeSoundFile.actual_override=None;FakeSoundFile.header_override=None;FakeSoundFile.shift_seek=False
        FakeSoundFile.short_sequential_reads=False
        p=patch.object(M,'print',create=True);p.start();self.addCleanup(p.stop)

    def test_all500_freeze_only_then_validate_without_redecode(self):
        c=M.freeze(self.root,self.out)
        self.assertEqual(c['schema_version'],2);self.assertEqual(len(c['rows']),500)
        self.assertFalse((self.out/'audio').exists());self.assertFalse((self.out/'inference_manifest.csv').exists())
        count=FakeSoundFile.calls
        self.assertEqual(M.validate_contract(c,self.root,self.out),c['rows'])
        self.assertEqual(FakeSoundFile.calls,count)
        self.assertEqual([r['id'] for r in c['rows']],[r['id'] for r in self.c['selected']])
        changed=next(r for r in c['rows'] if r['id']=='97136')
        self.assertEqual((changed['native_frames'],changed['sf_header_frames'],changed['sf_actual_read_frames']),(12,9581607,11))
        self.assertEqual(changed['decoder_admission_claim'],M.CLAIM)
        self.assertTrue(all(r['role']==M.ROLE for r in c['rows']))

    def test_actual_eof_short_despite_inflated_header(self):
        FakeSoundFile.actual_override=9
        rows,_=M.audit_inputs(self.root)
        row=next(r for r in rows if r['id']=='97136')
        with self.assertRaisesRegex(ValueError,'Actual sequential EOF'):
            M.verify_interval(self.root,row,9581607)
        with self.assertRaisesRegex(ValueError,'Actual sequential EOF'):
            M.freeze(self.root,self.out)
        self.assertFalse((self.out/'materialization_contract.json').exists())

    def test_short_reads_continue_to_actual_empty_without_padding(self):
        FakeSoundFile.short_sequential_reads=True
        c=M.freeze(self.root,self.out)
        changed=next(r for r in c['rows'] if r['id']=='97136')
        self.assertEqual(changed['sf_actual_read_frames'],11)
        self.assertEqual(changed['crop_frames'],8)

    def test_float64_seek_mismatch_despite_same_float32(self):
        FakeSoundFile.shift_seek=True
        with self.assertRaisesRegex(ValueError,'exactly match sequential float64'):
            M.freeze(self.root,self.out)

    def test_unexpected_header_mismatch(self):
        FakeSoundFile.header_override=13
        with self.assertRaisesRegex(ValueError,'header mismatch inventory'):
            M.freeze(self.root,self.out)

    def test_source_and_missing_remote_audit(self):
        path=self.root/'raw'/self.c['selected'][0]['path'];path.write_bytes(b'y')
        with self.assertRaisesRegex(ValueError,'bytes/hash'):
            M.freeze(self.root,self.out)
        path.write_bytes(b'x');(self.root/'remote_copy_audit_v1.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing regular'):
            M.freeze(self.root,self.out)

    def test_evidence_hash_changed(self):
        (self.audit/'mureka97136_decoder_diagnosis_v3.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'evidence missing or changed'):
            M.freeze(self.root,self.out)

    def test_ranking_probe_source_admission(self):
        for change in (lambda c:c['selected'].reverse(),lambda c:c['ranked_candidates'].reverse(),
                       lambda c:c.__setitem__('repo','other/repo'),lambda c:c.__setitem__('classifier_authorized',True),
                       lambda c:c['excluded_probe_ids'].__setitem__(0,c['selected'][0]['id'])):
            c=copy.deepcopy(self.c);change(c)
            with self.assertRaises(ValueError):M.selected_rows(c)

    def test_foreign_destination_and_repeated_freeze(self):
        self.out.mkdir();(self.out/'foreign').write_bytes(b'preserve')
        with self.assertRaisesRegex(ValueError,'foreign/unreceipted'):M.freeze(self.root,self.out)
        (self.out/'foreign').unlink();M.freeze(self.root,self.out)
        with self.assertRaisesRegex(ValueError,'already frozen'):M.freeze(self.root,self.out)

    def test_canonical_code_runtime_observation_and_alias_corruption(self):
        c=M.freeze(self.root,self.out)
        bad=copy.deepcopy(c);bad['role']='development'
        with self.assertRaisesRegex(ValueError,'canonical hash'):M.validate_contract(bad,self.root,self.out)
        for mutation in (lambda b:b.__setitem__('code_sha256','0'*64),lambda b:b.__setitem__('runtime',{}),
                         lambda b:b['rows'][0].__setitem__('sf_actual_read_frames',0),
                         lambda b:b['rows'][0].__setitem__('native_crop_float64_sha256','0'*64)):
            bad=copy.deepcopy(c);mutation(bad)
            bad['contract_sha256']=M.canonical_hash({k:v for k,v in bad.items() if k!='contract_sha256'})
            with self.assertRaises(ValueError):M.validate_contract(bad,self.root,self.out)

    def test_no_materialization_without_freeze(self):
        with self.assertRaises(FileNotFoundError):M.materialize(self.root,self.out)

    def test_no_publication_on_item_failure(self):
        M.freeze(self.root,self.out)
        with patch.object(M,'one_item',side_effect=ValueError('item failed')):
            with self.assertRaisesRegex(ValueError,'item failed'):M.materialize(self.root,self.out)
        self.assertFalse((self.out/'inference_manifest.csv').exists())
        self.assertFalse(list(self.out.glob('inference_shard_*.txt')))

    def test_all500_manifest_native_fields_and_21_shards(self):
        M.freeze(self.root,self.out)
        def item(source,output,row,ch):
            i='music8k_mureka_v9_'+row['id'];put(output/'items'/(i+'.json'),{'synthetic':True})
            return dict(item_id=i,standardized_path=str(output/'audio'/(i+'.wav')),source_audio_path=str(source/'raw'/row['path']),
                standardized_file_sha256='a'*64,crop_start_frame=row['crop_start_frame'],source_total_frames=row['native_frames'],
                acquisition_ffmpeg_frames=row['native_frames'],crop_end_frame_exclusive=row['crop_start_frame']+M.FRAMES,
                group_id=row['reference_group_id'],source_audio_sha256=row['sha256'],
                **{k:row[k] for k in ['sf_header_frames','sf_actual_read_frames','sf_header_minus_acquisition_frames',
                    'sf_actual_minus_acquisition_frames','center_float64_sha256','center_float32_sha256',
                    'native_crop_float64_sha256','native_crop_float32_sha256']})
        with patch.object(M,'one_item',side_effect=item):result=M.materialize(self.root,self.out)
        self.assertEqual([s['rows'] for s in result['shards']],[24]*20+[20])
        self.assertEqual(result['header_mismatch_ids'],['97136']);self.assertEqual(result['actual_eof_mismatch_ids'],['97136'])
        import csv
        with (self.out/'native_metadata_60s.csv').open() as f:
            native=list(csv.DictReader(f))
        self.assertEqual(len(native),500)
        row=next(r for r in native if r['id'].endswith('_97136'))
        self.assertEqual((row['physical_frames'],row['sf_header_frames'],row['sf_actual_read_frames']),('12','9581607','11'))
        self.assertEqual(row['native_crop_float64_sha256'],row['center_float64_sha256'])


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'source';self.out=Path(self.tmp.name)/'output'
        self.path=self.root/'raw/mureka_v9/1.mp3';self.path.parent.mkdir(parents=True)
        self.data=(np.arange(26,dtype=np.float64).reshape(13,2)/10)+1e-12
        # Synthetic DOUBLE WAV under .mp3 suffix, never an acquired recording.
        sf.write(self.path,self.data,M.SR,format='WAV',subtype='DOUBLE')
        p=patch.object(M,'FRAMES',8);p.start();self.addCleanup(p.stop)
        self.row=dict(id='1',path='mureka_v9/1.mp3',bytes=self.path.stat().st_size,sha256=M.sha_file(self.path),
                      native_frames=13,crop_start_frame=2,crop_frames=8,reference_group_id='music8k_reference_1',
                      acquisition_decoded_float32_sha256='f'*64)
        self.row.update(M.verify_interval(self.root,self.row,13))

    def test_float64_float32_exact_preservation_odd_center_and_resume(self):
        r=M.one_item(self.root,self.out,self.row,'c'*64)
        back,sr=sf.read(r['standardized_path'],dtype='float32',always_2d=True)
        self.assertTrue(np.array_equal(back,self.data[2:10].astype(np.float32)))
        self.assertEqual(r['native_crop_float64_sha256'],M.waveform_hash(self.data[2:10],'<f8'))
        self.assertGreater(back.max(),1);self.assertGreater(back.mean(),0)
        self.assertEqual(r,M.one_item(self.root,self.out,self.row,'c'*64))

    def test_frozen_center_hash_mismatch(self):
        self.row['center_float64_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'center sample hash'):M.one_item(self.root,self.out,self.row,'c'*64)

    def test_resume_output_source_and_contract_corruption(self):
        r=M.one_item(self.root,self.out,self.row,'c'*64)
        with self.assertRaisesRegex(ValueError,'receipt/contract'):M.one_item(self.root,self.out,self.row,'d'*64)
        with Path(r['standardized_path']).open('r+b') as f:f.seek(-4,2);f.write(b'\x00'*4)
        with self.assertRaisesRegex(ValueError,'output hash'):M.one_item(self.root,self.out,self.row,'c'*64)
        with self.path.open('r+b') as f:f.seek(-4,2);f.write(b'\x00'*4)
        with self.assertRaisesRegex(ValueError,'bytes/hash'):M.one_item(self.root,self.out,self.row,'c'*64)

    def test_no_overwrite_orphan(self):
        path=self.out/'audio/music8k_mureka_v9_1.wav';path.parent.mkdir(parents=True);path.write_bytes(b'owned')
        with self.assertRaisesRegex(ValueError,'Unreceipted'):M.one_item(self.root,self.out,self.row,'c'*64)
        self.assertEqual(path.read_bytes(),b'owned')


if __name__=='__main__':unittest.main(verbosity=2)
