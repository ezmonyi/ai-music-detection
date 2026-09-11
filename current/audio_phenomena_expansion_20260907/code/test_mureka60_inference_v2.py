"""CPU-only synthetic checks. Never reads real Mureka media or invokes inference."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import prepare_mureka60_inputs_v2 as P
import run_mureka60_inference_v2 as M
import verify_extract_mureka60_v2 as A


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(P.json_bytes(value))


def prepared_fixture(root):
    source, output = root/'source', root/'prepared'
    source.mkdir(); output.mkdir()
    src, receipts, native = [], [], []
    for i in range(500):
        raw = source/'raw/mureka_v9'/f'{i}.mp3'
        raw.parent.mkdir(parents=True, exist_ok=True); raw.write_bytes(b'synthetic-native-'+str(i).encode())
        row = dict(id=str(i), path=f'mureka_v9/{i}.mp3', sha256=P.sha_file(raw), native_frames=P.FRAMES+3,
                   crop_start_frame=1, crop_frames=P.FRAMES, reference_group_id=f'music8k_reference_{i}',
                   sf_header_frames=P.FRAMES+3, sf_actual_read_frames=P.FRAMES+3,
                   native_crop_float64_sha256='a'*64, native_crop_float32_sha256='b'*64)
        src.append(row)
    c = dict(status='frozen_before_materialization', purpose='measurement_only', role=M.ROLE,
             classifier_admission_authorized=False, source_root=str(source), output_dir=str(output),
             source_repo='homura23/MUSIC8K', source_revision=P.REVISION, acquisition_contract_sha256=P.ACQUISITION_SHA,
             code_sha256=P.sha_file(P.__file__), configuration=P.CONFIG, configuration_sha256=P.canonical_hash(P.CONFIG),
             runtime={}, runtime_sha256=P.canonical_hash({}), rows=src, selected_ids=[r['id'] for r in src],
             input_file_sha256={}, decoder_evidence_sha256={})
    c['contract_sha256'] = P.canonical_hash(c)
    for row in src:
        i = row['id']; item = f'music8k_mureka_v9_{i}'; audio = output/'audio'/(item+'.wav')
        audio.parent.mkdir(exist_ok=True); audio.write_bytes(b'synthetic-wave-'+i.encode())
        r = dict(item_id=item, role=M.ROLE, source_id='Mureka_v9', source_group='Mureka_v9', label=1,
                 status='verified', classifier_admission_authorized=False, acquisition_role='reserved_unscored',
                 contract_sha256=c['contract_sha256'], input_row_sha256=P.canonical_hash(row),
                 source_audio_path=str(source/'raw'/row['path']), source_audio_sha256=row['sha256'],
                 group_id=row['reference_group_id'], source_total_frames=row['native_frames'], crop_start_frame=1,
                 crop_frames=P.FRAMES, crop_end_frame_exclusive=P.FRAMES+1, standardized_path=str(audio), duration=60,
                 audio_offset_s=0, requires_crop=0, standardized_sr=44100, native_sr=44100,
                 standardized_channels=2, standardized_frames=P.FRAMES,
                 standardized_file_sha256=P.sha_file(audio), standardized_file_bytes=audio.stat().st_size)
        r.update({key: row[key] for key in ('sf_header_frames', 'sf_actual_read_frames',
                  'native_crop_float64_sha256', 'native_crop_float32_sha256')})
        r['standardized_waveform_sha256'] = row['native_crop_float32_sha256']
        put(output/'items'/(item+'.json'),r); receipts.append(r)
        native.append(dict(id=item, role=M.ROLE, source_group='Mureka_v9', label=1, classifier_admission_authorized=False,
                           group_id=r['group_id'], acquisition_role='reserved_unscored', native_sample_rate_hz=44100,
                           physical_sample_rate_hz=44100, physical_channels=2, physical_frames=row['native_frames'],
                           audio_path=r['source_audio_path'], source_audio_sha256=r['source_audio_sha256'],
                           crop_start_frame=1, crop_frames=P.FRAMES, crop_end_frame_exclusive=P.FRAMES+1, audio_offset_s=1/44100))
    for n, row in zip(native, src):
        n.update({key: row[key] for key in ('sf_header_frames', 'sf_actual_read_frames',
                    'native_crop_float64_sha256', 'native_crop_float32_sha256')})
    (output/'inference_manifest.csv').write_bytes(P.csv_bytes(receipts))
    (output/'native_metadata_60s.csv').write_bytes(P.csv_bytes(native))
    shards=[]
    for i in range(21):
        path = output/f'inference_shard_{i:02d}.txt'
        part=receipts[i*24:i*24+24]
        path.write_text(''.join(r['standardized_path']+'\n' for r in part))
        shards.append(dict(index=i, path=str(path), rows=len(part), sha256=P.sha_file(path)))
    s=dict(role=M.ROLE, classifier_admission_authorized=False, status='verified', rows=500,
           contract_sha256=c['contract_sha256'], manifest_sha256=P.sha_file(output/'inference_manifest.csv'),
           native_metadata_sha256=P.sha_file(output/'native_metadata_60s.csv'), shards=shards,
           receipts_sha256={r['item_id']+'.json':P.sha_file(output/'items'/(r['item_id']+'.json')) for r in receipts})
    put(output/'materialization_contract.json', c); put(output/'materialization_summary.json',s)
    return source, output, src, c, s, [{k:str(v) for k,v in r.items()} for r in receipts]


class IntervalEvidenceTests(unittest.TestCase):
    def evidence(self):
        source = dict(sf_header_frames=9000000, sf_actual_read_frames=6000000,
                      native_crop_float64_sha256='a'*64, native_crop_float32_sha256='b'*64)
        receipt = dict(source, crop_end_frame_exclusive=4400000,
                       standardized_waveform_sha256='b'*64)
        native = {k:str(v) for k,v in source.items()}
        return receipt, native, source

    def test_header_is_not_actual_eof(self):
        M.validate_interval_evidence(*self.evidence())

    def test_real_eof_before_crop_is_rejected_despite_large_header(self):
        receipt, native, source = self.evidence()
        source['sf_actual_read_frames'] = receipt['sf_actual_read_frames'] = 4399999
        native['sf_actual_read_frames'] = '4399999'
        with self.assertRaisesRegex(ValueError, 'real decoded EOF'):
            M.validate_interval_evidence(receipt, native, source)

    def test_native_window_hash_and_standardized_hash_are_bound(self):
        receipt, native, source = self.evidence()
        native['native_crop_float64_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'sample hash'):
            M.validate_interval_evidence(receipt, native, source)
        receipt, native, source = self.evidence()
        receipt['standardized_waveform_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'standardized samples'):
            M.validate_interval_evidence(receipt, native, source)


class PreparedTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        self.source,self.out,self.src,self.c,self.s,self.rows=prepared_fixture(self.root)
        for obj,name,value in ((P,'SOURCE',self.source),(P,'OUTPUT',self.out),
                              (P,'validate_contract',lambda contract,source,output:self.src)):
            p=patch.object(obj,name,value); p.start(); self.addCleanup(p.stop)

    def verify(self):
        return M.validate_prepared(self.out, decode=False)

    def test_all500_and_21_shards_valid(self):
        rows,shards,bindings=self.verify()
        self.assertEqual(len(rows),500); self.assertEqual([len(p) for _,p in shards],[24]*20+[20])
        self.assertTrue(all(r['role']==M.ROLE for r in rows)); self.assertGreater(len(bindings),500)

    def mutate_contract(self,key,value):
        c=copy.deepcopy(self.c); c[key]=value
        c['contract_sha256']=P.canonical_hash({k:v for k,v in c.items() if k!='contract_sha256'})
        s=dict(self.s,contract_sha256=c['contract_sha256'])
        put(self.out/'materialization_contract.json',c); put(self.out/'materialization_summary.json',s)

    def test_wrong_source_rejected(self):
        self.mutate_contract('source_repo','other/source')
        with self.assertRaisesRegex(ValueError,'Wrong source'): self.verify()

    def test_development_role_rejected(self):
        self.mutate_contract('role','development')
        with self.assertRaisesRegex(ValueError,'Forbidden preparation'): self.verify()

    def test_classifier_admission_rejected(self):
        self.mutate_contract('classifier_admission_authorized',True)
        with self.assertRaisesRegex(ValueError,'Forbidden preparation'): self.verify()

    def test_499_summary_rejected(self):
        put(self.out/'materialization_summary.json',dict(self.s,rows=499))
        with self.assertRaisesRegex(ValueError,'complete500'): self.verify()

    def test_selection_order_rejected(self):
        self.mutate_contract('selected_ids',list(reversed(self.c['selected_ids'])))
        with self.assertRaisesRegex(ValueError,'selection/evidence'): self.verify()

    def test_source_receipt_provenance_rejected(self):
        self.src[0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'selection/evidence'): self.verify()

    def test_audio_tampering_rejected(self):
        Path(self.rows[0]['standardized_path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'audio bytes'): self.verify()

    def test_native_metadata_group_tampering_rejected(self):
        path=self.out/'native_metadata_60s.csv'
        with path.open() as f: rows=list(csv.DictReader(f))
        rows[0]['group_id']='wrong-reference-group'
        path.write_bytes(P.csv_bytes(rows))
        s=dict(self.s,native_metadata_sha256=P.sha_file(path))
        put(self.out/'materialization_summary.json',s)
        with self.assertRaisesRegex(ValueError,'Native FHM'): self.verify()

    def test_missing_preparation_receipt_rejected(self):
        (self.out/'items'/(self.rows[0]['item_id']+'.json')).unlink()
        with self.assertRaisesRegex(ValueError,'receipt union'): self.verify()

    def test_shard_reordering_even_with_updated_hash_rejected(self):
        path=self.out/'inference_shard_00.txt'; lines=path.read_text().splitlines()
        path.write_text('\n'.join(reversed(lines))+'\n')
        s=copy.deepcopy(self.s); s['shards'][0]['sha256']=P.sha_file(path)
        put(self.out/'materialization_summary.json',s)
        with self.assertRaisesRegex(ValueError,'order mismatch'): self.verify()

    def test_duplicate_shard_and_500_identity_rejected(self):
        rows=copy.deepcopy(self.rows); rows[1]=rows[0]
        with self.assertRaisesRegex(ValueError,'unique rows'): M.validate_shards(self.out,self.s,rows)
        with self.assertRaisesRegex(ValueError,'unique rows'): M.validate_shards(self.out,self.s,self.rows[:-1])

    def test_no_gpu_reached_on_invalid_preparation(self):
        self.mutate_contract('role','development')
        a=SimpleNamespace(aio_gpus=[6],beat_gpus=[7],output_root=self.out/'inference',prepared_dir=self.out)
        with patch.object(M.original,'require_idle_5090') as gpu, patch.object(M,'runtime_snapshot') as runtime:
            with self.assertRaises(ValueError): M.run_locked(a)
            gpu.assert_not_called(); runtime.assert_not_called()


class LedgerTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve(); self.output=self.root/'inference'
        self.rows=[]
        for i in range(500):
            path=self.root/f'item{i}.wav'; path.write_bytes(str(i).encode())
            self.rows.append(dict(item_id=f'item{i}',standardized_path=str(path),standardized_file_sha256=P.sha_file(path),
                                  standardized_file_bytes=str(path.stat().st_size)))
        self.shards=[(i,[r['standardized_path'] for r in self.rows[i*24:i*24+24]]) for i in range(21)]
        self.contract=M.sealed(dict(M.scope(),runtime_root=str(self.root/'runtime'),allinone_gpus=[6],beats_gpus=[7],dependencies_sha256={}))
        put(self.output/'run_contract.json',self.contract); self.contract_sha=P.sha_file(self.output/'run_contract.json')
        for name in ('demix','structure','spec','beats','logs','receipts'): (self.output/name).mkdir()

    def receipt(self,stage,index):
        rows=self.rows[index*24:index*24+24]
        outputs={}
        for row in rows:
            for path in M.original.product_paths(stage,row['item_id'],self.output):
                path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'synthetic-product')
                outputs[str(path)]=M.file_record(path)
        log=self.output/'logs'/f'{stage}_{index:02d}.log'; log.write_bytes(b'synthetic-log')
        r=M.sealed(dict(M.scope(),status='passed',exit_code=0,stage=stage,shard_index=index,
            run_contract_sha256=self.contract_sha,item_ids=[r['item_id'] for r in rows],gpu=6 if stage=='allinone' else 7,
            command=M.original.command(stage,[r['standardized_path'] for r in rows],self.root/'runtime',self.output),
            dependencies_sha256={},inputs=M.input_records(rows),outputs=outputs,log_path=str(log),log=M.file_record(log)))
        put(self.output/'receipts'/f'{stage}_{index:02d}.json',r)
        return r,rows

    def verify(self,receipt,stage='beats',index=0):
        M.validate_receipt(receipt,self.contract,self.contract_sha,stage,index,self.rows[index*24:index*24+24],self.output)

    def test_resume_revalidates_output_input_log_hashes(self):
        r,rows=self.receipt('beats',0); self.verify(r)
        for path in [Path(next(iter(r['outputs']))),Path(rows[0]['standardized_path']),Path(r['log_path'])]:
            old=path.read_bytes(); path.write_bytes(b'tampered')
            with self.assertRaises(ValueError): self.verify(r)
            path.write_bytes(old)

    def test_missing_product_and_extra_receipt_product_rejected(self):
        r,_=self.receipt('beats',0)
        bad={k:v for k,v in r.items() if k!='canonical_sha256'}; bad['outputs']=dict(r['outputs'],unknown={})
        with self.assertRaisesRegex(ValueError,'union mismatch'): self.verify(M.sealed(bad))
        Path(next(iter(r['outputs']))).unlink()
        with self.assertRaises(ValueError): self.verify(r)

    def test_nonzero_exit_and_scoring_receipts_rejected(self):
        r,_=self.receipt('beats',0)
        for key,value in [('exit_code',1),('scores_generated',True),('classifier_fitted',True),('role','development')]:
            bad={k:v for k,v in r.items() if k!='canonical_sha256'}; bad[key]=value
            with self.assertRaises(ValueError): self.verify(M.sealed(bad))

    def test_unreceipted_products_fail_closed(self):
        path=self.output/'beats/item0.beats'; path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'Unreceipted'): M.verify_all_receipts(self.output,self.contract,self.rows,self.shards)

    def test_unreceipted_logs_fail_closed(self):
        (self.output/'logs/beats_00.log').write_text('failed original inference')
        with self.assertRaisesRegex(ValueError,'Unreceipted'): M.verify_all_receipts(self.output,self.contract,self.rows,self.shards)

    def test_missing_completion_before_audit_or_extraction(self):
        a=SimpleNamespace(inference_root=self.output,prepared_dir=self.root)
        with patch.object(M,'validate_prepared') as preparation, patch.object(A.subprocess,'run') as extraction:
            with self.assertRaises(ValueError): A.audit(a)
            preparation.assert_not_called(); extraction.assert_not_called()

    def test_full42_union_and_missing_receipt_rejected(self):
        for stage in M.STAGES:
            for i in range(21): self.receipt(stage,i)
        receipts=M.verify_all_receipts(self.output,self.contract,self.rows,self.shards,require_complete=True)
        self.assertEqual(len(receipts),42)
        completion=M.sealed(dict(M.scope(),status='passed',rows=500,completed_stage_shards=42,
            run_contract_sha256=self.contract_sha,receipts_sha256=receipts,dependencies_sha256={}))
        put(self.output/'completion.json',completion)
        M.verify_completion(self.output,self.contract,self.rows,self.shards)
        (self.output/'receipts/beats_20.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing stage-shard'): M.verify_completion(self.output,self.contract,self.rows,self.shards)

    def test_immutable_publication_preserves_existing(self):
        path=self.root/'contract.json'; P.publish(path,b'original')
        with self.assertRaises(ValueError): P.publish(path,b'changed')
        self.assertEqual(path.read_bytes(),b'original')

    def test_nonzero_process_is_fatal_and_publishes_no_completion(self):
        (self.output/'run_contract.json').unlink()
        a=SimpleNamespace(aio_gpus=[6],beat_gpus=[7],output_root=self.output,prepared_dir=self.root,
                          runtime_root=self.root/'runtime')
        with patch.object(M,'validate_prepared',return_value=(self.rows,self.shards,{})), \
             patch.object(M,'runtime_snapshot',return_value={}), \
             patch.object(M.original,'require_idle_5090'), \
             patch.object(M.subprocess,'run',side_effect=M.subprocess.CalledProcessError(1,['synthetic'])):
            with self.assertRaises(M.subprocess.CalledProcessError): M.run_locked(a)
        self.assertFalse((self.output/'completion.json').exists())
        self.assertEqual(list((self.output/'receipts').iterdir()),[])


class SafeguardTests(unittest.TestCase):
    def test_gpu_explicit_disjoint_reserved_and_idle_checks(self):
        M.validate_gpus([6],[7])
        for a,b in [([], [7]),([6],[6]),([0],[7]),([6,6],[7]),([6],[-1])]:
            with self.assertRaises(ValueError): M.validate_gpus(a,b)
        for data in ['NVIDIA GeForce RTX 5090, 1500, 0','NVIDIA GeForce RTX 5090, 0, 20','NVIDIA A100, 0, 0']:
            with patch.object(M.original.subprocess,'check_output',return_value=data):
                with self.assertRaises(RuntimeError): M.original.require_idle_5090(6)

    def test_pinned_helpers_and_original_command_parameters(self):
        self.assertGreater(len(M.dependency_snapshot()),3)
        c=M.original.command('beats',['input.wav'],Path('/runtime'),Path('/out'))
        self.assertEqual(c,['/runtime/venv/bin/beat_this','input.wav','-o','/out/beats','--model',
            '/runtime/checkpoints/hub/checkpoints/beat_this-final0.ckpt','--no-dbn','--gpu','0','--float16','--skip-existing'])
        c=M.original.command('allinone',['input.wav'],Path('/runtime'),Path('/out'))
        self.assertEqual(c,['/runtime/venv/bin/all-in-one-infer','input.wav','-o','/out/structure','-m','harmonix-all',
                           '-d','cuda','-k','--demix-dir','/out/demix','--spec-dir','/out/spec'])

    def test_synthetic_exact60_float_media_and_missing_beat(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); wave=root/'exact.wav'
            data=np.zeros((P.FRAMES,2),dtype=np.float32); data[:,1]=1.25
            sf.write(wave,data,44100,subtype='FLOAT')
            M.inspect_audio(wave,subtype='FLOAT')
            sf.write(root/'wrong.wav',data[:-1],44100,subtype='FLOAT')
            with self.assertRaises(ValueError): M.inspect_audio(root/'wrong.wav',subtype='FLOAT')
            with self.assertRaises(ValueError): M.original.inspect_beats(root/'missing.beats')
            self.assertFalse((root/'missing.beats').exists())

    def test_synthetic_stems_structure_spec_and_beat_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); wave=root/'input.wav'
            sf.write(wave,np.zeros((P.FRAMES,2),dtype=np.float32),44100,subtype='FLOAT')
            row={'item_id':'synthetic','standardized_path':str(wave)}
            for stem in M.original.STEMS:
                path=root/'demix/htdemucs/synthetic'/(stem+'.wav')
                path.parent.mkdir(parents=True,exist_ok=True); os.link(wave,path)
            put(root/'structure/synthetic.json',{'path':str(wave),'segments':[{'start':0,'end':60}]})
            (root/'spec').mkdir()
            spec=np.zeros((4,6000,81),dtype=np.float32); np.save(root/'spec/synthetic.npy',spec)
            products=M.original.verify_products('allinone',[row],root)
            self.assertEqual(len(products),6)
            np.save(root/'spec/synthetic.npy',spec.astype(np.float64))
            with self.assertRaisesRegex(ValueError,'spectrogram'): M.original.verify_products('allinone',[row],root)
            put(root/'structure/synthetic.json',{'path':str(wave),'segments':[{'start':1,'end':60}]})
            with self.assertRaises(ValueError): M.original.inspect_structure(root/'structure/synthetic.json',str(wave))
            beats=root/'wrong.beats'; beats.write_text('0\t1\n0\t2\n')
            with self.assertRaises(ValueError): M.original.inspect_beats(beats)

    def test_feature_wrapper_only_pinned_extractor_no_fit(self):
        a=SimpleNamespace(old_code_root=Path('/old'),prepared_dir=Path('/new'),bias=Path('/bias'),
                          inference_root=Path('/new/inference'),output_dir=Path('/new/measurements'),workers=2)
        command=A.extraction_command(a)
        self.assertEqual(command[1],'/old/extract_expanded_four_family.py')
        self.assertIn('--strict',command)
        self.assertFalse(any('fit' in word or 'score' in word or 'evaluate' in word for word in command))
        for flag in ('classifier_fitted','scores_generated','classifier_admission_authorized'):
            with self.assertRaises(ValueError): M.measurement_only(dict(M.scope(),**{flag:True}))


if __name__=='__main__':
    unittest.main(verbosity=2)
