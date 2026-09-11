"""Frozen prior2174 native60-to-native30 FLOAT materialization, never admission.

Default preflight binds metadata/code/runtime only. Explicit run uses two bounded
sequential prefix observations through each frozen region end, then one native
center30 resampling. Existing standardized60 products are evidence, not inputs.
Ordinary1571 historical interval hashes remain null; Mureka500/Saraga103 hashes
are mandatory. Interrupted outputs and failure receipts are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import uuid

VERSION='materialize_native30_prior2174_v1'
PLAN_SHA='94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964'
REVIEW_SHA='a5a5619116564f0d320c550404db57839842409b5e6330621f8075966362ecfa'
SCREEN_SHA='ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87'
HELPER_SHA='165f28d4e553ee679772aa1ac47b31f9a485b94d3a8ac409f25595577abe395a'
DSP_SHA='2e05a4d0d2e7789888aa41a53bece46e8de97fad34ef34f21354d165e77b55be'
REGION_DSP_SHA='1d8695e1312565687489aeda03beab5f7c42de78e8c14d30ca4ee545e62a847d'
REGION_TEST_SHA='0245b80577f5c03e5468d2890d87d6fc26d9c3ee15dd05eb0bcac4eb8dc8083e'
SOURCES={'MTG-Jamendo':481,'Mureka_v9':500,'Suno':396,'human_maestro_v3':300,
         'human_medleydb':156,'human_moisesdb':238,'human_saraga_hindustani_v1':103}
EXPECTED={'plan':3869,'prior':2174,'new':1695,'human':1278,'ai':896,
          'ordinary':1571,'mandatory_hash':603,'sources':SOURCES}
SPECIAL={'Mureka_v9','human_saraga_hindustani_v1'}
BOUNDS_ONLY='frozen_native_frame_bounds_from_receipt'
HASH_VERIFIED='verified_native_interval_hash'


def require(ok,message):
    if not ok:
        raise ValueError(message)


def pinned_module(name,digest):
    path=Path(__file__).resolve().with_name(name+'.py')
    require(path.is_file() and not path.is_symlink(),'pinned module must be regular')
    require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'pinned module SHA mismatch: '+name)
    module=importlib.import_module(name)
    require(Path(module.__file__).resolve()==path,'unexpected imported module path: '+name)
    return module


base=pinned_module('materialize_native30_new1695_v1',HELPER_SHA)


def hash_string(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _reconcile(plan,review,screen,expected=EXPECTED):
    """Metadata reconciliation; reduced count expectations are test-only Python input."""
    require(plan['schema_version']=='native30-origin-plan-v2'
            and plan['status']=='draft_not_admitted_not_frozen_for_execution','accepted plan schema/status')
    require(review['status']=='metadata_consistency_accepted_not_execution_frozen_or_audio_admitted'
            and review['plan_sha256']==PLAN_SHA,'parent metadata review')
    require(review['physical_source_audio_opened_by_review']==0 and review['classifier_fits']==0,'review scope')
    require(plan['counts']['audio_files_opened']==0 and plan['counts']['classifier_fits']==0,'plan scope')
    require(plan['counts']['planned_rows']==expected['plan'] and plan['counts']['prior60_rows']==expected['prior']
            and plan['counts']['new_native30_stereo_rows']==expected['new'],'plan population counts')
    require(screen['classifier_fits']==0 and screen['feature_extraction_authorized'] is False,'screen scope')
    all_rows=plan['rows']
    require(len(all_rows)==expected['plan'] and len({r['id'] for r in all_rows})==len(all_rows),'all plan identities')
    screens={r['id']:r for r in screen['rows']}
    components={r['component_id']:r for r in screen['components']}
    require(len(screens)==len(screen['rows']) and len(components)==len(screen['components']),'screen duplicate identity')
    selected=[]
    for row in sorted(all_rows,key=lambda r:r['id']):
        ident=row['id']
        require(isinstance(ident,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',ident),'safe item id')
        require(ident in screens,'plan row missing from full screen')
        screened=screens[ident]
        require(all(row[k]==screened[k] for k in ('id','label','role','source_group','group_id','component_id',
                'input_occurrences','selected_metadata_input')),'full plan/screen row disagreement')
        require(row['role']=='development' and row['label'] in ('0','1') and isinstance(row['group_id'],str)
                and row['group_id'] and screened['duration_exposure_candidate'] is True
                and not screened['exclusion_reasons'],'nondevelopment/excluded row')
        component=components[row['component_id']]
        require(ident in component['members'] and ident in component['candidate_screen_members']
                and not component['protected_relationships'],'protected/component membership mismatch')
        require(row['origin_set'] in ('prior60','native30_evidence_v2'),'unknown origin family')
        if row['origin_set']!='prior60':
            continue
        native,region,planned=row['source_origin'],row['native_region'],row['planned_native30']
        path=Path(native['path'])
        require(path.is_absolute() and path.parts[:2]==('/','mnt') and '..' not in path.parts,
                'prior native source must use the frozen /mnt original path')
        require(native['channels']==2 and type(native['channels']) is int
                and type(native['sample_rate_hz']) is int and native['sample_rate_hz']>0,'native stereo/rate evidence')
        require(screened['recorded_native_channels']==2 and screened['recorded_native_channel_status']=='stereo','screen native channels')
        require(hash_string(native['sha256']) and region['source_sha256']==native['sha256'],'native source SHA binding')
        rate=native['sample_rate_hz']
        require(type(region['start_frame']) is int and region['start_frame']>=0
                and type(region['frames']) is int and region['frames']==60*rate,'exact frozen native60 bounds')
        special=row['source_group'] in SPECIAL
        require(region['evidence_strength']==(HASH_VERIFIED if special else BOUNDS_ONLY),'historical evidence strength')
        if special:
            require(hash_string(region['float64_sha256']) and region['seek_equals_sequential_float64'] is True,
                    'mandatory Mureka/Saraga historical hash')
        else:
            require(region['float64_sha256'] is None,'ordinary historical hash must remain null')
        require(isinstance(region['decoder_limit'],str) and region['decoder_limit'],'decoder limits retained')
        require(planned=={'coordinate_space':'native_source_frames','derived_from_native_region':True,
            'frames':30*rate,'must_reproduce_native_region_before_subcrop':special,
            'policy':'center_exact30_within_approved_native60_region',
            'start_frame':region['start_frame']+15*rate},'frozen native30 coordinates/policy')
        prior=row['prior_standardized60_evidence_not_origin']
        require(native['path']!=prior['path'] and hash_string(prior['sha256']),'standardized60 cannot become native input')
        require(native['scope']==('accepted_native60_interval_from_full_original' if row['source_group']=='Mureka_v9' else
            'accepted_native60_interval_from_archive_member_original' if row['source_group']=='human_saraga_hindustani_v1'
            else 'full_original_file'),'native origin scope')
        selected.append({**row,'original_native_path':native['path'],'execution_native_path':native['path'],
            'local_origin_copy':False,'native_evidence':native,
            'origin_plan_row_sha256':base.value_hash(row),'screen_row_sha256':base.value_hash(screened),
            'component_sha256':base.value_hash(component)})
    counts=Counter(r['source_group'] for r in selected)
    labels=Counter(r['label'] for r in selected)
    require(len(selected)==expected['prior'] and len(all_rows)-len(selected)==expected['new'],'2174 / 1695 boundary')
    require(dict(counts)==expected['sources'] and labels=={'0':expected['human'],'1':expected['ai']},'prior source/label roster')
    require(sum(r['native_region']['float64_sha256'] is None for r in selected)==expected['ordinary']
            and sum(r['native_region']['float64_sha256'] is not None for r in selected)==expected['mandatory_hash'],'1571 / 603 historical evidence split')
    require(plan['prior60_source_counts']==expected['sources'],'plan source accounting')
    return selected


def prepare(plan_path,review_path,screen_path,output):
    code=Path(__file__).absolute()
    paths={'plan':plan_path,'parent_metadata_review':review_path,'screen':screen_path,
        'materializer':code,'tests':code.with_name('test_materialize_native30_prior2174_v1.py'),
        'shared_materialization_helper':code.with_name('materialize_native30_new1695_v1.py'),
        'shared_helper_tests':code.with_name('test_materialize_native30_new1695_v1.py'),
        'native_dsp':code.with_name('standardize_native30_v1.py'),
        'native_dsp_tests':code.with_name('test_standardize_native30_v1.py'),
        'region_dsp':code.with_name('standardize_native30_frozen_region_v1.py'),
        'region_dsp_tests':code.with_name('test_standardize_native30_frozen_region_v1.py')}
    bindings={key:base.file_binding(path) for key,path in paths.items()}
    for key,digest in [('plan',PLAN_SHA),('parent_metadata_review',REVIEW_SHA),('screen',SCREEN_SHA),
        ('shared_materialization_helper',HELPER_SHA),('native_dsp',DSP_SHA),('region_dsp',REGION_DSP_SHA),('region_dsp_tests',REGION_TEST_SHA)]:
        require(bindings[key]['sha256']==digest,'fixed input SHA mismatch: '+key)
    plan,review,screen=[base.read_json(path) for path in (plan_path,review_path,screen_path)]
    selected=_reconcile(plan,review,screen)
    native_dsp=pinned_module('standardize_native30_v1',DSP_SHA)
    region_dsp=pinned_module('standardize_native30_frozen_region_v1',REGION_DSP_SHA)
    require(region_dsp.dsp is native_dsp,'region uses the pinned native resampler')
    runtime=base.runtime_binding(native_dsp)
    require((runtime['numpy'],runtime['scipy'],runtime['soundfile'])==('1.26.4','1.17.1','0.14.0'),'frozen numerical runtime')
    root=Path(output).absolute()
    require(root.parent.is_dir() and '..' not in root.parts and not root.is_symlink(),'safe output parent/root')
    require(all(not Path(r['execution_native_path']).is_relative_to(root) for r in selected),'output overlaps native sources')
    require(all(not Path(path).absolute().is_relative_to(root) for path in paths.values()),'output contains bound metadata/code')
    base.recheck(bindings)
    contract={'version':VERSION,'status':'frozen_before_any_audio_processing',
        'scope':'prior2174_frozen_native60_to_native30_DSP_only','expected_count':EXPECTED['prior'],
        'output_root':str(root),'rows':selected,'configuration':region_dsp.CONFIG,'runtime':runtime,'bindings':bindings,
        'source_counts':SOURCES,'human':EXPECTED['human'],'ai':EXPECTED['ai'],
        'ordinary_null_historical_region_hashes':1571,'mandatory_historical_region_hashes':603,
        'source_provenance_bindings_from_accepted_plan':plan['input_bindings'],
        'source_provenance_scope':'accepted plan and parent metadata review are rehashed; historical upstream receipts are bound by that plan, not re-decoded here',
        'row_order':'id_ascending','classifier_fits':0,'cohort_admitted':False,'feature_extraction_authorized':False,
        'new1695_rows_processed':0,'full_stream_completeness_claim':False,'independent_decoder_agreement_claim':False,
        'resume_policy':'verify_contract_row_source_receipt_output_bytes; unreceipted_audio_fails'}
    return contract,region_dsp


def verify_region_audit(audit,row,configuration):
    native,region=row['native_evidence'],row['native_region']
    rate,start,frames=native['sample_rate_hz'],region['start_frame'],region['frames']
    crop_start=start+(frames-30*rate)//2
    expected={'region_start_frame':start,'region_frames':frames,'region_end_frame_exclusive':start+frames,
        'crop_start_frame':crop_start,'crop_frames':30*rate,'crop_end_frame_exclusive':crop_start+30*rate,
        'crop_offset_within_native_region':crop_start-start}
    require(audit['status']=='verified_bounded_native60_to30_DSP_only' and audit['region_kind']=='frozen_native60_interval'
            and audit['configuration']==configuration and audit['coordinates']==expected,'bounded DSP audit coordinates/configuration')
    require(audit['source_path']==row['execution_native_path'] and audit['source_sha256_before']==native['sha256']
            and audit['source_sha256_after']==native['sha256'] and audit['native_rate_hz']==rate and audit['native_channels']==2,
            'bounded DSP audit native source')
    require(audit['source_byte_hash_scope']=='whole_file_bytes_not_full_waveform_decode'
            and audit['full_stream_completeness_established'] is False
            and audit['independent_decoder_agreement_established'] is False,'no unsupported full-stream/decoder claims')
    require(audit['evidence_strength']==region['evidence_strength'] and
            audit['historical_expected_region_float64_sha256']==region['float64_sha256'],'historical evidence cannot change')
    special=region['float64_sha256'] is not None
    require(audit['historical_region_waveform_hash_available'] is special
            and audit['historical_region_hash_match'] is (True if special else None),'historical null/match semantics')
    observed=audit['observed_current_region_float64_sha256']
    require(hash_string(observed) and (not special or observed==region['float64_sha256']),'current native60 hash')
    passes=audit['sequential_passes']
    require(len(passes)==2 and passes[0]==passes[1] and audit['two_sequential_observations_equal'] is True,'two bounded observations')
    for observed_pass in passes:
        require(observed_pass['decoded_through_frame_exclusive']==start+frames
                and observed_pass['prefix_frames_before_region']==start
                and observed_pass['retained_native_region_frames']==frames
                and observed_pass['observed_region_float64_sha256']==observed
                and observed_pass['observation']=='bounded_sequential_prefix_through_region_end'
                and observed_pass['empty_eof_probe_performed'] is False
                and observed_pass['full_stream_completeness_established'] is False,'bounded prefix observation accounting')
    require(audit['decoded_through_frame_exclusive']==start+frames and audit['output_frames']==1323000
            and audit['output_rate_hz']==44100 and audit['output_channels']==2 and audit['sample_count']==2646000,
            'native30/output accounting')
    require(hash_string(audit['native_crop_float64_sha256']) and hash_string(audit['output_waveform_float32_sha256']),
            'required waveform hashes')
    require(audit['classifier_admitted'] is False and audit['cohort_admitted'] is False
            and audit['feature_extraction_authorized'] is False,'DSP-only audit scope')


def verify_receipt(output,row,contract_sha,configuration):
    envelope=base._verify_receipt(output,row,contract_sha)
    receipt=envelope['payload']
    require(all(receipt[key]==row[key] for key in ('id','source_group','group_id','component_id','label','role')),
            'receipt outer identity/label/role differs from frozen row')
    verify_region_audit(receipt['audit'],row,configuration)
    require(receipt['waveform_bit_exact_roundtrip'] is True
            and receipt['waveform_float32_sha256']==receipt['audit']['output_waveform_float32_sha256'],'receipt roundtrip/hash agreement')
    return envelope


def one(output,row,contract_sha,region_dsp,configuration):
    ident=row['id'];receipt=output/'items'/(ident+'.json');wav=output/'audio'/(ident+'.wav')
    if receipt.exists() or receipt.is_symlink():
        return verify_receipt(output,row,contract_sha,configuration)
    require(not wav.exists() and not wav.is_symlink(),'unreceipted waveform retained; explicit review required')
    require(not list((output/'audio').glob('.'+ident+'.wav.*.tmp')),'unreceipted temporary waveform retained')
    native,region=row['native_evidence'],row['native_region']
    values,audit=region_dsp.standardize(row['execution_native_path'],native['sha256'],native['sample_rate_hz'],native['channels'],
        region_start_frame=region['start_frame'],region_frames=region['frames'],evidence_strength=region['evidence_strength'],
        expected_region_float64_sha256=region['float64_sha256'])
    verify_region_audit(audit,row,configuration)
    base.publish_waveform(region_dsp.dsp,wav,values)
    payload={'status':'materialized_DSP_only','contract_sha256':contract_sha,
        **{key:row[key] for key in ('id','source_group','group_id','component_id','label','role')},
        'row_sha256':base.value_hash(row),'row':row,'standardized_path':str(wav),
        'file_sha256':base.digest(wav),'file_bytes':wav.stat().st_size,
        'waveform_float32_sha256':audit['output_waveform_float32_sha256'],'waveform_bit_exact_roundtrip':True,'audit':audit}
    envelope={'payload':payload,'receipt_sha256':base.value_hash(payload)}
    base.write_new(receipt,envelope)
    return envelope


def run_contract(contract,region_dsp,workers=4):
    """Small contracts are reachable by synthetic tests only, not CLI options."""
    require(type(workers) is int and 1<=workers<=16,'workers must be in 1..16')
    rows=contract['rows'];output=Path(contract['output_root']);sha=base.value_hash(contract)
    require(len(rows)==contract['expected_count'] and len({r['id'] for r in rows})==len(rows)
            and rows==sorted(rows,key=lambda r:r['id']),'fixed deterministic contract rows')
    with base.writer_lock(output):
        base.check_contract_files(contract)
        contract_path=output/'contract.json'
        if contract_path.exists():
            require(not contract_path.is_symlink() and base.read_json(contract_path)==contract
                    and base.digest(contract_path)==sha,'resume contract conflict')
        else:
            require({p.name for p in output.iterdir()}=={'writer.lock'},'nonempty output without contract')
            base.write_new(contract_path,contract)
        for name in ('items','audio','failures','runs'):
            path=output/name;path.mkdir(exist_ok=True)
            require(path.is_dir() and not path.is_symlink(),'safe output subdirectory')
        base.validate_inventory(output,rows)
        commit_path=output/'COMMIT.json';committed=commit_path.exists()
        if committed:
            commit=base.read_json(commit_path)
            require(commit['status']=='committed_prior2174_DSP_only' and commit['completed']==len(rows)
                    and commit['contract_sha256']==sha and commit['products']==base._products(output),'existing COMMIT inventory')
        run_id=uuid.uuid4().hex
        def process(row):
            try:
                envelope=verify_receipt(output,row,sha,contract['configuration']) if committed else one(
                    output,row,sha,region_dsp,contract['configuration'])
                return {'id':row['id'],'ok':True,'receipt':envelope}
            except Exception as exc:
                failure={'id':row['id'],'contract_sha256':sha,'row_sha256':base.value_hash(row),
                    'exception':type(exc).__name__,'message':str(exc),'run_id':run_id}
                if not committed:
                    base.write_new(output/'failures'/(row['id']+'.'+run_id+'.json'),failure)
                return {'id':row['id'],'ok':False,'failure':failure}
        collected={}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending=[executor.submit(process,row) for row in rows]
            for count,future in enumerate(as_completed(pending),1):
                result=future.result();collected[result['id']]=result
                if count%25==0 or count==len(rows):
                    passed=sum(r['ok'] for r in collected.values())
                    print(json.dumps({'event':'prior2174_materialization_progress','reviewed':count,'planned':len(rows),
                        'passed_or_resumed':passed,'failed':count-passed},sort_keys=True),file=sys.stderr,flush=True)
        ordered=[collected[row['id']] for row in rows]
        base.check_contract_files(contract)
        good=[r for r in ordered if r['ok']];failures=[r['failure'] for r in ordered if not r['ok']]
        summary={'status':'partial_no_COMMIT' if failures else 'complete_DSP_only','contract_sha256':sha,
            'expected':len(rows),'completed':len(good),'failed':len(failures),'failures':failures,'run_id':run_id,
            'classifier_fits':0,'cohort_admitted':False,'feature_extraction_authorized':False}
        if committed:
            require(not failures,'committed materialization source/receipt verification failed')
            return {'status':'verified_existing_COMMIT','completed':len(good),'commit_sha256':base.digest(commit_path)}
        base.write_new(output/'runs'/(run_id+'.json'),summary)
        if failures:
            return summary
        base.validate_inventory(output,rows,complete=True)
        manifest={'version':VERSION,'status':'all_prior2174_DSP_materialized_not_cohort_admitted',
            'contract_sha256':sha,'count':len(rows),'records':[r['receipt']['payload'] for r in good],
            'source_counts':dict(Counter(r['source_group'] for r in rows)),
            'ordinary_null_historical_region_hashes':sum(r['native_region']['float64_sha256'] is None for r in rows),
            'mandatory_historical_region_hashes':sum(r['native_region']['float64_sha256'] is not None for r in rows),
            'classifier_fits':0,'cohort_admitted':False,'feature_extraction_authorized':False,
            'full_stream_completeness_claim':False,'new1695_rows_processed':0}
        manifest_path=output/'manifest.json'
        if manifest_path.exists():
            require(not manifest_path.is_symlink() and base.read_json(manifest_path)==manifest,'existing manifest conflict')
        else:
            base.write_new(manifest_path,manifest)
        base.check_contract_files(contract)
        base.validate_inventory(output,rows,complete=True)
        base.write_new(commit_path,{'status':'committed_prior2174_DSP_only','contract_sha256':sha,
            'completed':len(rows),'products':base._products(output)})
        return {**summary,'commit_sha256':base.digest(commit_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','parent-review','screen','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--mode',choices=('preflight','run'),default='preflight')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    contract,region_dsp=prepare(args.plan,args.parent_review,args.screen,args.output)
    if args.mode=='preflight':
        print(json.dumps({'status':'metadata_preflight_only_no_audio_reads_or_writes','selected':len(contract['rows']),
            'contract_sha256':base.value_hash(contract),'output_root':contract['output_root'],
            'source_counts':contract['source_counts'],'human':contract['human'],'ai':contract['ai'],
            'ordinary_null_historical_region_hashes':1571,'mandatory_historical_region_hashes':603},sort_keys=True))
        return 0
    result=run_contract(contract,region_dsp,args.workers)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 1 if result['status']=='partial_no_COMMIT' else 0


if __name__=='__main__':
    raise SystemExit(main())
