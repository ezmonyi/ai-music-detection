#!/usr/bin/env python3
"""Score-only Saraga103 frozen transfer. Draft never freezes or predicts.

Uses unchanged stored-model audit/replay helpers, explicit Human identities,
completed measurement proofs, and NFS-safe exclusive hardlink COMMIT publication.
No fitting, neural execution, waveform decoding, feature/threshold selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import score_mureka60_frozen_v4 as M

A, P = M.A, M.P
require, digest, read_json = M.require, M.digest, M.read_json
Bindings, rows, seal, replay, write_json = M.Bindings, M.rows, M.seal, M.replay, M.write_json
CAPS, COMBINATIONS, INDEX, DESCRIPTORS = M.CAPS, M.COMBINATIONS, M.INDEX, M.DESCRIPTORS
ROLE = 'external_human_unscored'
SOURCE = 'human_saraga_hindustani_v1'
STAGE = 'saraga103_frozen_v4_transfer_scoring_only'
META = ['id', 'label', 'source_group', 'group_id', 'role', 'evaluation_allowed', 'classifier_admission_authorized']
RD = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
PREPARED = RD/'saraga_external103_intervals_v1'
FHM_LAUNCH = '4a7271aa05f117005b8dcf1d743cb56d658d3f387d856ff6e724ca57f4a2c7be'
FHM_ACCEPTANCE = '4b9aabb4e251a807e32861a74aa1dfe5da1341b4aefad33e4c085b1420382f27'
FHM_CSV = 'dfd9317235b49756ab2e883c583ba1a21e96ffeb8d54f28982bd647cc9fa5679'
INTERVAL_COMMIT = '8cf0c7bc743441f699a74ea7d9095634ab08f81004e8a7e11349046fc2e3e820'
OLD4_FREEZE = '6911d0f4c02debddc922c02d6b08145e608b6fffad5c4d224f0074e0aa04e544'
OLD4_CONTRACT = '10d40d1bd89c5dfba8a2a7b80ef065c6ceebfeeb7ef8aaed11341dc02868f92b'
ID_SHA = 'a0df862441fb0cf9f214d739a6fca55dc54626457fdd410810aeb1690d35321a'
GROUP_COUNTS = [72, 14, 10, 5, 2]
OVERVIEW = ['S','D','R','P','F','H','M','S+D+R+P','S+D+R+P+F','S+D+R+P+H','S+D+R+P+M','S+D+R+P+F+H+M']
PROTOCOL = Path(__file__).resolve().parent.parent/'SARAGA103_FROZEN_TRANSFER_PROTOCOL_EN.md'


def check_identity(metadata, features, synthetic=False):
    require(list(metadata) == META and list(features) == ['id', *DESCRIPTORS], 'Predictor/metadata schema changed')
    require(metadata.id.tolist() == features.id.tolist() and not metadata.id.duplicated().any()
            and metadata.id.str.strip().ne('').all(), 'Missing/duplicate/reordered measurement rows')
    require(set(metadata.role) == {ROLE} and set(metadata.source_group) == {SOURCE}
            and set(metadata.label) == {'0'} and set(metadata.evaluation_allowed) == {'False'}
            and set(metadata.classifier_admission_authorized) == {'False'}
            and metadata.group_id.str.strip().ne('').all(), 'Human role/label/source/admission changed')
    if synthetic:
        require(0 < len(metadata) <= 16 and metadata.id.str.startswith('synthetic_saraga_v4_').all()
                and metadata.group_id.str.startswith('synthetic_').all(), 'Synthetic fixture identity/size invalid')
    else:
        require(len(metadata) == 103 and metadata.id.str.fullmatch(r'saraga_hindustani_[0-9a-f-]{36}').all(),
                'Real mode requires exactly103 frozen Saraga IDs')
        require(hashlib.sha256('\n'.join(sorted(i.removeprefix('saraga_hindustani_') for i in metadata.id)).encode()).hexdigest() == ID_SHA,
                'Frozen Saraga identity set changed')
        require(sorted(Counter(metadata.group_id).values(), reverse=True) == GROUP_COUNTS, 'Five fixed component sizes changed')
    converted = features.set_index('id').copy()
    for col in DESCRIPTORS:
        values = converted[col]
        missing = values.str.strip().str.lower().isin(['', 'nan', 'na', 'null'])
        numeric = pd.to_numeric(values.mask(missing), errors='raise')
        require(np.isfinite(numeric[~missing].to_numpy(float)).all(), 'Nonfinite descriptor value')
        converted[col] = numeric
    return converted


def validate_model(model, combination):
    M.validate_model(model, combination)
    require(all('saraga' not in name.lower() for name in model['training_source_counts']), 'Saraga training leakage')


def validate_index(index, models):
    result = M.validate_index(index, models)
    for combo, key in result[['combination','model_sha256']].drop_duplicates().itertuples(index=False, name=None):
        validate_model(models[key], combo)
    return result


def load_old(a, b):
    models, index, oldmeta = M.load_old(a, b)
    return models, validate_index(index, models), oldmeta


def measurement_scope(value):
    require(value.get('purpose') == 'measurement_only' and value.get('role') == ROLE
            and value.get('classifier_fitted') is False and value.get('scores_generated') is False
            and value.get('classifier_admission_authorized') is False
            and value.get('evaluation_allowed') is False, 'Measurement role/admission changed')


def sealed_measurement(path, b):
    value = b.json(path)
    seal(value)
    measurement_scope(value)
    return value


def verify_commit(root, b, expected=None):
    b.file(root/'COMMIT.json', expected)
    marker = b.json(root/'COMMIT.json')
    require(marker['status'] == 'committed' and bool(marker['files_sha256']), 'Uncommitted interval')
    for name, sha in marker['files_sha256'].items():
        require(Path(name).name == name and name not in ('.','..','COMMIT.json'), 'Unsafe committed name')
        b.file(root/name, sha)
    return marker


def validate_wav_proof(proof):
    """Accepted Saraga materializer schema, distinct from old standardizers."""
    require(proof['wav']['float32_sha256'] == proof['interval_proof']['standardized_float32_sha256']
            and proof['wav']['readback_verified'] is True, 'Standardized PCM proof changed')


def validate_alias(row,original,root):
    alias=Path(row['standardized_path'])
    require(alias.parent == root/'inputs' and alias.name == row['item_id']+'.wav'
            and not alias.is_symlink() and alias.samefile(original['standardized_path'])
            and row['standardized_file_sha256'] == original['standardized_file_sha256']
            and row['accepted_standardized_path'] == original['standardized_path'], 'Alias interval changed')


def validate_intervals(a, b, launch):
    require(a.prepared_dir == PREPARED, 'Only fixed accepted Saraga103 intervals allowed')
    verify_commit(a.prepared_dir/'final', b, INTERVAL_COMMIT)
    native = rows(a.prepared_dir/'final/native_metadata_60s.csv', 'id', b)
    manifest = rows(a.prepared_dir/'final/inference_manifest.csv', 'item_id', b)
    summary = b.json(a.prepared_dir/'final/materialization_summary.json')
    require(summary['passed'] == summary['expected'] == 103 and summary['failed'] == 0
            and summary['final_passed_source_and_output_hash_recheck'] is True
            and summary['synthetic_test_only'] is False and summary['classifier_admission'] is False,
            'Incomplete physical interval cohort')
    ids = native.id.tolist()
    require(manifest.item_id.tolist() == ids == launch['selected_ids'], 'Native/standardized order mismatch')
    inventory = b.json(a.prepared_dir/'final/item_proof_inventory.json')
    require(set(inventory) == set(ids), 'Item proof inventory mismatch')
    for n, standard in zip(native.to_dict('records'), manifest.to_dict('records')):
        item = a.prepared_dir/'items'/n['id']
        marker = verify_commit(item, b)
        proof = b.json(item/'proof.json')
        q = proof['interval_proof']
        require(marker['files_sha256']['proof.json'] == inventory[n['id']]
                and proof['item_id'] == n['id'] and proof['role'] == ROLE and proof['classifier_admission'] is False
                and proof['status'] == 'passed_interval_materialization_not_classifier_admission'
                and proof['contract_sha256'] == summary['contract_sha256'], 'Item proof identity/contract changed')
        sr = int(n['native_sample_rate_hz'])
        start, count = int(n['crop_start_frame']), int(n['crop_frames'])
        require(sr in (44100,48000) and count == sr*60 and start == (int(n['sf_actual_read_frames'])-count)//2
                and start == q['crop_start_frame'] and start+count == int(n['crop_end_frame_exclusive']) == q['crop_end_frame_exclusive']
                and q['observed_actual_frames'] == int(n['sf_actual_read_frames'])
                and q['observed_header']['header_frames'] == int(n['sf_header_frames'])
                and q['native_sample_rate_hz'] == sr and q['native_channels'] == 2
                and q['real_empty_read_observed'] is True and q['seek_proof']['exact_bytes_equal'] is True,
                'Native center60/EOF/seek proof changed')
        require(n['audio_path'] == q['source_audio_path']
                and n['source_audio_sha256'] == n['registered_raw_sha256'] == q['raw_hashes_after']['sha256']
                and n['native_crop_float64_sha256'] == q['native_float64_sha256'], 'Native raw/PCM hash distinction changed')
        b.file(n['audio_path'], n['source_audio_sha256'])
        require(standard['item_id'] == n['id'] and standard['group_id'] == n['group_id']
                and standard['label'] == n['label'] == '0' and standard['source_id'] == n['source_id'] == SOURCE
                and standard['role'] == n['role'] == ROLE
                and standard['standardized_path'] == str(item/'audio.wav')
                and standard['standardized_file_sha256'] == proof['wav']['file_sha256'],
                'Standardized old4 input must bind the same native interval')
        validate_wav_proof(proof)
        b.file(standard['standardized_path'], standard['standardized_file_sha256'])
    require(Counter(native.native_sample_rate_hz) == {'44100':96,'48000':7}, 'Native rate inventory changed')
    return native, manifest


def validate_old4(a, b, native, manifest):
    root = a.old4_dir.parent
    require(a.old4_dir == RD/'saraga_external103_old4_v1/measurements', 'Only isolated accepted Saraga old4 namespace allowed')
    audit = sealed_measurement(a.old4_dir/'strict_inference_audit.json', b)
    receipt = sealed_measurement(a.old4_dir/'extraction_receipt.json', b)
    ledger = sealed_measurement(a.old4_dir/'measurement_roles.json', b)
    run = sealed_measurement(root/'run_contract.json', b)
    completion = sealed_measurement(root/'completion.json', b)
    require(audit['status'] == receipt['status'] == completion['status'] == 'passed'
            and audit['rows'] == receipt['rows'] == completion['rows'] == run['rows'] == 103
            and audit['all10_stage_receipts_verified'] is True and audit['exact_product_and_log_union_verified'] is True
            and audit['all_input_and_stem_samples_decoded'] is True
            and audit['same_process_cuda_guard_required'] is True
            and receipt['before_after_all_provenance_inputs_products_dependencies_verified'] is True,
            'Full103 strict guarded measurement acceptance required')
    b.file(a.old4_dir/'strict_inference_audit.json', receipt['strict_audit_sha256'])
    b.file(root/'run_contract.json', audit['run_contract_sha256'])
    b.file(root/'completion.json', audit['completion_sha256'])
    require(completion['run_contract_sha256'] == audit['run_contract_sha256'] and completion['completed_stage_shards'] == 10
            and run['item_ids'] == native.id.tolist() and run['source_id'] == SOURCE and run['label'] == 0
            and run['output_root'] == str(root) and run['prepared_dir'] == str(a.prepared_dir), 'Inference identity mismatch')
    frozen = b.json(a.old4_frozen)
    b.file(a.old4_frozen, OLD4_FREEZE)
    b.file(a.old4_frozen, audit['authorization_file_sha256'])
    require(frozen['status'] == 'frozen' and frozen['authorized_stage'] == 'saraga_external103_old4_measurement_only'
            and frozen['contract'] == run and frozen['contract_sha256'] == run['canonical_sha256'] == OLD4_CONTRACT
            and frozen['independent_review']['approved'] is True, 'Old4 freeze changed')
    review = frozen['independent_review']
    require(not Path(review['evidence']).is_absolute() and '..' not in Path(review['evidence']).parts, 'Unsafe review path')
    b.file(a.root/review['evidence'], review['evidence_sha256'])
    for key in ('dependencies_sha256','preparation_sha256','runtime_sha256'):
        require(audit[key] == run[key], 'Strict audit/run provenance mismatch')
        b.hashes(run[key])
    require(receipt['dependencies_sha256'] == run['dependencies_sha256'] == completion['dependencies_sha256'], 'Stale extraction dependencies')
    b.records(audit['inputs']); b.records(audit['products']); b.records(audit['cuda_guard_receipts'])
    b.records({str(root/'inputs/COMMIT.json'):audit['alias_commit'], str(root/'inputs/inference_manifest.csv'):audit['adapted_manifest']})
    adapted = rows(root/'inputs/inference_manifest.csv', 'item_id', b)
    require(adapted.item_id.tolist() == native.id.tolist(), 'Alias manifest IDs changed')
    require({p.name for p in (root/'inputs').iterdir()} == {i+'.wav' for i in native.id}|{'inference_manifest.csv','COMMIT.json'},
            'Unexpected/missing alias inventory')
    verify_commit(root/'inputs',b)
    expected_inputs = {}
    for row, original in zip(adapted.to_dict('records'), manifest.to_dict('records')):
        # Receipt paths are authoritative immutable aliases; require their exact
        # path, hash, and samefile linkage to the accepted standardized WAV.
        validate_alias(row,original,root)
        expected_inputs[row['standardized_path']] = {'sha256':row['standardized_file_sha256'], 'bytes':int(row['standardized_file_bytes'])}
    require(audit['inputs'] == expected_inputs, 'Audited standardized input coverage mismatch')
    paths = {str(root/'receipts'/f'{stage}_{i:02d}.json') for stage in ('allinone','beats') for i in range(5)}
    require(set(completion['receipts_sha256']) == paths, 'All10 stage receipts required')
    require({str(p) for p in (root/'receipts').iterdir()} == paths, 'Unexpected stage receipt inventory')
    b.hashes(completion['receipts_sha256'])
    products = {}
    guards = {}
    for path in sorted(paths):
        item = sealed_measurement(path, b)
        stage, shard = Path(path).stem.rsplit('_',1)
        i = int(shard)
        part = adapted.iloc[i*24:(i+1)*24]
        gpu = run[stage+'_gpus'][i % len(run[stage+'_gpus'])]
        original_command = [v for token in run['original_commands'][stage]
                            for v in (part.standardized_path.tolist() if token == '<frozen shard inputs>' else [token])]
        expected_products = {str(root/'beats'/(iid+'.beats')) for iid in part.item_id} if stage == 'beats' else {
            str(p) for iid in part.item_id for p in [*(root/'demix/htdemucs'/iid/(s+'.wav') for s in ('bass','drums','other','vocals')),
                root/'structure'/(iid+'.json'),root/'spec'/(iid+'.npy')]}
        require(item['status'] == 'passed' and item['exit_code'] == 0 and item['stage'] == stage
                and item['shard_index'] == i and item['item_ids'] == part.item_id.tolist()
                and item['gpu'] == gpu and item['run_contract_sha256'] == audit['run_contract_sha256']
                and item['dependencies_sha256'] == run['dependencies_sha256']
                and item['command'] == run['commands'][stage][str(i)]
                and item['original_command'] == original_command
                and set(item['outputs']) == expected_products
                and item['inputs'] == {p:expected_inputs[p] for p in part.standardized_path}, 'Stage identity/command/input mismatch')
        b.records(item['outputs']); b.records({item['log_path']:item['log']})
        require(not set(products).intersection(item['outputs']), 'Duplicate product coverage')
        products.update(item['outputs'])
        gp = root/'logs'/f'{stage}_{i:02d}.cuda_guard.json'
        b.records({str(gp):item['cuda_guard']}); guards[str(gp)] = item['cuda_guard']
        guard = b.json(gp)
        require(guard['status'] == 'passed_same_process_cuda_guard' and guard['stage'] == stage
                and guard['guard_sha256'] == run['cuda_guard_sha256'] == audit['cuda_guard_sha256']
                and guard['guard_version'] == run['cuda_guard_version'] == audit['cuda_guard_version']
                and guard['original_command'] == item['original_command']
                and guard['original_entrypoint_sha256'] == run['runtime_sha256'][item['original_command'][0]]
                and guard['before'] == guard['after'] and guard['before']['cuda_available'] is True
                and guard['before']['physical_gpu_index'] == gpu and guard['before']['allocation_device'] == 'cuda:0'
                and guard['before']['cuda_visible_devices'] == str(gpu) and guard['before']['visible_device_count'] == 1
                and guard['before']['logical_device_index'] == 0 and guard['before']['cuda_initialized'] is True
                and 'RTX 5090' in guard['before']['device_name'] and guard['before']['device_total_memory_bytes'] > 0
                and guard['before']['allocation_and_synchronization_passed'] is True
                and guard['entrypoint_dispatched_same_process'] is True and guard['original_argv_preserved'] is True
                and guard['entrypoint_returned_successfully'] is True and guard['cpu_fallback_authorized'] is False
                and guard['model_or_torch_monkeypatched'] is False, 'CUDA guard proof changed')
        require(guard['process']['pid'] > 0 and bool(guard['process']['hostname'])
                and b.file(guard['process']['python_executable']) == b.file((Path(run['runtime_root'])/'venv/bin/python').resolve()),
                'Guard inference interpreter evidence changed')
    require(products == audit['products'] and guards == audit['cuda_guard_receipts'], 'Product/guard union changed')
    actual_products={str(p) for name in ('demix','structure','spec','beats') for p in (root/name).rglob('*') if p.is_file()}
    expected_logs={str(root/'logs'/f'{stage}_{i:02d}{suffix}') for stage in ('allinone','beats') for i in range(5)
                   for suffix in ('.log','.cuda_guard.json')}
    require(actual_products == set(products) and {str(p) for p in (root/'logs').iterdir()} == expected_logs,
            'Unreceipted/missing neural products or logs')
    b.records(receipt['features'])
    expected = {str(a.old4_dir/'features'/n) for n in ('expanded_features_60s.csv','expanded_features_60s.jsonl','expanded_features_60s_metadata.json')}
    require(set(receipt['features']) == expected, 'Feature inventory changed')
    require({str(p) for p in (a.old4_dir/'features').rglob('*') if p.is_file()} == expected, 'Unexpected feature artifacts')
    b.records({str(a.old4_dir/'extraction.log'):receipt['log'], str(a.old4_dir/'measurement_roles.json'):receipt['role_ledger']})
    require(ledger['items'] == [dict(item_id=n['id'],role=ROLE,label=0,source_id=SOURCE,group_id=n['group_id'],
            evaluation_allowed=False,classifier_admission_authorized=False) for n in native.to_dict('records')], 'Human role ledger changed')
    frame = rows(a.old4_dir/'features/expanded_features_60s.csv','item_id',b)
    meta = b.json(a.old4_dir/'features/expanded_features_60s_metadata.json')
    payload = meta['run_payload']
    require(meta['rows'] == meta['complete'] == 103 and meta['partial'] == 0 and meta['duration_sec'] == 60
            and meta['run_fingerprint'] == digest(payload)
            and payload['extractor_sha256'] == P.OLD_CODE['extract_expanded_four_family.py']
            and payload['core_sha256'] == P.OLD_CODE['expanded_feature_definitions.py']
            and payload['bias_sha256'] == P.BIAS_SHA, 'Old4 numerical contract changed')
    for family, prefix, key in [('S','s8__','s8_features'),('D','d__','d_features'),('R','r__','r_features'),('P','p__','p_features')]:
        require([prefix+n for n in meta[key]] == P.OLD_COLUMNS[family], 'Old descriptor schema changed')
    jsonl = [json.loads(s) for s in (a.old4_dir/'features/expanded_features_60s.jsonl').read_text().splitlines()]
    require(frame.item_id.tolist() == native.id.tolist() == [p['item_id'] for p in jsonl], 'Old4 row order changed')
    for n, row, item, aliases in zip(native.to_dict('records'), frame.to_dict('records'), jsonl, adapted.to_dict('records')):
        require(row['label'] == '0' and row['source_id'] == SOURCE and row['group_id'] == n['group_id']
                and row['status'] == 'complete' and row['run_fingerprint'] == meta['run_fingerprint'], 'Old4 identity/fingerprint changed')
        hashes = {'source_audio_sha256':aliases['standardized_file_sha256'],
            **{stem+'_sha256':products[str(root/'demix/htdemucs'/n['id']/(stem+'.wav'))]['sha256'] for stem in ('bass','drums','other','vocals')},
            'beats_sha256':products[str(root/'beats'/(n['id']+'.beats'))]['sha256'],
            'structure_sha256':products[str(root/'structure'/(n['id']+'.json'))]['sha256']}
        require(json.loads(row['input_hashes']) == hashes, 'Old4 standardized/stem/event hash binding changed')
        for key, value in item.items():
            require(json.loads(row[key]) == value if isinstance(value,(dict,list)) else str(value) == row[key], 'CSV/JSONL mismatch: '+key)
    return frame


def validate_measurements(a, b):
    acceptance = b.json(a.fhm_dir/'acceptance.json'); b.file(a.fhm_dir/'acceptance.json', FHM_ACCEPTANCE)
    frozen = b.json(a.fhm_dir/'launch_contract.json')
    launch = frozen['contract']
    require(frozen['status'] == 'frozen' and frozen['authorized_stage'] == 'saraga103_native60_fhm_measurement_only'
            and frozen['contract_sha256'] == digest(launch) == FHM_LAUNCH, 'Accepted FHM frozen launch changed')
    require(acceptance['status'] == 'passed_measurement_not_classifier_admission' and acceptance['rows'] == 103
            and acceptance['role'] == ROLE and acceptance['classifier_fitted'] is False and acceptance['scores_generated'] is False
            and acceptance['all_descriptor_values_checked'] is True and acceptance['original_sources_rehashed_after_extraction'] is True,
            'Complete FHM acceptance required')
    b.file(a.fhm_dir/'launch_contract.json', acceptance['launch_contract_sha256'])
    b.file(a.fhm_dir/'extraction.log', acceptance['extraction_log_sha256'])
    b.hashes(launch['input_sha256']); b.hashes(acceptance['outputs_sha256'])
    native, manifest = validate_intervals(a,b,launch)
    meta = native[META].copy()
    reference = b.json(a.fhm_reference); seal(reference,'contract_hash')
    require(reference['contract_hash'] == P.FHM_CONTRACT, 'Historical FHM reference changed')
    froot = a.fhm_dir/'features'
    contract = b.json(froot/'contract.json'); seal(contract,'contract_hash')
    for key in ('duration','preflight_only','input_config','F_config','feature_names','code_sha256','runtime'):
        require(contract[key] == reference[key], 'FHM numerical parity changed: '+key)
    require(contract['metadata_sha256'] == b.file(a.prepared_dir/'final/native_metadata_60s.csv')
            and contract['selected_ids'] == native.id.tolist() and contract['feature_names'] == P.NEW_COLUMNS, 'FHM input/order changed')
    b.file(froot/'features.csv',FHM_CSV)
    fhm = rows(froot/'features.csv','id',b)
    require(fhm.id.tolist() == native.id.tolist() and fhm.extraction_status.eq('ok').all(), 'Incomplete FHM rows')
    itempaths = {str(froot/'items'/(hashlib.sha256(i.encode()).hexdigest()+'.json')) for i in native.id}
    require(set(acceptance['outputs_sha256']) == itempaths | {str(froot/n) for n in ('contract.json','features.csv','summary.json','process.json')}, 'FHM inventory changed')
    for n, record in zip(native.to_dict('records'),fhm.to_dict('records')):
        item = b.json(froot/'items'/(hashlib.sha256(n['id'].encode()).hexdigest()+'.json'))
        require(item['input_row_hash'] == digest(n) and item['extraction_contract_hash'] == contract['contract_hash']
                and item['source_audio_sha256'] == n['source_audio_sha256'] and item['source_audio_path'] == n['audio_path']
                and item['crop_start_frame'] == int(n['crop_start_frame']) and item['crop_frames'] == int(n['crop_frames'])
                and item['source_total_frames'] == int(n['sf_header_frames'])
                and item['analysis_frames'] == 960000 and item['analysis_sr'] == 16000, 'FHM native interval mismatch')
        for key in ('id','label','source_group','source_id','role','group_id'):
            require(item[key] == record[key] == n[key], 'FHM identity mismatch')
        for key in [*sum(P.NEW_COLUMNS.values(),[]),'F_status','H_status','M_status','extraction_status','source_audio_sha256','analysis_waveform_sha256']:
            require(record.get(key,'') == ('' if item.get(key) is None else str(item[key])), 'FHM CSV/item mismatch')
        require(all(item[f+'_status'] == 'ok' for f in ('F','H','M'))
                and all(np.isfinite(float(item[c])) for c in sum(P.NEW_COLUMNS.values(),[])), 'Accepted27 FHM finite descriptors changed')
    summary, process = b.json(froot/'summary.json'), b.json(froot/'process.json')
    require(summary['expected'] == summary['recorded'] == 103 and summary['complete_accounting'] is True
            and summary['features_csv_sha256'] == FHM_CSV and summary['contract_hash'] == contract['contract_hash']
            and process['state'] == 'finished' and process['complete_accounting'] is True, 'FHM completion changed')
    old = validate_old4(a,b,native,manifest)
    raw = pd.DataFrame({'id':native.id})
    for column in DESCRIPTORS:
        raw[column] = (old if column in sum(P.OLD_COLUMNS.values(),[]) else fhm)[column]
    return meta, raw


def load_synthetic(root,b):
    require(b.json(root/'synthetic_fixture.json') == {'synthetic_test_only':True,'purpose':'small_handwritten_saraga_replay_fixture_no_training'}, 'Explicit Saraga synthetic marker required')
    models = b.json(root/'fold_models.json')
    index = validate_index(rows(root/'model_index.csv','fixture_row',b).drop(columns='fixture_row'),models)
    return models,index,rows(root/'metadata.csv','id',b),rows(root/'features.csv','id',b)


def assemble(a):
    b = Bindings()
    if a.synthetic_test_only:
        require(a.synthetic_fixture is not None,'Synthetic fixture directory required')
        models,index,meta,raw = load_synthetic(a.synthetic_fixture,b)
    else:
        require(a.synthetic_fixture is None,'Synthetic fixture refused in real mode')
        for name in ('root','old_results','old_package','old_preregistration','old_audit','prepared_dir','old4_dir','old4_frozen','fhm_dir','fhm_reference'):
            require(getattr(a,name) is not None,'Real mode missing required input: '+name)
        models,index,oldmeta = load_old(a,b)
        meta,raw = validate_measurements(a,b)
        require(not set(meta.id)&set(oldmeta.id) and not set(meta.group_id)&set(oldmeta.group_id),'Old/new ID/group leakage')
    features = check_identity(meta,raw,a.synthetic_test_only)
    for path in (Path(__file__).resolve(),Path(M.__file__).resolve(),Path(A.__file__).resolve(),Path(P.__file__).resolve(),
                 Path(sys.executable).resolve(),Path(np.__file__).resolve(),Path(pd.__file__).resolve(),PROTOCOL):
        b.file(path)
    contract = dict(schema_version=1,authorized_stage=STAGE,synthetic_test_only=a.synthetic_test_only,
        measurement_interface='synthetic_fixture' if a.synthetic_test_only else 'saraga103_old4_guard_v1_native_fhm_v1',
        old_development_admission=False,refitting=False,model_selection=False,threshold_tuning=False,
        families=P.COLUMNS,combinations=list(COMBINATIONS),caps=list(CAPS),fold_models=5,
        feature_mode='values_plus_missing',threshold=.5,positive_rule='score >= 0.5',prediction_link='identity_unclipped',
        rows=len(meta),model_instances=3175,prediction_rows=3175*len(meta),summary_rows=3175,
        component_sizes=dict(sorted(Counter(meta.group_id).items())),overview_combinations=OVERVIEW,
        identity_sha256=digest(meta.to_dict('records')),model_index_sha256=digest(index.to_dict('records')),
        input_files_sha256=dict(sorted(b.files.items())),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),
        endpoints=['fp','tn','false_positive_rate','specificity','equal_component_false_positive_rate','equal_component_specificity'],
        forbidden_endpoints=['balanced_accuracy','roc_auc','two_class_accuracy','source_transfer_J','rank','winner'],
        comparison_tolerance=dict(score_atol=1e-12,score_rtol=1e-12,threshold_decisions='exact',integer_summaries='exact'),
        uncertainty='five-model min/max are descriptive ranges, not confidence intervals; same103 dependent observations',
        component_weighting='equal mean of five connected components; distinct from primary per-recording weights',
        measurement_role=ROLE,scoring_role='external_human_frozen_v4_transfer',publication='exclusive directory reservation and hardlink COMMIT last; no overwrite or resume')
    require(a.synthetic_test_only or all(x == '1' for x in contract['runtime']['threads'].values()), 'Real scoring requires all three thread settings=1')
    b.recheck()
    return contract,models,index,meta,features,b


def authorize(contract,path):
    require(path is not None,'Scoring requires independently frozen receipt')
    value = read_json(path)
    require(value.get('status') == 'frozen' and value.get('authorized_stage') == STAGE and value.get('contract') == contract
            and value.get('contract_sha256') == digest(contract), 'Receipt draft/stale/wrong mode or stage')
    review = value.get('independent_review')
    require(isinstance(review,dict) and review.get('approved') is True and review.get('reviewer') == 'root'
            and isinstance(review.get('reviewed_utc'),str) and review['reviewed_utc'].strip(), 'Independent root review required')
    return value


def write_scores(root,contract,models,index,meta,features):
    fields = INDEX+META+['scoring_role','synthetic_test_only','score','threshold','predicted_label']
    summaries,components = [],[]
    identities = meta.to_dict('records')
    masks = {g:meta.group_id.eq(g).to_numpy() for g in sorted(meta.group_id.unique())}
    with (root/'predictions.csv').open('x',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fields); writer.writeheader()
        for common in index.to_dict('records'):
            scores = replay(features,models[common['model_sha256']]); positive = scores >= .5
            for identity,score,pred in zip(identities,scores,positive):
                writer.writerow(dict(common,**identity,scoring_role=contract['scoring_role'],synthetic_test_only=contract['synthetic_test_only'],
                    score=format(float(score),'.17g'),threshold=.5,predicted_label=int(pred)))
            group_rates = []
            for group,mask in masks.items():
                n,fp = int(mask.sum()),int(positive[mask].sum())
                group_rates.append(fp/n)
                components.append(dict(common,synthetic_test_only=contract['synthetic_test_only'],group_id=group,rows=n,
                    fp=fp,tn=n-fp,false_positive_rate=fp/n,specificity=(n-fp)/n))
            n,fp = len(meta),int(positive.sum())
            summaries.append(dict(common,synthetic_test_only=contract['synthetic_test_only'],rows=n,unique_ids=meta.id.nunique(),
                unique_groups=len(masks),fp=fp,tn=n-fp,threshold=.5,false_positive_rate=fp/n,specificity=(n-fp)/n,
                equal_component_false_positive_rate=float(np.mean(group_rates)),equal_component_specificity=float(np.mean([1-r for r in group_rates]))))
    summaries,components = pd.DataFrame(summaries),pd.DataFrame(components)
    summaries.to_csv(root/'per_model_specificity.csv',index=False)
    components.to_csv(root/'per_component_specificity.csv',index=False)
    records=[]
    for (combo,cap),part in summaries.groupby(['combination','quantity'],sort=True):
        row=dict(combination=combo,quantity=cap,fold_models=5,rows_per_model=len(meta),synthetic_test_only=contract['synthetic_test_only'])
        require(len(part)==5,'Missing five-model cell')
        for col in ('fp','tn','false_positive_rate','specificity','equal_component_false_positive_rate','equal_component_specificity'):
            row.update({col+'_mean':float(part[col].mean()),col+'_min':float(part[col].min()),col+'_max':float(part[col].max())})
        records.append(row)
    cells=pd.DataFrame(records)
    cells.to_csv(root/'all635_cells.csv',index=False)
    cells[cells.combination.isin(OVERVIEW)].to_csv(root/'predefined_overview.csv',index=False)
    write_json(root/'output_accounting.json',dict(prediction_rows=len(meta)*len(index),model_summary_rows=len(summaries),
        component_summary_rows=len(components),aggregate_cells=len(cells),integer_counts_consistent=bool((summaries.fp+summaries.tn).eq(len(meta)).all()),
        independent_numerical_audit_performed=False))


def publish_directory(source,destination):
    """NFS supports link(2) NOREPLACE. An orphan is never accepted or overwritten."""
    destination.mkdir()  # exclusive reservation; existing empty directories refused
    for path in sorted(source.iterdir()):
        require(path.is_file() and not path.is_symlink(),'Invalid publication product')
        with path.open('rb') as file:
            os.fsync(file.fileno())
        os.link(path,destination/path.name)
    files={p.name:{'sha256':P.sha(p),'bytes':p.stat().st_size} for p in sorted(destination.iterdir())}
    marker=source/'COMMIT.json'
    write_json(marker,dict(status='committed',files=files,publication='exclusive hardlinks, COMMIT last'))
    with marker.open('rb') as file:
        os.fsync(file.fileno())
    os.link(marker,destination/'COMMIT.json')
    verify_publication(destination)


def verify_publication(root):
    require(not root.is_symlink(),'Symlink publication')
    marker=read_json(root/'COMMIT.json')
    require(marker['status']=='committed' and set(p.name for p in root.iterdir())==set(marker['files'])|{'COMMIT.json'},'Orphan/unexpected publication')
    for name,record in marker['files'].items():
        require(Path(name).name==name and name not in ('.','..') and not (root/name).is_symlink()
                and P.sha(root/name)==record['sha256'] and (root/name).stat().st_size==record['bytes'],'Published product changed')
    return marker


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['draft','score'],required=True)
    for name in ('root','old-results','old-package','old-preregistration','old-audit','prepared-dir','old4-dir','old4-frozen','fhm-dir','fhm-reference','synthetic-fixture','receipt'):
        parser.add_argument('--'+name,type=lambda p:Path(p).absolute())
    parser.add_argument('--synthetic-test-only',action='store_true')
    parser.add_argument('--output-dir',type=lambda p:Path(p).absolute(),required=True)
    a=parser.parse_args(argv)
    require(not a.output_dir.exists() and not a.output_dir.is_symlink(),'Refusing existing output path')
    contract,models,index,meta,features,b=assemble(a)
    if a.stage=='score':
        authorize(contract,a.receipt); b.file(a.receipt)
    a.output_dir.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=a.output_dir.name+'.tmp.',dir=a.output_dir.parent))
    try:
        index.to_csv(temporary/'model_index.csv',index=False); meta.to_csv(temporary/'identity_roles.csv',index=False)
        if a.stage=='draft':
            write_json(temporary/'scoring_draft.json',dict(status='draft',authorized_stage=STAGE,contract=contract,
                contract_sha256=digest(contract),independent_review=None,
                freeze_instruction='Root reviews all real proofs and writes NEW independently frozen receipt; this program cannot freeze.'))
        else:
            write_scores(temporary,contract,models,index,meta,features)
            write_json(temporary/'scoring_receipt.json',read_json(a.receipt))
        b.recheck()
        write_json(temporary/'publication_manifest.json',dict(status='draft' if a.stage=='draft' else 'scored',
            synthetic_test_only=a.synthetic_test_only,contract_sha256=digest(contract),classifier_fitted=False,
            original_v4_development_admission=False,unique_new_ids=len(meta),model_instances=3175,
            prediction_rows=0 if a.stage=='draft' else 3175*len(meta),
            files={p.name:{'sha256':P.sha(p),'bytes':p.stat().st_size} for p in sorted(temporary.iterdir())}))
        publish_directory(temporary,a.output_dir)
    finally:
        shutil.rmtree(temporary)


if __name__=='__main__':
    main()
