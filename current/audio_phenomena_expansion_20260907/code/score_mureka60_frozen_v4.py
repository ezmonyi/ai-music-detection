#!/usr/bin/env python3
"""Score-only Mureka transfer adapter. Drafts never score or freeze themselves.

No evaluator imports, fitting, model selection, media decoding or inference.
Real admission requires the existing independent v4 auditor and complete v2
measurement receipts. Synthetic fixtures have a deliberately separate entry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import ctypes
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import audit_equal60_v4_results as A
import prepare_evaluation_inputs_v4 as P

ROLE = 'external_generator_unscored'
STAGE = 'mureka500_frozen_v4_transfer_scoring_only'
CAPS = ('25', '50', '100', '200', 'all')
COMBINATIONS = tuple('+'.join(c) for n in range(1, 8) for c in itertools.combinations(P.COLUMNS, n))
MODEL_KEYS = {'columns', 'medians', 'mean', 'scale', 'coefficients_with_intercept', 'ridge',
              'threshold', 'feature_mode', 'training_rows', 'training_source_counts', 'weighting',
              'missing_value_policy', 'observed_fraction_by_column', 'model_type', 'prediction_link'}
META = ['id', 'label', 'source_group', 'group_id', 'role', 'acquisition_role']
DESCRIPTORS = sum(P.COLUMNS.values(), [])
INDEX = ['combination', 'quantity', 'fold_index', 'fold_uid', 'model_sha256',
         'candidate_key', 'train_id_set_sha256', 'test_id_set_sha256']
require, digest, read_json = A.require, A.digest, A.strict_json


class Bindings:
    """Every admitted local file is rehashed before atomic publication."""
    def __init__(self):
        self.files = {}

    def file(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.resolve() == path, f'Missing/noncanonical regular file: {path}')
        # Large source/stem files recur in multiple receipts. Bind them once per
        # snapshot; recheck() independently rereads all bytes before publication.
        value = self.files[str(path)] if str(path) in self.files else P.sha(path)
        require(expected is None or value == expected, f'Stale/tampered file: {path}')
        require(str(path) not in self.files or self.files[str(path)] == value, f'File changed: {path}')
        self.files[str(path)] = value
        return value

    def json(self, path):
        self.file(path)
        return read_json(path)

    def hashes(self, values, base=None):
        require(isinstance(values, dict) and bool(values), 'Empty provenance hash map')
        for path, value in values.items():
            self.file((Path(base) / path) if base is not None else path, value)

    def records(self, values):
        require(isinstance(values, dict) and bool(values), 'Empty product inventory')
        for path, value in values.items():
            require(set(value) == {'sha256', 'bytes'}, 'Invalid file record')
            self.file(path, value['sha256'])
            require(Path(path).stat().st_size == value['bytes'], 'File size changed')

    def recheck(self):
        for path, value in self.files.items():
            require(P.sha(path) == value and not Path(path).is_symlink(), f'Bound input changed: {path}')


def seal(value, key='canonical_sha256'):
    require(value.get(key) == digest({k: v for k, v in value.items() if k != key}), 'Invalid canonical seal')


def measurement_scope(value):
    require(value.get('purpose') == 'measurement_only' and value.get('role') == ROLE
            and value.get('classifier_fitted') is False and value.get('scores_generated') is False
            and value.get('classifier_admission_authorized') is False, 'Measurement role/admission changed')


def rows(path, key, bindings):
    bindings.file(path)
    frame = A.read_csv(path)
    require(key in frame and len(frame) > 0 and frame[key].str.strip().ne('').all()
            and not frame[key].duplicated().any(), f'Duplicate/blank/missing IDs: {path}')
    return frame


def check_identity(metadata, features, synthetic=False):
    require(list(metadata) == META and list(features) == ['id', *DESCRIPTORS], 'Predictor/metadata schema changed')
    require(metadata.id.tolist() == features.id.tolist() and not metadata.id.duplicated().any()
            and not features.id.duplicated().any(), 'Missing/duplicate/reordered measurement rows')
    require(set(metadata.role) == {ROLE} and set(metadata.acquisition_role) == {'reserved_unscored'}
            and set(metadata.source_group) == {'Mureka_v9'} and set(metadata.label) == {'1'}
            and metadata.group_id.str.strip().ne('').all(), 'Role/label/source leakage')
    if synthetic:
        require(0 < len(metadata) <= 16 and metadata.id.str.startswith('synthetic_mureka_v4_').all()
                and metadata.group_id.str.startswith('synthetic_').all(), 'Synthetic fixture identity/size invalid')
    else:
        require(len(metadata) == 500 and metadata.id.str.fullmatch(r'music8k_mureka_v9_\d+').all(),
                'Real mode requires exactly500 acquired Mureka IDs; synthetic data refused')
    converted = features.set_index('id').copy()
    for col in DESCRIPTORS:
        values = converted[col]
        missing = values.str.strip().str.lower().isin(['', 'nan', 'na', 'null'])
        numeric = pd.to_numeric(values.mask(missing), errors='raise')
        require(np.isfinite(numeric[~missing].to_numpy(float)).all(), 'Nonfinite descriptor value')
        converted[col] = numeric
    return converted


def validate_model(model, combination):
    cols = [c for f in combination.split('+') for c in P.COLUMNS[f]]
    require(set(model) == MODEL_KEYS and model['columns'] == cols, 'Frozen model schema/descriptor mismatch')
    for key, value in dict(ridge=10.0, threshold=0.5, feature_mode='values_plus_missing',
                           weighting=A.WEIGHTING, missing_value_policy=A.POLICIES['values_plus_missing'],
                           model_type='weighted_ridge_linear_probability', prediction_link='identity').items():
        require(model[key] == value, 'Frozen model restriction: ' + key)
    require(type(model['training_rows']) is int and model['training_rows'] > 0
            and all(not ('Mureka' in s or 'synthetic_mureka' in s) for s in model['training_source_counts'])
            and sum(model['training_source_counts'].values()) == model['training_rows'], 'Training source/count leakage')
    for key, length in [('medians', len(cols)), ('mean', 2*len(cols)), ('scale', 2*len(cols)),
                        ('coefficients_with_intercept', 2*len(cols)+1)]:
        value = np.asarray(model[key], float)
        require(value.shape == (length,) and np.isfinite(value).all(), 'Invalid frozen transform: ' + key)
    require((np.asarray(model['scale']) > 0).all(), 'Invalid frozen scale')
    observed = model['observed_fraction_by_column']
    require(set(observed) == set(cols) and all(math.isfinite(v) and 0 <= v <= 1 for v in observed.values()),
            'Invalid frozen observed fractions')


def validate_index(index, models):
    require(list(index) == INDEX, 'Model-index schema changed')
    keys = ['combination', 'quantity', 'fold_index']
    expected = set(itertools.product(COMBINATIONS, CAPS, map(str, range(5))))
    require(len(index) == 3175 and not index.duplicated(keys).any()
            and set(index[keys].itertuples(index=False, name=None)) == expected, 'Exact127x5x5 model coverage required')
    require(index.groupby(['quantity', 'fold_index']).fold_uid.nunique().eq(1).all()
            and index[['quantity', 'fold_index', 'fold_uid']].drop_duplicates().fold_uid.nunique() == 25,
            'Fold/cap identity changed')
    for combo, key in index[['combination', 'model_sha256']].drop_duplicates().itertuples(index=False, name=None):
        require(key in models and key == digest({'model': models[key], 'threshold': 0.5}), 'Model hash mismatch')
        validate_model(models[key], combo)
    return index.sort_values(['combination', 'quantity', 'fold_index']).reset_index(drop=True)


def replay(features, model):
    """Identity-link raw score; no clipping, fitting, medians or scaling learned here."""
    x = features[model['columns']].to_numpy(float)
    missing = ~np.isfinite(x)
    z = np.concatenate([np.where(missing, model['medians'], x), missing.astype(float)], axis=1)
    z = (z - model['mean']) / model['scale']
    beta = np.asarray(model['coefficients_with_intercept'])
    scores = np.column_stack((np.ones(len(z)), z)) @ beta
    require(np.isfinite(scores).all(), 'Nonfinite replay output')
    return scores


def load_old(a, bindings):
    report = bindings.json(a.old_audit)
    require(report.get('status') == 'passed' and report.get('synthetic_test_only') is False
            and report.get('rows') == 1604 and report.get('candidates') == 127
            and report.get('pooled_metrics') == 3175 and report.get('caps') == [25, 50, 100, 200, 'all']
            and report.get('model_fitting_performed') is False, 'Real independently passed1604 v4 audit required')
    require(report.get('result_manifest_sha256') == bindings.file(a.old_results/'run_manifest.json')
            and report.get('auditor_sha256') == bindings.file(Path(A.__file__).resolve()), 'Stale independent audit')
    # The auditor verifies frozen receipt, all original data/code hashes, train-only
    # transforms, ridge stationarity and the complete predictions; it never fits.
    require(A.audit(a.old_results, a.old_package, a.old_preregistration, synthetic=False) == report,
            'Fresh independent audit differs from supplied passed report')
    manifest = bindings.json(a.old_results/'run_manifest.json')
    bindings.hashes(manifest['files_sha256'], a.old_results)
    bindings.file(a.old_preregistration)
    proof = bindings.json(a.old_package/'preparation_audit.json')
    bindings.hashes(proof['files_sha256'], a.old_package)
    for name, value in manifest['contract']['code_sha256'].items():
        bindings.file(Path(__file__).with_name(name).resolve(), value)
    models = bindings.json(a.old_results/'fold_models.json')
    pooled = A.read_csv(a.old_results/'development_group_cv_pooled_fold_metrics.csv')
    index = validate_index(pooled[INDEX], models)
    oldmeta = A.read_csv(a.old_package/'metadata_60s.csv')
    return models, index, oldmeta


def validate_release_proof(proof, b, old_ids):
    """Rebuild the exact v3 resource-release identity proof, without GPU calls."""
    seal(proof)
    fixed = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/equal60_development_v2')
    inference = fixed/'inference'
    require(proof.get('schema_version') == 1 and proof.get('status') == 'verified_full134_original1604'
            and proof.get('rows') == 1604 and proof.get('stage_shards') == 134
            and proof.get('fixed_prepared_root') == str(fixed) and proof.get('fixed_inference_root') == str(inference)
            and proof.get('prepared_manifest_sha256') == P.PREPARED_SHA
            and proof.get('materialization_contract_sha256') == P.MATERIALIZATION_CONTRACT
            and proof.get('original_runner_sha256') == 'a33f667ac9ef5cc5b4e5e646adfb752a68380b1af5f42b0768b1a10e3d031858',
            'Wrong fixed original1604 GPU release proof')
    expected_files = {str(inference/n) for n in ('completion.json','run_contract.json')}
    expected_files |= {str(fixed/n) for n in ('inference_manifest.csv','materialization_summary.json')}
    expected_files |= {str(fixed/f'inference_shard_{i:02d}.txt') for i in range(67)}
    expected_files |= {str(inference/'receipts'/f'{stage}_{i:02d}.json') for stage in ('allinone','beats') for i in range(67)}
    require(set(proof['proof_files']) == expected_files, 'Original release proof file inventory changed')
    b.records(proof['proof_files'])
    completion, run, summary = b.json(inference/'completion.json'), b.json(inference/'run_contract.json'), b.json(fixed/'materialization_summary.json')
    manifest = rows(fixed/'inference_manifest.csv', 'item_id', b)
    require(b.file(fixed/'inference_manifest.csv') == P.PREPARED_SHA
            and summary.get('contract_sha256') == P.MATERIALIZATION_CONTRACT
            and run.get('code_sha256') == 'a33f667ac9ef5cc5b4e5e646adfb752a68380b1af5f42b0768b1a10e3d031858',
            'Original release production manifest/materialization/code pins changed')
    require(len(manifest) == 1604 and set(manifest.item_id) == set(old_ids) and manifest.role.eq('development').all(),
            'Release proof is not the audited old1604 development cohort')
    require(completion.get('status') == 'passed' and completion.get('rows') == 1604
            and completion.get('completed_stage_shards') == 134
            and completion.get('run_contract_sha256') == proof['run_contract_sha256'] == b.file(inference/'run_contract.json')
            and proof['completion_sha256'] == b.file(inference/'completion.json')
            and proof['prepared_manifest_sha256'] == b.file(fixed/'inference_manifest.csv'), 'Stale original completion proof')
    require(run.get('rows') == 1604 and run.get('output_root') == str(inference)
            and run.get('manifest_sha256') == proof['prepared_manifest_sha256']
            and run.get('aio_gpus') == [0,1,2,3] and run.get('beat_gpus') == [4,5]
            and run.get('shards') == summary.get('shards') and len(summary['shards']) == 67, 'Original release run contract changed')
    assignment = {stage:{f'{i:02d}':gpus[i % len(gpus)] for i in range(67)}
                  for stage,gpus in [('allinone',[0,1,2,3]),('beats',[4,5])]}
    require(proof['original_stage_gpu_assignment'] == assignment, 'Original GPU assignment changed')
    identities, all_inputs = {}, []
    by_path = manifest.set_index('standardized_path')
    require(by_path.index.is_unique, 'Duplicate original standardized path')
    runtime = Path(run['runtime_root'])
    for i, shard in enumerate(summary['shards']):
        path = fixed/f'inference_shard_{i:02d}.txt'
        b.file(path, shard['sha256'])
        inputs = path.read_text().splitlines()
        require(set(shard) == {'index','rows','sha256'} and shard['index'] == i
                and shard['rows'] == len(inputs), 'Original shard identity changed')
        require(all(p in by_path.index for p in inputs), 'Unknown original shard input')
        all_inputs.extend(inputs)
        ids = by_path.loc[inputs].item_id.tolist()
        for stage in ('allinone','beats'):
            name = f'{stage}_{i:02d}.json'
            receipt = b.json(inference/'receipts'/name)
            command = ([str(runtime/'venv/bin/all-in-one-infer'), *inputs, '-o', str(inference/'structure'), '-m','harmonix-all','-d','cuda','-k',
                        '--demix-dir',str(inference/'demix'),'--spec-dir',str(inference/'spec')] if stage == 'allinone' else
                       [str(runtime/'venv/bin/beat_this'), *inputs, '-o',str(inference/'beats'),'--model',
                        str(runtime/'checkpoints/hub/checkpoints/beat_this-final0.ckpt'),'--no-dbn','--gpu','0','--float16','--skip-existing'])
            identity = dict(stage=stage, shard_index=i, item_ids=ids, gpu=assignment[stage][f'{i:02d}'], command=command)
            require(receipt.get('status') == 'passed' and receipt.get('run_contract_sha256') == proof['run_contract_sha256']
                    and all(receipt.get(k) == v for k,v in identity.items() if k != 'command')
                    and receipt.get('command',receipt.get('original_shard_command')) == command
                    and isinstance(receipt.get('outputs'),dict) and receipt['outputs'], 'Original release stage receipt changed')
            identities[name] = identity
    require(all_inputs == manifest.standardized_path.tolist() and identities == proof['receipt_identities'], 'Original release cohort/receipt proof mismatch')


def validate_measurement_version(version, run, audit, receipt, b, old_ids):
    require(version in ('v2','v3'), 'Unknown measurement version')
    require(run.get('schema_version') == (1 if version == 'v2' else 3), 'Explicit measurement version/schema mismatch')
    gpus = [run['allinone_gpus'], run['beats_gpus']]
    require(all(isinstance(g,list) and g and len(g)==len(set(g)) and all(type(i) is int and i>=0 for i in g) for g in gpus)
            and not set(gpus[0]) & set(gpus[1]), 'Invalid/disjoint GPU assignments')
    assignment = {stage:{f'{i:02d}':g[i % len(g)] for i in range(21)} for stage,g in zip(('allinone','beats'),gpus)}
    if version == 'v2':
        require(not set(sum(gpus,[])) & set(range(6)) and 'original_gpu_release_proof' not in run
                and 'original_gpu_release_proof' not in audit and 'stage_gpu_assignment' not in run,
                'v3 release metadata cannot enter v2 route')
    else:
        require('original_gpu_release_proof' in run and 'original_gpu_release_proof' in audit
                and run['original_gpu_release_proof'] == audit['original_gpu_release_proof']
                and audit.get('explicit_stage_gpu_assignment_verified') is True
                and run.get('stage_gpu_assignment') == assignment, 'v3 release/assignment audit changed')
        proof = run['original_gpu_release_proof']
        require(proof is not None or not set(sum(gpus,[])) & set(range(6)), 'v3 reserved GPUs lack release proof')
        if proof is not None:
            validate_release_proof(proof,b,old_ids)
    flag = ('before_after_all_provenance_inputs_products_dependencies_verified' if version == 'v2'
            else 'before_after_all_provenance_inputs_products_dependencies_release_proof_verified')
    require(receipt.get(flag) is True, 'Measurement extraction acceptance/version flag changed')
    return assignment


def validate_measurements(a, b, oldmeta):
    """Read acceptance proofs and rehash their whole input/product/code graph.

    Physical decoding was completed by the strict auditor; this adapter checks
    that precisely those decoded bytes and all measured values are unchanged.
    """
    prepared, old, fhm = a.prepared_dir, a.prepared_dir/'measurements', a.fhm_dir
    version = a.measurement_version
    inference = prepared/('inference' if version == 'v2' else 'inference_v3')
    require((inference/'completion.json').is_file() and (old/'extraction_receipt.json').is_file(),
            'Complete explicitly selected '+version+' inference/measurements required')
    native = rows(prepared/'native_metadata_60s.csv', 'id', b)
    manifest = rows(prepared/'inference_manifest.csv', 'item_id', b)
    ids = native.id.tolist()
    require(len(ids) == 500 and ids == manifest.item_id.tolist(), 'Complete500 measurement identity required')
    metadata = native[META].copy()
    require(native.classifier_admission_authorized.eq('False').all(), 'Native development admission forbidden')
    mc, ms = b.json(prepared/'materialization_contract.json'), b.json(prepared/'materialization_summary.json')
    seal(mc, 'contract_sha256')
    require(mc['status'] == 'frozen_before_materialization' and mc['purpose'] == 'measurement_only'
            and mc['role'] == ROLE and mc['classifier_admission_authorized'] is False
            and mc['acquisition_role'] == 'reserved_unscored' and mc['source_repo'] == 'homura23/MUSIC8K'
            and mc['source_revision'] == '05232438ba76a7bc55cbebfdc6d5f4011c980bba'
            and mc['acquisition_contract_sha256'] == 'e985adc956116851a7358f3ed9279241a6f92b65d28f22b0eb40e94023de8d24'
            and mc['output_dir'] == str(prepared), 'Materialization contract changed')
    require(ms['status'] == 'verified' and ms['rows'] == 500 and ms['role'] == ROLE
            and ms['classifier_admission_authorized'] is False and ms['contract_sha256'] == mc['contract_sha256'],
            'Incomplete materialization')
    b.file(prepared/'native_metadata_60s.csv', ms['native_metadata_sha256'])
    b.file(prepared/'inference_manifest.csv', ms['manifest_sha256'])
    b.file(Path(__file__).with_name('prepare_mureka60_inputs_v2.py').resolve(), mc['code_sha256'])
    b.hashes(mc['input_file_sha256'], mc['source_root'])
    b.hashes(mc['decoder_evidence_sha256'])
    require(mc['input_file_sha256'].get('contract.json') == mc['acquisition_contract_sha256'], 'Acquisition contract binding missing')
    acquisition = b.json(Path(mc['source_root'])/'contract.json')
    require(acquisition['status'] == 'frozen_for_acquisition_only' and acquisition['classifier_authorized'] is False
            and acquisition['selected_count'] == 500 and acquisition['revision'] == mc['source_revision']
            and [r['id'] for r in acquisition['selected']] == mc['selected_ids'], 'Acquisition selection/role mismatch')
    config = mc['configuration']
    require(mc['configuration_sha256'] == digest(config) and mc['runtime_sha256'] == digest(mc['runtime']), 'Materialization runtime/config seal changed')
    for key, value in dict(sample_rate_hz=44100, channels=2, crop_frames=2646000, duration_sec=60,
                           resampling=False, padding=False, gain=False, dc_removal=False, limiting=False,
                           output_format='WAV', output_subtype='FLOAT', role=ROLE,
                           classifier_admission_authorized=False, selection_features_used=False).items():
        require(config[key] == value, 'Materialization representation changed: '+key)
    require(['music8k_mureka_v9_'+i for i in mc['selected_ids']] == ids
            and [r['id'] for r in mc['rows']] == mc['selected_ids'], 'Frozen acquisition selection changed')
    require(set(ms['receipts_sha256']) == {i+'.json' for i in ids}, 'Materialization receipt union changed')
    b.hashes(ms['receipts_sha256'], prepared/'items')
    for n, row, source in zip(native.to_dict('records'), manifest.to_dict('records'), mc['rows']):
        receipt = b.json(prepared/'items'/(n['id']+'.json'))
        require(row == {k: str(v) for k, v in receipt.items()}, 'Manifest/receipt mismatch')
        start = (int(source['native_frames']) - 2646000)//2
        require(receipt['status'] == 'verified' and receipt['role'] == ROLE and receipt['label'] == 1
                and receipt['source_id'] == 'Mureka_v9' and receipt['acquisition_role'] == 'reserved_unscored'
                and receipt['classifier_admission_authorized'] is False and receipt['group_id'] == n['group_id'] == source['reference_group_id']
                and receipt['input_row_sha256'] == digest(source) and receipt['contract_sha256'] == mc['contract_sha256'],
                'Measurement identity/role/provenance changed')
        require(receipt['crop_start_frame'] == int(n['crop_start_frame']) == start
                and receipt['crop_frames'] == int(n['crop_frames']) == 2646000
                and receipt['crop_end_frame_exclusive'] == int(n['crop_end_frame_exclusive']) == start+2646000
                and int(n['sf_actual_read_frames']) == receipt['sf_actual_read_frames'] == source['sf_actual_read_frames']
                and receipt['sf_actual_read_frames'] >= start+2646000
                and int(n['sf_header_frames']) == receipt['sf_header_frames'] == source['sf_header_frames'],
                'Native interval/actual EOF mismatch')
        for key in ('native_crop_float64_sha256', 'native_crop_float32_sha256'):
            require(n[key] == receipt[key] == source[key], 'Native crop sample hash mismatch')
        require(receipt['standardized_waveform_sha256'] == source['native_crop_float32_sha256']
                and receipt['standardized_frames'] == 2646000 and receipt['standardized_sr'] == 44100
                and receipt['standardized_channels'] == 2 and source['sequential_all_samples_finite'] is True
                and source['center_seek_equals_sequential_float64'] is True, 'Standardized interval mismatch')
        b.file(row['standardized_path'], row['standardized_file_sha256'])
        require(n['source_audio_sha256'] == row['source_audio_sha256'] == source['sha256'], 'Raw source hash mismatch')
        b.file(row['source_audio_path'], source['sha256'])

    audit, receipt, ledger = (b.json(old/name) for name in ('strict_inference_audit.json', 'extraction_receipt.json', 'measurement_roles.json'))
    for proof in (audit, receipt, ledger):
        seal(proof)
        measurement_scope(proof)
    require(audit['status'] == receipt['status'] == 'passed' and audit['rows'] == receipt['rows'] == 500
            and audit['all42_stage_receipts_verified'] is True and audit['exact_product_and_log_union_verified'] is True
            and audit['all_input_and_stem_samples_decoded'] is True,
            'Full strict physical/measurement acceptance required')
    require(ledger['items'] == [dict(item_id=n['id'], role=ROLE, label=1, source_id='Mureka_v9',
            group_id=n['group_id'], classifier_admission_authorized=False) for n in metadata.to_dict('records')], 'Role ledger changed')
    b.file(old/'strict_inference_audit.json', receipt['strict_audit_sha256'])
    b.records({str(old/'measurement_roles.json'): receipt['role_ledger'], str(old/'extraction.log'): receipt['log']})
    b.records(receipt['features'])
    expected_features = {str(old/'features'/name) for name in ('expanded_features_60s.csv', 'expanded_features_60s.jsonl', 'expanded_features_60s_metadata.json')}
    require(set(receipt['features']) == expected_features, 'Old feature inventory changed')
    for key in ('dependencies_sha256', 'preparation_sha256', 'runtime_sha256'):
        b.hashes(audit[key])
    b.hashes(receipt['dependencies_sha256'])
    code_names = ['run_mureka60_inference_v2.py', 'verify_extract_mureka60_v2.py', 'prepare_mureka60_inputs_v2.py',
                  'run_equal60_inference_batches.py', 'verify_extract_equal60.py', 'materialize_equal60_inputs_v2.py']
    if version == 'v3':
        code_names.extend(['run_mureka60_inference_v3.py','verify_extract_mureka60_v3.py'])
    require(set(audit['dependencies_sha256']) == {str(Path(__file__).with_name(n).resolve()) for n in code_names},
            'Exact version-specific measurement code inventory changed')
    for name in code_names:
        current = Path(__file__).with_name(name).resolve()
        require(audit['dependencies_sha256'].get(str(current)) == b.file(current), 'Stale measurement auditor code')
    b.records(audit['inputs'])
    b.records(audit['products'])
    b.file(inference/'run_contract.json', audit['run_contract_sha256'])
    b.file(inference/'completion.json', audit['completion_sha256'])
    completion = b.json(inference/'completion.json')
    run = b.json(inference/'run_contract.json')
    for value in (completion, run):
        seal(value)
        measurement_scope(value)
    assignment = validate_measurement_version(version,run,audit,receipt,b,oldmeta.id)
    require(completion['status'] == 'passed' and completion['rows'] == 500
            and completion['completed_stage_shards'] == 42 and run['item_ids'] == ids
            and run['rows'] == 500 and run['output_root'] == str(inference) and run['prepared_dir'] == str(prepared)
            and completion['run_contract_sha256'] == audit['run_contract_sha256']
            and completion['dependencies_sha256'] == run['dependencies_sha256'] == audit['dependencies_sha256'] == receipt['dependencies_sha256']
            and run['preparation_sha256'] == audit['preparation_sha256']
            and run['runtime_sha256'] == audit['runtime_sha256'], 'Incomplete/stale inference')
    expected_inputs = {r['standardized_path']: {'sha256':r['standardized_file_sha256'], 'bytes':int(r['standardized_file_bytes'])}
                       for r in manifest.to_dict('records')}
    require(audit['inputs'] == expected_inputs, 'Strict audited input inventory changed')
    paths = {str(inference/'receipts'/f'{stage}_{i:02d}.json') for stage in ('allinone', 'beats') for i in range(21)}
    require(set(completion['receipts_sha256']) == paths, 'Exact42 receipt inventory required')
    b.hashes(completion['receipts_sha256'])
    union = {}
    for path in sorted(paths):
        value = b.json(path)
        seal(value)
        measurement_scope(value)
        require(value['status'] == 'passed' and value['exit_code'] == 0
                and value['run_contract_sha256'] == audit['run_contract_sha256']
                and value['dependencies_sha256'] == run['dependencies_sha256'], 'Unpassed/stale stage receipt')
        stage, shard = Path(path).stem.rsplit('_', 1)
        part = manifest.iloc[int(shard)*24:(int(shard)+1)*24]
        require(value['stage'] == stage and value['shard_index'] == int(shard)
                and value['item_ids'] == part.item_id.tolist()
                and value['gpu'] == assignment[stage][shard]
                and value['inputs'] == {p:expected_inputs[p] for p in part.standardized_path}, 'Stage-shard ID/input coverage changed')
        b.records(value['inputs'])
        b.records(value['outputs'])
        b.records({value['log_path']: value['log']})
        require(not set(union) & set(value['outputs']), 'Duplicate stage products')
        union.update(value['outputs'])
    require(union == audit['products'], 'Strict product/receipt union changed')

    oldframe = rows(old/'features/expanded_features_60s.csv', 'item_id', b)
    om = b.json(old/'features/expanded_features_60s_metadata.json')
    require(oldframe.item_id.tolist() == ids and oldframe.status.eq('complete').all()
            and om['rows'] == 500 and om['duration_sec'] == 60 and om['sample_rate_hz'] == 44100,
            'Incomplete old measurements')
    require(digest(om['run_payload']) == om['run_fingerprint'], 'Old feature fingerprint mismatch')
    for key, value in (('extractor_sha256', P.OLD_CODE['extract_expanded_four_family.py']),
                       ('core_sha256', P.OLD_CODE['expanded_feature_definitions.py']), ('bias_sha256', P.BIAS_SHA)):
        require(om['run_payload'][key] == value, 'Old feature definition changed')
    for code, prefix, key in [('S','s8__','s8_features'), ('D','d__','d_features'), ('R','r__','r_features'), ('P','p__','p_features')]:
        require([prefix+n for n in om[key]] == P.OLD_COLUMNS[code], 'Old descriptor schema changed')
    for n, row, measured in zip(native.to_dict('records'), manifest.to_dict('records'), oldframe.to_dict('records')):
        require(measured['label'] == '1' and measured['source_id'] == 'Mureka_v9' and measured['group_id'] == n['group_id']
                and measured['run_fingerprint'] == om['run_fingerprint'] and measured['bias_sha256'] == P.BIAS_SHA
                and float(measured['duration_sec']) == 60, 'Old measurement identity/context changed')
        expected = {'source_audio_sha256': row['standardized_file_sha256'],
                    **{s+'_sha256': union[str(inference/'demix/htdemucs'/n['id']/(s+'.wav'))]['sha256'] for s in ('bass','drums','other','vocals')},
                    'beats_sha256': union[str(inference/'beats'/(n['id']+'.beats'))]['sha256'],
                    'structure_sha256': union[str(inference/'structure'/(n['id']+'.json'))]['sha256']}
        require(json.loads(measured['input_hashes']) == expected, 'Old extractor input provenance changed')

    accept, launch = b.json(fhm/'acceptance.json'), b.json(fhm/'launch_contract.json')
    require(accept['status'] == 'passed' and accept['rows'] == 500 and accept['role'] == ROLE
            and accept['classifier_fitted'] is False and accept['scores_generated'] is False
            and accept['all_descriptor_values_checked'] is True and accept['original_sources_rehashed_after_extraction'] is True,
            'Full500 FHM acceptance required')
    require(launch['purpose'] == 'measurement_only' and launch['role'] == ROLE and launch['rows'] == 500
            and launch['classifier_fitted'] is False and launch['scores_generated'] is False, 'FHM scope changed')
    b.file(fhm/'launch_contract.json', accept['launch_contract_sha256'])
    b.file(fhm/'extraction.log', accept['extraction_log_sha256'])
    b.hashes(launch['input_sha256'])
    b.hashes(accept['outputs_sha256'])
    froot = fhm/'features'
    reference = b.json(a.fhm_reference)
    seal(reference, 'contract_hash')
    require(reference['contract_hash'] == P.FHM_CONTRACT, 'Frozen historical FHM reference changed')
    contract = b.json(froot/'contract.json')
    seal(contract, 'contract_hash')
    for key in ('duration', 'preflight_only', 'input_config', 'F_config', 'feature_names', 'code_sha256', 'runtime'):
        require(contract[key] == reference[key], 'FHM numerical/runtime parity changed: '+key)
    require(contract['feature_names'] == P.NEW_COLUMNS and contract['selected_ids'] == ids
            and contract['metadata_sha256'] == b.file(prepared/'native_metadata_60s.csv'), 'FHM contract input changed')
    for name, value in contract['code_sha256'].items():
        b.file(Path(__file__).with_name(name).resolve(), value)
    frame = rows(froot/'features.csv', 'id', b)
    require(frame.id.tolist() == ids and frame.extraction_status.eq('ok').all(), 'Incomplete FHM rows')
    itempaths = {str(froot/'items'/(hashlib.sha256(i.encode()).hexdigest()+'.json')) for i in ids}
    require(set(accept['outputs_sha256']) == itempaths | {str(froot/n) for n in ('contract.json','features.csv','summary.json','process.json')},
            'FHM acceptance inventory changed')
    for n, measured in zip(native.to_dict('records'), frame.to_dict('records')):
        item = b.json(froot/'items'/(hashlib.sha256(n['id'].encode()).hexdigest()+'.json'))
        require(item['input_row_hash'] == digest(n) and item['extraction_contract_hash'] == contract['contract_hash']
                and item['extraction_status'] == 'ok' and item['source_audio_sha256'] == n['source_audio_sha256']
                and item['source_audio_path'] == n['audio_path'] and item['role'] == ROLE and item['group_id'] == n['group_id']
                and item['crop_start_frame'] == int(n['crop_start_frame']) and item['crop_frames'] == 2646000
                and item['source_total_frames'] == int(n['sf_header_frames'])
                and item['analysis_frames'] == 960000 and item['analysis_sr'] == 16000, 'FHM interval/row binding changed')
        for key in [*sum(P.NEW_COLUMNS.values(), []), 'F_status','H_status','M_status','extraction_status','source_audio_sha256','analysis_waveform_sha256']:
            require(measured.get(key, '') == ('' if item.get(key) is None else str(item[key])), 'FHM descriptor CSV/item mismatch')
    summary, process = b.json(froot/'summary.json'), b.json(froot/'process.json')
    require(summary['expected'] == summary['recorded'] == 500 and summary['complete_accounting'] is True
            and process['state'] == 'finished' and process['complete_accounting'] is True
            and summary['contract_hash'] == contract['contract_hash']
            and summary['features_csv_sha256'] == b.file(froot/'features.csv'), 'FHM completion changed')
    require(summary['status_counts'] == dict(Counter(frame.extraction_status))
            and summary['family_status_counts'] == {f+'_status': dict(Counter(frame[f+'_status'])) for f in ('F','H','M')}, 'FHM status counts changed')
    features = pd.DataFrame({'id': ids})
    for col in DESCRIPTORS:
        features[col] = (oldframe if col in sum(P.OLD_COLUMNS.values(), []) else frame)[col]
    return metadata, features


def load_synthetic(root, b):
    marker = b.json(root/'synthetic_fixture.json')
    require(marker == {'synthetic_test_only': True, 'purpose': 'small_handwritten_replay_fixture_no_training'},
            'Explicit synthetic-only fixture marker required')
    models = b.json(root/'fold_models.json')
    index = validate_index(rows(root/'model_index.csv', 'fixture_row', b).drop(columns=['fixture_row']), models)
    metadata = rows(root/'metadata.csv', 'id', b)
    features = rows(root/'features.csv', 'id', b)
    return models, index, metadata, features


def assemble(a):
    b = Bindings()
    synthetic = a.synthetic_test_only
    if synthetic:
        require(a.synthetic_fixture is not None, 'Synthetic fixture directory required')
        models, index, meta, raw = load_synthetic(a.synthetic_fixture, b)
    else:
        require(a.synthetic_fixture is None, 'Synthetic fixture refused in real mode')
        for name in ('old_results','old_package','old_preregistration','old_audit','prepared_dir','fhm_dir','fhm_reference'):
            require(getattr(a, name) is not None, 'Real mode missing required input: '+name)
        models, index, oldmeta = load_old(a, b)
        meta, raw = validate_measurements(a, b, oldmeta)
        require(not set(meta.id) & set(oldmeta.id) and not set(meta.group_id) & set(oldmeta.group_id), 'Old/new ID or group leakage')
    features = check_identity(meta, raw, synthetic)
    for path in (Path(__file__).resolve(), Path(A.__file__).resolve(), Path(P.__file__).resolve(),
                 Path(sys.executable).resolve(), Path(np.__file__).resolve(), Path(pd.__file__).resolve()):
        b.file(path)
    contract = dict(schema_version=1, authorized_stage=STAGE, synthetic_test_only=synthetic,
        measurement_interface='synthetic_fixture' if synthetic else 'mureka_'+a.measurement_version,
        old_development_admission=False, refitting=False, model_selection=False, threshold_tuning=False,
        families=P.COLUMNS, combinations=list(COMBINATIONS), caps=list(CAPS), fold_models=5,
        feature_mode='values_plus_missing', threshold=0.5, positive_rule='score >= 0.5', prediction_link='identity_unclipped',
        rows=len(meta), model_instances=3175, prediction_rows=3175*len(meta), summary_rows=3175,
        identity_sha256=digest(meta.to_dict('records')), model_index_sha256=digest(index.to_dict('records')),
        input_files_sha256=dict(sorted(b.files.items())),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),
        endpoints=['ai_sensitivity', 'false_negative_rate'], forbidden_endpoints=['balanced_accuracy','roc_auc','source_transfer_J'],
        uncertainty='none; repeated model predictions are dependent observations of the same songs',
        acquisition_role='reserved_unscored', measurement_role=ROLE, scoring_role='external_generator_frozen_v4_transfer')
    require(synthetic or all(v == '1' for v in contract['runtime']['threads'].values()), 'Real scoring requires all three thread settings=1')
    b.recheck()
    return contract, models, index, meta, features, b


def authorize(contract, path):
    require(path is not None, 'Scoring requires an independently reviewed frozen receipt; draft cannot score')
    receipt = read_json(path)
    require(receipt.get('status') == 'frozen' and receipt.get('authorized_stage') == STAGE
            and receipt.get('contract') == contract and receipt.get('contract_sha256') == digest(contract),
            'Receipt is draft, stale, synthetic/real mismatched, or not independently frozen for this contract')
    review = receipt.get('independent_review')
    require(isinstance(review, dict) and review.get('approved') is True and isinstance(review.get('reviewer'), str) and review['reviewer'].strip()
            and isinstance(review.get('reviewed_utc'), str) and review['reviewed_utc'].strip(), 'Independent root review record required')
    return receipt


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def publish_directory(source, destination):
    """Atomic no-replace rename on the supported Linux/macOS workstations."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = fn(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith('linux'):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = fn(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise ValueError('Atomic no-replace directory publication requires Linux or macOS')
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def write_scores(root, contract, models, index, meta, features):
    pred_fields = INDEX + META + ['scoring_role', 'synthetic_test_only', 'score', 'threshold', 'predicted_label']
    summary_fields = INDEX + ['synthetic_test_only','rows','unique_ids','unique_groups','tp','fn','threshold','ai_sensitivity','false_negative_rate']
    count = 0
    with (root/'predictions.csv').open('x', newline='') as pf, (root/'per_model_sensitivity.csv').open('x', newline='') as sf:
        pw, sw = csv.DictWriter(pf, fieldnames=pred_fields), csv.DictWriter(sf, fieldnames=summary_fields)
        pw.writeheader(); sw.writeheader()
        identities = meta.to_dict('records')
        for common in index.to_dict('records'):
            score = replay(features, models[common['model_sha256']])
            positive = score >= 0.5
            for identity, value, predicted in zip(identities, score, positive):
                pw.writerow(dict(common, **identity, scoring_role=contract['scoring_role'],
                                 synthetic_test_only=contract['synthetic_test_only'], score=float(value), threshold=0.5, predicted_label=int(predicted)))
                count += 1
            tp = int(positive.sum())
            sw.writerow(dict(common, synthetic_test_only=contract['synthetic_test_only'], rows=len(meta), unique_ids=meta.id.nunique(),
                             unique_groups=meta.group_id.nunique(), tp=tp, fn=len(meta)-tp, threshold=0.5,
                             ai_sensitivity=tp/len(meta), false_negative_rate=(len(meta)-tp)/len(meta)))
    require(count == contract['prediction_rows'] and len(index) == contract['summary_rows'], 'Output row inventory mismatch')
    # Re-read publication CSVs in bounded chunks to ensure all model/ID cells,
    # identities and fixed-threshold endpoints are represented exactly once.
    audit_output(root, contract, index, meta)


def audit_output(root, contract, index, meta):
    expected_ids = meta.id.tolist()
    chunks = pd.read_csv(root/'predictions.csv', dtype=str, keep_default_na=False, chunksize=len(meta))
    count, observed_tp = 0, []
    try:
        for number, frame in enumerate(chunks):
            require(number < len(index) and len(frame) == len(meta) and frame.id.tolist() == expected_ids, 'Missing/duplicate prediction rows')
            common = index.iloc[number]
            require(all(frame[k].eq(common[k]).all() for k in INDEX), 'Prediction model grid mismatch')
            require(frame[META].reset_index(drop=True).equals(meta.reset_index(drop=True)), 'Prediction roles/identity changed')
            # Python float preserves the round-trip side of 0.5, including
            # adjacent representable values; pandas numeric parsing may not.
            scores = np.asarray([float(v) for v in frame.score], dtype=float)
            require(np.isfinite(scores).all() and frame.threshold.eq('0.5').all()
                    and frame.predicted_label.tolist() == (scores >= .5).astype(int).astype(str).tolist(), 'Prediction threshold mismatch')
            require(frame.scoring_role.eq(contract['scoring_role']).all()
                    and frame.synthetic_test_only.eq(str(contract['synthetic_test_only'])).all(), 'Prediction scope marker changed')
            observed_tp.append(int((scores >= .5).sum()))
            count += len(frame)
    finally:
        chunks.close()
    require(count == contract['prediction_rows'], 'Missing prediction grid')
    summary = A.read_csv(root/'per_model_sensitivity.csv')
    require(len(summary) == 3175 and summary[INDEX].equals(index), 'Missing/duplicate summary grid')
    require(summary.rows.eq(str(len(meta))).all() and summary.unique_ids.eq(str(len(meta))).all()
            and summary.unique_groups.eq(str(meta.group_id.nunique())).all()
            and summary.synthetic_test_only.eq(str(contract['synthetic_test_only'])).all()
            and summary.threshold.eq('0.5').all(), 'Summary denominator/threshold mismatch')
    tp, fn = summary.tp.astype(int), summary.fn.astype(int)
    require(tp.tolist() == observed_tp and ((tp >= 0) & (fn >= 0) & (tp+fn == len(meta))).all()
            and np.allclose(summary.ai_sensitivity.astype(float), tp/len(meta), rtol=0, atol=1e-15)
            and np.allclose(summary.false_negative_rate.astype(float), fn/len(meta), rtol=0, atol=1e-15), 'Summary endpoints mismatch')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['draft','score'])
    for name in ('old-results','old-package','old-preregistration','old-audit','prepared-dir','fhm-dir','fhm-reference','synthetic-fixture','receipt'):
        p.add_argument('--'+name, type=lambda s: Path(s).absolute())
    p.add_argument('--output-dir', type=lambda s: Path(s).absolute(), required=True)
    p.add_argument('--synthetic-test-only', action='store_true')
    p.add_argument('--measurement-version', choices=['v2','v3'], default='v2',
                   help='Explicit accepted neural/auditor interface; native preparation/FHM remain v2')
    a = p.parse_args(argv)
    require(not a.output_dir.exists() and not a.output_dir.is_symlink(), 'Refusing existing output path')
    contract, models, index, meta, features, bindings = assemble(a)
    if a.stage == 'score':
        authorize(contract, a.receipt)
        bindings.file(a.receipt)
    a.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=a.output_dir.name+'.tmp.', dir=a.output_dir.parent))
    try:
        index.to_csv(temp/'model_index.csv', index=False)
        meta.to_csv(temp/'identity_roles.csv', index=False)
        if a.stage == 'draft':
            write_json(temp/'scoring_draft.json', dict(status='draft', authorized_stage=STAGE,
                contract=contract, contract_sha256=digest(contract),
                independent_review=None, freeze_instruction='Parent/root must review exact real inputs/audit/model inventory before copying this receipt to a NEW path, setting status=frozen and adding independent_review. This program cannot freeze.'))
        else:
            write_scores(temp, contract, models, index, meta, features)
            write_json(temp/'scoring_receipt.json', read_json(a.receipt))
        bindings.recheck()
        inventory = {path.name: {'sha256': P.sha(path), 'bytes': path.stat().st_size} for path in sorted(temp.iterdir())}
        write_json(temp/'publication_manifest.json', dict(status='draft' if a.stage == 'draft' else 'scored',
            synthetic_test_only=a.synthetic_test_only, contract_sha256=digest(contract), classifier_fitted=False,
            original_v4_development_admission=False, unique_new_ids=len(meta), model_instances=3175,
            prediction_rows=0 if a.stage == 'draft' else 3175*len(meta), summary_rows=0 if a.stage == 'draft' else 3175,
            files=inventory))
        require(not a.output_dir.exists(), 'Output appeared during run')
        publish_directory(temp, a.output_dir)
    except BaseException:
        shutil.rmtree(temp)
        raise


if __name__ == '__main__':
    main()
