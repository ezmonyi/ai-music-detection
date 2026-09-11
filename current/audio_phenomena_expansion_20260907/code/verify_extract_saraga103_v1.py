#!/usr/bin/env python3
"""Strict completion-gated Saraga103 unchanged old4 extraction.

All103 identities and external human role remain in the companion ledger.
Empty actually emitted beat files produce unavailable features; never fabricated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import run_saraga103_inference_v1 as runner
from run_saraga103_inference_v1 import (COUNT, ROLE, check_seal, dependency_snapshot, file_record,
    json_bytes, measurement_only, publish, read_json, regular, require, scope, sealed, sha_file)
from verify_extract_equal60 import STEMS


def audit(a, decode=True):
    """A full10 receipt completeness gate precedes expensive decoding."""
    require(a.inference_root == a.output_root, 'Wrong isolated inference root')
    runner.validate_paths(a)
    completion_path = regular(a.inference_root / 'completion.json')
    c = read_json(regular(a.inference_root / 'run_contract.json'))
    check_seal(c)
    measurement_only(c)
    completion = read_json(completion_path)
    check_seal(completion)
    require(completion.get('completed_stage_shards') == 10 and completion.get('rows') == COUNT,
            'All10 stage receipts required before extraction')
    for stage in runner.STAGES:
        for index in range(runner.SHARDS):
            regular(a.inference_root / 'receipts' / f'{stage}_{index:02d}.json')
    a.aio_gpus, a.beat_gpus = c['allinone_gpus'], c['beats_gpus']
    expected, rows, shards = runner.current_contract(a, decode=decode)
    require(c == expected, 'Inference contract differs from current full input/runtime/dependency bindings')
    runner.verify_authorization(a, c)
    runner.validate_aliases(a.inference_root, rows)
    runner.verify_completion(a.inference_root, c, rows, shards, decode=decode)
    products = {str(p): file_record(p) for row in rows for stage in runner.STAGES
                for p in runner.original.product_paths(stage, row['item_id'], a.inference_root)}
    require(dependency_snapshot() == c['dependencies_sha256'], 'Audit dependencies changed')
    for path, digest in {**c['preparation_sha256'], **c['runtime_sha256']}.items():
        require(sha_file(path) == digest, 'Audit provenance/runtime changed')
    result = sealed(dict(scope(), status='passed', rows=COUNT,
        run_contract_sha256=sha_file(a.inference_root / 'run_contract.json'),
        completion_sha256=sha_file(completion_path),
        authorization_file_sha256=sha_file(a.frozen),
        dependencies_sha256=c['dependencies_sha256'],
        preparation_sha256=c['preparation_sha256'], runtime_sha256=c['runtime_sha256'],
        inputs=runner.input_records(rows), products=products,
        alias_commit=file_record(a.inference_root / 'inputs/COMMIT.json'),
        adapted_manifest=file_record(a.inference_root / 'inputs/inference_manifest.csv'),
        all10_stage_receipts_verified=True, exact_product_and_log_union_verified=True,
        same_process_cuda_guard_required=True, cuda_guard_version=runner.cuda_guard.VERSION,
        cuda_guard_sha256=c['cuda_guard_sha256'],
        cuda_guard_receipts={str(runner.guard_path(a.inference_root, stage, index)):
            file_record(runner.guard_path(a.inference_root, stage, index))
            for stage in runner.STAGES for index in range(runner.SHARDS)},
        all_input_and_stem_samples_decoded=decode, spectrogram_shape=[4, 6000, 81],
        spectrogram_dtype='float32',
        empty_beats_policy='only actually emitted receipted file; unavailable, never fabricated or repaired',
        native_fhm_metadata='separate accepted native metadata; not extracted by old four-family extractor'))
    return rows, result


def extraction_command(a):
    return [sys.executable, str(a.old_code_root/'extract_expanded_four_family.py'),
        '--manifest', str(a.inference_root/'inputs/inference_manifest.csv'), '--bias', str(a.bias), '--duration', '60',
        '--demix-root', str(a.inference_root/'demix'), '--beat-root', str(a.inference_root/'beats'),
        '--structure-root', str(a.inference_root/'structure'), '--output-dir', str(a.output_dir/'features'),
        '--workers', str(a.workers), '--strict']


def feature_records(a, rows, products):
    path = regular(a.output_dir/'features/expanded_features_60s.csv')
    with path.open() as f:
        records = list(csv.DictReader(f))
    require([r['item_id'] for r in records] == [r['item_id'] for r in rows]
            and len(records) == COUNT and all(r['status'] == 'complete' for r in records),
            'Incomplete/misaligned old feature extraction')
    forbidden = {'prediction', 'predictions', 'probability', 'probabilities', 'predicted_label',
                 'fold', 'fold_id', 'split', 'classifier_score', 'ai_score', 'threshold'}
    require(not set(records[0]).intersection(forbidden), 'Unexpected classifier/scoring output columns')
    jsonl = regular(a.output_dir / 'features/expanded_features_60s.jsonl')
    payloads = [json.loads(line) for line in jsonl.read_text().splitlines()]
    require([r['item_id'] for r in payloads] == [r['item_id'] for r in rows]
            and len(payloads) == COUNT, 'JSONL cohort omission/duplication/order mismatch')
    metadata = read_json(a.output_dir / 'features/expanded_features_60s_metadata.json')
    expected_payload = dict(extractor_sha256=sha_file(a.old_code_root / 'extract_expanded_four_family.py'),
        core_sha256=sha_file(a.old_code_root / 'expanded_feature_definitions.py'),
        bias_sha256=sha_file(a.bias), duration=60.0,
        demix_roots=[str(a.inference_root / 'demix')], beat_roots=[str(a.inference_root / 'beats')],
        structure_roots=[str(a.inference_root / 'structure')])
    fingerprint = hashlib.sha256(json.dumps(expected_payload, sort_keys=True,
        separators=(',', ':'), allow_nan=True).encode()).hexdigest()
    require(metadata['run_payload'] == expected_payload and metadata['run_fingerprint'] == fingerprint
            and metadata['rows'] == metadata['complete'] == COUNT and metadata['partial'] == 0
            and metadata['duration_sec'] == 60, 'Extractor metadata/runtime fingerprint mismatch')
    for record, row, payload in zip(records, rows, payloads):
        require(payload['run_fingerprint'] == fingerprint
                and payload['bias_sha256'] == expected_payload['bias_sha256'], 'Feature fingerprint/bias changed')
        require(payload['label'] == '0' and payload['source_id'] == runner.SOURCE
                and payload['group_id'] == row['group_id'] and payload['status'] == 'complete',
                'JSONL identity/role scope mismatch')
        require(str(payload['native_sample_rate_hz']) == row['native_sr']
                and record['native_sample_rate_hz'] == row['native_sr'],
                'Original native sample rate lost in extractor')
        require(not set(payload).intersection(forbidden), 'JSONL contains classifier output')
        require(record['label'] == '0' and record['source_id'] == runner.SOURCE
                and record['group_id'] == row['group_id'], 'Extracted identity/source/label changed')
        expected = {'source_audio_sha256': row['standardized_file_sha256'],
                    **{stem+'_sha256': products[str(a.inference_root/'demix/htdemucs'/row['item_id']/(stem+'.wav'))]['sha256']
                       for stem in STEMS},
                    'beats_sha256': products[str(a.inference_root/'beats'/(row['item_id']+'.beats'))]['sha256'],
                    'structure_sha256': products[str(a.inference_root/'structure'/(row['item_id']+'.json'))]['sha256']}
        require(json.loads(record['input_hashes']) == json.loads(payload['input_hashes']) == expected,
                'Extractor input hashes differ from audited products')
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                require(json.loads(record[key]) == value, 'CSV/JSONL payload mismatch: ' + key)
            else:
                require(str(value) == record[key], 'CSV/JSONL payload mismatch: ' + key)
    return records


def feature_inventory(output):
    root = output/'features'
    expected = {root/'expanded_features_60s.csv', root/'expanded_features_60s.jsonl',
                root/'expanded_features_60s_metadata.json'}
    actual = {p for p in root.rglob('*') if p.is_file()}
    require(actual == expected and not any(p.is_symlink() for p in root.rglob('*')), 'Unexpected/missing extractor products')
    return {str(p):file_record(p) for p in sorted(expected)}


def verify_extraction_receipt(a, rows, audited):
    receipt = read_json(regular(a.output_dir/'extraction_receipt.json'))
    check_seal(receipt)
    measurement_only(receipt)
    require(receipt['status'] == 'passed' and receipt['rows'] == COUNT and receipt['command'] == extraction_command(a)
            and receipt['strict_audit_sha256'] == sha_file(a.output_dir/'strict_inference_audit.json')
            and receipt['dependencies_sha256'] == dependency_snapshot(), 'Extraction receipt binding changed')
    require(receipt['features'] == feature_inventory(a.output_dir)
            and receipt['log'] == file_record(a.output_dir/'extraction.log')
            and receipt['role_ledger'] == file_record(a.output_dir/'measurement_roles.json'), 'Extraction artifacts changed')
    roles = read_json(a.output_dir/'measurement_roles.json')
    check_seal(roles)
    measurement_only(roles)
    require(roles['items'] == role_items(rows), 'Measurement role ledger changed')
    feature_records(a, rows, audited['products'])


def role_items(rows):
    return [{'item_id':r['item_id'], 'role':ROLE, 'label':0, 'source_id':runner.SOURCE,
             'group_id':r['group_id'], 'evaluation_allowed':False, 'classifier_admission_authorized':False} for r in rows]


def run_locked(a):
    rows, before = audit(a, decode=True)
    publish(a.output_dir/'strict_inference_audit.json', json_bytes(before))
    if not a.extract:
        return
    if (a.output_dir/'extraction_receipt.json').exists():
        verify_extraction_receipt(a, rows, before)
        return
    require(not (a.output_dir/'features').exists() and not (a.output_dir/'extraction.log').exists()
            and not (a.output_dir/'measurement_roles.json').exists(), 'Unreceipted extraction products; manual review required')
    cmd = extraction_command(a)
    with (a.output_dir/'extraction.log').open('x') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = feature_records(a, rows, before['products'])
    after_rows, after = audit(a, decode=False)
    # Full hashes are rechecked after extraction; full media decoding was before extraction.
    after_body = {k:v for k,v in after.items() if k not in ('canonical_sha256', 'all_input_and_stem_samples_decoded')}
    before_body = {k:v for k,v in before.items() if k not in ('canonical_sha256', 'all_input_and_stem_samples_decoded')}
    require(after_rows == rows and after_body == before_body, 'Source/product/runtime/provenance changed during extraction')
    roles = sealed(dict(scope(), schema_version=1, items=role_items(rows),
        note='Authoritative role companion to unchanged old feature CSV; no development or classifier admission.'))
    publish(a.output_dir/'measurement_roles.json', json_bytes(roles))
    receipt = sealed(dict(scope(), status='passed', rows=COUNT, command=cmd,
        strict_audit_sha256=sha_file(a.output_dir/'strict_inference_audit.json'),
        dependencies_sha256=dependency_snapshot(), features=feature_inventory(a.output_dir),
        log=file_record(a.output_dir/'extraction.log'), role_ledger=file_record(a.output_dir/'measurement_roles.json'),
        availability={k:dict(Counter(r[k] for r in records)) for k in ('s8_computed','d_eligible','r_eligible','p_eligible')},
        before_after_all_provenance_inputs_products_dependencies_verified=True))
    publish(a.output_dir/'extraction_receipt.json', json_bytes(receipt))
    verify_extraction_receipt(a, rows, before)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    runner.common_arguments(p)
    p.add_argument('--inference-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--frozen-sha256', required=True)
    p.add_argument('--extract', action='store_true')
    a = p.parse_args()
    a.output_root = a.inference_root
    runner.validate_paths(a)
    require(a.output_dir == a.inference_root / 'measurements'
            and a.output_dir.resolve() == a.output_dir and a.workers > 0,
            'Only dedicated Saraga old4 measurements destination permitted')
    regular(a.inference_root / 'completion.json')
    a.output_dir.mkdir(exist_ok=True)
    with regular(a.inference_root / 'writer.lock').open('r+') as inference_lock, \
            (a.output_dir / 'writer.lock').open('a') as lock:
        fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(a)
    print(json.dumps(dict(status='passed', rows=COUNT, extracted=a.extract, **scope())))


if __name__ == '__main__':
    main()
