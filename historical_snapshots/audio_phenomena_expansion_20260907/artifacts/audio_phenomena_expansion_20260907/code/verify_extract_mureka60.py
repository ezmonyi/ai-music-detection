#!/usr/bin/env python3
"""Completion-gated Mureka500 output audit and unchanged old feature extraction.

Measurement only. Raw old extractor outputs retain their original schema; an
explicit companion identity/role ledger forbids development/scoring admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import run_mureka60_inference as runner
from run_mureka60_inference import (COUNT, ROLE, check_seal, dependency_snapshot, file_record,
    json_bytes, measurement_only, publish, read_json, regular, require, scope, sealed, sha_file)
from verify_extract_equal60 import STEMS


def audit(a, decode=True):
    """Completion and every receipt required BEFORE any expensive full decoding."""
    require(a.inference_root == a.prepared_dir/'inference' and a.inference_root.resolve() == a.inference_root,
            'Only dedicated Mureka inference accepted')
    completion_path = regular(a.inference_root/'completion.json')
    c = read_json(regular(a.inference_root/'run_contract.json'))
    check_seal(c)
    measurement_only(c)
    runner.validate_gpus(c['allinone_gpus'], c['beats_gpus'])
    # Cheap completeness gate; detailed ledger verification follows preparation identity checks.
    completion = read_json(completion_path)
    check_seal(completion)
    require(completion.get('completed_stage_shards') == 42 and completion.get('rows') == COUNT,
            'Full42 stage receipts required before extraction')
    for stage in runner.STAGES:
        for index in range(runner.SHARDS):
            regular(a.inference_root/'receipts'/f'{stage}_{index:02d}.json')
    deps = dependency_snapshot()
    rows, shards, provenance = runner.validate_prepared(a.prepared_dir, decode=decode)
    runtime = runner.runtime_snapshot(a)
    expected = runner.build_contract(SimpleNamespace(**vars(a), output_root=a.inference_root,
        aio_gpus=c['allinone_gpus'], beat_gpus=c['beats_gpus']), rows, shards, provenance, runtime, deps)
    require(c == expected, 'Inference contract differs from current provenance/runtime/dependencies')
    runner.verify_completion(a.inference_root, c, rows, shards, decode=decode)
    products = {str(p): file_record(p) for row in rows for stage in runner.STAGES
                for p in runner.original.product_paths(stage, row['item_id'], a.inference_root)}
    require(dependency_snapshot() == deps, 'Audit dependencies changed')
    for path, digest in {**provenance, **runtime}.items():
        require(sha_file(path) == digest, 'Audit provenance/runtime changed')
    result = sealed(dict(scope(), status='passed', rows=COUNT, run_contract_sha256=sha_file(a.inference_root/'run_contract.json'),
        completion_sha256=sha_file(completion_path), dependencies_sha256=deps, preparation_sha256=provenance,
        runtime_sha256=runtime, inputs=runner.input_records(rows), products=products,
        all42_stage_receipts_verified=True, exact_product_and_log_union_verified=True,
        all_input_and_stem_samples_decoded=decode, spectrogram_shape=[4,6000,81], spectrogram_dtype='float32',
        empty_beats_policy='only an actually emitted receipted file; unavailable, never fabricated or repaired',
        native_fhm_metadata='separate frozen native_metadata_60s.csv; not extracted by old four-family extractor'))
    return rows, result


def extraction_command(a):
    return [sys.executable, str(a.old_code_root/'extract_expanded_four_family.py'),
        '--manifest', str(a.prepared_dir/'inference_manifest.csv'), '--bias', str(a.bias), '--duration', '60',
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
    for record, row in zip(records, rows):
        require(record['label'] == '1' and record['source_id'] == 'Mureka_v9'
                and record['group_id'] == row['group_id'], 'Extracted identity/source/label changed')
        expected = {'source_audio_sha256': row['standardized_file_sha256'],
                    **{stem+'_sha256': products[str(a.inference_root/'demix/htdemucs'/row['item_id']/(stem+'.wav'))]['sha256']
                       for stem in STEMS},
                    'beats_sha256': products[str(a.inference_root/'beats'/(row['item_id']+'.beats'))]['sha256'],
                    'structure_sha256': products[str(a.inference_root/'structure'/(row['item_id']+'.json'))]['sha256']}
        require(json.loads(record['input_hashes']) == expected, 'Extractor input hashes differ from audited products')
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
    return [{'item_id':r['item_id'], 'role':ROLE, 'label':1, 'source_id':'Mureka_v9',
             'group_id':r['group_id'], 'classifier_admission_authorized':False} for r in rows]


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
    p.add_argument('--extract', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    runner.prep.validate_paths(runner.prep.SOURCE, a.prepared_dir)
    require(a.output_dir == a.prepared_dir/'measurements' and a.output_dir.resolve() == a.output_dir,
            'Only dedicated Mureka measurements destination permitted')
    require(a.workers > 0, 'Positive worker count required')
    require(a.inference_root == a.prepared_dir/'inference', 'Wrong inference root')
    regular(a.inference_root/'completion.json')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    # The inference lock excludes writers while auditing/extracting the immutable products.
    with (a.inference_root/'writer.lock').open('a') as inference_lock, (a.output_dir/'writer.lock').open('a') as lock:
        fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(a)
    print(json.dumps(dict(status='passed', rows=COUNT, extracted=a.extract, **scope())))


if __name__ == '__main__':
    main()
