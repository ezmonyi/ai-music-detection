#!/usr/bin/env python3
"""Strictly audit Mureka500 inference_v3 and optionally run old extraction.

Released original GPUs never bypass receipt, product, runtime, preparation, or
measurement-only checks.  Any embedded original1604 release proof is rebuilt
from current fixed-root bytes before v3 output is accepted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import run_mureka60_inference_v3 as runner
import verify_extract_mureka60_v2 as v2audit


COUNT, ROLE = runner.COUNT, runner.ROLE
check_seal = runner.check_seal
file_record = runner.file_record
json_bytes = runner.json_bytes
measurement_only = runner.measurement_only
publish = runner.publish
read_json = runner.read_json
regular = runner.regular
require = runner.require
scope = runner.scope
sealed = runner.sealed
sha_file = runner.sha_file


def audit(a, decode=True, fixed_original_root=runner.ORIGINAL_PREPARED_ROOT,
          synthetic_test=False):
    require(a.inference_root == a.prepared_dir / 'inference_v3'
            and a.inference_root.resolve() == a.inference_root,
            'Only dedicated Mureka inference_v3 accepted')
    completion_path = regular(a.inference_root / 'completion.json')
    contract = read_json(regular(a.inference_root / 'run_contract.json'))
    check_seal(contract)
    measurement_only(contract)
    release_proof = contract.get('original_gpu_release_proof')
    runner.validate_gpus(contract['allinone_gpus'], contract['beats_gpus'], release_proof)
    if release_proof is not None:
        require(Path(release_proof['fixed_prepared_root']) == Path(fixed_original_root),
                'GPU release proof is not for the fixed original1604 root')
        runner.check_seal(release_proof)
        current_proof = runner.validate_original_completion(
            Path(fixed_original_root) / 'inference/completion.json', fixed_original_root,
            synthetic_test=synthetic_test)
        require(release_proof == current_proof, 'Original GPU release proof is stale or changed')
    completion = read_json(completion_path)
    check_seal(completion)
    require(completion.get('completed_stage_shards') == 42 and completion.get('rows') == COUNT,
            'Full42 stage receipts required before extraction')
    for stage in runner.STAGES:
        for index in range(runner.SHARDS):
            regular(a.inference_root / 'receipts' / f'{stage}_{index:02d}.json')
    dependencies = runner.dependency_snapshot()
    rows, shards, provenance = runner.validate_prepared(a.prepared_dir, decode=decode)
    runtime = runner.v2.runtime_snapshot(a)
    expected_args = dict(vars(a), output_root=a.inference_root,
                         aio_gpus=contract['allinone_gpus'], beat_gpus=contract['beats_gpus'])
    expected = runner.build_contract(SimpleNamespace(**expected_args),
        rows, shards, provenance, runtime, dependencies, release_proof)
    require(contract == expected,
            'Inference v3 contract differs from current provenance/runtime/dependencies/release proof')
    runner.verify_completion(a.inference_root, contract, rows, shards, decode=decode)
    products = {str(path): file_record(path) for row in rows for stage in runner.STAGES
                for path in runner.original.product_paths(stage, row['item_id'], a.inference_root)}
    require(runner.dependency_snapshot() == dependencies, 'Audit dependencies changed')
    for path, digest in {**provenance, **runtime}.items():
        require(sha_file(path) == digest, 'Audit provenance/runtime changed')
    if release_proof is not None:
        require(runner.validate_original_completion(
            Path(fixed_original_root) / 'inference/completion.json', fixed_original_root,
            synthetic_test=synthetic_test) == release_proof,
            'Original GPU release proof changed during audit')
    result = sealed(dict(scope(), status='passed', rows=COUNT,
        run_contract_sha256=sha_file(a.inference_root / 'run_contract.json'),
        completion_sha256=sha_file(completion_path), dependencies_sha256=dependencies,
        preparation_sha256=provenance, runtime_sha256=runtime,
        original_gpu_release_proof=release_proof, inputs=runner.input_records(rows), products=products,
        all42_stage_receipts_verified=True, exact_product_and_log_union_verified=True,
        explicit_stage_gpu_assignment_verified=True,
        all_input_and_stem_samples_decoded=decode, spectrogram_shape=[4, 6000, 81],
        spectrogram_dtype='float32',
        empty_beats_policy='only an actually emitted receipted file; unavailable, never fabricated or repaired',
        native_fhm_metadata='separate frozen native_metadata_60s.csv; not extracted by old four-family extractor'))
    return rows, result


def extraction_command(a):
    return v2audit.extraction_command(a)


def feature_records(a, rows, products):
    return v2audit.feature_records(a, rows, products)


def feature_inventory(output):
    return v2audit.feature_inventory(output)


def role_items(rows):
    return v2audit.role_items(rows)


def verify_extraction_receipt(a, rows, audited):
    receipt = read_json(regular(a.output_dir / 'extraction_receipt.json'))
    check_seal(receipt)
    measurement_only(receipt)
    require(receipt['status'] == 'passed' and receipt['rows'] == COUNT
            and receipt['command'] == extraction_command(a)
            and receipt['strict_audit_sha256'] == sha_file(a.output_dir / 'strict_inference_audit.json')
            and receipt['dependencies_sha256'] == runner.dependency_snapshot(),
            'Extraction receipt binding changed')
    require(receipt['features'] == feature_inventory(a.output_dir)
            and receipt['log'] == file_record(a.output_dir / 'extraction.log')
            and receipt['role_ledger'] == file_record(a.output_dir / 'measurement_roles.json'),
            'Extraction artifacts changed')
    roles = read_json(a.output_dir / 'measurement_roles.json')
    check_seal(roles)
    measurement_only(roles)
    require(roles['items'] == role_items(rows), 'Measurement role ledger changed')
    feature_records(a, rows, audited['products'])


def run_locked(a):
    rows, before = audit(a, decode=True)
    publish(a.output_dir / 'strict_inference_audit.json', json_bytes(before))
    if not a.extract:
        return
    if (a.output_dir / 'extraction_receipt.json').exists():
        verify_extraction_receipt(a, rows, before)
        return
    require(not (a.output_dir / 'features').exists()
            and not (a.output_dir / 'extraction.log').exists()
            and not (a.output_dir / 'measurement_roles.json').exists(),
            'Unreceipted extraction products; manual review required')
    command = extraction_command(a)
    with (a.output_dir / 'extraction.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = feature_records(a, rows, before['products'])
    after_rows, after = audit(a, decode=False)
    after_body = {key: value for key, value in after.items()
                  if key not in ('canonical_sha256', 'all_input_and_stem_samples_decoded')}
    before_body = {key: value for key, value in before.items()
                   if key not in ('canonical_sha256', 'all_input_and_stem_samples_decoded')}
    require(after_rows == rows and after_body == before_body,
            'Source/product/runtime/provenance/release proof changed during extraction')
    roles = sealed(dict(scope(), schema_version=1, items=role_items(rows),
        note='Authoritative role companion to unchanged old feature CSV; no development or classifier admission.'))
    publish(a.output_dir / 'measurement_roles.json', json_bytes(roles))
    receipt = sealed(dict(scope(), status='passed', rows=COUNT, command=command,
        strict_audit_sha256=sha_file(a.output_dir / 'strict_inference_audit.json'),
        dependencies_sha256=runner.dependency_snapshot(), features=feature_inventory(a.output_dir),
        log=file_record(a.output_dir / 'extraction.log'),
        role_ledger=file_record(a.output_dir / 'measurement_roles.json'),
        availability={key: dict(Counter(row[key] for row in records))
                      for key in ('s8_computed', 'd_eligible', 'r_eligible', 'p_eligible')},
        before_after_all_provenance_inputs_products_dependencies_release_proof_verified=True))
    publish(a.output_dir / 'extraction_receipt.json', json_bytes(receipt))
    verify_extraction_receipt(a, rows, before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runner.common_arguments(parser)
    parser.add_argument('--inference-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    a = parser.parse_args()
    runner.v2.prep.validate_paths(runner.v2.prep.SOURCE, a.prepared_dir)
    require(a.output_dir == a.prepared_dir / 'measurements'
            and a.output_dir.resolve() == a.output_dir,
            'Only dedicated Mureka measurements destination permitted')
    require(a.workers > 0, 'Positive worker count required')
    require(a.inference_root == a.prepared_dir / 'inference_v3', 'Wrong inference_v3 root')
    regular(a.inference_root / 'completion.json')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.inference_root / 'writer.lock').open('a') as inference_lock, \
            (a.output_dir / 'writer.lock').open('a') as output_lock:
        fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(a)
    print(json.dumps(dict(status='passed', rows=COUNT, extracted=a.extract, **scope())))


if __name__ == '__main__':
    main()
