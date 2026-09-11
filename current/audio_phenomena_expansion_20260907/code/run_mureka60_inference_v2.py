#!/usr/bin/env python3
"""Interval-verified v2 Mureka500 inference; explicit idle GPUs, strict receipts.

No scoring, fitting, downloads, output repair, or empty-beat fabrication.
The original exact60 runner and feature functions are imported without edits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading

import prepare_mureka60_inputs_v2 as prep
from prepare_mureka60_inputs_v2 import canonical_hash, json_bytes, publish, read_json, require, sha_file
import run_equal60_inference_batches as original
from verify_extract_equal60 import inspect_audio, verify_runtime

ROLE = 'external_generator_unscored'
COUNT, SHARDS = 500, 21
STAGES = ('allinone', 'beats')
PINNED_HELPERS = {
    'run_equal60_inference_batches.py': 'a33f667ac9ef5cc5b4e5e646adfb752a68380b1af5f42b0768b1a10e3d031858',
    'verify_extract_equal60.py': 'c921bc04988f58a382efea080df2c5493b1d4faca3e48bc38797b59981316771',
    'materialize_equal60_inputs_v2.py': 'ab2a82f3dd2cf3f30786b2c680e33e2665ca3da8f3c2fab46fa5659019868aba',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path,
            f'Missing/noncanonical regular file: {path}')
    return path


def file_record(path):
    path = regular(path)
    return {'sha256': sha_file(path), 'bytes': path.stat().st_size}


def sealed(body):
    require('canonical_sha256' not in body, 'Already sealed')
    return dict(body, canonical_sha256=canonical_hash(body))


def check_seal(value):
    require(value.get('canonical_sha256') == canonical_hash({k: v for k, v in value.items()
                                                           if k != 'canonical_sha256'}),
            'Canonical receipt/contract hash mismatch')


def measurement_only(value):
    require(value.get('purpose') == 'measurement_only' and value.get('role') == ROLE
            and value.get('classifier_fitted') is False and value.get('scores_generated') is False
            and value.get('classifier_admission_authorized') is False, 'Forbidden scoring/development role')


def scope():
    return dict(purpose='measurement_only', role=ROLE, classifier_fitted=False,
                scores_generated=False, classifier_admission_authorized=False)


def dependency_snapshot():
    root = Path(__file__).resolve().parent
    for name, expected in PINNED_HELPERS.items():
        require(sha_file(root / name) == expected, 'Frozen helper changed: ' + name)
    names = [*PINNED_HELPERS, 'prepare_mureka60_inputs_v2.py', 'run_mureka60_inference_v2.py',
             'verify_extract_mureka60_v2.py']
    return {str(root / name): sha_file(regular(root / name)) for name in names}


def validate_shards(prepared, summary, rows):
    require(len(rows) == COUNT and len({r['item_id'] for r in rows}) == COUNT
            and len({r['standardized_path'] for r in rows}) == COUNT, 'Exactly500 unique rows required')
    require(len(summary['shards']) == SHARDS, 'Exactly21 shards required')
    shards, all_paths = [], []
    for index, shard in enumerate(summary['shards']):
        path = prepared / f'inference_shard_{index:02d}.txt'
        require(shard['index'] == index and shard['path'] == str(path)
                and shard['rows'] == (24 if index < 20 else 20), 'Shard index/count/path mismatch')
        require(sha_file(regular(path)) == shard['sha256'], 'Shard bytes changed')
        inputs = path.read_text().splitlines()
        require(len(inputs) == shard['rows'], 'Shard count changed')
        shards.append((index, inputs))
        all_paths.extend(inputs)
    require(all_paths == [r['standardized_path'] for r in rows], 'Shard omission/duplication/order mismatch')
    require({p.name for p in prepared.glob('inference_shard_*.txt')}
            == {f'inference_shard_{i:02d}.txt' for i in range(SHARDS)}, 'Unexpected shard inventory')
    return shards


def validate_interval_evidence(receipt, native, source):
    for key in ('sf_header_frames', 'sf_actual_read_frames'):
        require(receipt[key] == source[key] and int(native[key]) == source[key],
                'Decoder observation identity mismatch: ' + key)
    for key in ('native_crop_float64_sha256', 'native_crop_float32_sha256'):
        require(receipt[key] == source[key] == native[key] and len(source[key]) == 64,
                'Verified native interval sample hash mismatch: ' + key)
    require(source['sf_actual_read_frames'] >= receipt['crop_end_frame_exclusive']
            and receipt['standardized_waveform_sha256'] == source['native_crop_float32_sha256'],
            'Requested interval exceeds real decoded EOF or standardized samples changed')


def validate_prepared(prepared, decode=True):
    """All500 selection, acquisition proof, metadata, receipts and audio before GPU."""
    prepared = Path(prepared)
    prep.validate_paths(prep.SOURCE, prepared)
    c = read_json(regular(prepared / 'materialization_contract.json'))
    s = read_json(regular(prepared / 'materialization_summary.json'))
    require(c['contract_sha256'] == canonical_hash({k:v for k,v in c.items() if k != 'contract_sha256'}),
            'Preparation canonical hash mismatch')
    require(c['purpose'] == 'measurement_only' and c['role'] == s['role'] == ROLE
            and c['classifier_admission_authorized'] is False and s['classifier_admission_authorized'] is False,
            'Forbidden preparation role/admission')
    require(c['status'] == 'frozen_before_materialization' and s['status'] == 'verified' and s['rows'] == COUNT
            and c['contract_sha256'] == s['contract_sha256'], 'Missing complete500 preparation')
    require(c['source_root'] == str(prep.SOURCE) and c['output_dir'] == str(prepared)
            and c['source_repo'] == 'homura23/MUSIC8K' and c['source_revision'] == prep.REVISION
            and c['acquisition_contract_sha256'] == prep.ACQUISITION_SHA, 'Wrong source/acquisition')
    require(c['code_sha256'] == sha_file(prep.__file__) and c['configuration'] == prep.CONFIG
            and c['configuration_sha256'] == canonical_hash(prep.CONFIG)
            and c['runtime_sha256'] == canonical_hash(c['runtime']), 'Preparation code/config/runtime binding changed')
    # V2 validates the complete frozen interval evidence and source bindings;
    # audit_inputs alone contains acquisition fields, not measured EOF/crop proof.
    source_rows = prep.validate_contract(c, prep.SOURCE, prepared)
    source_bindings = c['input_file_sha256']
    require(c['rows'] == source_rows and c['input_file_sha256'] == source_bindings
            and c['selected_ids'] == [r['id'] for r in source_rows] and len(source_rows) == COUNT,
            'Acquisition all500 selection/evidence mismatch')
    manifest = regular(prepared / 'inference_manifest.csv')
    native = regular(prepared / 'native_metadata_60s.csv')
    require(sha_file(manifest) == s['manifest_sha256'] and sha_file(native) == s['native_metadata_sha256'],
            'Prepared manifest/native metadata changed')
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    with native.open() as f:
        native_rows = list(csv.DictReader(f))
    ids = ['music8k_mureka_v9_' + r['id'] for r in source_rows]
    require([r['item_id'] for r in rows] == ids and [r['id'] for r in native_rows] == ids,
            'Prepared identity/order mismatch')
    expected_receipts = {i + '.json' for i in ids}
    require(set(s['receipts_sha256']) == expected_receipts
            and {p.name for p in (prepared / 'items').iterdir()} == expected_receipts, 'Preparation receipt union mismatch')
    require({str(p) for p in (prepared / 'audio').iterdir()}
            == {str(prepared / 'audio' / (i + '.wav')) for i in ids}, 'Prepared audio inventory mismatch')
    bindings = {str(prepared / n): sha_file(prepared / n) for n in
                ('materialization_contract.json', 'materialization_summary.json', 'inference_manifest.csv', 'native_metadata_60s.csv')}
    for row, n, src in zip(rows, native_rows, source_rows):
        item_id = row['item_id']
        path = regular(prepared / 'items' / (item_id + '.json'))
        receipt = read_json(path)
        require(sha_file(path) == s['receipts_sha256'][path.name], 'Preparation item receipt changed')
        bindings[str(path)] = sha_file(path)
        require(row == {k: str(v) for k,v in receipt.items()}, 'Manifest differs from full receipt')
        require(receipt['role'] == ROLE and receipt['source_id'] == receipt['source_group'] == 'Mureka_v9'
                and receipt['label'] == 1 and receipt['status'] == 'verified'
                and receipt['classifier_admission_authorized'] is False
                and receipt['acquisition_role'] == 'reserved_unscored', 'Wrong row source/role/admission')
        require(receipt['contract_sha256'] == c['contract_sha256']
                and receipt['input_row_sha256'] == canonical_hash(src)
                and receipt['source_audio_path'] == str(prep.SOURCE / 'raw' / src['path'])
                and receipt['source_audio_sha256'] == src['sha256']
                and receipt['group_id'] == src['reference_group_id']
                and receipt['source_total_frames'] == src['native_frames']
                and receipt['crop_start_frame'] == (src['native_frames'] - prep.FRAMES)//2
                and receipt['crop_frames'] == prep.FRAMES
                and receipt['crop_end_frame_exclusive'] == receipt['crop_start_frame'] + prep.FRAMES,
                'Native crop/source provenance mismatch')
        require(receipt['standardized_path'] == str(prepared / 'audio' / (item_id + '.wav'))
                and receipt['duration'] == 60 and receipt['audio_offset_s'] == receipt['requires_crop'] == 0
                and receipt['standardized_sr'] == receipt['native_sr'] == 44100
                and receipt['standardized_channels'] == 2 and receipt['standardized_frames'] == prep.FRAMES,
                'Exact60 representation mismatch')
        require(n['role'] == ROLE and n['source_group'] == 'Mureka_v9' and n['label'] == '1'
                and n['classifier_admission_authorized'] == 'False'
                and n['group_id'] == receipt['group_id'] and n['acquisition_role'] == 'reserved_unscored'
                and int(n['native_sample_rate_hz']) == int(n['physical_sample_rate_hz']) == 44100
                and int(n['physical_channels']) == 2 and int(n['physical_frames']) == src['native_frames']
                and n['audio_path'] == receipt['source_audio_path']
                and n['source_audio_sha256'] == receipt['source_audio_sha256']
                and int(n['crop_start_frame']) == receipt['crop_start_frame']
                and int(n['crop_frames']) == prep.FRAMES
                and int(n['crop_end_frame_exclusive']) == receipt['crop_end_frame_exclusive']
                and float(n['audio_offset_s']) == receipt['crop_start_frame']/44100
                and round(float(n['audio_offset_s'])*44100) == receipt['crop_start_frame'],
                'Native FHM metadata interval/source mismatch')
        validate_interval_evidence(receipt, n, src)
        audio = regular(row['standardized_path'])
        require(file_record(audio) == {'sha256': row['standardized_file_sha256'],
                                      'bytes': int(row['standardized_file_bytes'])}, 'Prepared audio bytes changed')
        if decode:
            inspect_audio(audio, row['standardized_file_sha256'], subtype='FLOAT')
    shards = validate_shards(prepared, s, rows)
    for index, _ in shards:
        path = prepared / f'inference_shard_{index:02d}.txt'
        bindings[str(path)] = sha_file(path)
    bindings.update({str(prep.SOURCE / k): v for k,v in source_bindings.items()})
    bindings.update(c['decoder_evidence_sha256'])
    return rows, shards, bindings


def runtime_snapshot(a):
    checks = verify_runtime(a.runtime_contract, a.checkpoint_manifest, a.old_code_root, a.bias)
    frozen = read_json(a.runtime_contract)
    require(str(a.runtime_root) == frozen['runtime_root']
            and Path(sys.executable).samefile(a.runtime_root / 'venv/bin/python'), 'Wrong inference interpreter/root')
    snap = frozen['library_runtime_snapshot']
    require(platform.python_version() == snap['python'] and sha_file(sys.executable) == snap['python_executable_sha256'],
            'Frozen Python runtime changed')
    for package, version in snap['packages'].items():
        if version is not None:
            require(importlib.metadata.version(package) == version, 'Package version changed: ' + package)
    import torch
    checkpoint_manifest = read_json(a.checkpoint_manifest)
    require(Path(torch.hub.get_dir())/'checkpoints' == Path(checkpoint_manifest['cache_root']),
            'Torch cache override would change checkpoint resolution')
    # Bind every implementation/config file in inference packages, plus clean Beat This commit.
    for package in ('allin1_infer', 'demucs_infer', 'beat_this'):
        spec = importlib.util.find_spec(package)
        require(spec is not None and spec.origin is not None, 'Missing runtime package: ' + package)
        root = Path(spec.origin).resolve().parent
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.toml', '.json'):
                checks[str(path)] = sha_file(regular(path))
        if package == 'beat_this':
            repo = next((p for p in (root, *root.parents) if (p / '.git').exists()), None)
            require(repo is not None, 'Beat This source repository missing')
            head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], text=True)
            require(head == frozen['beat_this']['repository_commit'] and not dirty, 'Beat This revision/worktree changed')
    for path in (a.runtime_contract, a.checkpoint_manifest, Path(sys.executable).resolve(),
                 a.runtime_root/'venv/bin/all-in-one-infer', a.runtime_root/'venv/bin/beat_this'):
        checks[str(path)] = sha_file(path)
    return checks


def validate_gpus(aio, beats):
    for gpus in (aio, beats):
        require(bool(gpus) and len(gpus) == len(set(gpus)) and all(type(g) is int and g >= 0 for g in gpus),
                'Explicit nonempty unique GPU indices required')
    require(not set(aio).intersection(beats), 'Inference GPU stages must be disjoint')
    require(not set(aio + beats).intersection(range(6)), 'GPUs0-5 reserved for original1604 run')


def input_records(rows):
    records = {r['standardized_path']: file_record(r['standardized_path']) for r in rows}
    for row in rows:
        require(records[row['standardized_path']] == {'sha256': row['standardized_file_sha256'],
                'bytes': int(row['standardized_file_bytes'])}, 'Inference input changed')
    return records


def expected_products(stage, rows, output):
    return {str(p) for r in rows for p in original.product_paths(stage, r['item_id'], output)}


def validate_receipt(receipt, contract, contract_sha, stage, index, rows, output, decode=False):
    check_seal(receipt)
    measurement_only(receipt)
    expected_cmd = original.command(stage, [r['standardized_path'] for r in rows], Path(contract['runtime_root']), output)
    require(receipt.get('status') == 'passed' and receipt.get('exit_code') == 0
            and receipt.get('run_contract_sha256') == contract_sha and receipt.get('stage') == stage
            and receipt.get('shard_index') == index and receipt.get('item_ids') == [r['item_id'] for r in rows]
            and receipt.get('command') == expected_cmd
            and receipt.get('gpu') == contract[stage+'_gpus'][index % len(contract[stage+'_gpus'])]
            and receipt.get('dependencies_sha256') == contract['dependencies_sha256'], 'Receipt execution identity mismatch')
    require(receipt.get('inputs') == input_records(rows), 'Receipt input hashes changed')
    paths = expected_products(stage, rows, output)
    require(set(receipt.get('outputs', {})) == paths, 'Receipt exact output union mismatch')
    for path, expected in receipt['outputs'].items():
        require(file_record(path) == expected, 'Receipted product tampered: ' + path)
    log = output / 'logs' / f'{stage}_{index:02d}.log'
    require(receipt.get('log_path') == str(log) and receipt.get('log') == file_record(log), 'Receipt log changed/missing')
    if decode:
        require(original.verify_products(stage, rows, output) == receipt['outputs'], 'Product decode/hash changed')


def validate_inventory(output, rows, completed):
    """Every product must belong to exactly one verified receipt, including resume."""
    expected = set().union(*(expected_products(stage, part, output) for stage, part in completed)) if completed else set()
    actual = set()
    for name in ('demix', 'structure', 'spec', 'beats'):
        for path in (output / name).rglob('*'):
            require(not path.is_symlink(), 'Symlink output forbidden')
            if path.is_file():
                actual.add(str(path))
    require(actual == expected, 'Unreceipted/missing/unexpected output products')


def verify_all_receipts(output, contract, rows, shards, require_complete=False, decode=False):
    contract_sha = sha_file(output / 'run_contract.json')
    by_path = {r['standardized_path']: r for r in rows}
    completed, receipts = [], {}
    for stage in STAGES:
        for index, inputs in shards:
            part = [by_path[p] for p in inputs]
            path = output / 'receipts' / f'{stage}_{index:02d}.json'
            if not path.exists():
                require(not require_complete, 'Missing stage-shard receipt: ' + str(path))
                continue
            validate_receipt(read_json(regular(path)), contract, contract_sha, stage, index, part, output, decode)
            completed.append((stage, part))
            receipts[str(path)] = sha_file(path)
    require({str(p) for p in (output / 'receipts').iterdir()} == set(receipts), 'Unexpected receipt inventory')
    validate_inventory(output, rows, completed)
    expected_logs = {str(output/'logs'/f'{read_json(p)["stage"]}_{read_json(p)["shard_index"]:02d}.log') for p in receipts}
    require({str(p) for p in (output/'logs').iterdir()} == expected_logs, 'Unreceipted/unexpected logs; manual review required')
    return receipts


def verify_completion(output, contract, rows, shards, decode=False):
    completion = read_json(regular(output / 'completion.json'))
    check_seal(completion)
    measurement_only(completion)
    require(completion.get('status') == 'passed' and completion.get('rows') == COUNT
            and completion.get('completed_stage_shards') == 42
            and completion.get('run_contract_sha256') == sha_file(output/'run_contract.json'), 'Incomplete500 inference')
    receipts = verify_all_receipts(output, contract, rows, shards, require_complete=True, decode=decode)
    require(completion.get('receipts_sha256') == receipts
            and completion.get('dependencies_sha256') == contract['dependencies_sha256'], 'Completion binding mismatch')
    return completion


def build_contract(a, rows, shards, provenance, runtime, dependencies):
    return sealed(dict(scope(), schema_version=1, rows=COUNT, prepared_dir=str(a.prepared_dir),
        output_root=str(a.output_root), runtime_root=str(a.runtime_root),
        item_ids=[r['item_id'] for r in rows], preparation_sha256=provenance,
        dependencies_sha256=dependencies, runtime_sha256=runtime,
        allinone_gpus=a.aio_gpus, beats_gpus=a.beat_gpus,
        shards=[{'index':i, 'inputs':p} for i,p in shards],
        commands={s:original.command(s, ['<frozen shard inputs>'], a.runtime_root, a.output_root) for s in STAGES},
        resume_policy='verify full receipts and inventories; reject all unreceipted products/logs',
        recovery_authorized=False, neural_reproducibility='published output hashes; unchanged unseeded Demucs shifts=1'))


def run_locked(a):
    validate_gpus(a.aio_gpus, a.beat_gpus)
    require(a.output_root == a.prepared_dir / 'inference' and a.output_root.resolve() == a.output_root,
            'Only dedicated Mureka prepared/inference output permitted')
    dependencies = dependency_snapshot()
    rows, shards, provenance = validate_prepared(a.prepared_dir)
    runtime = runtime_snapshot(a)
    contract = build_contract(a, rows, shards, provenance, runtime, dependencies)
    publish(a.output_root / 'run_contract.json', json_bytes(contract))
    contract_sha = sha_file(a.output_root / 'run_contract.json')
    for name in ('structure', 'demix', 'spec', 'beats', 'receipts', 'logs'):
        (a.output_root / name).mkdir(exist_ok=True)
    verify_all_receipts(a.output_root, contract, rows, shards, decode=True)
    if (a.output_root / 'completion.json').exists():
        verify_completion(a.output_root, contract, rows, shards)
        return
    by_path = {r['standardized_path']:r for r in rows}
    stop = threading.Event()

    def check_bindings():
        require(dependency_snapshot() == dependencies, 'Code dependencies changed during inference')
        for path, digest in {**provenance, **runtime}.items():
            require(sha_file(path) == digest, 'Bound provenance/runtime changed: ' + path)

    def worker(stage, gpu, assigned):
        for index, inputs in assigned:
            if stop.is_set():
                return
            selected = [by_path[p] for p in inputs]
            receipt_path = a.output_root/'receipts'/f'{stage}_{index:02d}.json'
            try:
                if receipt_path.exists():
                    validate_receipt(read_json(receipt_path), contract, contract_sha, stage, index, selected, a.output_root)
                    continue
                check_bindings()
                before = input_records(selected)
                require(not any(Path(p).exists() for p in expected_products(stage, selected, a.output_root)),
                        'Unreceipted product; manual review required')
                original.require_idle_5090(gpu)
                cmd = original.command(stage, inputs, a.runtime_root, a.output_root)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                log_path = a.output_root/'logs'/f'{stage}_{index:02d}.log'
                started = now()
                print(f'start stage={stage} shard={index} gpu={gpu} rows={len(inputs)}', flush=True)
                with log_path.open('x') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                # A suppressed CLI failure or missing .beats is fatal. No recovery adapter is called.
                outputs = original.verify_products(stage, selected, a.output_root)
                require(input_records(selected) == before, 'Inputs changed during inference')
                check_bindings()
                receipt = sealed(dict(scope(), status='passed', exit_code=0, stage=stage, shard_index=index,
                    run_contract_sha256=contract_sha, item_ids=[r['item_id'] for r in selected], gpu=gpu, command=cmd,
                    started_utc=started, completed_utc=now(), inputs=before, outputs=outputs,
                    log_path=str(log_path), log=file_record(log_path), dependencies_sha256=dependencies))
                publish(receipt_path, json_bytes(receipt))
                print(f'passed stage={stage} shard={index}', flush=True)
            except Exception:
                stop.set()
                raise

    work = [(stage, gpu, assigned) for stage, gpus in (('allinone', a.aio_gpus), ('beats', a.beat_gpus))
            for gpu, assigned in original.assignments(shards, gpus)]
    with ThreadPoolExecutor(max_workers=len(work)) as pool:
        futures = [pool.submit(worker, *job) for job in work]
        for future in futures:
            future.result()
    check_bindings()
    require(runtime_snapshot(a) == runtime, 'Runtime changed during inference')
    after_rows, after_shards, after_provenance = validate_prepared(a.prepared_dir)
    require((after_rows, after_shards, after_provenance) == (rows, shards, provenance), 'Prepared data changed')
    receipts = verify_all_receipts(a.output_root, contract, rows, shards, require_complete=True)
    publish(a.output_root/'completion.json', json_bytes(sealed(dict(scope(), status='passed', rows=COUNT,
        completed_stage_shards=42, run_contract_sha256=contract_sha, receipts_sha256=receipts,
        dependencies_sha256=dependencies))))
    verify_completion(a.output_root, contract, rows, shards)


def common_arguments(parser):
    for name in ('prepared-dir', 'runtime-root', 'runtime-contract', 'checkpoint-manifest', 'old-code-root', 'bias'):
        parser.add_argument('--'+name, type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--aio-gpus', nargs='+', type=int, required=True)
    parser.add_argument('--beat-gpus', nargs='+', type=int, required=True)
    a = parser.parse_args()
    validate_gpus(a.aio_gpus, a.beat_gpus)
    prep.validate_paths(prep.SOURCE, a.prepared_dir)
    require(a.output_root == a.prepared_dir/'inference' and a.output_root.resolve() == a.output_root,
            'Only dedicated Mureka inference output permitted')
    a.output_root.mkdir(parents=True, exist_ok=True)
    with (a.output_root/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(a)
    print('Mureka500 measurement inference passed; no scoring or fitting', flush=True)


if __name__ == '__main__':
    main()
