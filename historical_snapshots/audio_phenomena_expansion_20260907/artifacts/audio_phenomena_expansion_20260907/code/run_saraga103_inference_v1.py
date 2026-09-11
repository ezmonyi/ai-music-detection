#!/usr/bin/env python3
"""Saraga103 old S/D/R/P measurement adapter; independent draft/freeze/run gate.

Only immutable identity aliases and orchestration differ from Mureka v2.
Numerical/model helpers remain hash-pinned and unchanged. Never scores/fits.
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

import materialize_saraga_external103_v1 as materializer
from types import SimpleNamespace
import hashlib
import numpy as np
import soundfile as sf
from prepare_mureka60_inputs_v2 import canonical_hash, json_bytes, publish, read_json, require, sha_file
import run_equal60_inference_batches as original
from verify_extract_equal60 import inspect_audio, verify_runtime
import saraga103_cuda_guard_v1 as cuda_guard

ROLE = 'external_human_unscored'
SOURCE = 'human_saraga_hindustani_v1'
COUNT, SHARDS = 103, 5
PREPARED = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/saraga_external103_intervals_v1')
MEASUREMENT_FREEZE = 'preregistration/saraga_external103_measurement_frozen_v1.json'
MEASUREMENT_SHA = '0cf1e78ccdb29e63c421a46bae48e2da76ef0f3116aad442a3a3d91d6e044440'
MEASUREMENT_CONTRACT_SHA = '79fbbf5ab9711a52d29d187b210678fcb2613186eb1cf8abb55a2f96370e52df'
STAGE = 'saraga_external103_old4_measurement_only'
STAGES = ('allinone', 'beats')
GUARD = Path(__file__).resolve().with_name('saraga103_cuda_guard_v1.py')
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
                scores_generated=False, classifier_admission_authorized=False, evaluation_allowed=False)


def dependency_snapshot():
    root = Path(__file__).resolve().parent
    for name, expected in PINNED_HELPERS.items():
        require(sha_file(regular(root / name)) == expected, 'Frozen helper changed: ' + name)
    names = [*PINNED_HELPERS, 'prepare_mureka60_inputs_v2.py',
             'materialize_saraga_external103_v1.py', 'prepare_saraga_external_cohort_v1.py',
             'saraga_interval_audio_v1.py', 'run_saraga103_inference_v1.py',
             'verify_extract_saraga103_v1.py', 'saraga103_cuda_guard_v1.py']
    return {str(root / name): sha_file(regular(root / name)) for name in names}


def validate_paths(a):
    require(a.prepared_dir == PREPARED and a.prepared_dir.resolve() == a.prepared_dir,
            'Only fixed accepted Saraga103 prepared directory allowed')
    require(a.output_root == PREPARED.parent / 'saraga_external103_old4_v1'
            and a.output_root.resolve() == a.output_root,
            'Only isolated Saraga old4 v1 output namespace allowed')
    materializer.path_ok(a.root)
    materializer.path_ok(a.output_root)


def read_csv(path):
    with regular(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def validate_row(row, native, selected, saved, prepared):
    item = selected['item_id']
    proof = saved['interval_proof']
    expected = dict(item_id=item, id=item, label='0', source_id=SOURCE,
        source_group=SOURCE, group_id=selected['group_id'], role=ROLE,
        standardized_path=str(prepared / 'items' / item / 'audio.wav'),
        audio_path=str(prepared / 'items' / item / 'audio.wav'),
        audio_offset_s='0', requires_crop='0', duration='60', duration_sec='60',
        native_sample_rate_hz=selected['native_sample_rate_hz'],
        native_sr=selected['native_sample_rate_hz'],
        original_source_audio_path=selected['source_audio_path'],
        original_source_audio_sha256=selected['acquisition_raw_sha256'],
        standardized_sr='44100', standardized_frames='2646000', standardized_channels='2',
        standardized_file_sha256=saved['wav']['file_sha256'],
        evaluation_allowed='False', classifier_admission_authorized='False')
    require(row == expected, 'Inference row differs from frozen identity/representation')
    require(native == {k: str(v) for k, v in materializer.native_row(selected, saved).items()},
            'Native metadata differs from accepted interval')
    sr = int(selected['native_sample_rate_hz'])
    count = 60 * sr
    start = (int(selected['actual_eof_frames']) - count) // 2
    require(sr in (44100, 48000) and count <= int(selected['actual_eof_frames'])
            and proof['native_sample_rate_hz'] == sr
            and proof['observed_actual_frames'] == int(selected['actual_eof_frames'])
            and proof['physical_header_frames'] == int(selected['soundfile_header_frames'])
            and proof['crop_start_frame'] == start and proof['crop_frames'] == count
            and proof['crop_end_frame_exclusive'] == start + count
            and round(float(selected['source_offset_seconds']) * sr) == start,
            'Exact native center coordinates/EOF/header mismatch')
    require(proof['raw_hashes_before'] == proof['raw_hashes_after']
            and proof['raw_hashes_after']['sha256'] == selected['acquisition_raw_sha256']
            and saved['role'] == ROLE and saved['classifier_admission'] is False
            and saved['item_id'] == item
            and saved['status'] == 'passed_interval_materialization_not_classifier_admission'
            and saved['selection_row_sha256'] == materializer.util.seal(selected)
            and saved['contract_sha256'] == MEASUREMENT_CONTRACT_SHA,
            'Accepted proof identity/raw/role/contract mismatch')


def validate_prepared(a, decode=True):
    """Current full103 commits, frozen inputs, raw hashes, WAVs and sample proof."""
    validate_paths(a)
    prepared = a.prepared_dir
    bindings = {}
    def bind(path, expected=None):
        path = regular(path)
        digest = sha_file(path)
        require(expected is None or digest == expected, 'Bound input changed: ' + str(path))
        bindings[str(path)] = digest
        return path
    frozen = read_json(bind(a.root / MEASUREMENT_FREEZE, MEASUREMENT_SHA))
    materializer.verify_publication(prepared, allow_extra=True)
    bind(prepared / 'COMMIT.json')
    bind(prepared / 'frozen_measurement_contract.json', MEASUREMENT_SHA)
    c = frozen['contract']
    require(frozen['status'] == 'frozen' and frozen['authorized_stage'] == materializer.STAGE
            and frozen['contract_sha256'] == materializer.util.seal(c) == MEASUREMENT_CONTRACT_SHA
            and c['output'] == str(prepared) and c['synthetic_test_only'] is False,
            'Wrong interval measurement freeze')
    review = frozen['independent_review']
    require(review['approved'] is True and review['reviewer'] == 'root', 'Missing root interval review')
    bind(a.root / review['evidence'], review['evidence_sha256'])
    for relative, digest in c['inputs_sha256'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe binding')
        bind(a.root / relative, digest)
    for name, digest in c['code_sha256'].items():
        bind(a.root / 'code' / name, digest)
    materializer.verify_publication(prepared / 'final')
    for path in (prepared / 'final').iterdir():
        bind(path)
    summary = read_json(prepared / 'final/materialization_summary.json')
    require(summary['status'] == 'passed_all_interval_materialization_not_classifier_admission'
            and summary['contract_sha256'] == MEASUREMENT_CONTRACT_SHA
            and summary['expected'] == summary['passed'] == COUNT
            and summary['failed'] == 0 and summary['failed_ids'] == []
            and summary['final_passed_source_and_output_hash_recheck'] is True
            and summary['classifier_admission'] is False
            and summary['synthetic_test_only'] is False, 'Full103 interval acceptance required')
    require((prepared / 'failures').is_dir() and not list((prepared / 'failures').iterdir()),
            'Recorded interval failures forbid admission')
    rows = read_csv(prepared / 'final/inference_manifest.csv')
    native = read_csv(prepared / 'final/native_metadata_60s.csv')
    ids = c['selected_ids']
    require(len(ids) == COUNT and len(set(ids)) == COUNT and ids == sorted(ids)
            and [r['item_id'] for r in rows] == ids
            and [r['id'] for r in native] == ids
            and [r['item_id'] for r in c['rows']] == ids, 'All103 fixed identities/order required')
    require({p.name for p in (prepared / 'items').iterdir()} == set(ids), 'Item union mismatch')
    inventory = read_json(prepared / 'final/item_proof_inventory.json')
    require(set(inventory) == set(ids), 'Proof inventory mismatch')
    for row, n, selected in zip(rows, native, c['rows']):
        directory = prepared / 'items' / row['item_id']
        materializer.verify_publication(directory)
        require({p.name for p in directory.iterdir()} == {'audio.wav', 'proof.json', 'COMMIT.json'},
                'Unexpected interval product inventory')
        bind(directory / 'COMMIT.json')
        saved = read_json(bind(directory / 'proof.json', inventory[row['item_id']]))
        validate_row(row, n, selected, saved, prepared)
        receipt = read_json(a.root / materializer.util.PHYSICAL / 'receipts' / (selected['mbid'] + '.json'))
        materializer.check_proof(selected, receipt['record'], saved['interval_proof'])
        bind(Path(selected['source_audio_path']), selected['acquisition_raw_sha256'])
        require(file_record(selected['source_audio_path']) == {
            k: saved['interval_proof']['raw_hashes_after'][k] for k in ('sha256', 'bytes')},
            'Original raw bytes changed')
        audio = bind(directory / 'audio.wav', saved['wav']['file_sha256'])
        require(audio.stat().st_size == saved['wav']['bytes'], 'WAV size changed')
        if decode:
            observed = materializer.verify_written(audio, SimpleNamespace(
                sf=sf, np=np, _sample_sha=lambda x, dtype: hashlib.sha256(
                    np.asarray(x, dtype=dtype).tobytes(order='C')).hexdigest()),
                saved['interval_proof']['standardized_float32_sha256'])
            require(observed == saved['wav'], 'WAV sample identity changed')
        row['standardized_file_bytes'] = str(saved['wav']['bytes'])
    require(sum(r['native_sr'] == '44100' for r in rows) == 96
            and sum(r['native_sr'] == '48000' for r in rows) == 7, 'Native rate counts changed')
    return rows, bindings


def adapted_rows(rows, output):
    """Unique basename aliases are essential: every accepted source is audio.wav."""
    adapted = []
    for row in rows:
        require(Path(row['item_id']).name == row['item_id'] and row['item_id'] not in ('.', '..'),
                'Unsafe alias identity')
        alias = output / 'inputs' / (row['item_id'] + '.wav')
        adapted.append(dict(row, accepted_standardized_path=row['standardized_path'],
                            standardized_path=str(alias), audio_path=str(alias)))
    require(len({r['standardized_path'] for r in adapted}) == len(rows), 'Duplicate alias')
    return adapted


def shards_for(rows):
    return [(i // 24, [r['standardized_path'] for r in rows[i:i + 24]])
            for i in range(0, len(rows), 24)]


def validate_aliases(output, rows):
    materializer.verify_publication(output / 'inputs')
    expected = {r['item_id'] + '.wav' for r in rows} | {'inference_manifest.csv', 'COMMIT.json'}
    require({p.name for p in (output / 'inputs').iterdir()} == expected, 'Alias inventory differs')
    require((output / 'inputs/inference_manifest.csv').read_bytes() == materializer.csv_bytes(rows),
            'Adapted manifest differs from current exact rows')
    for row in rows:
        alias = regular(row['standardized_path'])
        source = regular(row['accepted_standardized_path'])
        require(alias.samefile(source) and file_record(alias) == {
            'sha256': row['standardized_file_sha256'], 'bytes': int(row['standardized_file_bytes'])},
            'Alias is not the exact accepted hardlink')
    return input_records(rows)


def publish_aliases(output, rows):
    destination = output / 'inputs'
    if destination.exists():
        validate_aliases(output, rows)
        return
    destination.mkdir()
    for row in rows:
        source = regular(row['accepted_standardized_path'])
        require(file_record(source) == {'sha256': row['standardized_file_sha256'],
                                      'bytes': int(row['standardized_file_bytes'])}, 'Alias source changed')
        os.link(source, destination / (row['item_id'] + '.wav'), follow_symlinks=False)
    publish(destination / 'inference_manifest.csv', materializer.csv_bytes(rows))
    materializer.commit_directory(destination, {p.name for p in destination.iterdir()})
    validate_aliases(output, rows)



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
                 a.runtime_root/'venv/bin/all-in-one-infer', a.runtime_root/'venv/bin/beat_this', GUARD):
        checks[str(path)] = sha_file(path)
    return checks


def validate_gpus(aio, beats):
    for gpus in (aio, beats):
        require(bool(gpus) and len(gpus) == len(set(gpus)) and all(type(g) is int and g >= 0 for g in gpus),
                'Explicit nonempty unique GPU indices required')
    require(not set(aio).intersection(beats), 'Inference GPU stages must be disjoint')
    require(0 not in aio + beats, 'GPU0 is excluded from this bounded Saraga run')


def input_records(rows):
    records = {r['standardized_path']: file_record(r['standardized_path']) for r in rows}
    for row in rows:
        require(records[row['standardized_path']] == {'sha256': row['standardized_file_sha256'],
                'bytes': int(row['standardized_file_bytes'])}, 'Inference input changed')
    return records


def expected_products(stage, rows, output):
    return {str(p) for r in rows for p in original.product_paths(stage, r['item_id'], output)}


def guard_path(output, stage, index):
    return output / 'logs' / f'{stage}_{index:02d}.cuda_guard.json'


def guarded_command(stage, inputs, runtime, output, gpu, index):
    command = original.command(stage, inputs, runtime, output)
    return [str(runtime / 'venv/bin/python'), str(GUARD), '--stage', stage,
        '--physical-gpu', str(gpu), '--entrypoint', command[0],
        '--evidence', str(guard_path(output, stage, index)), '--', *command[1:]]


def validate_guard(output, stage, index, gpu, expected_command, contract):
    path = regular(guard_path(output, stage, index))
    evidence = read_json(path)
    require(evidence.get('status') == 'passed_same_process_cuda_guard'
            and evidence.get('guard_version') == cuda_guard.VERSION and evidence.get('stage') == stage
            and evidence.get('original_command') == expected_command
            and evidence.get('guard_sha256') == contract['dependencies_sha256'][str(GUARD)]
            and evidence.get('original_entrypoint_sha256') == contract['runtime_sha256'][expected_command[0]]
            and evidence.get('entrypoint_dispatched_same_process') is True
            and evidence.get('original_argv_preserved') is True
            and evidence.get('entrypoint_returned_successfully') is True
            and evidence.get('model_or_torch_monkeypatched') is False
            and evidence.get('cpu_fallback_authorized') is False,
            'Missing/invalid same-process CUDA guard execution evidence')
    device = evidence['before']
    require(device == evidence['after'] and device.get('cuda_available') is True
            and device.get('visible_device_count') == 1 and device.get('cuda_initialized') is True
            and device.get('allocation_device') == 'cuda:0'
            and device.get('allocation_and_synchronization_passed') is True
            and device.get('cuda_visible_devices') == str(gpu)
            and device.get('physical_gpu_index') == gpu and device.get('logical_device_index') == 0
            and 'RTX 5090' in device.get('device_name', '')
            and isinstance(device.get('device_total_memory_bytes'), int)
            and device['device_total_memory_bytes'] > 0,
            'CUDA guard device/assignment/allocation mismatch')
    process = evidence['process']
    require(type(process.get('pid')) is int and process['pid'] > 0 and bool(process.get('hostname'))
            and process.get('python_executable') == str(Path(sys.executable).resolve()),
            'CUDA guard process/runtime evidence mismatch')
    return file_record(path)


def validate_receipt(receipt, contract, contract_sha, stage, index, rows, output, decode=False):
    check_seal(receipt)
    measurement_only(receipt)
    expected_cmd = original.command(stage, [r['standardized_path'] for r in rows], Path(contract['runtime_root']), output)
    gpu = contract[stage + '_gpus'][index % len(contract[stage + '_gpus'])]
    guarded = guarded_command(stage, [r['standardized_path'] for r in rows],
                              Path(contract['runtime_root']), output, gpu, index)
    require(receipt.get('status') == 'passed' and receipt.get('exit_code') == 0
            and receipt.get('run_contract_sha256') == contract_sha and receipt.get('stage') == stage
            and receipt.get('shard_index') == index and receipt.get('item_ids') == [r['item_id'] for r in rows]
            and receipt.get('command') == guarded and receipt.get('original_command') == expected_cmd
            and receipt.get('gpu') == gpu
            and receipt.get('dependencies_sha256') == contract['dependencies_sha256'], 'Receipt execution identity mismatch')
    require(receipt.get('cuda_guard') == validate_guard(output, stage, index, gpu, expected_cmd, contract),
            'CUDA guard evidence hash changed')
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
    expected_logs.update(str(guard_path(output, read_json(p)['stage'], read_json(p)['shard_index'])) for p in receipts)
    require({str(p) for p in (output/'logs').iterdir()} == expected_logs, 'Unreceipted/unexpected logs; manual review required')
    return receipts


def verify_completion(output, contract, rows, shards, decode=False):
    completion = read_json(regular(output / 'completion.json'))
    check_seal(completion)
    measurement_only(completion)
    require(completion.get('status') == 'passed' and completion.get('rows') == COUNT
            and completion.get('completed_stage_shards') == 10
            and completion.get('run_contract_sha256') == sha_file(output/'run_contract.json'), 'Incomplete103 inference')
    receipts = verify_all_receipts(output, contract, rows, shards, require_complete=True, decode=decode)
    require(completion.get('receipts_sha256') == receipts
            and completion.get('dependencies_sha256') == contract['dependencies_sha256'], 'Completion binding mismatch')
    return completion


def build_contract(a, rows, shards, provenance, runtime, dependencies):
    return sealed(dict(scope(), schema_version=2, rows=COUNT, prepared_dir=str(a.prepared_dir),
        measurement_freeze_sha256=MEASUREMENT_SHA, measurement_contract_sha256=MEASUREMENT_CONTRACT_SHA,
        adapted_manifest_sha256=hashlib.sha256(materializer.csv_bytes(rows)).hexdigest(),
        measurement_workers=a.workers, source_id=SOURCE, label=0,
        output_root=str(a.output_root), runtime_root=str(a.runtime_root),
        item_ids=[r['item_id'] for r in rows], preparation_sha256=provenance,
        dependencies_sha256=dependencies, runtime_sha256=runtime,
        allinone_gpus=a.aio_gpus, beats_gpus=a.beat_gpus,
        shards=[{'index':i, 'inputs':p} for i,p in shards],
        cuda_guard_version=cuda_guard.VERSION, cuda_guard_sha256=dependencies[str(GUARD)],
        original_commands={s:original.command(s, ['<frozen shard inputs>'], a.runtime_root, a.output_root) for s in STAGES},
        commands={stage: {str(index): guarded_command(stage, inputs, a.runtime_root, a.output_root,
                     gpus[index % len(gpus)], index) for index, inputs in shards}
                  for stage, gpus in (('allinone', a.aio_gpus), ('beats', a.beat_gpus))},
        resume_policy='verify full receipts and inventories; reject all unreceipted products/logs',
        recovery_authorized=False, neural_reproducibility='published output hashes; unchanged unseeded Demucs shifts=1'))


def run_locked(a):
    validate_gpus(a.aio_gpus, a.beat_gpus)
    validate_paths(a)
    contract, rows, shards = current_contract(a, decode=True)
    verify_authorization(a, contract)
    dependencies = contract['dependencies_sha256']
    provenance = contract['preparation_sha256']
    runtime = contract['runtime_sha256']
    publish_aliases(a.output_root, rows)
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
        validate_aliases(a.output_root, rows)
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
                original_cmd = original.command(stage, inputs, a.runtime_root, a.output_root)
                cmd = guarded_command(stage, inputs, a.runtime_root, a.output_root, gpu, index)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                log_path = a.output_root/'logs'/f'{stage}_{index:02d}.log'
                started = now()
                print(f'start stage={stage} shard={index} gpu={gpu} rows={len(inputs)}', flush=True)
                with log_path.open('x') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                guard_record = validate_guard(a.output_root, stage, index, gpu, original_cmd, contract)
                # A suppressed CLI failure or missing .beats is fatal. No recovery adapter is called.
                outputs = original.verify_products(stage, selected, a.output_root)
                require(input_records(selected) == before, 'Inputs changed during inference')
                check_bindings()
                receipt = sealed(dict(scope(), status='passed', exit_code=0, stage=stage, shard_index=index,
                    run_contract_sha256=contract_sha, item_ids=[r['item_id'] for r in selected], gpu=gpu, command=cmd,
                    original_command=original_cmd, cuda_guard=guard_record,
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
    after_contract, after_rows, after_shards = current_contract(a, decode=True)
    validate_aliases(a.output_root, after_rows)
    require((after_contract, after_rows, after_shards) == (contract, rows, shards), 'Prepared data changed')
    receipts = verify_all_receipts(a.output_root, contract, rows, shards, require_complete=True)
    publish(a.output_root/'completion.json', json_bytes(sealed(dict(scope(), status='passed', rows=COUNT,
        completed_stage_shards=10, run_contract_sha256=contract_sha, receipts_sha256=receipts,
        dependencies_sha256=dependencies))))
    verify_completion(a.output_root, contract, rows, shards)


def current_contract(a, decode=True):
    validate_gpus(a.aio_gpus, a.beat_gpus)
    dependencies = dependency_snapshot()
    accepted, provenance = validate_prepared(a, decode=decode)
    rows = adapted_rows(accepted, a.output_root)
    shards = shards_for(rows)
    require(len(shards) == SHARDS and [len(p) for _, p in shards] == [24, 24, 24, 24, 7],
            'Fixed103 shard membership differs')
    runtime = runtime_snapshot(a)
    require(type(a.workers) is int and a.workers > 0, 'Positive extraction workers required')
    return build_contract(a, rows, shards, provenance, runtime, dependencies), rows, shards


def verify_authorization(a, contract):
    require(a.frozen is not None and a.frozen_sha256 is not None, 'Independent old4 freeze required')
    require(sha_file(regular(a.frozen)) == a.frozen_sha256, 'Old4 frozen wrapper hash mismatch')
    frozen = read_json(a.frozen)
    require(frozen.get('status') == 'frozen' and frozen.get('authorized_stage') == STAGE
            and frozen.get('contract') == contract
            and frozen.get('contract_sha256') == contract['canonical_sha256'],
            'Old4 freeze scope/current contract mismatch')
    review = frozen['independent_review']
    require(review['approved'] is True and review['reviewer'] == 'root', 'Independent root review required')
    relative = Path(review['evidence'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe review path')
    require(sha_file(regular(a.root / relative)) == review['evidence_sha256'], 'Root review evidence changed')
    return frozen


def common_arguments(parser):
    for name in ('root', 'prepared-dir', 'runtime-root', 'runtime-contract',
                 'checkpoint-manifest', 'old-code-root', 'bias'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('draft', 'run'))
    common_arguments(parser)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--aio-gpus', nargs='+', type=int, required=True)
    parser.add_argument('--beat-gpus', nargs='+', type=int, required=True)
    parser.add_argument('--draft-dir', type=Path)
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--frozen-sha256')
    a = parser.parse_args()
    validate_paths(a)
    if a.stage == 'draft':
        require(a.draft_dir is not None and not a.output_root.exists(), 'Fresh output and draft destination required')
        contract, _, _ = current_contract(a, decode=True)
        materializer.publish_directory(a.draft_dir, {'old4_contract_draft.json': json_bytes({
            'status': 'draft', 'authorized_stage': STAGE, 'contract': contract,
            'contract_sha256': contract['canonical_sha256'],
            'neural_inference_started': False, 'classifier_admission_authorized': False})})
        print('Saraga103 old4 draft only; no neural inference/extraction/scoring')
        return
    # Validate authorization before reserving any execution output.
    contract, _, _ = current_contract(a, decode=True)
    verify_authorization(a, contract)
    a.output_root.mkdir(exist_ok=True)
    with (a.output_root / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(a)
    print('Saraga103 old4 inference complete; extraction still requires strict audit')


if __name__ == '__main__':
    main()
