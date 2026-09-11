#!/usr/bin/env python3
"""Integrity-hardened native30 All-In-One and Beat This batch inference.

This version preserves v1 as a pinned dependency and strengthens receipt,
runtime-resolution and final-evidence validation.  Default preflight opens no
cohort audio.  Run still requires a separately published parent freeze and is
not a Beat This recovery mechanism.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


VERSION = 'run_native30_inference_batches_v2'
V1_SHA = '84dce45426596888b3a4e3b20e4ad1c61dc976645506f269c7ec7990b12f3933'
FREEZE_VERSION = 'native30-inference-parent-freeze-v2'
FREEZE_STATUS = 'parent_frozen_for_native30_neural_inference'
SUCCESS_SCOPE = {'classifier_fits': 0, 'cohort_admitted': False,
                 'feature_extraction_authorized': False, 'recovery_applied': False}
TORCH_HOME = '/home/yi/.cache/torch'
TORCH_HUB = '/home/yi/.cache/torch/hub'
CHILD_BASE_ENVIRONMENT = {
    'LANG': 'C.UTF-8',
    'LC_ALL': 'C.UTF-8',
    'TORCH_HOME': TORCH_HOME,
    'OMP_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '2',
    'MKL_NUM_THREADS': '2',
}
CLEARED_CHILD_ENVIRONMENT = ['HF_HOME', 'HOME', 'LD_LIBRARY_PATH', 'PYTHONHOME', 'PYTHONPATH',
                             'TMPDIR', 'XDG_CACHE_HOME']
EXTRA_RESOLUTION_CODE = {
    'repos/all-in-one-infer/src/allin1_infer/models/loaders.py': 'ddb2c76203b8da23386f97aad0766c7bf721d226a9fb5dfecefdc21ed11282d6',
    'repos/all-in-one-infer/src/allin1_infer/checkpoints.py': '4c567e2db33a6cf8f8d48a432b25b6ebadbd3023bef788e1bbb098347e07b887',
    'repos/all-in-one-infer/src/allin1_infer/config/checkpoints.toml': '5ea1427404e47ff691255fd902fc07f83bf9bf889bc1b1813d95ad00fb748f4a',
    'venv/lib/python3.11/site-packages/demucs_infer/pretrained.py': '8e0488acd2b41462facee84342e1745607e2164932183c58f18cdbf876ce5a98',
    'venv/lib/python3.11/site-packages/demucs_infer/repo.py': 'a04b871cff21a6b042b2fc3717e827af95d6e2fc15ddecaa1348fe3a6f5369c6',
    'venv/lib/python3.11/site-packages/demucs_infer/remote/files.txt': '7258eee911e5963e6270983d7072832e90e1ed96b25bb26aadf12f9867109bf9',
    'venv/lib/python3.11/site-packages/demucs_infer/remote/htdemucs.yaml': '239c445d0b14454d541ad8bd9bb271c9e536d267e8a4625208744cbb2e7bb66c',
}


def _load_v1():
    path = Path(__file__).resolve().with_name('run_native30_inference_batches_v1.py')
    if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != V1_SHA:
        raise RuntimeError('pinned native30 inference v1 dependency changed')
    module = importlib.import_module('run_native30_inference_batches_v1')
    if Path(module.__file__).resolve() != path:
        raise RuntimeError('native30 inference v1 import path changed')
    return module


b = _load_v1()


def expected_child_environment(runtime, gpu):
    runtime = b.safe_path(runtime)
    return {**CHILD_BASE_ENVIRONMENT,
        'PATH': str(runtime / 'venv/bin') + ':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        'CUDA_VISIBLE_DEVICES': str(gpu),
    }


def child_environment(runtime, gpu):
    """Exact child environment; no parent cache, import, or loader paths leak in."""
    return expected_child_environment(runtime, gpu)


def child_environment_policy(runtime):
    environment = expected_child_environment(runtime, 0)
    environment.pop('CUDA_VISIBLE_DEVICES')
    return {'exact_base': environment,
            'per_shard': {'CUDA_VISIBLE_DEVICES': 'assigned physical GPU decimal'},
            'inherited': [], 'cleared': CLEARED_CHILD_ENVIRONMENT}


def bind_canonical_json(path, expected):
    """Bind a committed JSON value to both its parsed value and canonical bytes."""
    b.require(b.read_json(path) == expected, 'committed JSON value changed: ' + str(path))
    record = b.binding(path)
    b.require(record['sha256'] == b.value_hash(expected), 'committed JSON encoding/hash changed: ' + str(path))
    return record


def validate_freeze(path, expected_sha, *, expected=b.EXPECTED, helper_loader=b.load_cohort_helper,
                    plan_sha=b.PLAN_SHA, screen_sha=b.SCREEN_SHA):
    path = b.safe_path(path)
    b.require(b.hash_string(expected_sha) and b.digest(path) == expected_sha, 'caller parent-freeze SHA mismatch')
    freeze = b.read_json(path)
    b.require(freeze.get('version') == FREEZE_VERSION and freeze.get('status') == FREEZE_STATUS, 'v2 parent freeze schema/status')
    b.require(all(freeze.get(key) == value for key, value in SUCCESS_SCOPE.items() if key != 'recovery_applied'), 'parent freeze scope')
    b.require(freeze.get('duration_s') == 30 and freeze.get('input_format') ==
              {'format': 'WAV', 'subtype': 'FLOAT', 'sample_rate_hz': 44100, 'channels': 2, 'frames': 1323000},
              'frozen input format')
    for key in ('cohort_contract', 'cohort_helper', 'plan', 'screen', 'runner', 'tests', 'v1_dependency'):
        b._binding_matches(freeze[key], audio=False)
    b.require(freeze['runner'] == b.binding(Path(__file__).resolve())
              and freeze['tests'] == b.binding(Path(__file__).resolve().with_name('test_' + VERSION + '.py')),
              'v2 runner/test not parent-pinned')
    b.require(freeze['v1_dependency'] == b.binding(Path(b.__file__).resolve())
              and freeze['v1_dependency']['sha256'] == V1_SHA, 'v1 dependency not parent-pinned')
    cohort = b.read_json(freeze['cohort_contract']['path'])
    b.require(b.value_hash(cohort) == freeze['cohort_contract']['sha256'], 'cohort contract canonical/hash binding')
    b.require(cohort.get('version') == 'run_native30_fhsc_cohort_v1'
              and cohort.get('status') == 'frozen_before_any_measurement_audio_reads'
              and cohort.get('expected_count') == expected['total'], 'prepared cohort schema/count')
    b.validate_rows(cohort['rows'], expected)
    b.require(cohort.get('input_format') == freeze['input_format']
              and cohort.get('source_counts') == expected['sources']
              and cohort.get('human') == expected['human'] and cohort.get('ai') == expected['ai'], 'cohort accounting')
    for row in cohort['rows']:
        b.require(cohort['bindings'].get(row['input']['path']) == row['input'], 'input absent from cohort graph')
    helper = helper_loader(freeze['cohort_helper'])
    b.verify_source_graph(freeze, cohort, helper, expected, plan_sha=plan_sha, screen_sha=screen_sha)
    b.validate_shards(freeze['shards'], cohort['rows'])
    output = b.safe_path(freeze['output_root'])
    b.require(output.parent.is_dir() and output != Path(cohort['output_root'])
              and not output.is_relative_to(Path(freeze['upstream_commits']['new']['path']).parent)
              and not output.is_relative_to(Path(freeze['upstream_commits']['prior']['path']).parent), 'unsafe output overlap')
    b.require(not output.exists() or (output.is_dir() and not output.is_symlink()), 'unsafe output root')
    aio, beats = freeze['aio_gpus'], freeze['beat_gpus']
    b.require(aio and beats and len(aio) == len(set(aio)) and len(beats) == len(set(beats))
              and not set(aio) & set(beats) and all(type(x) is int and x >= 0 for x in aio + beats), 'disjoint GPU sets')
    policy = freeze.get('child_environment')
    b.require(policy == child_environment_policy(freeze['runtime_root']), 'child environment policy not frozen')
    b.require(freeze.get('torch_hub_dir') == TORCH_HUB, 'Torch hub directory not frozen')
    return freeze, cohort, helper


def probe_child_runtime(runtime, gpu=0):
    runtime = b.safe_path(runtime)
    expected_origins = {
        'allin1_infer': runtime / 'repos/all-in-one-infer/src/allin1_infer/__init__.py',
        'beat_this': runtime / 'repos/beat_this/beat_this/__init__.py',
        'demucs_infer': runtime / 'venv/lib/python3.11/site-packages/demucs_infer/__init__.py',
    }
    probe = ('import importlib.util,json,os,shutil,sys,torch; '
             'names=("allin1_infer","beat_this","demucs_infer"); '
             'print(json.dumps({"environment":dict(os.environ),"python":sys.executable,'
             '"launchers":{n:shutil.which(n) for n in ("all-in-one-infer","beat_this")},'
             '"torch_hub_dir":torch.hub.get_dir(),"import_origins":'
             '{n:importlib.util.find_spec(n).origin for n in names}},sort_keys=True))')
    environment = child_environment(runtime, gpu)
    child = json.loads(subprocess.check_output([str(runtime / 'venv/bin/python'), '-c', probe],
                                               env=environment, text=True))
    child_origins = {name: str(Path(path).resolve()) for name, path in child['import_origins'].items()}
    launchers = {name: str(Path(path).resolve()) for name, path in child['launchers'].items()}
    b.require(child['torch_hub_dir'] == TORCH_HUB, 'child runtime Torch hub resolution changed')
    b.require(child_origins == {name: str(path.resolve()) for name, path in expected_origins.items()},
              'child runtime import origins changed')
    b.require(child['environment'] == environment, 'child process environment changed')
    b.require(Path(child['python']).resolve() == (runtime / 'venv/bin/python').resolve(), 'child Python changed')
    b.require(launchers == {name: str((runtime / 'venv/bin' / name).resolve())
                            for name in ('all-in-one-infer', 'beat_this')}, 'child CLI resolution changed')
    launcher_help = {}
    for name in ('all-in-one-infer', 'beat_this'):
        completed = subprocess.run([launchers[name], '--help'], env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        b.require(completed.stdout, 'empty child CLI help: ' + name)
        launcher_help[name] = {'bytes': len(completed.stdout),
                               'sha256': hashlib.sha256(completed.stdout).hexdigest()}
    return {'environment': environment, 'python': str(Path(child['python']).resolve()),
            'launchers': launchers, 'torch_hub_dir': child['torch_hub_dir'],
            'import_origins': child_origins, 'launcher_help': launcher_help}


def verify_runtime(freeze):
    evidence = b.verify_runtime(freeze)
    runtime = b.safe_path(freeze['runtime_root'])
    checked = dict(evidence['checked_sha256'])
    for relative, expected in EXTRA_RESOLUTION_CODE.items():
        path = runtime / relative
        checked[str(path)] = b.require_hash(path, expected)
    expected_origins = {
        'allin1_infer': runtime / 'repos/all-in-one-infer/src/allin1_infer/__init__.py',
        'beat_this': runtime / 'repos/beat_this/beat_this/__init__.py',
        'demucs_infer': runtime / 'venv/lib/python3.11/site-packages/demucs_infer/__init__.py',
    }
    parent_origins = {}
    for name, expected in expected_origins.items():
        spec = importlib.util.find_spec(name)
        b.require(spec and spec.origin and Path(spec.origin).resolve() == expected.resolve(), 'import origin mismatch: ' + name)
        parent_origins[name] = str(Path(spec.origin).resolve())
    import torch
    b.require(torch.hub.get_dir() == TORCH_HUB, 'parent runtime Torch hub resolution changed')
    child = probe_child_runtime(runtime)
    manifest = b.read_json(freeze['checkpoint_manifest']['path'])
    for row in manifest['checkpoints']:
        path = Path(row['path'])
        if row['role'] == 'beat_this_final0':
            b.require(path == runtime / 'checkpoints/hub/checkpoints/beat_this-final0.ckpt', 'Beat This checkpoint path changed')
        else:
            b.require(path.parent == Path(TORCH_HUB) / 'checkpoints', 'Torch checkpoint outside frozen hub')
    evidence.update(checked_sha256=checked, parent_import_origins=parent_origins,
                    child_runtime=child, torch_hub_dir=TORCH_HUB,
                    child_environment_policy=freeze['child_environment'])
    return evidence


def assigned_gpu(freeze, stage, shard_index):
    gpus = freeze['aio_gpus'] if stage == 'allinone' else freeze['beat_gpus']
    return gpus[shard_index % len(gpus)]


def receipt_path(output, stage, shard_index):
    return output / 'receipts' / f'{stage}_{shard_index:03d}.json'


def log_path(output, stage, shard_index):
    return output / 'logs' / f'{stage}_{shard_index:03d}.log'


RECEIPT_KEYS = {'status', 'stage', 'shard_index', 'item_ids', 'run_contract_sha256', 'gpu', 'command',
                'child_environment', 'started_utc', 'completed_utc', 'inputs', 'outputs', 'log',
                'ordinary_empty_beat_allowed_only_after_this_successful_command', *SUCCESS_SCOPE}


def validate_receipt(path, run_sha, stage, shard, rows, freeze, output):
    path = b.safe_path(path)
    envelope = b.read_json(path)
    b.require(set(envelope) == {'payload', 'receipt_sha256'}
              and envelope['receipt_sha256'] == b.value_hash(envelope['payload']), 'stage receipt envelope/hash')
    receipt = envelope['payload']; ids = shard['item_ids']; gpu = assigned_gpu(freeze, stage, shard['index'])
    b.require(set(receipt) == RECEIPT_KEYS and receipt['status'] == 'passed'
              and receipt['stage'] == stage and receipt['shard_index'] == shard['index']
              and receipt['item_ids'] == ids and receipt['run_contract_sha256'] == run_sha
              and receipt['gpu'] == gpu and all(receipt[k] == v for k, v in SUCCESS_SCOPE.items()),
              'stage receipt identity/GPU/scope')
    b.require(isinstance(receipt['started_utc'], str) and isinstance(receipt['completed_utc'], str), 'stage receipt timestamps')
    expected_command = b.command(stage, [row['input']['path'] for row in rows], Path(freeze['runtime_root']), output)
    b.require(receipt['command'] == expected_command, 'stage receipt command changed')
    b.require(receipt['child_environment'] == child_environment(freeze['runtime_root'], gpu),
              'stage receipt child environment changed')
    expected_log = b.binding(log_path(output, stage, shard['index']))
    b.require(receipt['log'] == expected_log, 'stage receipt log changed')
    expected_inputs = {row['id']: b.inspect_audio(row['input']['path'], row, input_audio=True) for row in rows}
    b.require(receipt['inputs'] == expected_inputs, 'stage receipt input PCM/hash changed')
    current = b.verify_products(stage, rows, output)
    expected_paths = {str(path) for ident in ids for path in b.product_paths(stage, ident, output)}
    b.require(set(receipt['outputs']) == expected_paths == set(current), 'stage receipt exact output path set')
    b.require(receipt['outputs'] == current, 'stage receipt product semantics/hash changed')
    b.require(receipt['ordinary_empty_beat_allowed_only_after_this_successful_command'] is (stage == 'beats'),
              'empty-beat success provenance')
    return {'receipt': b.binding(path), 'payload_sha256': envelope['receipt_sha256'],
            'log': expected_log, 'outputs': current, 'inputs': expected_inputs}


def source_graph_bindings(freeze, helper):
    result = {}
    for key in ('plan', 'screen'):
        helper.add_binding(result, freeze[key])
    new_entry, prior_entry = freeze['upstream_commits']['new'], freeze['upstream_commits']['prior']
    new_root, prior_root = Path(new_entry['path']).parent, Path(prior_entry['path']).parent
    _, new = helper.bind_commit(new_root, new_entry['sha256'], 'new', result)
    _, prior = helper.bind_commit(prior_root, prior_entry['sha256'], 'prior', result)
    upstream = helper.base.read_json(new_root / 'upstream_bindings.json')
    helper.collect_upstream(upstream, result)
    helper.bind_receipts(Path(new['original_root']), new, result, 'new')
    helper.bind_receipts(prior_root, prior, result, 'prior')
    return result


def rehash_source_graph(freeze, cohort, helper):
    graph = source_graph_bindings(freeze, helper)
    for path, entry in graph.items():
        b.require(cohort['bindings'].get(path) == entry, 'source graph declaration changed: ' + path)
        b._binding_matches(entry, audio=True)
    return {'files': len(graph), 'bindings_sha256': b.value_hash(graph)}


def audit_all(run_sha, freeze, cohort, output):
    by_id = {row['id']: row for row in cohort['rows']}
    receipts, logs, products, inputs = {}, {}, {}, {}
    for stage in ('allinone', 'beats'):
        for shard in freeze['shards']:
            rows = [by_id[ident] for ident in shard['item_ids']]
            path = receipt_path(output, stage, shard['index'])
            evidence = validate_receipt(path, run_sha, stage, shard, rows, freeze, output)
            receipts[str(path)] = {'binding': evidence['receipt'], 'payload_sha256': evidence['payload_sha256']}
            b.require(evidence['log']['path'] not in logs, 'duplicate log evidence')
            logs[evidence['log']['path']] = evidence['log']
            for product, record in evidence['outputs'].items():
                b.require(product not in products, 'duplicate stage product')
                products[product] = record
            for ident, record in evidence['inputs'].items():
                previous = inputs.get(ident)
                b.require(previous is None or previous == record, 'input evidence differs between stages')
                inputs[ident] = record
    expected_products = {str(path) for row in cohort['rows'] for stage in ('allinone', 'beats')
                         for path in b.product_paths(stage, row['id'], output)}
    b.require(set(products) == expected_products and set(inputs) == set(by_id), 'final evidence set mismatch')
    validate_output_inventory(output, cohort['rows'], freeze['shards'], final_audit=(output / 'final_audit.json').exists(),
                              completion=(output / 'completion.json').exists())
    return {'stage_receipts': receipts, 'logs': logs, 'products': products, 'inputs': inputs,
            'stage_receipts_sha256': b.value_hash(receipts), 'logs_sha256': b.value_hash(logs),
            'products_sha256': b.value_hash(products), 'inputs_sha256': b.value_hash(inputs)}


def expected_run_record(freeze_path, freeze_sha, freeze, cohort, runtime_start, source_start):
    runtime, output = Path(freeze['runtime_root']), Path(freeze['output_root'])
    rows = cohort['rows']
    return {'version': VERSION, 'status': 'parent_frozen_execution',
            'parent_freeze': {'path': str(freeze_path), 'sha256': freeze_sha},
            'cohort_contract': freeze['cohort_contract'], 'rows': len(rows),
            'row_ids_sha256': b.value_hash([row['id'] for row in rows]), 'shards': freeze['shards'],
            'aio_gpus': freeze['aio_gpus'], 'beat_gpus': freeze['beat_gpus'],
            'runtime_root': str(runtime), 'output_root': str(output), 'runtime_start': runtime_start,
            'source_graph_start': source_start, 'child_environment_policy': freeze['child_environment'],
            'commands': {stage: b.command(stage, ['<frozen shard inputs>'], runtime, output)
                         for stage in ('allinone', 'beats')},
            'resume_policy': 'semantic receipt revalidation; retain/fail on unreceipted partial products',
            'beat_serializer_recovery': 'not_in_v2', **SUCCESS_SCOPE}


def expected_audit_record(run_path, freeze_path, freeze_sha, evidence):
    return {'status': 'all_stage_receipts_products_logs_semantically_revalidated',
            'run_contract': b.binding(run_path),
            'parent_freeze': {'path': str(freeze_path), 'sha256': freeze_sha},
            **evidence, **{k: v for k, v in SUCCESS_SCOPE.items() if k != 'recovery_applied'}}


def expected_completion_record(rows, run_path, freeze_path, freeze_sha, audit_path,
                               source_end, runtime_end, evidence):
    return {'status': 'passed_native30_inference_not_feature_extraction', 'rows': len(rows),
            'run_contract': b.binding(run_path),
            'parent_freeze': {'path': str(freeze_path), 'sha256': freeze_sha},
            'final_audit': b.binding(audit_path), 'source_graph_end': source_end,
            'runtime_end': runtime_end, **evidence,
            **{k: v for k, v in SUCCESS_SCOPE.items() if k != 'recovery_applied'}}


def verify_completion(freeze_path, freeze_sha):
    """Read-only exhaustive verifier for downstream native30 consumers."""
    freeze_path = b.safe_path(freeze_path)
    freeze, cohort, helper = validate_freeze(freeze_path, freeze_sha)
    output = b.safe_path(freeze['output_root'])
    runtime_end = verify_runtime(freeze)
    source_end = rehash_source_graph(freeze, cohort, helper)
    run_path = output / 'run_contract.json'
    expected_run = expected_run_record(freeze_path, freeze_sha, freeze, cohort, runtime_end, source_end)
    run_binding = bind_canonical_json(run_path, expected_run)
    evidence = audit_all(run_binding['sha256'], freeze, cohort, output)
    audit_path = output / 'final_audit.json'
    audit = expected_audit_record(run_path, freeze_path, freeze_sha, evidence)
    audit_binding = bind_canonical_json(audit_path, audit)
    completion_path = output / 'completion.json'
    completion = expected_completion_record(cohort['rows'], run_path, freeze_path, freeze_sha, audit_path,
                                            source_end, runtime_end, evidence)
    completion_binding = bind_canonical_json(completion_path, completion)
    validate_output_inventory(output, cohort['rows'], freeze['shards'], final_audit=True, completion=True)
    b.require(bind_canonical_json(run_path, expected_run) == run_binding
              and bind_canonical_json(audit_path, audit) == audit_binding
              and bind_canonical_json(completion_path, completion) == completion_binding,
              'completion graph changed during verification')
    return {'status': 'verified_complete_native30_inference', 'rows': len(cohort['rows']),
            'completion': completion_binding, 'final_audit': audit_binding,
            'run_contract': run_binding, 'source_graph': source_end, 'runtime': runtime_end,
            'evidence_sha256': b.value_hash(evidence)}


def validate_output_inventory(output, rows, shards, *, final_audit=False, completion=False):
    ids = {row['id'] for row in rows}
    roots = {'writer.lock', 'run_contract.json', 'structure', 'demix', 'spec', 'beats', 'receipts', 'logs'}
    if final_audit: roots.add('final_audit.json')
    if completion: roots.add('completion.json')
    b.require({path.name for path in output.iterdir()} == roots, 'inference root inventory mismatch')
    for path in output.rglob('*'):
        b.require(not path.is_symlink(), 'inference symlink forbidden')
    b.require({p.name for p in (output / 'structure').iterdir()} == {x + '.json' for x in ids}, 'structure inventory')
    b.require({p.name for p in (output / 'spec').iterdir()} == {x + '.npy' for x in ids}, 'spectrogram inventory')
    b.require({p.name for p in (output / 'beats').iterdir()} == {x + '.beats' for x in ids}, 'beat inventory')
    demix = output / 'demix'
    b.require({p.name for p in demix.iterdir()} == {'htdemucs'}
              and {p.name for p in (demix / 'htdemucs').iterdir()} == ids, 'demix inventory')
    for ident in ids:
        b.require({p.name for p in (demix / 'htdemucs' / ident).iterdir()} == {x + '.wav' for x in b.STEMS}, 'stem inventory')
    receipt_names = {f'{stage}_{shard["index"]:03d}.json' for stage in ('allinone', 'beats') for shard in shards}
    log_names = {f'{stage}_{shard["index"]:03d}.log' for stage in ('allinone', 'beats') for shard in shards}
    b.require({p.name for p in (output / 'receipts').iterdir()} == receipt_names, 'receipt inventory')
    b.require({p.name for p in (output / 'logs').iterdir()} == log_names, 'log inventory')


def run_contract(freeze_path, freeze_sha, freeze, cohort, helper, runtime_start):
    output, runtime = Path(freeze['output_root']), Path(freeze['runtime_root'])
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'writer.lock'
    b.require(not lock_path.is_symlink(), 'writer lock symlink forbidden')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('structure', 'demix', 'spec', 'beats', 'receipts', 'logs'):
            (output / name).mkdir(exist_ok=True)
            b.require((output / name).is_dir() and not (output / name).is_symlink(), 'unsafe output directory')
        source_start = rehash_source_graph(freeze, cohort, helper)
        rows = cohort['rows']; by_id = {row['id']: row for row in rows}
        for row in rows:
            b.inspect_audio(row['input']['path'], row, input_audio=True)
        run = expected_run_record(freeze_path, freeze_sha, freeze, cohort, runtime_start, source_start)
        run_path = output / 'run_contract.json'
        if run_path.exists():
            b.require(b.read_json(run_path) == run, 'existing run contract conflict')
        else:
            b.require({p.name for p in output.iterdir()} == {'writer.lock', 'structure', 'demix', 'spec', 'beats', 'receipts', 'logs'},
                      'nonempty inference output before run contract')
            b.write_new(run_path, run)
        run_sha = b.digest(run_path); stopped = threading.Event()

        def worker(stage, gpu, assigned):
            for shard in assigned:
                if stopped.is_set(): return
                ids = shard['item_ids']; selected = [by_id[ident] for ident in ids]
                path = receipt_path(output, stage, shard['index'])
                try:
                    b.require(b.digest(freeze_path) == freeze_sha, 'parent freeze changed during inference')
                    if path.exists():
                        validate_receipt(path, run_sha, stage, shard, selected, freeze, output)
                        continue
                    expected_paths = [p for ident in ids for p in b.product_paths(stage, ident, output)]
                    b.require(not any(p.exists() or p.is_symlink() for p in expected_paths), 'unreceipted partial products')
                    log = log_path(output, stage, shard['index'])
                    b.require(not log.exists() and not log.is_symlink(), 'unreceipted stage log retained')
                    before = {row['id']: b.inspect_audio(row['input']['path'], row, input_audio=True) for row in selected}
                    b.require_idle_5090(gpu)
                    command = b.command(stage, [row['input']['path'] for row in selected], runtime, output)
                    environment = child_environment(runtime, gpu)
                    started = datetime.now(timezone.utc).isoformat()
                    with log.open('x') as stream:
                        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    products = b.verify_products(stage, selected, output)
                    after = {row['id']: b.inspect_audio(row['input']['path'], row, input_audio=True) for row in selected}
                    b.require(before == after, 'input changed during inference')
                    payload = {'status': 'passed', 'stage': stage, 'shard_index': shard['index'], 'item_ids': ids,
                               'run_contract_sha256': run_sha, 'gpu': gpu, 'command': command,
                               'child_environment': environment,
                               'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                               'inputs': before, 'outputs': products, 'log': b.binding(log),
                               'ordinary_empty_beat_allowed_only_after_this_successful_command': stage == 'beats',
                               **SUCCESS_SCOPE}
                    b.write_new(path, {'payload': payload, 'receipt_sha256': b.value_hash(payload)})
                    validate_receipt(path, run_sha, stage, shard, selected, freeze, output)
                except Exception:
                    stopped.set(); raise

        jobs = [(stage, gpu, assigned) for stage, gpus in (('allinone', freeze['aio_gpus']), ('beats', freeze['beat_gpus']))
                for gpu, assigned in b.assignments(freeze['shards'], gpus)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(worker, *job) for job in jobs]
            for future in futures: future.result()

        evidence = audit_all(run_sha, freeze, cohort, output)
        audit = expected_audit_record(run_path, freeze_path, freeze_sha, evidence)
        audit_path = output / 'final_audit.json'
        if audit_path.exists(): b.require(b.read_json(audit_path) == audit, 'existing final audit conflict')
        else: b.write_new(audit_path, audit)
        audit_binding = bind_canonical_json(audit_path, audit)
        b.require(audit_all(run_sha, freeze, cohort, output) == evidence, 'evidence changed after final audit')

        final_freeze, final_cohort, final_helper = validate_freeze(freeze_path, freeze_sha)
        b.require(final_freeze == freeze and final_cohort == cohort, 'parent/cohort freeze changed at final audit')
        source_end = rehash_source_graph(freeze, cohort, final_helper)
        runtime_end = verify_runtime(freeze)
        b.require(source_end == source_start and runtime_end == runtime_start, 'source graph/runtime changed during inference')
        final_evidence = audit_all(run_sha, freeze, cohort, output)
        b.require(final_evidence == evidence and bind_canonical_json(audit_path, audit) == audit_binding,
                  'final evidence/audit mutation')
        completion = expected_completion_record(rows, run_path, freeze_path, freeze_sha, audit_path,
                                                source_end, runtime_end, final_evidence)
        completion_path = output / 'completion.json'
        if completion_path.exists(): b.require(b.read_json(completion_path) == completion, 'existing completion conflict')
        else: b.write_new(completion_path, completion)
        completion_binding = bind_canonical_json(completion_path, completion)
        validate_output_inventory(output, rows, freeze['shards'], final_audit=True, completion=True)
        b.require(bind_canonical_json(completion_path, completion) == completion_binding, 'completion commitment changed')
        return {'status': completion['status'], 'rows': len(rows), 'completion_sha256': b.digest(completion_path),
                'final_audit_sha256': b.digest(audit_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--frozen-sha256', required=True)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    args = parser.parse_args()
    freeze, cohort, helper = validate_freeze(args.frozen, args.frozen_sha256)
    runtime = verify_runtime(freeze)
    if args.mode == 'preflight':
        result = {'status': 'metadata_preflight_only_no_audio_reads_or_writes', 'rows': len(cohort['rows']),
                  'shards': len(freeze['shards']), 'parent_freeze_sha256': args.frozen_sha256,
                  'runtime': runtime, **{k: v for k, v in SUCCESS_SCOPE.items() if k != 'recovery_applied'}}
    else:
        result = run_contract(args.frozen.resolve(), args.frozen_sha256, freeze, cohort, helper, runtime)
    print(b.canonical(result).decode().strip(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
