"""Separately frozen, append-only continuation of a partial native30 v2 run.

Ordinary v2 receipts remain unchanged. Recovered shards have distinct receipts
outside the original product root and can never satisfy the ordinary v2 gate.
No classifier, feature extraction, replacement input or fabricated beat event.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

import numpy as np

VERSION = 'continue_native30_empty_beats_v1'
V2_SHA = '1aaf8f2498986fbffe23f818ae6db2d37f59303eb239362447fbf8d8f88b6274'
FREEZE_VERSION = 'native30-recovery-continuation-parent-freeze-v1'
FREEZE_STATUS = 'parent_frozen_for_native30_recovery_continuation'
CHECKPOINT_SHA = '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
ERROR = 'Not all downbeats are beats.'
FAILURE_PATTERN = r'Could not process "([^"]+)"\. Rerun with this file alone for details\.'
POLICY = {'beat_array': 'one_dimensional_empty', 'downbeat_array': 'one_dimensional_nonempty_finite_strictly_increasing',
          'downbeat_range_closed_seconds': [0, 30], 'exception_type': 'ValueError', 'exception_message': ERROR,
          'float16': True, 'dbn': False, 'checkpoint_sha256': CHECKPOINT_SHA,
          'missing_outputs_exactly_equal_unique_logged_failed_paths': True,
          'preserve_existing_products_and_failed_logs': True, 'replace_inputs': False, 'fabricate_beats': False}
SCOPE = {'classifier_fits': 0, 'cohort_admitted': False, 'feature_extraction_authorized': False}
EVIDENCE_KEYS = ('stage_receipts', 'logs', 'products', 'inputs')


def load_runner():
    path = Path(__file__).resolve().with_name('run_native30_inference_batches_v2.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != V2_SHA or path.is_symlink():
        raise ValueError('pinned ordinary v2 runner changed')
    module = importlib.import_module('run_native30_inference_batches_v2')
    if Path(module.__file__).resolve() != path:
        raise ValueError('wrong ordinary v2 import origin')
    return module


v2 = load_runner()
b = v2.b
require = b.require


def utc():
    return datetime.now(timezone.utc).isoformat()


def put(path, payload):
    b.write_new(path, {'payload': payload, 'receipt_sha256': b.value_hash(payload)})


def get(path):
    value = b.read_json(path)
    require(set(value) == {'payload', 'receipt_sha256'} and b.value_hash(value['payload']) == value['receipt_sha256'],
            'continuation envelope/hash mismatch')
    return value['payload']


def files(root):
    found = {}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'continuation/original symlink forbidden')
        if path.is_file() and path != root / 'writer.lock':
            found[str(path)] = path
        else:
            require(path.is_dir() or (path == root / 'writer.lock' and path.is_file()), 'nonregular inventory entry')
    return found


def validate_arrays(beats, downbeats):
    beats, downbeats = np.asarray(beats), np.asarray(downbeats)
    require(beats.ndim == downbeats.ndim == 1 and len(beats) == 0 and len(downbeats) > 0,
            'not the reviewed empty-beat/nonempty-downbeat case')
    require(np.issubdtype(beats.dtype, np.number) and np.issubdtype(downbeats.dtype, np.number), 'nonnumeric raw arrays')
    require(np.isfinite(downbeats).all() and (downbeats >= 0).all() and (downbeats <= 30).all()
            and (np.diff(downbeats) > 0).all(), 'invalid raw downbeat times')


def array_record(values):
    values = np.asarray(values)
    return {'values': values.tolist(), 'dtype': str(values.dtype), 'shape': list(values.shape),
            'float64_le_sha256': hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes()).hexdigest()}


def read_array(record):
    values = np.asarray(record['values'], dtype=record['dtype'])
    require(array_record(values) == record, 'raw-array representation/hash changed')
    return values


def validate_initial_snapshot(freeze, original, *, audio, first=False):
    snapshot = freeze['initial_partial_inventory']
    require(isinstance(snapshot, dict) and str(original / 'run_contract.json') in snapshot, 'initial snapshot missing run contract')
    current = files(original)
    require(set(snapshot) <= set(current), 'previously existing original file removed')
    if first:
        require(set(snapshot) == set(current), 'first continuation inventory differs from parent-frozen partial snapshot')
    for path, entry in snapshot.items():
        require(entry['path'] == path and Path(path).is_relative_to(original), 'snapshot path escapes original root')
        b._binding_matches(entry, audio=audio)
    require(not (original / 'completion.json').exists() and not (original / 'final_audit.json').exists(),
            'original v2 run must remain partial without its own completion')


def validate_freeze(path, expected_sha, *, audio=False):
    path = b.safe_path(path)
    b.require_hash(path, expected_sha)
    freeze = b.read_json(path)
    require(freeze.get('version') == FREEZE_VERSION and freeze.get('status') == FREEZE_STATUS and
            freeze.get('policy') == POLICY and all(freeze.get(k) == v for k, v in SCOPE.items()), 'continuation freeze scope/policy')
    for key in ('original_parent_freeze', 'original_run_contract', 'v2_runner', 'driver', 'tests'):
        b._binding_matches(freeze[key], audio=False)
    require(freeze['v2_runner'] == b.binding(Path(v2.__file__).resolve()) and freeze['v2_runner']['sha256'] == V2_SHA,
            'ordinary runner not frozen')
    require(freeze['driver'] == b.binding(Path(__file__).resolve()) and
            freeze['tests'] == b.binding(Path(__file__).resolve().with_name('test_' + VERSION + '.py')), 'continuation code/tests not frozen')
    original_freeze = freeze['original_parent_freeze']
    prior, cohort, helper = v2.validate_freeze(original_freeze['path'], original_freeze['sha256'])
    original, evidence_root = Path(prior['output_root']), b.safe_path(freeze['output_root'])
    require(not evidence_root.is_relative_to(original) and not original.is_relative_to(evidence_root)
            and evidence_root.parent.is_dir(), 'separate continuation evidence root required')
    require(freeze['original_run_contract']['path'] == str(original / 'run_contract.json'), 'original run contract path')
    run = b.read_json(original / 'run_contract.json')
    expected_run = v2.expected_run_record(Path(original_freeze['path']), original_freeze['sha256'], prior, cohort,
                                         run['runtime_start'], run['source_graph_start'])
    require(run == expected_run, 'original v2 run contract does not match parent authority')
    require(freeze['initial_partial_inventory'].get(str(original / 'run_contract.json')) == freeze['original_run_contract'],
            'snapshot/original run binding')
    validate_initial_snapshot(freeze, original, audio=audio)
    return freeze, prior, cohort, helper


def paths_for(shard, rows, original):
    return {row['input']['path']: original / 'beats' / (row['id'] + '.beats') for row in rows}


def failed_partition(shard, rows, original, log):
    require([row['id'] for row in rows] == shard['item_ids'], 'failed shard row order')
    paths = paths_for(shard, rows, original)
    logged = re.findall(FAILURE_PATTERN, Path(log).read_text())
    missing = [path for path, target in paths.items() if not target.exists()]
    require(missing and len(logged) == len(set(logged)) and set(logged) == set(missing),
            'missing products must exactly match unique logged failures')
    require(set(logged) <= set(paths), 'logged failure outside frozen shard')
    existing = {}
    for target in paths.values():
        if target.exists():
            existing[str(target)] = b.inspect_beats(target)
        else:
            require(not target.is_symlink(), 'missing target symlink')
    return missing, existing


def name(stage, shard):
    return f'{stage}_{shard["index"]:03d}.json'


def validate_execution(root, stage, shard, rows, prior, run_sha, freeze):
    intent_path = root / 'executions' / name(stage, shard)
    result_path = root / 'execution_results' / name(stage, shard)
    intent, result = get(intent_path), get(result_path)
    original = Path(prior['output_root']); gpu = v2.assigned_gpu(prior, stage, shard['index'])
    require(intent['status'] == 'continuation_stage_execution_intent' and intent['stage'] == stage and
            intent['shard_index'] == shard['index'] and intent['item_ids'] == shard['item_ids'] and
            intent['original_run_contract_sha256'] == run_sha and intent['gpu'] == gpu and
            intent['driver'] == freeze['driver'] and intent['continuation_freeze_sha256'] == freeze['_sha256'] and
            intent['command'] == b.command(stage, [r['input']['path'] for r in rows], Path(prior['runtime_root']), original) and
            intent['child_environment'] == v2.child_environment(prior['runtime_root'], gpu), 'continuation execution intent mismatch')
    require(result['status'] == 'continuation_stage_subprocess_exited' and result['intent'] == b.binding(intent_path) and
            result['returncode'] == 0 and result['log'] == b.binding(v2.log_path(original, stage, shard['index'])),
            'continuation subprocess result/log mismatch')
    expected_inputs = {r['id']: b.inspect_audio(r['input']['path'], r, input_audio=True) for r in rows}
    require(intent['inputs'] == result['inputs_after'] == expected_inputs, 'continuation execution input hashes changed')
    return {'intent': b.binding(intent_path), 'result': b.binding(result_path)}


def validate_plan(plan, freeze, prior, shard, rows, root):
    original = Path(prior['output_root']); run_sha = freeze['original_run_contract']['sha256']
    log = v2.log_path(original, 'beats', shard['index'])
    require(plan['status'] == 'frozen_missing_beat_recovery_transaction' and plan['policy'] == POLICY and
            plan['original_run_contract_sha256'] == run_sha and plan['shard_index'] == shard['index'] and
            plan['item_ids'] == shard['item_ids'] and plan['original_log'] == b.binding(log) and
            plan['original_shard_command'] == b.command('beats', [r['input']['path'] for r in rows], Path(prior['runtime_root']), original),
            'recovery transaction lineage')
    logged = re.findall(FAILURE_PATTERN, log.read_text())
    by_path = {r['input']['path']: r for r in rows}
    require(len(logged) == len(set(logged)) and set(logged) == set(plan['missing_input_paths']) and set(logged) <= set(by_path),
            'recovery transaction logged failure set')
    expected_missing = [r['id'] for r in rows if r['input']['path'] in set(logged)]
    require(plan['missing_ids'] == expected_missing and len(expected_missing) > 0, 'recovery transaction ordered missing IDs')
    existing_ids = {r['id'] for r in rows} - set(expected_missing)
    expected_paths = {str(original / 'beats' / (i + '.beats')) for i in existing_ids}
    require(set(plan['preserved_outputs']) == expected_paths, 'recovery preserved output set')
    for path, record in plan['preserved_outputs'].items():
        require(b.inspect_beats(path) == record, 'preexisting beat product changed')
    current_inputs = {r['id']: b.inspect_audio(r['input']['path'], r, input_audio=True) for r in rows}
    require(plan['inputs'] == current_inputs, 'recovery input PCM/hash changed')
    if plan['original_attempt_kind'] == 'original_v2_partial':
        require(str(log) in freeze['initial_partial_inventory'] and freeze['initial_partial_inventory'][str(log)] == b.binding(log),
                'original failed log absent from parent partial snapshot')
        require(all(str(original / 'beats' / (i + '.beats')) not in freeze['initial_partial_inventory'] for i in expected_missing),
                'recovered target was not initially missing')
    else:
        require(plan['original_attempt_kind'] == 'continuation_subprocess', 'unknown original attempt kind')
        require(plan['execution'] == validate_execution(root, 'beats', shard, rows, prior, run_sha, freeze), 'failed continuation provenance')
    return current_inputs


def validate_raw(raw, plan, request_binding, freeze, prior):
    gpu = v2.assigned_gpu(prior, 'beats', plan['shard_index'])
    require(raw['status'] == 'reproduced_exact_empty_beat_serializer_failure' and raw['request'] == request_binding and
            raw['policy'] == POLICY and raw['gpu'] == gpu and 'RTX 5090' in raw['gpu_name'] and
            all(raw.get(k) == v for k, v in SCOPE.items()) and raw['cuda_device'] == 'cuda:0' and
            raw['child_environment'] == v2.child_environment(prior['runtime_root'], gpu) and
            raw['checkpoint'] == b.binding(Path(prior['runtime_root']) / 'checkpoints/hub/checkpoints/beat_this-final0.ckpt') and
            raw['checkpoint']['sha256'] == CHECKPOINT_SHA and raw['driver_sha256'] == freeze['driver']['sha256'] and
            raw['runtime_sha256'] == b.value_hash(b.read_json(freeze['original_run_contract']['path'])['runtime_start']),
            'raw replay runtime/checkpoint/driver provenance')
    require(raw['recovery_api'] == 'File2Beats(checkpoint,device=cuda:0,float16=True,dbn=False)(input_path)', 'recovery API provenance')
    records = raw['records']
    require([r['id'] for r in records] == plan['missing_ids'], 'raw replay exact ID order')
    for record in records:
        require(record['input'] == plan['inputs'][record['id']] and record['exception_type'] == 'ValueError'
                and record['exception_message'] == ERROR, 'raw replay input/exception')
        validate_arrays(read_array(record['beats']), read_array(record['downbeats']))
    return records


def validate_recovered(path, freeze, prior, shard, rows, root):
    payload = get(path); original = Path(prior['output_root'])
    request_path = root / 'recovery_requests' / name('beats', shard)
    raw_path = root / 'raw_replays' / name('beats', shard)
    request, raw = get(request_path), get(raw_path)
    inputs = validate_plan(request, freeze, prior, shard, rows, root)
    records = validate_raw(raw, request, b.binding(request_path), freeze, prior)
    require(payload['status'] == 'recovered_serializer_v1' and payload['request'] == b.binding(request_path) and
            payload['raw_replay'] == b.binding(raw_path) and payload['original_run_contract_sha256'] == freeze['original_run_contract']['sha256'] and
            payload['recovery_applied'] is True and payload['ordinary_success'] is False and payload['item_ids'] == shard['item_ids'] and
            all(payload.get(k) == v for k, v in SCOPE.items()), 'typed recovered receipt')
    replay_log = root / 'replay_logs' / (name('beats', shard)[:-5] + '.log')
    require(payload['replay_log'] == b.binding(replay_log), 'replay log changed')
    expected_invocation = replay_command(freeze, prior, request_path)
    require(raw['recovery_invocation'] == expected_invocation and payload['recovery_invocation'] == expected_invocation,
            'actual replay invocation mismatch')
    products = b.verify_products('beats', rows, original)
    require(payload['outputs'] == products and payload['inputs'] == inputs, 'recovered shard physical evidence')
    for record in records:
        target = original / 'beats' / (record['id'] + '.beats')
        require(products[str(target)]['bytes'] == 0 and products[str(target)]['status'] == 'empty_unavailable'
                and products[str(target)]['beat_count'] == 0 and products[str(target)]['sha256'] == hashlib.sha256(b'').hexdigest(),
                'recovered product must be genuinely empty/unavailable')
    require(not v2.receipt_path(original, 'beats', shard['index']).exists(), 'recovery must not have a fake ordinary receipt')
    return {'receipt': b.binding(path), 'payload_sha256': b.value_hash(payload), 'inputs': inputs, 'outputs': products,
            'log': request['original_log'], 'raw_replay': b.binding(raw_path), 'request': b.binding(request_path),
            'replay_log': payload['replay_log'], 'missing_ids': request['missing_ids']}


def replay_command(freeze, prior, request_path):
    return [str(Path(prior['runtime_root']) / 'venv/bin/python'), str(Path(__file__).resolve()),
            '--frozen', freeze['_path'], '--frozen-sha256', freeze['_sha256'], '--mode', 'replay',
            '--request', str(request_path), '--request-sha256', b.digest(request_path)]


def replay_worker(freeze, prior, cohort, request_path, request_sha):
    """Called only in a separate exact-environment subprocess after parent freeze."""
    root = Path(freeze['output_root']); request_path = b.safe_path(request_path)
    b.require_hash(request_path, request_sha); plan = get(request_path)
    shard = prior['shards'][plan['shard_index']]
    require(request_path == root / 'recovery_requests' / name('beats', shard), 'replay request path')
    rows = [r for r in cohort['rows'] if r['id'] in shard['item_ids']]
    validate_plan(plan, freeze, prior, shard, rows, root)
    gpu = v2.assigned_gpu(prior, 'beats', shard['index'])
    require(dict(os.environ) == v2.child_environment(prior['runtime_root'], gpu), 'replay child environment')
    b.require_idle_5090(gpu)
    import torch
    from beat_this.inference import File2Beats
    from beat_this.utils import infer_beat_numbers
    require(torch.cuda.is_available() and 'RTX 5090' in torch.cuda.get_device_name(0), 'replay requires mapped RTX5090')
    checkpoint = Path(prior['runtime_root']) / 'checkpoints/hub/checkpoints/beat_this-final0.ckpt'
    b.require_hash(checkpoint, CHECKPOINT_SHA)
    model = File2Beats(str(checkpoint), device='cuda:0', float16=True, dbn=False)
    by_id = {r['id']: r for r in rows}; records = []
    for ident in plan['missing_ids']:
        row = by_id[ident]; observed = b.inspect_audio(row['input']['path'], row, input_audio=True)
        beats, downbeats = model(row['input']['path'])
        validate_arrays(beats, downbeats)
        try:
            infer_beat_numbers(beats, downbeats)
        except ValueError as exc:
            require(str(exc) == ERROR, 'different serializer exception')
        else:
            raise ValueError('reviewed serializer exception was not reproduced')
        require(b.inspect_audio(row['input']['path'], row, input_audio=True) == observed, 'input changed during replay')
        records.append({'id': ident, 'input': observed, 'beats': array_record(beats), 'downbeats': array_record(downbeats),
                        'exception_type': 'ValueError', 'exception_message': ERROR})
    payload = {'status': 'reproduced_exact_empty_beat_serializer_failure', 'request': b.binding(request_path),
               'policy': POLICY, 'gpu': gpu, 'gpu_name': torch.cuda.get_device_name(0), 'cuda_device': 'cuda:0',
               'child_environment': dict(os.environ), 'checkpoint': b.binding(checkpoint), 'driver_sha256': freeze['driver']['sha256'],
               'recovery_api': 'File2Beats(checkpoint,device=cuda:0,float16=True,dbn=False)(input_path)',
               'recovery_invocation': [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
               'runtime_sha256': b.value_hash(b.read_json(freeze['original_run_contract']['path'])['runtime_start']),
               'records': records, 'completed_utc': utc(), **SCOPE}
    put(root / 'raw_replays' / name('beats', shard), payload)


def recover_shard(freeze, prior, shard, rows, root):
    original = Path(prior['output_root']); log = v2.log_path(original, 'beats', shard['index'])
    request_path = root / 'recovery_requests' / name('beats', shard)
    if request_path.exists():
        plan = get(request_path)
    else:
        missing, existing = failed_partition(shard, rows, original, log)
        initial = str(log) in freeze['initial_partial_inventory']
        execution = None if initial else validate_execution(root, 'beats', shard, rows, prior, freeze['original_run_contract']['sha256'], freeze)
        plan = {'status': 'frozen_missing_beat_recovery_transaction', 'policy': POLICY,
                'original_run_contract_sha256': freeze['original_run_contract']['sha256'], 'shard_index': shard['index'],
                'item_ids': shard['item_ids'], 'missing_input_paths': missing,
                'missing_ids': [r['id'] for r in rows if r['input']['path'] in set(missing)],
                'original_log': b.binding(log), 'original_attempt_kind': 'original_v2_partial' if initial else 'continuation_subprocess',
                'execution': execution, 'preserved_outputs': existing,
                'inputs': {r['id']: b.inspect_audio(r['input']['path'], r, input_audio=True) for r in rows},
                'original_shard_command': b.command('beats', [r['input']['path'] for r in rows], Path(prior['runtime_root']), original)}
        put(request_path, plan)
    validate_plan(plan, freeze, prior, shard, rows, root)
    raw_path = root / 'raw_replays' / name('beats', shard)
    replay_log = root / 'replay_logs' / (name('beats', shard)[:-5] + '.log')
    command = replay_command(freeze, prior, request_path)
    if not raw_path.exists():
        require(not replay_log.exists(), 'interrupted/unreceipted replay log retained; explicit review required')
        gpu = v2.assigned_gpu(prior, 'beats', shard['index']); b.require_idle_5090(gpu)
        with replay_log.open('x') as stream:
            subprocess.run(command, env=v2.child_environment(prior['runtime_root'], gpu), stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    raw = get(raw_path)
    validate_raw(raw, plan, b.binding(request_path), freeze, prior)
    require(raw['recovery_invocation'] == command, 'raw replay actual invocation')
    for ident in plan['missing_ids']:
        target = original / 'beats' / (ident + '.beats')
        if not target.exists():
            with target.open('xb') as stream:
                stream.flush(); os.fsync(stream.fileno())
        require(b.inspect_beats(target)['bytes'] == 0, 'recovery target changed/nonempty')
    validate_plan(plan, freeze, prior, shard, rows, root)
    outputs = b.verify_products('beats', rows, original)
    payload = {'status': 'recovered_serializer_v1', 'request': b.binding(request_path), 'raw_replay': b.binding(raw_path),
               'original_run_contract_sha256': freeze['original_run_contract']['sha256'], 'item_ids': shard['item_ids'],
               'recovery_applied': True, 'ordinary_success': False, 'inputs': plan['inputs'], 'outputs': outputs,
               'replay_log': b.binding(replay_log), 'recovery_invocation': command, **SCOPE}
    path = root / 'recovered_receipts' / name('beats', shard)
    if not path.exists(): put(path, payload)
    return validate_recovered(path, freeze, prior, shard, rows, root)


def ordinary_shard(freeze, prior, shard, rows, stage, root):
    original, runtime = Path(prior['output_root']), Path(prior['runtime_root'])
    run_sha = freeze['original_run_contract']['sha256']; gpu = v2.assigned_gpu(prior, stage, shard['index'])
    log = v2.log_path(original, stage, shard['index'])
    targets = [p for r in rows for p in b.product_paths(stage, r['id'], original)]
    require(not log.exists() and not any(p.exists() or p.is_symlink() for p in targets), 'unreceipted ordinary products/log retained')
    intent_path = root / 'executions' / name(stage, shard); result_path = root / 'execution_results' / name(stage, shard)
    require(not intent_path.exists() and not result_path.exists(), 'interrupted stage intent retained')
    inputs = {r['id']: b.inspect_audio(r['input']['path'], r, input_audio=True) for r in rows}
    command = b.command(stage, [r['input']['path'] for r in rows], runtime, original)
    environment = v2.child_environment(runtime, gpu); started = utc()
    put(intent_path, {'status': 'continuation_stage_execution_intent', 'stage': stage, 'shard_index': shard['index'],
                     'item_ids': shard['item_ids'], 'original_run_contract_sha256': run_sha, 'gpu': gpu,
                     'command': command, 'child_environment': environment, 'inputs': inputs,
                     'driver': freeze['driver'], 'continuation_freeze_sha256': freeze['_sha256'], 'started_utc': started})
    b.require_idle_5090(gpu)
    with log.open('x') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    after = {r['id']: b.inspect_audio(r['input']['path'], r, input_audio=True) for r in rows}; completed = utc()
    put(result_path, {'status': 'continuation_stage_subprocess_exited', 'intent': b.binding(intent_path),
                      'returncode': result.returncode, 'log': b.binding(log), 'inputs_after': after, 'completed_utc': completed})
    require(result.returncode == 0 and inputs == after, 'ordinary command/input failure retained')
    if stage == 'beats' and any(not p.exists() for p in targets):
        return recover_shard(freeze, prior, shard, rows, root)
    products = b.verify_products(stage, rows, original)
    payload = {'status': 'passed', 'stage': stage, 'shard_index': shard['index'], 'item_ids': shard['item_ids'],
               'run_contract_sha256': run_sha, 'gpu': gpu, 'command': command, 'child_environment': environment,
               'started_utc': started, 'completed_utc': completed, 'inputs': inputs, 'outputs': products, 'log': b.binding(log),
               'ordinary_empty_beat_allowed_only_after_this_successful_command': stage == 'beats', **v2.SUCCESS_SCOPE}
    path = v2.receipt_path(original, stage, shard['index']); put(path, payload)
    return v2.validate_receipt(path, run_sha, stage, shard, rows, prior, original)


def audit_all(freeze, prior, cohort, root):
    original = Path(prior['output_root']); by_id = {r['id']: r for r in cohort['rows']}
    stage_receipts, logs, products, inputs, provenance, recovery, executions = {}, {}, {}, {}, {}, {}, {}
    expected_ordinary = set(); expected_external = {'driver_run.json', 'writer.lock'}
    for stage in ('allinone', 'beats'):
        for shard in prior['shards']:
            rows = [by_id[i] for i in shard['item_ids']]
            ordinary = v2.receipt_path(original, stage, shard['index'])
            recovered = root / 'recovered_receipts' / name('beats', shard)
            if ordinary.exists():
                require(stage != 'beats' or not recovered.exists(), 'duplicate ordinary/recovered shard evidence')
                ev = v2.validate_receipt(ordinary, freeze['original_run_contract']['sha256'], stage, shard, rows, prior, original)
                path = ordinary; kind = 'ordinary_v2'; expected_ordinary.add(ordinary.name)
                if str(ordinary) not in freeze['initial_partial_inventory']:
                    execution = validate_execution(root, stage, shard, rows, prior, freeze['original_run_contract']['sha256'], freeze)
                    executions[str(ordinary)] = execution
                    expected_external.update({'executions/' + name(stage, shard), 'execution_results/' + name(stage, shard)})
            else:
                require(stage == 'beats' and recovered.exists(), 'missing ordinary stage receipt')
                ev = validate_recovered(recovered, freeze, prior, shard, rows, root); path = recovered; kind = 'recovered_serializer_v1'
                recovery[str(path)] = {k: ev[k] for k in ('request', 'raw_replay', 'replay_log', 'missing_ids')}
                expected_external.update({folder + '/' + name('beats', shard) for folder in
                                          ('recovery_requests', 'raw_replays', 'recovered_receipts')})
                expected_external.add('replay_logs/' + name('beats', shard)[:-5] + '.log')
                if str(v2.log_path(original, 'beats', shard['index'])) not in freeze['initial_partial_inventory']:
                    execution = validate_execution(root, stage, shard, rows, prior, freeze['original_run_contract']['sha256'], freeze)
                    executions[str(path)] = execution
                    expected_external.update({'executions/' + name(stage, shard), 'execution_results/' + name(stage, shard)})
            stage_receipts[str(path)] = {'binding': ev['receipt'], 'payload_sha256': ev['payload_sha256']}
            require(ev['log']['path'] not in logs, 'duplicate stage log'); logs[ev['log']['path']] = ev['log']
            for p, record in ev['outputs'].items():
                require(p not in products, 'duplicate physical stage product'); products[p] = record
            for ident, record in ev['inputs'].items():
                require(ident not in inputs or inputs[ident] == record, 'cross-stage input disagreement'); inputs[ident] = record
                if stage == 'beats':
                    provenance[ident] = {'kind': kind, 'receipt': ev['receipt'],
                        'item_status': ('recovered_empty_unavailable' if ident in ev.get('missing_ids', []) else
                                        'preserved_cli_success' if kind == 'recovered_serializer_v1' else 'ordinary_v2_success')}
                    if kind == 'recovered_serializer_v1': provenance[ident]['raw_replay'] = ev['raw_replay']
    expected_products = {str(p) for r in cohort['rows'] for stage in ('allinone', 'beats') for p in b.product_paths(stage, r['id'], original)}
    require(set(products) == expected_products and set(inputs) == set(provenance) == set(by_id), 'combined evidence exact cohort/product sets')
    expected_original_files = expected_products | {str(original / 'run_contract.json')} | set(logs) | {
        str(original / 'receipts' / n) for n in expected_ordinary}
    require(set(files(original)) == expected_original_files, 'combined original root exhaustive file inventory')
    actual_external = {str(p.relative_to(root)) for p in files(root).values()}
    expected_external.discard('writer.lock')
    for optional in ('final_audit.json', 'completion.json'):
        if (root / optional).exists(): expected_external.add(optional)
    require(actual_external == expected_external, 'continuation external evidence exhaustive file inventory')
    evidence = {'stage_receipts': stage_receipts, 'logs': logs, 'products': products, 'inputs': inputs}
    evidence.update({key + '_sha256': b.value_hash(evidence[key]) for key in EVIDENCE_KEYS})
    return evidence, provenance, recovery, executions


def setup_context(freeze_path, freeze_sha, *, audio):
    freeze, prior, cohort, helper = validate_freeze(freeze_path, freeze_sha, audio=audio)
    freeze = {**freeze, '_path': str(Path(freeze_path).absolute()), '_sha256': freeze_sha}
    runtime = v2.verify_runtime(prior)
    run = b.read_json(freeze['original_run_contract']['path'])
    require(runtime == run['runtime_start'], 'frozen original runtime changed')
    return freeze, prior, cohort, helper, runtime


def final_records(freeze, prior, cohort, helper, runtime):
    root, original = Path(freeze['output_root']), Path(prior['output_root'])
    b.require_hash(freeze['_path'], freeze['_sha256'])
    for key in ('original_parent_freeze', 'original_run_contract', 'v2_runner', 'driver', 'tests'):
        b._binding_matches(freeze[key], audio=True)
    validate_initial_snapshot(freeze, original, audio=True)
    source = v2.rehash_source_graph(prior, cohort, helper)
    original_run = b.read_json(freeze['original_run_contract']['path'])
    require(source == original_run['source_graph_start'], 'original source graph changed')
    require(get(root / 'driver_run.json') == driver_record(freeze, runtime), 'driver execution record changed')
    evidence, provenance, recovery, executions = audit_all(freeze, prior, cohort, root)
    return {'status': 'passed_native30_inference_with_frozen_recovery_not_feature_extraction', 'version': VERSION,
            'rows': len(cohort['rows']), 'original_v2_status': 'partial_without_own_completion',
            'continuation_parent_freeze': {'path': freeze['_path'], 'sha256': freeze['_sha256']},
            'original_parent_freeze': freeze['original_parent_freeze'], 'original_run_contract': freeze['original_run_contract'],
            'cohort_contract': prior['cohort_contract'],
            'driver_execution': b.binding(root / 'driver_run.json'), 'source_graph': source, 'runtime': runtime,
            **evidence, 'beat_provenance': provenance, 'recovery_evidence': recovery,
            'ordinary_continuation_executions': executions, 'recovered_ids': sorted(i for i, p in provenance.items()
                if p['item_status'] == 'recovered_empty_unavailable'), **SCOPE}


def verify_completion(freeze_path, freeze_sha):
    freeze, prior, cohort, helper, runtime = setup_context(freeze_path, freeze_sha, audio=True)
    root = Path(freeze['output_root']); expected = final_records(freeze, prior, cohort, helper, runtime)
    audit_binding = v2.bind_canonical_json(root / 'final_audit.json', expected)
    completion = {**expected, 'final_audit': audit_binding}
    binding = v2.bind_canonical_json(root / 'completion.json', completion)
    require(final_records(freeze, prior, cohort, helper, v2.verify_runtime(prior)) == expected, 'combined evidence changed during verification')
    return {'status': 'verified_complete_native30_inference_with_frozen_recovery', 'rows': len(cohort['rows']),
            'completion': binding, 'final_audit': audit_binding,
            'evidence_sha256': b.value_hash({k: expected[k] for key in EVIDENCE_KEYS for k in (key, key + '_sha256')}),
            'beat_provenance': expected['beat_provenance']}


def driver_record(freeze, runtime):
    return {'status': 'separately_frozen_continuation_execution',
            'parent_freeze': {'path': freeze['_path'], 'sha256': freeze['_sha256']},
            'driver': freeze['driver'], 'tests': freeze['tests'], 'v2_runner': freeze['v2_runner'],
            'original_run_contract': freeze['original_run_contract'],
            'initial_partial_inventory_sha256': b.value_hash(freeze['initial_partial_inventory']),
            'policy': POLICY, 'runtime': runtime, **SCOPE}


def run(freeze_path, freeze_sha):
    freeze, prior, cohort, helper, runtime = setup_context(freeze_path, freeze_sha, audio=True)
    root, original = Path(freeze['output_root']), Path(prior['output_root'])
    root.mkdir(exist_ok=True)
    with (original / 'writer.lock').open('a') as original_lock, (root / 'writer.lock').open('a') as lock:
        fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_initial_snapshot(freeze, original, audio=True, first=not (root / 'driver_run.json').exists())
        for folder in ('executions', 'execution_results', 'recovery_requests', 'raw_replays', 'replay_logs', 'recovered_receipts'):
            (root / folder).mkdir(exist_ok=True)
            require((root / folder).is_dir() and not (root / folder).is_symlink(), 'unsafe continuation directory')
        driver = driver_record(freeze, runtime)
        driver_path = root / 'driver_run.json'
        if driver_path.exists(): require(get(driver_path) == driver, 'existing continuation driver conflict')
        else: put(driver_path, driver)
        by_id = {r['id']: r for r in cohort['rows']}; stopped = threading.Event()

        def worker(stage, gpu, assigned):
            for shard in assigned:
                if stopped.is_set(): return
                rows = [by_id[i] for i in shard['item_ids']]
                ordinary = v2.receipt_path(original, stage, shard['index'])
                recovered = root / 'recovered_receipts' / name('beats', shard)
                try:
                    b.require_hash(freeze_path, freeze_sha)
                    if ordinary.exists():
                        v2.validate_receipt(ordinary, freeze['original_run_contract']['sha256'], stage, shard, rows, prior, original)
                    elif stage == 'beats' and recovered.exists():
                        validate_recovered(recovered, freeze, prior, shard, rows, root)
                    elif stage == 'beats' and v2.log_path(original, stage, shard['index']).exists():
                        recover_shard(freeze, prior, shard, rows, root)
                    else:
                        ordinary_shard(freeze, prior, shard, rows, stage, root)
                except Exception:
                    stopped.set(); raise

        jobs = [(stage, gpu, assigned) for stage, gpus in (('allinone', prior['aio_gpus']), ('beats', prior['beat_gpus']))
                for gpu, assigned in b.assignments(prior['shards'], gpus)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(worker, *job) for job in jobs]
            for future in futures: future.result()
        final_freeze, final_prior, final_cohort, final_helper, final_runtime = setup_context(freeze_path, freeze_sha, audio=True)
        require((final_freeze, final_prior, final_cohort, final_runtime) == (freeze, prior, cohort, runtime), 'continuation authority/runtime changed')
        expected = final_records(freeze, prior, cohort, final_helper, runtime)
        audit_path = root / 'final_audit.json'
        if audit_path.exists(): v2.bind_canonical_json(audit_path, expected)
        else: b.write_new(audit_path, expected)
        completion = {**expected, 'final_audit': v2.bind_canonical_json(audit_path, expected)}
        require(final_records(freeze, prior, cohort, helper, v2.verify_runtime(prior)) == expected, 'end evidence mutation')
        path = root / 'completion.json'
        if path.exists(): v2.bind_canonical_json(path, completion)
        else: b.write_new(path, completion)
        return {'status': completion['status'], 'completion': b.binding(path), 'recovered_ids': completion['recovered_ids']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', required=True); parser.add_argument('--frozen-sha256', required=True)
    parser.add_argument('--mode', choices=('preflight', 'run', 'replay'), default='preflight')
    parser.add_argument('--request'); parser.add_argument('--request-sha256')
    args = parser.parse_args()
    if args.mode == 'run': result = run(args.frozen, args.frozen_sha256)
    else:
        freeze, prior, cohort, helper, runtime = setup_context(args.frozen, args.frozen_sha256, audio=False)
        if args.mode == 'replay':
            require(args.request and args.request_sha256, 'pinned replay request required')
            replay_worker(freeze, prior, cohort, args.request, args.request_sha256); result = {'status': 'replay_complete'}
        else:
            result = {'status': 'metadata_preflight_only_no_audio_reads_or_inference', 'rows': len(cohort['rows']),
                      'initial_files': len(freeze['initial_partial_inventory']), 'runtime': runtime, **SCOPE}
    print(b.canonical(result).decode().strip(), flush=True)


if __name__ == '__main__':
    main()
