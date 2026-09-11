"""Separately frozen operational dispatch successor; numerical helpers unchanged.

The previous recovery root is immutable. An observed GPU-guard abort is not a
subprocess execution. New attempts have their own root, driver and placement.
No authorizing freeze is produced by this program.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import socket
import sys
import threading

import continue_native30_empty_beats_v2 as r

VERSION = 'resume_native30_gpu_guard_v2'
BASE_SHA = '91f5fec0fb25b163f69d6b46e032a5f9b27fb7206f93ac7aa4afb4933ae0e6bf'
BASE_FREEZE_SHA = 'bcbe3aafe98ab775bc769739352d467a9df1c33998d1245b0c918af0b7441485'
BLOCKED_SHA = '9b0837bb31c5f31488567ac98d268889ab741c575a01ef919411868481ae5d7f'
TERMINAL_SHA = 'ce77d219363ef6267bb6f97331711591044f1fe57648b14faee68dcd3b8c8e0a'
HOST = '10-60-169-42'
PLACEMENT = {'host': HOST, 'allinone_logical': [0, 1, 2, 3], 'allinone_physical': [7],
             'beats_logical': [4, 5], 'beats_physical': [6, 5]}
FREEZE_VERSION = 'native30-gpu-guard-operational-parent-freeze-v2'
FREEZE_STATUS = 'parent_frozen_for_native30_gpu_guard_operational_continuation_v2'
STATUS = 'passed_native30_operational_continuation_not_feature_extraction'
b, v2, require = r.b, r.v2, r.require
EVIDENCE_KEYS = r.EVIDENCE_KEYS
b.require_hash(Path(r.__file__).resolve(), BASE_SHA)
require(Path(r.__file__).resolve() == Path(__file__).resolve().with_name('continue_native30_empty_beats_v2.py'), 'base import origin')


def effective_prior(prior, placement):
    require(placement == PLACEMENT and prior['aio_gpus'] == PLACEMENT['allinone_logical'] and
            prior['beat_gpus'] == PLACEMENT['beats_logical'], 'only reviewed physical GPU dispatch remap permitted')
    return {**prior, 'aio_gpus': list(PLACEMENT['allinone_physical']), 'beat_gpus': list(PLACEMENT['beats_physical'])}


def snapshots(freeze, prior, *, audio, first=False):
    for key, root, exact in (('stopped_original_inventory', Path(prior['output_root']), first),
                             ('stopped_recovery_inventory', Path(freeze['_base']['output_root']), True)):
        current, expected = r.files(root), freeze[key]
        require(set(expected) <= set(current) and (not exact or set(expected) == set(current)), 'stopped snapshot file set changed')
        for path, entry in expected.items():
            require(entry['path'] == path and b.safe_path(path).is_relative_to(root), 'snapshot path escaped root')
            b._binding_matches(entry, audio=audio)
    require(not any((Path(prior['output_root']) / n).exists() or (Path(freeze['_base']['output_root']) / n).exists()
                    for n in ('completion.json', 'final_audit.json')), 'historical roots must remain partial')


def aborted_attempt(freeze, prior, cohort, *, first=False):
    """Exact incident, not a general exception/orphan bypass; absences snapshot-bound."""
    oldroot = Path(freeze['_base']['output_root']); original = Path(prior['output_root'])
    shard = prior['shards'][96]; require(shard['index'] == 96 and len(shard['item_ids']) == 24, 'reviewed blocked shard only')
    path = oldroot / 'executions/beats_096.json'; terminal = freeze['terminal_log']
    require(freeze['blocked_intent'] == b.binding(path) and freeze['blocked_intent']['sha256'] == BLOCKED_SHA,
            'exact historical blocked intent required')
    require(terminal['sha256'] == TERMINAL_SHA, 'exact reviewed terminal log required'); b._binding_matches(terminal, audio=True)
    intent = r.get(path); base = freeze['_base']
    rows = {x['id']: x for x in cohort['rows']}; ordered = [rows[i] for i in shard['item_ids']]
    require(intent['status'] == 'continuation_stage_execution_intent' and intent['stage'] == 'beats' and
            intent['shard_index'] == 96 and intent['item_ids'] == shard['item_ids'] and intent['gpu'] == 4 and
            intent['started_utc'] == '2026-09-08T06:49:42.978609+00:00' and intent['driver'] == base['driver'] and
            intent['continuation_freeze_sha256'] == BASE_FREEZE_SHA and
            intent['original_run_contract_sha256'] == base['original_run_contract']['sha256'] and
            intent['command'] == b.command('beats', [x['input']['path'] for x in ordered], Path(prior['runtime_root']), original) and
            intent['child_environment'] == v2.child_environment(prior['runtime_root'], 4), 'blocked intent authority/command mismatch')
    absent = [v2.log_path(original, 'beats', 96), v2.receipt_path(original, 'beats', 96),
              oldroot / 'execution_results/beats_096.json',
              *[p for row in ordered for p in b.product_paths('beats', row['id'], original)]]
    snapshot = {**freeze['stopped_original_inventory'], **freeze['stopped_recovery_inventory']}
    require(snapshot.get(str(path)) == freeze['blocked_intent'] and all(str(p) not in snapshot for p in absent),
            'blocked prelaunch absences not bound to stopped snapshots')
    require(not (oldroot / 'execution_results/beats_096.json').exists(), 'blocked intent must never gain historical execution result')
    if first: require(all(not p.exists() and not p.is_symlink() for p in absent), 'blocked attempt has launch/output artifacts')
    return {'status': 'reviewed_aborted_before_subprocess_launch', 'stage': 'beats', 'shard_index': 96,
            'item_ids': shard['item_ids'], 'historical_intent': freeze['blocked_intent'], 'terminal_log': terminal,
            'source': freeze['base_driver'], 'source_order': ['intent_written', 'require_idle_5090_raised', 'log_open_not_reached', 'subprocess_run_not_reached'],
            'subprocess_launched': False, 'execution_result_created': False,
            'absence_paths_at_stopped_snapshot': [str(p) for p in absent],
            'stopped_original_inventory_sha256': b.value_hash(freeze['stopped_original_inventory']),
            'stopped_recovery_inventory_sha256': b.value_hash(freeze['stopped_recovery_inventory'])}


def setup_context(path, sha, *, audio):
    path = b.safe_path(path); b.require_hash(path, sha); f = b.read_json(path)
    require(f.get('version') == FREEZE_VERSION and f.get('status') == FREEZE_STATUS and f.get('placement') == PLACEMENT and
            f.get('policy') == r.POLICY and all(f.get(k) == v for k, v in r.SCOPE.items()), 'operational freeze scope')
    for key in ('base_freeze', 'base_driver', 'driver', 'tests', 'terminal_log', 'blocked_intent'):
        b._binding_matches(f[key], audio=True)
    require(f['base_freeze']['sha256'] == BASE_FREEZE_SHA and f['base_driver'] == b.binding(Path(r.__file__).resolve()) and
            f['driver'] == b.binding(Path(__file__).resolve()) and
            f['tests'] == b.binding(Path(__file__).resolve().with_name('test_' + VERSION + '.py')), 'operational code/parent authority')
    base, prior, cohort, helper, runtime = r.setup_context(f['base_freeze']['path'], f['base_freeze']['sha256'], audio=audio)
    root = b.safe_path(f['output_root']); roots = [root, Path(base['output_root']), Path(prior['output_root'])]
    require(root.parent.is_dir() and all(not a.is_relative_to(c) for a in roots for c in roots if a != c) and len(set(roots)) == 3,
            'three disjoint evidence/product roots required')
    f = {**f, '_path': str(path), '_sha256': sha, '_base': base}
    effective = effective_prior(prior, f['placement'])
    snapshots(f, prior, audio=audio); aborted_attempt(f, prior, cohort)
    # Only new products are considered a new execution epoch by reused helpers.
    ctx = {**base, **f, 'original_run_contract': base['original_run_contract'],
           'initial_partial_inventory': f['stopped_original_inventory'], 'resume_continuation_inventory': {}}
    return ctx, prior, effective, cohort, helper, runtime


def dispatch_record(f, prior, effective, stage, shard):
    return {'status': 'operational_physical_dispatch', 'driver': f['driver'],
            'parent_freeze': {'path': f['_path'], 'sha256': f['_sha256']}, 'host': HOST,
            'stage': stage, 'shard_index': shard['index'], 'item_ids': shard['item_ids'],
            'logical_gpu': v2.assigned_gpu(prior, stage, shard['index']),
            'physical_gpu': v2.assigned_gpu(effective, stage, shard['index']),
            'placement': PLACEMENT, 'blocked_attempt': b.binding(Path(f['output_root']) / 'blocked_attempt.json')
                if stage == 'beats' and shard['index'] == 96 else None}


def immutable_put(path, payload):
    if path.exists(): require(r.get(path) == payload, 'conflicting immutable operational evidence')
    else: r.put(path, payload)


def launch_stage(f, prior, effective, stage, shard, rows):
    require(socket.gethostname() == HOST, 'dispatch host mismatch')
    root = Path(f['output_root']); original = Path(prior['output_root'])
    gpu = v2.assigned_gpu(effective, stage, shard['index'])
    b.require_idle_5090(gpu)  # Before a new dispatch or intent; unchanged helper guards again before subprocess.
    immutable_put(root / 'dispatch' / r.name(stage, shard), dispatch_record(f, prior, effective, stage, shard))
    if stage == 'beats' and v2.log_path(original, stage, shard['index']).exists():
        return r.recover_shard(f, effective, shard, rows, root)
    return r.ordinary_shard(f, effective, shard, rows, stage, root)


def stage_evidence(f, prior, effective, stage, shard, rows):
    """Route by stopped bytes, never by a relabelled historical GPU field."""
    original, oldroot, root = Path(prior['output_root']), Path(f['_base']['output_root']), Path(f['output_root'])
    ordinary = v2.receipt_path(original, stage, shard['index'])
    recovered_old = oldroot / 'recovered_receipts' / r.name('beats', shard)
    recovered_new = root / 'recovered_receipts' / r.name('beats', shard)
    historical = str(ordinary) in f['stopped_original_inventory'] or (stage == 'beats' and recovered_old.exists())
    authority, config, evidence_root = (f['_base'], prior, oldroot) if historical else (f, effective, root)
    run_sha = f['original_run_contract']['sha256']; execution = None
    if ordinary.exists():
        require(stage != 'beats' or not (recovered_old.exists() or recovered_new.exists()), 'duplicate beat receipt')
        ev = v2.validate_receipt(ordinary, run_sha, stage, shard, rows, config, original); kind = 'ordinary_v2'
        if str(ordinary) not in authority['initial_partial_inventory']:
            execution = r.validate_execution(evidence_root, stage, shard, rows, config, run_sha, authority)
    else:
        require(stage == 'beats', 'missing AIO receipt'); target = recovered_old if historical else recovered_new
        require(target.exists() and not (recovered_old.exists() and recovered_new.exists()), 'missing/duplicate recovered receipt')
        ev = r.validate_recovered(target, authority, config, shard, rows, evidence_root); kind = 'recovered_serializer_v1'
        if str(v2.log_path(original, stage, shard['index'])) not in authority['initial_partial_inventory']:
            execution = r.validate_execution(evidence_root, stage, shard, rows, config, run_sha, authority)
    dispatch = None
    if not historical:
        dispatch_path = root / 'dispatch' / r.name(stage, shard)
        require(r.get(dispatch_path) == dispatch_record(f, prior, effective, stage, shard), 'new dispatch provenance mismatch')
        dispatch = b.binding(dispatch_path)
        require(execution is not None, 'new stage cannot bypass execution provenance')
    return ev, kind, ('preserved_recovery_v2' if historical else 'operational_dispatch_v2'), execution, dispatch


def audit_all(f, prior, effective, cohort):
    root, original = Path(f['output_root']), Path(prior['output_root'])
    evidence = {k: {} for k in EVIDENCE_KEYS}; provenance, executions, recovery = {}, {}, {}
    expected_external = {'driver_run.json', 'blocked_attempt.json'}
    for stage in ('allinone', 'beats'):
        for shard in prior['shards']:
            by_id = {row['id']: row for row in cohort['rows']}; rows = [by_id[i] for i in shard['item_ids']]
            ev, kind, epoch, execution, dispatch = stage_evidence(f, prior, effective, stage, shard, rows)
            path = ev['receipt']['path']; evidence['stage_receipts'][path] = {'binding': ev['receipt'], 'payload_sha256': ev['payload_sha256']}
            require(ev['log']['path'] not in evidence['logs'], 'duplicate stage log'); evidence['logs'][ev['log']['path']] = ev['log']
            for p, item in ev['outputs'].items():
                require(p not in evidence['products'], 'duplicate product'); evidence['products'][p] = item
            for ident, item in ev['inputs'].items():
                require(ident not in evidence['inputs'] or evidence['inputs'][ident] == item, 'cross-stage input disagreement')
                evidence['inputs'][ident] = item
                if stage == 'beats':
                    provenance[ident] = {'kind': kind, 'receipt': ev['receipt'], 'execution_epoch': epoch,
                        'item_status': ('recovered_empty_unavailable' if ident in ev.get('missing_ids', []) else
                                        'preserved_cli_success' if kind == 'recovered_serializer_v1' else 'ordinary_v2_success')}
                    if kind == 'recovered_serializer_v1': provenance[ident]['raw_replay'] = ev['raw_replay']
                    if dispatch: provenance[ident]['physical_dispatch'] = dispatch
            if execution: executions[path] = {'execution_epoch': epoch, **execution, 'dispatch': dispatch}
            if kind == 'recovered_serializer_v1': recovery[path] = {k: ev[k] for k in ('request', 'raw_replay', 'replay_log', 'missing_ids')}
            if epoch == 'operational_dispatch_v2':
                expected_external.update(folder + '/' + r.name(stage, shard) for folder in ('dispatch', 'executions', 'execution_results'))
                if kind == 'recovered_serializer_v1':
                    expected_external.update(folder + '/' + r.name(stage, shard) for folder in ('recovery_requests', 'raw_replays', 'recovered_receipts'))
                    expected_external.add('replay_logs/' + r.name(stage, shard)[:-5] + '.log')
    expected_products = {str(p) for row in cohort['rows'] for stage in ('allinone', 'beats') for p in b.product_paths(stage, row['id'], original)}
    require(set(evidence['products']) == expected_products and set(evidence['inputs']) == set(provenance) == {r['id'] for r in cohort['rows']}, 'exact cohort/product sets')
    expected_original = expected_products | set(evidence['logs']) | {str(original / 'run_contract.json')} | {
        p for p in evidence['stage_receipts'] if Path(p).is_relative_to(original)}
    require(set(r.files(original)) == expected_original, 'original exhaustive inventory')
    for optional in ('final_audit.json', 'completion.json'):
        if (root / optional).exists(): expected_external.add(optional)
    require({str(p.relative_to(root)) for p in r.files(root).values()} == expected_external, 'operational exhaustive inventory')
    evidence.update({k + '_sha256': b.value_hash(evidence[k]) for k in EVIDENCE_KEYS})
    return evidence, provenance, executions, recovery


def driver_record(f, runtime):
    return {'status': 'separately_frozen_operational_dispatch_execution', 'version': VERSION,
            'driver': f['driver'], 'tests': f['tests'], 'base_driver': f['base_driver'], 'base_freeze': f['base_freeze'],
            'parent_freeze': {'path': f['_path'], 'sha256': f['_sha256']}, 'placement': PLACEMENT,
            'runtime': runtime, 'stopped_original_inventory_sha256': b.value_hash(f['stopped_original_inventory']),
            'stopped_recovery_inventory_sha256': b.value_hash(f['stopped_recovery_inventory']), **r.SCOPE}


def final_records(f, prior, effective, cohort, helper, runtime):
    root = Path(f['output_root']); b.require_hash(f['_path'], f['_sha256'])
    for key in ('base_freeze', 'base_driver', 'driver', 'tests', 'terminal_log', 'blocked_intent'):
        b._binding_matches(f[key], audio=True)
    snapshots(f, prior, audio=True)
    source = v2.rehash_source_graph(prior, cohort, helper); run = b.read_json(f['original_run_contract']['path'])
    require(source == run['source_graph_start'] and runtime == run['runtime_start'], 'original scientific/runtime graph changed')
    base = f['_base']; oldroot = Path(base['output_root'])
    require(r.get(oldroot / 'driver_run.json') == r.load_v1().driver_record(base['_previous'], runtime) and
            r.get(oldroot / 'driver_run_v2.json') == r.driver_record(base, runtime), 'historical driver authority')
    require(r.get(root / 'driver_run.json') == driver_record(f, runtime), 'operational driver authority')
    require(r.get(root / 'blocked_attempt.json') == aborted_attempt(f, prior, cohort), 'aborted attempt certificate changed')
    evidence, provenance, executions, recovery = audit_all(f, prior, effective, cohort)
    new_attempt = executions[str(v2.receipt_path(Path(prior['output_root']), 'beats', 96))] if str(v2.receipt_path(Path(prior['output_root']), 'beats', 96)) in executions else executions[str(root / 'recovered_receipts/beats_096.json')]
    return {'status': STATUS, 'version': VERSION, 'rows': len(cohort['rows']), 'original_v2_status': 'partial_without_own_completion',
            'base_recovery_v2_status': 'partial_preserved_without_own_completion', 'operational_parent_freeze': {'path': f['_path'], 'sha256': f['_sha256']},
            'base_freeze': f['base_freeze'], 'base_driver': f['base_driver'], 'driver_execution': b.binding(root / 'driver_run.json'),
            'original_parent_freeze': base['original_parent_freeze'], 'original_run_contract': base['original_run_contract'],
            'cohort_contract': prior['cohort_contract'], 'placement': PLACEMENT, 'source_graph': source, 'runtime': runtime,
            'blocked_attempt': b.binding(root / 'blocked_attempt.json'), 'replacement_attempt_096': new_attempt,
            **evidence, 'beat_provenance': provenance, 'stage_executions': executions, 'recovery_evidence': recovery,
            'recovered_ids': sorted(i for i, p in provenance.items() if p['item_status'] == 'recovered_empty_unavailable'), **r.SCOPE}


def verify_completion(path, sha):
    context = setup_context(path, sha, audio=True); f, prior, effective, cohort, helper, runtime = context
    with ExitStack() as stack:
        for folder in (Path(prior['output_root']), Path(f['_base']['output_root']), Path(f['output_root'])):
            lock = stack.enter_context((folder / 'writer.lock').open('rb'))
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return verify_locked(context)


def verify_locked(context):
    f, prior, effective, cohort, helper, runtime = context
    root = Path(f['output_root']); expected = final_records(*context)
    audit = v2.bind_canonical_json(root / 'final_audit.json', expected)
    completion = v2.bind_canonical_json(root / 'completion.json', {**expected, 'final_audit': audit})
    require(final_records(f, prior, effective, cohort, helper, v2.verify_runtime(prior)) == expected, 'end evidence mutation')
    return {'status': 'verified_complete_native30_operational_continuation', 'rows': len(cohort['rows']), 'completion': completion,
            'final_audit': audit, 'evidence_sha256': b.value_hash({k: expected[k] for key in EVIDENCE_KEYS for k in (key, key + '_sha256')}),
            'beat_provenance': expected['beat_provenance']}


def run(path, sha):
    context = setup_context(path, sha, audio=True); f, prior, effective, cohort, helper, runtime = context
    require(socket.gethostname() == HOST, 'operational dispatch host mismatch')
    root, original, oldroot = Path(f['output_root']), Path(prior['output_root']), Path(f['_base']['output_root'])
    root.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for folder, mode, operation in ((original, 'r+b', fcntl.LOCK_EX), (oldroot, 'rb', fcntl.LOCK_SH), (root, 'a+b', fcntl.LOCK_EX)):
            lock = stack.enter_context((folder / 'writer.lock').open(mode)); fcntl.flock(lock, operation | fcntl.LOCK_NB)
        first = not (root / 'driver_run.json').exists(); snapshots(f, prior, audio=True, first=first)
        certificate = aborted_attempt(f, prior, cohort, first=first)
        for folder in ('dispatch', 'executions', 'execution_results', 'recovery_requests', 'raw_replays', 'replay_logs', 'recovered_receipts'):
            (root / folder).mkdir(exist_ok=True); require(not (root / folder).is_symlink(), 'unsafe evidence folder')
        immutable_put(root / 'blocked_attempt.json', certificate); immutable_put(root / 'driver_run.json', driver_record(f, runtime))
        by_id = {row['id']: row for row in cohort['rows']}; stopped = threading.Event()
        def worker(stage, gpu, shards):
            for shard in shards:
                if stopped.is_set(): return
                rows = [by_id[i] for i in shard['item_ids']]
                try:
                    b.require_hash(path, sha); require(socket.gethostname() == HOST, 'dispatch host changed')
                    ordinary = v2.receipt_path(original, stage, shard['index'])
                    recovered = [x / 'recovered_receipts' / r.name('beats', shard) for x in (oldroot, root)]
                    if ordinary.exists() or (stage == 'beats' and any(p.exists() for p in recovered)):
                        stage_evidence(f, prior, effective, stage, shard, rows); continue
                    launch_stage(f, prior, effective, stage, shard, rows)
                    stage_evidence(f, prior, effective, stage, shard, rows)
                except Exception: stopped.set(); raise
        jobs = [(stage, gpu, shards) for stage, gpus in (('allinone', effective['aio_gpus']), ('beats', effective['beat_gpus']))
                for gpu, shards in b.assignments(prior['shards'], gpus)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(worker, *job) for job in jobs]
            for future in futures: future.result()
        require(setup_context(path, sha, audio=True) == context, 'operational authority/runtime changed')
        expected = final_records(*context); audit = root / 'final_audit.json'
        if not audit.exists(): b.write_new(audit, expected)
        audit_binding = v2.bind_canonical_json(audit, expected)
        require(final_records(f, prior, effective, cohort, helper, v2.verify_runtime(prior)) == expected, 'end evidence mutation')
        completion = {**expected, 'final_audit': audit_binding}; target = root / 'completion.json'
        if not target.exists(): b.write_new(target, completion)
        return {'status': STATUS, 'completion': v2.bind_canonical_json(target, completion)}


def replay_worker(freeze, prior, cohort, request_path, request_sha):
    """Same pinned recovery numerics; records THIS actual operational CLI invocation."""
    require(socket.gethostname() == HOST, 'replay host mismatch')
    root = Path(freeze['output_root']); request_path = b.safe_path(request_path)
    b.require_hash(request_path, request_sha); plan = r.get(request_path)
    shard = prior['shards'][plan['shard_index']]
    require(request_path == root / 'recovery_requests' / r.name('beats', shard), 'replay request path')
    rows = [row for row in cohort['rows'] if row['id'] in shard['item_ids']]
    r.validate_plan(plan, freeze, prior, shard, rows, root)
    gpu = v2.assigned_gpu(prior, 'beats', shard['index'])
    require(dict(os.environ) == v2.child_environment(prior['runtime_root'], gpu), 'replay child environment')
    before_cuda = dict(os.environ); b.require_idle_5090(gpu)
    import torch
    from beat_this.inference import File2Beats
    from beat_this.utils import infer_beat_numbers
    require(torch.cuda.is_available() and 'RTX 5090' in torch.cuda.get_device_name(0), 'replay mapped RTX5090')
    checkpoint = Path(prior['runtime_root']) / 'checkpoints/hub/checkpoints/beat_this-final0.ckpt'
    b.require_hash(checkpoint, r.CHECKPOINT_SHA)
    model = File2Beats(str(checkpoint), device='cuda:0', float16=True, dbn=False)
    by_id = {row['id']: row for row in rows}; records = []
    for ident in plan['missing_ids']:
        row = by_id[ident]; observed = b.inspect_audio(row['input']['path'], row, input_audio=True)
        beats, downbeats = model(row['input']['path']); r.validate_arrays(beats, downbeats)
        try: infer_beat_numbers(beats, downbeats)
        except ValueError as exc: require(str(exc) == r.ERROR, 'different serializer exception')
        else: raise ValueError('reviewed serializer exception was not reproduced')
        require(b.inspect_audio(row['input']['path'], row, input_audio=True) == observed, 'input changed during replay')
        records.append({'id': ident, 'input': observed, 'beats': r.array_record(beats), 'downbeats': r.array_record(downbeats),
                        'exception_type': 'ValueError', 'exception_message': r.ERROR})
    r.put(root / 'raw_replays' / r.name('beats', shard), {
        'status': 'reproduced_exact_empty_beat_serializer_failure', 'request': b.binding(request_path), 'policy': r.POLICY,
        'gpu': gpu, 'gpu_name': torch.cuda.get_device_name(0), 'cuda_device': 'cuda:0', 'child_environment': dict(os.environ),
        'checkpoint': b.binding(checkpoint), 'driver_sha256': freeze['driver']['sha256'], 'child_environment_before_cuda_init': before_cuda,
        'environment_transition': r.ENVIRONMENT_TRANSITION, 'recovery_api': 'File2Beats(checkpoint,device=cuda:0,float16=True,dbn=False)(input_path)',
        'recovery_invocation': [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        'runtime_sha256': b.value_hash(b.read_json(freeze['original_run_contract']['path'])['runtime_start']),
        'records': records, 'completed_utc': r.utc(), **r.SCOPE})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--frozen', required=True); p.add_argument('--frozen-sha256', required=True)
    p.add_argument('--mode', choices=('preflight', 'run', 'replay', 'verify'), default='preflight')
    p.add_argument('--request'); p.add_argument('--request-sha256'); a = p.parse_args()
    if a.mode == 'run': result = run(a.frozen, a.frozen_sha256)
    elif a.mode == 'verify': result = verify_completion(a.frozen, a.frozen_sha256)
    else:
        f, prior, effective, cohort, helper, runtime = setup_context(a.frozen, a.frozen_sha256, audio=False)
        if a.mode == 'replay':
            require(a.request and a.request_sha256, 'pinned replay request required')
            replay_worker(f, effective, cohort, a.request, a.request_sha256); result = {'status': 'replay_complete'}
        else: result = {'status': 'metadata_preflight_only_no_inference', 'rows': len(cohort['rows']), 'placement': PLACEMENT, **r.SCOPE}
    print(b.canonical(result).decode().strip(), flush=True)


if __name__ == '__main__': main()

