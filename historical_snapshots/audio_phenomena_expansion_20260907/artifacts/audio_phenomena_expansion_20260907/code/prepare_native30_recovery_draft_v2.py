"""Read-only stopped-run inventory and non-authorizing recovery freeze draft.

Hashes original bytes under an exclusive existing writer lock; never decodes
waveforms, invokes models, repairs products, or changes original files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import os
from pathlib import Path
import re
import socket
import sys

HERE = Path(__file__).resolve().parent
RECOVERY_SHA = '2e10ff670964a3dc6840c858565fb2056d09ee067a3fcd9ed3334d3c3dcb60a8'
RECOVERY_TESTS_SHA = '088ef96d6c252d5adc4d7b79924bfb394b94275576357581cc7fbd1b3e691a32'
path = HERE / 'continue_native30_empty_beats_v1.py'
if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != RECOVERY_SHA:
    raise ValueError('approved continuation driver changed')
r = importlib.import_module('continue_native30_empty_beats_v1')
if Path(r.__file__).resolve() != path: raise ValueError('continuation import origin')
b, v2, require = r.b, r.v2, r.require


def stat_key(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def stable_binding(path):
    before = stat_key(path); binding = b.binding(path)
    require(stat_key(path) == before, 'file changed during snapshot hashing: ' + str(path))
    return str(path), binding, before


def receipt_metadata(path, stage, shard, rows, prior, run_sha, snapshot):
    envelope = b.read_json(path); payload = envelope['payload']
    require(set(envelope) == {'payload', 'receipt_sha256'} and envelope['receipt_sha256'] == b.value_hash(payload), 'receipt envelope/hash')
    require(set(payload) == v2.RECEIPT_KEYS and payload['status'] == 'passed' and payload['stage'] == stage and
            payload['shard_index'] == shard['index'] and payload['item_ids'] == shard['item_ids'] and
            payload['run_contract_sha256'] == run_sha and payload['gpu'] == v2.assigned_gpu(prior, stage, shard['index']) and
            all(payload[k] == v for k, v in v2.SUCCESS_SCOPE.items()), 'receipt identity/authority/scope')
    original = Path(prior['output_root'])
    require(payload['command'] == b.command(stage, [row['input']['path'] for row in rows], Path(prior['runtime_root']), original) and
            payload['child_environment'] == v2.child_environment(prior['runtime_root'], payload['gpu']), 'receipt command/environment')
    log = str(v2.log_path(original, stage, shard['index']))
    require(payload['log'] == snapshot[log] and isinstance(payload['started_utc'], str) and isinstance(payload['completed_utc'], str), 'receipt log/timestamps')
    expected = {str(p) for row in rows for p in b.product_paths(stage, row['id'], original)}
    require(set(payload['outputs']) == expected, 'receipt exact product set')
    for path, declared in payload['outputs'].items():
        require({k: declared[k] for k in ('path', 'bytes', 'sha256')} == snapshot[path], 'receipt product bytes changed: ' + path)
    require(set(payload['inputs']) == {row['id'] for row in rows}, 'receipt input set')
    for row in rows:
        declared = payload['inputs'][row['id']]
        require({k: declared[k] for k in ('path', 'bytes', 'sha256')} == row['input'] and
                declared['waveform_float32_sha256'] == row['waveform_float32_sha256'], 'receipt frozen input declaration')
    require(payload['ordinary_empty_beat_allowed_only_after_this_successful_command'] is (stage == 'beats'), 'ordinary empty provenance')
    return {'payload_sha256': envelope['receipt_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--parent-freeze', required=True)
    parser.add_argument('--parent-freeze-sha256', required=True); parser.add_argument('--run-sha256', required=True)
    parser.add_argument('--continuation-root', required=True); parser.add_argument('--audit', required=True); parser.add_argument('--draft', required=True)
    args = parser.parse_args()
    original = b.safe_path(args.root); continuation = b.safe_path(args.continuation_root)
    audit_path, draft_path = b.safe_path(args.audit), b.safe_path(args.draft)
    require(not audit_path.exists() and not draft_path.exists() and audit_path.parent.is_dir() and draft_path.parent.is_dir(), 'new metadata outputs required')
    require(not continuation.exists() and continuation.parent.is_dir() and not continuation.is_relative_to(original) and
            not original.is_relative_to(continuation), 'new separate proposed continuation root required')
    require(not any(p.is_relative_to(original) or p.is_relative_to(continuation) for p in (audit_path, draft_path)), 'metadata output overlap')
    b.require_hash(args.parent_freeze, args.parent_freeze_sha256)
    b.require_hash(original / 'run_contract.json', args.run_sha256)
    b.require_hash(HERE / 'test_continue_native30_empty_beats_v1.py', RECOVERY_TESTS_SHA)
    prior, cohort, _ = v2.validate_freeze(args.parent_freeze, args.parent_freeze_sha256)
    require(prior['output_root'] == str(original), 'original parent output root')
    run = b.read_json(original / 'run_contract.json')
    require(run == v2.expected_run_record(Path(args.parent_freeze), args.parent_freeze_sha256, prior, cohort,
                run['runtime_start'], run['source_graph_start']), 'original run authority')
    lock = b.safe_path(original / 'writer.lock'); require(lock.is_file(), 'existing writer lock required')
    started = datetime.now(timezone.utc).isoformat()
    lock_before = (stat_key(lock), b.binding(lock))
    with lock.open('r+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (original / 'completion.json').exists() and not (original / 'final_audit.json').exists(), 'original must remain partial')
        inventory = r.files(original); snapshot, stats = {}, {}
        print(b.canonical({'event': 'exclusive_original_writer_lock_acquired', 'files': len(inventory),
                           'bytes': sum(p.stat().st_size for p in inventory.values()), 'pid': os.getpid(), 'host': socket.gethostname()}).decode().strip(), flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, (path, entry, observed) in enumerate(pool.map(stable_binding, sorted(inventory.values())), 1):
                snapshot[path] = entry; stats[path] = observed
                if index % 1000 == 0 or index == len(inventory):
                    print(b.canonical({'event': 'snapshot_hash_progress', 'files': index, 'total': len(inventory)}).decode().strip(), flush=True)
        by_id = {row['id']: row for row in cohort['rows']}; shards = []; anomalies = []; allowed = {str(original / 'run_contract.json')}
        for stage in ('allinone', 'beats'):
            for shard in prior['shards']:
                rows = [by_id[i] for i in shard['item_ids']]; targets = [p for row in rows for p in b.product_paths(stage, row['id'], original)]
                receipt, log = v2.receipt_path(original, stage, shard['index']), v2.log_path(original, stage, shard['index'])
                allowed.update(map(str, [*targets, receipt, log])); present = [p for p in targets if str(p) in snapshot]
                record = {'stage': stage, 'index': shard['index'], 'item_ids': shard['item_ids'], 'expected_products': len(targets),
                          'present_products': len(present), 'receipt_present': str(receipt) in snapshot, 'log_present': str(log) in snapshot}
                try:
                    if str(receipt) in snapshot:
                        receipt_metadata(receipt, stage, shard, rows, prior, args.run_sha256, snapshot)
                        record['status'] = 'ordinary_receipt_and_all_product_byte_bindings_validated'
                    elif not present and str(log) not in snapshot:
                        record['status'] = 'unstarted_no_products_or_log'
                    else:
                        require(stage == 'beats', 'unreceipted partial All-In-One shard')
                        missing, preserved = r.failed_partition(shard, rows, original, log)
                        record.update(status='candidate_known_serializer_replay_required_not_yet_eligible',
                                      missing_ids=[row['id'] for row in rows if row['input']['path'] in set(missing)],
                                      missing_input_paths=missing, preserved_outputs=preserved, original_log=snapshot[str(log)])
                except Exception as exc:
                    record['status'] = 'anomaly'; record['error'] = type(exc).__name__ + ': ' + str(exc)
                    anomalies.append(record)
                shards.append(record)
        unexpected = sorted(set(snapshot) - allowed)
        require(set(r.files(original)) == set(snapshot) and all(stat_key(Path(p)) == stats[p] for p in snapshot), 'snapshot changed during inventory')
        require(snapshot[str(original / 'run_contract.json')]['sha256'] == args.run_sha256, 'original run contract changed before locked snapshot')
        b.require_hash(args.parent_freeze, args.parent_freeze_sha256)
        if unexpected: anomalies.append({'unexpected_files': unexpected})
        draft = {'version': r.FREEZE_VERSION, 'status': 'draft_requires_parent_review_not_execution_authorized', 'policy': r.POLICY,
                 'original_parent_freeze': b.binding(Path(args.parent_freeze)), 'original_run_contract': snapshot[str(original / 'run_contract.json')],
                 'v2_runner': b.binding(Path(v2.__file__).resolve()), 'driver': b.binding(Path(r.__file__).resolve()),
                 'tests': b.binding(HERE / 'test_continue_native30_empty_beats_v1.py'), 'output_root': str(continuation),
                 'initial_partial_inventory': snapshot, **r.SCOPE}
        audit = {'version': 'native30-recovery-eligibility-inventory-v1',
                 'status': 'blocked_anomalies_no_draft' if anomalies else 'metadata_inventory_passed_replay_predicate_unverified',
                 'host': socket.gethostname(), 'pid': os.getpid(), 'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                 'helper': b.binding(Path(__file__).resolve()), 'invocation': [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                 'original_parent_freeze': draft['original_parent_freeze'], 'original_run_contract': draft['original_run_contract'],
                 'exclusive_original_writer_lock': 'held_without_content_changes_through_hashing_inventory_and_publication_then_released',
                 'snapshot_sha256': b.value_hash(snapshot), 'snapshot_files': len(snapshot), 'snapshot_bytes': sum(e['bytes'] for e in snapshot.values()),
                 'shard_status_counts': dict(Counter(s['stage'] + ':' + s['status'] for s in shards)), 'shards': shards, 'anomalies': anomalies,
                 'source_audio_decoded': False, 'output_audio_decoded': False, 'neural_inference_performed': False,
                 'semantic_product_replay_performed': False, 'recovery_predicate_replayed': False,
                 'original_files_modified': False, 'draft_authorizes_execution': False}
        if not anomalies:
            b.write_new(draft_path, draft); audit['draft'] = b.binding(draft_path)
        require((stat_key(lock), b.binding(lock)) == lock_before, 'writer lock file content/metadata changed')
        audit['writer_lock_before_after_identical'] = True
        b.write_new(audit_path, audit)
        print(b.canonical({'status': audit['status'], 'audit': b.binding(audit_path), 'draft': audit.get('draft'),
                           'snapshot_files': len(snapshot), 'snapshot_bytes': audit['snapshot_bytes'], 'shard_status_counts': audit['shard_status_counts'],
                           'anomalies': anomalies}).decode().strip(), flush=True)
    return int(bool(anomalies))


if __name__ == '__main__':
    raise SystemExit(main())

