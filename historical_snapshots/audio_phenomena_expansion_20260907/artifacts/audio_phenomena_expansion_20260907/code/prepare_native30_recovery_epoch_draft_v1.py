"""Byte-snapshot both stopped roots and prepare a nonauthorizing v2 epoch draft.

No waveform decoding, numerical product replay, model calls, or recovery writes.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
PINS = {'continue_native30_empty_beats_v2.py': '91f5fec0fb25b163f69d6b46e032a5f9b27fb7206f93ac7aa4afb4933ae0e6bf',
        'test_continue_native30_empty_beats_v2.py': 'd73cde748119301e2a878301e5ac0309ec0e291fc9f6577e359d8f466b6e7a6f',
        'prepare_native30_recovery_draft_v2.py': 'd547951f458aca91391f21929f7f6cbaeeb3f3b1972f5bfe0f045ea8f03941f9'}
for name, expected in PINS.items():
    path = HERE / name
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected: raise ValueError('helper dependency pin: ' + name)
r = importlib.import_module('continue_native30_empty_beats_v2')
h = importlib.import_module('prepare_native30_recovery_draft_v2')
b, require = r.b, r.require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-freeze', required=True); parser.add_argument('--previous-freeze-sha256', required=True)
    parser.add_argument('--audit', required=True); parser.add_argument('--draft', required=True)
    args = parser.parse_args()
    audit_path, draft_path = b.safe_path(args.audit), b.safe_path(args.draft)
    require(not audit_path.exists() and not draft_path.exists(), 'new metadata artifacts only')
    old_binding = b.binding(b.safe_path(args.previous_freeze)); require(old_binding['sha256'] == args.previous_freeze_sha256, 'previous authority pin')
    old, prior, cohort, _ = r.load_v1().validate_freeze(args.previous_freeze, args.previous_freeze_sha256, audio=False)
    old = {**old, '_path': args.previous_freeze, '_sha256': args.previous_freeze_sha256}
    original, continuation = Path(prior['output_root']), Path(old['output_root'])
    require(not any(p.is_relative_to(root) for p in (audit_path, draft_path) for root in (original, continuation)), 'metadata output overlap')
    runtime = b.read_json(old['original_run_contract']['path'])['runtime_start']
    require(r.get(continuation / 'driver_run.json') == r.load_v1().driver_record(old, runtime), 'original v1 driver record mismatch')
    started = datetime.now(timezone.utc).isoformat()
    with ExitStack() as stack:
        locks = {}
        for root in sorted((original, continuation)):
            lock = b.safe_path(root / 'writer.lock'); locks[str(lock)] = (h.stat_key(lock), b.binding(lock))
            stream = stack.enter_context(lock.open('r+b')); fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(b.canonical({'event': 'both_stopped_root_exclusive_locks_acquired', 'host': socket.gethostname()}).decode().strip(), flush=True)
        require(not any((root / name).exists() for root in (original, continuation) for name in ('completion.json', 'final_audit.json')) and
                not (continuation / 'driver_run_v2.json').exists(), 'partial v1 epoch before any v2 writer required')
        inventory = {**r.files(original), **r.files(continuation)}; snapshot, stats = {}, {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, (path, entry, observed) in enumerate(pool.map(h.stable_binding, sorted(inventory.values())), 1):
                snapshot[path] = entry; stats[path] = observed
                if index % 1000 == 0 or index == len(inventory):
                    print(b.canonical({'event': 'two_root_snapshot_hash_progress', 'files': index, 'total': len(inventory)}).decode().strip(), flush=True)
        originals = {p: e for p, e in snapshot.items() if Path(p).is_relative_to(original)}
        historical = {p: e for p, e in snapshot.items() if Path(p).is_relative_to(continuation)}
        by_id = {row['id']: row for row in cohort['rows']}; stages = []; allowed_original = {str(original / 'run_contract.json')}
        for stage in ('allinone', 'beats'):
            for shard in prior['shards']:
                rows = [by_id[i] for i in shard['item_ids']]
                targets = [p for row in rows for p in b.product_paths(stage, row['id'], original)]
                receipt, log = r.v2.receipt_path(original, stage, shard['index']), r.v2.log_path(original, stage, shard['index'])
                allowed_original.update(map(str, [*targets, receipt, log])); present = [p for p in targets if str(p) in originals]
                record = {'stage': stage, 'index': shard['index'], 'rows': len(rows), 'present_products': len(present)}
                if str(receipt) in originals:
                    h.receipt_metadata(receipt, stage, shard, rows, prior, old['original_run_contract']['sha256'], originals)
                    record['status'] = 'ordinary_receipt_product_byte_bindings_validated'
                elif not present and str(log) not in originals: record['status'] = 'unstarted'
                else:
                    require(stage == 'beats', 'unreceipted partial AIO is not resumable by this correction')
                    missing, preserved = r.failed_partition(shard, rows, original, log)
                    record.update(status='failed_beats_exact_logged_missing', missing_ids=[row['id'] for row in rows if row['input']['path'] in missing],
                                  preserved_count=len(preserved), original_log=originals[str(log)])
                stages.append(record)
        require(set(originals) <= allowed_original, 'unexpected original file')
        draft = {k: v for k, v in old.items() if not k.startswith('_')}
        draft.update(version=r.FREEZE_VERSION, status='draft_requires_parent_review_not_execution_authorized',
                     driver=b.binding(HERE / 'continue_native30_empty_beats_v2.py'), tests=b.binding(HERE / 'test_continue_native30_empty_beats_v2.py'),
                     previous_continuation_freeze=old_binding, environment_transition=r.ENVIRONMENT_TRANSITION,
                     torch_cuda_source=b.binding(Path(prior['runtime_root']) / 'venv/lib/python3.11/site-packages/torch/cuda/__init__.py'),
                     resume_original_inventory=originals, resume_continuation_inventory=historical)
        context = {**draft, '_previous': old, '_path': str(draft_path), '_sha256': '0' * 64}
        allowed_history = {str(continuation / 'driver_run.json')}; execution_records = []
        for intent_path in sorted((continuation / 'executions').glob('*.json')):
            intent = r.get(intent_path); stage = intent['stage']; shard = prior['shards'][intent['shard_index']]
            require(intent_path.name == r.name(stage, shard), 'execution filename/shard mismatch')
            result_path = continuation / 'execution_results' / intent_path.name; result = r.get(result_path)
            receipt = b.read_json(r.v2.receipt_path(original, stage, shard['index']))['payload']
            require(intent['status'] == 'continuation_stage_execution_intent' and intent['item_ids'] == shard['item_ids'] and
                    intent['driver'] == old['driver'] and intent['continuation_freeze_sha256'] == old['_sha256'] and
                    intent['original_run_contract_sha256'] == old['original_run_contract']['sha256'] and
                    intent['gpu'] == receipt['gpu'] and intent['command'] == receipt['command'] and
                    intent['child_environment'] == receipt['child_environment'] and intent['inputs'] == receipt['inputs'], 'v1 execution provenance metadata')
            require(result['status'] == 'continuation_stage_subprocess_exited' and result['returncode'] == 0 and
                    result['intent'] == historical[str(intent_path)] and result['log'] == receipt['log'] and result['inputs_after'] == intent['inputs'],
                    'v1 execution result metadata')
            execution_records.append({'intent': historical[str(intent_path)], 'result': historical[str(result_path)], 'stage': stage, 'rows': len(shard['item_ids'])})
            allowed_history.update((str(intent_path), str(result_path)))
        raw_records = []
        for raw_path in sorted((continuation / 'raw_replays').glob('*.json')):
            raw = r.get(raw_path); request_path = continuation / 'recovery_requests' / raw_path.name; request = r.get(request_path)
            shard = prior['shards'][request['shard_index']]; rows = [by_id[i] for i in shard['item_ids']]
            log = r.v2.log_path(original, 'beats', shard['index']); missing, preserved = r.failed_partition(shard, rows, original, log)
            require(request['original_log'] == originals[str(log)] and request['missing_input_paths'] == missing and
                    request['preserved_outputs'] == preserved and request['missing_ids'] == [row['id'] for row in rows if row['input']['path'] in missing],
                    'historical request missing/log/preserved metadata')
            require(request['policy'] == r.POLICY and request['original_run_contract_sha256'] == old['original_run_contract']['sha256'] and
                    request['item_ids'] == shard['item_ids'] and request['original_shard_command'] == b.command('beats', [x['input']['path'] for x in rows], Path(prior['runtime_root']), original),
                    'historical request authority')
            for row in rows:
                declared = request['inputs'][row['id']]
                require({k: declared[k] for k in ('path', 'bytes', 'sha256')} == row['input'] and
                        declared['waveform_float32_sha256'] == row['waveform_float32_sha256'], 'historical input declaration/cohort join')
            r.validate_raw(raw, request, historical[str(request_path)], context, prior)
            require(raw['recovery_invocation'] == r.load_v1().replay_command(old, prior, request_path), 'actual v1 replay invocation')
            replay_log = continuation / 'replay_logs' / (raw_path.stem + '.log')
            allowed_history.update(map(str, (raw_path, request_path, replay_log)))
            expected = r.v2.child_environment(prior['runtime_root'], raw['gpu'])
            raw_records.append({'raw': historical[str(raw_path)], 'request': historical[str(request_path)], 'replay_log': historical[str(replay_log)],
                                'environment_added': {k: v for k, v in raw['child_environment'].items() if k not in expected},
                                'environment_removed': sorted(set(expected) - set(raw['child_environment'])),
                                'environment_changed': {k: [v, raw['child_environment'].get(k)] for k, v in expected.items() if raw['child_environment'].get(k) != v},
                                'recorded_predicates': [{'id': row['id'], 'beats': row['beats'], 'downbeats': row['downbeats'],
                                                         'exception_type': row['exception_type'], 'exception_message': row['exception_message']} for row in raw['records']],
                                'raw_authority_epoch': 'v1_snapshot_bound', 'pre_cuda_environment': 'inferred_from_pinned_v1_pre_cuda_assertion_not_recorded'})
        require(set(historical) == allowed_history, 'unexpected historical continuation file')
        require(set(inventory) == set(r.files(original)) | set(r.files(continuation)) and
                all(h.stat_key(Path(p)) == stats[p] for p in snapshot), 'stopped snapshot changed during inspection')
        require(all((h.stat_key(Path(p)), b.binding(Path(p))) == before for p, before in locks.items()), 'lock content/metadata changed')
        b._binding_matches(old_binding, audio=True)
        b.write_new(draft_path, draft)
        audit = {'version': 'native30-recovery-v2-epoch-draft-inventory-v1', 'status': 'historical_recorded_predicates_and_exact_environment_transition_verified_no_replay',
                 'helper': b.binding(Path(__file__).resolve()), 'host': socket.gethostname(), 'started_utc': started,
                 'completed_utc': datetime.now(timezone.utc).isoformat(), 'previous_continuation_freeze': old_binding,
                 'draft': b.binding(draft_path), 'both_exclusive_writer_locks_held_and_unchanged': True,
                 'snapshot_counts': {name: {'files': len(value), 'bytes': sum(x['bytes'] for x in value.values()), 'sha256': b.value_hash(value)}
                                     for name, value in (('original', originals), ('continuation', historical))},
                 'stage_counts': dict(Counter(row['stage'] + ':' + row['status'] for row in stages)), 'stages': stages,
                 'v1_execution_metadata': execution_records, 'raw_evidence': raw_records, 'torch_cuda_source': draft['torch_cuda_source'],
                 'source_excerpt': Path(draft['torch_cuda_source']['path']).read_text().splitlines()[409:412],
                 'original_bytes_modified': False, 'waveforms_decoded': False, 'semantic_product_replay_performed': False,
                 'neural_inference_performed': False, 'recovery_performed': False, 'draft_authorizes_execution': False}
        b.write_new(audit_path, audit)
        print(b.canonical({'status': audit['status'], 'audit': b.binding(audit_path), 'draft': audit['draft'],
                          'snapshot_counts': audit['snapshot_counts'], 'stage_counts': audit['stage_counts']}).decode().strip(), flush=True)


if __name__ == '__main__': main()
