"""Read-only two-root stopped-byte snapshot; writes only new NONAUTHORIZING metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
from pathlib import Path
import socket
import subprocess

import resume_native30_gpu_guard_v1 as s

DRIVER_SHA = '8cf3f5fc6e67d0a633076c04fcc895d24144385d71006b5319f1bab163b372cb'
TEST_SHA = '028db520fb1f9edad4316ea7c6cfadbc0891b313096908a4f119788a55dc3b55'
EXPECTED_SNAPSHOT_SHA = '17db12d2dece3a98c3e9837748e703df48d2cf52cca58db657b4ce18ae124e02'


def stat(path):
    x = path.stat(); return (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)


def stable_binding(path):
    before = stat(path); value = s.b.binding(path)
    s.require(stat(path) == before, 'file changed while hashing')
    return str(path), value, before


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-freeze', required=True); p.add_argument('--terminal-log', required=True)
    p.add_argument('--operational-root', required=True); p.add_argument('--draft', required=True); p.add_argument('--audit', required=True)
    a = p.parse_args()
    s.b.require_hash(Path(s.__file__).resolve(), DRIVER_SHA)
    tests = Path(s.__file__).resolve().with_name('test_resume_native30_gpu_guard_v1.py'); s.b.require_hash(tests, TEST_SHA)
    base, prior, cohort, helper, runtime = s.r.setup_context(a.base_freeze, s.BASE_FREEZE_SHA, audio=False)
    original, historical = Path(prior['output_root']), Path(base['output_root'])
    output, draft_path, audit_path = map(s.b.safe_path, (a.operational_root, a.draft, a.audit))
    s.require(socket.gethostname() == s.HOST and not output.exists(), 'exact host and unused operational root required')
    s.require(all(not target.is_relative_to(root) for target in (output, draft_path, audit_path) for root in (original, historical)), 'metadata/output root overlaps history')
    s.require(not draft_path.exists() and not audit_path.exists() and draft_path != audit_path, 'new metadata outputs only')
    s.b.require_hash(a.terminal_log, s.TERMINAL_SHA)
    lock_records = {}; started = s.r.utc()
    with ExitStack() as stack:
        for root in (original, historical):
            path = root / 'writer.lock'; before = (stat(path), s.b.binding(path))
            lock = stack.enter_context(path.open('r+b'))  # No truncate or write; NFS exclusive flock needs writable descriptor.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); lock_records[str(path)] = before
        print(s.b.canonical({'event': 'both_stopped_root_exclusive_locks_acquired_without_writes', 'utc': s.r.utc()}).decode().strip(), flush=True)
        paths = {**s.r.files(original), **s.r.files(historical)}; bindings, states = {}, {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(stable_binding, path) for path in paths.values()]
            for future in as_completed(futures):
                path, binding, state = future.result(); bindings[path] = binding; states[path] = state
                if len(bindings) % 2000 == 0:
                    print(s.b.canonical({'event': 'snapshot_hash_progress', 'files': len(bindings), 'total': len(paths)}).decode().strip(), flush=True)
        s.require(s.b.value_hash(bindings) == EXPECTED_SNAPSHOT_SHA, 'stopped inventory differs from independently reviewed byte audit')
        original_snapshot = {p: e for p, e in bindings.items() if Path(p).is_relative_to(original)}
        historical_snapshot = {p: e for p, e in bindings.items() if Path(p).is_relative_to(historical)}
        draft = {'version': s.FREEZE_VERSION, 'status': 'draft_nonauthorizing_pending_parent_publication',
                 'base_freeze': s.b.binding(a.base_freeze), 'base_driver': s.b.binding(Path(s.r.__file__).resolve()),
                 'driver': s.b.binding(Path(s.__file__).resolve()), 'tests': s.b.binding(tests),
                 'terminal_log': s.b.binding(a.terminal_log), 'blocked_intent': s.b.binding(historical / 'executions/beats_096.json'),
                 'placement': s.PLACEMENT, 'policy': s.r.POLICY, 'output_root': str(output),
                 'stopped_original_inventory': original_snapshot, 'stopped_recovery_inventory': historical_snapshot, **s.r.SCOPE}
        context = {**draft, '_base': base}
        s.snapshots(context, prior, audio=False, first=True)
        certificate = s.aborted_attempt(context, prior, cohort, first=True)
        unmatched = sorted(Path(p).name for p in historical_snapshot if Path(p).parent == historical / 'executions'
                           and str(historical / 'execution_results' / Path(p).name) not in historical_snapshot)
        s.require(unmatched == ['beats_096.json'], 'unexpected unmatched execution intent')
        s.require(set(paths) == set(s.r.files(original)) | set(s.r.files(historical)) and
                  all(stat(Path(path)) == value for path, value in states.items()), 'snapshot changed before publication')
        s.require(all((stat(Path(path)), s.b.binding(path)) == value for path, value in lock_records.items()), 'existing writer.lock changed')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
        audit = {'status': 'readonly_stopped_snapshot_for_nonauthorizing_operational_draft', 'started_utc': started, 'completed_utc': s.r.utc(),
                 'builder': s.b.binding(Path(__file__).resolve()), 'host': socket.gethostname(), 'gpu_observation_only': gpu,
                 'original_files': len(original_snapshot), 'historical_recovery_files': len(historical_snapshot),
                 'files': len(bindings), 'bytes': sum(e['bytes'] for e in bindings.values()), 'snapshot_sha256': s.b.value_hash(bindings),
                 'original_snapshot_sha256': s.b.value_hash(original_snapshot), 'historical_snapshot_sha256': s.b.value_hash(historical_snapshot),
                 'locks': {'mode': 'exclusive_nonblocking_no_writes', 'unchanged_at_end': True, 'paths': list(lock_records)},
                 'unmatched_intents': unmatched, 'aborted_attempt': certificate,
                 'waveform_decoding_performed': False, 'inference_launched': False, 'freeze_authorized': False}
        s.b.write_new(audit_path, audit)
        s.b.write_new(draft_path, {**draft, 'snapshot_audit': s.b.binding(audit_path), 'snapshot_builder': audit['builder']})
    print(s.b.canonical({'status': audit['status'], 'draft': s.b.binding(draft_path), 'audit': s.b.binding(audit_path),
                         'files': len(bindings), 'bytes': audit['bytes'], 'snapshot_sha256': audit['snapshot_sha256']}).decode().strip(), flush=True)


if __name__ == '__main__': main()
