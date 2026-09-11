#!/usr/bin/env python3
"""Resumable, hash-audited exact60 inference on explicitly idle RTX 5090s only.

Completed receipts are verified on resume. Unreceipted partial outputs fail
closed and are never silently deleted or accepted. No classifier fitting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading

import numpy as np

from materialize_equal60_inputs_v2 import sha_file, preserve_json
from verify_extract_equal60 import (inspect_audio, inspect_beats,
                                    inspect_structure, verify_runtime, STEMS)


def now():
    return datetime.now(timezone.utc).isoformat()


def assignments(shards, gpus):
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('GPUs must be nonempty and unique')
    return [(gpu, shards[i::len(gpus)]) for i, gpu in enumerate(gpus)]


def command(stage, inputs, runtime, output):
    if stage == 'allinone':
        return [str(runtime/'venv/bin/all-in-one-infer'), *inputs,
                '-o', str(output/'structure'), '-m', 'harmonix-all', '-d', 'cuda', '-k',
                '--demix-dir', str(output/'demix'), '--spec-dir', str(output/'spec')]
    if stage == 'beats':
        return [str(runtime/'venv/bin/beat_this'), *inputs, '-o', str(output/'beats'),
                '--model', str(runtime/'checkpoints/hub/checkpoints/beat_this-final0.ckpt'),
                '--no-dbn', '--gpu', '0', '--float16', '--skip-existing']
    raise ValueError('Unknown stage')


def product_paths(stage, item_id, output):
    if stage == 'beats':
        return [output/'beats'/f'{item_id}.beats']
    return [*(output/'demix/htdemucs'/item_id/f'{s}.wav' for s in STEMS),
            output/'structure'/f'{item_id}.json', output/'spec'/f'{item_id}.npy']


def verify_products(stage, rows, output):
    files = {}
    for row in rows:
        item_id = row['item_id']
        if stage == 'beats':
            inspect_beats(output/'beats'/f'{item_id}.beats')
        else:
            for stem in STEMS:
                inspect_audio(output/'demix/htdemucs'/item_id/f'{stem}.wav')
            inspect_structure(output/'structure'/f'{item_id}.json', row['standardized_path'])
            arr = np.load(output/'spec'/f'{item_id}.npy', mmap_mode='r', allow_pickle=False)
            if arr.shape != (4, 6000, 81) or arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise ValueError('Invalid exact60 spectrogram: ' + item_id)
        for path in product_paths(stage, item_id, output):
            files[str(path)] = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
    return files


def validate_resume(receipt, contract_sha, stage, index, ids):
    if (receipt.get('status') != 'passed' or receipt.get('run_contract_sha256') != contract_sha
            or receipt.get('stage') != stage or receipt.get('shard_index') != index
            or receipt.get('item_ids') != ids):
        raise ValueError('Receipt identity/contract mismatch')
    for path, expected in receipt['outputs'].items():
        if sha_file(path) != expected['sha256'] or Path(path).stat().st_size != expected['bytes']:
            raise ValueError('Previously completed output changed: ' + path)


def require_idle_5090(gpu):
    data = subprocess.check_output(['nvidia-smi', '--id='+str(gpu),
        '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    name, memory, util = [x.strip() for x in data.strip().split(',')]
    if 'RTX 5090' not in name or int(memory) > 1024 or int(util) > 5:
        raise RuntimeError(f'Refusing unavailable/busy RTX 5090 {gpu}: {data.strip()}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-dir', 'output-root', 'materialization-audit', 'runtime-root',
                 'runtime-contract', 'checkpoint-manifest', 'old-code-root', 'bias'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--aio-gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--beat-gpus', nargs='+', type=int, default=[4, 5])
    a = p.parse_args()
    if set(a.aio_gpus) & set(a.beat_gpus):
        raise ValueError('Inference stages must have disjoint GPUs')
    a.output_root.mkdir(parents=True, exist_ok=True)
    with (a.output_root/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(a)


def run_locked(a):
    audit = json.loads(a.materialization_audit.read_text())
    manifest = a.prepared_dir/'inference_manifest.csv'
    summary = json.loads((a.prepared_dir/'materialization_summary.json').read_text())
    rows = list(csv.DictReader(manifest.open()))
    if (audit['status'] != 'passed' or not audit['all_output_files_rehashed'] or len(rows) != 1604
            or audit['prepared_manifest_sha256'] != sha_file(manifest)
            or audit['materialization_contract_sha256'] != summary['contract_sha256']
            or [r['item_id'] for r in audit['records']] != [r['item_id'] for r in rows]
            or any(r['role'] != 'development' for r in rows)):
        raise ValueError('Full materialization audit required')
    runtime_checks = verify_runtime(a.runtime_contract, a.checkpoint_manifest, a.old_code_root, a.bias)
    by_path = {r['standardized_path']: r for r in rows}
    shards, concatenated = [], []
    for shard in summary['shards']:
        path = a.prepared_dir/f"inference_shard_{shard['index']:02d}.txt"
        if sha_file(path) != shard['sha256']:
            raise ValueError('Shard changed')
        inputs = path.read_text().splitlines()
        if len(inputs) != shard['rows']:
            raise ValueError('Shard count changed')
        shards.append((shard['index'], inputs))
        concatenated.extend(inputs)
    if concatenated != list(by_path) or len(by_path) != len(rows):
        raise ValueError('Shard omission/duplication/order mismatch')
    for name in ('structure', 'demix', 'spec', 'beats', 'receipts', 'logs'):
        (a.output_root/name).mkdir(exist_ok=True)
    contract = {'code_sha256': sha_file(__file__), 'rows': len(rows),
        'manifest_sha256': sha_file(manifest), 'audit_sha256': sha_file(a.materialization_audit),
        'dependencies': {name: sha_file(Path(__file__).parent/name) for name in
                         ('verify_extract_equal60.py', 'materialize_equal60_inputs_v2.py')},
        'runtime_code_checkpoint_bias_checks': runtime_checks,
        'shards': summary['shards'], 'aio_gpus': a.aio_gpus, 'beat_gpus': a.beat_gpus,
        'output_root': str(a.output_root), 'runtime_root': str(a.runtime_root),
        'resume_policy': 'verify completed receipts; fail on unreceipted partial products',
        'neural_reproducibility': 'published output hashes; unseeded established Demucs shift default',
        'commands': {s: command(s, ['<frozen shard inputs>'], a.runtime_root, a.output_root)
                     for s in ('allinone', 'beats')}}
    preserve_json(a.output_root/'run_contract.json', contract)
    contract_sha = sha_file(a.output_root/'run_contract.json')
    stopped = threading.Event()

    def worker(stage, gpu, assigned):
        for index, inputs in assigned:
            if stopped.is_set():
                return
            selected = [by_path[path] for path in inputs]
            ids = [r['item_id'] for r in selected]
            receipt_path = a.output_root/'receipts'/f'{stage}_{index:02d}.json'
            try:
                if receipt_path.exists():
                    validate_resume(json.loads(receipt_path.read_text()), contract_sha, stage, index, ids)
                    print(f'validated resume stage={stage} shard={index}', flush=True)
                    continue
                if any(path.exists() for item_id in ids for path in product_paths(stage, item_id, a.output_root)):
                    raise ValueError(f'Unreceipted partial outputs: {stage} shard {index}')
                for row in selected:
                    if sha_file(row['standardized_path']) != row['standardized_file_sha256']:
                        raise ValueError('Input changed before inference')
                require_idle_5090(gpu)
                cmd = command(stage, inputs, a.runtime_root, a.output_root)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
                started = now()
                log_path = a.output_root/'logs'/f'{stage}_{index:02d}.log'
                print(f'start {started} stage={stage} gpu={gpu} shard={index} rows={len(inputs)}', flush=True)
                with log_path.open('a') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                outputs = verify_products(stage, selected, a.output_root)
                for row in selected:
                    if sha_file(row['standardized_path']) != row['standardized_file_sha256']:
                        raise ValueError('Input changed during inference')
                preserve_json(receipt_path, {'status': 'passed', 'stage': stage, 'shard_index': index,
                    'run_contract_sha256': contract_sha, 'item_ids': ids, 'gpu': gpu, 'command': cmd,
                    'started_utc': started, 'completed_utc': now(), 'outputs': outputs,
                    'log_sha256': sha_file(log_path)})
                print(f'passed {now()} stage={stage} shard={index}', flush=True)
            except Exception:
                stopped.set()
                raise

    work = [(stage, gpu, assigned) for stage, gpus in (('allinone', a.aio_gpus), ('beats', a.beat_gpus))
            for gpu, assigned in assignments(shards, gpus)]
    with ThreadPoolExecutor(max_workers=len(work)) as pool:
        futures = [pool.submit(worker, *job) for job in work]
        for future in futures:
            future.result()
    preserve_json(a.output_root/'completion.json', {'status': 'passed', 'rows': len(rows),
        'run_contract_sha256': contract_sha, 'completed_stage_shards': 2*len(shards)})
    print('All exact60 inference stage-shards completed and verified', flush=True)


if __name__ == '__main__':
    main()
