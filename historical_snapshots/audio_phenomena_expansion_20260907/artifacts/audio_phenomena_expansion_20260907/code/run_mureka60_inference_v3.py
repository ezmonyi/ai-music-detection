#!/usr/bin/env python3
"""Mureka500 v3 inference with completion-gated release of original GPUs0--5.

The default is the v2 reservation (only GPUs >=6).  An explicit, fully verified
completion proof for the fixed original1604 inference may release GPUs0--5.
No model, command, representation, shard, or measurement role is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import threading

import run_mureka60_inference_v2 as v2


COUNT, SHARDS, STAGES, ROLE = v2.COUNT, v2.SHARDS, v2.STAGES, v2.ROLE
ORIGINAL_COUNT, ORIGINAL_SHARDS = 1604, 67
ORIGINAL_PREPARED_ROOT = Path(
    '/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/equal60_development_v2')
ORIGINAL_STAGE_GPUS = {'allinone': [0, 1, 2, 3], 'beats': [4, 5]}
ORIGINAL_MANIFEST_SHA256 = 'f52b93bf4ff820bd35845749b91a2c7139b901588ebde2ebf62f4442f583d3bf'
ORIGINAL_MATERIALIZATION_CONTRACT_SHA256 = '6b51d502a68f3030592f2ddabda0596aa18f27471b79bfaf328fa08d6e6ed29c'
ORIGINAL_RUNNER_SHA256 = 'a33f667ac9ef5cc5b4e5e646adfb752a68380b1af5f42b0768b1a10e3d031858'

# Re-export the reviewed v2 primitives used by receipts and the strict auditor.
check_seal = v2.check_seal
file_record = v2.file_record
input_records = v2.input_records
json_bytes = v2.json_bytes
measurement_only = v2.measurement_only
now = v2.now
original = v2.original
publish = v2.publish
read_json = v2.read_json
regular = v2.regular
require = v2.require
scope = v2.scope
sealed = v2.sealed
sha_file = v2.sha_file
validate_prepared = v2.validate_prepared


def dependency_snapshot():
    """Bind the frozen helpers, v2 adapters, and both new v3 adapters."""
    checks = v2.dependency_snapshot()
    root = Path(__file__).resolve().parent
    for name in ('run_mureka60_inference_v3.py', 'verify_extract_mureka60_v3.py'):
        checks[str(root / name)] = sha_file(regular(root / name))
    return checks


def stage_gpu_assignment(aio_gpus, beat_gpus, shard_count=SHARDS):
    return {stage: {f'{index:02d}': gpus[index % len(gpus)] for index in range(shard_count)}
            for stage, gpus in (('allinone', aio_gpus), ('beats', beat_gpus))}


def validate_gpus(aio, beats, release_proof=None):
    """Keep v2's default reservation; release only with a current full proof."""
    for gpus in (aio, beats):
        require(bool(gpus) and len(gpus) == len(set(gpus))
                and all(type(g) is int and g >= 0 for g in gpus),
                'Explicit nonempty unique GPU indices required')
    require(not set(aio).intersection(beats), 'Inference GPU stages must be disjoint')
    released = set(aio + beats).intersection(range(6))
    if release_proof is not None:
        check_seal(release_proof)
        require(release_proof.get('status') == 'verified_full134_original1604'
                and release_proof.get('rows') == ORIGINAL_COUNT
                and release_proof.get('stage_shards') == 2 * ORIGINAL_SHARDS,
                'Invalid original1604 GPU release proof')
    require(not released or release_proof is not None,
            'GPUs0-5 remain reserved without verified original1604 completion')


def _original_shards(prepared, contract, summary, rows, proof_files):
    require(contract.get('shards') == summary.get('shards'),
            'Original prepared shard ledger differs from run contract')
    require(len(summary.get('shards', [])) == ORIGINAL_SHARDS,
            'Original inference must have exactly67 shards')
    all_paths, shards = [], []
    for index, shard in enumerate(summary['shards']):
        path = regular(prepared / f'inference_shard_{index:02d}.txt')
        require(set(shard) == {'index', 'rows', 'sha256'}
                and shard.get('index') == index
                and sha_file(path) == shard.get('sha256'), 'Original shard identity/hash mismatch')
        inputs = path.read_text().splitlines()
        require(len(inputs) == shard.get('rows'), 'Original shard row count mismatch')
        proof_files[str(path)] = file_record(path)
        shards.append((index, inputs))
        all_paths.extend(inputs)
    require(all_paths == [r['standardized_path'] for r in rows]
            and len(set(all_paths)) == ORIGINAL_COUNT,
            'Original shard cohort omission/duplication/order mismatch')
    return shards


def validate_production_pins(contract, summary, manifest_sha):
    require(manifest_sha == ORIGINAL_MANIFEST_SHA256
            and summary.get('contract_sha256') == ORIGINAL_MATERIALIZATION_CONTRACT_SHA256
            and contract.get('code_sha256') == ORIGINAL_RUNNER_SHA256,
            'Original1604 manifest/materialization/runner pin mismatch')


def validate_original_completion(completion_path, fixed_root=ORIGINAL_PREPARED_ROOT,
                                 synthetic_test=False):
    """Return a sealed, byte-level proof for the fixed completed original run.

    A nonproduction `fixed_root` is accepted only under explicit synthetic-test
    mode. CLI callers always use the production root and immutable cohort pins.
    """
    fixed_root = Path(fixed_root)
    require(fixed_root.is_absolute() and fixed_root.resolve() == fixed_root,
            'Original prepared root must be canonical absolute path')
    require((synthetic_test and fixed_root != ORIGINAL_PREPARED_ROOT)
            or (not synthetic_test and fixed_root == ORIGINAL_PREPARED_ROOT),
            'Nonproduction original root is allowed only for isolated synthetic tests')
    inference = fixed_root / 'inference'
    completion_path = regular(completion_path)
    require(completion_path == inference / 'completion.json',
            'Completion proof is not the fixed original1604 inference')
    paths = {
        'completion': completion_path,
        'run_contract': regular(inference / 'run_contract.json'),
        'prepared_manifest': regular(fixed_root / 'inference_manifest.csv'),
        'materialization_summary': regular(fixed_root / 'materialization_summary.json'),
    }
    proof_files = {str(path): file_record(path) for path in paths.values()}
    completion = read_json(paths['completion'])
    contract = read_json(paths['run_contract'])
    summary = read_json(paths['materialization_summary'])
    with paths['prepared_manifest'].open() as stream:
        rows = list(csv.DictReader(stream))
    require(completion.get('status') == 'passed' and completion.get('rows') == ORIGINAL_COUNT
            and completion.get('completed_stage_shards') == 2 * ORIGINAL_SHARDS
            and completion.get('run_contract_sha256') == sha_file(paths['run_contract']),
            'Original completion is not passed full134')
    require(contract.get('rows') == ORIGINAL_COUNT
            and contract.get('output_root') == str(inference)
            and contract.get('manifest_sha256') == sha_file(paths['prepared_manifest'])
            and contract.get('aio_gpus') == ORIGINAL_STAGE_GPUS['allinone']
            and contract.get('beat_gpus') == ORIGINAL_STAGE_GPUS['beats'],
            'Original run contract/root/GPU identity mismatch')
    if not synthetic_test:
        validate_production_pins(contract, summary, sha_file(paths['prepared_manifest']))
    require(len(rows) == ORIGINAL_COUNT
            and len({r.get('item_id') for r in rows}) == ORIGINAL_COUNT
            and all(r.get('role') == 'development' for r in rows),
            'Original prepared manifest is not the exact development1604 cohort')
    shards = _original_shards(fixed_root, contract, summary, rows, proof_files)
    by_path = {r['standardized_path']: r for r in rows}
    expected_names = {f'{stage}_{index:02d}.json' for stage in STAGES
                      for index in range(ORIGINAL_SHARDS)}
    receipts_dir = inference / 'receipts'
    require(receipts_dir.is_dir() and not receipts_dir.is_symlink()
            and {p.name for p in receipts_dir.iterdir()} == expected_names,
            'Original receipt inventory must be exactly134 stage receipts')
    original_assignment = stage_gpu_assignment(
        ORIGINAL_STAGE_GPUS['allinone'], ORIGINAL_STAGE_GPUS['beats'], ORIGINAL_SHARDS)
    receipt_identities = {}
    for stage in STAGES:
        for index, inputs in shards:
            name = f'{stage}_{index:02d}.json'
            path = regular(receipts_dir / name)
            receipt = read_json(path)
            ids = [by_path[p]['item_id'] for p in inputs]
            require(receipt.get('status') == 'passed' and receipt.get('stage') == stage
                    and receipt.get('shard_index') == index
                    and receipt.get('run_contract_sha256') == sha_file(paths['run_contract'])
                    and receipt.get('item_ids') == ids
                    and receipt.get('gpu') == original_assignment[stage][f'{index:02d}'],
                    'Original receipt status/stage/cohort/GPU identity mismatch: ' + name)
            expected_command = original.command(stage, inputs, Path(contract['runtime_root']), inference)
            command = receipt.get('command', receipt.get('original_shard_command'))
            require(command == expected_command, 'Original receipt command identity mismatch: ' + name)
            require(isinstance(receipt.get('outputs'), dict) and receipt['outputs'],
                    'Original receipt lacks output ledger: ' + name)
            proof_files[str(path)] = file_record(path)
            receipt_identities[name] = {'stage': stage, 'shard_index': index,
                'item_ids': ids, 'gpu': receipt['gpu'], 'command': command}
    return sealed({'schema_version': 1, 'status': 'verified_full134_original1604',
        'fixed_prepared_root': str(fixed_root), 'fixed_inference_root': str(inference),
        'rows': ORIGINAL_COUNT, 'stage_shards': 2 * ORIGINAL_SHARDS,
        'run_contract_sha256': sha_file(paths['run_contract']),
        'prepared_manifest_sha256': sha_file(paths['prepared_manifest']),
        'materialization_contract_sha256': summary.get('contract_sha256'),
        'original_runner_sha256': contract.get('code_sha256'),
        'completion_sha256': sha_file(paths['completion']),
        'original_stage_gpu_assignment': original_assignment,
        'receipt_identities': receipt_identities, 'proof_files': proof_files})


@contextmanager
def original_writer_lock(release_proof):
    """Exclude restart of the old GPU writer for the entire released-GPU run."""
    if release_proof is None:
        yield
        return
    lock_path = regular(Path(release_proof['fixed_inference_root']) / 'writer.lock')
    with lock_path.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def build_contract(a, rows, shards, provenance, runtime, dependencies, release_proof):
    return sealed(dict(scope(), schema_version=3, rows=COUNT,
        prepared_dir=str(a.prepared_dir), output_root=str(a.output_root),
        runtime_root=str(a.runtime_root), item_ids=[r['item_id'] for r in rows],
        preparation_sha256=provenance, dependencies_sha256=dependencies,
        runtime_sha256=runtime, allinone_gpus=a.aio_gpus, beats_gpus=a.beat_gpus,
        stage_gpu_assignment=stage_gpu_assignment(a.aio_gpus, a.beat_gpus),
        original_gpu_release_proof=release_proof,
        shards=[{'index': index, 'inputs': paths} for index, paths in shards],
        commands={stage: original.command(stage, ['<frozen shard inputs>'],
                 a.runtime_root, a.output_root) for stage in STAGES},
        resume_policy='verify full receipts and inventories; reject all unreceipted products/logs',
        recovery_authorized=False,
        neural_reproducibility='published output hashes; unchanged unseeded Demucs shifts=1'))


def validate_receipt(receipt, contract, contract_sha, stage, index, rows, output, decode=False):
    v2.validate_receipt(receipt, contract, contract_sha, stage, index, rows, output, decode)
    require(receipt.get('gpu') == contract['stage_gpu_assignment'][stage][f'{index:02d}'],
            'Receipt differs from explicit stage GPU assignment')


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
            validate_receipt(read_json(regular(path)), contract, contract_sha, stage, index,
                             part, output, decode)
            completed.append((stage, part))
            receipts[str(path)] = sha_file(path)
    require({str(p) for p in (output / 'receipts').iterdir()} == set(receipts),
            'Unexpected receipt inventory')
    v2.validate_inventory(output, rows, completed)
    expected_logs = {str(output / 'logs' /
                     f'{read_json(path)["stage"]}_{read_json(path)["shard_index"]:02d}.log')
                     for path in receipts}
    require({str(p) for p in (output / 'logs').iterdir()} == expected_logs,
            'Unreceipted/unexpected logs; manual review required')
    return receipts


def verify_completion(output, contract, rows, shards, decode=False):
    completion = read_json(regular(output / 'completion.json'))
    check_seal(completion)
    measurement_only(completion)
    require(completion.get('status') == 'passed' and completion.get('rows') == COUNT
            and completion.get('completed_stage_shards') == 42
            and completion.get('run_contract_sha256') == sha_file(output / 'run_contract.json'),
            'Incomplete500 inference')
    receipts = verify_all_receipts(output, contract, rows, shards, True, decode)
    require(completion.get('receipts_sha256') == receipts
            and completion.get('dependencies_sha256') == contract['dependencies_sha256'],
            'Completion binding mismatch')
    return completion


def run_locked(a, release_proof=None):
    validate_gpus(a.aio_gpus, a.beat_gpus, release_proof)
    if release_proof is not None:
        require(validate_original_completion(
            Path(release_proof['fixed_inference_root']) / 'completion.json',
            Path(release_proof['fixed_prepared_root'])) == release_proof,
            'Original GPU release proof is stale before inference')
    require(a.output_root == a.prepared_dir / 'inference_v3'
            and a.output_root.resolve() == a.output_root,
            'Only dedicated Mureka prepared/inference_v3 output permitted')
    dependencies = dependency_snapshot()
    rows, shards, provenance = validate_prepared(a.prepared_dir)
    runtime = v2.runtime_snapshot(a)
    contract = build_contract(a, rows, shards, provenance, runtime, dependencies, release_proof)
    publish(a.output_root / 'run_contract.json', json_bytes(contract))
    contract_sha = sha_file(a.output_root / 'run_contract.json')
    for name in ('structure', 'demix', 'spec', 'beats', 'receipts', 'logs'):
        (a.output_root / name).mkdir(exist_ok=True)
    verify_all_receipts(a.output_root, contract, rows, shards, decode=True)
    if (a.output_root / 'completion.json').exists():
        verify_completion(a.output_root, contract, rows, shards)
        return
    by_path = {r['standardized_path']: r for r in rows}
    stop = threading.Event()

    def check_bindings():
        require(dependency_snapshot() == dependencies, 'Code dependencies changed during inference')
        if release_proof is not None:
            require(validate_original_completion(
                Path(release_proof['fixed_inference_root']) / 'completion.json',
                Path(release_proof['fixed_prepared_root'])) == release_proof,
                'Original GPU release proof changed during inference')
        for path, digest in {**provenance, **runtime}.items():
            require(sha_file(path) == digest, 'Bound provenance/runtime changed: ' + path)

    def worker(stage, gpu, assigned):
        for index, inputs in assigned:
            if stop.is_set():
                return
            selected = [by_path[path] for path in inputs]
            receipt_path = a.output_root / 'receipts' / f'{stage}_{index:02d}.json'
            try:
                if receipt_path.exists():
                    validate_receipt(read_json(receipt_path), contract, contract_sha, stage,
                                     index, selected, a.output_root)
                    continue
                check_bindings()
                before = input_records(selected)
                require(not any(Path(path).exists() for path in
                                v2.expected_products(stage, selected, a.output_root)),
                        'Unreceipted product; manual review required')
                original.require_idle_5090(gpu)
                cmd = original.command(stage, inputs, a.runtime_root, a.output_root)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1')
                log_path = a.output_root / 'logs' / f'{stage}_{index:02d}.log'
                started = now()
                print(f'start stage={stage} shard={index} gpu={gpu} rows={len(inputs)}', flush=True)
                with log_path.open('x') as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                outputs = original.verify_products(stage, selected, a.output_root)
                require(input_records(selected) == before, 'Inputs changed during inference')
                check_bindings()
                receipt = sealed(dict(scope(), status='passed', exit_code=0, stage=stage,
                    shard_index=index, run_contract_sha256=contract_sha,
                    item_ids=[r['item_id'] for r in selected], gpu=gpu, command=cmd,
                    started_utc=started, completed_utc=now(), inputs=before, outputs=outputs,
                    log_path=str(log_path), log=file_record(log_path),
                    dependencies_sha256=dependencies))
                publish(receipt_path, json_bytes(receipt))
                print(f'passed stage={stage} shard={index}', flush=True)
            except Exception:
                stop.set()
                raise

    work = [(stage, gpu, assigned)
            for stage, gpus in (('allinone', a.aio_gpus), ('beats', a.beat_gpus))
            for gpu, assigned in original.assignments(shards, gpus)]
    with ThreadPoolExecutor(max_workers=len(work)) as pool:
        futures = [pool.submit(worker, *job) for job in work]
        for future in futures:
            future.result()
    check_bindings()
    require(v2.runtime_snapshot(a) == runtime, 'Runtime changed during inference')
    after = validate_prepared(a.prepared_dir)
    require(after == (rows, shards, provenance), 'Prepared data changed')
    receipts = verify_all_receipts(a.output_root, contract, rows, shards, True)
    publish(a.output_root / 'completion.json', json_bytes(sealed(dict(scope(), status='passed',
        rows=COUNT, completed_stage_shards=42, run_contract_sha256=contract_sha,
        receipts_sha256=receipts, dependencies_sha256=dependencies))))
    verify_completion(a.output_root, contract, rows, shards)


def common_arguments(parser):
    v2.common_arguments(parser)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--aio-gpus', nargs='+', type=int, required=True)
    parser.add_argument('--beat-gpus', nargs='+', type=int, required=True)
    parser.add_argument('--original-inference-completion', type=Path)
    a = parser.parse_args()
    release_proof = (validate_original_completion(a.original_inference_completion)
                     if a.original_inference_completion else None)
    validate_gpus(a.aio_gpus, a.beat_gpus, release_proof)
    v2.prep.validate_paths(v2.prep.SOURCE, a.prepared_dir)
    require(a.output_root == a.prepared_dir / 'inference_v3'
            and a.output_root.resolve() == a.output_root,
            'Only dedicated Mureka inference_v3 output permitted')
    a.output_root.mkdir(parents=True, exist_ok=True)
    with (a.output_root / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with original_writer_lock(release_proof):
            # Re-read after acquiring the old writer lock to close the proof/lock race.
            if release_proof is not None:
                release_proof = validate_original_completion(a.original_inference_completion)
            run_locked(a, release_proof)
    print('Mureka500 v3 measurement inference passed; no scoring or fitting', flush=True)


if __name__ == '__main__':
    main()
