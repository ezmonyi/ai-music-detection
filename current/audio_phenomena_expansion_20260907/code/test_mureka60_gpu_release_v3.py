"""Synthetic CPU-only v3 GPU-release tests; never runs neural inference."""
from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

# The local lightweight test environment may omit libsndfile. No media is
# opened here, but never replace a real module during broader test discovery.
try:
    import soundfile  # noqa: F401
except ModuleNotFoundError:
    sys.modules['soundfile'] = ModuleType('soundfile')

import run_mureka60_inference_v3 as M
import verify_extract_mureka60_v3 as A


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.json_bytes(value))


def csv_put(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_fixture(root, wrong_role=False):
    prepared = (root / 'equal60_development_v2').resolve()
    inference = prepared / 'inference'
    inference.mkdir(parents=True)
    (inference / 'writer.lock').write_bytes(b'')
    rows = [{'item_id': f'original_{index:04d}',
             'standardized_path': str(prepared / 'audio' / f'original_{index:04d}.wav'),
             'standardized_file_sha256': f'{index:064x}'[-64:],
             'role': ('external_generator_unscored' if wrong_role and index == 0 else 'development')}
            for index in range(M.ORIGINAL_COUNT)]
    csv_put(prepared / 'inference_manifest.csv', rows)
    shards = []
    for index in range(M.ORIGINAL_SHARDS):
        part = rows[index * 24:index * 24 + 24]
        path = prepared / f'inference_shard_{index:02d}.txt'
        path.write_text(''.join(row['standardized_path'] + '\n' for row in part))
        shards.append({'index': index, 'rows': len(part), 'sha256': M.sha_file(path)})
    summary = {'contract_sha256': 'synthetic-materialization-contract', 'shards': shards}
    put(prepared / 'materialization_summary.json', summary)
    contract = {'rows': M.ORIGINAL_COUNT, 'output_root': str(inference),
        'runtime_root': str(root / 'runtime'),
        'code_sha256': 'synthetic-original-runner',
        'manifest_sha256': M.sha_file(prepared / 'inference_manifest.csv'),
        'aio_gpus': [0, 1, 2, 3], 'beat_gpus': [4, 5], 'shards': shards}
    put(inference / 'run_contract.json', contract)
    contract_sha = M.sha_file(inference / 'run_contract.json')
    (inference / 'receipts').mkdir()
    for stage, gpus in (('allinone', contract['aio_gpus']), ('beats', contract['beat_gpus'])):
        for index, shard in enumerate(shards):
            inputs = (prepared / f'inference_shard_{index:02d}.txt').read_text().splitlines()
            ids = [rows[index * 24 + offset]['item_id'] for offset in range(len(inputs))]
            receipt = {'status': 'passed', 'stage': stage, 'shard_index': index,
                'run_contract_sha256': contract_sha, 'item_ids': ids,
                'gpu': gpus[index % len(gpus)],
                'command': M.original.command(stage, inputs, Path(contract['runtime_root']), inference),
                'outputs': {str(inference / stage / f'{ids[0]}.synthetic'): {
                    'sha256': '0' * 64, 'bytes': 0}}}
            put(inference / 'receipts' / f'{stage}_{index:02d}.json', receipt)
    put(inference / 'completion.json', {'status': 'passed', 'rows': M.ORIGINAL_COUNT,
        'completed_stage_shards': 134, 'run_contract_sha256': contract_sha})
    return prepared, inference / 'completion.json'


def new500_fixture(root, release_proof):
    prepared = (root / 'mureka60_inputs_v2').resolve()
    output = prepared / 'inference_v3'
    output.mkdir(parents=True)
    rows = []
    for index in range(M.COUNT):
        path = prepared / 'audio' / f'new_{index:03d}.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'input-{index}'.encode())
        rows.append({'item_id': f'new_{index:03d}', 'standardized_path': str(path),
            'standardized_file_sha256': M.sha_file(path),
            'standardized_file_bytes': str(path.stat().st_size),
            'group_id': f'group_{index}'})
    shards = [(index, [r['standardized_path'] for r in rows[index * 24:index * 24 + 24]])
              for index in range(M.SHARDS)]
    dependencies = M.dependency_snapshot()
    args = SimpleNamespace(prepared_dir=prepared, output_root=output,
        inference_root=output, runtime_root=(root / 'runtime').resolve(),
        aio_gpus=[0], beat_gpus=[1], old_code_root=root / 'old', bias=root / 'bias',
        runtime_contract=root / 'runtime.json', checkpoint_manifest=root / 'checkpoints.json')
    contract = M.build_contract(args, rows, shards, {}, {}, dependencies, release_proof)
    put(output / 'run_contract.json', contract)
    contract_sha = M.sha_file(output / 'run_contract.json')
    for name in ('structure', 'demix', 'spec', 'beats', 'receipts', 'logs'):
        (output / name).mkdir()
    receipt_hashes = {}
    for stage in M.STAGES:
        for index, inputs in shards:
            selected = rows[index * 24:index * 24 + len(inputs)]
            outputs = {}
            for row in selected:
                for path in M.original.product_paths(stage, row['item_id'], output):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f'{stage}-{row["item_id"]}-{path.name}'.encode())
                    outputs[str(path)] = M.file_record(path)
            log = output / 'logs' / f'{stage}_{index:02d}.log'
            log.write_text('synthetic-only\n')
            receipt = M.sealed(dict(M.scope(), status='passed', exit_code=0, stage=stage,
                shard_index=index, run_contract_sha256=contract_sha,
                item_ids=[r['item_id'] for r in selected],
                gpu=contract['stage_gpu_assignment'][stage][f'{index:02d}'],
                command=M.original.command(stage, inputs, args.runtime_root, output),
                dependencies_sha256=dependencies, inputs=M.input_records(selected), outputs=outputs,
                log_path=str(log), log=M.file_record(log)))
            receipt_path = output / 'receipts' / f'{stage}_{index:02d}.json'
            put(receipt_path, receipt)
            receipt_hashes[str(receipt_path)] = M.sha_file(receipt_path)
    put(output / 'completion.json', M.sealed(dict(M.scope(), status='passed', rows=M.COUNT,
        completed_stage_shards=42, run_contract_sha256=contract_sha,
        receipts_sha256=receipt_hashes, dependencies_sha256=dependencies)))
    return args, rows, shards


class OriginalReleaseProofTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_positive_full134_fake_proof(self):
        prepared, completion = original_fixture(self.root)
        summary = json.loads((prepared / 'materialization_summary.json').read_text())
        self.assertEqual(set(summary['shards'][0]), {'index', 'rows', 'sha256'})
        proof = M.validate_original_completion(completion, prepared, synthetic_test=True)
        self.assertEqual(proof['status'], 'verified_full134_original1604')
        self.assertEqual(len(proof['receipt_identities']), 134)
        self.assertEqual(len(proof['proof_files']), 4 + 67 + 134)
        M.check_seal(proof)
        M.validate_gpus([0, 6], [1, 7], proof)

    def test_production_manifest_materialization_and_runner_pins(self):
        contract = {'code_sha256': M.ORIGINAL_RUNNER_SHA256}
        summary = {'contract_sha256': M.ORIGINAL_MATERIALIZATION_CONTRACT_SHA256}
        M.validate_production_pins(contract, summary, M.ORIGINAL_MANIFEST_SHA256)
        for changed_contract, changed_summary, changed_manifest in (
                ({'code_sha256': '0' * 64}, summary, M.ORIGINAL_MANIFEST_SHA256),
                (contract, {'contract_sha256': '0' * 64}, M.ORIGINAL_MANIFEST_SHA256),
                (contract, summary, '0' * 64)):
            with self.assertRaisesRegex(ValueError, 'pin mismatch'):
                M.validate_production_pins(changed_contract, changed_summary, changed_manifest)

    def test_nonproduction_root_requires_explicit_synthetic_mode(self):
        prepared, completion = original_fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'isolated synthetic'):
            M.validate_original_completion(completion, prepared)

    def test_missing_and_mutated_proof_rejected(self):
        prepared, completion = original_fixture(self.root)
        missing = prepared / 'inference/receipts/beats_66.json'
        missing.unlink()
        with self.assertRaisesRegex(ValueError, 'exactly134'):
            M.validate_original_completion(completion, prepared, synthetic_test=True)
        prepared, completion = original_fixture((self.root / 'second').resolve())
        receipt = prepared / 'inference/receipts/allinone_00.json'
        value = json.loads(receipt.read_text())
        value['gpu'] = 5
        put(receipt, value)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            M.validate_original_completion(completion, prepared, synthetic_test=True)

    def test_wrong_cohort_and_stale_contract_rejected(self):
        prepared, completion = original_fixture(self.root / 'wrong', wrong_role=True)
        with self.assertRaisesRegex(ValueError, 'development1604'):
            M.validate_original_completion(completion, prepared, synthetic_test=True)
        prepared, completion = original_fixture(self.root / 'stale')
        contract_path = prepared / 'inference/run_contract.json'
        contract = json.loads(contract_path.read_text())
        contract['runtime_root'] += '-changed'
        put(contract_path, contract)
        with self.assertRaisesRegex(ValueError, 'full134'):
            M.validate_original_completion(completion, prepared, synthetic_test=True)

    def test_busy_original_writer_lock_rejected(self):
        prepared, completion = original_fixture(self.root)
        proof = M.validate_original_completion(completion, prepared, synthetic_test=True)
        lock_path = prepared / 'inference/writer.lock'
        with lock_path.open('r+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises((BlockingIOError, OSError)):
                with M.original_writer_lock(proof):
                    self.fail('busy lock was admitted')


class GPUAndContractTests(unittest.TestCase):
    def test_default_reservation_and_disjoint_guards(self):
        M.validate_gpus([6], [7])
        for aio, beats in [([0], [7]), ([6], [6]), ([], [7]), ([6, 6], [7]), ([6], [-1])]:
            with self.assertRaises(ValueError):
                M.validate_gpus(aio, beats)

    def test_actual_idle_rtx5090_guard_is_never_bypassed(self):
        for value in ('NVIDIA A100, 0, 0', 'NVIDIA GeForce RTX 5090, 2000, 0',
                      'NVIDIA GeForce RTX 5090, 0, 20'):
            with patch.object(M.original.subprocess, 'check_output', return_value=value):
                with self.assertRaises(RuntimeError):
                    M.original.require_idle_5090(0)
        with patch.object(M.original.subprocess, 'check_output',
                          return_value='NVIDIA GeForce RTX 5090, 20, 0'):
            M.original.require_idle_5090(0)

    def test_contract_binds_full_proof_dependencies_and_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prepared, completion = original_fixture(root)
            proof = M.validate_original_completion(completion, prepared, synthetic_test=True)
            args = SimpleNamespace(prepared_dir=root / 'new', output_root=root / 'new/inference_v3',
                runtime_root=root / 'runtime', aio_gpus=[0, 6], beat_gpus=[1, 7])
            rows = [{'item_id': 'one'}]
            contract = M.build_contract(args, rows, [(0, ['/one'])], {'p': 'h'}, {'r': 'h'},
                                        {'code': 'h'}, proof)
            self.assertEqual(contract['original_gpu_release_proof'], proof)
            self.assertEqual(contract['stage_gpu_assignment']['allinone']['00'], 0)
            self.assertEqual(contract['stage_gpu_assignment']['allinone']['01'], 6)
            self.assertEqual(contract['dependencies_sha256'], {'code': 'h'})
            changed = json.loads(json.dumps(proof))
            changed['completion_sha256'] = 'f' * 64
            other = M.build_contract(args, rows, [(0, ['/one'])], {'p': 'h'}, {'r': 'h'},
                                     {'code': 'h'}, changed)
            self.assertNotEqual(contract['canonical_sha256'], other['canonical_sha256'])


class StrictAuditorIntegrationTests(unittest.TestCase):
    def test_released_gpus_and_all42_receipts_are_strictly_revalidated(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        original_prepared, completion = original_fixture(root / 'original')
        proof = M.validate_original_completion(completion, original_prepared, synthetic_test=True)
        args, rows, shards = new500_fixture(root / 'new', proof)
        with patch.object(M, 'validate_prepared', return_value=(rows, shards, {})), \
                patch.object(M.v2, 'runtime_snapshot', return_value={}):
            audited_rows, result = A.audit(args, decode=False,
                fixed_original_root=original_prepared, synthetic_test=True)
            self.assertEqual(audited_rows, rows)
            self.assertTrue(result['all42_stage_receipts_verified'])
            self.assertTrue(result['explicit_stage_gpu_assignment_verified'])
            receipt = args.inference_root / 'receipts/beats_20.json'
            value = json.loads(receipt.read_text())
            value['gpu'] = 0
            put(receipt, value)
            with self.assertRaises(ValueError):
                A.audit(args, decode=False, fixed_original_root=original_prepared,
                        synthetic_test=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
