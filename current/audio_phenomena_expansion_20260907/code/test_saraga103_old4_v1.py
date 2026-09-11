"""Synthetic only; never launches neural inference or opens real Saraga audio."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import os
import sys

import run_saraga103_inference_v1 as R
import verify_extract_saraga103_v1 as A
import saraga103_cuda_guard_v1 as G


class Old4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / 'output'
        self.out.mkdir()

    def rows(self, count=103):
        rows = []
        for i in range(count):
            item = f'synthetic_saraga_{i:03d}'
            audio = self.root / 'accepted' / item / 'audio.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(f'synthetic wave bytes {i}'.encode())
            rows.append(dict(item_id=item, label='0', source_id=R.SOURCE,
                source_group=R.SOURCE, group_id='synthetic_group', role=R.ROLE,
                standardized_path=str(audio), audio_path=str(audio),
                standardized_file_sha256=R.sha_file(audio), standardized_file_bytes=str(audio.stat().st_size),
                native_sr='48000' if i < 7 else '44100', audio_offset_s='0', requires_crop='0'))
        return R.adapted_rows(rows, self.out)

    def test_unique_basename_aliases_preserve_all103(self):
        rows = self.rows()
        self.assertEqual(len(set(Path(r['standardized_path']).stem for r in rows)), 103)
        self.assertEqual({Path(r['accepted_standardized_path']).name for r in rows}, {'audio.wav'})
        self.assertEqual([len(p) for _, p in R.shards_for(rows)], [24, 24, 24, 24, 7])
        self.assertEqual([p for _, part in R.shards_for(rows) for p in part], [r['standardized_path'] for r in rows])

    def test_alias_commit_no_replace_and_resume(self):
        rows = self.rows(3)
        R.publish_aliases(self.out, rows)
        R.publish_aliases(self.out, rows)
        self.assertEqual(len(R.validate_aliases(self.out, rows)), 3)
        with self.assertRaises(FileExistsError):
            R.materializer.commit_directory(self.out / 'inputs', {'inference_manifest.csv'})

    def test_orphan_alias_directory_fails_closed(self):
        rows = self.rows(1)
        (self.out / 'inputs').mkdir()
        with self.assertRaises((ValueError, FileNotFoundError)):
            R.publish_aliases(self.out, rows)

    def test_changed_alias_manifest_rejected(self):
        rows = self.rows(1)
        R.publish_aliases(self.out, rows)
        (self.out / 'inputs/inference_manifest.csv').write_text('tamper')
        with self.assertRaises(ValueError):
            R.validate_aliases(self.out, rows)

    def test_same_bytes_copy_is_not_hardlink(self):
        rows = self.rows(1)
        R.publish_aliases(self.out, rows)
        alias = Path(rows[0]['standardized_path'])
        data = alias.read_bytes()
        alias.unlink()
        alias.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'hardlink'):
            R.validate_aliases(self.out, rows)

    def test_changed_source_also_invalidates_alias(self):
        rows = self.rows(1)
        R.publish_aliases(self.out, rows)
        Path(rows[0]['accepted_standardized_path']).write_bytes(b'tamper')
        with self.assertRaises(ValueError):
            R.validate_aliases(self.out, rows)

    def test_alias_symlink_rejected(self):
        rows = self.rows(1)
        R.publish_aliases(self.out, rows)
        alias = Path(rows[0]['standardized_path'])
        alias.unlink()
        alias.symlink_to(rows[0]['accepted_standardized_path'])
        with self.assertRaises(ValueError):
            R.validate_aliases(self.out, rows)

    def test_duplicate_or_unsafe_identity_rejected(self):
        rows = self.rows(1)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            R.adapted_rows(rows + rows, self.out)
        rows[0]['item_id'] = '../bad'
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            R.adapted_rows(rows, self.out)

    def test_roles_are_human_and_all103_remain(self):
        rows = self.rows()
        ledger = A.role_items(rows)
        self.assertEqual(len(ledger), 103)
        self.assertTrue(all(x['role'] == R.ROLE and x['label'] == 0
            and x['source_id'] == R.SOURCE and not x['evaluation_allowed']
            and not x['classifier_admission_authorized'] for x in ledger))

    def test_measurement_scope_rejects_generator_or_scoring(self):
        R.measurement_only(R.scope())
        for key, value in [('role', 'external_generator_unscored'), ('scores_generated', True),
                           ('classifier_admission_authorized', True)]:
            with self.assertRaises(ValueError):
                R.measurement_only(dict(R.scope(), **{key: value}))

    def test_explicit_disjoint_nonzero_gpus(self):
        R.validate_gpus([6], [7])
        R.validate_gpus([1, 2, 4, 5, 6], [7])
        for aio, beats in [([0], [7]), ([6], [6]), ([], [7]), ([6, 6], [7]), ([-1], [7])]:
            with self.assertRaises(ValueError):
                R.validate_gpus(aio, beats)

    def test_dynamic_extraction_manifest_and_strict_command(self):
        a = SimpleNamespace(old_code_root=self.root / 'old', prepared_dir=self.root / 'native',
            inference_root=self.out, output_dir=self.out / 'measurements', bias=self.root / 'bias', workers=2)
        command = A.extraction_command(a)
        self.assertEqual(command[command.index('--manifest') + 1], str(self.out / 'inputs/inference_manifest.csv'))
        self.assertIn('--strict', command)
        self.assertNotIn('--limit', command)

    def test_partial_completion_rejected_before_decode(self):
        a = SimpleNamespace(inference_root=self.out, output_root=self.out)
        R.publish(self.out / 'run_contract.json', R.json_bytes(R.sealed(R.scope())))
        R.publish(self.out / 'completion.json', R.json_bytes(R.sealed(dict(rows=102, completed_stage_shards=9))))
        with patch.object(R, 'validate_paths'), patch.object(R, 'current_contract') as current:
            with self.assertRaisesRegex(ValueError, 'All10'):
                A.audit(a)
            current.assert_not_called()

    def test_independent_freeze_and_hash_required(self):
        a = SimpleNamespace(root=self.root, frozen=None, frozen_sha256=None)
        contract = R.sealed(dict(R.scope(), rows=103))
        with self.assertRaisesRegex(ValueError, 'freeze required'):
            R.verify_authorization(a, contract)
        evidence = self.root / 'review.json'
        evidence.write_text('synthetic independent review')
        a.frozen = self.root / 'frozen.json'
        wrapper = dict(status='frozen', authorized_stage=R.STAGE, contract=contract,
            contract_sha256=contract['canonical_sha256'], independent_review=dict(approved=True,
                reviewer='root', evidence='review.json', evidence_sha256=R.sha_file(evidence)))
        a.frozen.write_bytes(R.json_bytes(wrapper))
        a.frozen_sha256 = R.sha_file(a.frozen)
        R.verify_authorization(a, contract)
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            R.verify_authorization(a, R.sealed(dict(R.scope(), rows=102)))
        evidence.write_text('changed review')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            R.verify_authorization(a, contract)

    def test_selection_or_interval_freeze_cannot_authorize_old4(self):
        a = SimpleNamespace(root=self.root, frozen=self.root / 'f.json')
        contract = R.sealed(dict(R.scope(), rows=103))
        a.frozen.write_bytes(R.json_bytes(dict(status='frozen', authorized_stage=R.materializer.STAGE,
            contract=contract, contract_sha256=contract['canonical_sha256'])))
        a.frozen_sha256 = R.sha_file(a.frozen)
        with self.assertRaisesRegex(ValueError, 'scope'):
            R.verify_authorization(a, contract)

    def test_original_model_command_remains_gpu_only(self):
        for stage in R.STAGES:
            command = R.original.command(stage, ['synthetic.wav'], self.root / 'runtime', self.out)
            if stage == 'allinone':
                self.assertEqual(command[command.index('-d') + 1], 'cuda')
                self.assertIn('harmonix-all', command)
            else:
                self.assertIn('--float16', command)
                self.assertIn('--no-dbn', command)

    def test_input_record_tamper_rejected(self):
        rows = self.rows(1)
        R.publish_aliases(self.out, rows)
        rows[0]['standardized_file_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'input changed'):
            R.input_records(rows)

    def test_unreceipted_empty_beats_not_admitted(self):
        rows = self.rows(1)
        (self.out / 'beats').mkdir()
        beat = self.out / 'beats' / (rows[0]['item_id'] + '.beats')
        beat.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'Unreceipted'):
            R.validate_inventory(self.out, rows, [])
        R.validate_inventory(self.out, rows, [('beats', rows)])
        # The inventory check alone is not acceptance; receipt verification is required.

    def test_frozen_helpers_are_unchanged(self):
        current = R.dependency_snapshot()
        for name, digest in R.PINNED_HELPERS.items():
            self.assertEqual(current[str(Path(R.__file__).resolve().parent / name)], digest)

    def native_fixture(self, sr):
        item = 'synthetic_native'
        selected = dict(item_id=item, mbid='synthetic_native', group_id='synthetic_group',
            native_sample_rate_hz=str(sr), source_audio_path=str(self.root / 'native.mp3'),
            acquisition_raw_sha256='a' * 64, actual_eof_frames=str(60 * sr + 5),
            soundfile_header_frames=str(60 * sr + 31), source_offset_seconds=str(2 / sr))
        proof = dict(native_sample_rate_hz=sr, observed_actual_frames=60 * sr + 5,
            physical_header_frames=60 * sr + 31, crop_start_frame=2, crop_frames=60 * sr,
            crop_end_frame_exclusive=2 + 60 * sr, native_float64_sha256='b' * 64,
            native_float32_sha256='c' * 64,
            raw_hashes_before=dict(sha256='a' * 64, bytes=20), raw_hashes_after=dict(sha256='a' * 64, bytes=20))
        saved = dict(item_id=item, role=R.ROLE, classifier_admission=False,
            contract_sha256=R.MEASUREMENT_CONTRACT_SHA,
            status='passed_interval_materialization_not_classifier_admission',
            selection_row_sha256=R.materializer.util.seal(selected), interval_proof=proof,
            wav=dict(file_sha256='d' * 64, bytes=100))
        native = {k: str(v) for k, v in R.materializer.native_row(selected, saved).items()}
        row = dict(item_id=item, id=item, label='0', source_id=R.SOURCE, source_group=R.SOURCE,
            group_id=selected['group_id'], role=R.ROLE,
            standardized_path=str(self.out / 'items' / item / 'audio.wav'),
            audio_path=str(self.out / 'items' / item / 'audio.wav'), audio_offset_s='0', requires_crop='0',
            duration='60', duration_sec='60', native_sample_rate_hz=str(sr), native_sr=str(sr),
            original_source_audio_path=selected['source_audio_path'], original_source_audio_sha256='a' * 64,
            standardized_sr='44100', standardized_frames='2646000', standardized_channels='2',
            standardized_file_sha256='d' * 64, evaluation_allowed='False', classifier_admission_authorized='False')
        return row, native, selected, saved, self.out

    def test_native_44100_and_48000_preserve_distinct_native_counts(self):
        for sr in (44100, 48000):
            fixture = self.native_fixture(sr)
            R.validate_row(*fixture)
            self.assertEqual(int(fixture[1]['crop_frames']), 60 * sr)
            self.assertEqual(fixture[0]['standardized_frames'], '2646000')

    def test_native_header_cannot_replace_eof(self):
        row, native, selected, saved, output = self.native_fixture(48000)
        selected['actual_eof_frames'] = selected['soundfile_header_frames']
        saved['selection_row_sha256'] = R.materializer.util.seal(selected)
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            R.validate_row(row, native, selected, saved, output)

    def test_native_source_and_role_tampering_rejected(self):
        for key, value in [('source_id', 'Mureka_v9'), ('label', '1'), ('role', 'development'),
                           ('native_sr', '44100'), ('audio_offset_s', '1')]:
            row, native, selected, saved, output = self.native_fixture(48000)
            row[key] = value
            with self.assertRaises(ValueError):
                R.validate_row(row, native, selected, saved, output)

    def feature_fixture(self):
        rows = self.rows()
        a = SimpleNamespace(inference_root=self.out, output_dir=self.out / 'measurements',
            old_code_root=self.root / 'old', bias=self.root / 'bias')
        a.old_code_root.mkdir()
        for name in ('extract_expanded_four_family.py', 'expanded_feature_definitions.py'):
            (a.old_code_root / name).write_text('synthetic code fixture only')
        a.bias.write_text('synthetic bias fixture only')
        features = a.output_dir / 'features'
        features.mkdir(parents=True)
        expected_payload = dict(extractor_sha256=R.sha_file(a.old_code_root / 'extract_expanded_four_family.py'),
            core_sha256=R.sha_file(a.old_code_root / 'expanded_feature_definitions.py'),
            bias_sha256=R.sha_file(a.bias), duration=60.0, demix_roots=[str(self.out / 'demix')],
            beat_roots=[str(self.out / 'beats')], structure_roots=[str(self.out / 'structure')])
        fingerprint = hashlib.sha256(json.dumps(expected_payload, sort_keys=True,
            separators=(',', ':'), allow_nan=True).encode()).hexdigest()
        metadata = dict(rows=103, complete=103, partial=0, duration_sec=60,
            run_payload=expected_payload, run_fingerprint=fingerprint)
        (features / 'expanded_features_60s_metadata.json').write_text(json.dumps(metadata))
        products, payloads = {}, []
        for row in rows:
            inputs = {'source_audio_sha256': row['standardized_file_sha256']}
            for stem in A.STEMS:
                path = self.out / 'demix/htdemucs' / row['item_id'] / (stem + '.wav')
                products[str(path)] = dict(sha256='e' * 64, bytes=100)
                inputs[stem + '_sha256'] = 'e' * 64
            for directory, suffix, key in [('beats', '.beats', 'beats_sha256'),
                                            ('structure', '.json', 'structure_sha256')]:
                path = self.out / directory / (row['item_id'] + suffix)
                products[str(path)] = dict(sha256=hashlib.sha256(b'').hexdigest(), bytes=0)
                inputs[key] = products[str(path)]['sha256']
            payloads.append(dict(item_id=row['item_id'], label='0', source_id=R.SOURCE,
                group_id=row['group_id'], native_sample_rate_hz=int(row['native_sr']), status='complete',
                run_fingerprint=fingerprint, bias_sha256=R.sha_file(a.bias),
                input_hashes=json.dumps(inputs), r__ibi_cv=float('nan'), r_eligible=0,
                p__section_duration_cv=float('nan'), p_eligible=0))
        self.write_features(features, payloads)
        return a, rows, products, payloads

    def write_features(self, features, payloads):
        with (features / 'expanded_features_60s.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(payloads[0]))
            writer.writeheader()
            writer.writerows(payloads)
        (features / 'expanded_features_60s.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in payloads))

    def test_all103_unavailable_values_are_retained(self):
        a, rows, products, payloads = self.feature_fixture()
        records = A.feature_records(a, rows, products)
        self.assertEqual(len(records), 103)
        self.assertTrue(all(r['r__ibi_cv'] == 'nan' and r['r_eligible'] == '0' for r in records))

    def test_extracted_wrong_source_or_omitted_row_rejected(self):
        a, rows, products, payloads = self.feature_fixture()
        payloads[0]['source_id'] = 'Mureka_v9'
        self.write_features(a.output_dir / 'features', payloads)
        with self.assertRaisesRegex(ValueError, 'JSONL identity'):
            A.feature_records(a, rows, products)
        self.write_features(a.output_dir / 'features', payloads[1:])
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            A.feature_records(a, rows, products)

    def test_extracted_input_hash_and_prediction_rejected(self):
        a, rows, products, payloads = self.feature_fixture()
        inputs = json.loads(payloads[0]['input_hashes'])
        inputs['beats_sha256'] = 'f' * 64
        payloads[0]['input_hashes'] = json.dumps(inputs)
        self.write_features(a.output_dir / 'features', payloads)
        with self.assertRaisesRegex(ValueError, 'input hashes'):
            A.feature_records(a, rows, products)
        for payload in payloads:
            payload['prediction'] = 0
        self.write_features(a.output_dir / 'features', payloads)
        with self.assertRaisesRegex(ValueError, 'scoring'):
            A.feature_records(a, rows, products)


class FakeCUDA:
    def __init__(self):
        self.available = True
        self.count = 1
        self.initialized = False
        self.name = 'NVIDIA GeForce RTX 5090'
        self.init_error = False
        self.sync_error = False

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def init(self):
        if self.init_error:
            raise RuntimeError('synthetic CUDA init error')
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def set_device(self, index):
        assert index == 0

    def get_device_properties(self, index):
        assert index == 0
        return SimpleNamespace(name=self.name, uuid='GPU-synthetic-fixture', total_memory=32 * 1024**3)

    def synchronize(self, index):
        assert index == 0
        if self.sync_error:
            raise RuntimeError('synthetic CUDA synchronize error')


class FakeTorch:
    __version__ = 'synthetic_torch'
    version = SimpleNamespace(cuda='synthetic_cuda')
    float32 = 'synthetic_float32'

    def __init__(self):
        self.cuda = FakeCUDA()
        self.allocation_error = False
        self.allocation_device = 'cuda'

    def empty(self, shape, dtype, device):
        assert shape == (1,) and dtype == self.float32 and device == 'cuda:0'
        if self.allocation_error:
            raise RuntimeError('synthetic CUDA allocation error')
        return SimpleNamespace(device=SimpleNamespace(type=self.allocation_device, index=0))


class CUDAGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.entry = self.root / 'beat_this'
        self.entry.write_text('synthetic console fixture; not executed by a model')
        self.evidence = self.root / 'guard.json'
        self.torch = FakeTorch()
        self.env = patch.dict(os.environ, CUDA_VISIBLE_DEVICES='7')
        self.env.start()
        self.addCleanup(self.env.stop)

    def launch(self, dispatch):
        return G.launch('beats', 7, self.entry, ['synthetic.wav', '--gpu', '0'],
                        self.evidence, torch=self.torch, dispatch=dispatch)

    def rejected_before_dispatch(self, pattern):
        dispatch = Mock()
        with self.assertRaisesRegex(RuntimeError, pattern):
            self.launch(dispatch)
        dispatch.assert_not_called()
        self.assertFalse(self.evidence.exists())

    def test_cuda_unavailable_never_dispatches_cpu(self):
        self.torch.cuda.available = False
        self.rejected_before_dispatch('CUDA unavailable')

    def test_wrong_visible_model_rejected(self):
        self.torch.cuda.name = 'NVIDIA RTX 4090'
        self.rejected_before_dispatch('not an RTX 5090')

    def test_zero_or_multiple_visible_devices_rejected(self):
        for count in (0, 2):
            self.torch.cuda.count = count
            self.rejected_before_dispatch('Exactly one')

    def test_cuda_assignment_mismatch_rejected(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='6'):
            self.rejected_before_dispatch('assignment mismatch')

    def test_cuda_initialization_failure_never_dispatches(self):
        self.torch.cuda.init_error = True
        self.rejected_before_dispatch('init error')

    def test_cuda_allocation_failure_never_dispatches(self):
        self.torch.allocation_error = True
        self.rejected_before_dispatch('allocation error')

    def test_cpu_allocation_never_dispatches(self):
        self.torch.allocation_device = 'cpu'
        self.rejected_before_dispatch('wrong device')

    def test_cuda_synchronize_failure_never_dispatches(self):
        self.torch.cuda.sync_error = True
        self.rejected_before_dispatch('synchronize error')

    def test_same_process_dispatch_exact_argv_and_guard_evidence(self):
        argv_before = sys.argv
        observations = []
        def dispatch(path, run_name):
            observations.append((path, run_name, sys.argv.copy(), os.getpid()))
            raise SystemExit(0)
        result = self.launch(dispatch)
        self.assertIs(sys.argv, argv_before)
        self.assertEqual(observations, [(str(self.entry), '__main__',
            [str(self.entry), 'synthetic.wav', '--gpu', '0'], os.getpid())])
        self.assertEqual(result['before'], result['after'])
        self.assertTrue(result['entrypoint_dispatched_same_process'])
        self.assertFalse(result['model_or_torch_monkeypatched'])
        self.assertEqual(json.loads(self.evidence.read_text()), result)

    def test_nonzero_entrypoint_exit_no_success_evidence(self):
        with self.assertRaisesRegex(RuntimeError, 'unsuccessfully'):
            self.launch(Mock(side_effect=SystemExit(2)))
        self.assertFalse(self.evidence.exists())

    def test_post_dispatch_cuda_loss_rejects_success(self):
        def dispatch(*args, **kwargs):
            self.torch.cuda.available = False
        with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
            self.launch(dispatch)
        self.assertFalse(self.evidence.exists())

    def test_existing_evidence_forbids_redispatch(self):
        self.launch(Mock())
        dispatch = Mock()
        with self.assertRaisesRegex(RuntimeError, 'Existing guard evidence'):
            self.launch(dispatch)
        dispatch.assert_not_called()

    def test_both_original_arguments_wrapped_without_mutation(self):
        for stage, gpu in [('allinone', 6), ('beats', 7)]:
            runtime, output = self.root / 'runtime', self.root / 'output'
            original = R.original.command(stage, ['one.wav', 'two.wav'], runtime, output)
            guarded = R.guarded_command(stage, ['one.wav', 'two.wav'], runtime, output, gpu, 3)
            self.assertEqual(guarded[guarded.index('--') + 1:], original[1:])
            self.assertEqual(guarded[guarded.index('--entrypoint') + 1], original[0])
            self.assertEqual(guarded[guarded.index('--physical-gpu') + 1], str(gpu))

    def test_actual_runpy_dispatch_of_synthetic_console_entrypoint(self):
        self.entry.write_text('import sys\nassert sys.argv[1:] == ["synthetic.wav", "--gpu", "0"]\nraise SystemExit(0)\n')
        G.launch('beats', 7, self.entry, ['synthetic.wav', '--gpu', '0'], self.evidence,
                 torch=self.torch)
        self.assertTrue(json.loads(self.evidence.read_text())['entrypoint_returned_successfully'])

    def test_auditor_reconciles_device_and_command_evidence(self):
        result = self.launch(Mock())
        output = self.root / 'output'
        (output / 'logs').mkdir(parents=True)
        path = R.guard_path(output, 'beats', 0)
        path.write_text(json.dumps(result))
        contract = dict(dependencies_sha256={str(R.GUARD): G.file_sha(G.__file__)},
                        runtime_sha256={str(self.entry): G.file_sha(self.entry)})
        original = [str(self.entry), 'synthetic.wav', '--gpu', '0']
        R.validate_guard(output, 'beats', 0, 7, original, contract)
        with self.assertRaisesRegex(ValueError, 'assignment'):
            R.validate_guard(output, 'beats', 0, 6, original, contract)
        with self.assertRaisesRegex(ValueError, 'execution evidence'):
            R.validate_guard(output, 'beats', 0, 7, ['wrong'], contract)


if __name__ == '__main__':
    unittest.main()
