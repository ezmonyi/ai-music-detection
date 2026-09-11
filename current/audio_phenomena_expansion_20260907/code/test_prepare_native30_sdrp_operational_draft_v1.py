import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('operational_sdrp_draft_under_test',
    HERE / 'prepare_native30_sdrp_operational_draft_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class Fixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.code = self.root / 'code'; self.code.mkdir()
        self.original = self.root / 'original'; self.original.mkdir()
        self.recovery = self.root / 'recovery'; self.recovery.mkdir()
        self.operational_root = self.root / 'operational'; self.operational_root.mkdir()
        self.output = self.root / 'sdrp-output'
        self.loader_audio = []
        self.files = {}
        for name in ('operational_freeze', 'operational_runner', 'base_freeze', 'base_driver',
                     'v2_runner', 'inference_freeze', 'cohort', 'plan', 'extractor', 'core',
                     'bias', 'previous_freeze', 'torch_cuda'):
            path = self.code / name
            path.write_bytes((name + '\n').encode())
            self.files[name] = m.binding(path)
        core_path = self.code / 'expanded_feature_definitions.py'
        core_path.write_bytes(b'core\n')
        self.files['core'] = m.binding(core_path)
        self.runtime = {'python': 'synthetic-cpu-runtime'}
        run = {'source_graph_start': {'synthetic': 'source-graph'}}
        run_path = self.original / 'run_contract.json'; m.write_json_new(run_path, run)
        self.run_binding = m.binding(run_path)
        self.rows = [{'id': f'id{i:04d}'} for i in range(m.EXPECTED_ROWS)]
        self.prior = {'output_root': str(self.original), 'cohort_contract': self.files['cohort'],
            'plan': self.files['plan'], 'extractor': self.files['extractor'],
            'bias': self.files['bias']}
        self.continuation = {'output_root': str(self.recovery),
            'v2_runner': self.files['v2_runner'],
            'original_parent_freeze': self.files['inference_freeze'],
            'previous_continuation_freeze': self.files['previous_freeze'],
            'torch_cuda_source': self.files['torch_cuda']}
        self.operational = {'_path': self.files['operational_freeze']['path'],
            '_sha256': self.files['operational_freeze']['sha256'],
            '_base': self.continuation, 'output_root': str(self.operational_root),
            'base_freeze': self.files['base_freeze'], 'base_driver': self.files['base_driver'],
            'driver': self.files['operational_runner'],
            'original_run_contract': self.run_binding}
        self.engine = types.SimpleNamespace(VERSION='resume_native30_gpu_guard_v2',
            STATUS='passed_native30_operational_continuation_not_feature_extraction',
            PLACEMENT={'host': 'synthetic'}, __file__=self.files['operational_runner']['path'])
        self.backend = types.SimpleNamespace(
            FREEZE_VERSION='native30-sdrp-operational-extraction-parent-freeze-v1',
            FREEZE_STATUS='parent_frozen_for_native30_S_D_R_P_operational_aware_extraction_v1',
            SCOPE={'classifier_fits': 0, 'cohort_admitted': False,
                   'neural_inference_performed': False, 'recovery_applied': False,
                   'M_predictor': False, 'recovery_evidence_consumed': True,
                   'recovery_provenance_is_predictor': False},
            ordinary_path=Path(self.files['base_driver']['path']),
            ORDINARY_SHA=self.files['base_driver']['sha256'],
            o=types.SimpleNamespace(CORE_SHA=self.files['core']['sha256'],
                                    runtime_snapshot=lambda: {'cpu': 'synthetic'}),
            load_operational=lambda: self.engine,
            completion_epoch=lambda completion, bindings, root: {'epoch': 'synthetic'})
        self.publish_completion()

    def close(self):
        self.tmp.cleanup()

    def context_loader(self, path, sha, *, audio):
        self.loader_audio.append(audio)
        return (self.operational, self.prior, {'effective': True}, {'rows': self.rows},
                {'helper': True}, self.runtime)

    def publish_completion(self, *, status=None, map_hash_valid=True):
        for name in ('completion.json', 'final_audit.json'):
            path = self.operational_root / name
            if path.exists(): path.unlink()
        maps = {'stage_receipts': {'receipt': {'sha256': 'a' * 64}},
                'logs': {'log': {'sha256': 'b' * 64}},
                'products': {'product': {'sha256': 'c' * 64}},
                'inputs': {'id0000': {'sha256': 'd' * 64}}}
        base = {'version': self.engine.VERSION, 'status': status or self.engine.STATUS,
            'rows': m.EXPECTED_ROWS, 'original_v2_status': 'partial_without_own_completion',
            'base_recovery_v2_status': 'partial_preserved_without_own_completion',
            'operational_parent_freeze': {'path': self.operational['_path'],
                                          'sha256': self.operational['_sha256']},
            'base_freeze': self.operational['base_freeze'],
            'base_driver': self.operational['base_driver'], 'placement': self.engine.PLACEMENT,
            'runtime': self.runtime, 'cohort_contract': self.prior['cohort_contract'],
            'original_parent_freeze': self.continuation['original_parent_freeze'],
            'source_graph': {'synthetic': 'source-graph'},
            'driver_execution': self.files['operational_runner'],
            'blocked_attempt': self.files['operational_runner'],
            'replacement_attempt_096': {'synthetic': True},
            'beat_provenance': {row['id']: {'kind': 'ordinary_v2'} for row in self.rows},
            'recovered_ids': [],
            'stage_executions': {'receipt': {'attempt': 'replacement'}},
            'recovery_evidence': {'receipt': {'recovery': True}}, **maps}
        base['replacement_attempt_096'] = base['stage_executions']['receipt']
        for key, value in maps.items():
            base[key + '_sha256'] = m.value_hash(value) if map_hash_valid else '0' * 64
        m.write_json_new(self.operational_root / 'final_audit.json', base)
        completion = {**base, 'final_audit': m.binding(self.operational_root / 'final_audit.json')}
        m.write_json_new(self.operational_root / 'completion.json', completion)
        self.completion = m.binding(self.operational_root / 'completion.json')

    def build(self, **changes):
        values = {'operational_freeze': self.operational['_path'],
            'operational_freeze_sha': self.operational['_sha256'],
            'completion': self.completion['path'], 'completion_sha': self.completion['sha256'],
            'output_root': str(self.output),
            'builder_tests': str((HERE / 'test_prepare_native30_sdrp_operational_draft_v1.py').resolve()),
            'backend': self.backend, 'context_loader': self.context_loader,
            'cpu_runtime': {'cpu': 'synthetic'}}
        values.update(changes)
        with mock.patch.object(m, 'OPERATIONAL_FREEZE_SHA', self.operational['_sha256']), \
             mock.patch.object(m, 'OPERATIONAL_SHA', self.files['operational_runner']['sha256']):
            return m.build(**values)


class OperationalSDRPDraftTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.addCleanup(self.f.close)

    def test_valid_terminal_metadata_builds_only_nonauthorizing_draft(self):
        draft = self.f.build()
        self.assertEqual(draft['status'], m.DRAFT_STATUS)
        self.assertFalse(draft['feature_extraction_authorized'])
        self.assertFalse(draft['authorization_boundary']['this_draft_authorizes_extraction'])
        self.assertEqual(draft['inference_completion'], self.f.completion)
        self.assertEqual(draft['terminal_completion_metadata_validation']['rows'], 3830)
        self.assertEqual(self.f.loader_audio, [False])

    def test_value_hash_matches_actual_frozen_engine_for_unicode(self):
        # Do not prove compatibility by hashing both sides with this builder.
        # The actual operational engine exposes the frozen inference helper.
        with mock.patch.dict(sys.modules, {'soundfile': types.ModuleType('soundfile')}):
            import resume_native30_gpu_guard_v2 as actual_engine
        payload = {'source_group': '人声-гитара', 'ids': ['café', '한국어']}
        self.assertEqual(m.value_hash(payload), actual_engine.b.value_hash(payload))
        self.assertEqual(m.canonical(payload), actual_engine.b.canonical(payload))

    def test_draft_has_every_exact_prepare_authority(self):
        draft = self.f.build()
        keys = ('runner', 'tests', 'ordinary_extractor', 'recovery_runner',
                'recovery_parent_freeze', 'inference_runner', 'inference_parent_freeze',
                'inference_completion', 'cohort_contract', 'origin_plan', 'extractor', 'core',
                'bias', 'previous_continuation_freeze', 'torch_cuda_source',
                'operational_runner', 'operational_parent_freeze')
        self.assertTrue(all(draft[key] == m.binding(draft[key]['path']) for key in keys))
        self.assertEqual(draft['duration_s'], 30)
        self.assertTrue(draft['cpu_only'])

    def test_missing_completion_is_hard_failure(self):
        missing = self.f.operational_root / 'missing.json'
        with self.assertRaisesRegex(ValueError, 'safe absolute'):
            self.f.build(completion=str(missing), completion_sha='0' * 64)

    def test_caller_cannot_omit_or_invent_completion_sha(self):
        with self.assertRaisesRegex(ValueError, 'exact completion SHA'):
            self.f.build(completion_sha='')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.f.build(completion_sha='0' * 64)

    def test_nonterminal_completion_is_rejected_even_when_resealed(self):
        self.f.publish_completion(status='running_not_terminal')
        with self.assertRaisesRegex(ValueError, 'nonterminal'):
            self.f.build()

    def test_bad_internal_evidence_hash_is_rejected(self):
        self.f.publish_completion(map_hash_valid=False)
        with self.assertRaisesRegex(ValueError, 'evidence map/hash'):
            self.f.build()

    def test_completion_final_audit_disagreement_is_rejected(self):
        audit = json.loads((self.f.operational_root / 'final_audit.json').read_text())
        audit['rows'] = 1
        (self.f.operational_root / 'final_audit.json').unlink()
        m.write_json_new(self.f.operational_root / 'final_audit.json', audit)
        completion = json.loads(Path(self.f.completion['path']).read_text())
        completion['final_audit'] = m.binding(self.f.operational_root / 'final_audit.json')
        Path(self.f.completion['path']).unlink()
        m.write_json_new(Path(self.f.completion['path']), completion)
        self.f.completion = m.binding(Path(self.f.completion['path']))
        with self.assertRaisesRegex(ValueError, 'canonical payload mismatch'):
            self.f.build()

    def test_existing_or_overlapping_output_rejected(self):
        self.f.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'new canonical'):
            self.f.build()
        self.f.output.rmdir()
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.f.build(output_root=str(self.f.original / 'nested'))

    def test_publish_is_new_and_stays_nonauthorizing(self):
        draft = self.f.build()
        path = self.f.root / 'draft.json'
        m.write_json_new(path, draft)
        saved = json.loads(path.read_text())
        self.assertEqual(saved, draft)
        self.assertFalse(saved['feature_extraction_authorized'])
        with self.assertRaisesRegex(ValueError, 'new safe draft'):
            m.write_json_new(path, draft)


if __name__ == '__main__':
    unittest.main()
