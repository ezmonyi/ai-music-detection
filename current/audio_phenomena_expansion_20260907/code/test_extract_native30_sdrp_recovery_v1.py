"""Synthetic waveform/provenance fixtures only; no real cohort or model runs."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import extract_native30_sdrp_recovery_v1 as m
ordinary_tests = m.o.load_module(m.HERE / 'test_extract_native30_sdrp_v1.py',
    'fa3c15ca8f3e97197f0379da9a7d4e5e36b7a4c2780ee5928893b1c52790ac69', 'test_extract_native30_sdrp_v1')


class RecoveryExtractionTests(unittest.TestCase):
    write_beat_receipt = ordinary_tests.ExtractionTests.write_beat_receipt
    audit = ordinary_tests.ExtractionTests.audit
    legacy = ordinary_tests.ExtractionTests.legacy

    @classmethod
    def setUpClass(cls):
        ordinary_tests.ExtractionTests.setUpClass.__func__(cls)

    def setUp(self):
        ordinary_tests.ExtractionTests.setUp(self)
        self.recovery = self.root / 'recovery'; self.recovery.mkdir(); (self.recovery / 'writer.lock').touch()
        self.contract.update(version=m.VERSION, recovery_root=str(self.recovery), **m.SCOPE)
        self.provenance = {'kind': 'ordinary_v2', 'receipt': m.base.file_binding(self.receipt), 'item_status': 'ordinary_v2_success'}
        self.mapped['beat_provenance'] = self.provenance
        self.audited.update(beat_provenance={'synthetic': self.provenance}, recovery_evidence={}, recovered_ids=[])

    def recovered(self, missing=True):
        self.receipt = self.recovery / 'beats_000.json'; self.mapped['beat_receipt_path'] = str(self.receipt)
        evidence = {}
        for key in ('request', 'raw_replay', 'replay_log'):
            path = self.recovery / (key + '.json'); m.base.write_new(path, {'synthetic': key})
            evidence[key] = m.base.file_binding(path)
        evidence['missing_ids'] = ['synthetic'] if missing else []
        self.payload = {'status': 'recovered_serializer_v1', 'recovery_applied': True, 'ordinary_success': False,
                        'item_ids': ['synthetic'], **{k: evidence[k] for k in ('request', 'raw_replay', 'replay_log')}}
        self.write_beat_receipt()
        self.provenance = {'kind': 'recovered_serializer_v1', 'receipt': m.base.file_binding(self.receipt),
                           'raw_replay': evidence['raw_replay'], 'item_status': 'recovered_empty_unavailable' if missing else 'preserved_cli_success'}
        self.mapped['beat_provenance'] = self.provenance
        self.audited = self.audit()
        self.audited.update(beat_provenance={'synthetic': self.provenance}, recovery_evidence={str(self.receipt): evidence},
                            recovered_ids=['synthetic'] if missing else [])

    def run_fixture(self):
        return m.run_contract(self.contract, self.extractor, self.bias, lambda _: self.audited)

    def test_pinned_backends_unchanged(self):
        self.assertEqual(m.base.digest(m.ordinary_path), m.ORDINARY_SHA)
        self.assertEqual(m.base.digest(m.HERE / 'continue_native30_empty_beats_v1.py'), m.RECOVERY_SHA)
        self.assertEqual(len(m.o.feature_names(self.extractor)), 43)

    def test_ordinary_branch_reuses_strict_existing_gate(self):
        inputs = m.inspect_inputs(self.contract, self.mapped, self.audited)
        self.assertEqual(inputs['beat_provenance']['kind'], 'ordinary_v2')
        self.payload['recovery_applied'] = True; self.write_beat_receipt()
        with self.assertRaises(ValueError): m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_ordinary_item_numerics_unchanged_in_separate_backend(self):
        baseline = m.o.clean(self.legacy()); self.assertEqual(self.run_fixture()['completed'], 1)
        path = Path(self.contract['output_root']) / 'items/synthetic.json'
        item = m.base.read_json(path)['payload']
        self.assertEqual(item['legacy_result'], baseline)
        self.assertEqual(item['inputs']['beat_provenance']['item_status'], 'ordinary_v2_success')
        item['inputs']['beat_provenance']['item_status'] = 'recovered_empty_unavailable'
        path.unlink(); m.base.write_new(path, {'payload': item, 'receipt_sha256': m.base.value_hash(item)})
        with self.assertRaisesRegex(ValueError, 'existing COMMIT changed'): self.run_fixture()

    def test_recovered_empty_is_distinct_scientific_missing_no_new_predictors(self):
        self.recovered(); baseline = m.o.clean(self.legacy())
        result = self.run_fixture(); self.assertEqual(result['completed'], 1)
        output = Path(self.contract['output_root']); receipt = m.base.read_json(output / 'items/synthetic.json')['payload']
        self.assertEqual(receipt['legacy_result'], baseline)
        self.assertEqual(receipt['inputs']['beat_provenance']['item_status'], 'recovered_empty_unavailable')
        self.assertEqual(receipt['legacy_result']['r_eligible'], 0)
        self.assertTrue(all(receipt['legacy_result']['r__' + name] is None for name in self.extractor.R_FEATURES))
        self.assertEqual(set(k for k in receipt['legacy_result'] if k.startswith(m.o.PREFIXES)), set(m.o.feature_names(self.extractor)))
        self.assertNotIn('NaN', (output / 'items/synthetic.json').read_text())
        commit = m.base.read_json(output / 'COMMIT.json')
        self.assertEqual(commit['status'], m.COMMIT_STATUS); self.assertFalse(commit['recovery_applied'])
        self.assertTrue(commit['recovery_evidence_consumed']); self.assertFalse(commit['recovery_provenance_is_predictor'])
        self.assertEqual(m.base.read_json(output / 'manifest.json')['recovered_ids'], ['synthetic'])
        before = m.o.products(output)
        with patch.object(self.extractor, 'process', side_effect=AssertionError('must not recompute')):
            self.assertEqual(self.run_fixture()['status'], 'verified_existing_COMMIT')
        self.assertEqual(before, m.o.products(output))

    def test_preserved_success_has_identical_nonempty_numerics(self):
        self.beat.write_text('0.5\t1\n1.0\t2\n1.5\t3\n2.0\t4\n2.5\t1\n3.0\t2\n')
        self.recovered(missing=False)
        baseline = m.o.clean(self.legacy()); self.assertEqual(self.run_fixture()['completed'], 1)
        item = m.base.read_json(Path(self.contract['output_root']) / 'items/synthetic.json')['payload']
        self.assertEqual(item['legacy_result'], baseline)
        self.assertEqual(item['inputs']['beat_provenance']['item_status'], 'preserved_cli_success')

    def test_recovered_cannot_pass_ordinary_gate(self):
        self.recovered()
        with self.assertRaisesRegex(ValueError, 'successful nonrecovery'):
            m.o.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_repaired_nonempty_output_rejected_even_if_hashes_rebound(self):
        self.beat.write_text('1.0\t1\n'); self.recovered()
        with self.assertRaisesRegex(ValueError, 'genuinely empty'): m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_provenance_type_and_set_mutations_rejected(self):
        self.recovered()
        for mutation in (lambda a: a['beat_provenance']['synthetic'].update(item_status='preserved_cli_success'),
                         lambda a: a['beat_provenance']['synthetic'].update(kind='ordinary_v2'),
                         lambda a: a.update(recovered_ids=[]),
                         lambda a: a['recovery_evidence'][str(self.receipt)].update(missing_ids=[])):
            changed = copy.deepcopy(self.audited); mutation(changed)
            with self.assertRaises(ValueError): m.inspect_inputs(self.contract, self.mapped, changed)

    def test_raw_sidecar_mutation_rejected_after_extraction(self):
        self.recovered(); inputs = m.inspect_inputs(self.contract, self.mapped, self.audited)
        Path(self.provenance['raw_replay']['path']).write_text('changed raw arrays')
        with self.assertRaisesRegex(ValueError, 'recovery evidence changed'): m.recheck_item_inputs(inputs)

    def test_physical_duration_and_input_float_gates_unchanged(self):
        import soundfile as sf
        self.recovered()
        for samples, subtype in ((self.samples[:-1], 'FLOAT'), (self.samples, 'PCM_16')):
            sf.write(self.source, samples, 44100, subtype=subtype)
            with self.assertRaises(ValueError): m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_failure_is_retained_without_commit(self):
        self.recovered()
        with patch.object(self.extractor, 'process', side_effect=RuntimeError('synthetic failure')):
            result = self.run_fixture()
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        output = Path(self.contract['output_root']); self.assertFalse((output / 'COMMIT.json').exists())
        self.assertEqual(len(list((output / 'failures').iterdir())), 1)

    def test_derivative_verifier_binding_and_provenance_are_not_trusted_without_check(self):
        self.recovered()
        parent = self.root / 'parent.json'; m.base.write_new(parent, {'synthetic': 'authority'}); binding = m.base.file_binding(parent)
        self.contract['bindings'].update(recovery_parent_freeze=binding, inference_parent_freeze=binding, cohort_contract=binding)
        completion = {**self.audited['evidence'], 'beat_provenance': self.audited['beat_provenance'],
                      'recovery_evidence': self.audited['recovery_evidence'], 'recovered_ids': self.audited['recovered_ids'],
                      'original_parent_freeze': binding, 'cohort_contract': binding}
        path = self.root / 'completion.json'; m.base.write_new(path, completion)
        self.contract['bindings']['inference_completion'] = m.base.file_binding(path)
        proof = {'status': 'verified_complete_native30_inference_with_frozen_recovery', 'rows': 1,
                 'completion': m.base.file_binding(path), 'evidence_sha256': m.base.value_hash(self.audited['evidence']),
                 'beat_provenance': self.audited['beat_provenance']}
        backend = SimpleNamespace(verify_completion=lambda *args: proof)
        with patch.object(m, 'load_recovery', return_value=backend):
            audited = m.verify_completed_inference(self.contract)
            self.assertEqual(audited['recovered_ids'], ['synthetic'])
            proof['evidence_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'audited evidence'): m.verify_completed_inference(self.contract)


if __name__ == '__main__':
    unittest.main()
