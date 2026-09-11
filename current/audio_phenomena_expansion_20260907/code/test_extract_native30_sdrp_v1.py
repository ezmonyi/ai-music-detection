"""Synthetic CPU-only fixtures; no real cohort or neural inference execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import extract_native30_sdrp_v1 as m


OLD = Path(__file__).resolve().parents[2] / 'source_diversity_expansion_20260905/code'


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = m.load_extractor(OLD)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.infer = self.root / 'inference'
        for name in ('demix/htdemucs/synthetic', 'beats', 'structure', 'spec', 'receipts'):
            (self.infer / name).mkdir(parents=True, exist_ok=True)
        (self.infer / 'writer.lock').touch()
        # Exactly 30 seconds, 44.1k stereo; plan native rate is deliberately 48k.
        self.samples = np.zeros((1323000, 2), dtype=np.float32)
        self.source = self.root / 'synthetic.wav'
        sf.write(self.source, self.samples, 44100, subtype='FLOAT')
        for stem in m.physical.STEMS:
            sf.write(self.infer / 'demix/htdemucs/synthetic' / (stem + '.wav'), self.samples, 44100, subtype='PCM_16')
        self.beat = self.infer / 'beats/synthetic.beats'
        self.beat.touch()
        self.structure = self.infer / 'structure/synthetic.json'
        self.structure.write_bytes(m.base.canonical({'path': str(self.source), 'segments': [
            {'start': 0, 'end': 7}, {'start': 7, 'end': 18}, {'start': 18, 'end': 30}]}))
        np.save(self.infer / 'spec/synthetic.npy', np.zeros((4, 3000, 81), dtype=np.float32))
        self.origin = {'id': 'synthetic', 'source_group': 'FMA', 'label': '0', 'role': 'development',
                       'group_id': 'g_shared', 'component_id': 'c_shared',
                       'source_origin': {'sample_rate_hz': 48000, 'channels': 2}}
        self.row = {**{k: self.origin[k] for k in m.physical.IDENTITY},
                    'origin_plan_row_sha256': m.base.value_hash(self.origin), 'input': m.base.file_binding(self.source),
                    'waveform_float32_sha256': hashlib.sha256(self.samples.astype('<f4').tobytes()).hexdigest()}
        self.infer_freeze = {'output_root': str(self.infer), 'shards': [{'index': 0, 'item_ids': ['synthetic']}]}
        self.mapped = m.mapped_rows([self.row], {'rows': [self.origin]}, self.infer_freeze)[0]
        self.receipt = Path(self.mapped['beat_receipt_path'])
        self.payload = {'status': 'passed', 'stage': 'beats', 'item_ids': ['synthetic'],
                        'ordinary_empty_beat_allowed_only_after_this_successful_command': True, 'recovery_applied': False}
        self.write_beat_receipt()
        self.bias_path = self.root / 'synthetic_bias.npz'
        self.bias = np.zeros(2049, dtype=np.float64)
        np.savez(self.bias_path, frequencies_hz=np.fft.rfftfreq(4096, 1 / 44100), selected_bias_db=self.bias)
        self.contract = {'version': m.VERSION, 'rows': [self.mapped], 'expected_count': 1,
                         'output_root': str(self.root / 'extraction'), 'inference_root': str(self.infer),
                         'bindings': {'bias': m.base.file_binding(self.bias_path)}, **m.SCOPE}
        self.audited = self.audit()

    def write_beat_receipt(self):
        self.receipt.write_bytes(m.base.canonical({'payload': self.payload, 'receipt_sha256': m.base.value_hash(self.payload)}))

    def audit(self):
        evidence = {'inputs': {'synthetic': m.physical.inspect_audio(self.source, self.row, input_audio=True)},
                    'products': {**m.physical.verify_products('allinone', [self.row], self.infer),
                                 **m.physical.verify_products('beats', [self.row], self.infer)},
                    'stage_receipts': {str(self.receipt): {'binding': m.base.file_binding(self.receipt),
                                                         'payload_sha256': m.base.value_hash(self.payload)}}, 'logs': {}}
        for key in ('inputs', 'products', 'stage_receipts', 'logs'):
            evidence[key + '_sha256'] = m.base.value_hash(evidence[key])
        return {'verification': {'status': 'synthetic_verified_only', 'evidence_sha256': m.base.value_hash(evidence)}, 'evidence': evidence}

    def run_fixture(self):
        return m.run_contract(self.contract, self.extractor, self.bias, lambda _: self.audited)

    def legacy(self, row=None):
        args = SimpleNamespace(duration=30, demix_root=[self.infer / 'demix'], beat_root=[self.infer / 'beats'],
                               structure_root=[self.infer / 'structure'])
        return self.extractor.process(row or self.mapped['extractor_row'], args, self.bias,
                                      self.contract['bindings']['bias']['sha256'], m.base.value_hash(self.contract))

    def test_pins_original_module_linkage_and_feature_count(self):
        self.assertEqual(m.base.digest(OLD / 'extract_expanded_four_family.py'), m.EXTRACTOR_SHA)
        self.assertEqual(m.base.digest(OLD / 'expanded_feature_definitions.py'), m.CORE_SHA)
        self.assertEqual(m.base.digest(m.HERE / 'run_native30_inference_batches_v2.py'), m.INFERENCE_V2_SHA)
        self.assertEqual(len(m.feature_names(self.extractor)), 43)
        self.assertIs(self.extractor.read_audio, __import__('expanded_feature_definitions').read_audio)

    def test_source_group_alias_native_rate_sha_join(self):
        row = self.mapped['extractor_row']
        self.assertEqual(row['source_id'], 'FMA')
        self.assertEqual(row['native_sample_rate_hz'], '48000')
        self.assertNotEqual(int(row['native_sample_rate_hz']), sf.info(self.source).samplerate)
        for field in ('source_group', 'label', 'group_id', 'component_id', 'role'):
            changed = copy.deepcopy(self.row)
            changed[field] = 'wrong'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'identity SHA join'):
                m.mapped_rows([changed], {'rows': [self.origin]}, self.infer_freeze)
        changed = copy.deepcopy(self.origin)
        changed['source_origin']['sample_rate_hz'] = 44100
        with self.assertRaisesRegex(ValueError, 'identity SHA join'):
            m.mapped_rows([self.row], {'rows': [changed]}, self.infer_freeze)

    def test_shard_partition_and_native_evidence_fail_closed(self):
        for shards in ([], [{'index': 0, 'item_ids': ['synthetic', 'synthetic']}], [{'index': 0, 'item_ids': ['wrong']}]):
            with self.subTest(shards=shards), self.assertRaisesRegex(ValueError, 'shard/cohort'):
                m.mapped_rows([self.row], {'rows': [self.origin]}, {**self.infer_freeze, 'shards': shards})
        for rate in (None, '48000', 0, True):
            origin = copy.deepcopy(self.origin)
            origin['source_origin']['sample_rate_hz'] = rate
            row = {**self.row, 'origin_plan_row_sha256': m.base.value_hash(origin)}
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, 'native sample-rate'):
                m.mapped_rows([row], {'rows': [origin]}, self.infer_freeze)

    def test_genuine_empty_is_scientific_missing_and_commit_resumes_without_process(self):
        baseline = m.clean(self.legacy())
        result = self.run_fixture()
        self.assertEqual(result['completed'], 1)
        output = Path(self.contract['output_root'])
        raw = (output / 'items/synthetic.json').read_text()
        self.assertNotIn('NaN', raw)
        receipt = json.loads(raw)['payload']
        measured = receipt['legacy_result']
        self.assertEqual(measured, baseline)
        self.assertEqual((measured['status'], measured['beat_output_status'], measured['errors']), ('complete', 'empty', ''))
        self.assertEqual((measured['r_eligible'], measured['p_eligible'], measured['source_id'], measured['native_sample_rate_hz']), (0, 1, 'FMA', 48000))
        self.assertTrue(all(measured['r__' + key] is None for key in self.extractor.R_FEATURES))
        self.assertTrue(any(measured['p__' + key] is not None for key in self.extractor.P_FEATURES))
        before = m.products(output)
        with patch.object(self.extractor, 'process', side_effect=AssertionError('resume must not recompute')):
            resumed = self.run_fixture()
        self.assertEqual(resumed['status'], 'verified_existing_COMMIT')
        self.assertEqual(m.products(output), before)
        self.assertEqual(m.base.read_json(output / 'COMMIT.json')['products'], before)

    def test_label_source_do_not_change_fixed_correction_or_numerics(self):
        first = self.legacy()
        second = self.legacy({**self.mapped['extractor_row'], 'label': '1', 'source_id': 'ACE-Step'})
        self.assertEqual(first['feature_payload_sha256'], second['feature_payload_sha256'])
        self.assertEqual(first['bias_sha256'], second['bias_sha256'])
        self.assertEqual(first['input_hashes'], second['input_hashes'])

    def test_nonzero_stereo_harmonics_transients_nonempty_beat_parity(self):
        t = np.arange(1323000, dtype=np.float64) / 44100
        phase = np.mod(t, 0.5)
        # Repeated harmonic attacks with varying amplitude create >8 measured
        # dynamics events. Independent stereo channels stay safely below 1.
        envelope = np.exp(-phase / 0.10) * (0.4 + 0.6 * np.mod(np.floor(t * 2), 7) / 6)
        stems = {}
        for index, (stem, frequency) in enumerate(zip(m.physical.STEMS, (110, 330, 440, 220))):
            carrier = sum(np.sin(2 * np.pi * frequency * harmonic * t + index * 0.1) / harmonic
                          for harmonic in range(1, 9))
            signal = 0.08 * envelope * carrier
            if stem == 'vocals':
                signal += 0.025 * np.sin(2 * np.pi * 6100 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 6 * t))
            stereo = np.column_stack((signal, 0.85 * np.roll(signal, 13 + index))).astype(np.float32)
            path = self.infer / 'demix/htdemucs/synthetic' / (stem + '.wav')
            sf.write(path, stereo, 44100, subtype='PCM_16')
            stems[stem] = stereo
        self.samples = sum(stems.values()).astype(np.float32)
        self.assertGreater(float(np.std(self.samples)), 0.01)
        self.assertFalse(np.array_equal(self.samples[:, 0], self.samples[:, 1]))
        sf.write(self.source, self.samples, 44100, subtype='FLOAT')
        self.row['input'] = m.base.file_binding(self.source)
        self.row['waveform_float32_sha256'] = hashlib.sha256(self.samples.astype('<f4').tobytes()).hexdigest()
        times = np.arange(0.2, 29.5, 0.5)
        times += 0.018 * np.sin(np.arange(len(times)) * 0.7)
        self.beat.write_text(''.join(f'{time:.10f} {index % 4 + 1}\n' for index, time in enumerate(times)))
        self.audited = self.audit()
        baseline = m.clean(self.legacy())
        alternate = m.clean(self.legacy({**self.mapped['extractor_row'], 'label': '1', 'source_id': 'ACE-Step'}))
        self.assertEqual(baseline['status'], 'complete')
        self.assertEqual(baseline['beat_output_status'], 'nonempty')
        self.assertEqual([baseline[k] for k in ('s8_computed', 'd_eligible', 'r_eligible', 'p_eligible')], [1, 1, 1, 1])
        self.assertGreater(baseline['d__dynamics_span'], 0)
        self.assertGreater(baseline['r__tempo_tv'], 0)
        self.assertGreater(baseline['p__section_duration_cv'], 0)
        for prefix in ('s8__', 'd__', 'r__', 'p__'):
            selected = [key for key in m.feature_names(self.extractor) if key.startswith(prefix)]
            self.assertTrue(all(baseline[key] is not None and np.isfinite(baseline[key]) for key in selected), prefix)
        self.assertEqual(baseline['feature_payload_sha256'], alternate['feature_payload_sha256'])
        self.assertEqual({k: baseline[k] for k in m.feature_names(self.extractor)},
                         {k: alternate[k] for k in m.feature_names(self.extractor)})
        result = self.run_fixture()
        self.assertEqual(result['completed'], 1)
        measured = m.base.read_json(Path(self.contract['output_root']) / 'items/synthetic.json')['payload']['legacy_result']
        self.assertEqual(measured, baseline)

    def test_frozen_musdb_power_correction_sign_and_scale(self):
        core = __import__('expanded_feature_definitions')
        waveform = np.random.default_rng(20260907).normal(0, 0.1, 44100 * 2)
        raw, freq = core._active_corrected_power(waveform, np.zeros(2049))
        corrected, corrected_freq = core._active_corrected_power(waveform, np.full(2049, 10.0))
        np.testing.assert_array_equal(freq, corrected_freq)
        np.testing.assert_allclose(corrected, raw * 0.1, rtol=1e-14, atol=0)

    def test_exact_frame_gate_prevents_legacy_padding_and_clipping(self):
        for frames in (1322999, 1323001, 2646000):
            with self.subTest(frames=frames):
                sf.write(self.source, np.zeros((frames, 2), dtype=np.float32), 44100, subtype='FLOAT')
                with patch.object(self.extractor, 'process', side_effect=AssertionError('physical gate must precede process')) as call:
                    result = self.run_fixture()
                self.assertEqual(result['status'], 'partial_no_COMMIT')
                self.assertIn('exact native30', result['failures'][0]['message'])
                call.assert_not_called()
        self.assertEqual(len(list((Path(self.contract['output_root']) / 'failures').iterdir())), 3)

    def test_rate_channel_and_float_gate(self):
        for rate, channels, subtype in ((48000, 2, 'FLOAT'), (44100, 1, 'FLOAT'), (44100, 2, 'PCM_16')):
            with self.subTest(rate=rate, channels=channels, subtype=subtype):
                sf.write(self.source, self.samples[:, :channels], rate, subtype=subtype)
                with self.assertRaises(ValueError):
                    m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_nonfinite_input_fails(self):
        self.samples[3, 0] = np.nan
        sf.write(self.source, self.samples, 44100, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_changed_pcm_and_container_hashes_rejected(self):
        self.samples[3, 0] = 0.25
        sf.write(self.source, self.samples, 44100, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            m.inspect_inputs(self.contract, self.mapped, self.audited)
        self.row['input'] = m.base.file_binding(self.source)
        with self.assertRaisesRegex(ValueError, 'PCM hash changed'):
            m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_short_stem_and_bad_spec_are_not_missingness(self):
        stem = self.infer / 'demix/htdemucs/synthetic/vocals.wav'
        sf.write(stem, self.samples[:-1], 44100, subtype='PCM_16')
        with self.assertRaisesRegex(ValueError, 'exact native30'):
            m.inspect_inputs(self.contract, self.mapped, self.audited)
        sf.write(stem, self.samples, 44100, subtype='PCM_16')
        np.save(self.infer / 'spec/synthetic.npy', np.zeros((4, 2999, 81), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'spectrogram'):
            m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_missing_and_malformed_beats_fail_not_synthetic_empty(self):
        for contents in ('0 1\n0 2\n', '31 1\n', '1 1.5\n', 'nan 1\n'):
            self.beat.write_text(contents)
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                m.inspect_inputs(self.contract, self.mapped, self.audited)
        self.beat.unlink()
        with self.assertRaises(FileNotFoundError):
            m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_structure_requires_true_native30_and_input_provenance(self):
        for payload in ({'path': str(self.source), 'segments': [{'start': 0, 'end': 60}]},
                        {'path': '/wrong.wav', 'segments': [{'start': 0, 'end': 30}]},
                        {'path': str(self.source), 'segments': [{'start': 0, 'end': 10}, {'start': 11, 'end': 30}]}):
            self.structure.write_bytes(m.base.canonical(payload))
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_empty_requires_successful_ordinary_receipt_not_recovery(self):
        for change in ({'status': 'failed'}, {'recovery_applied': True},
                       {'ordinary_empty_beat_allowed_only_after_this_successful_command': False}, {'item_ids': []}):
            saved = self.payload
            self.payload = {**saved, **change}
            self.write_beat_receipt()
            audited = self.audit()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'nonrecovery inference'):
                m.inspect_inputs(self.contract, self.mapped, audited)
            self.payload = saved
        self.write_beat_receipt()
        self.audited['evidence']['stage_receipts'] = {}
        with self.assertRaisesRegex(ValueError, 'absent'):
            m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_processing_failure_retained_then_retry_without_erasing_failure(self):
        with patch.object(self.extractor, 'read_spectral_audio', side_effect=RuntimeError('synthetic numerical failure')):
            result = self.run_fixture()
        output = Path(self.contract['output_root'])
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertIn('synthetic numerical failure', result['failures'][0]['message'])
        self.assertFalse((output / 'COMMIT.json').exists())
        self.assertFalse((output / 'items/synthetic.json').exists())
        failed = m.products(output)
        self.assertEqual(self.run_fixture()['completed'], 1)
        for path, binding in failed.items():
            self.assertEqual(m.products(output)[path], binding)

    def test_input_mutation_during_process_fails_before_receipt(self):
        original = self.extractor.process
        def mutate(*args):
            result = original(*args)
            self.beat.write_text('1 1\n')
            return result
        with patch.object(self.extractor, 'process', side_effect=mutate):
            result = self.run_fixture()
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertIn('input changed during extraction', result['failures'][0]['message'])
        self.assertFalse((Path(self.contract['output_root']) / 'items/synthetic.json').exists())

    def test_runtime_or_binding_changed_rejected(self):
        self.bias_path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bound file changed'):
            self.run_fixture()
        with patch.object(m, 'runtime_snapshot', return_value={'actual': 2}), self.assertRaisesRegex(ValueError, 'CPU runtime changed'):
            m.check_contract({'bindings': {}, 'cpu_runtime': {'expected': 1}})

    def test_bias_array_must_equal_bound_file(self):
        self.bias = np.ones(2049)
        with self.assertRaisesRegex(ValueError, 'bound calibration'):
            self.run_fixture()

    def test_result_processing_infinity_feature_schema_and_hash_rejected(self):
        result = m.clean(self.legacy())
        inputs = m.inspect_inputs(self.contract, self.mapped, self.audited)
        for change in ({'errors': 'failure'}, {'s8__bogus': 1.0}, {'feature_payload_sha256': 'a' * 64},
                       {m.feature_names(self.extractor)[0]: float('inf')}, {'native_sample_rate_hz': 44100}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_result({**result, **change}, self.mapped, m.base.value_hash(self.contract),
                                  self.contract['bindings']['bias']['sha256'], inputs, self.extractor)

    def test_committed_receipt_tamper_fails_without_overwrite(self):
        self.run_fixture()
        output = Path(self.contract['output_root'])
        item = output / 'items/synthetic.json'
        item.write_bytes(item.read_bytes() + b' ')
        before = item.read_bytes()
        with self.assertRaisesRegex(ValueError, 'COMMIT changed'):
            self.run_fixture()
        self.assertEqual(item.read_bytes(), before)

    def test_uncommitted_tampered_receipt_retained_and_not_recomputed(self):
        self.run_fixture()
        output = Path(self.contract['output_root'])
        (output / 'COMMIT.json').unlink()  # Synthetic interruption reconstruction only.
        item = output / 'items/synthetic.json'
        envelope = m.base.read_json(item)
        envelope['payload']['row_sha256'] = '0' * 64
        envelope['receipt_sha256'] = m.base.value_hash(envelope['payload'])
        item.write_bytes(m.base.canonical(envelope))
        with patch.object(self.extractor, 'process', side_effect=AssertionError('do not overwrite bad receipt')):
            result = self.run_fixture()
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertIn('lineage', result['failures'][0]['message'])

    def test_changed_audit_graph_prevents_commit(self):
        changed = copy.deepcopy(self.audited)
        changed['verification']['changed'] = True
        audits = iter([self.audited, changed])
        with self.assertRaisesRegex(ValueError, 'graph changed'):
            m.run_contract(self.contract, self.extractor, self.bias, lambda _: next(audits))

    def test_separate_parent_freeze_required_before_metadata_preflight(self):
        path = self.root / 'freeze.json'
        path.write_bytes(m.base.canonical({'version': 'native30-inference-parent-freeze-v2'}))
        with self.assertRaisesRegex(ValueError, 'extraction freeze scope'), patch.object(m.physical, 'inspect_audio', side_effect=AssertionError('no audio')):
            m.prepare(path, m.base.digest(path))
        with self.assertRaisesRegex(ValueError, 'caller extraction freeze SHA'):
            m.prepare(path, '0' * 64)

    def test_positive_metadata_preflight_does_not_open_audio_or_products(self):
        plan = self.root / 'plan.json'
        plan.write_bytes(m.base.canonical({'rows': [self.origin]}))
        cohort = self.root / 'cohort.json'
        cohort.write_bytes(m.base.canonical({'rows': [self.row]}))
        parent_freeze = self.root / 'inference_freeze.json'
        parent_freeze.write_bytes(m.base.canonical(self.infer_freeze))
        completion = self.infer / 'completion.json'
        completion.write_bytes(m.base.canonical({'status': 'passed_native30_inference_not_feature_extraction', 'rows': 1}))
        paths = {'runner': m.HERE / (m.VERSION + '.py'), 'tests': Path(__file__).resolve(),
                 'inference_runner': m.HERE / 'run_native30_inference_batches_v2.py',
                 'inference_parent_freeze': parent_freeze, 'inference_completion': completion,
                 'cohort_contract': cohort, 'origin_plan': plan,
                 'extractor': OLD / 'extract_expanded_four_family.py', 'core': OLD / 'expanded_feature_definitions.py',
                 'bias': self.bias_path}
        freeze = {'version': m.FREEZE_VERSION, 'status': m.FREEZE_STATUS, 'duration_s': 30, 'cpu_only': True,
                  'feature_extraction_authorized': True, 'output_root': str(self.root / 'prepared'), 'cpu_runtime': {'synthetic': True},
                  **{key: m.base.file_binding(path) for key, path in paths.items()}, **m.SCOPE}
        path = self.root / 'extraction_freeze.json'
        path.write_bytes(m.base.canonical(freeze))
        infer_freeze = {**self.infer_freeze, 'cohort_contract': freeze['cohort_contract'], 'plan': freeze['origin_plan'],
                        'bias': freeze['bias'], 'old_code_root': str(OLD)}
        original_loader = m.load_module
        backend = SimpleNamespace(validate_freeze=lambda p, sha: (infer_freeze, {'rows': [self.row]}, None))
        def loader(path, sha, name):
            return backend if name == 'run_native30_inference_batches_v2' else original_loader(path, sha, name)
        with patch.object(m, 'load_module', side_effect=loader), patch.object(m, 'PLAN_SHA', m.base.digest(plan)), \
             patch.object(m, 'BIAS_SHA', m.base.digest(self.bias_path)), patch.object(m.physical, 'EXPECTED', {'total': 1, 'sources': {'FMA': 1}}), \
             patch.object(m.physical, 'validate_rows'), patch.object(m, 'runtime_snapshot', return_value={'synthetic': True}), \
             patch.object(m.physical, 'inspect_audio', side_effect=AssertionError('metadata must not decode audio')), \
             patch.object(m.physical, 'verify_products', side_effect=AssertionError('metadata must not read products')), \
             patch.object(self.extractor, 'process', side_effect=AssertionError('metadata must not process')), \
             patch.object(self.extractor, 'load_bias', side_effect=AssertionError('metadata does not load numerical bias')):
            contract, extractor = m.prepare(path, m.base.digest(path))
        self.assertEqual(contract['rows'], [self.mapped])
        self.assertIs(extractor, self.extractor)
        self.assertFalse(Path(contract['output_root']).exists())

    def test_completion_verifier_dispatch_binds_complete_evidence(self):
        completion = self.root / 'completion.json'
        completion.write_bytes(m.base.canonical(self.audited['evidence']))
        bindings = {'inference_parent_freeze': {'path': '/synthetic/freeze.json', 'sha256': 'b' * 64},
                    'inference_completion': m.base.file_binding(completion)}
        proof = {'status': 'verified_complete_native30_inference', 'rows': 1,
                 'completion': bindings['inference_completion'], 'evidence_sha256': m.base.value_hash(self.audited['evidence'])}
        runner = SimpleNamespace(verify_completion=lambda p, sha: proof)
        with patch.object(m, 'load_module', return_value=runner):
            audited = m.verify_completed_inference({'bindings': bindings, 'expected_count': 1})
            self.assertEqual(audited['evidence'], self.audited['evidence'])
            proof['evidence_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                m.verify_completed_inference({'bindings': bindings, 'expected_count': 1})

    def test_symlink_code_and_unknown_inventory_rejected(self):
        link = self.root / 'linked.py'
        link.symlink_to(m.HERE / 'run_native30_inference_batches_v2.py')
        with self.assertRaisesRegex(ValueError, 'noncanonical code path'):
            m.load_module(link, m.INFERENCE_V2_SHA, 'bad_alias')
        output = Path(self.contract['output_root'])
        output.mkdir()
        (output / 'unexpected').touch()
        with self.assertRaisesRegex(ValueError, 'nonempty output'):
            self.run_fixture()


if __name__ == '__main__':
    unittest.main()
