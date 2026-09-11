import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import run_native30_inference_batches_v1 as r


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.canonical(value))
    return r.binding(path)


def make_audio(path, *, frames=1323000, subtype='FLOAT'):
    value = hashlib.sha256()
    block = np.zeros((65536, 2), dtype='<f4')
    remaining = frames
    with sf.SoundFile(path, 'x', samplerate=44100, channels=2, format='WAV', subtype=subtype) as stream:
        while remaining:
            current = block[:min(remaining, len(block))]
            stream.write(current)
            value.update(current.tobytes())
            remaining -= len(current)
    return value.hexdigest()


class FakeBase:
    read_json = staticmethod(r.read_json)
    value_hash = staticmethod(r.value_hash)
    file_binding = staticmethod(r.binding)


class FakeHelper:
    base = FakeBase()
    EXPECTED = {'plan': 1, 'new': 1, 'prior': 0, 'excluded': 0, 'total': 1,
                'human': 0, 'ai': 1, 'sources': {'Synthetic': 1}}

    def __init__(self, rows):
        self.rows = rows

    @staticmethod
    def add_binding(bindings, entry):
        old = bindings.get(entry['path'])
        if old is not None and old != entry:
            raise ValueError('conflicting binding')
        bindings[entry['path']] = entry

    def bind_commit(self, root, digest, kind, bindings):
        commit_entry = r.binding(root / 'COMMIT.json')
        if commit_entry['sha256'] != digest:
            raise ValueError('commit mismatch')
        self.add_binding(bindings, commit_entry)
        return r.read_json(root / 'COMMIT.json'), r.read_json(root / 'manifest.json')

    @staticmethod
    def collect_upstream(value, bindings):
        del value, bindings

    @staticmethod
    def tree_files(root, exclude=()):
        return {p.relative_to(root).as_posix() for p in root.rglob('*')
                if p.is_file() and p.relative_to(root).as_posix() not in exclude}

    @staticmethod
    def bind_receipts(root, manifest, bindings, origin):
        del root, manifest, bindings, origin

    def reconcile(self, plan, screen, new, prior, expected):
        del plan, screen, new, prior
        if expected['total'] != len(self.rows):
            raise ValueError('expected mismatch')
        return self.rows


class RunnerTests(unittest.TestCase):
    def test_exact_float_input_and_pcm_stem_loading(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            input_path, stem_path = root / 'input.wav', root / 'stem.wav'
            pcm = make_audio(input_path)
            make_audio(stem_path, subtype='PCM_16')
            expected = {'input': r.binding(input_path), 'waveform_float32_sha256': pcm}
            observed = r.inspect_audio(input_path, expected, input_audio=True)
            self.assertEqual(observed['subtype'], 'FLOAT')
            self.assertEqual(r.inspect_audio(stem_path)['subtype'], 'PCM_16')
            with self.assertRaisesRegex(ValueError, 'WAV FLOAT'):
                r.inspect_audio(stem_path, input_audio=True)

    def test_short_input_is_rejected_without_padding(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'short.wav'
            make_audio(path, frames=1322999)
            with self.assertRaisesRegex(ValueError, 'frame mismatch'):
                r.inspect_audio(path, input_audio=True)

    def test_beats_empty_is_explicit_and_nonempty_is_strict(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            empty = root / 'empty.beats'; empty.write_bytes(b'')
            self.assertEqual(r.inspect_beats(empty)['status'], 'empty_unavailable')
            valid = root / 'valid.beats'; valid.write_text('0.5\t1\n1.0\t2\n30.0\t1\n')
            self.assertEqual(r.inspect_beats(valid)['beat_count'], 3)
            invalid = root / 'invalid.beats'; invalid.write_text('0.5\t1\n30.1\t2\n')
            with self.assertRaisesRegex(ValueError, 'outside native30'):
                r.inspect_beats(invalid)

    def test_structure_requires_exact_path_contiguity_and_span(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); audio = root / 'input.wav'
            good = root / 'good.json'
            put(good, {'path': str(audio), 'segments': [{'start': 0, 'end': 10}, {'start': 10, 'end': 30}]})
            self.assertEqual(r.inspect_structure(good, audio)['segment_count'], 2)
            bad = root / 'bad.json'
            put(bad, {'path': str(audio), 'segments': [{'start': 0, 'end': 29.9}]})
            with self.assertRaisesRegex(ValueError, 'span native30'):
                r.inspect_structure(bad, audio)

    def test_spectrogram_shape_dtype_and_finiteness(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / 'spec').mkdir()
            path = root / 'spec' / 'x.npy'
            np.save(path, np.zeros((4, 3000, 81), dtype=np.float32))
            rows = [{'id': 'x', 'input': {'path': str(root / 'input.wav')}}]
            with patch.object(r, 'inspect_audio', return_value={'path': '/stem'}), \
                 patch.object(r, 'inspect_structure', return_value={'path': '/structure'}):
                result = r.verify_products('allinone', rows, root)
            self.assertEqual(result[str(path)]['shape'], [4, 3000, 81])
            np.save(path, np.zeros((4, 2999, 81), dtype=np.float32))
            with patch.object(r, 'inspect_audio', return_value={'path': '/stem'}), \
                 patch.object(r, 'inspect_structure', return_value={'path': '/structure'}), \
                 self.assertRaisesRegex(ValueError, 'spectrogram'):
                r.verify_products('allinone', rows, root)

    def test_commands_preserve_established_recipes(self):
        runtime, output = Path('/runtime'), Path('/output')
        aio = r.command('allinone', ['/a.wav'], runtime, output)
        beats = r.command('beats', ['/a.wav'], runtime, output)
        self.assertEqual(aio[aio.index('-m') + 1], 'harmonix-all')
        self.assertIn('-k', aio)
        self.assertEqual(beats[beats.index('--model') + 1], '/runtime/checkpoints/hub/checkpoints/beat_this-final0.ckpt')
        self.assertIn('--no-dbn', beats); self.assertIn('--float16', beats); self.assertIn('--skip-existing', beats)

    def test_shards_must_exactly_partition_ordered_rows(self):
        rows = [{'id': x} for x in ('a', 'b', 'c')]
        shards = [{'index': 0, 'rows': 2, 'item_ids': ['a', 'b'], 'item_ids_sha256': r.value_hash(['a', 'b'])},
                  {'index': 1, 'rows': 1, 'item_ids': ['c'], 'item_ids_sha256': r.value_hash(['c'])}]
        r.validate_shards(shards, rows)
        shards[1]['item_ids'] = ['b']
        shards[1]['item_ids_sha256'] = r.value_hash(['b'])
        with self.assertRaisesRegex(ValueError, 'omit/duplicate/reorder'):
            r.validate_shards(shards, rows)

    def build_freeze(self, root):
        audio = root / 'synthetic_1.wav'; audio.write_bytes(b'synthetic-not-opened-by-preflight')
        row = {'id': 'synthetic_1', 'source_group': 'Synthetic', 'label': '1', 'role': 'development',
               'group_id': 'g1', 'component_id': 'c1', 'origin_family': 'new',
               'origin_plan_row_sha256': '1' * 64, 'screen_row_sha256': '2' * 64,
               'component_sha256': '3' * 64, 'producer_receipt_sha256': '4' * 64,
               'input': r.binding(audio), 'waveform_float32_sha256': '5' * 64}
        plan = put(root / 'plan.json', {'rows': [{'id': 'synthetic_1'}]})
        screen = put(root / 'screen.json', {'rows': [{'id': 'synthetic_1'}]})
        original = root / 'original'; original.mkdir()
        new_root, prior_root = root / 'new', root / 'prior'; new_root.mkdir(); prior_root.mkdir()
        put(new_root / 'upstream_bindings.json', {})
        put(new_root / 'manifest.json', {'original_root': str(original), 'records': []})
        put(prior_root / 'manifest.json', {'records': []})
        new_commit = put(new_root / 'COMMIT.json', {'status': 'physical_intake_closed_with_explicit_exclusions',
                         'version': 'finalize_native30_partial_intake_v3', 'eligible': 1, 'excluded_short': 0, 'products': {}})
        prior_commit = put(prior_root / 'COMMIT.json', {'status': 'committed_prior2174_DSP_only', 'completed': 0, 'products': {}})
        helper_file = root / 'helper.py'; helper_file.write_text('# synthetic helper\n')
        declared = {x['path']: x for x in (plan, screen, new_commit, prior_commit, row['input'])}
        cohort = {'version': 'run_native30_fhsc_cohort_v1', 'status': 'frozen_before_any_measurement_audio_reads',
                  'expected_count': 1, 'rows': [row], 'bindings': declared,
                  'upstream_commits': {'new': {'path': new_commit['path'], 'sha256': new_commit['sha256']},
                                       'prior': {'path': prior_commit['path'], 'sha256': prior_commit['sha256']}},
                  'source_counts': {'Synthetic': 1}, 'human': 0, 'ai': 1,
                  'output_root': str(root / 'measurement_output'),
                  'upstream_inventories': {
                      str(new_root): sorted(FakeHelper.tree_files(new_root, ('writer.lock',))),
                      str(prior_root): sorted(FakeHelper.tree_files(prior_root, ('writer.lock',))),
                      str(original): sorted(FakeHelper.tree_files(original, ('writer.lock',))),
                  },
                  'input_format': {'format': 'WAV', 'subtype': 'FLOAT', 'sample_rate_hz': 44100, 'channels': 2, 'frames': 1323000}}
        cohort_entry = put(root / 'cohort.json', cohort)
        output = root / 'output'; output.mkdir()
        freeze = {'version': r.FREEZE_VERSION, 'status': r.FREEZE_STATUS,
                  'classifier_fits': 0, 'cohort_admitted': False, 'feature_extraction_authorized': False,
                  'duration_s': 30, 'input_format': cohort['input_format'], 'cohort_contract': cohort_entry,
                  'cohort_helper': r.binding(helper_file), 'plan': plan, 'screen': screen,
                  'upstream_commits': {'new': new_commit, 'prior': prior_commit},
                  'runner': r.binding(Path(r.__file__).resolve()),
                  'tests': r.binding(Path(__file__).resolve()), 'output_root': str(output),
                  'aio_gpus': [0], 'beat_gpus': [1],
                  'shards': [{'index': 0, 'rows': 1, 'item_ids': ['synthetic_1'],
                              'item_ids_sha256': r.value_hash(['synthetic_1'])}]}
        freeze_entry = put(root / 'freeze.json', freeze)
        expected = {'total': 1, 'new': 1, 'prior': 0, 'excluded': 0,
                    'human': 0, 'ai': 1, 'sources': {'Synthetic': 1}}
        return freeze_entry, expected, FakeHelper([row]), plan, screen

    def test_parent_freeze_replays_source_graph_without_audio_read(self):
        with tempfile.TemporaryDirectory() as name:
            entry, expected, helper, plan, screen = self.build_freeze(Path(name))
            with patch.object(r, 'sf') as forbidden_audio:
                freeze, cohort = r.validate_freeze(entry['path'], entry['sha256'], expected=expected,
                    helper_loader=lambda record: helper, plan_sha=plan['sha256'], screen_sha=screen['sha256'])
            self.assertEqual(len(cohort['rows']), 1)
            forbidden_audio.SoundFile.assert_not_called()
            self.assertEqual(freeze['status'], r.FREEZE_STATUS)

    def test_changed_plan_fails_even_when_freeze_boolean_stays_true(self):
        with tempfile.TemporaryDirectory() as name:
            entry, expected, helper, plan, screen = self.build_freeze(Path(name))
            Path(plan['path']).write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, 'size changed|hash changed'):
                r.validate_freeze(entry['path'], entry['sha256'], expected=expected,
                    helper_loader=lambda record: helper, plan_sha=plan['sha256'], screen_sha=screen['sha256'])

    def test_receipt_resume_rehashes_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'product'; path.write_bytes(b'abc')
            receipt = {'status': 'passed', 'run_contract_sha256': 'a' * 64, 'stage': 'beats',
                       'shard_index': 0, 'item_ids': ['x'], 'outputs': {str(path): r.binding(path)}}
            r.validate_receipt(receipt, 'a' * 64, 'beats', 0, ['x'])
            path.write_bytes(b'abd')
            with self.assertRaisesRegex(ValueError, 'changed'):
                r.validate_receipt(receipt, 'a' * 64, 'beats', 0, ['x'])

    def test_write_new_never_overwrites(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'x.json'; r.write_new(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                r.write_new(path, {'a': 2})
            self.assertEqual(r.read_json(path), {'a': 1})


if __name__ == '__main__':
    unittest.main()
