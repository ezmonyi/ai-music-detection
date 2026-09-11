"""Bounded stdlib fixtures only: no evaluator, producer, fit or real-data reads."""
import copy
import csv
import math
from pathlib import Path
import tempfile
import unittest

import check_equal60_v5_presentation as C


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def model(arm='human', combo=C.BASE, fold='f0'):
    return dict(fold_type='human_source_holdout', heldout_source=arm, combination=combo,
                feature_mode='values_plus_missing', quantity='all', fold_uid=fold)


def increment_fixture():
    pairs, sources, index = [], [], {}
    rates = [(C.BASE, .5, .5, .25, .5), (C.ADDED[0], .6, .4, .5, .25),
             (C.ADDED[1], .5, .5, .25, .5), (C.ADDED[2], .4, .3, .1, .2),
             (C.ADDED[3], .7, .8, .75, .8)]
    for combo, auc, ba, record, component in rates:
        m = model(combo=combo)
        index[combo] = m | dict(train_id_set_sha256='a'*64, test_id_set_sha256='b'*64)
        axes = {k: m[k] for k in C.AXES}
        pairs.append(axes | dict(summary_level='held_source_arm', held_source_arms=1,
            valid_folds=1, metric_cells=1, roc_auc=auc, balanced_accuracy=ba))
        sources.append(axes | dict(source_group='human', label='0', endpoint='human_specificity',
            unique_oof_recordings=4, global_components=2, correct_recordings=1,
            recording_rate=record, equal_component_rate=component,
            equal_component_minus_recording_pp=100*(component-record), valid_folds=1))
    return pairs, sources, index


class SourceReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.index = {'m0': model(fold='f0'), 'm1': model(fold='f1')}
        self.schedule = {'f0': {'test': {'ids': ['h0', 'h1', 'h2']}}, 'f1': {'test': {'ids': ['h3']}}}
        self.audit = {'streams': {'primary': {'predictions': 4}}}
        self.rows = [dict(model_uid='m0' if i < 3 else 'm1', row_id='h'+str(i), label='0',
            source_group='human', group_id='large' if i < 3 else 'small', role='development',
            score='1' if i < 3 else '0', threshold='0.5', predicted_label='1' if i < 3 else '0') for i in range(4)]

    def tearDown(self):
        self.temp.cleanup()

    def run_fixture(self):
        write_rows(self.root/'primary_predictions.csv', self.rows)
        return C.reconstruct_sources(self.root, 'primary', self.index, self.schedule, self.audit)

    def test_unequal_groups_reconstruct_recording_and_component_oof_rates(self):
        rows, count = self.run_fixture()
        self.assertEqual(count, 4)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r['unique_oof_recordings'], r['global_components'], r['valid_folds']), (4, 2, 2))
        self.assertEqual(r['correct_recordings'], 1)
        self.assertEqual(r['recording_rate'], .25)
        self.assertEqual(r['equal_component_rate'], .5)
        self.assertEqual(r['equal_component_minus_recording_pp'], 25.)

    def test_ai_endpoint_uses_positive_decisions(self):
        for r in self.rows:
            r.update(label='1', source_group='ai')
        rows, _ = self.run_fixture()
        self.assertEqual(rows[0]['endpoint'], 'ai_sensitivity')
        self.assertEqual(rows[0]['recording_rate'], .75)
        self.assertEqual(rows[0]['equal_component_rate'], .5)

    def test_duplicate_oof_recording_across_models_is_rejected(self):
        self.schedule['f1']['test']['ids'] = ['h0']
        self.rows[-1] = self.rows[0] | {'model_uid': 'm1'}
        with self.assertRaisesRegex(ValueError, 'Repeated OOF'):
            self.run_fixture()

    def test_missing_and_wrong_threshold_decisions_are_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, 'Model predictions absent'):
            self.run_fixture()
        self.rows[0]['predicted_label'] = '0'
        with self.assertRaisesRegex(ValueError, 'Invalid saved decision'):
            self.run_fixture()


class MacroReconstructionTests(unittest.TestCase):
    def test_unequal_pairs_folds_and_arms_and_separate_pooled_companion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = {uid: model(arm=arm, fold=uid) for uid, arm in
                     [('a1', 'A'), ('a2', 'A'), ('b1', 'B'), ('b2', 'B'), ('b3', 'B')]}
            pairs = []
            for uid, values in [('a1', [0., 1.]), ('a2', [1.]), ('b1', [.25]), ('b2', [.25]), ('b3', [.25])]:
                for i, value in enumerate(values):
                    pairs.append(dict(model_uid=uid, human_source=index[uid]['heldout_source'], ai_source=str(i),
                                      roc_auc=value, balanced_accuracy=value))
            write_rows(root/'primary_pair_metrics.csv', pairs)
            audit = {'streams': {'primary': {'pair_metrics': 6, 'pooled_metrics': 5}}}
            result = C.reconstruct_macros(root, 'primary', 'pair', index, audit)
            lookup = {r['heldout_source']: r for r in result}
            self.assertEqual(lookup['A']['roc_auc'], .75)
            self.assertEqual(lookup['B']['roc_auc'], .25)
            self.assertEqual(lookup['__equal_arms__']['roc_auc'], .5)
            self.assertEqual((lookup['A']['valid_folds'], lookup['A']['metric_cells']), (2, 3))
            self.assertEqual((lookup['__equal_arms__']['held_source_arms'], lookup['__equal_arms__']['valid_folds'],
                              lookup['__equal_arms__']['metric_cells']), (2, 5, 6))
            pooled = [dict(model_uid=uid, roc_auc=v, balanced_accuracy=v)
                      for uid, v in [('a1', .1), ('a2', .3), ('b1', .8), ('b2', .8), ('b3', .8)]]
            write_rows(root/'primary_pooled_metrics.csv', pooled)
            lookup = {r['heldout_source']: r for r in C.reconstruct_macros(root, 'primary', 'pooled', index, audit)}
            self.assertAlmostEqual(lookup['A']['roc_auc'], .2)
            self.assertAlmostEqual(lookup['B']['roc_auc'], .8)
            self.assertAlmostEqual(lookup['__equal_arms__']['roc_auc'], .5)
            self.assertEqual(lookup['__equal_arms__']['metric_cells'], 5)


class IncrementEncodingTests(unittest.TestCase):
    def test_positive_negative_zero_and_different_component_directions(self):
        pair, sources, index = increment_fixture()
        output, identity_hash = C.reconstruct_increments(pair, sources, index)
        self.assertEqual(len(output), 16)
        self.assertEqual(len(identity_hash), 64)
        self.assertEqual({r['direction'] for r in output}, {'positive', 'negative', 'zero'})
        f = [r for r in output if r['scope'] == 'source_endpoint' and r['added_combination'] == C.ADDED[0]]
        self.assertEqual({r['direction'] for r in f}, {'positive', 'negative'})
        self.assertTrue(all(r['paired_direction_comparison'] == 'different_direction' for r in f))
        h = [r for r in output if r['scope'] == 'source_endpoint' and r['added_combination'] == C.ADDED[1]]
        self.assertTrue(all(r['direction'] == 'zero' and r['paired_direction_comparison'] == 'same_direction' for r in h))

    def test_one_ulp_saved_encoding_preserves_sign_after_independent_comparison(self):
        pair, sources, index = increment_fixture()
        saved_pair, saved_sources = copy.deepcopy(pair), copy.deepcopy(sources)
        saved_pair[2]['roc_auc'] = math.nextafter(.5, 1.)
        saved_sources[2]['recording_rate'] = math.nextafter(.25, 0.)
        # Validate the saved tokens against independently reconstructed rates first.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_rows(root/'pair.csv', saved_pair)
            write_rows(root/'source.csv', saved_sources)
            validated_pair = C.compare_table(root/'pair.csv', pair, C.MACRO_KEYS, {})
            validated_sources = C.compare_table(root/'source.csv', sources, C.SOURCE_KEYS, {})
            out, _ = C.reconstruct_increments(pair, sources, index, validated_pair, validated_sources)
            h = [r for r in out if r['added_combination'] == C.ADDED[1]]
            auc = next(r for r in h if r['endpoint'] == 'roc_auc')
            rec = next(r for r in h if r['endpoint'] == 'human_specificity_recording_rate')
            self.assertEqual(auc['direction'], 'positive')
            self.assertEqual(rec['direction'], 'negative')
            self.assertEqual(rec['paired_direction_comparison'], 'different_direction')
            self.assertEqual(auc['added_minus_baseline_pp'], 100*(math.nextafter(.5, 1.)-.5))
            write_rows(root/'delta.csv', out)
            C.compare_table(root/'delta.csv', out, C.DELTA_KEYS, {}, exact_floats=('baseline_rate', 'added_rate', 'added_minus_baseline_pp'))
            altered = copy.deepcopy(out)
            next(r for r in altered if C.key(r, C.DELTA_KEYS) == C.key(auc, C.DELTA_KEYS))['added_minus_baseline_pp'] = 0.
            write_rows(root/'delta.csv', altered)
            with self.assertRaisesRegex(ValueError, 'Rate differs'):
                C.compare_table(root/'delta.csv', out, C.DELTA_KEYS, {}, exact_floats=('baseline_rate', 'added_rate', 'added_minus_baseline_pp'))
            altered = copy.deepcopy(out)
            next(r for r in altered if C.key(r, C.DELTA_KEYS) == C.key(auc, C.DELTA_KEYS))['direction'] = 'zero'
            write_rows(root/'delta.csv', altered)
            with self.assertRaisesRegex(ValueError, 'Field differs'):
                C.compare_table(root/'delta.csv', out, C.DELTA_KEYS, {})

    def test_materially_changed_rates_and_unmatched_training_ids_are_rejected(self):
        pair, sources, index = increment_fixture()
        saved = copy.deepcopy(pair)
        saved[2]['roc_auc'] += .01
        with self.assertRaisesRegex(ValueError, 'Serialized source rate differs'):
            C.reconstruct_increments(pair, sources, index, saved, sources)
        index[C.ADDED[0]]['train_id_set_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'training/test hashes differ'):
            C.reconstruct_increments(pair, sources, index)


class TableTamperTests(unittest.TestCase):
    def test_value_count_duplicate_omission_and_extra_column_tampering(self):
        expected = [dict(id='a', count=3, rate=.25), dict(id='b', count=1, rate=.75)]
        variants = []
        for field, value in [('rate', .35), ('count', 4)]:
            r = copy.deepcopy(expected)
            r[0][field] = value
            variants.append(r)
        variants.extend([expected[:1], [expected[0], expected[0]], [r | {'extra': 'x'} for r in expected]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'table.csv'
            write_rows(path, expected)
            self.assertEqual(C.compare_table(path, expected, ('id',), {}), expected)
            for rows in variants:
                with self.subTest(rows=rows):
                    write_rows(path, rows)
                    with self.assertRaises(ValueError):
                        C.compare_table(path, expected, ('id',), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
