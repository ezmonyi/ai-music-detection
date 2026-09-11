"""Synthetic reporting fixtures only: no feature values, fitted models or fits."""
import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_native30_evaluation_v2 as m
import summarize_native30_evaluation_v1 as old


def pred(ident, label, source, group, is_correct, score=None):
    outcome = label if is_correct else 1 - label
    return {'id': ident, 'label': label, 'source_group': source, 'group_id': group,
            'component_id': 'component_' + group, 'role': 'development',
            'score': (0.75 if outcome else 0.25) if score is None else score,
            'threshold': 0.5, 'predicted_label': outcome}


def write(path, value):
    path.write_bytes(m.canonical(value))


def synthetic_publication(directory):
    root = directory / 'evaluation'
    root.mkdir()
    for name in ('models', 'runs', 'failures'):
        (root / name).mkdir()
    (root / 'writer.lock').touch()
    rows = []
    source_labels = {'H1': 0, 'H2': 0, 'A1': 1, 'A2': 1}
    for source, label in source_labels.items():
        for bucket, count in ((0, 3), (1, 1)):
            for n in range(count):
                rows.append({'id': f'{source}_{bucket}_{n}', 'source_group': source, 'label': label,
                             'bucket': bucket, 'group_id': f'{source}_g{bucket}'})
    expected = {'rows': len(rows), 'sources': dict(Counter(r['source_group'] for r in rows)),
                'labels': dict(Counter(r['label'] for r in rows)), 'buckets': 2, 'caps': (25, 'all'),
                'combinations': ['S', m.REFERENCE]}
    tasks = []
    held_units = [(m.PROTOCOLS[label], source) for source, label in source_labels.items()] + [(m.PROTOCOLS[2], '__all_sources__')]
    lookup = {r['id']: r for r in rows}
    for protocol, source in held_units:
        for cap in expected['caps']:
            for bucket in range(2):
                if protocol == m.PROTOCOLS[2]:
                    test = [r for r in rows if r['bucket'] == bucket]
                    train = [r for r in rows if r['bucket'] != bucket]
                else:
                    label = source_labels[source]
                    test = [r for r in rows if r['bucket'] == bucket and (r['source_group'] == source or r['label'] != label)]
                    train = [r for r in rows if r['bucket'] != bucket and r['source_group'] != source]
                identity = {'fold_type': protocol, 'heldout_source': source, 'opposite_group_fold': bucket}
                omitted = source == 'H2' and bucket == 1
                reference = {**identity, 'quantity': cap, 'fold_uid': m.value_hash(identity)[:24],
                             'train_ids': sorted(r['id'] for r in train), 'test_ids': sorted(r['id'] for r in test),
                             'train_id_set_sha256': m.id_hash(r['id'] for r in train),
                             'test_id_set_sha256': m.id_hash(r['id'] for r in test),
                             'status': 'omitted' if omitted else 'eligible_metadata_cell',
                             'omission_reasons': ['synthetic_predeclared_coverage_omission'] if omitted else []}
                reference['schedule_uid'] = m.value_hash(reference)[:24]
                for combination in expected['combinations']:
                    for mode in (m.MODES if cap == 'all' else m.MODES[:1]):
                        tasks.append({**reference, 'combination': combination, 'feature_mode': mode,
                                      'analysis_role': 'primary' if mode == m.MODES[0] else 'diagnostic'})
    valid = [t for t in tasks if t['status'] == 'eligible_metadata_cell']
    omissions = [t for t in tasks if t['status'] == 'omitted']
    accounting = {}
    for role in ('primary', 'diagnostic'):
        intended = [t for t in tasks if t['analysis_role'] == role]
        chosen = [t for t in valid if t['analysis_role'] == role]
        accounting[role] = {'intended': len(intended), 'eligible': len(chosen), 'omitted': len(intended) - len(chosen),
                            'prediction_rows': sum(len(t['test_ids']) for t in chosen)}
    expected.update(intended=len(tasks), eligible=len(valid), primary=accounting['primary']['eligible'],
                    diagnostic=accounting['diagnostic']['eligible'])
    code_bindings = {name: m.binding(m.HERE / name) for name in ('evaluate_native30_v2.py', 'test_evaluate_native30_v2.py')}
    contract = {'version': 'evaluate_native30_v2', 'stage': 'native30_seven_family_schedule_only_development_evaluation_v2',
                'output_root': str(root), 'expected_count': len(rows), 'bindings': code_bindings,
                'accounting': accounting, 'package_commit': {'path': '/synthetic/not-read/package/COMMIT.json', 'sha256': 'a' * 64},
                'schedule_document': {'path': '/synthetic/not-read/schedule.json', 'sha256': 'b' * 64},
                'schedule_sha256': 'c' * 64, **m.PRODUCER_SCOPE}
    contract_sha = m.value_hash(contract)
    write(root / 'contract.json', contract)
    freeze_path = directory / 'frozen.json'
    write(freeze_path, {'version': 'native30-evaluation-parent-freeze-v2', 'status': 'parent_frozen_for_native30_fold_only_fitting',
                       'stage': contract['stage'],
                       'contract': contract, 'contract_sha256': contract_sha, 'fitting_authorized': True, 'scoring_authorized': True,
                       'independent_review': {'reviewer': 'root', 'approved': True}, **m.PRODUCER_SCOPE})
    write(root / 'authorization.json', {'status': 'parent_frozen_verified', 'contract_sha256': contract_sha,
                                       'binding': m.binding(freeze_path)})
    models = {}
    for task in valid:
        predictions = []
        for ident in task['test_ids']:
            row = lookup[ident]
            correct = task['combination'] == m.REFERENCE or (task['quantity'] == 25 and row['bucket'] == 0)
            predictions.append(pred(ident, row['label'], row['source_group'], row['group_id'], correct))
        # Explicit synthetic metadata stand-in: no numerical fit is fabricated
        # or executed by this test. Reporting uses only bound prediction data.
        model = {'threshold': 0.5, 'ridge': 10.0, 'feature_mode': task['feature_mode'], 'training_rows': len(task['train_ids'])}
        payload = {'status': 'fold_only_model_and_predictions_not_selected', 'contract_sha256': contract_sha,
                   'task': task, 'task_sha256': m.value_hash(task), 'model': model, 'model_sha256': m.value_hash(model),
                   'predictions': predictions, 'predictions_sha256': m.value_hash(predictions),
                   'metrics': {'within_fold_pooled_descriptive': {'roc_auc': m.auc(predictions)}},
                   'training_numerical_audit': {'status': 'training_transforms_and_ridge_equations_verified_without_refit',
                       'normal_equation_residual_infinity': 0.0, 'roundoff_tolerance': 1e-12}, **m.PRODUCER_SCOPE}
        uid = m.model_uid(task, contract_sha)
        path = root / 'models' / (uid + '.json')
        write(path, {'payload': payload, 'receipt_sha256': m.value_hash(payload)})
        models[uid] = m.binding(path)
    write(root / 'omitted.json', omissions)
    write(root / 'manifest.json', {'version': 'evaluate_native30_v2', 'contract_sha256': contract_sha, 'model_instances': len(valid),
        'omitted_instances': len(omissions), 'accounting': accounting,
        'training_transform_equation_prediction_replay_performed': True,
        'independent_full_publication_numerical_audit_performed': False, 'model_receipts': models, **m.PRODUCER_SCOPE})
    write(root / 'runs' / ('0' * 32 + '.json'), {'synthetic_fixture': True, 'completed': len(valid)})
    write(root / 'COMMIT.json', {'version': 'evaluate_native30_v2', 'status': 'committed_fold_only_evaluation_not_model_selection',
        'contract_sha256': contract_sha, 'model_instances': len(valid), 'products': {},
        'all_input_output_runtime_bindings_end_rehashed': True, **m.PRODUCER_SCOPE})
    reseal(root)
    return root, expected


def reseal(root):
    manifest = m.read(root / 'manifest.json')
    manifest['model_receipts'] = {p.stem: m.binding(p) for p in (root / 'models').iterdir()}
    write(root / 'manifest.json', manifest)
    commit = m.read(root / 'COMMIT.json')
    commit['products'] = m.evaluation_inventory(root)
    write(root / 'COMMIT.json', commit)


class AggregationTests(unittest.TestCase):
    def test_full127_fivecap_and_allcap_diagnostic_output_without_topn_filter(self):
        cap_records, pools = {}, {}
        for protocol, source in ((m.PROTOCOLS[0], 'H'), (m.PROTOCOLS[1], 'A'), (m.PROTOCOLS[2], '__all_sources__')):
            for cap in m.CAPS:
                references = []
                for bucket in range(5):
                    uid = m.value_hash([protocol, cap, bucket])[:24]
                    cap_records[uid] = {'opposite_group_fold': bucket, 'train_ids': [], 'status': 'omitted',
                                        'omission_reasons': ['synthetic_no_eligible_test_predictions']}
                    references.append({'entry': None, 'schedule_uid': uid})
                for combination in m.COMBINATIONS:
                    for mode in (m.MODES if cap == 'all' else m.MODES[:1]):
                        pools[(protocol, source, cap, combination, mode)] = references
        published = {'fold_rows': [], 'pools': pools, 'cap_records': cap_records,
                     'expected': m.EXPECTED, 'contract': {'accounting': {}}}
        result = m.summarize(published)
        self.assertEqual(len(m.COMBINATIONS), 127)
        self.assertEqual(len(result['primary_protocol_cells']), 3 * 127 * 5)
        self.assertEqual(len(result['diagnostic_protocol_cells']), 3 * 127 * 2)
        self.assertTrue(all(r['macro_equal_group_equal_source_BA'] is None for r in result['primary_protocol_cells']))
        self.assertTrue(all(r['omitted_model_count'] == 5 for r in result['primary_protocol_cells']))
        self.assertEqual(len(result['comparisons']['adjacent_caps']), 3 * 127 * 4)
        self.assertEqual(len(result['comparisons']['diagnostic_vs_values_plus_missing']), 3 * 127 * 2)

    def test_unequal_group_sizes_and_source_group_counts_have_distinct_estimands(self):
        rows = [pred('h' + str(n), 0, 'H1', 'large', True) for n in range(9)]
        rows += [pred('small', 0, 'H1', 'small', False), pred('other', 0, 'H2', 'solo', True), pred('ai', 1, 'A', 'ai', False)]
        result = m.pool_predictions(rows)
        source = {r['source_group']: r for r in result['per_source']}
        self.assertEqual(source['H1']['equal_group_recall'], 0.5)
        self.assertEqual(source['H1']['recording_weighted_recall'], 0.9)
        self.assertEqual(result['equal_group_equal_source_human_recall'], 0.75)
        self.assertNotEqual(result['equal_group_equal_source_human_recall'], 2 / 3)
        self.assertEqual(result['equal_group_equal_source_BA'], 0.375)
        self.assertAlmostEqual(result['recording_weighted_within_held_source_pool']['balanced_accuracy'], 5 / 11)

    def test_duplicate_test_ids_within_pool_rejected(self):
        a = pred('shared', 0, 'H', 'g', True)
        with self.assertRaisesRegex(ValueError, 'repeated test ID'):
            m.pool_predictions([a, a])

    def test_source_conditioned_groups_can_share_global_names_across_labels(self):
        result = m.pool_predictions([pred('h', 0, 'H', 'shared', True), pred('a', 1, 'A', 'shared', False)])
        self.assertEqual(result['test_groups'], 1)
        self.assertEqual([x['groups'] for x in result['per_source']], [1, 1])
        self.assertEqual(result['equal_group_equal_source_BA'], 0.5)

    def test_missing_pool_unavailable_not_chance(self):
        result = m.pool_predictions([])
        self.assertEqual(result['status'], 'no_eligible_bucket_predictions')
        self.assertIsNone(result['equal_group_equal_source_BA'])

    def test_auc_ties_and_separately_shifted_model_scores(self):
        first = [pred('h0', 0, 'H', 'h0', True, -0.2), pred('a0', 1, 'A', 'a0', False, -0.1)]
        second = [pred('h1', 0, 'H', 'h1', False, 3.1), pred('a1', 1, 'A', 'a1', True, 3.2)]
        self.assertEqual(m.auc(first), 1)
        self.assertEqual(m.auc(second), 1)
        self.assertEqual(m.auc(first + second), 0.75)  # Shows why this is NOT a reporting headline.
        tied = [pred('h', 0, 'H', 'h', False, 0.5), pred('a', 1, 'A', 'a', True, 0.5)]
        self.assertEqual(m.auc(tied), 0.5)

    def test_negative_reference_delta_and_coverage_mismatch(self):
        row = {'fold_type': m.PROTOCOLS[0], 'quantity': 25, 'combination': 'S', 'feature_mode': m.MODES[0],
               'status': 'available', 'coverage_key': 'a', 'macro_equal_group_equal_source_BA': 0.3,
               'macro_recording_weighted_within_experiment_BA': 0.4}
        reference = {**row, 'combination': m.REFERENCE, 'macro_equal_group_equal_source_BA': 0.6,
                     'macro_recording_weighted_within_experiment_BA': 0.7}
        self.assertAlmostEqual(m.compare(row, reference, 'fixed')['delta_macro_equal_group_equal_source_BA'], -0.3)
        reference['coverage_key'] = 'different'
        result = m.compare(row, reference, 'fixed')
        self.assertEqual(result['status'], 'coverage_mismatch')
        self.assertIsNone(result['delta_macro_equal_group_equal_source_BA'])


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.root, self.expected = synthetic_publication(self.directory)

    def inspect(self):
        return m.inspect_evaluation(self.root, m.binding(self.root / 'COMMIT.json')['sha256'], self.expected)


    def test_v2_report_numerical_and_english_markdown_parity_with_v1(self):
        self.assertEqual(m.binding(Path(old.__file__))['sha256'],
                         'ccf7ae2b714eb576ee46edeb7569350535c2b80ab2fa8dbd8845a19f85d2c6bb')
        published = self.inspect()
        previous = old.summarize(published)
        current = m.summarize(published)
        self.assertEqual(previous.pop('version'), old.VERSION)
        self.assertEqual(current.pop('version'), m.VERSION)
        self.assertEqual(current, previous)
        self.assertEqual(m.markdown_products(current, published),
                         old.markdown_products(previous, published))
        self.assertEqual((m.EXPECTED, m.SPEC_SHA, m.PRODUCER_SCOPE),
                         (old.EXPECTED, old.SPEC_SHA, old.PRODUCER_SCOPE))
        with self.assertRaisesRegex(ValueError, 'complete pinned evaluator'):
            old.inspect_evaluation(self.root, m.binding(self.root / 'COMMIT.json')['sha256'], self.expected)

    def test_v2_exact_evaluator_code_and_test_pins_required(self):
        self.assertEqual(m.binding(m.HERE / 'evaluate_native30_v2.py')['sha256'], m.EVALUATOR_SHA)
        self.assertEqual(m.binding(m.HERE / 'test_evaluate_native30_v2.py')['sha256'], m.EVALUATOR_TEST_SHA)
        for key in ('EVALUATOR_SHA', 'EVALUATOR_TEST_SHA'):
            with patch.object(m, key, '0' * 64):
                with self.assertRaisesRegex(ValueError, 'evaluator code lineage'):
                    self.inspect()

    def test_complete_input_pooling_repeated_opposite_ids_allowed_between_held_sources(self):
        published = self.inspect()
        result = m.summarize(published)
        pools = [r for r in result['held_source_cells'] if r['fold_type'] == m.PROTOCOLS[0]
                 and r['quantity'] == 25 and r['combination'] == 'S' and r['feature_mode'] == m.MODES[0]]
        self.assertEqual(len(pools), 2)
        self.assertEqual({r['heldout_source'] for r in pools}, {'H1', 'H2'})
        # H1/H2 both include A1/A2; that repetition is allowed between pools.
        self.assertTrue(all({'A1', 'A2'} <= {s['source_group'] for s in r['per_source']} for r in pools))
        macro = next(r for r in result['primary_protocol_cells'] if r['fold_type'] == m.PROTOCOLS[0]
                     and r['quantity'] == 25 and r['combination'] == 'S')
        self.assertEqual(macro['macro_equal_group_equal_source_BA'], 0.75)
        self.assertEqual(macro['available_held_source_experiments'], 2)
        self.assertEqual(macro['omitted_model_count'], 1)
        self.assertEqual(macro['prediction_occurrences_not_independent_rows'], sum(r['test_rows'] for r in pools))
        weighted = sum(r['test_rows'] * r['equal_group_equal_source_BA'] for r in pools) / sum(r['test_rows'] for r in pools)
        self.assertNotEqual(macro['macro_equal_group_equal_source_BA'], weighted)

    def test_all_cells_modes_caps_protocols_and_negative_deltas_retained(self):
        result = m.summarize(self.inspect())
        self.assertEqual(len(result['primary_protocol_cells']), 3 * 2 * 2)
        self.assertEqual(len(result['diagnostic_protocol_cells']), 3 * 2 * 2)
        self.assertEqual(len(result['held_source_cells']), 5 * 2 * 4)
        self.assertEqual({r['fold_type'] for r in result['primary_protocol_cells']}, set(m.PROTOCOLS))
        self.assertTrue(all(r['quantity'] == 'all' for r in result['diagnostic_protocol_cells']))
        adjacent = [r for r in result['comparisons']['adjacent_caps'] if r['combination'] == 'S']
        self.assertTrue(all(r['status'] == 'matched' and r['delta_macro_equal_group_equal_source_BA'] < 0 for r in adjacent))
        reference = [r for r in result['comparisons']['fixed_S_D_R_P_reference'] if r['combination'] == 'S']
        self.assertTrue(all(r['delta_macro_equal_group_equal_source_BA'] < 0 for r in reference))
        self.assertFalse(result['cross_model_pooled_auc_computed'])
        self.assertFalse(result['confidence_intervals_claimed'])
        self.assertFalse(result['winner_selection'])

    def test_json_english_markdown_complete_publication_and_no_overwrite(self):
        published = self.inspect()
        output = self.directory / 'report'
        refs = {'specification': m.binding(m.SPEC, m.SPEC_SHA), 'reporter': m.binding(Path(m.__file__).resolve())}
        result = m.publish(published, output, refs)
        self.assertEqual(result['classifier_fits'], 0)
        commit = m.read(output / 'COMMIT.json')
        self.assertTrue(commit['all_evaluation_products_end_rehashed'])
        self.assertEqual(set(commit['products']), {'summary.json', 'per_fold_auc.json', 'omitted_cells.json', 'coverage.json',
                                                 'tables.md', 'held_source_tables.md', 'comparisons.md', 'per_fold_auc.md'})
        self.assertEqual(m.read(output / 'omitted_cells.json'), published['omissions'])
        tables = (output / 'tables.md').read_text()
        self.assertIn('No confidence interval', tables)
        self.assertIn('ordinary_group_holdout_descriptive', tables)
        self.assertIn('S+D+R+P', tables)
        self.assertEqual(len(m.read(output / 'per_fold_auc.json')), self.expected['eligible'])
        before = m.binding(output / 'COMMIT.json')
        with self.assertRaisesRegex(ValueError, 'new nonoverlapping'):
            m.publish(published, output, refs)
        self.assertEqual(m.binding(output / 'COMMIT.json'), before)

    def test_missing_commit_wrong_pin_and_partial_status_rejected(self):
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            m.inspect_evaluation(self.root, '0' * 64, self.expected)
        commit = m.read(self.root / 'COMMIT.json')
        commit['status'] = 'partial_no_COMMIT'
        write(self.root / 'COMMIT.json', commit)
        with self.assertRaisesRegex(ValueError, 'complete pinned'):
            self.inspect()
        (self.root / 'COMMIT.json').unlink()
        with self.assertRaisesRegex(ValueError, 'regular bound'):
            self.inspect()

    def test_unbound_or_tampered_products_rejected(self):
        path = next((self.root / 'models').iterdir())
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'inventory/hash'):
            self.inspect()
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'noncanonical'):
            self.inspect()

    def test_resealed_prediction_and_model_evidence_hash_errors(self):
        path = next((self.root / 'models').iterdir())
        envelope = m.read(path)
        envelope['payload']['predictions'][0]['score'] = 0.123
        envelope['receipt_sha256'] = m.value_hash(envelope['payload'])
        write(path, envelope)
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'model/prediction evidence'):
            self.inspect()

    def test_resealed_threshold_or_single_fold_auc_change_rejected(self):
        path = next((self.root / 'models').iterdir())
        original = m.read(path)
        for defect in ('threshold', 'auc'):
            envelope = copy.deepcopy(original)
            record = envelope['payload']
            if defect == 'threshold':
                record['predictions'][0]['threshold'] = 0.6
                record['predictions_sha256'] = m.value_hash(record['predictions'])
            else:
                record['metrics']['within_fold_pooled_descriptive']['roc_auc'] = 0.123456
            envelope['receipt_sha256'] = m.value_hash(record)
            write(path, envelope)
            reseal(self.root)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.inspect()

    def test_source_identity_drift_across_models_rejected(self):
        path = next((self.root / 'models').iterdir())
        envelope = m.read(path)
        record = envelope['payload']
        record['predictions'][0]['component_id'] = 'different_component'
        record['predictions_sha256'] = m.value_hash(record['predictions'])
        envelope['receipt_sha256'] = m.value_hash(record)
        write(path, envelope)
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'identity changes'):
            self.inspect()

    def test_omission_reasons_and_complete_catalogue_required(self):
        omissions = m.read(self.root / 'omitted.json')
        omissions[0]['omission_reasons'] = []
        write(self.root / 'omitted.json', omissions)
        reseal(self.root)
        with self.assertRaisesRegex(ValueError, 'omission reasons'):
            self.inspect()

    def test_changed_parent_authority_rejected(self):
        freeze = self.directory / 'frozen.json'
        freeze.write_bytes(freeze.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'authorization evidence changed'):
            self.inspect()

    def test_input_change_after_inspection_prevents_report_commit(self):
        published = self.inspect()
        path = next((self.root / 'models').iterdir())
        path.write_bytes(path.read_bytes() + b' ')
        output = self.directory / 'report'
        with self.assertRaises(ValueError):
            m.publish(published, output, {})
        self.assertFalse((output / 'COMMIT.json').exists())

    def test_symlink_inventory_and_overlapping_output_rejected(self):
        published = self.inspect()
        with self.assertRaisesRegex(ValueError, 'new nonoverlapping'):
            m.publish(published, self.root / 'report', {})
        extra = self.root / 'runs' / ('f' * 32 + '.json')
        extra.symlink_to(self.root / 'contract.json')
        with self.assertRaisesRegex(ValueError, 'unredirected'):
            self.inspect()


if __name__ == '__main__':
    unittest.main()
