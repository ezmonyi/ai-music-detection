"""Synthetic-only schedule replay, numerical parity and immutable publication."""
import copy
from collections import Counter
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_native30_v1 as m
from test_plan_native30_evaluation_schedule_v1 import fixture, screen_for, bucket_group, row


def numerical_fixture():
    raw, screen = fixture(paired=True)
    origins, cohort, metadata, features = [], [], [], []
    for index, original in enumerate(raw):
        origin = {**original, 'source_origin': {'sample_rate_hz': 48000, 'channels': 2}}
        native = {**original, 'origin_plan_row_sha256': m.io.value_hash(origin),
                  'input': {'path': '/synthetic/' + original['id'] + '.wav', 'bytes': 42, 'sha256': 'a' * 64},
                  'waveform_float32_sha256': 'b' * 64}
        meta = {**original, 'duration_view_s': 30, 'native_sample_rate_hz': 48000,
                'input_sha256': 'a' * 64, 'waveform_float32_sha256': 'b' * 64}
        values = {name: (None if (index + col) % 11 == 0 else float(np.sin(index * 0.21 + col) + col / 17))
                  for col, name in enumerate(m.FEATURES)}
        values[m.FEATURES[0]] = None  # Fully unobserved column must remain present.
        values[m.FAMILIES['D'][0]] = 3.0  # Near-zero scale path.
        origins.append(origin)
        cohort.append(native)
        metadata.append(meta)
        features.append({'id': original['id'], **values})
    table = m.make_table(metadata, features, cohort, origins)
    schedule = m.planner.make_schedule(cohort, screen)
    return table, schedule, metadata, features, cohort, origins, screen


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = numerical_fixture()

    def setUp(self):
        self.table, self.full_schedule, self.metadata, self.features, self.rows, self.origins, self.screen = copy.deepcopy(self.fixture)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        # Reduced fit roster is strictly Python-test-only. Full replay tests
        # still enumerate every one of the 127 subsets and all five caps.
        schedules = [x for x in self.full_schedule['cap_schedules'] if x['quantity'] == 'all'
                     and x['status'] == 'eligible_metadata_cell'][:1]
        ids = {x['schedule_uid'] for x in schedules}
        self.schedule = copy.deepcopy(self.full_schedule)
        self.schedule['combination_schedule_cells'] = [x for x in self.schedule['combination_schedule_cells']
            if x['schedule_uid'] in ids and x['combination'] in {'S', 'D', 'S+D+R+P+F+H+SC'}]
        self.contract = {'version': m.VERSION, 'stage': m.STAGE, 'expected_count': len(self.table),
                         'output_root': str(self.root / 'output'), 'bindings': {}, 'runtime': {},
                         'table_sha256': m.io.value_hash(m.table_payload(self.table)),
                         'schedule_sha256': m.io.value_hash(self.schedule), 'accounting': m.schedule_counts(self.schedule), **m.SCOPE}
        self.freeze = self.root / 'fit_freeze.json'
        self.write_freeze()

    def write_freeze(self):
        frozen = {'version': m.FREEZE_VERSION, 'status': 'parent_frozen_for_native30_fold_only_fitting',
                  'stage': m.STAGE, 'fitting_authorized': True, 'scoring_authorized': True,
                  'contract': self.contract, 'contract_sha256': m.io.value_hash(self.contract),
                  'independent_review': {'reviewer': 'root', 'approved': True}, **m.SCOPE}
        self.freeze.write_bytes(m.io.canonical(frozen))
        self.authorization = m.authorize(self.contract, self.freeze, m.io.digest(self.freeze))

    def task(self, mode='values_plus_missing', combination='S+D+R+P+F+H+SC'):
        cell = next(c for c in self.schedule['combination_schedule_cells']
                    if c['feature_mode'] == mode and c['combination'] == combination)
        return m.resolve_task(cell, self.schedule)

    def run_fixture(self):
        return m.run_context(self.contract, self.table, self.schedule, self.authorization)

    def test_exact54_schema_and_frozen_pins(self):
        self.assertEqual(list(m.FAMILIES), ['S', 'D', 'R', 'P', 'F', 'H', 'SC'])
        self.assertEqual([len(x) for x in m.FAMILIES.values()], [15, 3, 3, 6, 15, 6, 6])
        self.assertEqual(len(m.FEATURES), 54)
        self.assertEqual(len(set(m.FEATURES)), 54)
        self.assertFalse(any(x.startswith(('M_', 's16__')) for x in m.FEATURES))
        self.assertEqual(m.io.digest(m.HERE / 'frozen_evaluate_expanded_20260905.py'), m.BASE_SHA)
        self.assertEqual(m.io.digest(m.HERE / 'plan_native30_evaluation_schedule_v1.py'), m.PLANNER_SHA)
        # Source-only config import: no inputs are opened and no fitting occurs.
        prep = m.load_module(m.HERE / 'prepare_evaluation_inputs_v4.py',
            '65793112b2be5174d9585de2285fee9ffad56e7248f86c7c0f9f50a8e05ba1de', '_native30_test_column_reference')
        for family in ('S', 'D', 'R', 'P', 'F', 'H'):
            self.assertEqual(m.FAMILIES[family], prep.COLUMNS[family])

    def test_no_v6_group_guard_or_feature_availability_filter(self):
        self.assertGreater(self.table.groupby('__group')['__label'].nunique().max(), 1)
        self.assertGreater(self.table.groupby('__group')['__source'].nunique().max(), 1)
        self.assertEqual(len(self.table), len(self.metadata))
        self.assertTrue(self.table[m.FEATURES[0]].isna().all())
        self.assertEqual(set(self.table['__id']), {r['id'] for r in self.metadata})

    def test_feature_and_identity_provenance_fail_closed(self):
        for mutation in ('extra', 'absent', 'infinite', 'bool', 'identity', 'native_rate', 'duration', 'order', 'origin'):
            metadata, features, rows, origins = copy.deepcopy((self.metadata, self.features, self.rows, self.origins))
            if mutation == 'extra': features[0]['M_fake'] = 1
            if mutation == 'absent': features[0].pop(m.FEATURES[-1])
            if mutation == 'infinite': features[0][m.FEATURES[0]] = float('inf')
            if mutation == 'bool': features[0][m.FEATURES[0]] = True
            if mutation == 'identity': metadata[0]['group_id'] = 'wrong'
            if mutation == 'native_rate': metadata[0]['native_sample_rate_hz'] = 44100
            if mutation == 'duration': metadata[0]['duration_view_s'] = 60
            if mutation == 'order': features.reverse()
            if mutation == 'origin': origins[0]['source_origin']['sample_rate_hz'] = 44100
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.make_table(metadata, features, rows, origins)

    def test_full_schedule_replay_preserves_mixed_group_purge_caps_and_modes(self):
        replayed = m.replay_schedule(self.full_schedule, self.rows, self.screen)
        self.assertEqual(replayed, self.full_schedule)
        self.assertEqual(len(replayed['combinations']), 127)
        self.assertEqual(len(replayed['cap_schedules']), 25 * 5)
        self.assertEqual(len(replayed['combination_schedule_cells']), 25 * 127 * 7)
        cells = replayed['combination_schedule_cells']
        for cap in replayed['cap_schedules']:
            matches = [x for x in cells if x['schedule_uid'] == cap['schedule_uid']]
            self.assertEqual(len(matches), 127 * (3 if cap['quantity'] == 'all' else 1))
        for fold in replayed['fold_cells']:
            self.assertFalse(set(fold['test']['groups']) & set(fold['uncapped_train']['groups']))
            self.assertFalse(set(fold['test']['components']) & set(fold['uncapped_train']['components']))
            if fold['heldout_source'] == 'H1':
                self.assertIn('A1', fold['sources_fully_removed_by_dependency_purge'])

    def test_tampered_schedule_or_protected_closure_rejected(self):
        for defect in ('train', 'mode', 'omission', 'reference'):
            schedule = copy.deepcopy(self.full_schedule)
            if defect == 'train': schedule['cap_schedules'][0]['train']['ids'].pop()
            if defect == 'mode': schedule['combination_schedule_cells'][0]['feature_mode'] = 'missingness_only'
            if defect == 'omission': schedule['cap_schedules'][0]['omission_reasons'] = ['invented']
            if defect == 'reference': schedule['fold_cells'][0]['v6_style_reference_train_before_dependency_purge']['ids'].pop()
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, 'pinned metadata replay'):
                m.replay_schedule(schedule, self.rows, self.screen)
        screen = copy.deepcopy(self.screen)
        screen['components'][0]['protected_relationships'] = ['synthetic locked exposure']
        with self.assertRaisesRegex(ValueError, 'pinned metadata replay'):
            m.replay_schedule(self.full_schedule, self.rows, screen)

    def test_11source_intended_accounting_not_historical38(self):
        rows = []
        for index, source in enumerate(m.planner.SOURCES):
            label = '0' if source in {'FMA', 'MTG-Jamendo'} or source.startswith('human_') else '1'
            for bucket in range(5):
                if source == 'human_saraga_hindustani_v1' and bucket in (0, 4):
                    continue
                for ordinal in range(3):
                    group = bucket_group(source, bucket, ordinal)
                    rows.append(row(f's{index}_{bucket}_{ordinal}', source, label, group))
        schedule = m.planner.make_schedule(rows, screen_for(rows))
        accounting = m.schedule_counts(schedule)
        self.assertEqual(sum(x['intended'] for x in accounting.values()), 53340)
        self.assertEqual(accounting['primary']['eligible'], 36830)
        self.assertEqual(accounting['diagnostic']['eligible'], 14732)
        self.assertEqual(schedule['accounting']['eligible_uncapped_fold_cells'], 58)
        self.assertEqual(len(schedule['omitted_fold_cells']), 2)

    def test_exact_core_model_and_prediction_parity_three_modes(self):
        for mode in (m.planner.PRIMARY_MODE, *m.planner.DIAGNOSTIC_MODES):
            task = self.task(mode)
            train, test = m.task_tables(self.table, task)
            baseline = m.BASE.fit_model(train, m.FEATURES, feature_mode=mode)
            receipt = m.model_receipt(self.contract, self.table, task)
            self.assertEqual(receipt['model'], baseline)
            np.testing.assert_array_equal([r['score'] for r in receipt['predictions']], m.BASE.predict(test, baseline))
            self.assertEqual(baseline['ridge'], 10)
            self.assertEqual(baseline['threshold'], 0.5)
            self.assertEqual(baseline['medians'][0], 0)
            self.assertEqual(baseline['scale'][0], 1)

    def test_train_only_transforms_test_values_cannot_change_model(self):
        task = self.task()
        original = m.model_receipt(self.contract, self.table, task)
        changed = self.table.copy()
        chosen = changed['__id'].isin(task['test_ids'])
        changed.loc[chosen, m.FEATURES] = 1e6
        second = m.model_receipt(self.contract, changed, task)
        self.assertEqual(original['model'], second['model'])
        self.assertNotEqual(original['predictions_sha256'], second['predictions_sha256'])

    def test_shared_group_weight_hierarchy_unchanged(self):
        train = self.table
        self.assertGreater(train.groupby('__group')['__label'].nunique().max(), 1)
        weights = m.BASE.sample_weights(train)
        self.assertAlmostEqual(weights.sum(), len(train))
        for label in (0, 1):
            self.assertAlmostEqual(weights[train['__label'].to_numpy() == label].sum(), len(train) / 2)
        self.assertTrue(np.isfinite(weights).all())

    def test_parent_authorization_precedes_any_fit(self):
        with patch.object(m.BASE, 'fit_model', side_effect=AssertionError('no unauthorized fitting')) as fit:
            with self.assertRaises(ValueError):
                m.run_context(self.contract, self.table, self.schedule, {})
            with self.assertRaises(ValueError):
                m.authorize(self.contract, None, None)
            fit.assert_not_called()
        self.assertFalse(Path(self.contract['output_root']).exists())
        frozen = m.io.read_json(self.freeze)
        frozen['winner_selection'] = True
        self.freeze.write_bytes(m.io.canonical(frozen))
        with self.assertRaisesRegex(ValueError, 'scope/contract'):
            m.authorize(self.contract, self.freeze, m.io.digest(self.freeze))

    def test_all_modes_publish_and_resume_without_refit(self):
        result = self.run_fixture()
        self.assertEqual(result['completed'], 9)
        output = Path(self.contract['output_root'])
        before = m.products(output)
        with patch.object(m.BASE, 'fit_model', side_effect=AssertionError('resumption cannot fit')):
            resumed = self.run_fixture()
        self.assertEqual(resumed['status'], 'verified_existing_COMMIT_no_refit')
        self.assertEqual(m.products(output), before)
        commit = m.io.read_json(output / 'COMMIT.json')
        self.assertEqual(commit['products'], before)
        self.assertFalse(commit['winner_selection'])
        for path in (output / 'models').iterdir():
            receipt = m.io.read_json(path)['payload']
            self.assertNotIn('NaN', path.read_text())
            self.assertEqual({r['id'] for r in receipt['predictions']}, set(receipt['task']['test_ids']))

    def test_processing_failure_retained_retry_resumes_good_models(self):
        original = m.BASE.fit_model
        calls = 0
        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise np.linalg.LinAlgError('synthetic interrupted solve')
            return original(*args, **kwargs)
        with patch.object(m.BASE, 'fit_model', side_effect=fail_once):
            result = self.run_fixture()
        self.assertEqual((result['completed'], result['failed']), (8, 1))
        output = Path(self.contract['output_root'])
        self.assertFalse((output / 'COMMIT.json').exists())
        failure = list((output / 'failures').iterdir())[0]
        before = failure.read_bytes()
        with patch.object(m.BASE, 'fit_model', wraps=original) as fit:
            final = self.run_fixture()
            self.assertEqual(fit.call_count, 1)
        self.assertEqual(final['completed'], 9)
        self.assertEqual(failure.read_bytes(), before)

    def test_omitted_cells_reasons_published_without_fit(self):
        rows = [r for r in self.rows if not (r['source_group'] == 'H2' and m.planner.hash_fold(r['group_id']) == 0)]
        ids = {r['id'] for r in rows}
        self.table = self.table[self.table['__id'].isin(ids)].reset_index(drop=True)
        self.schedule = m.planner.make_schedule(rows, self.screen)
        valid = next(c for c in self.schedule['cap_schedules'] if c['quantity'] == 'all' and c['status'] == 'eligible_metadata_cell')
        omitted = next(c for c in self.schedule['cap_schedules'] if c['quantity'] == 'all' and c['status'] == 'omitted')
        selected = {valid['schedule_uid'], omitted['schedule_uid']}
        self.schedule['combination_schedule_cells'] = [c for c in self.schedule['combination_schedule_cells']
            if c['schedule_uid'] in selected and c['combination'] in {'S', 'D', 'S+D+R+P+F+H+SC'}]
        self.contract.update(expected_count=len(self.table), table_sha256=m.io.value_hash(m.table_payload(self.table)),
                             schedule_sha256=m.io.value_hash(self.schedule), accounting=m.schedule_counts(self.schedule))
        self.write_freeze()
        with patch.object(m.BASE, 'fit_model', wraps=m.BASE.fit_model) as fit:
            result = self.run_fixture()
        self.assertEqual(fit.call_count, 9)
        self.assertEqual(result['omitted'], 9)
        omitted_records = m.io.read_json(Path(self.contract['output_root']) / 'omitted.json')
        self.assertTrue(all(c['schedule_uid'] == omitted['schedule_uid'] and c['omission_reasons'] == omitted['omission_reasons']
                            for c in omitted_records))

    def test_tampered_receipt_rejected_without_overwrite(self):
        self.run_fixture()
        output = Path(self.contract['output_root'])
        path = next((output / 'models').iterdir())
        path.write_bytes(path.read_bytes() + b' ')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'COMMIT changed'):
            self.run_fixture()
        self.assertEqual(path.read_bytes(), before)

    def test_uncommitted_prediction_tamper_rejected_even_when_resealed(self):
        self.run_fixture()
        output = Path(self.contract['output_root'])
        (output / 'COMMIT.json').unlink()  # Reconstruct synthetic interruption only.
        path = next((output / 'models').iterdir())
        envelope = m.io.read_json(path)
        receipt = envelope['payload']
        receipt['predictions'][0]['score'] += 0.1
        receipt['predictions_sha256'] = m.io.value_hash(receipt['predictions'])
        envelope['receipt_sha256'] = m.io.value_hash(receipt)
        path.write_bytes(m.io.canonical(envelope))
        with patch.object(m.BASE, 'fit_model', side_effect=AssertionError('must not overwrite receipt')):
            result = self.run_fixture()
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertEqual(result['failed'], 1)

    def test_midrun_table_mutation_prevents_commit(self):
        original = m.BASE.fit_model
        calls = 0
        def mutate(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 1:
                self.table.loc[0, m.FEATURES[-1]] = 123.0
            return result
        with patch.object(m.BASE, 'fit_model', side_effect=mutate), self.assertRaisesRegex(ValueError, 'in-memory table/schedule changed'):
            self.run_fixture()
        self.assertFalse((Path(self.contract['output_root']) / 'COMMIT.json').exists())

    def test_binding_runtime_and_freeze_mutations_prevent_fitting(self):
        for defect in ('binding', 'runtime', 'freeze'):
            contract = copy.deepcopy(self.contract)
            if defect == 'binding': contract['bindings']['missing'] = {'path': str(self.root / 'missing'), 'bytes': 1, 'sha256': 'a' * 64}
            if defect == 'runtime': contract['runtime'] = {'unexpected': True}
            if defect == 'freeze': self.freeze.write_bytes(self.freeze.read_bytes() + b' ')
            with self.subTest(defect=defect), patch.object(m, 'runtime_snapshot', return_value={}), \
                 patch.object(m.BASE, 'fit_model', side_effect=AssertionError('no fit')) as fit, self.assertRaises(ValueError):
                m.run_context(contract, self.table, self.schedule, self.authorization)
            fit.assert_not_called()

    def test_precise_task_row_order_and_group_leakage_gate(self):
        task = self.task()
        train, test = m.task_tables(self.table.sample(frac=1, random_state=7), task)
        self.assertEqual(train['__id'].tolist(), task['train_ids'])
        self.assertEqual(test['__id'].tolist(), task['test_ids'])
        bad = copy.deepcopy(task)
        bad['train_ids'][0] = bad['test_ids'][0]
        bad['train_id_set_sha256'] = m.planner.id_set_hash(bad['train_ids'])
        with self.assertRaisesRegex(ValueError, 'leakage'):
            m.task_tables(self.table, bad)

    def test_mode_cap_restriction_and_model_infinity(self):
        task = self.task('median_only')
        task['quantity'] = 25
        with self.assertRaisesRegex(ValueError, 'all-cap'):
            m.columns_for(task)
        task = self.task()
        train, _ = m.task_tables(self.table, task)
        model = m.BASE.fit_model(train, m.FEATURES)
        model['coefficients_with_intercept'][0] = float('inf')
        with self.assertRaisesRegex(ValueError, 'model parameter'):
            m.validate_model(model, train, task)

    def test_train_transform_and_coefficient_replay_without_solver(self):
        for mode in (m.planner.PRIMARY_MODE, *m.planner.DIAGNOSTIC_MODES):
            task = self.task(mode)
            train, _ = m.task_tables(self.table, task)
            model = m.BASE.fit_model(train, m.FEATURES, feature_mode=mode)
            with patch.object(np.linalg, 'solve', side_effect=AssertionError('validation must not fit')):
                proof = m.validate_model(model, train, task)
            self.assertLessEqual(proof['normal_equation_residual_infinity'], proof['roundoff_tolerance'])
            for defect in ('medians', 'mean', 'scale', 'coefficients_with_intercept'):
                broken = copy.deepcopy(model)
                broken[defect][0] += 0.1
                with self.subTest(mode=mode, defect=defect), self.assertRaises(ValueError):
                    m.validate_model(broken, train, task)

    def test_equation_audit_allmissing_collinear_and_large_offset_designs(self):
        n = len(self.table)
        x = np.linspace(-1.0, 1.0, n)
        for condition in ('all_missing', 'perfect_collinear', 'large_offset_near_collinear'):
            table = self.table.copy()
            for index, column in enumerate(m.FEATURES):
                table[column] = (np.full(n, np.nan) if condition == 'all_missing' else
                                 x * (index + 1) + index if condition == 'perfect_collinear' else
                                 1e12 + (index + 1) * 0.001 * x + 0.0001 * np.sin(np.arange(n)))
            for mode in (m.planner.PRIMARY_MODE, *m.planner.DIAGNOSTIC_MODES):
                task = self.task(mode)
                train, _ = m.task_tables(table, task)
                model = m.BASE.fit_model(train, m.FEATURES, feature_mode=mode)
                with self.subTest(condition=condition, mode=mode), patch.object(np.linalg, 'solve', side_effect=AssertionError('no refit')):
                    proof = m.validate_model(model, train, task)
                    self.assertLessEqual(proof['normal_equation_residual_infinity'], proof['roundoff_tolerance'])

    def test_resealed_coefficient_and_consistent_predictions_still_fail_equations(self):
        task = self.task()
        receipt = m.model_receipt(self.contract, self.table, task)
        receipt['model']['coefficients_with_intercept'][0] += 0.1
        receipt['model_sha256'] = m.io.value_hash(receipt['model'])
        _, test = m.task_tables(self.table, task)
        receipt['predictions'] = m.prediction_evidence(test, receipt['model'])
        receipt['predictions_sha256'] = m.io.value_hash(receipt['predictions'])
        receipt['metrics'] = m.metric_evidence(receipt['predictions'], task)
        path = self.root / 'resealed.json'
        m.io.write_new(path, {'payload': receipt, 'receipt_sha256': m.io.value_hash(receipt)})
        with patch.object(np.linalg, 'solve', side_effect=AssertionError('validation cannot refit')), \
             self.assertRaisesRegex(ValueError, 'normal-equation residual'):
            m.verify_model_receipt(path, self.contract, self.table, task)

    def test_missing_assembler_pin_cannot_authorize_production(self):
        with patch.object(m, 'PREPARER_SHA', None), patch.object(m.BASE, 'fit_model', side_effect=AssertionError('no fit')):
            with self.assertRaisesRegex(ValueError, 'approval pin'):
                m.build_context('/synthetic/package', 'a' * 64, '/synthetic/out')

    def test_positive_context_preflight_replays_package_schedule_without_fit(self):
        code = self.root / 'code'
        code.mkdir()
        for name in (m.VERSION + '.py', 'test_' + m.VERSION + '.py', 'frozen_evaluate_expanded_20260905.py',
                     'plan_native30_evaluation_schedule_v1.py', 'materialize_native30_new1695_v1.py'):
            shutil.copyfile(m.HERE / name, code / name)
        assembler_path = code / (m.PREPARER_NAME + '.py')
        assembler_path.write_text('# synthetic interface stand-in; no data loaders\n')
        refs = {}
        for name, value in (('cohort_contract', {'rows': self.rows}), ('origin_plan', {'rows': self.origins}), ('screen', self.screen)):
            path = self.root / (name + '.json')
            m.io.write_new(path, value)
            refs[name] = m.io.file_binding(path)
        schedule = copy.deepcopy(self.full_schedule)
        schedule['prepared_contract_sha256'] = refs['cohort_contract']['sha256']
        schedule['input_bindings'] = {}
        schedule_root = self.root / 'schedule'
        schedule_root.mkdir()
        m.io.write_new(schedule_root / 'schedule_draft.json', schedule)
        refs['schedule_draft'] = m.io.file_binding(schedule_root / 'schedule_draft.json')
        m.io.write_new(schedule_root / 'COMMIT.json', {'status': 'committed_metadata_draft_not_evaluation_authorization',
            'products': {'schedule_draft.json': refs['schedule_draft']}, 'draft_sha256': refs['schedule_draft']['sha256']})
        refs['schedule_commit'] = m.io.file_binding(schedule_root / 'COMMIT.json')
        assembled = {'status': 'assembled_not_evaluated_not_authorized', 'expected_count': len(self.table),
                     'family_config': m.FAMILIES, 'feature_names': m.FEATURES, 'bindings': refs}
        package = self.root / 'package'
        package.mkdir()
        for name, value in (('contract', assembled), ('metadata', self.metadata), ('features', self.features), ('lineage', [])):
            m.io.write_new(package / (name + '.json'), value)
        m.io.write_new(package / 'COMMIT.json', {'synthetic': True})
        bundle = {'contract': assembled, 'metadata': self.metadata, 'features': self.features, 'lineage': [],
                  'commit': m.io.file_binding(package / 'COMMIT.json')}
        backend = SimpleNamespace(verify_package=lambda p, sha: bundle)
        accounting = m.schedule_counts(schedule)
        expected = {'total': len(self.table), 'labels': dict(Counter(self.table['__label'])),
                    'intended': sum(a['intended'] for a in accounting.values()),
                    'primary': accounting['primary']['eligible'], 'diagnostic': accounting['diagnostic']['eligible']}
        with patch.object(m, 'HERE', code), patch.object(m, 'PREPARER_SHA', m.io.digest(assembler_path)), \
             patch.object(m, 'EXPECTED', expected), patch.object(m, 'SCHEDULE_SHA', refs['schedule_draft']['sha256']), \
             patch.object(m, 'SCHEDULE_COMMIT_SHA', refs['schedule_commit']['sha256']), \
             patch.object(m.planner, 'PLAN_SHA', refs['origin_plan']['sha256']), patch.object(m.planner, 'SCREEN_SHA', refs['screen']['sha256']), \
             patch.object(m.planner, 'SOURCES', dict(Counter(self.table['__source']))), \
             patch.object(m.planner, 'validate_prepared_contract', return_value=self.rows), \
             patch.object(m, 'load_module', return_value=backend), patch.object(m, 'runtime_snapshot', return_value={}), \
             patch.object(m.BASE, 'fit_model', side_effect=AssertionError('preflight cannot fit')) as fit, \
             patch.object(m.BASE, 'predict', side_effect=AssertionError('preflight cannot predict')):
            contract, table, replay = m.build_context(package, bundle['commit']['sha256'], self.root / 'planned_output')
        self.assertEqual(m.table_payload(table), m.table_payload(self.table))
        self.assertEqual(replay, schedule)
        self.assertEqual(contract['accounting'], accounting)
        self.assertFalse(Path(contract['output_root']).exists())
        fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
