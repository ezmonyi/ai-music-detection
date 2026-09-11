"""Synthetic-only tests for the Native30 BC evaluator adapter.

These tests use small metadata fixtures.  They do not open audio, inspect a
live BC measurement root, or authorize a production fit.
"""
import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_native30_bc_v1 as m
import evaluate_native30_v3 as old
from test_plan_native30_evaluation_schedule_v1 import fixture


def numerical_fixture():
    rows, screen = fixture(paired=True)
    origins, cohort, metadata, features = [], [], [], []
    for index, original in enumerate(rows):
        origin = {**original, 'source_origin': {'sample_rate_hz': 48000, 'channels': 2}}
        native = {**original,
                  'origin_plan_row_sha256': m.io.value_hash(origin),
                  'input': {'path': '/synthetic/' + original['id'] + '.wav',
                            'bytes': 42, 'sha256': 'a' * 64},
                  'waveform_float32_sha256': 'b' * 64}
        meta = {**original, 'duration_view_s': 30,
                'native_sample_rate_hz': 48000,
                'input_sha256': 'a' * 64,
                'waveform_float32_sha256': 'b' * 64}
        values = {
            name: (None if (index + column) % 13 == 0
                   else float(np.sin(index * 0.17 + column) + column / 19))
            for column, name in enumerate(m.FEATURES)
        }
        # One old column is entirely missing, and BC has a mixture of finite
        # and missing rows.  Both are legitimate scientific missingness.
        values[m.FAMILIES['S'][0]] = None
        values[m.BC_FEATURE] = None if index % 7 == 0 else float(index / 500)
        origins.append(origin)
        cohort.append(native)
        metadata.append(meta)
        features.append({'id': original['id'], **values})
    table = m.make_table(metadata, features, cohort, origins)
    old_schedule = m.planner.make_schedule(cohort, screen)
    schedule = m.schedule_adapter.derive_schedule(old_schedule, old_schedule,
                                                   m.OLD_FAMILY_CONFIG)
    return table, schedule, metadata, features, cohort, origins, screen


def put(path, value):
    """Write one canonical fixture product and return its real binding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.io.write_new(path, value)
    return m.io.file_binding(path)


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = numerical_fixture()

    def setUp(self):
        self.table, self.full_schedule, self.metadata, self.features, self.rows, self.origins, self.screen = copy.deepcopy(self.fixture)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        eligible = next(c for c in self.full_schedule['cap_schedules']
                        if c['quantity'] == 'all' and c['status'] == 'eligible_metadata_cell')
        uid = eligible['schedule_uid']
        self.schedule = copy.deepcopy(self.full_schedule)
        self.schedule['combination_schedule_cells'] = [
            cell for cell in self.schedule['combination_schedule_cells']
            if cell['schedule_uid'] == uid
            and cell['combination'] in {'S', 'BC', 'S+BC'}
        ]
        # Keep the deliberately reduced synthetic schedule internally
        # consistent.  Production schedules carry this same declaration, but
        # it must always be recomputed from the retained cells rather than
        # copied from the full 255-combination fixture.
        roles = {}
        for role in ('primary', 'diagnostic'):
            cells = [cell for cell in self.schedule['combination_schedule_cells']
                     if cell['analysis_role'] == role]
            eligible_cells = [cell for cell in cells
                              if cell['status'] == 'eligible_metadata_cell']
            roles[role] = {
                'intended_cells': len(cells),
                'eligible_cells': len(eligible_cells),
                'omitted_cells': len(cells) - len(eligible_cells),
            }
        self.schedule['accounting'].update({
            'intended_combination_cap_cells': sum(
                item['intended_cells'] for item in roles.values()),
            'eligible_combination_cap_cells': sum(
                item['eligible_cells'] for item in roles.values()),
            'omitted_combination_cap_cells': sum(
                item['omitted_cells'] for item in roles.values()),
            **roles,
        })
        self.contract = {
            'version': m.VERSION,
            'stage': m.STAGE,
            'expected_count': len(self.table),
            'output_root': str(self.root / 'output'),
            'bindings': {},
            'runtime': {},
            'table_sha256': m.io.value_hash(m.table_payload(self.table)),
            'schedule_sha256': m.io.value_hash(self.schedule),
            'accounting': m.schedule_counts(self.schedule),
            **m.SCOPE,
            **m.BC_SCOPE,
        }
        self.freeze = self.root / 'fit_freeze.json'
        self.write_freeze()

    def write_freeze(self):
        frozen = {
            'version': m.FREEZE_VERSION,
            'status': 'parent_frozen_for_native30_fold_only_fitting',
            'stage': m.STAGE,
            'fitting_authorized': True,
            'scoring_authorized': True,
            'contract': self.contract,
            'contract_sha256': m.io.value_hash(self.contract),
            'independent_review': {'reviewer': 'root', 'approved': True},
            **m.SCOPE,
        }
        self.freeze.write_bytes(m.io.canonical(frozen))
        self.authorization = m.authorize(self.contract, self.freeze,
                                         m.io.digest(self.freeze))

    def task(self, mode='values_plus_missing', combination='S+BC'):
        cell = next(cell for cell in self.schedule['combination_schedule_cells']
                    if cell['feature_mode'] == mode
                    and cell['combination'] == combination)
        return m.resolve_task(cell, self.schedule)

    def run_fixture(self):
        return m.run_context(self.contract, self.table, self.schedule,
                             self.authorization)

    def test_exact_eight_family_catalogue_and_old_projection(self):
        self.assertEqual(list(m.FAMILIES), ['S', 'D', 'R', 'P', 'F', 'H', 'SC', 'BC'])
        self.assertEqual(len(m.FEATURES), 55)
        self.assertEqual(len(set(m.FEATURES)), 55)
        self.assertEqual(m.FEATURES[:-1], old.FEATURES)
        self.assertEqual(len(m.planner.COMBINATIONS), 255)
        self.assertEqual([c for c in m.planner.COMBINATIONS
                          if 'BC' not in c.split('+')], old.planner.COMBINATIONS)
        self.assertEqual(m.BC_FEATURE, 'BC_b2_500_750_1250hz_center8s_median')

    def test_pinned_sources_and_no_old_global_mutation(self):
        self.assertEqual(m.io.digest(m.HERE / 'evaluate_native30_v3.py'), m.V3_SHA)
        self.assertEqual(m.io.digest(m.HERE / 'frozen_evaluate_expanded_20260905.py'), m.BASE_SHA)
        self.assertEqual(m.io.digest(m.HERE / 'plan_native30_evaluation_schedule_v1.py'), m.PLANNER_SHA)
        self.assertEqual(m.io.digest(m.HERE / 'prepare_native30_evaluation_inputs_bc_v1.py'), m.PREPARER_SHA)
        before = (copy.deepcopy(old.FAMILIES), list(old.FEATURES),
                  list(old.planner.COMBINATIONS))
        m.replay_schedule(self.full_schedule, self.rows, self.screen)
        m.columns_for(self.task())
        self.assertEqual((old.FAMILIES, old.FEATURES, old.planner.COMBINATIONS), before)

    def test_changed_pins_fail_closed_and_private_rebinding_restores(self):
        for path, name in ((m.HERE / 'evaluate_native30_v3.py', '_reject_v3'),
                           (m.HERE / 'plan_native30_evaluation_schedule_v1.py',
                            '_reject_planner')):
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, 'unapproved/changed code pin'):
                m.load_module(path, '0' * 64, name)
        with patch.object(m, 'PLANNER_SHA', '0' * 64), \
             self.assertRaisesRegex(ValueError, 'unapproved/changed source pin'):
            m._pinned_binding(m.HERE / 'plan_native30_evaluation_schedule_v1.py',
                              m.PLANNER_SHA)
        before = (m.CORE.FAMILIES, m.CORE.FEATURES, m.CORE.planner,
                  m.CORE.input_guard, m.CORE.validate_model)
        with self.assertRaises(ValueError):
            m.validate_model({}, self.table, self.task())
        self.assertIs(m.CORE.FAMILIES, before[0])
        self.assertEqual(m.CORE.FEATURES, before[1])
        self.assertIs(m.CORE.planner, before[2])
        self.assertIs(m.CORE.input_guard, before[3])
        self.assertIs(m.CORE.validate_model, before[4])

    def test_schedule_replay_is_exact_255_and_preserves_caps_and_folds(self):
        replayed = m.replay_schedule(self.full_schedule, self.rows, self.screen)
        self.assertEqual(replayed, self.full_schedule)
        self.assertEqual(len(replayed['combinations']), 255)
        self.assertEqual(len(replayed['combination_schedule_cells']),
                         25 * 255 * 7)
        projected = [cell for cell in replayed['combination_schedule_cells']
                     if 'BC' not in cell['combination'].split('+')]
        old_schedule = m.planner.make_schedule(self.rows, self.screen)
        self.assertEqual(projected, old_schedule['combination_schedule_cells'])
        self.assertEqual(replayed['fold_cells'], old_schedule['fold_cells'])
        self.assertEqual(replayed['cap_schedules'], old_schedule['cap_schedules'])

    def test_schedule_accounting_is_derived_and_tamper_rejected(self):
        counts = m.schedule_counts(self.full_schedule)
        self.assertEqual(counts['primary']['intended'], 25 * 5 * 255)
        self.assertEqual(counts['diagnostic']['intended'], 25 * 2 * 255)
        broken = copy.deepcopy(self.full_schedule)
        broken['accounting']['eligible_combination_cap_cells'] += 1
        with self.assertRaisesRegex(ValueError, 'accounting replay'):
            m.replay_schedule(broken, self.rows, self.screen)
        broken = copy.deepcopy(self.full_schedule)
        broken['combination_schedule_cells'][0]['combination'] = 'S+M'
        with self.assertRaises(ValueError):
            m.schedule_counts(broken)

    def test_build_context_replays_bc_package_contract_and_schedule_graph(self):
        """Exercise the package-shaped adapter boundary, not a hand-built context.

        The package verifier is a small stand-in for the caller-pinned
        assembler because this synthetic fixture intentionally has 120 rows,
        whereas the production assembler is fixed at 3830.  Its contract,
        products, authority request, parent product graph, and old/new schedule
        documents use the production schemas and are all rehashed by
        ``build_context``.
        """
        class PackageBackend:
            def __init__(self, bundle):
                self.bundle = bundle
                self.calls = []

            def verify_package(self, package, commit_sha):
                self.calls.append((Path(package), commit_sha))
                return copy.deepcopy(self.bundle)

        root = self.root / 'package_schema'
        old_schedule = m.planner.make_schedule(self.rows, self.screen)
        derived = m.schedule_adapter.derive_schedule(
            old_schedule, old_schedule, m.OLD_FAMILY_CONFIG)

        cohort_doc = {
            'version': 'run_native30_fhsc_cohort_v1',
            'status': 'frozen_before_any_measurement_audio_reads',
            'expected_count': len(self.rows), 'rows': self.rows,
            'source_counts': dict(Counter(r['source_group'] for r in self.rows)),
            'human': 60, 'ai': 60, 'bindings': {}, 'classifier_fits': 0,
            'cohort_admitted': False, 'source_selection_changed': False,
        }
        cohort_entry = put(root / 'authority/cohort.json', cohort_doc)
        plan_entry = put(root / 'authority/plan.json', {'rows': self.origins})
        screen_entry = put(root / 'authority/screen.json', self.screen)
        old_entry = put(root / 'old_schedule/schedule_draft.json', old_schedule)
        old_commit = {
            'status': 'committed_metadata_draft_not_evaluation_authorization',
            'version': m.planner.VERSION,
            'products': {'schedule_draft.json': old_entry},
            'draft_sha256': old_entry['sha256'], 'classifier_fits': 0,
            'fitting_authorized': False,
        }
        old_commit_entry = put(root / 'old_schedule/COMMIT.json', old_commit)
        parent_contract = {
            'version': 'prepare_native30_evaluation_inputs_v3',
            'status': 'assembled_not_evaluated_not_authorized',
            'expected_count': len(self.rows),
            'family_config': copy.deepcopy(m.OLD_FAMILY_CONFIG),
            'feature_names': list(m._OLD_FEATURES),
            'cohort_contract': cohort_entry,
            'bindings': {
                'cohort_contract': cohort_entry, 'origin_plan': plan_entry,
                'screen': screen_entry, 'schedule_draft': old_entry,
                'schedule_commit': old_commit_entry,
            },
        }
        parent_root = root / 'parent_package'
        parent_contract_entry = put(parent_root / 'contract.json', parent_contract)
        parent_products = {'contract.json': parent_contract_entry}
        for name, value in (
                ('metadata.json', self.metadata), ('features.json', self.features),
                ('lineage.json', self.origins)):
            parent_products[name] = put(parent_root / name, value)
        parent_commit_entry = put(parent_root / 'COMMIT.json', {
            'version': 'prepare_native30_evaluation_inputs_v3',
            'status': 'assembled_not_evaluated_not_authorized',
            'expected_count': len(self.rows), 'products': parent_products,
            'synthetic_parent_fixture': True,
        })
        freeze_entry = put(root / 'bc_authority/freeze.json',
                           {'synthetic_bc_freeze': True})
        bc_commit_entry = put(root / 'bc_authority/COMMIT.json',
                              {'synthetic_bc_measurement_commit': True})

        schedule = {
            **derived,
            'prepared_contract_sha256': cohort_entry['sha256'],
            'input_bindings': {
                'cohort': cohort_entry, 'plan': plan_entry, 'screen': screen_entry,
            },
            'metadata_read_scope': 'synthetic metadata only', 'runtime': {},
            'historical_original_schedule_provenance': {
                'schedule_sha256': old_schedule['schedule_sha256'],
            },
        }
        schedule_entry = put(root / 'bc_schedule/schedule_draft.json', schedule)
        schedule_commit_entry = put(root / 'bc_schedule/COMMIT.json', {
            'version': m.schedule_adapter.VERSION,
            'status': 'committed_BC_metadata_schedule_draft_not_measurement_or_evaluation_authorization',
            'products': {'schedule_draft.json': schedule_entry},
            'draft_sha256': schedule_entry['sha256'],
            'original_schedule': old_entry, 'original_commit': old_commit_entry,
            'classifier_fits': 0, 'fitting_authorized': False,
            'measurement_authorized': False,
        })
        request = {
            'parent_commit': parent_commit_entry, 'bc_freeze': freeze_entry,
            'bc_commit': bc_commit_entry, 'schedule_draft': schedule_entry,
            'schedule_commit': schedule_commit_entry,
        }

        package = root / 'bc_package'
        package.mkdir(parents=True)
        assembled = {
            'version': 'prepare_native30_evaluation_inputs_bc_v1',
            'status': 'assembled_not_evaluated_not_authorized',
            'expected_count': len(self.table),
            'family_config': copy.deepcopy(m.FAMILIES),
            'feature_names': list(m.FEATURES), 'feature_count': 55,
            'metadata_schema': ['id', 'label', 'source_group', 'group_id',
                                'component_id', 'role', 'duration_view_s',
                                'native_sample_rate_hz', 'input_sha256',
                                'waveform_float32_sha256'],
            'feature_schema': ['id', *m.FEATURES],
            'lineage_schema': ['id'],
            'source_counts': dict(Counter(self.table['__source'])),
            'scientific_missingness': 'JSON null retained; no availability filtering',
            'request': request, 'request_sha256': m.io.value_hash(request),
            'code_bindings': {
                'assembler': m.io.file_binding(
                    m.HERE / (m.PREPARER_NAME + '.py')),
                'tests': m.io.file_binding(
                    m.HERE / ('test_' + m.PREPARER_NAME + '.py')),
            },
            'parent_products': parent_products,
            'bc_proof': {'synthetic': True},
            'original54_features_sha256': parent_products['features.json']['sha256'],
            'schedule_accounting': schedule['accounting'],
            'schedule_sha256': schedule['schedule_sha256'],
            'cohort_contract': cohort_entry,
            'analysis_duration_s': 8, 'duration_view_s': 30,
            **m.BC_SCOPE, **m.SCOPE,
        }
        package_values = {
            'contract.json': assembled, 'metadata.json': self.metadata,
            'features.json': self.features, 'lineage.json': self.origins,
            'bc_coverage.json': {
                'version': 'prepare_native30_evaluation_inputs_bc_v1',
                'status': 'descriptive_BC_availability_not_predictors_not_admission',
                **m.BC_SCOPE, **m.SCOPE,
            },
        }
        package_products = {}
        for name, value in package_values.items():
            put(package / name, value)
            package_products[name] = m.io.file_binding(package / name)
        package_commit = {
            'version': 'prepare_native30_evaluation_inputs_bc_v1',
            'status': 'assembled_not_evaluated_not_authorized',
            'expected_count': len(self.table), 'products': package_products,
            'request_sha256': m.io.value_hash(request),
            'all_source_evidence_deep_verified_before_and_after': True,
            'audio_files_opened': 0, **m.BC_SCOPE,
        }
        package_commit_entry = put(package / 'COMMIT.json', package_commit)
        bundle = {
            'contract': assembled, 'metadata': self.metadata,
            'features': self.features, 'lineage': self.origins,
            'coverage': package_values['bc_coverage.json'],
            'commit': package_commit_entry, 'products': package_products,
        }
        backend = PackageBackend(bundle)
        counts = m.schedule_counts(schedule)
        expected = {
            'total': len(self.table),
            'labels': dict(Counter(self.table['__label'])),
            'intended': sum(value['intended'] for value in counts.values()),
            'primary': counts['primary']['eligible'],
            'diagnostic': counts['diagnostic']['eligible'],
        }
        with patch.object(m, 'load_module', return_value=backend), \
             patch.object(m, 'runtime_snapshot', return_value={'synthetic': True}), \
             patch.object(m, 'EXPECTED', expected), \
             patch.object(m.planner, 'SOURCES',
                          dict(Counter(self.table['__source']))), \
             patch.object(m, 'SCHEDULE_SHA', old_entry['sha256']), \
             patch.object(m, 'SCHEDULE_COMMIT_SHA', old_commit_entry['sha256']), \
             patch.object(m.planner, 'PLAN_SHA', plan_entry['sha256']), \
             patch.object(m.planner, 'SCREEN_SHA', screen_entry['sha256']), \
             patch.object(m.planner, 'validate_prepared_contract',
                          return_value=self.rows):
            contract, table, replayed = m.build_context(
                package, package_commit_entry['sha256'], root / 'evaluation')
        self.assertEqual(len(backend.calls), 2)  # build + final deep recheck
        self.assertEqual(m.io.value_hash(m.table_payload(table)),
                         m.io.value_hash(m.table_payload(self.table)))
        self.assertEqual(replayed, schedule)
        self.assertEqual(contract['families'], m.FAMILIES)
        self.assertEqual(contract['feature_names'], m.FEATURES)
        self.assertTrue(contract['schedule_accounting_recomputed_from_cells'])
        self.assertEqual(contract['accounting'], counts)
        self.assertFalse((root / 'evaluation').exists())

    def test_table_appends_one_scalar_and_retains_null_without_filtering(self):
        self.assertEqual(len(self.table), len(self.rows))
        self.assertTrue(self.table[m.FAMILIES['S'][0]].isna().all())
        self.assertGreater(self.table[m.BC_FEATURE].isna().sum(), 0)
        self.assertLess(self.table[m.BC_FEATURE].notna().sum(), len(self.table))
        self.assertEqual(set(self.table['__label']), {0, 1})
        self.assertGreater(self.table.groupby('__group')['__label'].nunique().max(), 1)

    def test_all_three_modes_use_pinned_core_and_bc_columns(self):
        for mode in (m.planner.PRIMARY_MODE, *m.planner.DIAGNOSTIC_MODES):
            task = self.task(mode)
            columns = m.columns_for(task)
            self.assertEqual(columns, m.FAMILIES['S'] + [m.BC_FEATURE])
            train, test = m.task_tables(self.table, task)
            model = m.BASE.fit_model(train, columns, feature_mode=mode)
            receipt = m.model_receipt(self.contract, self.table, task)
            self.assertEqual(receipt['model'], model)
            np.testing.assert_array_equal(
                [row['score'] for row in receipt['predictions']],
                m.BASE.predict(test, model))

    def test_old127_model_and_prediction_parity(self):
        task = self.task('values_plus_missing', 'S')
        adapter = m.model_receipt(self.contract, self.table, task)
        old_contract = copy.deepcopy(self.contract)
        old_contract['version'] = old.VERSION
        old_contract['stage'] = old.STAGE
        old_contract['families'] = old.FAMILIES
        old_contract['feature_names'] = old.FEATURES
        reference = old.model_receipt(old_contract, self.table, task)
        self.assertEqual(adapter['model'], reference['model'])
        self.assertEqual(adapter['predictions'], reference['predictions'])
        self.assertEqual(adapter['metrics'], reference['metrics'])

    def test_entirely_missing_bc_is_valid_for_all_modes(self):
        table = self.table.copy()
        table[m.BC_FEATURE] = np.nan
        for mode in (m.planner.PRIMARY_MODE, *m.planner.DIAGNOSTIC_MODES):
            task = self.task(mode)
            receipt = m.model_receipt(self.contract, table, task)
            self.assertTrue(np.isfinite(np.asarray(
                receipt['model']['coefficients_with_intercept'], dtype=float)).all())
            self.assertEqual(receipt['model']['observed_fraction_by_column'][m.BC_FEATURE], 0.0)

    def test_train_only_values_cannot_change_model(self):
        task = self.task()
        original = m.model_receipt(self.contract, self.table, task)
        changed = self.table.copy()
        changed.loc[changed['__id'].isin(task['test_ids']), m.BC_FEATURE] = 99.0
        second = m.model_receipt(self.contract, changed, task)
        self.assertEqual(original['model'], second['model'])
        self.assertNotEqual(original['predictions_sha256'], second['predictions_sha256'])

    def test_parent_authorization_precedes_any_fit(self):
        with patch.object(m.BASE, 'fit_model', side_effect=AssertionError('no fit')) as fit:
            with self.assertRaises(ValueError):
                m.run_context(self.contract, self.table, self.schedule, {})
            with self.assertRaises(ValueError):
                m.authorize(self.contract, None, None)
            fit.assert_not_called()
        frozen = m.io.read_json(self.freeze)
        frozen['winner_selection'] = True
        self.freeze.write_bytes(m.io.canonical(frozen))
        with self.assertRaisesRegex(ValueError, 'scope/contract'):
            m.authorize(self.contract, self.freeze, m.io.digest(self.freeze))

    def test_publish_resume_without_refit_and_immutable_inventory(self):
        result = self.run_fixture()
        self.assertEqual(result['completed'], 9)
        output = Path(self.contract['output_root'])
        before = m.products(output)
        with patch.object(m.BASE, 'fit_model', side_effect=AssertionError('resume refit')):
            resumed = self.run_fixture()
        self.assertEqual(resumed['status'], 'verified_existing_COMMIT_no_refit')
        self.assertEqual(m.products(output), before)
        self.assertEqual(m.io.read_json(output / 'COMMIT.json')['products'], before)

    def test_failure_receipt_is_retained_and_retry_fits_only_missing_task(self):
        original = m.BASE.fit_model
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise np.linalg.LinAlgError('synthetic interruption')
            return original(*args, **kwargs)

        with patch.object(m.BASE, 'fit_model', side_effect=fail_once):
            result = self.run_fixture()
        self.assertEqual((result['completed'], result['failed']), (8, 1))
        output = Path(self.contract['output_root'])
        failure = next((output / 'failures').iterdir())
        before = failure.read_bytes()
        with patch.object(m.BASE, 'fit_model', wraps=original) as fit:
            final = self.run_fixture()
            self.assertEqual(fit.call_count, 1)
        self.assertEqual(final['completed'], 9)
        self.assertEqual(failure.read_bytes(), before)

    def test_tampered_receipt_is_not_overwritten(self):
        self.run_fixture()
        output = Path(self.contract['output_root'])
        path = next((output / 'models').iterdir())
        path.write_bytes(path.read_bytes() + b' ')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'COMMIT changed'):
            self.run_fixture()
        self.assertEqual(path.read_bytes(), before)

    def test_resealed_prediction_tamper_fails_deep_replay(self):
        self.run_fixture()
        output = Path(self.contract['output_root'])
        (output / 'COMMIT.json').unlink()
        path = next((output / 'models').iterdir())
        envelope = m.io.read_json(path)
        payload = envelope['payload']
        payload['predictions'][0]['score'] += .1
        payload['predictions_sha256'] = m.io.value_hash(payload['predictions'])
        envelope['receipt_sha256'] = m.io.value_hash(payload)
        path.write_bytes(m.io.canonical(envelope))
        with patch.object(m.BASE, 'fit_model', side_effect=AssertionError('must not overwrite')):
            result = self.run_fixture()
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertEqual(result['failed'], 1)

    def test_diagnostic_modes_are_all_cap_only(self):
        task = self.task('median_only')
        task['quantity'] = 25
        with self.assertRaisesRegex(ValueError, 'all-cap'):
            m.columns_for(task)
        self.assertEqual(m.columns_for(self.task('missingness_only')),
                         m.FAMILIES['S'] + [m.BC_FEATURE])

    def test_coefficient_validation_does_not_call_solver(self):
        task = self.task()
        train, _ = m.task_tables(self.table, task)
        model = m.BASE.fit_model(train, m.columns_for(task),
                                 feature_mode=task['feature_mode'])
        with patch.object(np.linalg, 'solve', side_effect=AssertionError('no refit')):
            proof = m.validate_model(model, train, task)
        self.assertLessEqual(proof['normal_equation_residual_infinity'],
                             proof['roundoff_tolerance'])
        broken = copy.deepcopy(model)
        broken['coefficients_with_intercept'][0] += .1
        with self.assertRaisesRegex(ValueError, 'normal-equation residual'):
            m.validate_model(broken, train, task)


if __name__ == '__main__':
    unittest.main()
