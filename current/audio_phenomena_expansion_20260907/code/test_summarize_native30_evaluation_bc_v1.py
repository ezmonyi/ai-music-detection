"""Synthetic and metadata-only tests for the Native30 BC report adapter."""
from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import plan_native30_evaluation_schedule_v1 as planner
import plan_native30_evaluation_schedule_bc_v1 as schedule_adapter
import summarize_native30_evaluation_bc_v1 as m
import summarize_native30_evaluation_v3 as old


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.canonical(value))
    return m.binding(path)


def synthetic_rows():
    rows = []

    def bucket_group(prefix, bucket, ordinal):
        number = 0
        found = []
        while len(found) <= ordinal:
            candidate = prefix + '_g' + str(number)
            if planner.hash_fold(candidate) == bucket:
                found.append(candidate)
            number += 1
        return found[-1]

    for source, label in (('H1', '0'), ('H2', '0'), ('A1', '1'), ('A2', '1')):
        for bucket in range(5):
            for ordinal in range(3):
                group = bucket_group(source, bucket, ordinal)
                for sample in range(2):
                    rows.append({'id': f'{source}_{bucket}_{ordinal}_{sample}',
                                 'source_group': source, 'label': label,
                                 'role': 'development', 'group_id': group,
                                 'component_id': 'component_' + group})
    return sorted(rows, key=lambda row: row['id'])


def synthetic_screen(rows):
    components = {}
    for row in rows:
        component = components.setdefault(
            row['component_id'], {'component_id': row['component_id'], 'members': [],
                                  'candidate_screen_members': [],
                                  'protected_relationships': [], 'link_tokens': []})
        component['members'].append(row['id'])
        component['candidate_screen_members'].append(row['id'])
    return {'rows': copy.deepcopy(rows), 'components': list(components.values())}


def reduced_expected(rows, combinations=('S', 'S+BC')):
    return {'rows': len(rows),
            'sources': dict(Counter(row['source_group'] for row in rows)),
            'labels': {0: sum(row['label'] == '0' for row in rows),
                       1: sum(row['label'] == '1' for row in rows)},
            'buckets': 5, 'caps': (25, 'all'),
            'combinations': list(combinations), 'intended': 0,
            'eligible': 0, 'primary': 0, 'diagnostic': 0}


def resolve_cell(cell, schedule):
    caps = {row['schedule_uid']: row for row in schedule['cap_schedules']}
    folds = {row['fold_uid']: row for row in schedule['fold_cells']}
    cap = caps[cell['schedule_uid']]
    fold = folds[cap['fold_uid']]
    return {**cell, 'quantity': cap['quantity'], 'fold_uid': cap['fold_uid'],
            'fold_type': cap['fold_type'], 'heldout_source': cap['heldout_source'],
            'opposite_group_fold': cap['opposite_group_fold'],
            'train_ids': cap['train']['ids'], 'test_ids': fold['test']['ids'],
            'train_id_set_sha256': cap['train']['id_set_sha256'],
            'test_id_set_sha256': fold['test']['id_set_sha256'],
            'omission_reasons': cap['omission_reasons']}


def make_package(directory, rows, schedule, expected):
    package = Path(directory) / 'bc_package'
    authorities = Path(directory) / 'authorities'
    request = {}
    for name in ('parent_commit', 'bc_freeze', 'bc_commit', 'schedule_draft',
                 'schedule_commit'):
        request[name] = put(authorities / (name + '.json'), {'authority': name})

    metadata = []
    features = []
    lineage = []
    for row in rows:
        metadata.append({'id': row['id'], 'label': row['label'],
                         'source_group': row['source_group'],
                         'group_id': row['group_id'],
                         'component_id': row['component_id'], 'role': row['role'],
                         'duration_view_s': 30, 'native_sample_rate_hz': 44100,
                         'input_sha256': 'input_' + row['id'],
                         'waveform_float32_sha256': 'wave_' + row['id']})
        features.append({'id': row['id'], **{name: None for name in m.FEATURES}})
        lineage.append({'id': row['id']})
    metadata_schema = list(metadata[0])
    lineage_schema = ['id']
    code_bindings = {
        'assembler': m.binding(m.HERE / 'prepare_native30_evaluation_inputs_bc_v1.py',
                               m.PREPARER_SHA),
        'tests': m.binding(m.HERE / 'test_prepare_native30_evaluation_inputs_bc_v1.py',
                           m.PREPARER_TEST_SHA),
    }
    package_contract = {
        'version': 'prepare_native30_evaluation_inputs_bc_v1',
        'status': 'assembled_not_evaluated_not_authorized',
        'expected_count': len(rows), 'family_config': m.FAMILY_CONFIG,
        'feature_names': m.FEATURES, 'feature_count': 55,
        'metadata_schema': metadata_schema,
        'feature_schema': ['id', *m.FEATURES], 'lineage_schema': lineage_schema,
        'source_counts': dict(Counter(row['source_group'] for row in metadata)),
        'scientific_missingness': 'JSON null retained; no imputation or availability filtering',
        'request': request, 'request_sha256': m.value_hash(request),
        'code_bindings': code_bindings,
        'schedule_accounting': schedule['accounting'],
        'schedule_sha256': schedule['schedule_sha256'],
        'cohort_contract': request['parent_commit'],
        **m.BC_SCOPE, **m.SCOPE,
    }
    values = {'contract.json': package_contract, 'metadata.json': metadata,
              'features.json': features, 'lineage.json': lineage,
              'bc_coverage.json': {
                  'version': 'prepare_native30_evaluation_inputs_bc_v1',
                  'status': 'descriptive_BC_availability_not_predictors_not_admission',
                  **m.BC_SCOPE, **m.SCOPE}}
    products = {}
    for name, value in values.items():
        products[name] = put(package / name, value)
    package_commit = {
        'version': 'prepare_native30_evaluation_inputs_bc_v1',
        'status': 'assembled_not_evaluated_not_authorized',
        'expected_count': len(rows), 'products': products,
        'request_sha256': m.value_hash(request),
        'all_source_evidence_deep_verified_before_and_after': True,
        'audio_files_opened': 0, **m.BC_SCOPE,
    }
    package_entry = put(package / 'COMMIT.json', package_commit)
    return package_entry


def make_evaluation(directory):
    directory = Path(directory).resolve()
    rows = synthetic_rows()
    screen = synthetic_screen(rows)
    old_schedule = planner.make_schedule(rows, screen)
    schedule = schedule_adapter.derive_schedule(old_schedule, old_schedule,
                                                m._OLD_FAMILY_CONFIG)
    expected = reduced_expected(rows)
    selected = set(expected['combinations'])
    selected_caps = set(expected['caps'])
    schedule_path = directory / 'schedule' / 'schedule_draft.json'
    schedule_entry = put(schedule_path, schedule)
    package_entry = make_package(directory, rows, schedule, expected)

    tasks = []
    for cell in schedule['combination_schedule_cells']:
        task = resolve_cell(cell, schedule)
        if task['combination'] in selected and task['quantity'] in selected_caps:
            tasks.append(task)
    valid = [task for task in tasks if task['status'] == 'eligible_metadata_cell']
    omissions = [task for task in tasks if task['status'] == 'omitted']
    accounting = {}
    for role in ('primary', 'diagnostic'):
        chosen = [task for task in valid if task['analysis_role'] == role]
        all_tasks = [task for task in tasks if task['analysis_role'] == role]
        accounting[role] = {
            'intended': len(all_tasks), 'eligible': len(chosen),
            'omitted': len(all_tasks) - len(chosen),
            'prediction_rows': sum(len(task['test_ids']) for task in chosen),
        }
    expected.update(intended=len(tasks), eligible=len(valid),
                   primary=accounting['primary']['eligible'],
                   diagnostic=accounting['diagnostic']['eligible'])

    package_root = Path(package_entry['path']).parent
    code = {
        'evaluate_native30_bc_v1.py': m.binding(m.HERE / 'evaluate_native30_bc_v1.py', m.EVALUATOR_SHA),
        'test_evaluate_native30_bc_v1.py': m.binding(m.HERE / 'test_evaluate_native30_bc_v1.py', m.EVALUATOR_TEST_SHA),
        'evaluate_native30_v3.py': m.binding(m.HERE / 'evaluate_native30_v3.py', m.V3_EVALUATOR_SHA),
        'test_evaluate_native30_v3.py': m.binding(m.HERE / 'test_evaluate_native30_v3.py', m.V3_EVALUATOR_TEST_SHA),
        'prepare_native30_evaluation_inputs_bc_v1.py': m.binding(m.HERE / 'prepare_native30_evaluation_inputs_bc_v1.py', m.PREPARER_SHA),
        'test_prepare_native30_evaluation_inputs_bc_v1.py': m.binding(m.HERE / 'test_prepare_native30_evaluation_inputs_bc_v1.py', m.PREPARER_TEST_SHA),
        'plan_native30_evaluation_schedule_bc_v1.py': m.binding(m.HERE / 'plan_native30_evaluation_schedule_bc_v1.py', m.BC_SCHEDULE_CODE_SHA),
        'test_plan_native30_evaluation_schedule_bc_v1.py': m.binding(m.HERE / 'test_plan_native30_evaluation_schedule_bc_v1.py', m.BC_SCHEDULE_TEST_SHA),
    }
    contract = {
        'version': 'evaluate_native30_bc_v1',
        'stage': 'native30_eight_family_schedule_only_development_evaluation_bc_v1',
        'package_commit': package_entry, 'bindings': code,
        'output_root': str(directory / 'evaluation'), 'expected_count': len(rows),
        'families': m.FAMILY_CONFIG, 'feature_names': m.FEATURES,
        'table_sha256': 'synthetic_table_not_loaded_by_reporter',
        'accounting': accounting, 'schedule_document': schedule_entry,
        'schedule_sha256': m.value_hash(schedule),
        **m.BC_SCOPE, **m.SCOPE,
    }
    contract_sha = m.value_hash(contract)
    evaluation = directory / 'evaluation'
    for name in ('models', 'failures', 'runs'):
        (evaluation / name).mkdir(parents=True)
    (evaluation / 'writer.lock').touch()
    put(evaluation / 'contract.json', contract)
    freeze = {
        'version': 'native30-evaluation-parent-freeze-bc-v1',
        'status': 'parent_frozen_for_native30_fold_only_fitting',
        'stage': contract['stage'], 'contract': contract,
        'contract_sha256': contract_sha, 'fitting_authorized': True,
        'scoring_authorized': True,
        'independent_review': {'reviewer': 'root', 'approved': True},
        **m.SCOPE,
    }
    freeze_entry = put(directory / 'freeze.json', freeze)
    put(evaluation / 'authorization.json', {
        'status': 'parent_frozen_verified', 'contract_sha256': contract_sha,
        'binding': freeze_entry})
    put(evaluation / 'omitted.json', omissions)

    row_by_id = {row['id']: row for row in rows}
    models = {}
    for task in valid:
        predictions = []
        for ident in task['test_ids']:
            row = row_by_id[ident]
            label = int(row['label'])
            predictions.append({
                'id': ident, 'label': label, 'source_group': row['source_group'],
                'group_id': row['group_id'], 'component_id': row['component_id'],
                'role': 'development', 'score': 0.75 if label else 0.25,
                'threshold': 0.5, 'predicted_label': label,
            })
        model = {'threshold': 0.5, 'ridge': 10.0,
                 'feature_mode': task['feature_mode'],
                 'training_rows': len(task['train_ids'])}
        payload = {
            'status': 'fold_only_model_and_predictions_not_selected',
            'contract_sha256': contract_sha, 'task': task,
            'task_sha256': m.value_hash(task), 'model': model,
            'model_sha256': m.value_hash(model), 'predictions': predictions,
            'predictions_sha256': m.value_hash(predictions),
            'metrics': {'within_fold_pooled_descriptive': {'roc_auc': m.auc(predictions)}},
            'training_numerical_audit': {
                'status': 'training_transforms_and_ridge_equations_verified_without_refit',
                'normal_equation_residual_infinity': 0.0,
                'roundoff_tolerance': 1e-12,
            },
            **m.PRODUCER_SCOPE,
        }
        uid = m.model_uid(task, contract_sha)
        path = evaluation / 'models' / (uid + '.json')
        put(path, {'payload': payload, 'receipt_sha256': m.value_hash(payload)})
        models[uid] = m.binding(path)
    put(evaluation / 'manifest.json', {
        'version': 'evaluate_native30_bc_v1', 'contract_sha256': contract_sha,
        'model_instances': len(valid), 'omitted_instances': len(omissions),
        'accounting': accounting,
        'training_transform_equation_prediction_replay_performed': True,
        'independent_full_publication_numerical_audit_performed': False,
        'model_receipts': models, **m.SCOPE,
    })
    put(evaluation / 'runs' / ('0' * 32 + '.json'),
        {'synthetic_fixture': True, 'completed': len(valid)})
    # The v3 inventory deliberately requires a terminal COMMIT entry before
    # it will produce the product map.  The real evaluator writes it last;
    # this placeholder is replaced immediately below in the synthetic setup.
    put(evaluation / 'COMMIT.json', {})
    commit = {
        'version': 'evaluate_native30_bc_v1',
        'status': 'committed_fold_only_evaluation_not_model_selection',
        'contract_sha256': contract_sha, 'model_instances': len(valid),
        'products': m.evaluation_inventory(evaluation),
        'all_input_output_runtime_bindings_end_rehashed': True, **m.SCOPE,
    }
    commit_entry = put(evaluation / 'COMMIT.json', commit)
    return evaluation, expected, schedule, schedule_entry, package_entry, commit_entry


class AdapterTests(unittest.TestCase):
    def test_exact_eight_family_255_catalogue_and_old127_projection(self):
        self.assertEqual(m.FAMILIES[:7], tuple(old.FAMILIES))
        self.assertEqual(m.OLD_COMBINATIONS, list(old.COMBINATIONS))
        self.assertEqual(len(m.FAMILIES), 8)
        self.assertEqual(len(m.FEATURES), 55)
        self.assertEqual(len(m.COMBINATIONS), 255)
        self.assertEqual([c for c in m.COMBINATIONS if 'BC' not in c.split('+')],
                         m.OLD_COMBINATIONS)

    def test_old127_report_core_and_markdown_are_exact_v3_delegation(self):
        cap_records, pools = {}, {}
        for protocol, source in ((old.PROTOCOLS[0], 'H'),
                                 (old.PROTOCOLS[1], 'A'),
                                 (old.PROTOCOLS[2], '__all_sources__')):
            for cap in old.CAPS:
                references = []
                for bucket in range(5):
                    uid = old.value_hash([protocol, cap, bucket])[:24]
                    cap_records[uid] = {
                        'opposite_group_fold': bucket, 'train_ids': [],
                        'status': 'omitted',
                        'omission_reasons': ['synthetic_no_eligible_test_predictions']}
                    references.append({'entry': None, 'schedule_uid': uid})
                for combination in old.COMBINATIONS:
                    modes = old.MODES if cap == 'all' else old.MODES[:1]
                    for mode in modes:
                        pools[(protocol, source, cap, combination, mode)] = references
        published = {'fold_rows': [], 'pools': pools, 'cap_records': cap_records,
                     'expected': old.EXPECTED, 'contract': {'accounting': {}}}
        expected = old.summarize(published)
        with m._bound_core():
            delegated = m.CORE.summarize(published)
        self.assertEqual(delegated.pop('version'), m.VERSION)
        self.assertEqual(expected.pop('version'), old.VERSION)
        self.assertEqual(delegated, expected)
        self.assertEqual(m.markdown_products(delegated, published),
                         old.markdown_products(expected, published))

    def test_actual_committed_schedule_recomputes_103530_and_old51562(self):
        path = (m.HERE.parent / 'results' /
                'native30_bc_evaluation_schedule_draft_v1' /
                'schedule_draft.json').resolve()
        schedule = m.read(path)
        proof = m.schedule_audit(schedule)
        self.assertEqual(proof['counts']['total']['intended_cells'], 107100)
        self.assertEqual(proof['counts']['total']['eligible_cells'], 103530)
        self.assertEqual(proof['counts']['primary']['eligible_cells'], 73950)
        self.assertEqual(proof['counts']['diagnostic']['eligible_cells'], 29580)
        self.assertEqual(proof['old_projection']['accounting']['total']['eligible_cells'], 51562)
        self.assertEqual(proof['old_projection']['accounting']['total']['intended_cells'], 53340)
        self.assertEqual(proof['additional_BC_projection']['accounting']['total']['eligible_cells'], 51968)

        old_path = (m.HERE.parent / 'results' /
                    'native30_evaluation_schedule_draft_v1' /
                    'schedule_draft.json').resolve()
        old_schedule = m.read(old_path)
        self.assertEqual(proof['old_cells'], old_schedule['combination_schedule_cells'])
        self.assertEqual(proof['old_projection']['combinations'], old_schedule['combinations'])

    def test_private_rebinding_does_not_mutate_public_v3_globals(self):
        before = {name: copy.deepcopy(getattr(old, name)) for name in
                  ('FAMILIES', 'COMBINATIONS', 'CAPS', 'MODES', 'PROTOCOLS',
                   'REFERENCE', 'EXPECTED', 'PRODUCER_SCOPE', 'SCOPE', 'VERSION')}
        self.assertEqual(m.auc([
            {'score': 0.1, 'label': 0}, {'score': 0.9, 'label': 1}]), 1.0)
        self.assertEqual(m.pool_predictions([])['status'], 'no_eligible_bucket_predictions')
        after = {name: getattr(old, name) for name in before}
        self.assertEqual(after, before)

    def test_actual_shaped_package_schedule_and_evaluation_integration(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation, expected, schedule, schedule_entry, package_entry, commit_entry = make_evaluation(temporary)
            published = m.inspect_evaluation(evaluation, commit_entry['sha256'], expected)
            self.assertEqual(published['package_proof']['metadata_rows'], len(synthetic_rows()))
            self.assertEqual(published['schedule_proof']['counts']['total']['eligible_cells'],
                             44625)
            self.assertEqual(len(published['fold_rows']), expected['eligible'])
            result = m.summarize(published)
            self.assertEqual(result['feature_count'], 55)
            self.assertEqual(result['combination_count'], 255)
            self.assertFalse(result['cross_model_pooled_auc_computed'])
            self.assertFalse(result['winner_selection'])
            self.assertEqual(len(result['comparisons']['predeclared_BC_matched_comparisons']), 12)
            self.assertIn('No confidence interval', '\n'.join(result['scope_notes']))
            report_bindings = {
                'reporter': m.binding(Path(m.__file__).resolve()),
                'tests': m.binding(m.HERE / ('test_' + m.VERSION + '.py')),
                'specification': m.binding(m.SPEC, m.SPEC_SHA),
            }
            report = m.publish(published, Path(temporary).resolve() / 'report',
                               report_bindings)
            self.assertEqual(report['status'], 'committed_reporting_only_no_selection')
            report_commit = m.read(Path(report['commit']['path']))
            self.assertEqual(report_commit['products']['summary.json']['sha256'],
                             m.binding(Path(temporary).resolve() / 'report' / 'summary.json')['sha256'])

    def test_schedule_accounting_is_derived_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, schedule, _, _, _ = make_evaluation(temporary)
            broken = copy.deepcopy(schedule)
            broken['accounting']['eligible_combination_cap_cells'] += 1
            with self.assertRaisesRegex(ValueError, 'accounting replay'):
                m.schedule_audit(broken, reduced_expected(synthetic_rows()))

    def test_evaluator_pin_rejection_and_receipt_mutation_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation, expected, _, _, _, commit_entry = make_evaluation(temporary)
            published = m.inspect_evaluation(evaluation, commit_entry['sha256'], expected)
            with patch.object(m, 'EVALUATOR_SHA', '0' * 64):
                with self.assertRaisesRegex(ValueError, 'frozen source lineage'):
                    m.inspect_evaluation(evaluation, commit_entry['sha256'], expected)
            path = next((evaluation / 'models').iterdir())
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'inventory/hash'):
                m.inspect_evaluation(evaluation, commit_entry['sha256'], expected)
            self.assertEqual(published['contract_sha256'], m.value_hash(m.read(evaluation / 'contract.json')))


if __name__ == '__main__':
    unittest.main()
