"""Schedule-driven native30 seven-family development evaluation.

No historical v6 loader, splitting, cap selection, winner selection, refit or
threshold tuning. Default preflight prepares a contract without fitting. The
run requires a separate caller-SHA-pinned parent authorization. Production
input assembly must be independently approved before installing its code pin.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import pandas as pd

VERSION = 'evaluate_native30_v2'
STAGE = 'native30_seven_family_schedule_only_development_evaluation_v2'
FREEZE_VERSION = 'native30-evaluation-parent-freeze-v2'
PREPARER_SHA = '11f163914c6007e1ba75032da224eed565a06fc33e3cd0cbef8285c5673d3361'
PREPARER_NAME = 'prepare_native30_evaluation_inputs_v2'
BASE_SHA = 'd2ed30d9833fbe122f023de1223e44a40c1b4f95f99f63f0ba27647c87cc7232'
PLANNER_SHA = 'ff6965b3f9c834ee9d39bbf0f994595f44fa2e261d9c7c6b60a153cb9f72ce24'
IO_SHA = '165f28d4e553ee679772aa1ac47b31f9a485b94d3a8ac409f25595577abe395a'
SCHEDULE_SHA = '4e12300a7fe43b4f3024c1cd9e675c25b0ade3af25dded0d688b8d8f2fcab7a6'
SCHEDULE_COMMIT_SHA = '5d0e329d9ef0a794e32acbf5bee06a4a09a58b453e1741ba6232fbfb32347233'
EXPECTED = {'total': 3830, 'labels': {0: 1664, 1: 2166}, 'intended': 53340, 'primary': 36830, 'diagnostic': 14732}
SCOPE = {'winner_selection': False, 'threshold_tuning': False, 'full_cohort_refit': False,
         'historical_locked_or_pilot_scoring': False, 'M_predictor': False,
         'audio_or_neural_inference_performed': False, 'source_ranking': False}
FAMILIES = {
    'S': ['s8__' + x for x in ('tilt_1_5k_db_oct', 'hf_tilt_5_7p5k_db_oct', 'hf_ratio_5_7p5_db',
          'sibilance_ratio_5_7p5_db', 'hf_flatness_5_7p5', 'hf_entropy_5_7p5', 'hf_crest_5_7p5_db',
          'fakeprint_peak_density_5_7p5_per_khz', 'fakeprint_periodicity_5_7p5', 'hf_flux_5_7p5',
          'hf_frame_similarity_5_7p5', 'hf_power_sd_5_7p5_db', 'hf_mod_4_12_share_5_7p5',
          'sibilance_contrast_5_7p5_db', 'sibilance_burst_rate_5_7p5_hz')],
    'D': ['d__' + x for x in ('dynamics_span', 'dynamics_iqr', 'dynamics_adjacent_change')],
    'R': ['r__' + x for x in ('ibi_cv', 'tempo_tv', 'tempo_entropy')],
    'P': ['p__' + x for x in ('section_duration_cv', 'section_duration_entropy', 'section_bars_cv',
          'section_bars_offmode_fraction', 'section_duration_median', 'section_bars_median')],
    'F': ['F_' + name + '_' + region for name in ('phase_residual_cvar', 'group_delay_iqr_ms',
          'group_delay_cross_band_iqr_ms') for region in ('all', 'attack', 'sustain', 'decay')]
         + ['F_' + name + '_decay_minus_sustain' for name in ('phase_residual_cvar', 'group_delay_iqr_ms', 'group_delay_cross_band_iqr_ms')],
    'H': ['H_' + x for x in ('pitch_class_entropy_norm', 'pc_token_entropy_norm', 'chroma_path_change_median',
          'chroma_path_change_iqr', 'pc_path_step_median', 'pc_path_large_step_rate')],
    'SC': [f'SC_{lo}_{hi}hz_{metric}_median' for lo, hi in ((80, 500), (500, 2000), (2000, 6000))
           for metric in ('abs_iid_db', 'side_energy_fraction')],
}
FEATURES = sum(FAMILIES.values(), [])
META = ['id', 'source_group', 'label', 'role', 'group_id', 'component_id', 'duration_view_s',
        'native_sample_rate_hz', 'input_sha256', 'waveform_float32_sha256']
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, pin, name):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(), 'canonical code file required')
    require(isinstance(pin, str) and hashlib.sha256(path.read_bytes()).hexdigest() == pin, 'unapproved/changed code pin: ' + name)
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path, 'wrong loaded module origin')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


io = load_module(HERE / 'materialize_native30_new1695_v1.py', IO_SHA, 'materialize_native30_new1695_v1')
planner = load_module(HERE / 'plan_native30_evaluation_schedule_v1.py', PLANNER_SHA, 'plan_native30_evaluation_schedule_v1')
BASE = load_module(HERE / 'frozen_evaluate_expanded_20260905.py', BASE_SHA, '_native30_fixed_evaluation_core')


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float):
        require(not math.isinf(value), 'infinite numerical result')
        return None if math.isnan(value) else value
    return value


def runtime_snapshot():
    require(sys.flags.optimize == 0, 'optimized Python not authorized')
    require(platform.python_version() == '3.11.15' and np.__version__ == '1.26.4', 'frozen numerical Python/NumPy versions')
    files = {}
    for module in (np, pd):
        root = Path(module.__file__).resolve().parent
        paths = list(root.rglob('*'))
        libraries = root.with_name(root.name + '.libs')
        if libraries.is_dir():
            paths += list(libraries.rglob('*'))
        for path in paths:
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}:
                files[str(path)] = io.file_binding(path)
    return {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'executable': io.file_binding(Path(sys.executable).resolve()), 'package_files': files,
            'module_origins': {m.__name__: str(Path(m.__file__).resolve()) for m in (np, pd, BASE)},
            'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}


def check_bindings(contract):
    io.recheck(contract['bindings'])
    if contract.get('package_commit'):
        # Named COMMIT bindings alone cannot detect a changed descendant whose
        # COMMIT file was left untouched. Reuse the assembler's read-only deep
        # product/receipt reconstruction at both ends of evaluation/resumption.
        require(PREPARER_SHA is not None, 'assembler approval pin not installed')
        assembler = load_module(HERE / (PREPARER_NAME + '.py'), PREPARER_SHA, PREPARER_NAME)
        entry = contract['package_commit']
        proof = assembler.verify_package(Path(entry['path']).parent, entry['sha256'])
        require(proof['commit'] == entry, 'deep prepared-package replay changed')
    if contract.get('runtime'):
        require(runtime_snapshot() == contract['runtime'], 'frozen numerical runtime changed')


def table_payload(table):
    return clean(table.to_dict(orient='records'))


def make_table(metadata, features, cohort_rows, plan_rows):
    """Pure API: permits reduced synthetic fixtures, never a CLI size override."""
    ids = [r['id'] for r in metadata]
    require(ids == sorted(set(ids)) == [r['id'] for r in features], 'exact sorted unique metadata/feature IDs required')
    cohort = planner.indexed(cohort_rows)
    origins = planner.indexed(plan_rows)
    require(set(ids) == set(cohort), 'prepared input/cohort roster mismatch')
    for meta, feature in zip(metadata, features):
        row, origin = cohort[meta['id']], origins[meta['id']]
        require(set(meta) == set(META) and all(meta[k] == row[k] for k in planner.IDENTITY), 'metadata identity/schema mismatch')
        require(meta['role'] == 'development' and meta['duration_view_s'] == 30
                and meta['input_sha256'] == row['input']['sha256']
                and meta['waveform_float32_sha256'] == row['waveform_float32_sha256'], 'native30 provenance mismatch')
        require(io.value_hash(origin) == row['origin_plan_row_sha256']
                and type(meta['native_sample_rate_hz']) is int
                and meta['native_sample_rate_hz'] == origin['source_origin']['sample_rate_hz'], 'native-rate origin-plan SHA mismatch')
        require(set(feature) == {'id', *FEATURES} and all(feature[c] is None or
                (type(feature[c]) in (int, float) and math.isfinite(feature[c])) for c in FEATURES), 'exact54 finite-or-null features required')
    table = pd.DataFrame(metadata).rename(columns={'id': '__id', 'label': '__label', 'source_group': '__source',
                       'group_id': '__group', 'component_id': '__component', 'role': '__role'})
    require(set(table['__label']) == {'0', '1'}, 'both immutable binary labels required')
    table['__label'] = table['__label'].astype(int)
    require(table.groupby('__source')['__label'].nunique().max() == 1, 'source label conflict')
    # Global groups deliberately MAY cross source and label; never rename them.
    for column in FEATURES:
        table[column] = [float('nan') if r[column] is None else r[column] for r in features]
    return table


def replay_schedule(schedule, rows, screen):
    replayed = planner.make_schedule(rows, screen)
    require(all(schedule.get(k) == value for k, value in replayed.items()), 'committed schedule differs from pinned metadata replay')
    require(len({r['schedule_uid'] for r in schedule['cap_schedules']}) == len(schedule['cap_schedules']), 'duplicate schedule UID')
    return replayed


def schedule_counts(schedule):
    folds = {r['fold_uid']: r for r in schedule['fold_cells']}
    caps = {r['schedule_uid']: r for r in schedule['cap_schedules']}
    result = {}
    for role in ('primary', 'diagnostic'):
        cells = [x for x in schedule['combination_schedule_cells'] if x['analysis_role'] == role]
        valid = [x for x in cells if x['status'] == 'eligible_metadata_cell']
        result[role] = {'intended': len(cells), 'eligible': len(valid), 'omitted': len(cells) - len(valid),
                        'prediction_rows': sum(folds[caps[x['schedule_uid']]['fold_uid']]['test']['rows'] for x in valid)}
    return result


def build_context(package, package_commit_sha, output):
    require(PREPARER_SHA is not None, 'assembler approval pin not installed')
    assembler = load_module(HERE / (PREPARER_NAME + '.py'), PREPARER_SHA, PREPARER_NAME)
    package = planner.path_checked(package)
    bundle = assembler.verify_package(package, package_commit_sha)
    assembled, bindings = bundle['contract'], {}
    require(assembled['status'] == 'assembled_not_evaluated_not_authorized' and assembled['expected_count'] == EXPECTED['total']
            and assembled['family_config'] == FAMILIES and assembled['feature_names'] == FEATURES, 'native30 assembled package contract')
    for entry in assembled['bindings'].values():
        require(planner.binding(entry['path']) == entry, 'assembled upstream binding changed')
        bindings[entry['path']] = entry
    require(planner.binding(package / 'COMMIT.json', package_commit_sha) == bundle['commit'], 'package COMMIT mismatch')
    for path in package.iterdir():
        bindings[str(path)] = planner.binding(path)
    refs = assembled['bindings']
    require(refs['schedule_draft']['sha256'] == SCHEDULE_SHA and refs['schedule_commit']['sha256'] == SCHEDULE_COMMIT_SHA
            and refs['origin_plan']['sha256'] == planner.PLAN_SHA and refs['screen']['sha256'] == planner.SCREEN_SHA, 'fixed schedule/plan/screen pins')
    cohort, plan, screen, schedule = [io.read_json(refs[k]['path']) for k in ('cohort_contract', 'origin_plan', 'screen', 'schedule_draft')]
    schedule_commit = io.read_json(refs['schedule_commit']['path'])
    require(schedule_commit['status'] == 'committed_metadata_draft_not_evaluation_authorization'
            and schedule_commit['products'] == {'schedule_draft.json': refs['schedule_draft']}
            and schedule_commit['draft_sha256'] == SCHEDULE_SHA, 'schedule COMMIT binding/status')
    require(schedule['prepared_contract_sha256'] == refs['cohort_contract']['sha256'] == io.value_hash(cohort), 'shared prepared cohort contract')
    for entry in schedule['input_bindings'].values():
        require(planner.binding(entry['path']) == entry, 'schedule provenance binding changed')
        bindings[entry['path']] = entry
    rows = planner.validate_prepared_contract(cohort, plan, screen)
    replay_schedule(schedule, rows, screen)
    table = make_table(bundle['metadata'], bundle['features'], rows, plan['rows'])
    require(len(table) == EXPECTED['total'] and dict(Counter(table['__source'])) == planner.SOURCES
            and dict(Counter(table['__label'])) == EXPECTED['labels'], 'actual native30 exact roster')
    accounting = schedule_counts(schedule)
    require(sum(v['intended'] for v in accounting.values()) == EXPECTED['intended']
            and accounting['primary']['eligible'] == EXPECTED['primary']
            and accounting['diagnostic']['eligible'] == EXPECTED['diagnostic'], 'actual committed schedule accounting')
    for name, pin in ((VERSION + '.py', None), ('test_' + VERSION + '.py', None),
                      ('frozen_evaluate_expanded_20260905.py', BASE_SHA), ('plan_native30_evaluation_schedule_v1.py', PLANNER_SHA),
                      ('materialize_native30_new1695_v1.py', IO_SHA), (PREPARER_NAME + '.py', PREPARER_SHA)):
        path = HERE / name
        bindings[str(path)] = planner.binding(path, pin)
    output = planner.path_checked(output)
    committed_roots = {Path(p).parent for p in bindings if Path(p).name in {'COMMIT.json', 'contract.json'}}
    require(output.parent.is_dir() and not any(Path(p).is_relative_to(output) for p in bindings)
            and not output.is_relative_to(package) and not any(output.is_relative_to(p) for p in committed_roots),
            'output overlaps evaluation provenance')
    contract = {'version': VERSION, 'stage': STAGE, 'package_commit': bundle['commit'], 'bindings': bindings,
                'output_root': str(output), 'runtime': runtime_snapshot(), 'expected_count': len(table), 'families': FAMILIES,
                'feature_names': FEATURES, 'table_sha256': io.value_hash(table_payload(table)), 'accounting': accounting,
                'schedule_document': refs['schedule_draft'], 'schedule_sha256': io.value_hash(schedule),
                'schedule_protocol': schedule['protocol'], 'ridge': 10.0, 'threshold': 0.5, 'prediction_link': 'raw_identity_unclipped',
                'weighting': 'equal classes/sources within class/groups within source/items within group; sum training n',
                'preprocessing': 'unchanged pinned BASE.fit_model; train-only median/weighted scale; no availability filtering',
                'coefficient_validation': 'no refit; infinity residual <= 128*float64_eps*design_columns*max(1, norm(A,inf)*norm(beta,inf), norm(b,inf))',
                'validation_scope': 'in-process transform/equation/prediction replay; not an independent full-publication numerical audit',
                'comparison_scope': 'new cohort/duration/purging/caps; no causal historical outcome subtraction', **SCOPE}
    check_bindings(contract)
    return contract, table, schedule


def authorize(contract, path, expected_sha):
    require(path is not None and expected_sha is not None, 'separate parent fitting freeze required')
    path = planner.path_checked(path)
    binding = planner.binding(path, expected_sha)
    frozen = io.read_json(path)
    require(frozen.get('version') == FREEZE_VERSION and frozen.get('status') == 'parent_frozen_for_native30_fold_only_fitting'
            and frozen.get('fitting_authorized') is True and frozen.get('scoring_authorized') is True
            and frozen.get('stage') == STAGE and frozen.get('contract') == contract
            and frozen.get('contract_sha256') == io.value_hash(contract)
            and frozen.get('independent_review', {}).get('approved') is True
            and frozen.get('independent_review', {}).get('reviewer') == 'root'
            and all(frozen.get(k) == v for k, v in SCOPE.items()), 'parent fit freeze scope/contract mismatch')
    return {'status': 'parent_frozen_verified', 'contract_sha256': io.value_hash(contract), 'binding': binding}


def resolve_task(cell, schedule):
    caps = {r['schedule_uid']: r for r in schedule['cap_schedules']}
    folds = {r['fold_uid']: r for r in schedule['fold_cells']}
    cap = caps[cell['schedule_uid']]
    fold = folds[cap['fold_uid']]
    return {**cell, 'quantity': cap['quantity'], 'fold_uid': cap['fold_uid'], 'fold_type': cap['fold_type'],
            'heldout_source': cap['heldout_source'], 'opposite_group_fold': cap['opposite_group_fold'],
            'train_ids': cap['train']['ids'], 'test_ids': fold['test']['ids'],
            'train_id_set_sha256': cap['train']['id_set_sha256'], 'test_id_set_sha256': fold['test']['id_set_sha256'],
            'omission_reasons': cap['omission_reasons']}


def task_uid(task, contract_sha):
    return io.value_hash({'contract_sha256': contract_sha, 'schedule_uid': task['schedule_uid'],
                          'combination': task['combination'], 'feature_mode': task['feature_mode']})[:32]


def task_tables(table, task):
    indexed = table.set_index('__id', drop=False)
    train = indexed.loc[task['train_ids']].reset_index(drop=True)
    test = indexed.loc[task['test_ids']].reset_index(drop=True)
    require(planner.id_set_hash(train['__id']) == task['train_id_set_sha256']
            and planner.id_set_hash(test['__id']) == task['test_id_set_sha256'], 'task row-reference hash mismatch')
    require(set(train['__label']) == set(test['__label']) == {0, 1}
            and not set(train['__id']) & set(test['__id'])
            and not set(train['__group']) & set(test['__group'])
            and not set(train['__component']) & set(test['__component']), 'task class/group/component leakage')
    return train, test


def columns_for(task):
    require(task['combination'] in planner.COMBINATIONS and task['feature_mode'] in
            (planner.PRIMARY_MODE, *planner.DIAGNOSTIC_MODES), 'unknown combination/feature mode')
    require(task['feature_mode'] == planner.PRIMARY_MODE or task['quantity'] == 'all', 'diagnostics only at all-cap')
    return sum((FAMILIES[key] for key in task['combination'].split('+')), [])


def validate_model(model, train, task):
    columns = columns_for(task)
    dims = len(columns) * (2 if task['feature_mode'] == planner.PRIMARY_MODE else 1)
    require(set(model) == {'columns', 'medians', 'mean', 'scale', 'coefficients_with_intercept', 'ridge', 'threshold',
            'feature_mode', 'training_rows', 'training_source_counts', 'weighting', 'missing_value_policy',
            'observed_fraction_by_column'}, 'unchanged core model schema required')
    require(model['columns'] == columns and model['feature_mode'] == task['feature_mode']
            and model['ridge'] == 10.0 and model['threshold'] == 0.5 and model['training_rows'] == len(train)
            and model['training_source_counts'] == BASE.source_key(train).value_counts().sort_index().to_dict(), 'model frozen numerical metadata mismatch')
    for key, count in (('medians', len(columns)), ('mean', dims), ('scale', dims), ('coefficients_with_intercept', dims + 1)):
        values = np.asarray(model[key], dtype=float)
        require(values.shape == (count,) and np.isfinite(values).all(), 'malformed model parameter: ' + key)
    require(np.all(np.asarray(model['scale']) > 0), 'nonpositive model scale')
    require(set(model['observed_fraction_by_column']) == set(columns), 'model observed-column schema')
    for column in columns:
        finite = np.isfinite(train[column].to_numpy(float))
        require(model['observed_fraction_by_column'][column] == float(finite.mean()), 'training missingness mismatch')
    # The pinned helper reproduces train-only transforms without fitting. It
    # prepares values+indicators; selecting either half gives the diagnostic
    # design without changing any numerical producer or solving again.
    prepared = BASE.prepare_training(train, columns)
    count = len(columns)
    indices = (np.arange(2 * count) if task['feature_mode'] == planner.PRIMARY_MODE else
               np.arange(count) if task['feature_mode'] == 'median_only' else np.arange(count, 2 * count))
    require(np.array_equal(np.asarray(model['medians']), prepared['medians'])
            and np.array_equal(np.asarray(model['mean']), prepared['mean'][indices])
            and np.array_equal(np.asarray(model['scale']), prepared['scale'][indices]), 'train-only transform replay mismatch')
    design = np.column_stack((np.ones(len(train)), prepared['z'][:, indices]))
    weights = np.sqrt(prepared['weights'])
    weighted = design * weights[:, None]
    penalty = np.eye(design.shape[1]) * BASE.RIDGE
    penalty[0, 0] = 0.0
    normal = weighted.T @ weighted + penalty
    target = weighted.T @ (prepared['y'] * weights)
    coefficients = np.asarray(model['coefficients_with_intercept'])
    residual = float(np.linalg.norm(normal @ coefficients - target, ord=np.inf))
    magnitude = max(1.0, float(np.linalg.norm(normal, ord=np.inf) * np.linalg.norm(coefficients, ord=np.inf)),
                    float(np.linalg.norm(target, ord=np.inf)))
    # Roundoff-only audit tolerance; not an optimization/selection tolerance.
    tolerance = float(128 * np.finfo(np.float64).eps * design.shape[1] * magnitude)
    require(residual <= tolerance, 'fixed ridge normal-equation residual mismatch')
    return {'status': 'training_transforms_and_ridge_equations_verified_without_refit',
            'normal_equation_residual_infinity': residual, 'roundoff_tolerance': tolerance,
            'normal_equation_dimension': design.shape[1]}


def prediction_evidence(test, model):
    scores = BASE.predict(test, model)
    require(scores.shape == (len(test),) and np.isfinite(scores).all(), 'nonfinite/wrong-shape raw predictions')
    predictions = []
    for row, score in zip(test.to_dict(orient='records'), scores):
        predictions.append({'id': row['__id'], 'label': int(row['__label']), 'source_group': row['__source'],
                            'group_id': row['__group'], 'component_id': row['__component'], 'role': row['__role'],
                            'score': float(score), 'threshold': 0.5, 'predicted_label': int(score >= 0.5)})
    return predictions


def metric_evidence(predictions, task):
    scored = pd.DataFrame(predictions)
    source = []
    for name, part in scored.groupby('source_group', sort=True):
        positive = part['predicted_label'].to_numpy()
        rates = part.groupby('group_id')['predicted_label'].mean().to_list()
        label = int(part['label'].iloc[0])
        source.append({'source_group': name, 'label': label, 'rows': len(part), 'groups': len(rates),
                       'components': part['component_id'].nunique(), 'recording_positive_rate': float(positive.mean()),
                       'equal_group_positive_rate': math.fsum(rates) / len(rates),
                       **BASE.metrics(part['label'].to_numpy(), part['score'].to_numpy())})
    humans = sorted(scored.loc[scored['label'] == 0, 'source_group'].unique())
    generators = sorted(scored.loc[scored['label'] == 1, 'source_group'].unique())
    pairs = []
    for human in humans:
        for generator in generators:
            if task['fold_type'] == 'human_source_holdout' and human != task['heldout_source']:
                continue
            if task['fold_type'] == 'generator_holdout' and generator != task['heldout_source']:
                continue
            part = scored[scored['source_group'].isin([human, generator])]
            pairs.append({'human_source': human, 'ai_source': generator, 'rows': len(part),
                          **BASE.metrics(part['label'].to_numpy(), part['score'].to_numpy())})
    return clean({'per_source': source, 'source_pairs': pairs,
                  'within_fold_pooled_descriptive': BASE.metrics(scored['label'].to_numpy(), scored['score'].to_numpy())})


def model_receipt(contract, table, task):
    train, test = task_tables(table, task)
    model = BASE.fit_model(train, columns_for(task), feature_mode=task['feature_mode'])
    numerical_audit = validate_model(model, train, task)
    predictions = prediction_evidence(test, model)
    payload = {'status': 'fold_only_model_and_predictions_not_selected', 'contract_sha256': io.value_hash(contract),
               'task': task, 'task_sha256': io.value_hash(task), 'model': clean(model),
               'model_sha256': io.value_hash(clean(model)), 'predictions': predictions,
               'predictions_sha256': io.value_hash(predictions), 'metrics': metric_evidence(predictions, task),
               'training_numerical_audit': numerical_audit, **SCOPE}
    return payload


def verify_model_receipt(path, contract, table, task):
    require(path.is_file() and not path.is_symlink(), 'regular model receipt required')
    envelope = io.read_json(path)
    require(set(envelope) == {'payload', 'receipt_sha256'} and io.digest(path) == io.value_hash(envelope)
            and envelope['receipt_sha256'] == io.value_hash(envelope['payload']), 'canonical model receipt hash')
    receipt = envelope['payload']
    require(set(receipt) == {'status', 'contract_sha256', 'task', 'task_sha256', 'model', 'model_sha256',
            'predictions', 'predictions_sha256', 'metrics', 'training_numerical_audit', *SCOPE}, 'model receipt schema')
    require(receipt['contract_sha256'] == io.value_hash(contract) and receipt['task'] == task
            and receipt['task_sha256'] == io.value_hash(task) and all(receipt[k] == v for k, v in SCOPE.items())
            and receipt['status'] == 'fold_only_model_and_predictions_not_selected', 'model receipt lineage/scope')
    train, test = task_tables(table, task)
    require(receipt['training_numerical_audit'] == validate_model(receipt['model'], train, task), 'training numerical audit changed')
    require(receipt['model_sha256'] == io.value_hash(receipt['model']), 'model parameter hash')
    predictions = prediction_evidence(test, receipt['model'])
    require(receipt['predictions'] == predictions and receipt['predictions_sha256'] == io.value_hash(predictions), 'saved prediction replay mismatch')
    require(receipt['metrics'] == metric_evidence(predictions, task), 'saved metric replay mismatch')
    return receipt


def input_guard(contract, table, schedule, authorization):
    check_bindings(contract)
    require(io.value_hash(table_payload(table)) == contract['table_sha256']
            and io.value_hash(schedule) == contract['schedule_sha256'], 'in-memory table/schedule changed')
    require(authorization.get('status') == 'parent_frozen_verified'
            and authorization.get('contract_sha256') == io.value_hash(contract), 'verified parent fitting authorization required')
    require(planner.binding(authorization['binding']['path']) == authorization['binding'], 'parent freeze changed')
    require(authorize(contract, authorization['binding']['path'], authorization['binding']['sha256']) == authorization, 'parent authorization replay changed')


def inventory(output, uids):
    model_names = {uid + '.json' for uid in uids}
    require({p.name for p in output.iterdir()} <= {'writer.lock', 'contract.json', 'authorization.json', 'models',
            'failures', 'runs', 'omitted.json', 'manifest.json', 'COMMIT.json'}, 'unexpected evaluation root product')
    for path in output.rglob('*'):
        require(not path.is_symlink(), 'evaluation symlink forbidden')
        if path.parent == output:
            require(path.is_dir() if path.name in {'models', 'failures', 'runs'} else path.is_file(), 'evaluation root entry type')
        elif path.parent == output / 'models':
            require(path.is_file() and path.name in model_names, 'unknown model receipt')
        elif path.parent in {output / 'failures', output / 'runs'}:
            require(path.is_file() and path.suffix == '.json' and len(path.stem) == 32
                    and set(path.stem) <= set('0123456789abcdef'), 'unexpected failure/run evidence')
        else:
            raise ValueError('unexpected nested evaluation product')


def products(output):
    return {str(p.relative_to(output)): io.file_binding(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name not in {'COMMIT.json', 'writer.lock'}}


def commit_record(contract_sha, bound, count):
    return {'version': VERSION, 'status': 'committed_fold_only_evaluation_not_model_selection',
            'contract_sha256': contract_sha, 'model_instances': count, 'products': bound,
            'all_input_output_runtime_bindings_end_rehashed': True, **SCOPE}


def run_context(contract, table, schedule, authorization):
    """Reduced contracts permitted only through Python for synthetic tests."""
    input_guard(contract, table, schedule, authorization)
    output = planner.path_checked(contract['output_root'])
    sha = io.value_hash(contract)
    tasks = [resolve_task(c, schedule) for c in schedule['combination_schedule_cells']]
    valid = [task for task in tasks if task['status'] == 'eligible_metadata_cell']
    omitted = [task for task in tasks if task['status'] == 'omitted']
    require(len(valid) + len(omitted) == len(tasks) and schedule_counts(schedule) == contract['accounting'], 'task accounting changed')
    uids = [task_uid(task, sha) for task in valid]
    require(len(uids) == len(set(uids)), 'duplicate model task UID')
    with io.writer_lock(output):
        contract_path = output / 'contract.json'
        if contract_path.exists():
            require(io.read_json(contract_path) == contract and io.digest(contract_path) == sha, 'resumed evaluator contract conflict')
        else:
            require({p.name for p in output.iterdir()} == {'writer.lock'}, 'nonempty unbound evaluator output')
            io.write_new(contract_path, contract)
        for name in ('models', 'failures', 'runs'):
            (output / name).mkdir(exist_ok=True)
        for name, value in (('authorization.json', authorization), ('omitted.json', omitted)):
            path = output / name
            if path.exists():
                require(io.read_json(path) == value and io.digest(path) == io.value_hash(value), 'retained metadata evidence conflict')
            else:
                io.write_new(path, value)
        inventory(output, uids)
        committed, commit = (output / 'COMMIT.json').exists(), output / 'COMMIT.json'
        if committed:
            require(io.read_json(commit) == commit_record(sha, products(output), len(valid))
                    and io.digest(commit) == io.value_hash(io.read_json(commit)), 'existing evaluation COMMIT changed')
        completed, failed = [], []
        run_id = uuid.uuid4().hex
        for index, (task, uid) in enumerate(zip(valid, uids), 1):
            path = output / 'models' / (uid + '.json')
            try:
                if path.exists() or path.is_symlink():
                    verify_model_receipt(path, contract, table, task)
                else:
                    require(not committed, 'missing committed model')
                    payload = model_receipt(contract, table, task)
                    io.write_new(path, {'payload': payload, 'receipt_sha256': io.value_hash(payload)})
                completed.append(uid)
            except Exception as exc:
                failure = {'contract_sha256': sha, 'model_uid': uid, 'task': task,
                           'exception': type(exc).__name__, 'message': str(exc), 'run_id': run_id}
                failed.append(failure)
                if not committed:
                    io.write_new(output / 'failures' / (uuid.uuid4().hex + '.json'), failure)
            if index % 100 == 0 or index == len(valid):
                print(io.canonical({'event': 'native30_evaluation_progress', 'reviewed': index,
                                   'planned': len(valid), 'valid_models': len(completed), 'failed': len(failed)}).decode().strip(), flush=True)
        input_guard(contract, table, schedule, authorization)
        if committed:
            require(not failed and io.read_json(commit) == commit_record(sha, products(output), len(valid)), 'committed evaluator verification failed')
            return {'status': 'verified_existing_COMMIT_no_refit', 'model_instances': len(valid), 'commit_sha256': io.digest(commit)}
        summary = {'status': 'partial_no_COMMIT' if failed else 'all_fold_models_complete_not_selected',
                   'contract_sha256': sha, 'run_id': run_id, 'completed': len(completed), 'failed': len(failed),
                   'expected': len(valid), 'omitted': len(omitted), **SCOPE}
        io.write_new(output / 'runs' / (run_id + '.json'), summary)
        if failed:
            return summary
        # Re-read all newly written receipts; replay predictions without refitting.
        for task, uid in zip(valid, uids):
            verify_model_receipt(output / 'models' / (uid + '.json'), contract, table, task)
        manifest = {'version': VERSION, 'contract_sha256': sha, 'accounting': contract['accounting'],
                    'model_instances': len(valid), 'omitted_instances': len(omitted),
                    'training_transform_equation_prediction_replay_performed': True,
                    'independent_full_publication_numerical_audit_performed': False,
                    'model_receipts': {uid: io.file_binding(output / 'models' / (uid + '.json')) for uid in uids}, **SCOPE}
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            require(io.read_json(manifest_path) == manifest, 'retained evaluation manifest conflict')
        else:
            io.write_new(manifest_path, manifest)
        input_guard(contract, table, schedule, authorization)
        inventory(output, uids)
        require({p.stem for p in (output / 'models').iterdir()} == set(uids), 'incomplete final model inventory')
        bound = products(output)
        require(products(output) == bound, 'evaluation product changed during final rehash')
        io.write_new(commit, commit_record(sha, bound, len(valid)))
        return {**summary, 'commit_sha256': io.digest(commit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--package-commit-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    parser.add_argument('--frozen')
    parser.add_argument('--frozen-sha256')
    args = parser.parse_args()
    contract, table, schedule = build_context(args.package, args.package_commit_sha256, args.output)
    if args.mode == 'preflight':
        result = {'status': 'preflight_no_fitting', 'contract': contract, 'contract_sha256': io.value_hash(contract), 'classifier_fits': 0}
    else:
        authorization = authorize(contract, args.frozen, args.frozen_sha256)
        result = run_context(contract, table, schedule, authorization)
    print(io.canonical(result).decode().strip())
    return 1 if result['status'] == 'partial_no_COMMIT' else 0


if __name__ == '__main__':
    raise SystemExit(main())
