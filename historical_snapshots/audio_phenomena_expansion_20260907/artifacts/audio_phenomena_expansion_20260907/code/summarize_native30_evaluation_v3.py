"""Reporting only: complete native30 evaluation receipts to JSON/English tables.

No imports of numerical fitting/feature code; no refit, selection, threshold
tuning, cross-model pooled AUC, confidence interval or historical causal delta.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

VERSION = 'summarize_native30_evaluation_v3'
EVALUATOR_SHA = 'c2544fe4ce98d88b906f502a576f788da46a4e28117ea7f3ce41802495fe7d1c'
EVALUATOR_TEST_SHA = '32a9a5c3b4f645ad624d1f40720c66133e22727c8a1a509c4b3842f3b4fc411c'
SPEC_SHA = 'a8a123dcd414336ae7b16917ba7acf15a33e37220dfd071db5135d75d98a205f'
HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / 'NATIVE30_REPORTING_CONTRACT_EN.md'
FAMILIES = ('S', 'D', 'R', 'P', 'F', 'H', 'SC')
COMBINATIONS = ['+'.join(c) for n in range(1, 8) for c in itertools.combinations(FAMILIES, n)]
CAPS = (25, 50, 100, 200, 'all')
MODES = ('values_plus_missing', 'median_only', 'missingness_only')
PROTOCOLS = ('human_source_holdout', 'generator_holdout', 'ordinary_group_holdout_descriptive')
REFERENCE = 'S+D+R+P'
SOURCES = {'ACE-Step': 400, 'FMA': 354, 'HeartMuLa': 366, 'MTG-Jamendo': 500,
           'Mureka_v9': 500, 'Suno': 400, 'Udio': 500, 'human_maestro_v3': 300,
           'human_medleydb': 168, 'human_moisesdb': 239, 'human_saraga_hindustani_v1': 103}
EXPECTED = {'rows': 3830, 'sources': SOURCES, 'labels': {0: 1664, 1: 2166}, 'buckets': 5,
            'caps': CAPS, 'combinations': COMBINATIONS, 'intended': 53340, 'eligible': 51562,
            'primary': 36830, 'diagnostic': 14732}
PRODUCER_SCOPE = {'winner_selection': False, 'threshold_tuning': False, 'full_cohort_refit': False,
                  'historical_locked_or_pilot_scoring': False, 'M_predictor': False,
                  'audio_or_neural_inference_performed': False, 'source_ranking': False}
SCOPE = {'classifier_fits': 0, 'models_rerun': 0, 'threshold_tuning': False, 'winner_selection': False,
         'confidence_intervals_claimed': False, 'cross_model_pooled_auc_computed': False,
         'historical_causal_deltas_computed': False, 'independent_publication_numerical_audit_performed': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def id_hash(ids):
    return hashlib.sha256(''.join(x + '\n' for x in sorted(set(ids))).encode()).hexdigest()


def safe(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.resolve() == path and not path.is_symlink(), 'canonical unredirected path required')
    return path


def binding(path, expected=None):
    path = safe(path)
    require(path.is_file(), 'regular bound file required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(expected is None or digest == expected, 'file SHA mismatch: ' + str(path))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def read(path, canonical_required=True):
    raw = Path(path).read_bytes()
    def invalid(value):
        raise ValueError('nonfinite JSON token: ' + value)
    value = json.loads(raw, parse_constant=invalid)
    require(not canonical_required or raw == canonical(value), 'noncanonical JSON: ' + str(path))
    return value


def write_new(path, data):
    path = Path(path)
    data = data if isinstance(data, bytes) else canonical(data)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def auc(predictions):
    """Single fitted model only; exact average-tie-rank Mann–Whitney AUC."""
    ordered = sorted((r['score'], r['label']) for r in predictions)
    n1 = sum(label for _, label in ordered)
    n0 = len(ordered) - n1
    if not n0 or not n1:
        return None
    positive_ranks = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank = (index + 1 + end) / 2.0
        positive_ranks += rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_ranks - n1 * (n1 + 1) / 2) / (n0 * n1)


def correct(row):
    return int(row['predicted_label'] == row['label'])


def recording_recalls(predictions):
    recalls = {str(label): mean(correct(r) for r in predictions if r['label'] == label) for label in (0, 1)}
    return {'human_recall': recalls['0'], 'ai_recall': recalls['1'],
            'balanced_accuracy': mean(recalls.values()) if None not in recalls.values() else None}


def evaluation_inventory(root):
    required = {'writer.lock', 'contract.json', 'authorization.json', 'models', 'failures', 'runs',
                'omitted.json', 'manifest.json', 'COMMIT.json'}
    require({p.name for p in root.iterdir()} == required, 'complete evaluator root inventory required')
    found = {}
    for path in sorted(root.rglob('*')):
        safe(path)
        relative = path.relative_to(root)
        if path.is_dir():
            require(len(relative.parts) == 1 and relative.name in {'models', 'failures', 'runs'}, 'unexpected evaluation directory')
            continue
        require(path.is_file(), 'nonregular evaluation product')
        if len(relative.parts) == 2:
            require(relative.parts[0] in {'models', 'failures', 'runs'} and re.fullmatch(r'[0-9a-f]{32}\.json', relative.name),
                    'unexpected evaluation receipt filename')
        else:
            require(len(relative.parts) == 1, 'unexpected nested evaluation file')
        if relative.name not in {'writer.lock', 'COMMIT.json'}:
            found[str(relative)] = binding(path)
    return found


def task_key(task):
    return (task['fold_type'], task['heldout_source'], task['quantity'], task['combination'], task['feature_mode'])


def model_uid(task, contract_sha):
    return value_hash({'contract_sha256': contract_sha, 'schedule_uid': task['schedule_uid'],
                       'combination': task['combination'], 'feature_mode': task['feature_mode']})[:32]


def validate_task(task, expected):
    require(task['fold_type'] in PROTOCOLS and task['combination'] in expected['combinations']
            and task['quantity'] in expected['caps'] and task['feature_mode'] in MODES
            and type(task['opposite_group_fold']) is int and 0 <= task['opposite_group_fold'] < expected['buckets'], 'unknown reporting task')
    require(task['feature_mode'] == MODES[0] or task['quantity'] == 'all', 'diagnostic outside all-cap')
    require(task['analysis_role'] == ('primary' if task['feature_mode'] == MODES[0] else 'diagnostic'), 'mode/analysis-role mismatch')
    require(task['status'] in {'eligible_metadata_cell', 'omitted'}, 'unknown task status')
    require(isinstance(task['omission_reasons'], list)
            and bool(task['omission_reasons']) == (task['status'] == 'omitted'), 'omission reasons/status mismatch')
    identity = {k: task[k] for k in ('fold_type', 'heldout_source', 'opposite_group_fold')}
    require(task['fold_uid'] == value_hash(identity)[:24] and re.fullmatch('[0-9a-f]{24}', task['schedule_uid']), 'fold/schedule identity mismatch')
    for prefix in ('train', 'test'):
        ids = task[prefix + '_ids']
        require(isinstance(ids, list) and all(isinstance(x, str) and x for x in ids)
                and ids == sorted(set(ids)) and task[prefix + '_id_set_sha256'] == id_hash(ids), 'task ordered unique ID hash mismatch')
    require(not set(task['train_ids']) & set(task['test_ids']), 'task train/test ID overlap')


def receipt(path, entry, contract_sha, expected):
    require(binding(path) == entry, 'receipt binding changed')
    envelope = read(path)
    require(set(envelope) == {'payload', 'receipt_sha256'} and envelope['receipt_sha256'] == value_hash(envelope['payload']), 'receipt payload hash mismatch')
    record = envelope['payload']
    task = record['task']
    validate_task(task, expected)
    require(record['status'] == 'fold_only_model_and_predictions_not_selected' and task['status'] == 'eligible_metadata_cell'
            and record['contract_sha256'] == contract_sha and record['task_sha256'] == value_hash(task)
            and all(record.get(k) == v for k, v in PRODUCER_SCOPE.items()), 'receipt producer lineage/scope mismatch')
    require(path.stem == model_uid(task, contract_sha) and record['model_sha256'] == value_hash(record['model'])
            and record['predictions_sha256'] == value_hash(record['predictions']), 'model/prediction evidence hash mismatch')
    require(record['model']['threshold'] == 0.5 and record['model']['ridge'] == 10.0
            and record['model']['feature_mode'] == task['feature_mode']
            and record['model']['training_rows'] == len(task['train_ids']), 'fixed model reporting metadata mismatch')
    audit = record['training_numerical_audit']
    require(audit['status'] == 'training_transforms_and_ridge_equations_verified_without_refit'
            and 0 <= audit['normal_equation_residual_infinity'] <= audit['roundoff_tolerance']
            and math.isfinite(audit['roundoff_tolerance']), 'producer training-validation evidence absent')
    predictions = record['predictions']
    require([p['id'] for p in predictions] == task['test_ids'], 'prediction/task exact ordered IDs mismatch')
    for prediction in predictions:
        require(set(prediction) == {'id', 'label', 'source_group', 'group_id', 'component_id', 'role', 'score', 'threshold', 'predicted_label'}, 'prediction schema')
        require(type(prediction['label']) is int and prediction['label'] in (0, 1) and prediction['role'] == 'development'
                and type(prediction['score']) in (int, float) and math.isfinite(prediction['score'])
                and prediction['threshold'] == 0.5 and type(prediction['predicted_label']) is int
                and prediction['predicted_label'] == int(prediction['score'] >= 0.5), 'prediction label/score/threshold mismatch')
        require(all(isinstance(prediction[k], str) and prediction[k] for k in ('id', 'source_group', 'group_id', 'component_id')), 'prediction identity missing')
    require({r['label'] for r in predictions} == {0, 1}, 'eligible fold test class absent')
    require(record['metrics']['within_fold_pooled_descriptive']['roc_auc'] == auc(predictions), 'single-model AUC receipt mismatch')
    return record


def inspect_evaluation(root, commit_sha, expected=EXPECTED):
    """Reduced expected rosters are Python synthetic tests only, never CLI."""
    root = safe(root)
    commit_entry = binding(root / 'COMMIT.json', commit_sha)
    commit, contract = read(root / 'COMMIT.json'), read(root / 'contract.json')
    contract_sha = value_hash(contract)
    require(commit['version'] == 'evaluate_native30_v3' and commit['status'] == 'committed_fold_only_evaluation_not_model_selection'
            and commit['contract_sha256'] == contract_sha and commit['all_input_output_runtime_bindings_end_rehashed'] is True
            and commit['model_instances'] == expected['eligible']
            and all(commit.get(k) == v for k, v in PRODUCER_SCOPE.items()), 'complete pinned evaluator COMMIT required')
    require(contract['version'] == 'evaluate_native30_v3' and contract['stage'] == 'native30_seven_family_schedule_only_development_evaluation_v3'
            and contract['output_root'] == str(root) and contract['expected_count'] == expected['rows']
            and all(contract.get(k) == v for k, v in PRODUCER_SCOPE.items()), 'evaluation contract identity/count/scope mismatch')
    external_bindings = {}
    for name, pin in (('evaluate_native30_v3.py', EVALUATOR_SHA), ('test_evaluate_native30_v3.py', EVALUATOR_TEST_SHA)):
        entries = [e for e in contract['bindings'].values() if Path(e['path']).name == name]
        require(len(entries) == 1 and entries[0]['sha256'] == pin and binding(entries[0]['path']) == entries[0], 'frozen evaluator code lineage changed')
        external_bindings[name] = entries[0]
    actual = evaluation_inventory(root)
    require(commit['products'] == actual, 'complete evaluation product inventory/hash mismatch')
    manifest, authorization = read(root / 'manifest.json'), read(root / 'authorization.json')
    require(manifest['version'] == 'evaluate_native30_v3' and manifest['contract_sha256'] == contract_sha and manifest['model_instances'] == expected['eligible']
            and manifest['accounting'] == contract['accounting']
            and manifest['training_transform_equation_prediction_replay_performed'] is True
            and manifest['independent_full_publication_numerical_audit_performed'] is False
            and all(manifest.get(k) == v for k, v in PRODUCER_SCOPE.items()), 'evaluation manifest mismatch')
    require(authorization['status'] == 'parent_frozen_verified' and authorization['contract_sha256'] == contract_sha
            and binding(authorization['binding']['path']) == authorization['binding'], 'evaluation authorization evidence changed')
    frozen = read(authorization['binding']['path'])
    require(frozen['version'] == 'native30-evaluation-parent-freeze-v3'
            and frozen['status'] == 'parent_frozen_for_native30_fold_only_fitting'
            and frozen['stage'] == contract['stage']
            and frozen['contract'] == contract and frozen['contract_sha256'] == contract_sha
            and frozen['fitting_authorized'] is True and frozen['scoring_authorized'] is True
            and frozen['independent_review']['reviewer'] == 'root' and frozen['independent_review']['approved'] is True
            and all(frozen.get(k) == v for k, v in PRODUCER_SCOPE.items()), 'original parent fitting authority mismatch')
    models = {Path(name).stem: entry for name, entry in actual.items() if name.startswith('models/')}
    require(models == manifest['model_receipts'] and len(models) == expected['eligible'], 'complete model receipt inventory required')
    omissions = read(root / 'omitted.json')
    require(len(omissions) == manifest['omitted_instances'] == expected['intended'] - expected['eligible'], 'omitted task count mismatch')
    pools, cap_records, catalogue, roster = defaultdict(list), {}, {}, {}
    fold_rows = []
    def register(task, entry=None):
        validate_task(task, expected)
        key = (*task_key(task), task['opposite_group_fold'])
        require(key not in catalogue, 'duplicate intended task')
        catalogue[key] = task['status']
        reference = {k: v for k, v in task.items() if k not in {'combination', 'feature_mode', 'analysis_role'}}
        uid = task['schedule_uid']
        require(uid not in cap_records or cap_records[uid] == reference, 'candidate schedules do not share exact rows/status/reasons')
        cap_records[uid] = reference
        pools[task_key(task)].append({'entry': entry, 'schedule_uid': uid})
    for task in omissions:
        register(task)
    for index, (uid, entry) in enumerate(sorted(models.items()), 1):
        record = receipt(Path(entry['path']), entry, contract_sha, expected)
        task, predictions = record['task'], record['predictions']
        register(task, entry)
        for p in predictions:
            metadata = {k: p[k] for k in ('id', 'label', 'source_group', 'group_id', 'component_id', 'role')}
            require(p['id'] not in roster or roster[p['id']] == metadata, 'item identity changes between evaluation models')
            roster[p['id']] = metadata
        fold_rows.append({'model_uid': uid, **{k: task[k] for k in ('fold_type', 'heldout_source', 'quantity', 'combination', 'feature_mode', 'opposite_group_fold', 'schedule_uid')},
                          'train_rows': len(task['train_ids']), 'test_rows': len(predictions),
                          'single_model_fold_auc': auc(predictions),
                          'single_model_recording_weighted_BA': recording_recalls(predictions)['balanced_accuracy']})
        if index % 1000 == 0:
            print(json.dumps({'event': 'report_receipts_verified', 'verified': index, 'expected': len(models)}), flush=True)
    require(len(roster) == expected['rows'] and dict(Counter(r['source_group'] for r in roster.values())) == expected['sources']
            and dict(Counter(r['label'] for r in roster.values())) == expected['labels'], 'reporting roster/source/label mismatch')
    source_labels = defaultdict(set)
    for row in roster.values():
        source_labels[row['source_group']].add(row['label'])
    require(all(len(labels) == 1 for labels in source_labels.values()), 'source label changes')
    held_units = [(PROTOCOLS[label], source) for source, labels in source_labels.items() for label in labels]
    held_units.append((PROTOCOLS[2], '__all_sources__'))
    wanted = {(protocol, source, cap, combination, mode, bucket)
              for protocol, source in held_units for cap in expected['caps'] for combination in expected['combinations']
              for mode in (MODES if cap == 'all' else MODES[:1]) for bucket in range(expected['buckets'])}
    require(set(catalogue) == wanted and len(wanted) == expected['intended'], 'incomplete/extra intended reporting catalogue')
    for record in cap_records.values():
        require(set(record['train_ids']) | set(record['test_ids']) <= set(roster), 'unknown task identity')
        train, test = ([roster[x] for x in record[name]] for name in ('train_ids', 'test_ids'))
        require(not {r['group_id'] for r in train} & {r['group_id'] for r in test}
                and not {r['component_id'] for r in train} & {r['component_id'] for r in test}, 'reporting train/test group/component overlap')
        if record['fold_type'] != PROTOCOLS[2]:
            require(record['heldout_source'] not in {r['source_group'] for r in train}, 'held source appears in training')
    for role in ('primary', 'diagnostic'):
        selected = [(k, status) for k, status in catalogue.items() if (k[4] == MODES[0]) == (role == 'primary')]
        eligible = sum(status == 'eligible_metadata_cell' for _, status in selected)
        predictions = sum(r['test_rows'] for r in fold_rows if (r['feature_mode'] == MODES[0]) == (role == 'primary'))
        require(contract['accounting'][role] == {'intended': len(selected), 'eligible': eligible,
                'omitted': len(selected) - eligible, 'prediction_rows': predictions} and eligible == expected[role], 'producer role accounting mismatch')
    return {'root': root, 'commit': commit_entry, 'contract': contract, 'contract_sha256': contract_sha,
            'snapshot': actual, 'authorization': authorization, 'pools': dict(pools), 'cap_records': cap_records,
            'roster': roster, 'omissions': omissions, 'fold_rows': fold_rows, 'expected': expected,
            'external_bindings': external_bindings}


def pool_predictions(predictions):
    """Exactly one protocol/held-source/cap/subset/mode pool, never cross-held."""
    ids = [p['id'] for p in predictions]
    require(len(ids) == len(set(ids)), 'repeated test ID across pooled bucket models')
    grouped = defaultdict(lambda: defaultdict(list))
    labels = {}
    for p in predictions:
        source, group = p['source_group'], p['group_id']
        require(source not in labels or labels[source] == p['label'], 'pooled source label conflict')
        labels[source] = p['label']
        grouped[source][group].append(p)
    sources = []
    for source in sorted(grouped):
        groups = grouped[source]
        items = [p for values in groups.values() for p in values]
        sources.append({'source_group': source, 'label': labels[source], 'groups': len(groups), 'rows': len(items),
                        'components': len({p['component_id'] for p in items}),
                        'equal_group_recall': mean(mean(correct(p) for p in values) for values in groups.values()),
                        'recording_weighted_recall': mean(correct(p) for p in items)})
    human = mean(s['equal_group_recall'] for s in sources if s['label'] == 0)
    ai = mean(s['equal_group_recall'] for s in sources if s['label'] == 1)
    return {'status': 'available' if human is not None and ai is not None else 'no_eligible_bucket_predictions',
            'test_rows': len(predictions), 'test_groups': len({p['group_id'] for p in predictions}),
            'test_components': len({p['component_id'] for p in predictions}), 'test_id_set_sha256': id_hash(ids),
            'per_source': sources, 'equal_group_equal_source_human_recall': human,
            'equal_group_equal_source_ai_recall': ai,
            'equal_group_equal_source_BA': mean((human, ai)) if human is not None and ai is not None else None,
            'recording_weighted_within_held_source_pool': recording_recalls(predictions)}


def summarize(published):
    held_results = []
    fold_lookup = {(r['schedule_uid'], r['combination'], r['feature_mode']): r for r in published['fold_rows']}
    fold_groups = defaultdict(list)
    for row in published['fold_rows']:
        fold_groups[(row['fold_type'], row['quantity'], row['combination'], row['feature_mode'])].append(row)
    def pool_order(key):
        protocol, source, cap, combination, mode = key
        return (PROTOCOLS.index(protocol), source, published['expected']['combinations'].index(combination),
                list(published['expected']['caps']).index(cap), MODES.index(mode))
    for key, references in sorted(published['pools'].items(), key=lambda x: pool_order(x[0])):
        protocol, source, cap, combination, mode = key
        predictions, eligible_buckets, omitted, training, aucs = [], [], [], [], []
        group_buckets = {}
        for reference in sorted(references, key=lambda x: published['cap_records'][x['schedule_uid']]['opposite_group_fold']):
            cap_record = published['cap_records'][reference['schedule_uid']]
            bucket = cap_record['opposite_group_fold']
            training.append({'bucket': bucket, 'training_rows': len(cap_record['train_ids']), 'status': cap_record['status']})
            if reference['entry'] is None:
                omitted.append({'bucket': bucket, 'reasons': cap_record['omission_reasons']})
                continue
            record = receipt(Path(reference['entry']['path']), reference['entry'], published['contract_sha256'], published['expected'])
            require(task_key(record['task']) == key, 'pool references another held-source experiment')
            eligible_buckets.append(bucket)
            for row in record['predictions']:
                group = row['group_id']
                require(group not in group_buckets or group_buckets[group] == bucket, 'global group appears in multiple pooled buckets')
                group_buckets[group] = bucket
            predictions.extend(record['predictions'])
            aucs.append(fold_lookup[(reference['schedule_uid'], combination, mode)]['single_model_fold_auc'])
        pooled = pool_predictions(predictions)
        held_recall = next((r['equal_group_recall'] for r in pooled['per_source'] if r['source_group'] == source), None)
        coverage = {'eligible_buckets': eligible_buckets, 'omitted_buckets': omitted, 'test_id_set_sha256': pooled['test_id_set_sha256'],
                    'per_source_rows_groups': [{k: r[k] for k in ('source_group', 'label', 'rows', 'groups')} for r in pooled['per_source']]}
        held_results.append({'fold_type': protocol, 'heldout_source': source, 'quantity': cap, 'combination': combination,
                             'feature_mode': mode, **pooled, 'held_source_equal_group_recall': held_recall,
                             'intended_bucket_count': published['expected']['buckets'], 'eligible_bucket_count': len(eligible_buckets),
                             'eligible_buckets': eligible_buckets, 'omitted_buckets': omitted, 'training_rows_by_bucket': training,
                             'descriptive_mean_single_model_fold_auc': mean(x for x in aucs if x is not None),
                             'descriptive_auc_fold_count': sum(x is not None for x in aucs),
                             'coverage_key': value_hash(coverage)})
    grouped = defaultdict(list)
    for row in held_results:
        grouped[(row['fold_type'], row['quantity'], row['combination'], row['feature_mode'])].append(row)
    protocol_rows = []
    for key, rows in sorted(grouped.items(), key=lambda x: pool_order((x[0][0], '', x[0][1], x[0][2], x[0][3]))):
        protocol, cap, combination, mode = key
        available = [r for r in rows if r['status'] == 'available']
        folds = fold_groups[key]
        protocol_rows.append({'fold_type': protocol, 'quantity': cap, 'combination': combination, 'feature_mode': mode,
            'status': 'available' if available else 'no_eligible_held_source_results',
            'intended_held_source_experiments': len(rows), 'available_held_source_experiments': len(available),
            'macro_equal_group_equal_source_BA': mean(r['equal_group_equal_source_BA'] for r in available),
            'macro_human_recall': mean(r['equal_group_equal_source_human_recall'] for r in available),
            'macro_ai_recall': mean(r['equal_group_equal_source_ai_recall'] for r in available),
            'macro_recording_weighted_within_experiment_BA': mean(r['recording_weighted_within_held_source_pool']['balanced_accuracy'] for r in available),
            'prediction_occurrences_not_independent_rows': sum(r['test_rows'] for r in rows),
            'eligible_model_count': len(folds), 'omitted_model_count': sum(len(r['omitted_buckets']) for r in rows),
            'descriptive_mean_single_model_fold_auc': mean(r['single_model_fold_auc'] for r in folds if r['single_model_fold_auc'] is not None),
            'coverage_key': value_hash(sorted((r['heldout_source'], r['coverage_key']) for r in rows)),
            'held_source_coverage': [{'heldout_source': r['heldout_source'], 'status': r['status'],
                'test_rows': r['test_rows'], 'test_groups': r['test_groups'], 'eligible_buckets': r['eligible_buckets'],
                'omitted_buckets': r['omitted_buckets'], 'held_source_equal_group_recall': r['held_source_equal_group_recall']} for r in rows]})
    lookups = {(r['fold_type'], r['quantity'], r['combination'], r['feature_mode']): r for r in protocol_rows}
    comparisons = {'fixed_S_D_R_P_reference': [], 'adjacent_caps': [], 'diagnostic_vs_values_plus_missing': []}
    for row in protocol_rows:
        key = (row['fold_type'], row['quantity'], REFERENCE, row['feature_mode'])
        comparisons['fixed_S_D_R_P_reference'].append(compare(row, lookups.get(key), 'fixed_S+D+R+P'))
        cap_index = list(published['expected']['caps']).index(row['quantity'])
        if row['feature_mode'] == MODES[0] and cap_index:
            previous = published['expected']['caps'][cap_index - 1]
            key = (row['fold_type'], previous, row['combination'], row['feature_mode'])
            comparisons['adjacent_caps'].append(compare(row, lookups.get(key), 'previous_cap'))
        if row['feature_mode'] != MODES[0]:
            key = (row['fold_type'], 'all', row['combination'], MODES[0])
            comparisons['diagnostic_vs_values_plus_missing'].append(compare(row, lookups.get(key), 'values_plus_missing'))
    return {'version': VERSION, 'status': 'complete_reporting_only_no_selection',
            'primary_protocol_cells': [r for r in protocol_rows if r['feature_mode'] == MODES[0]],
            'diagnostic_protocol_cells': [r for r in protocol_rows if r['feature_mode'] != MODES[0]],
            'held_source_cells': held_results, 'comparisons': comparisons,
            'evaluation_accounting': published['contract']['accounting'],
            'scope_notes': ['Human-source holdout, generator holdout, and ordinary grouped descriptive protocols remain separate.',
                'Pool disjoint bucket test predictions only within one protocol/held-source/cap/subset/mode; require unique IDs.',
                'Primary BA: group-mean correctness, equal-source recall within each class, then equal-class mean; protocol mean weights held-source experiments equally.',
                'Recording-weighted BA is separately labeled and computed within each held-source experiment before its macro mean.',
                'Opposite-class predictions repeated across held-source experiments are dependent occurrences, not independent samples.',
                'AUC is computed only within a single fitted fold; its mean is descriptive, not cross-model pooled AUC.',
                'No confidence interval is inferred from five fixed buckets. No winner, tuning, historical causal delta or independent numerical audit is claimed.',
                'F: phase/group delay. H: pitch-class/chroma organization, not acoustic overtone shape. SC: frozen six descriptors.'], **SCOPE}


def compare(current, reference, kind):
    available = reference is not None and current['status'] == reference['status'] == 'available'
    matched = available and current['coverage_key'] == reference['coverage_key']
    return {'comparison': kind, **{k: current[k] for k in ('fold_type', 'quantity', 'combination', 'feature_mode')},
            'reference_quantity': reference['quantity'] if reference else None,
            'reference_combination': reference['combination'] if reference else REFERENCE,
            'reference_feature_mode': reference['feature_mode'] if reference else None,
            'status': 'matched' if matched else 'coverage_mismatch' if available else 'unavailable_reference_or_result',
            'current_coverage_key': current['coverage_key'], 'reference_coverage_key': reference['coverage_key'] if reference else None,
            'delta_macro_equal_group_equal_source_BA': current['macro_equal_group_equal_source_BA'] - reference['macro_equal_group_equal_source_BA'] if matched else None,
            'delta_macro_recording_weighted_within_experiment_BA': current['macro_recording_weighted_within_experiment_BA'] - reference['macro_recording_weighted_within_experiment_BA'] if matched else None}


def markdown_table(rows, columns):
    def cell(value):
        if value is None:
            return 'unavailable'
        if isinstance(value, float):
            return format(value, '.6f')
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True, separators=(',', ':'))
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |',
                      *['| ' + ' | '.join(cell(row.get(column)) for column in columns) + ' |' for row in rows]])


def markdown_products(result, published):
    notes = '\n\n'.join(result['scope_notes'])
    columns = ['combination', 'quantity', 'feature_mode', 'macro_equal_group_equal_source_BA', 'macro_human_recall',
               'macro_ai_recall', 'macro_recording_weighted_within_experiment_BA', 'available_held_source_experiments',
               'eligible_model_count', 'omitted_model_count', 'descriptive_mean_single_model_fold_auc']
    tables = ['# Native30 exhaustive evaluation tables', notes]
    for category in ('primary_protocol_cells', 'diagnostic_protocol_cells'):
        for protocol in PROTOCOLS:
            rows = [r for r in result[category] if r['fold_type'] == protocol]
            tables.extend(['## ' + category + ' — ' + protocol, markdown_table(rows, columns)])
    held = ['# Per-held-source results and coverage', notes,
            markdown_table(result['held_source_cells'], ['fold_type', 'heldout_source', 'quantity', 'combination', 'feature_mode',
                'equal_group_equal_source_BA', 'held_source_equal_group_recall', 'equal_group_equal_source_human_recall',
                'equal_group_equal_source_ai_recall', 'test_rows', 'test_groups', 'eligible_bucket_count', 'omitted_buckets', 'training_rows_by_bucket'])]
    comparisons = ['# Fixed comparisons — no winner selection', 'Negative increments are retained. Nonmatching coverage is not subtracted.']
    for name, rows in result['comparisons'].items():
        comparisons.extend(['## ' + name, markdown_table(rows, ['fold_type', 'quantity', 'combination', 'feature_mode',
            'reference_quantity', 'reference_combination', 'reference_feature_mode', 'status',
            'delta_macro_equal_group_equal_source_BA', 'delta_macro_recording_weighted_within_experiment_BA'])])
    folds = ['# Per-fold AUC — separately fitted models', 'These are single-model fold AUCs. Means are descriptive; no pooled raw-score AUC or confidence interval is claimed.',
             markdown_table(published['fold_rows'], ['model_uid', 'fold_type', 'heldout_source', 'quantity', 'combination',
                 'feature_mode', 'opposite_group_fold', 'train_rows', 'test_rows', 'single_model_fold_auc', 'single_model_recording_weighted_BA'])]
    return {'tables.md': '\n\n'.join(tables) + '\n', 'held_source_tables.md': '\n\n'.join(held) + '\n',
            'comparisons.md': '\n\n'.join(comparisons) + '\n', 'per_fold_auc.md': '\n\n'.join(folds) + '\n'}


def recheck(published):
    require(binding(published['commit']['path']) == published['commit']
            and evaluation_inventory(published['root']) == published['snapshot']
            and binding(published['authorization']['binding']['path']) == published['authorization']['binding'], 'evaluation changed during reporting')
    require(all(binding(e['path']) == e for e in published['external_bindings'].values()), 'frozen evaluator code changed during reporting')


def publish(published, output, report_bindings):
    output = safe(output)
    require(not output.exists() and output.parent.is_dir()
            and not output.is_relative_to(published['root']) and not published['root'].is_relative_to(output)
            and not any(Path(e['path']).is_relative_to(output) for e in report_bindings.values()), 'new nonoverlapping report output required')
    result = summarize(published)
    authorities = {'evaluation_COMMIT': published['commit'], 'evaluation_contract': published['snapshot']['contract.json'],
                   'evaluation_authorization': published['authorization'], 'package_COMMIT': published['contract'].get('package_commit'),
                   'frozen_evaluator_code': published['external_bindings'],
                   'schedule_document': published['contract'].get('schedule_document'),
                   'schedule_sha256': published['contract'].get('schedule_sha256'),
                   'source_graph_policy': 'source authorities retained by immutable evaluation contract; no feature/audio loading or model rerun by reporter',
                   'report_bindings': report_bindings}
    result['source_authorities'] = authorities
    recheck(published)
    for entry in report_bindings.values():
        require(binding(entry['path']) == entry, 'report specification/code changed')
    output.mkdir()
    write_new(output / 'summary.json', result)
    write_new(output / 'per_fold_auc.json', published['fold_rows'])
    write_new(output / 'omitted_cells.json', published['omissions'])
    write_new(output / 'coverage.json', list(published['cap_records'].values()))
    for name, text in markdown_products(result, published).items():
        write_new(output / name, text.encode())
    for entry in report_bindings.values():
        require(binding(entry['path']) == entry, 'report specification/code changed before COMMIT')
    recheck(published)
    products = {p.name: binding(p) for p in sorted(output.iterdir())}
    require(all(binding(entry['path']) == entry for entry in products.values()), 'report output changed before COMMIT')
    write_new(output / 'COMMIT.json', {'version': VERSION, 'status': 'committed_reporting_only_no_selection',
                                     'evaluation_COMMIT': published['commit'], 'products': products,
                                     'all_evaluation_products_end_rehashed': True, **SCOPE})
    return {'status': 'committed_reporting_only_no_selection', 'output': str(output),
            'primary_cells': len(result['primary_protocol_cells']), 'diagnostic_cells': len(result['diagnostic_protocol_cells']),
            'held_source_cells': len(result['held_source_cells']), 'commit': binding(output / 'COMMIT.json'), **SCOPE}


@contextmanager
def read_lock(root):
    path = safe(Path(root) / 'writer.lock')
    require(path.is_file(), 'completed evaluation writer lock required')
    with path.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--evaluation-commit-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    refs = {'reporter': binding(Path(__file__).resolve()), 'tests': binding(HERE / ('test_' + VERSION + '.py')),
            'specification': binding(SPEC, SPEC_SHA)}
    with read_lock(args.evaluation):
        published = inspect_evaluation(args.evaluation, args.evaluation_commit_sha256)
        result = publish(published, args.output, refs)
    print(canonical(result).decode().strip())


if __name__ == '__main__':
    main()
