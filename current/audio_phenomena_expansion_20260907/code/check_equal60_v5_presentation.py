#!/usr/bin/env python3
"""Independent stdlib-only numerical table check; stdout receipt only on success.

Never imports the producer, evaluator or numerical auditor. Does not fit, replay
scores, calculate AUC from pooled scores, rank candidates, or modify inputs.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

FAMILIES = tuple('SDRPFHM')
COMBOS = {'+'.join(x) for n in range(1, 8) for x in itertools.combinations(FAMILIES, n)}
CAPS = {'25', '50', '100', '200', 'all'}
BASE = 'S+D+R+P'
ADDED = (BASE+'+F', BASE+'+H', BASE+'+M', BASE+'+F+H+M')
FIXED = set(FAMILIES) | {BASE, *ADDED}
AXES = ('fold_type', 'heldout_source', 'combination', 'feature_mode', 'quantity')
MACRO_KEYS = ('summary_level', *AXES)
SOURCE_KEYS = (*AXES, 'source_group')
DELTA_KEYS = ('scope', 'summary_level', 'fold_type', 'heldout_source', 'source_group', 'endpoint',
              'quantity', 'baseline_combination', 'added_combination')
RANGE_KEYS = ('summary_level', 'fold_type', 'heldout_source', 'quantity', 'training_source')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def unique(items):
        out = {}
        for k, v in items:
            require(k not in out, 'Duplicate JSON key: '+k)
            out[k] = v
        return out
    def reject(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=reject)


def csv_rows(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'Invalid CSV header')
        for r in reader:
            require(None not in r and None not in r.values(), 'Malformed CSV: '+str(path))
            yield r


def key(row, fields):
    return tuple(str(row[k]) for k in fields)


def average(values):
    values = list(values)
    require(bool(values), 'Undefined empty average')
    return math.fsum(values)/len(values)


def direction(value):
    return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def verify_commit(root):
    marker = strict_json(root/'COMMIT.json')
    require(marker['status'] == 'committed', 'Uncommitted publication')
    files = marker['files']
    require(set(p.name for p in root.iterdir()) == set(files)|{'COMMIT.json'}, 'Publication inventory differs')
    for name, spec in files.items():
        p = root/name
        require(Path(name).name == name and p.is_file() and not p.is_symlink(), 'Unsafe publication member')
        require(p.stat().st_size == spec['bytes'] and sha(p) == spec['sha256'], 'Publication bytes changed: '+str(p))
    return {name: spec['sha256'] for name, spec in files.items()} | {'COMMIT.json': sha(root/'COMMIT.json')}


def compare_table(path, expected, keys, totals, exact_floats=()):
    expected = list(expected)
    lookup = {key(r, keys): r for r in expected}
    require(len(lookup) == len(expected), 'Duplicate independently reconstructed key: '+path.name)
    seen = set()
    numeric = 0
    validated = []
    for r in csv_rows(path):
        k = key(r, keys)
        require(k in lookup and k not in seen, 'Extra or duplicate table key: '+str((path.name, k)))
        seen.add(k)
        wanted = lookup[k]
        require(set(r) == set(wanted), 'Table columns differ: '+path.name)
        parsed = {}
        for field, value in wanted.items():
            token = r[field]
            if isinstance(value, int):
                require(token == str(value), 'Integer differs: '+str((path.name, k, field, token, value)))
                numeric += 1
                parsed[field] = int(token)
            elif isinstance(value, float):
                actual = float(token)
                agrees = actual == value if field in exact_floats else math.isclose(actual, value, rel_tol=1e-11, abs_tol=1e-12)
                require(math.isfinite(actual) and agrees,
                        'Rate differs: '+str((path.name, k, field, actual, value)))
                numeric += 1
                parsed[field] = actual
            else:
                require(token == str(value), 'Field differs: '+str((path.name, k, field, token, value)))
                parsed[field] = token
        validated.append(parsed)
    require(seen == set(lookup), 'Missing table rows: '+path.name)
    totals[path.name] = {'rows': len(seen), 'numeric_values_checked': numeric}
    return validated


def check_index(root, category, schedule, audit):
    index = {}
    identities = set()
    for r in csv_rows(root/(category+'_model_index.csv')):
        uid = r['model_uid']
        require(uid not in index and r['fold_uid'] in schedule, 'Duplicate model or unknown fold')
        f = schedule[r['fold_uid']]
        for field in ('fold_type', 'heldout_source', 'quantity', 'fold_index', 'train_id_set_sha256', 'test_id_set_sha256'):
            require(str(f[field]) == r[field], 'Index/frozen fold mismatch: '+field)
        require(int(r['training_rows']) == f['train']['rows'] and int(r['test_rows']) == f['test']['rows'], 'Model denominator mismatch')
        identity = (r['fold_uid'], r['combination'], r['feature_mode'])
        require(identity not in identities, 'Duplicate model grid cell')
        identities.add(identity)
        index[uid] = r
    expected = {(uid, combo, mode) for uid, f in schedule.items() for combo in COMBOS
                for mode in (('values_plus_missing',) if category == 'primary' else ('median_only', 'missingness_only'))
                if category == 'primary' or str(f['quantity']) == 'all'}
    require(identities == expected and len(index) == audit['streams'][category]['model_index'], 'Incomplete exact model grid')
    return index


def reconstruct_sources(root, category, index, schedule, audit):
    all_ids = sorted({iid for f in schedule.values() for iid in f['test']['ids']})
    bit = {iid: 1 << i for i, iid in enumerate(all_ids)}
    expected_masks = {uid: sum(bit[iid] for iid in f['test']['ids']) for uid, f in schedule.items()}
    seen_models = defaultdict(int)
    identity = {}
    groups = {}
    endpoints = {}
    count = 0
    for r in csv_rows(root/(category+'_predictions.csv')):
        uid, iid = r['model_uid'], r['row_id']
        require(uid in index and iid in bit, 'Prediction has unknown model/ID')
        m = index[uid]
        mask = bit[iid]
        require(expected_masks[m['fold_uid']] & mask and not seen_models[uid] & mask, 'Missing scope or repeated per-model ID')
        seen_models[uid] |= mask
        row_identity = key(r, ('label', 'source_group', 'group_id', 'role'))
        require(identity.setdefault(iid, row_identity) == row_identity and r['role'] == 'development', 'Identity/role changed')
        require(groups.setdefault(r['group_id'], (r['label'], r['source_group'])) == (r['label'], r['source_group']), 'Global group crosses source/label')
        label, predicted = int(r['label']), int(r['predicted_label'])
        score = float(r['score'])
        require(label in (0, 1) and predicted in (0, 1) and r['threshold'] == '0.5'
                and math.isfinite(score) and predicted == int(score >= .5), 'Invalid saved decision')
        k = key(m, AXES)+(r['source_group'],)
        e = endpoints.setdefault(k, {'seen': 0, 'groups': {}, 'folds': set(), 'label': label})
        require(not e['seen'] & mask and e['label'] == label, 'Repeated OOF ID or mixed source label')
        e['seen'] |= mask
        e['folds'].add(m['fold_uid'])
        g = e['groups'].setdefault(r['group_id'], [0, 0])
        g[0] += int(label == predicted)
        g[1] += 1
        count += 1
    require(set(seen_models) == set(index), 'Model predictions absent')
    for uid, mask in seen_models.items():
        require(mask == expected_masks[index[uid]['fold_uid']], 'Model test coverage differs')
    require(count == audit['streams'][category]['predictions'], 'Prediction row count differs from numerical audit')
    output = []
    for k, e in endpoints.items():
        correct = sum(v[0] for v in e['groups'].values())
        n = sum(v[1] for v in e['groups'].values())
        require(n == e['seen'].bit_count(), 'OOF count differs from unique IDs')
        rate = correct/n
        component = average(a/b for a, b in e['groups'].values())
        output.append(dict(zip(SOURCE_KEYS, k)) | {'label': str(e['label']),
            'endpoint': 'ai_sensitivity' if e['label'] else 'human_specificity',
            'unique_oof_recordings': n, 'global_components': len(e['groups']), 'correct_recordings': correct,
            'recording_rate': rate, 'equal_component_rate': component,
            'equal_component_minus_recording_pp': 100*(component-rate), 'valid_folds': len(e['folds'])})
    return output, count


def reconstruct_macros(root, category, kind, index, audit):
    per_model = defaultdict(list)
    seen = set()
    name = category+('_pair_metrics.csv' if kind == 'pair' else '_pooled_metrics.csv')
    for r in csv_rows(root/name):
        uid = r['model_uid']
        require(uid in index, 'Metric has unknown model')
        identity = (uid, r['human_source'], r['ai_source']) if kind == 'pair' else (uid,)
        require(identity not in seen, 'Duplicate raw metric key')
        seen.add(identity)
        values = [float(r[k]) for k in ('roc_auc', 'balanced_accuracy')]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'Invalid audited metric')
        per_model[uid].append(values)
    require(set(per_model) == set(index) and len(seen) == audit['streams'][category][kind+'_metrics'], 'Metric grid differs')
    arms = defaultdict(list)
    for uid, values in per_model.items():
        arms[key(index[uid], AXES)].append((len(values), [average(v[i] for v in values) for i in (0, 1)]))
    output = []
    for k, values in arms.items():
        output.append({'summary_level': 'held_source_arm', **dict(zip(AXES, k)), 'held_source_arms': 1,
            'valid_folds': len(values), 'metric_cells': sum(v[0] for v in values),
            'roc_auc': average(v[1][0] for v in values), 'balanced_accuracy': average(v[1][1] for v in values)})
    types = defaultdict(list)
    for r in output:
        types[(r['fold_type'], r['combination'], r['feature_mode'], r['quantity'])].append(r)
    for (fold_type, combo, mode, cap), values in types.items():
        output.append({'summary_level': 'fold_type_macro', 'fold_type': fold_type, 'heldout_source': '__equal_arms__',
            'combination': combo, 'feature_mode': mode, 'quantity': cap, 'held_source_arms': len(values),
            'valid_folds': sum(v['valid_folds'] for v in values), 'metric_cells': sum(v['metric_cells'] for v in values),
            'roc_auc': average(v['roc_auc'] for v in values), 'balanced_accuracy': average(v['balanced_accuracy'] for v in values)})
    return output


def reconstruct_ranges(schedule):
    scopes = defaultdict(list)
    for f in schedule.values():
        for source in [*f['train']['source_rows'], '__total__']:
            n = f['train']['rows'] if source == '__total__' else f['train']['source_rows'][source]
            g = len(f['train']['groups']) if source == '__total__' else f['train']['source_groups'][source]
            for level, held in [('held_source_arm', f['heldout_source']), ('fold_type_macro', '__all_arms__')]:
                scopes[(level, f['fold_type'], held, str(f['quantity']), source)].append((n, g))
    return [dict(zip(RANGE_KEYS, k)) | {'valid_folds': len(v), 'min_training_rows': min(x[0] for x in v),
        'max_training_rows': max(x[0] for x in v), 'min_training_groups': min(x[1] for x in v),
        'max_training_groups': max(x[1] for x in v)} for k, v in scopes.items()]


def reconstruct_increments(pair, sources, index, saved_pair=None, saved_sources=None):
    # Registry layout is the documented CSV provenance encoding, not a numerical producer call.
    registry = {}
    for m in index.values():
        if m['combination'] in {BASE, *ADDED}:
            k = key(m, ('fold_type', 'heldout_source', 'feature_mode', 'quantity', 'fold_uid', 'combination'))
            require(k not in registry, 'Duplicate matched identity registry key')
            registry[k] = (m['train_id_set_sha256'], m['test_id_set_sha256'])
    for k, v in registry.items():
        if k[-1] == BASE:
            for c in ADDED:
                require(registry.get((*k[:-1], c)) == v, 'Increment training/test hashes differ')
    registry_sha = digest(sorted(registry.items()))
    output = []
    for rows, saved, fields, scope in [(pair, saved_pair, MACRO_KEYS, 'two_class_pair_macro'),
                                      (sources, saved_sources, SOURCE_KEYS, 'source_endpoint')]:
        lookup = {key(r, fields): r for r in rows}
        encoded = {key(r, fields): r for r in (rows if saved is None else saved)}
        require(len(encoded) == len(lookup) == len(rows) and set(encoded) == set(lookup), 'Validated rate encoding keys differ')
        metrics = ('roc_auc', 'balanced_accuracy') if scope == 'two_class_pair_macro' else ('recording_rate', 'equal_component_rate')
        # Numerical correctness remains independently checked. Only the literal
        # encoding/sign uses round-trippable saved rate tokens, because fsum and
        # producer averaging may differ by one ULP around a mathematical zero.
        for k, expected in lookup.items():
            require(all(math.isfinite(encoded[k][m]) and math.isclose(encoded[k][m], expected[m], rel_tol=1e-11, abs_tol=1e-12)
                        for m in metrics), 'Serialized source rate differs from independent reconstruction')
        for r in rows:
            if r['combination'] != BASE:
                continue
            for c in ADDED:
                other = dict(r, combination=c)
                saved_left, saved_right = encoded[key(r, fields)], encoded[key(other, fields)]
                deltas = [saved_right[m]-saved_left[m] for m in metrics]
                paired = 'not_applicable' if scope == 'two_class_pair_macro' else ('same_direction' if len({direction(x) for x in deltas}) == 1 else 'different_direction')
                for metric, delta in zip(metrics, deltas):
                    output.append({'scope': scope, 'summary_level': r.get('summary_level', 'held_source_arm'),
                        'fold_type': r['fold_type'], 'heldout_source': r['heldout_source'],
                        'source_group': r.get('source_group', '__two_class__'),
                        'endpoint': metric if scope == 'two_class_pair_macro' else r['endpoint']+'_'+metric,
                        'quantity': r['quantity'], 'baseline_combination': BASE, 'added_combination': c,
                        'baseline_rate': saved_left[metric], 'added_rate': saved_right[metric], 'added_minus_baseline_pp': 100*delta,
                        'direction': direction(delta), 'paired_identity_registry_sha256': registry_sha,
                        'paired_direction_comparison': paired})
    return output, registry_sha


def verify_grids(rows, keys, category):
    scopes = defaultdict(set)
    for r in rows:
        scopes[key(r, keys)].add((r['combination'], r['quantity']))
    expected = set(itertools.product(COMBOS, CAPS if category == 'primary' else {'all'}))
    require(scopes and all(v == expected for v in scopes.values()), 'Incomplete 635/127 reporting grid')
    return len(scopes)


def self_checks():
    # Unequal groups, unequal pair counts, and unequal arm fold counts must not change estimands.
    require(average([0/3, 1/1]) == .5 and (0+1)/(3+1) == .25, 'Component self-check failed')
    require(average([average([0., 1.]), average([1.])]) == .75, 'Pair-fold macro self-check failed')
    require(average([average([0., 0., 0.]), average([1.])]) == .5, 'Equal-arm macro self-check failed')


def check(args):
    self_checks()
    root, pres = args.results_dir.resolve(), args.presentation_dir.resolve()
    require(root != pres and args.audit_sha256 == sha(args.audit), 'Wrong audit binding or overlapping roots')
    audit = strict_json(args.audit)
    require(audit['status'] == 'passed' and audit['synthetic_test_only'] is args.synthetic_test_only
            and audit['metrics_independently_recomputed'] is True and audit['model_fitting_performed'] is False,
            'Independent numerical audit/mode is invalid')
    raw_hashes, presentation_hashes = verify_commit(root), verify_commit(pres)
    require(audit['result_files_sha256'] == raw_hashes and audit['result_commit_sha256'] == raw_hashes['COMMIT.json'], 'Audit/raw hash map differs')
    receipt = strict_json(pres/'presentation_receipt.json')
    run = strict_json(root/'run_manifest.json')
    require(receipt['synthetic_test_only'] is args.synthetic_test_only and run['synthetic_test_only'] is args.synthetic_test_only,
            'Publication synthetic/real mode differs')
    require(receipt['contract_sha256'] == audit['contract_sha256'] == digest(run['contract']), 'Contract binding differs')
    require(receipt['input_commit_sha256'] == raw_hashes['COMMIT.json'] and receipt['independent_audit']['sha256'] == args.audit_sha256,
            'Presenter audit/COMMIT binding differs')
    require((audit['rows'] <= 16) if args.synthetic_test_only else (audit['rows'] == 2207 and audit['valid_folds'] == 43), 'Cohort gate differs')
    bound = {str(args.audit.resolve()): args.audit_sha256, str(Path(__file__).resolve()): sha(__file__)}
    for support in (Path(__file__).with_name('test_check_equal60_v5_presentation.py'),
                    Path(__file__).resolve().parent.parent/'EQUAL60_V5_PRESENTATION_INDEPENDENT_CHECK_EN.md'):
        bound[str(support.resolve())] = sha(support)
    for path, expected in receipt['inputs_sha256'].items():
        require(sha(path) == expected, 'Presenter input binding changed: '+path)
        bound[path] = expected
    for name, spec in receipt['outputs_before_receipt_sha256'].items():
        require(presentation_hashes[name] == spec['sha256'] and (pres/name).stat().st_size == spec['bytes'], 'Presenter output receipt differs')
    schedule_list = [json.loads(line) for line in (root/'fold_registry.jsonl').read_text().splitlines()]
    require(schedule_list == run['contract']['schedule'], 'Frozen schedule differs')
    schedule = {f['fold_uid']: f for f in schedule_list}
    require(len(schedule) == len(schedule_list), 'Duplicate frozen fold UID')
    totals, counts, grids, reconstructed = {}, {}, {}, {}
    for category in ('primary', 'diagnostic'):
        index = check_index(root, category, schedule, audit)
        sources, prediction_count = reconstruct_sources(root, category, index, schedule, audit)
        pair = reconstruct_macros(root, category, 'pair', index, audit)
        pooled = reconstruct_macros(root, category, 'pooled', index, audit)
        prefix = 'primary_' if category == 'primary' else 'diagnostic_all_cap_'
        saved_sources = compare_table(pres/(prefix+'source_endpoints_all_arms.csv'), sources, SOURCE_KEYS, totals)
        saved_pair = compare_table(pres/(prefix+'pair_macro_all_arms.csv'), pair, MACRO_KEYS, totals)
        compare_table(pres/(prefix+'pooled_fold_companion_all_arms.csv'), pooled, MACRO_KEYS, totals)
        grids[category] = {'pair_reporting_scopes': verify_grids(pair, ('summary_level', 'fold_type', 'heldout_source', 'feature_mode'), category),
            'source_reporting_scopes': verify_grids(sources, ('fold_type', 'heldout_source', 'source_group', 'feature_mode'), category)}
        counts[category] = {'models': len(index), 'predictions': prediction_count}
        reconstructed[category] = (index, sources, pair, saved_sources, saved_pair)
    index, sources, pair, saved_sources, saved_pair = reconstructed['primary']
    compare_table(pres/'primary_fixed12_overview.csv', [r for r in pair if r['combination'] in FIXED], MACRO_KEYS, totals)
    increments, registry_sha = reconstruct_increments(pair, sources, index, saved_pair, saved_sources)
    compare_table(pres/'primary_fixed_increments.csv', increments, DELTA_KEYS, totals,
                  exact_floats=('baseline_rate', 'added_rate', 'added_minus_baseline_pp'))
    compare_table(pres/'training_row_group_ranges.csv', reconstruct_ranges(schedule), RANGE_KEYS, totals)
    # Verify the published Markdown/LaTeX fixed overview numbers without executing the renderer.
    wanted = {(r['fold_type'], r['combination'], r['quantity']): r for r in saved_pair if r['summary_level'] == 'fold_type_macro' and r['combination'] in FIXED}
    md_seen, tex_seen = set(), set()
    for line in (pres/'EQUAL60_V5_PRESENTATION_TABLES_EN.md').read_text().splitlines():
        if line.startswith('| ') and any(t in line for t in ('human_source_holdout', 'generator_holdout', 'ordinary_group_holdout_descriptive')):
            fields = [v.strip() for v in line.strip('|').split('|')]
            k = tuple(fields[:3]); require(k in wanted and k not in md_seen, 'Markdown overview identity mismatch'); md_seen.add(k)
            require(fields[3:] == [f"{wanted[k][m]:.6f}" for m in ('roc_auc', 'balanced_accuracy')], 'Markdown overview values differ')
    types = {'H-source': 'human_source_holdout', 'Generator': 'generator_holdout', 'Group': 'ordinary_group_holdout_descriptive'}
    for line in (pres/'equal60_v5_presentation_tables_en.tex').read_text().splitlines():
        if line.split(' & ')[0] in types:
            fields = [v.strip() for v in line.removesuffix('\\\\').split('&')]
            k = (types[fields[0]], fields[1].replace('{+}', '+'), fields[2]); require(k in wanted and k not in tex_seen, 'LaTeX overview identity mismatch'); tex_seen.add(k)
            require(fields[3:] == [str(wanted[k]['valid_folds']), *[f"{wanted[k][m]:.6f}" for m in ('roc_auc', 'balanced_accuracy')]], 'LaTeX overview values differ')
    require(md_seen == tex_seen == set(wanted), 'Missing Markdown/LaTeX overview rows')
    require(verify_commit(root) == raw_hashes and verify_commit(pres) == presentation_hashes, 'Publication changed during check')
    for path, expected in bound.items():
        require(sha(path) == expected, 'Bound input/checker changed during check')
    return {'status': 'passed_independent_presentation_table_check', 'schema_version': 1,
        'synthetic_test_only': args.synthetic_test_only, 'checker_sha256': sha(__file__), 'python': platform.python_version(),
        'raw_result_commit_sha256': raw_hashes['COMMIT.json'], 'presentation_commit_sha256': presentation_hashes['COMMIT.json'],
        'numerical_audit_sha256': args.audit_sha256, 'contract_sha256': audit['contract_sha256'],
        'raw_files_sha256': raw_hashes, 'presentation_files_sha256': presentation_hashes,
        'bound_small_inputs_sha256': bound, 'counts': counts, 'tables': totals, 'reporting_grids': grids,
        'total_numeric_table_values_checked': sum(v['numeric_values_checked'] for v in totals.values()),
        'markdown_overview_rows_checked': len(md_seen), 'latex_overview_rows_checked': len(tex_seen),
        'paired_identity_registry_sha256': registry_sha, 'self_checks': 3,
        'source_endpoints_from_saved_decisions': True, 'pair_fold_arm_macros_reconstructed': True,
        'serialized_delta_directions_from_independently_validated_rate_tokens': True,
        'rendered_numbers_from_independently_validated_rate_tokens': True,
        'fitting_performed': False, 'scoring_performed': False, 'pooled_oof_auc_computed': False,
        'producer_evaluator_auditor_imported': False, 'ranking_or_selection_performed': False,
        'scope': 'Numerical presentation consistency against independently audited saved outputs; not another raw-score or media audit.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--presentation-dir', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--audit-sha256', required=True)
    p.add_argument('--synthetic-test-only', action='store_true')
    args = p.parse_args()
    print(json.dumps(check(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
