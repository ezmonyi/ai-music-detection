"""Metadata-only NEW draft native30 evaluation schedule, never fitting.

Preserves v6 hash buckets and group-ranking hash, but permits shared groups
across sources/classes. Removes transitive dependencies of ALL held-source
rows, not only test-bucket rows. Incremental coupled whole-group caps preserve
nesting and hard source caps. These are substantive new protocol semantics;
historical v6 outcome differences cannot be interpreted causally.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import sys

VERSION = 'plan_native30_evaluation_schedule_v1'
SEED, BUCKETS = 20260907, 5
CAPS = (25, 50, 100, 200, 'all')
FAMILIES = ('S', 'D', 'R', 'P', 'F', 'H', 'SC')
COMBINATIONS = ['+'.join(c) for n in range(1, 8) for c in itertools.combinations(FAMILIES, n)]
PRIMARY_MODE = 'values_plus_missing'
DIAGNOSTIC_MODES = ('median_only', 'missingness_only')
PLAN_SHA = '94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964'
SCREEN_SHA = 'ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87'
PINS = {'frozen_evaluate_expanded_20260905.py': 'd2ed30d9833fbe122f023de1223e44a40c1b4f95f99f63f0ba27647c87cc7232',
        'evaluate_new_phenomena_v2.py': '4014492f3e3ddda2d9cbea9f37147b37be0124ba071f0fee803845537526bd2e',
        'evaluate_new_phenomena_v5.py': '4326c53debdfef06fa7cfd9f9f79c119a7473da19b5cd3de82e46ea483d5cb6c',
        'evaluate_new_phenomena_v6.py': '42aa98a6bece876415af502cd1e8109d0a24bff0f4a7ed1447d3084ca6d8d31b',
        'run_native30_fhsc_cohort_v1.py': 'dc26465459df133b89d6023bb1c732e91ebab5b63ff02143280b50c383e7b3cc'}
IDENTITY = ('id', 'source_group', 'label', 'role', 'group_id', 'component_id')
SOURCES = {'ACE-Step': 400, 'FMA': 354, 'HeartMuLa': 366, 'MTG-Jamendo': 500,
           'Mureka_v9': 500, 'Suno': 400, 'Udio': 500, 'human_maestro_v3': 300,
           'human_medleydb': 168, 'human_moisesdb': 239, 'human_saraga_hindustani_v1': 103}
SCOPE = {'status': 'new_protocol_metadata_draft_not_frozen_for_evaluation',
         'audio_files_opened': 0, 'feature_files_opened': 0, 'classifier_fits': 0,
         'predictions_computed': 0, 'cohort_admitted': False, 'fitting_authorized': False,
         'scoring_authorized': False, 'winner_selection': False, 'threshold_tuning': False,
         'model_freeze': False, 'M_predictor': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hash_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def path_checked(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.resolve() == path and not path.is_symlink(),
            'canonical unredirected absolute path required')
    return path


def binding(path, expected=None):
    path = path_checked(path)
    require(path.is_file(), 'regular metadata/code file required')
    observed = sha(path)
    require(expected is None or observed == expected, 'pinned metadata/code SHA mismatch: ' + str(path))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': observed}


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def hash_fold(value, folds=BUCKETS, seed=SEED):
    """Exact base hash_fold arithmetic and seed|group serialization."""
    digest = hashlib.sha256(f'{seed}|{value}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % folds


def rank_hash(group, seed=SEED):
    """Exact base deterministic_quantity ranking salt and hex digest."""
    return hashlib.sha256(f'{seed}|{group}'.encode()).hexdigest()


def id_set_hash(values):
    """Exact v2 id_set_hash; deliberately distinct from JSON object hashes."""
    payload = ''.join(f'{value}\n' for value in sorted({str(value) for value in values}))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def indexed(rows):
    result = {}
    for row in rows:
        require(all(isinstance(row[k], str) and row[k] for k in IDENTITY), 'nonempty string identities required')
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['id']) is not None, 'unsafe row ID')
        require(row['id'] not in result, 'duplicate row ID')
        require(row['label'] in {'0', '1'}, 'binary item labels required')
        result[row['id']] = row
    return result


class Dependencies:
    """Closure over the FULL pinned graph, not only currently selected rows."""
    def __init__(self, screen, selected):
        self.parents = {}
        def union(a, b):
            left, right = self.root(a), self.root(b)
            if left != right:
                self.parents[max(left, right)] = min(left, right)
        component_ids, member_ids, protected = set(), {}, set()
        for component in screen['components']:
            cid = component['component_id']
            require(cid not in component_ids, 'duplicate screen component')
            component_ids.add(cid)
            node = ('component', cid)
            self.root(node)
            for ident in component['members']:
                require(ident not in member_ids, 'member occurs in multiple declared screen components')
                member_ids[ident] = cid
                union(node, ('id', ident))
            for token in component.get('link_tokens', []):
                union(node, ('token', token))
                if token.startswith('conditioning:'):
                    union(node, ('group', token[len('conditioning:'):]))
            if component['protected_relationships']:
                protected.add(node)
        for row in screen['rows']:
            require(row['component_id'] in component_ids and member_ids.get(row['id']) == row['component_id'], 'screen graph row membership')
            union(('id', row['id']), ('group', row['group_id']))
            if row['role'] != 'development':
                protected.add(('id', row['id']))
        for row in selected:
            require(member_ids.get(row['id']) == row['component_id'], 'selected component membership')
            union(('id', row['id']), ('group', row['group_id']))
        self.protected_roots = {self.root(node) for node in protected}
        self.row_roots = {row['id']: self.root(('id', row['id'])) for row in selected}

    def root(self, node):
        self.parents.setdefault(node, node)
        while self.parents[node] != node:
            self.parents[node] = self.parents[self.parents[node]]
            node = self.parents[node]
        return node

    def roots(self, rows):
        return {self.row_roots[row['id']] for row in rows}


def population(rows):
    groups = defaultdict(set)
    for row in rows:
        groups[row['source_group']].add(row['group_id'])
    return {'rows': len(rows), 'ids': sorted(r['id'] for r in rows),
            'groups': sorted({r['group_id'] for r in rows}), 'components': sorted({r['component_id'] for r in rows}),
            'label_counts': dict(sorted(Counter(r['label'] for r in rows).items())),
            'source_rows': dict(sorted(Counter(r['source_group'] for r in rows).items())),
            'source_groups': {s: len(g) for s, g in sorted(groups.items())},
            'id_set_sha256': id_set_hash(r['id'] for r in rows)}


def shared_group_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    return {'global_groups': len(groups),
            'cross_source_groups': sum(len({r['source_group'] for r in rs}) > 1 for rs in groups.values()),
            'cross_label_groups': sum(len({r['label'] for r in rs}) > 1 for rs in groups.values())}


def reference_cell(rows, kind, source, bucket):
    """Base make_folds masks, before its class check; no v6 load_table guard.

    This intentionally records masks for mixed groups unsupported by the v6
    loader. It is a metadata comparator, not a claim v6 accepted this cohort.
    """
    if kind == 'ordinary_group_holdout_descriptive':
        test = [r for r in rows if hash_fold(r['group_id']) == bucket]
        train = [r for r in rows if hash_fold(r['group_id']) != bucket]
    else:
        label = '0' if kind == 'human_source_holdout' else '1'
        test = [r for r in rows if hash_fold(r['group_id']) == bucket and
                (r['source_group'] == source and r['label'] == label or r['label'] != label)]
        test_groups = {r['group_id'] for r in test}
        train = [r for r in rows if not (r['label'] == label and r['source_group'] == source)
                 and hash_fold(r['group_id']) != bucket and r['group_id'] not in test_groups]
    return train, test


def coupled_caps(rows):
    """New incremental selector: whole GLOBAL groups, hard caps per source.

    Always construct all five caps in order. At each step retain accepted
    groups, then scan unselected groups in the one frozen hash order. A shared
    group costs one unit in each represented remaining-training source.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['group_id']].append(row)
    ordering = sorted(grouped, key=lambda group: (rank_hash(group), group))
    sources = {group: {r['source_group'] for r in current} for group, current in grouped.items()}
    selected, counts, previous, results = set(), Counter(), set(), []
    available = Counter(source for group in grouped for source in sources[group])
    for cap in CAPS:
        added, blocked = [], []
        for group in ordering:
            if group in selected:
                continue
            if cap == 'all' or all(counts[source] < cap for source in sources[group]):
                selected.add(group)
                added.append(group)
                counts.update(sources[group])
            else:
                blocked.append(group)
        current = [r for r in rows if r['group_id'] in selected]
        ids = {r['id'] for r in current}
        require(previous <= ids, 'caps must be nested')
        require(all({r['id'] for r in grouped[g]} <= ids for g in selected), 'partial global group selected')
        require(cap == 'all' or all(n <= cap for n in counts.values()), 'hard source cap exceeded')
        previous = ids
        target = {s: available[s] if cap == 'all' else min(cap, available[s]) for s in sorted(available)}
        results.append((cap, current, {'selected_global_groups': len(selected), 'added_groups': added,
                                      'blocked_groups': blocked, 'available_source_groups': dict(sorted(available.items())),
                                      'selected_source_groups': {s: counts[s] for s in sorted(available)},
                                      'target_source_groups': target,
                                      'underfilled_sources': {s: {'target': target[s], 'selected': counts[s], 'deficit': target[s] - counts[s]}
                                                             for s in target if counts[s] < target[s]}}))
    require({r['id'] for r in results[-1][1]} == {r['id'] for r in rows}, 'all cap must retain all eligible training rows')
    return results


def coverage_reasons(train, test):
    reasons = []
    for name, rows in (('train', train), ('test', test)):
        for label in ('0', '1'):
            groups = {r['group_id'] for r in rows if r['label'] == label}
            if not groups:
                reasons.append(name + '_class_' + label + '_absent')
            elif name == 'train' and len(groups) < 2:
                reasons.append('train_class_' + label + '_fewer_than_two_global_groups')
    return reasons


def make_schedule(rows, screen):
    """Pure metadata Python API; reduced fixtures are not available via CLI."""
    original = indexed(rows)
    require(all(r['role'] == 'development' for r in rows), 'input rows must be development')
    source_labels = defaultdict(set)
    for row in rows:
        source_labels[row['source_group']].add(row['label'])
    require(all(len(labels) == 1 for labels in source_labels.values()), 'source has conflicting immutable labels')
    graph = Dependencies(screen, rows)
    protected = [r for r in rows if graph.row_roots[r['id']] in graph.protected_roots]
    eligible = [r for r in rows if graph.row_roots[r['id']] not in graph.protected_roots]
    require({r['id'] for r in eligible}.isdisjoint(r['id'] for r in protected)
            and len(eligible) + len(protected) == len(original), 'protected/eligible exact partition')
    kinds = [(label, kind) for label, kind in (('0', 'human_source_holdout'), ('1', 'generator_holdout'))]
    intended = [(kind, source, bucket) for label, kind in kinds for source in
                sorted(s for s, labels in source_labels.items() if labels == {label}) for bucket in range(BUCKETS)]
    intended += [('ordinary_group_holdout_descriptive', '__all_sources__', bucket) for bucket in range(BUCKETS)]
    cells, schedules, omitted = [], [], []
    for kind, source, bucket in intended:
        identity = {'fold_type': kind, 'heldout_source': source, 'opposite_group_fold': bucket}
        fold_uid = value_hash(identity)[:24]
        reference, test = reference_cell(eligible, kind, source, bucket)
        held = [] if kind == 'ordinary_group_holdout_descriptive' else [r for r in eligible if r['source_group'] == source]
        held_roots, test_roots = graph.roots(held), graph.roots(test)
        uncapped = [r for r in reference if graph.row_roots[r['id']] not in held_roots | test_roots]
        test_ids, train_ids = {r['id'] for r in test}, {r['id'] for r in uncapped}
        exclusions = defaultdict(list)
        for row in eligible:
            if row['id'] in test_ids or row['id'] in train_ids:
                continue
            root = graph.row_roots[row['id']]
            reason = ('all_held_source_dependency_closure' if root in held_roots else
                      'test_dependency_closure' if root in test_roots else
                      'global_hash_bucket_exclusion' if hash_fold(row['group_id']) == bucket else None)
            require(reason is not None, 'unaccounted excluded row')
            exclusions[reason].append(row)
        require(not test_ids & train_ids and len(test_ids) + len(train_ids) + sum(map(len, exclusions.values())) == len(eligible),
                'fold exact item partition')
        require(not {r['group_id'] for r in uncapped} & {r['group_id'] for r in test}
                and not {r['component_id'] for r in uncapped} & {r['component_id'] for r in test}
                and not graph.roots(uncapped) & (held_roots | test_roots | graph.protected_roots), 'global/component/transitive leakage')
        purged = [r for r in reference if r['id'] not in train_ids]
        reference_sources = {r['source_group'] for r in reference}
        remaining_sources = {r['source_group'] for r in uncapped}
        reasons = coverage_reasons(uncapped, test)
        cell = {**identity, 'fold_uid': fold_uid, 'status': 'omitted' if reasons else 'eligible_metadata_cell',
                'omission_reasons': reasons, 'test': population(test), 'uncapped_train': population(uncapped),
                'v6_style_reference_train_before_dependency_purge': population(reference),
                'additional_dependency_purged_from_reference': population(purged),
                'sources_fully_removed_by_dependency_purge': sorted(reference_sources - remaining_sources),
                'all_held_source_seed': population(held),
                'excluded_partition': {reason: population(items) for reason, items in sorted(exclusions.items())}}
        cells.append(cell)
        if reasons:
            omitted.append({**identity, 'fold_uid': fold_uid, 'reasons': reasons})
        previous = set()
        for cap, train, diagnostics in coupled_caps(uncapped):
            ids = {r['id'] for r in train}
            require(previous <= ids <= train_ids and not ids & test_ids, 'cap membership/nesting')
            previous = ids
            reasons = coverage_reasons(train, test)
            record = {**identity, 'fold_uid': fold_uid, 'quantity': cap, 'train': population(train),
                      'test_population_ref': fold_uid, 'uncapped_train_population_ref': fold_uid,
                      'test_id_set_sha256': id_set_hash(test_ids), 'uncapped_train_id_set_sha256': id_set_hash(train_ids),
                      'cap_diagnostics': diagnostics, 'status': 'omitted' if reasons else 'eligible_metadata_cell',
                      'omission_reasons': reasons}
            record['schedule_uid'] = value_hash(record)[:24]
            schedules.append(record)
    experiments = [{'combination': combination, 'schedule_uid': schedule['schedule_uid'], 'status': schedule['status'],
                    'feature_mode': mode, 'analysis_role': 'primary' if mode == PRIMARY_MODE else 'diagnostic'}
                   for schedule in schedules for combination in COMBINATIONS
                   for mode in ((PRIMARY_MODE, *DIAGNOSTIC_MODES) if schedule['quantity'] == 'all' else (PRIMARY_MODE,))]
    viable = [r for r in schedules if r['status'] == 'eligible_metadata_cell']
    mode_accounting = {role: {'intended_cells': sum(e['analysis_role'] == role for e in experiments),
                             'eligible_cells': sum(e['analysis_role'] == role and e['status'] == 'eligible_metadata_cell' for e in experiments),
                             'omitted_cells': sum(e['analysis_role'] == role and e['status'] == 'omitted' for e in experiments)}
                       for role in ('primary', 'diagnostic')}
    return {'version': VERSION, 'protocol': 'new_transitive_source_purge_and_incremental_coupled_global_group_caps_v1',
            'seed': SEED, 'global_hash_buckets': BUCKETS, 'quantities': list(CAPS), 'families': list(FAMILIES),
            'combinations': COMBINATIONS, 'primary_feature_mode': PRIMARY_MODE,
            'diagnostic_feature_modes': list(DIAGNOSTIC_MODES), 'diagnostic_quantities': ['all'],
            'input_population': population(rows), 'eligible_population': population(eligible),
            'protected_excluded_population': population(protected), 'shared_groups': shared_group_summary(eligible),
            'fold_cells': cells, 'cap_schedules': schedules, 'combination_schedule_cells': experiments,
            'omitted_fold_cells': omitted, 'schedule_sha256': value_hash(schedules),
            'accounting': {'intended_fold_cells': len(cells), 'eligible_uncapped_fold_cells': len(cells) - len(omitted),
                           'omitted_uncapped_fold_cells': len(omitted), 'intended_cap_cells': len(schedules),
                           'eligible_cap_cells': len(viable), 'omitted_cap_cells': len(schedules) - len(viable),
                           'intended_combination_cap_cells': len(experiments),
                           'eligible_combination_cap_cells': sum(a['eligible_cells'] for a in mode_accounting.values()),
                           'primary': mode_accounting['primary'], 'diagnostic': mode_accounting['diagnostic'],
                           'eligible_fold_types': dict(Counter(c['fold_type'] for c in cells if c['status'] != 'omitted'))},
            'semantics': {'v6_identical_primitives': ['hash_fold(seed|group)', 'group rank SHA256(seed|group)', 'id_set_hash(sorted unique IDs plus newlines)'],
                          'new_vs_v6': ['global groups may cross both source and item label',
                                        'transitive components and global groups of ALL held-source rows are purged from training',
                                        'full protected graph closure excluded before every fold',
                                        'incremental coupled whole-global-group caps replace independent per-source ranking'],
                          'cap_order_required': list(CAPS), 'hard_cap_unit': 'unique global groups represented within each remaining training source',
                          'cap_rows': 'retain every uncapped-training row of each selected global group',
                          'group_tie_break': 'group_id lexical only if rank hashes collide',
                          'reference_comparator': 'base make_folds masks only; literal v6 loader would reject mixed-source/label groups',
                          'shared_candidate_rows': 'all127 family subsets share exact schedule references at every cap, without feature availability filtering',
                          'diagnostic_modes': 'median_only and missingness_only enumerate all127 subsets at all-cap only; exact primary schedule IDs/rows reused',
                          'ordinary_folds': 'descriptive only; separate from Human-source and generator-heldout arms',
                          'class_coverage': 'both classes in train and test; at least two global groups per training class',
                          'historical_comparison': 'new cohort, duration and purging/caps; cannot interpret differences from historical v6 outcomes causally'}, **SCOPE}


def validate_prepared_contract(contract, plan, screen):
    require(contract.get('version') == 'run_native30_fhsc_cohort_v1'
            and contract.get('status') == 'frozen_before_any_measurement_audio_reads', 'prepared cohort contract version/status')
    require(contract.get('expected_count') == 3830 and contract.get('source_counts') == SOURCES
            and contract.get('human') == 1664 and contract.get('ai') == 2166, 'prepared exact cohort counts')
    require(contract.get('classifier_fits') == 0 and contract.get('cohort_admitted') is False
            and contract.get('source_selection_changed') is False, 'prepared scope')
    rows, planned, screened = indexed(contract['rows']), indexed(plan['rows']), indexed(screen['rows'])
    require(len(rows) == 3830 and len(planned) == 3869 and set(rows) <= set(planned), 'prepared/plan IDs')
    require(Counter(r['source_group'] for r in rows.values()) == SOURCES
            and Counter(r['label'] for r in rows.values()) == {'0': 1664, '1': 2166}, 'prepared source/label roster')
    absent = set(planned) - set(rows)
    require(len(absent) == 39 and all(planned[i]['source_group'] == 'FMA' and planned[i]['label'] == '0'
                                    and planned[i]['origin_set'] == 'native30_evidence_v2' for i in absent), 'only39 FMA short exclusions')
    components = {r['component_id']: r for r in screen['components']}
    require(len(components) == len(screen['components']), 'duplicate components')
    for ident, row in rows.items():
        p, s = planned[ident], screened[ident]
        c = components[row['component_id']]
        require(all(row[k] == p[k] == s[k] for k in IDENTITY), 'immutable cohort/plan/screen identity mismatch')
        require(row['origin_plan_row_sha256'] == value_hash(p) and row['screen_row_sha256'] == value_hash(s)
                and row['component_sha256'] == value_hash(c), 'prepared row graph/hash binding')
        require(row['role'] == 'development' and s['duration_exposure_candidate'] is True and not s['exclusion_reasons']
                and ident in c['members'] and ident in c['candidate_screen_members'] and not c['protected_relationships'],
                'prepared cohort contains excluded/protected identity')
    return [rows[i] for i in sorted(rows)]


def build(contract_path, contract_sha, plan_path, screen_path):
    require(hash_string(contract_sha), 'parent-pinned prepared contract SHA is mandatory')
    paths = [path_checked(p) for p in (contract_path, plan_path, screen_path)]
    # Deliberately read only these three metadata JSONs and pinned CODE files.
    # Do not follow contract.bindings into WAVs or descriptors/feature products.
    require(all(p.suffix == '.json' for p in paths), 'JSON metadata inputs only')
    bindings = {str(p): binding(p, digest) for p, digest in zip(paths, (contract_sha, PLAN_SHA, SCREEN_SHA))}
    here = Path(__file__).resolve().parent
    for name, expected in PINS.items():
        path = here / name
        bindings[str(path)] = binding(path, expected)
    for path in (Path(__file__).resolve(), here / ('test_' + VERSION + '.py')):
        bindings[str(path)] = binding(path)
    contract, plan, screen = map(read, paths)
    for path, expected in ((paths[1], PLAN_SHA), (paths[2], SCREEN_SHA)):
        require(contract['bindings'].get(str(path), {}).get('sha256') == expected, 'prepared contract must bind supplied plan/screen')
    rows = validate_prepared_contract(contract, plan, screen)
    result = make_schedule(rows, screen)
    require(result['eligible_population']['rows'] == 3830 and result['protected_excluded_population']['rows'] == 0,
            'prepared3830 must already exclude protected dependencies')
    result.update(input_bindings=bindings, prepared_contract_sha256=contract_sha,
                  metadata_read_scope='three input JSON metadata files, listed source code and interpreter executable only; upstream audio/features not opened or rehashed',
                  runtime={'python': platform.python_version(), 'executable': binding(Path(sys.executable).resolve()),
                           'implementation': 'Python standard library only; no numerical/model imports'})
    for entry in bindings.values():
        require(binding(entry['path']) == entry, 'metadata/code changed while scheduling')
    return result


def publish(result, output):
    output = path_checked(output)
    require(not output.exists() and output.parent.is_dir(), 'new output directory required; no overwrite')
    for entry in result['input_bindings'].values():
        require(binding(entry['path']) == entry, 'bound metadata/code changed before publication')
        require(not Path(entry['path']).is_relative_to(output), 'output overlaps input metadata/code')
    output.mkdir()
    path = output / 'schedule_draft.json'
    with path.open('xb') as stream:
        stream.write(canonical(result))
    require(read(path) == result, 'draft JSON roundtrip changed')
    product = binding(path)
    for entry in result['input_bindings'].values():
        require(binding(entry['path']) == entry, 'bound metadata/code changed before COMMIT')
    with (output / 'COMMIT.json').open('xb') as stream:
        stream.write(canonical({'status': 'committed_metadata_draft_not_evaluation_authorization',
                                'version': VERSION, 'products': {'schedule_draft.json': product},
                                'draft_sha256': product['sha256'], 'classifier_fits': 0, 'fitting_authorized': False}))
    return {'status': SCOPE['status'], 'output': str(output), 'draft_sha256': product['sha256'],
            'commit_sha256': sha(output / 'COMMIT.json'), 'accounting': result['accounting'], 'classifier_fits': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort-contract', 'cohort-contract-sha256', 'plan', 'screen', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = build(args.cohort_contract, args.cohort_contract_sha256, args.plan, args.screen)
    print(canonical(publish(result, args.output)).decode().strip())


if __name__ == '__main__':
    main()
