"""Reporting-only adapter for the Native30 eight-family BC evaluation.

The seven-family reporting implementation is loaded from the exact frozen v3
reporter under a private module name.  This file changes only the catalogue,
evaluator identity and immutable-boundary checks: all AUC, recall, pooling,
comparison and Markdown definitions remain those of the v3 reporter.  No
feature values are loaded, no numerical fit is run, and no model is selected.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid


VERSION = 'summarize_native30_evaluation_bc_v2'
V3_REPORTER_SHA = 'c6e8b1652c6225b47efbe1d65e31ce0948cd7d3c2da73d1686ab0c0b4d5729ad'
V3_EVALUATOR_SHA = 'c2544fe4ce98d88b906f502a576f788da46a4e28117ea7f3ce41802495fe7d1c'
V3_EVALUATOR_TEST_SHA = '32a9a5c3b4f645ad624d1f40720c66133e22727c8a1a509c4b3842f3b4fc411c'
EVALUATOR_SHA = 'a40d9074d9c017423aa002778ae857a155e52ad8a49fe7cef9a865b292d8b1b1'
EVALUATOR_TEST_SHA = '5ca444bef96b1db78dcc8ea8fea66ef018f2b591630a837808af4e005d03482d'
SPEC_SHA = 'a8a123dcd414336ae7b16917ba7acf15a33e37220dfd071db5135d75d98a205f'

# The BC package/schedule pins are provenance authorities.  They are checked
# by filename and SHA in the evaluator contract and again at report time.
PREPARER_SHA = '617de4ed687868a221c68664930d5757174d0dadec32df4663ee39e5621a413f'
PREPARER_TEST_SHA = 'cdbb9b567541fef898498dd111783a942be5ac710a3c175cebbdf4bb82fd0039'
BC_SCHEDULE_CODE_SHA = 'c432815b11e6dc315191f68f9722764b06be8e477b2d5db75308067ffe6764e5'
BC_SCHEDULE_TEST_SHA = 'be470106aabb8fd4864744b636d4057a9b1822bccf0a4261396f4c8be1bccf5d'
OLD_SCHEDULE_SHA = '4e12300a7fe43b4f3024c1cd9e675c25b0ade3af25dded0d688b8d8f2fcab7a6'
OLD_SCHEDULE_COMMIT_SHA = '5d0e329d9ef0a794e32acbf5bee06a4a09a58b453e1741ba6232fbfb32347233'

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / 'NATIVE30_REPORTING_CONTRACT_EN.md'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.resolve() == path
            and not path.is_symlink(), 'canonical unredirected path required')
    return path


def binding(path, expected=None):
    path = safe(path)
    require(path.is_file(), 'regular bound file required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(expected is None or digest == expected, 'file SHA mismatch: ' + str(path))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def id_hash(ids):
    return hashlib.sha256(''.join(x + '\n' for x in sorted(set(ids))).encode()).hexdigest()


def read(path, canonical_required=True):
    path = safe(path)
    raw = path.read_bytes()

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


def load_module(path, pin, name):
    path = safe(path)
    require(path.is_file(), 'regular pinned code file required')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == pin,
            'unapproved/changed code pin: ' + name)
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path, 'wrong loaded module origin')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'module loader unavailable: ' + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


# Load the unchanged reporter privately.  In particular, importing this file
# does not import NumPy, pandas, BASE, or the evaluator implementation.
CORE = load_module(HERE / 'summarize_native30_evaluation_v3.py',
                   V3_REPORTER_SHA, '_native30_bc_reporter_v3_core')


def _literal_assignment(path, name):
    """Read one top-level literal assignment without executing source code."""
    path = safe(path)
    tree = ast.parse(path.read_text())
    candidates = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            candidates.append(node)
    require(len(candidates) == 1, 'one literal source assignment required: ' + name)
    try:
        return ast.literal_eval(candidates[0].value)
    except Exception as error:
        raise ValueError('nonliteral source assignment: ' + name) from error


# Derive the old order from the pinned v3 evaluator source rather than
# duplicating 54 feature names.  The pinned schedule adapter's restricted AST
# reader handles the source's literal/list-comprehension catalogue without
# executing evaluator code.  The BC source pin is separately checked and its
# literal BC_FEATURE assignment is checked before appending it.
_OLD_EVALUATOR_PATH = HERE / 'evaluate_native30_v3.py'
_BC_EVALUATOR_PATH = HERE / 'evaluate_native30_bc_v1.py'
_SCHEDULE_ADAPTER = load_module(
    HERE / 'plan_native30_evaluation_schedule_bc_v1.py', BC_SCHEDULE_CODE_SHA,
    '_native30_bc_schedule_adapter_for_reporting')
_OLD_FAMILY_CONFIG = _SCHEDULE_ADAPTER.family_config_from_source(
    _OLD_EVALUATOR_PATH.read_text())


def _bc_feature_from_source(path):
    tree = ast.parse(safe(path).read_text())
    assignments = [node for node in tree.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'FAMILIES']
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict),
            'one source family expansion assignment required')
    for key, value in zip(assignments[0].value.keys, assignments[0].value.values):
        if isinstance(key, ast.Constant) and key.value == 'BC':
            require(isinstance(value, ast.List) and len(value.elts) == 1
                    and isinstance(value.elts[0], ast.Constant)
                    and isinstance(value.elts[0].value, str),
                    'literal BC feature source assignment required')
            return value.elts[0].value
    raise ValueError('literal BC family source assignment missing')


BC_FEATURE = _bc_feature_from_source(_BC_EVALUATOR_PATH)
require(list(_OLD_FAMILY_CONFIG) == ['S', 'D', 'R', 'P', 'F', 'H', 'SC']
        and all(isinstance(value, list) for value in _OLD_FAMILY_CONFIG.values()),
        'pinned seven-family source catalogue required')
require(BC_FEATURE == 'BC_b2_500_750_1250hz_center8s_median',
        'pinned BC feature assignment changed')
FAMILY_CONFIG = {**_OLD_FAMILY_CONFIG, 'BC': [BC_FEATURE]}
FAMILIES = tuple(FAMILY_CONFIG)
OLD_FEATURES = sum((_OLD_FAMILY_CONFIG[key] for key in _OLD_FAMILY_CONFIG), [])
FEATURES = sum((FAMILY_CONFIG[key] for key in FAMILY_CONFIG), [])
OLD_COMBINATIONS = ['+'.join(c) for n in range(1, 8)
                    for c in itertools.combinations(FAMILIES[:7], n)]
COMBINATIONS = ['+'.join(c) for n in range(1, 9)
                for c in itertools.combinations(FAMILIES, n)]
CAPS = (25, 50, 100, 200, 'all')
MODES = ('values_plus_missing', 'median_only', 'missingness_only')
PROTOCOLS = ('human_source_holdout', 'generator_holdout', 'ordinary_group_holdout_descriptive')
REFERENCE = 'S+D+R+P'
SOURCES = dict(CORE.SOURCES)

require(len(OLD_FEATURES) == 54 and len(FEATURES) == 55
        and len(set(FEATURES)) == 55 and len(OLD_COMBINATIONS) == 127
        and len(COMBINATIONS) == 255 and [c for c in COMBINATIONS if 'BC' not in c.split('+')] == OLD_COMBINATIONS,
        'exact seven-to-eight family catalogue projection required')

EXPECTED = {'rows': 3830, 'sources': SOURCES, 'labels': dict(CORE.EXPECTED['labels']),
            'buckets': 5, 'caps': CAPS, 'combinations': COMBINATIONS,
            'intended': 107100, 'eligible': 103530, 'primary': 73950,
            'diagnostic': 29580}
PRODUCER_SCOPE = dict(CORE.PRODUCER_SCOPE)
SCOPE = dict(CORE.SCOPE)
BC_SCOPE = {'BC_classifier_admitted': False, 'BC_diagnostics_are_predictors': False}


@contextmanager
def _bound_core():
    """Rebind only the private v3 reporter during delegated calls."""
    names = {
        'VERSION': VERSION, 'EVALUATOR_SHA': EVALUATOR_SHA,
        'EVALUATOR_TEST_SHA': EVALUATOR_TEST_SHA, 'SPEC_SHA': SPEC_SHA,
        'FAMILIES': FAMILIES, 'COMBINATIONS': COMBINATIONS, 'CAPS': CAPS,
        'MODES': MODES, 'PROTOCOLS': PROTOCOLS, 'REFERENCE': REFERENCE,
        'SOURCES': SOURCES, 'EXPECTED': EXPECTED,
        'PRODUCER_SCOPE': PRODUCER_SCOPE, 'SCOPE': SCOPE,
    }
    old = {name: getattr(CORE, name) for name in names}
    for name, value in names.items():
        setattr(CORE, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(CORE, name, value)


# Public aliases retain the v3 numerical/reporting definitions.  Calls are
# wrapped so a normal import of summarize_native30_evaluation_v3 is untouched.
def auc(predictions):
    with _bound_core():
        return CORE.auc(predictions)


def correct(row):
    with _bound_core():
        return CORE.correct(row)


def recording_recalls(predictions):
    with _bound_core():
        return CORE.recording_recalls(predictions)


def pool_predictions(predictions):
    with _bound_core():
        return CORE.pool_predictions(predictions)


def compare(current, reference, kind):
    with _bound_core():
        return CORE.compare(current, reference, kind)


def model_uid(task, contract_sha):
    with _bound_core():
        return CORE.model_uid(task, contract_sha)


def validate_task(task, expected):
    with _bound_core():
        return CORE.validate_task(task, expected)


def receipt(path, entry, contract_sha, expected):
    with _bound_core():
        return CORE.receipt(path, entry, contract_sha, expected)


def evaluation_inventory(root):
    with _bound_core():
        return CORE.evaluation_inventory(root)


def _cell_accounting(cells, folds, caps):
    result = {}
    for role in ('primary', 'diagnostic'):
        selected = [cell for cell in cells if cell.get('analysis_role') == role]
        eligible = [cell for cell in selected if cell.get('status') == 'eligible_metadata_cell']
        result[role] = {'intended_cells': len(selected), 'eligible_cells': len(eligible),
                        'omitted_cells': len(selected) - len(eligible)}
    result['total'] = {key: sum(result[role][key] for role in ('primary', 'diagnostic'))
                       for key in ('intended_cells', 'eligible_cells', 'omitted_cells')}
    return result


def _combination_features(combination):
    return sum((FAMILY_CONFIG[key] for key in combination.split('+')), [])


def schedule_audit(schedule, expected=EXPECTED):
    """Recompute schedule cells, cap/fold counts, and the old-127 projection."""
    require(isinstance(schedule, dict), 'BC schedule document required')
    require(schedule.get('version') == 'plan_native30_evaluation_schedule_bc_v1'
            and schedule.get('families') == list(FAMILIES)
            and schedule.get('combinations') == COMBINATIONS
            and schedule.get('family_config') == FAMILY_CONFIG
            and schedule.get('feature_names') == FEATURES,
            'exact eight-family schedule schema')
    require(schedule.get('combination_feature_names') ==
            {combination: _combination_features(combination) for combination in COMBINATIONS},
            'combination feature order/schema changed')
    require(schedule.get('primary_feature_mode') == MODES[0]
            and schedule.get('diagnostic_feature_modes') == list(MODES[1:])
            and schedule.get('quantities') == list(CAPS)
            and schedule.get('diagnostic_quantities') == ['all'],
            'arm/cap/missingness schedule schema changed')
    require(schedule.get('BC_standalone_reference') is None,
            'BC standalone reference must remain unavailable')
    expected_pairs = [{'reference_combination': combination,
                       'bc_combination': combination + '+BC'}
                      for combination in OLD_COMBINATIONS]
    require(schedule.get('bc_matched_comparisons') == expected_pairs,
            'predeclared BC matched comparison catalogue changed')

    fold_cells = schedule.get('fold_cells')
    cap_schedules = schedule.get('cap_schedules')
    cells = schedule.get('combination_schedule_cells')
    require(isinstance(fold_cells, list) and isinstance(cap_schedules, list)
            and isinstance(cells, list), 'schedule fold/cap/cell lists required')
    folds = {row.get('fold_uid'): row for row in fold_cells}
    caps = {row.get('schedule_uid'): row for row in cap_schedules}
    require(len(folds) == len(fold_cells) and len(caps) == len(cap_schedules)
            and all(isinstance(uid, str) and re.fullmatch(r'[0-9a-f]{24}', uid)
                    for uid in folds | caps), 'schedule UID uniqueness/schema')
    for fold in fold_cells:
        require(fold.get('fold_type') in PROTOCOLS and type(fold.get('opposite_group_fold')) is int
                and 0 <= fold['opposite_group_fold'] < expected['buckets']
                and fold.get('status') in {'eligible_metadata_cell', 'omitted'},
                'fold schema/status')
        test = fold.get('test')
        require(isinstance(test, dict) and type(test.get('rows')) is int and test['rows'] >= 0
                and isinstance(test.get('ids'), list) and len(test['ids']) == test['rows'],
                'fold test population schema')
    for cap in cap_schedules:
        require(cap.get('quantity') in CAPS and cap.get('fold_uid') in folds
                and cap.get('status') in {'eligible_metadata_cell', 'omitted'}
                and isinstance(cap.get('train'), dict)
                and type(cap['train'].get('rows')) is int and cap['train']['rows'] >= 0,
                'cap schedule schema/status')
    seen_cells = set()
    for cell in cells:
        key = (cell.get('schedule_uid'), cell.get('combination'), cell.get('feature_mode'))
        require(key not in seen_cells, 'duplicate combination schedule cell')
        seen_cells.add(key)
        require(cell.get('schedule_uid') in caps
                and cell.get('combination') in COMBINATIONS
                and cell.get('feature_mode') in MODES
                and cell.get('status') in {'eligible_metadata_cell', 'omitted'},
                'combination schedule cell schema')
        role = cell.get('analysis_role')
        require(role in {'primary', 'diagnostic'}, 'unknown schedule analysis role')
        require((role == 'primary') == (cell['feature_mode'] == MODES[0]),
                'schedule analysis role/missingness mode mismatch')
        cap = caps[cell['schedule_uid']]
        require(cell['status'] == cap['status'], 'cell/cap status mismatch')
        require(cell['feature_mode'] == MODES[0] or cap['quantity'] == 'all',
                'diagnostic outside all-cap')

    counts = _cell_accounting(cells, folds, caps)
    schedule_accounting = schedule.get('accounting')
    require(isinstance(schedule_accounting, dict), 'schedule accounting missing')
    declared = {
        'intended_fold_cells': len(fold_cells),
        'eligible_uncapped_fold_cells': sum(row['status'] == 'eligible_metadata_cell' for row in fold_cells),
        'omitted_uncapped_fold_cells': sum(row['status'] == 'omitted' for row in fold_cells),
        'intended_cap_cells': len(cap_schedules),
        'eligible_cap_cells': sum(row['status'] == 'eligible_metadata_cell' for row in cap_schedules),
        'omitted_cap_cells': sum(row['status'] == 'omitted' for row in cap_schedules),
        'intended_combination_cap_cells': counts['total']['intended_cells'],
        'eligible_combination_cap_cells': counts['total']['eligible_cells'],
        'omitted_combination_cap_cells': counts['total']['omitted_cells'],
        'eligible_fold_types': dict(Counter(row['fold_type'] for row in fold_cells
                                            if row['status'] == 'eligible_metadata_cell')),
        'primary': counts['primary'], 'diagnostic': counts['diagnostic'],
    }
    for key, value in declared.items():
        require(schedule_accounting.get(key) == value,
                'schedule accounting replay mismatch: ' + key)

    projection = schedule.get('original_seven_family_projection')
    require(isinstance(projection, dict)
            and projection.get('combinations') == OLD_COMBINATIONS
            and projection.get('feature_names') == OLD_FEATURES,
            'old127 projection catalogue changed')
    old_cells = [cell for cell in cells if 'BC' not in cell['combination'].split('+')]
    bc_cells = [cell for cell in cells if 'BC' in cell['combination'].split('+')]
    require(projection.get('experiment_cells_sha256') == value_hash(old_cells)
            and projection.get('accounting') == _cell_accounting(old_cells, folds, caps),
            'old127 projection cells/accounting changed')
    additional = schedule.get('additional_BC_projection')
    require(isinstance(additional, dict) and additional.get('combination_count') == 128
            and additional.get('accounting') == _cell_accounting(bc_cells, folds, caps),
            'BC projection accounting changed')

    # The prospective counts are assertions only after the list-derived
    # accounting above has passed.  Reduced Python fixtures are intentionally
    # permitted for tests; the CLI always uses EXPECTED and the full roster.
    if expected.get('rows') == EXPECTED['rows']:
        require(counts['total']['intended_cells'] == EXPECTED['intended']
                and counts['primary']['eligible_cells'] == EXPECTED['primary']
                and counts['diagnostic']['eligible_cells'] == EXPECTED['diagnostic'],
                'actual 255-cell accounting differs from approved protocol')
        require(counts['total']['eligible_cells'] == EXPECTED['eligible'],
                'actual 103530 eligible accounting differs from approved protocol')
        require(len(fold_cells) == 60 and len(cap_schedules) == 300
                and sum(row['status'] == 'eligible_metadata_cell'
                        for row in fold_cells) == 58,
                'actual 58-eligible-fold/300-cap schedule shape required')
        inputs = schedule.get('input_bindings')
        require(isinstance(inputs, dict)
                and inputs.get('original_schedule', {}).get('sha256') == OLD_SCHEDULE_SHA
                and inputs.get('original_commit', {}).get('sha256') == OLD_SCHEDULE_COMMIT_SHA,
                'original schedule/COMMIT SHA pins changed')
        original_schedule_path = Path(inputs['original_schedule']['path'])
        local_original_schedule_path = (HERE.parent / 'results' /
                                        'native30_evaluation_schedule_draft_v1' /
                                        'schedule_draft.json').resolve()
        candidates = [original_schedule_path, local_original_schedule_path]
        original_schedule = None
        for candidate in candidates:
            if candidate.is_file() and not candidate.is_symlink():
                try:
                    binding(candidate, OLD_SCHEDULE_SHA)
                except ValueError:
                    continue
                original_schedule = read(candidate)
                break
        require(original_schedule is not None
                and schedule.get('schedule_sha256') ==
                original_schedule.get('schedule_sha256'),
                'old schedule semantic hash changed')
    return {'counts': counts, 'folds': folds, 'caps': caps,
            'cells': cells, 'old_cells': old_cells, 'bc_cells': bc_cells,
            'schedule_sha256': value_hash(schedule),
            'schedule_accounting': schedule_accounting,
            'old_projection': projection, 'additional_BC_projection': additional}


def _entry_by_basename(bindings, basename, pin):
    entries = [entry for entry in bindings.values()
               if isinstance(entry, dict) and Path(entry.get('path', '')).name == basename]
    require(len(entries) == 1 and entries[0].get('sha256') == pin
            and binding(entries[0]['path']) == entries[0],
            'frozen source lineage changed: ' + basename)
    return entries[0]


def _verify_package(package_entry, expected, schedule):
    """Verify the actual BC package-shaped product boundary without fits."""
    require(isinstance(package_entry, dict)
            and set(package_entry) == {'path', 'bytes', 'sha256'},
            'BC package COMMIT binding schema')
    commit_path = safe(package_entry['path'])
    package_root = commit_path.parent
    require(binding(commit_path) == package_entry, 'BC package COMMIT changed')
    require({path.name for path in package_root.iterdir()} ==
            {'contract.json', 'metadata.json', 'features.json', 'lineage.json',
             'bc_coverage.json', 'COMMIT.json'}, 'BC package exact root inventory')
    commit = read(commit_path)
    require(commit.get('version') == 'prepare_native30_evaluation_inputs_bc_v1'
            and commit.get('status') == 'assembled_not_evaluated_not_authorized'
            and commit.get('expected_count') == expected['rows']
            and commit.get('all_source_evidence_deep_verified_before_and_after') is True
            and commit.get('audio_files_opened') == 0,
            'BC package COMMIT schema/scope')
    for key, value in BC_SCOPE.items():
        require(commit.get(key) == value, 'BC package COMMIT admission scope')
    product_names = ('contract.json', 'metadata.json', 'features.json',
                     'lineage.json', 'bc_coverage.json')
    products = {name: binding(package_root / name) for name in product_names}
    require(commit.get('products') == products, 'BC package product inventory/hash mismatch')
    package_contract, metadata, features, lineage, coverage = [
        read(package_root / name) for name in product_names]
    require(package_contract.get('version') == 'prepare_native30_evaluation_inputs_bc_v1'
            and package_contract.get('status') == 'assembled_not_evaluated_not_authorized'
            and package_contract.get('expected_count') == expected['rows']
            and package_contract.get('family_config') == FAMILY_CONFIG
            and package_contract.get('feature_names') == FEATURES
            and package_contract.get('feature_count') == 55,
            'BC package contract 55-feature schema')
    request = package_contract.get('request')
    require(isinstance(request, dict)
            and package_contract.get('request_sha256') == value_hash(request),
            'BC package request/hash schema')
    require(package_contract.get('schedule_accounting') == schedule['accounting']
            and package_contract.get('schedule_sha256') == schedule.get('schedule_sha256'),
            'BC package/schedule authority join')
    code_bindings = package_contract.get('code_bindings')
    require(isinstance(code_bindings, dict), 'BC package code binding map')
    _entry_by_basename(code_bindings, 'prepare_native30_evaluation_inputs_bc_v1.py', PREPARER_SHA)
    _entry_by_basename(code_bindings, 'test_prepare_native30_evaluation_inputs_bc_v1.py', PREPARER_TEST_SHA)
    metadata_schema = package_contract.get('metadata_schema')
    feature_schema = package_contract.get('feature_schema')
    lineage_schema = package_contract.get('lineage_schema')
    require(isinstance(metadata_schema, list) and isinstance(lineage_schema, list)
            and feature_schema == ['id', *FEATURES], 'BC package declared row schemas')
    require(isinstance(metadata, list) and isinstance(features, list)
            and isinstance(lineage, list)
            and len(metadata) == len(features) == len(lineage) == expected['rows'],
            'BC package row counts')
    ids = [row.get('id') for row in metadata]
    require(ids == sorted(ids) and len(ids) == len(set(ids))
            and ids == [row.get('id') for row in features]
            and ids == [row.get('id') for row in lineage],
            'BC package ordered unique ID join')
    require(all(set(row) == set(metadata_schema) for row in metadata)
            and all(set(row) == set(feature_schema) for row in features)
            and all(set(row) == set(lineage_schema) for row in lineage),
            'BC package exact metadata/feature/lineage row schema')
    require(all(all(value is None or (type(value) in (int, float) and math.isfinite(value))
                   for key, value in row.items() if key != 'id') for row in features),
            'BC package scalar/null missingness schema')
    source_counts = Counter(row.get('source_group') for row in metadata)
    labels = Counter(int(row.get('label')) for row in metadata)
    require(dict(source_counts) == expected['sources'] and dict(labels) == expected['labels'],
            'BC package source/label roster')
    require(isinstance(coverage, dict)
            and coverage.get('version') == 'prepare_native30_evaluation_inputs_bc_v1'
            and coverage.get('status') == 'descriptive_BC_availability_not_predictors_not_admission'
            and all(coverage.get(key) == value for key, value in BC_SCOPE.items()),
            'BC coverage scope schema')
    return {'root': package_root, 'commit': commit, 'contract': package_contract,
            'products': products, 'metadata_rows': len(metadata),
            'feature_rows': len(features), 'lineage_rows': len(lineage)}


def _validate_external_code_bindings(contract):
    bindings = contract.get('bindings')
    require(isinstance(bindings, dict), 'evaluator source binding map required')
    values = {
        'evaluate_native30_bc_v1.py': EVALUATOR_SHA,
        'test_evaluate_native30_bc_v1.py': EVALUATOR_TEST_SHA,
        'evaluate_native30_v3.py': V3_EVALUATOR_SHA,
        'test_evaluate_native30_v3.py': V3_EVALUATOR_TEST_SHA,
        'prepare_native30_evaluation_inputs_bc_v1.py': PREPARER_SHA,
        'test_prepare_native30_evaluation_inputs_bc_v1.py': PREPARER_TEST_SHA,
        'plan_native30_evaluation_schedule_bc_v1.py': BC_SCHEDULE_CODE_SHA,
        'test_plan_native30_evaluation_schedule_bc_v1.py': BC_SCHEDULE_TEST_SHA,
    }
    result = {}
    for basename, pin in values.items():
        result[basename] = _entry_by_basename(bindings, basename, pin)
    return result


def _validate_authorization(root, contract, contract_sha, expected):
    authorization = read(root / 'authorization.json')
    require(authorization.get('status') == 'parent_frozen_verified'
            and authorization.get('contract_sha256') == contract_sha
            and isinstance(authorization.get('binding'), dict)
            and binding(authorization['binding']['path']) == authorization['binding'],
            'evaluation authorization evidence changed')
    frozen = read(authorization['binding']['path'])
    require(frozen.get('version') == 'native30-evaluation-parent-freeze-bc-v1'
            and frozen.get('status') == 'parent_frozen_for_native30_fold_only_fitting'
            and frozen.get('stage') == contract['stage']
            and frozen.get('contract') == contract
            and frozen.get('contract_sha256') == contract_sha
            and frozen.get('fitting_authorized') is True
            and frozen.get('scoring_authorized') is True
            and frozen.get('independent_review', {}).get('reviewer') == 'root'
            and frozen.get('independent_review', {}).get('approved') is True
            and all(frozen.get(key) == value for key, value in PRODUCER_SCOPE.items()),
            'BC parent fit freeze scope/contract mismatch')
    for key, value in BC_SCOPE.items():
        if key in frozen:
            require(frozen[key] == value, 'BC parent freeze admission scope')
    return authorization


def _validate_contract_and_schedule(root, commit, contract, commit_entry, expected):
    contract_sha = value_hash(contract)
    require(commit.get('version') == 'evaluate_native30_bc_v1'
            and commit.get('status') == 'committed_fold_only_evaluation_not_model_selection'
            and commit.get('contract_sha256') == contract_sha
            and commit.get('all_input_output_runtime_bindings_end_rehashed') is True
            and commit.get('model_instances') == expected['eligible']
            and all(commit.get(key) == value for key, value in PRODUCER_SCOPE.items()),
            'complete pinned BC evaluator COMMIT required')
    require(contract.get('version') == 'evaluate_native30_bc_v1'
            and contract.get('stage') == 'native30_eight_family_schedule_only_development_evaluation_bc_v1'
            and contract.get('output_root') == str(root)
            and contract.get('expected_count') == expected['rows']
            and contract.get('families') == FAMILY_CONFIG
            and contract.get('feature_names') == FEATURES
            and contract.get('accounting') is not None
            and all(contract.get(key) == value for key, value in (*BC_SCOPE.items(), *PRODUCER_SCOPE.items())),
            'BC evaluator contract identity/count/scope mismatch')
    schedule_entry = contract.get('schedule_document')
    require(isinstance(schedule_entry, dict)
            and set(schedule_entry) == {'path', 'bytes', 'sha256'},
            'BC evaluator schedule binding required')
    schedule_path = safe(schedule_entry['path'])
    require(binding(schedule_path) == schedule_entry
            and value_hash(read(schedule_path)) == contract.get('schedule_sha256')
            and schedule_entry['sha256'] == contract.get('schedule_sha256'),
            'BC evaluator schedule binding/hash mismatch')
    schedule = read(schedule_path)
    schedule_proof = schedule_audit(schedule, expected)
    package_proof = _verify_package(contract.get('package_commit'), expected, schedule)
    external = _validate_external_code_bindings(contract)
    manifest = read(root / 'manifest.json')
    require(manifest.get('version') == 'evaluate_native30_bc_v1'
            and manifest.get('contract_sha256') == contract_sha
            and manifest.get('model_instances') == expected['eligible']
            and manifest.get('accounting') == contract['accounting']
            and manifest.get('training_transform_equation_prediction_replay_performed') is True
            and manifest.get('independent_full_publication_numerical_audit_performed') is False
            and all(manifest.get(key) == value for key, value in PRODUCER_SCOPE.items()),
            'BC evaluator manifest mismatch')
    authorization = _validate_authorization(root, contract, contract_sha, expected)
    return contract_sha, schedule, schedule_proof, package_proof, external, manifest, authorization


def inspect_evaluation(root, commit_sha, expected=EXPECTED):
    """Verify a complete BC evaluation and return receipt references only."""
    root = safe(root)
    commit_entry = binding(root / 'COMMIT.json', commit_sha)
    commit, contract = read(root / 'COMMIT.json'), read(root / 'contract.json')
    (contract_sha, schedule, schedule_proof, package_proof, external,
     manifest, authorization) = _validate_contract_and_schedule(
         root, commit, contract, commit_entry, expected)
    actual = evaluation_inventory(root)
    require(commit.get('products') == actual, 'complete BC evaluation product inventory/hash mismatch')
    models = {Path(name).stem: entry for name, entry in actual.items() if name.startswith('models/')}
    require(models == manifest.get('model_receipts') and len(models) == expected['eligible'],
            'complete BC model receipt inventory required')
    omissions = read(root / 'omitted.json')
    require(len(omissions) == manifest.get('omitted_instances') == expected['intended'] - expected['eligible'],
            'BC omitted task count mismatch')

    pools, cap_records, catalogue, roster = defaultdict(list), {}, {}, {}
    fold_rows = []

    def register(task, entry=None):
        validate_task(task, expected)
        key = (*CORE.task_key(task), task['opposite_group_fold'])
        require(key not in catalogue, 'duplicate BC intended task')
        catalogue[key] = task['status']
        reference = {key: value for key, value in task.items()
                     if key not in {'combination', 'feature_mode', 'analysis_role'}}
        uid = task['schedule_uid']
        require(uid not in cap_records or cap_records[uid] == reference,
                'BC candidate schedules do not share exact rows/status/reasons')
        cap_records[uid] = reference
        pools[CORE.task_key(task)].append({'entry': entry, 'schedule_uid': uid})

    for task in omissions:
        register(task)
    for index, (uid, entry) in enumerate(sorted(models.items()), 1):
        record = receipt(Path(entry['path']), entry, contract_sha, expected)
        task, predictions = record['task'], record['predictions']
        require(uid == model_uid(task, contract_sha), 'BC model UID/task mismatch')
        register(task, entry)
        for prediction in predictions:
            metadata = {key: prediction[key] for key in
                        ('id', 'label', 'source_group', 'group_id', 'component_id', 'role')}
            require(prediction['id'] not in roster or roster[prediction['id']] == metadata,
                    'BC item identity changes between evaluation models')
            roster[prediction['id']] = metadata
        fold_rows.append({'model_uid': uid, **{key: task[key] for key in
                          ('fold_type', 'heldout_source', 'quantity', 'combination',
                           'feature_mode', 'opposite_group_fold', 'schedule_uid')},
                          'train_rows': len(task['train_ids']),
                          'test_rows': len(predictions),
                          'single_model_fold_auc': auc(predictions),
                          'single_model_recording_weighted_BA':
                              recording_recalls(predictions)['balanced_accuracy']})
        if index % 1000 == 0:
            print(json.dumps({'event': 'report_bc_receipts_verified',
                              'verified': index, 'expected': len(models)}), flush=True)

    require(len(roster) == expected['rows']
            and dict(Counter(row['source_group'] for row in roster.values())) == expected['sources']
            and dict(Counter(row['label'] for row in roster.values())) == expected['labels'],
            'BC reporting roster/source/label mismatch')
    source_labels = defaultdict(set)
    for row in roster.values():
        source_labels[row['source_group']].add(row['label'])
    require(all(len(labels) == 1 for labels in source_labels.values()), 'BC source label changes')
    held_units = [(PROTOCOLS[label], source)
                  for source, labels in source_labels.items() for label in labels]
    held_units.append((PROTOCOLS[2], '__all_sources__'))
    wanted = {(protocol, source, cap, combination, mode, bucket)
              for protocol, source in held_units for cap in expected['caps']
              for combination in expected['combinations']
              for mode in (MODES if cap == 'all' else MODES[:1])
              for bucket in range(expected['buckets'])}
    require(set(catalogue) == wanted and len(wanted) == expected['intended'],
            'incomplete/extra BC intended reporting catalogue')
    for record in cap_records.values():
        require(set(record['train_ids']) | set(record['test_ids']) <= set(roster),
                'unknown BC task identity')
        train = [roster[item] for item in record['train_ids']]
        test = [roster[item] for item in record['test_ids']]
        require(not {row['group_id'] for row in train} & {row['group_id'] for row in test}
                and not {row['component_id'] for row in train} &
                {row['component_id'] for row in test},
                'BC reporting train/test group/component overlap')
        if record['fold_type'] != PROTOCOLS[2]:
            require(record['heldout_source'] not in {row['source_group'] for row in train},
                    'BC held source appears in training')
    for role in ('primary', 'diagnostic'):
        selected = [(key, status) for key, status in catalogue.items()
                    if (key[4] == MODES[0]) == (role == 'primary')]
        eligible = sum(status == 'eligible_metadata_cell' for _, status in selected)
        predictions = sum(row['test_rows'] for row in fold_rows
                          if (row['feature_mode'] == MODES[0]) == (role == 'primary'))
        require(contract['accounting'][role] == {
                    'intended': len(selected), 'eligible': eligible,
                    'omitted': len(selected) - eligible,
                    'prediction_rows': predictions}
                and eligible == expected[role],
                'BC evaluator role accounting mismatch')
    return {'root': root, 'commit': commit_entry, 'contract': contract,
            'contract_sha256': contract_sha, 'snapshot': actual,
            'authorization': authorization, 'pools': dict(pools),
            'cap_records': cap_records, 'roster': roster,
            'omissions': omissions, 'fold_rows': fold_rows,
            'expected': expected, 'external_bindings': external,
            'schedule': schedule, 'schedule_entry': contract['schedule_document'],
            'schedule_proof': schedule_proof, 'package_proof': package_proof,
            'manifest': manifest}


def summarize(published):
    """Run the unchanged v3 descriptive aggregation plus BC audit metadata."""
    with _bound_core():
        result = CORE.summarize(published)
    result.update({'version': VERSION, 'families': list(FAMILIES),
                   'family_config': FAMILY_CONFIG,
                   'feature_count': len(FEATURES),
                   'combination_count': len(COMBINATIONS), **BC_SCOPE})
    # ``inspect_evaluation`` always supplies these proofs.  Keeping this
    # small fallback makes the unchanged aggregation independently testable
    # with a reduced Python-only context, as in the frozen v3 reporter tests;
    # it cannot reach the CLI because the CLI first performs full inspection.
    if 'schedule_proof' in published and 'package_proof' in published:
        result.update({
            'schedule_accounting': published['schedule_proof']['counts'],
            'old127_schedule_accounting': published['schedule_proof']['old_projection']['accounting'],
            'BC_schedule_accounting': published['schedule_proof']['additional_BC_projection']['accounting'],
            'schedule_accounting_recomputed_from_cells': True,
            'old127_projection_verified': True,
            'BC_matched_comparison_catalogue_verified': True,
            'BC_package_rows_verified': published['package_proof']['metadata_rows'],
        })
    matched = []
    lookup = {(row['fold_type'], row['quantity'], row['combination'], row['feature_mode']): row
              for row in result['primary_protocol_cells'] + result['diagnostic_protocol_cells']}
    for combination in OLD_COMBINATIONS:
        reference = combination
        candidate = combination + '+BC'
        for mode in MODES:
            if mode != MODES[0]:
                quantity_values = ('all',)
            else:
                quantity_values = CAPS
            for protocol in PROTOCOLS:
                for quantity in quantity_values:
                    current = lookup.get((protocol, quantity, candidate, mode))
                    baseline = lookup.get((protocol, quantity, reference, mode))
                    if current is not None:
                        matched.append(compare(current, baseline, 'predeclared_BC_vs_reference'))
    result['comparisons']['predeclared_BC_matched_comparisons'] = matched
    result['scope_notes'].extend([
        'BC is a predeclared candidate scalar: it remains descriptive and is not admitted as a classifier or predictor.',
        'The eight-family schedule has 255 subsets and 55 features; intended/eligible cells are recomputed from the immutable cell list before approved full-cohort counts are asserted.',
        'The old seven-family 127-subset schedule projection is retained and hash-checked; BC comparisons are predeclared X versus X+BC pairs only.',
        'No pooled cross-model AUC, posthoc winner claim, threshold tuning, confidence interval, or causal historical subtraction is produced.',
    ])
    # Keep the v3 scope fields visible even if a caller supplied a custom
    # synthetic expected roster.  This is descriptive metadata, not selection.
    result.update(SCOPE)
    return result


def markdown_table(rows, columns):
    with _bound_core():
        return CORE.markdown_table(rows, columns)


def markdown_products(result, published):
    with _bound_core():
        return CORE.markdown_products(result, published)


def _recheck_schedule_and_package(published):
    schedule_entry = published['schedule_entry']
    require(binding(schedule_entry['path']) == schedule_entry,
            'BC schedule changed during reporting')
    schedule = read(schedule_entry['path'])
    require(value_hash(schedule) == published['contract']['schedule_sha256'],
            'BC schedule hash changed during reporting')
    proof = schedule_audit(schedule, published['expected'])
    require(proof['counts'] == published['schedule_proof']['counts'],
            'BC schedule accounting changed during reporting')
    package = _verify_package(published['contract']['package_commit'],
                              published['expected'], schedule)
    require(package['products'] == published['package_proof']['products'],
            'BC package products changed during reporting')


def recheck(published):
    require(binding(published['commit']['path']) == published['commit']
            and evaluation_inventory(published['root']) == published['snapshot']
            and binding(published['authorization']['binding']['path']) ==
            published['authorization']['binding'],
            'evaluation changed during reporting')
    for entry in published['external_bindings'].values():
        require(binding(entry['path']) == entry,
                'frozen evaluator code changed during reporting')
    _recheck_schedule_and_package(published)


@contextmanager
def read_lock(root):
    """Hold the evaluator's shared writer lock for the whole report."""
    path = safe(Path(root) / 'writer.lock')
    require(path.is_file(), 'completed BC evaluation writer lock required')
    with path.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def publish(published, output, report_bindings):
    output = safe(output)
    require(not output.exists() and output.parent.is_dir()
            and not output.is_relative_to(published['root'])
            and not published['root'].is_relative_to(output)
            and not any(Path(entry['path']).is_relative_to(output)
                        for entry in report_bindings.values()),
            'new nonoverlapping report output required')
    result = summarize(published)
    result['source_authorities'] = {
        'evaluation_COMMIT': published['commit'],
        'evaluation_contract': published['snapshot']['contract.json'],
        'evaluation_authorization': published['authorization'],
        'package_COMMIT': published['contract'].get('package_commit'),
        'package_products': published['package_proof']['products'],
        'schedule_document': published['schedule_entry'],
        'schedule_sha256': published['contract'].get('schedule_sha256'),
        'frozen_evaluator_code': published['external_bindings'],
        'source_graph_policy': 'immutable BC package/schedule authorities are rehashed; reporter performs no feature/audio loading or model rerun',
        'report_bindings': report_bindings,
    }
    recheck(published)
    for entry in report_bindings.values():
        require(binding(entry['path']) == entry,
                'report specification/code changed')
    output.mkdir()
    write_new(output / 'summary.json', result)
    write_new(output / 'per_fold_auc.json', published['fold_rows'])
    write_new(output / 'omitted_cells.json', published['omissions'])
    write_new(output / 'coverage.json', list(published['cap_records'].values()))
    for name, text in markdown_products(result, published).items():
        write_new(output / name, text.encode())
    for entry in report_bindings.values():
        require(binding(entry['path']) == entry,
                'report specification/code changed before COMMIT')
    recheck(published)
    products = {path.name: binding(path) for path in sorted(output.iterdir())}
    require(all(binding(entry['path']) == entry for entry in products.values()),
            'report output changed before COMMIT')
    write_new(output / 'COMMIT.json', {
        'version': VERSION,
        'status': 'committed_reporting_only_no_selection',
        'evaluation_COMMIT': published['commit'],
        'products': products,
        'all_evaluation_products_end_rehashed': True,
        **BC_SCOPE, **SCOPE,
    })
    return {'status': 'committed_reporting_only_no_selection',
            'output': str(output),
            'primary_cells': len(result['primary_protocol_cells']),
            'diagnostic_cells': len(result['diagnostic_protocol_cells']),
            'held_source_cells': len(result['held_source_cells']),
            'commit': binding(output / 'COMMIT.json'),
            'feature_count': len(FEATURES), 'combination_count': len(COMBINATIONS),
            **BC_SCOPE, **SCOPE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--evaluation-commit-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    refs = {'reporter': binding(Path(__file__).resolve()),
            'tests': binding(HERE / ('test_' + VERSION + '.py')),
            'specification': binding(SPEC, SPEC_SHA)}
    with read_lock(args.evaluation):
        published = inspect_evaluation(args.evaluation, args.evaluation_commit_sha256)
        result = publish(published, args.output, refs)
    print(canonical(result).decode().strip())


if __name__ == '__main__':
    main()
