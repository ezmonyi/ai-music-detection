"""Schedule-driven Native30 eight-family evaluation adapter.

This adapter extends the pinned seven-family v3 evaluator with the one-column
BC candidate.  It consumes only a deeply verified BC input package and the
metadata-only eight-family schedule.  The numerical estimator, train-only
transforms, weights, prediction transform, metric functions and receipt
guards are the unchanged v3 core.  A separate caller-pinned fit freeze is
required before any fold fit is attempted.

The v3 core is loaded under a private module name.  Its globals are rebound
only for the duration of a delegated call and restored in ``finally``; the
imported ``evaluate_native30_v3`` module and its seven-family globals are
never edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd


VERSION = 'evaluate_native30_bc_v1'
STAGE = 'native30_eight_family_schedule_only_development_evaluation_bc_v1'
FREEZE_VERSION = 'native30-evaluation-parent-freeze-bc-v1'

# Immutable source pins.  The two old schedule pins remain exposed under the
# v3 names because the seven-family projection must be byte-identical.
V3_SHA = 'c2544fe4ce98d88b906f502a576f788da46a4e28117ea7f3ce41802495fe7d1c'
BASE_SHA = 'd2ed30d9833fbe122f023de1223e44a40c1b4f95f99f63f0ba27647c87cc7232'
PLANNER_SHA = 'ff6965b3f9c834ee9d39bbf0f994595f44fa2e261d9c7c6b60a153cb9f72ce24'
IO_SHA = '165f28d4e553ee679772aa1ac47b31f9a485b94d3a8ac409f25595577abe395a'
SCHEDULE_SHA = '4e12300a7fe43b4f3024c1cd9e675c25b0ade3af25dded0d688b8d8f2fcab7a6'
SCHEDULE_COMMIT_SHA = '5d0e329d9ef0a794e32acbf5bee06a4a09a58b453e1741ba6232fbfb32347233'

# Final caller-pinned BC package adapter and the separately versioned
# metadata schedule adapter.  These are source-code pins, not measurement or
# fit authority.
PREPARER_SHA = '617de4ed687868a221c68664930d5757174d0dadec32df4663ee39e5621a413f'
PREPARER_NAME = 'prepare_native30_evaluation_inputs_bc_v1'
PREPARER_TEST_SHA = 'cdbb9b567541fef898498dd111783a942be5ac710a3c175cebbdf4bb82fd0039'
BC_SCHEDULE_CODE_SHA = 'c432815b11e6dc315191f68f9722764b06be8e477b2d5db75308067ffe6764e5'
BC_SCHEDULE_TEST_SHA = 'be470106aabb8fd4864744b636d4057a9b1822bccf0a4261396f4c8be1bccf5d'

EXPECTED = {
    'total': 3830,
    'labels': {0: 1664, 1: 2166},
    'intended': 107100,
    'primary': 73950,
    'diagnostic': 29580,
}

# Keep the v3 scope byte-for-byte.  BC package scope is independently checked
# by the input adapter; it is not silently folded into old model receipts.
SCOPE = {
    'winner_selection': False,
    'threshold_tuning': False,
    'full_cohort_refit': False,
    'historical_locked_or_pilot_scoring': False,
    'M_predictor': False,
    'audio_or_neural_inference_performed': False,
    'source_ranking': False,
}
BC_SCOPE = {
    'BC_classifier_admitted': False,
    'BC_diagnostics_are_predictors': False,
}

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, pin, name):
    """Load a canonical regular file only after checking its exact SHA."""
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), 'canonical code file required')
    require(isinstance(pin, str) and hashlib.sha256(path.read_bytes()).hexdigest() == pin,
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


# Load the immutable v3 numerical/control implementation privately.  This
# deliberately does not import it as ``evaluate_native30_v3`` and does not
# overwrite that module if a caller already imported it.
CORE = load_module(HERE / 'evaluate_native30_v3.py', V3_SHA,
                   '_native30_bc_evaluator_v3_core')
io = CORE.io
BASE = CORE.BASE
_CORE_VALIDATE_MODEL = CORE.validate_model
_OLD_PLANNER = CORE.planner
_OLD_FAMILIES = copy.deepcopy(CORE.FAMILIES)
_OLD_FEATURES = list(CORE.FEATURES)
_OLD_COMBINATIONS = list(_OLD_PLANNER.COMBINATIONS)

FAMILIES = {**copy.deepcopy(_OLD_FAMILIES),
            'BC': ['BC_b2_500_750_1250hz_center8s_median']}
BC_FEATURE = FAMILIES['BC'][0]
FEATURES = sum(FAMILIES.values(), [])
META = list(CORE.META)
OLD_FAMILY_CONFIG = copy.deepcopy(_OLD_FAMILIES)
_ALL_FAMILY_KEYS = tuple((*_OLD_PLANNER.FAMILIES, 'BC'))


class _PlannerFacade:
    """v3 metadata functions plus the eight-family catalogue constants.

    ``make_schedule`` remains the pinned seven-family function.  The BC
    schedule extension is derived by the separately pinned schedule module.
    Keeping this facade separate prevents accidental mutation of the original
    planner's ``FAMILIES``/``COMBINATIONS`` globals.
    """

    FAMILIES = _ALL_FAMILY_KEYS
    COMBINATIONS = ['+'.join(c) for n in range(1, 9)
                    for c in itertools.combinations(_ALL_FAMILY_KEYS, n)]
    CAPS = _OLD_PLANNER.CAPS
    PRIMARY_MODE = _OLD_PLANNER.PRIMARY_MODE
    DIAGNOSTIC_MODES = _OLD_PLANNER.DIAGNOSTIC_MODES
    PLAN_SHA = _OLD_PLANNER.PLAN_SHA
    SCREEN_SHA = _OLD_PLANNER.SCREEN_SHA
    IDENTITY = _OLD_PLANNER.IDENTITY
    SOURCES = _OLD_PLANNER.SOURCES

    def __getattr__(self, name):
        return getattr(_OLD_PLANNER, name)


planner = _PlannerFacade()
# Public convenience alias retained for callers that inspect the expanded
# catalogue directly.  Validation below always reads the facade so that the
# pinned eight-family catalogue remains the single source of truth.
COMBINATIONS = list(planner.COMBINATIONS)
schedule_adapter = load_module(
    HERE / 'plan_native30_evaluation_schedule_bc_v1.py', BC_SCHEDULE_CODE_SHA,
    'plan_native30_evaluation_schedule_bc_v1')


@contextmanager
def _bound_core():
    """Temporarily give delegated v3 functions this adapter's constants.

    The private core is the only module mutated here.  Every value is restored
    even if a fit, validation, or file operation raises.
    """
    names = {
        'VERSION': VERSION,
        'STAGE': STAGE,
        'FREEZE_VERSION': FREEZE_VERSION,
        'PREPARER_SHA': PREPARER_SHA,
        'PREPARER_NAME': PREPARER_NAME,
        'BASE_SHA': BASE_SHA,
        'PLANNER_SHA': PLANNER_SHA,
        'IO_SHA': IO_SHA,
        'SCHEDULE_SHA': SCHEDULE_SHA,
        'SCHEDULE_COMMIT_SHA': SCHEDULE_COMMIT_SHA,
        'EXPECTED': EXPECTED,
        'SCOPE': SCOPE,
        'FAMILIES': FAMILIES,
        'FEATURES': FEATURES,
        'META': META,
        'HERE': HERE,
        'planner': planner,
        'io': io,
        'BASE': BASE,
    }
    old = {name: getattr(CORE, name) for name in names}
    for name, value in names.items():
        setattr(CORE, name, value)
    # run_context calls this global.  It must use the adapter's package and
    # schedule guard, not v3's seven-family package guard.
    old_input_guard = CORE.input_guard
    old_validate_model = CORE.validate_model
    CORE.input_guard = input_guard
    CORE.validate_model = _validate_model_bound
    try:
        yield
    finally:
        CORE.validate_model = old_validate_model
        CORE.input_guard = old_input_guard
        for name, value in old.items():
            setattr(CORE, name, value)


def clean(value):
    with _bound_core():
        return CORE.clean(value)


def table_payload(table):
    with _bound_core():
        return CORE.table_payload(table)


def runtime_snapshot():
    with _bound_core():
        return CORE.runtime_snapshot()


def make_table(metadata, features, cohort_rows, plan_rows):
    require(len(FEATURES) == 55 and len(set(FEATURES)) == 55,
            'exact55 feature catalogue required')
    with _bound_core():
        return CORE.make_table(metadata, features, cohort_rows, plan_rows)


def _original_schedule(rows, screen):
    """Replay the original seven-family schedule without reading files."""
    return planner.make_schedule(rows, screen)


_SCHEDULE_DYNAMIC_KEYS = {
    'input_bindings', 'prepared_contract_sha256', 'metadata_read_scope',
    'runtime', 'historical_original_schedule_provenance',
}


def _expected_extended_schedule(rows, screen):
    old = _original_schedule(rows, screen)
    # The planner adapter consumes a completed schedule plus a pure replay.
    # For synthetic rows the completed schedule has no build metadata; for
    # production schedule metadata is ignored by derive_schedule's BUILD_KEYS.
    return schedule_adapter.derive_schedule(old, old, OLD_FAMILY_CONFIG)


def _validate_schedule_accounting(schedule, counts):
    """Check both cell-derived accounting and the schedule's declared counts."""
    require(set(counts) == {'primary', 'diagnostic'}, 'schedule accounting roles')
    declared = schedule.get('accounting')
    require(isinstance(declared, dict), 'schedule accounting missing')
    expected_declared = {
        'intended_combination_cap_cells': sum(v['intended'] for v in counts.values()),
        'eligible_combination_cap_cells': sum(v['eligible'] for v in counts.values()),
        'omitted_combination_cap_cells': sum(v['omitted'] for v in counts.values()),
        'primary': {
            'intended_cells': counts['primary']['intended'],
            'eligible_cells': counts['primary']['eligible'],
            'omitted_cells': counts['primary']['omitted'],
        },
        'diagnostic': {
            'intended_cells': counts['diagnostic']['intended'],
            'eligible_cells': counts['diagnostic']['eligible'],
            'omitted_cells': counts['diagnostic']['omitted'],
        },
    }
    for key, value in expected_declared.items():
        require(declared.get(key) == value, 'schedule accounting replay mismatch: ' + key)


def schedule_counts(schedule):
    """Recompute primary/diagnostic cell counts from the actual cell list."""
    folds = {row['fold_uid']: row for row in schedule.get('fold_cells', [])}
    caps = {row['schedule_uid']: row for row in schedule.get('cap_schedules', [])}
    require(len(folds) == len(schedule.get('fold_cells', []))
            and len(caps) == len(schedule.get('cap_schedules', [])),
            'duplicate schedule fold/cap UID')
    require(isinstance(schedule.get('combination_schedule_cells'), list),
            'schedule cell list required')
    require(all(cell.get('analysis_role') in ('primary', 'diagnostic')
                for cell in schedule['combination_schedule_cells']),
            'unknown schedule analysis role')
    result = {}
    for role in ('primary', 'diagnostic'):
        cells = [cell for cell in schedule['combination_schedule_cells']
                 if cell.get('analysis_role') == role]
        valid = []
        for cell in cells:
            require(cell.get('status') in ('eligible_metadata_cell', 'omitted'),
                    'unknown schedule cell status')
            require(cell.get('combination') in planner.COMBINATIONS,
                    'unknown eight-family combination')
            require(cell.get('feature_mode') in
                    (planner.PRIMARY_MODE, *planner.DIAGNOSTIC_MODES),
                    'unknown feature mode')
            if role == 'diagnostic':
                require(cell['feature_mode'] != planner.PRIMARY_MODE,
                        'primary mode marked diagnostic')
            else:
                require(cell['feature_mode'] == planner.PRIMARY_MODE,
                        'diagnostic mode marked primary')
            require(cell['schedule_uid'] in caps, 'cell cap reference missing')
            cap = caps[cell['schedule_uid']]
            require(cap['fold_uid'] in folds, 'cap fold reference missing')
            require(cell['status'] == cap['status'], 'cell/cap status mismatch')
            if cell['status'] == 'eligible_metadata_cell':
                valid.append(cell)
        result[role] = {
            'intended': len(cells),
            'eligible': len(valid),
            'omitted': len(cells) - len(valid),
            'prediction_rows': sum(
                folds[caps[cell['schedule_uid']]['fold_uid']]['test']['rows']
                for cell in valid),
        }
    _validate_schedule_accounting(schedule, result)
    return result


def replay_schedule(schedule, rows, screen):
    """Validate a committed BC schedule against the pinned seven-family replay."""
    # Recompute declarations from the actual cell/cap/fold graph before
    # comparing the full document.  This makes a forged accounting field fail
    # as an accounting replay error rather than being hidden behind a generic
    # document-difference message.
    schedule_counts(schedule)
    expected = _expected_extended_schedule(rows, screen)
    require(set(schedule) <= set(expected) | _SCHEDULE_DYNAMIC_KEYS,
            'unexpected schedule fields')
    for key, value in expected.items():
        require(schedule.get(key) == value,
                'committed schedule differs from pinned metadata replay: ' + key)
    require(schedule.get('families') == list(FAMILIES)
            and schedule.get('combinations') == planner.COMBINATIONS
            and schedule.get('family_config') == FAMILIES
            and schedule.get('feature_names') == FEATURES,
            'exact eight-family catalogue replay')
    require(schedule.get('original_seven_family_projection', {}).get('combinations')
            == _OLD_COMBINATIONS,
            'old127 combination projection changed')
    counts = schedule_counts(schedule)
    return schedule


def _add_binding(bindings, entry, *, name=None):
    require(isinstance(entry, dict) and set(entry) == {'path', 'bytes', 'sha256'},
            'malformed source binding')
    observed = io.file_binding(entry['path'])
    require(observed == entry, 'bound source changed: ' + entry['path'])
    bindings[name or entry['path']] = entry


def _pinned_binding(path, pin):
    entry = io.file_binding(path)
    require(entry['sha256'] == pin,
            'unapproved/changed source pin: ' + str(path))
    return entry


def _read_binding(entry):
    _add_binding({}, entry)
    return io.read_json(entry['path'])


def _package_contract_graph(assembled, bundle, package):
    """Recover the immutable old schedule/plan/screen graph from BC package."""
    require(isinstance(assembled.get('request'), dict), 'BC package request missing')
    request = assembled['request']
    require(set(request) == {'parent_commit', 'bc_freeze', 'bc_commit',
                             'schedule_draft', 'schedule_commit'},
            'BC package authority request schema')
    require(assembled.get('request_sha256') == io.value_hash(request),
            'BC package request hash changed')
    for entry in request.values():
        _add_binding({}, entry)
    parent_products = assembled.get('parent_products')
    require(isinstance(parent_products, dict)
            and isinstance(parent_products.get('contract.json'), dict),
            'BC parent product graph missing')
    parent_entry = parent_products['contract.json']
    parent_contract = _read_binding(parent_entry)
    require(parent_contract.get('version') == 'prepare_native30_evaluation_inputs_v3'
            and parent_contract.get('status') == 'assembled_not_evaluated_not_authorized'
            and parent_contract.get('family_config') == _OLD_FAMILIES
            and parent_contract.get('feature_names') == _OLD_FEATURES,
            'pinned seven-family parent package required')
    parent_bindings = parent_contract.get('bindings')
    require(isinstance(parent_bindings, dict), 'parent binding map missing')
    for entry in parent_bindings.values():
        _add_binding({}, entry)
    refs = parent_bindings
    for key in ('cohort_contract', 'origin_plan', 'screen', 'schedule_draft',
                'schedule_commit'):
        require(key in refs, 'parent schedule graph missing: ' + key)
    require(parent_contract.get('cohort_contract') == refs['cohort_contract'],
            'parent cohort binding changed')
    require(refs['schedule_draft']['sha256'] == SCHEDULE_SHA
            and refs['schedule_commit']['sha256'] == SCHEDULE_COMMIT_SHA
            and refs['origin_plan']['sha256'] == planner.PLAN_SHA
            and refs['screen']['sha256'] == planner.SCREEN_SHA,
            'fixed original schedule/plan/screen pins')
    original_schedule = io.read_json(refs['schedule_draft']['path'])
    original_commit = io.read_json(refs['schedule_commit']['path'])
    require(original_commit == {
        'status': 'committed_metadata_draft_not_evaluation_authorization',
        'version': planner.VERSION,
        'products': {'schedule_draft.json': refs['schedule_draft']},
        'draft_sha256': SCHEDULE_SHA,
        'classifier_fits': 0,
        'fitting_authorized': False,
    }, 'original schedule COMMIT/status changed')
    schedule_entry = request['schedule_draft']
    schedule_commit_entry = request['schedule_commit']
    require(Path(schedule_entry['path']).name == 'schedule_draft.json'
            and Path(schedule_commit_entry['path']).name == 'COMMIT.json'
            and Path(schedule_entry['path']).parent == Path(schedule_commit_entry['path']).parent,
            'BC schedule authority root')
    schedule = io.read_json(schedule_entry['path'])
    schedule_commit = io.read_json(schedule_commit_entry['path'])
    require(schedule_commit == {
                'version': schedule_adapter.VERSION,
                'status': 'committed_BC_metadata_schedule_draft_not_measurement_or_evaluation_authorization',
                'products': {'schedule_draft.json': schedule_entry},
                'draft_sha256': schedule_entry['sha256'],
                'original_schedule': refs['schedule_draft'],
                'original_commit': refs['schedule_commit'],
                'classifier_fits': 0, 'fitting_authorized': False,
                'measurement_authorized': False,
            },
            'BC schedule COMMIT/status changed')
    return (request, parent_contract, refs, original_schedule, schedule,
            schedule_entry, schedule_commit_entry)


def _bind_contract_files(assembled, bundle, package, parent_contract, refs,
                         schedule_entry, schedule_commit_entry):
    bindings = {}
    for entry in assembled.get('code_bindings', {}).values():
        _add_binding(bindings, entry)
    for entry in assembled.get('parent_products', {}).values():
        _add_binding(bindings, entry)
    for entry in assembled.get('request', {}).values():
        _add_binding(bindings, entry)
    for entry in refs.values():
        _add_binding(bindings, entry)
    _add_binding(bindings, schedule_entry)
    _add_binding(bindings, schedule_commit_entry)
    commit_entry = io.file_binding(package / 'COMMIT.json')
    require(commit_entry == bundle['commit'], 'BC package COMMIT binding mismatch')
    for path in package.iterdir():
        require(path.is_file() and not path.is_symlink(), 'BC package product must be regular')
        _add_binding(bindings, io.file_binding(path))
    # These are the evaluator's explicit, caller-visible source pins.  The
    # self binding has no expected hash because the running file is the source
    # being bound; recheck still catches any mutation after construction.
    for path, pin in (
        (HERE / 'evaluate_native30_v3.py', V3_SHA),
        (HERE / 'frozen_evaluate_expanded_20260905.py', BASE_SHA),
        (HERE / 'plan_native30_evaluation_schedule_v1.py', PLANNER_SHA),
        (HERE / 'materialize_native30_new1695_v1.py', IO_SHA),
        (HERE / (PREPARER_NAME + '.py'), PREPARER_SHA),
        (HERE / 'plan_native30_evaluation_schedule_bc_v1.py', BC_SCHEDULE_CODE_SHA),
        (HERE / ('test_' + PREPARER_NAME + '.py'), PREPARER_TEST_SHA),
        (HERE / 'test_plan_native30_evaluation_schedule_bc_v1.py', BC_SCHEDULE_TEST_SHA),
    ):
        _add_binding(bindings, _pinned_binding(path, pin), name=str(path))
    _add_binding(bindings, io.file_binding(HERE / (VERSION + '.py')), name=str(HERE / (VERSION + '.py')))
    _add_binding(bindings, io.file_binding(HERE / ('test_' + VERSION + '.py')),
                 name=str(HERE / ('test_' + VERSION + '.py')))
    return bindings, commit_entry


def check_bindings(contract):
    """Rehash every bound file and deeply replay the committed BC package."""
    io.recheck(contract.get('bindings', {}))
    if contract.get('package_commit'):
        require(PREPARER_SHA is not None, 'assembler approval pin not installed')
        assembler = load_module(HERE / (PREPARER_NAME + '.py'), PREPARER_SHA,
                                PREPARER_NAME)
        entry = contract['package_commit']
        package = planner.path_checked(Path(entry['path']).parent)
        proof = assembler.verify_package(package, entry['sha256'])
        require(proof['commit'] == entry, 'deep prepared-package replay changed')
        require(proof['contract']['feature_names'] == FEATURES
                and proof['contract']['family_config'] == FAMILIES
                and proof['contract']['expected_count'] == EXPECTED['total'],
                'BC package catalogue/count changed')
    if contract.get('schedule_document'):
        entry = contract['schedule_document']
        require(io.file_binding(entry['path']) == entry, 'schedule document changed')
        schedule = io.read_json(entry['path'])
        require(io.value_hash(schedule) == contract.get('schedule_sha256'),
                'schedule document hash changed')
        require(schedule_counts(schedule) == contract.get('accounting'),
                'schedule accounting changed')
    if contract.get('runtime'):
        require(runtime_snapshot() == contract['runtime'], 'frozen numerical runtime changed')


def _actual_expected_counts(counts, row_count):
    """Assert production accounting only after recomputing it from cells."""
    if row_count == EXPECTED['total']:
        require(sum(v['intended'] for v in counts.values()) == EXPECTED['intended'],
                'actual intended cell accounting mismatch')
        require(counts['primary']['eligible'] == EXPECTED['primary']
                and counts['diagnostic']['eligible'] == EXPECTED['diagnostic'],
                'actual eligible cell accounting mismatch')


def build_context(package, package_commit_sha, output):
    """Deep-verify a BC package and schedule, but perform no fit/prediction."""
    require(PREPARER_SHA is not None, 'assembler approval pin not installed')
    assembler = load_module(HERE / (PREPARER_NAME + '.py'), PREPARER_SHA,
                            PREPARER_NAME)
    package = planner.path_checked(package)
    bundle = assembler.verify_package(package, package_commit_sha)
    assembled = bundle['contract']
    require(assembled.get('status') == 'assembled_not_evaluated_not_authorized'
            and assembled.get('expected_count') == EXPECTED['total']
            and assembled.get('family_config') == FAMILIES
            and assembled.get('feature_names') == FEATURES
            and assembled.get('feature_count') == 55,
            'native30 BC assembled package contract')
    (request, parent_contract, refs, original_schedule, schedule,
     schedule_entry, schedule_commit_entry) = _package_contract_graph(
         assembled, bundle, package)
    cohort = io.read_json(refs['cohort_contract']['path'])
    plan = io.read_json(refs['origin_plan']['path'])
    screen = io.read_json(refs['screen']['path'])
    rows = planner.validate_prepared_contract(cohort, plan, screen)
    replay_original = planner.make_schedule(rows, screen)
    # Verify the original schedule's build metadata against the pure replay.
    pure_original = {k: v for k, v in original_schedule.items()
                     if k not in schedule_adapter.BUILD_KEYS}
    require(pure_original == replay_original,
            'original seven-family schedule replay mismatch')
    replay_schedule(schedule, rows, screen)
    require(schedule['prepared_contract_sha256'] == refs['cohort_contract']['sha256']
            and schedule['input_population']['ids'] == [r['id'] for r in rows]
            and schedule['eligible_population']['ids'] == [r['id'] for r in rows]
            and schedule['protected_excluded_population']['rows'] == 0,
            'BC schedule/cohort population join')
    table = make_table(bundle['metadata'], bundle['features'], rows, plan['rows'])
    require(len(table) == EXPECTED['total']
            and dict(Counter(table['__source'])) == planner.SOURCES
            and dict(Counter(table['__label'])) == EXPECTED['labels'],
            'actual native30 exact roster')
    counts = schedule_counts(schedule)
    _actual_expected_counts(counts, len(table))
    require(assembled.get('schedule_accounting') == schedule['accounting']
            and assembled.get('schedule_sha256') == schedule['schedule_sha256'],
            'BC package/schedule accounting join')
    bindings, commit_entry = _bind_contract_files(
        assembled, bundle, package, parent_contract, refs, schedule_entry,
        schedule_commit_entry)
    output = planner.path_checked(output)
    committed_roots = {Path(path).parent for path in bindings
                       if Path(path).name in {'COMMIT.json', 'contract.json'}}
    bound_roots = {Path(entry['path']).parent for entry in bindings.values()}
    require(output.parent.is_dir()
            and not any(Path(entry['path']).is_relative_to(output)
                        for entry in bindings.values())
            and not any(output.is_relative_to(root) or root.is_relative_to(output)
                        for root in bound_roots)
            and not output.is_relative_to(package)
            and not any(output.is_relative_to(root) for root in committed_roots),
            'output overlaps evaluation provenance')
    contract = {
        'version': VERSION,
        'stage': STAGE,
        'package_commit': commit_entry,
        'bindings': bindings,
        'output_root': str(output),
        'runtime': runtime_snapshot(),
        'expected_count': len(table),
        'families': FAMILIES,
        'feature_names': FEATURES,
        'table_sha256': io.value_hash(table_payload(table)),
        'accounting': counts,
        'schedule_document': schedule_entry,
        'schedule_sha256': io.value_hash(schedule),
        'schedule_protocol': schedule['protocol'],
        'schedule_accounting_recomputed_from_cells': True,
        'ridge': 10.0,
        'threshold': 0.5,
        'prediction_link': 'raw_identity_unclipped',
        'weighting': 'equal classes/sources within class/groups within source/items within group; sum training n',
        'preprocessing': 'unchanged pinned BASE.fit_model; train-only median/weighted scale; no availability filtering',
        'coefficient_validation': 'no refit; infinity residual <= 128*float64_eps*design_columns*max(1, norm(A,inf)*norm(beta,inf), norm(b,inf))',
        'validation_scope': 'in-process transform/equation/prediction replay; not an independent full-publication numerical audit',
        'comparison_scope': 'new BC cohort/duration/purging/caps; matched X+BC versus X; no causal historical outcome subtraction',
        'original_seven_family_projection': {
            'combinations': _OLD_COMBINATIONS,
            'feature_names': _OLD_FEATURES,
            'combination_cells_sha256': io.value_hash([
                c for c in schedule['combination_schedule_cells']
                if 'BC' not in c['combination'].split('+')]),
            'accounting': copy.deepcopy(
                schedule['original_seven_family_projection']['accounting']),
            'experiment_cells_sha256':
                schedule['original_seven_family_projection']['experiment_cells_sha256'],
        },
        'bc_matched_comparisons': [
            {'reference_combination': combination,
             'bc_combination': combination + '+BC'}
            for combination in _OLD_COMBINATIONS
        ],
        **BC_SCOPE,
        **SCOPE,
    }
    check_bindings(contract)
    return contract, table, schedule


def authorize(contract, path, expected_sha):
    """Verify a separate parent fit/scoring freeze for this exact contract."""
    require(path is not None and expected_sha is not None,
            'separate parent fitting freeze required')
    path = planner.path_checked(path)
    binding = planner.binding(path, expected_sha)
    frozen = io.read_json(path)
    require(frozen.get('version') == FREEZE_VERSION
            and frozen.get('status') == 'parent_frozen_for_native30_fold_only_fitting'
            and frozen.get('fitting_authorized') is True
            and frozen.get('scoring_authorized') is True
            and frozen.get('stage') == STAGE
            and frozen.get('contract') == contract
            and frozen.get('contract_sha256') == io.value_hash(contract)
            and frozen.get('independent_review', {}).get('approved') is True
            and frozen.get('independent_review', {}).get('reviewer') == 'root'
            and all(frozen.get(key) == value for key, value in SCOPE.items())
            and frozen.get('winner_selection') is False,
            'parent fit freeze scope/contract mismatch')
    return {'status': 'parent_frozen_verified',
            'contract_sha256': io.value_hash(contract), 'binding': binding}


def resolve_task(cell, schedule):
    with _bound_core():
        return CORE.resolve_task(cell, schedule)


def task_uid(task, contract_sha):
    with _bound_core():
        return CORE.task_uid(task, contract_sha)


def task_tables(table, task):
    with _bound_core():
        return CORE.task_tables(table, task)


def columns_for(task):
    require(task.get('combination') in planner.COMBINATIONS
            and task.get('feature_mode') in
            (planner.PRIMARY_MODE, *planner.DIAGNOSTIC_MODES),
            'unknown eight-family combination/feature mode')
    require(task['feature_mode'] == planner.PRIMARY_MODE
            or task.get('quantity') == 'all', 'diagnostics only at all-cap')
    return sum((FAMILIES[key] for key in task['combination'].split('+')), [])


def validate_model(model, train, task):
    with _bound_core():
        return _validate_model_bound(model, train, task)


def _fit_transform_metadata(train, columns, feature_mode):
    """Replay BASE.fit_model's train-only metadata without solving."""
    raw = train[columns].to_numpy(float)
    missing = ~np.isfinite(raw)
    medians = np.asarray([
        float(np.median(column[np.isfinite(column)]))
        if np.isfinite(column).any() else 0.0
        for column in raw.T
    ])
    imputed = np.where(missing, medians, raw)
    if feature_mode == planner.PRIMARY_MODE:
        values = np.concatenate((imputed, missing.astype(float)), axis=1)
    elif feature_mode == 'median_only':
        values = imputed
    elif feature_mode == 'missingness_only':
        values = missing.astype(float)
    else:
        raise ValueError('unknown feature mode')
    weights = BASE.sample_weights(train)
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return medians, mean, scale


def _validate_model_bound(model, train, task):
    """Run the v3 model audit with an exact diagnostic transform bridge."""
    try:
        return _CORE_VALIDATE_MODEL(model, train, task)
    except ValueError as error:
        # ``fit_model`` computes a diagnostic mode on one half of the
        # values+missingness matrix, while ``prepare_training`` computes both
        # halves in one weighted reduction.  A one-column BC subset can
        # therefore differ by one float64 ulp despite identical train-only
        # inputs.  Replay BASE.fit_model's selected transform exactly, then
        # replay every original v3 guard on an exact prepared copy.  No fit or
        # solver is introduced here.
        if str(error) != 'train-only transform replay mismatch':
            raise
        columns = columns_for(task)
        expected_medians, expected_mean, expected_scale = (
            _fit_transform_metadata(train, columns, task['feature_mode']))
        require(np.array_equal(np.asarray(model['medians'], dtype=float),
                               expected_medians)
                and np.array_equal(np.asarray(model['mean'], dtype=float),
                                   expected_mean)
                and np.array_equal(np.asarray(model['scale'], dtype=float),
                                   expected_scale),
                'train-only transform replay mismatch')

        # CORE.validate_model's prepared matrix is mathematically identical,
        # but its combined weighted reduction can differ by one ulp for a
        # one-column diagnostic subset.  Once the strict BASE.fit_model
        # metadata replay above has passed, use the prepared values solely to
        # satisfy the delegated core's exact representation check.
        prepared = BASE.prepare_training(train, columns)
        count = len(columns)
        indices = (np.arange(2 * count)
                   if task['feature_mode'] == planner.PRIMARY_MODE else
                   np.arange(count)
                   if task['feature_mode'] == 'median_only' else
                   np.arange(count, 2 * count))
        exact = copy.deepcopy(model)
        exact['medians'] = prepared['medians'].tolist()
        exact['mean'] = prepared['mean'][indices].tolist()
        exact['scale'] = prepared['scale'][indices].tolist()
        return _CORE_VALIDATE_MODEL(exact, train, task)


def prediction_evidence(test, model):
    with _bound_core():
        return CORE.prediction_evidence(test, model)


def metric_evidence(predictions, task):
    with _bound_core():
        return CORE.metric_evidence(predictions, task)


def model_receipt(contract, table, task):
    with _bound_core():
        return CORE.model_receipt(contract, table, task)


def verify_model_receipt(path, contract, table, task):
    with _bound_core():
        return CORE.verify_model_receipt(path, contract, table, task)


def input_guard(contract, table, schedule, authorization):
    check_bindings(contract)
    require(io.value_hash(table_payload(table)) == contract['table_sha256']
            and io.value_hash(schedule) == contract['schedule_sha256'],
            'in-memory table/schedule changed')
    require(authorization.get('status') == 'parent_frozen_verified'
            and authorization.get('contract_sha256') == io.value_hash(contract),
            'verified parent fitting authorization required')
    require(planner.binding(authorization['binding']['path']) ==
            authorization['binding'], 'parent freeze changed')
    require(authorize(contract, authorization['binding']['path'],
                      authorization['binding']['sha256']) == authorization,
            'parent authorization replay changed')


def inventory(output, uids):
    with _bound_core():
        return CORE.inventory(output, uids)


def products(output):
    with _bound_core():
        return CORE.products(output)


def commit_record(contract_sha, bound, count):
    with _bound_core():
        return CORE.commit_record(contract_sha, bound, count)


def run_context(contract, table, schedule, authorization):
    """Run/resume fold-only models after a verified, separate fit freeze."""
    with _bound_core():
        return CORE.run_context(contract, table, schedule, authorization)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--package-commit-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    parser.add_argument('--frozen')
    parser.add_argument('--frozen-sha256')
    args = parser.parse_args()
    contract, table, schedule = build_context(
        args.package, args.package_commit_sha256, args.output)
    if args.mode == 'preflight':
        result = {'status': 'preflight_no_fitting',
                  'contract': contract,
                  'contract_sha256': io.value_hash(contract),
                  'classifier_fits': 0,
                  'predictions_computed': 0,
                  **BC_SCOPE, **SCOPE}
    else:
        authorization = authorize(contract, args.frozen, args.frozen_sha256)
        result = run_context(contract, table, schedule, authorization)
    print(io.canonical(result).decode().strip())
    return 1 if result['status'] == 'partial_no_COMMIT' else 0


if __name__ == '__main__':
    raise SystemExit(main())
