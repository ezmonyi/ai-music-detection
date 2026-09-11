#!/usr/bin/env python3
"""Isolated native-stereo v6 orchestration; draft never fits or predicts."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import itertools
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

import evaluate_new_phenomena_v5 as V5
import prepare_evaluation_inputs_v6 as PREP6

V2, BASE, PREP5 = V5.V2, V5.BASE, V5.PREP
require, canonical, digest, sha = V5.require, V5.canonical, V5.digest, V5.sha
read_json, write_json, read_csv = V5.read_json, V5.write_json, V5.read_csv
ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / 'EXPLORATORY_V6_EVALUATOR_PROTOCOL_EN.md'
STAGE = 'exploratory_v6_native_stereo_development_cv_only'
SEED, FOLDS = 20260907, 5
QUANTITIES = (25, 50, 100, 200, 'all')
SC_COLUMNS = [f'SC_{lo}_{hi}hz_{metric}_median'
              for lo, hi in ((80, 500), (500, 2000), (2000, 6000))
              for metric in ('abs_iid_db', 'side_energy_fraction')]
FAMILIES = {**{k: list(v) for k, v in PREP5.COLUMNS.items()}, 'SC': SC_COLUMNS}
DESCRIPTORS = sum(FAMILIES.values(), [])
COMBINATIONS = ['+'.join(c) for n in range(1, len(FAMILIES) + 1)
                for c in itertools.combinations(FAMILIES, n)]
MODES, POLICIES = V5.MODES, V5.POLICIES
COUNTS = {k: v for k, v in PREP5.COUNTS.items() if k != 'human_urmp'}
EXPECTED_REAL = dict(valid_folds=38, primary_fits=48450, primary_prediction_rows=14017350,
                     diagnostic_fits=19380, diagnostic_prediction_rows=5606940)
PRODUCTS = {'metadata_60s.csv', 'features_60s.csv', 'family_config.json', 'origin_ledger.csv'}


def family_config():
    return dict(schema_version=6, families=FAMILIES)


def bound_file(path, expected, bindings):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            'Canonical regular bound file required: ' + str(path))
    require(sha(path) == expected, 'Bound file SHA mismatch: ' + str(path))
    require(str(path) not in bindings or bindings[str(path)] == expected, 'Conflicting binding')
    bindings[str(path)] = expected
    return path


def validate_package(package, synthetic=False):
    """Strict publication and externally reproducible lineage; never grants fitting."""
    package = Path(package)
    PREP6.checked_runtime()
    require(package.is_absolute() and package.resolve() == package and not package.is_symlink(),
            'Canonical package required')
    marker = read_json(package / 'COMMIT.json')
    names = PRODUCTS | {'preparation_audit.json'}
    require(marker.get('status') == 'committed' and set(marker.get('files', {})) == names
            and {p.name for p in package.iterdir()} == names | {'COMMIT.json'}, 'Incomplete/orphan package')
    bindings = {}
    for name, record in marker['files'].items():
        path = bound_file(package / name, record['sha256'], bindings)
        require(path.stat().st_size == record['bytes'], 'Package size changed')
    bindings[str(package / 'COMMIT.json')] = sha(package / 'COMMIT.json')
    proof = read_json(package / 'preparation_audit.json')
    require(proof.get('schema_version') == 6 and proof.get('status') == 'prepared_not_authorized_for_fitting'
            and proof.get('synthetic_test_only') is synthetic and proof.get('fitting_authorized') is False
            and proof.get('scoring_authorized') is False, 'Invalid v6 preparation proof')
    require(proof.get('files_sha256') == {n: marker['files'][n]['sha256'] for n in PRODUCTS},
            'Preparation product bindings differ')
    contract = proof['contract']
    require(proof.get('contract_sha256') == digest(contract), 'Preparation contract SHA mismatch')
    require(read_json(package / 'family_config.json') == family_config(), 'Exact60/eight-family configuration required')
    # Parent preparation must independently validate original values, stereo provenance,
    # crop/hash correspondence, and locked/pilot ID AND global-group exclusion.
    require(contract.get('lineage_validated') is True
            and contract.get('original_54_values_preserved') is True
            and contract.get('native_stereo_only') is True
            and contract.get('historical_locked_pilot_ids_and_groups_excluded') is True,
            'Required cohort lineage checks missing')
    lineage = contract.get('input_files_sha256', {})
    require(bool(lineage), 'No bound preparation lineage')
    for path, expected in lineage.items():
        bound_file(path, expected, bindings)
    if synthetic:
        require(contract.get('synthetic_fixture_only') is True, 'Explicit synthetic fixture boundary required')
    else:
        # Reconstruct every metadata/ledger/descriptor token and hash every
        # committed SC product at context construction, including post-run.
        require(PREP6.validate_package(package, False) == proof, 'Strict v6 lineage replay differs')
        gate_ref = contract['external_gate']
        gate_path = bound_file(gate_ref['receipt_path'], gate_ref['receipt_sha256'], bindings)
        gate = read_json(gate_path)
        require(gate.get('measurement_gate_passed') is True and gate.get('candidate') == 'SC_L6'
                and gate.get('candidate_features') == SC_COLUMNS and gate.get('classifier_fitted') is False
                and gate.get('decision') == 'eligible_for_separately_frozen_exploratory_study'
                and gate.get('primary_checks') == gate.get('primary_passed') == 196
                and gate.get('codec_pairs') == gate.get('eligible_codec_pairs') == 28,
                'SC_L6 external music measurement gate not passed')
        freeze_path = bound_file(gate_ref['freeze_path'], gate['freeze_sha256'], bindings)
        raw_path = bound_file(gate_ref['raw_commit_path'], gate['raw_commit_sha256'], bindings)
        require(raw_path.name == 'COMMIT.json', 'Gate raw COMMIT required')
        freeze = read_json(freeze_path)
        require(freeze.get('candidate') == 'SC_L6' and freeze.get('candidate_features') == SC_COLUMNS
                and len(freeze.get('selected', [])) == 7, 'Gate freeze candidate/roster mismatch')
        # Original freeze paths belong to the gate host and need not exist on
        # an evaluator host. Bind the freeze bytes and independently replayed
        # raw publication; do not reinterpret historical absolute paths.
        require(bool(freeze.get('bindings')) and all(isinstance(p, str) and isinstance(h, str)
                and len(h) == 64 and set(h) <= set('0123456789abcdef')
                for p, h in freeze['bindings'].items()), 'Invalid gate protocol/code bindings')
        raw = read_json(raw_path)
        require(raw.get('freeze_sha256') == gate['freeze_sha256'] and bool(raw.get('products')),
                'Gate raw publication/freeze mismatch')
        review_ref = contract['independent_gate_review']
        review = read_json(bound_file(review_ref['receipt_path'], review_ref['receipt_sha256'], bindings))
        require(review.get('passed') is True and review.get('gate_receipt_sha256') == gate_ref['receipt_sha256']
                and review.get('freeze_sha256') == gate['freeze_sha256']
                and review.get('raw_commit_sha256') == gate['raw_commit_sha256']
                and review.get('primary_checks_replayed') == 196 and review.get('ratios_replayed') == 168
                and review.get('classifier_fitted') is False, 'Independent gate review missing or stale')
        require(any(Path(p).name == 'COMMIT.json' and p != str(raw_path) for p in lineage),
                'Extraction/cohort publication lineage lacks COMMIT')
    V5.recheck({'input_files_sha256': bindings, 'runtime': {
        'python_executable': str(Path(sys.executable).resolve()),
        'python_executable_sha256': sha(Path(sys.executable).resolve())}})
    return proof, bindings


def load_table(package, synthetic=False):
    metadata = read_csv(package / 'metadata_60s.csv', PREP5.META)
    features = read_csv(package / 'features_60s.csv', ['id', *DESCRIPTORS])
    ids = [r['id'] for r in metadata]
    require(ids == [r['id'] for r in features] and len(ids) == len(set(ids)), 'Exact ID order/uniqueness required')
    require(all(r['role'] == 'development' and r['duration_view'] == '60s' and r['id'].strip()
                and r['group_id'].strip() for r in metadata), 'Only exact60 development rows allowed')
    if synthetic:
        require(0 < len(ids) <= 16 and all(i.startswith('synthetic_v6_') for i in ids), 'Synthetic16 boundary')
    else:
        require(len(ids) == 2174 and Counter(r['source_group'] for r in metadata) == COUNTS
                and Counter(r['label'] for r in metadata) == {'0': 1278, '1': 896}, 'Exact2174 composition required')
    table = pd.DataFrame(metadata).rename(columns={'id': '__id', 'label': '__label', 'source_group': '__source',
                                                  'group_id': '__group', 'role': '__role'})
    table['__label'] = pd.to_numeric(table['__label'], errors='raise')
    require(set(table['__label']) == {0, 1} and table.groupby('__source')['__label'].nunique().max() == 1,
            'Binary source labels required')
    require(table.groupby('__group')['__label'].nunique().max() == 1
            and table.groupby('__group')['__source'].nunique().max() == 1, 'Global group crosses label/source')
    for column in DESCRIPTORS:
        values = [float('nan') if row[column].strip().lower() in ('', 'nan', 'na', 'null')
                  else float(row[column]) for row in features]
        require(all(math.isnan(v) or math.isfinite(v) for v in values), 'Infinite predictor')
        table[column] = values
    return table


def make_schedule(table, synthetic=False):
    require(set(table['__role']) == {'development'}, 'Role leakage into scheduling')
    folds = BASE.make_folds(table, FOLDS, SEED)
    omitted = V5.fold_omissions(table, folds)  # Same fixed seed/folds; no mutable v5 globals.
    schedule = []
    for index, fold in enumerate(folds):
        previous = set()
        for cap in QUANTITIES:
            train, test = BASE.deterministic_quantity(fold['train'], cap, SEED), fold['test']
            valid, reason = V2.valid_train(train, 2)
            require(valid, 'Invalid capped training: ' + reason)
            require(not set(train['__id']) & set(test['__id'])
                    and not set(train['__group']) & set(test['__group']), 'Train/test ID/group leakage')
            require(set(train['__id']) <= set(fold['train']['__id']) and len(train) < len(table), 'Improper training subset')
            require(fold['fold_type'] == 'ordinary_group_holdout_descriptive'
                    or fold['heldout_source'] not in set(train['__source']), 'Held source leakage')
            require(cap == 'all' or train.groupby('__source')['__group'].nunique().max() <= cap, 'Group cap exceeded')
            require(previous <= set(train['__id']), 'Non-nested caps')
            previous = set(train['__id'])
            record = dict(fold_index=index, quantity=cap, fold_type=fold['fold_type'], heldout_source=fold['heldout_source'],
                          opposite_group_fold=fold['opposite_group_fold'], train=V5.population(train), test=V5.population(test),
                          uncapped_train=V5.population(fold['train']), train_id_set_sha256=V2.id_set_hash(train['__id']),
                          test_id_set_sha256=V2.id_set_hash(test['__id']))
            record['fold_uid'] = digest(record)[:24]
            schedule.append((record, train, test, fold))
    n = len(COMBINATIONS)
    accounting = dict(valid_folds=len(folds), primary_fits=len(schedule)*n,
                      primary_prediction_rows=sum(len(t)*n for _, _, t, _ in schedule),
                      diagnostic_fits=sum(r['quantity'] == 'all' for r, *_ in schedule)*n*2,
                      diagnostic_prediction_rows=sum(len(t)*n*2 for r, _, t, _ in schedule if r['quantity'] == 'all'))
    if not synthetic:
        require(accounting == EXPECTED_REAL, 'Real38 schedule accounting changed')
        require(Counter(f['fold_type'] for f in folds) == {'human_source_holdout': 23, 'generator_holdout': 10,
                                                         'ordinary_group_holdout_descriptive': 5}, 'Real fold types changed')
        require({(r['fold_type'], r['heldout_source'], r['opposite_group_fold']) for r in omitted}
                == {('human_source_holdout', 'human_saraga_hindustani_v1', i) for i in (0, 4)}, 'Real omissions changed')
    return schedule, omitted, accounting


def build_context(package, synthetic=False):
    proof, bindings = validate_package(package, synthetic)
    table = load_table(Path(package), synthetic)
    require(proof.get('rows') == len(table), 'Preparation row count differs')
    schedule, omitted, accounting = make_schedule(table, synthetic)
    for name, expected in V5.FIXED_CORE.items():
        bound_file(ROOT / 'code' / name, expected, bindings)
    for path in (Path(__file__).resolve(), Path(V5.__file__).resolve(), Path(PREP5.__file__).resolve(), Path(PREP6.__file__).resolve(),
                 Path(PREP5.P4.__file__).resolve(), Path(PREP5.U.__file__).resolve(), PROTOCOL):
        bindings[str(path)] = sha(path)
    records = [r for r, *_ in schedule]
    executable = str(Path(sys.executable).resolve())
    contract = dict(schema_version=6, authorized_stage=STAGE, synthetic_test_only=synthetic,
        input_package=str(package), package_contract_sha256=proof['contract_sha256'], input_files_sha256=bindings,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     python_executable=executable, python_executable_sha256=sha(executable),
                     threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}),
        rows=len(table), population=V5.population(table), families=FAMILIES, combinations=COMBINATIONS,
        seed=SEED, quantities=list(QUANTITIES), opposite_group_folds=FOLDS, fixed_ridge=10., fixed_threshold=.5,
        positive_rule='score >= 0.5', prediction_link='raw_identity_unclipped', primary_feature_mode=MODES[0],
        all_cap_diagnostics=list(MODES[1:]), schedule=records, schedule_sha256=digest(records),
        omitted_source_or_group_cells=omitted, accounting=accounting, selection=False, threshold_tuning=False,
        full_cohort_refit=False, historical_locked_or_pilot_scoring=False, source_transfer_J_computed=False,
        source_ranking=False, winner_selection=False,
        sample_weighting='equal classes; equal sources within class; equal groups within source; equal samples within group; sum weights = training n',
        quantity_confound='ridge stays10 while weight sum equals training n; cap changes composition and effective regularization',
        preprocessing='train-only medians and weighted means/scales; all-missing train median0; near-zero scale1; no complete-case filtering',
        matched_comparison='same2174 cohort and exact fold/cap/test IDs for subsets with and without SC; no direct v5 outcome subtraction',
        exploratory_reuse='Mureka/Saraga consumed development; seven music controls new to stereo gate within reused MedleyDB corpus',
        reporting='source arms separate; unique OOF IDs within arm; equal group OOF rates; pair then fold then arm macro; no pooled OOF AUC')
    require(synthetic or all(v == '1' for v in contract['runtime']['threads'].values()), 'Real runtime requires three thread settings1')
    V5.recheck(contract)
    return contract, table, schedule


def authorize(contract, receipt, receipt_sha256):
    require(receipt is not None and receipt_sha256 is not None, 'Parent-frozen receipt required before fitting')
    require(sha(receipt) == receipt_sha256, 'Frozen receipt SHA mismatch')
    value = read_json(receipt)
    require(value.get('status') == 'frozen' and value.get('authorized_stage') == STAGE
            and value.get('contract') == contract and value.get('contract_sha256') == digest(contract), 'Frozen contract mismatch')
    review = value.get('independent_review', {})
    require(review.get('approved') is True and review.get('reviewer') == 'root'
            and isinstance(review.get('reviewed_utc'), str) and review['reviewed_utc'].strip(), 'Parent review required')
    return dict(status='frozen_verified', contract_sha256=digest(contract), receipt_sha256=receipt_sha256)


def model_result(train, test, record, combination, mode, line):
    require(combination in COMBINATIONS and mode in MODES, 'Unknown candidate/mode')
    columns = sum((FAMILIES[f] for f in combination.split('+')), [])
    model = V2.fit_candidate(train, columns, 'ridge', feature_mode=mode)
    model['missing_value_policy'] = POLICIES[mode]
    require(model['threshold'] == .5 and model['ridge'] == 10. and model['columns'] == columns
            and model['training_rows'] == len(train) and model['feature_mode'] == mode, 'Frozen fit changed')
    scores = V2.predict_candidate(test, model)
    require(scores.shape == (len(test),) and np.isfinite(scores).all(), 'Invalid raw scores')
    identity = dict(combination=combination, feature_mode=mode, quantity=record['quantity'], fold_uid=record['fold_uid'])
    common = dict(model_uid=digest(identity)[:24], model_sha256=digest({'model': model, 'threshold': .5}),
                  model_jsonl_line=line, **identity, **{k: record[k] for k in ('fold_index', 'fold_type', 'heldout_source',
                  'opposite_group_fold', 'train_id_set_sha256', 'test_id_set_sha256')}, training_rows=len(train), test_rows=len(test))
    return model, scores, common


def run_evaluation(output, schedule, contract, authorization):
    require(authorization.get('status') == 'frozen_verified' and authorization.get('contract_sha256') == digest(contract),
            'Verified frozen authorization required before any fit')
    require([r for r, *_ in schedule] == contract['schedule'], 'Schedule changed before fitting')
    count = 0
    with ExitStack() as stack:
        streams = {category: {name: V5.Stream(stack, output/(category+'_'+name+'.csv'), fields)
                   for name, fields in [('model_index', V5.INDEX), ('predictions', V5.PRED), ('pair_metrics', V5.PAIR),
                                        ('pooled_metrics', V5.POOLED), ('per_source_endpoints', V5.SOURCE)]}
                   for category in ('primary', 'diagnostic')}
        models = stack.enter_context((output/'fold_models.jsonl').open('x'))
        for record, train, test, fold in schedule:
            V5.recheck(contract)
            test = test.reset_index(drop=True)
            for combination in COMBINATIONS:
                for mode in (MODES if record['quantity'] == 'all' else MODES[:1]):
                    count += 1
                    model, scores, common = model_result(train, test, record, combination, mode, count)
                    uid = common['model_uid']; s = streams['primary' if mode == MODES[0] else 'diagnostic']
                    models.write(canonical(dict(common, model=model)).decode()+'\n'); s['model_index'].write(common)
                    for identity, score in zip(test[['__id', '__label', '__source', '__group', '__role']].itertuples(index=False, name=None), scores):
                        iid, label, source, group, role = identity
                        s['predictions'].write(dict(model_uid=uid, row_id=iid, label=int(label), source_group=source,
                            group_id=group, role=role, score=format(float(score), '.17g'), threshold=.5, predicted_label=int(score >= .5)))
                    s['pair_metrics'].many({k: (uid if k == 'model_uid' else row[k]) for k in V5.PAIR}
                                           for row in V2.pair_metrics(test, scores, fold, .5, 1))
                    s['pooled_metrics'].write(dict(model_uid=uid, test_rows=len(test), test_groups=test['__group'].nunique(),
                        threshold=.5, **V2.metrics_at_threshold(test['__label'].to_numpy(int), scores, .5)))
                    s['per_source_endpoints'].many(V5.source_endpoints(test, scores, uid))
            models.flush()
            print(canonical(dict(event='completed_fold_cap', fold_index=record['fold_index'], quantity=record['quantity'],
                                 completed_model_instances=count)).decode(), flush=True)
        observed = {category: {name: stream.count for name, stream in group.items()} for category, group in streams.items()}
    for category in ('primary', 'diagnostic'):
        require(observed[category]['model_index'] == contract['accounting'][category+'_fits']
                and observed[category]['predictions'] == contract['accounting'][category+'_prediction_rows']
                and observed[category]['pooled_metrics'] == contract['accounting'][category+'_fits'], 'Stream accounting mismatch')
    require(count == sum(contract['accounting'][k+'_fits'] for k in ('primary', 'diagnostic')), 'Model total mismatch')
    return dict(status='completed_exploratory_cv', model_instances=count, stream_counts=observed,
                all_models_are_training_fold_only=True, independent_numerical_audit_performed=False,
                full_cohort_refit=False, selection=False, threshold_tuning=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('draft', 'run'), required=True)
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path); parser.add_argument('--receipt-sha256')
    parser.add_argument('--synthetic-test-only', action='store_true')
    args = parser.parse_args(argv)
    require(args.output_dir.is_absolute() and args.output_dir.parent.resolve() == args.output_dir.parent
            and not args.output_dir.exists() and not args.output_dir.is_symlink(), 'New canonical output required')
    contract, _, schedule = build_context(args.package_dir, args.synthetic_test_only)
    authorization = authorize(contract, args.receipt, args.receipt_sha256) if args.stage == 'run' else None
    args.output_dir.mkdir()
    with (args.output_dir/'fold_registry.jsonl').open('x') as stream:
        for record, *_ in schedule: stream.write(canonical(record).decode()+'\n')
    write_json(args.output_dir/'omitted_fold_cells.json', contract['omitted_source_or_group_cells'])
    if args.stage == 'draft':
        write_json(args.output_dir/'preregistration_draft.json', dict(status='draft', authorized_stage=STAGE,
                   contract=contract, contract_sha256=digest(contract), fitting_started=False, independent_review=None))
    else:
        write_json(args.output_dir/'frozen_authorization.json', read_json(args.receipt))
        result = run_evaluation(args.output_dir, schedule, contract, authorization)
        after, _, _ = build_context(args.package_dir, args.synthetic_test_only)
        require(after == contract and sha(args.receipt) == authorization['receipt_sha256'], 'Inputs/receipt changed during fitting')
        write_json(args.output_dir/'run_manifest.json', dict(schema_version=6, stage=STAGE,
                   synthetic_test_only=args.synthetic_test_only, contract=contract, authorization=authorization, **result))
    V5.recheck(contract); V5.commit_output(args.output_dir)
    print(canonical(dict(status='draft_no_fit' if args.stage == 'draft' else 'committed_exploratory_cv',
                         output=str(args.output_dir), contract_sha256=digest(contract), accounting=contract['accounting'])).decode())


if __name__ == '__main__':
    main()
