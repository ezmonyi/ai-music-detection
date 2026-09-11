#!/usr/bin/env python3
"""Independent Saraga numerical transfer audit; no fit, inference or published replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import math
from pathlib import Path
import re

import numpy as np

# Structural parsing, schema, model-policy and immutable publication helpers only.
# Do not import the Saraga scorer or invoke any other module's prediction function.
import audit_mureka60_transfer_v1 as structure

require, sha, digest = structure.require, structure.sha, structure.digest
strict_json, csv_rows, Bindings = structure.strict_json, structure.csv_rows, structure.Bindings
FAMILIES, DESCRIPTORS, COMBINATIONS, CAPS, INDEX = (structure.FAMILIES, structure.DESCRIPTORS,
    structure.COMBINATIONS, structure.CAPS, structure.INDEX)
ROLE = 'external_human_unscored'
SOURCE = 'human_saraga_hindustani_v1'
SCORING_ROLE = 'external_human_frozen_v4_transfer'
STAGE = 'saraga103_frozen_v4_transfer_scoring_only'
META = ['id', 'label', 'source_group', 'group_id', 'role', 'evaluation_allowed', 'classifier_admission_authorized']
PRED = INDEX + META + ['scoring_role', 'synthetic_test_only', 'score', 'threshold', 'predicted_label']
METRICS = ['fp', 'tn', 'false_positive_rate', 'specificity',
           'equal_component_false_positive_rate', 'equal_component_specificity']
SUMMARY = INDEX + ['synthetic_test_only', 'rows', 'unique_ids', 'unique_groups', 'fp', 'tn', 'threshold', *METRICS[2:]]
COMPONENT = INDEX + ['synthetic_test_only', 'group_id', 'rows', 'fp', 'tn', 'false_positive_rate', 'specificity']
CELL = ['combination', 'quantity', 'fold_models', 'rows_per_model', 'synthetic_test_only'] + [m + '_' + s for m in METRICS for s in ('mean', 'min', 'max')]
OVERVIEW = ['S','D','R','P','F','H','M','S+D+R+P','S+D+R+P+F','S+D+R+P+H','S+D+R+P+M','S+D+R+P+F+H+M']
GROUP_COUNTS = [72, 14, 10, 5, 2]
ID_SHA = 'a0df862441fb0cf9f214d739a6fca55dc54626457fdd410810aeb1690d35321a'
ATOL = RTOL = 1e-12
SUMMARY_ATOL = 1e-15
FILES = {'model_index.csv', 'identity_roles.csv', 'predictions.csv', 'per_model_specificity.csv',
         'per_component_specificity.csv', 'all635_cells.csv', 'predefined_overview.csv',
         'output_accounting.json', 'scoring_receipt.json'}
PROTOCOL = Path(__file__).resolve().parent.parent / 'SARAGA103_TRANSFER_AUDIT_PROTOCOL_EN.md'


def independent_scores(features, model):
    """Accumulate each frozen affine term separately; no matmul/scorer replay."""
    coefficients = model['coefficients_with_intercept']
    result = np.full(len(next(iter(features.values()))), coefficients[0], dtype=np.float64)
    width = len(model['columns'])
    for j, column in enumerate(model['columns']):
        values = features[column]
        missing = np.isnan(values)
        filled = values.copy()
        filled[missing] = model['medians'][j]
        result += (filled - model['mean'][j]) / model['scale'][j] * coefficients[j + 1]
    for j, column in enumerate(model['columns']):
        missing = np.isnan(features[column]).astype(np.float64)
        result += (missing - model['mean'][width + j]) / model['scale'][width + j] * coefficients[width + j + 1]
    require(np.isfinite(result).all(), 'Independent score is nonfinite')
    return result


def contract_check(receipt, synthetic):
    require(receipt.get('status') == 'frozen' and receipt.get('authorized_stage') == STAGE, 'Unfrozen/wrong-stage scoring receipt')
    c = receipt['contract']
    require(receipt.get('contract_sha256') == digest(c), 'Frozen contract digest mismatch')
    review = receipt.get('independent_review', {})
    require(review.get('approved') is True and review.get('reviewer') == 'root'
            and isinstance(review.get('reviewed_utc'), str) and review['reviewed_utc'].strip(), 'Independent root approval missing')
    n = c['rows']
    require(type(n) is int and ((synthetic and 0 < n <= 16) or (not synthetic and n == 103)), 'Real103/synthetic16 row boundary')
    fixed = dict(schema_version=1, authorized_stage=STAGE, synthetic_test_only=synthetic,
        measurement_interface='synthetic_fixture' if synthetic else 'saraga103_old4_guard_v1_native_fhm_v1',
        old_development_admission=False, refitting=False, model_selection=False, threshold_tuning=False,
        families=FAMILIES, combinations=COMBINATIONS, caps=CAPS, fold_models=5,
        feature_mode='values_plus_missing', threshold=.5, positive_rule='score >= 0.5', prediction_link='identity_unclipped',
        model_instances=3175, prediction_rows=3175*n, summary_rows=3175, overview_combinations=OVERVIEW,
        endpoints=METRICS,
        forbidden_endpoints=['balanced_accuracy','roc_auc','two_class_accuracy','source_transfer_J','rank','winner'],
        comparison_tolerance=dict(score_atol=ATOL,score_rtol=RTOL,threshold_decisions='exact',integer_summaries='exact'),
        uncertainty='five-model min/max are descriptive ranges, not confidence intervals; same103 dependent observations',
        component_weighting='equal mean of five connected components; distinct from primary per-recording weights',
        measurement_role=ROLE,scoring_role=SCORING_ROLE,
        publication='exclusive directory reservation and hardlink COMMIT last; no overwrite or resume')
    require(set(c) == set(fixed) | {'rows','component_sizes','identity_sha256','model_index_sha256','input_files_sha256','runtime'}, 'Unknown/missing scoring contract field')
    require(all(type(c[k]) is type(v) and c[k] == v for k,v in fixed.items()), 'Scoring protocol changed')
    require(isinstance(c['component_sizes'], dict) and all(type(v) is int and v > 0 for v in c['component_sizes'].values())
            and sum(c['component_sizes'].values()) == n, 'Component denominators invalid')
    require(synthetic or sorted(c['component_sizes'].values(), reverse=True) == GROUP_COUNTS, 'Real component sizes changed')
    require(isinstance(c['input_files_sha256'], dict) and c['input_files_sha256'], 'Empty frozen input bindings')
    require(all(Path(p).is_absolute() and re.fullmatch('[0-9a-f]{64}', h) for p,h in c['input_files_sha256'].items()), 'Invalid frozen file binding')
    runtime = c['runtime']
    require(isinstance(runtime, dict) and set(runtime) == {'python','numpy','pandas','threads'}, 'Scorer runtime fields changed')
    require(synthetic or runtime['threads'] == {k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}, 'Real scorer threads uncontrolled')
    return c


def publication_check(scored, receipt, c, b, synthetic):
    require(scored.is_absolute() and scored.resolve() == scored and not scored.is_symlink(), 'Noncanonical scored directory')
    b.file(scored/'COMMIT.json')
    marker = strict_json(scored/'COMMIT.json')
    require(set(marker) == {'status','files','publication'} and marker['status'] == 'committed'
            and marker['publication'] == 'exclusive hardlinks, COMMIT last', 'Missing/invalid score publication COMMIT')
    require(set(marker['files']) == FILES | {'publication_manifest.json'}
            and {p.name for p in scored.iterdir()} == set(marker['files']) | {'COMMIT.json'}, 'Scored artifact inventory changed')
    for name, record in marker['files'].items():
        require(set(record) == {'sha256','bytes'} and type(record['bytes']) is int and record['bytes'] >= 0, 'Publication record invalid')
        b.file(scored/name, record['sha256'])
        require((scored/name).stat().st_size == record['bytes'], 'Published artifact size changed')
    manifest = strict_json(scored/'publication_manifest.json')
    expected = dict(status='scored',synthetic_test_only=synthetic,contract_sha256=digest(c),classifier_fitted=False,
        original_v4_development_admission=False,unique_new_ids=c['rows'],model_instances=3175,prediction_rows=3175*c['rows'])
    require(set(manifest) == set(expected) | {'files'} and all(type(manifest[k]) is type(v) and manifest[k] == v for k,v in expected.items()), 'Publication contract/count/scope mismatch')
    require(manifest['files'] == {k:v for k,v in marker['files'].items() if k != 'publication_manifest.json'}, 'Nested publication bindings differ')
    require(strict_json(scored/'scoring_receipt.json') == receipt, 'Published scoring receipt differs from approved receipt')


def source_data(c, inputs, b, synthetic):
    """Caller paths must be in the score freeze; no traversal of the raw-media graph."""
    require(set(inputs) == {'old4_csv','fhm_csv','models','model_index'}, 'Exact four numerical input paths required')
    for path in inputs.values():
        require(str(path) in c['input_files_sha256'], 'Numerical input absent from frozen bindings: '+str(path))
        b.file(path, c['input_files_sha256'][str(path)])
    models = strict_json(inputs['models'])
    index = sorted([{k:r[k] for k in INDEX} for r in csv_rows(inputs['model_index'])],
                   key=lambda r:(r['combination'],r['quantity'],r['fold_index']))
    structure.validate_index(index, models)
    for model in models.values():
        require(all('saraga' not in s.lower() for s in model['training_source_counts']), 'Saraga training leakage')
    old = csv_rows(inputs['old4_csv'], key='item_id')
    fhm = csv_rows(inputs['fhm_csv'], key='id')
    meta = [{k:r[k] for k in META} for r in fhm]
    ids = [r['id'] for r in meta]
    require(len(ids) == c['rows'] and [r['item_id'] for r in old] == ids, 'Old4/FHM exact identity order mismatch')
    require(all(r['label'] == '0' and r['source_group'] == SOURCE and r['role'] == ROLE
                and r['evaluation_allowed'] == r['classifier_admission_authorized'] == 'False'
                and r['group_id'].strip() for r in meta), 'Human cohort/source/role/admission changed')
    if synthetic:
        require(all(r['id'].startswith('synthetic_saraga_v4_') and r['group_id'].startswith('synthetic_') for r in meta), 'Synthetic identity marker missing')
    else:
        require(all(re.fullmatch(r'saraga_hindustani_[0-9a-f-]{36}', i) for i in ids)
                and hashlib.sha256('\n'.join(sorted(i.removeprefix('saraga_hindustani_') for i in ids)).encode()).hexdigest() == ID_SHA,
                'Frozen real identity set changed')
    require(dict(sorted(Counter(r['group_id'] for r in meta).items())) == c['component_sizes'], 'Component identity/count mismatch')
    values = {name:[] for name in DESCRIPTORS}
    for identity, sd, fh in zip(meta, old, fhm):
        require(sd['label'] == '0' and sd['source_id'] == SOURCE and sd['group_id'] == identity['group_id']
                and sd['status'] == 'complete' and fh['extraction_status'] == 'ok', 'Failed or misassigned measurement row')
        require(sd['native_sample_rate_hz'] == fh['native_sample_rate_hz'] and sd['native_sample_rate_hz'] in ('44100','48000'), 'Native sample rate lost/mismatched')
        for family, names in FAMILIES.items():
            row = sd if family in ('S','D','R','P') else fh
            for name in names:
                token = row[name].strip()
                missing = token.lower() in ('','nan','na','null')
                number = float('nan') if missing else float(token)
                require(missing or math.isfinite(number), 'Infinite/non-numeric descriptor: '+name)
                values[name].append(number)
    require(digest(meta) == c['identity_sha256'] and digest(index) == c['model_index_sha256'], 'Frozen numerical identity/model index digest changed')
    return models, index, meta, {k:np.asarray(v,dtype=np.float64) for k,v in values.items()}


def number_equal(token, expected, message, tolerance=SUMMARY_ATOL):
    value = float(token)
    require(math.isfinite(value) and abs(value - expected) <= tolerance, message)


def audit(scored, receipt_path, receipt_sha256, inputs, synthetic=False):
    scored, receipt_path = Path(scored), Path(receipt_path)
    b = Bindings()
    for path in (Path(__file__).resolve(), Path(structure.__file__).resolve(), PROTOCOL):
        b.file(path)
    b.file(receipt_path, receipt_sha256)
    receipt = strict_json(receipt_path)
    c = contract_check(receipt, synthetic)
    publication_check(scored, receipt, c, b, synthetic)
    models,index,meta,features = source_data(c, inputs, b, synthetic)
    require(csv_rows(scored/'model_index.csv', INDEX) == index, 'Published model index differs')
    require(csv_rows(scored/'identity_roles.csv', META, 'id') == meta, 'Published identities differ')
    summaries = csv_rows(scored/'per_model_specificity.csv', SUMMARY)
    components = csv_rows(scored/'per_component_specificity.csv', COMPONENT)
    groups = sorted(c['component_sizes'])
    require(len(summaries) == 3175 and len(components) == 3175*len(groups), 'Summary/component row counts changed')
    masks = {g:np.asarray([r['group_id'] == g for r in meta]) for g in groups}
    expected_cells = defaultdict(list)
    count, max_abs, max_rel, closest = 0, 0., 0., math.inf
    with (scored/'predictions.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == PRED, 'Prediction schema changed')
        for instance,(modelrow,summary) in enumerate(zip(index,summaries)):
            reconstructed = independent_scores(features, models[modelrow['model_sha256']])
            positive = reconstructed >= .5
            for identity, score in zip(meta,reconstructed):
                row = next(reader,None)
                require(row is not None and None not in row and None not in row.values(), 'Prediction missing/malformed')
                require(all(row[k] == v for k,v in modelrow.items()) and all(row[k] == v for k,v in identity.items()), 'Prediction identity/model/role mismatch')
                require(row['scoring_role'] == SCORING_ROLE and row['synthetic_test_only'] == str(synthetic)
                        and row['threshold'] == '0.5', 'Prediction threshold/scope changed')
                observed = float(row['score'])
                require(math.isfinite(observed), 'Published score nonfinite')
                error = abs(observed - float(score))
                require(error <= ATOL + RTOL*abs(float(score)), 'Independent numerical score mismatch')
                require(row['predicted_label'] == str(int(observed >= .5)) == str(int(score >= .5)), 'Exact threshold decision mismatch; tolerance cannot excuse crossing')
                max_abs = max(max_abs,error)
                max_rel = max(max_rel,error/max(1.,abs(float(score))))
                closest = min(closest,abs(float(score)-.5))
                count += 1
            rates = []
            for j,group in enumerate(groups):
                component = components[instance*len(groups)+j]
                n,fp = c['component_sizes'][group],int(positive[masks[group]].sum())
                exact = dict(modelrow,synthetic_test_only=str(synthetic),group_id=group,rows=str(n),fp=str(fp),tn=str(n-fp))
                require(all(component[k] == v for k,v in exact.items()), 'Component identity/denominator/integer count mismatch')
                number_equal(component['false_positive_rate'],fp/n,'Component FPR mismatch')
                number_equal(component['specificity'],(n-fp)/n,'Component specificity mismatch')
                rates.append(fp/n)
            n,fp = len(meta),int(positive.sum())
            exact = dict(modelrow,synthetic_test_only=str(synthetic),rows=str(n),unique_ids=str(n),unique_groups=str(len(groups)),fp=str(fp),tn=str(n-fp),threshold='0.5')
            require(all(summary[k] == v for k,v in exact.items()), 'Per-model integer denominator/count/model mismatch')
            metrics = dict(fp=fp,tn=n-fp,false_positive_rate=fp/n,specificity=(n-fp)/n,
                equal_component_false_positive_rate=math.fsum(rates)/len(groups),
                equal_component_specificity=math.fsum(1-r for r in rates)/len(groups))
            for name in METRICS[2:]:
                number_equal(summary[name],metrics[name],'Per-model metric mismatch: '+name)
            expected_cells[(modelrow['combination'],modelrow['quantity'])].append(metrics)
        require(next(reader,None) is None, 'Extra prediction rows')
    require(count == 3175*len(meta), 'Prediction accounting mismatch')
    cells = csv_rows(scored/'all635_cells.csv', CELL)
    keys = sorted(expected_cells)
    require(len(cells) == len(keys) == 635, 'Expected635 aggregate cells')
    for cell,key in zip(cells,keys):
        values = expected_cells[key]
        require(len(values) == 5 and cell['combination'] == key[0] and cell['quantity'] == key[1]
                and cell['fold_models'] == '5' and cell['rows_per_model'] == str(len(meta))
                and cell['synthetic_test_only'] == str(synthetic), 'Aggregate cell identity/count mismatch')
        for metric in METRICS:
            column = [v[metric] for v in values]
            expected = {'mean':math.fsum(column)/5,'min':min(column),'max':max(column)}
            for statistic,value in expected.items():
                number_equal(cell[metric+'_'+statistic],value,'Aggregate statistic mismatch: '+metric+'_'+statistic)
    overview = csv_rows(scored/'predefined_overview.csv', CELL)
    require(overview == [row for row in cells if row['combination'] in OVERVIEW] and len(overview) == 60, 'Predefined overview changed or selected after scoring')
    require(strict_json(scored/'output_accounting.json') == dict(prediction_rows=count,model_summary_rows=3175,
        component_summary_rows=3175*len(groups),aggregate_cells=635,integer_counts_consistent=True,
        independent_numerical_audit_performed=False), 'Scorer output accounting changed')
    b.recheck()
    return dict(schema_version=1,status='passed',synthetic_test_only=synthetic,rows=len(meta),
        model_instances=3175,prediction_rows=count,model_summary_rows=3175,component_summary_rows=len(components),aggregate_cells=635,
        component_sizes=c['component_sizes'],descriptors=54,missing_by_descriptor={k:int(np.isnan(v).sum()) for k,v in features.items()},
        score_atol=ATOL,score_rtol=RTOL,summary_atol=SUMMARY_ATOL,score_max_absolute_error=max_abs,
        score_max_relative_error_with_unit_floor=max_rel,minimum_independent_distance_to_threshold=closest,
        all_decisions_exact=True,all_models_all_ids_checked=True,all_integer_counts_exact=True,
        independent_method='columnwise affine accumulation of frozen median-filled values and missing indicators; no matmul or scorer import',
        equal_component_and_per_recording_metrics_separately_checked=True,all635_mean_min_max_checked=True,
        endpoints=METRICS,model_fitting_performed=False,model_selection_performed=False,threshold_tuning_performed=False,
        source_media_rehashed=False,source_media_decoded=False,
        scope='Bound accepted numerical inputs and score publication only; not a new raw-media admission or training-optimality audit.',
        frozen_receipt_sha256=b.files[str(receipt_path)],contract_sha256=digest(c),
        publication_commit_sha256=b.files[str(scored/'COMMIT.json')],checked_files_sha256=dict(sorted(b.files.items()))),b


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scored-dir','receipt','old4-csv','fhm-csv','models','model-index','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--synthetic-test-only',action='store_true')
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), 'Audit output must be new')
    inputs = {k:getattr(args,k) for k in ('old4_csv','fhm_csv','models','model_index')}
    report,b = audit(args.scored_dir,args.receipt,args.receipt_sha256,inputs,args.synthetic_test_only)
    structure.publish_report(report,b,args.output)
    print('Saraga numerical audit passed: '+str(args.output))


if __name__ == '__main__':
    main()
