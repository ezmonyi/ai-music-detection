#!/usr/bin/env python3
"""Independent read-only audit of frozen Mureka transfer scores; never fit or score for publication."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

# Intentionally duplicated protocol constants: no scoring/evaluation/preparation imports.
FAMILIES = {
    'S': ['s8__'+x for x in ('tilt_1_5k_db_oct', 'hf_tilt_5_7p5k_db_oct', 'hf_ratio_5_7p5_db', 'sibilance_ratio_5_7p5_db', 'hf_flatness_5_7p5', 'hf_entropy_5_7p5', 'hf_crest_5_7p5_db', 'fakeprint_peak_density_5_7p5_per_khz', 'fakeprint_periodicity_5_7p5', 'hf_flux_5_7p5', 'hf_frame_similarity_5_7p5', 'hf_power_sd_5_7p5_db', 'hf_mod_4_12_share_5_7p5', 'sibilance_contrast_5_7p5_db', 'sibilance_burst_rate_5_7p5_hz')],
    'D': ['d__'+x for x in ('dynamics_span', 'dynamics_iqr', 'dynamics_adjacent_change')],
    'R': ['r__'+x for x in ('ibi_cv', 'tempo_tv', 'tempo_entropy')],
    'P': ['p__'+x for x in ('section_duration_cv', 'section_duration_entropy', 'section_bars_cv', 'section_bars_offmode_fraction', 'section_duration_median', 'section_bars_median')],
    'F': ['F_'+x for x in ('phase_residual_cvar_all', 'phase_residual_cvar_attack', 'phase_residual_cvar_sustain', 'phase_residual_cvar_decay', 'group_delay_iqr_ms_all', 'group_delay_iqr_ms_attack', 'group_delay_iqr_ms_sustain', 'group_delay_iqr_ms_decay', 'group_delay_cross_band_iqr_ms_all', 'group_delay_cross_band_iqr_ms_attack', 'group_delay_cross_band_iqr_ms_sustain', 'group_delay_cross_band_iqr_ms_decay', 'phase_residual_cvar_decay_minus_sustain', 'group_delay_iqr_ms_decay_minus_sustain', 'group_delay_cross_band_iqr_ms_decay_minus_sustain')],
    'H': ['H_'+x for x in ('pitch_class_entropy_norm', 'pc_token_entropy_norm', 'chroma_path_change_median', 'chroma_path_change_iqr', 'pc_path_step_median', 'pc_path_large_step_rate')],
    'M': ['M_'+x for x in ('recurrence_peak_similarity', 'recurrence_density', 'recurrence_lag_contrast', 'best_lag_sec', 'best_transposition_semitones', 'returning_pattern_count')],
}
DESCRIPTORS = sum(FAMILIES.values(), [])
COMBINATIONS = ['+'.join(c) for n in range(1, 8) for c in itertools.combinations(FAMILIES, n)]
CAPS = ['25', '50', '100', '200', 'all']
META = ['id', 'label', 'source_group', 'group_id', 'role', 'acquisition_role']
INDEX = ['combination', 'quantity', 'fold_index', 'fold_uid', 'model_sha256', 'candidate_key', 'train_id_set_sha256', 'test_id_set_sha256']
PRED = INDEX + META + ['scoring_role', 'synthetic_test_only', 'score', 'threshold', 'predicted_label']
SUMMARY = INDEX + ['synthetic_test_only', 'rows', 'unique_ids', 'unique_groups', 'tp', 'fn', 'threshold', 'ai_sensitivity', 'false_negative_rate']
ROLE = 'external_generator_unscored'
SCORING_ROLE = 'external_generator_frozen_v4_transfer'
STAGE = 'mureka500_frozen_v4_transfer_scoring_only'
WEIGHTING = 'classes equal; sources equal within class; groups equal within source; samples equal within group'
MODEL_KEYS = {'columns', 'medians', 'mean', 'scale', 'coefficients_with_intercept', 'ridge', 'threshold', 'feature_mode', 'training_rows', 'training_source_counts', 'weighting', 'missing_value_policy', 'observed_fraction_by_column', 'model_type', 'prediction_link'}
RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
RD = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
REAL_INPUTS = {'old_results': RC/'results/equal60_results_v4_with_recovery_v1',
               'prepared': RD/'mureka60_inputs_v2', 'fhm': RC/'results/measurements_mureka60_fhm_v2'}
ATOL = 2e-12
RTOL = 2e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, 'Duplicate JSON key: '+k)
            out[k] = v
        return out
    def bad_constant(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    result = json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=bad_constant)
    require(isinstance(result, dict), 'Expected JSON object')
    return result


class Bindings:
    def __init__(self):
        self.files = {}

    def file(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve() == path,
                'Missing/noncanonical regular file: '+str(path))
        value = sha(path)
        require(expected is None or value == expected, 'Hash mismatch: '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == value, 'Input changed: '+str(path))
        self.files[str(path)] = value
        return value

    def recheck(self):
        for name, expected in list(self.files.items()):
            self.file(name, expected)


def csv_rows(path, header=None, key=None):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), 'Missing/duplicate CSV header')
        require(header is None or reader.fieldnames == header, 'CSV schema changed: '+str(path))
        rows = list(reader)
    require(rows and all(None not in r and None not in r.values() for r in rows), 'Empty/malformed CSV')
    if key:
        ids = [r[key] for r in rows]
        require(all(x.strip() for x in ids) and len(ids) == len(set(ids)), 'Duplicate/empty CSV IDs')
    return rows


def validate_model(model, combo):
    columns = sum((FAMILIES[f] for f in combo.split('+')), [])
    require(set(model) == MODEL_KEYS and model['columns'] == columns, 'Model fields/columns changed')
    expected = dict(ridge=10.0, threshold=.5, feature_mode='values_plus_missing', weighting=WEIGHTING,
                    missing_value_policy='train-only median plus feature-missing indicators',
                    model_type='weighted_ridge_linear_probability', prediction_link='identity')
    require(all(model[k] == v for k, v in expected.items()), 'Model policy changed')
    counts = model['training_source_counts']
    require(type(model['training_rows']) is int and model['training_rows'] > 0 and isinstance(counts, dict)
            and counts and all(type(n) is int and n > 0 and 'mureka' not in s.lower() for s, n in counts.items())
            and sum(counts.values()) == model['training_rows'], 'Training source/count leakage')
    for k, n in [('medians', len(columns)), ('mean', 2*len(columns)), ('scale', 2*len(columns)), ('coefficients_with_intercept', 2*len(columns)+1)]:
        array = np.asarray(model[k], dtype=float)
        require(array.shape == (n,) and np.isfinite(array).all(), 'Invalid frozen parameter: '+k)
    require((np.asarray(model['scale']) > 0).all(), 'Nonpositive scale')
    observed = model['observed_fraction_by_column']
    require(set(observed) == set(columns) and all(math.isfinite(v) and 0 <= v <= 1 for v in observed.values()), 'Invalid observation fractions')


def validate_index(index, models):
    require(len(index) == 3175 and all(list(r) == INDEX for r in index), 'Exactly3175 model instances required')
    triples = [(r['combination'], r['quantity'], r['fold_index']) for r in index]
    require(len(set(triples)) == 3175 and set(triples) == set(itertools.product(COMBINATIONS, CAPS, map(str, range(5)))), 'Incomplete/duplicate model grid')
    require(index == sorted(index, key=lambda r: (r['combination'], r['quantity'], r['fold_index'])), 'Model index order changed')
    folds = {}
    for r in index:
        key = (r['quantity'], r['fold_index'])
        fold_identity = tuple(r[k] for k in ('fold_uid', 'train_id_set_sha256', 'test_id_set_sha256'))
        require(key not in folds or folds[key] == fold_identity, 'Fold identity varies across subsets')
        folds[key] = fold_identity
        require(all(r[k].strip() for k in INDEX), 'Blank model index field')
        require(all(re.fullmatch('[0-9a-f]{64}', r[k]) for k in ('model_sha256', 'train_id_set_sha256', 'test_id_set_sha256')), 'Invalid model/index hash')
    require(len({v[0] for v in folds.values()}) == 25, 'Fold IDs not unique across caps')
    for combo, key in {(r['combination'], r['model_sha256']) for r in index}:
        require(key in models and key == digest({'model': models[key], 'threshold': .5}), 'Frozen model hash mismatch')
        validate_model(models[key], combo)


def independent_scores(features, model):
    """Columnwise affine accumulation, independently implemented without BLAS replay."""
    n = len(next(iter(features.values())))
    result = np.full(n, model['coefficients_with_intercept'][0], dtype=np.float64)
    d = len(model['columns'])
    for j, name in enumerate(model['columns']):
        values = features[name]
        absent = np.isnan(values)
        filled = values.copy()
        filled[absent] = model['medians'][j]
        result += ((filled-model['mean'][j])/model['scale'][j])*model['coefficients_with_intercept'][j+1]
    for j, name in enumerate(model['columns']):
        indicator = np.isnan(features[name]).astype(float)
        result += ((indicator-model['mean'][d+j])/model['scale'][d+j])*model['coefficients_with_intercept'][d+j+1]
    require(np.isfinite(result).all(), 'Nonfinite independently reconstructed score')
    return result


def contract_check(receipt, synthetic):
    require(receipt.get('status') == 'frozen' and receipt.get('authorized_stage') == STAGE, 'Unfrozen/wrong-stage receipt')
    c = receipt['contract']
    require(receipt.get('contract_sha256') == digest(c), 'Frozen contract hash mismatch')
    review = receipt.get('independent_review')
    require(isinstance(review, dict) and review.get('approved') is True and all(isinstance(review.get(k), str) and review[k].strip() for k in ('reviewer', 'reviewed_utc')), 'Missing independent approval')
    n = c['rows']
    require(type(n) is int and ((synthetic and 0 < n <= 16) or (not synthetic and n == 500)), 'Real mode requires500 rows; synthetic limited16')
    fixed = dict(schema_version=1, authorized_stage=STAGE, synthetic_test_only=synthetic,
                 measurement_interface='synthetic_fixture' if synthetic else 'mureka_v3',
                 old_development_admission=False, refitting=False, model_selection=False, threshold_tuning=False,
                 families=FAMILIES, combinations=COMBINATIONS, caps=CAPS, fold_models=5,
                 feature_mode='values_plus_missing', threshold=.5, positive_rule='score >= 0.5', prediction_link='identity_unclipped',
                 model_instances=3175, prediction_rows=3175*n, summary_rows=3175,
                 endpoints=['ai_sensitivity', 'false_negative_rate'], forbidden_endpoints=['balanced_accuracy', 'roc_auc', 'source_transfer_J'],
                 uncertainty='none; repeated model predictions are dependent observations of the same songs',
                 acquisition_role='reserved_unscored', measurement_role=ROLE, scoring_role=SCORING_ROLE)
    require(set(c) == set(fixed) | {'rows', 'identity_sha256', 'model_index_sha256', 'input_files_sha256', 'runtime'}, 'Unknown/missing contract field')
    require(all(type(c[k]) is type(v) and c[k] == v for k, v in fixed.items()), 'Transfer protocol contract changed')
    require(isinstance(c['input_files_sha256'], dict) and c['input_files_sha256'], 'Empty frozen input bindings')
    for path, value in c['input_files_sha256'].items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}', value), 'Invalid frozen input binding')
    require(isinstance(c['runtime'], dict) and set(c['runtime']) == {'python', 'numpy', 'pandas', 'threads'}, 'Missing scorer runtime')
    require(synthetic or c['runtime']['threads'] == {k:'1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}, 'Uncontrolled real scoring threads')
    return c


def source_data(c, roots, b, synthetic):
    """Bind only compact numerical/provenance inputs; never walk raw media graph."""
    if not synthetic:
        require(roots == REAL_INPUTS, 'Real numerical inputs must use exact canonical remote paths; no remapping')
    def bound(path, kind='json'):
        path = Path(path)
        require(str(path) in c['input_files_sha256'], 'Required input absent from frozen contract: '+str(path))
        b.file(path, c['input_files_sha256'][str(path)])
        if kind == 'hash_only':
            return None
        return strict_json(path) if kind == 'json' else csv_rows(path)
    old, prepared, fhm = roots['old_results'], roots['prepared'], roots['fhm']
    if not synthetic:
        # The scoring code is part of the reviewed receipt, but is never imported.
        bound(RC/'code/score_mureka60_frozen_v4.py', 'hash_only')
    if synthetic:
        marker = strict_json(prepared/'synthetic_audit_fixture.json')
        b.file(prepared/'synthetic_audit_fixture.json')
        require(marker == {'synthetic_test_only': True, 'purpose': 'independent_audit_handwritten_fixture_no_training'}, 'Synthetic marker missing')
    old_manifest = bound(old/'run_manifest.json')
    models = bound(old/'fold_models.json')
    pooled = bound(old/'development_group_cv_pooled_fold_metrics.csv', 'csv')
    for name in ('fold_models.json', 'development_group_cv_pooled_fold_metrics.csv'):
        require(old_manifest['files_sha256'].get(name) == b.files[str(old/name)], 'Original result model/index binding changed')
    require(old_manifest.get('schema_version') == 4 and old_manifest.get('stage') == 'dev', 'Wrong old result stage/schema')
    index = sorted([{k:r[k] for k in INDEX} for r in pooled], key=lambda r:(r['combination'], r['quantity'], r['fold_index']))
    validate_index(index, models)
    native = bound(prepared/'native_metadata_60s.csv', 'csv')
    inference = bound(prepared/'inference_manifest.csv', 'csv')
    sdrp = bound(prepared/'measurements/features/expanded_features_60s.csv', 'csv')
    extraction = bound(prepared/'measurements/extraction_receipt.json')
    acceptance = bound(fhm/'acceptance.json')
    fhmrows = bound(fhm/'features/features.csv', 'csv')
    for receipt in (extraction, acceptance):
        require(receipt.get('status') == 'passed' and receipt.get('rows') == c['rows'] and receipt.get('role') == ROLE
                and receipt.get('classifier_fitted') is False and receipt.get('scores_generated') is False, 'Measurement acceptance/scope mismatch')
    require(extraction.get('classifier_admission_authorized') is False
            and extraction.get('before_after_all_provenance_inputs_products_dependencies_release_proof_verified') is True,
            'v3 extraction acceptance missing')
    require(extraction.get('canonical_sha256') == digest({k:v for k,v in extraction.items() if k != 'canonical_sha256'}), 'Extraction receipt seal changed')
    require(acceptance.get('all_descriptor_values_checked') is True and acceptance.get('original_sources_rehashed_after_extraction') is True, 'FHM acceptance incomplete')
    sdpath = str(prepared/'measurements/features/expanded_features_60s.csv')
    record = extraction['features'].get(sdpath, {})
    require(record.get('sha256') == b.files[sdpath] and record.get('bytes') == Path(sdpath).stat().st_size, 'SDRP CSV acceptance hash mismatch')
    fhpath = str(fhm/'features/features.csv')
    require(acceptance['outputs_sha256'].get(fhpath) == b.files[fhpath], 'FHM CSV acceptance hash mismatch')
    def by_id(rows, key):
        ids = [r[key] for r in rows]
        require(len(ids) == len(set(ids)) and all(i.strip() for i in ids), 'Duplicate/empty measurement ID')
        return dict(zip(ids, rows))
    native_map, sdrp_map, fhm_map, inference_map = [by_id(rows, key) for rows,key in ((native,'id'), (sdrp,'item_id'), (fhmrows,'id'), (inference,'item_id'))]
    require(len(native) == c['rows'] and set(native_map) == set(sdrp_map) == set(fhm_map) == set(inference_map), 'Missing/excluded measurement IDs')
    meta = [{k:r[k] for k in META} for r in native]
    pattern = r'synthetic_mureka_v4_\d+' if synthetic else r'music8k_mureka_v9_\d+'
    require(all(re.fullmatch(pattern, r['id']) and r['label'] == '1' and r['source_group'] == 'Mureka_v9'
                and r['role'] == ROLE and r['acquisition_role'] == 'reserved_unscored' and r['group_id'].strip()
                and (not synthetic or r['group_id'].startswith('synthetic_')) for r in meta), 'Invalid cohort/roles')
    require(all(r.get('classifier_admission_authorized') == 'False' for r in native), 'Native development admission changed')
    values = {name:[] for name in DESCRIPTORS}
    for r in meta:
        sd, fh, inf = sdrp_map[r['id']], fhm_map[r['id']], inference_map[r['id']]
        require(sd['status'] == 'complete' and fh['extraction_status'] == 'ok', 'Failed measurement row')
        require(all(fh[k] == r[k] for k in META) and fh.get('classifier_admission_authorized') == 'False', 'FHM metadata role mismatch')
        require(sd['label'] == r['label'] and sd['source_id'] == r['source_group'] and sd['group_id'] == r['group_id'], 'SDRP metadata mismatch')
        require(inf['role'] == ROLE and inf['label'] == '1' and inf['source_id'] == 'Mureka_v9' and inf['group_id'] == r['group_id']
                and inf.get('acquisition_role') == 'reserved_unscored' and inf.get('classifier_admission_authorized') == 'False', 'Prepared metadata mismatch')
        for family, columns in FAMILIES.items():
            source = sd if family in ('S','D','R','P') else fh
            for name in columns:
                text = source[name].strip()
                number = float('nan') if text.lower() in ('', 'na', 'nan', 'null') else float(text)
                require(math.isnan(number) if text.lower() in ('', 'na', 'nan', 'null') else math.isfinite(number), 'Nonfinite descriptor')
                values[name].append(number)
    return models, index, meta, {k:np.asarray(v, dtype=float) for k,v in values.items()}


def audit(scored, receipt_path, roots, synthetic=False):
    b = Bindings()
    b.file(Path(__file__).resolve())
    b.file(receipt_path)
    receipt = strict_json(receipt_path)
    c = contract_check(receipt, synthetic)
    manifest_path = scored/'publication_manifest.json'
    b.file(manifest_path)
    publication = strict_json(manifest_path)
    fields = dict(status='scored', synthetic_test_only=synthetic, contract_sha256=digest(c), classifier_fitted=False,
                  original_v4_development_admission=False, unique_new_ids=c['rows'], model_instances=3175,
                  prediction_rows=3175*c['rows'], summary_rows=3175)
    require(set(publication) == set(fields)|{'files'} and all(type(publication[k]) is type(v) and publication[k] == v for k,v in fields.items()), 'Publication contract/count mismatch')
    names = {'model_index.csv', 'identity_roles.csv', 'predictions.csv', 'per_model_sensitivity.csv', 'scoring_receipt.json'}
    require(set(publication['files']) == names, 'Publication file inventory mismatch')
    require({p.name for p in scored.iterdir()} == names|{'publication_manifest.json'}, 'Unexpected/missing scored artifact')
    for name, record in publication['files'].items():
        require(set(record) == {'sha256', 'bytes'} and type(record['bytes']) is int, 'Invalid publication record')
        b.file(scored/name, record['sha256'])
        require((scored/name).stat().st_size == record['bytes'], 'Published file size mismatch')
    require(strict_json(scored/'scoring_receipt.json') == receipt, 'Published receipt differs from approved receipt')
    models, index, meta, features = source_data(c, roots, b, synthetic)
    require(csv_rows(scored/'model_index.csv', INDEX) == index and digest(index) == c['model_index_sha256'], 'Published model index differs from frozen old models')
    require(csv_rows(scored/'identity_roles.csv', META, 'id') == meta and digest(meta) == c['identity_sha256'], 'Published identities differ from all measurement IDs')
    summaries = csv_rows(scored/'per_model_sensitivity.csv', SUMMARY)
    require(len(summaries) == 3175, 'Exactly3175 sensitivity summaries required')
    n, groups = len(meta), len({r['group_id'] for r in meta})
    count, max_abs, max_rel, closest = 0, 0., 0., math.inf
    with (scored/'predictions.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == PRED, 'Prediction schema changed')
        for modelrow, summary in zip(index, summaries):
            expected = independent_scores(features, models[modelrow['model_sha256']])
            tp = 0
            for identity, reconstructed in zip(meta, expected):
                row = next(reader, None)
                require(row is not None and None not in row and None not in row.values(), 'Missing/malformed prediction row')
                require(all(row[k] == v for k,v in modelrow.items()) and all(row[k] == v for k,v in identity.items()), 'Prediction ID/model/role mismatch')
                require(row['scoring_role'] == SCORING_ROLE and row['synthetic_test_only'] == str(synthetic) and row['threshold'] == '0.5', 'Prediction scope/threshold changed')
                observed = float(row['score'])  # Preserve adjacent floats around .5.
                require(math.isfinite(observed), 'Nonfinite published score')
                error = abs(observed-float(reconstructed))
                require(error <= ATOL + RTOL*abs(float(reconstructed)), 'Independent numerical score mismatch')
                published_decision = int(observed >= .5)
                independent_decision = int(reconstructed >= .5)
                require(row['predicted_label'] == str(published_decision) == str(independent_decision), 'Decision mismatch; tolerance cannot excuse threshold crossing')
                max_abs, max_rel = max(max_abs, error), max(max_rel, error/max(1., abs(float(reconstructed))))
                closest = min(closest, abs(float(reconstructed)-.5))
                tp += independent_decision
                count += 1
            require(all(summary[k] == v for k,v in modelrow.items()), 'Summary model identity mismatch')
            exact = dict(synthetic_test_only=str(synthetic), rows=str(n), unique_ids=str(n), unique_groups=str(groups), tp=str(tp), fn=str(n-tp), threshold='0.5')
            require(all(summary[k] == v for k,v in exact.items()), 'Summary denominator/count/scope mismatch')
            require(abs(float(summary['ai_sensitivity'])-tp/n) <= 1e-15 and abs(float(summary['false_negative_rate'])-(n-tp)/n) <= 1e-15, 'Summary sensitivity/FNR mismatch')
        require(next(reader, None) is None, 'Extra prediction rows')
    require(count == 3175*n, 'Prediction count mismatch')
    b.recheck()
    report = dict(schema_version=1, status='passed', synthetic_test_only=synthetic, rows=n,
                  unique_ids=n, positive_class_count=n, unique_groups=groups,
                  group_counts=dict(sorted(Counter(r['group_id'] for r in meta).items())),
                  model_instances=3175, combinations=127, caps=CAPS, fold_models_per_cap=5,
                  prediction_rows=count, sensitivity_summary_rows=3175, descriptors=54,
                  missing_by_descriptor={k:int(np.isnan(v).sum()) for k,v in features.items()},
                  all_ids_in_every_model=True, all_decisions_match_published_and_independent_scores=True,
                  score_absolute_tolerance=ATOL, score_relative_tolerance=RTOL,
                  score_max_absolute_error=max_abs, score_max_relative_error_with_unit_floor=max_rel,
                  minimum_independent_distance_to_threshold=closest, summary_absolute_tolerance=1e-15,
                  endpoints=['ai_sensitivity', 'false_negative_rate'], denominator_per_model=n,
                  model_fitting_performed=False, new_source_transform_learning_performed=False,
                  source_media_rehashed=False, source_media_decoded=False,
                  scope='Numerical transfer audit of frozen accepted measurement CSVs and saved models; no new raw-media verification or training-optimality audit.',
                  uncertainty='none; repeated model predictions are dependent observations of the same songs',
                  frozen_receipt_sha256=b.files[str(receipt_path)], contract_sha256=digest(c),
                  publication_manifest_sha256=b.files[str(manifest_path)],
                  auditor_sha256=b.files[str(Path(__file__).resolve())], checked_files_sha256=dict(sorted(b.files.items())))
    return report, b


def publish_report(report, bindings, output):
    """Same-directory hard-link publication: atomic, no replace, works without renameat2."""
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), 'Audit output must be a NEW absolute file path')
    require(output.parent.is_dir() and output.parent.resolve() == output.parent, 'Audit parent must exist and be canonical')
    fd, name = tempfile.mkstemp(prefix='.'+output.name+'.', dir=output.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(report, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        bindings.recheck()
        os.link(name, output)  # Fails if any destination exists; no clobber fallback.
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scored-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic-test-only', action='store_true')
    parser.add_argument('--synthetic-root', type=Path)
    args = parser.parse_args(argv)
    require(all(p.is_absolute() for p in (args.scored_dir, args.receipt, args.output)), 'All paths must be absolute')
    require(not args.output.exists() and not args.output.is_symlink(), 'Refusing existing audit output')
    require(args.synthetic_test_only == (args.synthetic_root is not None), 'Synthetic root and synthetic flag must be supplied together')
    roots = REAL_INPUTS
    if args.synthetic_test_only:
        require(args.synthetic_root.is_absolute(), 'Synthetic root must be absolute')
        roots = {k:args.synthetic_root/k for k in REAL_INPUTS}
    report, bindings = audit(args.scored_dir, args.receipt, roots, args.synthetic_test_only)
    publish_report(report, bindings, args.output)
    print(json.dumps({'status': 'passed', 'output': str(args.output), 'synthetic_test_only': args.synthetic_test_only,
                      'prediction_rows': report['prediction_rows'], 'audit_sha256': sha(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
