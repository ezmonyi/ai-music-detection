#!/usr/bin/env python3
"""Handwritten synthetic numerical fixtures, never real scoring or fitting."""
import ast
import copy
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import audit_mureka60_transfer_v1 as A


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def table(path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    roots = {k:root/k for k in A.REAL_INPUTS}
    old, prepared, fhm = roots['old_results'], roots['prepared'], roots['fhm']
    put(prepared/'synthetic_audit_fixture.json', {'synthetic_test_only': True, 'purpose': 'independent_audit_handwritten_fixture_no_training'})
    meta = [dict(id=f'synthetic_mureka_v4_{i}', label='1', source_group='Mureka_v9', group_id=f'synthetic_group_{i//2}', role=A.ROLE, acquisition_role='reserved_unscored') for i in range(3)]
    native = [dict(r, classifier_admission_authorized='False') for r in meta]
    table(prepared/'native_metadata_60s.csv', native)
    table(prepared/'inference_manifest.csv', [dict(item_id=r['id'], label='1', source_id='Mureka_v9', role=A.ROLE, group_id=r['group_id'], acquisition_role='reserved_unscored', classifier_admission_authorized='False') for r in meta])
    sdrp, fhmrows = [], []
    for i, r in enumerate(meta):
        value = ['1', '', '9'][i]
        sdrp.append(dict(item_id=r['id'], label='1', source_id='Mureka_v9', group_id=r['group_id'], status='complete', **{c:value for fam in ('S','D','R','P') for c in A.FAMILIES[fam]}))
        fhmrows.append(dict(native[i], extraction_status='ok', **{c:value for fam in ('F','H','M') for c in A.FAMILIES[fam]}))
    sdpath, fhpath = prepared/'measurements/features/expanded_features_60s.csv', fhm/'features/features.csv'
    # Deliberately permute both tables: auditor must join exact ID, never row offset.
    table(sdpath, sdrp[::-1])
    table(fhpath, fhmrows[1:]+fhmrows[:1])
    extraction = dict(status='passed', rows=3, role=A.ROLE, classifier_fitted=False, scores_generated=False,
                      classifier_admission_authorized=False, before_after_all_provenance_inputs_products_dependencies_release_proof_verified=True,
                      features={str(sdpath):dict(sha256=A.sha(sdpath), bytes=sdpath.stat().st_size)})
    extraction['canonical_sha256'] = A.digest(extraction)
    put(prepared/'measurements/extraction_receipt.json', extraction)
    put(fhm/'acceptance.json', dict(status='passed', rows=3, role=A.ROLE, classifier_fitted=False, scores_generated=False,
                                  all_descriptor_values_checked=True, original_sources_rehashed_after_extraction=True,
                                  outputs_sha256={str(fhpath):A.sha(fhpath)}))
    models, index = {}, []
    for combo in A.COMBINATIONS:
        cols = sum((A.FAMILIES[f] for f in combo.split('+')), [])
        d = len(cols)
        beta = [.5]+[0.]*(2*d)
        beta[1], beta[d+1] = .25, -.1
        model = dict(columns=cols, medians=[1.]*d, mean=[1.]*d+[0.]*d, scale=[2.]*d+[1.]*d,
                     coefficients_with_intercept=beta, ridge=10., threshold=.5, feature_mode='values_plus_missing',
                     training_rows=2, training_source_counts={'0:synthetic_human':1, '1:Suno':1}, weighting=A.WEIGHTING,
                     missing_value_policy='train-only median plus feature-missing indicators',
                     observed_fraction_by_column={c:.5 for c in cols}, model_type='weighted_ridge_linear_probability', prediction_link='identity')
        key = A.digest({'model':model, 'threshold':.5})
        models[key] = model
        for cap in A.CAPS:
            for fold in range(5):
                index.append(dict(combination=combo, quantity=cap, fold_index=str(fold), fold_uid=f'synthetic_{cap}_{fold}',
                                  model_sha256=key, candidate_key='synthetic_'+combo,
                                  train_id_set_sha256=A.digest(['old_train']), test_id_set_sha256=A.digest(['old_test'])))
    index.sort(key=lambda r:(r['combination'], r['quantity'], r['fold_index']))
    put(old/'fold_models.json', models)
    table(old/'development_group_cv_pooled_fold_metrics.csv', index, A.INDEX)
    put(old/'run_manifest.json', dict(schema_version=4, stage='dev', files_sha256={n:A.sha(old/n) for n in ('fold_models.json','development_group_cv_pooled_fold_metrics.csv')}))
    paths = [old/'fold_models.json', old/'development_group_cv_pooled_fold_metrics.csv', old/'run_manifest.json',
             prepared/'native_metadata_60s.csv', prepared/'inference_manifest.csv', sdpath,
             prepared/'measurements/extraction_receipt.json', fhm/'acceptance.json', fhpath]
    c = dict(schema_version=1, authorized_stage=A.STAGE, synthetic_test_only=True, measurement_interface='synthetic_fixture',
             old_development_admission=False, refitting=False, model_selection=False, threshold_tuning=False,
             families=A.FAMILIES, combinations=A.COMBINATIONS, caps=A.CAPS, fold_models=5,
             feature_mode='values_plus_missing', threshold=.5, positive_rule='score >= 0.5', prediction_link='identity_unclipped',
             rows=3, model_instances=3175, prediction_rows=9525, summary_rows=3175,
             identity_sha256=A.digest(meta), model_index_sha256=A.digest(index), input_files_sha256={str(p):A.sha(p) for p in paths},
             runtime=dict(python='test', numpy='test', pandas='test', threads={}), endpoints=['ai_sensitivity','false_negative_rate'],
             forbidden_endpoints=['balanced_accuracy','roc_auc','source_transfer_J'],
             uncertainty='none; repeated model predictions are dependent observations of the same songs',
             acquisition_role='reserved_unscored', measurement_role=A.ROLE, scoring_role=A.SCORING_ROLE)
    receipt = dict(status='frozen', authorized_stage=A.STAGE, contract=c, contract_sha256=A.digest(c),
                   independent_review=dict(approved=True, reviewer='synthetic_test_only', reviewed_utc='2026-09-07T00:00:00Z'))
    put(root/'receipt.json', receipt)
    scored = root/'scored'
    put(scored/'scoring_receipt.json', receipt)
    table(scored/'model_index.csv', index, A.INDEX)
    table(scored/'identity_roles.csv', meta, A.META)
    # Analytic fixture values, NOT produced by auditor/scorer replay.
    predictions = [dict(m, **r, scoring_role=A.SCORING_ROLE, synthetic_test_only='True', score=s,
                        threshold='0.5', predicted_label=p) for m in index for r,s,p in zip(meta,['0.5','0.4','1.5'],['1','0','1'])]
    table(scored/'predictions.csv', predictions, A.PRED)
    table(scored/'per_model_sensitivity.csv', [dict(m, synthetic_test_only='True', rows='3', unique_ids='3', unique_groups='2',
                                                tp='2', fn='1', threshold='0.5', ai_sensitivity=str(2/3), false_negative_rate=str(1/3)) for m in index], A.SUMMARY)
    publication = dict(status='scored', synthetic_test_only=True, contract_sha256=A.digest(c), classifier_fitted=False,
                       original_v4_development_admission=False, unique_new_ids=3, model_instances=3175, prediction_rows=9525, summary_rows=3175,
                       files={p.name:dict(sha256=A.sha(p), bytes=p.stat().st_size) for p in scored.iterdir()})
    put(scored/'publication_manifest.json', publication)
    return roots


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.roots = fixture(self.root)
        self.scored = self.root/'scored'
        self.receipt = self.root/'receipt.json'

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self):
        return A.audit(self.scored, self.receipt, self.roots, True)

    def republish(self):
        manifest = A.strict_json(self.scored/'publication_manifest.json')
        for name in manifest['files']:
            path = self.scored/name
            manifest['files'][name] = dict(sha256=A.sha(path), bytes=path.stat().st_size)
        put(self.scored/'publication_manifest.json', manifest)

    def update_receipt(self, callback):
        receipt = A.strict_json(self.receipt)
        callback(receipt)
        receipt['contract_sha256'] = A.digest(receipt['contract'])
        put(self.receipt, receipt)
        put(self.scored/'scoring_receipt.json', receipt)
        manifest = A.strict_json(self.scored/'publication_manifest.json')
        manifest['contract_sha256'] = receipt['contract_sha256']
        put(self.scored/'publication_manifest.json', manifest)
        self.republish()

    def alter_table(self, name, callback):
        rows = A.csv_rows(self.scored/name)
        callback(rows)
        table(self.scored/name, rows)
        self.republish()

    def alter_source_table(self, source, callback):
        rows = A.csv_rows(source)
        callback(rows)
        table(source, rows)
        sdpath = self.roots['prepared']/'measurements/features/expanded_features_60s.csv'
        fhpath = self.roots['fhm']/'features/features.csv'
        if source == sdpath:
            receipt_path = self.roots['prepared']/'measurements/extraction_receipt.json'
            receipt = A.strict_json(receipt_path)
            receipt['features'][str(source)] = dict(sha256=A.sha(source), bytes=source.stat().st_size)
            receipt['canonical_sha256'] = A.digest({k:v for k,v in receipt.items() if k != 'canonical_sha256'})
            put(receipt_path, receipt)
        if source == fhpath:
            receipt_path = self.roots['fhm']/'acceptance.json'
            receipt = A.strict_json(receipt_path)
            receipt['outputs_sha256'][str(source)] = A.sha(source)
            put(receipt_path, receipt)
        def refresh(receipt):
            hashes = receipt['contract']['input_files_sha256']
            receipt['contract']['input_files_sha256'] = {p:A.sha(p) for p in hashes}
        self.update_receipt(refresh)

    def test_complete_independent_join_and_numeric_audit(self):
        with patch.object(np.linalg, 'solve', side_effect=AssertionError('No fitting')), patch.object(np, 'median', side_effect=AssertionError('No median learning')):
            report, bindings = self.audit()
        self.assertEqual(report['prediction_rows'], 9525)
        self.assertEqual(report['sensitivity_summary_rows'], 3175)
        self.assertEqual(report['positive_class_count'], 3)
        self.assertEqual(report['unique_groups'], 2)
        self.assertEqual(report['score_max_absolute_error'], 0.)
        self.assertEqual(report['missing_by_descriptor'], {k:1 for k in A.DESCRIPTORS})
        self.assertFalse(report['source_media_rehashed'])
        self.assertFalse(report['source_media_decoded'])
        self.assertEqual(report['minimum_independent_distance_to_threshold'], 0.)
        output = self.root/'audit.json'
        A.publish_report(report, bindings, output)
        self.assertEqual(A.strict_json(output), report)
        with self.assertRaisesRegex(ValueError, 'NEW'):
            A.publish_report(report, bindings, output)

    def test_score_tamper_even_with_new_publication_hash(self):
        self.alter_table('predictions.csv', lambda r:r[0].update(score='0.6'))
        with self.assertRaisesRegex(ValueError, 'numerical'):
            self.audit()

    def test_borderline_tolerance_does_not_excuse_decision_crossing(self):
        self.alter_table('predictions.csv', lambda r:r[0].update(score=repr(math.nextafter(.5, 0)), predicted_label='0'))
        with self.assertRaisesRegex(ValueError, 'Decision mismatch'):
            self.audit()

    def test_label_must_match_published_score(self):
        self.alter_table('predictions.csv', lambda r:r[0].update(predicted_label='0'))
        with self.assertRaisesRegex(ValueError, 'Decision mismatch'):
            self.audit()

    def test_score_tolerance_on_same_decision(self):
        self.alter_table('predictions.csv', lambda r:r[2].update(score=repr(1.5+1e-13)))
        report, _ = self.audit()
        self.assertGreater(report['score_max_absolute_error'], 0)

    def test_summary_denominator_rejected(self):
        self.alter_table('per_model_sensitivity.csv', lambda r:r[0].update(rows='2'))
        with self.assertRaisesRegex(ValueError, 'denominator'):
            self.audit()

    def test_summary_fn_and_endpoint_rejected(self):
        self.alter_table('per_model_sensitivity.csv', lambda r:r[0].update(false_negative_rate='0.5'))
        with self.assertRaisesRegex(ValueError, 'sensitivity/FNR'):
            self.audit()

    def test_summary_group_count_rejected(self):
        self.alter_table('per_model_sensitivity.csv', lambda r:r[0].update(unique_groups='3'))
        with self.assertRaisesRegex(ValueError, 'denominator'):
            self.audit()

    def test_duplicate_prediction_id_rejected(self):
        self.alter_table('predictions.csv', lambda r:r[1].update(id=r[0]['id']))
        with self.assertRaisesRegex(ValueError, 'ID/model/role'):
            self.audit()

    def test_missing_and_extra_prediction_rows_rejected(self):
        self.alter_table('predictions.csv', lambda r:r.pop())
        with self.assertRaisesRegex(ValueError, 'Missing/malformed'):
            self.audit()

    def test_extra_prediction_row_rejected(self):
        self.alter_table('predictions.csv', lambda r:r.append(r[-1].copy()))
        with self.assertRaisesRegex(ValueError, 'Extra prediction'):
            self.audit()

    def test_role_leakage_rejected(self):
        self.alter_table('identity_roles.csv', lambda r:r[0].update(role='development'))
        with self.assertRaisesRegex(ValueError, 'identities'):
            self.audit()

    def test_model_grid_tamper_rejected(self):
        self.alter_table('model_index.csv', lambda r:r[0].update(model_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'model index'):
            self.audit()

    def test_no_forbidden_endpoint_column(self):
        self.alter_table('per_model_sensitivity.csv', lambda r:[row.update(roc_auc='1') for row in r])
        with self.assertRaisesRegex(ValueError, 'CSV schema'):
            self.audit()

    def test_no_draft_receipt(self):
        self.update_receipt(lambda r:r.update(status='draft'))
        with self.assertRaisesRegex(ValueError, 'Unfrozen'):
            self.audit()

    def test_contract_policy_rejected(self):
        self.update_receipt(lambda r:r['contract'].update(threshold=.4))
        with self.assertRaisesRegex(ValueError, 'protocol contract'):
            self.audit()

    def test_source_csv_tamper_rejected(self):
        path = self.roots['fhm']/'features/features.csv'
        rows = A.csv_rows(path)
        rows[0][A.FAMILIES['M'][0]] = '123'
        table(path, rows)
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            self.audit()

    def test_measurement_duplicate_id_rejected_after_rebinding(self):
        path = self.roots['fhm']/'features/features.csv'
        self.alter_source_table(path, lambda r:r[0].update(id=r[1]['id']))
        with self.assertRaisesRegex(ValueError, 'Duplicate/empty measurement'):
            self.audit()

    def test_measurement_exclusion_rejected_after_rebinding(self):
        path = self.roots['fhm']/'features/features.csv'
        self.alter_source_table(path, lambda r:r.pop())
        with self.assertRaisesRegex(ValueError, 'Missing/excluded measurement'):
            self.audit()

    def test_measurement_infinity_rejected_after_rebinding(self):
        path = self.roots['fhm']/'features/features.csv'
        self.alter_source_table(path, lambda r:r[0].update({A.FAMILIES['M'][0]:'inf'}))
        with self.assertRaisesRegex(ValueError, 'Nonfinite descriptor'):
            self.audit()

    def test_changed_measurement_value_detected_numerically(self):
        path = self.roots['fhm']/'features/features.csv'
        self.alter_source_table(path, lambda r:r[0].update({A.FAMILIES['F'][0]:'77'}))
        with self.assertRaisesRegex(ValueError, 'numerical score'):
            self.audit()

    def test_failed_measurement_not_silently_excluded(self):
        path = self.roots['fhm']/'features/features.csv'
        self.alter_source_table(path, lambda r:r[0].update(extraction_status='failed'))
        with self.assertRaisesRegex(ValueError, 'Failed measurement'):
            self.audit()

    def test_missing_indicator_or_median_change_detected(self):
        models = A.strict_json(self.roots['old_results']/'fold_models.json')
        model = next(iter(models.values()))
        features = {name:np.asarray([1., np.nan, 9.]) for name in A.DESCRIPTORS}
        np.testing.assert_array_equal(A.independent_scores(features, model), [.5, .4, 1.5])
        changed = copy.deepcopy(model)
        changed['medians'][0] = 5.
        np.testing.assert_array_equal(A.independent_scores(features, changed), [.5, .9, 1.5])
        changed = copy.deepcopy(model)
        changed['coefficients_with_intercept'][len(model['columns'])+1] = 0.
        np.testing.assert_array_equal(A.independent_scores(features, changed), [.5, .5, 1.5])

    def test_symlink_publication_refused(self):
        path = self.scored/'model_index.csv'
        backup = self.root/'linked_index.csv'
        shutil.move(path, backup)
        path.symlink_to(backup)
        with self.assertRaisesRegex(ValueError, 'noncanonical regular file'):
            self.audit()

    def test_publication_hash_tamper_rejected(self):
        with (self.scored/'predictions.csv').open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            self.audit()

    def test_inputs_rechecked_before_publication(self):
        report, bindings = self.audit()
        path = self.roots['prepared']/'native_metadata_60s.csv'
        with path.open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            A.publish_report(report, bindings, self.root/'audit.json')
        self.assertFalse((self.root/'audit.json').exists())

    def test_real_mode_rejects_synthetic_receipt_and_paths(self):
        with self.assertRaisesRegex(ValueError, '500 rows'):
            A.audit(self.scored, self.receipt, self.roots, False)
        with self.assertRaisesRegex(ValueError, 'canonical remote'):
            A.source_data({}, self.roots, A.Bindings(), False)

    def test_real_code_hash_only_branch_does_not_parse_python(self):
        code = A.RC/'code/score_mureka60_frozen_v4.py'
        manifest = A.REAL_INPUTS['old_results']/'run_manifest.json'
        contract = {'input_files_sha256': {str(code):'1'*64, str(manifest):'2'*64}}
        bindings = A.Bindings()
        # Stop at the next legitimate JSON read: no real inputs or500-row bypass.
        with patch.object(bindings, 'file') as bind, patch.object(A, 'strict_json', side_effect=RuntimeError('stop_at_old_manifest')) as read, \
             patch.object(A, 'csv_rows', side_effect=AssertionError('Python source must not be parsed as CSV')):
            with self.assertRaisesRegex(RuntimeError, 'stop_at_old_manifest'):
                A.source_data(contract, A.REAL_INPUTS, bindings, False)
        self.assertEqual(bind.call_args_list[0].args, (code, '1'*64))
        read.assert_called_once_with(manifest)

    def test_no_scorer_import_or_fit_api(self):
        tree = ast.parse(Path(A.__file__).read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(any(name and ('score_mureka' in name or 'evaluate_' in name or 'prepare_evaluation' in name or 'audit_equal' in name) for name in imports))
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in ('fit', 'fit_transform', 'solve', 'lstsq', 'median', 'nanmedian') for node in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
