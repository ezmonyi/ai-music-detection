#!/usr/bin/env python3
"""Small handwritten synthetic replay only: no data acquisition, inference or fit."""
import ast
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import score_mureka60_frozen_v4 as S


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def fixture(root):
    root.mkdir()
    put(root/'synthetic_fixture.json', {'synthetic_test_only': True,
        'purpose': 'small_handwritten_replay_fixture_no_training'})
    meta = pd.DataFrame([dict(id=f'synthetic_mureka_v4_{i}', label='1', source_group='Mureka_v9',
        group_id=f'synthetic_group_{i//2}', role=S.ROLE, acquisition_role='reserved_unscored') for i in range(3)], columns=S.META)
    features = pd.DataFrame({'id': meta.id, **{c: ['0', '', '8'] for c in S.DESCRIPTORS}})
    meta.to_csv(root/'metadata.csv', index=False)
    features.to_csv(root/'features.csv', index=False)
    models, index = {}, []
    for combo in S.COMBINATIONS:
        cols = [c for family in combo.split('+') for c in S.P.COLUMNS[family]]
        n = len(cols)
        beta = [.5] + [0.]*(2*n)
        beta[1], beta[n+1] = .25, -.1
        model = dict(columns=cols, medians=[0.]*n, mean=[0.]*(2*n), scale=[1.]*(2*n),
            coefficients_with_intercept=beta, ridge=10., threshold=.5, feature_mode='values_plus_missing',
            training_rows=2, training_source_counts={'0:synthetic_human':1, '1:Suno':1},
            weighting=S.A.WEIGHTING, missing_value_policy=S.A.POLICIES['values_plus_missing'],
            observed_fraction_by_column={c:.5 for c in cols}, model_type='weighted_ridge_linear_probability', prediction_link='identity')
        key = S.digest({'model': model, 'threshold':.5})
        models[key] = model
        for cap in S.CAPS:
            for fold in range(5):
                index.append(dict(combination=combo, quantity=cap, fold_index=str(fold), fold_uid=f'synthetic_{cap}_{fold}',
                    model_sha256=key, candidate_key='synthetic_candidate_'+combo,
                    train_id_set_sha256=S.digest(['synthetic_old_train']), test_id_set_sha256=S.digest(['synthetic_old_test'])))
    put(root/'fold_models.json', models)
    frame = pd.DataFrame(index, columns=S.INDEX)
    frame.insert(0, 'fixture_row', [str(i) for i in range(len(frame))])
    frame.to_csv(root/'model_index.csv', index=False)
    return models, meta, features


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.fixture = cls.root/'fixture'
        cls.models, cls.meta, cls.raw = fixture(cls.fixture)
        cls.draft = cls.root/'draft'
        cls.score = cls.root/'score'
        cls.run_cli('draft', cls.draft)
        receipt = S.read_json(cls.draft/'scoring_draft.json')
        receipt.update(status='frozen', independent_review=dict(approved=True,
            reviewer='synthetic_test_root', reviewed_utc='2026-09-07T00:00:00Z'))
        cls.receipt = cls.root/'synthetic_frozen_receipt.json'
        put(cls.receipt, receipt)
        with patch.object(np.linalg, 'solve', side_effect=AssertionError('refit prohibited')), \
             patch.object(np, 'median', side_effect=AssertionError('new-source imputation prohibited')), \
             patch.object(S.A, 'audit', side_effect=AssertionError('synthetic cannot masquerade as real audit')):
            cls.run_cli('score', cls.score, cls.receipt)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_cli(cls, stage, output, receipt=None, fixture_dir=None):
        argv = ['--stage',stage,'--synthetic-test-only','--synthetic-fixture',str(fixture_dir or cls.fixture),'--output-dir',str(output)]
        if receipt is not None:
            argv.extend(['--receipt',str(receipt)])
        S.main(argv)

    def test_full_exact_grid_and_only_ai_endpoints(self):
        pred = S.A.read_csv(self.score/'predictions.csv')
        summary = S.A.read_csv(self.score/'per_model_sensitivity.csv')
        self.assertEqual(len(pred), 9525)
        self.assertEqual(len(summary), 3175)
        self.assertEqual(pred.id.nunique(), 3)
        self.assertEqual(pred.groupby('id').size().tolist(), [3175]*3)
        self.assertEqual(summary.groupby('combination').size().tolist(), [25]*127)
        self.assertTrue(summary.tp.eq('2').all() and summary.fn.eq('1').all())
        self.assertTrue(summary.unique_groups.eq('2').all())
        self.assertTrue(np.allclose(summary.ai_sensitivity.astype(float), 2/3))
        self.assertFalse(set(summary).intersection(['balanced_accuracy','roc_auc','source_transfer_J','rank','winner']))
        manifest = S.read_json(self.score/'publication_manifest.json')
        self.assertEqual(manifest['prediction_rows'], 9525)
        for name, record in manifest['files'].items():
            self.assertEqual(S.P.sha(self.score/name), record['sha256'])
        self.assertTrue(manifest['synthetic_test_only'])
        self.assertFalse(manifest['classifier_fitted'])

    def test_known_replay_missing_values_tie_and_identity_link(self):
        model = next(iter(self.models.values()))
        features = S.check_identity(self.meta, self.raw, True)
        with patch.object(np, 'median', side_effect=AssertionError('no reimputation')):
            scores = S.replay(features, model)
        np.testing.assert_allclose(scores, [.5,.4,2.5], rtol=0, atol=1e-15)
        self.assertEqual((scores >= .5).tolist(), [True,False,True])
        changed = features.copy(); changed.iloc[2] = 1000
        np.testing.assert_allclose(S.replay(changed, model)[:2], scores[:2])
        self.assertTrue(features.iloc[1].isna().all())
        # Independently construct the frozen original matrix, not a call to replay.
        x = features[model['columns']].to_numpy(float)
        mask = ~np.isfinite(x)
        matrix = np.column_stack([np.ones(len(x)),
            (np.hstack([np.where(mask,np.asarray(model['medians']),x),mask.astype(float)])
             -np.asarray(model['mean']))/np.asarray(model['scale'])])
        np.testing.assert_array_equal(scores, matrix @ np.asarray(model['coefficients_with_intercept']))

    def test_draft_does_not_score_or_freeze(self):
        self.assertFalse((self.draft/'predictions.csv').exists())
        self.assertEqual(S.read_json(self.draft/'scoring_draft.json')['status'], 'draft')
        with self.assertRaisesRegex(ValueError, 'draft|independently'):
            self.run_cli('score', self.root/'draft_refused', self.draft/'scoring_draft.json')

    def test_no_fitting_or_inference_imports(self):
        tree = ast.parse(Path(S.__file__).read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [v.name for n in ast.walk(tree) if isinstance(n, ast.Import) for v in n.names]
        self.assertFalse(any(any(x in (name or '') for x in ('evaluate_new_phenomena','torch','soundfile','run_mureka','sklearn')) for name in imports))
        forbidden = {'fit','fit_candidate','lstsq','solve','median','nanmedian','quantile'}
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree)))

    def test_real_mode_refuses_synthetic_and_missing_inputs(self):
        for args in [[], ['--synthetic-fixture',str(self.fixture)]]:
            with self.assertRaisesRegex(ValueError, 'Real mode|Synthetic fixture refused'):
                S.main(['--stage','draft','--output-dir',str(self.root/'real_refused'), *args])
        with self.assertRaisesRegex(ValueError, 'exactly500'):
            S.check_identity(self.meta, self.raw, False)

    def test_real_mode_refuses_synthetic_old_report(self):
        from types import SimpleNamespace
        path = self.root/'synthetic_old_audit.json'
        put(path, dict(status='passed', synthetic_test_only=True, rows=1604, candidates=127,
                      caps=[25,50,100,200,'all'], pooled_metrics=3175, model_fitting_performed=False))
        with self.assertRaisesRegex(ValueError, 'Real independently passed'):
            S.load_old(SimpleNamespace(old_audit=path), S.Bindings())

    def test_exact_model_grid_missing_duplicate_and_changed_cap(self):
        index = S.A.read_csv(self.fixture/'model_index.csv').drop(columns='fixture_row')
        for frame in [index.iloc[:-1], pd.concat([index.iloc[:-1], index.iloc[:1]]), index.assign(quantity='all')]:
            with self.subTest(rows=len(frame)), self.assertRaisesRegex(ValueError, 'Exact127x5x5'):
                S.validate_index(frame, self.models)

    def test_model_tampering_and_metadata_descriptor_refused(self):
        index = S.A.read_csv(self.fixture/'model_index.csv').drop(columns='fixture_row')
        models = copy.deepcopy(self.models)
        models[index.model_sha256.iloc[0]]['coefficients_with_intercept'][0] = .9
        with self.assertRaisesRegex(ValueError, 'Model hash'):
            S.validate_index(index, models)
        model = copy.deepcopy(next(iter(self.models.values())))
        model['columns'][0] = 'duration_sec'
        with self.assertRaisesRegex(ValueError, 'descriptor'):
            S.validate_model(model, 'S')
        model = copy.deepcopy(next(iter(self.models.values())))
        model['training_source_counts'] = {'1:Mureka_v9':2}
        with self.assertRaisesRegex(ValueError, 'leakage'):
            S.validate_model(model, 'S')

    def test_measurement_identity_missing_duplicates_and_changed_roles(self):
        for metadata, features in [(self.meta.iloc[:-1],self.raw),
                                   (pd.concat([self.meta.iloc[:-1],self.meta.iloc[:1]]),self.raw),
                                   (self.meta.assign(role='development'),self.raw),
                                   (self.meta.assign(acquisition_role='development'),self.raw),
                                   (self.meta,self.raw.assign(duration_sec='60'))]:
            with self.subTest(rows=len(metadata)), self.assertRaises(ValueError):
                S.check_identity(metadata, features, True)

    def test_tamper_after_freeze_and_new_path_only(self):
        receipt = S.read_json(self.receipt)
        changed = copy.deepcopy(receipt['contract']); changed['threshold'] = .4
        with self.assertRaisesRegex(ValueError, 'stale'):
            S.authorize(changed, self.receipt)
        with self.assertRaisesRegex(ValueError, 'existing output'):
            self.run_cli('score', self.score, self.receipt)
        binding = S.Bindings()
        p = self.root/'tamper.txt'; p.write_text('original')
        binding.file(p); p.write_text('modified')
        with self.assertRaisesRegex(ValueError, 'changed'):
            binding.recheck()

    def test_review_and_synthetic_mode_must_match(self):
        receipt = S.read_json(self.receipt)
        receipt['independent_review'] = None
        path = self.root/'unreviewed.json'; put(path, receipt)
        with self.assertRaisesRegex(ValueError, 'Independent root review'):
            S.authorize(receipt['contract'], path)
        changed = copy.deepcopy(receipt['contract']); changed['synthetic_test_only'] = False
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            S.authorize(changed, self.receipt)

    def test_invalid_descriptor_and_duplicate_json(self):
        for bad in ('inf','-inf','not_numeric'):
            frame = self.raw.copy(); frame.loc[0,S.DESCRIPTORS[0]] = bad
            with self.assertRaises(ValueError):
                S.check_identity(self.meta,frame,True)
        path = self.root/'duplicate.json'; path.write_text('{"id":1,"id":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            S.read_json(path)

    def test_atomic_publication_refuses_even_empty_existing_directory(self):
        source, destination = self.root/'atomic_source', self.root/'atomic_existing'
        source.mkdir(); destination.mkdir()
        (source/'marker').write_text('preserve')
        with self.assertRaises(OSError):
            S.publish_directory(source, destination)
        self.assertTrue((source/'marker').exists())
        self.assertEqual(list(destination.iterdir()), [])

    def test_output_inventory_rejects_duplicate_missing_role_and_threshold_tampering(self):
        root = self.root/'output_tamper'
        shutil.copytree(self.score, root)
        path = root/'predictions.csv'
        original = S.A.read_csv(path)
        contract = S.read_json(self.receipt)['contract']
        index = S.A.read_csv(root/'model_index.csv')
        cases = [original.iloc[:-1],
                 pd.concat([original.iloc[[0]], original.iloc[0:-1]], ignore_index=True),
                 original.assign(role='development'), original.assign(predicted_label='0')]
        for frame in cases:
            frame.to_csv(path,index=False)
            with self.subTest(rows=len(frame)), self.assertRaises(ValueError):
                S.audit_output(root,contract,index,self.meta)

    def test_adjacent_threshold_values_roundtrip_without_changing_side(self):
        expected = np.array([np.nextafter(.5,0),.5,np.nextafter(.5,1)])
        model = copy.deepcopy(next(iter(self.models.values())))
        model['coefficients_with_intercept'] = [0.] * len(model['coefficients_with_intercept'])
        model['coefficients_with_intercept'][1] = 1.
        features = S.check_identity(self.meta,self.raw,True)
        features.loc[:,model['columns'][0]] = expected
        np.testing.assert_array_equal(S.replay(features,model),expected)
        root = self.root/'threshold_roundtrip'; shutil.copytree(self.score,root)
        frame = S.A.read_csv(root/'predictions.csv')
        frame.loc[:2,'score'] = [repr(float(v)) for v in expected]
        frame.loc[:2,'predicted_label'] = ['0','1','1']
        frame.to_csv(root/'predictions.csv',index=False)
        S.audit_output(root,S.read_json(self.receipt)['contract'],S.A.read_csv(root/'model_index.csv'),self.meta)

    def test_explicit_v3_version_and_release_proof_gates(self):
        run = dict(schema_version=3, allinone_gpus=[6], beats_gpus=[7], original_gpu_release_proof=None,
                   stage_gpu_assignment={s:{f'{i:02d}':g for i in range(21)} for s,g in [('allinone',6),('beats',7)]})
        audit = dict(original_gpu_release_proof=None, explicit_stage_gpu_assignment_verified=True)
        receipt = dict(before_after_all_provenance_inputs_products_dependencies_release_proof_verified=True)
        self.assertEqual(S.validate_measurement_version('v3',run,audit,receipt,S.Bindings(),[]),run['stage_gpu_assignment'])
        with self.assertRaisesRegex(ValueError,'version/schema'):
            S.validate_measurement_version('v2',run,audit,receipt,S.Bindings(),[])
        wrong = copy.deepcopy(run); wrong['allinone_gpus']=[0]
        wrong['stage_gpu_assignment']['allinone']={f'{i:02d}':0 for i in range(21)}
        with self.assertRaisesRegex(ValueError,'lack release'):
            S.validate_measurement_version('v3',wrong,audit,receipt,S.Bindings(),[])
        wrong = copy.deepcopy(run); wrong['stage_gpu_assignment']['allinone']['00']=7
        with self.assertRaisesRegex(ValueError,'assignment audit'):
            S.validate_measurement_version('v3',wrong,audit,receipt,S.Bindings(),[])
        wrong_proof = dict(schema_version=1,status='verified_full134_original1604',rows=1604,stage_shards=134,
                           fixed_prepared_root='/synthetic/not_the_original',fixed_inference_root='/synthetic/inference')
        wrong_proof['canonical_sha256']=S.digest(wrong_proof)
        with self.assertRaisesRegex(ValueError,'fixed original1604'):
            S.validate_release_proof(wrong_proof,S.Bindings(),[])

    def test_v2_route_preserved_and_v3_flag_cannot_substitute(self):
        run = dict(schema_version=1, allinone_gpus=[6],beats_gpus=[7])
        receipt = dict(before_after_all_provenance_inputs_products_dependencies_verified=True)
        S.validate_measurement_version('v2',run,{},receipt,S.Bindings(),[])
        run['original_gpu_release_proof']=None
        with self.assertRaisesRegex(ValueError,'cannot enter v2'):
            S.validate_measurement_version('v2',run,{},receipt,S.Bindings(),[])


if __name__ == '__main__':
    unittest.main()
