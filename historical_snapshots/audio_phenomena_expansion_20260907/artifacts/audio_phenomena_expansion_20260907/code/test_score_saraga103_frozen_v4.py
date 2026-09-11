#!/usr/bin/env python3
"""Handwritten, isolated synthetic fixtures. No real Human scoring/admission."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import score_saraga103_frozen_v4 as S


def put(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def fixture(root):
    root.mkdir()
    put(root/'synthetic_fixture.json',{'synthetic_test_only':True,'purpose':'small_handwritten_saraga_replay_fixture_no_training'})
    meta=pd.DataFrame([dict(id=f'synthetic_saraga_v4_{i}',label='0',source_group=S.SOURCE,
        group_id=f'synthetic_group_{max(0,i-2)}',role=S.ROLE,evaluation_allowed='False',classifier_admission_authorized='False')
        for i in range(7)],columns=S.META)
    raw=pd.DataFrame({'id':meta.id,**{c:['0','','8','-8','0','','8'] for c in S.DESCRIPTORS}})
    meta.to_csv(root/'metadata.csv',index=False); raw.to_csv(root/'features.csv',index=False)
    models,index={},[]
    for combo in S.COMBINATIONS:
        cols=[c for f in combo.split('+') for c in S.P.COLUMNS[f]]; n=len(cols)
        beta=[.5]+[0.]*(2*n); beta[1]=.25; beta[n+1]=-.1
        model=dict(columns=cols,medians=[0.]*n,mean=[0.]*(2*n),scale=[1.]*(2*n),coefficients_with_intercept=beta,
            ridge=10.,threshold=.5,feature_mode='values_plus_missing',training_rows=2,
            training_source_counts={'0:synthetic_old_human':1,'1:Suno':1},weighting=S.A.WEIGHTING,
            missing_value_policy=S.A.POLICIES['values_plus_missing'],observed_fraction_by_column={c:.5 for c in cols},
            model_type='weighted_ridge_linear_probability',prediction_link='identity')
        key=S.digest({'model':model,'threshold':.5}); models[key]=model
        for cap in S.CAPS:
            for fold in range(5):
                index.append(dict(combination=combo,quantity=cap,fold_index=str(fold),fold_uid=f'synthetic_{cap}_{fold}',
                    model_sha256=key,candidate_key='synthetic_candidate_'+combo,
                    train_id_set_sha256=S.digest(['synthetic_old_train']),test_id_set_sha256=S.digest(['synthetic_old_test'])))
    put(root/'fold_models.json',models)
    index=pd.DataFrame(index,columns=S.INDEX); index.insert(0,'fixture_row',list(map(str,range(len(index)))))
    index.to_csv(root/'model_index.csv',index=False)
    return models,meta,raw


class SaragaTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(); cls.root=Path(cls.temporary.name).resolve()
        cls.fixture=cls.root/'fixture'; cls.models,cls.meta,cls.raw=fixture(cls.fixture)
        cls.draft,cls.scored=cls.root/'draft',cls.root/'scored'
        cls.run_cli('draft',cls.draft)
        receipt=S.read_json(cls.draft/'scoring_draft.json')
        receipt.update(status='frozen',independent_review=dict(approved=True,reviewer='root',reviewed_utc='synthetic_test_time'))
        cls.receipt=cls.root/'synthetic_only_frozen.json'; put(cls.receipt,receipt)
        with patch.object(np.linalg,'solve',side_effect=AssertionError('fit forbidden')), \
             patch.object(np,'median',side_effect=AssertionError('imputation fit forbidden')), \
             patch.object(S.M,'load_old',side_effect=AssertionError('real audit forbidden in fixture')), \
             patch.object(S,'validate_measurements',side_effect=AssertionError('real Human admission forbidden')):
            cls.run_cli('score',cls.scored,cls.receipt)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def run_cli(cls,stage,output,receipt=None):
        argv=['--stage',stage,'--synthetic-test-only','--synthetic-fixture',str(cls.fixture),'--output-dir',str(output)]
        if receipt is not None: argv += ['--receipt',str(receipt)]
        S.main(argv)

    def test_full_grid_predictions_and_integer_counts(self):
        pred=S.A.read_csv(self.scored/'predictions.csv'); summary=S.A.read_csv(self.scored/'per_model_specificity.csv')
        self.assertEqual(len(pred),3175*7); self.assertEqual(len(summary),3175)
        self.assertTrue(pred.groupby('id').size().eq(3175).all())
        self.assertTrue(summary.fp.eq('4').all() and summary.tn.eq('3').all())
        self.assertTrue(np.allclose(summary.false_positive_rate.astype(float),4/7))
        self.assertTrue(np.allclose(summary.specificity.astype(float),3/7))
        self.assertTrue(pred.label.eq('0').all() and pred.role.eq(S.ROLE).all())
        self.assertTrue(pred.evaluation_allowed.eq('False').all() and pred.classifier_admission_authorized.eq('False').all())

    def test_five_component_rates_differ_from_record_weighting(self):
        comp=S.A.read_csv(self.scored/'per_component_specificity.csv'); summary=S.A.read_csv(self.scored/'per_model_specificity.csv')
        self.assertEqual(len(comp),3175*5)
        self.assertTrue(comp.groupby(S.INDEX,dropna=False).fp.apply(lambda x:x.astype(int).sum()).eq(4).all())
        self.assertTrue(comp[comp.group_id.eq('synthetic_group_0')].rows.eq('3').all())
        self.assertTrue(np.allclose(summary.equal_component_false_positive_rate.astype(float),8/15))
        self.assertFalse(np.allclose(summary.equal_component_false_positive_rate.astype(float),4/7))

    def test_635_cells_and_fixed_overview(self):
        cells=S.A.read_csv(self.scored/'all635_cells.csv'); overview=S.A.read_csv(self.scored/'predefined_overview.csv')
        self.assertEqual(len(cells),635); self.assertEqual(len(overview),60)
        self.assertEqual(set(overview.combination),set(S.OVERVIEW))
        self.assertTrue(cells.fold_models.eq('5').all())
        for suffix in ('mean','min','max'):
            np.testing.assert_allclose(cells['false_positive_rate_'+suffix].astype(float),4/7)
        self.assertFalse(any(c in cells for c in ('balanced_accuracy','roc_auc','rank','winner','two_class_accuracy')))

    def test_raw_identity_scores_tie_missing_and_no_reimputation(self):
        model=next(iter(self.models.values())); x=S.check_identity(self.meta,self.raw,True)
        scores=S.replay(x,model)
        np.testing.assert_allclose(scores,[.5,.4,2.5,-1.5,.5,.4,2.5],rtol=0,atol=1e-15)
        self.assertEqual((scores>=.5).tolist(),[True,False,True,False,True,False,True])
        values=x[model['columns']].to_numpy(float); missing=~np.isfinite(values)
        independent=np.column_stack([np.ones(len(values)),(np.hstack([np.where(missing,model['medians'],values),missing.astype(float)])-model['mean'])/model['scale']])
        np.testing.assert_array_equal(scores,independent@model['coefficients_with_intercept'])
        changed=x.copy(); changed.iloc[2]=999
        np.testing.assert_array_equal(S.replay(changed,model)[:2],scores[:2])

    def test_draft_has_no_scores_and_cannot_authorize(self):
        self.assertFalse((self.draft/'predictions.csv').exists())
        self.assertEqual(S.read_json(self.draft/'scoring_draft.json')['status'],'draft')
        with self.assertRaisesRegex(ValueError,'draft'):
            self.run_cli('score',self.root/'draft_refused',self.draft/'scoring_draft.json')

    def test_real_mode_refuses_fixtures_and_missing_arguments(self):
        for extra in ([],['--synthetic-fixture',str(self.fixture)]):
            with self.assertRaises(ValueError):
                S.main(['--stage','draft','--output-dir',str(self.root/'real_refused'),*extra])
        with self.assertRaisesRegex(ValueError,'exactly103'):
            S.check_identity(self.meta,self.raw,False)

    def test_wrong_human_role_label_source_and_admission(self):
        for key,value in [('role','development'),('label','1'),('source_group','Mureka_v9'),
                          ('evaluation_allowed','True'),('classifier_admission_authorized','True')]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Human'):
                S.check_identity(self.meta.assign(**{key:value}),self.raw,True)

    def test_synthetic_ids_cannot_alias_real_identities(self):
        meta=self.meta.copy(); raw=self.raw.copy(); meta.loc[0,'id']=raw.loc[0,'id']='saraga_hindustani_fake'
        with self.assertRaisesRegex(ValueError,'Synthetic'): S.check_identity(meta,raw,True)
        with self.assertRaisesRegex(ValueError,'Synthetic'): S.check_identity(self.meta.assign(group_id='real_group'),self.raw,True)

    def test_schema_duplicate_reorder_and_infinity(self):
        for meta,raw in [(self.meta.iloc[:-1],self.raw),(self.meta,self.raw.iloc[::-1]),
            (self.meta,self.raw.assign(duration='60')),(self.meta.assign(id='duplicate'),self.raw.assign(id='duplicate'))]:
            with self.assertRaises(ValueError): S.check_identity(meta,raw,True)
        raw=self.raw.copy(); raw.loc[0,S.DESCRIPTORS[0]]='inf'
        with self.assertRaisesRegex(ValueError,'Nonfinite'): S.check_identity(self.meta,raw,True)

    def test_frozen_grid_missing_duplicate_and_cap(self):
        idx=S.A.read_csv(self.fixture/'model_index.csv').drop(columns='fixture_row')
        for frame in (idx.iloc[:-1],pd.concat([idx.iloc[:-1],idx.iloc[:1]]),idx.assign(quantity='all')):
            with self.assertRaisesRegex(ValueError,'Exact127x5x5'): S.validate_index(frame,self.models)

    def test_saraga_training_leakage_and_model_tamper(self):
        model=copy.deepcopy(next(iter(self.models.values())))
        model['training_source_counts']={'0:human_saraga_hindustani_v1':2}
        with self.assertRaisesRegex(ValueError,'Saraga training'): S.validate_model(model,'S')
        idx=S.A.read_csv(self.fixture/'model_index.csv').drop(columns='fixture_row'); models=copy.deepcopy(self.models)
        models[idx.model_sha256.iloc[0]]['threshold']=.4
        with self.assertRaises(ValueError): S.validate_index(idx,models)

    def test_frozen_transform_restrictions(self):
        original=next(iter(self.models.values()))
        for key,value in [('threshold',.4),('prediction_link','sigmoid'),('feature_mode','values'),('scale',[0.]*len(original['scale']))]:
            model=copy.deepcopy(original); model[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): S.validate_model(model,'S')

    def test_nontrivial_frozen_medians_means_scales(self):
        model=copy.deepcopy(next(iter(self.models.values()))); n=len(model['columns'])
        model.update(medians=[2.]*n,mean=[.1]*(2*n),scale=[2.]*(2*n))
        scores=S.replay(S.check_identity(self.meta,self.raw,True),model)
        np.testing.assert_allclose(scores[:3],[.4925,.6925,1.4925],rtol=0,atol=1e-15)

    def test_cell_ranges_follow_five_different_frozen_models(self):
        idx=S.A.read_csv(self.fixture/'model_index.csv').drop(columns='fixture_row').iloc[:5].copy()
        models={}
        for i in range(5):
            model=copy.deepcopy(next(iter(self.models.values()))); model['coefficients_with_intercept'][0]+=.2*i
            key=S.digest({'model':model,'threshold':.5}); models[key]=model; idx.loc[i,'model_sha256']=key
        directory=self.root/'different_model_ranges'; directory.mkdir()
        contract=S.read_json(self.receipt)['contract']
        S.write_scores(directory,contract,models,idx,self.meta,S.check_identity(self.meta,self.raw,True))
        cell=S.A.read_csv(directory/'all635_cells.csv').iloc[0]
        self.assertAlmostEqual(float(cell.false_positive_rate_min),4/7)
        self.assertAlmostEqual(float(cell.false_positive_rate_max),6/7)
        self.assertAlmostEqual(float(cell.false_positive_rate_mean),.8)

    def test_wrong_review_stale_contract_and_mode(self):
        frozen=S.read_json(self.receipt)
        for key,value in [('threshold',.4),('synthetic_test_only',False),('authorized_stage',S.M.STAGE)]:
            contract=dict(frozen['contract'],**{key:value})
            with self.assertRaises(ValueError): S.authorize(contract,self.receipt)
        frozen['independent_review']['reviewer']='not_root'; path=self.root/'bad_review.json'; put(path,frozen)
        with self.assertRaisesRegex(ValueError,'root'): S.authorize(frozen['contract'],path)

    def test_publication_and_all_hashes_verified(self):
        marker=S.verify_publication(self.scored)
        self.assertIn('predictions.csv',marker['files'])
        self.assertEqual(S.read_json(self.scored/'publication_manifest.json')['prediction_rows'],22225)
        self.assertFalse(S.read_json(self.scored/'output_accounting.json')['independent_numerical_audit_performed'])

    def test_publication_never_overwrites_existing(self):
        with self.assertRaisesRegex(ValueError,'existing output'): self.run_cli('score',self.scored,self.receipt)
        src=self.root/'publish_src'; dst=self.root/'publish_dst'; src.mkdir(); dst.mkdir(); (src/'value').write_text('new'); (dst/'value').write_text('old')
        with self.assertRaises(FileExistsError): S.publish_directory(src,dst)
        self.assertEqual((dst/'value').read_text(),'old')

    def test_publication_link_failure_leaves_uncommitted_orphan(self):
        src=self.root/'fail_src'; dst=self.root/'fail_dst'; src.mkdir(); (src/'value').write_text('new')
        with patch.object(S.os,'link',side_effect=OSError('NFS link unavailable')):
            with self.assertRaises(OSError): S.publish_directory(src,dst)
        self.assertFalse((dst/'COMMIT.json').exists())
        with self.assertRaises(FileNotFoundError): S.verify_publication(dst)
        with self.assertRaises(FileExistsError): S.publish_directory(src,dst)

    def test_bound_file_change_and_symlink_rejected(self):
        path=self.root/'bound'; path.write_text('a'); bindings=S.Bindings(); bindings.file(path); path.write_text('b')
        with self.assertRaisesRegex(ValueError,'changed'): bindings.recheck()
        link=self.root/'symlink'; link.symlink_to(path)
        with self.assertRaises(ValueError): S.Bindings().file(link)

    def test_measurement_gate_never_accepts_generator_or_unpassed_flags(self):
        good=dict(purpose='measurement_only',role=S.ROLE,classifier_fitted=False,scores_generated=False,
            classifier_admission_authorized=False,evaluation_allowed=False)
        S.measurement_scope(good)
        for key,value in [('role',S.M.ROLE),('scores_generated',True),('evaluation_allowed',True)]:
            with self.assertRaises(ValueError): S.measurement_scope(dict(good,**{key:value}))

    def test_actual_materializer_wav_schema_and_hash_distinction(self):
        # Field names copied from the accepted Saraga proof, values deliberately synthetic.
        proof={'wav':{'bytes':8,'file_sha256':'container_hash','float32_sha256':'pcm_hash','readback_verified':True},
               'interval_proof':{'standardized_float32_sha256':'pcm_hash','raw_hashes_after':{'sha256':'native_mp3_hash'}}}
        S.validate_wav_proof(proof)
        self.assertNotEqual(proof['wav']['file_sha256'],proof['interval_proof']['raw_hashes_after']['sha256'])
        wrong=copy.deepcopy(proof); wrong['wav']['float32_sha256']='container_hash'
        with self.assertRaises(ValueError): S.validate_wav_proof(wrong)
        wrong=copy.deepcopy(proof); wrong['wav']['waveform_sha256']=wrong['wav'].pop('float32_sha256')
        with self.assertRaises(KeyError): S.validate_wav_proof(wrong)

    def test_actual_alias_layout_uses_direct_inputs_hardlink(self):
        root=self.root/'alias_schema'; (root/'inputs').mkdir(parents=True)
        original=root/'synthetic_source.wav'; original.write_bytes(b'synthetic bytes, not audio')
        alias=root/'inputs/synthetic_saraga_v4_schema.wav'; os.link(original,alias)
        row=dict(item_id='synthetic_saraga_v4_schema',standardized_path=str(alias),
            accepted_standardized_path=str(original),standardized_file_sha256=S.P.sha(original))
        source=dict(standardized_path=str(original),standardized_file_sha256=S.P.sha(original))
        S.validate_alias(row,source,root)
        (root/'inputs/audio').mkdir(); nested=root/'inputs/audio'/alias.name; os.link(original,nested)
        with self.assertRaises(ValueError): S.validate_alias(dict(row,standardized_path=str(nested)),source,root)
        alias.unlink(); alias.write_bytes(original.read_bytes())
        with self.assertRaises(ValueError): S.validate_alias(row,source,root)

    def test_no_mutated_globals_inference_or_fitting_imports(self):
        self.assertEqual(S.M.ROLE,'external_generator_unscored')
        tree=ast.parse(Path(S.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(any(x in (n or '') for x in ('torch','soundfile','run_saraga','sklearn','evaluate_new')) for n in imports))
        forbidden={'fit','solve','lstsq','median','nanmedian','quantile','decode','subprocess'}
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree)))


if __name__=='__main__':
    unittest.main()
