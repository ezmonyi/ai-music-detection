"""Generated data only; no real package plan, fitting, scoring or GPU use."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluate_new_phenomena_v5 as E


def put(path,value): path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def group_for(source,bucket,ordinal=0):
    for i in range(10000):
        group=f'synthetic_v5_group_{source}_{ordinal}_{i}'
        if E.BASE.hash_fold(group,5,E.SEED)==bucket:return group
    raise AssertionError('Synthetic hash bucket not found')


def synthetic_table(groups_per_source=4):
    records=[]
    for source,label in [('MTG-Jamendo',0),('Suno',1),('Mureka_v9',1),('human_saraga_hindustani_v1',0)]:
        for i in range(groups_per_source):
            iid=f'synthetic_v5_{source}_{i}'
            records.append(dict(__id=iid,__label=label,__source=source,__group=group_for(source,i%4,i),__role='development',
                **{c:float('nan') if (i+j)%3==0 else float(i-label) for j,c in enumerate(E.DESCRIPTORS)}))
    return pd.DataFrame(records)


def package_fixture(root):
    fixture=root/'source';fixture.mkdir()
    put(fixture/'synthetic_fixture.json',{'synthetic_test_only':True,'purpose':'derived_v5_package_fixture_no_real_admission'})
    table=synthetic_table()
    for kind,sources,role in [('old',{'MTG-Jamendo','Suno'},'development'),('mureka',{'Mureka_v9'},'external_generator_unscored'),
                              ('saraga',{'human_saraga_hindustani_v1'},'external_human_unscored')]:
        metadata=[];features=[]
        for row in table[table['__source'].isin(sources)].to_dict('records'):
            metadata.append(dict(id=row['__id'],label=str(row['__label']),source_group=row['__source'],group_id=row['__group'],role=role,
                duration_view='60s',native_sample_rate_hz='44100'))
            features.append(dict(id=row['__id'],**{c:'' if np.isnan(row[c]) else str(row[c]) for c in E.DESCRIPTORS}))
        write_csv(fixture/(kind+'_metadata.csv'),metadata);write_csv(fixture/(kind+'_features.csv'),features)
    package=root/'package'
    E.PREP.prepare({'synthetic_fixture':str(fixture)},package,True)
    return package


class ScheduleTests(unittest.TestCase):
    def setUp(self):self.table=synthetic_table()

    def test_exact_base_schedule_roles_source_and_group_exclusion(self):
        schedule,omitted,count=E.make_schedule(self.table,True)
        base=E.BASE.make_folds(self.table,5,E.SEED)
        self.assertEqual(len(schedule),len(base)*5)
        self.assertEqual(count['primary_fits'],len(base)*5*127)
        self.assertEqual(count['diagnostic_fits'],len(base)*2*127)
        for record,train,test,fold in schedule:
            self.assertFalse(set(train['__id'])&set(test['__id']))
            self.assertFalse(set(train['__group'])&set(test['__group']))
            if fold['fold_type']!='ordinary_group_holdout_descriptive':self.assertNotIn(fold['heldout_source'],set(train['__source']))
            self.assertEqual(record['test']['ids'],sorted(test['__id']))
        self.assertTrue(omitted)
        self.assertTrue(all(r['opposite_group_fold']==4 for r in omitted))
        with self.assertRaisesRegex(ValueError,'Role leakage'):E.make_schedule(self.table.assign(__role='locked'),True)

    def test_omissions_not_filled_and_real_counts_cannot_accept_synthetic(self):
        schedule,omitted,_=E.make_schedule(self.table,True)
        self.assertEqual(len(omitted),5)  # Four source arms and ordinary bucket4.
        self.assertEqual(len({(r['fold_type'],r['heldout_source'],r['opposite_group_fold']) for r in omitted}),5)
        with self.assertRaisesRegex(ValueError,'Real schedule differs'):E.make_schedule(self.table,False)

    def test_caps_nested_by_source_keep_all_selected_group_rows(self):
        table=synthetic_table(52)
        duplicate=table.iloc[[0]].assign(__id='synthetic_v5_duplicate_component_recording')
        table=pd.concat([table,duplicate],ignore_index=True)
        previous=set()
        for cap in E.QUANTITIES:
            selected=E.BASE.deterministic_quantity(table,cap,E.SEED)
            self.assertTrue(previous<=set(selected['__id']));previous=set(selected['__id'])
            if cap!='all':self.assertTrue(selected.groupby('__source')['__group'].nunique().le(cap).all())
            for group in set(selected['__group']):
                self.assertEqual(set(selected.loc[selected['__group']==group,'__id']),set(table.loc[table['__group']==group,'__id']))
        self.assertEqual(previous,set(table['__id']))
        E.make_schedule(table,True)  # Runtime nesting guard also exercises actual cap truncation.

    def test_exact_ids_shared_across_caps_and_all127_candidates(self):
        schedule,_,_=E.make_schedule(self.table,True)
        for i in range(0,len(schedule),5):
            self.assertEqual(len({r['test_id_set_sha256'] for r,*_ in schedule[i:i+5]}),1)
        self.assertEqual(len(E.COMBINATIONS),127)
        self.assertEqual(len(set(E.COMBINATIONS)),127)
        self.assertEqual(sum(len(c) for c in E.FAMILIES.values()),54)
        self.assertEqual(E.MODES,('values_plus_missing','median_only','missingness_only'))


class FitTests(unittest.TestCase):
    def setUp(self):
        self.table=synthetic_table()
        self.schedule,_,_=E.make_schedule(self.table,True)
        self.record,self.train,self.test,self.fold=next(x for x in self.schedule if x[0]['quantity']=='all')

    def test_train_only_transform_and_fixed_threshold_raw_link(self):
        model,scores,_=E.model_result(self.train,self.test,self.fold,self.record,'D','values_plus_missing',1)
        changed=self.test.copy();changed[E.DESCRIPTORS]=999999
        second,_,_=E.model_result(self.train,changed,self.fold,self.record,'D','values_plus_missing',1)
        self.assertEqual(model,second)
        for j,column in enumerate(E.FAMILIES['D']):
            finite=self.train[column].to_numpy(float);finite=finite[np.isfinite(finite)]
            self.assertEqual(model['medians'][j],float(np.median(finite)) if len(finite) else 0.)
        self.assertEqual(model['threshold'],.5);self.assertEqual(model['ridge'],10.)
        self.assertEqual(model['prediction_link'],'identity')
        np.testing.assert_array_equal(scores,E.BASE.predict(self.test,model))

    def test_missing_modes_dimensions_and_all_missing_zero_fallback(self):
        train=self.train.copy();column=E.FAMILIES['D'][0];train[column]=np.nan
        for mode in E.MODES:
            model,_,_=E.model_result(train,self.test,self.fold,self.record,'D',mode,1)
            self.assertEqual(model['medians'][0],0.)
            width=3*(2 if mode=='values_plus_missing' else 1)
            self.assertEqual(len(model['mean']),width);self.assertEqual(len(model['coefficients_with_intercept']),width+1)
            self.assertEqual(model['missing_value_policy'],E.POLICIES[mode])

    def test_weight_sum_n_balanced_sources_groups_and_ridge_confounded(self):
        table=pd.concat([self.train,self.train.iloc[[0]].assign(__id='synthetic_extra')],ignore_index=True)
        weights=E.BASE.sample_weights(table)
        self.assertAlmostEqual(weights.sum(),len(table))
        for label in (0,1):self.assertAlmostEqual(weights[table['__label'].eq(label)].sum(),len(table)/2)
        for source,part in table.groupby('__source'):
            group_mass=[weights[table['__group'].eq(g)].sum() for g in part['__group'].unique()]
            np.testing.assert_allclose(group_mass,np.full(len(group_mass),group_mass[0]))
        self.assertEqual(E.BASE.RIDGE,10.)

    def test_recording_vs_component_source_denominators(self):
        table=pd.DataFrame([dict(__id=str(i),__label=0,__source='human',__group='large' if i<3 else 'small') for i in range(4)])
        row=E.source_endpoints(table,np.array([1.,1.,1.,0.]),'synthetic_model')[0]
        self.assertEqual(row['rows'],4);self.assertEqual(row['components'],2)
        self.assertEqual(row['recording_human_specificity'],.25)
        self.assertEqual(row['equal_component_human_specificity'],.5)
        self.assertEqual(row['fp'],3);self.assertEqual(row['tn'],1)
        self.assertIsNone(row['recording_ai_sensitivity'])

    def test_streamed_three_mode_unit_saves_every_model_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp).resolve()
            contract=dict(schedule=[self.record],accounting=dict(primary_fits=1,primary_prediction_rows=len(self.test),
                diagnostic_fits=2,diagnostic_prediction_rows=2*len(self.test)))
            auth=dict(status='frozen_verified',contract_sha256=E.digest(contract))
            # One candidate/one fold synthetic unit; production combinations are not configurable.
            with patch.object(E,'COMBINATIONS',['D']),patch.object(E,'recheck'), \
                 patch.object(E.V2,'choose_train_threshold',side_effect=AssertionError('threshold selection forbidden')):
                result=E.run_evaluation(output,self.table,[(self.record,self.train,self.test,self.fold)],contract,auth)
            self.assertEqual(result['model_instances'],3)
            models=[json.loads(line) for line in (output/'fold_models.jsonl').read_text().splitlines()]
            self.assertEqual({r['feature_mode'] for r in models},set(E.MODES))
            self.assertTrue(all(r['training_rows']<len(self.table) and r['test_id_set_sha256']==self.record['test_id_set_sha256'] for r in models))
            pred=E.read_csv(output/'primary_predictions.csv',E.PRED)
            self.assertEqual({r['row_id'] for r in pred},set(self.test['__id']))
            self.assertTrue(all(r['threshold']=='0.5' and int(r['predicted_label'])==int(float(r['score'])>=.5) for r in pred))

    def test_internal_run_rejects_unverified_authorization_before_fit(self):
        with patch.object(E.V2,'fit_candidate',side_effect=AssertionError('should not fit')):
            with self.assertRaisesRegex(ValueError,'authorization'):
                E.run_evaluation(Path('unused'),self.table,self.schedule,{'schedule':[]},{'status':'draft'})


class PackageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve();cls.package=package_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_no_fit_draft_contract_binds_folds_runtime_and_parent_protocol(self):
        output=self.root/'draft'
        with patch.object(E.V2,'fit_candidate',side_effect=AssertionError('draft fitting forbidden')), \
             patch.object(E.V2,'predict_candidate',side_effect=AssertionError('draft prediction forbidden')):
            E.main(['--stage','draft','--package-dir',str(self.package),'--output-dir',str(output),'--synthetic-test-only'])
        E.verify_publication(output)
        receipt=E.read_json(output/'preregistration_draft.json');contract=receipt['contract']
        self.assertFalse(receipt['fitting_started']);self.assertEqual(receipt['status'],'draft')
        self.assertIn(str(E.PARENT_PROTOCOL),contract['input_files_sha256'])
        self.assertEqual(contract['schedule_sha256'],E.digest(contract['schedule']))
        self.assertEqual(contract['fixed_threshold'],.5)
        self.assertFalse(contract['full_cohort_refit']);self.assertFalse(contract['historical_locked_or_pilot_scoring'])
        self.assertFalse((output/'fold_models.jsonl').exists())

    def test_missing_draft_wrong_stage_and_stale_freeze_rejected(self):
        contract,_,_=E.build_context(self.package,True)
        with self.assertRaisesRegex(ValueError,'parent-frozen'):E.authorize(contract,None,None)
        path=self.root/'synthetic_freeze_unit.json'
        receipt=dict(status='frozen',authorized_stage=E.STAGE,contract=contract,contract_sha256=E.digest(contract),
            independent_review=dict(approved=True,reviewer='root',reviewed_utc='synthetic_unit'))
        put(path,receipt)
        self.assertEqual(E.authorize(contract,path,E.sha(path))['status'],'frozen_verified')
        for key,value in [('status','draft'),('authorized_stage','dev')]:
            changed=copy.deepcopy(receipt);changed[key]=value;put(path,changed)
            with self.assertRaisesRegex(ValueError,'contract mismatch'):E.authorize(contract,path,E.sha(path))
        put(path,receipt)
        changed=copy.deepcopy(contract);changed['fixed_threshold']=.4
        with self.assertRaisesRegex(ValueError,'contract mismatch'):E.authorize(changed,path,E.sha(path))
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):E.authorize(contract,path,'0'*64)

    def test_run_without_receipt_does_not_reserve_output_or_fit(self):
        output=self.root/'never_created'
        with patch.object(E.V2,'fit_candidate',side_effect=AssertionError('fit forbidden')):
            with self.assertRaisesRegex(ValueError,'parent-frozen'):
                E.main(['--stage','run','--package-dir',str(self.package),'--output-dir',str(output),'--synthetic-test-only'])
        self.assertFalse(output.exists())

    def test_role_id_missingness_and_source_group_loading(self):
        table=E.load_table(self.package,True)
        self.assertEqual(len(table),16);self.assertEqual(set(table['__role']),{'development'})
        self.assertTrue(table[E.DESCRIPTORS].isna().any().any())
        with self.assertRaisesRegex(ValueError,'Exact2207'):E.load_table(self.package,False)


class PublicationTests(unittest.TestCase):
    def test_no_replace_or_orphan_acceptance_and_hash_recheck(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();output=root/'new';output.mkdir();(output/'small.csv').write_text('id\nsynthetic\n')
            with self.assertRaises(FileNotFoundError):E.verify_publication(output)
            E.commit_output(output);before=(output/'COMMIT.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'orphan'):E.commit_output(output)
            self.assertEqual((output/'COMMIT.json').read_bytes(),before)
            (output/'small.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'artifact changed'):E.verify_publication(output)

    def test_hardlink_failure_retains_uncommitted_orphan(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();(root/'small.csv').write_text('synthetic')
            with patch.object(E.os,'link',side_effect=OSError('synthetic NFS failure')):
                with self.assertRaises(OSError):E.commit_output(root)
            self.assertTrue((root/'small.csv').exists());self.assertFalse((root/'COMMIT.json').exists())
            self.assertFalse((root/'.COMMIT.pending').exists())


if __name__=='__main__':unittest.main(verbosity=2)
