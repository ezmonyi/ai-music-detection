"""Handwritten synthetic oracle; no scorer imports, training or real replay."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import audit_saraga103_transfer_v1 as A


def write_json(path, value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def write_csv(path, records, fields=None):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def rebind_outputs(root):
    manifest = A.strict_json(root/'publication_manifest.json')
    manifest['files']={name:{'sha256':A.sha(root/name),'bytes':(root/name).stat().st_size} for name in sorted(A.FILES)}
    write_json(root/'publication_manifest.json',manifest)
    write_json(root/'COMMIT.json',dict(status='committed',publication='exclusive hardlinks, COMMIT last',
        files={name:{'sha256':A.sha(root/name),'bytes':(root/name).stat().st_size} for name in sorted(A.FILES|{'publication_manifest.json'})}))


def make_fixture(root):
    source=root/'numerical';source.mkdir()
    output=root/'scored';output.mkdir()
    tokens=['','1','2','na','-1','nan']
    # Analytic values, independent of the auditor's vector/affine implementation.
    filled=[2.,1.,2.,2.,-1.,2.]
    missing=[1,0,0,1,0,1]
    meta=[dict(id=f'synthetic_saraga_v4_{i}',label='0',source_group=A.SOURCE,
        group_id='synthetic_large' if i<4 else 'synthetic_small',role=A.ROLE,
        evaluation_allowed='False',classifier_admission_authorized='False') for i in range(6)]
    old,fhm=[],[]
    for i,identity in enumerate(meta):
        native='44100' if i<4 else '48000'
        sd=dict(item_id=identity['id'],label='0',source_id=A.SOURCE,group_id=identity['group_id'],status='complete',native_sample_rate_hz=native)
        fh=dict(identity,source_id=A.SOURCE,extraction_status='ok',native_sample_rate_hz=native)
        for family,names in A.FAMILIES.items():
            for name in names:
                (sd if family in ('S','D','R','P') else fh)[name]=tokens[i]
        old.append(sd);fhm.append(fh)
    inputs=dict(old4_csv=source/'old4.csv',fhm_csv=source/'fhm.csv',models=source/'fold_models.json',model_index=source/'model_index.csv')
    write_csv(inputs['old4_csv'],old);write_csv(inputs['fhm_csv'],fhm)
    models,index={},[]
    for combo in A.COMBINATIONS:
        columns=sum((A.FAMILIES[f] for f in combo.split('+')),[])
        width=len(columns)
        for cap in A.CAPS:
            for fold in range(5):
                intercept=-.25+.2*fold
                coefficients=[intercept]+[0.]*(2*width)
                coefficients[1]=.3;coefficients[width+1]=.45
                model=dict(columns=columns,medians=[2.]*width,mean=[0.]*(2*width),scale=[1.]*(2*width),
                    coefficients_with_intercept=coefficients,ridge=10.,threshold=.5,feature_mode='values_plus_missing',
                    training_rows=3,training_source_counts={'synthetic_training_only':3},weighting=A.structure.WEIGHTING,
                    missing_value_policy='train-only median plus feature-missing indicators',
                    observed_fraction_by_column={name:.5 for name in columns},
                    model_type='weighted_ridge_linear_probability',prediction_link='identity')
                key=A.digest({'model':model,'threshold':.5});models[key]=model
                index.append(dict(combination=combo,quantity=cap,fold_index=str(fold),fold_uid=f'synthetic_{cap}_{fold}',
                    model_sha256=key,candidate_key='synthetic_handwritten',
                    train_id_set_sha256=hashlib.sha256(f'train_{cap}_{fold}'.encode()).hexdigest(),
                    test_id_set_sha256=hashlib.sha256(f'test_{cap}_{fold}'.encode()).hexdigest()))
    index.sort(key=lambda r:(r['combination'],r['quantity'],r['fold_index']))
    write_json(inputs['models'],models);write_csv(inputs['model_index'],index,A.INDEX)
    sizes={'synthetic_large':4,'synthetic_small':2}
    contract=dict(schema_version=1,authorized_stage=A.STAGE,synthetic_test_only=True,measurement_interface='synthetic_fixture',
        old_development_admission=False,refitting=False,model_selection=False,threshold_tuning=False,
        families=A.FAMILIES,combinations=A.COMBINATIONS,caps=A.CAPS,fold_models=5,
        feature_mode='values_plus_missing',threshold=.5,positive_rule='score >= 0.5',prediction_link='identity_unclipped',
        rows=6,model_instances=3175,prediction_rows=19050,summary_rows=3175,component_sizes=sizes,overview_combinations=A.OVERVIEW,
        identity_sha256=A.digest(meta),model_index_sha256=A.digest(index),input_files_sha256={str(p):A.sha(p) for p in inputs.values()},
        runtime=dict(python='synthetic',numpy='synthetic',pandas='synthetic',threads={}),endpoints=A.METRICS,
        forbidden_endpoints=['balanced_accuracy','roc_auc','two_class_accuracy','source_transfer_J','rank','winner'],
        comparison_tolerance=dict(score_atol=1e-12,score_rtol=1e-12,threshold_decisions='exact',integer_summaries='exact'),
        uncertainty='five-model min/max are descriptive ranges, not confidence intervals; same103 dependent observations',
        component_weighting='equal mean of five connected components; distinct from primary per-recording weights',
        measurement_role=A.ROLE,scoring_role=A.SCORING_ROLE,
        publication='exclusive directory reservation and hardlink COMMIT last; no overwrite or resume')
    receipt=dict(status='frozen',authorized_stage=A.STAGE,contract=contract,contract_sha256=A.digest(contract),
        independent_review=dict(approved=True,reviewer='root',reviewed_utc='synthetic_handwritten_test'))
    receipt_path=root/'frozen.json';write_json(receipt_path,receipt)
    write_json(output/'scoring_receipt.json',receipt)
    write_csv(output/'model_index.csv',index,A.INDEX);write_csv(output/'identity_roles.csv',meta,A.META)
    summaries,components=[],[]
    with (output/'predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=A.PRED);writer.writeheader()
        for common in index:
            intercept=-.25+.2*int(common['fold_index'])
            # Written independently from frozen model arrays and any replay function.
            scores=[intercept+.3*v+.45*m for v,m in zip(filled,missing)]
            decisions=[int(s>=.5) for s in scores]
            for identity,score,decision in zip(meta,scores,decisions):
                writer.writerow(dict(common,**identity,scoring_role=A.SCORING_ROLE,synthetic_test_only=True,
                    score=format(score,'.17g'),threshold=.5,predicted_label=decision))
            rates=[]
            for group,positions in [('synthetic_large',range(4)),('synthetic_small',range(4,6))]:
                count=len(positions);fp=sum(decisions[j] for j in positions);rates.append(fp/count)
                components.append(dict(common,synthetic_test_only=True,group_id=group,rows=count,fp=fp,tn=count-fp,
                    false_positive_rate=fp/count,specificity=(count-fp)/count))
            fp=sum(decisions)
            summaries.append(dict(common,synthetic_test_only=True,rows=6,unique_ids=6,unique_groups=2,fp=fp,tn=6-fp,
                threshold=.5,false_positive_rate=fp/6,specificity=(6-fp)/6,
                equal_component_false_positive_rate=sum(rates)/2,equal_component_specificity=sum(1-r for r in rates)/2))
    write_csv(output/'per_model_specificity.csv',summaries,A.SUMMARY)
    write_csv(output/'per_component_specificity.csv',components,A.COMPONENT)
    cells=[]
    for offset in range(0,len(summaries),5):
        values=summaries[offset:offset+5]
        cell=dict(combination=values[0]['combination'],quantity=values[0]['quantity'],fold_models=5,rows_per_model=6,synthetic_test_only=True)
        for metric in A.METRICS:
            vals=[r[metric] for r in values]
            cell.update({metric+'_mean':sum(vals)/5,metric+'_min':min(vals),metric+'_max':max(vals)})
        cells.append(cell)
    write_csv(output/'all635_cells.csv',cells,A.CELL)
    write_csv(output/'predefined_overview.csv',[r for r in cells if r['combination'] in A.OVERVIEW],A.CELL)
    write_json(output/'output_accounting.json',dict(prediction_rows=19050,model_summary_rows=3175,component_summary_rows=6350,
        aggregate_cells=635,integer_counts_consistent=True,independent_numerical_audit_performed=False))
    write_json(output/'publication_manifest.json',dict(status='scored',synthetic_test_only=True,contract_sha256=A.digest(contract),
        classifier_fitted=False,original_v4_development_admission=False,unique_new_ids=6,model_instances=3175,prediction_rows=19050,files={}))
    rebind_outputs(output)
    return output,receipt_path,inputs


class NumericalTests(unittest.TestCase):
    def test_nontrivial_median_center_scale_missing_and_unclipped_scores(self):
        model=dict(columns=['a','b'],medians=[4.,8.],mean=[1.,2.,.25,.5],scale=[2.,3.,.5,2.],
            coefficients_with_intercept=[.7,2.,-3.,4.,-5.])
        features={'a':np.array([np.nan,3.]),'b':np.array([5.,np.nan])}
        expected=np.array([.7+((4-1)/2)*2+((5-2)/3)*-3+((1-.25)/.5)*4+((0-.5)/2)*-5,
            .7+((3-1)/2)*2+((8-2)/3)*-3+((0-.25)/.5)*4+((1-.5)/2)*-5])
        np.testing.assert_array_equal(A.independent_scores(features,model),expected)
        self.assertGreater(expected[0],1);self.assertLess(expected[1],0)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template=tempfile.TemporaryDirectory()
        cls.root=Path(cls.template.name).resolve()
        cls.base_scored,cls.receipt,cls.inputs=make_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.template.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.scored=Path(self.temp.name).resolve()/'scored'
        shutil.copytree(self.base_scored,self.scored)

    def run_audit(self,**overrides):
        args=dict(scored=self.scored,receipt_path=self.receipt,receipt_sha256=A.sha(self.receipt),inputs=self.inputs,synthetic=True)
        args.update(overrides)
        return A.audit(**args)

    def mutate_table(self,name,mutation):
        records=A.csv_rows(self.scored/name);mutation(records);write_csv(self.scored/name,records)
        rebind_outputs(self.scored)

    def test_full_grid_handwritten_oracle_missing_and_component_weights(self):
        report,_=self.run_audit()
        self.assertEqual(report['prediction_rows'],19050)
        self.assertEqual(report['model_summary_rows'],3175)
        self.assertEqual(report['component_summary_rows'],6350)
        self.assertEqual(report['aggregate_cells'],635)
        self.assertTrue(all(n==3 for n in report['missing_by_descriptor'].values()))
        summaries=A.csv_rows(self.scored/'per_model_specificity.csv')
        self.assertTrue(any(r['false_positive_rate']!=r['equal_component_false_positive_rate'] for r in summaries))

    def test_committed_product_tampering_rejected_before_numerics(self):
        with (self.scored/'predictions.csv').open('a') as stream:stream.write('\n')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.run_audit()

    def test_resealed_wrong_score_rejected(self):
        self.mutate_table('predictions.csv',lambda r:r[0].update(score='42'))
        with self.assertRaisesRegex(ValueError,'numerical score mismatch'):self.run_audit()

    def test_threshold_crossing_not_excused_by_tolerance(self):
        # Override only the independent test seam to isolate an adjacent-float crossing.
        from unittest.mock import patch
        self.mutate_table('predictions.csv',lambda r:r[0].update(score='0.49999999999999994',predicted_label='0'))
        original=A.independent_scores
        def adjacent(features,model):
            result=original(features,model);result[0]=.5;return result
        with patch.object(A,'independent_scores',side_effect=adjacent):
            with self.assertRaisesRegex(ValueError,'Exact threshold decision mismatch'):self.run_audit()

    def test_wrong_model_integer_denominator_rejected(self):
        self.mutate_table('per_model_specificity.csv',lambda r:r[0].update(fp='5'))
        with self.assertRaisesRegex(ValueError,'integer denominator'):self.run_audit()

    def test_wrong_component_count_rejected(self):
        self.mutate_table('per_component_specificity.csv',lambda r:r[0].update(rows='6'))
        with self.assertRaisesRegex(ValueError,'Component identity/denominator'):self.run_audit()

    def test_primary_metric_cannot_replace_equal_component_metric(self):
        def mutation(records):
            row=next(r for r in records if r['false_positive_rate']!=r['equal_component_false_positive_rate'])
            row['equal_component_false_positive_rate']=row['false_positive_rate']
        self.mutate_table('per_model_specificity.csv',mutation)
        with self.assertRaisesRegex(ValueError,'Per-model metric mismatch'):self.run_audit()

    def test_all635_aggregate_statistic_checked(self):
        self.mutate_table('all635_cells.csv',lambda r:r[-1].update(specificity_max='999'))
        with self.assertRaisesRegex(ValueError,'Aggregate statistic mismatch'):self.run_audit()

    def test_predefined_overview_cannot_be_reranked(self):
        self.mutate_table('predefined_overview.csv',lambda r:r.reverse())
        with self.assertRaisesRegex(ValueError,'Predefined overview'):self.run_audit()

    def test_omitted_prediction_rejected(self):
        self.mutate_table('predictions.csv',lambda r:r.pop())
        with self.assertRaisesRegex(ValueError,'Prediction missing'):self.run_audit()

    def test_caller_input_must_be_exact_frozen_path(self):
        copy_path=Path(self.temp.name)/'fhm.csv';shutil.copy2(self.inputs['fhm_csv'],copy_path)
        with self.assertRaisesRegex(ValueError,'absent from frozen bindings'):
            self.run_audit(inputs=dict(self.inputs,fhm_csv=copy_path))

    def test_receipt_hash_and_real_boundary(self):
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.run_audit(receipt_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'Real103/synthetic16'):self.run_audit(synthetic=False)

    def test_orphan_and_extra_artifact_rejected(self):
        (self.scored/'COMMIT.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing/noncanonical'):self.run_audit()

    def test_report_no_replace_and_postaudit_input_recheck(self):
        report,bindings=self.run_audit()
        path=Path(self.temp.name).resolve()/'audit.json'
        A.structure.publish_report(report,bindings,path)
        original=path.read_bytes()
        with self.assertRaises(ValueError):A.structure.publish_report(report,bindings,path)
        self.assertEqual(path.read_bytes(),original)
        with (self.scored/'predictions.csv').open('a') as stream:stream.write('\n')
        after=path.with_name('after_tamper.json')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):A.structure.publish_report(report,bindings,after)
        self.assertFalse(after.exists())

    def test_contract_threshold_role_and_component_denominator_rejected(self):
        receipt=A.strict_json(self.receipt)
        for field,value in [('threshold',.51),('measurement_role','development'),('component_sizes',{'synthetic_large':5})]:
            changed=copy.deepcopy(receipt)
            changed['contract'][field]=value
            changed['contract_sha256']=A.digest(changed['contract'])
            with self.subTest(field=field),self.assertRaises(ValueError):A.contract_check(changed,True)

    def test_nonpositive_scale_rejected_before_numerical_use(self):
        models=A.strict_json(self.inputs['models'])
        model=copy.deepcopy(next(iter(models.values())))
        model['scale'][0]=0
        combo=next(r['combination'] for r in A.csv_rows(self.inputs['model_index']) if r['model_sha256']==next(iter(models)))
        with self.assertRaisesRegex(ValueError,'Nonpositive scale'):A.structure.validate_model(model,combo)

    def test_duplicate_json_keys_rejected(self):
        path=Path(self.temp.name)/'duplicate.json'
        path.write_text('{"status":"frozen","status":"draft"}')
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):A.strict_json(path)


if __name__=='__main__':
    unittest.main(verbosity=2)
