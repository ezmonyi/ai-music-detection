"""Synthetic-only independent v6 audit mutation and interoperability checks."""
import ast
import copy
import csv
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import audit_equal60_v6_results as A
import test_audit_equal60_v5_results as OLD_TEST
import evaluate_new_phenomena_v6 as E  # Fixture generation only; never imported by auditor.
import test_evaluate_new_phenomena_v6 as ET


def overwrite_json(path,value):
    path.write_text(A.canonical(value).decode()+'\n')


def resign_package(package):
    marker=A.read_json(package/'COMMIT.json'); proof=A.read_json(package/'preparation_audit.json')
    proof['files_sha256']={n:A.sha(package/n) for n in A.PRODUCTS}
    overwrite_json(package/'preparation_audit.json',proof)
    marker['files']={n:dict(sha256=A.sha(package/n),bytes=(package/n).stat().st_size)
                     for n in A.PRODUCTS|{'preparation_audit.json'}}
    overwrite_json(package/'COMMIT.json',marker)


def overwrite_csv(path,fields,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


class V6AuditTests(unittest.TestCase):
    def test_independent_import_boundary_and_grid(self):
        tree=ast.parse(Path(A.__file__).read_text())
        imports=[alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        self.assertFalse(any('evaluate_new' in name or 'frozen_evaluate' in name for name in imports))
        self.assertEqual(A.sha(Path(A.A5.__file__)),A.HELPER_SHA)
        self.assertEqual(A.DESCRIPTORS,E.DESCRIPTORS)
        self.assertEqual(A.COMBINATIONS,E.COMBINATIONS)
        self.assertEqual(len(A.COMBINATIONS),255)
        self.assertEqual(len(A.A5.COMBINATIONS),127)

    def test_independent_schedule_matches_core_and_rejects_membership_mutation(self):
        table=ET.table()
        schedule,omitted,counts=A.build_schedule(table)
        expected,expected_omitted,expected_counts=E.make_schedule(table,True)
        self.assertEqual([r for r,*_ in schedule],[r for r,*_ in expected])
        self.assertEqual(omitted,expected_omitted);self.assertEqual(counts,expected_counts)
        self.assertEqual(counts['primary_fits'],len(schedule)*255)
        mutated=copy.deepcopy([schedule[0][0]]);mutated[0]['test_id_set_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'membership/order'):
            A.A5.validate_schedule_records(mutated,[schedule[0][0]])

    def test_sc_medians_missing_vectors_scales_and_stationarity_mutations(self):
        train=ET.table();train[A.SC[0]]=np.nan
        cache=A.TrainingCache(train)
        for mode in A.MODES:
            model=OLD_TEST.synthetic_model(cache,A.SC,mode)
            _,residual=A.A5.validate_model(model,cache,A.SC,mode)
            self.assertLess(residual,1e-8)
            self.assertEqual(model['medians'][0],0.)
            for field in ('medians','mean','scale','coefficients_with_intercept'):
                bad=copy.deepcopy(model);bad[field][-1]+=.1
                with self.subTest(mode=mode,field=field),self.assertRaises(ValueError):
                    A.A5.validate_model(bad,cache,A.SC,mode)
            bad=copy.deepcopy(model);bad['observed_fraction_by_column'][A.SC[0]]=1.
            with self.assertRaisesRegex(ValueError,'Observed-fraction'):
                A.A5.validate_model(bad,cache,A.SC,mode)
            bad=copy.deepcopy(model);bad['training_rows']+=1
            with self.assertRaisesRegex(ValueError,'training-row'):
                A.A5.validate_model(bad,cache,A.SC,mode)

    def test_sc_test_values_affect_raw_replay_and_prediction_binding(self):
        train=ET.table();cache=A.TrainingCache(train)
        model=OLD_TEST.synthetic_model(cache,A.SC,A.MODES[0]);test=train.iloc[[0]].copy()
        score=float(A.A5.replay_scores(test,A.SC,A.MODES[0],model)[0])
        identity=dict(model_uid='synthetic',row_id='r',label=1,source_group='ai',group_id='g',role='development')
        prediction={k:str(v) for k,v in identity.items()}
        prediction.update(score=format(score,'.17g'),threshold='0.5',predicted_label=str(int(score>=.5)))
        A.A5.validate_prediction_row(prediction,identity,score,'valid')
        for key,value in (('row_id','wrong'),('score',str(score+.01)),('threshold','0.4'),('label','0')):
            bad=dict(prediction);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                A.A5.validate_prediction_row(bad,identity,score,'tampered')
        test[A.SC]=1e6
        self.assertNotEqual(float(A.A5.replay_scores(test,A.SC,A.MODES[0],model)[0]),score)

    def test_package_sc_binding_order_and_row_mutations_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();package=ET.fixture(root)
            table,_,_=A.load_package(package,True);self.assertEqual(len(table),16)
            fields=['id',*A.DESCRIPTORS];rows=A.read_csv(package/'features_60s.csv',fields)
            old=rows[0][A.SC[0]];rows[0][A.SC[0]]='123.5'
            overwrite_csv(package/'features_60s.csv',fields,rows)
            with self.assertRaisesRegex(ValueError,'product changed'):A.load_package(package,True)
            rows[0][A.SC[0]]=old;rows[0],rows[1]=rows[1],rows[0]
            overwrite_csv(package/'features_60s.csv',fields,rows);resign_package(package)
            with self.assertRaisesRegex(ValueError,'ID/order'):A.load_package(package,True)
            rows[0],rows[1]=rows[1],rows[0];rows[1]['id']=rows[0]['id']
            overwrite_csv(package/'features_60s.csv',fields,rows);resign_package(package)
            with self.assertRaisesRegex(ValueError,'ID/order'):A.load_package(package,True)

    def test_complete_synthetic_authorized_run_and_audit_then_mutation(self):
        # One SC family, but the full generated schedule and all three modes;
        # production255 constants remain unconfigurable from CLI.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();package=ET.fixture(root);result=root/'result';receipt=root/'frozen.json'
            with patch.object(E,'COMBINATIONS',['SC']),patch.object(A,'COMBINATIONS',['SC']),redirect_stdout(io.StringIO()):
                c,_,_=E.build_context(package,True)
                E.write_json(receipt,dict(status='frozen',authorized_stage=E.STAGE,contract=c,contract_sha256=E.digest(c),
                             independent_review=dict(approved=True,reviewer='root',reviewed_utc='synthetic_fixture')))
                E.main(['--stage','run','--package-dir',str(package),'--output-dir',str(result),
                        '--receipt',str(receipt),'--receipt-sha256',E.sha(receipt),'--synthetic-test-only'])
                with patch.object(E.V2,'fit_candidate',side_effect=AssertionError('auditor must not fit')), \
                     patch.object(E.V2,'predict_candidate',side_effect=AssertionError('auditor must not score')):
                    audit=A.audit(result,package,receipt,A.sha(receipt),True)
                self.assertEqual(audit['status'],'passed')
                self.assertFalse(audit['model_fitting_performed']);self.assertFalse(audit['scorer_predict_called'])
                self.assertEqual(audit['models']['total'],c['accounting']['primary_fits']+c['accounting']['diagnostic_fits'])
                self.assertEqual(audit['oof_source_endpoints']['duplicate_oof_ids_within_arm_source'],0)
                path=result/'primary_predictions.csv';rows=A.read_csv(path,A.A5.PRED)
                rows[0]['score']=str(float(rows[0]['score'])+.05);overwrite_csv(path,A.A5.PRED,rows)
                # Re-sign container hashes to require numerical replay to catch the change.
                marker=A.read_json(result/'COMMIT.json');marker['files'][path.name]=dict(sha256=A.sha(path),bytes=path.stat().st_size)
                overwrite_json(result/'COMMIT.json',marker)
                with self.assertRaisesRegex(ValueError,'independent affine replay'):
                    A.audit(result,package,receipt,A.sha(receipt),True)


if __name__=='__main__':unittest.main()
