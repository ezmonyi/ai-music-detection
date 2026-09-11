"""Handwritten presentation fixtures; no numerical scoring or real admission."""
import ast
import copy
import csv
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import unittest
from unittest.mock import patch

import present_saraga_mureka_frozen_transfer_v1 as P


def put(path,value): path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def csv_write(path,records,fields=None):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields or list(records[0]));writer.writeheader();writer.writerows(records)


def fixture(root):
    root.mkdir()
    put(root/'presentation_fixture.json',{'synthetic_test_only':True,'purpose':'handwritten_presentation_fixture_no_inference_or_scores'})
    index=[]
    for combo,cap,fold in itertools_grid():
        index.append(dict(combination=combo,quantity=cap,fold_index=fold,fold_uid=f'synthetic_{cap}_{fold}',
            model_sha256=P.digest([combo,cap,fold]),candidate_key='synthetic_'+combo,
            train_id_set_sha256=P.digest(['train',cap,fold]),test_id_set_sha256=P.digest(['test',cap,fold])))
    index.sort(key=P.index_key)
    for kind,n in [('mureka',10),('saraga',7)]:
        directory=root/kind;directory.mkdir()
        ids=[]
        for i in range(n):
            value=dict(id=f'synthetic_{kind}_v4_{i}',label='1' if kind=='mureka' else '0',
                source_group='Mureka_v9' if kind=='mureka' else 'human_saraga_hindustani_v1',
                group_id=f'synthetic_group_{i if kind=="mureka" else max(0,i-2)}',
                role='external_generator_unscored' if kind=='mureka' else 'external_human_unscored')
            value.update(dict(acquisition_role='reserved_unscored') if kind=='mureka' else dict(evaluation_allowed='False',classifier_admission_authorized='False'))
            ids.append(value)
        groups=dict(sorted(P.Counter(i['group_id'] for i in ids).items()))
        csv_write(directory/'identity_roles.csv',ids,P.META[kind]);csv_write(directory/'model_index.csv',index,P.INDEX)
        contract=dict(authorized_stage=P.STAGES[kind],rows=n,synthetic_test_only=True,threshold=.5,model_instances=3175,
            refitting=False,model_selection=False,threshold_tuning=False,model_index_sha256=P.digest(index),identity_sha256=P.digest(ids))
        if kind=='saraga':contract['component_sizes']=groups
        put(directory/'scoring_receipt.json',dict(status='frozen',authorized_stage=P.STAGES[kind],contract=contract,contract_sha256=P.digest(contract)))
        summaries,components=[],[]
        for model in index:
            fold=int(model['fold_index']);combo=model['combination']
            row=dict(model,synthetic_test_only='True',rows=n,unique_ids=n,unique_groups=len(groups),threshold=.5)
            if kind=='mureka':
                increment={P.CONTRASTS[0]:fold%3,P.CONTRASTS[1]:1,P.CONTRASTS[2]:-1,P.CONTRASTS[3]:2}.get(combo,0)
                tp=3+fold+increment;row.update(tp=tp,fn=n-tp,ai_sensitivity=tp/n,false_negative_rate=(n-tp)/n)
            else:
                fplist=[fold%4,0,0,0,0]
                if combo==P.CONTRASTS[0]:fplist[1]=int(fold%2==0)
                if combo==P.CONTRASTS[1]:fplist[0]=max(0,fplist[0]-1)
                if combo==P.CONTRASTS[2]:fplist[1]=fplist[2]=1
                if combo==P.CONTRASTS[3]:fplist[3]=1
                for (group,count),fp in zip(groups.items(),fplist):
                    components.append(dict(model,synthetic_test_only='True',group_id=group,rows=count,fp=fp,tn=count-fp,
                        false_positive_rate=fp/count,specificity=(count-fp)/count))
                fp=sum(fplist);fpr=statistics.mean(fp/count for fp,count in zip(fplist,groups.values()))
                row.update(fp=fp,tn=n-fp,false_positive_rate=fp/n,specificity=(n-fp)/n,
                           equal_component_false_positive_rate=fpr,equal_component_specificity=1-fpr)
            summaries.append(row)
        csv_write(directory/('per_model_sensitivity.csv' if kind=='mureka' else 'per_model_specificity.csv'),summaries,P.SUMMARY[kind])
        # The presentation reader verifies hashes of a previously audited file;
        # it deliberately does not score or inspect prediction values here.
        (directory/'predictions.csv').write_text('synthetic_prior_publication_placeholder\n')
        if kind=='saraga':
            csv_write(directory/'per_component_specificity.csv',components,P.COMPONENT)
            for name in ('all635_cells.csv','predefined_overview.csv'): (directory/name).write_text('synthetic_prior_presentation_placeholder\n')
            put(directory/'output_accounting.json',{'synthetic_placeholder':True})
        rebind(directory,kind,n,groups)


def itertools_grid():
    for combo in P.COMBINATIONS:
        for cap in P.CAPS:
            for fold in map(str,range(5)): yield combo,cap,fold


def rebind(directory,kind,n=None,groups=None):
    c=P.read_json(directory/'scoring_receipt.json')['contract'];n=n or c['rows'];groups=groups or c.get('component_sizes')
    manifest=dict(status='scored',synthetic_test_only=True,contract_sha256=P.digest(c),classifier_fitted=False,
        original_v4_development_admission=False,unique_new_ids=n,model_instances=3175,prediction_rows=3175*n,
        files={name:{'sha256':P.sha(directory/name),'bytes':(directory/name).stat().st_size} for name in sorted(P.FILES[kind])})
    put(directory/'publication_manifest.json',manifest)
    if kind=='saraga':
        put(directory/'COMMIT.json',dict(status='committed',publication='exclusive hardlinks, COMMIT last',
            files=dict(manifest['files'],**{'publication_manifest.json':{'sha256':P.sha(directory/'publication_manifest.json'),'bytes':(directory/'publication_manifest.json').stat().st_size}})))
    audited={str(path):P.sha(path) for path in directory.iterdir()}
    code=Path(P.__file__).with_name(P.AUDIT_NAMES[kind]);audited[str(code)]=P.sha(code)
    audit=dict(status='passed',synthetic_test_only=True,model_fitting_performed=False,rows=n,model_instances=3175,
        prediction_rows=3175*n,contract_sha256=P.digest(c),frozen_receipt_sha256=P.sha(directory/'scoring_receipt.json'),checked_files_sha256=audited)
    if kind=='mureka':audit.update(all_ids_in_every_model=True,all_decisions_match_published_and_independent_scores=True,
        sensitivity_summary_rows=3175,new_source_transform_learning_performed=False,publication_manifest_sha256=P.sha(directory/'publication_manifest.json'))
    else:audit.update(all_decisions_exact=True,all_models_all_ids_checked=True,all_integer_counts_exact=True,
        equal_component_and_per_recording_metrics_separately_checked=True,all635_mean_min_max_checked=True,
        model_summary_rows=3175,aggregate_cells=635,model_selection_performed=False,threshold_tuning_performed=False,
        component_sizes=groups,component_summary_rows=3175*len(groups),publication_commit_sha256=P.sha(directory/'COMMIT.json'))
    put(directory.parent/(kind+'_audit.json'),audit)


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve();cls.data=cls.root/'fixture';fixture(cls.data)
        cls.sources={k:P.load_source(cls.data/k,cls.data/(k+'_audit.json'),P.sha(cls.data/(k+'_audit.json')),k,P.Bindings(),True) for k in ('mureka','saraga')}
        cls.tables=P.build_tables(**cls.sources);cls.output=cls.root/'presentation';cls.run_cli(cls.data,cls.output)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    @classmethod
    def run_cli(cls,data,out,extra=None):
        argv=['--output-dir',str(out),'--synthetic-test-only','--synthetic-fixture-root',str(data)]
        for k in ('mureka','saraga'):
            argv += ['--'+k+'-results',str(data/k),'--'+k+'-audit',str(data/(k+'_audit.json')),
                     '--'+k+'-audit-sha256',P.sha(data/(k+'_audit.json'))]
        P.main(argv+(extra or []))
    def clone(self,name):
        dst=self.root/name;shutil.copytree(self.data,dst);return dst

    def test_complete_grids_and_fixed_rows(self):
        for name,count in [('matched_per_model.csv',3175),('all635_side_by_side.csv',635),('fixed60_overview.csv',60),
                           ('matched100_deltas_pp.csv',100),('fixed20_delta_cells_pp.csv',20),('saraga_component_cells.csv',3175)]:
            self.assertEqual(len(self.tables[name]),count)
        self.assertEqual({r['combination'] for r in self.tables['fixed60_overview.csv']},set(P.OVERVIEW))

    def test_source_specific_mean_rates_and_ranges(self):
        row=next(r for r in self.tables['all635_side_by_side.csv'] if r['combination']==P.BASELINE and r['quantity']=='25')
        self.assertAlmostEqual(row['mureka_ai_sensitivity_mean'],.5)
        self.assertAlmostEqual(row['mureka_ai_sensitivity_min'],.3);self.assertAlmostEqual(row['mureka_ai_sensitivity_max'],.7)
        self.assertAlmostEqual(row['saraga_false_positive_rate_mean'],6/35)
        self.assertAlmostEqual(row['saraga_equal_component_false_positive_rate_mean'],.08)
        self.assertEqual((row['mureka_rows_per_model'],row['saraga_rows_per_model']),(10,7))

    def test_paired_pp_delta_direction_and_ranges(self):
        row=next(r for r in self.tables['fixed20_delta_cells_pp.csv'] if r['candidate']==P.CONTRASTS[0] and r['quantity']=='25')
        self.assertAlmostEqual(row['mureka_ai_sensitivity_delta_pp_mean'],8.)
        self.assertAlmostEqual(row['mureka_ai_sensitivity_delta_pp_min'],0.)
        self.assertAlmostEqual(row['mureka_ai_sensitivity_delta_pp_max'],20.)
        self.assertAlmostEqual(row['mureka_false_negative_rate_delta_pp_mean'],-8.)
        self.assertAlmostEqual(row['saraga_specificity_delta_pp_mean'],-60/7)
        self.assertAlmostEqual(row['saraga_false_positive_rate_delta_pp_mean'],60/7)
        self.assertAlmostEqual(row['saraga_equal_component_specificity_delta_pp_mean'],-12.)
        self.assertAlmostEqual(row['saraga_equal_component_false_positive_rate_delta_pp_mean'],12.)
        # Pair ranges differ from extrema of the separate five-model rates.
        self.assertLess(row['mureka_ai_sensitivity_delta_pp_max'],50.)

    def test_exact_model_hash_and_train_test_match_required(self):
        for key in P.INDEX:
            human=copy.deepcopy(self.sources['saraga']);human['index'][0][key]='changed'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'exact cap/fold'):P.matching_rows(self.sources['mureka'],human)

    def test_contrast_requires_common_old_fold_and_train_test(self):
        ai,human=copy.deepcopy(self.sources['mureka']),copy.deepcopy(self.sources['saraga'])
        for source in (ai,human):
            row=next(r for r in source['index'] if r['combination']==P.CONTRASTS[0] and r['quantity']=='25' and r['fold_index']=='0')
            row['train_id_set_sha256']=P.digest('changed')
        with self.assertRaisesRegex(ValueError,'Contrast folds'):P.build_tables(ai,human)

    def test_unpassed_or_wrong_scope_audit_rejected(self):
        report=P.read_json(self.data/'saraga_audit.json')
        for key,value in [('status','running'),('synthetic_test_only',False),('all_decisions_exact',False),('model_selection_performed',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):P.verify_audit(dict(report,**{key:value}),'saraga',True,7)

    def test_tampered_publication_fails_hash_validation(self):
        data=self.clone('changed_bytes');(data/'mureka/per_model_sensitivity.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_cli(data,self.root/'changed_result')

    def test_changed_audit_hash_rejected(self):
        path=self.data/'saraga_audit.json'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):P.load_source(self.data/'saraga',path,'0'*64,'saraga',P.Bindings(),True)

    def test_original_and_reserialized_frozen_receipt_hashes_may_differ(self):
        data=self.clone('normalized_receipt');path=data/'mureka_audit.json';report=P.read_json(path)
        original=data/'original_frozen_receipt.json'
        original.write_text(json.dumps(P.read_json(data/'mureka/scoring_receipt.json'),indent=4)+'\n')
        self.assertNotEqual(P.sha(original),P.sha(data/'mureka/scoring_receipt.json'))
        report['frozen_receipt_sha256']=P.sha(original);report['checked_files_sha256'][str(original)]=P.sha(original);put(path,report)
        source=P.load_source(data/'mureka',path,P.sha(path),'mureka',P.Bindings(),True)
        self.assertEqual(source['count'],10)
        self.assertEqual(source['original_frozen_receipt_sha256'],P.sha(original))
        self.assertEqual(source['published_scoring_receipt_sha256'],P.sha(data/'mureka/scoring_receipt.json'))

    def test_resigned_summary_does_not_escape_integer_validation(self):
        data=self.clone('changed_counts');path=data/'saraga/per_model_specificity.csv';rows=P.read_csv(path,P.SUMMARY['saraga']);rows[0]['tn']='999';csv_write(path,rows)
        rebind(data/'saraga','saraga')
        with self.assertRaisesRegex(ValueError,'Integer counts'):P.load_source(data/'saraga',data/'saraga_audit.json',P.sha(data/'saraga_audit.json'),'saraga',P.Bindings(),True)

    def test_component_sum_and_equal_weight_validation(self):
        data=self.clone('changed_components');path=data/'saraga/per_component_specificity.csv';rows=P.read_csv(path,P.COMPONENT)
        rows[0].update(fp='1',tn='2',false_positive_rate=str(1/3),specificity=str(2/3));csv_write(path,rows);rebind(data/'saraga','saraga')
        with self.assertRaisesRegex(ValueError,'Component/recording'):P.load_source(data/'saraga',data/'saraga_audit.json',P.sha(data/'saraga_audit.json'),'saraga',P.Bindings(),True)

    def test_missing_duplicate_grid_and_fold_identity(self):
        idx=self.sources['mureka']['index']
        for changed in (idx[:-1],idx[:-1]+idx[:1]):
            with self.assertRaises(ValueError):P.validate_index(changed)
        idx=copy.deepcopy(idx);idx[0]['fold_uid']='new'
        with self.assertRaisesRegex(ValueError,'varies across'):P.validate_index(idx)

    def test_output_commit_input_hashes_and_no_pooling(self):
        marker=P.read_json(self.output/'COMMIT.json');P.record_files(self.output,marker['files'],P.Bindings())
        receipt=P.read_json(self.output/'presentation_receipt.json')
        self.assertTrue(receipt['exact_model_identity_match']);self.assertFalse(receipt['source_pooling_performed'])
        self.assertFalse(receipt['classifier_scores_generated']);self.assertFalse(receipt['raw_media_admission_repeated'])
        self.assertIn(str(self.data/'mureka_audit.json'),receipt['input_files_sha256'])
        for records in self.tables.values():
            self.assertFalse(set(records[0])&{'balanced_accuracy','roc_auc','source_transfer_J','rank','winner'})

    def test_markdown_explains_separate_rates_and_delta_sign(self):
        text=(self.output/'MATCHED_TRANSFER_TABLES_EN.md').read_text()
        self.assertIn('Synthetic test fixture only',text);self.assertIn('not confidence intervals',text)
        self.assertIn('higher error rates',text);self.assertIn('3, 1, 1, 1, 1',text)

    def test_no_overwrite_and_hardlink_failure_no_commit(self):
        with self.assertRaisesRegex(ValueError,'Existing output'):self.run_cli(self.data,self.output)
        source=self.root/'link_failure_source';source.mkdir();(source/'x').write_text('x');dest=self.root/'link_failure_dest'
        with patch.object(P.os,'link',side_effect=OSError('unavailable')):
            with self.assertRaises(OSError):P.publish(source,dest)
        self.assertFalse((dest/'COMMIT.json').exists())
        with self.assertRaises(FileExistsError):P.publish(source,dest)

    def test_real_mode_refuses_synthetic_marker_or_audit(self):
        with self.assertRaises(ValueError):P.verify_audit(P.read_json(self.data/'saraga_audit.json'),'saraga',False,103)
        with self.assertRaisesRegex(ValueError,'Only accepted'):P.load_source(self.data/'mureka',self.data/'mureka_audit.json',P.sha(self.data/'mureka_audit.json'),'mureka',P.Bindings(),False)

    def test_read_only_no_classifier_or_raw_media_imports(self):
        tree=ast.parse(Path(P.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(any(token in (name or '') for token in ('score_','audit_','torch','numpy','soundfile','sklearn')) for name in imports))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','predict','replay','decode','quantile','median') for n in ast.walk(tree)))


if __name__=='__main__':unittest.main()
