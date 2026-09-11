#!/usr/bin/env python3
"""Isolated exploratory v5 source/group CV; draft never fits or predicts.

The historical v2/base numerical implementation is unchanged. External-source
reuse is explicit, diagnostics cannot select, and every fit is train-fold only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

import prepare_evaluation_inputs_v5 as PREP
import evaluate_new_phenomena_v2 as V2

BASE = V2.BASE
ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT/'EXPLORATORY_V5_EVALUATOR_PROTOCOL_EN.md'
PARENT_PROTOCOL = ROOT/'EQUAL60_EXPLORATORY_V5_PROTOCOL_EN.md'
STAGE = 'exploratory_v5_development_cv_only'
SEED, FOLDS = 20260907, 5
QUANTITIES = (25,50,100,200,'all')
FAMILIES = PREP.COLUMNS
DESCRIPTORS = sum(FAMILIES.values(),[])
COMBINATIONS = ['+'.join(c) for n in range(1,8) for c in itertools.combinations(FAMILIES,n)]
MODES = ('values_plus_missing','median_only','missingness_only')
POLICIES = {'values_plus_missing':'train-only median plus feature-missing indicators',
            'median_only':'train-only median; no indicators','missingness_only':'feature-missing indicators only'}
FIXED_CORE = {'evaluate_new_phenomena_v2.py':'4014492f3e3ddda2d9cbea9f37147b37be0124ba071f0fee803845537526bd2e',
              'frozen_evaluate_expanded_20260905.py':'d2ed30d9833fbe122f023de1223e44a40c1b4f95f99f63f0ba27647c87cc7232'}
INDEX = ['model_uid','model_sha256','model_jsonl_line','combination','feature_mode','quantity','fold_uid','fold_index',
         'fold_type','heldout_source','opposite_group_fold','train_id_set_sha256','test_id_set_sha256','training_rows','test_rows']
PRED = ['model_uid','row_id','label','source_group','group_id','role','score','threshold','predicted_label']
METRICS = ['roc_auc','balanced_accuracy','ai_sensitivity','human_specificity','tp','tn','fp','fn']
PAIR = ['model_uid','human_source','ai_source','test_human','test_ai','test_human_groups','test_ai_groups','threshold',*METRICS]
POOLED = ['model_uid','test_rows','test_groups','threshold',*METRICS]
SOURCE = ['model_uid','source_group','label','rows','components','threshold','tp','tn','fp','fn',
          'recording_positive_rate','recording_negative_rate','recording_ai_sensitivity','recording_human_specificity',
          'equal_component_positive_rate','equal_component_negative_rate','equal_component_ai_sensitivity','equal_component_human_specificity']
EXPECTED_REAL = dict(valid_folds=43,primary_fits=27305,primary_prediction_rows=7633970,
                     diagnostic_fits=10922,diagnostic_prediction_rows=3053588)


def require(condition,message):
    if not condition: raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha(path): return PREP.sha(path)


def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'Duplicate JSON key: '+key);result[key]=value
        return result
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=reject)


def write_json(path,value):
    with Path(path).open('x') as stream:
        stream.write(canonical(value).decode()+'\n');stream.flush();os.fsync(stream.fileno())


def read_csv(path,fields):
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==fields,'CSV schema changed: '+str(path))
        result=list(reader)
    require(result and all(None not in row and None not in row.values() for row in result),'Empty/malformed CSV')
    return result


def load_table(package,synthetic=False):
    metadata=read_csv(package/'metadata_60s.csv',PREP.META)
    features=read_csv(package/'features_60s.csv',['id',*DESCRIPTORS])
    ids=[r['id'] for r in metadata]
    require(ids==[r['id'] for r in features] and len(ids)==len(set(ids)),'Exact ID order/uniqueness required')
    require(all(r['role']=='development' and r['duration_view']=='60s' and r['id'].strip() and r['group_id'].strip() for r in metadata),
            'Only derived exact60 development rows allowed')
    if synthetic: require(0<len(ids)<=16 and all(i.startswith('synthetic_v5_') for i in ids),'Synthetic16 boundary')
    else:
        require(len(ids)==2207 and Counter(r['source_group'] for r in metadata)==PREP.COUNTS
                and Counter(r['label'] for r in metadata)=={'0':1311,'1':896},'Exact2207 exploratory composition required')
    table=pd.DataFrame(metadata).rename(columns={'id':'__id','label':'__label','source_group':'__source','group_id':'__group','role':'__role'})
    table['__label']=pd.to_numeric(table['__label'],errors='raise')
    require(set(table['__label'])=={0,1},'Both binary labels required')
    require(table.groupby('__source')['__label'].nunique().max()==1,'Source crosses labels')
    require(table.groupby('__group')['__label'].nunique().max()==1 and table.groupby('__group')['__source'].nunique().max()==1,
            'Global group crosses labels or sources')
    for column in DESCRIPTORS:
        numbers=[]
        for row in features:
            token=row[column].strip()
            value=float('nan') if token.lower() in ('','nan','na','null') else float(token)
            require(math.isnan(value) or math.isfinite(value),'Infinite descriptor: '+column)
            numbers.append(value)
        table[column]=numbers
    return table


def population(table):
    return dict(rows=len(table),ids=sorted(table['__id']),groups=sorted(set(table['__group'])),
        label_counts={str(k):int(v) for k,v in table['__label'].value_counts().sort_index().items()},
        source_rows={str(k):int(v) for k,v in table['__source'].value_counts().sort_index().items()},
        source_groups={str(k):int(v) for k,v in table.groupby('__source')['__group'].nunique().items()})


def fold_omissions(table,folds):
    """Independently enumerate intended source/inner cells; do not fill omissions."""
    present={(f['fold_type'],f['heldout_source'],f['opposite_group_fold']) for f in folds}
    require(len(present)==len(folds),'Duplicate base fold identity')
    group_fold=table['__group'].map(lambda g:BASE.hash_fold(str(g),FOLDS,SEED))
    expected=[]
    for label,kind in ((0,'human_source_holdout'),(1,'generator_holdout')):
        expected.extend((kind,s,i) for s in sorted(table.loc[table['__label']==label,'__source'].unique()) for i in range(FOLDS))
    expected.extend(('ordinary_group_holdout_descriptive','__all_sources__',i) for i in range(FOLDS))
    require(present<=set(expected),'Unexpected base fold')
    omitted=[]
    for kind,source,inner in expected:
        if (kind,source,inner) in present: continue
        if kind=='ordinary_group_holdout_descriptive':
            test=table[group_fold==inner];train=table[group_fold!=inner]
        else:
            label=0 if kind=='human_source_holdout' else 1
            held=(table['__label']==label)&(table['__source']==source)
            test=table[(group_fold==inner)&(held|(table['__label']!=label))]
            train=table[(~held)&(group_fold!=inner)&(~table['__group'].isin(set(test['__group'])))]
        require(train['__label'].nunique()!=2 or test['__label'].nunique()!=2,'Base omitted an otherwise valid fold')
        omitted.append(dict(fold_type=kind,heldout_source=source,opposite_group_fold=inner,
            reason='one_or_both_classes_absent_in_train_or_test',train=population(train),test=population(test)))
    return omitted


def make_schedule(table,synthetic=False):
    require(set(table['__role'])=={'development'},'Role leakage into scheduling')
    folds=BASE.make_folds(table,FOLDS,SEED)
    omitted=fold_omissions(table,folds)
    schedule=[]
    for index,fold in enumerate(folds):
        previous_ids=set()
        for cap in QUANTITIES:
            train=BASE.deterministic_quantity(fold['train'],cap,SEED)
            test=fold['test']
            valid,reason=V2.valid_train(train,2)
            require(valid,'Invalid capped training fold: '+reason)
            require(not set(train['__id'])&set(test['__id']) and not set(train['__group'])&set(test['__group']),'Train/test ID/group leakage')
            require(set(train['__id'])<=set(fold['train']['__id']) and len(train)<len(table),'Capped training is not a proper base-training subset')
            if fold['fold_type']!='ordinary_group_holdout_descriptive':
                require(fold['heldout_source'] not in set(train['__source']),'Held-out source present in training')
            if cap!='all': require(train.groupby('__source')['__group'].nunique().max()<=cap,'Group cap exceeded')
            require(previous_ids<=set(train['__id']),'Training group caps are not nested')
            previous_ids=set(train['__id'])
            record=dict(fold_index=index,quantity=cap,fold_type=fold['fold_type'],heldout_source=fold['heldout_source'],
                opposite_group_fold=fold['opposite_group_fold'],train=population(train),test=population(test),
                uncapped_train=population(fold['train']),train_id_set_sha256=V2.id_set_hash(train['__id']),test_id_set_sha256=V2.id_set_hash(test['__id']))
            record['fold_uid']=digest(record)[:24]
            schedule.append((record,train,test,fold))
    accounting=dict(valid_folds=len(folds),primary_fits=len(schedule)*127,
        primary_prediction_rows=sum(len(test)*127 for _,_,test,_ in schedule),
        diagnostic_fits=sum(r['quantity']=='all' for r,*_ in schedule)*127*2,
        diagnostic_prediction_rows=sum(len(test)*127*2 for r,_,test,_ in schedule if r['quantity']=='all'))
    if not synthetic:
        require(accounting==EXPECTED_REAL,'Real schedule differs from expected43folds/counts; review before freezing')
        require(Counter(f['fold_type'] for f in folds)=={'human_source_holdout':28,'generator_holdout':10,'ordinary_group_holdout_descriptive':5},'Real fold-type counts changed')
        require(len(omitted)==2 and all(r['fold_type']=='human_source_holdout' and r['heldout_source']=='human_saraga_hindustani_v1' for r in omitted),'Expected two empty Saraga source folds')
    return schedule,omitted,accounting


def code_bindings():
    for name,expected in FIXED_CORE.items(): require(sha(ROOT/'code'/name)==expected,'Frozen numerical core changed: '+name)
    paths=[Path(__file__).resolve(),Path(PREP.__file__).resolve(),Path(PREP.P4.__file__).resolve(),
           Path(PREP.U.__file__).resolve(),Path(V2.__file__).resolve(),V2.BASE_EVALUATOR_PATH.resolve(),PROTOCOL,PREP.PROTOCOL,PARENT_PROTOCOL]
    return {str(p):sha(p) for p in paths}


def build_context(package,synthetic=False):
    package=Path(package)
    require(package.is_absolute() and package.resolve()==package and not package.is_symlink(),'Canonical package path required')
    proof=PREP.validate_package(package,synthetic)
    config=read_json(package/'family_config.json')
    require(config==PREP.P4.build_config(),'Unchanged54-feature family config required')
    table=load_table(package,synthetic)
    schedule,omitted,accounting=make_schedule(table,synthetic)
    files={str(package/name):value for name,value in proof['files_sha256'].items()}
    files.update({str(package/name):sha(package/name) for name in ('preparation_audit.json','COMMIT.json')})
    files.update(code_bindings())
    records=[record for record,*_ in schedule]
    contract=dict(schema_version=5,authorized_stage=STAGE,synthetic_test_only=synthetic,
        input_package=str(package),package_contract_sha256=proof['contract_sha256'],input_files_sha256=files,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            python_executable=str(Path(sys.executable).resolve()),python_executable_sha256=sha(Path(sys.executable).resolve()),
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),
        rows=len(table),label_counts={str(k):int(v) for k,v in table['__label'].value_counts().sort_index().items()},
        source_counts={str(k):int(v) for k,v in table['__source'].value_counts().sort_index().items()},
        families=FAMILIES,combinations=COMBINATIONS,seed=SEED,quantities=list(QUANTITIES),opposite_group_folds=FOLDS,
        fixed_ridge=10.,fixed_threshold=.5,positive_rule='score >= 0.5',prediction_link='raw_identity_unclipped',
        primary_feature_mode=MODES[0],all_cap_diagnostics=list(MODES[1:]),
        schedule=records,omitted_source_or_group_cells=omitted,schedule_sha256=digest(records),accounting=accounting,
        quantity_unit='up to cap global group IDs per source within the training fold; retain all rows of chosen source-groups',
        sample_weighting='equal classes; equal sources within class; equal groups within source; equal samples within group; sum weights = training n',
        quantity_confound='ridge stays10 while weight sum equals training n; cap changes both composition and effective regularization',
        preprocessing='train-only medians and weighted means/scales; all-missing train median0; near-zero scale1; no complete-case filtering',
        selection=False,threshold_tuning=False,full_cohort_refit=False,historical_locked_or_pilot_scoring=False,
        source_transfer_J_computed=False,source_ranking=False,winner_selection=False,
        exploratory_reuse='Mureka500 and Saraga103 are consumed exploratory development; not independent confirmation',
        metric_scope='fold-specific test rows; primary and diagnostics separate; per-source recording rates and equal-component rates both reported',
        cross_fold_reporting=dict(source_recording='within arm/candidate/cap/mode/source: correct decisions / unique out-of-fold recording count',
            source_component='within that same scope: unweighted mean of global-group out-of-fold correct rates; never mean of fold component means',
            source_holdout_two_class='mean source-pair metrics within fold, then mean valid folds within held-source arm, then equal arms within fold type',
            ordinary_two_class='mean source-pair metrics within fold, then mean ordinary folds; separate descriptive arm',
            pooled_out_of_fold_auc=False,pooling_across_held_source_arms=False,presentation_aggregation_implemented_here=False),
        publication='exclusive mkdir reservation; streamed artifacts; hardlink COMMIT last; no replacement, orphan acceptance or resume')
    require(synthetic or all(v=='1' for v in contract['runtime']['threads'].values()),'Real fitting requires all three thread settings1')
    recheck(contract)
    return contract,table,schedule


def recheck(contract):
    for path,expected in contract['input_files_sha256'].items(): require(sha(path)==expected,'Bound input/code changed: '+path)
    require(sha(contract['runtime']['python_executable'])==contract['runtime']['python_executable_sha256'],'Python executable changed')


def authorize(contract,receipt,receipt_sha256):
    require(receipt is not None and receipt_sha256 is not None,'Run requires parent-frozen exact receipt, including synthetic runs')
    require(sha(receipt)==receipt_sha256,'Frozen receipt SHA mismatch')
    value=read_json(receipt)
    require(value.get('status')=='frozen' and value.get('authorized_stage')==STAGE and value.get('contract')==contract
            and value.get('contract_sha256')==digest(contract),'Frozen input/code/runtime/fold contract mismatch')
    review=value.get('independent_review',{})
    require(review.get('approved') is True and review.get('reviewer')=='root' and isinstance(review.get('reviewed_utc'),str)
            and review['reviewed_utc'].strip(),'Independent parent review required')
    return dict(status='frozen_verified',contract_sha256=digest(contract),receipt_sha256=receipt_sha256)


def source_endpoints(test,scores,model_uid):
    result=[]
    for source,part in test.groupby('__source',sort=True):
        positions=test.index.get_indexer(part.index)
        positive=scores[positions]>=.5
        label=int(part['__label'].iloc[0]);n=len(part)
        group_rates=[]
        for group in sorted(set(part['__group'])):
            mask=part['__group'].eq(group).to_numpy()
            group_rates.append(float(positive[mask].mean()))
        pos=int(positive.sum());rate=pos/n;equal=math.fsum(group_rates)/len(group_rates)
        result.append(dict(model_uid=model_uid,source_group=source,label=label,rows=n,components=len(group_rates),threshold=.5,
            tp=pos if label else 0,tn=n-pos if not label else 0,fp=pos if not label else 0,fn=n-pos if label else 0,
            recording_positive_rate=rate,recording_negative_rate=1-rate,recording_ai_sensitivity=rate if label else None,
            recording_human_specificity=1-rate if not label else None,equal_component_positive_rate=equal,
            equal_component_negative_rate=1-equal,equal_component_ai_sensitivity=equal if label else None,
            equal_component_human_specificity=1-equal if not label else None))
    return result


class Stream:
    def __init__(self,stack,path,fields):
        self.file=stack.enter_context(path.open('x',newline=''))
        self.writer=csv.DictWriter(self.file,fieldnames=fields)
        self.writer.writeheader();self.count=0
    def write(self,record): self.writer.writerow(record);self.count+=1
    def many(self,records):
        for record in records:self.write(record)


def model_result(train,test,fold,record,combination,mode,line):
    columns=sum((FAMILIES[f] for f in combination.split('+')),[])
    model=V2.fit_candidate(train,columns,'ridge',feature_mode=mode)
    model['missing_value_policy']=POLICIES[mode]
    require(model['threshold']==.5 and model['ridge']==10. and model['feature_mode']==mode
            and model['columns']==columns and model['training_rows']==len(train),'Inherited fit contract changed')
    scores=V2.predict_candidate(test,model)
    require(scores.shape==(len(test),) and np.isfinite(scores).all(),'Invalid prediction array')
    model_sha=digest({'model':model,'threshold':.5})
    identity=dict(combination=combination,feature_mode=mode,quantity=record['quantity'],fold_uid=record['fold_uid'])
    model_uid=digest(identity)[:24]
    common=dict(model_uid=model_uid,model_sha256=model_sha,model_jsonl_line=line,**identity,
        fold_index=record['fold_index'],fold_type=record['fold_type'],heldout_source=record['heldout_source'],
        opposite_group_fold=record['opposite_group_fold'],train_id_set_sha256=record['train_id_set_sha256'],
        test_id_set_sha256=record['test_id_set_sha256'],training_rows=len(train),test_rows=len(test))
    return model,scores,common


def run_evaluation(output,table,schedule,contract,authorization):
    require(authorization.get('status')=='frozen_verified' and authorization.get('contract_sha256')==digest(contract),
            'Verified frozen authorization required before any fit')
    require([r for r,*_ in schedule]==contract['schedule'],'Schedule changed before fitting')
    count_models=0
    with ExitStack() as stack:
        streams={}
        for category in ('primary','diagnostic'):
            streams[category]={name:Stream(stack,output/(category+'_'+name+'.csv'),fields)
                for name,fields in [('model_index',INDEX),('predictions',PRED),('pair_metrics',PAIR),('pooled_metrics',POOLED),('per_source_endpoints',SOURCE)]}
        models=stack.enter_context((output/'fold_models.jsonl').open('x'))
        for record,train,test,fold in schedule:
            recheck(contract)
            # IDs follow package order; stable row indices make source slicing unambiguous.
            test=test.reset_index(drop=True)
            for combination in COMBINATIONS:
                modes=MODES if record['quantity']=='all' else MODES[:1]
                for mode in modes:
                    count_models+=1
                    model,scores,common=model_result(train,test,fold,record,combination,mode,count_models)
                    category='primary' if mode==MODES[0] else 'diagnostic';s=streams[category]
                    models.write(canonical(dict(common,model=model)).decode()+'\n')
                    s['model_index'].write(common)
                    for identity,score in zip(test[['__id','__label','__source','__group','__role']].itertuples(index=False,name=None),scores):
                        iid,label,source,group,role=identity
                        s['predictions'].write(dict(model_uid=common['model_uid'],row_id=iid,label=int(label),source_group=source,
                            group_id=group,role=role,score=format(float(score),'.17g'),threshold=.5,predicted_label=int(score>=.5)))
                    pair=V2.pair_metrics(test,scores,fold,.5,1)
                    s['pair_metrics'].many({k:(common['model_uid'] if k=='model_uid' else r[k]) for k in PAIR} for r in pair)
                    s['pooled_metrics'].write(dict(model_uid=common['model_uid'],test_rows=len(test),test_groups=test['__group'].nunique(),
                        threshold=.5,**V2.metrics_at_threshold(test['__label'].to_numpy(int),scores,.5)))
                    s['per_source_endpoints'].many(source_endpoints(test,scores,common['model_uid']))
            models.flush()
            print(json.dumps(dict(event='completed_fold_cap',fold_index=record['fold_index'],fold_type=record['fold_type'],
                heldout_source=record['heldout_source'],quantity=record['quantity'],completed_model_instances=count_models)),flush=True)
        observed={category:{name:stream.count for name,stream in group.items()} for category,group in streams.items()}
    expected=contract['accounting']
    for category in ('primary','diagnostic'):
        require(observed[category]['model_index']==expected[category+'_fits']
                and observed[category]['predictions']==expected[category+'_prediction_rows']
                and observed[category]['pooled_metrics']==expected[category+'_fits'],'Streamed model/prediction accounting mismatch')
    require(count_models==expected['primary_fits']+expected['diagnostic_fits'],'Saved model count mismatch')
    return dict(status='completed_exploratory_cv',model_instances=count_models,stream_counts=observed,
        all_models_are_training_fold_only=True,full_cohort_refit=False,selection=False,threshold_tuning=False,
        historical_locked_or_pilot_scoring=False,source_transfer_J_computed=False,
        independent_numerical_audit_performed=False)


def commit_output(root):
    paths=sorted(root.iterdir())
    require(paths and all(p.is_file() and not p.is_symlink() for p in paths) and not (root/'COMMIT.json').exists(),'Invalid/orphan output products')
    for path in paths:
        with path.open('rb') as stream:os.fsync(stream.fileno())
    marker=dict(status='committed',publication='exclusive directory reservation; hardlink COMMIT last',
        files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths})
    temporary=root/'.COMMIT.pending'
    write_json(temporary,marker)
    try:os.link(temporary,root/'COMMIT.json')
    finally:temporary.unlink()
    verify_publication(root)
    return marker


def verify_publication(root):
    marker=read_json(root/'COMMIT.json')
    require(marker['status']=='committed' and marker['publication']=='exclusive directory reservation; hardlink COMMIT last'
            and set(p.name for p in root.iterdir())==set(marker['files'])|{'COMMIT.json'},'Uncommitted/orphan/unexpected output')
    for name,record in marker['files'].items():
        path=root/name
        require(Path(name).name==name and name not in ('.','..') and path.is_file() and not path.is_symlink()
                and sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],'Published artifact changed')
    return marker


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('draft','run'),required=True)
    parser.add_argument('--package-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--synthetic-test-only',action='store_true')
    args=parser.parse_args(argv)
    require(args.output_dir.is_absolute() and args.output_dir.parent.resolve()==args.output_dir.parent
            and not args.output_dir.exists() and not args.output_dir.is_symlink(),'New canonical output path required')
    contract,table,schedule=build_context(args.package_dir,args.synthetic_test_only)
    authorization=authorize(contract,args.receipt,args.receipt_sha256) if args.stage=='run' else None
    args.output_dir.mkdir()  # Exclusive reservation. Failures remain visible uncommitted orphans.
    with (args.output_dir/'fold_registry.jsonl').open('x') as stream:
        for record,*_ in schedule:stream.write(canonical(record).decode()+'\n')
    write_json(args.output_dir/'omitted_fold_cells.json',contract['omitted_source_or_group_cells'])
    if args.stage=='draft':
        write_json(args.output_dir/'preregistration_draft.json',dict(status='draft',authorized_stage=STAGE,
            contract=contract,contract_sha256=digest(contract),fitting_started=False,independent_review=None))
    else:
        write_json(args.output_dir/'frozen_authorization.json',read_json(args.receipt))
        result=run_evaluation(args.output_dir,table,schedule,contract,authorization)
        # Full compact package revalidation and a fresh schedule precede COMMIT.
        after,_,_=build_context(args.package_dir,args.synthetic_test_only)
        require(after==contract and sha(args.receipt)==authorization['receipt_sha256'],'Inputs/code/runtime/receipt changed during fitting')
        write_json(args.output_dir/'run_manifest.json',dict(schema_version=5,stage=STAGE,
            synthetic_test_only=args.synthetic_test_only,contract=contract,authorization=authorization,**result))
    recheck(contract)
    commit_output(args.output_dir)
    print(json.dumps(dict(status='draft_no_fit' if args.stage=='draft' else 'committed_exploratory_cv',
        output=str(args.output_dir),contract_sha256=digest(contract),accounting=contract['accounting'])),flush=True)


if __name__=='__main__':main()
