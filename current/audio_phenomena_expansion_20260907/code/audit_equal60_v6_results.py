#!/usr/bin/env python3
"""Independent streaming v6 audit, without evaluator, scorer or optimizer imports.

The SHA-pinned v5 auditor supplies independent generic I/O, weights, metrics,
stationarity/replay and OOF helpers. V6 owns its schema,60-column cache,
lineage/schedule reconstruction and contracts. No imported globals are changed.
Only COMMIT-complete runs can enter audit; no classification fit is performed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import audit_equal60_v5_results as A5

require, canonical, digest = A5.require, A5.canonical, A5.digest
sha, read_json, read_csv = A5.sha256_file, A5.read_json, A5.read_small_csv
STAGE = 'exploratory_v6_native_stereo_development_cv_only'
SEED, FOLDS, CAPS, MODES = 20260907, 5, (25, 50, 100, 200, 'all'), A5.MODES
HELPER_SHA = '1a928135698d1fce75eea4792696c89b546a914994cc317381c30af1493332fe'
SC = [f'SC_{lo}_{hi}hz_{metric}_median' for lo, hi in ((80,500),(500,2000),(2000,6000))
      for metric in ('abs_iid_db','side_energy_fraction')]
FAMILIES = {**{k:list(v) for k,v in A5.FAMILIES.items()}, 'SC':SC}
DESCRIPTORS = sum(FAMILIES.values(), [])
COMBINATIONS = ['+'.join(c) for n in range(1,len(FAMILIES)+1) for c in itertools.combinations(FAMILIES,n)]
COUNTS = {k:v for k,v in A5.EXPECTED_REAL_SOURCES.items() if k != 'human_urmp'}
EXPECTED_REAL = dict(valid_folds=38, primary_fits=48450, primary_prediction_rows=14017350,
                     diagnostic_fits=19380, diagnostic_prediction_rows=5606940)
LEDGER = A5.LEDGER + ['sc_input_file_sha256','sc_measurement_sha256','sc_native_channels',
                     'sc_extraction_root','sc_gate_sha256']
PRODUCTS = {'metadata_60s.csv','features_60s.csv','origin_ledger.csv','family_config.json'}
OLD_COMMIT = '7cc03aa840c910df919e531153b61daba864e80794abc91ebf29d3859f70947a'


def bindings_check(mapping):
    require(isinstance(mapping,dict) and bool(mapping), 'Empty input bindings')
    for name, expected in mapping.items():
        p=Path(name)
        require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink()
                and A5.SHA_RE.fullmatch(str(expected)) and sha(p)==expected, 'Bound input changed: '+name)


def unique_binding(mapping, expected=None, basename=None):
    paths=[Path(p) for p,h in mapping.items() if (expected is None or h==expected)
           and (basename is None or Path(p).name==basename)]
    require(len(paths)==1, 'Ambiguous/missing lineage binding: '+str(basename or expected))
    return paths[0]


def jsonl_rows(path):
    with path.open() as stream:
        return [A5.strict_loads(line, str(path)+':'+str(i)) for i,line in enumerate(stream,1)]


def validate_gate(c):
    ref=c['external_gate']; review_ref=c['independent_gate_review']
    gate_path=Path(ref['receipt_path']); freeze_path=Path(ref['freeze_path']); raw_path=Path(ref['raw_commit_path'])
    bindings_check({str(gate_path):ref['receipt_sha256'],review_ref['receipt_path']:review_ref['receipt_sha256']})
    gate=read_json(gate_path); review=read_json(Path(review_ref['receipt_path']))
    require(gate.get('candidate')=='SC_L6' and gate.get('candidate_features')==SC
            and gate.get('measurement_gate_passed') is True and gate.get('classifier_fitted') is False
            and gate.get('decision')=='eligible_for_separately_frozen_exploratory_study'
            and gate.get('primary_checks')==gate.get('primary_passed')==196
            and gate.get('codec_pairs')==gate.get('eligible_codec_pairs')==28, 'Measurement gate failed')
    bindings_check({str(freeze_path):gate['freeze_sha256'],str(raw_path):gate['raw_commit_sha256']})
    freeze=read_json(freeze_path); raw=read_json(raw_path)
    require(freeze.get('candidate_features')==SC and freeze.get('candidate')=='SC_L6'
            and len(freeze.get('selected',[]))==7 and bool(freeze.get('bindings'))
            and raw.get('freeze_sha256')==gate['freeze_sha256'] and bool(raw.get('products')), 'Gate hash chain changed')
    require(review.get('passed') is True and review.get('classifier_fitted') is False
            and review.get('gate_receipt_sha256')==ref['receipt_sha256']
            and review.get('freeze_sha256')==gate['freeze_sha256']
            and review.get('raw_commit_sha256')==gate['raw_commit_sha256']
            and review.get('primary_checks_replayed')==196 and review.get('ratios_replayed')==168,
            'Independent gate replay failed/stale')


def validate_lineage(prep, metadata, features, ledger):
    """Reconstruct old tokens, SC tokens/provenance, and protected ID/group exclusion."""
    c=prep['contract']; bound=c['input_files_sha256']; bindings_check(bound); validate_gate(c)
    require(c.get('old_package_commit_sha256')==OLD_COMMIT, 'Unaccepted old v5 origin')
    old=unique_binding(bound,OLD_COMMIT,'COMMIT.json').parent
    old_commit=read_json(old/'COMMIT.json')
    for name,record in old_commit['files'].items():
        require(Path(name).name==name and (old/name).is_file() and not (old/name).is_symlink()
                and sha(old/name)==record['sha256'] and (old/name).stat().st_size==record['bytes'], 'Old origin publication changed')
    original_meta=read_csv(old/'metadata_60s.csv',A5.META)
    original_features=read_csv(old/'features_60s.csv',['id',*A5.DESCRIPTORS])
    original_ledger=read_csv(old/'origin_ledger.csv',A5.LEDGER)
    old_ids=[r['id'] for r in original_meta]
    require(len(old_ids)==2207 and len(set(old_ids))==2207 and old_ids==[r['id'] for r in original_features]
            ==[r['id'] for r in original_ledger], 'Old origin identities changed')
    native=unique_binding(bound,basename='native.csv')
    with native.open(newline='') as stream: protected=[r for r in csv.DictReader(stream) if r['role'] in ('locked','pilot')]
    ids=[r['id'] for r in metadata]; wanted=set(ids)
    require(len(protected)==488 and not wanted & {r['id'] for r in protected}
            and not {r['group_id'] for r in metadata} & {r['group_id'] for r in protected}, 'Protected ID/group leakage')
    require(metadata==[r for r in original_meta if r['id'] in wanted], 'Metadata differs from matched old cohort')
    old_features={r['id']:r for r in original_features}; old_ledgers={r['id']:r for r in original_ledger}
    ecommit_path=unique_binding(bound,c['extraction_commit_sha256'],'COMMIT.json'); extraction=ecommit_path.parent
    ec=read_json(ecommit_path); freeze_path=unique_binding(bound,ec['freeze_sha256']); frozen=read_json(freeze_path)
    require(frozen.get('status')=='frozen_for_measurement_only' and frozen.get('candidate_features')==SC, 'Extraction freeze changed')
    require(bool(frozen.get('bindings')) and all(bound.get(p)==h for p,h in frozen['bindings'].items()),
            'Extraction origin/code bindings absent from preparation')
    # Hash all compact measurements and frame arrays, not raw audio or neural outputs.
    products=ec['products']; actual={str(p.relative_to(extraction)) for p in extraction.rglob('*') if p.is_file() and p!=ecommit_path}
    require(actual==set(products), 'Extraction products missing/extra')
    for name,record in products.items():
        p=extraction/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and p.resolve()==p
                and not p.is_symlink() and p.stat().st_size==record['bytes'] and sha(p)==record['sha256'], 'SC extraction product changed')
    summary=read_json(extraction/'summary.json'); measurements=jsonl_rows(extraction/'selected_features.jsonl')
    require(summary.get('rows')==2174 and summary.get('classifier_fitted') is False
            and summary.get('quality_did_not_filter_rows') is True and summary.get('candidate_features')==SC
            and summary.get('freeze_sha256')==ec['freeze_sha256']
            and summary.get('selected_jsonl_sha256')==sha(extraction/'selected_features.jsonl'), 'SC extraction summary changed')
    selected=frozen['selected']; excluded=frozen['excluded']
    require(ids==[r['metadata']['id'] for r in selected]==[r['id'] for r in measurements], 'SC extraction ID order/coverage changed')
    require(len(excluded)==33 and set(old_ids)-wanted=={r['metadata']['id'] for r in excluded}
            and all(r['native_channels']==1 and r['metadata']['source_group']=='human_urmp' for r in excluded), 'Native-mono exclusion changed')
    for m,f,l,s,x in zip(metadata,features,ledger,selected,measurements):
        iid=m['id']; item=extraction/'items'/iid/'measurement.json'; measured=read_json(item)
        require(all(f[k]==old_features[iid][k] for k in ['id',*A5.DESCRIPTORS]), 'Original54 feature token changed')
        require(all(l[k]==old_ledgers[iid][k] for k in A5.LEDGER), 'Original ledger changed')
        require(s['metadata']==m and s['native_channels']==x['native_channels']==2
                and all(x[k]==m[k] for k in ('id','label','source_group','group_id'))
                and x['input_path']==s['standardized_path'] and x['input_file_sha256']==s['standardized_file_sha256'], 'SC cohort/channel/input identity changed')
        require(all(measured[k]==x[k] for k in x) and set(x['candidate_features'])==set(SC), 'SC measurement/JSONL mismatch')
        for key in SC:
            value=x['candidate_features'][key]
            require(value is None or (isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)), 'Nonfinite SC measurement')
            require(f[key]==('' if value is None else format(value,'.17g')), 'SC feature token differs from bound measurement')
        require(x['candidate_missing']==[key for key in SC if x['candidate_features'][key] is None], 'SC missingness declaration changed')
        require(l['sc_input_file_sha256']==x['input_file_sha256'] and l['sc_measurement_sha256']==sha(item)
                and l['sc_native_channels']=='2' and l['sc_extraction_root']==str(extraction)
                and l['sc_gate_sha256']==c['external_gate']['receipt_sha256'], 'SC provenance ledger changed')
    require(c.get('feature_tokens_sha256')==digest(features) and c.get('metadata_sha256')==digest(metadata)
            and c.get('origin_ledger_sha256')==digest(ledger) and c.get('descriptor_columns')==DESCRIPTORS
            and c.get('native_mono_excluded')==33 and c.get('historical_protected_id_count')==488
            and c.get('rows')==len(metadata) and c.get('source_counts')==dict(Counter(r['source_group'] for r in metadata))
            and summary.get('source_counts')==c['source_counts']
            and summary.get('candidate_missing_counts')=={key:sum(key in x['candidate_missing'] for x in measurements) for key in SC},
            'Preparation content digests/denominators changed')


def load_package(package, synthetic=False):
    package=Path(package)
    require(package.is_absolute() and package.resolve()==package and not package.is_symlink(), 'Canonical package required')
    marker=read_json(package/'COMMIT.json'); names=PRODUCTS|{'preparation_audit.json'}
    require(marker.get('status')=='committed' and set(marker.get('files',{}))==names
            and {p.name for p in package.iterdir()}==names|{'COMMIT.json'}, 'Incomplete/orphan package')
    for name,record in marker['files'].items():
        p=package/name
        require(p.is_file() and not p.is_symlink() and record=={'sha256':sha(p),'bytes':p.stat().st_size}, 'Package product changed: '+name)
    prep=read_json(package/'preparation_audit.json'); c=prep['contract']
    require(prep.get('schema_version')==6 and prep.get('status')=='prepared_not_authorized_for_fitting'
            and prep.get('synthetic_test_only') is synthetic and prep.get('fitting_authorized') is False
            and prep.get('scoring_authorized') is False and prep.get('contract_sha256')==digest(c), 'Preparation proof changed')
    require(prep.get('files_sha256')=={n:marker['files'][n]['sha256'] for n in PRODUCTS}, 'Preparation product binding mismatch')
    require(read_json(package/'family_config.json')==dict(schema_version=6,families=FAMILIES), 'Exact60 feature mapping changed')
    metadata=read_csv(package/'metadata_60s.csv',A5.META); features=read_csv(package/'features_60s.csv',['id',*DESCRIPTORS])
    ledger=read_csv(package/'origin_ledger.csv',['id'] if synthetic else LEDGER)
    ids=[r['id'] for r in metadata]
    require(ids==[r['id'] for r in features]==[r['id'] for r in ledger] and len(ids)==len(set(ids)), 'Package ID/order/duplicates changed')
    require(all(r['id'].strip() and r['group_id'].strip() and r['source_group'].strip() and r['role']=='development'
                and r['duration_view']=='60s' and r['label'] in ('0','1') for r in metadata), 'Metadata identity/role changed')
    for flag in ('lineage_validated','original_54_values_preserved','native_stereo_only','historical_locked_pilot_ids_and_groups_excluded'):
        require(c.get(flag) is True, 'Lineage flag absent: '+flag)
    bindings_check(c['input_files_sha256'])
    if synthetic:
        require(c.get('synthetic_fixture_only') is True and 0<len(ids)<=16
                and all(i.startswith('synthetic_v6_') for i in ids), 'Synthetic16 boundary')
    else:
        require(len(ids)==2174 and Counter(r['source_group'] for r in metadata)==COUNTS
                and Counter(r['label'] for r in metadata)=={'0':1278,'1':896}, 'Exact2174 cohort changed')
        validate_lineage(prep,metadata,features,ledger)
    table=pd.DataFrame(metadata).rename(columns={'id':'__id','label':'__label','source_group':'__source','group_id':'__group','role':'__role'})
    table['__label']=pd.to_numeric(table['__label'],errors='raise').astype(int)
    require(set(table['__label'])=={0,1} and table.groupby('__source')['__label'].nunique().max()==1
            and table.groupby('__group')['__label'].nunique().max()==1
            and table.groupby('__group')['__source'].nunique().max()==1, 'Source/group crosses label/source')
    for column in DESCRIPTORS:
        values=[math.nan if r[column].strip().lower() in ('','nan','na','null') else float(r[column]) for r in features]
        require(all(math.isnan(v) or math.isfinite(v) for v in values), 'Infinite descriptor')
        table[column]=values
    require(prep.get('rows')==len(table), 'Preparation row count changed')
    return table,prep,sha(package/'COMMIT.json')


class TrainingCache(A5.TrainingCache):
    """Same independent transform calculations, explicit60-column input cache."""
    def __init__(self, train):
        self.train=train.reset_index(drop=True); self.raw=self.train[DESCRIPTORS].to_numpy(float)
        self.missing=~np.isfinite(self.raw)
        self.medians=np.asarray([float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else 0. for c in self.raw.T])
        self.imputed=np.where(self.missing,self.medians,self.raw); self.weights=A5.independent_weights(self.train)
        self.y=self.train['__label'].to_numpy(float)
        self.value_mean=np.average(self.imputed,axis=0,weights=self.weights)
        self.value_scale=np.sqrt(np.average((self.imputed-self.value_mean)**2,axis=0,weights=self.weights))
        self.value_scale[self.value_scale<1e-8]=1.
        miss=self.missing.astype(float); self.missing_mean=np.average(miss,axis=0,weights=self.weights)
        self.missing_scale=np.sqrt(np.average((miss-self.missing_mean)**2,axis=0,weights=self.weights))
        self.missing_scale[self.missing_scale<1e-8]=1.
        self.positions={c:i for i,c in enumerate(DESCRIPTORS)}
        self.source_counts=Counter((self.train['__label'].astype(int).astype(str)+':'+self.train['__source'].astype(str)).tolist())


def build_schedule(table):
    require(set(table['__role'])=={'development'}, 'Role leakage')
    folds,omitted=A5.base_folds(table); schedule=[]
    for index,fold in enumerate(folds):
        previous=set()
        for cap in CAPS:
            train,test=A5.cap_training(fold['train'],cap),fold['test'].copy()
            require(A5.valid_train(train),'Invalid capped training')
            require(not set(train['__id'])&set(test['__id']) and not set(train['__group'])&set(test['__group']), 'ID/group leakage')
            require(set(train['__id'])<=set(fold['train']['__id']) and previous<=set(train['__id']), 'Non-nested training subset')
            require(fold['heldout_source']=='__all_sources__' or fold['heldout_source'] not in set(train['__source']), 'Held source leakage')
            require(cap=='all' or train.groupby('__source')['__group'].nunique().max()<=cap, 'Cap exceeded')
            previous=set(train['__id'])
            r=dict(fold_index=index,quantity=cap,fold_type=fold['fold_type'],heldout_source=fold['heldout_source'],
                   opposite_group_fold=fold['opposite_group_fold'],train=A5.population(train),test=A5.population(test),
                   uncapped_train=A5.population(fold['train']),train_id_set_sha256=A5.id_set_hash(train['__id']),test_id_set_sha256=A5.id_set_hash(test['__id']))
            r['fold_uid']=digest(r)[:24]; schedule.append((r,train,test,fold))
    n=len(COMBINATIONS)
    counts=dict(valid_folds=len(folds),primary_fits=len(schedule)*n,primary_prediction_rows=sum(len(t)*n for _,_,t,_ in schedule),
                diagnostic_fits=sum(r['quantity']=='all' for r,*_ in schedule)*n*2,
                diagnostic_prediction_rows=sum(len(t)*n*2 for r,_,t,_ in schedule if r['quantity']=='all'))
    return schedule,omitted,counts


def validate_contract(c,table,prep,package,schedule,omitted,counts,synthetic):
    records=[r for r,*_ in schedule]
    require(c.get('schema_version')==6 and c.get('authorized_stage')==STAGE and c.get('synthetic_test_only') is synthetic, 'Contract stage changed')
    require(c.get('input_package')==str(package) and c.get('package_contract_sha256')==prep['contract_sha256']
            and c.get('rows')==len(table) and c.get('population')==A5.population(table), 'Contract cohort/package changed')
    require(c.get('families')==FAMILIES and c.get('combinations')==COMBINATIONS and c.get('quantities')==list(CAPS)
            and c.get('seed')==SEED and c.get('opposite_group_folds')==FOLDS, 'Contract family/grid changed')
    require(c.get('fixed_ridge')==10. and c.get('fixed_threshold')==.5 and c.get('positive_rule')=='score >= 0.5'
            and c.get('prediction_link')=='raw_identity_unclipped' and c.get('primary_feature_mode')==MODES[0]
            and c.get('all_cap_diagnostics')==list(MODES[1:]), 'Contract numerical method changed')
    require(c.get('schedule')==records and c.get('schedule_sha256')==digest(records)
            and c.get('accounting')==counts and c.get('omitted_source_or_group_cells')==omitted, 'Independent schedule/accounting mismatch')
    for flag in ('selection','threshold_tuning','full_cohort_refit','historical_locked_or_pilot_scoring','source_transfer_J_computed','source_ranking','winner_selection'):
        require(c.get(flag) is False,'Forbidden contract flag: '+flag)
    files=c['input_files_sha256']; bindings_check(files); A5.validate_fixed_core_bindings(files)
    for path,expected in prep['contract']['input_files_sha256'].items():
        require(files.get(path)==expected,'Preparation lineage not frozen by evaluation')
    for name in PRODUCTS|{'COMMIT.json','preparation_audit.json'}:
        require(files.get(str(package/name))==sha(package/name),'Package artifact not frozen: '+name)
    for name in ('evaluate_new_phenomena_v6.py','evaluate_new_phenomena_v5.py','prepare_evaluation_inputs_v6.py','prepare_evaluation_inputs_v5.py',
                 'prepare_evaluation_inputs_v4.py','present_saraga_mureka_frozen_transfer_v1.py','EXPLORATORY_V6_EVALUATOR_PROTOCOL_EN.md'):
        unique_binding(files,basename=name)
    runtime=c['runtime']; exe=Path(runtime['python_executable'])
    require(exe.is_absolute() and exe.resolve()==exe and exe.is_file() and sha(exe)==runtime['python_executable_sha256'], 'Frozen executable changed')
    require(all(isinstance(runtime.get(k),str) and runtime[k] for k in ('python','numpy','pandas')), 'Incomplete frozen runtime')
    if not synthetic:
        require(runtime['threads']=={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}, 'Frozen threads changed')
        require(counts==EXPECTED_REAL and Counter(r['fold_type'] for r,*_ in schedule if r['quantity']=='all')
                =={'human_source_holdout':23,'generator_holdout':10,'ordinary_group_holdout_descriptive':5}, 'Real38 accounting changed')
        require({(r['fold_type'],r['heldout_source'],r['opposite_group_fold']) for r in omitted}
                =={('human_source_holdout','human_saraga_hindustani_v1',i) for i in (0,4)}, 'Real omitted cells changed')
        for refname in ('external_gate','independent_gate_review'):
            ref=prep['contract'][refname]
            require(files.get(ref['receipt_path'])==ref['receipt_sha256'], 'Gate/review absent from execution freeze')


def validate_authorization(path,expected,manifest,c,saved):
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path)==expected,'Frozen receipt hash changed')
    value=read_json(path); review=value.get('independent_review',{})
    require(value.get('status')=='frozen' and value.get('authorized_stage')==STAGE and value.get('contract')==c
            and value.get('contract_sha256')==digest(c) and saved==value,'Frozen authorization mismatch')
    require(review.get('approved') is True and review.get('reviewer')=='root'
            and isinstance(review.get('reviewed_utc'),str) and review['reviewed_utc'].strip(),'Parent review absent')
    require(manifest.get('authorization')==dict(status='frozen_verified',contract_sha256=digest(c),receipt_sha256=expected),'Authorization proof changed')


def audit_streams(result,marker,schedule,counts):
    """Replay every saved model/row and recompute all fold/source/OOF endpoints."""
    checked={}; streams={}; model_stream=None; oof=A5.OOFAccumulator(schedule)
    nmodels=npred=nprimary=ndiag=0; max_residual=max_error=0.; min_margin=math.inf
    fold_stream=A5.StrictJSONLStream(result/'fold_registry.jsonl',marker['files']['fold_registry.jsonl']['sha256'],marker['files']['fold_registry.jsonl']['bytes'])
    try:
        for i,(record,*_) in enumerate(schedule): require(fold_stream.next(str(i))==record,'Saved fold registry mismatch')
        checked['fold_registry.jsonl']=fold_stream.finish()
    finally: fold_stream.close()
    try:
        for category in ('primary','diagnostic'):
            streams[category]={}
            for name,fields in (('model_index',A5.INDEX),('predictions',A5.PRED),('pair_metrics',A5.PAIR),('pooled_metrics',A5.POOLED),('per_source_endpoints',A5.SOURCE)):
                filename=category+'_'+name+'.csv'; p=marker['files'][filename]
                streams[category][name]=A5.StrictCSVStream(result/filename,fields,p['sha256'],p['bytes'])
        p=marker['files']['fold_models.jsonl']; model_stream=A5.StrictJSONLStream(result/'fold_models.jsonl',p['sha256'],p['bytes'])
        for record,train,original_test,_ in schedule:
            test=original_test.reset_index(drop=True); cache=TrainingCache(train)
            identities=list(test[['__id','__label','__source','__group','__role']].itertuples(index=False,name=None))
            for combination in COMBINATIONS:
                columns=sum((FAMILIES[f] for f in combination.split('+')),[])
                for mode in (MODES if record['quantity']=='all' else MODES[:1]):
                    nmodels+=1; category='primary' if mode==MODES[0] else 'diagnostic'; group=streams[category]
                    nprimary+=int(category=='primary'); ndiag+=int(category=='diagnostic')
                    saved=model_stream.next('model '+str(nmodels)); require(isinstance(saved,dict) and 'model' in saved,'Malformed saved model')
                    model=saved['model']; common=A5.expected_common(record,combination,mode,nmodels,model)
                    require(set(saved)==set(common)|{'model'} and all(saved[k]==v for k,v in common.items()),'Model identity/order/hash changed')
                    A5.compare_record(group['model_index'].next('index'),common,
                        integer={'model_jsonl_line','fold_index','opposite_group_fold','training_rows','test_rows'},context='index')
                    _,residual=A5.validate_model(model,cache,columns,mode); max_residual=max(max_residual,residual)
                    replay=A5.replay_scores(test,columns,mode,model); scores=np.empty(len(test)); decisions=np.empty(len(test),int)
                    for i,(identity,replayed) in enumerate(zip(identities,replay)):
                        row=group['predictions'].next('prediction'); iid,label,source,gid,role=identity
                        expected=dict(model_uid=common['model_uid'],row_id=str(iid),label=int(label),source_group=str(source),group_id=str(gid),role=str(role))
                        score,decision=A5.validate_prediction_row(row,expected,float(replayed),'model '+str(nmodels)+' row '+str(i))
                        scores[i]=score; decisions[i]=decision; npred+=1
                        max_error=max(max_error,abs(score-float(replayed))); min_margin=min(min_margin,abs(score-.5))
                    for expected in A5.expected_pair_rows(test,scores,record,common['model_uid']):
                        A5.compare_record(group['pair_metrics'].next('pair'),expected,
                            integer={'test_human','test_ai','test_human_groups','test_ai_groups','tp','tn','fp','fn'},
                            floating={'threshold','roc_auc','balanced_accuracy','ai_sensitivity','human_specificity'},context='pair')
                    expected=dict(model_uid=common['model_uid'],test_rows=len(test),test_groups=int(test['__group'].nunique()),threshold=.5,
                                  **A5.independent_metrics(test['__label'],scores))
                    A5.compare_record(group['pooled_metrics'].next('pooled'),expected,integer={'test_rows','test_groups','tp','tn','fp','fn'},
                        floating={'threshold','roc_auc','balanced_accuracy','ai_sensitivity','human_specificity'},context='pooled')
                    for expected in A5.expected_source_rows(test,scores,common['model_uid']):
                        A5.compare_record(group['per_source_endpoints'].next('source'),expected,integer={'label','rows','components','tp','tn','fp','fn'},
                            floating={'threshold','recording_positive_rate','recording_negative_rate','recording_ai_sensitivity','recording_human_specificity',
                                      'equal_component_positive_rate','equal_component_negative_rate','equal_component_ai_sensitivity','equal_component_human_specificity'},context='source')
                    oof.add(category,combination,mode,record['quantity'],record,test,decisions)
            print(canonical(dict(event='audited_fold_cap',fold_index=record['fold_index'],quantity=record['quantity'],models_checked=nmodels,predictions_checked=npred)).decode(),flush=True)
        require(nprimary==counts['primary_fits'] and ndiag==counts['diagnostic_fits']
                and nmodels==nprimary+ndiag and npred==counts['primary_prediction_rows']+counts['diagnostic_prediction_rows'], 'Stream model/row accounting changed')
        checked['fold_models.jsonl']=model_stream.finish(); stream_counts={}
        for category,group in streams.items():
            stream_counts[category]={}
            for name,stream in group.items():
                checked[category+'_'+name+'.csv']=stream.finish(); stream_counts[category][name]=stream.count
    finally:
        if model_stream is not None:model_stream.close()
        for group in streams.values():
            for stream in group.values():stream.close()
    return dict(total=nmodels,primary=nprimary,diagnostic=ndiag,scores_replayed=npred,
                training_transforms_checked=nmodels,normal_equation_residual_checked=nmodels,
                maximum_normal_equation_absolute_residual=max_residual,maximum_absolute_score_replay_error=max_error,
                minimum_absolute_saved_score_distance_from_threshold=min_margin),stream_counts,checked,oof.finalize()


def audit(result_dir,package_dir,receipt_path,receipt_sha,synthetic=False):
    result,package,receipt_path=Path(result_dir),Path(package_dir),Path(receipt_path)
    self_path=Path(__file__).resolve(); helper=Path(A5.__file__).resolve()
    require(sha(helper)==HELPER_SHA,'Immutable independent helper changed')
    auditor_hash=sha(self_path); marker,snapshots=A5.verify_result_commit(result)
    table,prep,package_commit=load_package(package,synthetic)
    manifest=read_json(result/'run_manifest.json'); c=manifest['contract']
    schedule,omitted,counts=build_schedule(table)
    validate_contract(c,table,prep,package,schedule,omitted,counts,synthetic)
    validate_authorization(receipt_path,receipt_sha,manifest,c,read_json(result/'frozen_authorization.json'))
    require(manifest.get('schema_version')==6 and manifest.get('stage')==STAGE and manifest.get('synthetic_test_only') is synthetic
            and manifest.get('status')=='completed_exploratory_cv','Run stage/status changed')
    require(manifest.get('model_instances')==counts['primary_fits']+counts['diagnostic_fits']
            and manifest.get('all_models_are_training_fold_only') is True
            and manifest.get('independent_numerical_audit_performed') is False
            and all(manifest.get(k) is False for k in ('full_cohort_refit','selection','threshold_tuning')), 'Run scope/accounting changed')
    require(read_json(result/'omitted_fold_cells.json')==omitted,'Saved omissions changed')
    models,streams,checked,oof=audit_streams(result,marker,schedule,counts)
    require(manifest.get('stream_counts')==streams,'Run stream-count declarations changed')
    for name,record in marker['files'].items():
        if name not in checked:checked[name]=sha(result/name)
        require(checked[name]==record['sha256'],'Committed product changed: '+name)
    require(set(checked)==A5.EXPECTED_RESULT_FILES,'Unchecked result product')
    A5.recheck_snapshots(result,snapshots); bindings_check(c['input_files_sha256'])
    require(sha(receipt_path)==receipt_sha and sha(self_path)==auditor_hash and sha(helper)==HELPER_SHA,'Audit dependency changed during audit')
    require(sha(package/'COMMIT.json')==package_commit,'Package changed during audit')
    require(sha(Path(c['runtime']['python_executable']))==c['runtime']['python_executable_sha256'],
            'Frozen executable changed during audit')
    checked['COMMIT.json']=sha(result/'COMMIT.json')
    executable=Path(sys.executable).resolve()
    return dict(status='passed',schema_version=6,stage=STAGE,synthetic_test_only=synthetic,rows=len(table),
        label_counts=c['population']['label_counts'],source_counts=c['population']['source_rows'],
        families={k:len(v) for k,v in FAMILIES.items()},candidates=len(COMBINATIONS),caps=list(CAPS),
        valid_folds=counts['valid_folds'],omitted_fold_cells=len(omitted),accounting=counts,
        contract_sha256=digest(c),schedule_sha256=c['schedule_sha256'],frozen_receipt_sha256=receipt_sha,
        package_commit_sha256=package_commit,result_commit_sha256=checked['COMMIT.json'],
        run_manifest_sha256=checked['run_manifest.json'],result_files_sha256=checked,
        models=models,streams=streams,oof_source_endpoints=oof,model_fitting_performed=False,scorer_predict_called=False,
        training_stationarity_checked=True,metrics_independently_recomputed=True,
        protected_metadata_and_feature_lineage_independently_reconstructed=not synthetic,
        auditor_sha256=auditor_hash,immutable_helper_sha256=HELPER_SHA,
        audit_runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                           python_executable=str(executable),python_executable_sha256=sha(executable)),
        interpretation='exploratory matched native-stereo cohort; source arms separate; no pooled cross-model OOF AUC')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('result-dir','package-dir','frozen-receipt','audit-output'):
        parser.add_argument('--'+name,type=lambda p:Path(p).absolute(),required=True)
    parser.add_argument('--frozen-receipt-sha256',required=True)
    parser.add_argument('--synthetic-test-only',action='store_true')
    parser.add_argument('--parent-authorized-completed-audit',action='store_true')
    args=parser.parse_args(argv)
    require(args.parent_authorized_completed_audit,'Explicit parent authorization required for completed-run audit')
    A5.validate_new_receipt_path(args.audit_output)
    require(not args.audit_output.is_relative_to(args.result_dir),'Audit receipt must be outside immutable results')
    receipt=audit(args.result_dir,args.package_dir,args.frozen_receipt,args.frozen_receipt_sha256,args.synthetic_test_only)
    A5.publish_receipt(args.audit_output,receipt)
    print(canonical(dict(status='passed',audit_output=str(args.audit_output),audit_sha256=sha(args.audit_output),models=receipt['models']['total'])).decode())


if __name__=='__main__':main()
