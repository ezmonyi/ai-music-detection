"""YuE2-expanded schedule executor using the original fit and receipt audit APIs.

Default preflight does not fit. Run requires an independently supplied freeze
whose contract exactly matches the verified feature package and schedule.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code')
RD=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0,str(RC))
import evaluate_native30_bc_v1 as evaluator
import pandas as pd
import numpy as np
io=evaluator.io
PLAN_SHA='b1b3619b0b0d54e56f1e054541e388a35422b4b2ca0f2267b8496072b6fc4156'
_TABLE=None
_CONTRACT=None
_OUT=None


def verified(root):
    commit=io.read_json(root/'COMMIT.json')
    for name,b in commit['products'].items():assert io.file_binding(root/name)==b
    return commit


def build():
    assert io.digest(RC/'evaluate_native30_bc_v1.py')=='a40d9074d9c017423aa002778ae857a155e52ad8a49fe7cef9a865b292d8b1b1'
    plan=RD/'expanded_native30_plan_v1'
    assert io.digest(plan/'COMMIT.json')==PLAN_SHA
    verified(plan)
    package=RD/'expanded_native30_features_v1'
    verified(package)
    metadata=io.read_json(package/'metadata.json')
    features=io.read_json(package/'features.json')
    locked=io.read_json(package/'locked_metadata.json')
    assert len(metadata)==len(features)==4228 and len(locked)==100
    assert all(r['role']=='development' for r in metadata)
    assert {r['id'] for r in metadata}.isdisjoint(r['id'] for r in locked)
    lookup={r['id']:r for r in features}
    assert len(lookup)==4228
    table=pd.DataFrame(metadata).rename(columns={'id':'__id','label':'__label','source_group':'__source',
             'group_id':'__group','component_id':'__component','role':'__role'})
    table['__label']=table['__label'].astype(int)
    for name in evaluator.FEATURES:
        table[name]=[np.nan if lookup[uid][name] is None else lookup[uid][name] for uid in table['__id']]
    assert len(evaluator.FEATURES)==55
    schedule=io.read_json(plan/'schedule.json')
    assert schedule['combinations']==evaluator.COMBINATIONS and schedule['quantities']==[25,50,100,200,'all']
    caps={r['schedule_uid']:r for r in schedule['cap_schedules']}
    folds={r['fold_uid']:r for r in schedule['fold_cells']}
    tasks=[]
    for cell in schedule['combination_schedule_cells']:
        cap=caps[cell['schedule_uid']];fold=folds[cap['fold_uid']]
        tasks.append({**cell,'quantity':cap['quantity'],'fold_uid':cap['fold_uid'],
                      'fold_type':cap['fold_type'],'heldout_source':cap['heldout_source'],
                      'opposite_group_fold':cap['opposite_group_fold'],
                      'train_ids':cap['train']['ids'],'test_ids':fold['test']['ids'],
                      'train_id_set_sha256':cap['train']['id_set_sha256'],
                      'test_id_set_sha256':fold['test']['id_set_sha256'],
                      'omission_reasons':cap['omission_reasons']})
    valid=[task for task in tasks if task['status']=='eligible_metadata_cell']
    assert len(valid)==112455 and len(tasks)==116025
    # Validate class, exact ID hash, group and component exclusion for each cap once.
    checked=set()
    for task in valid:
        if task['schedule_uid'] not in checked:
            evaluator.CORE.task_tables(table,task)
            checked.add(task['schedule_uid'])
    contract=dict(version='yue2_expanded_native30_evaluation_v1',
                  driver=io.file_binding(Path(__file__).resolve()),
                  evaluator=io.file_binding(RC/'evaluate_native30_bc_v1.py'),
                  plan_commit=io.file_binding(plan/'COMMIT.json'),
                  feature_commit=io.file_binding(package/'COMMIT.json'),
                  expected_models=112455,primary_models=80325,diagnostic_models=32130,
                  omitted_models=3570,development_rows=4228,locked_rows_excluded=100,
                  families=evaluator.FAMILIES,ridge=10.0,threshold=0.5,
                  table_sha256=io.value_hash(evaluator.table_payload(table)),
                  runtime={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                           'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}},
                  winner_selection=False,threshold_tuning=False,full_cohort_refit=False,
                  historical_locked_scoring=False,independent_implementation_audit=False)
    return contract,table,tasks


def one(task):
    uid=evaluator.task_uid(task,io.value_hash(_CONTRACT))
    path=_OUT/'models'/(uid+'.json')
    if not path.exists():
        payload=evaluator.model_receipt(_CONTRACT,_TABLE,task)
        io.write_new(path,{'payload':payload,'receipt_sha256':io.value_hash(payload)})
    evaluator.verify_model_receipt(path,_CONTRACT,_TABLE,task)
    return uid,io.file_binding(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['preflight','run'],default='preflight')
    parser.add_argument('--wait-for-features',action='store_true')
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    while args.wait_for_features and not (RD/'expanded_native30_features_v1/COMMIT.json').exists():
        try:os.kill(2968633,0)
        except ProcessLookupError:raise RuntimeError('Feature assembler ended without COMMIT')
        time.sleep(30)
    global _CONTRACT,_TABLE,_OUT
    _CONTRACT,_TABLE,tasks=build()
    preflight=dict(status='preflight_passed_no_fitting',contract=_CONTRACT,contract_sha256=io.value_hash(_CONTRACT))
    if args.mode=='preflight':
        io.write_new(RD/'expanded_native30_evaluation_preflight_v1.json',preflight)
        print(json.dumps({'status':preflight['status'],'models':112455}),flush=True)
        return
    assert args.freeze and io.digest(args.freeze)==args.freeze_sha256
    freeze=io.read_json(args.freeze)
    assert freeze['fitting_authorized'] is True and freeze['contract']==_CONTRACT
    assert freeze['contract_sha256']==io.value_hash(_CONTRACT)
    _OUT=RD/'expanded_native30_evaluation_v1'
    with io.writer_lock(_OUT):
        path=_OUT/'contract.json'
        if path.exists():assert io.read_json(path)==_CONTRACT
        else:io.write_new(path,_CONTRACT)
        (_OUT/'models').mkdir(exist_ok=True)
        valid=[task for task in tasks if task['status']=='eligible_metadata_cell']
        bound={}
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
            for uid,binding in pool.map(one,valid,chunksize=10):
                bound[uid]=binding
                if len(bound)%100==0:print(f'Expanded evaluated and replay-verified {len(bound)}/112455',flush=True)
        assert len(bound)==112455
        final_contract,_,_=build()
        assert final_contract==_CONTRACT
        for entry in bound.values():assert io.file_binding(entry['path'])==entry
        io.write_new(_OUT/'COMMIT.json',dict(status='all_scheduled_models_and_prediction_replays_complete',
                      models=112455,model_receipts=bound,contract=io.file_binding(path),
                      freeze=io.file_binding(args.freeze),locked_test_scored=False))
        print('Expanded evaluation complete; reporting remains.',flush=True)


if __name__=='__main__':main()
