"""Prediction-only locked YuE2 sensitivity evaluation of all fixed old models."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code')
ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OLD = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/native30_bc_evaluation_v1_20260911')
sys.path.insert(0, str(RC))
import evaluate_native30_bc_v1 as evaluator
import summarize_native30_evaluation_bc_v3 as report
io = evaluator.io
PREFLIGHT = 'f47d130c47602762c1e05264b7700ecafaa8cc7a9a0cfdcd3c9473997cd47949'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    assert io.digest(args.freeze) == args.freeze_sha256
    freeze = io.read_json(args.freeze)
    assert freeze['scoring_authorized'] is True and freeze['classifier_fits'] == 0
    assert freeze['preflight_sha256'] == PREFLIGHT
    assert io.digest(Path(__file__)) == freeze['scorer_sha256']
    planpath = ROOT/'locked_yue2_preflight_v1.json'
    assert io.digest(planpath) == PREFLIGHT
    plan = io.read_json(planpath)
    assert io.file_binding(OLD/'COMMIT.json') == plan['original_evaluation']
    package = ROOT/'expanded_native30_features_v1'
    assert io.file_binding(package/'COMMIT.json') == plan['feature_package']
    products = io.read_json(package/'COMMIT.json')['products']
    for name in ['metadata.json','locked_metadata.json','locked_features.json']:
        assert io.file_binding(package/name) == products[name]
    metadata = io.read_json(package/'locked_metadata.json')
    assert [r['id'] for r in metadata] == plan['locked_ids'] and len(metadata) == 100
    assert all(r['role']=='locked_test' and int(r['label'])==1 for r in metadata)
    development = {r['id']:r for r in io.read_json(package/'metadata.json')}
    features = {r['id']:r for r in io.read_json(package/'locked_features.json')}
    table = pd.DataFrame(metadata).rename(columns={'id':'__id','label':'__label','source_group':'__source',
        'group_id':'__group','component_id':'__component','role':'__role'})
    for name in evaluator.FEATURES:
        table[name] = [np.nan if features[uid][name] is None else features[uid][name] for uid in table['__id']]
    commit = io.read_json(OLD/'COMMIT.json')
    contract = io.read_json(OLD/'contract.json')
    contract_sha = io.value_hash(contract)
    out = ROOT/'locked_yue2_scoring_v1'
    out.mkdir(exist_ok=False)
    (out/'models').mkdir()
    summary = []
    for i, selection in enumerate(plan['models'],1):
        uid = selection['model_uid']
        entry = commit['products']['models/'+uid+'.json']
        record = report.receipt(Path(entry['path']), entry, contract_sha, report.EXPECTED)
        task = record['task']
        assert task['quantity']=='all' and task['feature_mode']=='values_plus_missing'
        assert task['fold_type']=='ordinary_group_holdout_descriptive'
        assert task['combination']==selection['combination'] and task['opposite_group_fold']==selection['fold']
        assert io.value_hash(task['train_ids']) == selection['training_ids_sha256']
        training = [development[uid] for uid in task['train_ids']]
        for key in ['id','group_id','component_id']:
            assert {r[key] for r in training}.isdisjoint(r[key] for r in metadata)
        predictions = evaluator.prediction_evidence(table, record['model'])
        assert len(predictions)==100 and all(p['role']=='locked_test' for p in predictions)
        # Deterministic replay, no fit; retain all scores and exact model source.
        assert predictions == evaluator.prediction_evidence(table, record['model'])
        sensitivity = sum(p['predicted_label'] for p in predictions)/100
        result = dict(model_uid=uid, combination=selection['combination'],fold=selection['fold'],
            ai_sensitivity=sensitivity, human_specificity=None, balanced_accuracy=None, roc_auc=None,
            source_model=entry, predictions=predictions, classifier_fits=0)
        io.write_new(out/'models'/(uid+'.json'),result)
        summary.append({k:v for k,v in result.items() if k not in ['predictions','source_model']})
        if i%100==0:
            print(f'Locked scoring {i}/1275 models',flush=True)
    assert len(summary)==1275
    io.write_new(out/'summary.json',dict(status='ai_only_locked_sensitivity_not_two_class_accuracy',
        per_model=summary, unique_test_items=100,prediction_occurrences=127500, model_selection=False,
        classifier_fits=0,ordering_deviation='scored_after_expanded_development_not_before'))
    assert io.file_binding(OLD/'COMMIT.json') == plan['original_evaluation']
    bound = {str(p.relative_to(out)):io.file_binding(p) for p in sorted(out.rglob('*.json'))}
    io.write_new(out/'COMMIT.json',dict(status='completed_prediction_only_locked_scoring',
        models=1275,unique_test_items=100,prediction_occurrences=127500,products=bound,
        freeze=io.file_binding(args.freeze),preflight=io.file_binding(planpath),classifier_fits=0))
    print('Locked YuE2 scoring committed',flush=True)

if __name__=='__main__':
    main()
