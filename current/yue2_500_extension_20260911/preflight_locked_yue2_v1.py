"""Metadata-only selection and leakage check for additive locked scoring."""
from collections import Counter
import json
from pathlib import Path
import sys

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code')
ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OLD = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/native30_bc_evaluation_v1_20260911')
sys.path.insert(0, str(RC))
import evaluate_native30_bc_v1 as evaluator
io = evaluator.io

def main():
    assert io.digest(OLD/'COMMIT.json') == '6d6acecbb11b6fcee72f7ba8730b1a2a26f2797c38d659ca69151074dd3dba06'
    package = ROOT/'expanded_native30_features_v1'
    assert io.digest(package/'COMMIT.json') == '353b170a41ac442f612460da5483e92fcf60a2ca0c140fd0eba29223fce212b5'
    products = io.read_json(package/'COMMIT.json')['products']
    for name in ['metadata.json','locked_metadata.json']:
        assert io.file_binding(package/name) == products[name]
    locked = io.read_json(package/'locked_metadata.json')
    metadata = {r['id']:r for r in io.read_json(package/'metadata.json')}
    assert len(locked) == 100 and all(r['role']=='locked_test' and int(r['label'])==1 for r in locked)
    contract = io.read_json(OLD/'contract.json')
    schedule_binding = contract['schedule_document']
    assert io.file_binding(schedule_binding['path']) == schedule_binding
    schedule = io.read_json(schedule_binding['path'])
    chosen = []
    for cell in schedule['combination_schedule_cells']:
        task = evaluator.resolve_task(cell, schedule)
        if not (task['fold_type']=='ordinary_group_holdout_descriptive' and task['quantity']=='all'
                and task['feature_mode']=='values_plus_missing'):
            continue
        assert task['status']=='eligible_metadata_cell'
        train = [metadata[uid] for uid in task['train_ids']]
        for key in ['id','group_id','component_id']:
            assert {r[key] for r in train}.isdisjoint(r[key] for r in locked), key
        chosen.append(dict(model_uid=evaluator.task_uid(task,io.value_hash(contract)),
                           combination=task['combination'],fold=task['opposite_group_fold'],
                           training_ids_sha256=io.value_hash(task['train_ids'])))
    assert len(chosen)==1275 and set(Counter(r['combination'] for r in chosen).values())=={5}
    out = ROOT/'locked_yue2_preflight_v1.json'
    io.write_new(out,dict(status='metadata_preflight_passed_no_scores',models=chosen,
        locked_ids=[r['id'] for r in locked],original_evaluation=io.file_binding(OLD/'COMMIT.json'),
        feature_package=io.file_binding(package/'COMMIT.json'),classifier_fits=0,scores_computed=0))
    print(json.dumps(dict(status='metadata_preflight_passed_no_scores',models=1275,locked_rows=100)),flush=True)

if __name__=='__main__':
    main()
