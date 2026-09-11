"""Metadata-only YuE2 extension with shared-prompt and protected-set closure.

Uses the existing pure planner with an explicitly extended eight-family
catalogue in this process only. Original code and contracts are not modified.
"""
import copy
import itertools
import json
from pathlib import Path
import sys

RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
RD=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
YC=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OLD=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/native30_evaluation_package_v3_20260911')
OUT=RD/'expanded_native30_plan_v1'
sys.path.insert(0,str(RC/'code'))
import plan_native30_evaluation_schedule_v1 as planner
import run_native30_inference_batches_v1 as io


def main():
    assert io.digest(RC/'code/plan_native30_evaluation_schedule_v1.py')=='ff6965b3f9c834ee9d39bbf0f994595f44fa2e261d9c7c6b60a153cb9f72ce24'
    committed=io.read_json(OLD/'COMMIT.json')
    for name,binding in committed['products'].items():
        assert io.binding(OLD/name)==binding
    parent=io.read_json(OLD/'contract.json')
    screen_binding=parent['bindings']['screen']
    assert io.binding(Path(screen_binding['path']))==screen_binding
    screen=copy.deepcopy(io.read_json(screen_binding['path']))
    metadata=io.read_json(OLD/'metadata.json')
    assert len(metadata)==3830
    inventory_path=RD/'analysis_inputs_v1/inventory.json'
    inventory=io.read_json(inventory_path)['metadata']
    prompts_path=YC/'prompts/prompt_manifest.jsonl'
    prompts={r['id']:r for r in map(json.loads,prompts_path.read_text().splitlines())}
    assert len(inventory)==len(prompts)==500
    known={r['id']:r for r in screen['rows']}
    existing_groups={r['group_id'] for r in metadata}
    extension=[]
    group_joins=0
    for row in inventory:
        prompt=prompts[row['prompt_id']]
        assert prompt['split']==row['split']
        uid=row['id']
        assert uid not in known
        group='muse:'+prompt['source_song_id']
        # Verify exact shared-prompt identities against the old screen where present.
        for prefix in ('ai_acestep_','ai_heartmula_'):
            prior=known.get(prefix+prompt['id'])
            if prior:
                assert prior['group_id']==group
        cid='yue2_extension_'+prompt['id']
        identity=dict(id=uid,label='1',source_group='YuE2',role=row['split'],group_id=group,component_id=cid)
        screen['components'].append(dict(component_id=cid,members=[uid],
                                    link_tokens=['conditioning:'+group],
                                    protected_relationships=['locked_test'] if row['split']!='development' else []))
        screen['rows'].append(identity)
        if not row['native30_eligible']:
            continue
        receipt=RD/'native30_fhsc_v1/items'/(uid+'.json')
        measured=io.read_json(receipt)
        assert measured['id']==uid and measured['prompt_id']==prompt['id']
        record={**identity,'duration_view_s':30,'native_sample_rate_hz':48000,
                'input_sha256':measured['view_sha256'],'input_path':measured['view_path'],
                'source_receipt':io.binding(receipt),'prompt_id':prompt['id']}
        extension.append(record)
        group_joins+=group in existing_groups
    assert len(extension)==498
    development=[r for r in extension if r['role']=='development']
    locked=[r for r in extension if r['role']=='locked_test']
    assert len(development)==398 and len(locked)==100
    rows=sorted([{k:r[k] for k in planner.IDENTITY} for r in metadata+development],key=lambda r:r['id'])
    planner.FAMILIES=(*planner.FAMILIES,'BC')
    planner.COMBINATIONS=['+'.join(c) for n in range(1,9) for c in itertools.combinations(planner.FAMILIES,n)]
    schedule=planner.make_schedule(rows,screen)
    assert len(planner.COMBINATIONS)==255
    assert not set(r['id'] for r in locked)&set(schedule['eligible_population']['ids'])
    contract=dict(status='metadata_plan_only_not_feature_admission_not_fit_authorization',
                  parent_commit=io.binding(OLD/'COMMIT.json'),parent_screen=screen_binding,
                  prompt_manifest=io.binding(prompts_path),inventory=io.binding(inventory_path),
                  driver=io.binding(Path(__file__).resolve()),planner=io.binding(RC/'code/plan_native30_evaluation_schedule_v1.py'),
                  development_rows=len(rows),yue2_development=398,yue2_locked=100,
                  yue2_matching_existing_groups=group_joins,feature_combinations=255,
                  classifier_fits=0,threshold_tuning=False,winner_selection=False,
                  note='The metadata catalogue expands; the frozen source-purge and coupled-cap algorithms are unchanged.')
    OUT.mkdir(exist_ok=False)
    for name,value in [('contract.json',contract),('screen.json',screen),('metadata.json',metadata+development),
                       ('yue2_locked_metadata.json',locked),('schedule.json',schedule)]:
        io.write_new(OUT/name,value)
    io.write_new(OUT/'COMMIT.json',dict(status=contract['status'],
                 products={p.name:io.binding(p) for p in sorted(OUT.iterdir())},classifier_fits=0))
    print(json.dumps({'development_rows':len(rows),'eligible':schedule['eligible_population']['rows'],
                      'locked_yue2':len(locked),'shared_group_joins':group_joins,
                      'accounting':schedule['accounting']}),flush=True)


if __name__=='__main__':
    main()
