"""Join complete, hash-verified YuE2 measurements to the frozen original table."""
import json
import math
import os
from pathlib import Path
import sys
import time

RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code')
RD=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OLD=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/native30_bc_evaluation_package_v1_20260911')
sys.path.insert(0,str(RC))
import evaluate_native30_v3 as evaluator
io=evaluator.io
FEATURES={**evaluator.FAMILIES,'BC':['BC_b2_500_750_1250hz_center8s_median']}


def verified(path):
    commit=io.read_json(path/'COMMIT.json')
    for name,binding in commit.get('products',{}).items():
        assert io.file_binding(path/name)==binding
    return commit


def main():
    while not (RD/'native30_sdrp_v1/COMMIT.json').exists():
        try:os.kill(2964016,0)
        except ProcessLookupError:raise RuntimeError('SDRP producer stopped without COMMIT; inspect its failure log')
        time.sleep(30)
    output=RD/'expanded_native30_features_v1'
    assert not output.exists(),'Do not overwrite a previous assembly'
    plan=RD/'expanded_native30_plan_v1'
    verified(plan)
    parent=verified(OLD)
    assert io.digest(OLD/'COMMIT.json')=='849a79da6e4e160e5629a910e7f8c851484060616af8c5fc8bf6995f54d379f9'
    sdrp=verified(RD/'native30_sdrp_v1')
    bc=verified(RD/'native30_bc_v1')
    fhsc=verified(RD/'native30_fhsc_v1')
    audit_path=RD/'native30_bc_repeatability_audit_v1.json'
    audit=io.read_json(audit_path)
    assert audit['rows']==498 and audit['status']=='passed_full_same_code_audio_to_arrays_and_scalar_replay'
    assert audit['measurement_commit']==io.file_binding(RD/'native30_bc_v1/COMMIT.json')
    original=io.read_json(OLD/'features.json')
    metadata=io.read_json(plan/'metadata.json')
    locked=io.read_json(plan/'yue2_locked_metadata.json')
    catalogue=sum(FEATURES.values(),[])
    assert len(catalogue)==55 and len(original)==3830
    all_features={r['id']:r for r in original}
    lineage=[]
    for meta in sorted(metadata+locked,key=lambda r:r['id']):
        uid=meta['id']
        if meta['source_group']!='YuE2':continue
        files={k:RD/folder/'items'/(uid+'.json') for k,folder in
               [('sdrp','native30_sdrp_v1'),('bc','native30_bc_v1'),('fhsc','native30_fhsc_v1')]}
        assert io.file_binding(files['sdrp'])==sdrp['items'][uid]
        assert io.file_binding(files['bc'])==bc['items'][uid]
        assert io.digest(files['fhsc'])==fhsc['items'][uid]
        s,b,f=[io.read_json(files[k]) for k in ('sdrp','bc','fhsc')]
        expected=meta['input_sha256']
        assert s['row']['input']['sha256']==b['input']['sha256']==f['view_sha256']==expected
        assert s['row']['prompt_id']==f['prompt_id']==meta['prompt_id']
        assert s['row']['split']==f['split']==meta['role']
        values={}
        for family,names in FEATURES.items():
            source=(s['features'] if family in ('S','D','R','P') else
                    f[family] if family in ('F','H') else f['SC']['features'] if family=='SC' else b['features'])
            for name in names:
                value=source[name]
                assert value is None or isinstance(value,(int,float)) and math.isfinite(value),name
                values[name]=value
        assert uid not in all_features
        all_features[uid]={'id':uid,**values}
        lineage.append({'id':uid,'receipts':{k:io.file_binding(v) for k,v in files.items()}})
    assert len(all_features)==4328 and len(lineage)==498
    development_ids={r['id'] for r in metadata}; locked_ids={r['id'] for r in locked}
    assert len(development_ids)==4228 and len(locked_ids)==100 and not development_ids&locked_ids
    coverage={family:sum(all(all_features[uid][name] is not None for name in names) for uid in development_ids)
              for family,names in FEATURES.items()}
    contract=dict(status='assembled_verified_receipts_not_independent_feature_audit_not_fitted',
                  original_commit=io.file_binding(OLD/'COMMIT.json'),plan_commit=io.file_binding(plan/'COMMIT.json'),
                  bc_replay_audit=io.file_binding(audit_path),driver=io.file_binding(Path(__file__).resolve()),
                  families=FEATURES,development_rows=4228,locked_rows=100,development_complete_family_rows=coverage,
                  classifier_fits=0,threshold_tuning=False,winner_selection=False)
    output.mkdir()
    products={'contract.json':contract,'metadata.json':metadata,'locked_metadata.json':locked,
              'features.json':[all_features[uid] for uid in sorted(development_ids)],
              'locked_features.json':[all_features[uid] for uid in sorted(locked_ids)],'yue2_lineage.json':lineage}
    for name,value in products.items():io.write_new(output/name,value)
    io.write_new(output/'COMMIT.json',{'status':contract['status'],
                 'products':{name:io.file_binding(output/name) for name in products}})
    print(json.dumps({'development':4228,'locked':100,'coverage':coverage}),flush=True)


if __name__=='__main__':main()
