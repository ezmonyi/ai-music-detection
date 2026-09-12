"""Full SDRP reproducibility replay and FHM record reconciliation; no inference.

SDRP uses the original extractor again: this is a reproducibility check, not
an independently implemented acoustic algorithm. FHM is reconciled to committed
per-record outputs, not recomputed from audio in this audit.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
sys.path.insert(0,str(RC/'code'))
import prepare_evaluation_inputs_v4 as reference
import extract_native60_sdrp_v1 as producer


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def main():
    package=ROOT/'native60_feature_package_v1'
    package_sha=sha(package/'COMMIT.json')
    commit=read(package/'COMMIT.json')
    for name,entry in commit['products'].items():
        assert sha(package/name)==entry['sha256'] and (package/name).stat().st_size==entry['bytes']
    features=read(package/'features.json');metadata=read(package/'metadata.json')
    assert len(features)==len(metadata)==276
    contract=read(ROOT/'native60_sdrp_v1/contract.json')
    fingerprint=sha(ROOT/'native60_sdrp_v1/contract.json')
    freeze=read(RC/'preregistration/native30_inference_parent_freeze_v2.json')
    extractor=producer.adapter.load_extractor(freeze['old_code_root'])
    assert sha(Path(extractor.__file__))==reference.OLD_CODE['extract_expanded_four_family.py']
    bias_path=Path(contract['bias']['path'])
    assert sha(bias_path)==reference.BIAS_SHA
    bias=extractor.load_bias(bias_path)
    neural=ROOT/'native60_neural_v1'
    opts=SimpleNamespace(duration=60,demix_root=[neural/'demix'],
                         beat_root=[neural/'beats'],structure_root=[neural/'structure'])
    source_rows={r['id']:r for r in contract['rows']}
    fcommit=read(ROOT/'native60_fhm_v1/COMMIT.json')
    out=ROOT/'native60_measurement_replay_v1';out.mkdir(exist_ok=False)

    def one(pair):
        feature,meta=pair;uid=meta['id'];assert feature['id']==uid
        row=source_rows[uid]
        path=Path(row['input']['path'])
        assert sha(path)==row['input']['sha256']
        mapped=dict(item_id=uid,source_id='YuE2',label='ai',group_id=meta['group_id'],
            standardized_path=str(path),native_sample_rate_hz='48000',duration='60',audio_offset_s='0')
        fresh=extractor.process(mapped,opts,bias,reference.BIAS_SHA,fingerprint)
        assert fresh['status']=='complete'
        differences=[]
        for column in sum(reference.OLD_COLUMNS.values(),[]):
            left,right=feature[column],fresh[column]
            if right is not None and not math.isfinite(right):right=None
            if left is None or right is None:assert left is right,(uid,column)
            else:
                assert math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-10),(uid,column,left,right)
                differences.append(abs(left-right))
        name='features/items/'+hashlib.sha256(uid.encode()).hexdigest()+'.json'
        path=ROOT/'native60_fhm_v1'/name
        assert sha(path)==fcommit['products'][name]['sha256']
        item=read(path)
        assert item['id']==uid and item['group_id']==meta['group_id'] and item['role']==meta['role']
        for column in sum(reference.NEW_COLUMNS.values(),[]):assert feature[column]==item[column],(uid,column)
        return dict(id=uid,role=meta['role'],SDRP_max_absolute_difference=max(differences,default=0),
                    FHM_committed_item_match=True)

    checked=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index,result in enumerate(pool.map(one,zip(features,metadata)),1):
            checked.append(result)
            if index%25==0:print(f'Numeric reproducibility checked {index}/276',flush=True)
    for name,entry in commit['products'].items():assert sha(package/name)==entry['sha256']
    assert sha(package/'COMMIT.json')==package_sha
    with (out/'per_record.json').open('x') as stream:json.dump(checked,stream,indent=2)
    receipt=dict(status='passed',rows=276,classifier_fits=0,predictions_generated=0,
        feature_package_commit_sha256=package_sha,
        SDRP_recomputed_rows=276,FHM_committed_item_reconciled_rows=276,
        independent_acoustic_algorithm=False,FHM_audio_recomputed=False,
        relative_tolerance=1e-9,absolute_tolerance=1e-10,
        maximum_absolute_difference=max(r['SDRP_max_absolute_difference'] for r in checked),
        auditor_sha256=sha(Path(__file__)),per_record_sha256=sha(out/'per_record.json'))
    with (out/'ACCEPTANCE.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print('Measurement replay passed',flush=True)


if __name__=='__main__':main()
