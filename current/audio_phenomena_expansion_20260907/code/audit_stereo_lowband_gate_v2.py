#!/usr/bin/env python3
"""Independent gate replay; no producer or extractor imports, no fitting."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/stereo_lowband_controls_v2'
GATE=ROOT/'results/stereo_lowband_gate_v2'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    freeze=ROOT/'preregistration/stereo_lowband_gate_frozen_v2.json'
    frozen=read(freeze); result=read(GATE/'gate_result.json')
    previous=read(ROOT/'preregistration/stereo_music_controls_frozen_v1.json')
    selected=frozen['selected']; assert len(selected)==7
    assert not {r['track_id'] for r in selected}&{r['track_id'] for r in previous['selected']}
    assert not {r['artist_proxy'] for r in selected}&{r['artist_proxy'] for r in previous['selected']}
    for path,digest in frozen['bindings'].items():
        assert sha(path)==digest
    counts={}
    for directory in [RAW,GATE]:
        commit=read(directory/'COMMIT.json')
        assert commit['freeze_sha256']==sha(freeze)
        products=commit['products']
        assert set(products)=={str(p.relative_to(directory)) for p in directory.rglob('*')
                               if p.is_file() and p.name!='COMMIT.json'}
        for name,record in products.items():
            p=directory/name
            assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
        counts[directory.name]=len(products)
    checks=0; ratio_checks=0; all_ratios={key:[] for key in frozen['candidate_features']}
    max_iid=0.; max_width=0.; alignment_count=0
    for row in selected:
        assert sha(row['mixture'])==row['input_sha256']
        directory=RAW/row['track_id']; comparisons=read(directory/'comparisons.json')['comparisons']
        with np.load(directory/'original_frames.npz') as z:
            iid=z['iid_db'].copy(); side=z['side_energy_fraction'].copy()
        baseline=read(directory/'original_features.json')['features']
        for name,c in comparisons.items():
            if 'alignment' in c:
                assert c['alignment']['eligible']; alignment_count+=1
                a=read(directory/(name+'_reference_features.json'))['features']
                b=read(directory/(name+'_features.json'))['features']
                targets=read(directory/(name+'_matched_target_features.json'))
                for key in frozen['candidate_features']:
                    target='iid_plus_6db' if 'abs_iid_db' in key else 'width_2'
                    assert a[key] is not None and b[key] is not None and targets[target][key] is not None
                    numerator=abs(b[key]-a[key]); denominator=abs(targets[target][key]-a[key])
                    assert denominator>1e-8
                    ratio=numerator/denominator
                    assert abs(ratio-c['target_response_ratios'][key]['ratio'])<=1e-12
                    all_ratios[key].append(ratio); ratio_checks+=1
                continue
            changed=read(directory/(name+'_features.json'))['features']
            with np.load(directory/(name+'_frames.npz')) as z:
                for band_index,(lo,hi) in enumerate([(80,500),(500,2000),(2000,6000),(6000,12000)]):
                    if name in ['gain_half','swap']:
                        keys=[k for k in baseline if k.startswith(f'SC_{lo}_{hi}hz_')]
                        assert all((baseline[k] is None)==(changed[k] is None) for k in keys)
                        values=[abs(changed[k]-baseline[k]) for k in keys if baseline[k] is not None]
                        passed=bool(values and max(values)<=1e-8)
                    else:
                        if name.startswith('iid'):
                            db=float(name.split('_')[-1][:-2]); target=iid[band_index]+db
                            target[(abs(iid[band_index])>=60)|(abs(target)>=60)]=np.nan
                            observed=z['iid_db'][band_index]
                        else:
                            w=float(name.split('_')[-1]); q=side[band_index]
                            target=w*w*q/(1-q+w*w*q); observed=z['side_energy_fraction'][band_index]
                        common=np.isfinite(target)&np.isfinite(observed)
                        errors=np.abs(target[common]-observed[common])
                        maximum=float(errors.max()) if len(errors) else float('inf')
                        passed=common.sum()>=30 and common.mean()>=.5 and maximum<=1e-8
                        if name.startswith('iid'):
                            max_iid=max(max_iid,maximum)
                        else:
                            max_width=max(max_width,maximum)
                    assert bool(passed)==c['band_checks'][band_index]['passed'] and passed
                    checks+=1
    feature_checks=[]
    for key,ratios in all_ratios.items():
        assert len(ratios)==28
        count=sum(v<=.1 for v in ratios); passed=count/28>=.9 and max(ratios)<=1
        recorded=next(r for r in result['feature_checks'] if r['feature']==key)
        assert recorded['available']==28 and recorded['count_at_most_one_tenth']==count
        assert recorded['passed']==passed and abs(recorded['maximum_ratio']-max(ratios))<=1e-12
        feature_checks.append(dict(feature=key,count_at_most_one_tenth=count,maximum_ratio=max(ratios),passed=passed))
    assert checks==196 and alignment_count==28 and ratio_checks==168
    assert result['measurement_gate_passed']==all(r['passed'] for r in feature_checks)
    for row in read(GATE/'target_waveform_bindings.json'):
        assert sha(row['resolved_wav'])==row['sha256']
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        code_sha256=sha(__file__),freeze_sha256=sha(freeze),
        raw_commit_sha256=sha(RAW/'COMMIT.json'),gate_receipt_sha256=sha(GATE/'gate_result.json'),
        products_hashed=counts,source_inputs_hashed=7,primary_checks_replayed=checks,
        alignment_eligible=alignment_count,ratios_replayed=ratio_checks,
        max_primary_iid_error_db=max_iid,max_primary_width_error=max_width,
        feature_checks=feature_checks,classifier_fitted=False,
        scope='independent implementation of prospective measurement-gate replay; not classification accuracy')
    with (ROOT/'audit/stereo_lowband_gate_parent_replay_v2.json').open('x') as f:
        json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
