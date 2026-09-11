"""YuE2 native-center30 F/H/SC using unchanged, pinned existing measurements."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys

RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code')
RD=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0,str(RC))
import standardize_native30_v1 as dsp
import run_native30_fhsc_pilot_v1 as pilot
import soundfile as sf
import numpy as np

OUT=RD/'native30_fhsc_v1'

def write_new(path,value):
    with path.open('x') as f: json.dump(value,f,sort_keys=True,allow_nan=False)

def one(row):
    uid=row['id']; receipt=OUT/'items'/(uid+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text())
        assert r['source_sha256']==row['source_sha256']
        assert dsp.digest(r['view_path'])==r['view_sha256']
        return r
    source=Path(row['source'])
    wave,proof=dsp.standardize(source,row['source_sha256'],48000,2)
    path=OUT/'audio'/(uid+'.wav')
    assert not path.exists(), 'Unreceipted input retained; investigate before resume'
    sf.write(path,wave,44100,format='WAV',subtype='FLOAT')
    reread,sr=sf.read(path,dtype='float32',always_2d=True)
    assert sr==44100 and np.array_equal(wave,reread)
    result,sc=pilot.measure(path)
    frames=OUT/'frames'/(uid+'.npz')
    np.savez_compressed(frames,**sc['per_frame'],**sc['valid_masks'])
    result.update(id=uid,source_sha256=row['source_sha256'],view_path=str(path),view_sha256=dsp.digest(path),
                  standardization=proof,prompt_id=row['prompt_id'],split=row['split'],language=row['language'],
                  frames_path=str(frames),frames_sha256=dsp.digest(frames))
    write_new(receipt,result)
    return result

def main():
    for name,sha in pilot.PINS.items(): assert dsp.digest(RC/name)==sha
    inventory=RD/'analysis_inputs_v1/inventory.json'
    rows=json.loads(inventory.read_text())['metadata']
    eligible=[r for r in rows if r['native30_eligible']]
    assert len(rows)==500 and len(eligible)==498
    OUT.mkdir(exist_ok=True)
    for folder in ['audio','items','frames']: (OUT/folder).mkdir(exist_ok=True)
    contract=dict(status='extraction_only_not_classifier_admission',rows=498,excluded_short=[r['id'] for r in rows if not r['native30_eligible']],
                  inventory_sha256=dsp.digest(inventory),code_pins=pilot.PINS,
                  driver_sha256=dsp.digest(Path(__file__)),pilot_sha256=dsp.digest(RC/'run_native30_fhsc_pilot_v1.py'),
                  crop='unchanged standardize_native30_v1 native-center30',M_role='diagnostic_not_predictor')
    write_new(OUT/'contract.json',contract)
    finished=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(one,eligible):
            finished.append(r)
            if len(finished)%10==0: print(f'F/H/SC completed {len(finished)}/498',flush=True)
    from collections import Counter
    summary=dict(rows=len(finished),F=dict(Counter(r['F']['F_status'] for r in finished)),
                 H=dict(Counter(r['H']['H_status'] for r in finished)),
                 SC=dict(Counter(r['SC']['selected_six_status'] for r in finished)),classifier_fits=0)
    write_new(OUT/'summary.json',summary)
    write_new(OUT/'COMMIT.json',dict(status='completed_extraction_not_independent_audit',summary=summary,
              items={r['id']:dsp.digest(OUT/'items'/(r['id']+'.json')) for r in finished}))
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
