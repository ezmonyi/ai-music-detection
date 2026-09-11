"""Independent stored-frame scalar replay; no producer import or classification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def audit(root,freeze,output):
    contract=json.loads(freeze.read_text()); commit=json.loads((root/'COMMIT.json').read_text())
    require(sha(freeze)==commit['freeze_sha256']=='06dc798c0551d7f9b49c67e226e2633609b4e434fec06ad16ef28aa08340f979','freeze')
    records=[json.loads(s) for s in (root/'selected_features.jsonl').read_text().splitlines()]
    require([r['id'] for r in records]==[r['metadata']['id'] for r in contract['selected']], 'roster')
    require(len(records)==2174,'rows')
    maximum=0.0; checked=0; missing=Counter()
    for index,row in enumerate(records,1):
        relative=f"items/{row['id']}/frames.npz"; path=root/relative
        require(sha(path)==commit['products'][relative]['sha256'],'frame hash')
        with np.load(path,allow_pickle=False) as archive:
            for band,(low,high) in enumerate([(80,500),(500,2000),(2000,6000)]):
                for metric,name in [('iid_db','abs_iid_db'),('side_energy_fraction','side_energy_fraction')]:
                    values=archive[metric]
                    require(values.shape==(5,2580), 'frame shape')
                    values=values[band]; values=values[np.isfinite(values)]
                    key=f'SC_{low}_{high}hz_{name}_median'
                    actual=row['candidate_features'][key]
                    if len(values)<3:
                        require(actual is None,'missing mismatch'); missing[key]+=1
                    else:
                        expected=float(np.median(np.abs(values) if metric=='iid_db' else values))
                        require(actual is not None and abs(expected-actual)<=1e-12,'median mismatch')
                        maximum=max(maximum,abs(expected-actual))
                    checked+=1
        if index%500==0: print(f'replayed {index}/2174',flush=True)
    receipt=dict(passed=True,rows=2174,scalar_checks=checked,maximum_absolute_error=maximum,
        missing_counts=dict(missing),freeze_sha256=sha(freeze),extraction_commit_sha256=sha(root/'COMMIT.json'),
        audit_code_sha256=sha(__file__),classifier_fitted=False,
        scope='independent stored-frame median replay; full waveform STFT not recomputed')
    with output.open('x') as f: json.dump(receipt,f,sort_keys=True,indent=2); f.write('\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); audit(a.root,a.freeze,a.output)
