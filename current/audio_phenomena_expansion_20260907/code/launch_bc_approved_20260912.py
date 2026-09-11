"""Root-reviewed fixed BC contract: persist fit freeze, run, then report."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
RD=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
CONTRACT='b35583db44857a2b3df1b93397a93027fd8896d4d01c6d8708488529b696d498'
PACKAGE='849a79da6e4e160e5629a910e7f8c851484060616af8c5fc8bf6995f54d379f9'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        assert os.environ.get(key) is None, 'Must preserve reviewed runtime'
    evaluator=RC/'code/evaluate_native30_bc_v1.py'
    assert sha(evaluator)=='a40d9074d9c017423aa002778ae857a155e52ad8a49fe7cef9a865b292d8b1b1'
    sys.path.insert(0,str(RC/'code'))
    import evaluate_native30_bc_v1 as e
    pre=json.loads((RC/'preregistration/native30_bc_evaluation_preflight_r2_20260911.json').read_text())
    c=pre['contract']
    assert pre['status']=='preflight_no_fitting' and pre['contract_sha256']==CONTRACT
    assert e.io.value_hash(c)==CONTRACT
    assert c['expected_count']==3830 and len(c['feature_names'])==55
    assert c['accounting']['primary']['eligible']==73950 and c['accounting']['diagnostic']['eligible']==29580
    assert c['ridge']==10 and c['threshold']==0.5
    assert c['families']['BC']==['BC_b2_500_750_1250hz_center8s_median']
    assert all(c[k]==v for k,v in e.SCOPE.items())
    assert sha(RD/'native30_bc_evaluation_package_v1_20260911/COMMIT.json')==PACKAGE
    frozen=dict(version=e.FREEZE_VERSION,status='parent_frozen_for_native30_fold_only_fitting',
                fitting_authorized=True,scoring_authorized=True,stage=e.STAGE,
                contract=c,contract_sha256=CONTRACT,
                independent_review={'approved':True,'reviewer':'root',
                'scope':'Exact successful preflight, accounting, fixed estimator, source pins and prior tests reviewed; not an independent publication audit.'},**e.SCOPE)
    path=RC/'preregistration/native30_bc_evaluation_parent_freeze_20260912.json'
    raw=e.io.canonical(frozen)
    if path.exists(): assert path.read_bytes()==raw
    else:
        with path.open('xb') as f: f.write(raw)
    freeze_sha=sha(path)
    e.authorize(c,str(path),freeze_sha)
    print(json.dumps({'stage':'root_freeze_verified','freeze_sha256':freeze_sha,'contract_sha256':CONTRACT}),flush=True)
    out=Path(c['output_root'])
    with (RC/'logs/native30_bc_evaluation_20260912.log').open('xb') as log:
        subprocess.run([sys.executable,str(evaluator),'--package',str(RD/'native30_bc_evaluation_package_v1_20260911'),
                        '--package-commit-sha256',PACKAGE,'--output',str(out),'--mode','run','--frozen',str(path),
                        '--frozen-sha256',freeze_sha],stdout=log,stderr=subprocess.STDOUT,check=True)
    commit=out/'COMMIT.json'
    assert commit.is_file()
    reporter=RC/'code/summarize_native30_evaluation_bc_v1.py'
    assert sha(reporter)=='c8cca9237826b7ee7a8b4d4a75d39ebd3998b6a71532355b69f32de408a0a2db'
    print('Evaluation terminal COMMIT verified; starting reporting.',flush=True)
    with (RC/'logs/native30_bc_report_20260912.log').open('xb') as log:
        subprocess.run([sys.executable,str(reporter),'--evaluation',str(out),'--evaluation-commit-sha256',sha(commit),
                        '--output',str(RD/'native30_bc_evaluation_report_v1_20260912')],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('BC evaluation and reporting completed. Local synchronization and thesis integration remain.',flush=True)

if __name__=='__main__': main()
