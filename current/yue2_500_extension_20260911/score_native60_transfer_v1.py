"""Frozen YuE2 exact60 score replay with independent scalar verification.

Requires a separately accepted measurement package; never admits its own inputs.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from native60_score_kernel_v1 import score, positive_summary

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0, str(RC/'code'))
import score_mureka60_frozen_v4 as historical


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurement-acceptance', type=Path, required=True)
    parser.add_argument('--acceptance-sha256', required=True)
    args = parser.parse_args()
    assert sha(args.measurement_acceptance)==args.acceptance_sha256
    gate = read(args.measurement_acceptance)
    assert gate['status']=='passed' and gate['classifier_fits']==0
    package = ROOT/'native60_feature_package_v1'
    assert sha(package/'COMMIT.json')==gate['feature_package_commit_sha256']
    commit = read(package/'COMMIT.json')
    for name, entry in commit['products'].items():
        assert sha(package/name)==entry['sha256'] and (package/name).stat().st_size==entry['bytes']
    acceptance = ROOT/'native60_model_acceptance_v1.json'
    assert sha(acceptance)=='642b53378fe1cf4acc5f83b569c38344b2a777be857bf80540edfe4de119f23f'
    approved = read(acceptance)
    modelpath = Path(approved['old_results'])/'fold_models.json'
    assert sha(modelpath)==approved['files_sha256'][str(modelpath)]
    models = read(modelpath)
    index = historical.validate_index(pd.DataFrame(approved['model_index']),models)
    features, metadata = read(package/'features.json'), read(package/'metadata.json')
    ids = [r['id'] for r in metadata]
    assert ids==[r['id'] for r in features] and len(ids)==len(set(ids))==276
    assert all(r['label']==1 for r in metadata)
    populations = {'all_eligible':list(range(276)),
                   'locked_test':[i for i,r in enumerate(metadata) if r['role']=='locked_test'],
                   'development':[i for i,r in enumerate(metadata) if r['role']=='development']}
    assert len(populations['locked_test'])==53 and len(populations['development'])==223
    frame = pd.DataFrame(features).set_index('id')
    out = ROOT/'native60_transfer_scores_v1'
    out.mkdir(exist_ok=False)
    summaries, maximum = [], 0.0
    with (out/'predictions.csv.gz').open('xb') as binary:
        with gzip.GzipFile(fileobj=binary,mode='wb',mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['combination','quantity','fold_index','model_sha256','id','role','raw_score','predicted_ai'])
                for ordinal, entry in enumerate(index.to_dict('records'),1):
                    model=models[entry['model_sha256']]
                    values=historical.replay(frame,model)
                    independent=[score(row,model) for row in features]
                    delta=max(abs(float(v)-other[0]) for v,other in zip(values,independent))
                    maximum=max(maximum,delta)
                    assert np.allclose(values,[v[0] for v in independent],rtol=1e-11,atol=1e-11)
                    decisions=[int(v>=.5) for v in values]
                    assert decisions==[v[1] for v in independent], 'Threshold disagreement'
                    for row,value,decision in zip(metadata,values,decisions):
                        writer.writerow([entry['combination'],entry['quantity'],entry['fold_index'],
                            entry['model_sha256'],row['id'],row['role'],repr(float(value)),decision])
                    for population, positions in populations.items():
                        summaries.append(dict(**entry,population=population,
                            M_secondary_diagnostic='M' in entry['combination'].split('+'),
                            **positive_summary(decisions[i] for i in positions)))
                    if ordinal%250==0:print(f'Independently verified {ordinal}/3175 model applications',flush=True)
    assert len(summaries)==9525
    with (out/'per_model_summary.json').open('x') as stream:json.dump(summaries,stream,indent=2)
    for name, entry in commit['products'].items():assert sha(package/name)==entry['sha256']
    assert sha(modelpath)==approved['files_sha256'][str(modelpath)]
    products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()}
    with (out/'COMMIT.json').open('x') as stream:
        json.dump(dict(status='completed_frozen_transfer_independent_scores_verified',
            models=3175,recordings=276,prediction_rows=876300,summary_rows=9525,
            products=products,max_independent_score_difference=maximum,classifier_fits=0,
            measurement_acceptance_sha256=args.acceptance_sha256,
            scorer_sha256=sha(Path(__file__)),kernel_sha256=sha(Path(__file__).with_name('native60_score_kernel_v1.py'))),stream,indent=2)
    print('Native60 frozen transfer committed',flush=True)


if __name__=='__main__':main()
