"""Fixed legacy frequency-band comparison on preserved disjoint holdout groups."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
CODE=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OLD=Path('/mnt/nfs-code/users/yi/open_models_spectral_500_20260901/code/analysis_lib')
sys.path.insert(0,str(OLD))
import analyze_bias_corrected_vocals as bias
import analyze_demucs_artifacts as legacy


def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(p,v):
    with p.open('x') as stream:json.dump(v,stream,indent=2,allow_nan=False)


def main():
    source=ROOT/'legacy_spectral_measurement_v1'
    commit=json.loads((source/'COMMIT.json').read_text())
    assert commit['status']=='completed_legacy_measurement_and_descriptive_maps'
    for name,binding in commit['products'].items():
        path=(source/name).resolve()
        assert path.is_relative_to(source) and digest(path)==binding['sha256']
    rows=[json.loads(line) for line in (source/'manifest.jsonl').read_text().splitlines()]
    prompts={r['id']:r for r in map(json.loads,(CODE/'prompts/prompt_manifest.jsonl').read_text().splitlines())}
    groups=[('human:'+str(r['group_id'])) if int(r['label'])==0 else 'muse:'+prompts[r['id']]['source_song_id'] for r in rows]
    train=np.array([i for i,r in enumerate(rows) if r['split']=='development'])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='locked_test'])
    labels=np.array([int(r['label']) for r in rows])
    assert len(rows)==1000 and len(train)==800 and len(test)==200
    assert set(groups[i] for i in train).isdisjoint(groups[i] for i in test)
    assert np.bincount(labels[train]).tolist()==[400,400] and np.bincount(labels[test]).tolist()==[100,100]
    metrics=list(csv.DictReader((source/'features/vocal_metrics_raw_and_corrected.csv').open()))
    matrices={}
    fullmix=ROOT/'legacy_fullmix_v1'
    fullcommit=json.loads((fullmix/'COMMIT.json').read_text())
    assert fullcommit['status']=='completed_fullmix_measurement_only'
    for name,binding in fullcommit['products'].items():
        path=(fullmix/name).resolve()
        assert path.is_relative_to(fullmix) and digest(path)==binding['sha256']
    fullrows=list(csv.DictReader((fullmix/'metrics.csv').open()))
    assert [r['track'] for r in fullrows]==[legacy.track_name(r) for r in rows]
    matrices['fullmix__spectral']=np.asarray([[float(r[k]) for k in legacy.stemlib.ALL_METRICS] for r in fullrows])
    with np.load(fullmix/'frequency_vectors.npz',allow_pickle=False) as vectors:
        for name in legacy.FREQUENCY_RANGES:
            matrices['fullmix__'+name]=vectors[name]
    with np.load(source/'features/frequency_vectors_raw_and_corrected.npz',allow_pickle=False) as vectors:
        for variant in ['raw','bias_corrected']:
            matrices[variant+'__vocal_spectral']=bias.matrix_for_variant(metrics,rows,variant)
            for name in legacy.FREQUENCY_RANGES:
                matrices[variant+'__'+name]=vectors[variant+'__'+name]
    out=ROOT/'legacy_spectral_locked_v1'
    out.mkdir(exist_ok=False)
    write(out/'PROTOCOL.json',dict(source_commit_sha256=digest(source/'COMMIT.json'),
          fullmix_commit_sha256=digest(fullmix/'COMMIT.json'),
          scripts={str(p):digest(p) for p in [Path(__file__),*sorted(OLD.glob('*.py'))]},
          alpha=10.0,threshold=0.5,train_items=800,test_items=200,group_disjoint=True,
          representations=list(matrices),model_selection=False,
          excluded_evaluation='legacy random row-wise development CV is not group-aware',
          limitation='historical human holdout reused; not a new independent external test; YuE2 present in training'))
    results=[]
    for name,x in matrices.items():
        assert x.shape[0]==1000 and np.isfinite(x).all()
        mean,scale,weights,intercept=legacy.stemlib.fit_ridge(x[train],labels[train],alpha=10.0)
        scores=((x[test]-mean)/scale)@weights+intercept
        assert np.allclose(scores,legacy.fit_predict(x[train],labels[train],x[test]),rtol=0,atol=1e-12)
        ba,auc=legacy.metrics(labels[test],scores)
        predicted=(scores>=0.5).astype(int)
        manual=(np.mean(predicted[labels[test]==0]==0)+np.mean(predicted[labels[test]==1]==1))/2
        assert abs(manual-ba)<1e-12
        result=dict(representation=name,balanced_accuracy=float(ba),roc_auc=float(auc),
                    human_specificity=float(np.mean(predicted[labels[test]==0]==0)),
                    ai_sensitivity=float(np.mean(predicted[labels[test]==1]==1)),dimension=int(x.shape[1]))
        write(out/(name+'.json'),dict(**result,mean=mean.tolist(),scale=scale.tolist(),weights=weights.tolist(),
              intercept=float(intercept),test_ids=[rows[i]['id'] for i in test],scores=scores.tolist(),labels=labels[test].tolist()))
        results.append(result)
        print(json.dumps(result),flush=True)
    write(out/'summary.json',dict(results=results,model_selection=False,train=800,test=200,
          interpretation='within-generator historical-holdout comparison, not generator-held-out generalization'))
    products={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.is_file()}
    write(out/'COMMIT.json',dict(status='completed_fixed_legacy_band_holdout_comparison',products=products))


if __name__=='__main__':main()
