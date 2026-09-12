"""Complete the full-mix control for the legacy YuE2 spectral extension."""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
LIB=Path('/mnt/nfs-code/users/yi/open_models_spectral_500_20260901/code/analysis_lib')
sys.path.insert(0,str(LIB))
import analyze_demucs_artifacts as legacy


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    source=ROOT/'legacy_spectral_measurement_v1'
    input_record=json.loads((source/'INPUTS.json').read_text())
    manifest=source/'manifest.jsonl'
    assert sha(manifest)==input_record['manifest_sha256']
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows)==1000
    bound={r['path']:r for r in input_record['inputs']}
    out=ROOT/'legacy_fullmix_v1'
    out.mkdir(exist_ok=False)
    save(out/'PROTOCOL.json',dict(input_record_sha256=sha(source/'INPUTS.json'),
        scripts={str(p):sha(p) for p in [Path(__file__),*sorted(LIB.glob('*.py'))]},
        rows=1000,classifier_fits=0,separator_correction=False,
        representation='legacy standardized first30 fullmix; same frozen DSP as vocal controls'))
    def one(row):
        name=legacy.track_name(row)
        path=(source/'clips'/(name+'.flac')).resolve()
        expected=bound[str(path)]
        assert sha(path)==expected['sha256']
        audio=legacy.decode(path)
        values=legacy.stemlib.spectral_metrics(audio)
        vectors=legacy.frequency_vectors(audio)
        assert all(np.isfinite(v) for v in values.values())
        assert all(np.isfinite(v).all() for v in vectors.values())
        assert sha(path)==expected['sha256']
        return dict(id=row['id'],track=name,label=row['label'],source=row['source'],split=row['split'],**values),vectors
    metrics=[]; matrices={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(values,vectors) in enumerate(pool.map(one,rows),1):
            metrics.append(values)
            for name,vector in vectors.items():matrices.setdefault(name,[]).append(vector)
            if i%25==0:print(f'Fullmix measurement {i}/1000',flush=True)
    with (out/'metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    np.savez_compressed(out/'frequency_vectors.npz',**{k:np.stack(v) for k,v in matrices.items()})
    products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}
    save(out/'COMMIT.json',dict(status='completed_fullmix_measurement_only',rows=1000,products=products,classifier_fits=0))
    print('Fullmix measurement committed',flush=True)


if __name__=='__main__':main()
