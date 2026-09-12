"""Hash a fixed set of preserved derived artifacts; do not copy or publish audio."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
NAMES=['analysis_inputs_v1','demucs_v1','native30_fhsc_v1','native30_neural_v1',
       'native60_inputs_v1','native60_neural_v1','legacy_spectral_measurement_v1']
OUT=ROOT/'derived_artifact_inventory_v1'


def main():
    OUT.mkdir(exist_ok=False)
    summary=[]
    with (OUT/'files.jsonl').open('x') as stream:
        for name in NAMES:
            paths=sorted(p for p in (ROOT/name).rglob('*') if p.is_file())
            total=0;counts=Counter()
            for index,p in enumerate(paths):
                before=p.stat();h=hashlib.sha256()
                with p.open('rb') as f:
                    for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
                after=p.stat()
                assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
                row=dict(path=p.relative_to(ROOT).as_posix(),bytes=after.st_size,
                         sha256=h.hexdigest(),is_symlink=p.is_symlink())
                stream.write(json.dumps(row)+'\n');stream.flush()
                total+=after.st_size;counts[p.suffix]+=1
                if (index+1)%500==0:print(f'{name}: hashed {index+1}/{len(paths)}',flush=True)
            summary.append(dict(directory=name,files=len(paths),logical_bytes=total,extensions=dict(counts)))
            print(json.dumps(summary[-1]),flush=True)
    content=(OUT/'files.jsonl').read_bytes()
    (OUT/'COMMIT.json').write_text(json.dumps(dict(status='fixed_scope_artifact_hash_inventory_complete',
        directories=summary,files=sum(r['files'] for r in summary),
        logical_bytes=sum(r['logical_bytes'] for r in summary),
        inventory_sha256=hashlib.sha256(content).hexdigest(),
        content_deduplicated=False,locally_mirrored=False,audio_uploaded=False,
        whole_project_complete=False),indent=2))


if __name__=='__main__':main()
