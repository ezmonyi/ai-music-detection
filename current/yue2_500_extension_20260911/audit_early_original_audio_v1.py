"""Hash preserved early-study originals without modifying or publishing audio."""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT=Path('/mnt/nfs-code/users/yi/macbook_reproducibility_20260909/workspace')
INPUT=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/early_experiment_inputs_v1.csv')
OUT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/early_original_audio_audit_v1')


def main():
    rows=list(csv.DictReader(INPUT.open()))
    selected=[r for r in rows if r['experiment']!='direct_band_500x500_20260815']
    assert len(selected)==420
    paths=sorted({r['relative_source_path'] for r in selected});assert len(paths)==400
    output=[]
    for index,relative in enumerate(paths):
        name=PurePosixPath(relative)
        assert not name.is_absolute() and '..' not in name.parts
        path=ROOT/relative;before=path.stat()
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        output.append(dict(relative_source_path=relative,bytes=after.st_size,sha256=digest,
            memberships=[dict(experiment=r['experiment'],legacy_id=r['legacy_id'],label=r['label'],source=r['source'])
                         for r in selected if r['relative_source_path']==relative]))
        if (index+1)%50==0:print(f'Hashed {index+1}/400 early originals',flush=True)
    OUT.mkdir(exist_ok=False)
    raw=json.dumps(output,indent=2).encode();(OUT/'originals.json').write_bytes(raw)
    (OUT/'COMMIT.json').write_text(json.dumps(dict(status='400_preserved_paths_freshly_hashed',
        files=400,memberships=420,bytes=sum(r['bytes'] for r in output),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        historical_expected_raw_hash_available=False,audio_uploaded=False,
        whole_project_complete=False),indent=2))


if __name__=='__main__':main()
