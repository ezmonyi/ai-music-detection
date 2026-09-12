"""Publish only the verified selected generated AIME originals."""
import json
from collections import Counter
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
import publish_maestro_native30_v1 as transport

PLAN=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/aime_originals_publication_v1')
OUT=PLAN.with_name('hf_aime_originals_v1')
PREFIX='audio/aime_generated_originals_v1/'


def validate_manifest(rows):
    expected={'AudioLDM 2 Large','AudioLDM 2 Music','MusicGen Large','MusicGen Medium',
              'MusicGen Small','Mustango','Riffusion','Stable Audio v1','Stable Audio v2','Udio'}
    assert len(rows)==len({r['id'] for r in rows})==5000
    counts=Counter(r['model'] for r in rows)
    assert set(counts)==expected and set(counts.values())=={500}
    assert len({r['path'] for r in rows})==5000
    for r in rows:
        assert Path(r['filename']).name==r['filename']
        assert r['path']==PREFIX+r['filename']
        assert r['source_dataset']=='disco-eth/AIME'
        assert r['source_parquet_revision']=='1bdacac93127439e361bdd19d575d8b596bca4e3'


def worker(token):
    api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    commit=json.loads((PLAN/'COMMIT.json').read_text())
    assert commit['status']=='original_bytes_materialized_not_uploaded' and commit['files']==5000
    assert transport.digest(PLAN/'manifest.json')==commit['manifest_sha256']
    rows=json.loads((PLAN/'manifest.json').read_text())
    validate_manifest(rows)
    for i,start in enumerate(range(0,5000,50)):
        batch=rows[start:start+50];receipt=OUT/f'batch_{i:03d}.json'
        for r in batch:
            assert Path(r['filename']).name==r['filename']
            assert r['model']!='MTG-Jamendo'
            assert transport.digest(PLAN/'audio'/r['filename'])==r['sha256']
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['files']==[r['path'] for r in batch]
            revision=old['revision']
        else:
            ops=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(PLAN/'audio'/r['filename'])) for r in batch]
            if i==0:ops += [CommitOperationAdd(path_in_repo=PREFIX+n,path_or_fileobj=str(PLAN/n)) for n in ['manifest.json','README.md']]
            result=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
                commit_message=f'Archive selected AIME generated originals {start+1}-{start+len(batch)}')
            revision=result.oid
        remote={x.path:x for x in api.get_paths_info(transport.REPO,paths=[r['path'] for r in batch],repo_type='dataset',revision=revision)}
        for r in batch:
            assert remote[r['path']].size==r['bytes'] and remote[r['path']].lfs.sha256==r['sha256']
        if not receipt.exists():transport.save(receipt,dict(revision=revision,files=[r['path'] for r in batch],sha256_verified=True))
        print(f'AIME verified {start+len(batch)}/5000',flush=True)
    transport.save(OUT/'COMMIT.json',dict(status='5000_generated_originals_uploaded_hash_verified',files=5000,
        manifest_sha256=transport.digest(PLAN/'manifest.json'),whole_project_complete=False))


if __name__=='__main__':
    transport.OUT=OUT;transport.worker=worker;transport.main()
