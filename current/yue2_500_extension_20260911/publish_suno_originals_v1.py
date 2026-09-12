"""Publish audited original Suno recordings with separate source attribution."""
from collections import Counter
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
PLAN=ROOT/'suno_original_publication_plan_v1'
OUT=ROOT/'hf_suno_originals_v1'
SOURCE=Path('/mnt/nfs-code/users/yi/macbook_reproducibility_20260909/workspace')
PREFIX='audio/historical_suno_originals_v1/'


def worker(token):
    api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    commit=json.loads((PLAN/'COMMIT.json').read_text())
    assert commit['status']=='500_originals_publication_plan_not_uploaded' and commit['files']==500
    assert transport.digest(PLAN/'manifest.json')==commit['manifest_sha256']
    rows=json.loads((PLAN/'manifest.json').read_text())
    assert len(rows)==len({r['id'] for r in rows})==len({r['path'] for r in rows})==500
    assert Counter(r['source_collection'] for r in rows)=={'humair_suno':250,'suno_unknown':250}
    assert sum(r['bytes'] for r in rows)==2282055145
    for r in rows:
        assert r['path']==PREFIX+r['source_collection']+'/'+r['id']+'.mp3'
        assert Path(r['id']).name==r['id']
        allowed='humair025-suno-audio-mp3' if r['source_collection']=='humair_suno' else 'suno-ai-music-dataset-audio'
        p=SOURCE/r['source_relative_path']
        assert p.is_relative_to(SOURCE/allowed) and '..' not in p.parts
    for i,start in enumerate(range(0,500,100)):
        batch=rows[start:start+100];receipt=OUT/f'batch_{i:03d}.json'
        for r in batch:
            p=SOURCE/r['source_relative_path']
            assert p.stat().st_size==r['bytes'] and transport.digest(p)==r['sha256']
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['files']==[r['path'] for r in batch]
            revision=old['revision']
        else:
            ops=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(SOURCE/r['source_relative_path'])) for r in batch]
            if i==0:
                ops += [CommitOperationAdd(path_in_repo=PREFIX+n,path_or_fileobj=str(PLAN/n))
                    for n in ('manifest.json','README.md','SOURCE_CARD_HUMAIR.md','SOURCE_CARD_KUKEDLC.md')]
            revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
                commit_message=f'Archive historical Suno originals {start+1}-{start+100}').oid
        remote={r.path:r for r in api.get_paths_info(transport.REPO,
            paths=[r['path'] for r in batch],repo_type='dataset',revision=revision)}
        for r in batch:
            f=remote[r['path']]
            assert f.size==r['bytes'] and f.lfs and f.lfs.sha256==r['sha256']
        if not receipt.exists():transport.save(receipt,dict(revision=revision,files=[r['path'] for r in batch],sha256_verified=True))
        print(f'Suno originals verified {start+100}/500',flush=True)
    transport.save(OUT/'COMMIT.json',dict(status='500_suno_originals_uploaded_hash_verified',files=500,
        manifest_sha256=transport.digest(PLAN/'manifest.json'),whole_project_complete=False))


if __name__=='__main__':
    transport.OUT=OUT;transport.worker=worker;transport.main()
