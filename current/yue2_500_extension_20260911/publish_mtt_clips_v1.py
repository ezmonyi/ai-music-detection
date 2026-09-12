"""Archive 500 small MagnaTagATune excerpts in one commit, with attribution."""
import json
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
import publish_maestro_native30_v1 as transport

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
PLAN=ROOT/'mtt_publication_plan_v1'
OUT=ROOT/'hf_mtt_clips_v1'
SOURCE=Path('/mnt/nfs-data/users/yi/source_diversity_expansion_20260905/native/human_magnatagatune')
PREFIX='audio/magnatagatune_selected_v1/'


def worker(token):
    api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    assert transport.digest(PLAN/'manifest.json')=='90efec2510f56a2e224fb091670a41f7a7b0203f7767d51de2205b79a6632a5d'
    rows=json.loads((PLAN/'manifest.json').read_text())
    assert len(rows)==len({r['id'] for r in rows})==500
    assert sum(r['bytes'] for r in rows)==58615981
    for r in rows:
        assert Path(r['id']).name==r['id'] and r['path']==PREFIX+r['id']+'.mp3'
        p=SOURCE/(r['id']+'.mp3')
        assert p.stat().st_size==r['bytes'] and transport.digest(p)==r['sha256']
    receipt=OUT/'batch_000.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['files']==[r['path'] for r in rows]
        revision=old['revision']
    else:
        ops=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(SOURCE/(r['id']+'.mp3'))) for r in rows]
        ops += [CommitOperationAdd(path_in_repo=PREFIX+n,path_or_fileobj=str(PLAN/n)) for n in ('manifest.json','README.md')]
        revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
            commit_message='Archive 500 attributed MagnaTagATune research clips').oid
    for start in range(0,500,50):
        batch=rows[start:start+50]
        remote={r.path:r for r in api.get_paths_info(transport.REPO,
            paths=[r['path'] for r in batch],repo_type='dataset',revision=revision)}
        for r in batch:
            f=remote[r['path']];assert f.size==r['bytes'] and f.lfs and f.lfs.sha256==r['sha256']
    if not receipt.exists():transport.save(receipt,dict(revision=revision,files=[r['path'] for r in rows],sha256_verified=True))
    transport.save(OUT/'COMMIT.json',dict(status='500_mtt_clips_uploaded_hash_verified',files=500,
        manifest_sha256=transport.digest(PLAN/'manifest.json'),whole_project_complete=False))
    print('500 MTT clips uploaded and batch-hash-verified',flush=True)


if __name__=='__main__':
    transport.OUT=OUT;transport.worker=worker;transport.main()
