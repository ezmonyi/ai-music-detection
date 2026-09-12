"""Publish the verified VocalSet source subset, without breath annotations."""
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport

PLAN = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/vocalset_publication_plan_v1')
SOURCE = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/external_validation/vocalset_breath_original/audio')
OUT = PLAN.with_name('hf_vocalset_originals_v1')
PREFIX = 'external_controls/vocalset_selected_originals_v1/'


def worker(token):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    commit = json.loads((PLAN/'COMMIT.json').read_text())
    for name, binding in commit['products'].items():
        assert transport.digest(PLAN/name) == binding['sha256']
        assert (PLAN/name).stat().st_size == binding['bytes']
    rows = json.loads((PLAN/'manifest.json').read_text())
    assert len(rows) == len({r['path'] for r in rows}) == 244
    for r in rows:
        assert Path(r['filename']).name == r['filename']
        assert transport.digest(SOURCE/r['filename']) == r['sha256']
    for index, start in enumerate(range(0,244,20)):
        batch = rows[start:start+20]
        receipt = OUT/f'batch_{index:03d}.json'
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            assert previous['files'] == [r['path'] for r in batch]
            revision = previous['revision']
        else:
            ops = [CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(SOURCE/r['filename'])) for r in batch]
            if index == 0:
                ops += [CommitOperationAdd(path_in_repo=PREFIX+name,path_or_fileobj=str(PLAN/name))
                        for name in ['manifest.json','README.md']]
            result = api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
                commit_message=f'Archive selected VocalSet originals {start+1}-{start+len(batch)}')
            revision = result.oid
        remote = {r.path:r for r in api.get_paths_info(transport.REPO,
            paths=[r['path'] for r in batch],repo_type='dataset',revision=revision)}
        for row in batch:
            assert remote[row['path']].size == row['bytes']
            assert remote[row['path']].lfs.sha256 == row['sha256']
        if not receipt.exists():
            transport.save(receipt,dict(revision=revision,files=[r['path'] for r in batch],sha256_verified=True))
        print(f'VocalSet verified {start+len(batch)}/244',flush=True)
    transport.save(OUT/'COMMIT.json',dict(status='244_selected_originals_uploaded_hash_verified',files=244,
        manifest_sha256=transport.digest(PLAN/'manifest.json'),all_project_audio_complete=False))


if __name__ == '__main__':
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
