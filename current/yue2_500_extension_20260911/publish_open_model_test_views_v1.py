"""Publish 800 additional project-generated open-model historical inputs."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify

ROOT = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_open_model_test_views_v1')
PREFIX = 'audio/project_open_models_v1/historical_test_views_v1/'
from publish_open_model_audio_v1 import NOTICE as SOURCE_NOTICE
NOTICE = '''# Additional historical ACE-Step and HeartMuLa test inputs

800 exact historical cropped/standardized files, 400 from each model.
These are additional versions of existing project-generated recordings, not
new generations. Memberships preserve experimental view and crop parameters;
original records preserve model and shared-prompt IDs. No new inference or
byte modification is performed for this release.

The following source-release notice describes the original/30s publication,
not the counts or encoding of this additional 800-file folder:

''' + SOURCE_NOTICE


def build():
    plan = ROOT/'historical_test_view_objects_v1.json'
    originals = OUT.parent/'hf_open_model_audio_v1/manifest.json'
    assert digest(plan) == 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92'
    assert digest(originals) == '7be5aaa118e77181aa489ef3462b254de6f37679e69cb99e690d069387e4f5f5'
    source = {}
    for r in json.loads(originals.read_text()): source.setdefault(r['id'], []).append(r)
    assert len(source) == 1000
    rows = []; paths = {}; ids = set()
    for obj in json.loads(plan.read_text()):
        members = obj['memberships']
        if not any(m['source_group'] in ['ACE-Step','HeartMuLa'] for m in members):
            continue
        assert all(m['source_group'] in ['ACE-Step','HeartMuLa'] for m in members)
        if obj['public_paths']: continue
        for m in members:
            assert m['id'] in source
            ids.add(m['id'])
        p = Path(obj['source_paths'][0])
        name = PREFIX + obj['sha256'] + p.suffix
        paths[name] = p
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            memberships=members,
            original_records=[r for i in sorted({m['id'] for m in members}) for r in source[i]]))
    assert len(rows) == 800 and len(ids) == 800
    return rows, paths


def worker(token):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    rows, paths = build()
    if (OUT/'manifest.json').exists():
        assert json.loads((OUT/'manifest.json').read_text()) == rows
    else:
        save(OUT/'manifest.json', rows)
    for index, start in enumerate(range(0, len(rows), 20)):
        batch = rows[start:start+20]; receipt = OUT/f'batch_{index:03d}.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['files'] == [r['path'] for r in batch]
            verify(api, batch, prior['revision']); continue
        ops = []
        for r in batch:
            p = paths[r['path']]
            assert p.stat().st_size == r['bytes'] and digest(p) == r['sha256']
            ops.append(CommitOperationAdd(path_in_repo=r['path'], path_or_fileobj=str(p)))
        if index == 0:
            ops += [CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(OUT/'manifest.json')),
                    CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode())]
        revision = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=ops,
            commit_message=f'Preserve open-model historical test views {start+1}-{start+len(batch)} of 800').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/800', flush=True)
    save(OUT/'COMMIT.json', dict(status='800_open_model_test_views_uploaded_hash_verified',
        objects=800, original_ids=800, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
