"""Preserve 1,958 source-bound historical open-model vocal stems."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify
from publish_open_model_audio_v1 import NOTICE as SOURCE_NOTICE

ROOT = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_open_model_vocals_v1')
PREFIX = 'audio/project_open_models_v1/historical_vocals_v1/'
BATCH = 100
NOTICE = '''# Historical ACE-Step / HeartMuLa vocal inputs

1,958 existing Demucs vocal files from 1,000 project-generated recordings:
1,000 ACE-Step views and 958 HeartMuLa views across historical 10s/30s runs.
These are separated vocal inputs, not new generations or original full mixes.
Files are published without further byte modification. Historical input hashes
were checked against existing bytes. Original release records and shared-prompt
identities remain attached; per-file memberships preserve duration and track ID.

The following notice is the provenance of the original release, whose file
counts describe that release rather than this vocal supplement:

''' + SOURCE_NOTICE


def build():
    plan = ROOT / 'open_model_vocal_publication_objects_v1.json'
    assert digest(plan) == '292d79a72c0345ed1ca5948c2011c1eec1a8d5d5f6e07c9a1acce4be3bb162ce'
    records = json.loads(plan.read_text())
    assert len(records) == 1958
    rows, paths = [], {}
    for obj in records:
        path = Path(obj['source_paths'][0])
        name = PREFIX + obj['sha256'] + '.wav'
        assert name not in paths and path.suffix == '.wav'
        paths[name] = path
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            source_group=obj['source_group'], memberships=obj['memberships'],
            original_records=obj['original_records'], transformation=obj['transformation']))
    assert sum(r['bytes'] for r in rows) == 6833822152
    return rows, paths


def worker(token):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    rows, paths = build()
    if (OUT / 'manifest.json').exists():
        assert json.loads((OUT / 'manifest.json').read_text()) == rows
    else:
        save(OUT / 'manifest.json', rows)
    for index, start in enumerate(range(0, len(rows), BATCH)):
        batch = rows[start:start+BATCH]
        receipt = OUT / f'batch_{index:03d}.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['files'] == [r['path'] for r in batch]
            verify(api, batch, prior['revision'])
            continue
        ops = []
        for r in batch:
            path = paths[r['path']]
            assert path.stat().st_size == r['bytes'] and digest(path) == r['sha256']
            ops.append(CommitOperationAdd(path_in_repo=r['path'], path_or_fileobj=str(path)))
        if index == 0:
            ops.extend([CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(OUT/'manifest.json')),
                        CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode())])
        revision = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=ops,
            commit_message=f'Preserve historical open-model vocals {start+1}-{start+len(batch)} of 1958').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/1958', flush=True)
    save(OUT/'COMMIT.json', dict(status='1958_open_model_vocals_uploaded_hash_verified',
        objects=1958, original_ids=1000, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
