"""Preserve 5,000 source-bound historical open-model vocal stems."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify
from publish_aime_test_views_v1 import NOTICE as SOURCE_NOTICE

ROOT = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_aime_vocals_v1')
PREFIX = 'audio/aime_generated_vocals_v1/'
BATCH = 100
NOTICE = '''# Historical AIME generated-music vocal inputs

5,000 existing Demucs vocal files from 5,000 AIME generated recordings,
ten generator groups with 500 recordings each. These are historical separated
vocal inputs, not new generations or full mixes. No further byte modification
was performed for publication. The manifest preserves original source records.
MTG human recordings and prompt descriptions are excluded.

The following source-release notice describes cropped/standardized test views;
its counts refer to that release, not this 5,000-vocal supplement. Its source
attribution and generated-audio CC BY 4.0 declaration also identify these sources:

''' + SOURCE_NOTICE


def build():
    plan = ROOT / 'aime_vocal_publication_objects_v1.json'
    assert digest(plan) == '21126d3da7eeabf6d251e0b312bb5e5797a48a5ac305ee410c625e60ed7b948a'
    records = json.loads(plan.read_text())
    assert len(records) == 5000
    rows, paths = [], {}
    for obj in records:
        path = Path(obj['source_paths'][0])
        name = PREFIX + obj['sha256'] + '.wav'
        assert name not in paths and path.suffix == '.wav'
        paths[name] = path
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            source_group=obj['source_group'], memberships=obj['memberships'],
            original_records=obj['original_records'], transformation=obj['transformation']))
    assert sum(r['bytes'] for r in rows) == 8820220000
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
            commit_message=f'Preserve historical open-model vocals {start+1}-{start+len(batch)} of 5000').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/5000', flush=True)
    save(OUT/'COMMIT.json', dict(status='5000_aime_vocals_uploaded_hash_verified',
        objects=5000, original_ids=5000, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()

