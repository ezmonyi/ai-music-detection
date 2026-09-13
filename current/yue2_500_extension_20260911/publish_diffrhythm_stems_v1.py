"""Preserve 400 source-bound historical maestro stem stems."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify
from publish_diffrhythm_test_views_v1 import NOTICE as SOURCE_NOTICE
import publish_diffrhythm_pilot_v1 as source

ROOT = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_diffrhythm_stems_v1')
PREFIX = 'audio/diffrhythm_historical_stems_v1/'
BATCH = 100
NOTICE = '''# Historical diffrhythm separated stems

400 exact existing Demucs stem files from 50 original recordings.
Transformation: bass/drums/other/vocals separation. No further byte modification
for publication. Apache-2.0 under the source publisher's audio declaration. Pilot status and conditioning groups remain unchanged.
Per-recording original release records remain in the manifest.

The following earlier release notice supplies source attribution and conditions.
Its file counts describe that earlier release rather than this stem supplement:

''' + SOURCE_NOTICE


def build():
    plan = ROOT / 'diffrhythm_stem_publication_objects_v1.json'
    assert digest(plan) == 'd72a62b62ce2a86618b8fe52f539da2632b0032b6000ae23eae5ac8870ed9bfc'
    records = json.loads(plan.read_text())
    assert len(records) == 400
    rows, paths = [], {}
    for obj in records:
        path = Path(obj['source_paths'][0])
        name = PREFIX + obj['sha256'] + '.wav'
        assert name not in paths and path.suffix == '.wav'
        paths[name] = path
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            source_group=obj['source_group'], memberships=obj['memberships'], license=obj['license'],
            original_records=obj['original_records'], transformation=obj['transformation']))
    assert sum(r['bytes'] for r in rows) == 1411217600
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
            for filename in ('LICENSE.txt', 'SOURCE_CARD.md'):
                assert digest(source.SOURCES/filename) == source.PINS[filename]
                ops.append(CommitOperationAdd(path_in_repo=PREFIX+filename, path_or_fileobj=str(source.SOURCES/filename)))
        revision = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=ops,
            commit_message=f'Preserve historical maestro stems {start+1}-{start+len(batch)} of 400').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/400', flush=True)
    save(OUT/'COMMIT.json', dict(status='400_diffrhythm_stems_uploaded_hash_verified',
        objects=400, original_ids=50, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()


