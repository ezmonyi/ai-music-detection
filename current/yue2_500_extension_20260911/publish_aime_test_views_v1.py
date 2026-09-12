"""Publish exact historical processed inputs for the 5000 selected AIME outputs."""
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
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_aime_test_views_v1')
PREFIX = 'audio/aime_generated_test_views_v1/'
from publish_aime_originals_v1 import validate_manifest
NOTICE = '''# AIME generated audio: historical processed test inputs

5,500 exact historical cropped/standardized input files from 5,000 selected
generated recordings, 500 originals per generator label. These are additional
versions, not new independent recordings. The manifest retains experiment IDs,
view/crop parameters, generator labels, original hashes and source shard/row.
No new inference or byte modification is performed for this publication.

Source and attribution: disco-eth/AIME and the authors of Benchmarking Music
Generation Models and Metrics via Human Preference Studies.
https://huggingface.co/datasets/disco-eth/AIME
https://ieeexplore.ieee.org/abstract/document/10887745
The publisher licenses generated audio CC BY 4.0; these cropped/standardized
derivatives are shared under those terms, with modifications indicated here.
https://creativecommons.org/licenses/by/4.0/
This relies on the publisher's declaration, not independent clearance of every
underlying service or third-party right. No endorsement is implied.

MTG human recordings are excluded. Prompt descriptions have separate terms and
are not reproduced. Historical evaluation roles and grouping remain unchanged;
publication is not a new untouched test split. Whole-project coverage remains
incomplete and no blanket license is assigned to unrelated archive contents.
'''


def build():
    plan = ROOT/'historical_test_view_objects_v1.json'
    originals = OUT.parent/'aime_originals_publication_v1/manifest.json'
    assert digest(plan) == 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92'
    assert digest(originals) == 'b8c0ac0e9cc21b431c1813bcc3594463786123510be422048758947ab1fe9f70'
    original_rows = json.loads(originals.read_text())
    validate_manifest(original_rows)
    source = {}
    for r in original_rows: source.setdefault(r['id'], []).append(r)
    assert len(source) == 5000
    rows = []; paths = {}; ids = set()
    for obj in json.loads(plan.read_text()):
        members = obj['memberships']
        if not any(m['id'] in source for m in members):
            continue
        assert all(m['id'] in source for m in members)
        assert all(m['source_group'] == source[m['id']][0]['model'] for m in members)
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
    assert len(rows) == 5500 and len(ids) == 5000
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
    for index, start in enumerate(range(0, len(rows), 50)):
        batch = rows[start:start+50]; receipt = OUT/f'batch_{index:03d}.json'
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
            commit_message=f'Preserve AIME historical test views {start+1}-{start+len(batch)} of 5500').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/5500', flush=True)
    save(OUT/'COMMIT.json', dict(status='5500_aime_test_views_uploaded_hash_verified',
        objects=5500, original_ids=5000, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
