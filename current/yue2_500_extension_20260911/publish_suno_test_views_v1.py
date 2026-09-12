"""Publish 900 historical Suno inputs in five source-attributed batches."""
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
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_suno_test_views_v1')
PREFIX = 'audio/historical_suno_test_views_v1/'
SOURCE_PLAN=OUT.parent/'suno_original_publication_plan_v1'
NOTICE = '''# Historical Suno processed test inputs

900 historical cropped/standardized inputs from 500 existing recordings.
They are additional versions, not new independent songs. Exact experimental
memberships, crop parameters, original hashes, creator handles and group IDs
are preserved. Files are copied unchanged from audited processed inputs.

humair_suno source: https://huggingface.co/datasets/humair025/suno-audio
Card revision 344c67dd2992063779b8f40504ff112f6083e7f2, publisher-declared MIT.
Attribution: humair025 / Humair332; upstream nyuuzyou/suno; individual recorded
creators in the manifest. Retain upstream notices. https://opensource.org/license/mit

suno_unknown source: https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset
Revision bdff424e70c10ef62dca13ba43659ad9e7e1fcdb. Attribution: Kukedlc.
Publisher-declared CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
Retain attribution and indicate processing. Historical unknown model labels
are not silently replaced by current card claims.

Both source cards are retained. This relies on publisher declarations, not
independent clearance of all generation-service or third-party rights. No
blanket license, endorsement or exclusive ownership is claimed. No prompts,
lyrics or artwork are included. Whole-project delivery remains incomplete.
'''


def build():
    plan = ROOT/'historical_test_view_objects_v1.json'
    originals = SOURCE_PLAN/'manifest.json'
    assert digest(plan) == 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92'
    assert digest(originals) == '5ddf39584dea9a8e22e93b0ecf65ddff31b0ac9c554bae9ad35c30958a8ff23b'
    original_rows = json.loads(originals.read_text())
    assert all(r['publisher_declared_license'] in ['MIT','CC-BY-4.0'] for r in original_rows)
    source = {}
    for r in original_rows: source.setdefault(r['id'], []).append(r)
    assert len(source) == 500
    rows = []; paths = {}; ids = set()
    for obj in json.loads(plan.read_text()):
        members = obj['memberships']
        if not any(m['id'] in source for m in members):
            continue
        assert all(m['id'] in source for m in members)
        assert all(m['source_group'] == 'Suno' for m in members)
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
    assert len(rows) == 900 and len(ids) == 500
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
    for index, start in enumerate(range(0, len(rows), 200)):
        batch = rows[start:start+200]; receipt = OUT/f'batch_{index:03d}.json'
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
            for name in ['SOURCE_CARD_HUMAIR.md','SOURCE_CARD_KUKEDLC.md']:
                ops.append(CommitOperationAdd(path_in_repo=PREFIX+name,path_or_fileobj=str(SOURCE_PLAN/name)))
        revision = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=ops,
            commit_message=f'Preserve attributed Suno test views {start+1}-{start+len(batch)} of 900').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/900', flush=True)
    save(OUT/'COMMIT.json', dict(status='900_suno_test_views_uploaded_hash_verified',
        objects=900, original_ids=500, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
