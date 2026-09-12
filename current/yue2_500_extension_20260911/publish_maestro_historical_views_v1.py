"""Publish exact historical MAESTRO test inputs with source-bound attribution."""
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
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_maestro_historical_views_v1')
PREFIX = 'audio/human_maestro_v3/historical_test_views_v1/'
NOTICE = '''# MAESTRO historical 10s/30s test inputs

600 exact processed files from 300 existing MAESTRO performances, distinct
from the previously released Native30 inputs. These are not 600 new songs.
The source files were cropped/standardized for the historical four-family
experiments. Files are copied unchanged from the audited experiment inputs;
the view label and recorded crop parameters are retained without claiming
unverified encoding parameters. No new inference or audio modification occurs
during this publication.

Source: Google LLC; International Piano-e-Competition; MAESTRO dataset authors.
https://magenta.tensorflow.org/datasets/maestro
License for these derivatives: CC BY-NC-SA 4.0.
https://creativecommons.org/licenses/by-nc-sa/4.0/
Noncommercial use only; retain attribution, indicate modifications, and share
adaptations under the same license. No endorsement is implied.
Citation: Curtis Hawthorne et al., Enabling Factorized Piano Music Modeling
and Generation with the MAESTRO Dataset, ICLR 2019.
https://openreview.net/forum?id=r1lYRjC9F7

Original publication paths/hashes and composer-based group metadata are bound
by exact experiment ID. Composer identity is not performer identity.
This source-specific license does not apply to unrelated archive content.
Whole-project audio coverage remains incomplete.
'''


def build():
    plan = ROOT/'historical_test_view_objects_v1.json'
    originals = ROOT/'maestro_originals_manifest_for_views_v1.json'
    assert digest(plan) == 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92'
    assert digest(originals) == '4fda23164b186c023481424018d3be1018de931bf39b8bc33f4fa2fd79546602'
    source = {r['id']:r for r in json.loads(originals.read_text())}
    assert len(source) == 300
    rows = []; paths = {}; ids = set()
    for obj in json.loads(plan.read_text()):
        members = obj['memberships']
        if not any(m['source_group'] == 'human_maestro_v3' for m in members):
            continue
        assert all(m['source_group'] == 'human_maestro_v3' for m in members)
        assert not obj['public_paths']
        for m in members:
            assert m['id'] in source and source[m['id']]['license'] == 'CC-BY-NC-SA-4.0'
            ids.add(m['id'])
        p = Path(obj['source_paths'][0])
        name = PREFIX + obj['sha256'] + p.suffix
        paths[name] = p
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            memberships=members, license='CC-BY-NC-SA-4.0',
            original_records=[{k:source[i][k] for k in ['id','path','sha256','group_id','attribution','source_url']}
                              for i in sorted({m['id'] for m in members})]))
    assert len(rows) == 600 and len(ids) == 300
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
            commit_message=f'Preserve historical MAESTRO test views {start+1}-{start+len(batch)} of 600').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/600', flush=True)
    save(OUT/'COMMIT.json', dict(status='600_historical_maestro_views_uploaded_hash_verified',
        objects=600, original_ids=300, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
