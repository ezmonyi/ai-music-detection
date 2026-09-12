"""Publish the 500 attributed historical MagnaTagATune processed inputs."""
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
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_mtt_test_views_v1')
PREFIX = 'audio/magnatagatune_selected_v1/historical_test_views_v1/'
NOTICE = '''# MagnaTagATune historical processed test inputs

500 exact historical cropped/standardized input files from 500 selected
MagnaTagATune excerpts, not new independent recordings or full songs.
The manifest retains experimental view/crop parameters and original records,
including artists, titles, original track URLs, source archive members and
hashes. Files are copied unchanged from the audited processed inputs.

Audio source: Magnatune. https://magnatune.com/info/api
These adaptations retain CC BY-NC-SA 1.0 (not 4.0):
https://creativecommons.org/licenses/by-nc-sa/1.0/
Noncommercial use only; retain attribution, identify modifications and share
adaptations under the same terms. No endorsement or warranty is implied.

Dataset collection: Edith Law, Olivier Gillet and collaborators; City
University MIRG. https://mirg.city.ac.uk/datasets/magnatagatune/index1.html
Edith Law, Kris West, Michael Mandel, Mert Bay and J. Stephen Downie (2009),
Evaluation of algorithms using games: the case of music annotation, ISMIR.
Source mirror revision and artist-specific links are retained per recording.
No paid member downloads were accessed. Whole-project coverage remains
incomplete; these terms do not license unrelated archive contents.
'''


def build():
    plan = ROOT/'historical_test_view_objects_v1.json'
    originals = OUT.parent/'mtt_publication_plan_v1/manifest.json'
    assert digest(plan) == 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92'
    assert digest(originals) == '90efec2510f56a2e224fb091670a41f7a7b0203f7767d51de2205b79a6632a5d'
    original_rows = json.loads(originals.read_text())
    assert all(r['license'] == 'CC-BY-NC-SA-1.0' and r['artist'] and r['title'] for r in original_rows)
    source = {}
    for r in original_rows: source.setdefault(r['id'], []).append(r)
    assert len(source) == 500
    rows = []; paths = {}; ids = set()
    for obj in json.loads(plan.read_text()):
        members = obj['memberships']
        if not any(m['id'] in source for m in members):
            continue
        assert all(m['id'] in source for m in members)
        assert all(m['source_group'] == 'human_magnatagatune' for m in members)
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
    assert len(rows) == 500 and len(ids) == 500
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
            commit_message=f'Preserve attributed MTT test views {start+1}-{start+len(batch)} of 500').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/500', flush=True)
    save(OUT/'COMMIT.json', dict(status='500_mtt_test_views_uploaded_hash_verified',
        objects=500, original_ids=500, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
