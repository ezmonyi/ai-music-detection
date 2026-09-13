"""Publish only 452 exact-ID, per-track-license-screened FMA stems."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
import re
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify

ROOT = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_fma_stems_v1')
PREFIX = 'audio/fma_historical_stems_explicit_derivative_v1/'
BATCH = 100
NOTICE = '''# FMA historical separated stems: per-recording licenses

452 existing historical Demucs stem files, not new independent recordings.
Transformation: source separation into the stem named in each experimental
membership; no further waveform modification at publication. Each file retains
its original artist, title, FMA track ID and exact license URL in the manifest.
Each is distributed under that individual license, including version and
jurisdiction; there is no blanket folder license. Preserve attribution and
modification notices. Noncommercial and share-alike conditions apply where
specified; CC0 material retains its dedication. No endorsement is implied.

Source: Free Music Archive; https://github.com/mdeff/fma
Citation: Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson,
FMA: A Dataset For Music Analysis, ISMIR 2017.

Selection uses exact recording IDs from previously screened original source
records. ND and unresolved candidates are excluded; 848 other FMA stem objects
are not included. Source declarations are not a warranty of every third-party
right. Whole-project preservation and publication remain incomplete.
'''


def build():
    plan = ROOT/'fma_stem_candidates_v1/objects.json'
    assert digest(plan) == 'a7302dfbb1c4a07bd3828c89edd9ecee84b22264f93e2e5f51b454cafe4947a5'
    rows, paths = [], {}
    for obj in json.loads(plan.read_text()):
        url = obj['license_url']
        assert re.fullmatch(r'https://creativecommons.org/licenses/(by|by-sa|by-nc|by-nc-sa)/(1\.0|2\.0|2\.5|3\.0|4\.0)/(us/)?', url) or url == 'https://creativecommons.org/publicdomain/zero/1.0/'
        originals = obj['original_records']
        assert {m['item_id'] for m in obj['memberships']} <= {r['id'] for r in originals}
        assert all(r['artist'] and r['title'] and r['license_url'] == url for r in originals)
        path = Path(obj['source_paths'][0])
        name = PREFIX+obj['sha256']+'.wav'
        assert name not in paths and path.suffix == '.wav'
        paths[name] = path
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            license_url=url, original_records=originals, memberships=obj['memberships'],
            modification=obj['transformation']))
    assert len(rows) == 452 and sum(r['bytes'] for r in rows) == 1414747888
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
    for index, start in enumerate(range(0, len(rows), BATCH)):
        batch = rows[start:start+BATCH]
        receipt = OUT/f'batch_{index:03d}.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['sha256_verified'] and prior['files'] == [r['path'] for r in batch]
            verify(api, batch, prior['revision'])
            continue
        ops = []
        for row in batch:
            path = paths[row['path']]
            assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
            ops.append(CommitOperationAdd(path_in_repo=row['path'], path_or_fileobj=str(path)))
        if index == 0:
            ops += [CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(OUT/'manifest.json')),
                    CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode())]
        revision = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=ops,
            commit_message=f'Preserve licensed FMA stems {start+1}-{start+len(batch)} of 452').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/452', flush=True)
    save(OUT/'COMMIT.json', dict(status='452_fma_stems_uploaded_hash_verified', objects=452,
        manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
