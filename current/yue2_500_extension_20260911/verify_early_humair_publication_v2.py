"""Independently verify public humair objects against pinned recovered records."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi

REPO = 'EZMONYI/music-ai-human-test-audio'
PREFIX = 'audio/early_humair_originals_v2/'
PIN = 'd3cba828ae9ce21dba5dbd5bf6fe1d8773793d1197a8f7e5fd1e9829b81a4343'


def verify(records, terminal):
    source = records.read_bytes()
    assert hashlib.sha256(source).hexdigest() == PIN
    expected = json.loads(source)
    assert len(expected) == len({r['id'] for r in expected}) == 100
    done = json.loads(terminal.read_text())
    assert done['status'] == '100_early_humair_originals_uploaded_hash_verified'
    assert done['audio_files'] == 100 and done['metadata_records_sha256'] == PIN
    revision = done['revision']
    api = HfApi(token=False)
    assert not api.dataset_info(REPO, revision=revision).private

    def fetch(name):
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    raw = fetch('manifest.json')
    assert hashlib.sha256(raw).hexdigest() == done['manifest_sha256']
    public = json.loads(raw)
    assert public == [dict(r, published_path=PREFIX+r['id']+'.mp3') for r in expected]
    files = {r['published_path']: r for r in public}
    names = list(files)
    for offset in range(0, 100, 50):
        paths = names[offset:offset+50]
        entries = api.get_paths_info(REPO, paths=paths, repo_type='dataset', revision=revision)
        assert {e.path for e in entries} == set(paths)
        for entry in entries:
            row = files[entry.path]
            assert entry.size == row['original_bytes']
            assert entry.lfs and entry.lfs.sha256 == row['original_sha256']
    notice = fetch('README.md')
    card = fetch('SOURCE_CARD.md')
    assert b'license: mit' in card[:200]
    assert b'multiple attribution variants' in notice
    assert b'dd95495c415eea043c250f12da595de2ad4cad7f' in notice
    return dict(revision=revision, verified_audio_files=100,
                checked_bytes=sum(r['original_bytes'] for r in expected),
                public_manifest_sha256=done['manifest_sha256'], source_records_sha256=PIN,
                notice_sha256=hashlib.sha256(notice).hexdigest(),
                source_card_sha256=hashlib.sha256(card).hexdigest(),
                source_records_bound=True, attribution_variants_preserved=True,
                verification='anonymous pinned-revision manifest equality and LFS size/SHA-256; not full audio redownload',
                final_acceptance=True, whole_project_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--terminal', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    result = verify(args.records, args.terminal)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
