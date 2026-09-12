"""Anonymous independent verification of Saraga originals and source notices."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from publish_saraga_originals_v1 import OUT, PREFIX, PIN, CARD, NOTICE, build_records

REPO = 'EZMONYI/music-ai-human-test-audio'


def verify():
    done = json.loads((OUT/'COMMIT.json').read_text())
    assert done['status'] == '108_saraga_originals_uploaded_hash_verified'
    assert done['audio_files'] == 108 and done['classifier_ids'] == 103
    assert done['source_audit_sha256'] == PIN
    expected = build_records()
    receipts = sorted(OUT.glob('batch_*.json'))
    assert [p.name for p in receipts] == [f'batch_{i:03d}.json' for i in range(11)]
    for index, path in enumerate(receipts):
        receipt = json.loads(path.read_text())
        assert receipt['sha256_verified'] is True
        assert receipt['files'] == [r['path'] for r in expected[index*10:(index+1)*10]]
    revision = json.loads(receipts[-1].read_text())['revision']
    api = HfApi(token=False)
    assert not api.dataset_info(REPO, revision=revision).private

    def fetch(name):
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    manifest = fetch('manifest.json')
    assert hashlib.sha256(manifest).hexdigest() == done['manifest_sha256']
    assert json.loads(manifest) == expected
    notice = fetch('README.md')
    license_text = fetch('SOURCE_LICENSE.md')
    assert notice == NOTICE.encode() and license_text == CARD.read_bytes()
    for start in range(0, len(expected), 50):
        batch = {r['path']: r for r in expected[start:start+50]}
        entries = api.get_paths_info(REPO, paths=list(batch), repo_type='dataset', revision=revision)
        assert {e.path for e in entries} == set(batch)
        for entry in entries:
            assert entry.size == batch[entry.path]['bytes']
            assert entry.lfs and entry.lfs.sha256 == batch[entry.path]['sha256']
    return dict(revision=revision, verified_audio_files=108, classifier_ids=103,
        checked_bytes=sum(r['bytes'] for r in expected), source_audit_sha256=PIN,
        manifest_sha256=done['manifest_sha256'], notice_sha256=hashlib.sha256(notice).hexdigest(),
        source_license_sha256=hashlib.sha256(license_text).hexdigest(),
        attribution_bound=True, detailed_performer_credit_gaps=2,
        verification='Anonymous pinned-revision manifest and notice equality, all LFS hashes/sizes; no full audio redownload',
        final_acceptance=True, whole_project_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    result = verify()
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
