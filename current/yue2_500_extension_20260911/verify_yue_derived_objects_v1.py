"""Verify all 5,900 mapped YuE objects after both derived publications finish."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from publish_yue_derived_objects_v1 import OUT, PLAN, PREFIX, PINS, NOTICE, records

REPO = 'EZMONYI/music-ai-human-test-audio'


def verify():
    terminal = json.loads((OUT/'COMMIT.json').read_text())
    assert terminal['status'] == '4900_additional_yue_objects_uploaded_hash_verified'
    assert terminal['audio_objects'] == 4900 and terminal['plan_pins'] == PINS
    expected = records()
    receipts = sorted(OUT.glob('batch_*.json'))
    assert [p.name for p in receipts] == [f'batch_{i:03d}.json' for i in range(98)]
    for i, path in enumerate(receipts):
        receipt = json.loads(path.read_text())
        assert receipt['sha256_verified'] is True
        assert receipt['files'] == [r['path'] for r in expected[i*50:(i+1)*50]]
    api = HfApi(token=False)
    # Latest revision is fixed once so the independent legacy publication can
    # finish either before or after the final additional-object batch.
    info = api.dataset_info(REPO)
    assert not info.private
    revision = info.sha

    def fetch(name):
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    raw = fetch('manifest.json')
    assert hashlib.sha256(raw).hexdigest() == terminal['manifest_sha256']
    assert json.loads(raw) == expected
    assert fetch('README.md') == NOTICE.encode()
    for name, pin in PINS.items():
        public = fetch(name)
        assert public == (PLAN/name).read_bytes()
        assert hashlib.sha256(public).hexdigest() == pin
    objects = json.loads((PLAN/'objects.json').read_text())
    assert len(objects) == 5900
    for start in range(0, len(objects), 50):
        batch = {r['proposed_path']: r for r in objects[start:start+50]}
        entries = api.get_paths_info(REPO, paths=list(batch), repo_type='dataset', revision=revision)
        assert {e.path for e in entries} == set(batch)
        for entry in entries:
            assert entry.size == batch[entry.path]['bytes']
            assert entry.lfs and entry.lfs.sha256 == batch[entry.path]['sha256']
    return dict(revision=revision, verified_unique_audio_objects=5900,
        additional_objects=4900, legacy_demucs_objects=1000, audio_path_memberships=6922,
        checked_bytes=sum(r['bytes'] for r in objects), plan_pins=PINS,
        manifest_sha256=terminal['manifest_sha256'], final_acceptance=True,
        verification='Anonymous pinned-revision exact manifests/notices and all 5900 LFS hashes/sizes; no full audio redownload',
        scope='Seven inventoried YuE directories only; not all historical project audio',
        whole_project_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    result = verify()
    with args.receipt.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result))
