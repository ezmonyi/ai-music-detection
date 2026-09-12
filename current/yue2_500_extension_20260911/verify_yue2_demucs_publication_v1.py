"""Anonymous terminal acceptance of every published legacy YuE Demucs stem."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from publish_yue2_demucs_v1 import OUT, REPO, PREFIX, INVENTORY_PIN, build_records


def verify():
    terminal = json.loads((OUT/'COMMIT.json').read_text())
    assert terminal['status'] == '1000_yue_demucs_stems_uploaded_hash_verified'
    assert terminal['audio_files'] == 1000 and terminal['recordings'] == 500
    assert terminal['source_inventory_sha256'] == INVENTORY_PIN
    expected = build_records()
    receipts = sorted(OUT.glob('batch_*.json'))
    assert [p.name for p in receipts] == [f'batch_{i:03d}.json' for i in range(20)]
    for i, path in enumerate(receipts):
        receipt = json.loads(path.read_text())
        assert receipt['sha256_verified'] is True
        assert receipt['files'] == [r['path'] for r in expected[i*50:(i+1)*50]]
    revision = json.loads(receipts[-1].read_text())['revision']
    api = HfApi(token=False)
    assert not api.dataset_info(REPO, revision=revision).private
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}manifest.json'
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest() == terminal['manifest_sha256']
    assert json.loads(raw) == expected
    for start in range(0, 1000, 50):
        batch = {r['path']: r for r in expected[start:start+50]}
        entries = api.get_paths_info(REPO, paths=list(batch), repo_type='dataset', revision=revision)
        assert {e.path for e in entries} == set(batch)
        for entry in entries:
            assert entry.size == batch[entry.path]['bytes']
            assert entry.lfs and entry.lfs.sha256 == batch[entry.path]['sha256']
    return dict(revision=revision, verified_audio_files=1000, recordings=500,
                checked_bytes=sum(r['bytes'] for r in expected),
                manifest_sha256=terminal['manifest_sha256'], source_inventory_sha256=INVENTORY_PIN,
                source_records_bound=True, final_acceptance=True, whole_project_complete=False,
                verification='Anonymous pinned-revision manifest equality and all LFS hashes/sizes; no full audio redownload')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    result = verify()
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
