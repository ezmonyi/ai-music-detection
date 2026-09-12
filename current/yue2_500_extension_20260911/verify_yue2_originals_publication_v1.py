"""Independent read-only publication check; no upload and no credentials needed."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi

REPO = 'EZMONYI/music-ai-human-test-audio'
ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_yue2_originals_v1')
PREFIX = 'audio/yue2/originals_v1/'


def check(root, partial=False):
    manifest = root / 'manifest.json'
    raw = manifest.read_bytes()
    rows = json.loads(raw)
    assert len(rows) == 500 and len({r['path'] for r in rows}) == 500
    receipts = sorted(root.glob('batch_*.json'))
    assert partial or len(receipts) == 50
    covered = []
    for i, path in enumerate(receipts):
        assert path.name == f'batch_{i:03d}.json'
        receipt = json.loads(path.read_text())
        expected = rows[10*i:10*i+10]
        assert receipt['sha256_verified'] is True
        assert receipt['files'] == [r['path'] for r in expected]
        covered.extend(expected)
    api = HfApi(token=False)
    revision = api.dataset_info(REPO).sha
    for start in range(0, len(covered), 50):
        batch = covered[start:start+50]
        remote = {r.path:r for r in api.get_paths_info(REPO,
            paths=[r['path'] for r in batch], repo_type='dataset', revision=revision)}
        for row in batch:
            item = remote[row['path']]
            assert item.size == row['bytes']
            assert item.lfs and item.lfs.sha256 == row['sha256']
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}manifest.json'
    with urllib.request.urlopen(url, timeout=60) as response:
        assert response.read() == raw
    terminal = root / 'COMMIT.json'
    if not partial:
        commit = json.loads(terminal.read_text())
        assert commit['files'] == 500
        assert commit['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
        assert commit['status'] == 'all_500_originals_uploaded_and_hash_verified'
    return dict(repo=REPO, revision=revision, checked_files=len(covered),
                checked_bytes=sum(r['bytes'] for r in covered),
                manifest_sha256=hashlib.sha256(raw).hexdigest(),
                public_manifest_bytes_verified=True,
                final_acceptance=not partial, whole_project_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root, args.allow_partial)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
