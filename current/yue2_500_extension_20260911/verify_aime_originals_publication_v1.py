"""Read-only, anonymous final acceptance of the selected AIME publication."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
REPO = 'EZMONYI/music-ai-human-test-audio'
PREFIX = 'audio/aime_generated_originals_v1/'
MODELS = {'AudioLDM 2 Large', 'AudioLDM 2 Music', 'MusicGen Large',
          'MusicGen Medium', 'MusicGen Small', 'Mustango', 'Riffusion',
          'Stable Audio v1', 'Stable Audio v2', 'Udio'}


def check(root):
    plan = root / 'aime_originals_publication_v1'
    output = root / 'hf_aime_originals_v1'
    raw = (plan / 'manifest.json').read_bytes()
    rows = json.loads(raw)
    assert len(rows) == len({r['id'] for r in rows}) == 5000
    assert len({r['path'] for r in rows}) == 5000
    counts = Counter(r['model'] for r in rows)
    assert set(counts) == MODELS and set(counts.values()) == {500}
    audit_raw = (root / 'aime_generated_byte_audit_v1/verified_rows.jsonl').read_bytes()
    assert hashlib.sha256(audit_raw).hexdigest() == 'f43a6ce1257357bd6e4ef4a81773192677bc4ca00c8e0147af75119c09c95435'
    audited = {r['id']: r for r in map(json.loads, audit_raw.splitlines())}
    for r in rows:
        assert Path(r['filename']).name == r['filename']
        assert r['path'] == PREFIX + r['filename']
        assert r['source_dataset'] == 'disco-eth/AIME'
        assert r['source_parquet_revision'] == '1bdacac93127439e361bdd19d575d8b596bca4e3'
        for key in ('model', 'bytes', 'sha256', 'shard', 'row'):
            assert r[key] == audited[r['id']][key]
    assert sum(r['bytes'] for r in rows) == 8281783640
    receipts = sorted(output.glob('batch_*.json'))
    assert len(receipts) == 100
    for i, path in enumerate(receipts):
        assert path.name == f'batch_{i:03d}.json'
        receipt = json.loads(path.read_text())
        assert receipt['sha256_verified'] is True
        assert receipt['files'] == [r['path'] for r in rows[i*50:(i+1)*50]]
    terminal = json.loads((output / 'COMMIT.json').read_text())
    assert terminal['status'] == '5000_generated_originals_uploaded_hash_verified'
    assert terminal['files'] == 5000
    assert terminal['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
    api = HfApi(token=False)
    info = api.dataset_info(REPO)
    assert not info.private
    revision = info.sha
    for start in range(0, 5000, 50):
        batch = rows[start:start+50]
        remote = {r.path: r for r in api.get_paths_info(REPO,
            paths=[r['path'] for r in batch], repo_type='dataset', revision=revision)}
        for row in batch:
            item = remote[row['path']]
            assert item.size == row['bytes']
            assert item.lfs and item.lfs.sha256 == row['sha256']
        print(f'Independent public hash check {start+len(batch)}/5000', flush=True)
    for name in ('manifest.json', 'README.md'):
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            assert response.read() == (plan / name).read_bytes()
    return dict(repo=REPO, revision=revision, checked_files=5000,
                checked_bytes=sum(r['bytes'] for r in rows),
                manifest_sha256=hashlib.sha256(raw).hexdigest(),
                source_audit_bound=True, public_metadata_bytes_verified=True,
                remote_lfs_sha256_verified=True, final_acceptance=True,
                whole_project_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
