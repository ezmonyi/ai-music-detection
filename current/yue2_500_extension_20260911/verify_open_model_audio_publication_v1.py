"""Anonymous final read-only check of all 2,000 original/view public files."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from collections import Counter
from huggingface_hub import HfApi

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
REPO = 'EZMONYI/music-ai-human-test-audio'
PREFIX = 'audio/project_open_models_v1/'


def check(root):
    output = root/'hf_open_model_audio_v1'
    raw = (output/'manifest.json').read_bytes()
    rows = json.loads(raw)
    assert len(rows) == len({r['path'] for r in rows}) == 2000
    assert len({r['id'] for r in rows}) == 1000
    assert Counter((r['model'], r['kind']) for r in rows) == {
        (m, k):500 for m in ('acestep','heartmula') for k in ('original','standardized30')}
    audit = root/'open_model_audio_audit_v1'
    audit_commit = json.loads((audit/'COMMIT.json').read_text())
    audit_raw = (audit/'verified_records.json').read_bytes()
    assert hashlib.sha256(audit_raw).hexdigest() == audit_commit['manifest_sha256']
    expected = {(r['id'], f['kind']):f for r in json.loads(audit_raw) for f in r['files']}
    for r in rows:
        f = expected[(r['id'],r['kind'])]
        assert r['sha256'] == f['sha256'] and r['bytes'] == f['bytes']
        assert r['path'] == PREFIX+r['model']+'/'+r['kind']+'/'+Path(f['path']).name
    receipts = sorted(output.glob('batch_*.json'))
    assert len(receipts) == 100
    for i,p in enumerate(receipts):
        assert p.name == f'batch_{i:03d}.json'
        receipt = json.loads(p.read_text())
        assert receipt['sha256_verified'] is True
        assert receipt['files'] == [r['path'] for r in rows[i*20:(i+1)*20]]
    commit = json.loads((output/'COMMIT.json').read_text())
    assert commit['status'] == '2000_original_view_files_uploaded_hash_verified'
    assert commit['files'] == 2000 and commit['identities'] == 1000
    assert commit['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
    api = HfApi(token=False)
    info = api.dataset_info(REPO)
    assert not info.private
    for start in range(0,2000,50):
        batch = rows[start:start+50]
        remote = {r.path:r for r in api.get_paths_info(REPO,
            paths=[r['path'] for r in batch],repo_type='dataset',revision=info.sha)}
        for r in batch:
            f = remote[r['path']]
            assert f.size == r['bytes'] and f.lfs and f.lfs.sha256 == r['sha256']
        print(f'Independent public hash check {start+len(batch)}/2000',flush=True)
    with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{info.sha}/{PREFIX}manifest.json',timeout=60) as response:
        assert response.read() == raw
    return dict(repo=REPO,revision=info.sha,checked_files=2000,identities=1000,
        checked_bytes=sum(r['bytes'] for r in rows),manifest_sha256=hashlib.sha256(raw).hexdigest(),
        public_manifest_bytes_verified=True,source_audit_bound=True,
        final_acceptance=True,whole_project_complete=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    result=check(args.root)
    with args.receipt.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
