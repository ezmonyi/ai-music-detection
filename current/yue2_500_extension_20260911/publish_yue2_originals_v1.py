"""Resumable original YuE2 upload; token only in stdin/process memory."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT = ROOT / 'hf_yue2_originals_v1'
REPO = 'EZMONYI/music-ai-human-test-audio'
PREFIX = 'audio/yue2/originals_v1/'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    data = json.dumps(value, indent=2).encode()
    if p.exists():
        assert p.read_bytes() == data
    else:
        with p.open('xb') as stream:
            stream.write(data)


def verify(api, rows, revision):
    found = {r.path: r for r in api.get_paths_info(REPO,
        paths=[r['path'] for r in rows], repo_type='dataset', revision=revision)}
    for row in rows:
        remote = found[row['path']]
        assert remote.size == row['bytes']
        assert remote.lfs and remote.lfs.sha256 == row['sha256']


def worker(token):
    lock = (OUT / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(REPO).private
    source = ROOT / 'native60_inputs_v1'
    assert digest(source / 'COMMIT.json') == 'cde960d1f4ddb29d9751ac8d3bbecf28b6806a5fc9dc05cf299618ec2d8b02ea'
    commit = json.loads((source / 'COMMIT.json').read_text())
    rows = []
    for name in ['metadata.json', 'excluded.json']:
        assert digest(source / name) == commit['products'][name]['sha256']
        rows.extend(json.loads((source / name).read_text()))
    assert len(rows) == len({r['id'] for r in rows}) == 500
    records, local = [], {}
    for r in sorted(rows, key=lambda r: r['id']):
        p = Path(r['source_path'])
        assert digest(p) == r['source_sha256']
        local[r['id']] = str(p)
        records.append(dict(id=r['id'], path=PREFIX+r['id']+'.flac',
            sha256=r['source_sha256'], bytes=p.stat().st_size,
            source_group='YuE2', label=1, role=r['role'], group_id=r['group_id'],
            native_frames=r['native_frames'], sample_rate_hz=r['native_sample_rate_hz'],
            source_receipt_sha256=r['source_receipt_sha256'],
            modification='None; original FLAC bytes with stable experiment filename'))
    save(OUT / 'manifest.json', records)
    notice = '''# YuE2 original generated recordings

500 original FLAC recordings generated locally for this research from the fixed
500 Muse text prompts. No Muse/Suno source audio is included. The two short
outputs remain present; duration eligibility and experiment roles are separate.
The manifest is the planned roster; per-batch receipts establish arrival.

Attribution: YuE2 authors, https://github.com/multimodal-art-projection/YuE ;
YuE2-3B revision 1a96eca688d6ae5d7f0feb88573fec89920fcd19 and YuE2-Vae revision
95535e72a97bc0f09b8ada125d26b4009428c0e8. Checkpoint weights are CC BY-NC 4.0;
first-party code is Apache 2.0. Neither is asserted here as an automatic license
for generated audio. No checkpoint weights are distributed in this folder.

Text conditions: bolshyC/Muse revision b1bf3bf906daab3a896e14f6dea58cc295848452,
https://huggingface.co/datasets/bolshyC/Muse . Its card declares MIT and describes
lyrics/style text as automatically generated; this is source-provided provenance,
not independent clearance of every possible third-party right. This research
archive does not grant blanket rights to other source datasets or endorsements.
Preserve provenance and shared Muse groups when reproducing experiments.
'''
    for index, start in enumerate(range(0, 500, 10)):
        batch = records[start:start+10]
        receipt = OUT / f'batch_{index:03d}.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            assert old['files'] == [r['path'] for r in batch]
            verify(api, batch, old['revision'])
            continue
        ops = [CommitOperationAdd(path_in_repo=r['path'], path_or_fileobj=local[r['id']]) for r in batch]
        if index == 0:
            ops += [CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(OUT/'manifest.json')),
                    CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=notice.encode())]
        result = api.create_commit(repo_id=REPO, repo_type='dataset', operations=ops,
            commit_message=f'Archive YuE2 originals {start+1}-{start+10} of 500')
        verify(api, batch, result.oid)
        save(receipt, dict(revision=result.oid, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'verified {start+10}/500', flush=True)
    save(OUT/'COMMIT.json', dict(status='all_500_originals_uploaded_and_hash_verified',
        files=500, manifest_sha256=digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    token = json.load(sys.stdin)['token']
    OUT.mkdir(exist_ok=True)
    pid = os.fork()
    if pid:
        print(json.dumps(dict(background_pid=pid, output=str(OUT))), flush=True)
    else:
        os.setsid()
        log = os.open(str(OUT/f'worker_{os.getpid()}.log'), os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
        os.dup2(log, 1); os.dup2(log, 2)
        null = os.open('/dev/null', os.O_RDONLY); os.dup2(null, 0)
        try:
            worker(token)
        except Exception as exc:
            print(type(exc).__name__+': '+str(exc).replace(token, '[REDACTED]'), flush=True)
            os._exit(1)
        os._exit(0)
