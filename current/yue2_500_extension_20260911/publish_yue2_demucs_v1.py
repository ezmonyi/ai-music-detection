"""Resume-safe publication of the 500 YuE legacy Demucs vocal/accompaniment pairs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from huggingface_hub import HfApi, CommitOperationAdd
from publish_yue2_originals_v1 import digest, save, verify

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT = ROOT/'hf_yue2_demucs_v1'
REPO = 'EZMONYI/music-ai-human-test-audio'
PREFIX = 'audio/yue2/legacy_demucs_v1/'
INVENTORY_PIN = 'f1ce7251de8a3b08f6fe1c0cb629c494cf3cbd354495dafcb135438e709eed8c'
ORIGINAL_PIN = '4981ad60bd2a658d33d07e79abfb1e4f17d4ff7886e0efabd3d741abc5ab3776'
NOTICE = '''# YuE generated music: legacy Demucs stems

These 1,000 WAV files are 500 paired vocal/accompaniment derivatives, not
1,000 new recordings. They are the preserved htdemucs outputs from this
project's legacy analysis inputs. They are not full-length original recordings
or Native60 inputs. No new separation, normalization or inference was run for
publication. The manifest binds exact saved bytes and original recording IDs.
Historical preprocessing, including any short-input padding, remains unchanged.

Original provenance and checkpoint/text-source notices:
../originals_v1/README.md and ../originals_v1/manifest.json, originally accepted
at revision 4389bbd7906137b9abbc340fcd743bca451ab092.
YuE: https://github.com/multimodal-art-projection/YuE
Demucs: https://github.com/facebookresearch/demucs
Only YuE-generated audio is included; no human-source recordings are included.
Checkpoint licensing is not asserted to automatically license generated audio.
This archive grants no blanket clearance of third-party rights or endorsement.
Preserve provenance and experimental grouping when reusing these derivatives.
The manifest is a planned roster; batch receipts establish publication progress.
'''


def build_records():
    inventory = ROOT/'derived_artifact_inventory_v1/files.jsonl'
    originals = ROOT/'hf_yue2_originals_v1/manifest.json'
    assert digest(inventory) == INVENTORY_PIN
    assert digest(originals) == ORIGINAL_PIN
    source = {r['id']: r for r in json.loads(originals.read_text())}
    assert len(source) == 500
    records = []
    for line in inventory.read_text().splitlines():
        row = json.loads(line)
        p = Path(row['path'])
        if p.parts[0] != 'demucs_v1':
            continue
        assert len(p.parts) == 4 and p.parts[1] == 'htdemucs'
        identity, stem = p.parts[2:]
        assert identity in source and stem in ('vocals.wav', 'no_vocals.wav')
        original = source[identity]
        records.append(dict(id=identity, stem=stem[:-4], path=PREFIX+identity+'/'+stem,
                            source_relative_path=row['path'], bytes=row['bytes'], sha256=row['sha256'],
                            original_path=original['path'], original_sha256=original['sha256'],
                            group_id=original['group_id'], role=original['role'], label=1,
                            modification='Preserved legacy htdemucs separated stem'))
    assert len(records) == len({r['path'] for r in records}) == 1000
    assert {r['id'] for r in records} == set(source)
    return sorted(records, key=lambda r: r['path'])


def worker(token):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(REPO).private
    records = build_records()
    save(OUT/'manifest.json', records)
    for index, start in enumerate(range(0, len(records), 50)):
        batch = records[start:start+50]
        receipt = OUT/f'batch_{index:03d}.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['files'] == [r['path'] for r in batch]
            verify(api, batch, prior['revision'])
            continue
        ops = []
        for row in batch:
            path = ROOT/row['source_relative_path']
            assert path.resolve().is_relative_to(ROOT.resolve())
            assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
            ops.append(CommitOperationAdd(path_in_repo=row['path'], path_or_fileobj=str(path)))
        if index == 0:
            ops.extend([CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(OUT/'manifest.json')),
                        CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode())])
        while True:
            try:
                result = api.create_commit(repo_id=REPO, repo_type='dataset', operations=ops,
                    commit_message=f'Preserve YuE legacy Demucs stems {start+1}-{start+len(batch)} of 1000')
                break
            except Exception as exc:
                response = getattr(exc, 'response', None)
                if response is None or response.status_code != 429:
                    raise
                retry = response.headers.get('Retry-After', '')
                delay = max(3900, int(retry)) if retry.isdigit() else 3900
                print(f'HTTP 429: retry after {delay} seconds', flush=True)
                time.sleep(delay)
        verify(api, batch, result.oid)
        save(receipt, dict(revision=result.oid, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/1000 stems', flush=True)
    save(OUT/'COMMIT.json', dict(status='1000_yue_demucs_stems_uploaded_hash_verified',
        audio_files=1000, recordings=500, manifest_sha256=digest(OUT/'manifest.json'),
        source_inventory_sha256=INVENTORY_PIN, whole_project_complete=False))


if __name__ == '__main__':
    token = json.load(sys.stdin)['token']
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = os.fork()
    if pid:
        print(json.dumps(dict(background_pid=pid, output=str(OUT))), flush=True)
    else:
        os.setsid()
        log = os.open(OUT/f'worker_{os.getpid()}.log', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.dup2(log, 1); os.dup2(log, 2)
        null = os.open('/dev/null', os.O_RDONLY); os.dup2(null, 0)
        try:
            worker(token)
        except Exception as exc:
            print('Worker failed: '+type(exc).__name__, flush=True)
            os._exit(1)
        os._exit(0)
