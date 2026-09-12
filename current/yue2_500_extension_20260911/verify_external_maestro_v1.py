"""Independently verify all earlier external MAESTRO audio/MIDI pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from publish_external_maestro_v1 import OUT,PREFIX,PIN,NOTICE,build

REPO='EZMONYI/music-ai-human-test-audio'


def verify():
    done=json.loads((OUT/'COMMIT.json').read_text())
    assert done['status']=='100_external_maestro_inputs_uploaded_hash_verified'
    assert done['audio_files']==done['midi_files']==50 and done['source_audit_sha256']==PIN
    rows,_=build();receipts=sorted(OUT.glob('batch_*.json'))
    assert [p.name for p in receipts]==[f'batch_{i:03d}.json' for i in range(5)]
    for i,path in enumerate(receipts):
        r=json.loads(path.read_text());assert r['sha256_verified'] is True
        assert r['files']==[x['path'] for x in rows[i*20:(i+1)*20]]
    revision=json.loads(receipts[-1].read_text())['revision'];api=HfApi(token=False)
    assert not api.dataset_info(REPO,revision=revision).private
    def fetch(name):
        with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}',timeout=60) as response:
            return response.read()
    raw=fetch('manifest.json');assert hashlib.sha256(raw).hexdigest()==done['manifest_sha256']
    assert json.loads(raw)==rows and fetch('README.md')==NOTICE.encode()
    for start in range(0,100,50):
        batch={r['path']:r for r in rows[start:start+50]}
        entries=api.get_paths_info(REPO,paths=list(batch),repo_type='dataset',revision=revision)
        assert {e.path for e in entries}==set(batch)
        for e in entries:
            expected=batch[e.path];assert e.size==expected['bytes']
            if e.lfs: assert e.lfs.sha256==expected['sha256']
            else:
                # Small MIDI objects may be ordinary Git blobs, not LFS.
                payload=fetch(e.path.removeprefix(PREFIX))
                assert hashlib.sha256(payload).hexdigest()==expected['sha256']
    return dict(revision=revision,verified_audio_files=50,verified_midi_files=50,
        checked_bytes=sum(r['bytes'] for r in rows),source_audit_sha256=PIN,
        manifest_sha256=done['manifest_sha256'],final_acceptance=True,whole_project_complete=False,
        verification='Anonymous exact metadata and LFS hashes/sizes; non-LFS MIDI downloaded and SHA-256 checked')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();assert not a.receipt.exists();result=verify()
    with a.receipt.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
