"""Anonymous independent acceptance of the scoped 39-recording CC0 publication."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from publish_musicnet_ishizaka_v1 import OUT,PREFIX,NOTICE,build

REPO='EZMONYI/music-ai-human-test-audio'


def verify():
    done=json.loads((OUT/'COMMIT.json').read_text())
    assert done['status']=='39_musicnet_ishizaka_recordings_uploaded_hash_verified' and done['audio_files']==39
    rows,_=build();revision=done['revision'];api=HfApi(token=False)
    assert not api.dataset_info(REPO,revision=revision).private
    def fetch(name):
        with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}',timeout=60) as response:
            return response.read()
    raw=fetch('manifest.json');assert hashlib.sha256(raw).hexdigest()==done['manifest_sha256']
    assert json.loads(raw)==rows and fetch('README.md')==NOTICE.encode()
    bypath={r['path']:r for r in rows}
    entries=api.get_paths_info(REPO,paths=list(bypath),repo_type='dataset',revision=revision)
    assert {e.path for e in entries}==set(bypath)
    for e in entries:
        assert e.size==bypath[e.path]['bytes'] and e.lfs and e.lfs.sha256==bypath[e.path]['sha256']
    return dict(revision=revision,verified_audio_files=39,checked_bytes=sum(r['bytes'] for r in rows),
        manifest_sha256=done['manifest_sha256'],source_records_bound=True,final_acceptance=True,
        verification='Anonymous manifest/notice equality and every LFS hash/size; not full audio redownload',
        whole_project_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();assert not a.receipt.exists();result=verify()
    with a.receipt.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
