"""Anonymous verification of pilot audio, grouping and attribution notices."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from publish_diffrhythm_pilot_v1 import OUT,PREFIX,SOURCES,NOTICE,build

REPO='EZMONYI/music-ai-human-test-audio'


def verify():
    done=json.loads((OUT/'COMMIT.json').read_text())
    assert done['status']=='50_diffrhythm_pilot_originals_uploaded_hash_verified'
    assert done['audio_files']==50 and done['conditions']==10
    expected,_=build();revision=done['revision'];api=HfApi(token=False)
    assert not api.dataset_info(REPO,revision=revision).private
    def fetch(name):
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}'
        with urllib.request.urlopen(url,timeout=60) as response:return response.read()
    raw=fetch('manifest.json')
    assert hashlib.sha256(raw).hexdigest()==done['manifest_sha256']
    assert json.loads(raw)==expected
    metadata={}
    for name,wanted in [('README.md',NOTICE.encode()),('SOURCE_CARD.md',(SOURCES/'SOURCE_CARD.md').read_bytes()),
                        ('LICENSE.txt',(SOURCES/'LICENSE.txt').read_bytes())]:
        actual=fetch(name);assert actual==wanted
        metadata[name]=hashlib.sha256(actual).hexdigest()
    bypath={r['path']:r for r in expected}
    entries=api.get_paths_info(REPO,paths=list(bypath),repo_type='dataset',revision=revision)
    assert {e.path for e in entries}==set(bypath)
    for entry in entries:
        assert entry.size==bypath[entry.path]['bytes']
        assert entry.lfs and entry.lfs.sha256==bypath[entry.path]['sha256']
    return dict(revision=revision,verified_audio_files=50,conditions=10,
        checked_bytes=sum(r['bytes'] for r in expected),manifest_sha256=done['manifest_sha256'],
        metadata_sha256=metadata,source_records_bound=True,final_acceptance=True,
        verification='Anonymous manifest/notices equality and all LFS hashes/sizes; not a full audio redownload',
        whole_project_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();assert not args.receipt.exists()
    result=verify()
    with args.receipt.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
