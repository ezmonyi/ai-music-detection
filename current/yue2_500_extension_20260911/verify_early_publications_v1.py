"""Anonymous post-upload verification of early FMA or Suno publication."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
REPO='EZMONYI/music-ai-human-test-audio'


def verify(kind,root):
    count=162 if kind=='fma' else 100
    prefix=f'audio/early_{kind}_originals_v1/'
    output=root/f'hf_early_{kind}_originals_v1'
    terminal=json.loads((output/'COMMIT.json').read_text())
    assert terminal['status']==f'{count}_early_{kind}_originals_uploaded_hash_verified'
    assert terminal['audio_files']==count
    receipt=json.loads((output/'batch_000.json').read_text())
    assert receipt['revision']==terminal['revision'] and receipt['sha256_verified'] is True
    revision=terminal['revision'];api=HfApi(token=False)
    assert not api.dataset_info(REPO,revision=revision).private
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{prefix}manifest.json'
    with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()
    assert hashlib.sha256(raw).hexdigest()==terminal['manifest_sha256']
    public=json.loads(raw)
    if kind=='fma':
        source=(root/'early_fma_publication_plan_v1/records.json').read_bytes()
        assert hashlib.sha256(source).hexdigest()=='b3a1b9f926873b1bcdc07a0ac3073e4fdadf73766625afa7c7b3e56a9008d01b'
        expected=json.loads(source)
        assert len(public)==len(expected)==200 and terminal['held_files']==38
    else:
        source=(root/'early_original_audio_audit_v1/originals.json').read_bytes()
        proof=json.loads((root/'early_suno_source_verification_v1.json').read_text())
        assert hashlib.sha256(source).hexdigest()==proof['local_audit_manifest_sha256']
        expected=[r for r in json.loads(source) if r['memberships'][0]['source']=='suno_unknown']
        assert len(public)==len(expected)==100
    assert len({r['relative_source_path'] for r in public})==len(public)
    by_path={r['relative_source_path']:r for r in public};files={};held=[]
    for row in expected:
        actual=by_path[row['relative_source_path']]
        assert all(actual[k]==v for k,v in row.items())
        if kind=='fma':
            path=prefix+str(row['fma_track_id'])+'.mp3'
            if row['status']=='hold_ambiguous_license':
                assert actual['published_path'] is None;held.append(path);continue
        else:path=prefix+Path(row['relative_source_path']).name
        assert actual['published_path']==path;files[path]=row
    assert len(files)==count
    assert set(receipt['files'])==set(files)|{prefix+'manifest.json',prefix+'README.md'}
    assert len(receipt['files'])==count+2
    paths=list(files)
    for start in range(0,count,50):
        entries=api.get_paths_info(REPO,paths=paths[start:start+50],repo_type='dataset',revision=revision)
        assert {e.path for e in entries}==set(paths[start:start+50])
        for e in entries:
            assert e.size==files[e.path]['bytes'] and e.lfs and e.lfs.sha256==files[e.path]['sha256']
    if held:assert not api.get_paths_info(REPO,paths=held,repo_type='dataset',revision=revision)
    return dict(kind=kind,revision=revision,verified_audio_files=count,excluded_files_absent=len(held),
        public_manifest_sha256=terminal['manifest_sha256'],source_records_bound=True,
        final_acceptance=True,whole_project_complete=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['fma','suno'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();result=verify(args.kind,args.root)
    with args.receipt.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
