"""Publish only the 316 per-track-license-screened FMA processed inputs."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import json
import re
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify

ROOT=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_fma_test_views_v1')
PREFIX='audio/fma_historical_test_views_explicit_derivative_v1/'
NOTICE='''# FMA historical processed inputs: per-recording licenses

316 historical cropped/standardized files, copied unchanged from experiment
inputs. They are processed versions, not new independent songs. Each manifest
record preserves the original artist, title, FMA track ID and exact license URL.
Each processed file is distributed under its own original license, including
the original version and jurisdiction. There is no blanket folder license.
Preserve attribution; noncommercial and share-alike obligations apply where
stated. CC0 material retains its dedication. No endorsement is implied.

Source: Free Music Archive; FMA research dataset https://github.com/mdeff/fma
Citation: Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson,
FMA: A Dataset For Music Analysis, ISMIR 2017.
Modification: historical crop/standardization; exact view parameters and
original source hashes are recorded. No waveform changes occur at publication.
390 ND processed objects and 194 unresolved/mixed-license objects are excluded.
Metadata records document source declarations, not a warranty of every right.
No license is inferred for other project content. Whole-project coverage is
incomplete.
'''


def build():
    p=ROOT/'fma_test_view_rights_v2.json'
    assert digest(p)=='e308b3bf3e2a4b82f61e3734b23b1db7c8538129841bb90a1f9d502aa6c694fd'
    rows=[]; paths={}
    for obj in json.loads(p.read_text()):
        if obj['status']!='derivative_license_candidate_requires_notice': continue
        sources=obj['source_records']; urls={s['license_url'] for s in sources}
        assert len(urls)==1
        url=next(iter(urls))
        assert (re.fullmatch(r'https://creativecommons.org/licenses/(by|by-sa|by-nc|by-nc-sa)/(1\.0|2\.0|2\.5|3\.0|4\.0)/(us/)?',url)
                or url=='https://creativecommons.org/publicdomain/zero/1.0/')
        assert all(s['artist'] and s['title'] for s in sources)
        path=Path(obj['source_paths'][0]); name=PREFIX+obj['sha256']+path.suffix
        paths[name]=path
        rows.append(dict(path=name,sha256=obj['sha256'],bytes=obj['bytes'],license_url=url,
            memberships=obj['memberships'], modification='Historical cropped/standardized test input',
            original_records=[{k:v for k,v in s.items() if k not in ['local_path','derivative_publication_authorized']} for s in sources]))
    assert len(rows)==316
    return rows,paths


def worker(token):
    api=HfApi(token=token); assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    rows,paths=build(); manifest=OUT/'manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==rows
    else: save(manifest,rows)
    for i,start in enumerate(range(0,316,40)):
        batch=rows[start:start+40]; receipt=OUT/f'batch_{i:03d}.json'
        if receipt.exists():
            old=json.loads(receipt.read_text()); assert old['files']==[r['path'] for r in batch]
            verify(api,batch,old['revision']); continue
        ops=[]
        for r in batch:
            p=paths[r['path']]; assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256']
            ops.append(CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(p)))
        if i==0: ops += [CommitOperationAdd(path_in_repo=PREFIX+'manifest.json',path_or_fileobj=str(manifest)),CommitOperationAdd(path_in_repo=PREFIX+'README.md',path_or_fileobj=NOTICE.encode())]
        revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,commit_message=f'Preserve licensed FMA processed inputs {start+1}-{start+len(batch)} of 316').oid
        verify(api,batch,revision)
        save(receipt,dict(revision=revision,files=[r['path'] for r in batch],sha256_verified=True))
        print(f'Uploaded and verified {start+len(batch)}/316',flush=True)
    save(OUT/'COMMIT.json',dict(status='316_fma_test_views_uploaded_hash_verified',objects=316,manifest_sha256=digest(manifest),whole_project_complete=False))


if __name__=='__main__':
    from bounded_hf_http_v1 import install
    install(); transport.OUT=OUT; transport.worker=worker; transport.main()
