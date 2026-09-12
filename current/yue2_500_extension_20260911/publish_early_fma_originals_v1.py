"""Publish 162 unchanged early FMA excerpts; preserve 38 explicit holds."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
from prepare_fma_original_publication_v1 import WORK,sha
import publish_maestro_native30_v1 as transport

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT=ROOT/'hf_early_fma_originals_v1'
PLAN=ROOT/'early_fma_publication_plan_v1/records.json'
PIN='b3a1b9f926873b1bcdc07a0ac3073e4fdadf73766625afa7c7b3e56a9008d01b'
PREFIX='audio/early_fma_originals_v1/'
NOTICE='''# Early FMA study: unchanged licensed source excerpts

162 unchanged original FMA medium MP3 excerpts from the early frequency study;
38 other selected recordings are excluded because their license labels remain
ambiguous. The manifest preserves all 200 records and distinguishes exclusions.
These are approximately 30-second source excerpts, not full-length songs.
No decoding, re-encoding, cropping, normalization or stem separation was applied.

Attribution: see each row's artist and title. FMA dataset: Michael Defferrard,
Kirell Benzi, Pierre Vandergheynst and Xavier Bresson, ISMIR 2017.
Official dataset and archive retrieval: https://github.com/mdeff/fma

Each recording retains its own recorded license and linked version. Preserve
attribution, noncommercial, share-alike and no-derivatives restrictions wherever
applicable. ND recordings are redistributed unchanged; this release gives no
permission to publish modified versions. No blanket unrestricted license applies.

Original archive locators, hashes and historical memberships are retained.
Selected stem-pilot inputs overlap the early frequency-study inputs. File counts
are not independent song counts. Manifests describe planned coverage; final
upload receipts and independent verification establish actual publication.
'''


def worker(token):
    api=HfApi(token=token);assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(transport.REPO).private and sha(PLAN)==PIN
    rows=json.loads(PLAN.read_text());chosen=[];operations=[]
    for row in rows:
        r=dict(row);r['published_path']=None
        if r['status']=='explicit_license_unchanged_original_candidate':
            assert r['license_url'] and r['artist'] and r['title']
            path=WORK/r['relative_source_path']
            assert path.resolve().is_relative_to(WORK.resolve())
            assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
            r['published_path']=PREFIX+str(r['fma_track_id'])+'.mp3'
            operations.append(CommitOperationAdd(path_in_repo=r['published_path'],path_or_fileobj=str(path)))
        chosen.append(r)
    selected=[r for r in chosen if r['published_path']]
    assert len(chosen)==200 and len(selected)==162
    manifest=json.dumps(chosen,indent=2).encode()
    metadata={'manifest.json':manifest,'README.md':NOTICE.encode()}
    expected={r['published_path']:(r['bytes'],r['sha256'],None) for r in selected}
    for name,raw in metadata.items():
        target=PREFIX+name
        expected[target]=(len(raw),hashlib.sha256(raw).hexdigest(),hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
        operations.append(CommitOperationAdd(path_in_repo=target,path_or_fileobj=raw))
    receipt=OUT/'batch_000.json'
    if receipt.exists():revision=json.loads(receipt.read_text())['revision']
    else:
        revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=operations,
            commit_message='Archive 162 attributed unchanged early FMA excerpts; retain 38 license holds').oid
    names=list(expected)
    for start in range(0,len(names),50):
        entries=api.get_paths_info(transport.REPO,paths=names[start:start+50],repo_type='dataset',revision=revision)
        assert {r.path for r in entries}==set(names[start:start+50])
        for entry in entries:
            size,digest,blob=expected[entry.path];assert entry.size==size
            assert entry.lfs.sha256==digest if entry.lfs else entry.blob_id==blob
    if not receipt.exists():transport.save(receipt,dict(revision=revision,files=names,sha256_verified=True))
    transport.save(OUT/'COMMIT.json',dict(status='162_early_fma_originals_uploaded_hash_verified',
        revision=revision,audio_files=162,held_files=38,source_plan_sha256=PIN,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),whole_project_complete=False))
    print('162 early FMA originals uploaded and verified',flush=True)
