"""Publish 100 early Suno originals bound to verified public source bytes."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
import publish_maestro_native30_v1 as transport
from prepare_fma_original_publication_v1 import WORK,sha

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT=ROOT/'hf_early_suno_originals_v1'
PREFIX='audio/early_suno_originals_v1/'
REV='bdff424e70c10ef62dca13ba43659ad9e7e1fcdb'
NOTICE='''# Early Suno source originals

100 unchanged MP3 originals from the early frequency study, with selected
stem-pilot memberships preserved. Source: Kukedlc/suno-ai-music-dataset,
revision bdff424e70c10ef62dca13ba43659ad9e7e1fcdb.
https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset
Publisher-declared CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
Attribution: dataset publisher Kukedlc; original source UUIDs and paths are
preserved in the manifest. No endorsement or independent third-party-rights
clearance is implied. This is not a blanket license for all project audio.

All 100 local source files matched the fixed public source revision's LFS
SHA-256 and size. No audio transformations were applied. No lyrics, prompts,
artwork or signed URLs are republished here. The historical source category is
retained; this publication does not relabel experiments or refit classifiers.
These earlier inputs are separate from the later 500-file Suno publication.
A planned manifest alone is not proof of upload completion.
'''


def worker(token):
    api=HfApi(token=token);assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    proof=json.loads((ROOT/'early_suno_source_verification_v1.json').read_text())
    assert proof['status']=='100_early_suno_originals_match_pinned_public_source'
    assert proof['source_repo']=='Kukedlc/suno-ai-music-dataset' and proof['source_revision']==REV
    audit=ROOT/'early_original_audio_audit_v1/originals.json'
    assert sha(audit)==proof['local_audit_manifest_sha256']
    selected=[r for r in json.loads(audit.read_text()) if r['memberships'][0]['source']=='suno_unknown']
    verified={r['source_path']:r for r in proof['files']};assert len(selected)==len(verified)==100
    rows=[];ops=[];expected={}
    for r in selected:
        name=Path(r['relative_source_path']).name;p=WORK/r['relative_source_path']
        assert p.resolve().is_relative_to(WORK.resolve())
        original=verified['audio/'+name]
        assert sha(p)==r['sha256']==original['sha256'] and p.stat().st_size==r['bytes']==original['bytes']
        target=PREFIX+name
        rows.append(dict(r,published_path=target,source_dataset=proof['source_repo'],source_revision=REV,
                         publisher_declared_license='CC-BY-4.0',transformation='none'))
        ops.append(CommitOperationAdd(path_in_repo=target,path_or_fileobj=str(p)))
        expected[target]=(r['bytes'],r['sha256'],None)
    manifest=json.dumps(rows,indent=2).encode()
    for name,raw in [('manifest.json',manifest),('README.md',NOTICE.encode())]:
        target=PREFIX+name;ops.append(CommitOperationAdd(path_in_repo=target,path_or_fileobj=raw))
        expected[target]=(len(raw),hashlib.sha256(raw).hexdigest(),hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    receipt=OUT/'batch_000.json'
    if receipt.exists():revision=json.loads(receipt.read_text())['revision']
    else:revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
        commit_message='Archive 100 unchanged early Suno originals verified against fixed public source').oid
    names=list(expected)
    for start in range(0,len(names),50):
        entries=api.get_paths_info(transport.REPO,paths=names[start:start+50],repo_type='dataset',revision=revision)
        assert {r.path for r in entries}==set(names[start:start+50])
        for e in entries:
            size,digest,blob=expected[e.path];assert e.size==size
            assert e.lfs.sha256==digest if e.lfs else e.blob_id==blob
    if not receipt.exists():transport.save(receipt,dict(revision=revision,files=names,sha256_verified=True))
    transport.save(OUT/'COMMIT.json',dict(status='100_early_suno_originals_uploaded_hash_verified',
        revision=revision,audio_files=100,manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        source_revision=REV,whole_project_complete=False))
    print('100 early Suno originals uploaded and hash verified',flush=True)
