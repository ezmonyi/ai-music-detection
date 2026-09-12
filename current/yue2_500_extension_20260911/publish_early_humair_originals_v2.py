"""Publish 100 unchanged early humair originals with recovered attribution variants."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
import publish_maestro_native30_v1 as transport
from prepare_fma_original_publication_v1 import WORK,sha

BASE=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_early_humair_originals_v2')
PREFIX='audio/early_humair_originals_v2/'
PIN='d3cba828ae9ce21dba5dbd5bf6fe1d8773793d1197a8f7e5fd1e9829b81a4343'
NOTICE='''# Early humair Suno originals with upstream attribution

100 unchanged original MP3 files from humair025/suno-audio, selected in the
early frequency study. Source publisher: humair025 / Humair332.
https://huggingface.co/datasets/humair025/suno-audio
Source card revision: 344c67dd2992063779b8f40504ff112f6083e7f2.
The publisher declares MIT; the complete source card is included. This is not
independent clearance of all third-party rights or a blanket project license.

Individual titles, creator display names and handles are supplied in the
manifest from exact UUID matches in nyuuzyou/suno revision
dd95495c415eea043c250f12da595de2ad4cad7f.
https://huggingface.co/datasets/nyuuzyou/suno
One UUID has multiple attribution variants, all preserved explicitly; none is
silently asserted to be the unique historical value. Duplicate source rows
do not create additional music samples. No lyrics, prompts or artwork are copied.

No audio transformation was performed. Historical memberships and raw hashes
are preserved. These are original files, not separated stems. This archive
does not relabel or rerun the experiments. See the manifest for source paths.
'''


def worker(token):
    api=HfApi(token=token);assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    source=BASE/'early_humair_upstream_v1/recovered_metadata_v2/records.json'
    assert sha(source)==PIN;rows=json.loads(source.read_text());assert len(rows)==100
    ops=[];expected={};public=[]
    for r in rows:
        p=WORK/r['relative_source_path'];assert p.resolve().is_relative_to(WORK.resolve())
        assert p.stat().st_size==r['original_bytes'] and sha(p)==r['original_sha256']
        assert any(v['display_name'] or v['handle'] for v in r['metadata_variants'])
        name=PREFIX+r['id']+'.mp3';public.append(dict(r,published_path=name))
        ops.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(p)))
        expected[name]=(r['original_bytes'],r['original_sha256'],None)
    card=(WORK/'humair025-suno-audio-meta/README.md').read_bytes()
    assert b'license: mit' in card[:200]
    manifest=json.dumps(public,ensure_ascii=False,indent=2).encode()
    for name,raw in [('manifest.json',manifest),('README.md',NOTICE.encode()),('SOURCE_CARD.md',card)]:
        name=PREFIX+name;ops.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=raw))
        expected[name]=(len(raw),hashlib.sha256(raw).hexdigest(),hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
        commit_message='Archive 100 early humair originals with recovered creator attribution variants').oid
    names=list(expected)
    for offset in range(0,len(names),50):
        entries=api.get_paths_info(transport.REPO,paths=names[offset:offset+50],repo_type='dataset',revision=revision)
        assert {e.path for e in entries}==set(names[offset:offset+50])
        for e in entries:
            size,digest,blob=expected[e.path];assert e.size==size
            assert e.lfs.sha256==digest if e.lfs else e.blob_id==blob
    transport.save(OUT/'COMMIT.json',dict(status='100_early_humair_originals_uploaded_hash_verified',
        revision=revision,audio_files=100,metadata_records_sha256=PIN,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),whole_project_complete=False))
    print('100 early humair originals published and verified',flush=True)


if __name__=='__main__':
    transport.OUT=OUT;transport.worker=worker;transport.main()
