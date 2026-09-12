"""Publish only unchanged FMA MP3s with explicit mapped license versions."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd

PLAN=Path(__file__).with_name('fma_original_publication_plan_v1.json')
PIN='2aae3374be02122ce65874a64bb5bb8996403965a23613c22d01c02f15350ccc'
PREFIX='fma_originals_explicit_license_v1/'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    token=json.load(sys.stdin)['token'];api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    assert sha(PLAN)==PIN
    plan=json.loads(PLAN.read_text());rows=plan['rows']
    selected=[r for r in rows if r['status']=='explicit_version_original_candidate']
    assert len(rows)==500 and len(selected)==392
    operations=[];bound={};public=[]
    for row in rows:
        item={k:v for k,v in row.items() if k!='local_path'}
        item['label']='human'
        if row in selected:
            path=Path(row['local_path'])
            assert path.suffix=='.mp3' and sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
            assert row['transformations']=='none; unmodified FMA medium source MP3'
            target=PREFIX+'audio/'+row['id']+'.mp3'
            operations.append(CommitOperationAdd(path_in_repo=target,path_or_fileobj=str(path)))
            bound[target]=dict(bytes=row['bytes'],sha256=row['sha256'])
            item['published_path']=target
            item['status']='original_audio_in_this_release'
        else:item['published_path']=None
        public.append(item)
    notice=b'''# FMA: unchanged source MP3s with explicit license versions

This release contains 392 byte-identical source MP3s from the FMA medium
distribution, selected from the 500 FMA recordings used by this project.
These are FMA's approximately 30-second excerpts, NOT full-length songs.
No cropping, decoding/re-encoding, resampling, normalization or stem separation
was performed for this release. File renaming changes no audio bytes.

Attribution: the individual artists and track titles in manifest.json; original
FMA dataset by Michael Defferrard, Kirell Benzi, Pierre Vandergheynst and Xavier
Bresson, FMA: A Dataset For Music Analysis, ISMIR 2017.
Dataset and original archive retrieval: https://github.com/mdeff/fma
Original archive members and FMA track IDs are retained in manifest.json.

Each recording retains its own stated license, including noncommercial,
share-alike and no-derivatives terms where applicable. Exact license URLs are
provided per track. This research archive does not replace those licenses with
a blanket dataset license or promise unrestricted/commercial reuse. Attribution,
license notices and relevant restrictions must accompany subsequent sharing.
No-derivatives recordings are provided unchanged; this release does not grant
permission to publish modified versions. CC0 items retain their CC0 notice.

108 additional selected recordings have unmapped/ambiguous license labels and
are excluded from the audio release pending review; their identification and
exclusion reasons remain in the manifest. Project-created views and stems are
not uploaded by this publisher. Thus this is not the complete tested corpus.
'''
    for name,raw in [('README.md',notice),('manifest.json',json.dumps(dict(source_plan_sha256=PIN,
                  audio_count=392,held_count=108,rows=public),indent=2).encode())]:
        target=PREFIX+name;operations.append(CommitOperationAdd(path_in_repo=target,path_or_fileobj=raw))
        bound[target]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                          git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    repo='EZMONYI/music-ai-human-test-audio';names=list(bound)
    def inspect(revision):
        entries=[]
        for i in range(0,len(names),50):
            entries.extend(api.get_paths_info(repo,paths=names[i:i+50],repo_type='dataset',revision=revision))
        return entries
    def verify(entries):
        assert {e.path for e in entries}==set(names)
        for entry in entries:
            target=bound[entry.path];assert entry.size==target['bytes']
            assert (entry.lfs.sha256==target['sha256']) if entry.lfs else (entry.blob_id==target['git_blob_sha1'])
    revision=api.repo_info(repo,repo_type='dataset').sha
    prior=inspect(revision)
    if prior:verify(prior)
    else:
        result=api.create_commit(repo_id=repo,repo_type='dataset',operations=operations,
            commit_message='Preserve 392 unchanged FMA source MP3s with per-track licenses')
        revision=result.oid;verify(inspect(revision))
    receipt=dict(repo=repo,revision=revision,prefix=PREFIX,verified_files=len(names),audio_files=392,
                 held_audio_files=108,all_remote_hashes_verified=True,derived_audio_uploaded=False)
    target=Path(__file__).with_name('fma_originals_publication_v1.json')
    if not target.exists():
        with target.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
