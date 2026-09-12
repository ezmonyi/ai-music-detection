"""Publish verified complete result bundles and thesis v6 (no audio)."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/completed_results_publication_v1')
PIN='d392cff45fc86beac373faca19fe55aa2349e47fdbf93a5b8111976b37bfd152'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    token=json.load(sys.stdin)['token'];api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi' and sha(ROOT/'COMMIT.json')==PIN
    commit=json.loads((ROOT/'COMMIT.json').read_text())
    products=dict(commit['products']);products['COMMIT.json']=dict(sha256=PIN,bytes=(ROOT/'COMMIT.json').stat().st_size)
    prefix='reports/completed_experiments_thesis_v6_20260912/'
    operations=[];expected={}
    for name,bound in products.items():
        path=ROOT/name;assert sha(path)==bound['sha256'] and path.stat().st_size==bound['bytes']
        operations.append(CommitOperationAdd(path_in_repo=prefix+name,path_or_fileobj=str(path)))
        expected[prefix+name]=dict(bound)
        if bound['bytes']<2_000_000:
            raw=path.read_bytes();expected[prefix+name]['git_blob_sha1']=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    repo='EZMONYI/music-ai-human-test-audio'
    def verify(entries):
        assert {e.path for e in entries}==set(expected)
        for e in entries:
            b=expected[e.path];assert e.size==b['bytes']
            assert (e.lfs.sha256==b['sha256']) if e.lfs else (e.blob_id==b['git_blob_sha1'])
    revision=api.repo_info(repo,repo_type='dataset').sha
    entries=api.get_paths_info(repo,paths=list(expected),repo_type='dataset',revision=revision)
    if entries:verify(entries)
    else:
        revision=api.create_commit(repo_id=repo,repo_type='dataset',operations=operations,
            commit_message='Preserve completed BC, expanded YuE2, Native60 results and thesis v6').oid
        verify(api.get_paths_info(repo,paths=list(expected),repo_type='dataset',revision=revision))
    receipt=dict(repo=repo,revision=revision,prefix=prefix,files_verified=len(expected),
        all_remote_hashes_verified=True,source_commit_sha256=PIN,audio_included=False,
        whole_project_delivery_complete=False)
    with Path(__file__).with_name('completed_results_publication_v1.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
