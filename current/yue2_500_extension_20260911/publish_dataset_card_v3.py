"""Publish the scoped status correction; preserve the prior card and verify bytes."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from huggingface_hub import HfApi, CommitOperationAdd

REPO='EZMONYI/music-ai-human-test-audio'
ROOT=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')


def get(revision):
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/README.md'
    with urllib.request.urlopen(url,timeout=60) as response:return response.read()


def main():
    payload=json.load(sys.stdin);api=HfApi(token=payload['token'])
    assert api.whoami()['name'].lower()=='ezmonyi'
    new=(ROOT/'HF_DATASET_CARD_20260913_v3.md').read_bytes()
    assert hashlib.sha256(new).hexdigest()==payload['new_sha256']
    info=api.dataset_info(REPO);assert not info.private
    old=get(info.sha)
    if old==new:
        revision=info.sha
    else:
        assert hashlib.sha256(old).hexdigest()==payload['old_sha256'],'Card changed; inspect before updating'
        prior=ROOT/'HF_DATASET_CARD_before_v3.md'
        if prior.exists():assert prior.read_bytes()==old
        else:
            with prior.open('xb') as stream:stream.write(old)
        revision=api.create_commit(repo_id=REPO,repo_type='dataset',parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=new)],
            commit_message='Correct completed experiment status and distinguish incomplete audio archive').oid
    assert get(revision)==new
    receipt=dict(repo=REPO,revision=revision,readme_sha256=payload['new_sha256'],
        prior_readme_sha256=payload['old_sha256'],public_bytes_verified=True,
        audio_uploaded=False,whole_project_complete=False)
    out=ROOT/'dataset_card_v3_publication.json'
    if out.exists():assert json.loads(out.read_text())['readme_sha256']==payload['new_sha256']
    else:
        with out.open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main()
