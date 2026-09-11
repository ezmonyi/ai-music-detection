"""Publish the private archival card using stdin-only ephemeral authentication."""
import json
import sys
from huggingface_hub import HfApi

payload = json.load(sys.stdin)
api = HfApi(token=payload['token'])
repo = 'EZMONYI/music-ai-human-interpretable-results'
assert api.whoami()['name'].lower() == 'ezmonyi'
assert api.dataset_info(repo).private
commit = api.upload_file(
    repo_id=repo, repo_type='dataset', path_in_repo='README.md',
    path_or_fileobj='/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/hf_backup_20260909/README.md',
    commit_message='Document private backup scope and incomplete upload status')
print(json.dumps({'commit': commit.oid, 'url': commit.commit_url}))
