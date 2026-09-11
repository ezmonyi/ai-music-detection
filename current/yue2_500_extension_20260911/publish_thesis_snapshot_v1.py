"""Publish the preserved English thesis snapshot, retaining earlier HF versions."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download


def main():
    token=json.load(sys.stdin)['token']
    api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    repo='EZMONYI/music-ai-human-test-audio'
    root=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/thesis_snapshot_v1')
    manifest=json.loads((root/'files.json').read_text())
    operations=[]
    for name,expected in manifest.items():
        path=root/name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected
        operations.append(CommitOperationAdd(path_in_repo='reports/thesis_working_20260912_v1/'+name,path_or_fileobj=str(path)))
    operations.append(CommitOperationAdd(path_in_repo='reports/thesis_working_20260912_v1/files.json',path_or_fileobj=str(root/'files.json')))
    old=Path(hf_hub_download(repo,'README.md',repo_type='dataset',token=token)).read_text()
    update='''

## Update: English thesis snapshot and audio transfer

`reports/thesis_working_20260912_v1/` preserves an English working PDF and
its LaTeX source bundle, with SHA-256 checksums. The seven-family Native30
results are included. BC and YuE2 final evaluations are explicitly pending.
This is not the final submitted thesis.

MAESTRO v3.0.0 Native30 test views are being uploaded under
`audio/human_maestro_v3/native30/`; the manifest lists the planned 300 files
and is not itself proof that every file has arrived. Consult actual files
and the source-specific README for CC BY-NC-SA 4.0 attribution and changes.
The complete tested audio collection is still not archived.
'''
    assert '## Update: English thesis snapshot and audio transfer' not in old
    operations.append(CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=(old+update).encode()))
    result=api.create_commit(repo_id=repo,repo_type='dataset',operations=operations,
                             commit_message='Preserve English working thesis PDF and source bundle; document audio transfer')
    names=api.list_repo_files(repo,repo_type='dataset',revision=result.oid)
    assert all(op.path_in_repo in names for op in operations)
    receipt=dict(repo=repo,revision=result.oid,files=[op.path_in_repo for op in operations],final_thesis=False)
    with (root.parent/'hf_thesis_publication_v1.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main()
