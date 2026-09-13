"""Publish already-public thesis release bytes; verify anonymous fixed revision."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from huggingface_hub import HfApi, CommitOperationAdd


def main():
    payload = json.load(sys.stdin)
    root = Path(payload['root'])
    api = HfApi(token=payload['token'])
    repo = 'EZMONYI/music-ai-human-test-audio'
    assert api.whoami()['name'].lower() == 'ezmonyi'
    info = api.dataset_info(repo)
    assert not info.private
    files = sorted(p for p in root.rglob('*') if p.is_file())
    assert files and all(p.suffix in {'.tex', '.bib', '.pdf', '.png', '.md'} for p in files)
    rows = []
    operations = []
    for p in files:
        data = p.read_bytes()
        path = 'reports/thesis_v8_20260913/' + p.relative_to(root).as_posix()
        rows.append(dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=data))
    revision = api.create_commit(repo_id=repo, repo_type='dataset', parent_commit=info.sha,
        operations=operations, commit_message='Preserve verified English thesis v8 and source release').oid
    for row in rows:
        url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/' + row['path']
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    receipt = dict(repo=repo, revision=revision, files=rows,
        anonymous_full_bytes_verified=True, audio_uploaded=False, whole_project_complete=False)
    out = root.parent / 'thesis_v8_publication_receipt.json'
    with out.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(dict(revision=revision, files=len(rows), verified=True)))


if __name__ == '__main__':
    main()
