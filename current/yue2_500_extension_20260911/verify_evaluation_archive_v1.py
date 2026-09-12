"""Stream-check every committed product inside an archive, without extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

def verify(path, commit_sha):
    members = {}
    commits = []
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and '..' not in name.parts, member.name
            if member.isdir():
                continue
            assert member.isfile() and member.name not in members, member.name
            stream = archive.extractfile(member)
            digest = hashlib.sha256()
            raw = bytearray() if len(name.parts) == 2 and name.name == 'COMMIT.json' else None
            size = 0
            while block := stream.read(1024 * 1024):
                digest.update(block)
                size += len(block)
                if raw is not None:
                    raw.extend(block)
            assert size == member.size
            members[member.name] = {'bytes': size, 'sha256': digest.hexdigest()}
            if raw is not None:
                assert digest.hexdigest() == commit_sha, 'source COMMIT hash mismatch'
                commits.append((name.parts[0], json.loads(raw)))
    assert len(commits) == 1
    prefix, commit = commits[0]
    expected = commit.get('products')
    if expected is None:
        expected = {'models/' + uid + '.json': binding
                    for uid, binding in commit['model_receipts'].items()}
        expected['contract.json'] = commit['contract']
    for relative, binding in expected.items():
        assert members.get(prefix + '/' + relative) == {
            'bytes': binding['bytes'], 'sha256': binding['sha256']}, relative
    return dict(status='all_committed_archive_members_hash_verified',
                products=len(expected), archive_files=len(members),
                source_commit_sha256=commit_sha,
                note='Additional uncommitted files, if any, are not experimental evidence.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--commit-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.commit_sha256), sort_keys=True))
