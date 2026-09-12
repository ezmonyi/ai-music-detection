"""Validate a copied inventory and quantify byte-identical objects, without deletion."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(folder):
    raw = (folder / 'files.jsonl').read_bytes()
    commit = json.loads((folder / 'COMMIT.json').read_text())
    assert commit['status'] == 'fixed_scope_artifact_hash_inventory_complete'
    assert hashlib.sha256(raw).hexdigest() == commit['inventory_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == commit['files']
    assert len({r['path'] for r in rows}) == len(rows)
    assert sum(r['bytes'] for r in rows) == commit['logical_bytes']
    for row in rows:
        path = PurePosixPath(row['path'])
        assert not path.is_absolute() and '..' not in path.parts
        assert row['bytes'] >= 0
        assert len(row['sha256']) == 64
        int(row['sha256'], 16)
    expected = {d['directory'] for d in commit['directories']}
    assert {PurePosixPath(r['path']).parts[0] for r in rows} == expected
    for directory in commit['directories']:
        subset = [r for r in rows if PurePosixPath(r['path']).parts[0] == directory['directory']]
        assert len(subset) == directory['files']
        assert sum(r['bytes'] for r in subset) == directory['logical_bytes']
        assert dict(Counter(PurePosixPath(r['path']).suffix for r in subset)) == directory['extensions']
    objects = {(r['sha256'], r['bytes']) for r in rows}
    return dict(status='copied_inventory_verified', files=len(rows),
                logical_bytes=commit['logical_bytes'], unique_byte_objects=len(objects),
                unique_object_bytes=sum(size for _, size in objects),
                duplicate_file_entries=len(rows)-len(objects),
                inventory_sha256=commit['inventory_sha256'],
                scope='Seven fixed YuE extension artifact directories; not all project audio.',
                limitations='Hash identity is not song identity. Source bytes were hashed by the server producer, not reread by this verifier. No audio copied, deleted or uploaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.folder)
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(encoded)
    print(encoded)
