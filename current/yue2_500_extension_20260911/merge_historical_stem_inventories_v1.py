"""Union two accepted stem inventories while preserving experimental memberships."""
import argparse
import hashlib
import json
from pathlib import Path


def main(datasets, out):
    assert not out.exists()
    merged, pins, total = {}, {}, 0
    for name in ('historical_four_stem_objects_v1', 'historical_spectral_stem_objects_v1'):
        folder = datasets / name
        commit = json.loads((folder / 'COMMIT.json').read_text())
        raw = (folder / 'objects.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == commit['objects_sha256']
        pins[name] = commit['objects_sha256']
        rows = json.loads(raw)
        assert len(rows) == commit['unique_objects']
        total += len(rows)
        for row in rows:
            key = row['sha256']
            obj = merged.setdefault(key, dict(sha256=key, bytes=row['bytes'],
                source_paths=[], memberships=[], input_inventories=[]))
            assert obj['bytes'] == row['bytes']
            obj['source_paths'] = sorted(set(obj['source_paths']) | set(row['source_paths']))
            obj['input_inventories'].append(name)
            for member in row['memberships']:
                if member not in obj['memberships']:
                    obj['memberships'].append(member)
    rows = sorted(merged.values(), key=lambda r: r['sha256'])
    out.mkdir()
    raw = (json.dumps(rows, separators=(',', ':')) + '\n').encode()
    (out / 'objects.json').write_bytes(raw)
    summary = dict(input_objects_sha256=pins, input_object_entries=total,
        unique_objects=len(rows), duplicate_object_entries=total-len(rows),
        unique_bytes=sum(r['bytes'] for r in rows),
        unique_memberships=sum(len(r['memberships']) for r in rows),
        objects_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Union of two accepted historical stem inventories only; byte identity, not song identity. No upload or rights acceptance.',
        whole_project_complete=False)
    (out / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.datasets, a.out)
