"""Deduplicate verified historical stems against an immutable public snapshot."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def checked(path, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, path
    return raw


def main(audit, snapshot, out):
    assert not out.exists()
    ac = json.loads((audit / 'COMMIT.json').read_text())
    sc = json.loads((snapshot / 'COMMIT.json').read_text())
    rows = [json.loads(line) for line in checked(audit / 'records.jsonl', ac['records_sha256']).splitlines()]
    public = json.loads(checked(snapshot / 'files.json', sc['records_sha256']))
    assert len(rows) == ac['memberships']
    assert dict(Counter(r['status'] for r in rows)) == ac['statuses']
    index = defaultdict(list)
    for row in public:
        if row.get('sha256'):
            index[row['sha256']].append(row['path'])
    objects = {}
    # Fail closed rather than silently dropping unmatched historical input hashes.
    for row in rows:
        assert row['status'] == 'verified', row
        key = row['sha256']
        assert key == row['expected_sha256'] and row['bytes'] > 0 and not row['error']
        obj = objects.setdefault(key, dict(sha256=key, bytes=row['bytes'],
            source_paths=[], memberships=[], public_paths=sorted(index[key]),
            publication_rights_status='not_assessed_by_this_inventory'))
        assert obj['bytes'] == row['bytes']
        if row['path'] not in obj['source_paths']:
            obj['source_paths'].append(row['path'])
        obj['memberships'].append({k: row[k] for k in ('run', 'item_id', 'stem', 'duration_sec')})
    records = sorted(objects.values(), key=lambda r: r['sha256'])
    missing = [r for r in records if not r['public_paths']]
    summary = dict(audit_records_sha256=ac['records_sha256'],
        public_records_sha256=sc['records_sha256'], public_revision=sc['revision'],
        memberships=len(rows), unique_objects=len(records),
        snapshot_matched_objects=len(records) - len(missing),
        absent_at_snapshot_objects=len(missing), absent_at_snapshot_bytes=sum(r['bytes'] for r in missing),
        whole_project_complete=False,
        scope='Only verified stem inputs from selected historical runs. Snapshot absence is not current absence or permission to publish. No musical-content deduplication.')
    out.mkdir()
    raw = (json.dumps(records, indent=2) + '\n').encode()
    (out / 'objects.json').write_bytes(raw)
    summary['objects_sha256'] = hashlib.sha256(raw).hexdigest()
    (out / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.audit, a.snapshot, a.out)
