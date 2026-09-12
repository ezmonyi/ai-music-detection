"""Deduplicate audited test inputs without inferring derivative publication rights."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main(audit, out):
    assert not out.exists()
    commit = json.loads((audit / 'COMMIT.json').read_text())
    raw = (audit / 'records.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == commit['records_sha256']
    rows = json.loads(raw)
    assert len(rows) == commit['memberships']
    objects = {}; by_source = defaultdict(set)
    for r in rows:
        assert r['error'] is None and r['sha256'] and r['bytes'] > 0
        key = r['sha256']
        obj = objects.setdefault(key, dict(sha256=key, bytes=r['bytes'],
            source_paths=[], memberships=[], public_paths=[], publication_authorized=False))
        assert obj['bytes'] == r['bytes']
        if r['audio_path'] not in obj['source_paths']:
            obj['source_paths'].append(r['audio_path'])
        obj['memberships'].append({k:r[k] for k in
            ['id', 'source_group', 'view', 'crop_start_s', 'audio_offset_s', 'requires_crop']})
        obj['public_paths'] = sorted(set(obj['public_paths']) | set(r['public_paths']))
        by_source[r['source_group']].add(key)
    records = sorted(objects.values(), key=lambda r:r['sha256'])
    missing = [r for r in records if not r['public_paths']]
    summary = dict(input_records_sha256=commit['records_sha256'],
        public_revision=commit['public_revision'], unique_objects=len(records),
        unpublished_objects=len(missing), unpublished_bytes=sum(r['bytes'] for r in missing),
        memberships=len(rows), whole_project_complete=False,
        scope='Audited audio_path files only; does not enumerate separated stems or render runtime crops',
        source_counts=[dict(source_group=s, unique_objects=len(keys),
            unpublished_objects=sum(not objects[k]['public_paths'] for k in keys),
            unpublished_bytes=sum(objects[k]['bytes'] for k in keys if not objects[k]['public_paths']))
            for s, keys in sorted(by_source.items())])
    out.mkdir()
    data = (json.dumps(records, indent=2)+'\n').encode()
    (out/'objects.json').write_bytes(data)
    summary['objects_sha256'] = hashlib.sha256(data).hexdigest()
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); main(a.audit, a.out)
