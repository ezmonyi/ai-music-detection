"""Bind historical open-model vocals to preserved original publication records."""
import argparse
import hashlib
import json
from pathlib import Path


def checked(path, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, path
    return json.loads(raw)


def main(objects, originals, out):
    assert not out.exists()
    rows = checked(objects, '49ff713dc524189723b4adc6d9874d252d6437dbf6e97d25a0acff4ad760db01')
    source = checked(originals, '7be5aaa118e77181aa489ef3462b254de6f37679e69cb99e690d069387e4f5f5')
    index = {}
    for r in source:
        index.setdefault(r['id'], []).append(r)
    assert len(index) == 1000
    records = []
    ids = set()
    for row in rows:
        if row['source_group'] not in ('ACE-Step', 'HeartMuLa'):
            continue
        member_ids = {m['item_id'] for m in row['memberships']}
        assert all(m['stem'] == 'vocals' for m in row['memberships'])
        assert member_ids <= index.keys()
        ids.update(member_ids)
        records.append(dict(**row, original_records=[r for i in sorted(member_ids) for r in index[i]],
            transformation='Historical Demucs vocal separation; unchanged existing file bytes',
            release_basis='Existing project-generated original release provenance; no blanket output-rights warranty'))
    assert len(records) == 1958 and len(ids) == 1000
    out.mkdir()
    raw = (json.dumps(records, indent=2) + '\n').encode()
    (out / 'objects.json').write_bytes(raw)
    summary = dict(objects=1958, original_ids=1000, bytes=sum(r['bytes'] for r in records),
        objects_sha256=hashlib.sha256(raw).hexdigest(),
        input_objects_sha256='49ff713dc524189723b4adc6d9874d252d6437dbf6e97d25a0acff4ad760db01',
        original_manifest_sha256='7be5aaa118e77181aa489ef3462b254de6f37679e69cb99e690d069387e4f5f5',
        status='source_bound_plan_not_uploaded', whole_project_complete=False)
    (out / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects', type=Path, required=True)
    p.add_argument('--originals', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.objects, a.originals, a.out)
