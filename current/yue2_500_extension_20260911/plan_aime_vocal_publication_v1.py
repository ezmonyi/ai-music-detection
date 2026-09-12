"""Bind historical AIME vocals to original generated-audio publication records."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from plan_open_model_vocal_publication_v1 import checked


def main(objects, originals, out):
    assert not out.exists()
    rows = checked(objects, '49ff713dc524189723b4adc6d9874d252d6437dbf6e97d25a0acff4ad760db01')
    source = checked(originals, 'b8c0ac0e9cc21b431c1813bcc3594463786123510be422048758947ab1fe9f70')
    index = {r['id']: r for r in source}
    assert len(index) == len(source) == 5000
    records, ids = [], set()
    for row in rows:
        member_ids = {m['item_id'] for m in row['memberships']}
        if not member_ids.intersection(index):
            continue
        assert member_ids <= index.keys()
        assert all(m['stem'] == 'vocals' and m['source_group'] == index[m['item_id']]['model']
                   for m in row['memberships'])
        assert row['source_group'] != 'MTG-Jamendo'
        ids.update(member_ids)
        records.append(dict(**row, original_records=[index[i] for i in sorted(member_ids)],
            transformation='Historical Demucs vocal separation; unchanged existing file bytes',
            release_basis='AIME publisher generated-audio CC BY 4.0 declaration; no independent underlying-rights warranty'))
    assert len(records) == len(ids) == 5000
    counts = Counter(r['source_group'] for r in records)
    assert len(counts) == 10 and set(counts.values()) == {500}
    out.mkdir()
    raw = (json.dumps(records, indent=2) + '\n').encode()
    (out / 'objects.json').write_bytes(raw)
    summary = dict(objects=5000, original_ids=5000, bytes=sum(r['bytes'] for r in records),
        models=dict(sorted(counts.items())), objects_sha256=hashlib.sha256(raw).hexdigest(),
        input_objects_sha256='49ff713dc524189723b4adc6d9874d252d6437dbf6e97d25a0acff4ad760db01',
        original_manifest_sha256='b8c0ac0e9cc21b431c1813bcc3594463786123510be422048758947ab1fe9f70',
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
