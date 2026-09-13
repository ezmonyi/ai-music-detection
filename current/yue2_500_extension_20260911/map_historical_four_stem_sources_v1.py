"""Bind verified four-stem objects to exact historical metadata; do not grant rights."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(datasets, out):
    assert not out.exists()
    inventory = datasets / 'historical_four_stem_objects_v1/objects.json'
    assert digest(inventory) == 'fd17c91421ae9693a8a15ba4eae5c3959dfcca3524a30861366ae864dc51a292'
    pins = {10: 'a03aec930130fcb942c40b35ba5fb49109ceee588894124010448f041d9d1ef4',
            30: 'eb313f97e2d69d743423efc170eecb24f1f00ee77ad789461772daecd058c242'}
    metadata = {}
    for duration, expected in pins.items():
        path = datasets / 'historical_manifest_sources_v1' / f'metadata_{duration}s.csv'
        assert digest(path) == expected
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        metadata[duration] = {r['id']: r for r in rows}
        assert len(metadata[duration]) == len(rows)
    records = json.loads(inventory.read_text())
    counts, sizes = Counter(), Counter()
    for obj in records:
        groups = set()
        for member in obj['memberships']:
            original = metadata[int(member['duration_sec'])][member['item_id']]
            member['source_group'] = original['source_group']
            groups.add(original['source_group'])
        assert len(groups) == 1, groups
        group = next(iter(groups))
        obj['source_group'] = group
        counts[group] += 1
        sizes[group] += obj['bytes']
    out.mkdir()
    (out / 'objects.json').write_text(json.dumps(records, indent=2) + '\n')
    summary = dict(objects=len(records), input_objects_sha256=digest(inventory),
        metadata_sha256=pins, objects_sha256=digest(out / 'objects.json'),
        sources=[dict(source_group=g, objects=n, bytes=sizes[g]) for g, n in sorted(counts.items())],
        scope='Exact ID and duration joins only. Per-recording attribution and derivative rights still require source-release joins.',
        publication_authorized_by_this_mapping=False, whole_project_complete=False)
    (out / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.datasets, a.out)

