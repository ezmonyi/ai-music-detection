"""Count cross-scope byte overlap without upgrading current-only evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main(report, out):
    assert not out.exists()
    p = report/'datasets/historical_stem_union_v1/objects.json'
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'c139675d81142fdc596d7e315c24ce3e32a223c533b8322bf3e0f92252eee975'
    prior = {r['sha256']: r for r in json.loads(raw)}
    p = report/'current_results/legacy_nonvocal_current_file_audit_v1/records.jsonl'
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '34cfb5894eb3ce71828eea3b6d9c63ed88772f8606ca9fbecfb216298c50d9f2'
    rows = [json.loads(line) for line in raw.splitlines()]
    current = {}
    for row in rows:
        if row['status'] != 'current_bytes_hashed':
            continue
        key = row['sha256']
        if key in current:
            assert current[key]['bytes'] == row['bytes']
        current[key] = row
    overlap = sorted(prior.keys() & current.keys())
    for key in overlap:
        assert prior[key]['bytes'] == current[key]['bytes']
    union = {**current, **prior}
    summary = dict(historical_hash_verified_objects=len(prior), current_only_inventory_objects=len(current),
        overlap_objects=len(overlap), unique_objects=len(union), unique_bytes=sum(r['bytes'] for r in union.values()),
        current_without_historical_hash_objects=len(current.keys()-prior.keys()),
        excluded_unanchored_memberships=sum(r['status'] != 'current_bytes_hashed' for r in rows),
        scope='Two selected stem inventory scopes; not all project audio. Current sibling path inference remains unproven historical provenance.',
        whole_project_complete=False)
    out.mkdir()
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'overlap.json').write_text(json.dumps([dict(sha256=k, historical=prior[k], current=current[k]) for k in overlap], indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); main(a.report, a.out)
