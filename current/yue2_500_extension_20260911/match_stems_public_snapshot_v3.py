"""Match two distinct stem evidence scopes to a pinned public file snapshot."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path, pin, lines=False):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin
    return [json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw)


def main(report, out):
    assert not out.exists()
    public = read(report/'current_results/public_audio_snapshot_20260913_v3/files.json',
        'bca7457d23d1fbf105bad2b1178b9bf9957c1c9782d9f3418a9dbf65f11895e2')
    lookup = {}
    for row in public:
        if row['sha256']:
            lookup.setdefault(row['sha256'], []).append(row)
    historical = read(report/'datasets/historical_stem_union_v1/objects.json',
        'c139675d81142fdc596d7e315c24ce3e32a223c533b8322bf3e0f92252eee975')
    current = read(report/'current_results/legacy_nonvocal_current_file_audit_v1/records.jsonl',
        '34cfb5894eb3ce71828eea3b6d9c63ed88772f8606ca9fbecfb216298c50d9f2', True)
    summaries, results = {}, {}
    for name, rows in [('historical_hash_verified', historical),
                       ('current_only_inventory', [r for r in current if r['status']=='current_bytes_hashed'])]:
        unique = {r['sha256']:r for r in rows}
        matched, missing = {}, []
        for key, row in unique.items():
            candidates = lookup.get(key, [])
            assert all(x['bytes']==row['bytes'] for x in candidates)
            if candidates:
                matched[key] = [x['path'] for x in candidates]
            else:
                missing.append(key)
        summaries[name] = dict(objects=len(unique), matched=len(matched), unmatched=len(missing),
            unmatched_bytes=sum(unique[k]['bytes'] for k in missing))
        results[name] = dict(matched=matched, unmatched=sorted(missing))
    summary = dict(revision='b5231d23713673e90c65e70767485ef5aaf70a56', scopes=summaries,
        limitations='Direct public files only; archive contents and private backups excluded. Scopes overlap; do not sum. Current-only matches do not establish historical provenance. No rights clearance implied.',
        whole_project_complete=False)
    out.mkdir()
    raw = (json.dumps(results, separators=(',', ':'))+'\n').encode()
    (out/'matches.json').write_bytes(raw)
    summary['matches_sha256'] = hashlib.sha256(raw).hexdigest()
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.report,a.out)
