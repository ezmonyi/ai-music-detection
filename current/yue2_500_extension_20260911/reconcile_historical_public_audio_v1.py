"""Join recorded historical source hashes to fixed-revision public audio hashes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

HIST_PIN = 'd3519aa072498ec7812bd73b8456c0a47caba7bbac9801b1e4049df32a477361'


def reconcile(historical, snapshot, out):
    assert hashlib.sha256(historical.read_bytes()).hexdigest() == HIST_PIN
    commit = json.loads((snapshot/'COMMIT.json').read_text())
    raw = (snapshot/'files.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == commit['records_sha256']
    public = json.loads(raw)
    assert len(public) == commit['audio_file_paths']
    by_hash = defaultdict(list)
    for row in public:
        if row['sha256']: by_hash[row['sha256']].append(row['path'])
    rows = list(csv.DictReader(historical.open()))
    assert len(rows) == len({r['id'] for r in rows}) == 10141
    records = []; counts = defaultdict(lambda:dict(total=0, matched=0, absent=0, missing_recorded_hash=0))
    for row in rows:
        digest = row['raw_sha256']; matches = sorted(by_hash.get(digest, []))
        status = 'matched_public_sha256' if matches else ('absent_at_snapshot' if digest else 'missing_recorded_hash')
        records.append(dict(id=row['id'], source_group=row['source_group'], role=row['role'],
            recorded_raw_sha256=digest, status=status, public_paths=matches))
        c = counts[row['source_group']]; c['total'] += 1
        c['matched' if matches else 'absent' if digest else 'missing_recorded_hash'] += 1
    out.mkdir(exist_ok=False)
    data = (json.dumps(records, indent=2)+'\n').encode()
    (out/'records.json').write_bytes(data)
    with (out/'counts_by_source.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_group','total','matched','absent','missing_recorded_hash'])
        writer.writeheader()
        for source, c in sorted(counts.items()): writer.writerow(dict(source_group=source, **c))
    summary = dict(historical_ids=10141, matched=sum(c['matched'] for c in counts.values()),
        absent_at_snapshot=sum(c['absent'] for c in counts.values()),
        missing_recorded_hash=sum(c['missing_recorded_hash'] for c in counts.values()),
        public_revision=commit['revision'], public_inventory_sha256=commit['records_sha256'],
        historical_metadata_sha256=HIST_PIN, records_sha256=hashlib.sha256(data).hexdigest(),
        scope='Historical 10141 recorded raw hashes only; excludes archive members and later-only identities',
        exact_test_view_coverage_proven=False, whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    for source, c in sorted(counts.items()): print(source, c)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(); reconcile(a.historical,a.snapshot,a.out)
