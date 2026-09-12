"""Hash actual historical audio_path files; do not confuse them with raw sources."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(metadata, snapshot, out):
    assert not out.exists()
    commit = json.loads((snapshot / 'COMMIT.json').read_text())
    assert sha(snapshot / 'files.json') == commit['records_sha256']
    public = defaultdict(list)
    for row in json.loads((snapshot / 'files.json').read_text()):
        if row['sha256']:
            public[row['sha256']].append(row['path'])
    cache = {}; records = []; pins = {}
    for source in metadata:
        pins[source.name] = sha(source)
        rows = list(csv.DictReader(source.open()))
        assert len(rows) == len({r['id'] for r in rows})
        for row in rows:
            path = Path(row['audio_path'])
            key = str(path)
            if key not in cache:
                try:
                    cache[key] = dict(sha256=sha(path), bytes=path.stat().st_size, error=None)
                except OSError as error:
                    cache[key] = dict(sha256=None, bytes=None, error=type(error).__name__)
                if len(cache) % 500 == 0:
                    print(f'Checked {len(cache)} unique paths', flush=True)
            result = cache[key]
            matches = public.get(result['sha256'], [])
            records.append(dict(id=row['id'], source_group=row['source_group'],
                view=row['duration_view'], audio_path=key,
                crop_start_s=row['crop_start_s'], audio_offset_s=row['audio_offset_s'],
                requires_crop=row['requires_crop'], **result, public_paths=matches,
                status='unreadable' if result['error'] else 'matched' if matches else 'unpublished_hash'))
    out.mkdir()
    data = (json.dumps(records, indent=2) + '\n').encode()
    (out / 'records.json').write_bytes(data)
    counts = Counter((r['view'], r['source_group'], r['status']) for r in records)
    with (out / 'counts.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['view', 'source_group', 'status', 'memberships'])
        writer.writerows([*key, value] for key, value in sorted(counts.items()))
    summary = dict(public_revision=commit['revision'], metadata_sha256=pins,
        public_inventory_sha256=commit['records_sha256'], memberships=len(records),
        unique_paths=len(cache), statuses=dict(Counter(r['status'] for r in records)),
        records_sha256=hashlib.sha256(data).hexdigest(),
        scope='Files referenced by audio_path only; crop settings retained, no claim of rendered crop or stem coverage',
        whole_project_complete=False)
    (out / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, nargs='+', required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.metadata, a.snapshot, a.out)
