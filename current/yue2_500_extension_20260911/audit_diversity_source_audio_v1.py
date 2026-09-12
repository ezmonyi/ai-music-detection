"""Read-only audit of historical diversity source paths; report every mismatch."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

SOURCE_ROOT = Path('/mnt/nfs-data/users/yi/source_diversity_expansion_20260905/native')


def main(metadata, out):
    raw = metadata.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'a03aec930130fcb942c40b35ba5fb49109ceee588894124010448f041d9d1ef4'
    rows = [r for r in csv.DictReader(raw.decode().splitlines())
            if Path(r['source_audio_path']).is_relative_to(SOURCE_ROOT)]
    assert len(rows) == len({r['id'] for r in rows}) == 2641
    out.mkdir(exist_ok=False)
    counts, sizes = Counter(), Counter()
    with (out/'records.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            p = Path(row['source_audio_path'])
            result = dict(id=row['id'], source_group=row['source_group'],
                          path=str(p), source_locator=row['source_locator'],
                          expected_sha256=row['raw_sha256'])
            try:
                before = p.stat()
                h = hashlib.sha256()
                with p.open('rb') as audio:
                    for chunk in iter(lambda:audio.read(1<<20),b''):h.update(chunk)
                after = p.stat()
                result.update(bytes=after.st_size, actual_sha256=h.hexdigest())
                if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                    result['status'] = 'changed_during_read'
                elif h.hexdigest() != row['raw_sha256']:
                    result['status'] = 'hash_mismatch'
                else:
                    result['status'] = 'hash_verified'
                    sizes[row['source_group']] += after.st_size
            except FileNotFoundError:
                result['status'] = 'missing'
            except OSError as exc:
                result.update(status='io_error', error_type=type(exc).__name__, errno=exc.errno)
            counts[(row['source_group'],result['status'])] += 1
            stream.write(json.dumps(result)+'\n');stream.flush()
            if (index+1)%100 == 0:print(f'Audited {index+1}/2641 source paths',flush=True)
    report = dict(status='source_path_audit_completed_not_publication_acceptance',
        rows=len(rows),input_sha256=hashlib.sha256(raw).hexdigest(),
        records_sha256=hashlib.sha256((out/'records.jsonl').read_bytes()).hexdigest(),
        counts=[dict(source_group=k[0],status=k[1],count=v) for k,v in sorted(counts.items())],
        verified_bytes_by_source=dict(sizes),audio_uploaded=False,whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.metadata,a.out)
