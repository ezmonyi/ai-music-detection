#!/usr/bin/env python3
"""Compare completed 10s raw prediction/fold copies with live remote hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def main():
    root = Path(__file__).resolve().parent.parent
    local = root/'results/evaluation_sdrfh_10s_v2'
    remote = '/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/results/evaluation_sdrfh_10s_v2'
    names = ['development_group_cv_predictions.csv', 'development_source_holdout_predictions.csv',
             'development_missingness_group_cv_predictions.csv',
             'development_missingness_source_holdout_predictions.csv', 'fold_registry.jsonl']
    command = 'sha256sum ' + ' '.join(remote+'/'+name for name in names)
    response = subprocess.check_output(['ssh', '-o', 'ConnectTimeout=15', '5090-2', command], text=True)
    expected = {Path(line.split()[1]).name: line.split()[0] for line in response.splitlines()}
    records = []
    for name in names:
        path = local/name
        before = path.stat()
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1 << 20), b''):
                h.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Local file changed: '+name)
        if h.hexdigest() != expected[name]:
            raise ValueError('Local/remote hash mismatch: '+name)
        records.append(dict(name=name, bytes=after.st_size, sha256=h.hexdigest()))
    receipt = dict(status='passed', verified_utc=datetime.now(timezone.utc).isoformat(),
                   remote_host='5090-2', remote_directory=remote, local_directory=str(local),
                   files=records, total_bytes=sum(r['bytes'] for r in records))
    out = root/'audit/local_10s_raw_sync_20260907.json'
    with out.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
