#!/usr/bin/env python3
"""Hash local probe media against the completed strict remote inference audit."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--local-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit = json.loads(a.audit.read_text())
    if audit['status'] != 'passed':
        raise ValueError('Remote audit did not pass')
    expected = {}
    for item in audit['items']:
        item_id = item['item_id']
        expected[f'audio/{item_id}.wav'] = item['input']['sha256']
        for stem, record in item['stems'].items():
            expected[f'inference/demix/htdemucs/{item_id}/{stem}.wav'] = record['sha256']
        for directory, field, suffix in (('beats', 'beats', 'beats'), ('structure', 'structure', 'json'), ('spec', 'spectrogram', 'npy')):
            expected[f'inference/{directory}/{item_id}.{suffix}'] = item[field]['sha256']
    rows = []
    for relative, expected_hash in sorted(expected.items()):
        path = a.local_root / relative
        digest = sha(path)
        if digest != expected_hash:
            raise ValueError(f'Local copy hash mismatch: {relative}')
        rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest})
    out = {'status': 'passed', 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
           'source_audit_sha256': sha(a.audit), 'local_root': str(a.local_root.resolve()), 'records': rows}
    if a.output.exists():
        raise ValueError('Preserve existing completed local copy audit')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k:v for k,v in out.items() if k != 'records'}))


if __name__ == '__main__':
    main()
