"""Verify recorded Demucs inputs, without inferring use from directory contents."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(results, out):
    out.mkdir(exist_ok=False)
    cache, pins, counts = {}, {}, Counter()
    total = 0
    with (out / 'records.jsonl').open('x') as sink:
        for folder in results:
            states = sorted(folder.glob('expanded_features_*.jsonl'))
            spectral_only = not states
            if spectral_only:
                states = sorted(folder.glob('expanded_spectral_*.jsonl'))
            assert len(states) == 1, folder
            state = states[0]
            meta_path = state.with_name(state.stem + '_metadata.json')
            meta = json.loads(meta_path.read_text())
            pins[str(state)] = sha(state)
            pins[str(meta_path)] = sha(meta_path)
            roots = [Path(p) for p in meta['run_payload']['demix_roots']]
            seen = set()
            for line in state.read_text().splitlines():
                row = json.loads(line)
                item = row['item_id']
                assert item not in seen
                seen.add(item)
                assert row['run_fingerprint'] == meta['run_fingerprint']
                hashes = ({'vocals_sha256': row.get('vocal_stem_sha256')}
                          if spectral_only else row['input_hashes'])
                if isinstance(hashes, str):
                    hashes = json.loads(hashes)
                for stem in (('vocals',) if spectral_only else ('bass', 'drums', 'other', 'vocals')):
                    expected = hashes.get(stem + '_sha256')
                    candidates = [r / 'htdemucs' / item / (stem + '.wav') for r in roots]
                    path = next((p for p in candidates if p.is_file() and (spectral_only or p.stat().st_size)), None)
                    actual, size, error = None, None, None
                    if expected and path is not None:
                        key = str(path)
                        if key not in cache:
                            try:
                                cache[key] = (sha(path), path.stat().st_size, None)
                            except OSError as exc:
                                cache[key] = (None, None, type(exc).__name__)
                        actual, size, error = cache[key]
                    status = ('not_recorded' if not expected else 'missing' if path is None
                              else 'unreadable' if error else 'verified' if actual == expected
                              else 'hash_mismatch')
                    record = dict(run=str(folder), item_id=item, stem=stem,
                                  duration_sec=row.get('duration_sec', meta['run_payload'].get('duration')), expected_sha256=expected,
                                  path=str(path) if path else None, sha256=actual,
                                  bytes=size, error=error, status=status)
                    sink.write(json.dumps(record, sort_keys=True) + '\n')
                    counts[status] += 1
                    total += 1
                    if total % 1000 == 0:
                        sink.flush()
                        print(json.dumps(dict(memberships=total, statuses=dict(counts))), flush=True)
    summary = dict(input_sha256=pins, memberships=total, unique_hashed_paths=len(cache),
                   statuses=dict(counts), records_sha256=sha(out / 'records.jsonl'),
                   scope='Recorded four-stem or spectral-only vocal inputs in the explicitly selected runs only; not publication acceptance or whole-project coverage.',
                   whole_project_complete=False)
    (out / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.results, a.out)
