"""Inventory current sibling stems; do not claim unavailable historical hashes."""
import csv
import hashlib
import json
from pathlib import Path
from audit_historical_stems_v1 import sha

BASE = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT = BASE / 'legacy_nonvocal_current_file_audit_v1'
SOURCES = [(10, Path('/mnt/nfs-code/users/yi/external_generator_500_heuristics_20260904/results/heuristic_features.csv'),
            'a8f14b7ac29d4f34ad9c6fb8e0e62906a0cb1cbe8276988aa605b56043ab2432'),
           (30, Path('/mnt/nfs-code/users/yi/dynamics_rhythm_change_detector_extension_20260903/detector_results/new_features.csv'),
            'fb28a2189b6a23d83082a61fb09e2998baf55279ed131fe85d93d5e982965a80')]


def main():
    assert not OUT.exists()
    vocals = BASE / 'historical_spectral_stem_audit_v1/records.jsonl'
    assert sha(vocals) == '6174d83a98e716cb7cc661e7900f8bacf4c55acb0e21354efffe727ab9024c96'
    index = {}
    for line in vocals.read_text().splitlines():
        row = json.loads(line)
        assert row['status'] == 'verified'
        key = (int(row['duration_sec']), row['item_id'])
        assert key not in index
        index[key] = Path(row['path']).parent
    from collections import Counter
    counts, cache = Counter(), {}
    OUT.mkdir()
    with (OUT / 'records.jsonl').open('x') as sink:
        for duration, table, pin in SOURCES:
            assert sha(table) == pin
            with table.open() as source:
                rows = list(csv.DictReader(source))
            for row in rows:
                assert row['dynamics_signal'] == 'demucs_bass_plus_drums_plus_other'
                parent = index.get((duration, row['track']))
                for stem in ('bass', 'drums', 'other'):
                    path = parent / (stem + '.wav') if parent else None
                    digest, size, error = None, None, None
                    if path is not None:
                        if str(path) not in cache:
                            try:
                                cache[str(path)] = (sha(path), path.stat().st_size, None)
                            except OSError as exc:
                                cache[str(path)] = (None, None, type(exc).__name__)
                        digest, size, error = cache[str(path)]
                    status = 'not_anchored_in_selected_vocal_audit' if path is None else 'unreadable' if error else 'current_bytes_hashed'
                    counts[status] += 1
                    sink.write(json.dumps(dict(item_id=row['track'], duration_sec=duration,
                        stem=stem, path=str(path) if path else None, sha256=digest, bytes=size,
                        status=status, error=error, historical_hash_available=False,
                        path_basis='Sibling of verified historical vocal; historical dynamics directory argument not independently proven')) + '\n')
                    if sum(counts.values()) % 1000 == 0:
                        sink.flush(); print(json.dumps(dict(records=sum(counts.values()), statuses=dict(counts))), flush=True)
    summary = dict(records=sum(counts.values()), statuses=dict(counts), unique_attempted_paths=len(cache),
        records_sha256=sha(OUT/'records.jsonl'), historical_integrity_verified=False,
        whole_project_complete=False, scope='Current reconstructed sibling files only; unanchored upstream rows retained.')
    (OUT/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
