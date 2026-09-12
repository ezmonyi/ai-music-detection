"""Audit existing ACE-Step/HeartMuLa originals and views; no regeneration/upload."""
import hashlib
import json
from pathlib import Path

BASE = Path('/mnt/nfs-code/users/yi/open_models_spectral_500_20260901')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/open_model_audio_audit_v1')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    records, inputs = [], {}
    for model in ('acestep', 'heartmula'):
        state = BASE / 'state' / f'{model}_standardization.jsonl'
        inputs[state.name] = digest(state)
        rows = [json.loads(line) for line in state.read_text().splitlines()]
        assert len(rows) == len({r['id'] for r in rows}) == 500
        for i, row in enumerate(rows):
            assert row['status'] == 'ok' and row['generator'] == model
            record = dict(id=f'ai_{model}_{row["id"]}', model=model,
                          prompt_id=row['id'], standardization_record=row,
                          files=[])
            for key, kind in [('source', 'original'), ('output', 'standardized30')]:
                path = Path(row[key])
                assert path.is_relative_to(BASE)
                expected = row[key + '_sha256']
                assert digest(path) == expected, str(path)
                record['files'].append(dict(kind=kind, path=str(path),
                    sha256=expected, bytes=path.stat().st_size))
            records.append(record)
            if (i+1) % 50 == 0:
                print(f'{model}: {i+1}/500 original/view pairs verified', flush=True)
    assert len({r['id'] for r in records}) == 1000
    manifest = OUT / 'verified_records.json'
    manifest.write_text(json.dumps(records, indent=2))
    (OUT / 'COMMIT.json').write_text(json.dumps(dict(
        status='1000_original_view_pairs_hash_verified_not_uploaded',
        identities=1000, audio_files=2000,
        bytes=sum(f['bytes'] for r in records for f in r['files']),
        source_state_sha256=inputs, manifest_sha256=digest(manifest),
        redistribution_clearance=False, classifier_membership_joined=False,
        whole_project_complete=False), indent=2))


if __name__ == '__main__':
    main()
