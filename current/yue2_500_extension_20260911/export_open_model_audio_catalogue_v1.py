"""Bind fresh original/view byte audit to historical experiment identities."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(audit, historical, out):
    commit = json.loads((audit / 'COMMIT.json').read_text())
    source = audit / 'verified_records.json'
    assert sha(source) == commit['manifest_sha256']
    assert commit['status'] == '1000_original_view_pairs_hash_verified_not_uploaded'
    rows = json.loads(source.read_text())
    history = {r['id']: r for r in csv.DictReader(historical.open())}
    selected = {k for k, r in history.items() if r['source_group'] in ('ACE-Step', 'HeartMuLa')}
    assert len(rows) == len(selected) == 1000 and {r['id'] for r in rows} == selected
    output = []
    for row in rows:
        h = history[row['id']]
        files = {r['kind']: r for r in row['files']}
        original, view = files['original'], files['standardized30']
        assert h['label'] == '1' and h['raw_sha256'] == original['sha256']
        s = row['standardization_record']
        output.append(dict(id=row['id'], source_group=h['source_group'],
            historical_role=h['role'], group_id=h['group_id'], prompt_id=row['prompt_id'],
            original_sha256=original['sha256'], original_bytes=original['bytes'],
            original_filename=Path(original['path']).name,
            original_sample_rate=s['source_sample_rate'], original_frames=s['source_frames'],
            original_duration_s=s['source_duration_s'],
            standardized_sha256=view['sha256'], standardized_bytes=view['bytes'],
            standardized_filename=Path(view['path']).name,
            standardized_sample_rate=s['output_sample_rate'],
            standardized_frames=s['output_frames'], padding_frames=s['padding_frames'],
            truncated_frames=s['truncated_frames']))
    out.mkdir(exist_ok=False)
    with (out / 'original_view_pairs.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(output)
    (out / 'README.md').write_text('''# ACE-Step and HeartMuLa original/view provenance

1,000 existing historical AI identities, 500 per model. Each original and its
standardized 30-second view passed a fresh server SHA-256 check against the
preserved standardization record. Original hashes and IDs also match the
historical 10-second input metadata. These are not 2,000 independent songs.

Original duration, sample rate, frame count, padding and truncation are explicit.
Do not treat the historical source_audio_path as proof that a file contains the
original bytes: some historical paths name standardized analysis-layout files.
The raw hash identifies the generated original; the separate standardized hash
identifies its analysis view. Shared Muse prompt groups must remain grouped
across models and with YuE2 wherever their prompt identities overlap.

This catalogue proves this original/view provenance join, not publication,
redistribution clearance, content deduplication or universal classifier use.
The complete unsanitized audit with server paths is preserved locally in the
report's current_results/open_model_audio_audit_v1 directory. Prompt text and
lyrics are not redistributed in this catalogue.
''')
    (out / 'COMMIT.json').write_text(json.dumps(dict(
        status='1000_original_view_pairs_audited_and_historical_ids_joined', rows=1000,
        audit_commit_sha256=sha(audit / 'COMMIT.json'),
        historical_manifest_sha256=sha(historical),
        products={p.name:dict(bytes=p.stat().st_size, sha256=sha(p)) for p in out.iterdir()},
        audio_uploaded=False, whole_project_complete=False), indent=2))
    print('Joined 1,000 identities and 2,000 original/view file hashes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('audit', 'historical', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.audit, args.historical, args.out)
