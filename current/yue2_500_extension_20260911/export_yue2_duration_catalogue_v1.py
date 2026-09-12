"""Export all 500 generated identities and explicit duration eligibility."""
import csv
import hashlib
import json
from pathlib import Path
import argparse


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source, datasets):
    commit = json.loads((source / 'COMMIT.json').read_text())
    bound = {}
    for name in ['metadata.json', 'excluded.json']:
        path = source / name
        assert sha(path) == commit['products'][name]['sha256']
        assert path.stat().st_size == commit['products'][name]['bytes']
        bound[name] = sha(path)
    eligible = json.loads((source / 'metadata.json').read_text())
    excluded = json.loads((source / 'excluded.json').read_text())
    assert len(eligible) == 276 and len(excluded) == 224
    all_rows = eligible + excluded
    assert len({r['id'] for r in all_rows}) == 500
    native30 = datasets / 'native30_expanded_v1'
    parent = json.loads((native30 / 'COMMIT.json').read_text())
    assert sha(native30 / 'cohort.csv') == parent['products']['cohort.csv']['sha256']
    cohort = {r['id']: r for r in csv.DictReader((native30 / 'cohort.csv').open()) if r['source_group'] == 'YuE2'}
    assert len(cohort) == 498
    fields = ['id', 'prompt_id', 'label', 'source_group', 'role', 'group_id',
              'native_frames', 'native_sample_rate_hz', 'source_sha256',
              'source_receipt_sha256', 'eligible_native30', 'eligible_native60',
              'native60_start_frame', 'native60_frames', 'native60_input_sha256']
    output = []
    for row in sorted(all_rows, key=lambda r: r['id']):
        n30 = row['native_frames'] >= 30 * row['native_sample_rate_hz']
        n60 = row['native_frames'] >= 60 * row['native_sample_rate_hz']
        assert n30 == (row['id'] in cohort)
        if n30:
            for key in ['role', 'group_id', 'source_group']:
                assert row[key] == cohort[row['id']][key]
        assert n60 == ('input_sha256' in row)
        record = {k: row[k] for k in fields[:10]}
        record.update(eligible_native30=int(n30), eligible_native60=int(n60),
                      native60_start_frame=row.get('start_frame', ''),
                      native60_frames=row.get('frames', ''),
                      native60_input_sha256=row.get('input_sha256', ''))
        output.append(record)
    out = datasets / 'yue2_duration_catalogue_v1'
    out.mkdir(exist_ok=False)
    with (out / 'generated_recordings.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    (out / 'README.md').write_text('''# YuE2 generation and duration catalogue

All 500 generated recording IDs, including two excluded from Native30.
Producer metadata and exclusion hashes were checked against the Native60 input
COMMIT. All 498 duration-eligible Native30 IDs, roles and groups were reconciled
with the published expanded Native30 catalogue. Native60 admits 276 recordings
and excludes 224; its exact crop offsets, lengths and input hashes are retained.
No padding is implied by either native-duration eligibility flag.

The two short recordings are ai_yue2_muse_cn_suno_cn_006375_0 (1,057,856 frames)
and ai_yue2_muse_en_suno_en_016463_0 (1,269,056 frames), both at 48 kHz.
Their exclusion from native-duration experiments does not erase the generated
source files or imply failed generation. Legacy padded-first30 analysis has a
different protocol and must not be treated as a native-duration measurement.

Source hashes identify original generated audio; Native60 input hashes identify
derived exact-length WAV files. Paths and private server details are omitted.
This metadata publication does not establish audio upload or redistribution
rights. Eligibility alone is not proof of measurement or classifier success;
refer to the separate committed experiment results. Preserve shared Muse prompt
groups across YuE2, ACE-Step and HeartMuLa when constructing splits.
''')
    (out / 'COMMIT.json').write_text(json.dumps(dict(
        rows=500, native30_eligible=498, native60_eligible=276,
        input_commit_sha256=sha(source / 'COMMIT.json'), source_bindings=bound,
        native30_catalogue_sha256=sha(native30 / 'cohort.csv'),
        products={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in out.iterdir()},
        audio_publication_verified=False), indent=2))
    print('Verified 500 generated IDs, 498 Native30 and 276 Native60 eligible')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--datasets', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.datasets)
