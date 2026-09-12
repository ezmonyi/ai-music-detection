"""Publish a path-free, hash-bound cohort index, not an audio redistribution grant."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from verify_local_package_v1 import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    verification = verify(args.package)
    rows = []
    fields = ['id', 'label', 'source_group', 'role', 'group_id', 'component_id',
              'duration_view_s', 'native_sample_rate_hz', 'input_sha256']
    for name in ['metadata.json', 'locked_metadata.json']:
        for original in json.loads((args.package/name).read_text()):
            row = {field: original[field] for field in fields}
            row['label'] = int(row['label'])
            assert row['label'] in (0, 1)
            assert len(row['input_sha256']) == 64
            rows.append(row)
    assert len(rows) == 4328 and len({r['id'] for r in rows}) == 4328
    assert Counter(r['role'] for r in rows) == {'development':4228,'locked_test':100}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'cohort.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((r['source_group'],r['label'],r['role']) for r in rows)
    description = '''# Native30 expanded dataset catalogue v1

This is a verified index of the completed expanded Native30 feature cohort,
not the entire historical corpus and not a claim that all audio is uploaded.
It contains 4,228 development rows and 100 locked YuE2 rows (4,328 unique IDs).
Label 0 means source-labelled human music; label 1 means source-labelled AI.
Labels describe source provenance, not a claim of forensic ground truth for
every recording. Shared source-song identities must not cross training/test
boundaries. Preserve both group_id and component_id when reproducing splits.
The development role denotes eligibility for development protocols, not one
universal training partition. Use the frozen per-protocol schedules.

The input hash identifies the analysis view, not necessarily the original
recording or any HF object. Absolute server paths and source receipts have
been omitted. This catalogue does not grant redistribution rights or imply
that every dataset member is public-domain material.

| Source | Label | Role | Rows |
|---|---:|---|---:|
'''
    for (source,label,role), count in sorted(counts.items()):
        description += f'| {source} | {label} | {role} | {count} |\n'
    description += '''
## Audio and scope

The audio delivery destination is
https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio .
Only per-file publication receipts establish uploaded content. At this
checkpoint the verified audio publication covers 300 MAESTRO originals and
300 derived views, representing the same 300 works, not 600 unique songs.
The remaining source audio needs source-specific rights review and upload.

YuE2 generation produced 500 recordings. Two were too short for Native30;
398 eligible rows are development and 100 are locked. This catalogue contains
498 YuE2 rows, not all generated audio. Historical spectral and native60
cohorts have different eligibility and are not interchangeable with this one.
'''
    (args.output/'README.md').write_text(description)
    products = {}
    for path in args.output.iterdir():
        raw = path.read_bytes()
        products[path.name] = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (args.output/'COMMIT.json').write_text(json.dumps(dict(source_verification=verification,
        rows=len(rows),products=products,audio_complete=False),indent=2))
    print(json.dumps(dict(rows=len(rows),source_role_cells=len(counts))))


if __name__ == '__main__':
    main()
