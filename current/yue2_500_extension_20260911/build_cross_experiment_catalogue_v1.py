"""Join published input catalogues without conflating experimental roles/views."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'deliverables/github_staging_20260912/datasets'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    specifications = [
        ('historical_input_inventory_v1', 'metadata_10s.csv', 'historical_10s'),
        ('historical_input_inventory_v1', 'metadata_30s.csv', 'historical_30s'),
        ('native30_expanded_v1', 'cohort.csv', 'expanded_native30'),
    ]
    records, memberships, bindings = {}, [], {}
    for folder, filename, cohort in specifications:
        path = root / folder / filename
        commit = json.loads((root / folder / 'COMMIT.json').read_text())
        binding = commit['products'][filename]
        assert digest(path) == binding['sha256']
        assert path.stat().st_size == binding['bytes']
        bindings[f'{folder}/{filename}'] = digest(path)
        rows = list(csv.DictReader(path.open()))
        assert len({r['id'] for r in rows}) == len(rows)
        for row in rows:
            identity = (row['label'], row['source_group'])
            assert row['id'] not in records or records[row['id']] == identity
            records[row['id']] = identity
            memberships.append(dict(id=row['id'], cohort=cohort, role=row['role'],
                                    group_id=row['group_id'],
                                    component_id=row.get('component_id', ''),
                                    source_index=f'{folder}/{filename}'))
    out = root / 'cross_experiment_catalogue_v1'
    out.mkdir(exist_ok=False)
    def write(name, fields, rows):
        with (out / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write('identities.csv', ['id', 'label', 'source_group'],
          [dict(id=k, label=v[0], source_group=v[1]) for k, v in sorted(records.items())])
    write('memberships.csv', ['id', 'cohort', 'role', 'group_id', 'component_id', 'source_index'], memberships)
    counts = Counter(records.values())
    write('source_counts.csv', ['label', 'source_group', 'distinct_ids'],
          [dict(label=k[0], source_group=k[1], distinct_ids=v) for k, v in sorted(counts.items())])
    (out / 'README.md').write_text('''# Cross-experiment catalogue v1

Exact-ID union of three hash-verified published catalogues: historical 10s,
historical 30s, and expanded Native30. Roles, groups and components remain
cohort-specific in memberships.csv; do not infer one universal split.
Historical rows establish input/measurement scope, not classifier membership.
Consult the separate historical measurement-status catalogue for failures and
per-family availability. No new training, admission or threshold selection is
performed by this join.

There are 11,242 distinct IDs across 29 sources. Expanded Native30 contributes
1,101 IDs absent from historical inputs: Mureka 500, YuE2 498 and Saraga 103.
The remaining 3,227 Native30 IDs overlap historical inputs. Historical 30s is
a subset of historical 10s. IDs are not content-deduplicated songs: alternate
IDs or shared prompts can remain related. Preserve original group contracts.

This is NOT the complete audio release: external phenomenon controls, two
short YuE2 generations used only in the legacy analysis, and any earlier-only
samples are not covered. Native60 view membership is not represented here.
Source CSVs retain their own waveform hashes; unlike-duration hashes must not
be interpreted as identical original audio. This index grants no audio rights
and does not establish public availability. Publication remains incomplete.
''')
    (out / 'COMMIT.json').write_text(json.dumps(dict(
        distinct_ids=len(records), membership_rows=len(memberships),
        source_count=len(counts), source_bindings=bindings,
        products={p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in out.iterdir()},
        whole_corpus_complete=False), indent=2))
    print(json.dumps(dict(distinct_ids=len(records), membership_rows=len(memberships), sources=len(counts))))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets-root', type=Path, default=ROOT)
    build(parser.parse_args().datasets_root)
