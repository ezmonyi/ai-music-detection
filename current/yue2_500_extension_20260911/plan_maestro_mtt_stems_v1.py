"""Join selected MAESTRO/MTT stems to preserved source-specific release terms."""
import argparse
import hashlib
import json
from pathlib import Path
from plan_open_model_vocal_publication_v1 import checked


def main(objects, maestro, mtt, out):
    assert not out.exists()
    objects = checked(objects, '56ff43a3e6a24ccb72990cac2326ea8062e84d20af50baf5193fd5d99565792d')
    specs = [('human_maestro_v3', maestro, '4fda23164b186c023481424018d3be1018de931bf39b8bc33f4fa2fd79546602',
              'CC-BY-NC-SA-4.0', 300, 2400),
             ('human_magnatagatune', mtt, '90efec2510f56a2e224fb091670a41f7a7b0203f7767d51de2205b79a6632a5d',
              'CC-BY-NC-SA-1.0', 500, 2000)]
    plans = []
    for group, path, pin, license_name, n_ids, n_objects in specs:
        source = checked(path, pin)
        index = {r['id']: r for r in source}
        assert len(index) == len(source) == n_ids
        assert all(r['license'] == license_name for r in source)
        if group == 'human_magnatagatune':
            assert all(r['artist'] and r['title'] for r in source)
        else:
            assert all(r['attribution'] and r['source_url'] for r in source)
        rows, ids = [], set()
        for obj in objects:
            if obj['source_group'] != group:
                continue
            member_ids = {m['item_id'] for m in obj['memberships']}
            assert member_ids <= index.keys()
            ids.update(member_ids)
            rows.append(dict(**obj, original_records=[index[i] for i in sorted(member_ids)],
                license=license_name,
                transformation='Historical Demucs separated stem; no further byte modification for publication'))
        assert len(rows) == n_objects and len(ids) == n_ids
        plans.append((group, rows, pin, license_name))
    out.mkdir()
    summaries = []
    for group, rows, pin, license_name in plans:
        raw = (json.dumps(rows, separators=(',', ':')) + '\n').encode()
        (out / (group + '.json')).write_bytes(raw)
        summaries.append(dict(source_group=group, objects=len(rows), bytes=sum(r['bytes'] for r in rows),
            original_manifest_sha256=pin, license=license_name, objects_sha256=hashlib.sha256(raw).hexdigest()))
    summary = dict(sources=summaries, status='source_bound_plans_not_uploaded', whole_project_complete=False)
    (out / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('objects', 'maestro', 'mtt', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    main(a.objects, a.maestro, a.mtt, a.out)
