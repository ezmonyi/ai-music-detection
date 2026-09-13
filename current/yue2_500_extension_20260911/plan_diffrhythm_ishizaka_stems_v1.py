"""Source-bind pilot/CC0 stems without clearing unrelated MusicNet recordings."""
import argparse
import hashlib
import json
from pathlib import Path
from plan_open_model_vocal_publication_v1 import checked


def main(report, out):
    assert not out.exists()
    objects = checked(report/'datasets/historical_four_stem_sources_v1/objects.json',
        '56ff43a3e6a24ccb72990cac2326ea8062e84d20af50baf5193fd5d99565792d')
    specs = [('diffrhythm', 'hf_diffrhythm_pilot_v1',
        '0f68b18de8d73ea99d406bd70f25df758a5b4fdcd3fa5f5d5a039743603ad1a7',
        'license', 'Apache-2.0 (source publisher audio declaration)', 50, 400),
        ('ishizaka', 'hf_musicnet_ishizaka_v1',
        'c5db937a1d1e308f533f2e8f2378febddf7db0a3d7197bc3f4e54eb3d649f7fe',
        'recording_license', 'CC0-1.0', 39, 312)]
    plans = []
    for name, folder, pin, license_key, license_name, n_ids, n_objects in specs:
        source = checked(report/'current_results'/folder/'manifest.json', pin)
        index = {r['id']: r for r in source}
        assert len(index) == len(source) == n_ids
        assert all(r[license_key] == license_name for r in source)
        selected, ids = [], set()
        for obj in objects:
            members = {m['item_id'] for m in obj['memberships']}
            if not members.intersection(index):
                continue
            assert members <= index.keys()
            ids.update(members)
            selected.append(dict(**obj, original_records=[index[i] for i in sorted(members)],
                license=license_name, transformation='Historical Demucs stem; no further byte modification',
                role='pilot_not_promoted' if name == 'diffrhythm' else 'source_specific_cc0_subset'))
        assert len(selected) == n_objects and len(ids) == n_ids
        plans.append((name, selected, pin))
    out.mkdir()
    summaries = []
    for name, rows, pin in plans:
        raw = (json.dumps(rows, separators=(',', ':'))+'\n').encode()
        (out/(name+'.json')).write_bytes(raw)
        summaries.append(dict(name=name, objects=len(rows), bytes=sum(r['bytes'] for r in rows),
            original_manifest_sha256=pin, objects_sha256=hashlib.sha256(raw).hexdigest()))
    summary = dict(sources=summaries, status='source_bound_plans_not_uploaded', whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); main(a.report, a.out)
