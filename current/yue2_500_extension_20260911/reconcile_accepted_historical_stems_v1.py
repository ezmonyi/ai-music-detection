"""Reconcile accepted stem releases against the frozen historical stem scope."""
import argparse
import hashlib
import json
from pathlib import Path


def main(report, out):
    assert not out.exists()
    raw = (report/'datasets/historical_stem_union_v1/objects.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'c139675d81142fdc596d7e315c24ce3e32a223c533b8322bf3e0f92252eee975'
    inventory = {r['sha256']: r for r in json.loads(raw)}
    releases = ['hf_open_model_vocals_v1', 'hf_aime_vocals_v1',
        'hf_maestro_stems_v1', 'hf_mtt_stems_v1', 'hf_diffrhythm_stems_v1',
        'hf_ishizaka_stems_v1', 'hf_remaining_suno_stems_v1',
        'hf_remaining_open_models_stems_v1', 'hf_remaining_udio_stems_v1']
    accepted, evidence = {}, []
    for name in releases:
        base = report/'current_results'/name
        receipt = json.loads((base/'INDEPENDENT_ACCEPTANCE.json').read_text())
        raw = (base/'manifest.json').read_bytes()
        assert receipt['final_acceptance'] is True
        assert hashlib.sha256(raw).hexdigest() == receipt['manifest_sha256']
        rows = json.loads(raw)
        assert len(rows) == receipt['verified_audio_objects']
        assert sum(r['bytes'] for r in rows) == receipt['checked_bytes']
        for row in rows:
            key = row['sha256']
            assert key in inventory and inventory[key]['bytes'] == row['bytes']
            accepted.setdefault(key, []).append(dict(release=name, path=row['path'], revision=receipt['revision']))
        evidence.append(dict(release=name, objects=len(rows), revision=receipt['revision'],
                             manifest_sha256=receipt['manifest_sha256']))
    remaining = sorted(inventory.keys()-accepted.keys())
    summary = dict(scope='Frozen 31,378 historical-hash-verified stems only; excludes current-only siblings and other scopes',
        inventory_objects=len(inventory), accepted_unique_objects=len(accepted),
        accepted_bytes=sum(inventory[k]['bytes'] for k in accepted),
        not_matched_to_these_releases_objects=len(remaining),
        not_matched_bytes=sum(inventory[k]['bytes'] for k in remaining),
        warning='Unmatched does not independently establish unpublished status or grant redistribution rights.',
        releases=evidence, whole_project_complete=False)
    out.mkdir()
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'remaining_hashes.json').write_text(json.dumps(remaining)+'\n')
    (out/'accepted_hashes.json').write_text(json.dumps(accepted, separators=(',', ':'))+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.report, a.out)
