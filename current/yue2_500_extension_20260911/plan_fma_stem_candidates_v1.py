"""Select historical FMA stems by exact IDs with previously screened attribution."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def checked(path, pin):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin
    return json.loads(raw)


def main(report, out):
    assert not out.exists()
    released = checked(report/'current_results/hf_fma_test_views_v1/manifest.json',
        'ab3aad594c129db50b5ad88f8367e00720175cb0fd78025ba2910375f242f8cf')
    source = {}
    for row in released:
        for original in row['original_records']:
            key = original['id']
            if key in source:
                assert source[key] == original
            source[key] = original
    inventory = {}
    for folder, pin in [('historical_four_stem_sources_v1', '56ff43a3e6a24ccb72990cac2326ea8062e84d20af50baf5193fd5d99565792d'),
                        ('historical_spectral_stem_sources_v1', '49ff713dc524189723b4adc6d9874d252d6437dbf6e97d25a0acff4ad760db01')]:
        for row in checked(report/'datasets'/folder/'objects.json', pin):
            if row['source_group'] != 'FMA':
                continue
            if row['sha256'] in inventory:
                prior = inventory[row['sha256']]
                assert prior['bytes'] == row['bytes']
                for member in row['memberships']:
                    if member not in prior['memberships']:
                        prior['memberships'].append(member)
            else:
                inventory[row['sha256']] = row
    selected, held = [], []
    for row in inventory.values():
        ids = {m['item_id'] for m in row['memberships']}
        if not ids <= source.keys():
            held.append(row['sha256'])
            continue
        originals = [source[k] for k in sorted(ids)]
        urls = {r['license_url'] for r in originals}
        assert len(urls) == 1
        url = next(iter(urls))
        assert re.fullmatch(r'https://creativecommons.org/licenses/(by|by-sa|by-nc|by-nc-sa)/(1\.0|2\.0|2\.5|3\.0|4\.0)/(us/)?', url) or url == 'https://creativecommons.org/publicdomain/zero/1.0/'
        assert all(r['artist'] and r['title'] for r in originals)
        selected.append(dict(**row, original_records=originals, license_url=url,
            transformation='Historical Demucs stem separation; existing bytes unchanged for publication'))
    assert len(inventory) == 1300
    out.mkdir()
    raw = (json.dumps(sorted(selected, key=lambda r:r['sha256']), separators=(',', ':'))+'\n').encode()
    (out/'objects.json').write_bytes(raw)
    (out/'held_hashes.json').write_text(json.dumps(sorted(held))+'\n')
    result = dict(candidate_objects=len(selected), candidate_bytes=sum(r['bytes'] for r in selected),
        held_objects=len(held), objects_sha256=hashlib.sha256(raw).hexdigest(),
        status='source_bound_candidates_not_uploaded; publication requires retained per-track notices',
        whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.report, a.out)
