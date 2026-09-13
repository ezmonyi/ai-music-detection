"""Bind additional generated stems, excluding accepted and already queued vocals."""
import argparse
import hashlib
import json
from pathlib import Path
from plan_open_model_vocal_publication_v1 import checked


def main(report, out):
    assert not out.exists()
    rows = checked(report/'datasets/historical_four_stem_sources_v1/objects.json', '56ff43a3e6a24ccb72990cac2326ea8062e84d20af50baf5193fd5d99565792d')
    rows += checked(report/'datasets/historical_spectral_stem_sources_v1/objects.json', '49ff713dc524189723b4adc6d9874d252d6437dbf6e97d25a0acff4ad760db01')
    already = checked(report/'current_results/hf_open_model_vocals_v1/manifest.json', 'f6e9f37c0b36ba3c3cd495a8728370f16d7c7d0f4982335717b99bdb84645a36')
    queued = checked(report/'datasets/aime_vocal_publication_plan_v1/objects.json', '21126d3da7eeabf6d251e0b312bb5e5797a48a5ac305ee410c625e60ed7b948a')
    excluded = {r['sha256'] for r in already+queued}
    specs = [('suno', {'Suno'}, 'suno_original_publication_plan_v1', '5ddf39584dea9a8e22e93b0ecf65ddff31b0ac9c554bae9ad35c30958a8ff23b', 1300),
             ('open_models', {'ACE-Step','HeartMuLa'}, 'hf_open_model_audio_v1', '7be5aaa118e77181aa489ef3462b254de6f37679e69cb99e690d069387e4f5f5', 600),
             ('udio', {'Udio'}, 'aime_originals_publication_v1', 'b8c0ac0e9cc21b431c1813bcc3594463786123510be422048758947ab1fe9f70', 2000)]
    plans = []
    for name, groups, folder, pin, count in specs:
        originals = checked(report/'current_results'/folder/'manifest.json', pin)
        index = {}
        for r in originals:
            index.setdefault(r['id'], []).append(r)
        selected = {}
        for row in rows:
            if row['source_group'] not in groups or row['sha256'] in excluded:
                continue
            member_ids = {m['item_id'] for m in row['memberships']}
            assert member_ids <= index.keys()
            item = selected.setdefault(row['sha256'], dict(**row,
                original_records=[r for i in sorted(member_ids) for r in index[i]],
                transformation='Historical Demucs separated stem; unchanged existing bytes'))
            assert item['bytes'] == row['bytes']
            for member in row['memberships']:
                if member not in item['memberships']:
                    item['memberships'].append(member)
        assert len(selected) == count
        plans.append((name, sorted(selected.values(), key=lambda r:r['sha256']), pin))
    out.mkdir()
    summaries = []
    for name, records, pin in plans:
        raw = (json.dumps(records, separators=(',', ':'))+'\n').encode()
        (out/(name+'.json')).write_bytes(raw)
        summaries.append(dict(name=name, objects=len(records), bytes=sum(r['bytes'] for r in records),
            objects_sha256=hashlib.sha256(raw).hexdigest(), original_manifest_sha256=pin))
    summary = dict(sources=summaries, status='source_bound_plans_not_uploaded',
        exclusion='Accepted open-model vocals and queued AIME vocal plan; queued is not accepted', whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(); main(a.report,a.out)
