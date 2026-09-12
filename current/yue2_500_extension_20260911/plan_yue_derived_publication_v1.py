"""Bind all inventoried YuE audio views to original IDs and byte-deduplicate them."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

IPIN = 'f1ce7251de8a3b08f6fe1c0cb629c494cf3cbd354495dafcb135438e709eed8c'
OPIN = '4981ad60bd2a658d33d07e79abfb1e4f17d4ff7886e0efabd3d741abc5ab3776'


def plan(inventory, originals, out):
    raw = inventory.read_bytes(); original_raw = originals.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == IPIN
    assert hashlib.sha256(original_raw).hexdigest() == OPIN
    sources = {r['id']: r for r in json.loads(original_raw)}
    assert len(sources) == 500
    memberships = []; excluded = []
    for line in raw.decode().splitlines():
        row = json.loads(line); path = Path(row['path'])
        if path.suffix.lower() not in {'.wav', '.flac', '.mp3'}:
            continue
        candidates = set(path.parts) | {path.stem}
        matched = candidates & sources.keys()
        if not matched:
            assert not any('ai_yue2_' in p for p in path.parts), row['path']
            excluded.append(dict(row, reason='Not a bound YuE generated identity; requires separate source review'))
            continue
        assert len(matched) == 1
        identity = matched.pop()
        memberships.append(dict(row, id=identity, group_id=sources[identity]['group_id'],
            role=sources[identity]['role'], object_key=row['sha256']+':'+str(row['bytes'])))
    objects = {}
    for row in memberships:
        obj = objects.setdefault(row['object_key'], dict(sha256=row['sha256'], bytes=row['bytes'],
            source_paths=[], recording_ids=[], existing_legacy_demucs_path=None))
        obj['source_paths'].append(row['path'])
        if row['id'] not in obj['recording_ids']: obj['recording_ids'].append(row['id'])
        if row['path'].startswith('demucs_v1/htdemucs/'):
            obj['existing_legacy_demucs_path'] = 'audio/yue2/legacy_demucs_v1/'+row['id']+'/'+Path(row['path']).name
    for obj in objects.values():
        obj['proposed_path'] = obj['existing_legacy_demucs_path'] or (
            'audio/yue2/derived_objects_v1/'+obj['sha256']+Path(obj['source_paths'][0]).suffix)
        obj['publication_verified'] = False
    out.mkdir(exist_ok=False)
    products = {}
    for name, value in [('memberships.json', memberships), ('objects.json', list(objects.values())),
                        ('excluded_non_yue_audio.json', excluded)]:
        data = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()
        (out/name).write_bytes(data)
        products[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    summary = dict(status='bound_derived_publication_plan_not_upload_acceptance',
        source_inventory_sha256=IPIN, original_manifest_sha256=OPIN,
        audio_memberships=len(memberships), unique_byte_objects=len(objects),
        logical_bytes=sum(r['bytes'] for r in memberships),
        unique_bytes=sum(o['bytes'] for o in objects.values()),
        separately_queued_legacy_objects=sum(o['existing_legacy_demucs_path'] is not None for o in objects.values()),
        remaining_objects=sum(o['existing_legacy_demucs_path'] is None for o in objects.values()),
        remaining_bytes=sum(o['bytes'] for o in objects.values() if o['existing_legacy_demucs_path'] is None),
        excluded_non_yue_audio_memberships=len(excluded),
        directory_memberships=dict(Counter(Path(r['path']).parts[0] for r in memberships)),
        products=products, whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='products'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--originals', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); plan(args.inventory, args.originals, args.out)
