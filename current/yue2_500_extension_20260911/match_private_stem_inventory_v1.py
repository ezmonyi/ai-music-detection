"""Local-only reconciliation of private HF metadata with current stem hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def checked(path, pin, lines=False):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin
    return [json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw)


def main(report, out):
    assert not out.exists()
    private = checked(report/'current_results/private_audio_snapshot_20260913_v1/files.json',
        '27425396423a7fbc537c8aa5b581097cd2cbe3d766f78bfc400e27e09e011be5')
    scope = checked(report/'datasets/stem_public_snapshot_match_v3/matches.json',
        '7b0ee6dc7a62abeab6acd94809f7e9f314a4f890ea86d3e44a9d8c25eaa83fea')
    current = checked(report/'current_results/legacy_nonvocal_current_file_audit_v1/records.jsonl',
        '34cfb5894eb3ce71828eea3b6d9c63ed88772f8606ca9fbecfb216298c50d9f2', True)
    source = {r['sha256']:r for r in current if r['status']=='current_bytes_hashed'}
    lookup = {}
    for row in private:
        if row['sha256']:
            lookup.setdefault(row['sha256'], []).append(row)
    pending = set(scope['current_only_inventory']['unmatched'])
    matched = []
    for key in sorted(pending & lookup.keys()):
        assert all(r['bytes'] == source[key]['bytes'] for r in lookup[key])
        matched.append(dict(sha256=key, bytes=source[key]['bytes'],
            source_record=source[key], private_paths=[r['path'] for r in lookup[key]]))
    remaining = sorted(pending-lookup.keys())
    assert len(matched)==3396
    out.mkdir(mode=0o700)
    raw=(json.dumps(matched,separators=(',',':'))+'\n').encode()
    (out/'matched_private.json').write_bytes(raw)
    (out/'unmatched_current_hashes.json').write_text(json.dumps(remaining)+'\n')
    summary=dict(private_revision='5ee4b8e31643421a3fadf6bc8388c1d7f77612c3',
        matched_hash_and_size_objects=len(matched), matched_bytes=sum(r['bytes'] for r in matched),
        current_only_unmatched_public_and_private=len(remaining),
        unmatched_bytes=sum(source[k]['bytes'] for k in remaining),
        historical_unmatched_hashes_found_private=len(set(scope['historical_hash_verified']['unmatched']) & lookup.keys()),
        matches_sha256=hashlib.sha256(raw).hexdigest(),
        evidence='Pinned remote LFS hash and size metadata; not a full download or proof of historical experimental input identity',
        public_release_authorized=False, whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.report,a.out)
