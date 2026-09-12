"""Partition FMA processed objects by exact recorded source license, not a blanket grant."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def is_derivative_candidate_license(url):
    return bool(re.fullmatch(r'https://creativecommons.org/licenses/(by|by-sa|by-nc|by-nc-sa)/(1\.0|2\.0|2\.5|3\.0|4\.0)/(us/)?', url)
                or url == 'https://creativecommons.org/publicdomain/zero/1.0/')


def load(p, pin):
    raw = p.read_bytes(); assert hashlib.sha256(raw).hexdigest() == pin
    return json.loads(raw)


def main(objects, sources, out):
    assert not out.exists()
    objs = load(objects, 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92')
    source = load(sources, '2aae3374be02122ce65874a64bb5bb8996403965a23613c22d01c02f15350ccc')
    source = {r['id']:r for r in source['rows']}
    rows = []
    for obj in objs:
        if not any(m['source_group'] == 'FMA' for m in obj['memberships']): continue
        assert all(m['source_group'] == 'FMA' for m in obj['memberships'])
        ids = sorted({m['id'] for m in obj['memberships']})
        assert all(i in source for i in ids)
        originals = [source[i] for i in ids]
        licenses = sorted({r.get('license_url', '') for r in originals})
        if all(r['status'] == 'explicit_version_original_candidate' for r in originals) and len(licenses) == 1:
            url = licenses[0]
            if is_derivative_candidate_license(url):
                status = 'derivative_license_candidate_requires_notice'
            elif '-nd/' in url:
                status = 'hold_no_derivatives'
            else:
                status = 'hold_other_terms'
        else:
            status = 'hold_unresolved_or_mixed_license'
        rows.append(dict(sha256=obj['sha256'], bytes=obj['bytes'],
            source_paths=obj['source_paths'], memberships=obj['memberships'],
            source_records=originals, status=status, publication_verified=False))
    assert len(rows) == 900
    out.mkdir()
    raw = (json.dumps(rows, indent=2)+'\n').encode()
    (out/'records.json').write_bytes(raw)
    summary = dict(objects=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
        candidate_bytes=sum(r['bytes'] for r in rows if r['status']=='derivative_license_candidate_requires_notice'),
        records_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Recorded per-track license screening only; not completed derivative publication',
        whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['objects','sources','out']: p.add_argument('--'+n, type=Path, required=True)
    a=p.parse_args(); main(a.objects,a.sources,a.out)
