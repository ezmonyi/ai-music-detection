"""Join audited source bytes, acquisition provenance and historical roles."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(audit, acquisition, historical, out):
    commit=json.loads((audit/'COMMIT.json').read_text())
    assert sha(audit/'records.jsonl')==commit['records_sha256']
    records=[json.loads(x) for x in (audit/'records.jsonl').read_text().splitlines()]
    arows=list(csv.DictReader(acquisition.open()))
    origins={r['item_id']:r for r in arows}
    history={r['id']:r for r in csv.DictReader(historical.open())}
    assert sha(historical)==commit['input_sha256']
    assert len(records)==len(arows)==len(origins)==2641
    assert {r['id'] for r in records}==set(origins)
    output=[]
    for r in records:
        a,h=origins[r['id']],history[r['id']]
        assert r['status']=='hash_verified'
        assert r['actual_sha256']==r['expected_sha256']==h['raw_sha256']
        assert {'human':'0','ai':'1'}[a['label']]==h['label']
        assert a['source_id']==r['source_group']
        locator=a['source_locator']
        # Public source locators must not embed credentials or signed queries.
        parsed=urlsplit(locator)
        assert not parsed.query and not parsed.password
        assert not any(word in locator.lower() for word in ('x-amz-','access_token=','signature='))
        output.append(dict(id=r['id'],label=h['label'],source_group=r['source_group'],
            source_revision=a['source_revision'],source_locator=locator,
            container_member=a['container_member'],source_sha256=r['actual_sha256'],
            source_bytes=r['bytes'],native_sample_rate_hz=h['native_sample_rate_hz'],
            native_duration_s=h['native_duration_s'],historical_role=h['role'],
            acquisition_role=a['role'],group_id=h['group_id'],
            artist_or_creator_recorded=a['artist_or_creator'],title_recorded=a['title'],
            license_recorded=a['license'],evaluation_allowed_recorded=a['evaluation_allowed'],
            provenance_grade=a['provenance_grade'],source_notes=a['notes']))
    out.mkdir(exist_ok=False)
    with (out/'retrieval.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(output)
    counts=Counter(r['source_group'] for r in output)
    (out/'README.md').write_text('''# Historical diversity source retrieval catalogue

2,641 exact historical identities joined across the frozen acquisition manifest,
10-second experiment metadata and fresh source-file hash audit. All recorded
source bytes passed verification. This is a retrieval/provenance index, not a
claim that the audio has been republished or that every upstream URL still works.

Use source_locator plus source_revision and container_member to identify the
recorded acquisition object. A locator may denote an archive or parquet member,
not a direct audio URL. source_sha256 describes the preserved source file; it is
not the checksum of its containing archive. Refer to source_notes and acquisition
code for decoding/mixing steps. Original and derived assets are not interchangeable.

Historical and acquisition roles are both preserved because later experimental
reuse differs from initial selection. Keep group_id intact. Stress, provisional,
pilot and locked data are not silently relabelled as primary training data.

License and evaluation fields are historical statements, not fresh legal
clearance. Artist/title fields are copied faithfully and may contain placeholders
(notably MoisesDB); non-empty fields do not imply attribution is complete.
No blanket license for these recordings is granted by this metadata release.
Use the source-integrity report and current publication review for open issues.

Private server paths, credentials and signed download URLs are excluded. Public
HF locators may retain revision syntax. These 2,641 IDs already occur in the
historical and cross-experiment catalogues; do not add them to corpus counts.
''')
    (out/'COMMIT.json').write_text(json.dumps(dict(
        status='audited_source_acquisition_and_historical_identity_join',rows=2641,
        source_counts=dict(counts),verified_source_bytes=sum(r['source_bytes'] for r in output),
        inputs=dict(audit_commit_sha256=sha(audit/'COMMIT.json'),
                    acquisition_sha256=sha(acquisition),historical_sha256=sha(historical)),
        products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()},
        audio_publication_established_by_this_catalogue=False,whole_project_complete=False),indent=2))
    print(json.dumps(dict(rows=len(output),source_counts=dict(counts))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('audit','acquisition','historical','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.audit,a.acquisition,a.historical,a.out)
