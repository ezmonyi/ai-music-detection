"""Publish a path-free projection of frozen historical input inventories."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

BASE=Path('/Users/yi/Documents/report/music_ai_detection_20260912/datasets/historical_manifest_sources_v1')
OUT=Path('/Users/yi/Documents/code/music/deliverables/github_staging_20260912/datasets/historical_input_inventory_v1')
PINS={'metadata_10s.csv':'a03aec930130fcb942c40b35ba5fb49109ceee588894124010448f041d9d1ef4',
      'metadata_30s.csv':'eb313f97e2d69d743423efc170eecb24f1f00ee77ad789461772daecd058c242'}
FIELDS=['id','label','source_group','role','original_role','group_id','condition_id',
        'native_sample_rate_hz','native_duration_s','raw_sha256','provenance_status',
        'acquisition','duration_sec','duration_view','crop_start_s','audio_offset_s',
        'requires_crop','eligible_common8','eligible_fullband','generator_family','evaluation_allowed']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    inventories={}
    for name,pin in PINS.items():
        assert sha(BASE/name)==pin
        with (BASE/name).open() as stream:inventories[name]=list(csv.DictReader(stream))
    ten={r['id']:r for r in inventories['metadata_10s.csv']}
    assert len(ten)==len(inventories['metadata_10s.csv'])==10141
    assert len(inventories['metadata_30s.csv'])==4497
    for row in inventories['metadata_30s.csv']:
        assert all(row[k]==ten[row['id']][k] for k in ['label','source_group','group_id'])
    OUT.mkdir(exist_ok=False)
    counts=[]
    for name,rows in inventories.items():
        with (OUT/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
            writer.writerows({k:r[k] for k in FIELDS} for r in rows)
        counts.extend(dict(view=name,source_group=source,role=role,rows=count)
                      for (source,role),count in sorted(Counter((r['source_group'],r['role']) for r in rows).items()))
    with (OUT/'counts_by_source_role.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    notice='''# Historical input inventory (not a final test cohort)

This is a field-limited projection of the September 5 frozen 10-second and
30-second input manifests. There are 10,141 distinct recorded IDs across 26
source groups; all 4,497 30-second IDs are also in the 10-second list, with
matching label, source and group. Therefore 14,638 view rows are NOT 14,638
independent songs. Distinct IDs are not a content-level deduplication claim.

Roles such as provisional, pilot, stress and locked remain unchanged.
Presence in an input manifest or historical availability does not prove
successful measurement, inclusion in a fitted classifier, or present-day audio
availability. Actual experiment/split membership must be joined to result
receipts. Blank evaluation permissions remain blank rather than being inferred.

`raw_sha256` is the hash recorded in the historical manifest; this publication
does not rehash the underlying audio and does not assert that an input was a
full original recording rather than an earlier prepared view. Native rates,
durations and crop fields are retained as recorded, not newly measured.

Server paths and creator/title/free-text fields are excluded from this public
projection. Complete original manifests are retained locally under
Documents/report/music_ai_detection_20260912/datasets/historical_manifest_sources_v1.
The COMMIT records source hashes and hashes of these projected products.
No audio is uploaded by this export, and no redistribution license is inferred.
This historical population is separate from the current Native30 classifier
package; unioning them requires explicit identity and actual-usage reconciliation.
'''
    with (OUT/'README.md').open('x') as stream:stream.write(notice)
    products={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir()}
    with (OUT/'COMMIT.json').open('x') as stream:
        json.dump(dict(status='verified_projection_not_final_test_membership',source_sha256=PINS,
            products=products,distinct_ids=10141,view_rows=14638,audio_rehashed=False,audio_uploaded=False),stream,indent=2)
    print('Published local projection: 10141 IDs, 14638 views, no audio or test-membership claim')


if __name__=='__main__':main()
