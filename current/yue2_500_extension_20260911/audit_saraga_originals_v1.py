"""Verify 108 preserved Saraga originals and distinguish the 103 classifier IDs."""
import hashlib
import json
from pathlib import Path

BASE=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
SOURCE=BASE/'saraga_hindustani_physical_v1'
OUT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/saraga_original_audit_v1')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    contract_path=SOURCE/'contract.json';contract=json.loads(contract_path.read_text())
    cohort_path=BASE/'native30_fhsc_cohort_v1/contract.json'
    cohort=json.loads(cohort_path.read_text())
    selected={r['id'] for r in cohort['rows'] if r['source_group']=='human_saraga_hindustani_v1'}
    assert len(selected)==103 and len(contract['items'])==108
    rows=[]
    for i,r in enumerate(contract['items']):
        p=SOURCE/r['raw_path']
        assert p==SOURCE/'raw'/(r['mbid']+'.mp3')
        assert p.stat().st_size==r['archive_member']['bytes']
        digest=sha(p);assert digest==r['archive_member']['sha256']
        identifier='saraga_hindustani_'+r['mbid']
        rows.append(dict(id=identifier,mbid=r['mbid'],bytes=p.stat().st_size,sha256=digest,
            archive_member=r['archive_member']['path'],metadata=r['reconciled_metadata'],
            included_in_native30_classifier_cohort=identifier in selected))
        if (i+1)%20==0:print(f'Verified {i+1}/108 Saraga originals',flush=True)
    assert sum(r['included_in_native30_classifier_cohort'] for r in rows)==103
    assert selected.issubset({r['id'] for r in rows})
    OUT.mkdir(exist_ok=False)
    (OUT/'records.json').write_text(json.dumps(rows,indent=2))
    (OUT/'COMMIT.json').write_text(json.dumps(dict(status='108_originals_hash_verified_103_cohort_ids_bound',
        files=108,classifier_ids=103,other_preserved_ids=5,bytes=sum(r['bytes'] for r in rows),
        records_sha256=sha(OUT/'records.json'),source_contract_sha256=sha(contract_path),
        cohort_contract_sha256=sha(cohort_path),archive_rehashed=False,
        audio_uploaded=False,whole_project_complete=False),indent=2))


if __name__=='__main__':main()
