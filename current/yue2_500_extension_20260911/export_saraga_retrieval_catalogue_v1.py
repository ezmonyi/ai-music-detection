"""Export audited Saraga source retrieval and attribution without republishing audio."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

PIN='5982e37c320db8287cb78f25bad22b49c33a7441f4c7a57fa2f09cd449ea8925'
REV='bd103295689ad11a5ed19038b7469e91b5c3c124'


def export(audit,out):
    raw=(audit/'records.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PIN
    commit=json.loads((audit/'COMMIT.json').read_text())
    assert commit['records_sha256']==PIN
    assert commit['status']=='108_originals_hash_verified_103_cohort_ids_bound'
    rows=json.loads(raw)
    assert len(rows)==len({r['id'] for r in rows})==108
    assert sum(r['included_in_native30_classifier_cohort'] for r in rows)==103
    assert sum(r['bytes'] for r in rows)==3956076186
    records=[]
    for row in rows:
        metadata=row['metadata']
        assert metadata['mbid']==row['mbid']
        assert metadata['archive_audio_path']==row['archive_member']
        records.append(dict(id=row['id'],source_group='human_saraga_hindustani_v1',
            included_in_native30_classifier_cohort=row['included_in_native30_classifier_cohort'],
            original_sha256=row['sha256'],original_bytes=row['bytes'],
            archive_url='https://zenodo.org/records/4301737/files/saraga1.5_hindustani.zip',
            archive_member=row['archive_member'],recording_mbid=row['mbid'],
            title=metadata['title'],album_artists=metadata['album_artists'],
            performer_credits=metadata['performer_credits'],
            release_or_concert_groups=metadata['release_or_concert_groups'],
            metadata_repository_revision=REV,metadata_path=metadata['catalog_metadata_path'],
            repository_audio_license='CC-BY-NC-4.0',
            source_licensing_note='Historical Zenodo metadata recorded CC-BY-NC-SA-4.0; preserve both statements. No new audio redistribution performed.',
            project_audio_publication_verified=False))
    out.mkdir(exist_ok=False)
    (out/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    fields=['id','source_group','included_in_native30_classifier_cohort','original_sha256',
            'original_bytes','archive_url','archive_member','recording_mbid','title']
    with (out/'retrieval.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in records)
    (out/'README.md').write_text('''# Saraga preserved-original retrieval catalogue

108 preserved Hindustani MP3 originals, of which exactly 103 belong to the
Native30 classifier cohort. Five other recordings are retained separately, not
counted as classifier test songs. These IDs overlap the cross-experiment index.

The fresh server audit matched every original SHA-256 and byte size to its
physical-source contract. Archive member paths, recording IDs, titles, album
artists, performer credits and concert/release groups are retained. The source
ZIP itself was not rehashed in this audit; retrieval URLs point to the original
versioned source, not to a verified project HF audio upload.

Official source: https://zenodo.org/records/4301737 (Saraga version 1.5).
Dataset creators: B. Bozkurt, A. Srinivasamurthy, S. Gulati and X. Serra;
Music Technology Group, Universitat Pompeu Fabra. See records.json for performers.

The pinned companion repository LICENSE.md explicitly assigns audio and
annotations CC BY-NC 4.0:
https://github.com/MTG/saraga/blob/bd103295689ad11a5ed19038b7469e91b5c3c124/LICENSE.md
Historical Zenodo metadata recorded CC BY-NC-SA 4.0. Both statements are retained;
this catalogue does not resolve the discrepancy or relicense the recordings.
No audio is included or claimed uploaded. Annotation presence does not establish
annotation accuracy, and this export does not modify experimental groupings.
''')
    products={p.name:dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir()}
    (out/'COMMIT.json').write_text(json.dumps(dict(status='audited_original_retrieval_catalogue',
        records=108,classifier_ids=103,source_records_sha256=PIN,products=products,
        audio_uploaded=False,whole_project_complete=False),indent=2))
    print('Exported 108 originals; 103 classifier IDs; no audio upload')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();export(args.audit,args.out)
