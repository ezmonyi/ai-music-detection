"""Publish unchanged Saraga originals with full attribution and both source notices."""
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save, verify

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT = ROOT/'hf_saraga_originals_v1'
SOURCE = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/saraga_hindustani_physical_v1/raw')
CARD = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/external_validation/saraga_catalog_v1/LICENSE.md')
PREFIX = 'audio/saraga_hindustani/originals_v1/'
PIN = '5982e37c320db8287cb78f25bad22b49c33a7441f4c7a57fa2f09cd449ea8925'
NOTICE = '''# Saraga 1.5 Hindustani: unchanged preserved originals

108 unchanged original MP3 recordings, of which 103 belong to the Native30
classifier cohort. Five others are preserved separately; these counts are not
new test-set additions. No resampling, cropping or separation was performed
for this publication. The manifest gives individual performer credits, titles,
concert/release metadata, original ZIP members and exact file hashes.
Two recordings have no instrument-level performer credits in the source;
their nonempty album-artist attribution is retained, without invented credits.

Dataset creators: B. Bozkurt, A. Srinivasamurthy, S. Gulati and X. Serra;
Music Technology Group, Universitat Pompeu Fabra. Individual performers are
credited in manifest.json. Source: https://zenodo.org/records/4301737
DOI: https://doi.org/10.5281/zenodo.4301737

The companion repository explicitly licenses audio and annotations CC BY-NC 4.0:
https://creativecommons.org/licenses/by-nc/4.0/
https://github.com/MTG/saraga/blob/bd103295689ad11a5ed19038b7469e91b5c3c124/LICENSE.md
Its full notice is retained in SOURCE_LICENSE.md. Historical Zenodo metadata
records CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
Both source statements are retained without claiming to reconcile or replace
them. This noncommercial thesis archive redistributes unchanged recordings;
it does not offer a new license. Preserve attribution and notices, respect
noncommercial restrictions and applicable ShareAlike terms for adaptations.
No endorsement or independent clearance of all other rights is implied.

No source code, new adaptations or manual annotations are included here.
The source request to tell the maintainers about research use has not yet
been fulfilled; no email was sent on the researcher's behalf by this publisher.
The manifest is a planned roster, not evidence that every batch has arrived.
'''


def build_records():
    audit = ROOT/'saraga_original_audit_v1/records.json'
    assert digest(audit) == PIN
    source = json.loads(audit.read_text())
    assert len(source) == len({r['id'] for r in source}) == 108
    records = []
    for row in source:
        metadata = row['metadata']
        assert metadata['mbid'] == row['mbid']
        assert metadata['title'] and metadata['album_artists']
        records.append(dict(id=row['id'], path=PREFIX+row['mbid']+'.mp3',
            sha256=row['sha256'], bytes=row['bytes'], recording_mbid=row['mbid'],
            included_in_native30_classifier_cohort=row['included_in_native30_classifier_cohort'],
            archive_url='https://zenodo.org/records/4301737/files/saraga1.5_hindustani.zip',
            archive_member=row['archive_member'], title=metadata['title'],
            album_artists=metadata['album_artists'], performer_credits=metadata['performer_credits'],
            detailed_performer_credits_available=bool(metadata['performer_credits']),
            release_or_concert_groups=metadata['release_or_concert_groups'],
            repository_audio_license='CC-BY-NC-4.0', historical_zenodo_license='CC-BY-NC-SA-4.0',
            modification='None; unchanged original MP3 bytes'))
    assert sum(r['included_in_native30_classifier_cohort'] for r in records) == 103
    assert sum(not r['detailed_performer_credits_available'] for r in records) == 2
    assert sum(r['bytes'] for r in records) == 3956076186
    return records


def worker(token):
    try:
        publish(token)
    except Exception as exc:
        # Prevent transport from logging third-party exception URLs/headers.
        raise RuntimeError(type(exc).__name__) from None


def publish(token):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    records = build_records()
    card = CARD.read_bytes()
    assert b'CC BY-NC 4.0' in card and b'audio recordings' in card
    save(OUT/'manifest.json', records)
    for index, offset in enumerate(range(0, 108, 10)):
        batch = records[offset:offset+10]
        receipt = OUT/f'batch_{index:03d}.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['files'] == [r['path'] for r in batch]
            verify(api, batch, prior['revision'])
            continue
        operations = []
        for row in batch:
            path = SOURCE/(row['recording_mbid']+'.mp3')
            assert path.resolve().is_relative_to(SOURCE.resolve())
            assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
            operations.append(CommitOperationAdd(path_in_repo=row['path'], path_or_fileobj=str(path)))
        if index == 0:
            operations.extend([
                CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(OUT/'manifest.json')),
                CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode()),
                CommitOperationAdd(path_in_repo=PREFIX+'SOURCE_LICENSE.md', path_or_fileobj=card)])
        result = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=operations,
            commit_message=f'Archive unchanged Saraga originals {offset+1}-{offset+len(batch)} of 108')
        verify(api, batch, result.oid)
        save(receipt, dict(revision=result.oid, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {offset+len(batch)}/108', flush=True)
    save(OUT/'COMMIT.json', dict(status='108_saraga_originals_uploaded_hash_verified', audio_files=108,
        classifier_ids=103, manifest_sha256=digest(OUT/'manifest.json'), source_audit_sha256=PIN,
        whole_project_complete=False))


if __name__ == '__main__':
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
