"""Prepare byte-verified VocalSet source subset; performs no network writes."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/external_validation/vocalset_breath_original')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/vocalset_publication_plan_v1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'acquisition_summary_v2.json'
    assert digest(source) == '99a981cf9bf14f4b890bc7dbcc1a81f735f751d4f8aaaa5d3cca3e6ef3fa72d9'
    data = json.loads(source.read_text())
    assert data['selected'] == data['downloaded_verified'] == 244
    rows = []
    for r in sorted(data['manifest'], key=lambda x:x['filename']):
        p = ROOT/'audio'/r['filename']
        assert p.name == r['filename'] and digest(p) == r['audio_sha256']
        assert p.stat().st_size == r['member_bytes']
        rows.append(dict(filename=r['filename'], singer=r['singer'],
            source_member=r['archive_member'], sha256=r['audio_sha256'],
            bytes=p.stat().st_size, frames=r['frames'], sample_rate=r['sample_rate'],
            channels=r['channels'], path='external_controls/vocalset_selected_originals_v1/audio/'+p.name,
            license='CC-BY-4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
            source_url='https://zenodo.org/records/1193957',
            attribution='Julia Wilkins, Prem Seetharaman, Alison Wahl, Bryan Pardo; VocalSet (2018)',
            modification='None; selected original archive-member bytes',
            scope='acquired_external_control_source_not_automatic_classifier_or_test_membership'))
    assert len({r['path'] for r in rows}) == 244
    OUT.mkdir(exist_ok=False)
    (OUT/'manifest.json').write_text(json.dumps(rows, indent=2))
    (OUT/'README.md').write_text('''# Selected VocalSet original recordings

Planned source subset: 244 original WAV files acquired for external breath and
vibrato controls. This manifest does not claim the audio is already uploaded or
that every acquired recording entered every assay. No breath annotations or
reviewer names are included. No new AI/human classifier labels are assigned.

Source: Julia Wilkins, Prem Seetharaman, Alison Wahl and Bryan Pardo,
VocalSet: A Singing Voice Dataset (2018), https://zenodo.org/records/1193957 .
The official record API was checked on 12 September 2026 and declares CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/ . Preserve attribution and indicate
modifications. The selected files are unchanged source-member bytes. This source
license does not license unrelated project audio or annotations.

All selected file sizes and SHA-256 values match the frozen acquisition summary.
This check does not claim a fresh checksum validation of the complete upstream
ZIP archive. Some archive filenames have ambiguous singer-directory mappings;
the retained source_member identifies the chosen historical acquisition. Preserve
the separate assay exclusions; source possession does not override them.
''')
    (OUT/'COMMIT.json').write_text(json.dumps(dict(status='prepared_not_uploaded',
        files=244, bytes=sum(r['bytes'] for r in rows),
        acquisition_sha256=digest(source),
        products={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in OUT.iterdir()}),indent=2))
    print(json.dumps(dict(files=244,bytes=sum(r['bytes'] for r in rows),status='prepared_not_uploaded')))


if __name__ == '__main__':main()
