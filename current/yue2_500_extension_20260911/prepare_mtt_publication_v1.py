"""Prepare attribution-preserving publication of 500 audited MagnaTagATune clips."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main(audit,acquisition,out):
    commit=json.loads((audit/'COMMIT.json').read_text())
    raw=(audit/'records.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==commit['records_sha256']
    records={r['id']:r for r in map(json.loads,raw.splitlines()) if r['source_group']=='human_magnatagatune'}
    selected=[r for r in csv.DictReader(acquisition.open()) if r['source_id']=='human_magnatagatune']
    assert len(records)==len(selected)==500
    rows=[]
    for r in selected:
        a=records[r['item_id']]
        assert a['status']=='hash_verified' and a['actual_sha256']==a['expected_sha256']
        assert r['artist_or_creator'] and r['title'] and r['group_id'].startswith('http://he3.magnatune.com/')
        rows.append(dict(id=r['item_id'],path='audio/magnatagatune_selected_v1/'+r['item_id']+'.mp3',
            sha256=a['actual_sha256'],bytes=a['bytes'],source_archive_member=r['container_member'],
            source_dataset='confit/magnatagatune',source_revision=r['source_revision'],
            artist=r['artist_or_creator'],title=r['title'],original_track_url=r['group_id'],
            license='CC-BY-NC-SA-1.0',license_url='https://creativecommons.org/licenses/by-nc-sa/1.0/',
            license_basis='Magnatune official API music licensing statement and institutional MagnaTagATune provenance',
            modification='No byte modification here; upstream MagnaTagATune excerpt, not the full recording'))
    assert len({r['id'] for r in rows})==500
    assert sum(r['bytes'] for r in rows)==58615981
    out.mkdir(exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(rows,indent=2))
    (out/'README.md').write_text('''# Selected MagnaTagATune research clips

500 upstream MP3 excerpts, 58,615,981 bytes, retained without byte modification.
These are the selected original dataset clips, not 500 full-length recordings
and not the further cropped 10-second analysis views. Individual artists,
titles, original Magnatune URLs and source archive members are in the manifest.

Audio source: Magnatune. Official API licensing statement identifies Creative
Commons Attribution-NonCommercial-ShareAlike 1.0:
https://magnatune.com/info/api
License: https://creativecommons.org/licenses/by-nc-sa/1.0/
Noncommercial use only; retain attribution and share adaptations under the same
license. This notice does not relicense the audio as CC BY-NC-SA 4.0. No warranty
or endorsement is implied; additional rights may affect particular uses.

Dataset collection: Edith Law, Olivier Gillet and collaborators; institutional
hosting: City University MIRG. Source and citation:
https://mirg.city.ac.uk/datasets/magnatagatune/index1.html
Edith Law, Kris West, Michael Mandel, Mert Bay and J. Stephen Downie (2009),
Evaluation of algorithms using games: the case of music annotation, ISMIR.

Selected bytes came from the pinned confit/magnatagatune HF mirror recorded in
the manifest. No complete Magnatune catalogue or paid member downloads were
accessed for this archive. These historical experiment IDs are already present
in the project catalogue; publication does not create a new independent cohort.
This is a source-byte-audited publication plan, not completed upload acceptance.
''')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'COMMIT.json').write_text(json.dumps(dict(status='500_mtt_clips_publication_plan_not_uploaded',
        files=500,bytes=58615981,manifest_sha256=sha(out/'manifest.json'),
        audit_commit_sha256=sha(audit/'COMMIT.json'),acquisition_sha256=sha(acquisition),
        whole_project_complete=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('audit','acquisition','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();main(a.audit,a.acquisition,a.out)
