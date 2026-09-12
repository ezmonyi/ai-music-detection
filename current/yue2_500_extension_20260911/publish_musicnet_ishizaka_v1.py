"""Publish only the 39 MusicNet Ishizaka WTC-I recordings with primary CC0 evidence."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import csv
import json
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest,save,verify
from bounded_hf_http_v1 import install

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
CODE=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT=ROOT/'hf_musicnet_ishizaka_v1'
PREFIX='audio/musicnet_ishizaka_cc0_v1/'
LICENSE_SOURCE='https://kimikoishizaka.bandcamp.com/album/bach-well-tempered-clavier-book-1'
NOTICE='''# MusicNet: Kimiko Ishizaka WTC-I subset

39 existing MusicNet WAV recordings attributed by official MusicNet metadata
to Kimiko Ishizaka, playing J.S. Bach's Well-Tempered Clavier Book I, BWV 846–868
within this selection. They are not a new experiment or 39 new independent
performers. No audio transformation was performed for this publication; the
exact preserved MusicNet WAV bytes are retained, not the artist's original
high-resolution album files.

The artist's own release page explicitly identifies the recording as CC0:
https://kimikoishizaka.bandcamp.com/album/bach-well-tempered-clavier-book-1
https://creativecommons.org/publicdomain/zero/1.0/
Performance attribution: Kimiko Ishizaka. Composition: Johann Sebastian Bach.
MusicNet dataset authors: John Thickstun, Zaid Harchaoui and Sham M. Kakade.
Dataset and official attribution metadata: https://zenodo.org/records/5120004
Paper: https://arxiv.org/abs/1611.09827

The selection is bound to official MusicNet performer/work metadata and the
project's freshly verified source hashes. Identity was not independently
established by downloading and comparing the original high-resolution album.
No album artwork, dedication texts, scores, MIDI or annotation files are included.
CC0 is not assigned to the remaining 291 MusicNet recordings or other datasets.
No endorsement is implied. Preserve the original experimental IDs and roles.
'''


def build():
    meta=CODE/'musicnet_metadata.csv'
    assert digest(meta)=='1308d938bafb594e3b0471f2bdda3630da352f881857f265f92114cf398de7ca'
    selected={r['id']:r for r in csv.DictReader(meta.open()) if r['source']=='Kimiko Ishizaka'}
    assert len(selected)==39
    assert all(r['ensemble']=='Solo Piano' and r['composition'].startswith('WTK I,') and 846<=int(r['catalog_name'][3:])<=869 for r in selected.values())
    audit=ROOT/'diversity_source_audio_audit_v1/records.jsonl'
    assert digest(audit)=='ec874be2d40b987facd745262a5664554495634fa72ffa2109691ff1679fba9a'
    public=[];paths={}
    for r in map(json.loads,audit.read_text().splitlines()):
        if r['source_group']!='human_musicnet':continue
        identity=r['id'].rsplit('_',1)[-1]
        if identity not in selected:continue
        credit=selected[identity];assert r['status']=='hash_verified' and r['actual_sha256']==r['expected_sha256']
        name=PREFIX+identity+'.wav';paths[name]=Path(r['path'])
        public.append(dict(id=r['id'],musicnet_id=identity,path=name,bytes=r['bytes'],sha256=r['actual_sha256'],
            source_url=r['source_locator'],performer=credit['source'],composer=credit['composer'],
            composition=credit['composition'],movement=credit['movement'],catalog_name=credit['catalog_name'],
            recording_license='CC0-1.0',license_source=LICENSE_SOURCE,modification='None to preserved MusicNet WAV bytes'))
    assert len(public)==39;return public,paths


def worker(token):
    try:
        api=HfApi(token=token);assert api.whoami()['name'].lower()=='ezmonyi'
        assert not api.dataset_info(transport.REPO).private
        rows,paths=build();save(OUT/'manifest.json',rows);ops=[]
        for row in rows:
            p=paths[row['path']];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
            ops.append(CommitOperationAdd(path_in_repo=row['path'],path_or_fileobj=str(p)))
        ops.extend([CommitOperationAdd(path_in_repo=PREFIX+'manifest.json',path_or_fileobj=str(OUT/'manifest.json')),
                    CommitOperationAdd(path_in_repo=PREFIX+'README.md',path_or_fileobj=NOTICE.encode())])
        revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
            commit_message='Preserve 39 MusicNet Ishizaka recordings with primary CC0 attribution').oid
        verify(api,rows,revision)
        save(OUT/'COMMIT.json',dict(status='39_musicnet_ishizaka_recordings_uploaded_hash_verified',revision=revision,
            audio_files=39,manifest_sha256=digest(OUT/'manifest.json'),whole_project_complete=False))
        print('Uploaded and verified 39 MusicNet recordings',flush=True)
    except Exception as exc:raise RuntimeError(type(exc).__name__) from None


if __name__=='__main__':
    install();transport.OUT=OUT;transport.worker=worker;transport.main()
