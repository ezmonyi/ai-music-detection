"""Preserve 50 early external MAESTRO recordings and their paired MIDI."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import json
import hashlib
from pathlib import Path
import urllib.request
from urllib.parse import quote
from huggingface_hub import HfApi,CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest,save

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT=ROOT/'hf_external_maestro_v1'
PREFIX='external_controls/maestro_dynamics_originals_v1/'
PIN='e16d269194b5aec1958db9267c7e5c5647f9036fd0f39c4a4adc1812e3055058'
NOTICE='''# MAESTRO v3.0.0 external dynamics benchmark inputs

50 unchanged original WAV recordings and 50 paired MIDI files used in the
early external dynamics benchmark. These are 50 performances, not 100 songs.
The experiment measured four 30-second windows per performance; these files
preserve full original inputs, not separate cropped-window exports.

Attribution: Google LLC; International Piano-e-Competition; Curtis Hawthorne,
Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman,
Erich Elsen, Jesse Engel and Douglas Eck. Enabling Factorized Piano Music
Modeling and Generation with the MAESTRO Dataset, ICLR 2019.
Source: https://magenta.tensorflow.org/datasets/maestro
https://openreview.net/forum?id=r1lYRjC9F7
License: Creative Commons Attribution Non-Commercial Share-Alike 4.0.
https://creativecommons.org/licenses/by-nc-sa/4.0/

Preserve attribution, license notices and noncommercial restrictions. No
endorsement is implied. No byte modification was performed for publication.
Composer/title and official archive member paths remain in the manifest;
composer identity is not performer identity. Official archive CRC/size checks
were recorded at acquisition, followed by fresh SHA-256 verification.
These external measurement controls are not newly added classifier test songs.
This source license does not apply to unrelated archive contents.
'''


def build():
    source=ROOT/'early_external_audio_audit_v1/records.json';assert digest(source)==PIN
    selected=[r for r in json.loads(source.read_text()) if r['dataset']=='MAESTRO']
    assert len(selected)==100 and sum(r['kind']=='audio' for r in selected)==50
    public=[];paths={}
    for r in selected:
        name=PREFIX+r['source_member'];paths[name]=Path(r['path'])
        public.append(dict(id=r['id'],kind=r['kind'],path=name,sha256=r['actual_sha256'],bytes=r['expected_bytes'],
            source_url=r['source_url'],source_member=r['source_member'],title=r['title'],composer=r['composer'],
            license='CC-BY-NC-SA-4.0',modification='None; original bytes',role='external_measurement_benchmark'))
    return public,paths


def verify(api,rows,revision):
    expected={r['path']:r for r in rows}
    entries=api.get_paths_info(transport.REPO,paths=list(expected),repo_type='dataset',revision=revision)
    assert {e.path for e in entries}==set(expected)
    for entry in entries:
        row=expected[entry.path];assert entry.size==row['bytes']
        if entry.lfs:
            assert entry.lfs.sha256==row['sha256']
        else:
            url=f'https://huggingface.co/datasets/{transport.REPO}/resolve/{revision}/{quote(entry.path,safe="/")}'
            with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()
            assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']


def worker(token):
    try:
        api=HfApi(token=token);assert api.whoami()['name'].lower()=='ezmonyi'
        assert not api.dataset_info(transport.REPO).private
        rows,paths=build();save(OUT/'manifest.json',rows)
        for index,start in enumerate(range(0,100,20)):
            batch=rows[start:start+20];receipt=OUT/f'batch_{index:03d}.json'
            if receipt.exists():
                prior=json.loads(receipt.read_text());assert prior['files']==[r['path'] for r in batch]
                verify(api,batch,prior['revision']);continue
            ops=[]
            for r in batch:
                p=paths[r['path']];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256']
                ops.append(CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(p)))
            if index==0:
                ops.extend([CommitOperationAdd(path_in_repo=PREFIX+'manifest.json',path_or_fileobj=str(OUT/'manifest.json')),
                            CommitOperationAdd(path_in_repo=PREFIX+'README.md',path_or_fileobj=NOTICE.encode())])
            revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
                commit_message=f'Preserve early external MAESTRO input files {start+1}-{start+len(batch)} of 100').oid
            verify(api,batch,revision)
            save(receipt,dict(revision=revision,files=[r['path'] for r in batch],sha256_verified=True))
            print(f'Uploaded and verified {start+len(batch)}/100',flush=True)
        save(OUT/'COMMIT.json',dict(status='100_external_maestro_inputs_uploaded_hash_verified',audio_files=50,midi_files=50,
            manifest_sha256=digest(OUT/'manifest.json'),source_audit_sha256=PIN,whole_project_complete=False))
    except Exception as exc:raise RuntimeError(type(exc).__name__) from None


if __name__=='__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT=OUT;transport.worker=worker;transport.main()
