"""Detached, resume-safe publication of the tested MAESTRO Native30 views.

Only explicitly selected CC BY-NC-SA 4.0 derivatives are uploaded. Stdin-only
authentication stays in process memory. All commits have local receipts.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd

RC=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
DATA=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
OUT=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/hf_maestro_native30_v1')
REPO='EZMONYI/music-ai-human-test-audio'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2)


def worker(token):
    api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(REPO).private
    cohort_path=DATA/'native30_fhsc_cohort_v1/contract.json'
    assert digest(cohort_path)=='730b9bb25d18715c5ed0191ec9eb7a921d473b9534c9e5bc9196c25f5ce44510'
    cohort=json.loads(cohort_path.read_text())
    rows=[r for r in cohort['rows'] if r['source_group']=='human_maestro_v3']
    assert len(rows)==300 and all(r['label']=='0' for r in rows)
    prefix='audio/human_maestro_v3/native30'
    records=[]
    for row in sorted(rows,key=lambda r:r['id']):
        p=Path(row['input']['path'])
        assert p.stat().st_size==row['input']['bytes'] and digest(p)==row['input']['sha256']
        records.append(dict(id=row['id'],path=f'{prefix}/{row["id"]}.wav',label='human',
                        dataset='MAESTRO v3.0.0',license='CC-BY-NC-SA-4.0',
                        license_url='https://creativecommons.org/licenses/by-nc-sa/4.0/',
                        original_dataset_url='https://magenta.tensorflow.org/datasets/maestro',
                        attribution='Google LLC; International Piano-e-Competition; MAESTRO dataset authors',
                        duration_s=30,sample_rate_hz=44100,channels=2,subtype='FLOAT',
                        modification='Native30 experimental standardized/cropped derivative; not the full original performance',
                        sha256=row['input']['sha256'],group_id=row['group_id'],
                        experiment_role=row['role'],experiment='Native30 seven/eight-family evaluation',
                        local_source=str(p)))
    manifest=OUT/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==records
    else:save(manifest,records)
    notice='''# MAESTRO v3.0.0: tested Native30 derivatives

These 300 files are the exact standardized 30-second views used in the
Native30 evaluation. They are not 300 new complete songs or the complete
MAESTRO dataset. Original files and other experimental views remain separate.

Source: Google LLC, in partnership with the International Piano-e-Competition.
Dataset: https://magenta.tensorflow.org/datasets/maestro
License: Creative Commons Attribution Non-Commercial Share-Alike 4.0.
https://creativecommons.org/licenses/by-nc-sa/4.0/

These cropped/resampled derivatives are shared under the same CC BY-NC-SA 4.0
terms. Noncommercial use only; preserve attribution, indicate modifications,
and apply the same license to adaptations. No endorsement is implied.

Modification: original stereo audio was standardized to the experiment's
44.1 kHz, 30-second FLOAT WAV input. Hashes bind the exact tested files.
Group metadata is retained; composer grouping is not performer identity.

Citation: Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon,
Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck.
Enabling Factorized Piano Music Modeling and Generation with the MAESTRO
Dataset. ICLR 2019. https://openreview.net/forum?id=r1lYRjC9F7

This license applies to this MAESTRO folder, not all other archive contents.
The wider music archive remains incomplete.
'''
    for index,offset in enumerate(range(0,300,20)):
        receipt=OUT/f'batch_{index:03d}.json'
        batch=records[offset:offset+20]
        if receipt.exists():
            prior=json.loads(receipt.read_text())
            files=api.list_repo_files(REPO,repo_type='dataset',revision=prior['revision'])
            assert all(r['path'] in files for r in batch)
            continue
        operations=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=r['local_source']) for r in batch]
        if index==0:
            public=[{k:v for k,v in r.items() if k!='local_source'} for r in records]
            operations += [CommitOperationAdd(path_in_repo=prefix+'/README.md',path_or_fileobj=notice.encode()),
                           CommitOperationAdd(path_in_repo=prefix+'/manifest.json',path_or_fileobj=json.dumps(public,indent=2).encode())]
        result=api.create_commit(repo_id=REPO,repo_type='dataset',operations=operations,
                                 commit_message=f'Archive tested MAESTRO Native30 views {offset+1}-{offset+20} with attribution')
        info=list(api.get_paths_info(REPO,paths=[r['path'] for r in batch],repo_type='dataset',revision=result.oid))
        bypath={r.path:r for r in info}
        for row in batch:
            remote=bypath[row['path']]
            assert remote.size==Path(row['local_source']).stat().st_size
            assert remote.lfs is not None and remote.lfs.sha256==row['sha256']
        save(receipt,dict(revision=result.oid,files=[r['path'] for r in batch],sha256_verified=True))
        print(f'MAESTRO uploaded and hash-verified {offset+20}/300',flush=True)
    save(OUT/'COMMIT.json',dict(repo=REPO,files=300,status='uploaded_tested_native30_views_not_all_project_audio',
                              manifest_sha256=digest(manifest)))


def main():
    token=json.load(sys.stdin)['token']
    OUT.mkdir(exist_ok=True)
    pid=os.fork()
    if pid:
        print(json.dumps({'background_pid':pid,'log':str(OUT/'worker.log')}),flush=True)
        return
    os.setsid()
    log=os.open(OUT/'worker.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.dup2(log,1);os.dup2(log,2)
    null=os.open('/dev/null',os.O_RDONLY);os.dup2(null,0)
    try:worker(token)
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc).replace(token,'[REDACTED]'),flush=True)
        os._exit(1)
    os._exit(0)


if __name__=='__main__':main()
