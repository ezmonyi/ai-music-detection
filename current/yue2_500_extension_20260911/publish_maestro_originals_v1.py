"""Archive original MAESTRO files underlying the 300 tested Native30 views."""
import json
from pathlib import Path
import soundfile as sf
from huggingface_hub import HfApi,CommitOperationAdd
import publish_maestro_native30_v1 as transport

OUT=transport.OUT.with_name('hf_maestro_originals_v1')


def worker(token):
    api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    plan_path=transport.RC/'audit/native30_origin_plan_v2.json'
    assert transport.digest(plan_path)=='94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964'
    plan=json.loads(plan_path.read_text())
    manifest_path=OUT.with_name('hf_maestro_native30_v1')/'manifest.json'
    tested=json.loads(manifest_path.read_text())
    ids={r['id'] for r in tested}
    assert len(ids)==300
    rows=[r for r in plan['rows'] if r['id'] in ids]
    assert len(rows)==300
    prefix='audio/human_maestro_v3/originals'
    records=[]
    for row in sorted(rows,key=lambda r:r['id']):
        origin=row['source_origin'];path=Path(origin['path'])
        assert origin['scope']=='full_original_file' and transport.digest(path)==origin['sha256']
        info=sf.info(path)
        records.append(dict(id=row['id'],path=f'{prefix}/{row["id"]}.wav',
                    sha256=origin['sha256'],bytes=path.stat().st_size,duration_s=info.duration,
                    sample_rate_hz=info.samplerate,channels=info.channels,subtype=info.subtype,
                    dataset='MAESTRO v3.0.0',label='human',group_id=row['group_id'],
                    attribution='Google LLC; International Piano-e-Competition; MAESTRO dataset authors',
                    license='CC-BY-NC-SA-4.0',license_url='https://creativecommons.org/licenses/by-nc-sa/4.0/',
                    source_url='https://magenta.tensorflow.org/datasets/maestro',
                    modification='Byte-identical experiment-bound original file, renamed to stable experiment ID',
                    tested_view=f'audio/human_maestro_v3/native30/{row["id"]}.wav',local_source=str(path)))
    transport.save(OUT/'manifest.json',records)
    notice='''# MAESTRO v3.0.0 original files underlying the tested views

This folder archives the original audio files bound by the experiment origin
plan for 300 MAESTRO recordings. Each is byte-identical to that bound source
file, renamed to the stable experiment ID. The matching 30-second test view
is linked from the manifest. These are not 300 additional independent songs.

Source and attribution: Google LLC, International Piano-e-Competition, and
the MAESTRO dataset authors. https://magenta.tensorflow.org/datasets/maestro
License: CC BY-NC-SA 4.0. Noncommercial use only; retain attribution and
indicate modifications. Adaptations must use the same license.
https://creativecommons.org/licenses/by-nc-sa/4.0/

Citation: Curtis Hawthorne et al., Enabling Factorized Piano Music Modeling
and Generation with the MAESTRO Dataset, ICLR 2019.
https://openreview.net/forum?id=r1lYRjC9F7

The source-specific license does not apply to all other archive content.
The manifest is a planned roster; inspect available files and upload receipts
before claiming all 300 originals have arrived. No endorsement is implied.
'''
    for index,offset in enumerate(range(0,300,10)):
        batch=records[offset:offset+10]
        operations=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=r['local_source']) for r in batch]
        if index==0:
            public=[{k:v for k,v in r.items() if k!='local_source'} for r in records]
            operations.extend([CommitOperationAdd(path_in_repo=prefix+'/README.md',path_or_fileobj=notice.encode()),
                               CommitOperationAdd(path_in_repo=prefix+'/manifest.json',path_or_fileobj=json.dumps(public,indent=2).encode())])
        result=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=operations,
                                commit_message=f'Archive original MAESTRO files {offset+1}-{offset+10} with provenance')
        remote={x.path:x for x in api.get_paths_info(transport.REPO,paths=[r['path'] for r in batch],repo_type='dataset',revision=result.oid)}
        for r in batch:
            assert remote[r['path']].size==r['bytes'] and remote[r['path']].lfs.sha256==r['sha256']
        transport.save(OUT/f'batch_{index:03d}.json',dict(revision=result.oid,files=[r['path'] for r in batch],sha256_verified=True))
        print(f'MAESTRO originals uploaded and hash-verified {offset+10}/300',flush=True)
    transport.save(OUT/'COMMIT.json',dict(status='uploaded_original_files_for_300_tested_maestro_recordings',
                            files=300,manifest_sha256=transport.digest(OUT/'manifest.json')))


if __name__=='__main__':
    # Reuse only the established stdin/fork/log transport; select a new worker
    # and output directory without modifying the earlier published driver.
    transport.OUT=OUT
    transport.worker=worker
    transport.main()
