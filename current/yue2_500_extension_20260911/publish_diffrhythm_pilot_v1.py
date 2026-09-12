"""Archive the existing 50-output/10-condition DiffRhythm pilot without relabelling."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import csv
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest,save,verify

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT=ROOT/'hf_diffrhythm_pilot_v1'
SOURCES=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/diffrhythm_publication_sources_v1')
PREFIX='audio/diffrhythm_pilot/originals_v1/'
PINS={'LICENSE.txt':'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
      'SOURCE_CARD.md':'3b4adc71e55af27df5e611e4d29df75d2cabc965c5e36f532894ffbfc8dfc45b',
      'retrieval.csv':'fb104e7eb2713d6278df19190559d05be6749c5b4ef2c4ec3637efcb4c25d839'}
NOTICE='''# Preserved DiffRhythm pilot originals

50 unchanged WAV outputs from Akjava/diffrhythm-instrument-cc0-oepngamearg-10x5-generated,
revision 2be7bd4fb18d317111c0fece3c077c3587bf6f23.
https://huggingface.co/datasets/Akjava/diffrhythm-instrument-cc0-oepngamearg-10x5-generated
Attribution: Akjava (dataset publisher); DiffRhythm authors Ziqian Ning,
Huakang Chen, Yuepeng Jiang, Chunbo Hao, Guobin Ma, Shuai Wang, Jixun Yao,
and Lei Xie. Paper: https://arxiv.org/abs/2503.01183

The source explicitly assigns audio and scripts Apache-2.0. The source card
and Apache license are retained. It reports generation without lyrics from
10 CC0 input clips; exact checkpoint architecture is unspecified. Those are
source-provided statements, not independent clearance of every input right.
No original input clips, new prompts or model weights are included here.

No modification to these WAV bytes was performed for publication. There are
50 outputs but only 10 input conditions; group IDs and historical roles remain
in the manifest. This pilot is not promoted to a new primary evaluation cohort.
Do not count outputs sharing a condition as independent held-out conditions.
No endorsement or blanket license for other project audio is implied.
'''


def build():
    for name,pin in PINS.items(): assert digest(SOURCES/name)==pin
    audit=ROOT/'diversity_source_audio_audit_v1/records.jsonl'
    assert digest(audit)=='ec874be2d40b987facd745262a5664554495634fa72ffa2109691ff1679fba9a'
    rows=[r for r in csv.DictReader((SOURCES/'retrieval.csv').open()) if r['source_group']=='ai_diffrhythm_pilot']
    actual={r['id']:r for r in map(json.loads,audit.read_text().splitlines()) if r['source_group']=='ai_diffrhythm_pilot'}
    assert len(rows)==len(actual)==50 and len({r['group_id'] for r in rows})==10
    public=[]; paths={}
    for row in rows:
        raw=actual[row['id']]; assert raw['status']=='hash_verified'
        assert raw['actual_sha256']==row['source_sha256']==raw['expected_sha256']
        name=PREFIX+row['id']+'.wav';paths[name]=Path(raw['path'])
        public.append(dict(id=row['id'],path=name,sha256=row['source_sha256'],bytes=int(row['source_bytes']),
            source_url=row['source_locator'],group_id=row['group_id'],role=row['historical_role'],
            acquisition_role=row['acquisition_role'],source_group=row['source_group'],
            license='Apache-2.0 (source publisher audio declaration)',modification='None'))
    return public,paths


def worker(token):
    try:
        api=HfApi(token=token);assert api.whoami()['name'].lower()=='ezmonyi'
        assert not api.dataset_info(transport.REPO).private
        rows,paths=build();save(OUT/'manifest.json',rows)
        ops=[]
        for row in rows:
            path=paths[row['path']]
            assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
            ops.append(CommitOperationAdd(path_in_repo=row['path'],path_or_fileobj=str(path)))
        for name in ['SOURCE_CARD.md','LICENSE.txt']:
            ops.append(CommitOperationAdd(path_in_repo=PREFIX+name,path_or_fileobj=str(SOURCES/name)))
        ops.extend([CommitOperationAdd(path_in_repo=PREFIX+'manifest.json',path_or_fileobj=str(OUT/'manifest.json')),
                    CommitOperationAdd(path_in_repo=PREFIX+'README.md',path_or_fileobj=NOTICE.encode())])
        revision=api.create_commit(repo_id=transport.REPO,repo_type='dataset',operations=ops,
            commit_message='Preserve 50 unchanged DiffRhythm pilot outputs with ten-condition grouping').oid
        verify(api,rows,revision)
        save(OUT/'COMMIT.json',dict(status='50_diffrhythm_pilot_originals_uploaded_hash_verified',revision=revision,
            audio_files=50,conditions=10,manifest_sha256=digest(OUT/'manifest.json'),whole_project_complete=False))
        print('Uploaded and verified 50 pilot originals',flush=True)
    except Exception as exc: raise RuntimeError(type(exc).__name__) from None


if __name__=='__main__':
    transport.OUT=OUT;transport.worker=worker;transport.main()
