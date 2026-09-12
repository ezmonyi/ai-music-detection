"""Publish only committed legacy spectral result files, not source audio."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
PINS={
    'legacy_fullmix_v1':'c32a466665cfe3c8189f343d5a619f5d52f7e484f84613c6fca427d5eaa2d54f',
    'legacy_spectral_measurement_v1':'926662f75af81b76cc0f5b0ccaf9b20be5b1cac685a1d4bf511efc3a0a65e84b',
    'legacy_spectral_locked_v2':'86a02ca2209396eed8fb40b767e5ca537d8bf1c8d4cac1637b64dc2273c2758a',
}


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    token=json.load(sys.stdin)['token'];api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    repo='EZMONYI/music-ai-human-test-audio';prefix='reports/yue2_legacy_spectral_v1/'
    files={};operations=[]
    for folder,pin in PINS.items():
        root=ROOT/folder;assert sha(root/'COMMIT.json')==pin
        commit=json.loads((root/'COMMIT.json').read_text())
        for name in ['COMMIT.json',*commit['products']]:
            path=(root/name).resolve();assert path.is_relative_to(root)
            assert path.is_file() and path.suffix not in ['.wav','.flac','.mp3']
            digest=sha(path);size=path.stat().st_size
            if name!='COMMIT.json':
                assert digest==commit['products'][name]['sha256'] and size==commit['products'][name]['bytes']
            target=prefix+folder+'/'+name
            files[target]=dict(bytes=size,sha256=digest,git_blob_sha1=hashlib.sha1(b'blob '+str(size).encode()+b'\0'+path.read_bytes()).hexdigest())
            operations.append(CommitOperationAdd(path_in_repo=target,path_or_fileobj=str(path)))
    notice=b'''# YuE2 legacy spectral results (completed sub-experiment)

1,000 first30 clips: 500 historical human and 500 YuE2. Fullmix, raw vocals,
and externally bias-corrected vocals are retained with fixed band comparisons.
The model comparison uses 400/100 development/test recordings per class and
disjoint artist/source-song groups. Human holdout recordings were used in prior
experiments. YuE2 is present in training: these are NOT unseen-generator results.
Two short YuE2 originals were padded in the legacy view, unlike Native30.

18 representations, ridge alpha 10, targets -1/+1, original threshold zero.
For the 22 spectral statistics: fullmix BA .925/AUC .9706; raw vocal BA
.950/AUC .9837; corrected vocal BA .950/AUC .9832. All outcomes are retained;
no winner or significance claim is made. The first comparison adapter stopped
on a threshold-check mismatch; v2 restores the historical zero threshold.

Shared external calibration does not isolate intrinsic AI causation. The
spectrograms are not phrase-aligned; excerpt activity and source confounding
remain. This is not the final whole-project report. No source audio is copied
by this result publication. COMMIT files bind each result package.
'''
    operations.append(CommitOperationAdd(path_in_repo=prefix+'README.md',path_or_fileobj=notice))
    result=api.create_commit(repo_id=repo,repo_type='dataset',operations=operations,
                             commit_message='Archive verified YuE2 legacy spectral comparisons and heatmaps')
    names=list(files)
    verified=set()
    for offset in range(0,len(names),20):
        for entry in api.get_paths_info(repo,paths=names[offset:offset+20],repo_type='dataset',revision=result.oid):
            bound=files[entry.path];assert entry.size==bound['bytes']
            if entry.lfs:assert entry.lfs.sha256==bound['sha256']
            else:assert entry.blob_id==bound['git_blob_sha1']
            verified.add(entry.path)
    assert verified==set(files)
    receipt=dict(repo=repo,revision=result.oid,files_verified=len(files),all_files_hash_verified=True,prefix=prefix,
                 project_complete=False)
    with (Path(__file__).parent/'legacy_results_publication_v1.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
