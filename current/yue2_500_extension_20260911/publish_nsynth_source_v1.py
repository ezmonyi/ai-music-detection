"""Archive the verified official CC-BY NSynth test release as measurement controls."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    token=json.load(sys.stdin)['token']
    api=HfApi(token=token)
    assert api.whoami()['name'].lower()=='ezmonyi'
    root=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/external_validation/nsynth_test_source_v1')
    archive=root/'nsynth-test.jsonwav.tar.gz'
    expected='0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c'
    assert sha(archive)==expected and archive.stat().st_size==349501546
    acquisition=json.loads((root/'acquisition_receipt.json').read_text())
    assert acquisition['release_sha256']==expected and acquisition['notes']==4096
    assert acquisition['license']=='CC BY 4.0' and acquisition['decoded_all'] is True
    notice=b'''# NSynth official test release: external measurement controls

Copyright/attribution: Google Inc.; Jesse Engel, Cinjon Resnick, Adam Roberts,
Sander Dieleman, Douglas Eck, Karen Simonyan, and Mohammad Norouzi (2017),
Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders,
https://arxiv.org/abs/1704.01279 .

Official source and license statement: https://magenta.tensorflow.org/datasets/nsynth
License: Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/ .
Official archive: https://storage.googleapis.com/download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

This is the unmodified official test archive (4,096 four-second notes and
upstream metadata). No audio transformations were applied to this archive.
It supports reconstruction of external phenomenon-validation controls.
The full source release is archived, not a claim that every note was used in
every experiment. These notes are NOT 4,096 human songs or generated AI songs.
NSynth's acoustic/electronic/synthetic instrument categories are not AI/human
authorship labels. This directory does not change the main classifier cohort.

SHA256: 0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c
'''
    prefix='external_controls/nsynth_test_source_v1/'
    repo='EZMONYI/music-ai-human-test-audio'
    provenance={k:acquisition[k] for k in ['download_url','release_sha256','release_bytes','release_md5',
                'license','license_url','attribution','notes','instruments','official_page','limitations']}
    result=api.create_commit(repo_id=repo,repo_type='dataset',commit_message='Preserve official CC-BY NSynth test source for external controls',
        operations=[CommitOperationAdd(path_in_repo=prefix+archive.name,path_or_fileobj=str(archive)),
                    CommitOperationAdd(path_in_repo=prefix+'README.md',path_or_fileobj=notice),
                    CommitOperationAdd(path_in_repo=prefix+'provenance.json',path_or_fileobj=json.dumps(provenance,indent=2).encode())])
    remote=api.get_paths_info(repo,paths=[prefix+archive.name],repo_type='dataset',revision=result.oid)[0]
    assert remote.size==archive.stat().st_size and remote.lfs.sha256==expected
    card=Path(hf_hub_download(repo,filename=prefix+'README.md',repo_type='dataset',revision=result.oid,token=token))
    assert card.read_bytes()==notice
    receipt=dict(repo=repo,revision=result.oid,archive_lfs_sha256_verified=True,license_notice_download_verified=True,
                 archive_sha256=expected,notes=4096,song_count_claim=False,path=prefix+archive.name)
    target=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/nsynth_publication_v1.json')
    with target.open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
