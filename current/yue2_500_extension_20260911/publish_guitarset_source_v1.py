"""Publish byte-identical GuitarSet controls with source-specific attribution."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/guitarset_archives_v2')
PINS = {
    'audio_mono-mic.zip': (656927981, '275966d6610ac34999b58426beb119c3', '237cdc58353d25c3c9683f4565a0f1cf2db30a9051abca545a919f8f1296dc28'),
    'annotation.zip': (39132574, 'b39b78e63d3446f2e54ddb7a54df9b10', '8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99'),
}
NOTICE = b'''# GuitarSet v1.1.0: external measurement controls

Attribution: Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye,
and Juan P. Bello. GuitarSet: A Dataset for Guitar Transcription (ISMIR 2018).
Official release: https://doi.org/10.5281/zenodo.3371780
License: Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/ .

The original mono microphone audio archive (360 recordings) and corrected
annotation archive are redistributed byte-for-byte without modification.
Other microphone/pickup variants are not included. These are source archives
supporting the project's external bicoherence controls, not a new 360-song
addition to the AI/human classifier cohort. Experiment-specific development,
reserved and unused partitions remain defined by the original experiment
manifests; publishing the source archive does not redefine those partitions.
Do not count annotations or derived windows as additional recordings.
'''


def main():
    token = json.load(sys.stdin)['token']
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    with urllib.request.urlopen('https://zenodo.org/api/records/3371780', timeout=40) as response:
        official = json.load(response)
    assert official['id'] == 3371780
    assert official['metadata']['license']['id'] == 'cc-by-4.0'
    upstream = {f['key']: f for f in official['files']}
    bindings = {}
    payloads = {}
    for name, (size, md5, digest) in PINS.items():
        path = ROOT / name
        assert path.stat().st_size == size == upstream[name]['size']
        assert upstream[name]['checksum'] == 'md5:' + md5
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'md5').hexdigest() == md5
        bindings[name] = dict(bytes=size, sha256=digest, md5=md5)
        payloads[name] = str(path)
    provenance = dict(doi=official['doi'], source='https://zenodo.org/records/3371780',
                      license='CC BY 4.0', creators=official['metadata']['creators'],
                      files=bindings.copy(), transformations='none', recordings=360,
                      classifier_cohort_addition=False)
    payloads['README.md'] = NOTICE
    payloads['provenance.json'] = json.dumps(provenance, indent=2).encode()
    for name in ('README.md', 'provenance.json'):
        raw = payloads[name]
        bindings[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                             git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    repo = 'EZMONYI/music-ai-human-test-audio'
    prefix = 'external_controls/guitarset_source_v1/'
    names = [prefix+n for n in bindings]
    def verify(entries):
        seen = set()
        for entry in entries:
            name = entry.path.removeprefix(prefix)
            bound = bindings[name]
            assert entry.size == bound['bytes']
            if entry.lfs:
                assert entry.lfs.sha256 == bound['sha256']
            else:
                assert entry.blob_id == bound['git_blob_sha1']
            seen.add(entry.path)
        assert seen == set(names)
    head = api.repo_info(repo, repo_type='dataset').sha
    existing = api.get_paths_info(repo, paths=names, repo_type='dataset', revision=head)
    if existing:
        verify(existing)  # Never overwrite a mismatched or partial versioned package.
        revision = head
    else:
        result = api.create_commit(repo_id=repo, repo_type='dataset',
            commit_message='Archive official GuitarSet audio and annotations for external controls',
            operations=[CommitOperationAdd(path_in_repo=prefix+n, path_or_fileobj=v) for n,v in payloads.items()])
        revision = result.oid
        verify(api.get_paths_info(repo, paths=names, repo_type='dataset', revision=revision))
    receipt = dict(repo=repo, revision=revision, prefix=prefix, files_verified=len(names),
                   all_files_hash_verified=True, recordings=360, classifier_cohort_addition=False)
    target = Path(__file__).with_name('guitarset_publication_v1.json')
    if not target.exists():
        with target.open('x') as stream:
            json.dump(receipt, stream, indent=2)
    else:
        saved = json.loads(target.read_text())
        verify(api.get_paths_info(repo, paths=names, repo_type='dataset', revision=saved['revision']))
        receipt = saved
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
