"""Publish verified completion boundaries, not an unsupported final report."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd

def main():
    token = json.load(sys.stdin)['token']
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    roots = Path('/mnt/nfs-data/users/yi')
    targets = {
        'bc': (roots/'audio_phenomena_expansion_20260907/native30_bc_evaluation_v1_20260911/COMMIT.json',
               '6d6acecbb11b6fcee72f7ba8730b1a2a26f2797c38d659ca69151074dd3dba06'),
        'yue2': (roots/'yue2_500_extension_20260911/expanded_native30_evaluation_v1/COMMIT.json',
                 '11f074f867d0e3721c6a586e8c8f86116ff1b0198147ba14f6379b5d6f55570d')}
    for path, pin in targets.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin
    repo = 'EZMONYI/music-ai-human-test-audio'
    assert not api.dataset_info(repo).private
    code = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
    manifests = [json.loads((code/folder/'manifest.json').read_text())
                 for folder in ['hf_maestro_native30_v1', 'hf_maestro_originals_v1']]
    verified = 0
    for manifest in manifests:
        for start in range(0, len(manifest), 50):
            batch = manifest[start:start+50]
            remote = {r.path: r for r in api.get_paths_info(repo, paths=[r['path'] for r in batch], repo_type='dataset')}
            for row in batch:
                item = remote[row['path']]
                expected_size = row.get('bytes')
                if expected_size is None:
                    expected_size = Path(row['local_source']).stat().st_size
                assert item.size == expected_size and item.lfs.sha256 == row['sha256']
                verified += 1
    assert verified == 600
    report = '''# Verified experiment checkpoint — 12 September 2026

This is an intermediate delivery checkpoint, not the final thesis report.

The original Native30 eight-family BC evaluation has completed 103,530
scheduled model instances with zero failures. The YuE2-expanded Native30
evaluation has completed 112,455 instances and prediction replays. These are
evaluation instances, not numbers of songs. The original development cohort
has 3,830 recordings; the expanded cohort has 4,228 development recordings.
The 100 locked YuE2 recordings remain excluded and have not been scored.

BC report generation initially failed because reporting-only scope fields
were incorrectly checked against producer records. The next attempt exposed
an unbound historical-test reference. Versioned reporting fixes preserve the
original model results and distinguish frozen lineage from report-time
references. Seven regression tests passed. Final BC reporting and YuE2
source-balanced synthesis are still pending; no new headline accuracy is
claimed in this checkpoint. Existing seven-family reports remain available.

The archive currently contains 300 tested MAESTRO recordings in two forms:
300 Native30 views and 300 full originals (600 audio files, not 600 songs).
This publication freshly checked all 600 remote sizes and LFS SHA-256 values
against the upload manifests. Their CC BY-NC-SA 4.0 source-specific terms and
attributions remain in the audio folders. Other tested sources are not yet
fully archived here. Public availability is not a blanket redistribution grant.

Remaining work includes report completion, protected-test scoring, applicable
additional YuE2 analysis arms, full permitted-audio archival, local result
verification and the final English thesis. Working PDFs and earlier reports
are retained as intermediate versions. No final winner has been selected.

Code: https://github.com/ezmonyi/ai-music-detection
'''
    provenance = dict(evaluation_commit_sha256={k:v[1] for k,v in targets.items()},
                      freshly_verified_maestro_audio_files=verified, distinct_maestro_recordings=300,
                      final_delivery_complete=False, locked_yue2_scored=False)
    prefix = 'reports/completion_checkpoint_20260912_v1/'
    payloads = {'REPORT_EN.md': report.encode(), 'provenance.json': json.dumps(provenance,indent=2).encode()}
    result = api.create_commit(repo_id=repo, repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=prefix+name,path_or_fileobj=data) for name,data in payloads.items()],
        commit_message='Preserve completed evaluation checkpoint and verify 600 MAESTRO audio files')
    from huggingface_hub import hf_hub_download
    for name, data in payloads.items():
        downloaded = Path(hf_hub_download(repo, prefix+name, repo_type='dataset',revision=result.oid,token=token))
        assert hashlib.sha256(downloaded.read_bytes()).digest() == hashlib.sha256(data).digest()
    receipt = dict(revision=result.oid, **provenance)
    with (code/'completion_checkpoint_publication_v1.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
