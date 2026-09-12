"""Publish audited project-generated originals and exact standardized views."""
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
AUDIT = ROOT / 'open_model_audio_audit_v1'
OUT = ROOT / 'hf_open_model_audio_v1'
PREFIX = 'audio/project_open_models_v1/'
NOTICE = '''# Project-generated ACE-Step and HeartMuLa audio

500 ACE-Step and 500 HeartMuLa recordings generated for this research project,
each with its original file and exact standardized 30-second analysis view.
2,000 files represent 1,000 recordings, not independent song populations.
Originals retain their bytes. Standardized views may be resampled, cropped or
padded; modification counts are preserved in the manifest. Forty-two HeartMuLa
views contain padding. Do not substitute them for duration-eligible Native30.

Model attribution and upstream licenses (not blanket audio-output licenses):
ACE-Step 1.5, ACE Studio / StepFun, MIT model:
https://huggingface.co/ACE-Step/Ace-Step1.5
https://arxiv.org/abs/2602.00744
HeartMuLa Team, HeartMuLa-oss-3B-happy-new-year, Apache-2.0 model:
https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year
https://arxiv.org/abs/2601.10547
HeartCodec-oss-20260123: https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123

Conditions came from bolshyC/Muse (publisher-declared MIT, automatically
generated lyrics/style): https://huggingface.co/datasets/bolshyC/Muse
https://arxiv.org/abs/2601.03973
No Muse source audio is included. Prompt text is not separately reproduced here.
The manifest preserves prompt IDs; maintain shared-prompt grouping across models.
The publisher's declarations do not independently clear every possible third-party
right. No endorsement, exclusive ownership, or blanket output license is claimed.
This is the user's requested research archive, not a universal rights warranty.

The project's open_model_audio_catalogue_v1 binds original hashes to historical
experiment roles and groups. Raw-source and standardized-view hashes differ.
Publication does not create a new untouched evaluation set. Other project audio,
derived views and separated stems remain outside this folder's coverage.
'''


def worker(token):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    commit = json.loads((AUDIT/'COMMIT.json').read_text())
    assert commit['status'] == '1000_original_view_pairs_hash_verified_not_uploaded'
    assert transport.digest(AUDIT/'verified_records.json') == commit['manifest_sha256']
    records = json.loads((AUDIT/'verified_records.json').read_text())
    assert len(records) == len({r['id'] for r in records}) == 1000
    rows, local = [], {}
    for record in records:
        s = record['standardization_record']
        for f in record['files']:
            path = PREFIX + record['model'] + '/' + f['kind'] + '/' + Path(f['path']).name
            local[path] = Path(f['path'])
            rows.append(dict(id=record['id'], model=record['model'], prompt_id=record['prompt_id'],
                path=path, kind=f['kind'], bytes=f['bytes'], sha256=f['sha256'],
                source_frames=s['source_frames'], source_sample_rate=s['source_sample_rate'],
                standardized_frames=s['output_frames'], standardized_sample_rate=s['output_sample_rate'],
                standardized_padding_frames=s['padding_frames'], standardized_truncated_frames=s['truncated_frames']))
    assert len(rows) == len(local) == 2000
    if (OUT/'manifest.json').exists():
        assert json.loads((OUT/'manifest.json').read_text()) == rows
    else:
        transport.save(OUT/'manifest.json', rows)
    for i, start in enumerate(range(0, 2000, 20)):
        batch = rows[start:start+20]
        receipt = OUT/f'batch_{i:03d}.json'
        for r in batch:
            assert local[r['path']].stat().st_size == r['bytes']
            assert transport.digest(local[r['path']]) == r['sha256']
        if receipt.exists():
            old = json.loads(receipt.read_text())
            assert old['files'] == [r['path'] for r in batch]
            revision = old['revision']
        else:
            ops = [CommitOperationAdd(path_in_repo=r['path'], path_or_fileobj=str(local[r['path']])) for r in batch]
            if i == 0:
                ops += [CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(OUT/'manifest.json')),
                        CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode())]
            revision = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=ops,
                commit_message=f'Archive project-generated original/view files {start+1}-{start+20}').oid
        remote = {r.path:r for r in api.get_paths_info(transport.REPO,
            paths=[r['path'] for r in batch], repo_type='dataset', revision=revision)}
        for r in batch:
            assert remote[r['path']].size == r['bytes']
            assert remote[r['path']].lfs and remote[r['path']].lfs.sha256 == r['sha256']
        if not receipt.exists():
            transport.save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Open-model audio verified {start+20}/2000', flush=True)
    transport.save(OUT/'COMMIT.json', dict(status='2000_original_view_files_uploaded_hash_verified',
        identities=1000, files=2000, manifest_sha256=transport.digest(OUT/'manifest.json'), whole_project_complete=False))


if __name__ == '__main__':
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
