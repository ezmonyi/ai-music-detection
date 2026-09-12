"""Preserve 100 audited DiffRhythm pilot test views with source terms."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
import publish_diffrhythm_pilot_v1 as source
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify

ROOT = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_diffrhythm_test_views_v1')
PREFIX = 'audio/diffrhythm_pilot/historical_test_views_v1/'
NOTICE = '''# DiffRhythm pilot: historical processed test inputs

100 exact historical 10s/30s input files from 50 outputs and ten conditioning
groups. These cropped/standardized versions differ from the original WAVs.
They are copied unchanged from audited experiment inputs; view labels and crop
parameters are preserved. Pilot status and condition grouping are unchanged.

Attribution: Akjava, source dataset publisher; DiffRhythm authors Ziqian Ning,
Huakang Chen, Yuepeng Jiang, Chunbo Hao, Guobin Ma, Shuai Wang, Jixun Yao,
and Lei Xie. https://arxiv.org/abs/2503.01183
Source: https://huggingface.co/datasets/Akjava/diffrhythm-instrument-cc0-oepngamearg-10x5-generated
Source revision: 2be7bd4fb18d317111c0fece3c077c3587bf6f23.
The publisher assigns audio Apache-2.0; the complete license and source card
are included. These processed files are distributed under Apache-2.0 with
this modification notice. The source reports ten CC0 conditioning clips;
this is not independent clearance of all underlying input rights. No input
clips, model weights or new generations are included. No endorsement implied.
The wider project archive remains incomplete.
'''


def build():
    p = ROOT/'historical_test_view_objects_v1.json'
    assert digest(p) == 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92'
    originals, _ = source.build()
    originals = {r['id']:r for r in originals}
    rows = []; paths = {}; ids = set()
    for obj in json.loads(p.read_text()):
        members = obj['memberships']
        if not any(m['source_group'] == 'ai_diffrhythm_pilot' for m in members): continue
        assert all(m['source_group'] == 'ai_diffrhythm_pilot' for m in members)
        assert not obj['public_paths']
        selected = sorted({m['id'] for m in members})
        assert all(i in originals for i in selected)
        ids.update(selected)
        path = Path(obj['source_paths'][0]); name = PREFIX+obj['sha256']+path.suffix
        paths[name] = path
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            memberships=members, original_records=[originals[i] for i in selected],
            license='Apache-2.0', modification='Historical cropped/standardized test input'))
    assert len(rows) == 100 and len(ids) == 50
    return rows, paths


def worker(token):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    rows, paths = build()
    manifest = OUT/'manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == rows
    else: save(manifest, rows)
    for index, start in enumerate(range(0, 100, 20)):
        batch = rows[start:start+20]; receipt = OUT/f'batch_{index:03d}.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['files'] == [r['path'] for r in batch]
            verify(api, batch, prior['revision']); continue
        ops = []
        for r in batch:
            p = paths[r['path']]
            assert p.stat().st_size == r['bytes'] and digest(p) == r['sha256']
            ops.append(CommitOperationAdd(path_in_repo=r['path'], path_or_fileobj=str(p)))
        if index == 0:
            ops += [CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(manifest)),
                    CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode())]
            for name in ['LICENSE.txt', 'SOURCE_CARD.md']:
                ops.append(CommitOperationAdd(path_in_repo=PREFIX+name, path_or_fileobj=str(source.SOURCES/name)))
        revision = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=ops,
            commit_message=f'Preserve DiffRhythm pilot test views {start+1}-{start+20} of 100').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {start+20}/100', flush=True)
    save(OUT/'COMMIT.json', dict(status='100_diffrhythm_test_views_uploaded_hash_verified',
        objects=100, original_ids=50, conditioning_groups=10,
        manifest_sha256=digest(manifest), whole_project_complete=False))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
