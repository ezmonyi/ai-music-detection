"""Publish three pinned, source-bound historical stem supplements resumably."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify
from publish_suno_test_views_v1 import NOTICE as SUNO_NOTICE, SOURCE_PLAN
from publish_open_model_audio_v1 import NOTICE as OPEN_NOTICE
from publish_aime_test_views_v1 import NOTICE as AIME_NOTICE

ROOT = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
BASE = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
REPO = 'EZMONYI/music-ai-human-test-audio'
BATCH = 100
SPECS = {
    'suno': (1300, 500, 4057257200, '81354e11d8311b86976516e580e109b9990791bdc8394d52d3fcb10605217aca', SUNO_NOTICE),
    'open_models': (600, 200, 1058426400, '80e994f1fa9a5fa1a5acd187eb1a6c192b059ed13ee371ea6d74da66c54f83b3', OPEN_NOTICE),
    'udio': (2000, 500, 10584088000, 'b0aa829539c47675e402bdcd7391aa930bdfc6be94f49880b3bcdbc311298cd6', AIME_NOTICE),
}
CARDS = {
    'SOURCE_CARD_HUMAIR.md': 'd99dbe56e4c4ae59c9b156d966d03bfb811013f827fb97614d3ff97a91174d61',
    'SOURCE_CARD_KUKEDLC.md': 'd86399e69181ce5678520f6fa9cf2560d4fa969c57a3c8f85af2435192537d51',
}


def build(source):
    count, ids_count, size, pin, source_notice = SPECS[source]
    plan = ROOT/'remaining_generated_stem_plans_v1'/(source+'.json')
    assert digest(plan) == pin
    objects = json.loads(plan.read_text())
    prefix = f'audio/remaining_{source}_stems_v1/'
    rows, paths, ids = [], {}, set()
    for obj in objects:
        path = Path(obj['source_paths'][0])
        name = prefix+obj['sha256']+'.wav'
        assert name not in paths and path.suffix == '.wav'
        members = {m['item_id'] for m in obj['memberships']}
        assert members <= {r['id'] for r in obj['original_records']}
        ids.update(members)
        paths[name] = path
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            source_group=obj['source_group'], memberships=obj['memberships'],
            original_records=obj['original_records'], transformation=obj['transformation']))
    assert len(rows) == count and len(ids) == ids_count
    assert sum(r['bytes'] for r in rows) == size
    notice = (f'# Additional historical {source} separated stems\n\n'
        f'{count} existing Demucs files from {ids_count} original recording IDs. '
        'These are historical separated stems, not new songs. Existing file bytes are unchanged. '
        'Original attribution, source declarations and experimental memberships are retained in the manifest. '
        'Previously published or queued vocal objects were excluded by hash.\n\n'
        'The source notice below describes an earlier release; its counts and processing description '
        'refer to that release, not this supplement. Its attribution and source-rights caveats remain applicable. '
        'No blanket output-rights warranty is made.\n\n'+source_notice)
    cards = {}
    if source == 'suno':
        for name, pin in CARDS.items():
            assert digest(SOURCE_PLAN/name) == pin
            cards[name] = (SOURCE_PLAN/name).read_bytes()
    return rows, paths, prefix, notice, cards


def worker(token, source):
    out = BASE/f'hf_remaining_{source}_stems_v1'
    out.mkdir(exist_ok=True)
    rows, paths, prefix, notice, cards = build(source)
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(REPO).private
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text()) == rows
    else:
        save(out/'manifest.json', rows)
    for index, start in enumerate(range(0, len(rows), BATCH)):
        batch = rows[start:start+BATCH]
        receipt = out/f'batch_{index:03d}.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['sha256_verified'] and prior['files'] == [r['path'] for r in batch]
            verify(api, batch, prior['revision'])
            continue
        ops = []
        for row in batch:
            path = paths[row['path']]
            assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
            ops.append(CommitOperationAdd(path_in_repo=row['path'], path_or_fileobj=str(path)))
        if index == 0:
            ops += [CommitOperationAdd(path_in_repo=prefix+'manifest.json', path_or_fileobj=str(out/'manifest.json')),
                    CommitOperationAdd(path_in_repo=prefix+'README.md', path_or_fileobj=notice.encode())]
            ops += [CommitOperationAdd(path_in_repo=prefix+name, path_or_fileobj=data) for name, data in cards.items()]
        revision = api.create_commit(repo_id=REPO, repo_type='dataset', operations=ops,
            commit_message=f'Preserve remaining {source} stems {start+1}-{start+len(batch)} of {len(rows)}').oid
        verify(api, batch, revision)
        save(receipt, dict(revision=revision, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {source} {start+len(batch)}/{len(rows)}', flush=True)
    save(out/'COMMIT.json', dict(status='uploaded_hash_verified', source=source,
        objects=len(rows), original_ids=SPECS[source][1], manifest_sha256=digest(out/'manifest.json'),
        independent_acceptance_required=True, whole_project_complete=False))
