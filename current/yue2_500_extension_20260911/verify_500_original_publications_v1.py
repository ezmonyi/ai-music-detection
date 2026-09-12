"""Anonymous acceptance of the frozen Mureka/Suno 500-original publications."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
REPO='EZMONYI/music-ai-human-test-audio'
CONFIG={
    'mureka':dict(plan='mureka_original_publication_plan_v1',output='hf_mureka_originals_v1',
        prefix='audio/mureka_originals_v1/',bytes=2411879065,
        manifest_sha256='ad67de34459614d78c929ec1df973a4f0c8567502d5fb178b5056b37fc558852',
        terminal='500_mureka_originals_uploaded_hash_verified',metadata=['manifest.json','README.md']),
    'suno':dict(plan='suno_original_publication_plan_v1',output='hf_suno_originals_v1',
        prefix='audio/historical_suno_originals_v1/',bytes=2282055145,
        manifest_sha256='5ddf39584dea9a8e22e93b0ecf65ddff31b0ac9c554bae9ad35c30958a8ff23b',
        terminal='500_suno_originals_uploaded_hash_verified',
        metadata=['manifest.json','README.md','SOURCE_CARD_HUMAIR.md','SOURCE_CARD_KUKEDLC.md'])}


def check_receipts(output,rows):
    receipts=sorted(output.glob('batch_*.json'))
    assert len(receipts)==5
    for i,path in enumerate(receipts):
        assert path.name==f'batch_{i:03d}.json'
        receipt=json.loads(path.read_text())
        assert receipt['sha256_verified'] is True
        assert receipt['files']==[r['path'] for r in rows[i*100:(i+1)*100]]


def check(root,source):
    c=CONFIG[source];plan=root/c['plan'];output=root/c['output']
    raw=(plan/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==c['manifest_sha256']
    rows=json.loads(raw)
    assert len(rows)==len({r['id'] for r in rows})==len({r['path'] for r in rows})==500
    assert sum(r['bytes'] for r in rows)==c['bytes']
    assert all(r['path'].startswith(c['prefix']) for r in rows)
    check_receipts(output,rows)
    terminal=json.loads((output/'COMMIT.json').read_text())
    assert terminal['status']==c['terminal'] and terminal['files']==500
    assert terminal['manifest_sha256']==c['manifest_sha256']
    api=HfApi(token=False);info=api.dataset_info(REPO)
    assert not info.private
    for start in range(0,500,50):
        batch=rows[start:start+50]
        remote={r.path:r for r in api.get_paths_info(REPO,
            paths=[r['path'] for r in batch],repo_type='dataset',revision=info.sha)}
        for r in batch:
            f=remote[r['path']]
            assert f.size==r['bytes'] and f.lfs and f.lfs.sha256==r['sha256']
        print(f'{source} independent hash verification {start+50}/500',flush=True)
    metadata_hashes={}
    for name in c['metadata']:
        local=(plan/name).read_bytes()
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{info.sha}/{c["prefix"]}{name}'
        with urllib.request.urlopen(url,timeout=60) as response:assert response.read()==local
        metadata_hashes[name]=hashlib.sha256(local).hexdigest()
    return dict(source=source,repo=REPO,revision=info.sha,checked_files=500,
        checked_bytes=c['bytes'],manifest_sha256=c['manifest_sha256'],
        public_metadata_sha256=metadata_hashes,remote_lfs_sha256_verified=True,
        final_acceptance=True,whole_project_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=sorted(CONFIG))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();result=check(a.root,a.source)
    with a.receipt.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
