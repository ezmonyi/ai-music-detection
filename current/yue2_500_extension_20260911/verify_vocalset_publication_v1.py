"""Read-only final verification of all selected VocalSet audio and metadata."""
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
PLAN = ROOT/'vocalset_publication_plan_v1'
OUT = ROOT/'hf_vocalset_originals_v1'
REPO = 'EZMONYI/music-ai-human-test-audio'
PREFIX = 'external_controls/vocalset_selected_originals_v1/'


def main():
    manifest = (PLAN/'manifest.json').read_bytes()
    rows = json.loads(manifest)
    terminal = json.loads((OUT/'COMMIT.json').read_text())
    assert terminal['files'] == len(rows) == 244
    assert terminal['manifest_sha256'] == hashlib.sha256(manifest).hexdigest()
    paths = []
    for index in range(13):
        receipt = json.loads((OUT/f'batch_{index:03d}.json').read_text())
        assert receipt['sha256_verified']
        assert receipt['files'] == [r['path'] for r in rows[index*20:index*20+20]]
        paths.extend(receipt['files'])
    assert len(paths) == len(set(paths)) == 244
    api = HfApi(token=False)
    revision = api.dataset_info(REPO).sha
    for start in range(0,244,50):
        batch = rows[start:start+50]
        remote = {r.path:r for r in api.get_paths_info(REPO,paths=[r['path'] for r in batch],repo_type='dataset',revision=revision)}
        for r in batch:
            assert remote[r['path']].size == r['bytes']
            assert remote[r['path']].lfs.sha256 == r['sha256']
    for name in ['manifest.json','README.md']:
        with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}',timeout=60) as response:
            assert response.read() == (PLAN/name).read_bytes()
    result = dict(status='independent_public_revision_verification_passed',revision=revision,
        audio_files=244,audio_bytes=sum(r['bytes'] for r in rows),metadata_files=2,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),whole_project_complete=False)
    with (OUT/'INDEPENDENT_ACCEPTANCE.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__ == '__main__':main()
