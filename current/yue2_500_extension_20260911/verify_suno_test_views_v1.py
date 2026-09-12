"""Anonymous fixed-revision acceptance of 900 Suno processed inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from publish_suno_test_views_v1 import OUT, PREFIX, NOTICE, build, SOURCE_PLAN
from publish_external_maestro_v1 import verify

REPO = 'EZMONYI/music-ai-human-test-audio'


def run():
    terminal = json.loads((OUT/'COMMIT.json').read_text())
    assert terminal['status'] == '900_suno_test_views_uploaded_hash_verified'
    assert terminal['objects'] == 900 and terminal['original_ids'] == 500
    rows, _ = build()
    receipts = sorted(OUT.glob('batch_*.json'))
    assert [p.name for p in receipts] == [f'batch_{i:03d}.json' for i in range(5)]
    for i, path in enumerate(receipts):
        r = json.loads(path.read_text())
        assert r['sha256_verified'] is True
        assert r['files'] == [x['path'] for x in rows[i*200:(i+1)*200]]
    revision = json.loads(receipts[-1].read_text())['revision']
    api = HfApi(token=False)
    assert not api.dataset_info(REPO, revision=revision).private
    def fetch(name):
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{PREFIX}{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    manifest = fetch('manifest.json')
    assert hashlib.sha256(manifest).hexdigest() == terminal['manifest_sha256']
    assert json.loads(manifest) == rows
    notice = fetch('README.md')
    assert notice == NOTICE.encode()
    for name, pin in {
        'SOURCE_CARD_HUMAIR.md':'d99dbe56e4c4ae59c9b156d966d03bfb811013f827fb97614d3ff97a91174d61',
        'SOURCE_CARD_KUKEDLC.md':'d86399e69181ce5678520f6fa9cf2560d4fa969c57a3c8f85af2435192537d51'
    }.items():
        content=fetch(name)
        assert content==(SOURCE_PLAN/name).read_bytes()
        assert hashlib.sha256(content).hexdigest()==pin
    for start in range(0, len(rows), 50):
        verify(api, rows[start:start+50], revision)
    return dict(revision=revision, verified_audio_objects=900, original_ids=500,
        checked_bytes=sum(r['bytes'] for r in rows),
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        notice_sha256=hashlib.sha256(notice).hexdigest(),
        verification='Anonymous exact manifest/notice and all remote hashes/sizes; not full audio redownload',
        final_acceptance=True, whole_project_complete=False)


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    assert not a.receipt.exists()
    result = run()
    with a.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
