"""Anonymous fixed-revision acceptance for each additional generated stem release."""
import argparse
import hashlib
import json
from urllib.request import urlopen
from huggingface_hub import HfApi
from publish_remaining_generated_stems_v1 import BASE, BATCH, REPO, SPECS, build
from publish_external_maestro_v1 import verify


def run(source):
    out = BASE/f'hf_remaining_{source}_stems_v1'
    destination = out/'INDEPENDENT_ACCEPTANCE.json'
    assert not destination.exists()
    rows, _, prefix, notice, cards = build(source)
    terminal = json.loads((out/'COMMIT.json').read_text())
    assert terminal['status'] == 'uploaded_hash_verified' and terminal['source'] == source
    assert terminal['objects'] == len(rows) and terminal['original_ids'] == SPECS[source][1]
    receipts = sorted(out.glob('batch_*.json'))
    assert [p.name for p in receipts] == [f'batch_{i:03d}.json' for i in range((len(rows)+BATCH-1)//BATCH)]
    for i, path in enumerate(receipts):
        receipt = json.loads(path.read_text())
        assert receipt['sha256_verified']
        assert receipt['files'] == [r['path'] for r in rows[i*BATCH:(i+1)*BATCH]]
    revision = receipt['revision']
    api = HfApi(token=False)
    assert not api.dataset_info(REPO, revision=revision).private
    def fetch(name):
        with urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{prefix}{name}', timeout=60) as response:
            return response.read()
    manifest = fetch('manifest.json')
    assert hashlib.sha256(manifest).hexdigest() == terminal['manifest_sha256']
    assert json.loads(manifest) == rows
    assert fetch('README.md') == notice.encode()
    for name, data in cards.items():
        assert fetch(name) == data
    for start in range(0, len(rows), 50):
        verify(api, rows[start:start+50], revision)
    result = dict(source=source, revision=revision, verified_audio_objects=len(rows),
        original_ids=SPECS[source][1], checked_bytes=sum(r['bytes'] for r in rows),
        manifest_sha256=terminal['manifest_sha256'], notice_sha256=hashlib.sha256(notice.encode()).hexdigest(),
        verification='Anonymous exact manifest/notice/source cards and all remote hashes/sizes; not full audio redownload',
        final_acceptance=True, whole_project_complete=False)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SPECS, required=True)
    run(parser.parse_args().source)
