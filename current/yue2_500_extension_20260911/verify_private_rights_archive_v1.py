"""Stream-verify local held-audio archive without extracting a second audio copy."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main(folder):
    receipt = folder / 'LOCAL_ACCEPTANCE.json'
    assert not receipt.exists()
    commit = json.loads((folder / 'COMMIT.json').read_text())
    raw = (folder / 'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == commit['manifest_sha256']
    records = json.loads(raw)
    expected = {r['archive_path']: r for r in records}
    assert len(expected) == commit['objects'] == 4484
    path = folder / 'rights_hold_4484_audio.tar'
    assert path.stat().st_size == commit['archive_bytes']
    seen = set()
    with tarfile.open(path, 'r|') as archive:
        for member in archive:
            assert member.isfile() and member.name not in seen
            seen.add(member.name)
            stream = archive.extractfile(member)
            assert stream is not None
            if member.name == 'manifest.json':
                assert stream.read() == raw
            elif member.name == 'README_PRIVATE.txt':
                assert stream.read().startswith(b'Private local research preservation only.')
            else:
                row = expected[member.name]
                assert member.size == row['bytes']
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
                assert h.hexdigest() == row['sha256'], member.name
    assert seen == set(expected) | {'manifest.json', 'README_PRIVATE.txt'}
    result = dict(status='local_archive_all_audio_sha256_verified', objects=4484,
        audio_bytes=commit['audio_bytes'], archive_bytes=path.stat().st_size,
        archive_path=str(path.resolve()), manifest_sha256=commit['manifest_sha256'],
        extracted_audio_copy=False, public_upload=False)
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    main(p.parse_args().folder)
