"""Privately archive the explicitly selected 4,484 held processed objects."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def read(path, pin):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin
    return json.loads(raw)


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        return data


def main(backlog, objects, out):
    rows = read(backlog, '60ac1d410744ff9939f259a30f034727000298343127c521769f0e2ef554954d')
    original = read(objects, 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92')
    index = {r['sha256']: r for r in original}
    categories = {'hold_fma_nd', 'hold_fma_unresolved', 'other_source_rights_or_provenance_unresolved'}
    selected = [dict(**r, source_paths=index[r['sha256']]['source_paths'],
        memberships=index[r['sha256']]['memberships'],
        archive_path='audio/' + r['sha256'] + Path(index[r['sha256']]['source_paths'][0]).suffix)
        for r in rows if r['next_action_category'] in categories]
    assert len(selected) == 4484 and sum(r['bytes'] for r in selected) == 9210478926
    out.mkdir(exist_ok=False)
    manifest = (json.dumps(selected, indent=2) + '\n').encode()
    (out / 'manifest.json').write_bytes(manifest)
    notice = b'Private local research preservation only. No public redistribution authorization. Includes ND and unresolved source terms. Exact selected processed bytes, not all project audio or stems. See manifest for provenance and hold category.\n'
    partial = out / 'rights_hold_4484_audio.tar.partial'
    with partial.open('xb') as target, tarfile.open(fileobj=target, mode='w') as archive:
        for name, data in [('manifest.json', manifest), ('README_PRIVATE.txt', notice)]:
            info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
        for i, row in enumerate(selected, 1):
            path = Path(row['source_paths'][0])
            assert path.stat().st_size == row['bytes']
            info = tarfile.TarInfo(row['archive_path']); info.size = row['bytes']; info.mode = 0o600
            with path.open('rb') as stream:
                reader = HashReader(stream)
                archive.addfile(info, reader)
                assert reader.hash.hexdigest() == row['sha256'], row['archive_path']
            if i % 250 == 0:
                print(f'Archived and hash-verified {i}/4484', flush=True)
    final = out / 'rights_hold_4484_audio.tar'
    partial.rename(final)
    summary = dict(status='archive_created_source_hashes_verified', objects=4484,
        audio_bytes=9210478926, archive_bytes=final.stat().st_size,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        private_preservation_only=True, local_download_verified=False)
    (out / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backlog', type=Path, required=True)
    p.add_argument('--objects', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); main(a.backlog, a.objects, a.out)
