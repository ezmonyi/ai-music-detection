#!/usr/bin/env python3
"""Verify a completed Saraga archive without extracting or admitting recordings."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

EXPECTED = {
    'saraga1.5_hindustani.zip': (4109172493, 'ea9ed2885ea37a1b10e42f60cf299702'),
    'saraga1.5_carnatic.zip': (14376627433, 'e4fcd380b4f6d025964cd16aee00273d'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            md5.update(block)
            sha.update(block)
    return dict(md5=md5.hexdigest(), sha256=sha.hexdigest())


def inspect_zip(path):
    records = []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) <= 20000 and len({m.filename for m in members}) == len(members),
                'Oversized/duplicate archive inventory')
        require(sum(m.file_size for m in members) <= 128*1024**3, 'Uncompressed size exceeds safety limit')
        for member in members:
            p = PurePosixPath(member.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in member.filename,
                    'Unsafe archive member path')
            require(not stat.S_ISLNK(member.external_attr >> 16) and not (member.flag_bits & 1),
                    'Symlink/encrypted archive member forbidden')
            require(member.file_size <= 4*1024**3, 'Single member exceeds safety limit')
            if member.is_dir():
                continue
            md5, sha, count = hashlib.md5(), hashlib.sha256(), 0
            # Reading to EOF verifies the ZIP CRC, not only its directory metadata.
            with archive.open(member) as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    count += len(block)
                    md5.update(block)
                    sha.update(block)
            require(count == member.file_size, 'Short archive member read')
            records.append(dict(path=member.filename, bytes=count, compressed_bytes=member.compress_size,
                crc32=f'{member.CRC:08x}', crc_verified=True, md5=md5.hexdigest(), sha256=sha.hexdigest()))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--zenodo-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.archive.resolve()
    name = path.name.removesuffix('.part')
    require(name in EXPECTED and path.is_file() and not args.archive.is_symlink(), 'Unexpected archive')
    record = json.loads(args.zenodo_record.read_text())
    item = next(f for f in record['files'] if f['key'] == name)
    size, md5 = EXPECTED[name]
    require(record['id'] == 4301737 and record['metadata']['version'] == '1.5'
            and item['size'] == size and item['checksum'] == 'md5:'+md5, 'Pinned archive metadata changed')
    require(path.stat().st_size == size, 'Partial/wrong-size archive')
    require(not args.output.exists(), 'Refusing existing archive audit')
    hashes = file_hashes(path)
    require(hashes['md5'] == md5, 'Whole-archive MD5 mismatch')
    print('Whole archive MD5 passed; reading every member to EOF for CRC and hashes', flush=True)
    records = inspect_zip(path)
    require(file_hashes(path) == hashes and path.stat().st_size == size, 'Archive changed during CRC scan')
    destination = path.with_name(name)
    if destination != path:
        require(not destination.exists(), 'Final archive already exists; manual review required')
        os.link(path, destination)
        path.unlink()
    result = dict(status='passed_archive_integrity_not_audio_admission', source_record=4301737,
        archive_path=str(destination), archive_bytes=size, archive_hashes=hashes,
        metadata_sha256=hashlib.sha256(args.zenodo_record.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        archive_members_verified=len(records), records=records,
        mp3_members=sum(r['path'].lower().endswith('.mp3') for r in records),
        extracted_audio_files=0, physically_decoded_audio_files=0, classifier_admission=False)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
