"""Fetch hash-pinned non-audio reproduction inputs from the user's Drive."""
import argparse
import hashlib
import json
from pathlib import Path

from .auth import drive_token
from .drive import DriveRangeReader, TarMember, download_member
from .reproduce import checked_archive_reader, digest


def fetch(manifest_path, destination, *, token=None):
    manifest = json.loads(Path(manifest_path).read_text())
    destination = Path(destination)
    receipts = []
    for item in manifest['objects']:
        if Path(item['name']).name != item['name'] or item['name'] in {'.', '..'}:
            raise ValueError('Unsafe destination name')
        path = destination / item['name']
        if path.exists():
            if path.is_symlink() or not path.is_file() or digest(path) != item['sha256']:
                raise ValueError('Existing destination failed pinned hash; refusing to overwrite it')
            receipts.append(dict(name=item['name'], sha256=item['sha256'], already_present=True))
            continue
        reader = checked_archive_reader(item, token=token)
        # Zero-offset whole-object range, not a TAR extraction. Same bounded
        # hash-verified commit primitive used for selected audio members.
        receipt = download_member(reader, TarMember(item['name'], 0, item['size']), path, item['sha256'])
        receipt.pop('destination', None)
        receipts.append(receipt)
    return dict(kind='pinned_drive_object_download', audio_downloaded=False,
                manifest_sha256=digest(manifest_path), receipts=receipts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--rclone-remote')
    args = parser.parse_args(argv)
    token = drive_token(args.rclone_remote)
    result = fetch(args.manifest, args.destination, token=lambda: token)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
