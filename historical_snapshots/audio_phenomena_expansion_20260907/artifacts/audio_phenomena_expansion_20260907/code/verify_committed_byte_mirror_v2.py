"""Verify trusted committed bytes with an explicitly selected manifest schema."""
import argparse
import hashlib
import json
from pathlib import Path
import socket


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, expected_hash, key):
    root = root.resolve(strict=True)
    commit = root / 'COMMIT.json'
    if sha(commit) != expected_hash:
        raise ValueError('untrusted commit bytes')
    manifest = json.loads(commit.read_text())
    products = manifest[key]
    if manifest.get('status') != 'committed' or not isinstance(products, dict) or not products:
        raise ValueError('invalid publication')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('symlink in mirror')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(products) | {'COMMIT.json'}:
        raise ValueError('mirror inventory mismatch')
    total = 0
    for name, record in products.items():
        path = root / name
        if Path(name).is_absolute() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('path escape')
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('mirror byte mismatch: ' + name)
        total += record['bytes']
    if sha(commit) != expected_hash:
        raise ValueError('commit changed during verification')
    return {'passed': True, 'scope': 'complete_byte_and_inventory_mirror_not_scientific_reanalysis',
        'host': socket.gethostname(), 'mirror_root': str(root), 'products': len(products),
        'product_bytes': total, 'commit_sha256': expected_hash, 'manifest_key': key,
        'code_sha256': sha(Path(__file__)), 'embedded_provenance_paths_rewritten': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--commit-sha256', required=True)
    parser.add_argument('--manifest-key', required=True, choices=['files', 'products'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError('new receipt outside mirror required')
    result = verify(root, args.commit_sha256, args.manifest_key)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
