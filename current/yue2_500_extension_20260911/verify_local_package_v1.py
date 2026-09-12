"""Verify recovered package products by relative name against producer COMMIT."""
import argparse
import hashlib
import json
from pathlib import Path

def verify(root):
    root = Path(root).resolve()
    raw = (root / 'COMMIT.json').read_bytes()
    commit = json.loads(raw)
    products = commit['products']
    for name, expected in products.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file(), name
        assert path.stat().st_size == expected['bytes'], name
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], name
    return dict(status='all_local_products_hash_verified', root=str(root),
                products=len(products), commit_sha256=hashlib.sha256(raw).hexdigest())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
