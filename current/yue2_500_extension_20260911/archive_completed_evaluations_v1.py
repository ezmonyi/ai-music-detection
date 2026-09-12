"""Preserve completed model receipts in compressed, versioned local-delivery archives."""
import hashlib
import json
from pathlib import Path
import subprocess

JOBS = [
    ('bc_native30', Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/native30_bc_evaluation_v1_20260911'),
     '6d6acecbb11b6fcee72f7ba8730b1a2a26f2797c38d659ca69151074dd3dba06'),
    ('yue2_expanded_native30', Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/expanded_native30_evaluation_v1'),
     '11f074f867d0e3721c6a586e8c8f86116ff1b0198147ba14f6379b5d6f55570d'),
]
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/completed_evaluation_archives_v1')

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    products = {}
    for name, source, pin in JOBS:
        assert digest(source / 'COMMIT.json') == pin
        target = OUT / (name + '.tar.gz')
        print('Archiving ' + name, flush=True)
        subprocess.run(['tar', '-czf', str(target), '-C', str(source.parent), source.name], check=True)
        assert digest(source / 'COMMIT.json') == pin
        # Validate the compressed stream and tar structure before publishing.
        subprocess.run(['tar', '-tzf', str(target)], stdout=subprocess.DEVNULL, check=True)
        products[target.name] = dict(bytes=target.stat().st_size, sha256=digest(target),
                                    source_commit_sha256=pin)
        print(json.dumps({name: products[target.name]}), flush=True)
    with (OUT / 'COMMIT.json').open('x') as stream:
        json.dump(dict(status='archives_created_and_tar_validated', products=products,
                       note='Local transfer and archive member hash verification still required.'), stream, sort_keys=True)

if __name__ == '__main__':
    main()
