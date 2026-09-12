"""Fetch two pinned public result bundles for report replay, not audio inference."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

BASE='https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio/resolve/eb0e35208bb9229e1a2ee1a389e181ce7ddee8d6/reports/completed_experiments_thesis_v6_20260912/'
BUNDLES={
    'native60_predictions.tar.gz':('native60_transfer_scores_v1',
        '283c8646dfbcb2be9b4f3b15a57771c50e7eb98ad7beea89c25ec97151f2006a',13941542),
    'native60_features.tar.gz':('native60_feature_package_v1',
        '254d8981603c0b0b724fade5a8ddbe61490ba600ff03d8dd02cd43d57e54250b',274223),
}


def fetch(destination):
    if destination.exists():
        raise FileExistsError('Destination must be new; no existing files are replaced')
    destination.parent.mkdir(parents=True,exist_ok=True)
    receipts=[]
    # Failed transfers remain unpublished. Stage on the same filesystem.
    with tempfile.TemporaryDirectory(prefix='native60-replay-',dir=destination.parent) as temporary:
        stage=Path(temporary);content=stage/'inputs';content.mkdir()
        for name,(folder,pin,size) in BUNDLES.items():
            archive=stage/name
            with urllib.request.urlopen(BASE+name,timeout=60) as response,archive.open('xb') as output:
                shutil.copyfileobj(response,output)
            with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if archive.stat().st_size!=size or digest!=pin:
                raise ValueError('Public archive failed fixed hash/size check: '+name)
            target=content/folder;target.mkdir()
            with tarfile.open(archive,'r:gz') as tar:
                members=tar.getmembers()
                if len({m.name for m in members})!=len(members):
                    raise ValueError('Duplicate archive member')
                for member in members:
                    path=PurePosixPath(member.name)
                    if not member.isfile() or path.is_absolute() or '..' in path.parts:
                        raise ValueError('Unsafe archive member')
                    output=target.joinpath(*path.parts)
                    output.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as source,output.open('xb') as stream:
                        shutil.copyfileobj(source,stream)
            receipts.append(dict(archive=name,sha256=pin,bytes=size,members=len(members)))
        (content/'DOWNLOAD_ACCEPTANCE.json').write_text(json.dumps(dict(
            status='pinned_public_bundles_downloaded_and_hash_verified',bundles=receipts,
            scientific_replay_performed=False,audio_downloaded=False),indent=2))
        if destination.exists():raise FileExistsError('Destination appeared during download')
        content.rename(destination)
    print('Verified public report inputs: '+str(destination))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();fetch(args.destination)
