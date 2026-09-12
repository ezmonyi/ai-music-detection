"""Snapshot anonymous public audio object metadata at one fixed HF revision."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

REPO = 'EZMONYI/music-ai-human-test-audio'
AUDIO = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus', '.aif', '.aiff', '.mid', '.midi'}


def main(out):
    assert not out.exists()
    api = HfApi(token=False)
    info = api.dataset_info(REPO)
    assert not info.private
    revision = info.sha
    records = []
    for item in api.list_repo_tree(REPO, repo_type='dataset', revision=revision, recursive=True):
        if Path(item.path).suffix.lower() not in AUDIO or not hasattr(item, 'size'):
            continue
        lfs = getattr(item, 'lfs', None)
        records.append(dict(path=item.path, bytes=item.size,
                            sha256=lfs.sha256 if lfs else None,
                            git_blob_id=getattr(item, 'blob_id', None)))
        if len(records)%5000 == 0: print(f'Enumerated {len(records)} audio objects', flush=True)
    assert len(records) == len({r['path'] for r in records})
    records.sort(key=lambda r:r['path'])
    out.mkdir()
    raw = (json.dumps(records, indent=2)+'\n').encode()
    (out/'files.json').write_bytes(raw)
    summary = dict(repo=REPO, revision=revision, audio_file_paths=len(records),
        logical_bytes=sum(r['bytes'] for r in records),
        lfs_sha256_available=sum(r['sha256'] is not None for r in records),
        records_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Direct audio/MIDI paths only; does not enumerate archive members',
        independent_source_coverage_proven=False, whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
