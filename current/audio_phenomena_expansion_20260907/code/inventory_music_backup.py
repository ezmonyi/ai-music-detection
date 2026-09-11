"""Inventory music under explicitly selected project roots; no uploads or deletion."""
import argparse
import json
import os
from pathlib import Path

AUDIO = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus', '.aif', '.aiff', '.mid', '.midi'}
SKIP = {'.git', '.cache', '__pycache__', 'venv', '.venv', 'node_modules', 'site-packages', 'hf_models', 'remote_demucs_wheels'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    records = {}
    for entry in plan['roots']:
        root = Path(entry['path'])
        if not root.is_dir():
            continue
        count = total = 0
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not Path(current, d).is_symlink())
            for name in sorted(files):
                path = Path(current, name)
                if path.suffix.lower() not in AUDIO or path.is_symlink() or not path.is_file():
                    continue
                size = path.stat().st_size
                records[str(path)] = {'path': str(path), 'bytes': size, 'root': str(root),
                    'repo_path': entry['repo_prefix'] + '/' + path.relative_to(root).as_posix()}
                count += 1
                total += size
        print(json.dumps({'root': str(root), 'files': count, 'bytes': total}), flush=True)
    result = {'status': 'inventory_not_uploaded', 'records': list(records.values()),
              'files': len(records), 'bytes': sum(r['bytes'] for r in records.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')

if __name__ == '__main__':
    main()
