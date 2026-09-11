#!/usr/bin/env python3
"""Archive public HF catalog metadata for possible long-audio expansion; no audio."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

REPOS = ('homura23/MUSIC8K', 'ai-music/ai-music-deduplicated', 'awsaf49/sonics')


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'music-thesis-catalog-audit/1.0'}), timeout=60) as response:
        return response.read()


def main():
    output = Path(__file__).resolve().parent.parent/'external_validation/long_source_catalogs_v1'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for repo in REPOS:
        info = json.loads(get('https://huggingface.co/api/datasets/'+repo+'?blobs=true'))
        revision = info['sha']
        readme = get(f'https://huggingface.co/datasets/{repo}/raw/{revision}/README.md')
        name = repo.replace('/', '__')
        for suffix, data in (('.json', json.dumps(info, indent=2).encode()), ('_README.md', readme)):
            with (output/(name+suffix)).open('xb') as f:
                f.write(data)
        files = info.get('siblings', [])
        formats = Counter(Path(f['rfilename']).suffix for f in files)
        groups = Counter(f['rfilename'].split('/')[0] for f in files if Path(f['rfilename']).suffix.lower() in ('.mp3', '.m4a', '.ogg', '.wav', '.flac'))
        entry = {'repo': repo, 'revision': revision, 'gated': info.get('gated'),
                 'license_metadata': info.get('cardData', {}).get('license'),
                 'readme_sha256': hashlib.sha256(readme).hexdigest(),
                 'file_count': len(files), 'extensions': dict(formats), 'audio_files_by_top_folder': dict(groups),
                 'auxiliary_files': [f['rfilename'] for f in files if Path(f['rfilename']).suffix in ('.json', '.jsonl', '.csv', '.py', '.md')],
                 'declared_file_bytes': sum(f.get('size', 0) for f in files),
                 'physically_decoded_audio': 0, 'admitted_to_experiment': False}
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=True), flush=True)
    summary = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'status': 'metadata_only',
               'selection_or_download_of_audio': False, 'sources': results}
    with (output/'summary.json').open('x') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
