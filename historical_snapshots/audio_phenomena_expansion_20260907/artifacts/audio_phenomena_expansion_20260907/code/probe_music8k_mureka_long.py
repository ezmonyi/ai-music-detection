#!/usr/bin/env python3
"""Freeze and decode twelve Mureka-v9 acquisition probes; no AI feature fitting."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

REVISION = '05232438ba76a7bc55cbebfdc6d5f4011c980bba'
SEED = 'music8k-mureka-v9-physical-probe-20260907'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def select(catalog):
    if catalog['sha'] != REVISION:
        raise ValueError('Unpinned catalog')
    entries = []
    for item in catalog['siblings']:
        path = Path(item['rfilename'])
        if path.parts[0] != 'mureka_v9':
            continue
        if len(path.parts) != 2 or path.suffix != '.mp3' or not path.stem.isdigit():
            raise ValueError('Unexpected candidate filename')
        if item['size'] != item['lfs']['size']:
            raise ValueError('LFS size mismatch')
        entries.append({'id': path.stem, 'path': str(path), 'bytes': item['size'],
                        'sha256': item['lfs']['sha256'],
                        'rank': hashlib.sha256((SEED+'|'+path.stem).encode()).hexdigest()})
    if len(entries) != 662 or len({r['id'] for r in entries}) != 662:
        raise ValueError('Candidate population mismatch')
    return sorted(entries, key=lambda r:r['rank'])


def decode(path, log_path):
    info = json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe', '-v', 'error',
        '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate,channels,codec_name,bit_rate',
        '-of', 'json', str(path)], text=True))['streams']
    if len(info) != 1:
        raise ValueError('Expected one selected audio stream')
    stream = info[0]
    rate, channels = int(stream['sample_rate']), int(stream['channels'])
    h, total, finite = hashlib.sha256(), 0, True
    cmd = ['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
           '-map', '0:a:0', '-c:a', 'pcm_f32le', '-f', 'f32le', '-']
    with log_path.open('xb') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)
        while True:
            block = proc.stdout.read(1 << 20)
            if not block:
                break
            if len(block) % 4:
                raise ValueError('Non-float-aligned decoded bytes')
            finite = finite and bool(np.isfinite(np.frombuffer(block, dtype='<f4')).all())
            h.update(block)
            total += len(block)
        if proc.wait() != 0:
            raise ValueError('FFmpeg full decoding failed')
    if total == 0 or total % (4*channels) or not finite:
        raise ValueError('Invalid decoded audio')
    frames = total//(4*channels)
    return dict(sample_rate=rate, channels=channels, codec=stream['codec_name'],
                decoded_frames=frames, decoded_duration_s=frames/rate,
                all_samples_finite=finite, decoded_native_float32_sha256=h.hexdigest(),
                eligible_native60=(frames >= 60*rate and rate >= 16000 and channels in (1,2)),
                command=cmd)


def main():
    root = Path(__file__).resolve().parent.parent
    catalog_path = root/'external_validation/long_source_catalogs_v1/homura23__MUSIC8K.json'
    metadata = root/'external_validation/music8k_metadata_v1/metadata'
    output = root/'external_validation/music8k_mureka_probe_v1'
    output.mkdir(parents=True, exist_ok=False)
    ranked = select(json.loads(catalog_path.read_text()))
    csv_rows = {r['id']: r for r in csv.DictReader((metadata/'pop_1000_unique_artist.csv').open())}
    json_rows = {str(r['id']):r for r in map(json.loads, (metadata/'songs_musiccaps.jsonl').open())}
    if any(r['id'] not in csv_rows or r['id'] not in json_rows for r in ranked):
        raise ValueError('Missing prompt/reference metadata')
    selected = ranked[:12]
    ffmpeg_version = subprocess.check_output(['/opt/homebrew/bin/ffmpeg', '-version'], text=True)
    contract = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'repo': 'homura23/MUSIC8K',
        'revision': REVISION, 'code_sha256': sha(__file__),
        'protocol_sha256': sha(root/'MUSIC8K_LONG_ACQUISITION_PROTOCOL_EN.md'),
        'catalog_sha256': sha(catalog_path), 'metadata_sha256': {p.name:sha(p) for p in metadata.iterdir() if p.is_file()},
        'ffmpeg_version': ffmpeg_version, 'numpy_version': np.__version__,
        'role': 'acquisition_probe_excluded_from_future_formal500', 'seed': SEED,
        'ranked_candidates': ranked, 'selected': selected, 'classifier_or_features': False}
    write_new(output/'contract.json', contract)
    cmd = ['hf', 'download', 'homura23/MUSIC8K', *(r['path'] for r in selected),
           '--repo-type', 'dataset', '--revision', REVISION, '--local-dir', str(output/'raw'), '--max-workers', '2']
    with (output/'download.log').open('x') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = []
    for row in selected:
        path = output/'raw'/row['path']
        result = dict(row)
        try:
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('Downloaded file LFS hash/size mismatch')
            result.update(decode(path, output/(row['id']+'_decode.log')))
            result['status'] = 'passed'
        except Exception as e:
            result.update(status='failed', error=str(e), eligible_native60=False)
        records.append(result)
        write_new(output/(row['id']+'_receipt.json'), result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('command', 'path', 'rank')}), flush=True)
    summary = {'status': 'completed', 'contract_sha256': sha(output/'contract.json'),
               'selected': 12, 'passed': sum(r['status']=='passed' for r in records),
               'eligible_native60': sum(r['eligible_native60'] for r in records),
               'downloaded_bytes': sum(r['bytes'] for r in records),
               'classifiers_fitted': 0, 'neural_inference': False,
               'receipts_sha256': {p.name:sha(p) for p in output.glob('*_receipt.json')}}
    write_new(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
