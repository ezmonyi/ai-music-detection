#!/usr/bin/env python3
"""Verify exact60 neural outputs before using unchanged September 5 features.

No model fitting or inference. An empty beat file is observed unavailability;
missing/short/mismatched files fail closed. Verification opens all stem samples.
"""
import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from materialize_equal60_inputs_v2 import sha_file, canonical_hash, preserve_json

FRAMES = 2646000
STEMS = ('bass', 'drums', 'other', 'vocals')
RUNTIME_SHA = 'c6581c737fdb584ff763cc88752dc98864d9ea583a3e4cf9dd552a42ce791268'
CHECKPOINT_SHA = 'ec65f4d774d21340b0cf40cd91ab526a92f9545bb5314c440a8db9d6ab618ad4'
OLD_CODE = {
    'extract_expanded_four_family.py': '98fc6caa8b54ed1370269db13fe8d49559d0b877b4b0d5a3a069c8409906c5fe',
    'expanded_feature_definitions.py': '8b9745085c7518e84613b2ba499dbae775a57e4dcf95670c5e86a05ab524ff00',
}
BIAS_SHA = 'bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e'


def require_hash(path, expected):
    actual = sha_file(path)
    if actual != expected:
        raise ValueError(f'Hash mismatch: {path}: {actual} != {expected}')
    return actual


def inspect_audio(path, expected_hash=None, frames=FRAMES, subtype=None):
    before = Path(path).stat()
    with sf.SoundFile(path) as f:
        if (f.samplerate, f.channels, f.frames) != (44100, 2, frames):
            raise ValueError(f'Exact-frame/rate/channel mismatch: {path}')
        if subtype and f.subtype != subtype:
            raise ValueError(f'Subtype mismatch: {path}')
        dtype = f.subtype
        finite = all(np.isfinite(block).all() for block in f.blocks(blocksize=65536, dtype='float32', always_2d=True))
    if not finite:
        raise ValueError(f'Nonfinite audio: {path}')
    digest = sha_file(path)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Audio changed during verification: {path}')
    if expected_hash and digest != expected_hash:
        raise ValueError(f'Audio input hash mismatch: {path}')
    return {'path': str(path), 'sha256': digest, 'bytes': after.st_size,
            'frames': frames, 'sample_rate': 44100, 'channels': 2, 'subtype': dtype, 'finite': True}


def inspect_beats(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Missing beat file: {path}')
    if not path.read_text().strip():
        return {'sha256': sha_file(path), 'status': 'empty_unavailable', 'beat_count': 0}
    data = np.loadtxt(path, ndmin=2)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError(f'Malformed beat data: {path}')
    if (np.diff(data[:, 0]) <= 0).any() or (data[:, 0] < 0).any() or (data[:, 0] > 60).any():
        raise ValueError(f'Beat time outside exact60 or unordered: {path}')
    if (data[:, 1] < 1).any() or not np.equal(data[:, 1], np.round(data[:, 1])).all():
        raise ValueError(f'Invalid beat positions: {path}')
    return {'sha256': sha_file(path), 'status': 'nonempty', 'beat_count': len(data)}


def inspect_structure(path, audio_path):
    payload = json.loads(Path(path).read_text())
    if payload.get('path') != str(audio_path):
        raise ValueError(f'Structure input provenance mismatch: {path}')
    segments = payload.get('segments', [])
    if not segments:
        raise ValueError('No structure segments')
    previous = 0.0
    for seg in segments:
        start, end = float(seg['start']), float(seg['end'])
        if not np.isfinite([start, end]).all() or start != previous or end <= start or end > 60:
            raise ValueError(f'Noncontiguous/out-of-context structure: {path}')
        previous = end
    if abs(previous - 60) > 1e-8:
        raise ValueError(f'Structure does not span exact60: {path}')
    return {'sha256': sha_file(path), 'segment_count': len(segments), 'input_path': payload['path']}


def verify_runtime(runtime_path, checkpoint_path, old_code_root, bias):
    require_hash(runtime_path, RUNTIME_SHA)
    require_hash(checkpoint_path, CHECKPOINT_SHA)
    runtime = json.loads(Path(runtime_path).read_text())
    checkpoints = json.loads(Path(checkpoint_path).read_text())
    checked = {row['path']: require_hash(row['path'], row['sha256']) for row in checkpoints['checkpoints']}
    for package, relative, key in (
        ('allin1_infer', 'cli.py', 'allin1_cli_sha256'),
        ('allin1_infer', 'stems.py', 'allin1_stems_sha256'),
        ('demucs_infer', 'apply.py', 'demucs_apply_sha256'),
    ):
        spec = importlib.util.find_spec(package)
        if not spec or not spec.origin:
            raise ValueError(f'Missing inference package: {package}')
        path = Path(spec.origin).parent / relative
        checked[str(path)] = require_hash(path, runtime['all_in_one'][key])
    for name, digest in OLD_CODE.items():
        checked[str(old_code_root / name)] = require_hash(old_code_root / name, digest)
    checked[str(bias)] = require_hash(bias, BIAS_SHA)
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--inference-root', type=Path, required=True)
    p.add_argument('--old-code-root', type=Path, required=True)
    p.add_argument('--runtime-contract', type=Path, required=True)
    p.add_argument('--checkpoint-manifest', type=Path, required=True)
    p.add_argument('--bias', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--extract', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    summary = json.loads((a.prepared_dir / 'materialization_summary.json').read_text())
    contract = json.loads((a.prepared_dir / 'materialization_contract.json').read_text())
    contract_hash = contract.pop('contract_sha256')
    if canonical_hash(contract) != contract_hash or summary['contract_sha256'] != contract_hash:
        raise ValueError('Materialization contract mismatch')
    manifest = a.prepared_dir / 'inference_manifest.csv'
    require_hash(manifest, summary['manifest_sha256'])
    rows = list(csv.DictReader(manifest.open()))
    if len(rows) != summary['rows'] or [r['item_id'] for r in rows] != contract['item_ids']:
        raise ValueError('Prepared row identity/order mismatch')
    runtime_checks = verify_runtime(a.runtime_contract, a.checkpoint_manifest, a.old_code_root, a.bias)
    items = []
    for row in rows:
        item_id = row['item_id']
        if row['role'] != 'development' or float(row['duration']) != 60 or float(row['audio_offset_s']) != 0:
            raise ValueError('Role/context mismatch')
        if row['contract_sha256'] != contract_hash:
            raise ValueError('Per-item contract mismatch')
        item = {'item_id': item_id, 'input': inspect_audio(row['standardized_path'], row['standardized_file_sha256'], subtype='FLOAT'),
                'stems': {stem: inspect_audio(a.inference_root / 'demix/htdemucs' / item_id / f'{stem}.wav') for stem in STEMS}}
        item['beats'] = inspect_beats(a.inference_root / 'beats' / f'{item_id}.beats')
        item['structure'] = inspect_structure(a.inference_root / 'structure' / f'{item_id}.json', row['standardized_path'])
        spec_path = a.inference_root / 'spec' / f'{item_id}.npy'
        spectrogram = np.load(spec_path, mmap_mode='r', allow_pickle=False)
        if spectrogram.shape != (4, 6000, 81) or not np.isfinite(spectrogram).all():
            raise ValueError(f'Exact60 spectrogram shape/finite mismatch: {spec_path}')
        item['spectrogram'] = {'sha256': sha_file(spec_path), 'shape': list(spectrogram.shape), 'dtype': str(spectrogram.dtype)}
        items.append(item)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {'status': 'passed', 'code_sha256': sha_file(__file__), 'rows': len(rows),
             'materialization_contract_sha256': contract_hash, 'manifest_sha256': sha_file(manifest),
             'neural_reproducibility': 'exact_published_outputs_not_bitwise_unseeded_demucs_regeneration',
             'runtime_code_checkpoint_bias_checks': runtime_checks, 'items': items}
    preserve_json(a.output_dir / 'strict_inference_audit.json', audit)
    if a.extract:
        cmd = [sys.executable, str(a.old_code_root / 'extract_expanded_four_family.py'),
               '--manifest', str(manifest), '--bias', str(a.bias), '--duration', '60',
               '--demix-root', str(a.inference_root / 'demix'), '--beat-root', str(a.inference_root / 'beats'),
               '--structure-root', str(a.inference_root / 'structure'), '--output-dir', str(a.output_dir / 'features'),
               '--workers', str(a.workers), '--strict']
        with (a.output_dir / 'extraction.log').open('a') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        features = a.output_dir / 'features/expanded_features_60s.csv'
        records = list(csv.DictReader(features.open()))
        if [r['item_id'] for r in records] != [r['item_id'] for r in rows] or any(r['status'] != 'complete' for r in records):
            raise ValueError('Incomplete or misaligned extracted rows')
        by_id = {r['item_id']: r for r in rows}
        audit_by_id = {r['item_id']: r for r in items}
        for record in records:
            hashes = json.loads(record['input_hashes'])
            verified = audit_by_id[record['item_id']]
            expected = {'source_audio_sha256': by_id[record['item_id']]['standardized_file_sha256'],
                        **{stem + '_sha256': verified['stems'][stem]['sha256'] for stem in STEMS},
                        'beats_sha256': verified['beats']['sha256'], 'structure_sha256': verified['structure']['sha256']}
            if hashes != expected:
                raise ValueError('Extractor opened changed/unverified inputs')
        preserve_json(a.output_dir / 'extraction_receipt.json', {
            'status': 'passed', 'rows': len(records), 'command': cmd,
            'strict_inference_audit_sha256': sha_file(a.output_dir / 'strict_inference_audit.json'),
            'features_sha256': sha_file(features),
            'availability': {key: dict(Counter(r[key] for r in records)) for key in ('s8_computed', 'd_eligible', 'r_eligible', 'p_eligible')},
            'no_short_padding_possible': True, 'classifier_fitted': False,
        })
    print(json.dumps({'status': 'passed', 'rows': len(rows), 'extraction_requested': a.extract}))


if __name__ == '__main__':
    main()
