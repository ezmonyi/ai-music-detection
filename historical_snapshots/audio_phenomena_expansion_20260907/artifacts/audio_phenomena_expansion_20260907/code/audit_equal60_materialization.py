#!/usr/bin/env python3
"""Independent full-file hash/property audit of the exact60 development views."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf

NATIVE_SHA = '00ff8671c589ba280f2fc2a5bf3019f2d907fdf7a8933afa573f433112a8ee1b'
FHM_SHA = '46ea2a81d4653fa8d830e7f31a858ad75999105e8b6ce813032862743c7e51fa'
FHM_CONTRACT = '7051466edec749af6b2563821eb78cad77b2f444aa8f2b97bfa9dd659e92af1e'
PREPARED_SHA = 'f52b93bf4ff820bd35845749b91a2c7139b901588ebde2ebf62f4442f583d3bf'
MATERIALIZER_SHA = 'ab2a82f3dd2cf3f30786b2c680e33e2665ca3da8f3c2fab46fa5659019868aba'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_identity(row, native, reference, contract_hash):
    item_id = row['item_id']
    require(item_id == native['id'] == reference['id'], 'ID mismatch')
    require(row['role'] == native['role'] == reference['role'] == 'development', 'Role mismatch')
    for field in ('label', 'group_id'):
        require(str(row[field]) == str(native[field]) == str(reference[field]), field + ' mismatch')
    require(row['source_id'] == native['source_group'] == reference['source_group'], 'Source group mismatch')
    require(row['input_row_sha256'] == canonical(native), 'Native row digest mismatch')
    require(row['contract_sha256'] == contract_hash, 'Materialization contract mismatch')
    require(reference['extraction_contract_hash'] == FHM_CONTRACT and reference['extraction_status'] == 'ok', 'FHM contract/status mismatch')
    require(float(reference['requested_duration_sec']) == float(row['duration']) == 60, 'Wrong analysis duration')
    require(float(row['audio_offset_s']) == 0 and int(row['requires_crop']) == 0, 'Prepared view must begin at zero')
    require(row['source_audio_path'] == native['audio_path'] == reference['source_audio_path'], 'Native path mismatch')
    require(row['source_audio_sha256'] == reference['source_audio_sha256'], 'Native source hash mismatch')
    for field, native_field in (('source_sample_rate', 'physical_sample_rate_hz'),
                                ('source_channels', 'physical_channels'),
                                ('source_total_frames', 'physical_frames'),
                                ('crop_start_frame', 'crop_start_frame'), ('crop_frames', 'crop_frames')):
        require(int(row[field]) == int(native[native_field]) == int(reference[field]), field + ' mismatch')
    require(int(row['crop_frames']) == 60 * int(row['source_sample_rate']), 'Native count not exact60')
    require(int(native['crop_end_frame_exclusive']) == int(row['crop_start_frame']) + int(row['crop_frames']), 'Native end mismatch')
    require(int(row['standardized_frames']) == 2646000 and int(row['standardized_sr']) == 44100 and int(row['standardized_channels']) == 2, 'Prepared dimensions mismatch')


def inspect_one(row, native, reference, prepared, contract_hash):
    validate_identity(row, native, reference, contract_hash)
    receipt = json.loads((prepared / 'items' / (row['item_id'] + '.json')).read_text())
    require(set(receipt) == set(row), 'Receipt/CSV field mismatch')
    require(all(str(receipt[k]) == row[k] for k in row), 'Receipt/CSV value mismatch')
    path = Path(row['standardized_path'])
    require(path.parent == prepared / 'audio' and path.name == row['item_id'] + '.wav', 'Unexpected prepared output path')
    before = path.stat()
    info = sf.info(path)
    require((info.frames, info.samplerate, info.channels, info.format, info.subtype) == (2646000, 44100, 2, 'WAV', 'FLOAT'), 'Physical WAV properties mismatch')
    require(before.st_size == int(row['standardized_file_bytes']), 'File size mismatch')
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Output changed during audit')
    require(digest == row['standardized_file_sha256'], 'Prepared file SHA mismatch')
    return {'item_id': row['item_id'], 'source_group': row['source_id'], 'label': row['label'],
            'group_id': row['group_id'], 'standardized_file_sha256': digest,
            'standardized_bytes': before.st_size, 'status': 'passed'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--native-manifest', type=Path, required=True)
    p.add_argument('--fhm-features', type=Path, required=True)
    p.add_argument('--materializer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    require(not a.output.exists(), 'Existing completed audit must be preserved')
    for path, expected in ((a.native_manifest, NATIVE_SHA), (a.fhm_features, FHM_SHA),
                           (a.prepared_dir/'inference_manifest.csv', PREPARED_SHA), (a.materializer, MATERIALIZER_SHA)):
        require(sha(path) == expected, 'Frozen input/code hash mismatch: ' + str(path))
    native = {r['id']: r for r in csv.DictReader(a.native_manifest.open())}
    references = {r['id']: r for r in csv.DictReader(a.fhm_features.open())}
    rows = list(csv.DictReader((a.prepared_dir/'inference_manifest.csv').open()))
    selected_ids = {k for k, v in native.items() if v['role'] == 'development'}
    require(len(rows) == len(selected_ids) == 1604 and {r['item_id'] for r in rows} == selected_ids, 'Development population mismatch')
    summary = json.loads((a.prepared_dir/'materialization_summary.json').read_text())
    contract = json.loads((a.prepared_dir/'materialization_contract.json').read_text())
    contract_hash = contract.pop('contract_sha256')
    require(canonical(contract) == contract_hash == summary['contract_sha256'], 'Contract digest mismatch')
    require(contract['item_ids'] == [r['item_id'] for r in rows], 'Contract ID order mismatch')
    require(contract['code_sha256'] == MATERIALIZER_SHA, 'Contract materializer mismatch')
    require(summary['manifest_sha256'] == PREPARED_SHA, 'Summary manifest mismatch')
    expected_paths = [r['standardized_path'] for r in rows]
    shard_paths = []
    for shard in summary['shards']:
        path = a.prepared_dir / f"inference_shard_{shard['index']:02d}.txt"
        require(sha(path) == shard['sha256'], 'Shard hash mismatch')
        paths = path.read_text().splitlines()
        require(len(paths) == shard['rows'], 'Shard count mismatch')
        shard_paths.extend(paths)
    require(shard_paths == expected_paths, 'Shard partition has duplicates/omissions/reordering')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        records = list(pool.map(lambda r: inspect_one(r, native[r['item_id']], references[r['item_id']], a.prepared_dir, contract_hash), rows))
    output = {'status': 'passed', 'rows': len(records), 'materialization_contract_sha256': contract_hash,
              'prepared_manifest_sha256': PREPARED_SHA, 'native_manifest_sha256': NATIVE_SHA,
              'fhm_feature_sha256': FHM_SHA, 'auditor_sha256': sha(__file__),
              'label_counts': dict(Counter(r['label'] for r in records)),
              'source_counts': dict(Counter(r['source_group'] for r in records)),
              'global_groups': len({r['group_id'] for r in records}),
              'shards': len(summary['shards']), 'output_bytes_rehashed': sum(r['standardized_bytes'] for r in records),
              'native_hash_policy': 'FHM frozen source hashes matched to materialization-time full-source hashes; native originals not rehashed again by this audit',
              'all_output_files_rehashed': True, 'old_features_inferred': False, 'records': records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output, indent=2) + '\n')
    temporary.replace(a.output)
    print(json.dumps({k:v for k,v in output.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
