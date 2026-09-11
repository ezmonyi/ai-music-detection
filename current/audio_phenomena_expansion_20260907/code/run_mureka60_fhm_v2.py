#!/usr/bin/env python3
"""Measure the unchanged F/H/M chains on all500 interval-verified native crops.

No training, classifier scoring, replacement or family-availability filtering.
"""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import prepare_mureka60_inputs_v2 as prep
import run_mureka60_inference_v2 as runner

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_SHA = '7051466edec749af6b2563821eb78cad77b2f444aa8f2b97bfa9dd659e92af1e'


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def code_and_reference():
    reference_path = ROOT/'results/features_60s_v1/contract.json'
    reference = prep.read_json(reference_path)
    body = {k:v for k,v in reference.items() if k != 'contract_hash'}
    prep.require(reference['contract_hash'] == REFERENCE_SHA == prep.canonical_hash(body),
                 'Historical60 feature contract changed')
    for name, digest in reference['code_sha256'].items():
        prep.require(prep.sha_file(ROOT/'code'/name) == digest, 'Frozen FHM code changed: '+name)
    current = {'python':platform.python_version(), **{name:importlib.metadata.version(name)
               for name in ('numpy','scipy','soundfile','librosa')}}
    prep.require(current == reference['runtime'], 'FHM runtime differs from historical60 before launch')
    return reference, {str(reference_path):prep.sha_file(reference_path),
                       **{str(ROOT/'code'/n):h for n,h in reference['code_sha256'].items()},
                       str(Path(__file__).resolve()):prep.sha_file(__file__)}


def validate_results(features, metadata, reference):
    """Reconcile every scalar descriptor and exact interval, not only row counts."""
    contract = prep.read_json(features/'contract.json')
    prep.require(prep.canonical_hash({k:v for k,v in contract.items() if k!='contract_hash'})
                 == contract['contract_hash'], 'FHM contract canonical hash changed')
    for key in ('duration', 'preflight_only', 'input_config', 'F_config', 'feature_names', 'code_sha256', 'runtime'):
        prep.require(contract[key] == reference[key], 'FHM numerical/runtime parity failed: '+key)
    rows = read_csv(metadata)
    ids = [r['id'] for r in rows]
    prep.require(len(ids) == len(set(ids)) == 500 and contract['selected_ids'] == ids
                 and contract['metadata_sha256'] == prep.sha_file(metadata), 'All500 native metadata identity changed')
    records = read_csv(features/'features.csv')
    prep.require([r['id'] for r in records] == ids, 'Feature rows missing/reordered/duplicated')
    names = [n for family in reference['feature_names'].values() for n in family]
    hashes = {}
    for source, record in zip(rows, records):
        path = features/'items'/(hashlib.sha256(source['id'].encode()).hexdigest()+'.json')
        item = prep.read_json(path)
        prep.require(item['input_row_hash'] == prep.canonical_hash(source)
                     and item['extraction_contract_hash'] == contract['contract_hash'], 'Item contract mismatch')
        prep.require(item['extraction_status'] == 'ok', 'Processing failure retained; inspect before acceptance: '+source['id'])
        prep.require(item['source_audio_sha256'] == source['source_audio_sha256']
                     and item['source_audio_path'] == source['audio_path']
                     and item['crop_start_frame'] == int(source['crop_start_frame'])
                     and item['crop_frames'] == prep.FRAMES
                     and item['source_total_frames'] == int(source['sf_header_frames'])
                     and item['analysis_frames'] == 960000 and item['analysis_sr'] == 16000
                     and item['role'] == prep.ROLE and item['group_id'] == source['group_id'],
                     'Native analysis interval/source/role mismatch')
        for key in (*names, 'F_status', 'H_status', 'M_status', 'extraction_status',
                    'source_audio_sha256', 'analysis_waveform_sha256'):
            value = item.get(key)
            prep.require(record.get(key, '') == ('' if value is None else str(value)),
                         'Descriptor CSV/item mismatch: '+source['id']+':'+key)
        hashes[str(path)] = prep.sha_file(path)
    expected_items = set(hashes)
    prep.require({str(p) for p in (features/'items').iterdir()} == expected_items, 'Unexpected item inventory')
    summary = prep.read_json(features/'summary.json')
    process = prep.read_json(features/'process.json')
    prep.require(summary['expected'] == summary['recorded'] == 500 and summary['complete_accounting'] is True
                 and summary['contract_hash'] == contract['contract_hash']
                 and summary['features_csv_sha256'] == prep.sha_file(features/'features.csv')
                 and process['state'] == 'finished' and process['complete_accounting'] is True,
                 'FHM completion summary mismatch')
    prep.require(summary['status_counts'] == dict(Counter(r['extraction_status'] for r in records))
                 and summary['family_status_counts'] == {f+'_status':dict(Counter(r[f+'_status'] for r in records))
                                                        for f in ('F','H','M')}, 'FHM summary observability mismatch')
    for name in ('contract.json', 'features.csv', 'summary.json', 'process.json'):
        hashes[str(features/name)] = prep.sha_file(features/name)
    return dict(status='passed', rows=500, classifier_fitted=False, scores_generated=False,
                role=prep.ROLE, all_descriptor_values_checked=True,
                source_total_frames_meaning='observed_soundfile_header_not_actual_eof',
                family_status_counts={f:dict(Counter(r[f+'_status'] for r in records)) for f in ('F','H','M')},
                outputs_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    prep.require(args.workers > 0, 'Positive worker count required')
    output = ROOT/'results/measurements_mureka60_fhm_v2'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('Validating all500 prepared native intervals and FLOAT views', flush=True)
        rows, shards, bindings = runner.validate_prepared(prep.OUTPUT, decode=True)
        reference, code_hashes = code_and_reference()
        bindings.update(code_hashes)
        bindings.update(runner.dependency_snapshot())
        metadata = prep.OUTPUT/'native_metadata_60s.csv'
        features = output/'features'
        command = [sys.executable, str(ROOT/'code/extract_features.py'), '--metadata',str(metadata),
                   '--output',str(features),'--duration','60','--workers',str(args.workers)]
        contract = dict(purpose='measurement_only', role=prep.ROLE, classifier_fitted=False,
                        scores_generated=False, rows=500, command=command, input_sha256=bindings)
        prep.publish(output/'launch_contract.json', prep.json_bytes(contract))
        if (output/'acceptance.json').exists():
            result = validate_results(features, metadata, reference)
            result.update(launch_contract_sha256=prep.sha_file(output/'launch_contract.json'),
                          extraction_log_sha256=prep.sha_file(output/'extraction.log'),
                          original_sources_rehashed_after_extraction=True)
            prep.require(prep.read_json(output/'acceptance.json') == result, 'Accepted FHM artifacts changed')
            print('Existing all500 FHM acceptance revalidated; no rerun', flush=True)
            return
        # Underlying extractor has its own contract and lock for validated resume.
        print('Starting unchanged native60 FHM extraction; no fitting or scoring', flush=True)
        with (output/'extraction.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = validate_results(features, metadata, reference)
        for path, digest in bindings.items():
            prep.require(prep.sha_file(path) == digest, 'Input/code evidence changed during extraction: '+path)
        # Rehash original source files via the unchanged acquisition evidence.
        prep.validate_contract(prep.read_json(prep.OUTPUT/'materialization_contract.json'), prep.SOURCE, prep.OUTPUT)
        result.update(launch_contract_sha256=prep.sha_file(output/'launch_contract.json'),
                      extraction_log_sha256=prep.sha_file(output/'extraction.log'),
                      original_sources_rehashed_after_extraction=True)
        prep.publish(output/'acceptance.json', prep.json_bytes(result))
        print(json.dumps({k:v for k,v in result.items() if k!='outputs_sha256'}), flush=True)


if __name__ == '__main__':
    main()
