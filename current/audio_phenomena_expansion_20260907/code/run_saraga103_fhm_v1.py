#!/usr/bin/env python3
"""Frozen native-60s F/H/M measurement only for the accepted Saraga103 cohort."""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
RD = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
INTERVALS = RD / 'saraga_external103_intervals_v1'
OUTPUT = ROOT / 'results/measurements_saraga103_fhm_v1'
ROLE = 'external_human_unscored'
SOURCE = 'human_saraga_hindustani_v1'
REFERENCE_HASH = '7051466edec749af6b2563821eb78cad77b2f444aa8f2b97bfa9dd659e92af1e'
SELECTION_SHA = '270d14dc10cc5f16811f799197f6848a2ff8ee213a99d85e890b52b79c1ed56b'
INTERVAL_SHA = '0cf1e78ccdb29e63c421a46bae48e2da76ef0f3116aad442a3a3d91d6e044440'
COMMIT_SHA = '8cf0c7bc743441f699a74ea7d9095634ab08f81004e8a7e11349046fc2e3e820'
ID_SHA = 'a0df862441fb0cf9f214d739a6fca55dc54626457fdd410810aeb1690d35321a'
STAGE = 'saraga103_native60_fhm_measurement_only'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def seal(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def publish(path, value):
    """Atomic no-replace hardlink; identical existing receipts are revalidated."""
    path = Path(path)
    content = canonical(value) + b'\n'
    if path.exists():
        require(path.read_bytes() == content, 'Existing immutable receipt differs')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.publish-', delete=False) as f:
        temporary = Path(f.name)
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def reference():
    p = ROOT / 'results/features_60s_v1/contract.json'
    ref = read(p)
    require(ref['contract_hash'] == REFERENCE_HASH == seal({k:v for k,v in ref.items() if k != 'contract_hash'}),
            'Historical feature contract changed')
    for name, digest in ref['code_sha256'].items():
        require(sha(ROOT/'code'/name) == digest, 'Historical numerical code changed: '+name)
    runtime = {'python': platform.python_version(), **{n: importlib.metadata.version(n)
               for n in ('numpy', 'scipy', 'soundfile', 'librosa')}}
    require(runtime == ref['runtime'], 'Historical measurement runtime mismatch')
    return ref


def validate_native_row(row, proof, selected):
    q = proof['interval_proof']
    iid = proof['item_id']
    require(row['id'] == iid == selected['item_id'], 'Identity mismatch')
    require(row['role'] == proof['role'] == selected['role'] == ROLE, 'Role mismatch')
    require(row['label'] == selected['label'] == '0', 'Label mismatch')
    require(row['source_id'] == row['source_group'] == selected['source_id'] == SOURCE, 'Source mismatch')
    require(row['group_id'] == selected['group_id'], 'Group mismatch')
    require(row['classifier_admission_authorized'] == row['evaluation_allowed'] == 'False'
            and proof['classifier_admission'] is False, 'Classifier admission forbidden')
    require(proof['status'] == 'passed_interval_materialization_not_classifier_admission', 'Unaccepted interval')
    require(row['audio_path'] == q['source_audio_path'] == selected['source_audio_path'], 'Native source path mismatch')
    require(Path(row['audio_path']) == RD/'saraga_hindustani_physical_v1/raw'/(q['mbid']+'.mp3'), 'Noncanonical source')
    require(row['source_audio_sha256'] == row['registered_raw_sha256'] == q['raw_hashes_after']['sha256']
            == selected['acquisition_raw_sha256'], 'Source hash provenance mismatch')
    sr = q['native_sample_rate_hz']
    require(sr in (44100, 48000) and q['native_channels'] == 2, 'Native format mismatch')
    for field, value in (('native_sample_rate_hz', sr), ('crop_start_frame', q['crop_start_frame']),
                         ('crop_frames', sr*60), ('crop_end_frame_exclusive', q['crop_end_frame_exclusive']),
                         ('sf_header_frames', q['observed_header']['header_frames']),
                         ('sf_actual_read_frames', q['observed_actual_frames'])):
        require(row[field] == str(value), 'Native metadata mismatch: '+field)
    require(int(row['crop_start_frame']) == (q['observed_actual_frames']-sr*60)//2, 'Not exact native center60')
    require(round(float(row['audio_offset_s'])*sr) == int(row['crop_start_frame']), 'Offset loses integer crop')
    require(row['native_crop_float64_sha256'] == q['native_float64_sha256'], 'Crop PCM provenance mismatch')
    require(q['real_empty_read_observed'] and q['seek_proof']['exact_bytes_equal'], 'Missing EOF/seek proof')


def build_contract(workers):
    require(isinstance(workers, int) and 1 <= workers <= 8, 'Workers must be 1..8')
    bindings = {}
    def bind(p, expected=None):
        p = Path(p)
        digest = sha(p)
        require(expected is None or digest == expected, 'Bound file changed: '+str(p))
        bindings[str(p)] = digest
        return read(p) if p.suffix == '.json' else None
    selection = bind(ROOT/'preregistration/saraga_external103_selection_frozen_v1.json', SELECTION_SHA)
    interval = bind(ROOT/'preregistration/saraga_external103_measurement_frozen_v1.json', INTERVAL_SHA)
    require(selection['status'] == interval['status'] == 'frozen', 'Unfrozen upstream contracts')
    commit = bind(INTERVALS/'final/COMMIT.json', COMMIT_SHA)
    require(commit['status'] == 'committed', 'Missing interval commit')
    for name, digest in commit['files_sha256'].items():
        require(Path(name).name == name, 'Unsafe committed name')
        bind(INTERVALS/'final'/name, digest)
    summary = read(INTERVALS/'final/materialization_summary.json')
    require(summary['passed'] == summary['expected'] == 103 and summary['failed'] == 0
            and summary['final_passed_source_and_output_hash_recheck'] is True, 'Interval cohort incomplete')
    selection_path = ROOT/selection['selection_draft_directory']/'metadata.csv'
    bind(selection_path, selection['selection_files_sha256']['metadata.csv'])
    selected = {r['item_id']:r for r in rows(selection_path)}
    metadata = INTERVALS/'final/native_metadata_60s.csv'
    native = rows(metadata)
    ids = [r['id'] for r in native]
    require(len(ids) == len(set(ids)) == 103 and set(ids) == set(selected), 'Cohort inventory mismatch')
    require(hashlib.sha256('\n'.join(sorted(i.removeprefix('saraga_hindustani_') for i in ids)).encode()).hexdigest() == ID_SHA,
            'Frozen identity set changed')
    inventory = read(INTERVALS/'final/item_proof_inventory.json')
    require(set(inventory) == set(ids), 'Proof inventory mismatch')
    for row in native:
        p = INTERVALS/'items'/row['id']
        proof = bind(p/'proof.json', inventory[row['id']])
        marker = bind(p/'COMMIT.json')
        require(marker['status'] == 'committed' and marker['files_sha256'] == {
            'proof.json': inventory[row['id']], 'audio.wav': proof['wav']['file_sha256']}, 'Item commit binding mismatch')
        require(proof['contract_sha256'] == summary['contract_sha256'] == interval['contract_sha256'], 'Measurement contract mismatch')
        validate_native_row(row, proof, selected[row['id']])
        bind(row['audio_path'], row['source_audio_sha256'])
        bind(p/'audio.wav', proof['wav']['file_sha256'])
    ref = reference()
    bind(ROOT/'results/features_60s_v1/contract.json')
    for name, digest in ref['code_sha256'].items():
        bind(ROOT/'code'/name, digest)
    bind(Path(__file__).resolve())
    return dict(stage=STAGE, classifier_fitted=False, scores_generated=False,
                neural_inference_performed=False, role=ROLE, selected_ids=ids, rows=103,
                metadata_path=str(metadata), output_path=str(OUTPUT), workers=workers,
                python_executable=sys.executable, runtime=ref['runtime'], reference_contract_hash=REFERENCE_HASH,
                command=[sys.executable,str(ROOT/'code/extract_features.py'),'--metadata',str(metadata),
                         '--output',str(OUTPUT/'features'),'--duration','60','--workers',str(workers)],
                thread_environment={k:os.environ.get(k) for k in
                                    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')},
                input_sha256=bindings)


def verify_native_pcm(native):
    import numpy as np
    import soundfile as sf
    for row in native:
        with sf.SoundFile(row['audio_path']) as f:
            require(f.samplerate == int(row['native_sample_rate_hz']) and f.channels == 2
                    and f.frames == int(row['sf_header_frames']), 'Decoder header mismatch')
            require(f.seek(int(row['crop_start_frame'])) == int(row['crop_start_frame']), 'Seek mismatch')
            x = f.read(int(row['crop_frames']), dtype='float64', always_2d=True)
        require(x.shape == (int(row['crop_frames']), 2) and np.isfinite(x).all(), 'Invalid native crop')
        require(hashlib.sha256(x.astype('<f8', copy=False).tobytes()).hexdigest() == row['native_crop_float64_sha256'],
                'Native seek crop no longer matches accepted sequential decode')


def validate_results(features, metadata, ref):
    contract = read(features/'contract.json')
    require(contract['contract_hash'] == seal({k:v for k,v in contract.items() if k != 'contract_hash'}), 'Feature contract hash mismatch')
    for key in ('duration','preflight_only','input_config','F_config','feature_names','code_sha256','runtime'):
        require(contract[key] == ref[key], 'Historical numerical parity mismatch: '+key)
    native = rows(metadata)
    records = rows(features/'features.csv')
    ids = [r['id'] for r in native]
    require(len(ids) == len(set(ids)) == 103 and contract['selected_ids'] == ids
            and contract['metadata_sha256'] == sha(metadata) and [r['id'] for r in records] == ids, 'Feature identity mismatch')
    names = [n for family in ref['feature_names'].values() for n in family]
    hashes = {}
    for source, record in zip(native, records):
        path = features/'items'/(hashlib.sha256(source['id'].encode()).hexdigest()+'.json')
        item = read(path)
        require(item['input_row_hash'] == seal(source) and item['extraction_contract_hash'] == contract['contract_hash'], 'Item binding mismatch')
        require(item['extraction_status'] == 'ok', 'Extraction failed; keep row and inspect: '+source['id'])
        for key in ('id','role','label','source_id','source_group','group_id'):
            require(item[key] == record[key] == source[key], 'Identity/provenance mismatch: '+key)
        require(item['source_audio_sha256'] == source['source_audio_sha256']
                and item['source_audio_path'] == source['audio_path']
                and item['crop_start_frame'] == int(source['crop_start_frame'])
                and item['crop_frames'] == int(source['native_sample_rate_hz'])*60
                and item['source_total_frames'] == int(source['sf_header_frames'])
                and item['analysis_frames'] == 960000 and item['analysis_sr'] == 16000, 'Native interval mismatch')
        for key in (*names,'F_status','H_status','M_status','extraction_status','source_audio_sha256','analysis_waveform_sha256'):
            value = item.get(key)
            require(record.get(key,'') == ('' if value is None else str(value)), 'CSV/item descriptor mismatch: '+key)
        hashes[str(path)] = sha(path)
    require({str(p) for p in (features/'items').iterdir()} == set(hashes), 'Unexpected item inventory')
    summary, process = read(features/'summary.json'), read(features/'process.json')
    require(summary['expected'] == summary['recorded'] == 103 and summary['complete_accounting'] is True
            and summary['contract_hash'] == contract['contract_hash'] and summary['features_csv_sha256'] == sha(features/'features.csv')
            and process['state'] == 'finished' and process['complete_accounting'] is True, 'Incomplete feature accounting')
    counts = {f+'_status':dict(Counter(r[f+'_status'] for r in records)) for f in ('F','H','M')}
    require(summary['status_counts'] == dict(Counter(r['extraction_status'] for r in records))
            and summary['family_status_counts'] == counts, 'Observability count mismatch')
    for name in ('contract.json','features.csv','summary.json','process.json'):
        hashes[str(features/name)] = sha(features/name)
    return dict(status='passed_measurement_not_classifier_admission', rows=103, role=ROLE,
                classifier_fitted=False, scores_generated=False, family_status_counts=counts,
                all_descriptor_values_checked=True, outputs_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('draft','run'))
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    body = build_contract(args.workers)
    if args.mode == 'draft':
        require(not OUTPUT.exists(), 'Output exists before draft')
        publish(args.receipt, dict(status='draft', authorized_stage=None, contract=body, contract_sha256=seal(body)))
        print(json.dumps({'status':'draft','contract_sha256':seal(body)}), flush=True)
        return
    require(args.receipt_sha256 is not None and sha(args.receipt) == args.receipt_sha256, 'Frozen receipt hash mismatch')
    frozen = read(args.receipt)
    require(frozen['status'] == 'frozen' and frozen['authorized_stage'] == STAGE
            and frozen['contract'] == body and frozen['contract_sha256'] == seal(body), 'Frozen contract mismatch')
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(OUTPUT/'launch_contract.json', frozen)
        ref = reference()
        metadata = Path(body['metadata_path'])
        if not (OUTPUT/'acceptance.json').exists():
            verify_native_pcm(rows(metadata))
            with (OUTPUT/'extraction.log').open('a') as log:
                subprocess.run(body['command'], check=True, stdout=log, stderr=subprocess.STDOUT)
        result = validate_results(OUTPUT/'features', metadata, ref)
        for path, digest in body['input_sha256'].items():
            require(sha(path) == digest, 'Input/code changed during extraction: '+path)
        result.update(launch_contract_sha256=sha(OUTPUT/'launch_contract.json'),
                      extraction_log_sha256=sha(OUTPUT/'extraction.log'), original_sources_rehashed_after_extraction=True)
        publish(OUTPUT/'acceptance.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'outputs_sha256'}), flush=True)


if __name__ == '__main__':
    main()
