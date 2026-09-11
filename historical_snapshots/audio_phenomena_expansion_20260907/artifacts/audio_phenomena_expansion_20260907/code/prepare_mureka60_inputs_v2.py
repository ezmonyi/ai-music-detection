#!/usr/bin/env python3
"""Version 2: prove each acquisition-center interval using true sequential EOF.

Standalone; preserves v1, raw MP3s and all acquisition/evidence artifacts.
No full-stream decoder-equality claim, inference or classifier admission.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import soundfile as sf

SOURCE = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/music8k_mureka500_v1')
OUTPUT = SOURCE.parent / 'mureka60_inputs_v2'
ACQUISITION_SHA = 'e985adc956116851a7358f3ed9279241a6f92b65d28f22b0eb40e94023de8d24'
REVISION = '05232438ba76a7bc55cbebfdc6d5f4011c980bba'
SEED = 'music8k-mureka-v9-formal500-20260907'
ROLE = 'external_generator_unscored'
SR, FRAMES, COUNT = 44100, 2646000, 500
AUDIT_ROOT = Path(__file__).resolve().parent.parent / 'audit'
EVIDENCE = {
    'mureka500_decoder_frames_v1.json': '1efc470a2346eac5514646cbf4717dc62449f0d4021d0e0f212228d770221ad8',
    'mureka97136_decoder_diagnosis_v3.json': 'ed6c87d33b3cf442d16b3e9f4b9bea1301aefd3f3324ca038923d4d362868d85',
}
KNOWN_HEADER_MISMATCHES = {'97136': 9581607}
BLOCK = 65536
CLAIM = 'acquisition_center_interval_verified_not_full_stream_decoder_agreement'
CONFIG = dict(sample_rate_hz=SR, channels=2, crop_frames=FRAMES,
              crop_rule='(native_frames - 2646000) // 2', duration_sec=60,
              decoder='soundfile', crop_dtype='float64_then_float32',
              admission_claim=CLAIM, actual_eof='SoundFile.read_until_empty',
              full_decode_block_frames=BLOCK, seek_agreement='exact_float64_array_equal',
              known_header_mismatch_ids=['97136'], full_stream_decoder_equality_required=False,
              output_format='WAV', output_subtype='FLOAT',
              resampling=False, padding=False, gain=False, dc_removal=False,
              limiting=False, shard_size=24, role=ROLE,
              classifier_admission_authorized=False, selection_features_used=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(path, body):
    """Atomic hard-link publication cannot replace an existing path, even in a race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Symlink publication forbidden')
    if path.exists():
        require(path.read_bytes() == body, f'Conflicting immutable publication: {path}')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.preparing-', delete=False) as f:
        temporary = Path(f.name)
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(not path.is_symlink() and path.read_bytes() == body, f'Publication conflict: {path}')
    finally:
        temporary.unlink()


def runtime():
    # libsndfile and its loaded codec libraries are bound in addition to versions.
    libraries = set()
    maps = Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            candidate = line.split()[-1]
            if candidate.startswith('/') and any(n in Path(candidate).name for n in ('sndfile', 'mpg123', 'mp3lame')):
                libraries.add(candidate)
    libname = str(getattr(sf, '_libname', ''))
    if Path(libname).is_file():
        libraries.add(str(Path(libname).resolve()))
    return dict(python=platform.python_version(), executable=str(Path(sys.executable).resolve()),
                executable_sha256=sha_file(sys.executable), platform=platform.platform(),
                numpy=np.__version__, soundfile=sf.__version__, libsndfile=sf.__libsndfile_version__,
                soundfile_module_sha256=sha_file(sf.__file__),
                decoder_libraries={p: sha_file(p) for p in sorted(libraries)})


def selected_rows(contract):
    require(contract['repo'] == 'homura23/MUSIC8K' and contract['revision'] == REVISION,
            'Wrong source or revision')
    require(contract['status'] == 'frozen_for_acquisition_only' and contract['classifier_authorized'] is False,
            'Acquisition role/admission changed')
    require(contract['seed'] == SEED and contract['selected_count'] == COUNT, 'Selection configuration changed')
    ranked, selected, probes = contract['ranked_candidates'], contract['selected'], contract['excluded_probe_ids']
    ids = [r['id'] for r in ranked]
    require(len(ids) == 650 and len(set(ids)) == 650 and all(i.isdigit() for i in ids), 'Candidate population changed')
    require(len(probes) == 12 and len(set(probes)) == 12 and not set(ids).intersection(probes), 'Probe exclusions changed')
    require(ids == sorted(ids, key=lambda i: hashlib.sha256((SEED + '|' + i).encode()).hexdigest()), 'Ranking changed')
    require(len(selected) == COUNT and selected == ranked[:COUNT], 'Frozen selected list changed')
    for row in ranked:
        i = row['id']
        require(row['rank'] == hashlib.sha256((SEED + '|' + i).encode()).hexdigest(), 'Rank digest changed')
        require(row['role'] == 'reserved_unscored' and row['source'] == 'Mureka v9', 'Source/role changed')
        require(row['path'] == f'mureka_v9/{i}.mp3' and row['reference_group_id'] == f'music8k_reference_{i}', 'Identity/path changed')
    return selected


def native_source(source, row):
    path = source / 'raw' / row['path']
    require(not path.is_symlink() and path.resolve() == path, 'Source symlink/path changed')
    require(path.stat().st_size == row['bytes'] and sha_file(path) == row['sha256'], 'Native bytes/hash mismatch')
    return path


def waveform_hash(data, dtype):
    return hashlib.sha256(np.ascontiguousarray(data, dtype=dtype).tobytes()).hexdigest()


def reviewed_headers(rows):
    """Bind the preserved survey and the independently diagnosed exact source."""
    bindings = {}
    for name, expected in EVIDENCE.items():
        path = AUDIT_ROOT / name
        require(path.is_file() and not path.is_symlink() and sha_file(path) == expected,
                f'Decoder evidence missing or changed: {name}')
        bindings[str(path)] = expected
    survey = read_json(AUDIT_ROOT / 'mureka500_decoder_frames_v1.json')
    diagnosis = read_json(AUDIT_ROOT / 'mureka97136_decoder_diagnosis_v3.json')
    require(survey['count'] == COUNT and survey['source_contract_sha256'] == ACQUISITION_SHA
            and survey['changed_admission'] is False, 'Survey contract/count changed')
    records = survey['records']
    require([r['id'] for r in records] == [r['id'] for r in rows], 'Survey full500 identity/order changed')
    require(len({r['id'] for r in records}) == COUNT, 'Duplicate survey ID')
    headers = {}
    for row, record in zip(rows, records):
        require(record['source_expected_sha256'] == row['sha256']
                and record['ffmpeg_decoded_frames'] == row['native_frames']
                and record['sf_sample_rate'] == record['ffmpeg_sample_rate'] == SR
                and record['sf_channels'] == record['ffmpeg_channels'] == 2, 'Survey source/format changed')
        headers[row['id']] = record['sf_header_frames']
    mismatches = {r['id']: headers[r['id']] for r in rows if headers[r['id']] != r['native_frames']}
    require(mismatches == KNOWN_HEADER_MISMATCHES and survey['header_mismatch_count'] == len(mismatches),
            'Unknown or changed header mismatch inventory')
    diagnosed = next(r for r in rows if r['id'] == '97136')
    require(diagnosis['status'] == 'read_only_diagnosis' and diagnosis['changed_admission'] is False
            and diagnosis['source_and_receipt_unchanged'] is True
            and diagnosis['source_sha256'] == diagnosed['sha256']
            and diagnosis['acquisition_frames'] == diagnosed['native_frames']
            and diagnosis['sf_header']['frames'] == headers['97136'], 'Exact-item diagnosis identity changed')
    return headers, bindings


def verify_interval(source, row, expected_header):
    """Same full sequential/direct-seek rule for every selected item; no blocks()."""
    path = native_source(source, row)
    before = sha_file(path)
    start, end = row['crop_start_frame'], row['crop_start_frame'] + FRAMES
    require(start == (row['native_frames'] - FRAMES)//2 and start >= 0 and row['crop_frames'] == FRAMES,
            'Acquisition center coordinates changed')
    total, crop_parts = 0, []
    full64, full32 = hashlib.sha256(), hashlib.sha256()
    with sf.SoundFile(path) as audio:
        header = audio.frames
        require((audio.samplerate, audio.channels) == (SR, 2), 'Decoder rate/channels mismatch')
        require(header == expected_header, 'Unknown or changed header mismatch inventory')
        while True:
            data = audio.read(BLOCK, dtype='float64', always_2d=True)
            require(data.ndim == 2 and data.shape[1] == 2 and np.isfinite(data).all(), 'Nonfinite/invalid sequential decode')
            if not len(data):
                break
            require(len(data) <= BLOCK, 'Decoder returned more frames than requested')
            full64.update(np.ascontiguousarray(data, dtype='<f8').tobytes())
            full32.update(np.ascontiguousarray(data, dtype='<f4').tobytes())
            left, right = max(start, total), min(end, total + len(data))
            if right > left:
                crop_parts.append(data[left-total:right-total].copy())
            total += len(data)
    require(total >= end, 'Actual sequential EOF does not contain complete center interval; padding forbidden')
    crop = np.concatenate(crop_parts, axis=0) if crop_parts else np.empty((0, 2), dtype=np.float64)
    require(crop.shape == (FRAMES, 2), 'Incomplete sequential center interval')
    with sf.SoundFile(path) as audio:
        require((audio.samplerate, audio.channels, audio.frames) == (SR, 2, header), 'Source format changed during interval audit')
        require(audio.seek(start) == start, 'Seek position mismatch')
        seek = audio.read(FRAMES, dtype='float64', always_2d=True)
    require(seek.shape == crop.shape and np.isfinite(seek).all() and np.array_equal(seek, crop),
            'Fresh seek does not exactly match sequential float64 center')
    after = sha_file(path)
    require(before == after == row['sha256'], 'Source changed during interval audit')
    return dict(sf_header_frames=header, sf_actual_read_frames=total,
                sf_header_minus_acquisition_frames=header-row['native_frames'],
                sf_actual_minus_acquisition_frames=total-row['native_frames'],
                sf_sequential_float64_sha256=full64.hexdigest(), sf_sequential_float32_sha256=full32.hexdigest(),
                center_float64_sha256=waveform_hash(crop, '<f8'), center_float32_sha256=waveform_hash(crop, '<f4'),
                native_crop_float64_sha256=waveform_hash(crop, '<f8'), native_crop_float32_sha256=waveform_hash(crop, '<f4'),
                source_sha256_before=before, source_sha256_after=after, sequential_all_samples_finite=True,
                center_seek_equals_sequential_float64=True, decoder_admission_claim=CLAIM)


def audit_inputs(source):
    """Reconcile all 500 IDs, source receipts and local/remote audits, then rehash."""
    names = ['contract.json', 'summary.json', 'local_physical_audit_v1.json', 'remote_copy_audit_v1.json']
    for name in names:
        require((source / name).is_file() and not (source / name).is_symlink(), f'Missing regular input: {name}')
    bindings = {name: sha_file(source / name) for name in names}
    require(bindings['contract.json'] == ACQUISITION_SHA, 'Frozen acquisition SHA mismatch')
    c, summary, local, remote = [read_json(source / name) for name in names]
    selected = selected_rows(c)
    require(local['status'] == 'passed' and local['selected'] == COUNT and local['eligible_native60'] == COUNT
            and local['all_original_files_rehashed'] is True and local['selection_independently_reconstructed'] is True
            and local['classifier_admission_authorized'] is False, 'Local physical audit failed/admitted')
    require(local['source_revision'] == REVISION and local['contract_sha256'] == ACQUISITION_SHA
            and local['summary_sha256'] == bindings['summary.json'], 'Local provenance mismatch')
    require(remote['status'] == 'passed' and remote['rows'] == COUNT
            and remote['local_audit_sha256'] == bindings['local_physical_audit_v1.json']
            and remote['remote_root'] == str(source) and remote['all_raw_files_rehashed'] is True
            and remote['all_item_receipts_rehashed'] is True and remote['classifier_fitted'] is False,
            'Missing/pending/incorrect remote copy audit')
    require(summary['status'] == 'completed' and summary['selected'] == summary['passed'] == summary['eligible_native60'] == COUNT
            and summary['contract_sha256'] == ACQUISITION_SHA and summary['classifiers_fitted'] == 0
            and summary['neural_inference'] is False, 'Acquisition summary mismatch')
    ids = [r['id'] for r in selected]
    require([r['id'] for r in local['rows']] == ids, 'Physical audit list/order differs from frozen selected500')
    expected_receipts = {i + '.json' for i in ids}
    require(set(summary['receipts_sha256']) == expected_receipts
            and {p.name for p in (source / 'items').iterdir()} == expected_receipts, 'Receipt inventory mismatch')
    require({str(p.relative_to(source / 'raw')) for p in (source / 'raw').rglob('*.mp3')}
            == {r['path'] for r in selected}, 'Full MP3 inventory mismatch')
    rows = []
    for row, physical in zip(selected, local['rows']):
        receipt_name = 'items/' + row['id'] + '.json'
        require(not (source / receipt_name).is_symlink(), 'Symlink receipt forbidden')
        receipt_sha = sha_file(source / receipt_name)
        require(receipt_sha == summary['receipts_sha256'][row['id'] + '.json'], 'Acquisition receipt hash changed')
        bindings[receipt_name] = receipt_sha
        receipt = read_json(source / receipt_name)
        require(all(receipt[k] == v for k, v in row.items()) and receipt['contract_sha256'] == ACQUISITION_SHA,
                'Acquisition receipt identity mismatch')
        require(receipt['status'] == 'passed' and receipt['eligible_native60'] is True and receipt['all_samples_finite'] is True,
                'Failed/nonfinite acquisition decode')
        require(physical['sha256'] == row['sha256'] and physical['bytes'] == row['bytes']
                and physical['eligible_native60'] is True, 'Physical raw identity mismatch')
        for a, b in [('sample_rate', 'sample_rate'), ('channels', 'channels'), ('frames', 'decoded_frames'),
                     ('decoded_native_float32_sha256', 'decoded_native_float32_sha256')]:
            require(physical[a] == receipt[b], 'Physical/receipt frame or format mismatch')
        frames = receipt['decoded_frames']
        require(type(frames) is int and frames >= FRAMES and receipt['sample_rate'] == SR and receipt['channels'] == 2,
                'Short source or unsupported native format')
        require(abs(receipt['decoded_duration_s'] - frames / SR) < 1e-9, 'Native duration mismatch')
        prepared = dict(row, native_frames=frames, crop_start_frame=(frames-FRAMES)//2,
                        crop_frames=FRAMES, acquisition_role=row['role'], role=ROLE,
                        acquisition_decoded_float32_sha256=receipt['decoded_native_float32_sha256'])
        native_source(source, prepared)
        rows.append(prepared)
    require(sum(r['bytes'] for r in rows) == c['intended_bytes'] == summary['bytes'] == local['bytes'] == remote['bytes'], 'Byte total mismatch')
    require(all(sha_file(source / name) == digest for name, digest in bindings.items()), 'Evidence mutated during audit')
    return rows, bindings


def validate_paths(source, output):
    require(source == SOURCE and source.resolve() == source, 'Only the designated NFS-data source is allowed')
    require(output == OUTPUT and output.resolve() == output, 'Only the new mureka60 NFS-data destination is allowed')


def freeze(source, output):
    require(not (output / 'materialization_contract.json').exists(), 'Contract already frozen; use materialize to resume')
    require(not output.exists() or {p.name for p in output.iterdir()} <= {'writer.lock'},
            'Freeze destination contains foreign/unreceipted artifacts')
    rows, bindings = audit_inputs(source)
    headers, evidence = reviewed_headers(rows)
    verified = []
    for row in rows:
        verified.append(dict(row, **verify_interval(source, row, headers[row['id']])))
        if len(verified) % 25 == 0:
            print(json.dumps(dict(stage='freeze',sequential_intervals_verified=len(verified),total=COUNT)),flush=True)
    rows = verified
    require({r['id']:r['sf_header_frames'] for r in rows if r['sf_header_frames'] != r['native_frames']}
            == KNOWN_HEADER_MISMATCHES, 'Observed header mismatch inventory changed')
    require(all(sha_file(source / name) == digest for name,digest in bindings.items())
            and all(sha_file(path) == digest for path,digest in evidence.items()), 'Evidence changed during sequential audit')
    rt = runtime()
    contract = dict(schema_version=2, status='frozen_before_materialization', purpose='measurement_only',
                    classifier_admission_authorized=False, role=ROLE, acquisition_role='reserved_unscored',
                    source_root=str(source), output_dir=str(output), source_repo='homura23/MUSIC8K', source_revision=REVISION,
                    acquisition_contract_sha256=ACQUISITION_SHA, configuration=CONFIG,
                    configuration_sha256=canonical_hash(CONFIG), runtime=rt, runtime_sha256=canonical_hash(rt),
                    code_sha256=sha_file(__file__), input_file_sha256=bindings,
                    decoder_evidence_sha256=evidence, decoder_admission_claim=CLAIM, rows=rows,
                    selected_ids=[r['id'] for r in rows])
    contract['contract_sha256'] = canonical_hash(contract)
    publish(output / 'materialization_contract.json', json_bytes(contract))
    return contract


def validate_contract(contract, source, output):
    body = {k: v for k, v in contract.items() if k != 'contract_sha256'}
    require(canonical_hash(body) == contract['contract_sha256'], 'Contract canonical hash mismatch')
    require(contract['source_root'] == str(source) and contract['output_dir'] == str(output), 'Frozen location mismatch')
    require(contract['code_sha256'] == sha_file(__file__), 'Frozen preparation code changed')
    require(contract['configuration'] == CONFIG and contract['configuration_sha256'] == canonical_hash(CONFIG), 'Frozen configuration changed')
    require(contract['purpose'] == 'measurement_only' and contract['role'] == ROLE
            and contract['classifier_admission_authorized'] is False
            and contract['acquisition_role'] == 'reserved_unscored'
            and contract['schema_version'] == 2 and contract['status'] == 'frozen_before_materialization'
            and contract['source_repo'] == 'homura23/MUSIC8K' and contract['source_revision'] == REVISION
            and contract['acquisition_contract_sha256'] == ACQUISITION_SHA,
            'Forbidden development/classifier admission or provenance change')
    rows, bindings = audit_inputs(source)
    headers, evidence = reviewed_headers(rows)
    require(evidence == contract['decoder_evidence_sha256'] and contract['decoder_admission_claim'] == CLAIM,
            'Frozen decoder evidence or admission claim changed')
    frozen_rows = contract['rows']
    require(len(frozen_rows) == COUNT, 'Frozen interval row count changed')
    for row, frozen in zip(rows, frozen_rows):
        require(all(frozen.get(k) == v for k, v in row.items()), 'Frozen acquisition row changed')
        for key in ('native_frames','crop_start_frame','crop_frames','sf_header_frames','sf_actual_read_frames',
                    'sf_header_minus_acquisition_frames','sf_actual_minus_acquisition_frames'):
            require(type(frozen[key]) is int, 'Frozen frame coordinates must be integers')
        require(frozen['sf_header_frames'] == headers[row['id']]
                and type(frozen['sf_actual_read_frames']) is int
                and frozen['sf_actual_read_frames'] >= frozen['crop_start_frame'] + FRAMES
                and frozen['sf_header_minus_acquisition_frames'] == frozen['sf_header_frames']-row['native_frames']
                and frozen['sf_actual_minus_acquisition_frames'] == frozen['sf_actual_read_frames']-row['native_frames']
                and frozen['center_seek_equals_sequential_float64'] is True
                and frozen['sequential_all_samples_finite'] is True
                and frozen['decoder_admission_claim'] == CLAIM
                and frozen['source_sha256_before'] == frozen['source_sha256_after'] == row['sha256'],
                'Frozen interval evidence invalid')
        for key in ('center_float64_sha256','center_float32_sha256','sf_sequential_float64_sha256',
                    'sf_sequential_float32_sha256','native_crop_float64_sha256','native_crop_float32_sha256'):
            value = frozen[key]
            require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),
                    'Invalid frozen waveform hash')
        require(frozen['native_crop_float64_sha256'] == frozen['center_float64_sha256']
                and frozen['native_crop_float32_sha256'] == frozen['center_float32_sha256'], 'Native crop hash aliases disagree')
        # Open every decoder before runtime fingerprinting, as during freeze.
        with sf.SoundFile(source / 'raw' / row['path']) as audio:
            require((audio.samplerate, audio.channels, audio.frames) == (SR, 2, headers[row['id']]),
                    'Frozen decoder header/rate/channels changed')
    # Match freeze's decoder initialization order, including lazily loaded codecs.
    rt = runtime()
    require(contract['runtime'] == rt and contract['runtime_sha256'] == canonical_hash(rt), 'Frozen runtime changed')
    require([r['id'] for r in rows] == contract['selected_ids']
            and bindings == contract['input_file_sha256'], 'Frozen selection/evidence changed')
    return frozen_rows


def decode_crop(source, row):
    path = native_source(source, row)
    with sf.SoundFile(path) as audio:
        require((audio.samplerate, audio.channels, audio.frames) == (SR, 2, row['sf_header_frames']), 'Frozen decoder format changed')
        require(audio.seek(row['crop_start_frame']) == row['crop_start_frame'], 'Seek position mismatch')
        data = audio.read(FRAMES, dtype='float64', always_2d=True)
    require(data.shape == (FRAMES, 2) and np.isfinite(data).all(), 'Incomplete/nonfinite crop')
    require(sha_file(path) == row['sha256'], 'Source changed while decoding')
    require(waveform_hash(data, '<f8') == row['center_float64_sha256']
            and waveform_hash(data, '<f4') == row['center_float32_sha256'], 'Frozen center sample hash mismatch')
    return np.ascontiguousarray(data, dtype='<f4')


def one_item(source, output, row, contract_hash):
    item_id = 'music8k_mureka_v9_' + row['id']
    destination, receipt_path = output / 'audio' / (item_id + '.wav'), output / 'items' / (item_id + '.json')
    crop = decode_crop(source, row)
    waveform_sha = hashlib.sha256(crop.tobytes()).hexdigest()
    base = dict(item_id=item_id, status='verified', role=ROLE, label=1, source_id='Mureka_v9', source_group='Mureka_v9',
                group_id=row['reference_group_id'], acquisition_role='reserved_unscored', classifier_admission_authorized=False,
                contract_sha256=contract_hash, input_row_sha256=canonical_hash(row),
                source_audio_path=str(source / 'raw' / row['path']), source_audio_sha256=row['sha256'],
                source_total_frames=row['native_frames'], source_sample_rate=SR, source_channels=2,
                acquisition_ffmpeg_frames=row['native_frames'], sf_header_frames=row['sf_header_frames'],
                sf_actual_read_frames=row['sf_actual_read_frames'],
                sf_header_minus_acquisition_frames=row['sf_header_minus_acquisition_frames'],
                sf_actual_minus_acquisition_frames=row['sf_actual_minus_acquisition_frames'],
                sf_sequential_float64_sha256=row['sf_sequential_float64_sha256'],
                sf_sequential_float32_sha256=row['sf_sequential_float32_sha256'],
                center_float64_sha256=row['center_float64_sha256'], center_float32_sha256=row['center_float32_sha256'],
                native_crop_float64_sha256=row['native_crop_float64_sha256'],
                decoder_admission_claim=CLAIM, center_seek_equals_sequential_float64=True,
                acquisition_decoded_float32_sha256=row['acquisition_decoded_float32_sha256'],
                crop_start_frame=row['crop_start_frame'], crop_frames=FRAMES,
                crop_end_frame_exclusive=row['crop_start_frame'] + FRAMES,
                native_crop_float32_sha256=waveform_sha, standardized_waveform_sha256=waveform_sha,
                standardized_path=str(destination), native_sr=SR, duration=60, audio_offset_s=0, requires_crop=0,
                standardized_sr=SR, standardized_channels=2, standardized_frames=FRAMES)
    if receipt_path.exists():
        require(not receipt_path.is_symlink() and not destination.is_symlink(), 'Symlink resume forbidden')
        receipt = read_json(receipt_path)
        require(all(receipt.get(k) == v for k, v in base.items()), 'Resume receipt/contract mismatch')
        require(sha_file(destination) == receipt['standardized_file_sha256']
                and destination.stat().st_size == receipt['standardized_file_bytes'], 'Resume output hash mismatch')
    else:
        require(not destination.exists() and not destination.is_symlink(), 'Unreceipted output conflict; manual recovery required')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.crop-', suffix='.wav', delete=False) as f:
            temporary = Path(f.name)
        try:
            sf.write(temporary, crop, SR, format='WAV', subtype='FLOAT')
            back, rate = sf.read(temporary, dtype='float32', always_2d=True)
            require(rate == SR and np.array_equal(back, crop), 'FLOAT roundtrip changed samples')
            with temporary.open('rb') as f:
                os.fsync(f.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink()
        receipt = dict(base, standardized_file_sha256=sha_file(destination), standardized_file_bytes=destination.stat().st_size)
        publish(receipt_path, json_bytes(receipt))
    with sf.SoundFile(destination) as audio:
        require((audio.samplerate, audio.channels, audio.frames, audio.subtype) == (SR, 2, FRAMES, 'FLOAT'), 'Output format changed')
        back = audio.read(dtype='float32', always_2d=True)
    require(np.array_equal(back, crop), 'Output samples differ from frozen native interval')
    return receipt


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def materialize(source, output):
    contract = read_json(output / 'materialization_contract.json')
    rows = validate_contract(contract, source, output)
    receipts = []
    for row in rows:
        receipts.append(one_item(source, output, row, contract['contract_sha256']))
        if len(receipts) % 25 == 0:
            print(json.dumps(dict(stage='materialize',items_verified=len(receipts),total=COUNT)),flush=True)
    require(len(receipts) == COUNT, 'No partial manifest publication')
    native = [dict(id=r['item_id'], audio_path=r['source_audio_path'], audio_offset_s=r['crop_start_frame']/SR,
                   native_sample_rate_hz=SR, label=1, source_group='Mureka_v9', role=ROLE, group_id=r['group_id'],
                   physical_frames=r['source_total_frames'], physical_sample_rate_hz=SR, physical_channels=2,
                   acquisition_ffmpeg_frames=r['acquisition_ffmpeg_frames'], sf_header_frames=r['sf_header_frames'],
                   sf_actual_read_frames=r['sf_actual_read_frames'],
                   sf_header_minus_acquisition_frames=r['sf_header_minus_acquisition_frames'],
                   sf_actual_minus_acquisition_frames=r['sf_actual_minus_acquisition_frames'],
                   center_float64_sha256=r['center_float64_sha256'], center_float32_sha256=r['center_float32_sha256'],
                   native_crop_float64_sha256=r['native_crop_float64_sha256'], native_crop_float32_sha256=r['native_crop_float32_sha256'],
                   decoder_admission_claim=CLAIM, center_seek_equals_sequential_float64=True,
                   crop_start_frame=r['crop_start_frame'], crop_frames=FRAMES, crop_end_frame_exclusive=r['crop_end_frame_exclusive'],
                   source_audio_sha256=r['source_audio_sha256'], acquisition_role='reserved_unscored',
                   classifier_admission_authorized=False) for r in receipts]
    manifest, metadata = csv_bytes(receipts), csv_bytes(native)
    publish(output / 'inference_manifest.csv', manifest)
    publish(output / 'native_metadata_60s.csv', metadata)
    shards = []
    for start in range(0, COUNT, 24):
        index = start // 24
        part = receipts[start:start+24]
        body = ''.join(r['standardized_path'] + '\n' for r in part).encode()
        name = f'inference_shard_{index:02d}.txt'
        publish(output / name, body)
        shards.append(dict(index=index, rows=len(part), sha256=hashlib.sha256(body).hexdigest(), path=str(output / name)))
    summary = dict(status='verified', rows=COUNT, contract_sha256=contract['contract_sha256'],
                   schema_version=2, decoder_admission_claim=CLAIM,
                   decoder_evidence_sha256=contract['decoder_evidence_sha256'],
                   header_mismatch_ids=[r['id'] for r in rows if r['sf_header_frames'] != r['native_frames']],
                   actual_eof_mismatch_ids=[r['id'] for r in rows if r['sf_actual_read_frames'] != r['native_frames']],
                   manifest_sha256=hashlib.sha256(manifest).hexdigest(), native_metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                   shards=shards, role=ROLE, classifier_admission_authorized=False,
                   receipts_sha256={r['item_id']+'.json':sha_file(output/'items'/(r['item_id']+'.json')) for r in receipts})
    publish(output / 'materialization_summary.json', json_bytes(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'materialize'), required=True)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    validate_paths(args.source_root, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = (freeze if args.stage == 'freeze' else materialize)(args.source_root, args.output_dir)
    print(json.dumps(dict(stage=args.stage,status=result['status'],
                          rows=len(result['rows']) if isinstance(result['rows'],list) else result['rows'],
                          contract_sha256=result['contract_sha256'])),flush=True)


if __name__ == '__main__':
    main()
