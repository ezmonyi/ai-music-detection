"""Fixed new1695 native-stereo materialization; default metadata-only preflight.

No full3869 admission, features, classifiers or neural inference. Run mode binds
all metadata/code/runtime before audio reads. Interrupted, unreceipted waveforms
are retained and rejected, never silently reused. Immutable failures survive
retries. POSIX advisory flock requires cooperating writers on the same storage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import uuid

VERSION = 'materialize_native30_new1695_v1'
REPORT_SHA = 'a3dcf87f703b7709b01d963e1a1c2e96f9a223d159180d1b793021337a3b160a'
SCREEN_SHA = 'ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87'
DSP_SHA = '2e05a4d0d2e7789888aa41a53bece46e8de97fad34ef34f21354d165e77b55be'
WORKSPACE = Path('/Users/yi/Documents/code/music')
SOURCE_COUNTS = {'ACE-Step': 400, 'FMA': 393, 'HeartMuLa': 366,
                 'MTG-Jamendo': 19, 'Suno': 4, 'Udio': 500,
                 'human_medleydb': 12, 'human_moisesdb': 1}
EXPECTED = {'candidates': 1746, 'selected': 1695, 'mono': 51, 'local': 397,
            'sources': SOURCE_COUNTS}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_binding(path):
    p = Path(path).absolute()
    require(p.is_file() and not p.is_symlink(), f'bound regular file required: {p}')
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)}


def recheck(bindings):
    for entry in bindings.values():
        require(file_binding(entry['path']) == entry, f'bound file changed: {entry["path"]}')


def write_new(path, value):
    """Atomic, non-overwriting JSON publication on the destination filesystem."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def execution_path(original, source, copy_root):
    p = Path(original)
    require(p.is_absolute() and '..' not in p.parts, 'unsafe native path')
    if p.is_relative_to(WORKSPACE):
        require(source in {'FMA', 'Suno'}, 'only FMA/Suno may use local-origin copies')
        relative = p.relative_to(WORKSPACE)
        trees = {'fma_medium'} if source == 'FMA' else {'suno-ai-music-dataset-audio', 'humair025-suno-audio-mp3'}
        require(relative.parts and relative.parts[0] in trees,
                'unexpected local native subtree')
        root = Path(copy_root).absolute()
        require('..' not in root.parts, 'unsafe execution copy root')
        return str(root / relative), True
    require(p.parts[:2] == ('/', 'mnt'), 'native source must be workspace-local or /mnt')
    return str(p), False


def _reconcile(report, screen, copy_root, expected):
    """Generic metadata helper; production always supplies fixed EXPECTED."""
    candidates = report['candidates']
    require(len(candidates) == expected['candidates'], 'evidence candidate count mismatch')
    require(len({r['id'] for r in candidates}) == len(candidates), 'duplicate evidence ids')
    screens = {r['id']: r for r in screen['rows']}
    components = {r['component_id']: r for r in screen['components']}
    require(len(screens) == len(screen['rows']), 'duplicate screen ids')
    require(len(components) == len(screen['components']), 'duplicate screen components')
    selected, excluded = [], []
    local_count = 0
    for evidence in sorted(candidates, key=lambda r: r['id']):
        ident = evidence['id']
        require(isinstance(ident, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', ident), 'unsafe id')
        require(ident in screens, 'evidence id absent from screen')
        row = screens[ident]
        require(all(evidence[k] == row[k] for k in ('id', 'source_group', 'label', 'component_id')),
                f'evidence/screen identity disagreement: {ident}')
        component = components[row['component_id']]
        require(ident in component['members'] and ident in component['candidate_screen_members'],
                'component membership disagreement')
        require(not component['protected_relationships'] and not row['exclusion_reasons']
                and row['duration_exposure_candidate'] is True and row['role'] == 'development',
                f'protected/nondevelopment candidate: {ident}')
        require(evidence['accepted60_freeze_relation'] != 'selected', 'prior2174 overlap forbidden')
        require(evidence['accepted60_freeze_relation'] in {'absent', 'excluded'}, 'unknown accepted60 relation')
        native = evidence['native_evidence']
        channels = native['channels']
        require(type(channels) is int and channels in (1, 2), 'unsupported native channel evidence')
        require(evidence['native_stereo_metadata_eligible'] is (channels == 2), 'native stereo flag mismatch')
        if channels == 1:
            excluded.append({'id': ident, 'source_group': row['source_group'], 'label': row['label'],
                             'component_id': row['component_id'], 'reason': 'native_mono', 'native_evidence': native})
            continue
        require(row['source_group'] in expected['sources'], 'protected/unsupported source')
        require(evidence['accepted60_freeze_relation'] == 'absent', 'new1695 must be absent from prior2174')
        require(row['full_native_duration_known'] is True, 'native duration evidence required')
        duration = native['duration_s']
        require(type(duration) in (int, float) and math.isfinite(duration) and 30 <= duration <= 60,
                'full-native duration outside [30,60]')
        require(type(native['sample_rate_hz']) is int and native['sample_rate_hz'] > 0, 'invalid native rate')
        require(isinstance(native['sha256'], str) and re.fullmatch('[0-9a-f]{64}', native['sha256']), 'invalid native SHA')
        require(row['label'] in ('0', '1') and isinstance(row['group_id'], str) and row['group_id'], 'label/group missing')
        mapped, local = execution_path(native['path'], row['source_group'], copy_root)
        local_count += int(local)
        selected.append({'id': ident, 'source_group': row['source_group'], 'label': row['label'],
                         'group_id': row['group_id'], 'component_id': row['component_id'], 'role': 'development',
                         'original_native_path': native['path'], 'execution_native_path': mapped,
                         'local_origin_copy': local, 'native_evidence': native,
                         'receipt_binding': evidence['receipt_binding'], 'receipt_row': evidence['receipt_row'],
                         'screen_row_sha256': value_hash(row), 'evidence_row_sha256': value_hash(evidence),
                         'component_sha256': value_hash(component), 'approved_region': None})
    require(len(selected) == expected['selected'] and len(excluded) == expected['mono'], '1695/51 split mismatch')
    require(dict(Counter(r['source_group'] for r in selected)) == expected['sources'], 'source roster mismatch')
    require(local_count == expected['local'], 'local-origin397 count mismatch')
    return selected, excluded


def symbol_mapping(address, maps_text):
    """Find exactly the executable Linux mapping containing a live symbol."""
    require(type(address) is int and address > 0, 'invalid live library symbol address')
    matches = []
    for line in maps_text.splitlines():
        fields = line.split(None, 5)
        require(len(fields) >= 5, 'malformed process mapping')
        limits = fields[0].split('-')
        require(len(limits) == 2, 'malformed mapping address range')
        start, end = (int(value, 16) for value in limits)
        if start <= address < end:
            require(len(fields) == 6 and 'x' in fields[1], 'symbol is not in a file-backed executable mapping')
            path = fields[5]
            require(path.startswith('/') and not path.endswith(' (deleted)'), 'loaded library path unavailable/deleted')
            device = fields[3].split(':')
            require(len(device) == 2, 'malformed mapped device')
            matches.append({'mapped_path': path, 'device_major': int(device[0], 16),
                            'device_minor': int(device[1], 16), 'inode': int(fields[4]),
                            'permissions': fields[1],
                            'symbol_file_offset': int(fields[2], 16) + address - start})
    require(len(matches) == 1, 'live symbol must resolve to exactly one mapping')
    return matches[0]


def loaded_sndfile_binding(soundfile):
    """Bind the file actually supplying the CFFI-loaded sf_version_string.

    _libname is absent in SoundFile 0.14 and loader names are not reliable file
    identities. The symbol pointer selects its actual /proc/self/maps entry;
    device/inode checks reject a replaced or deleted backing file. ASLR-dependent
    absolute addresses are deliberately omitted from the stable contract.
    """
    require(Path('/proc/self/maps').is_file(), 'Linux live-library mapping verification required')
    symbol = soundfile._ffi.addressof(soundfile._snd, 'sf_version_string')
    address = int(soundfile._ffi.cast('uintptr_t', symbol))
    mapping = symbol_mapping(address, Path('/proc/self/maps').read_text())
    path = Path(mapping['mapped_path']).resolve(strict=True)
    before = path.stat()
    require((os.major(before.st_dev), os.minor(before.st_dev), before.st_ino) ==
            (mapping['device_major'], mapping['device_minor'], mapping['inode']),
            'loaded libsndfile mapping/file device-inode mismatch')
    binding = file_binding(path)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'loaded libsndfile backing file changed during hashing')
    proof = {'method': 'soundfile_CFFI_live_symbol_linux_proc_maps', 'symbol': 'sf_version_string',
             **mapping}
    return binding, proof


def runtime_binding(dsp):
    """Bind versions, Python and loaded numerical module/extension files."""
    names = ['numpy', 'numpy.core._multiarray_umath', 'scipy', 'scipy.signal._signaltools',
             'scipy.signal._upfirdn', 'scipy.signal._upfirdn_apply', 'scipy.signal._fir_filter_design',
             'soundfile', '_cffi_backend']
    module_files = {name: file_binding(Path(importlib.import_module(name).__file__).resolve()) for name in names}
    library, resolution = loaded_sndfile_binding(dsp.sf)
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'executable': file_binding(Path(sys.executable).resolve()),
            'numpy': dsp.np.__version__, 'scipy': dsp.scipy.__version__,
            'soundfile': dsp.sf.__version__, 'libsndfile': dsp.sf.__libsndfile_version__,
            'libsndfile_binary': library, 'libsndfile_resolution': resolution, 'module_files': module_files}


def check_contract_files(contract):
    recheck(contract['bindings'])
    runtime = contract.get('runtime')
    if runtime:
        recheck({'python': runtime['executable'], 'libsndfile': runtime['libsndfile_binary'],
                 **runtime['module_files']})


def prepare(report_path, screen_path, copy_root, output):
    code = Path(__file__).absolute()
    paths = {'report': report_path, 'screen': screen_path, 'materializer': code,
             'tests': code.with_name('test_materialize_native30_new1695_v1.py'),
             'dsp': code.with_name('standardize_native30_v1.py'),
             'dsp_tests': code.with_name('test_standardize_native30_v1.py')}
    bindings = {key: file_binding(path) for key, path in paths.items()}
    for key, sha in [('report', REPORT_SHA), ('screen', SCREEN_SHA), ('dsp', DSP_SHA)]:
        require(bindings[key]['sha256'] == sha, f'fixed {key} SHA mismatch')
    report, screen = read_json(report_path), read_json(screen_path)
    require(report['input_bindings']['screen']['sha256'] == SCREEN_SHA, 'report/screen binding mismatch')
    selected, excluded = _reconcile(report, screen, copy_root, EXPECTED)
    dsp = importlib.import_module('standardize_native30_v1')
    require(Path(dsp.__file__).resolve() == Path(paths['dsp']).resolve(), 'wrong DSP module import')
    runtime = runtime_binding(dsp)
    recheck(bindings)
    contract = {'version': VERSION, 'status': 'frozen_before_any_audio_processing',
                'scope': 'new1695_native_stereo_DSP_only', 'expected_count': EXPECTED['selected'],
                'output_root': str(Path(output).absolute()), 'local_origin_copy_root': str(Path(copy_root).absolute()),
                'rows': selected, 'excluded_native_mono': excluded, 'configuration': dsp.CONFIG,
                'runtime': runtime, 'bindings': bindings, 'source_provenance_bindings': report['input_bindings'],
                'row_order': 'id_ascending', 'classifier_fits': 0, 'cohort_admitted': False,
                'feature_extraction_authorized': False, 'protected_sources': ['Mureka', 'Saraga', 'prior2174'],
                'resume_policy': 'verify_contract_row_source_receipt_output_bytes; unreceipted_audio_fails'}
    return contract, dsp


@contextmanager
def writer_lock(output):
    output = Path(output)
    output.mkdir(exist_ok=True)
    require(output.is_dir() and not output.is_symlink(), 'regular output directory required')
    lock = output / 'writer.lock'
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('exclusive output writer lock is held') from None
        yield
    finally:
        os.close(descriptor)


def publish_waveform(dsp, path, values):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            dsp.sf.write(stream, values, dsp.RATE, format='WAV', subtype='FLOAT')
            stream.flush(); os.fsync(stream.fileno())
        with dsp.sf.SoundFile(temporary) as handle:
            require(handle.samplerate == 44100 and handle.channels == 2 and
                    handle.frames == 1323000 and handle.format == 'WAV' and handle.subtype == 'FLOAT',
                    'output WAV format/rate/channels/frames mismatch')
            decoded = handle.read(dtype='float32', always_2d=True)
            require(not len(handle.read(1, dtype='float32', always_2d=True)), 'output not at EOF')
        require(decoded.shape == values.shape and decoded.tobytes() == values.tobytes(), 'output float32 roundtrip mismatch')
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_receipt(output, row, contract_sha):
    ident = row['id']; receipt_path = output / 'items' / (ident + '.json')
    require(receipt_path.is_file() and not receipt_path.is_symlink(), 'missing/unsafe receipt')
    envelope = read_json(receipt_path)
    receipt = envelope['payload']
    require(set(envelope) == {'payload', 'receipt_sha256'} and value_hash(receipt) == envelope['receipt_sha256'],
            'receipt payload hash mismatch')
    require(receipt['contract_sha256'] == contract_sha and receipt['row_sha256'] == value_hash(row)
            and receipt['row'] == row and receipt['status'] == 'materialized_DSP_only', 'receipt contract/row mismatch')
    source = Path(row['execution_native_path'])
    require(source.is_file() and not source.is_symlink() and digest(source) == row['native_evidence']['sha256'],
            'resume source bytes mismatch')
    wav = output / 'audio' / (ident + '.wav')
    require(receipt['standardized_path'] == str(wav) and wav.is_file() and not wav.is_symlink()
            and wav.stat().st_size == receipt['file_bytes'] and digest(wav) == receipt['file_sha256'],
            'resume waveform bytes/path mismatch')
    return envelope


def _one(output, row, contract_sha, dsp):
    ident = row['id']; receipt = output / 'items' / (ident + '.json'); wav = output / 'audio' / (ident + '.wav')
    if receipt.exists() or receipt.is_symlink():
        return _verify_receipt(output, row, contract_sha)
    require(not wav.exists() and not wav.is_symlink(), 'unreceipted waveform retained; quarantine/review required')
    require(not list((output / 'audio').glob('.' + ident + '.wav.*.tmp')), 'unreceipted temporary waveform retained')
    native = row['native_evidence']
    values, audit = dsp.standardize(row['execution_native_path'], native['sha256'],
                                    native['sample_rate_hz'], native['channels'], approved_region=None)
    require(audit['region_kind'] == 'full_native_sequential_decode' and audit['native_channels'] == 2,
            'unexpected DSP region/channels')
    require(audit['sequential_decode']['actual_frames'] <= 60 * native['sample_rate_hz'],
            'actual native duration exceeds60s; scope review required')
    publish_waveform(dsp, wav, values)
    payload = {'status': 'materialized_DSP_only', 'contract_sha256': contract_sha,
               **{key: row[key] for key in ('id', 'source_group', 'group_id', 'component_id', 'label', 'role')},
               'row_sha256': value_hash(row), 'row': row, 'standardized_path': str(wav),
               'file_sha256': digest(wav), 'file_bytes': wav.stat().st_size,
               'waveform_float32_sha256': audit['output_waveform_float32_sha256'],
               'waveform_bit_exact_roundtrip': True, 'audit': audit}
    envelope = {'payload': payload, 'receipt_sha256': value_hash(payload)}
    write_new(receipt, envelope)
    return envelope


def validate_inventory(output, rows, *, complete=False):
    """Reject extras, nested entries and symlinks rather than blessing them in COMMIT."""
    ident_set = {r['id'] for r in rows}
    roots = {'writer.lock', 'contract.json', 'manifest.json', 'COMMIT.json',
             'items', 'audio', 'failures', 'runs'}
    require({p.name for p in output.iterdir()} <= roots, 'unexpected output root file')
    for path in output.iterdir():
        require(not path.is_symlink(), 'output symlink forbidden')
        require(path.is_dir() if path.name in {'items', 'audio', 'failures', 'runs'} else path.is_file(),
                'output entry type mismatch')
    expected_wav = {ident + '.wav' for ident in ident_set}
    expected_receipts = {ident + '.json' for ident in ident_set}
    for directory in ['items', 'audio', 'failures', 'runs']:
        for path in (output / directory).iterdir():
            require(path.is_file() and not path.is_symlink(), 'nested/nonregular output entry')
            if directory == 'items':
                valid = path.name in expected_receipts
            elif directory == 'audio':
                temporary = re.fullmatch(r'\.(.+)\.wav\.[0-9a-f]{32}\.tmp', path.name)
                valid = path.name in expected_wav or (not complete and temporary and temporary[1] in ident_set)
            elif directory == 'failures':
                failure = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.json', path.name)
                valid = failure and failure[1] in ident_set
            else:
                valid = re.fullmatch(r'[0-9a-f]{32}\.json', path.name)
            require(valid, f'unexpected {directory} file: {path.name}')
    if complete:
        require({p.name for p in (output / 'audio').iterdir()} == expected_wav, 'incomplete audio inventory')
        require({p.name for p in (output / 'items').iterdir()} == expected_receipts, 'incomplete receipt inventory')


def _products(output):
    return {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for name in ['contract.json', 'manifest.json', 'items', 'audio', 'failures', 'runs']
            for p in ([output / name] if (output / name).is_file() else sorted((output / name).rglob('*')))
            if p.is_file()}


def _run_contract(contract, dsp, workers=4):
    """Generic runner for fixed prepared contracts; count override exists only via test fixtures.

    Production CLI exposes no count or injected-processor override.
    """
    require(type(workers) is int and 1 <= workers <= 16, 'workers must be in1..16')
    output = Path(contract['output_root'])
    rows = contract['rows']
    require(len(rows) == contract['expected_count'] and len({r['id'] for r in rows}) == len(rows), 'contract row count/identity')
    require(rows == sorted(rows, key=lambda r: r['id']), 'contract rows must be deterministic')
    sha = value_hash(contract)
    with writer_lock(output):
        check_contract_files(contract)
        contract_path = output / 'contract.json'
        if contract_path.exists():
            require(not contract_path.is_symlink() and read_json(contract_path) == contract
                    and digest(contract_path) == sha, 'existing contract conflict')
        else:
            require({p.name for p in output.iterdir()} == {'writer.lock'}, 'nonempty output without contract')
            write_new(contract_path, contract)
        for name in ['items', 'audio', 'failures', 'runs']:
            path = output / name
            path.mkdir(exist_ok=True)
            require(path.is_dir() and not path.is_symlink(), 'unsafe output subdirectory')
        validate_inventory(output, rows)
        commit_path = output / 'COMMIT.json'
        committed = commit_path.exists()
        if committed:
            require(not commit_path.is_symlink(), 'unsafe COMMIT')
            commit = read_json(commit_path)
            require(commit['contract_sha256'] == sha and commit['products'] == _products(output), 'COMMIT products mismatch')
        run_id = uuid.uuid4().hex
        def process(row):
            try:
                if committed:
                    result = _verify_receipt(output, row, sha)
                else:
                    result = _one(output, row, sha, dsp)
                return {'id': row['id'], 'ok': True, 'receipt': result}
            except Exception as exc:
                failure = {'id': row['id'], 'contract_sha256': sha, 'row_sha256': value_hash(row),
                           'exception': type(exc).__name__, 'message': str(exc), 'run_id': run_id}
                if not committed:
                    write_new(output / 'failures' / (row['id'] + '.' + run_id + '.json'), failure)
                return {'id': row['id'], 'ok': False, 'failure': failure}
        results_by_id = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process, row) for row in rows]
            for completed_count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results_by_id[result['id']] = result
                if completed_count % 25 == 0 or completed_count == len(rows):
                    passed = sum(r['ok'] for r in results_by_id.values())
                    print(json.dumps({'event': 'materialization_progress', 'reviewed': completed_count,
                                      'planned': len(rows), 'passed_or_resumed': passed,
                                      'failed': completed_count - passed}, sort_keys=True), file=sys.stderr, flush=True)
        results = [results_by_id[row['id']] for row in rows]
        check_contract_files(contract)
        good = [r for r in results if r['ok']]; failed = [r['failure'] for r in results if not r['ok']]
        summary = {'status': 'complete_DSP_only' if not failed else 'partial_no_COMMIT',
                   'contract_sha256': sha, 'expected': len(rows), 'completed': len(good),
                   'failed': len(failed), 'failures': failed, 'run_id': run_id,
                   'classifier_fits': 0, 'cohort_admitted': False}
        if committed:
            require(not failed, 'committed materialization source/receipt verification failed')
            return {'status': 'verified_existing_COMMIT', 'completed': len(good), 'commit_sha256': digest(commit_path)}
        write_new(output / 'runs' / (run_id + '.json'), summary)
        if failed:
            return summary
        validate_inventory(output, rows, complete=True)
        manifest = {'version': VERSION, 'status': 'all1695_DSP_materialized_not_cohort_admitted',
                    'contract_sha256': sha, 'count': len(rows), 'records': [r['receipt']['payload'] for r in good],
                    'excluded_native_mono': contract['excluded_native_mono'], 'classifier_fits': 0,
                    'cohort_admitted': False, 'feature_extraction_authorized': False}
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            require(not manifest_path.is_symlink() and read_json(manifest_path) == manifest, 'existing manifest conflict')
        else:
            write_new(manifest_path, manifest)
        check_contract_files(contract)
        validate_inventory(output, rows, complete=True)
        write_new(commit_path, {'status': 'committed_new1695_DSP_only', 'contract_sha256': sha,
                                'completed': len(rows), 'products': _products(output)})
        return {**summary, 'commit_sha256': digest(commit_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['report', 'screen', 'local-origin-copy-root', 'output']:
        parser.add_argument('--' + arg, required=True)
    parser.add_argument('--mode', choices=['preflight', 'run'], default='preflight')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    contract, dsp = prepare(args.report, args.screen, args.local_origin_copy_root, args.output)
    if args.mode == 'preflight':
        print(json.dumps({'status': 'metadata_preflight_only_no_audio_reads_or_writes',
                          'selected': len(contract['rows']), 'excluded_mono': len(contract['excluded_native_mono']),
                          'contract_sha256': value_hash(contract), 'output_root': contract['output_root'],
                          'source_counts': dict(Counter(r['source_group'] for r in contract['rows'])),
                          'local_origin_count': sum(r['local_origin_copy'] for r in contract['rows'])}, sort_keys=True))
        return 0
    result = _run_contract(contract, dsp, args.workers)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] != 'partial_no_COMMIT' else 1


if __name__ == '__main__':
    raise SystemExit(main())
