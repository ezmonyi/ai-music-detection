#!/usr/bin/env python3
"""Synthetic-only BC codec engineering probe; never reads study audio.

Generate four fixed 4 s, 16 kHz mono IEEE-FLOAT WAV fixtures, encode with
libmp3lame 128 kbit/s and libopus 96 kbit/s, decode explicitly to 16 kHz mono
IEEE-FLOAT WAV, and report raw length/alignment/amplitude behavior.  Alignment
is diagnostic only: decoded arrays are never shifted, trimmed, or padded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import uuid

import numpy as np


VERSION = 'probe_bc_codec_roundtrip_v1'
STATUS = 'committed_synthetic_codec_engineering_probe_not_BC_admission'
SAMPLE_RATE = 16000
SAMPLES = 64000
SCOPE = {'synthetic_only': True, 'reserved_audio_read': False,
         'development_audio_read': False, 'BC_measured': False,
         'BC_admitted': False, 'gate_threshold_chosen': False,
         'classifier_fits': 0, 'model_scoring': False,
         'automatic_alignment': False, 'automatic_trim': False,
         'automatic_padding': False, 'normalization': False,
         'clipping_applied': False}
CODECS = {
    'mp3_128k': {'encoder': 'libmp3lame', 'bitrate': '128k', 'extension': '.mp3',
                 'encoder_options': ['-reservoir', '1', '-abr', '0']},
    'opus_96k': {'encoder': 'libopus', 'bitrate': '96k', 'extension': '.opus',
                 'encoder_options': ['-application', 'audio', '-vbr', 'on',
                                     '-compression_level', '10', '-frame_duration', '20']},
}
OPERATION_ORDER = [
    'generate deterministic float64 mathematical fixture then cast once to float32',
    'write exact 16000 Hz mono IEEE-FLOAT WAV without normalization or clipping',
    'encode the complete WAV with the declared single-thread codec recipe',
    'decode the complete encoded stream to 16000 Hz mono pcm_f32le WAV',
    'measure raw decoded length, same-index error, cross-correlation lag, and amplitude without altering samples',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hash_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def safe_file(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and '..' not in path.parts,
            'absolute unredirected path required: ' + str(path))
    require(path.is_file() and not path.is_symlink(), 'regular file required: ' + str(path))
    return path


def binding(path, expected=None):
    path = safe_file(path)
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(expected is None or result['sha256'] == expected, 'file SHA mismatch: ' + str(path))
    return result


def write_new(path, payload):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_new(path, canonical(value))


def synthetic_signals():
    t = np.arange(SAMPLES, dtype=np.float64) / SAMPLE_RATE
    silence = np.zeros(SAMPLES, dtype=np.float32)
    impulse = np.zeros(SAMPLES, dtype=np.float32)
    impulse[[8000, 32000, 56000]] = np.array([0.8, -0.6, 0.4], dtype=np.float32)
    tone = (0.8 * np.sin(2 * np.pi * 997.0 * t + 0.37)).astype(np.float32)
    raw = (0.42 * np.sin(2 * np.pi * 500.0 * t + 0.1)
           + 0.31 * np.sin(2 * np.pi * 750.0 * t + 0.7)
           + 0.23 * np.sin(2 * np.pi * 1250.0 * t - 0.4))
    multitone = (raw * (0.85 / np.max(np.abs(raw)))).astype(np.float32)
    result = {'silence': silence, 'impulse': impulse, 'tone_997hz': tone,
              'multitone_500_750_1250hz': multitone}
    require(all(x.dtype == np.float32 and x.shape == (SAMPLES,) and np.isfinite(x).all()
                and float(np.max(np.abs(x))) < 1 for x in result.values()), 'invalid synthetic fixtures')
    return result


def wav_bytes(samples):
    samples = np.asarray(samples)
    require(samples.dtype == np.float32 and samples.ndim == 1 and np.isfinite(samples).all(),
            'WAV input must be finite mono float32')
    raw = samples.astype('<f4', copy=False).tobytes(order='C')
    fmt = struct.pack('<HHIIHH', 3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32)
    body = b'fmt ' + struct.pack('<I', len(fmt)) + fmt + b'data' + struct.pack('<I', len(raw)) + raw
    return b'RIFF' + struct.pack('<I', len(body) + 4) + b'WAVE' + body


def read_float_wav(path, *, return_info=False):
    data = safe_file(path).read_bytes()
    require(data[:4] == b'RIFF' and data[8:12] == b'WAVE', 'not RIFF/WAVE: ' + str(path))
    offset, fmt_chunk, raw = 12, None, None
    while offset + 8 <= len(data):
        name, size = data[offset:offset + 4], struct.unpack('<I', data[offset + 4:offset + 8])[0]
        start, stop = offset + 8, offset + 8 + size
        require(stop <= len(data), 'truncated WAV chunk: ' + str(path))
        if name == b'fmt ':
            require(size >= 16, 'short WAV fmt chunk')
            fmt_chunk = data[start:stop]
        elif name == b'data':
            require(raw is None, 'multiple WAV data chunks unsupported')
            raw = data[start:stop]
        offset = stop + (size & 1)
    require(fmt_chunk is not None and raw is not None, 'WAV fmt/data missing')
    code, channels, rate, byte_rate, block_align, bits = struct.unpack('<HHIIHH', fmt_chunk[:16])
    require((channels, rate, byte_rate, block_align, bits) ==
            (1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32), 'not exact 16k mono float32 WAV')
    info = {'format_tag': code, 'channels': channels, 'sample_rate_hz': rate,
            'bits_per_sample': bits, 'block_align': block_align}
    if code == 3:
        info['format'] = 'WAVE_FORMAT_IEEE_FLOAT'
    else:
        require(code == 65534 and len(fmt_chunk) >= 40, 'not IEEE float or WAVE_FORMAT_EXTENSIBLE')
        extension_size, valid_bits, channel_mask = struct.unpack('<HHI', fmt_chunk[16:24])
        subformat = fmt_chunk[24:40]
        ieee_float_guid = struct.pack('<IHH8s', 3, 0, 0x10, b'\x80\x00\x00\xaa\x00\x38\x9b\x71')
        require(extension_size >= 22 and valid_bits == 32 and subformat == ieee_float_guid,
                'invalid WAVE_FORMAT_EXTENSIBLE FLOAT subtype')
        info.update({'format': 'WAVE_FORMAT_EXTENSIBLE_IEEE_FLOAT',
                     'valid_bits_per_sample': valid_bits, 'channel_mask': channel_mask,
                     'subformat_guid_hex': subformat.hex()})
    require(len(raw) % 4 == 0, 'partial float sample')
    samples = np.frombuffer(raw, dtype='<f4').astype(np.float32, copy=True)
    require(np.isfinite(samples).all(), 'nonfinite decoded WAV')
    return (samples, info) if return_info else samples


def signal_stats(samples):
    samples = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return {'samples': len(samples), 'duration_s': len(samples) / SAMPLE_RATE,
            'min': float(np.min(samples)) if len(samples) else None,
            'max': float(np.max(samples)) if len(samples) else None,
            'peak_abs': peak, 'rms': float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0,
            'mean': float(np.mean(samples)) if len(samples) else None,
            'nonfinite_samples': int(np.count_nonzero(~np.isfinite(samples))),
            'abs_ge_1_samples': int(np.count_nonzero(np.abs(samples) >= 1.0)),
            'peak_over_unity': bool(peak > 1.0),
            'clipping_inference': 'not_inferred; FLOAT magnitude at or above 1 is only counted',
            'leading_exact_zero_samples': leading_zeros(samples),
            'trailing_exact_zero_samples': leading_zeros(samples[::-1])}


def leading_zeros(samples):
    nonzero = np.flatnonzero(samples != 0)
    return int(nonzero[0]) if len(nonzero) else len(samples)


def cross_correlation(input_samples, decoded):
    left = np.asarray(input_samples, dtype=np.float64)
    right = np.asarray(decoded, dtype=np.float64)
    left = left - np.mean(left); right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return {'status': 'undefined_zero_energy', 'peak_lag_samples': None,
                'peak_lag_seconds': None, 'normalized_peak': None}
    length = len(left) + len(right) - 1
    fft_length = 1 << (length - 1).bit_length()
    correlation = np.fft.irfft(np.fft.rfft(right, fft_length)
                               * np.fft.rfft(left[::-1], fft_length), fft_length)[:length]
    index = int(np.argmax(np.abs(correlation)))
    lag = index - (len(left) - 1)
    return {'status': 'diagnostic_only_no_alignment_applied', 'peak_lag_samples': lag,
            'peak_lag_seconds': lag / SAMPLE_RATE,
            'normalized_peak': float(correlation[index] / denominator)}


def compare_raw(input_samples, decoded):
    input_samples = np.asarray(input_samples, dtype=np.float32)
    decoded = np.asarray(decoded, dtype=np.float32)
    overlap = min(len(input_samples), len(decoded))
    left, right = input_samples[:overlap].astype(np.float64), decoded[:overlap].astype(np.float64)
    delta = right - left
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    same = None if denominator == 0 else float(np.dot(left, right) / denominator)
    return {'input_samples': len(input_samples), 'decoded_samples': len(decoded),
            'decoded_minus_input_samples': len(decoded) - len(input_samples),
            'raw_overlap_samples': overlap,
            'raw_same_index_rmse': float(np.sqrt(np.mean(delta * delta))) if overlap else None,
            'raw_same_index_max_abs_error': float(np.max(np.abs(delta))) if overlap else None,
            'raw_same_index_cosine': same,
            'cross_correlation': cross_correlation(input_samples, decoded),
            'no_alignment_trim_or_padding_applied': True}


def command_environment():
    return {'LANG': 'C', 'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'}


def run_command(command, log_path, *, parse_json=False):
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=command_environment(), check=False)
    receipt = {'command': [str(value) for value in command], 'environment': command_environment(),
               'returncode': completed.returncode,
               'stdout_utf8': completed.stdout.decode('utf-8', errors='strict'),
               'stderr_utf8': completed.stderr.decode('utf-8', errors='strict')}
    write_json(log_path, receipt)
    require(completed.returncode == 0, 'command failed: ' + ' '.join(command))
    if parse_json:
        return receipt, json.loads(receipt['stdout_utf8'])
    return receipt, None


def tool_record(ffmpeg, ffprobe, ffmpeg_sha, ffprobe_sha):
    ffmpeg, ffprobe = safe_file(ffmpeg), safe_file(ffprobe)
    ffmpeg_entry, ffprobe_entry = binding(ffmpeg, ffmpeg_sha), binding(ffprobe, ffprobe_sha)
    env = command_environment()
    version = subprocess.run([str(ffmpeg), '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env, check=True, text=True).stdout
    probe_version = subprocess.run([str(ffprobe), '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=env, check=True, text=True).stdout
    encoder_help = {}
    for recipe in CODECS.values():
        name = recipe['encoder']
        completed = subprocess.run([str(ffmpeg), '-hide_banner', '-h', 'encoder=' + name],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, check=False, text=True)
        require(completed.returncode == 0 and ('Encoder ' + name) in completed.stdout,
                'required encoder unavailable: ' + name)
        encoder_help[name] = completed.stdout
    linked = subprocess.run(['/usr/bin/ldd', str(ffmpeg)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env, check=True, text=True).stdout
    codec_libraries = {}
    for line in linked.splitlines():
        match = re.search(r'=>\s+(/\S+)\s+\(', line)
        if match and any(token in Path(match.group(1)).name for token in ('mp3lame', 'opus')):
            entry = binding(Path(match.group(1)).resolve())
            codec_libraries[Path(match.group(1)).name] = entry
    require(any('mp3lame' in key for key in codec_libraries)
            and any('opus' in key for key in codec_libraries), 'linked codec libraries unresolved')
    return {'ffmpeg': ffmpeg_entry, 'ffprobe': ffprobe_entry,
            'ffmpeg_version': version, 'ffprobe_version': probe_version,
            'encoder_help': encoder_help, 'ldd': linked,
            'codec_libraries': codec_libraries}


def tool_bindings(toolchain):
    current = {'ffmpeg': binding(toolchain['ffmpeg']['path']),
               'ffprobe': binding(toolchain['ffprobe']['path']),
               'codec_libraries': {name: binding(entry['path'])
                                   for name, entry in toolchain['codec_libraries'].items()}}
    expected = {'ffmpeg': toolchain['ffmpeg'], 'ffprobe': toolchain['ffprobe'],
                'codec_libraries': toolchain['codec_libraries']}
    require(current == expected, 'toolchain binary/library changed')
    return current


def encode_command(ffmpeg, input_path, output_path, recipe):
    return [str(ffmpeg), '-hide_banner', '-nostdin', '-n', '-loglevel', 'info',
            '-i', str(input_path), '-map', '0:a:0', '-vn', '-sn', '-dn',
            '-ac', '1', '-ar', str(SAMPLE_RATE), '-threads', '1',
            '-c:a', recipe['encoder'], '-b:a', recipe['bitrate'],
            *recipe['encoder_options'], '-map_metadata', '-1', str(output_path)]


def decode_command(ffmpeg, encoded_path, output_path):
    return [str(ffmpeg), '-hide_banner', '-nostdin', '-n', '-loglevel', 'info',
            '-i', str(encoded_path), '-map', '0:a:0', '-vn', '-sn', '-dn',
            '-ac', '1', '-ar', str(SAMPLE_RATE), '-threads', '1',
            '-c:a', 'pcm_f32le', '-map_metadata', '-1', str(output_path)]


def probe_command(ffprobe, path):
    return [str(ffprobe), '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]


def file_products(root):
    return {path.relative_to(root).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'COMMIT.json'}


def run_probe(ffmpeg, ffprobe, ffmpeg_sha, ffprobe_sha, output):
    output = Path(output)
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and output.parent.is_dir(), 'new absolute output directory required')
    tools = tool_record(ffmpeg, ffprobe, ffmpeg_sha, ffprobe_sha)
    output.mkdir()
    for name in ('synthetic', 'encoded', 'decoded', 'command_logs'):
        (output / name).mkdir()
    signals, rows = synthetic_signals(), []
    for signal_name, input_samples in signals.items():
        input_path = output / 'synthetic' / (signal_name + '.wav')
        write_new(input_path, wav_bytes(input_samples))
        require(np.array_equal(read_float_wav(input_path), input_samples), 'synthetic FLOAT WAV changed')
        for codec_name, recipe in CODECS.items():
            encoded_path = output / 'encoded' / (signal_name + '__' + codec_name + recipe['extension'])
            decoded_path = output / 'decoded' / (signal_name + '__' + codec_name + '.wav')
            encode_log = output / 'command_logs' / (signal_name + '__' + codec_name + '__encode.json')
            decode_log = output / 'command_logs' / (signal_name + '__' + codec_name + '__decode.json')
            probe_log = output / 'command_logs' / (signal_name + '__' + codec_name + '__probe.json')
            encode_receipt, _ = run_command(encode_command(ffmpeg, input_path, encoded_path, recipe), encode_log)
            decode_receipt, _ = run_command(decode_command(ffmpeg, encoded_path, decoded_path), decode_log)
            _, encoded_probe = run_command(probe_command(ffprobe, encoded_path), probe_log, parse_json=True)
            input_read, input_wav_format = read_float_wav(input_path, return_info=True)
            decoded_samples, decoded_wav_format = read_float_wav(decoded_path, return_info=True)
            require(np.array_equal(input_read, input_samples), 'input changed before measurement')
            rows.append({'signal': signal_name, 'codec': codec_name, 'recipe': recipe,
                         'input': binding(input_path), 'encoded': binding(encoded_path),
                         'decoded': binding(decoded_path), 'encode_log': binding(encode_log),
                         'decode_log': binding(decode_log), 'probe_log': binding(probe_log),
                         'encode_command': encode_receipt['command'],
                         'decode_command': decode_receipt['command'], 'encoded_probe': encoded_probe,
                         'input_wav_format': input_wav_format,
                         'decoded_wav_format': decoded_wav_format,
                         'input_float32_sha256': hashlib.sha256(input_samples.astype('<f4').tobytes()).hexdigest(),
                         'decoded_float32_sha256': hashlib.sha256(decoded_samples.astype('<f4').tobytes()).hexdigest(),
                         'input_stats': signal_stats(input_samples), 'decoded_stats': signal_stats(decoded_samples),
                         'raw_comparison': compare_raw(input_samples, decoded_samples)})
    here = Path(__file__).resolve().parent
    results = {'version': VERSION, 'status': 'synthetic_codec_roundtrip_characterized_not_admission',
               'sample_rate_hz': SAMPLE_RATE, 'channels': 1, 'input_subtype': 'FLOAT',
               'decoded_subtype': 'FLOAT', 'samples_per_fixture': SAMPLES,
               'operation_order': OPERATION_ORDER, 'codec_recipes': CODECS,
               'toolchain': tools, 'rows': rows,
               'runtime': {'python': sys.version, 'platform': platform.platform(),
                           'numpy': np.__version__}, **SCOPE}
    write_json(output / 'results.json', results)
    products = file_products(output)
    toolchain_end = tool_bindings(tools)
    commit = {'version': VERSION, 'status': STATUS, 'products': products,
              'products_sha256': value_hash(products), 'results_sha256': value_hash(results),
              'bindings': {'runner': binding(Path(__file__).resolve()),
                           'tests': binding(here / ('test_' + VERSION + '.py'))},
              'toolchain_bindings_before': {'ffmpeg': tools['ffmpeg'], 'ffprobe': tools['ffprobe'],
                                            'codec_libraries': tools['codec_libraries']},
              'toolchain_bindings_end': toolchain_end,
              'row_count': len(rows), 'signal_count': len(signals),
              'codec_count': len(CODECS), **SCOPE}
    write_json(output / 'COMMIT.json', commit)
    return verify_result(output, digest(output / 'COMMIT.json'))


def verify_result(root, commit_sha):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(),
            'safe result root required')
    require(hash_string(commit_sha), 'caller COMMIT SHA required')
    commit_path = safe_file(root / 'COMMIT.json')
    require(digest(commit_path) == commit_sha, 'COMMIT SHA mismatch')
    commit = json.loads(commit_path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    require(value_hash(commit) == commit_sha and commit.get('version') == VERSION
            and commit.get('status') == STATUS and all(commit.get(k) == v for k, v in SCOPE.items()),
            'COMMIT schema/scope')
    require(commit.get('toolchain_bindings_before') == commit.get('toolchain_bindings_end'),
            'toolchain changed during probe')
    products = file_products(root)
    require(commit.get('products') == products and commit.get('products_sha256') == value_hash(products),
            'product inventory/hash changed')
    for entry in commit['bindings'].values():
        require(binding(entry['path']) == entry, 'code/test binding changed')
    results = json.loads((root / 'results.json').read_text(),
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    require(value_hash(results) == commit.get('results_sha256')
            and results.get('version') == VERSION and results.get('codec_recipes') == CODECS
            and results.get('operation_order') == OPERATION_ORDER
            and all(results.get(k) == v for k, v in SCOPE.items()), 'results schema/scope/hash')
    signals = synthetic_signals()
    require(tool_bindings(results['toolchain']) == commit['toolchain_bindings_end'],
            'result toolchain/end binding mismatch')
    require(len(results.get('rows', [])) == commit.get('row_count') == len(signals) * len(CODECS),
            'result row count')
    expected_pairs = [(signal, codec) for signal in signals for codec in CODECS]
    require([(row['signal'], row['codec']) for row in results['rows']] == expected_pairs,
            'result row order/identity')
    for row in results['rows']:
        stem = row['signal'] + '__' + row['codec']
        recipe = CODECS[row['codec']]
        expected_paths = {'input': root / 'synthetic' / (row['signal'] + '.wav'),
                          'encoded': root / 'encoded' / (stem + recipe['extension']),
                          'decoded': root / 'decoded' / (stem + '.wav'),
                          'encode_log': root / 'command_logs' / (stem + '__encode.json'),
                          'decode_log': root / 'command_logs' / (stem + '__decode.json'),
                          'probe_log': root / 'command_logs' / (stem + '__probe.json')}
        require(all(row[key]['path'] == str(path) for key, path in expected_paths.items()),
                'row artifact path mapping changed')
        source = read_float_wav(row['input']['path'])
        decoded = read_float_wav(row['decoded']['path'])
        require(np.array_equal(source, signals[row['signal']]), 'synthetic generator/input mismatch')
        require(binding(row['input']['path']) == row['input']
                and binding(row['encoded']['path']) == row['encoded']
                and binding(row['decoded']['path']) == row['decoded']
                and binding(row['encode_log']['path']) == row['encode_log']
                and binding(row['decode_log']['path']) == row['decode_log']
                and binding(row['probe_log']['path']) == row['probe_log']
                and compare_raw(source, decoded) == row['raw_comparison']
                and signal_stats(source) == row['input_stats']
                and signal_stats(decoded) == row['decoded_stats'], 'row artifact/measurement mismatch')
        input_read, input_format = read_float_wav(row['input']['path'], return_info=True)
        decoded_read, decoded_format = read_float_wav(row['decoded']['path'], return_info=True)
        require(np.array_equal(input_read, source) and np.array_equal(decoded_read, decoded)
                and input_format == row['input_wav_format']
                and decoded_format == row['decoded_wav_format'], 'WAV format evidence changed')
        require(row['encode_command'] == encode_command(results['toolchain']['ffmpeg']['path'],
                row['input']['path'], row['encoded']['path'], CODECS[row['codec']])
                and row['decode_command'] == decode_command(results['toolchain']['ffmpeg']['path'],
                row['encoded']['path'], row['decoded']['path']), 'codec command recipe changed')
        logs = {key: json.loads(Path(row[key]['path']).read_text())
                for key in ('encode_log', 'decode_log', 'probe_log')}
        require(logs['encode_log']['command'] == row['encode_command']
                and logs['decode_log']['command'] == row['decode_command']
                and logs['probe_log']['command'] == probe_command(results['toolchain']['ffprobe']['path'],
                                                                    row['encoded']['path'])
                and all(record.get('returncode') == 0 and record.get('environment') == command_environment()
                        for record in logs.values()), 'command log provenance changed')
    return {'commit': binding(commit_path), 'results': results, 'products': products}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--ffmpeg-sha256', required=True)
    parser.add_argument('--ffprobe', required=True)
    parser.add_argument('--ffprobe-sha256', required=True)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    tools = tool_record(args.ffmpeg, args.ffprobe, args.ffmpeg_sha256, args.ffprobe_sha256)
    if args.mode == 'preflight':
        result = {'version': VERSION, 'status': 'synthetic_codec_preflight_codecs_available_no_media_read_or_written',
                  'toolchain': tools, 'codec_recipes': CODECS, **SCOPE}
    else:
        require(args.output is not None, '--output required for run')
        proof = run_probe(args.ffmpeg, args.ffprobe, args.ffmpeg_sha256, args.ffprobe_sha256,
                          args.output.resolve())
        result = {'version': VERSION, 'status': STATUS, 'commit': proof['commit'],
                  'row_count': len(proof['results']['rows']), **SCOPE}
    print(canonical(result).decode().strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
