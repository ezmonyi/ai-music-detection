"""Uniform native-origin exact30 DSP, with explicit optional approved region.

This component admits no cohort, labels, or decoder agreement. Its caller must
bind native-origin provenance and choose any approved region before execution.
Two sequential passes reach true empty EOF; seeking/header duration do not
choose crop coordinates. No audio is written here.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf

VERSION = 'standardize_native30_v1'
RATE = 44100
SECONDS = 30
FRAMES = RATE * SECONDS
BLOCK = 65536
CONFIG = {
    'version': VERSION, 'duration_s': SECONDS, 'output_rate_hz': RATE,
    'channels': 2, 'native_mono_policy': 'reject',
    'native_multichannel_policy': 'reject', 'crop_rule': 'floor_center_of_declared_region',
    'default_region': 'complete_sequential_native_decode',
    'approved_region': 'explicit_native_frames_and_float64_waveform_hash',
    'resampler': 'scipy.signal.resample_poly', 'window': ['kaiser', 5.0],
    'padtype': 'constant', 'resampling_boundary_extension': 'zero_filter_boundary_only',
    'short_audio_padding': False, 'limiting': False, 'normalization': False,
    'dc_removal': False, 'channel_mixing': False, 'spectral_correction': False,
    'output_dtype': 'little_endian_float32', 'intended_output_format': 'WAV_FLOAT',
    'decode': 'float64_two_sequential_passes_to_empty_EOF',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pcm_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes()).hexdigest()


def signature(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def center_coordinates(actual_frames, native_rate, region=None):
    """Choose native integer frames, never rounded seconds or header length."""
    require(integer(actual_frames) and integer(native_rate, 1), 'invalid native frame/rate count')
    if region is None:
        start, frames = 0, actual_frames
    else:
        require(isinstance(region, dict) and set(region) == {'start_frame', 'frames', 'float64_sha256'},
                'approved region requires exact bounds and float64 hash')
        start, frames = region['start_frame'], region['frames']
        require(integer(start) and integer(frames, 1) and hash_string(region['float64_sha256']),
                'invalid approved region fields')
        require(start + frames <= actual_frames, 'approved region extends beyond actual EOF')
    count = SECONDS * native_rate
    require(frames >= count, 'native region shorter than30s: padding forbidden')
    crop_start = start + (frames - count) // 2
    return {'region_start_frame': start, 'region_frames': frames,
            'crop_start_frame': crop_start, 'crop_frames': count,
            'crop_end_frame_exclusive': crop_start + count}


def sequential(path, expected_rate, capture=None):
    """Full finite decode; optionally retain a bounded frame interval."""
    total, calls = 0, 0
    full = hashlib.sha256()
    pieces = []
    with sf.SoundFile(path) as stream:
        require(stream.samplerate == expected_rate and stream.channels == 2,
                'native rate/channels disagree with provenance; no channel conversion')
        header = {'frames': int(stream.frames), 'sample_rate_hz': int(stream.samplerate),
                  'channels': int(stream.channels), 'format': stream.format, 'subtype': stream.subtype}
        while True:
            block = stream.read(BLOCK, dtype='float64', always_2d=True)
            calls += 1
            require(block.ndim == 2 and block.shape[1] == 2 and len(block) <= BLOCK
                    and np.isfinite(block).all(), 'invalid/nonfinite sequential audio')
            if not len(block):
                break
            full.update(np.ascontiguousarray(block, dtype='<f8').tobytes())
            if capture is not None:
                a, b = max(total, capture[0]), min(total + len(block), capture[1])
                if b > a:
                    pieces.append(block[a-total:b-total].copy())
            total += len(block)
    values = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 2), np.float64)
    return values, {'header': header, 'actual_frames': total,
                    'float64_pcm_sha256': full.hexdigest(), 'read_calls_including_empty_eof': calls,
                    'empty_eof_observed': True}


def resample_crop(crop, native_rate):
    require(integer(native_rate, 1) and isinstance(crop, np.ndarray)
            and crop.dtype == np.dtype('float64') and crop.shape == (SECONDS*native_rate, 2)
            and np.isfinite(crop).all(), 'exact finite float64 native30 stereo required')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        if native_rate == RATE:
            filtered = crop
        else:
            divisor = math.gcd(native_rate, RATE)
            filtered = resample_poly(crop, RATE//divisor, native_rate//divisor,
                                     axis=0, window=('kaiser', 5.0), padtype='constant')
        require(filtered.shape == (FRAMES, 2) and np.isfinite(filtered).all(), 'resampling output mismatch')
        result = np.ascontiguousarray(filtered, dtype='<f4')
    require(np.isfinite(result).all(), 'float32 conversion overflow')
    return result


def standardize(path, expected_sha256, native_rate, native_channels, *, approved_region=None):
    """Return float32 native-center30 view and proof; never write/mutate files.

    An approved region's float64 hash refers to native-rate decoded samples,
    not an existing resampled/FLOAT view. Full-stream decoder agreement with
    another decoder is not inferred, even when headers equal sequential EOF.
    """
    require(hash_string(expected_sha256) and integer(native_rate, 1)
            and type(native_channels) is int and native_channels == 2, 'invalid native provenance')
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'regular non-symlink native input required')
    before = signature(path)
    require(digest(path) == expected_sha256, 'source SHA256 mismatch')
    _, first = sequential(path, native_rate)
    coords = center_coordinates(first['actual_frames'], native_rate, approved_region)
    # Approved-region hash verification needs its full declared interval.
    start = coords['region_start_frame'] if approved_region is not None else coords['crop_start_frame']
    count = coords['region_frames'] if approved_region is not None else coords['crop_frames']
    values, second = sequential(path, native_rate, (start, start + count))
    require(first == second, 'sequential decode changed across passes')
    require(values.shape == (count, 2), 'incomplete captured interval')
    region_hash = None
    if approved_region is not None:
        region_hash = pcm_hash(values)
        require(region_hash == approved_region['float64_sha256'], 'approved native region hash mismatch')
    offset = coords['crop_start_frame'] - start
    crop = values[offset:offset + coords['crop_frames']]
    result = resample_crop(crop, native_rate)
    require(signature(path) == before and digest(path) == expected_sha256,
            'native source changed during processing')
    audit = {
        'status': 'verified_DSP_only_not_cohort_admission', 'configuration': CONFIG,
        'source_path': str(path), 'source_sha256_before': expected_sha256,
        'source_sha256_after': expected_sha256, 'source_stat_signature': list(before),
        'native_rate_hz': native_rate, 'native_channels': native_channels,
        'region_kind': 'approved_native_interval' if approved_region is not None else 'full_native_sequential_decode',
        'approved_region_float64_sha256': region_hash, 'coordinates': coords,
        'sequential_passes_identical': True, 'sequential_decode': first,
        'header_minus_actual_frames': first['header']['frames'] - first['actual_frames'],
        'native_crop_float64_sha256': pcm_hash(crop),
        'output_waveform_float32_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
        'output_frames': FRAMES, 'output_rate_hz': RATE, 'output_channels': 2,
        'peak_absolute': float(np.max(np.abs(result))),
        'samples_abs_above_one': int(np.count_nonzero(np.abs(result)>1)),
        'sample_count': int(result.size), 'classifier_admitted': False,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'soundfile': sf.__version__,
                    'libsndfile': sf.__libsndfile_version__},
    }
    return result, audit
