"""Exact30 inside a frozen native60 interval, with explicit evidence strength.

This module writes nothing and performs no cohort/feature/classifier admission.
Two fresh SoundFile handles decode from frame zero through region_end only.
They are repeat observations with one decoder, not independent algorithms or
full-stream completeness evidence. Whole-file byte hashing is separate from
this bounded waveform observation. No seeking or post-region decoding occurs.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import platform

import standardize_native30_v1 as dsp

VERSION = 'standardize_native30_frozen_region_v1'
DSP_SHA = '2e05a4d0d2e7789888aa41a53bece46e8de97fad34ef34f21354d165e77b55be'
BOUNDS_ONLY = 'frozen_native_frame_bounds_from_receipt'
HASH_VERIFIED = 'verified_native_interval_hash'
REGION_SECONDS = 60
CONFIG = {**copy.deepcopy(dsp.CONFIG), 'version': VERSION,
          'default_region': 'forbidden_explicit_frozen_native60_required',
          'approved_region': 'explicit_frozen_native_bounds_and_evidence_strength',
          'crop_rule': 'center_exact30_inside_frozen_native60_before_resampling',
          'decode': 'two_fresh_sequential_passes_from_zero_through_region_end_only',
          'region_seconds': REGION_SECONDS, 'full_stream_completeness_claim': False,
          'historical_hash_policy': 'mandatory_for_verified_native_interval_hash; null_for_bounds_only',
          'resample_crop_dependency_version': dsp.VERSION,
          'resample_crop_dependency_sha256': DSP_SHA}


def region_coordinates(native_rate, region_start_frame, region_frames, evidence_strength,
                       expected_region_float64_sha256):
    dsp.require(dsp.integer(native_rate, 1) and dsp.integer(region_start_frame)
                and dsp.integer(region_frames, 1), 'native rate/region bounds must be nonnegative integers')
    dsp.require(region_frames == REGION_SECONDS * native_rate, 'frozen region must be exactly60 native seconds')
    dsp.require(evidence_strength in (BOUNDS_ONLY, HASH_VERIFIED), 'unsupported interval evidence strength')
    if evidence_strength == HASH_VERIFIED:
        dsp.require(dsp.hash_string(expected_region_float64_sha256), 'mandatory historical native60 hash missing/invalid')
    else:
        dsp.require(expected_region_float64_sha256 is None, 'bounds-only evidence must retain null historical hash')
    crop_frames = dsp.SECONDS * native_rate
    crop_start = region_start_frame + (region_frames - crop_frames) // 2
    return {'region_start_frame': region_start_frame, 'region_frames': region_frames,
            'region_end_frame_exclusive': region_start_frame + region_frames,
            'crop_start_frame': crop_start, 'crop_frames': crop_frames,
            'crop_end_frame_exclusive': crop_start + crop_frames,
            'crop_offset_within_native_region': crop_start - region_start_frame}


def read_region(path, native_rate, region_start_frame, region_frames):
    """Observe a bounded interval sequentially, including finite prefix checks.

    Successful return means exactly region_end frames have been decoded through.
    No read is requested after that boundary, including an empty-EOF probe.
    Direct helper calls receive the same strict native60 bounds validation.
    """
    coordinates = region_coordinates(native_rate, region_start_frame, region_frames, BOUNDS_ONLY, None)
    end = coordinates['region_end_frame_exclusive']
    through, calls, pieces = 0, 0, []
    with dsp.sf.SoundFile(path) as stream:
        dsp.require(stream.samplerate == native_rate and stream.channels == 2,
                    'native rate/channels disagree with frozen provenance')
        header = {'sample_rate_hz': int(stream.samplerate), 'channels': int(stream.channels),
                  'format': stream.format, 'subtype': stream.subtype}
        while through < end:
            request = min(dsp.BLOCK, end - through)
            block = stream.read(request, dtype='float64', always_2d=True)
            calls += 1
            dsp.require(isinstance(block, dsp.np.ndarray) and block.dtype == dsp.np.dtype('float64')
                        and block.ndim == 2 and block.shape[1] == 2 and len(block) <= request
                        and dsp.np.isfinite(block).all(), 'invalid/nonfinite bounded sequential decode')
            dsp.require(len(block) > 0, 'native stream ended before frozen region_end; padding forbidden')
            first = max(through, region_start_frame)
            if first < through + len(block):
                pieces.append(block[first - through:].copy())
            through += len(block)
    retained = dsp.np.concatenate(pieces, axis=0) if pieces else dsp.np.empty((0, 2), dsp.np.float64)
    dsp.require(retained.shape == (region_frames, 2) and through == end, 'incomplete frozen native60 capture')
    return retained, {'header_without_duration_claim': header,
                      'decoded_through_frame_exclusive': through,
                      'prefix_frames_before_region': region_start_frame,
                      'retained_native_region_frames': region_frames, 'read_calls': calls,
                      'observation': 'bounded_sequential_prefix_through_region_end',
                      'empty_eof_probe_performed': False, 'full_stream_completeness_established': False,
                      'observed_region_float64_sha256': dsp.pcm_hash(retained)}


def _dependency_binding():
    path = Path(dsp.__file__).resolve()
    sha = dsp.digest(path)
    dsp.require(sha == DSP_SHA, 'pinned resample_crop dependency changed')
    return {'path': str(path), 'sha256': sha, 'version': dsp.VERSION}


def standardize(path, expected_sha256, native_rate, native_channels, *, region_start_frame,
                region_frames, evidence_strength, expected_region_float64_sha256):
    """Return FLOAT-compatible exact30 and a bounded-observation audit.

    Native region frames and start come from frozen receipts, never from a
    container header, a full-source center estimate or a standardized60 WAV.
    Callers must bind those receipt identities and the source/evidence-strength
    mapping; this scalar API does not infer a dataset's identity from its path.
    """
    dsp.require(dsp.hash_string(expected_sha256) and type(native_channels) is int and native_channels == 2,
                'valid source SHA and exactly two native channels required')
    coordinates = region_coordinates(native_rate, region_start_frame, region_frames,
                                     evidence_strength, expected_region_float64_sha256)
    dependency = _dependency_binding()
    path = Path(path).absolute()
    dsp.require(path.is_file() and not path.is_symlink(), 'regular non-symlink native source required')
    before = dsp.signature(path)
    dsp.require(dsp.digest(path) == expected_sha256, 'native source SHA256 mismatch')
    first_values, first = read_region(path, native_rate, region_start_frame, region_frames)
    second_values, second = read_region(path, native_rate, region_start_frame, region_frames)
    first_hash, second_hash = dsp.pcm_hash(first_values), dsp.pcm_hash(second_values)
    dsp.require(first_hash == first['observed_region_float64_sha256'] and
                second_hash == second['observed_region_float64_sha256'], 'pass arrays/hash disagree')
    dsp.require(first == second and first_hash == second_hash, 'native60 changed across sequential passes')
    historical_match = None
    if evidence_strength == HASH_VERIFIED:
        dsp.require(second_hash == expected_region_float64_sha256, 'historical native60 interval hash mismatch')
        historical_match = True
    offset = coordinates['crop_offset_within_native_region']
    crop = second_values[offset:offset + coordinates['crop_frames']]
    result = dsp.resample_crop(crop, native_rate)
    dsp.require(dsp.signature(path) == before and dsp.digest(path) == expected_sha256,
                'native source changed during bounded processing')
    dsp.require(_dependency_binding() == dependency, 'resample_crop dependency changed during processing')
    audit = {'status': 'verified_bounded_native60_to30_DSP_only', 'configuration': copy.deepcopy(CONFIG),
             'source_path': str(path), 'source_sha256_before': expected_sha256,
             'source_sha256_after': expected_sha256, 'source_stat_signature': list(before),
             'source_byte_hash_scope': 'whole_file_bytes_not_full_waveform_decode',
             'native_rate_hz': native_rate, 'native_channels': native_channels,
             'region_kind': 'frozen_native60_interval', 'coordinates': coordinates,
             'evidence_strength': evidence_strength,
             'historical_expected_region_float64_sha256': expected_region_float64_sha256,
             'observed_current_region_float64_sha256': second_hash,
             'historical_region_hash_match': historical_match,
             'historical_region_waveform_hash_available': evidence_strength == HASH_VERIFIED,
             'two_sequential_observations_equal': True, 'sequential_passes': [first, second],
             'decoded_through_frame_exclusive': coordinates['region_end_frame_exclusive'],
             'full_stream_completeness_established': False, 'independent_decoder_agreement_established': False,
             'native_crop_float64_sha256': dsp.pcm_hash(crop),
             'output_waveform_float32_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
             'output_frames': dsp.FRAMES, 'output_rate_hz': dsp.RATE, 'output_channels': 2,
             'peak_absolute': float(dsp.np.max(dsp.np.abs(result))),
             'samples_abs_above_one': int(dsp.np.count_nonzero(dsp.np.abs(result) > 1)),
             'sample_count': int(result.size), 'resample_crop_dependency': dependency,
             'classifier_admitted': False, 'cohort_admitted': False, 'feature_extraction_authorized': False,
             'runtime': {'python': platform.python_version(), 'numpy': dsp.np.__version__,
                         'scipy': dsp.scipy.__version__, 'soundfile': dsp.sf.__version__,
                         'libsndfile': dsp.sf.__libsndfile_version__}}
    return result, audit
