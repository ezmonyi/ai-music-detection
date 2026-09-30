"""Strict adapter around the byte-identical archived S8/D3/R3/P6 definitions.

Inference failures never become model missing-value indicators. Only the
original scientific availability rules may produce null measurements.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from .algorithms import expanded_feature_definitions as core
from .scoring import FAMILIES

BIAS_SHA256 = 'bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e'
CORE_SHA256 = '8b9745085c7518e84613b2ba499dbae775a57e4dcf95670c5e86a05ab524ff00'
NEURAL_FAMILIES = ('S', 'D', 'R', 'P')
STEMS = ('bass', 'drums', 'other', 'vocals')


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def selected_families(families) -> list[str]:
    families = list(families)
    if not families or len(set(families)) != len(families) or not set(families) <= set(NEURAL_FAMILIES):
        raise ValueError('Select nonempty, nonduplicated S/D/R/P families.')
    return [key for key in NEURAL_FAMILIES if key in families]


def check_audio(path: Path, *, subtype: str) -> None:
    if not Path(path).is_file() or Path(path).is_symlink():
        raise ValueError(f'Missing regular inference audio: {path}')
    info = sf.info(path)
    if (info.samplerate, info.channels, info.frames, info.subtype) != (44100, 2, 1323000, subtype):
        raise ValueError(f'Invalid Native30 {subtype} audio (no padding/resampling repair permitted): {path}')


def load_bias(path: Path) -> np.ndarray:
    if digest(path) != BIAS_SHA256:
        raise ValueError('MUSDB spectral-correction SHA256 mismatch.')
    with np.load(path, allow_pickle=False) as payload:
        frequencies = np.asarray(payload['frequencies_hz'], dtype=np.float64)
        bias = np.asarray(payload['selected_bias_db'], dtype=np.float64)
    expected = np.fft.rfftfreq(core.N_FFT, 1.0 / core.SR)
    if (frequencies.shape != expected.shape or not np.allclose(frequencies, expected)
            or bias.shape != expected.shape or not np.isfinite(bias).all()):
        raise ValueError('Invalid frozen MUSDB frequency grid or bias.')
    return bias


def extract_products(*, mix_path: Path, families, native_sample_rate: int,
                     stems_dir: Path | None = None, beats_path: Path | None = None,
                     structure_path: Path | None = None, bias_path: Path | None = None) -> dict:
    """Reduce real neural products using the original precision and file views."""
    selected = selected_families(families)
    if digest(Path(core.__file__)) != CORE_SHA256:
        raise ValueError('Frozen feature implementation changed.')
    check_audio(mix_path, subtype='FLOAT')
    if 'S' in selected and native_sample_rate < 16000:
        raise ValueError('S8 requires native sample rate >=16000 Hz; resampling cannot create bandwidth.')
    features, quality, metadata = {}, {}, {}
    hashes = {'mix': digest(mix_path)}
    if set(selected) & {'S', 'D', 'P'}:
        if stems_dir is None:
            raise ValueError('Real Demucs stems are required.')
        for name in STEMS:
            path = stems_dir / (name + '.wav')
            check_audio(path, subtype='PCM_16')
            hashes[name] = digest(path)
    if 'D' in selected:
        # Preserve float32 addition order of the archived extractor.
        nonvocal = [core.read_audio(stems_dir / (name + '.wav'), 30, strict=True)
                    for name in ('bass', 'drums', 'other')]
        values = core.dynamics_features(nonvocal[0] + nonvocal[1] + nonvocal[2])
        features.update({'d__' + k: v for k, v in values.items()})
        quality['D'] = 'eligible' if all(math.isfinite(v) for v in values.values()) else 'unavailable_fewer_than_8_events'
    if 'S' in selected:
        if bias_path is None:
            raise ValueError('Frozen MUSDB spectral correction is required for S.')
        bias = load_bias(bias_path)
        vocals = core.read_spectral_audio(stems_dir / 'vocals.wav', 30)
        _, values = core.spectral_families(vocals, bias)
        features.update({'s8__' + k: v for k, v in values.items()})
        metadata['vocal_activity'] = core.vocal_activity(core.read_audio(mix_path, 30, strict=True), vocals)
        metadata['s8_native_eligible'] = True
        metadata['vocal_activity_used_to_filter_primary_features'] = False
        hashes['musdb_bias'] = BIAS_SHA256
        quality['S'] = 'eligible'
    if set(selected) & {'R', 'P'}:
        if beats_path is None or not beats_path.is_file():
            raise ValueError('Missing Beat This output is an inference failure, not scientific missingness.')
        # The frozen loader filters timestamps to the analysis interval, which
        # would silently discard NaN/+Inf before the post-load checks below.
        # Validate raw products first; successful empty files remain legitimate.
        if beats_path.read_text().strip():
            raw_beats = np.loadtxt(beats_path, ndmin=2)
            if not np.isfinite(raw_beats).all():
                raise ValueError('Nonfinite raw Beat This output is an inference failure.')
        beats, numbers = core.load_beats(beats_path, 30)
        if (not np.isfinite(beats).all() or (beats < 0).any()
                or (np.diff(beats) <= 0).any() or (numbers < 1).any()):
            raise ValueError('Malformed beat output.')
        hashes['beats'] = digest(beats_path)
        metadata.update(n_beats=len(beats), n_downbeats=int((numbers == 1).sum()))
    if 'R' in selected:
        values = core.rhythm_features(beats)
        features.update({'r__' + k: v for k, v in values.items()})
        quality['R'] = 'eligible' if all(math.isfinite(v) for v in values.values()) else 'unavailable_fewer_than_16_beats_or_invalid_intervals'
    if 'P' in selected:
        if structure_path is None or not structure_path.is_file():
            raise ValueError('Missing All-In-One structure is an inference failure, not scientific missingness.')
        payload = json.loads(structure_path.read_text())
        if not isinstance(payload.get('segments'), list):
            raise ValueError('Malformed All-In-One segments.')
        # Cleaning in the frozen core clips infinities and drops NaN values.
        # Reject malformed raw boundaries without changing that historical core.
        try:
            raw_boundaries = [float(segment[key]) for segment in payload['segments']
                              for key in ('start', 'end')]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Malformed raw All-In-One segment boundaries.') from error
        if not np.isfinite(raw_boundaries).all():
            raise ValueError('Nonfinite raw All-In-One boundaries are an inference failure.')
        boundaries = core.allinone_boundaries(structure_path, 30)
        if not np.isfinite(boundaries).all():
            raise ValueError('Nonfinite All-In-One boundaries.')
        values = core.phrase_features(boundaries, beats[numbers == 1])
        features.update({'p__' + k: v for k, v in values.items()})
        hashes['structure'] = digest(structure_path)
        metadata['n_sections'] = len(boundaries) - 1
        quality['P'] = 'eligible' if any(math.isfinite(values[name]) for name in core.P_FEATURES[:4]) else 'unavailable_fewer_than_3_sections_or_downbeat_spans'
    expected = [name for family in selected for name in FAMILIES[family]]
    if set(features) != set(expected) or any(math.isinf(value) for value in features.values()):
        raise ValueError('Feature schema mismatch or infinite measurement.')
    return {'features': {k: float(features[k]) if math.isfinite(features[k]) else None for k in expected},
            'quality': quality, 'metadata': metadata, 'product_sha256': hashes,
            'feature_source_sha256': CORE_SHA256}
