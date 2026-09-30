"""Native30 input contract shared by uploads and reference-sample validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .algorithms import standardize_native30_v1 as native30

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_DURATION_SECONDS = 3600


@dataclass
class AudioView:
    stereo: np.ndarray
    mono_fh: np.ndarray
    mono_bc: np.ndarray
    metadata: dict
    provenance: dict


def load_audio(path: Path, *, display_name: str | None = None,
               approved_region: dict | None = None) -> AudioView:
    """Exactly reproduce archived Native30 DSP; never duplicate mono or pad.

    An approved historical region must include its native FLOAT64 PCM hash.
    Normal uploads use the full native recording's floor-centred 30 seconds.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Audio must be a regular file, not a symlink.')
    if not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
        raise ValueError('Audio file must be nonempty and at most 100 MiB.')
    try:
        info = sf.info(path)
    except (RuntimeError, sf.LibsndfileError) as error:
        raise ValueError('Unsupported or invalid audio. Use WAV, FLAC, MP3 or OGG supported by libsndfile.') from error
    if info.channels != 2:
        raise ValueError('This validated Native30 protocol requires native stereo (2 channels); mono is not duplicated.')
    if not 8000 <= info.samplerate <= 192000:
        raise ValueError('Sample rate is outside the supported 8–192 kHz range.')
    if not 30 <= info.duration <= MAX_DURATION_SECONDS:
        raise ValueError('This protocol requires 30–3600 seconds of audio; short recordings are never padded.')
    digest = native30.digest(path)
    stereo32, proof = native30.standardize(path, digest, info.samplerate, info.channels,
                                          approved_region=approved_region)
    # Inspect the same native crop before filtering: resampling a constant DC
    # input creates boundary ringing that must not masquerade as music.
    coordinates = proof['coordinates']
    native_crop, native_proof = native30.sequential(path, info.samplerate, capture=(
        coordinates['crop_start_frame'], coordinates['crop_end_frame_exclusive']))
    if native_proof['float64_pcm_sha256'] != proof['sequential_decode']['float64_pcm_sha256']:
        raise ValueError('Decoded source changed during input quality validation.')
    native_range = float(np.max(np.ptp(native_crop, axis=0)))
    if native_range < 1e-8:
        raise ValueError('The native 30-second region is silent or constant DC; no authorship probability is meaningful.')
    proof['native_signal_peak_to_peak'] = native_range
    del native_crop
    stereo = stereo32.astype(np.float64)
    mono = stereo.mean(axis=1)
    divisor = math.gcd(44100, 16000)
    mono_fh = resample_poly(mono - mono.mean(), 16000 // divisor, 44100 // divisor,
                            window=('kaiser', 5.0), padtype='constant')
    mono_bc = resample_poly(mono, 16000 // divisor, 44100 // divisor,
                            window=('kaiser', 5.0), padtype='constant')
    if mono_fh.shape != (480000,) or mono_bc.shape != (480000,):
        raise ValueError('Unexpected resampled analysis length.')
    proof.pop('source_path', None)
    proof.pop('source_stat_signature', None)
    proof['fh_waveform_float64_sha256'] = hashlib.sha256(np.asarray(mono_fh, dtype='<f8').tobytes()).hexdigest()
    proof['upload_policy'] = 'native stereo; floor-centred exact30; no peak normalization, limiting or padding'
    metadata = dict(name=display_name or path.name, duration=proof['sequential_decode']['actual_frames'] / info.samplerate,
                    sample_rate=info.samplerate, channels=info.channels, sha256=digest,
                    analysis_duration=30, analysis_sample_rate=44100,
                    crop_start_seconds=proof['coordinates']['crop_start_frame'] / info.samplerate)
    return AudioView(stereo, mono_fh, mono_bc, metadata, proof)
