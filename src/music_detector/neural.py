"""Real, receipt-pinned Native30 neural analysis, with explicit offline setup.

    python -m music_detector.neural setup --install-dependencies --download-weights
    python -m music_detector.neural status

No generation model is installed. Analysis never downloads assets, installs
packages, substitutes features, or converts processing errors into missingness.
CPU is the portable default; CUDA preserves the historical precision options.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import numpy as np
import soundfile as sf

from .neural_features import (NEURAL_FAMILIES, check_audio, digest, extract_products,
                              selected_families)

ASSETS = Path(__file__).parent / 'assets'
MANIFEST = ASSETS / 'neural_runtime.json'
_INFERENCE_LOCK = threading.Lock()


class NeuralUnavailable(RuntimeError):
    """Explicit setup or runtime correction is needed before inference."""


def cache_path(cache_dir=None) -> Path:
    return Path(cache_dir or os.environ.get('MUSIC_DETECTOR_NEURAL_CACHE', '')
                or Path.home() / '.cache' / 'music-artifact-detector' / 'neural-v1').expanduser().resolve()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _needed(families) -> tuple[list[str], set[str]]:
    selected = selected_families(families)
    packages = {'numpy', 'scipy', 'soundfile', 'torch', 'torchaudio'}
    if set(selected) & {'S', 'D', 'P'}:
        packages |= {'demucs-infer', 'all-in-one-infer', 'librosa', 'madmom-infer', 'julius', 'einops'}
    if set(selected) & {'R', 'P'}:
        packages |= {'beat-this', 'rotary-embedding-torch', 'soxr', 'einops'}
    return selected, packages


def runtime_status(*, families=NEURAL_FAMILIES, cache_dir=None, verify_hashes=True) -> dict:
    """Read-only status. Does not import torch or contact any server."""
    selected, packages = _needed(families)
    manifest = _manifest()
    cache = cache_path(cache_dir)
    errors, versions, sources, assets = [], {}, {}, []
    for name in sorted(packages):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = actual
        if actual is None or actual.split('+')[0] != manifest['requirements'][name]:
            errors.append(f'{name}: expected {manifest["requirements"][name]}, found {actual}')
    for relative, entry in manifest['source_files'].items():
        if entry['distribution'] not in packages or versions.get(entry['distribution']) is None:
            continue
        path = Path(importlib.metadata.distribution(entry['distribution']).locate_file(relative))
        valid = path.is_file() and (not verify_hashes or digest(path) == entry['sha256'])
        sources[relative] = valid
        if not valid:
            errors.append('Source hash mismatch: ' + relative)
    for entry in manifest['assets']:
        if not set(entry['families']) & set(selected):
            continue
        path = cache / 'checkpoints' / entry['name']
        valid = path.is_file() and not path.is_symlink() and path.stat().st_size == entry['bytes']
        if valid and verify_hashes:
            valid = digest(path) == entry['sha256']
        assets.append({'name': entry['name'], 'ready': valid, 'bytes': entry['bytes']})
        if not valid:
            errors.append('Missing or invalid analysis weight: ' + entry['name'])
    if 'S' in selected:
        bias = ASSETS / manifest['bias']['name']
        if not bias.is_file() or digest(bias) != manifest['bias']['sha256']:
            errors.append('Frozen MUSDB spectral correction missing or changed.')
    if set(selected) & {'S', 'D', 'P'}:
        yaml_path = cache / 'checkpoints' / 'htdemucs.yaml'
        if not yaml_path.is_file() or yaml_path.read_text() != "models: ['955717e8']\n":
            errors.append('Missing/changed local htdemucs.yaml; rerun explicit setup.')
    return {'ready': not errors, 'families': selected, 'cache_dir': str(cache), 'errors': errors,
            'versions': versions, 'source_files_verified': all(sources.values()),
            'assets': assets, 'manifest_sha256': digest(MANIFEST),
            'setup_command': 'python -m music_detector.neural setup --install-dependencies --download-weights',
            'supported_devices': ['cpu', 'cuda'], 'limitations': manifest['limitations']}


def setup(*, install_dependencies=False, download_weights=False, families=NEURAL_FAMILIES,
          cache_dir=None, progress=print) -> dict:
    """Explicit local installation/download; verified assets stay outside git."""
    selected, packages = _needed(families)
    manifest = _manifest()
    if not install_dependencies and not download_weights:
        raise ValueError('Explicit --install-dependencies and/or --download-weights required.')
    if install_dependencies:
        requirements = [f'{name}=={version}' for name, version in manifest['requirements'].items()
                        if name in packages and name != 'all-in-one-infer']
        if 'all-in-one-infer' in packages:
            requirements.append('all-in-one-infer @ ' + manifest['allinone_source_url'])
        progress('Installing version-pinned analysis dependencies; no generation models.')
        subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements], check=True)
        # PyPI and the frozen git tree both identify as 3.1.0. Pip may otherwise
        # retain the wrong same-version wheel; only this package is reinstalled.
        if 'all-in-one-infer' in packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall',
                            'all-in-one-infer @ ' + manifest['allinone_source_url']], check=True)
    if download_weights:
        directory = cache_path(cache_dir) / 'checkpoints'
        directory.mkdir(parents=True, exist_ok=True)
        for entry in manifest['assets']:
            if not set(entry['families']) & set(selected):
                continue
            destination = directory / entry['name']
            if destination.exists():
                if destination.is_symlink() or digest(destination) != entry['sha256']:
                    raise NeuralUnavailable(f'Existing asset mismatch; preserve and inspect it: {destination}')
                continue
            progress('Downloading analysis weight: ' + entry['name'])
            fd, temporary_name = tempfile.mkstemp(prefix=entry['name'] + '.', suffix='.partial', dir=directory)
            os.close(fd)
            temporary = Path(temporary_name)
            # A failed transfer remains a named .partial file for diagnosis.
            request = urllib.request.Request(entry['url'], headers={'User-Agent': 'MusicArtifactDetector/0.1'})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
                raise NeuralUnavailable('Downloaded checkpoint hash/size mismatch: ' + str(temporary))
            temporary.replace(destination)
        # Equivalent to upstream remote/htdemucs.yaml, but resolves only locally.
        yaml_path = directory / 'htdemucs.yaml'
        if not yaml_path.exists():
            yaml_path.write_text("models: ['955717e8']\n")
        elif yaml_path.read_text() != "models: ['955717e8']\n":
            raise NeuralUnavailable('Unexpected local Demucs model resolution YAML.')
    return runtime_status(families=selected, cache_dir=cache_dir)


def _device(device: str) -> str:
    import torch
    if device == 'cpu':
        return device
    if device == 'mps':
        raise NeuralUnavailable('MPS is not validated for the frozen chain (upstream removed support); use CPU on Apple Silicon.')
    if device == 'cuda' or (device.startswith('cuda:') and device[5:].isdigit()):
        index = 0 if device == 'cuda' else int(device[5:])
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return device
        raise NeuralUnavailable('Requested CUDA device is unavailable.')
    raise ValueError('Device must be cpu, cuda, or cuda:N; no silent device fallback.')


def _separate(mix_path, root, cache, device, progress):
    from allin1_infer.stems import DemucsProvider
    from demucs_infer.pretrained import get_model
    yaml_path = cache / 'checkpoints' / 'htdemucs.yaml'
    if not yaml_path.is_file() or yaml_path.read_text() != "models: ['955717e8']\n":
        raise NeuralUnavailable('Missing/changed local htdemucs.yaml; rerun explicit setup.')
    provider = DemucsProvider('htdemucs', device, demucs_overlap=.25, demucs_fp16=False)
    provider._model = get_model('htdemucs', repo=cache / 'checkpoints').to(device).eval()
    try:
        return provider.get_stems(mix_path, root / 'demix', progress_callback=lambda text, fraction: progress(text))
    finally:
        provider.clear_model_cache()
        gc.collect()


def _track_beats(mix_path, output, cache, device):
    from beat_this.inference import File2Beats
    from beat_this.utils import save_beat_tsv
    tracker = File2Beats(checkpoint_path=str(cache / 'checkpoints' / 'beat_this-final0.ckpt'),
                        device=device, float16=device.startswith('cuda'), dbn=False)
    beats, downbeats = tracker(str(mix_path))
    # Direct typed serializer preserves the upstream time/beat-number representation,
    # including a successful empty output. No array truth-value workaround/surrogate.
    save_beat_tsv(np.asarray(beats), np.asarray(downbeats), output)
    del tracker
    gc.collect()


def _structure(mix_path, stems_dir, output, root, cache, device):
    import torch
    from allin1_infer.spectrogram import extract_spectrograms
    from allin1_infer.models.loaders import _build_model_from_checkpoint
    from allin1_infer.models.ensemble import Ensemble
    from allin1_infer.helpers import run_inference, save_results
    # Same disk PCM16 -> spectrogram -> eight-fold -> postprocess chain as analyze(),
    # with local-only model loading instead of a download-capable alias resolver.
    spec_path = extract_spectrograms([stems_dir], root / 'spec', multiprocess=False)[0]
    entries = [a for a in _manifest()['assets'] if a['role'].startswith('harmonix_fold')]
    model = Ensemble([_build_model_from_checkpoint(str(cache / 'checkpoints' / a['name']), device)
                      for a in entries]).to(device).eval()
    if device.startswith('cuda'):
        torch.set_float32_matmul_precision('high')
    with torch.no_grad():
        result = run_inference(mix_path, spec_path, model, device, False, False)
    save_results(result, output.parent)
    del model
    gc.collect()
    return spec_path


def extract_sdrp(view, families, *, device='cpu', cache_dir=None, work_dir=None,
                 progress=lambda stage: None) -> dict:
    """Analyze an AudioView; return exact named features, quality and provenance.

    ``work_dir`` retains a new unique product directory for audit. Omit it to
    dispose of only this call's temporary audio/products after returning hashes.
    This call is synchronous and serialized for predictable memory consumption.
    """
    selected = selected_families(families)
    if np.asarray(view.stereo).shape != (1323000, 2) or not np.isfinite(view.stereo).all():
        raise ValueError('Expected finite, exact Native30 stereo AudioView.')
    native_rate = view.metadata['sample_rate']
    if 'S' in selected and native_rate < 16000:
        raise ValueError('S8 requires native sample rate >=16000 Hz.')
    status = runtime_status(families=selected, cache_dir=cache_dir)
    if not status['ready']:
        raise NeuralUnavailable('; '.join(status['errors']) + '. ' + status['setup_command'])
    device = _device(device)
    cache = cache_path(cache_dir)
    retained = work_dir is not None
    if retained:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='native30-', dir=work_dir))
        context = None
    else:
        context = tempfile.TemporaryDirectory(prefix='music-detector-neural-')
        root = Path(context.name)
    started = time.monotonic()
    try:
        with _INFERENCE_LOCK:
            mix_path = root / 'native30.wav'
            sf.write(mix_path, np.asarray(view.stereo, dtype=np.float32), 44100, subtype='FLOAT')
            check_audio(mix_path, subtype='FLOAT')
            stems_dir, beats_path, structure_path, spec_path = None, None, None, None
            if set(selected) & {'S', 'D', 'P'}:
                progress('Demucs htdemucs：真实四轨分离（CPU 可能需要数分钟）')
                stems_dir = _separate(mix_path, root, cache, device, progress)
                for name in ('bass', 'drums', 'other', 'vocals'):
                    check_audio(stems_dir / (name + '.wav'), subtype='PCM_16')
            if set(selected) & {'R', 'P'}:
                progress('Beat This final0：节拍与小节起点')
                beats_path = root / 'native30.beats'
                _track_beats(mix_path, beats_path, cache, device)
            if 'P' in selected:
                progress('All-In-One harmonix-all：八模型乐段边界分析')
                structure_path = root / 'structure' / 'native30.json'
                spec_path = _structure(mix_path, stems_dir, structure_path, root, cache, device)
            progress('按冻结 S8/D3/R3/P6 定义计算真实特征')
            result = extract_products(mix_path=mix_path, families=selected, native_sample_rate=native_rate,
                                      stems_dir=stems_dir, beats_path=beats_path,
                                      structure_path=structure_path, bias_path=ASSETS / 'demucs_frequency_bias.npz')
            if spec_path:
                result['product_sha256']['spectrogram'] = digest(spec_path)
            result['provenance'] = {
                'backend': 'real_native30_sdrp_v1', 'device': device,
                'python': platform.python_version(), 'platform': platform.platform(),
                'versions': status['versions'], 'manifest_sha256': status['manifest_sha256'],
                'feature_source_sha256': result['feature_source_sha256'],
                'product_sha256': result['product_sha256'],
                'checkpoint_sha256': {a['name']: a['sha256'] for a in _manifest()['assets']
                                      if set(a['families']) & set(selected)},
                'metadata': result['metadata'], 'elapsed_seconds': time.monotonic() - started,
                'retained_product_directory': str(root) if retained else None,
                'historical_bitwise_reproduction': False,
                'limitations': status['limitations'], 'protocol': _manifest()['protocol'],
                'licenses': _manifest()['licenses'],
            }
            return result
    finally:
        if context is not None:
            context.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'setup'))
    parser.add_argument('--families', default='S,D,R,P')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--install-dependencies', action='store_true')
    parser.add_argument('--download-weights', action='store_true')
    args = parser.parse_args(argv)
    kwargs = dict(families=args.families.split(','), cache_dir=args.cache_dir)
    if args.command == 'setup':
        result = setup(**kwargs, install_dependencies=args.install_dependencies,
                       download_weights=args.download_weights)
    else:
        result = runtime_status(**kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
