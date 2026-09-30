"""Offline protocol tests; synthetic audio is a test fixture, never a model substitute."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from music_detector import neural
from music_detector import neural_features as adapter
from music_detector.algorithms import expanded_feature_definitions as core
from music_detector.scoring import FAMILIES


@pytest.fixture
def products(tmp_path):
    t = np.arange(1323000) / 44100
    signal = (0.05 * np.sin(2 * np.pi * 220 * t) * (0.4 + 0.6 * np.cos(2 * np.pi * t) ** 8)).astype(np.float32)
    stereo = np.stack([signal, signal * .9], axis=1)
    mix = tmp_path / 'mix.wav'
    sf.write(mix, stereo, 44100, subtype='FLOAT')
    stems = tmp_path / 'stems'
    stems.mkdir()
    for i, name in enumerate(adapter.STEMS):
        sf.write(stems / (name + '.wav'), stereo * ((i + 1) / 10), 44100, subtype='PCM_16')
    beats = tmp_path / 'mix.beats'
    beats.write_text(''.join(f'{i * .5}\t{i % 4 + 1}\n' for i in range(60)))
    structure = tmp_path / 'mix.json'
    structure.write_text(json.dumps({'segments': [
        {'start': 0, 'end': 8}, {'start': 8, 'end': 16}, {'start': 16, 'end': 30}]}))
    return dict(mix_path=mix, stems_dir=stems, beats_path=beats,
                structure_path=structure, native_sample_rate=44100,
                bias_path=neural.ASSETS / 'demucs_frequency_bias.npz')


def test_frozen_sources_and_manifest_mirror():
    assert adapter.digest(Path(core.__file__)) == adapter.CORE_SHA256
    assert adapter.digest(neural.ASSETS / 'demucs_frequency_bias.npz') == adapter.BIAS_SHA256
    repo = Path(__file__).resolve().parents[1]
    assert neural.MANIFEST.read_bytes() == (repo / 'manifests/neural_runtime.json').read_bytes()
    assert len(neural._manifest()['assets']) == 10


def test_real_product_reduction_exactly_matches_archived_definitions(products):
    result = adapter.extract_products(**products, families=['S', 'D', 'R', 'P'])
    assert list(result['features']) == sum((FAMILIES[k] for k in ['S', 'D', 'R', 'P']), [])
    vocals = core.read_spectral_audio(products['stems_dir'] / 'vocals.wav', 30)
    _, s8 = core.spectral_families(vocals, adapter.load_bias(products['bias_path']))
    for name, value in s8.items():
        assert result['features']['s8__' + name] == value
    arrays = [core.read_audio(products['stems_dir'] / (name + '.wav'), 30, strict=True)
              for name in ('bass', 'drums', 'other')]
    dynamics = core.dynamics_features(arrays[0] + arrays[1] + arrays[2])
    for name, value in dynamics.items():
        assert result['features']['d__' + name] == value
    assert result['features']['r__ibi_cv'] == 0
    assert result['features']['r__tempo_tv'] == 0
    assert result['features']['r__tempo_entropy'] == 0
    assert result['features']['p__section_duration_median'] == 8
    assert result['features']['p__section_duration_cv'] == .375
    assert result['metadata']['n_beats'] == 60
    assert result['metadata']['n_downbeats'] == 15
    assert result['metadata']['vocal_activity_used_to_filter_primary_features'] is False


def test_scientific_missingness_is_not_processing_failure(products):
    products['beats_path'].write_text('')
    products['structure_path'].write_text('{"segments": []}')
    result = adapter.extract_products(**products, families=['R', 'P'])
    assert all(value is None for value in result['features'].values())
    assert result['quality']['R'].startswith('unavailable_fewer_than_16')
    products['beats_path'].unlink()
    with pytest.raises(ValueError, match='inference failure'):
        adapter.extract_products(**products, families=['R'])


def test_no_padding_bandwidth_or_wrong_bias_repair(products, tmp_path):
    products['native_sample_rate'] = 8000
    with pytest.raises(ValueError, match='bandwidth'):
        adapter.extract_products(**products, families=['S'])
    products['native_sample_rate'] = 44100
    wrong = tmp_path / 'wrong.npz'
    wrong.write_bytes(b'not the calibration')
    with pytest.raises(ValueError, match='SHA256'):
        adapter.extract_products(**{**products, 'bias_path': wrong}, families=['S'])
    sf.write(products['stems_dir'] / 'bass.wav', np.zeros((1322999, 2)), 44100, subtype='PCM_16')
    with pytest.raises(ValueError, match='no padding'):
        adapter.extract_products(**products, families=['D'])


def test_partial_phrase_availability_retains_individual_nulls(products):
    products['beats_path'].write_text('0\t1\n1\t2\n')
    result = adapter.extract_products(**products, families=['P'])
    assert result['features']['p__section_duration_median'] == 8
    assert result['features']['p__section_bars_cv'] is None
    assert result['quality']['P'] == 'eligible'


def test_invalid_and_nonfinite_beat_products_are_errors(products):
    products['beats_path'].write_text('1\t1\n0\t2\n')
    with pytest.raises(ValueError, match='Malformed'):
        adapter.extract_products(**products, families=['R'])


@pytest.mark.parametrize('raw', ['nan\t1\n', 'inf\t1\n', '-inf\t1\n', '0\tnan\n', '0\tinf\n'])
def test_nonfinite_raw_beats_fail_before_frozen_interval_filter(products, raw):
    products['beats_path'].write_text(raw)
    with pytest.raises(ValueError, match='Nonfinite raw Beat This'):
        adapter.extract_products(**products, families=['R'])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('key', ['start', 'end'])
def test_nonfinite_raw_segments_fail_before_frozen_boundary_cleanup(products, value, key):
    segments = [{'start': 0, 'end': 8}, {'start': 8, 'end': 16}, {'start': 16, 'end': 30}]
    # Also validate non-final ends, which the historical reducer does not consume.
    segments[0][key] = value
    products['structure_path'].write_text(json.dumps({'segments': segments}))
    with pytest.raises(ValueError, match='Nonfinite raw All-In-One'):
        adapter.extract_products(**products, families=['P'])


def test_setup_requires_explicit_actions_and_does_not_touch_network(tmp_path, monkeypatch):
    monkeypatch.setattr(neural.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('unexpected network'))
    with pytest.raises(ValueError, match='Explicit'):
        neural.setup(cache_dir=tmp_path)
    status = neural.runtime_status(families=['R'], cache_dir=tmp_path)
    assert not status['ready']
    assert [a['name'] for a in status['assets']] == ['beat_this-final0.ckpt']


def test_extraction_fails_before_inference_when_models_missing(tmp_path, monkeypatch):
    view = SimpleNamespace(stereo=np.zeros((1323000, 2)), metadata={'sample_rate': 44100})
    monkeypatch.setattr(neural, '_track_beats', lambda *a: pytest.fail('must not infer'))
    monkeypatch.setattr(neural.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('must not download'))
    with pytest.raises(neural.NeuralUnavailable, match='Missing or invalid analysis weight'):
        neural.extract_sdrp(view, ['R'], cache_dir=tmp_path)


def test_families_never_alias_surrogates():
    for families in ([], ['S', 'S'], ['F'], ['BC'], ['unknown']):
        with pytest.raises(ValueError):
            adapter.selected_families(families)
    assert adapter.selected_families(['P', 'R']) == ['R', 'P']
