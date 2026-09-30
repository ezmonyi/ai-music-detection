import numpy as np
import pytest
import soundfile as sf

from music_detector.audio import load_audio
from music_detector.analysis import analyze


@pytest.mark.parametrize('seconds,channels', [(2, 2), (30, 1)])
def test_rejects_invalid_native30_without_repair(tmp_path, seconds, channels):
    path = tmp_path / 'input.wav'
    sf.write(path, np.zeros((seconds * 16000, channels)), 16000)
    with pytest.raises(ValueError, match='native stereo|30–3600'):
        load_audio(path)


def test_silence_is_not_scored_and_link_is_not_followed(tmp_path):
    path = tmp_path / 'silence.wav'
    sf.write(path, np.zeros((30 * 16000, 2)), 16000)
    with pytest.raises(ValueError, match='silent'):
        analyze(path, ['F'])
    link = tmp_path / 'link.wav'
    link.symlink_to(path)
    with pytest.raises(ValueError, match='symlink'):
        load_audio(link)


@pytest.mark.parametrize('sample_rate', [16000, 44100, 48000])
def test_constant_dc_is_not_treated_as_human_music(tmp_path, sample_rate):
    path = tmp_path / 'dc.wav'
    sf.write(path, np.full((30 * sample_rate, 2), .1), sample_rate, subtype='FLOAT')
    with pytest.raises(ValueError, match='constant DC'):
        analyze(path, ['F', 'H'])


def test_all_missing_predictors_abstain_without_changing_frozen_model(monkeypatch):
    from music_detector import analysis
    from music_detector.audio import AudioView
    from music_detector.scoring import FAMILIES
    view = AudioView(np.asarray([[0., .1], [.1, 0.]]), np.ones(10), np.ones(10), {}, {})
    monkeypatch.setattr(analysis, 'load_audio', lambda *args, **kwargs: view)
    monkeypatch.setattr(analysis.phase_features, 'extract_phase_features',
                        lambda *args: dict.fromkeys(FAMILIES['F'], float('nan')) | {'F_status': 'unavailable'})
    result = analysis.analyze('not-read.wav', ['F'])
    assert result['prediction']['probability'] is None
    assert result['prediction']['decision'] == 'abstain'
    assert result['prediction']['raw_score'] is None
    assert result['features'][0]['status'] == 'unavailable'
