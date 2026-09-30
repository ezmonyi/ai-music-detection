from music_detector.explanations import explain_family, metric_unit


def test_missingness_is_not_intrinsic_audio_evidence():
    result = explain_family('P', {'p__section_bars_median': None}, {'contributions': [
        dict(feature='p__section_bars_median', family='P', value_contribution=.03,
             missingness_contribution=.2, missing=True)]})
    assert result['contribution'] == .23
    assert result['missingness_contribution'] == .2
    assert result['evidence'][0]['value'] is None
    assert '不能解释为音乐的固有 AI 特征' in result['reason']


def test_bc_never_gets_probability_evidence_and_units_are_explicit():
    result = explain_family('BC', {'BC_x': .1}, {})
    assert result['contribution'] is None
    assert metric_unit('F_group_delay_iqr_ms_all') == 'ms'
    assert metric_unit('s8__hf_tilt_5_7p5k_db_oct') == 'dB/oct'
