"""Synthetic tests for the online-RL reward boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_detector.rl.rewards import ArtifactReward, RewardGuards
from music_detector.scoring import FAMILIES


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "src/music_detector/assets/deployment_models.json"
BUNDLE_SHA256 = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()


def _audio(amplitude: float = 0.08, *, sample_rate: int = 44_100, seconds: float = 30.0) -> np.ndarray:
    count = int(sample_rate * seconds)
    t = np.arange(count, dtype=np.float64) / sample_rate
    left = amplitude * (
        0.45 * np.sin(2 * np.pi * 220 * t)
        + 0.25 * np.sin(2 * np.pi * 3_200 * t)
        + 0.20 * np.sin(2 * np.pi * 11_000 * t)
        + 0.10 * np.sin(2 * np.pi * 16_000 * t)
    )
    right = amplitude * (
        0.40 * np.sin(2 * np.pi * 220 * t + 0.2)
        + 0.30 * np.sin(2 * np.pi * 3_200 * t + 0.1)
        + 0.20 * np.sin(2 * np.pi * 11_000 * t + 0.4)
        + 0.10 * np.sin(2 * np.pi * 16_000 * t + 0.5)
    )
    return np.stack((left, right), axis=1).astype(np.float32)


def _broadband_complete_f_sc_audio(*, sample_rate: int = 44_100, seconds: float = 30.0) -> np.ndarray:
    """Deterministic native-stereo signal with complete F and SC coverage."""
    count = int(sample_rate * seconds)
    t = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(20261001)
    shared = rng.standard_normal(count)
    independent_left = rng.standard_normal(count)
    independent_right = rng.standard_normal(count)

    # Distinct log-envelope attack and decay regions make all F regions
    # eligible; broadband shared noise keeps every SC band energetic and
    # sufficiently coherent without using neural weights or a recording.
    envelope = np.ones(count, dtype=np.float64)
    attack = t < 0.8
    decay = t > seconds - 0.8
    envelope[attack] = 10.0 ** (-40.0 / 20.0 + (t[attack] / 0.8) * (40.0 / 20.0))
    envelope[decay] = 10.0 ** (-((t[decay] - (seconds - 0.8)) / 0.8) * (40.0 / 20.0))

    left = (
        0.045 * shared
        + 0.005 * independent_left
        + 0.020 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.012 * np.sin(2.0 * np.pi * 700.0 * t)
        + 0.009 * np.sin(2.0 * np.pi * 2_800.0 * t)
        + 0.006 * np.sin(2.0 * np.pi * 7_500.0 * t)
        + 0.005 * np.sin(2.0 * np.pi * 15_000.0 * t)
    )
    right = (
        0.041 * shared
        + 0.005 * independent_right
        + 0.018 * np.sin(2.0 * np.pi * 220.0 * t + 0.2)
        + 0.011 * np.sin(2.0 * np.pi * 700.0 * t + 0.4)
        + 0.008 * np.sin(2.0 * np.pi * 2_800.0 * t + 0.3)
        + 0.006 * np.sin(2.0 * np.pi * 7_500.0 * t + 0.5)
        + 0.004 * np.sin(2.0 * np.pi * 15_000.0 * t + 0.2)
    )
    return np.stack((envelope * left, envelope * right), axis=1).astype(np.float32)


def _fake_analyzer(path, families, **kwargs):
    values, _ = sf.read(path, dtype="float32", always_2d=True)
    columns = [name for family in families for name in FAMILIES[family]]
    # Deliberately retain input-dependent variation: a GRPO group at the base
    # policy must not receive an all-zero reward merely because its paired
    # baseline is identical.
    amplitude = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    feature_values = {name: 0.1 for name in columns}
    feature_values[columns[0]] = amplitude * 100.0
    return {"feature_values": feature_values}


def _reward(families=("F",)):
    return ArtifactReward(families, BUNDLE_SHA256, analyzer=_fake_analyzer)


def test_absolute_reward_is_nonzero_when_candidate_equals_baseline_and_delta_is_diagnostic_only():
    audio = _audio()
    result = _reward().score(audio, 44_100, audio.copy())
    assert result.valid
    assert result.reward != 0.0
    assert result.diagnostics["raw_score_delta"] == 0.0
    assert result.diagnostics["delta_scope"] == "paired_baseline_diagnostic_only"
    assert result.diagnostics["reward_kind"] == "bounded_negative_frozen_raw_score"
    assert -1.0 < result.reward < 1.0


def test_default_f_sc_analyzer_accepts_complete_native30_cpu_signal():
    audio = _broadband_complete_f_sc_audio()
    result = ArtifactReward(("F", "SC"), BUNDLE_SHA256).score(audio, 44_100, audio.copy())
    assert result.valid
    assert np.isfinite(result.reward)
    assert result.diagnostics["raw_score_delta"] == 0.0
    assert result.diagnostics["quality"]["candidate"]["duration_seconds"] == 30.0
    assert result.diagnostics["score"]["prediction"]["probability"] is not None


def test_reward_uses_absolute_raw_score_not_raw_score_delta():
    baseline = _audio(0.08)
    candidate = _audio(0.12)
    result = _reward().score(candidate, 44_100, baseline)
    assert result.valid
    assert result.diagnostics["raw_score_delta"] != 0.0
    expected = -np.tanh(result.diagnostics["raw_score"] / result.diagnostics["guard_config"]["score_scale"])
    assert result.reward == expected


def test_shape_layouts_are_supported_but_mono_is_rejected_before_analyzer():
    audio = _audio()
    result = _reward().score(audio.T, 44_100, audio.T)
    assert result.valid
    mono = audio.mean(axis=1)
    result = _reward().score(mono, 44_100, mono)
    assert not result.valid
    assert result.diagnostics["failure"] == "invalid_shape"


def test_silence_lowpass_and_clipping_are_invalid():
    baseline = _audio()
    silence = np.zeros_like(baseline)
    result = _reward().score(silence, 44_100, baseline)
    assert not result.valid and result.diagnostics["failure"] == "silence"

    lowpass = _audio()
    t = np.arange(lowpass.shape[0], dtype=np.float64) / 44_100
    lowpass[:, 0] = 0.08 * np.sin(2 * np.pi * 220 * t)
    lowpass[:, 1] = 0.08 * np.sin(2 * np.pi * 220 * t + 0.2)
    result = _reward().score(lowpass, 44_100, baseline)
    assert not result.valid
    assert result.diagnostics["failure"] in {"lowpass", "bandwidth_collapse"}

    clipped = baseline.copy()
    clipped[:10_000, 0] = 1.0
    result = _reward().score(clipped, 44_100, baseline)
    assert not result.valid
    assert result.diagnostics["failure"] in {"clipping_peak", "clipping_fraction"}


def test_missing_family_feature_is_invalid_not_missingness_reward():
    def missing_analyzer(path, families, **kwargs):
        columns = [name for family in families for name in FAMILIES[family]]
        return {"feature_values": {name: (None if name == columns[0] else 0.1) for name in columns}}

    result = ArtifactReward(("F",), BUNDLE_SHA256, analyzer=missing_analyzer).score(
        _audio(), 44_100, _audio()
    )
    assert not result.valid
    assert result.diagnostics["failure"] == "analysis"


def test_ineligible_baseline_is_distinguished_from_candidate_failure():
    result = _reward().score(_audio(), 44_100, np.zeros_like(_audio()))
    assert not result.valid
    assert result.diagnostics["failure"] == "baseline_ineligible"
    assert result.diagnostics["baseline_failure"] == "silence"


def test_numpy_scalar_features_are_normalized_before_frozen_scoring():
    def numpy_analyzer(path, families, **kwargs):
        columns = [name for family in families for name in FAMILIES[family]]
        return {"feature_values": {name: np.float32(0.1) for name in columns}}

    result = ArtifactReward(("F",), BUNDLE_SHA256, analyzer=numpy_analyzer).score(
        _audio(), 44_100, _audio()
    )
    assert result.valid
    assert np.isfinite(result.reward)


def test_guard_rejects_nan_and_non_worst_invalid_rewards():
    with pytest.raises(ValueError, match="finite"):
        RewardGuards(minimum_rms_dbfs=float("nan"))
    with pytest.raises(ValueError, match="no better"):
        RewardGuards(invalid_reward=-0.5)


def test_runtime_or_setup_oserror_is_not_silently_converted_to_candidate_invalid():
    def unavailable_analyzer(path, families, **kwargs):
        raise FileNotFoundError("checkpoint disappeared")

    bridge = ArtifactReward(("F",), BUNDLE_SHA256, analyzer=unavailable_analyzer)
    with pytest.raises(FileNotFoundError, match="checkpoint disappeared"):
        bridge.score(_audio(), 44_100, _audio())


def test_incomplete_baseline_analysis_is_not_reported_as_candidate_failure():
    calls = 0

    def baseline_missing_analyzer(path, families, **kwargs):
        nonlocal calls
        calls += 1
        columns = [name for family in families for name in FAMILIES[family]]
        if calls == 2:
            return {"feature_values": {name: None for name in columns}}
        return {"feature_values": {name: 0.1 for name in columns}}

    result = ArtifactReward(("F",), BUNDLE_SHA256, analyzer=baseline_missing_analyzer).score(
        _audio(), 44_100, _audio()
    )
    assert not result.valid
    assert result.diagnostics["failure"] == "baseline_analysis"


def test_bundle_checksum_mismatch_fails_fast():
    try:
        ArtifactReward(("F",), "0" * 64, analyzer=_fake_analyzer)
    except RuntimeError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("checksum mismatch must fail at setup")


def test_guard_mapping_is_explicit_and_invalid_reward_is_worst_case():
    guard = RewardGuards(score_scale=1.0, invalid_reward=-2.0)
    result = ArtifactReward(("F",), BUNDLE_SHA256, guard=guard, analyzer=_fake_analyzer).score(
        np.zeros_like(_audio()), 44_100, _audio()
    )
    assert not result.valid
    assert result.reward == -2.0
