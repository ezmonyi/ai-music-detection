"""No downloaded weights: isolated runtime receipt and failure contracts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from music_detector.rl.config import RewardConfig, config_from_dict
from music_detector.rl.isolated_analyzer import IsolatedAnalyzer
from music_detector.rl.rewards import FEATURE_IMPLEMENTATION_SHA256


BUNDLE = Path(__file__).resolve().parents[1] / "src/music_detector/assets/deployment_models.json"
BUNDLE_SHA = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()


def cfg(tmp_path):
    return RewardConfig(kind="artifact", families=["S", "R", "F"],
                        bundle_sha256=BUNDLE_SHA, analyzer_python=sys.executable,
                        analyzer_cache_dir=str(tmp_path))


def fake_run(command, **kwargs):
    request = json.loads(kwargs["input"])
    from music_detector.rl import analyzer_worker
    receipt = {"backend": "isolated_frozen_analyzer_v1", "families": request["families"],
               "device": request["device"], "seed": request["seed"],
               "bundle_sha256": request["bundle_sha256"],
               "feature_implementation_sha256": dict(FEATURE_IMPLEMENTATION_SHA256),
               "runtime": {"ready": True, "versions": {"torch": "2.8.0"}},
               "worker_sha256": hashlib.sha256(Path(analyzer_worker.__file__).read_bytes()).hexdigest()}
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert command[1:] == ["-m", "music_detector.rl.analyzer_worker"]
    return subprocess.CompletedProcess(command, 0, json.dumps({"receipt": receipt,
                                                               "result": {"feature_values": {}}}), "")


def test_isolated_preflight_and_call(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", fake_run)
    analyzer = IsolatedAnalyzer(cfg(tmp_path))
    assert analyzer.receipt["runtime"]["versions"]["torch"] == "2.8.0"
    assert analyzer(tmp_path / "audio.wav", ["S", "R", "F"], device="cpu") == {"feature_values": {}}
    with pytest.raises(ValueError, match="preflight"):
        analyzer(tmp_path / "audio.wav", ["F"], device="cpu")


@pytest.mark.parametrize("mutation", ["hash", "runtime", "malformed", "exit"])
def test_worker_failures_are_not_missingness_or_fake_rewards(monkeypatch, tmp_path, mutation):
    def bad_run(command, **kwargs):
        result = fake_run(command, **kwargs)
        if mutation == "exit":
            return subprocess.CompletedProcess(command, 1, "", "deliberate worker failure")
        if mutation == "malformed":
            result.stdout = "not json"
        else:
            value = json.loads(result.stdout)
            if mutation == "hash":
                value["receipt"]["worker_sha256"] = "0" * 64
            else:
                value["receipt"]["runtime"]["ready"] = False
            result.stdout = json.dumps(value)
        return result
    monkeypatch.setattr(subprocess, "run", bad_run)
    with pytest.raises(RuntimeError):
        IsolatedAnalyzer(cfg(tmp_path))


def test_receipt_must_remain_fixed_after_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", fake_run)
    analyzer = IsolatedAnalyzer(cfg(tmp_path))
    analyzer.receipt["runtime"]["versions"]["torch"] = "different"
    with pytest.raises(RuntimeError, match="changed after preflight"):
        analyzer(tmp_path / "audio.wav", ["S", "R", "F"], device="cpu")


def test_configuration_rejects_unpaired_or_relative_isolated_paths():
    with pytest.raises(ValueError, match="absolute"):
        config_from_dict({"reward": {"kind": "artifact", "analyzer_python": "python"}})
    with pytest.raises(ValueError, match="requires analyzer_python"):
        config_from_dict({"reward": {"analyzer_cache_dir": "/tmp/cache"}})


def test_real_cpu_worker_status_needs_no_neural_weights(tmp_path):
    pytest.importorskip("torch")
    config = RewardConfig(kind="artifact", families=["F", "SC"], bundle_sha256=BUNDLE_SHA,
                          analyzer_python=sys.executable, analyzer_cache_dir=str(tmp_path))
    analyzer = IsolatedAnalyzer(config)
    assert analyzer.receipt["runtime"]["ready"]
    assert analyzer.receipt["families"] == ["F", "SC"]
