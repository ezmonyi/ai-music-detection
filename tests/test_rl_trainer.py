"""CPU contract tests for the online RL trainer and its planning CLI.

The toy backend is deliberately used here: these tests exercise checkpoint,
replay, split-boundary, and optimizer semantics without downloading a model or
claiming anything about music quality.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
torch = pytest.importorskip("torch", reason="Install the optional rl extra")

from music_detector.rl.backends import ToyBackend
from music_detector.rl.config import config_from_dict, load_config
from music_detector.rl.rewards import RewardResult
from music_detector.rl.trainer import evaluate, train


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rl" / "toy_smoke.json"
TRAIN_FIXTURE = ROOT / "configs" / "rl" / "toy_train.jsonl"
VALIDATION_FIXTURE = ROOT / "configs" / "rl" / "toy_validation.jsonl"
TEST_FIXTURE = ROOT / "configs" / "rl" / "toy_test.jsonl"


def _materialize_jsonl(source: Path, destination: Path, *, split: str | None = None,
                       mutate: Any = None) -> Path:
    """Copy a fixture while tolerating the pre/post prompt_id fixture schema."""

    rows = []
    for index, line in enumerate(source.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        row.setdefault("prompt_id", row["source_id"])
        if split is not None:
            row["split"] = split
        if mutate is not None:
            mutate(row, index)
        rows.append(row)
    destination.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    return destination


def _train_data(tmp_path: Path) -> Path:
    return _materialize_jsonl(TRAIN_FIXTURE, tmp_path / "train.jsonl")


def _validation_data(tmp_path: Path) -> Path:
    return _materialize_jsonl(VALIDATION_FIXTURE, tmp_path / "validation.jsonl")


def _config(*, mode: str = "lora", updates: int = 2, group_size: int = 4,
            targets: str = "all_linear"):
    value = load_config(CONFIG_PATH).to_dict()
    value["policy"].update(mode=mode, targets=targets)
    value["training"].update(updates=updates, group_size=group_size, checkpoint_every=1)
    return config_from_dict(value)


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, type(right)) and len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class _ConstantReward:
    def score(self, audio, sample_rate, baseline_audio):
        return RewardResult(1.0, True, {"raw_score_delta": 0.0})


class _InvalidReward:
    def score(self, audio, sample_rate, baseline_audio):
        return RewardResult(-1.0, False, {"raw_score_delta": 0.0})


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_toy_optimizer_updates_and_mode_freezing(tmp_path: Path, mode: str):
    cfg = _config(mode=mode, updates=1)
    data = _train_data(tmp_path)
    backend = ToyBackend(device="cpu")
    before = {name: parameter.detach().clone() for name, parameter in backend.decoder.named_parameters()}

    summary = train(cfg, data, tmp_path / "run", backend=backend)

    assert summary["optimizer_updates"] == 1
    after = dict(backend.decoder.named_parameters())
    comparable = {
        name: parameter for name, parameter in after.items()
        if ".lora_" not in name
    }
    changed = [
        name for name, parameter in comparable.items()
        if not torch.equal(before[name.replace(".base", "")], parameter)
    ]
    changed += [name for name, parameter in after.items() if ".lora_" in name and not torch.equal(
        parameter, torch.zeros_like(parameter)
    )]
    assert changed, "a valid toy group must produce a real optimizer update"
    if mode == "lora":
        # Every original linearly parameter is frozen and remains identical;
        # only low-rank residuals may move.
        base_names = [name for name in after if ".base." in name]
        assert base_names and all(torch.equal(before[name.replace(".base", "")], after[name]) for name in base_names)
        assert any(".lora_B" in name for name in changed)
    else:
        assert all(parameter.requires_grad for parameter in after.values())
        assert all(".lora_" not in name for name in changed)


def test_resume_is_bitwise_equivalent_to_uninterrupted_run(tmp_path: Path):
    cfg = _config(updates=3)
    data = _train_data(tmp_path)
    whole = tmp_path / "whole"
    partial = tmp_path / "partial"
    resumed = tmp_path / "resumed"
    train(cfg, data, whole)
    train(cfg, data, partial, max_updates=1)
    train(cfg, data, resumed, resume=partial / "checkpoints" / "group_000001.pt")

    whole_checkpoint = _load_checkpoint(whole / "checkpoints" / "group_000003.pt")
    resumed_checkpoint = _load_checkpoint(resumed / "checkpoints" / "group_000003.pt")
    assert _nested_equal(whole_checkpoint["policy"], resumed_checkpoint["policy"])
    assert _nested_equal(whole_checkpoint["optimizer"], resumed_checkpoint["optimizer"])
    assert _nested_equal(whole_checkpoint["torch_rng"], resumed_checkpoint["torch_rng"])

    whole_rows = {row["groups_completed"]: row for row in map(json.loads, (whole / "metrics.jsonl").read_text().splitlines())}
    resumed_rows = {row["groups_completed"]: row for row in map(json.loads, (resumed / "metrics.jsonl").read_text().splitlines())}
    assert whole_rows.keys() >= {2, 3}
    assert resumed_rows.keys() == {2, 3}
    for group in (2, 3):
        for key in ("rewards", "loss", "reference_kl", "gradient_norm", "optimizer_updates", "skipped_zero_advantage"):
            assert whole_rows[group][key] == resumed_rows[group][key]


def test_resume_rejects_config_and_data_boundary_mismatch(tmp_path: Path):
    cfg = _config(updates=2)
    data = _train_data(tmp_path)
    partial = tmp_path / "partial"
    train(cfg, data, partial, max_updates=1)
    checkpoint = partial / "checkpoints" / "group_000001.pt"

    changed_config_value = cfg.to_dict()
    changed_config_value["training"]["learning_rate"] = 0.0007
    changed_config = config_from_dict(changed_config_value)
    with pytest.raises(ValueError, match="boundary"):
        train(changed_config, data, tmp_path / "bad-config", resume=checkpoint)

    changed_data = _materialize_jsonl(
        data,
        tmp_path / "changed-data.jsonl",
        mutate=lambda row, index: row.update(caption=row["caption"] + " changed") if index == 0 else None,
    )
    with pytest.raises(ValueError, match="boundary"):
        train(cfg, changed_data, tmp_path / "bad-data", resume=checkpoint)


def test_evaluation_rejects_training_overlap_and_accepts_holdout(tmp_path: Path):
    cfg = _config(updates=1)
    train_data = _train_data(tmp_path)
    train_dir = tmp_path / "train"
    train(cfg, train_data, train_dir)
    checkpoint = train_dir / "checkpoints" / "group_000001.pt"

    holdout = _validation_data(tmp_path)
    result = evaluate(cfg, holdout, checkpoint, tmp_path / "eval")
    assert result["count"] == 1
    assert result["valid_count"] == 1
    assert result["solver"] == "ODE"

    overlapping = _materialize_jsonl(
        TRAIN_FIXTURE,
        tmp_path / "overlap.jsonl",
        split="validation",
        mutate=lambda row, index: row.update(prompt_id="overlap-validation") if index == 0 else None,
    )
    with pytest.raises(ValueError, match="overlaps training"):
        evaluate(cfg, overlapping, checkpoint, tmp_path / "eval-overlap")


def test_all_invalid_reward_fails_without_update(tmp_path: Path):
    cfg = _config(updates=1)
    with pytest.raises(RuntimeError, match="All candidates failed"):
        train(cfg, _train_data(tmp_path), tmp_path / "invalid", reward=_InvalidReward())
    failure = json.loads((tmp_path / "invalid" / "failure.json").read_text())
    assert failure["optimizer_updates"] == 0
    assert not (tmp_path / "invalid" / "checkpoints" / "group_000001.pt").exists()


def test_zero_advantage_group_is_explicitly_skipped(tmp_path: Path):
    cfg = _config(updates=1)
    summary = train(cfg, _train_data(tmp_path), tmp_path / "zero", reward=_ConstantReward())
    assert summary["optimizer_updates"] == 0
    metric = json.loads((tmp_path / "zero" / "metrics.jsonl").read_text())
    assert metric["skipped_zero_advantage"] is True
    assert metric["gradient_norm"] == 0.0
    assert (tmp_path / "zero" / "checkpoints" / "group_000001.pt").exists()


def test_bounded_all_invalid_skip_never_updates_and_resumes_counters(tmp_path: Path):
    value = _config(updates=2).to_dict()
    value["training"].update(all_invalid_policy="skip_bounded", max_all_invalid_groups=2,
                             max_consecutive_all_invalid_groups=2)
    cfg = config_from_dict(value)
    data = _train_data(tmp_path)
    first = tmp_path / "skip-first"
    train(cfg, data, first, max_updates=1, reward=_InvalidReward())
    checkpoint = first / "checkpoints/group_000001.pt"
    saved = _load_checkpoint(checkpoint)
    assert saved["optimizer_updates"] == 0
    assert saved["optimizer"]["state"] == {}
    assert saved["admission_state"]["all_invalid_groups"] == 1
    resumed = tmp_path / "skip-resumed"
    summary = train(cfg, data, resumed, resume=checkpoint, reward=_InvalidReward())
    assert summary["groups_completed"] == 2 and summary["optimizer_updates"] == 0
    assert summary["all_invalid_groups"] == 2
    metric = json.loads((resumed / "metrics.jsonl").read_text())
    assert metric["skipped_all_invalid"] and metric["gradient_norm"] == 0.0
    final = _load_checkpoint(resumed / "checkpoints/group_000002.pt")
    assert _nested_equal(saved["policy"], final["policy"])
    assert _nested_equal(saved["optimizer"], final["optimizer"])


def test_bounded_all_invalid_budget_exhaustion_fails(tmp_path: Path):
    value = _config(updates=2).to_dict()
    value["training"].update(all_invalid_policy="skip_bounded", max_all_invalid_groups=1)
    cfg = config_from_dict(value)
    output = tmp_path / "bounded-failure"
    with pytest.raises(RuntimeError, match="safety budget"):
        train(cfg, _train_data(tmp_path), output, reward=_InvalidReward())
    failure = json.loads((output / "failure.json").read_text())
    assert failure["groups_completed"] == 1 and failure["optimizer_updates"] == 0


def test_valid_groups_continue_after_bounded_skip(tmp_path: Path):
    from music_detector.rl.trainer import create_reward
    value = _config(updates=2).to_dict()
    value["training"]["all_invalid_policy"] = "skip_bounded"
    cfg = config_from_dict(value)

    class FirstGroupInvalid:
        calls = 0
        regular = create_reward(cfg)

        def score(self, *args):
            self.calls += 1
            return (_InvalidReward().score(*args) if self.calls <= cfg.training.group_size else
                    self.regular.score(*args))

    summary = train(cfg, _train_data(tmp_path), tmp_path / "invalid-then-valid", reward=FirstGroupInvalid())
    assert summary["groups_completed"] == 2 and summary["optimizer_updates"] == 1
    assert summary["all_invalid_groups"] == 1 and summary["consecutive_all_invalid_groups"] == 0


def test_cli_ablation_plan_writes_45_planned_runs(tmp_path: Path):
    output = tmp_path / "plan"
    command = [sys.executable, "-m", "music_detector.rl.cli", "ablation-plan",
               "--config", str(CONFIG_PATH), "--output", str(output)]
    result = subprocess.run(command, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")},
                            text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["planned_runs"] == 45
    plan = json.loads((output / "plan.json").read_text())
    assert plan["status"] == "planned_not_run"
    assert len(plan["runs"]) == 45
    assert len(list(output.glob("*.json"))) == 46


@pytest.mark.parametrize("contents", [
    "{\"policy\": {\"targets\": \"not-a-profile\"}}",
    "{ malformed json",
])
def test_cli_validate_rejects_malformed_config(tmp_path: Path, contents: str):
    path = tmp_path / "bad.json"
    path.write_text(contents)
    command = [sys.executable, "-m", "music_detector.rl.cli", "validate", "--config", str(path)]
    result = subprocess.run(command, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")},
                            text=True, capture_output=True)
    assert result.returncode != 0
