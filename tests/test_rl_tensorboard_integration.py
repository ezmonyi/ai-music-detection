"""Integration contracts for RL TensorBoard logging and held-out evaluation.

These tests deliberately use the tiny CPU backend.  They verify event-file
semantics and checkpoint boundaries without downloading a model or making any
claim about music quality.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="Install the optional rl extra")
pytest.importorskip("tensorboard", reason="Install the optional rl extra for integration tests")

from music_detector.rl import trainer as trainer_module
from music_detector.rl.config import config_from_dict, load_config
from music_detector.rl.rewards import RewardResult
from music_detector.rl.trainer import evaluate, train


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rl" / "toy_smoke.json"


def _config(*, updates: int = 1, tensorboard: bool = False,
            evaluation_every: int = 0, evaluation_prompts: int = 10):
    value = load_config(CONFIG_PATH).to_dict()
    value["sampling"].update(steps=4, train_timesteps=2)
    value["training"].update(
        updates=updates,
        group_size=2,
        checkpoint_every=1,
        save_audio_every=100,
    )
    value["monitoring"] = {
        "tensorboard": tensorboard,
        "evaluation_every": evaluation_every,
        "evaluation_prompts": evaluation_prompts,
    }
    return config_from_dict(value)


def _write_prompts(path: Path, *, split: str, prefix: str, count: int = 2) -> Path:
    rows = []
    for index in range(count):
        rows.append({
            "prompt_id": f"{prefix}-prompt-{index}",
            "caption": f"A distinct {prefix} piano and drum arrangement {index}",
            "lyrics": "",
            "duration_s": 30,
            "seed": 1000 + index,
            "split": split,
            "source_dataset": "fixture/rl-tensorboard",
            "source_revision": "fixture-v1",
            "source_id": f"{prefix}-source-{index}",
            "license": "CC0-1.0",
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _event_accumulator(log_root: Path):
    """Load the sole real TensorBoard event stream below a run log root."""

    pytest.importorskip("tensorboard", reason="TensorBoard integration dependency is optional")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(log_root.rglob("events.out.tfevents.*"))
    assert event_files, f"No TensorBoard event file found below {log_root}"
    return EventAccumulator(str(event_files[0].parent))


def _scalar_events(log_root: Path):
    accumulator = _event_accumulator(log_root)
    accumulator.Reload()
    return accumulator, {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


class _SequenceReward:
    """Deterministic reward fixture for mixed-admission evaluation tests."""

    def __init__(self, outcomes: list[tuple[float, bool]]):
        self._outcomes = iter(outcomes)

    def score(self, audio, sample_rate, baseline_audio):
        reward, valid = next(self._outcomes)
        return RewardResult(
            float(reward),
            bool(valid),
            {"raw_score_delta": float(reward), "raw_score": abs(float(reward))},
        )


def _train_checkpoint(cfg, train_data: Path, output: Path):
    train(cfg, train_data, output)
    return output / "checkpoints" / f"group_{cfg.training.updates:06d}.pt"


def test_real_tensorboard_train_scalars_have_required_tags_and_group_steps(tmp_path: Path):
    cfg = _config(updates=2, tensorboard=True)
    train_data = _write_prompts(tmp_path / "train.jsonl", split="train", prefix="train")
    output = tmp_path / "run"

    train(cfg, train_data, output)

    accumulator, scalars = _scalar_events(output / "tensorboard")
    required = {
        "train/loss",
        "train/policy_loss",
        "train/reference_kl",
        "train/weighted_kl",
        "train/grad_norm",
        "train/learning_rate",
        "throughput/candidate_clips_per_second",
        "throughput/generated_clips_per_second",
        "throughput/audio_seconds_per_second",
        "throughput/groups_per_second",
        "system/host_peak_rss_gib",
    }
    assert required <= set(accumulator.Tags()["scalars"])
    for tag in required:
        events = scalars[tag]
        assert [event.step for event in events] == [1, 2]
        assert all(math.isfinite(event.value) for event in events)

    # A CPU run must never invent GPU memory metrics.
    assert not any(tag.startswith("system/cuda_") or tag.startswith("system/gpu")
                   for tag in accumulator.Tags()["scalars"])

    rows = _read_jsonl(output / "metrics.jsonl")
    assert len(rows) == 2
    for row in rows:
        assert row["loss"] == pytest.approx(row["policy_loss"] + row["weighted_kl"], abs=1e-6)


def test_tensorboard_steps_continue_after_resume(tmp_path: Path):
    cfg = _config(updates=3, tensorboard=True)
    train_data = _write_prompts(tmp_path / "train.jsonl", split="train", prefix="resume")
    partial = tmp_path / "partial"
    resumed = tmp_path / "resumed"

    train(cfg, train_data, partial, max_updates=1)
    checkpoint = partial / "checkpoints" / "group_000001.pt"
    train(cfg, train_data, resumed, resume=checkpoint)

    _, scalars = _scalar_events(resumed / "tensorboard")
    assert [event.step for event in scalars["train/loss"]] == [2, 3]
    assert [event.step for event in scalars["train/weighted_kl"]] == [2, 3]
    assert {row["groups_completed"] for row in _read_jsonl(resumed / "metrics.jsonl")} == {2, 3}


def test_standalone_evaluation_logs_admitted_reward_loss_and_mixed_fraction(tmp_path: Path):
    cfg = _config(updates=1, tensorboard=True)
    train_data = _write_prompts(tmp_path / "train.jsonl", split="train", prefix="standalone-train")
    validation = _write_prompts(tmp_path / "validation.jsonl", split="validation",
                                prefix="standalone-validation", count=2)
    checkpoint = _train_checkpoint(cfg, train_data, tmp_path / "train")
    output = tmp_path / "evaluation"

    summary = evaluate(
        cfg,
        validation,
        checkpoint,
        output,
        split="validation",
        reward=_SequenceReward([(0.75, True), (100.0, False)]),
    )

    assert summary["valid_count"] == 1
    assert summary["valid_fraction"] == pytest.approx(0.5)
    assert summary["reward_mean"] == pytest.approx(0.75)
    assert summary["loss"] == pytest.approx(-0.75)
    assert summary["loss_definition"] == "negative_mean_admitted_reward_not_policy_loss"

    accumulator, scalars = _scalar_events(output / "tensorboard")
    assert {"eval/loss", "eval/reward_mean", "eval/valid_fraction"} <= set(
        accumulator.Tags()["scalars"]
    )
    assert scalars["eval/loss"][0].step == 1
    assert scalars["eval/loss"][0].value == pytest.approx(-0.75)
    assert scalars["eval/reward_mean"][0].value == pytest.approx(0.75)
    assert scalars["eval/valid_fraction"][0].value == pytest.approx(0.5)


def test_all_invalid_standalone_evaluation_has_no_zero_loss_or_loss_event(tmp_path: Path):
    cfg = _config(updates=1, tensorboard=True)
    train_data = _write_prompts(tmp_path / "train.jsonl", split="train", prefix="invalid-train")
    validation = _write_prompts(tmp_path / "validation.jsonl", split="validation",
                                prefix="invalid-validation", count=2)
    checkpoint = _train_checkpoint(cfg, train_data, tmp_path / "train")
    output = tmp_path / "evaluation"

    summary = evaluate(
        cfg,
        validation,
        checkpoint,
        output,
        reward=_SequenceReward([(1.0, False), (2.0, False)]),
    )

    assert summary["valid_count"] == 0
    assert summary["valid_fraction"] == pytest.approx(0.0)
    assert summary["reward_mean"] is None
    assert summary["loss"] is None
    assert summary["loss_definition"] == "negative_mean_admitted_reward_not_policy_loss"
    accumulator, scalars = _scalar_events(output / "tensorboard")
    assert "eval/loss" not in accumulator.Tags()["scalars"]
    assert "eval/reward_mean" not in accumulator.Tags()["scalars"]
    assert scalars["eval/valid_fraction"][0].value == pytest.approx(0.0)


def test_periodic_validation_does_not_change_policy_optimizer_or_torch_rng(tmp_path: Path):
    train_data = _write_prompts(tmp_path / "train.jsonl", split="train", prefix="periodic-train")
    validation = _write_prompts(tmp_path / "validation.jsonl", split="validation",
                                prefix="periodic-validation", count=1)
    baseline_cfg = _config(updates=2, evaluation_every=0)
    periodic_cfg = _config(updates=2, evaluation_every=1, evaluation_prompts=1)

    baseline = tmp_path / "baseline"
    periodic = tmp_path / "periodic"
    train(baseline_cfg, train_data, baseline)
    train(periodic_cfg, train_data, periodic, validation_data=validation)

    baseline_checkpoint = torch.load(
        baseline / "checkpoints" / "group_000002.pt", map_location="cpu", weights_only=True
    )
    periodic_checkpoint = torch.load(
        periodic / "checkpoints" / "group_000002.pt", map_location="cpu", weights_only=True
    )
    for key in ("policy", "optimizer", "torch_rng", "cuda_rng"):
        assert _nested_equal(baseline_checkpoint[key], periodic_checkpoint[key]), key

    periodic_boundary = periodic_checkpoint["boundary"]["periodic_validation"]
    from music_detector.rl.data import file_sha256
    assert periodic_boundary["data_sha256"] == file_sha256(validation)
    assert periodic_boundary["prompt_ids"] == ["periodic-validation-prompt-0"]

    validation_rows = _read_jsonl(periodic / "validation_metrics.jsonl")
    assert [row["groups_completed"] for row in validation_rows] == [1, 2]


def test_periodic_validation_rejects_test_split_before_backend_initialization(tmp_path: Path, monkeypatch):
    cfg = _config(updates=1, evaluation_every=1)
    train_data = _write_prompts(tmp_path / "train.jsonl", split="train", prefix="preflight-train")
    test_data = _write_prompts(tmp_path / "test.jsonl", split="test", prefix="preflight-test")

    def backend_must_not_be_created(_cfg):
        pytest.fail("periodic validation split must be checked before backend initialization")

    monkeypatch.setattr(trainer_module, "create_backend", backend_must_not_be_created)
    with pytest.raises(ValueError, match="Expected only validation records"):
        train(cfg, train_data, tmp_path / "run", validation_data=test_data)


def test_periodic_validation_requires_complete_provenance_and_rejects_overlap_preflight(
    tmp_path: Path, monkeypatch,
):
    cfg = _config(updates=1, evaluation_every=1)
    train_data = _write_prompts(tmp_path / "train.jsonl", split="train", prefix="overlap-train")

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(json.dumps({
        "prompt_id": "missing-provenance",
        "caption": "A validation prompt",
        "lyrics": "",
        "duration_s": 30,
        "seed": 1,
        "split": "validation",
    }) + "\n")
    overlap = tmp_path / "overlap.jsonl"
    row = json.loads(train_data.read_text().splitlines()[0])
    row["split"] = "validation"
    overlap.write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(
        trainer_module,
        "create_backend",
        lambda _cfg: pytest.fail("validation preflight must precede backend initialization"),
    )
    with pytest.raises(ValueError, match="provenance-complete"):
        train(cfg, train_data, tmp_path / "incomplete-run", validation_data=incomplete)
    with pytest.raises(ValueError, match="overlaps training"):
        train(cfg, train_data, tmp_path / "overlap-run", validation_data=overlap)


def test_memory_scalars_report_cuda_current_peak_and_capacity_but_cpu_never_calls_cuda(
    monkeypatch,
):
    cuda_calls: list[str] = []

    def unexpected_cuda_call(*args, **kwargs):
        cuda_calls.append("called")
        raise AssertionError("CPU memory collection must not call CUDA")

    for name in (
        "mem_get_info", "max_memory_allocated", "max_memory_reserved",
        "memory_allocated", "memory_reserved",
    ):
        monkeypatch.setattr(getattr(torch, "cuda"), name, unexpected_cuda_call)
    cpu_memory = trainer_module._memory(torch.device("cpu"))
    assert not cuda_calls
    assert "host_peak_rss_bytes" in cpu_memory
    assert not any(name.startswith("cuda_") for name in cpu_memory)

    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (3 * gib, 8 * gib))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2 * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 512 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 768 * 2**20)
    memory = trainer_module._memory(torch.device("cuda:0"))
    scalars = trainer_module._memory_scalars(memory)

    assert scalars["system/cuda_peak_allocated_gib"] == pytest.approx(1.0)
    assert scalars["system/cuda_peak_reserved_gib"] == pytest.approx(2.0)
    assert scalars["system/cuda_allocated_gib"] == pytest.approx(0.5)
    assert scalars["system/cuda_reserved_gib"] == pytest.approx(0.75)
    assert scalars["system/cuda_device_free_gib"] == pytest.approx(3.0)
    assert scalars["system/cuda_device_total_gib"] == pytest.approx(8.0)
    assert scalars["system/cuda_device_used_gib"] == pytest.approx(5.0)


def test_evaluation_test_split_uses_test_namespace_and_never_periodic(tmp_path: Path):
    cfg = _config(updates=1, tensorboard=True, evaluation_every=0)
    train_data = _write_prompts(tmp_path / "train.jsonl", split="train", prefix="test-train")
    test_data = _write_prompts(tmp_path / "test.jsonl", split="test", prefix="test-holdout")
    checkpoint = _train_checkpoint(cfg, train_data, tmp_path / "train")
    output = tmp_path / "test-evaluation"

    evaluate(cfg, test_data, checkpoint, output, split="test",
             reward=_SequenceReward([(0.5, True), (0.25, True)]))
    accumulator, scalars = _scalar_events(output / "tensorboard")
    assert {"test/loss", "test/reward_mean", "test/valid_fraction"} <= set(
        accumulator.Tags()["scalars"]
    )
    assert not any(tag.startswith("eval/") for tag in accumulator.Tags()["scalars"])


@pytest.mark.parametrize("field,value", [
    ("tensorboard", "yes"), ("tensorboard", 1),
    ("evaluation_every", -1), ("evaluation_every", True), ("evaluation_every", 1.5),
    ("evaluation_prompts", 0), ("evaluation_prompts", True), ("evaluation_prompts", 2.5),
])
def test_monitoring_config_rejects_ambiguous_types_and_invalid_ranges(field, value):
    config = _config().to_dict()
    config["monitoring"][field] = value
    with pytest.raises(ValueError):
        config_from_dict(config)


def test_evaluation_state_restores_rng_and_mixed_train_flags_on_failure():
    import random
    import numpy as np
    from music_detector.rl.backends import ToyBackend

    backend = ToyBackend()
    backend.decoder.train()
    child = next(module for module in backend.decoder.modules() if module is not backend.decoder)
    child.eval()
    modes = [module.training for module in backend.decoder.modules()]
    torch_state = torch.get_rng_state().clone()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with trainer_module._evaluation_state(backend):
            torch.rand(3)
            np.random.random()
            random.random()
            assert not backend.decoder.training
            raise RuntimeError("evaluation failed")
    assert torch.equal(torch_state, torch.get_rng_state())
    assert _nested_equal(python_state, random.getstate())
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0]
    np.testing.assert_array_equal(restored[1], numpy_state[1])
    assert restored[2:] == numpy_state[2:]
    assert modes == [module.training for module in backend.decoder.modules()]
