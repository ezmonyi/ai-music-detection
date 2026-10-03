"""Sequential online flow-policy rollouts and one on-policy replay update/group.

No gradient passes through the audio decoder or the non-differentiable reward.
Detached stochastic transitions, not supervised denoising targets, provide the
policy gradient. CPU fixtures validate plumbing, not music quality or GPU fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Any

import numpy as np
import torch

from .backends import DecodedAudio, FlowBackend, ToyBackend
from .config import ExperimentConfig, validate_config
from .data import caption_hash, file_sha256
from .flow import (clipped_policy_loss, gaussian_kl, gaussian_log_prob,
                   group_advantages, sample_transition, transition_mean)
from .policy import configure_policy, export_adapter_state, load_adapter_state
from .monitoring import RunMonitor


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def append_json(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def read_prompts(path: str | Path, split: str) -> list[dict]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Prompt file is empty")
    seen_ids, seen_captions, seen_sources = set(), set(), set()
    for record in records:
        required = {"prompt_id", "caption", "lyrics", "duration_s", "seed", "split",
                    "source_dataset", "source_revision", "source_id", "license"}
        if not isinstance(record, dict) or required - set(record):
            raise ValueError("Use the provenance-complete JSONL produced by music_detector.rl.data")
        if record["split"] != split:
            raise ValueError(f"Expected only {split} records, found {record['split']}")
        if not isinstance(record["caption"], str) or not record["caption"].strip():
            raise ValueError("caption must be nonempty text")
        if not isinstance(record["lyrics"], str):
            raise ValueError("lyrics must be text; empty is allowed")
        if type(record["seed"]) is not int or record["seed"] < 0:
            raise ValueError("Every prompt needs a nonnegative integer seed")
        for key in ("prompt_id", "source_dataset", "source_revision", "source_id", "license"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"Missing text provenance: {key}")
        source = (record["source_dataset"], record["source_id"])
        hashed = caption_hash(record["caption"])
        if record["prompt_id"] in seen_ids or hashed in seen_captions or source in seen_sources:
            raise ValueError("Duplicate prompt/caption/source in split")
        seen_ids.add(record["prompt_id"])
        seen_captions.add(hashed)
        seen_sources.add(source)
    return sorted(records, key=lambda row: row["prompt_id"])


def derive_seed(master: int, record: dict, group: int, sample: int) -> int:
    value = f"{master}:{record['seed']}:{record['prompt_id']}:{group}:{sample}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def schedule(cfg: ExperimentConfig) -> list[float]:
    # Shift is explicit; SFT 50-step/shift=1 is not the historical Turbo solver.
    times = torch.linspace(1, 0, cfg.sampling.steps + 1, dtype=torch.float64)
    shift = cfg.sampling.time_shift
    return (shift * times / (1 + (shift - 1) * times)).tolist()


@dataclass
class Transition:
    x: torch.Tensor
    action: torch.Tensor
    t: float
    dt: float
    variance: torch.Tensor
    old_logprob: torch.Tensor


@torch.no_grad()
def rollout(backend: FlowBackend, condition: Any, cfg: ExperimentConfig, seed: int,
            *, reference: bool = False, stochastic: bool = True,
            collect: bool = True) -> tuple[DecodedAudio, list[Transition]]:
    generator = torch.Generator(device=backend.device).manual_seed(seed)
    x = torch.randn(backend.latent_shape(condition), device=backend.device,
                    dtype=torch.float32, generator=generator)
    times = schedule(cfg)
    eligible = [i for i, t in enumerate(times[:-1]) if stochastic and
                cfg.sampling.stochastic_t_min <= t <= cfg.sampling.stochastic_t_max]
    if collect and not eligible:
        raise ValueError("No stochastic timesteps in the configured schedule/window")
    # Seeded timestep subsampling keeps fixed actions on CPU, not full GPU graphs.
    chooser = np.random.default_rng(seed)
    selected = set(chooser.choice(eligible, min(len(eligible), cfg.sampling.train_timesteps),
                                  replace=False).tolist()) if collect else set()
    transitions = []
    for i, (t, next_t) in enumerate(zip(times, times[1:])):
        eta = cfg.sampling.noise_level if i in eligible else 0.0
        velocity = backend.velocity(x, t, condition, reference=reference).float()
        if velocity.shape != x.shape or not torch.isfinite(velocity).all():
            raise RuntimeError("Nonfinite or incorrectly shaped policy velocity")
        next_x, _, variance, logprob = sample_transition(
            x, velocity, t, next_t - t, eta, generator=generator,
            reduction=cfg.sampling.logprob_reduction)
        if not torch.isfinite(next_x).all():
            raise RuntimeError("Nonfinite flow trajectory")
        if i in selected:
            if variance is None or logprob is None:
                raise RuntimeError("Selected an unscoreable deterministic transition")
            transitions.append(Transition(x.detach().cpu(), next_x.detach().cpu(),
                                          t, next_t-t, variance.detach().cpu(), logprob.cpu()))
        x = next_x
    return backend.decode(x), transitions


class ToyReward:
    """Synthetic quadratic target; explicitly not the music artifact detector."""
    def score(self, audio, sample_rate, baseline_audio):
        from .rewards import RewardResult
        loss = float(np.mean((np.asarray(audio) - 0.15) ** 2))
        baseline = float(np.mean((np.asarray(baseline_audio) - 0.15) ** 2))
        return RewardResult(-loss, True, {"synthetic": True, "raw_score": loss,
                                        "baseline_raw_score": baseline,
                                        "raw_score_delta": loss-baseline})


def create_backend(cfg: ExperimentConfig) -> FlowBackend:
    if cfg.model.backend == "toy":
        return ToyBackend(device=cfg.model.device, duration_s=cfg.sampling.duration_s)
    from .acestep_backend import AceStepBackend
    return AceStepBackend(cfg.model)


def create_reward(cfg: ExperimentConfig):
    if cfg.reward.kind == "toy":
        return ToyReward()
    from .rewards import ArtifactReward
    analyzer = None
    if cfg.reward.analyzer_python is not None:
        from .isolated_analyzer import IsolatedAnalyzer
        analyzer = IsolatedAnalyzer(cfg.reward)
    return ArtifactReward(cfg.reward.families, cfg.reward.bundle_sha256,
                          device=cfg.reward.device, guard=cfg.reward.guards, analyzer=analyzer)


def setup_policy(cfg: ExperimentConfig, backend: FlowBackend):
    backend.prepare_reference(cfg.policy.mode)
    if cfg.policy.mode == "full":
        backend.decoder.float()
    metadata = configure_policy(backend.decoder, mode=cfg.policy.mode, rank=cfg.policy.rank,
                                alpha=cfg.policy.alpha, profile=cfg.policy.targets)
    # Rollout and replay use identical deterministic dropout behaviour.
    for module in backend.decoder.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    backend.decoder.train()
    parameters = [p for p in backend.decoder.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise RuntimeError("Trainable parameters must be nonempty and float32")
    return metadata, parameters


def _memory(device: torch.device) -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"host_peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                      cuda_device_free_bytes=free, cuda_device_total_bytes=total,
                      cuda_device_used_bytes=total-free)
    return result


def _memory_scalars(memory: dict) -> dict:
    # An unavailable CUDA measurement is absent, never a fabricated zero.
    return {"system/" + name.replace("_bytes", "_gib"): value / 2**30
            for name, value in memory.items() if name.endswith("_bytes")}


def _preflight_monitor(cfg: ExperimentConfig) -> None:
    if cfg.monitoring.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as error:
            raise RuntimeError("TensorBoard is enabled; install the optional rl extra before loading a model") from error


def _reject_overlap(records: list[dict], hashes: set, sources: set, prompt_ids: set) -> None:
    if any(r["prompt_id"] in prompt_ids or caption_hash(r["caption"]) in hashes or
           (r["source_dataset"], r["source_id"]) in sources for r in records):
        raise ValueError("Evaluation overlaps training prompt ID/caption/source")


@contextmanager
def _evaluation_state(backend):
    """Evaluation must not perturb future rollout RNG or decoder train modes."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    modes = [(module, module.training) for module in backend.decoder.modules()]
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            backend.decoder.eval()
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        for module, mode in modes:
            module.training = mode


def _evaluate_records(cfg, records, backend, reward, output, *, split, step,
                      stochastic=False, monitor=None, output_prepared=False) -> dict:
    """Paired deployment-like validation; no fictitious supervised/GRPO loss."""
    output = Path(output)
    if not output_prepared:
        output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deltas, admitted_rewards, raw_scores = [], [], []
    with _evaluation_state(backend), torch.no_grad():
        for row in records:
            if row["duration_s"] != cfg.sampling.duration_s:
                raise ValueError("Evaluation duration differs from training contract")
            condition = backend.condition(row)
            seed = derive_seed(cfg.training.seed, row, 0, 0)
            base, _ = rollout(backend, condition, cfg, seed, reference=True, stochastic=stochastic, collect=False)
            candidate, _ = rollout(backend, condition, cfg, seed, stochastic=stochastic, collect=False)
            if candidate.sample_rate != base.sample_rate:
                raise RuntimeError("Evaluation candidate/reference sample rates disagree")
            result = reward.score(candidate.samples, candidate.sample_rate, base.samples)
            if not math.isfinite(result.reward):
                raise RuntimeError("Nonfinite evaluation reward")
            append_json(output / "pairs.jsonl", {"prompt_id": row["prompt_id"], "seed": seed, **asdict(result)})
            safe_id = hashlib.sha256(row["prompt_id"].encode()).hexdigest()[:16]
            _save_audio(output / f"{safe_id}_base.wav", base)
            _save_audio(output / f"{safe_id}_candidate.wav", candidate)
            if result.valid:
                admitted_rewards.append(result.reward)
                for name, values in (("raw_score_delta", deltas), ("raw_score", raw_scores)):
                    value = result.diagnostics.get(name)
                    if value is not None:
                        if not math.isfinite(value):
                            raise RuntimeError(f"Nonfinite evaluation diagnostic: {name}")
                        values.append(value)
    _sync(backend.device)
    reward_mean = float(np.mean(admitted_rewards)) if admitted_rewards else None
    summary = {"count": len(records), "valid_count": len(admitted_rewards),
               "valid_fraction": len(admitted_rewards) / len(records),
               "groups_completed": step, "split": split,
               "solver": "SDE" if stochastic else "ODE",
               "loss": -reward_mean if reward_mean is not None else None,
               "loss_definition": "negative_mean_admitted_reward_not_policy_loss",
               "reward_mean": reward_mean,
               "raw_score_mean": float(np.mean(raw_scores)) if raw_scores else None,
               "paired_raw_score_delta_mean": float(np.mean(deltas)) if deltas else None,
               "elapsed_seconds": time.monotonic()-started,
               "music_quality_validated": False,
               "interpretation": "Eval loss is a frozen reward proxy, not policy/supervised loss or proof of better perceived music."}
    write_json(output / "summary.json", summary)
    if monitor is not None:
        namespace = "eval" if split == "validation" else "test"
        monitor.add_scalars({f"{namespace}/{key}": summary[key] for key in (
            "loss", "reward_mean", "valid_count", "valid_fraction", "raw_score_mean", "paired_raw_score_delta_mean",
            "elapsed_seconds")}, step)
    return summary


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_audio(path: Path, audio: DecodedAudio) -> None:
    import soundfile as sf
    # Never normalize or clip an invalid sample into an apparently valid one.
    sf.write(path, audio.samples, audio.sample_rate, subtype="FLOAT")


def _boundary(cfg: ExperimentConfig, data_path: Path, records: list[dict], backend, reward=None) -> dict:
    source = Path(__file__).parent
    modules = ["config.py", "data.py", "flow.py", "policy.py", "trainer.py", "backends.py", "rewards.py", "monitoring.py"]
    if cfg.model.backend == "acestep":
        modules.append("acestep_backend.py")
    if cfg.reward.analyzer_python is not None:
        modules.extend(["isolated_analyzer.py", "analyzer_worker.py"])
    return {"config_sha256": cfg.digest(), "data_sha256": file_sha256(data_path),
            "backend": backend.provenance(), "torch_version": str(torch.__version__),
            "implementation_sha256": {name: file_sha256(source / name) for name in modules},
            "parameter_dtypes": {
                "trainable": sorted({str(p.dtype) for p in backend.decoder.parameters() if p.requires_grad}),
                "frozen_decoder": sorted({str(p.dtype) for p in backend.decoder.parameters() if not p.requires_grad}),
                "configured_compute_precision": cfg.model.precision,
            },
            "analysis_receipt": getattr(getattr(reward, "_analyzer", None), "receipt", None),
            "training_caption_hashes": sorted(caption_hash(r["caption"]) for r in records),
            "training_prompt_ids": sorted(r["prompt_id"] for r in records),
            "training_sources": sorted([r["source_dataset"], r["source_id"]] for r in records)}


def save_checkpoint(path, cfg, boundary, backend, optimizer, groups, updates):
    payload = {"schema_version": 1, "config": cfg.to_dict(), "boundary": boundary,
               "policy": export_adapter_state(backend.decoder), "optimizer": optimizer.state_dict(),
               "groups_completed": groups, "optimizer_updates": updates,
               "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if backend.device.type == "cuda" else []}
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(cfg: ExperimentConfig, data_path: str | Path, output: str | Path, *,
          resume: str | Path | None = None, max_updates: int | None = None,
          backend: FlowBackend | None = None, reward=None,
          validation_data: str | Path | None = None) -> dict:
    validate_config(cfg)
    data_path, output = Path(data_path), Path(output)
    records = read_prompts(data_path, "train")
    if any(r["duration_s"] != cfg.sampling.duration_s for r in records):
        raise ValueError("Prompt and sampling durations disagree")
    validation_records = []
    if cfg.monitoring.evaluation_every:
        if validation_data is None:
            raise ValueError("Periodic evaluation requires --validation-data (validation split only)")
        validation_records = read_prompts(validation_data, "validation")
        _reject_overlap(validation_records, {caption_hash(r["caption"]) for r in records},
                        {(r["source_dataset"], r["source_id"]) for r in records},
                        {r["prompt_id"] for r in records})
        if any(r["duration_s"] != cfg.sampling.duration_s for r in validation_records):
            raise ValueError("Validation duration differs from training contract")
        # Deterministic subset independent of file order, pinned in manifest/resume.
        validation_records = sorted(validation_records, key=lambda r: hashlib.sha256(
            r["prompt_id"].encode()).hexdigest())[:cfg.monitoring.evaluation_prompts]
    elif validation_data is not None:
        raise ValueError("Set monitoring.evaluation_every > 0 to use --validation-data")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive integer")
    if output.exists():
        raise FileExistsError("Use a new output directory, including when resuming")
    _preflight_monitor(cfg)
    torch.manual_seed(cfg.training.seed)
    reward = reward or create_reward(cfg)  # Fail missing reward dependencies before loading a GPU model.
    backend = backend or create_backend(cfg)
    metadata, parameters = setup_policy(cfg, backend)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.training.learning_rate, weight_decay=0.0)
    boundary = _boundary(cfg, data_path, records, backend, reward)
    if validation_records:
        boundary["periodic_validation"] = {
            "data_sha256": file_sha256(validation_data),
            "prompt_ids": [r["prompt_id"] for r in validation_records],
            "solver": "ODE", "loss_definition": "negative_mean_admitted_reward_not_policy_loss"}
    start, optimizer_updates = 0, 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 1 or checkpoint["boundary"] != boundary:
            raise ValueError("Resume config/data/backend boundary does not match checkpoint")
        load_adapter_state(backend.decoder, checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, optimizer_updates = checkpoint["groups_completed"], checkpoint["optimizer_updates"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if backend.device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    stop = min(cfg.training.updates, start + max_updates) if max_updates else cfg.training.updates
    if start >= stop:
        raise ValueError("Checkpoint already reached requested group budget")
    output.mkdir(parents=True)
    (output / "audio").mkdir()
    (output / "checkpoints").mkdir()
    write_json(output / "config.json", cfg.to_dict())
    write_json(output / "manifest.json", {**boundary, "policy": metadata.as_dict(),
               "resume": str(Path(resume).resolve()) if resume else None,
               "protocol": "one on-policy replay pass/group; sequential candidate/reference rollouts",
               "music_evidence": cfg.model.backend != "toy"})
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    completed = start
    started = time.monotonic()
    monitor = None
    try:
        monitor = RunMonitor(output / "tensorboard", enabled=cfg.monitoring.tensorboard)
        monitor.add_text("run/config", json.dumps(cfg.to_dict(), indent=2), start)
        monitor.add_text("run/semantics", "train/loss = policy loss + weighted reference KL; "
                         "eval/loss = negative mean admitted validation reward, NOT policy loss. "
                         "Gradient norm is pre-clipping; peaks cover this process since this run began. "
                         "Toy runs are synthetic plumbing, not music evidence.", start)
        for group in range(start, stop):
            group_start = time.monotonic()
            row = records[group % len(records)]
            with torch.no_grad():
                condition = backend.condition(row)
            trajectories, rewards, diagnostics = [], [], []
            rollout_seconds = reward_seconds = 0.0
            for sample in range(cfg.training.group_size):
                seed = derive_seed(cfg.training.seed, row, group, sample)
                _sync(backend.device)
                before = time.monotonic()
                baseline, _ = rollout(backend, condition, cfg, seed, reference=True, collect=False)
                audio, transitions = rollout(backend, condition, cfg, seed)
                _sync(backend.device)
                rollout_seconds += time.monotonic() - before
                if audio.sample_rate != baseline.sample_rate:
                    raise RuntimeError("Candidate/reference sample rates disagree")
                before = time.monotonic()
                result = reward.score(audio.samples, audio.sample_rate, baseline.samples)
                reward_seconds += time.monotonic() - before
                if not math.isfinite(result.reward):
                    raise RuntimeError("Nonfinite reward")
                trajectories.append(transitions)
                rewards.append(result.reward)
                details = {"group": group, "sample": sample, "prompt_id": row["prompt_id"],
                           "seed": seed, **asdict(result)}
                diagnostics.append(details)
                append_json(output / "rollouts.jsonl", details)
                if group % cfg.training.save_audio_every == 0 or not result.valid:
                    stem = f"g{group:06d}_s{sample:02d}"
                    _save_audio(output / "audio" / f"{stem}_candidate.wav", audio)
                    _save_audio(output / "audio" / f"{stem}_base.wav", baseline)
            valid_count = sum(r["valid"] for r in diagnostics)
            if valid_count == 0:
                raise RuntimeError("All candidates failed reward admission; inspect rollouts.jsonl; no update performed")
            advantage = group_advantages(torch.tensor(rewards, dtype=torch.float32),
                                         clip=cfg.training.advantage_clip)
            optimizer.zero_grad(set_to_none=True)
            replay_start = time.monotonic()
            loss_total, policy_loss_total, kl_total, ratio_deviation = 0.0, 0.0, 0.0, 0.0
            grad_norm = 0.0
            skipped = not bool(advantage.abs().max() > 1e-7)
            if not skipped:
                count = sum(len(t) for t in trajectories)
                for sample, trajectory in enumerate(trajectories):
                    for transition in trajectory:
                        x = transition.x.to(backend.device).detach().requires_grad_(True)
                        action = transition.action.to(backend.device).detach()
                        variance = transition.variance.to(backend.device)
                        with torch.no_grad():
                            ref_v = backend.velocity(x.detach(), transition.t, condition, reference=True).float()
                            ref_mean = transition_mean(x.detach(), ref_v, transition.t, transition.dt,
                                                       cfg.sampling.noise_level)
                        velocity = backend.velocity(x, transition.t, condition).float()
                        mean = transition_mean(x, velocity, transition.t, transition.dt, cfg.sampling.noise_level)
                        logprob = gaussian_log_prob(action, mean, variance, cfg.sampling.logprob_reduction)
                        old = transition.old_logprob.to(backend.device)
                        kl = gaussian_kl(mean, ref_mean, variance, cfg.sampling.logprob_reduction).mean()
                        policy_loss = clipped_policy_loss(logprob, old, advantage[sample], cfg.training.clip_range)
                        loss = (policy_loss + cfg.training.kl_coefficient * kl) / count
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite policy objective; optimizer not stepped")
                        loss.backward()
                        loss_total += float(loss.detach())
                        policy_loss_total += float(policy_loss.detach()) / count
                        kl_total += float(kl.detach()) / count
                        ratio_deviation += float((logprob.detach()-old).abs().mean()) / count
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg.training.max_grad_norm,
                                                                error_if_nonfinite=True))
                if grad_norm == 0:
                    raise RuntimeError("Nonzero advantages produced zero gradients; inspect policy adapter")
                optimizer.step()
                optimizer_updates += 1
            _sync(backend.device)
            completed = group + 1
            replay_seconds = time.monotonic()-replay_start
            if completed % cfg.training.checkpoint_every == 0 or completed == stop:
                save_checkpoint(output / "checkpoints" / f"group_{completed:06d}.pt", cfg, boundary,
                                backend, optimizer, completed, optimizer_updates)
            group_seconds = time.monotonic()-group_start
            evaluation_seconds = 0.0
            if validation_records and (completed % cfg.monitoring.evaluation_every == 0 or
                                       completed == cfg.training.updates):
                evaluated = _evaluate_records(cfg, validation_records, backend, reward,
                    output / "validation" / f"group_{completed:06d}", split="validation",
                    step=completed, monitor=monitor)
                evaluation_seconds = evaluated["elapsed_seconds"]
                append_json(output / "validation_metrics.jsonl", evaluated)
            wall_seconds = time.monotonic()-group_start
            throughput = {"candidate_clips_per_second": cfg.training.group_size / group_seconds,
                          "generated_clips_per_second": 2*cfg.training.group_size / group_seconds,
                          "audio_seconds_per_second": 2*cfg.training.group_size*cfg.sampling.duration_s / group_seconds,
                          "groups_per_second": 1/group_seconds,
                          "optimizer_updates_per_second": int(not skipped)/group_seconds,
                          "end_to_end_groups_per_second": 1/wall_seconds}
            memory = _memory(backend.device)
            entry = {"groups_completed": completed, "optimizer_updates": optimizer_updates,
                     "prompt_id": row["prompt_id"], "rewards": rewards,
                     "reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards)),
                     "valid_fraction": valid_count / len(rewards), "loss": loss_total,
                     "policy_loss": policy_loss_total, "weighted_kl": cfg.training.kl_coefficient*kl_total,
                     "learning_rate": optimizer.param_groups[0]["lr"],
                     "reference_kl": kl_total, "old_new_logprob_abs_difference": ratio_deviation,
                     "gradient_norm": grad_norm, "skipped_zero_advantage": skipped,
                     "rollout_seconds": rollout_seconds, "reward_seconds": reward_seconds,
                     "replay_seconds": replay_seconds, "evaluation_seconds": evaluation_seconds,
                     "group_seconds": group_seconds, "whole_group_seconds": wall_seconds,
                     "throughput": throughput, **memory}
            append_json(output / "metrics.jsonl", entry)
            monitor.add_scalars({"train/loss": loss_total, "train/policy_loss": policy_loss_total,
                "train/reference_kl": kl_total, "train/weighted_kl": entry["weighted_kl"],
                "train/grad_norm": grad_norm, "train/learning_rate": entry["learning_rate"],
                "train/reward_mean": entry["reward_mean"], "train/valid_fraction": entry["valid_fraction"],
                "train/optimizer_updates": optimizer_updates, "train/skipped_zero_advantage": int(skipped),
                "train/old_new_logprob_abs_difference": ratio_deviation,
                "timing/rollout_seconds": rollout_seconds, "timing/reward_seconds": reward_seconds,
                "timing/replay_seconds": replay_seconds, "timing/group_seconds": group_seconds,
                "timing/whole_group_seconds": wall_seconds,
                **{f"throughput/{key}": value for key, value in throughput.items()},
                **_memory_scalars(memory)}, completed)
        summary = {"status": "complete" if completed == cfg.training.updates else "budget_slice_complete",
                   "groups_completed": completed, "optimizer_updates": optimizer_updates,
                   "elapsed_seconds": time.monotonic()-started, "music_quality_validated": False,
                   **_memory(backend.device)}
        write_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error),
                   "groups_completed": completed, "optimizer_updates": optimizer_updates}
        try:
            failure["memory"] = _memory(backend.device)
            if monitor is not None:
                monitor.add_text("run/failure", json.dumps(failure, indent=2), completed)
        except Exception as logging_error:
            failure["monitoring_error"] = str(logging_error)
        write_json(output / "failure.json", failure)
        raise
    finally:
        # A storage/writer failure must not hide the original model/OOM error.
        if monitor is not None:
            body_failed = sys.exc_info()[0] is not None
            try:
                monitor.close()
            except Exception as error:
                if not body_failed:
                    write_json(output / "failure.json", {"type": type(error).__name__,
                        "message": str(error), "phase": "monitor_close",
                        "groups_completed": completed, "optimizer_updates": optimizer_updates})
                    raise


def evaluate(cfg: ExperimentConfig, data_path, checkpoint_path, output, *,
             split="validation", stochastic=False, backend=None, reward=None) -> dict:
    validate_config(cfg)
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation requires validation or test split")
    records = read_prompts(data_path, split)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if checkpoint["boundary"]["config_sha256"] != cfg.digest():
        raise ValueError("Evaluation config differs from training checkpoint")
    for name, expected in checkpoint["boundary"]["implementation_sha256"].items():
        if file_sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Evaluation implementation differs from training: {name}")
    _reject_overlap(records, set(checkpoint["boundary"]["training_caption_hashes"]),
                    {tuple(s) for s in checkpoint["boundary"]["training_sources"]},
                    set(checkpoint["boundary"]["training_prompt_ids"]))
    if any(r["duration_s"] != cfg.sampling.duration_s for r in records):
        raise ValueError("Evaluation duration differs from training contract")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation output must be a new directory")
    _preflight_monitor(cfg)
    torch.manual_seed(cfg.training.seed)
    reward = reward or create_reward(cfg)
    if getattr(getattr(reward, "_analyzer", None), "receipt", None) != checkpoint["boundary"].get("analysis_receipt"):
        raise ValueError("Evaluation frozen analysis runtime differs from training checkpoint")
    backend = backend or create_backend(cfg)
    setup_policy(cfg, backend)
    if backend.provenance() != checkpoint["boundary"]["backend"]:
        raise ValueError("Evaluation backend differs from training checkpoint")
    load_adapter_state(backend.decoder, checkpoint["policy"])
    output.mkdir(parents=True)
    write_json(output / "manifest.json", {"config": cfg.to_dict(), "split": split,
               "data_sha256": file_sha256(data_path), "checkpoint_sha256": file_sha256(checkpoint_path),
               "solver": "windowed_SDE" if stochastic else "ODE", "backend": backend.provenance()})
    with RunMonitor(output / "tensorboard", enabled=cfg.monitoring.tensorboard) as monitor:
        monitor.add_text("run/eval_loss_definition", "Negative mean admitted held-out reward. "
                         "This is not a policy loss, not comparable in scale to train/loss, "
                         "and all-invalid evaluation has no loss value.", checkpoint["groups_completed"])
        summary = _evaluate_records(cfg, records, backend, reward, output, split=split,
                                    step=checkpoint["groups_completed"], stochastic=stochastic,
                                    monitor=monitor, output_prepared=True)
    return summary
