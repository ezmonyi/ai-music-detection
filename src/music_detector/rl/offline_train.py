"""Bounded replay-based Flow-GRPO continuation, explicitly not on-policy RL.

Each recorded transition retains its own behavior denominator. Dimension-mean
likelihood ratios follow the existing Flow-GRPO surrogate, not exact trajectory
importance sampling. Conservative drift gates cannot guarantee offline RL safety.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import caption_hash, file_sha256
from .flow import clipped_policy_loss, gaussian_kl, gaussian_log_prob, group_advantages, transition_mean
from .monitoring import RunMonitor
from .offline_cache import verify_index
from .offline_io import save_tensor, write_json
from .offline_protocol import (analysis_signature, checkpoint, digest, experiment_contract,
                               model_signature, reject_overlap, runtime_config)
from .offline_trajectory import verify_replay
from .policy import export_adapter_state, load_adapter_state
from .trainer import (append_json, create_backend, create_reward, derive_seed, read_prompts,
                      rollout, setup_policy, _evaluation_state, _memory, _memory_scalars, _save_audio, _sync)


@dataclass(frozen=True)
class OfflineOptions:
    updates: int = 1000
    max_epochs: int = 1
    train_timesteps: int = 4
    seed: int = 11
    learning_rate: float = 3e-5
    clip_range: float = 0.2
    kl_coefficient: float = 0.01
    kl_normalization: str = "exact_transition"
    advantage_clip: float = 5.0
    max_grad_norm: float = 1.0
    max_abs_log_ratio: float = 5.0
    max_clip_fraction: float = 0.8
    min_normalized_ess: float = 0.2
    max_behavior_kl: float = 0.1
    max_consecutive_skips: int = 25
    checkpoint_every: int = 50
    evaluation_every: int = 50
    evaluation_prompts: int = 10
    tensorboard: bool = True
    replay_steps: int = 4
    replay_groups_per_behavior: int = 2
    replay_tolerance: float = 1e-6

    def validate(self):
        for key in ("updates", "max_epochs", "train_timesteps", "max_consecutive_skips", "checkpoint_every",
                    "evaluation_prompts", "replay_steps", "replay_groups_per_behavior"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + " must be a positive integer")
        if type(self.seed) is not int or self.seed < 0 or type(self.evaluation_every) is not int or self.evaluation_every < 0:
            raise ValueError("Invalid seed/evaluation interval")
        if type(self.tensorboard) is not bool or self.kl_normalization not in {"exact_transition", "upstream_sigma"}:
            raise ValueError("Invalid monitoring/KL convention")
        for key in ("learning_rate", "kl_coefficient", "advantage_clip", "max_grad_norm", "max_abs_log_ratio",
                    "max_behavior_kl", "replay_tolerance"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be finite and positive")
        for key in ("clip_range", "max_clip_fraction", "min_normalized_ess"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
                raise ValueError(key + " must lie in (0,1)")


def load_options(path):
    value = json.loads(Path(path).read_text())
    if set(value) - set(OfflineOptions.__dataclass_fields__):
        raise ValueError("Unknown offline option")
    options = OfflineOptions(**value)
    options.validate()
    return options


def preflight(index_path, checkpoint_path, checkpoint_sha256, validation_data, test_data, options):
    options.validate()
    index = verify_index(index_path)
    saved, cfg = checkpoint(checkpoint_path, checkpoint_sha256)
    if index["initial_checkpoint_sha256"] != checkpoint_sha256 or index["experiment_contract"] != experiment_contract(cfg):
        raise ValueError("Merged dataset does not belong to this initialization checkpoint")
    for key, path, split in [("validation_sha256", validation_data, "validation"), ("test_sha256", test_data, "test")]:
        if file_sha256(path) != index[key]:
            raise ValueError("Held-out split changed after indexing")
        reject_overlap([g["record"] for g in index["groups"]], read_prompts(path, split), description=split)
    signal = sum(g["has_learning_signal"] for g in index["groups"])
    if options.updates > signal*options.max_epochs:
        raise ValueError(f"Only {signal} signal-bearing groups × {options.max_epochs} epochs for {options.updates} real updates; collect more data")
    if options.train_timesteps > cfg.sampling.steps:
        raise ValueError("Too many training timesteps")
    return {"status": "cpu_data_preflight_passed_gpu_not_tested", "prompt_groups": len(index["groups"]),
            "signal_bearing_groups": signal, "target_optimizer_updates": options.updates,
            "behavior_policies": len({s["behavior_policy_fingerprint"] for s in index["shards"]}),
            "offline_reuse_is_off_policy": True}, index, saved, cfg


def _runtime_boundary(index_path, init_sha, cfg, options, backend, reward):
    modules = ("offline_train.py", "offline_cache.py", "offline_protocol.py", "offline_trajectory.py",
               "flow.py", "policy.py", "trainer.py", "acestep_backend.py", "rewards.py")
    return {"index_sha256": file_sha256(index_path), "initial_checkpoint_sha256": init_sha,
            "experiment_contract": experiment_contract(cfg), "options": asdict(options),
            "model_signature": model_signature(backend.provenance()), "torch_version": str(torch.__version__),
            "analysis_signature": analysis_signature(getattr(getattr(reward, "_analyzer", None), "receipt", None)),
            "implementation_sha256": {n: file_sha256(Path(__file__).with_name(n)) for n in modules}}


def _evaluate_triplets(cfg, records, backend, reward, initial_policy, output, *, split, step, monitor=None):
    """Original SFT base, group-100 initialization and continued policy, same seeds."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    current = export_adapter_state(backend.decoder)
    scores, rewards, deltas, improvement = [], [], [], []
    initial_valid = joint_valid = 0
    started = time.monotonic()
    try:
        with _evaluation_state(backend), torch.no_grad():
            for row in records:
                if row["duration_s"] != cfg.sampling.duration_s:
                    raise ValueError("Evaluation duration mismatch")
                condition = backend.condition(row)
                seed = derive_seed(cfg.training.seed, row, 0, 0)
                base, _ = rollout(backend, condition, cfg, seed, reference=True, stochastic=False, collect=False)
                load_adapter_state(backend.decoder, initial_policy)
                initial, _ = rollout(backend, condition, cfg, seed, stochastic=False, collect=False)
                load_adapter_state(backend.decoder, current)
                candidate, _ = rollout(backend, condition, cfg, seed, stochastic=False, collect=False)
                initial_result = reward.score(initial.samples, initial.sample_rate, base.samples)
                result = reward.score(candidate.samples, candidate.sample_rate, base.samples)
                initial_valid += int(initial_result.valid)
                raw = result.diagnostics.get("raw_score")
                initial_raw = initial_result.diagnostics.get("raw_score")
                if result.valid:
                    rewards.append(result.reward)
                    if raw is not None:
                        scores.append(float(raw))
                    delta = result.diagnostics.get("raw_score_delta")
                    if delta is not None:
                        deltas.append(float(delta))
                difference = None
                if result.valid and initial_result.valid:
                    joint_valid += 1
                    if raw is not None and initial_raw is not None:
                        difference = float(raw)-float(initial_raw)
                        improvement.append(difference)
                append_json(output / "pairs.jsonl", {"prompt_id": row["prompt_id"], "seed": seed,
                    "initial": asdict(initial_result), "candidate": asdict(result),
                    "candidate_minus_initial_raw_score": difference})
                stem = digest(row["prompt_id"])[:16]
                for suffix, audio in [("base", base), ("initial", initial), ("candidate", candidate)]:
                    _save_audio(output / f"{stem}_{suffix}.wav", audio)
    finally:
        load_adapter_state(backend.decoder, current)
    summary = {"count": len(records), "valid_count": len(rewards), "initial_valid_count": initial_valid,
               "joint_valid_count": joint_valid, "valid_fraction": len(rewards)/len(records),
               "loss": -float(np.mean(rewards)) if rewards else None,
               "loss_definition": "negative_mean_admitted_frozen_reward_not_policy_loss",
               "raw_score_mean": float(np.mean(scores)) if scores else None,
               "paired_raw_score_delta_mean": float(np.mean(deltas)) if deltas else None,
               "candidate_minus_initial_raw_score_mean": float(np.mean(improvement)) if improvement else None,
               "comparison_note": "Joint-admitted subset is selection-conditioned; report coverage, not unconditional improvement",
               "split": split, "offline_optimizer_updates": step, "solver": "ODE",
               "elapsed_seconds": time.monotonic()-started, "music_quality_validated": False}
    write_json(output / "summary.json", summary)
    if monitor:
        monitor.add_scalars({f"eval/{k}": summary[k] for k in ("loss", "valid_fraction", "joint_valid_count",
                           "candidate_minus_initial_raw_score_mean", "elapsed_seconds")}, step)
    return summary


def _group_update(cfg, options, group, folder, behavior_policy, backend, parameters, optimizer, epoch):
    optimizer.zero_grad(set_to_none=True)
    rewards = torch.tensor([r["reward"] for r in group["results"]], dtype=torch.float32)
    advantage = (group_advantages(rewards, clip=options.advantage_clip) if group["valid_count"]
                 else torch.zeros_like(rewards))
    if not bool(advantage.abs().max() > 1e-7):
        return {"updated": False, "skip_reason": "all_invalid" if not group["valid_count"] else "zero_advantage",
                "loss": 0.0, "policy_loss": 0.0, "reference_kl": 0.0, "behavior_kl": 0.0,
                "gradient_norm": 0.0, "clip_fraction": 0.0, "normalized_surrogate_ess": 1.0,
                "maximum_abs_log_ratio": 0.0, "replay_transitions": 0}
    condition = torch.load(folder / "condition.pt", map_location="cpu", weights_only=True)
    current = export_adapter_state(backend.decoder)
    transitions = []
    chooser = random.Random(int(digest([options.seed, epoch, group["record"]["prompt_id"]])[:16], 16))
    load_adapter_state(backend.decoder, behavior_policy)
    try:
        with torch.no_grad():
            for sample in range(cfg.training.group_size):
                record = torch.load(folder / f"s{sample:02d}_trajectory.pt", map_location="cpu", weights_only=True)
                eligible = torch.where(record["likelihood_mask"])[0].tolist()
                indices = chooser.sample(eligible, min(len(eligible), options.train_timesteps))
                for i in indices:
                    x, action = record["states"][i].to(backend.device), record["states"][i+1].to(backend.device)
                    t, dt = float(record["timesteps"][i]), float(record["dt"][i])
                    variance = record["variance"][i].to(backend.device)
                    behavior_mean = transition_mean(x, backend.velocity(x, t, condition).float(), t, dt, cfg.sampling.noise_level)
                    actual_old = gaussian_log_prob(action, behavior_mean, variance, cfg.sampling.logprob_reduction)
                    old = record["old_logprobs"][i].to(backend.device)
                    if float((actual_old-old).abs().max()) > options.replay_tolerance:
                        raise RuntimeError("Behavior likelihood mismatch on a training transition; old probabilities were not overwritten")
                    reference_mean = transition_mean(x, backend.velocity(x, t, condition, reference=True).float(),
                                                     t, dt, cfg.sampling.noise_level)
                    transitions.append((sample, record["states"][i], record["states"][i+1], t, dt,
                                        record["variance"][i], record["old_logprobs"][i],
                                        behavior_mean.cpu(), reference_mean.cpu()))
    finally:
        load_adapter_state(backend.decoder, current)
    optimizer.zero_grad(set_to_none=True)
    ratios, differences = [], []
    loss_total = actor_total = reference_total = behavior_total = 0.0
    hard_drift = False
    count = len(transitions)
    for sample, x, action, t, dt, variance, old, behavior_mean, reference_mean in transitions:
        x = x.to(backend.device).detach().requires_grad_(True)
        action, variance, old = action.to(backend.device), variance.to(backend.device), old.to(backend.device)
        mean = transition_mean(x, backend.velocity(x, t, condition).float(), t, dt, cfg.sampling.noise_level)
        probability = gaussian_log_prob(action, mean, variance, cfg.sampling.logprob_reduction)
        difference = float((probability.detach()-old).item())
        if not math.isfinite(difference):
            raise RuntimeError("Nonfinite offline likelihood; optimizer not stepped")
        differences.append(abs(difference))
        hard_drift |= abs(difference) > options.max_abs_log_ratio
        # Diagnostic-only bound prevents overflow on a rejected group.
        ratios.append(math.exp(max(-20.0, min(20.0, difference))))
        kl_variance = variance if options.kl_normalization == "exact_transition" else variance/abs(dt)
        kl = gaussian_kl(mean, reference_mean.to(backend.device), kl_variance, cfg.sampling.logprob_reduction).mean()
        behavior_kl = gaussian_kl(mean, behavior_mean.to(backend.device), variance, cfg.sampling.logprob_reduction).mean()
        reference_total += float(kl.detach())/count
        behavior_total += float(behavior_kl.detach())/count
        if not hard_drift:
            actor = clipped_policy_loss(probability, old, advantage[sample], options.clip_range)
            loss = (actor + options.kl_coefficient*kl)/count
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite offline objective; optimizer not stepped")
            loss.backward()
            loss_total += float(loss.detach())
            actor_total += float(actor.detach())/count
    clip_fraction = sum(abs(r-1) > options.clip_range for r in ratios)/len(ratios)
    ess = sum(ratios)**2/(len(ratios)*sum(r*r for r in ratios))
    reason = ("log_ratio_drift" if hard_drift else "clip_fraction_drift" if clip_fraction > options.max_clip_fraction
              else "surrogate_ess_drift" if ess < options.min_normalized_ess
              else "behavior_kl_drift" if behavior_total > options.max_behavior_kl else None)
    grad_norm = 0.0
    if reason:
        optimizer.zero_grad(set_to_none=True)
        loss_total = actor_total = 0.0  # Explicit skipped-update placeholders, not an optimized objective.
    else:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, options.max_grad_norm, error_if_nonfinite=True))
        if grad_norm == 0:
            raise RuntimeError("Nonzero offline advantages produced zero gradients")
        optimizer.step()
    return {"updated": reason is None, "skip_reason": reason, "loss": loss_total, "policy_loss": actor_total,
            "reference_kl": reference_total, "behavior_kl": behavior_total, "gradient_norm": grad_norm,
            "clip_fraction": clip_fraction, "normalized_surrogate_ess": ess,
            "maximum_abs_log_ratio": max(differences), "replay_transitions": count}


def _train_offline(index_path, checkpoint_path, checkpoint_sha256, validation_data, test_data, output,
                  options, *, overrides=None, resume=None, max_updates=None, backend=None, reward=None):
    report, index, initial, source_cfg = preflight(index_path, checkpoint_path, checkpoint_sha256,
                                                validation_data, test_data, options)
    cfg = runtime_config(source_cfg, overrides)
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive real-update budget")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh run directory, including when resuming")
    # Check monitoring before loading GPU weights.
    if options.tensorboard:
        from .monitoring import _resolve_summary_writer
        _resolve_summary_writer()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    write_json(output / "pipeline_state.json", {"phase": "initializing", "target_updates": options.updates,
               "offline_optimizer_updates": 0})
    torch.manual_seed(options.seed)
    reward = reward or create_reward(cfg)
    backend = backend or create_backend(cfg)
    _, parameters = setup_policy(cfg, backend)
    load_adapter_state(backend.decoder, initial["policy"])
    initial_policy = initial["policy"]
    previous_updates = initial["optimizer_updates"]
    # A changed offline objective gets a fresh optimizer, not mislabeled exact resume.
    initial.pop("optimizer", None)
    optimizer = torch.optim.AdamW(parameters, lr=options.learning_rate, weight_decay=0.0)
    boundary = _runtime_boundary(index_path, checkpoint_sha256, cfg, options, backend, reward)
    if boundary["model_signature"] != index["model_signature"] or boundary["analysis_signature"] != index["analysis_signature"]:
        raise ValueError("Cloud model/analyzer differs from the verified dataset")
    roots = [Path(index_path).resolve().parent / s["root"] for s in index["shards"]]
    behaviors = [torch.load(root / "behavior_policy.pt", map_location="cpu", weights_only=True) for root in roots]
    replay_checks, checked = [], set()
    with torch.no_grad():
        for shard_number, shard in enumerate(index["shards"]):
            fingerprint = shard["behavior_policy_fingerprint"]
            if fingerprint in checked:
                continue
            load_adapter_state(backend.decoder, behaviors[shard_number])
            probes = [g for g in index["groups"] if g["shard"] == shard_number][:options.replay_groups_per_behavior]
            for group in probes:
                folder = roots[shard_number] / group["path"]
                condition = torch.load(folder / "condition.pt", map_location="cpu", weights_only=True)
                for sample in range(cfg.training.group_size):
                    record = torch.load(folder / f"s{sample:02d}_trajectory.pt", map_location="cpu", weights_only=True)
                    replay_checks.append({"behavior": fingerprint, "prompt_id": group["record"]["prompt_id"],
                        "sample": sample, **verify_replay(backend, condition, cfg, record,
                        maximum_steps=options.replay_steps, tolerance=options.replay_tolerance)})
            checked.add(fingerprint)
    load_adapter_state(backend.decoder, initial_policy)
    cursor = updates = consecutive_skips = 0
    if resume:
        resumed = torch.load(resume, map_location="cpu", weights_only=True)
        if resumed.get("schema_version") != 2 or resumed.get("offline_resume_boundary") != boundary:
            raise ValueError("Offline resume code/data/options/runtime boundary mismatch")
        load_adapter_state(backend.decoder, resumed["policy"])
        optimizer.load_state_dict(resumed["optimizer"])
        cursor, updates = resumed["offline_cursor"], resumed["offline_optimizer_updates"]
        consecutive_skips = resumed["offline_consecutive_skips"]
        torch.set_rng_state(resumed["torch_rng"])
        if backend.device.type == "cuda":
            torch.cuda.set_rng_state_all(resumed["cuda_rng"])
    order = []
    for epoch in range(options.max_epochs):
        indices = list(range(len(index["groups"])))
        random.Random(options.seed+epoch).shuffle(indices)
        order.extend((epoch, i) for i in indices)
    if cursor > len(order) or updates > options.updates:
        raise ValueError("Offline resume counters exceed the fixed budget")
    target = min(options.updates, updates+max_updates) if max_updates else options.updates
    write_json(output / "CONFIG.json", cfg.to_dict())
    write_json(output / "OFFLINE_OPTIONS.json", asdict(options))
    write_json(output / "PREFLIGHT.json", {**report, "gpu_behavior_replay": replay_checks})
    write_json(output / "MANIFEST.json", {"kind": "bounded_off_policy_flow_grpo_continuation", "boundary": boundary,
        "initial_optimizer_state": "fresh_at_offline_start_restored_only_for_offline_resume",
        "previous_optimizer_updates": previous_updates, "target_offline_updates": options.updates,
        "ratio_definition": "exp(dimension_mean_logprob_current_minus_recorded_behavior)",
        "not_exact_trajectory_importance_sampling": True, "guards_changed": False,
        "kl_note": "exact_transition matches old run; upstream_sigma divides regularizer by sigma_t squared instead of transition variance"})
    records = [g["record"] for g in index["groups"]]
    eval_records = sorted(read_prompts(validation_data, "validation"), key=lambda r: digest(r["prompt_id"]))[:options.evaluation_prompts]
    started = time.monotonic()
    monitor = RunMonitor(output / "tensorboard", enabled=options.tensorboard)

    def save(name):
        source_boundary = {**initial["boundary"], "backend": backend.provenance(),
            "training_prompt_ids": sorted(set(initial["boundary"].get("training_prompt_ids", [])) | {r["prompt_id"] for r in records}),
            "training_caption_hashes": sorted(set(initial["boundary"].get("training_caption_hashes", [])) | {caption_hash(r["caption"]) for r in records}),
            "training_sources": [list(s) for s in sorted(set(tuple(s) for s in initial["boundary"].get("training_sources", [])) |
                                       {(r["source_dataset"], r["source_id"]) for r in records})]}
        payload = {"schema_version": 2, "config": cfg.to_dict(), "boundary": source_boundary,
            "policy": export_adapter_state(backend.decoder), "optimizer": optimizer.state_dict(),
            "groups_completed": initial["groups_completed"]+cursor,
            "optimizer_updates": previous_updates+updates, "offline_optimizer_updates": updates,
            "offline_cursor": cursor, "offline_consecutive_skips": consecutive_skips,
            "offline_resume_boundary": boundary, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if backend.device.type == "cuda" else []}
        save_tensor(output / "checkpoints" / name, payload)

    try:
        while cursor < len(order) and updates < target:
            epoch, number = order[cursor]
            group = index["groups"][number]
            before = time.monotonic()
            stats = _group_update(cfg, options, group, roots[group["shard"]] / group["path"],
                                  behaviors[group["shard"]], backend, parameters, optimizer, epoch)
            _sync(backend.device)
            cursor += 1
            updates += int(stats["updated"])
            consecutive_skips = 0 if stats["updated"] else consecutive_skips+1
            seconds = time.monotonic()-before
            entry = {**stats, "groups_seen": cursor, "offline_optimizer_updates": updates,
                "total_optimizer_updates": previous_updates+updates, "epoch": epoch,
                "prompt_id": group["record"]["prompt_id"], "behavior_checkpoint_sha256": index["shards"][group["shard"]]["behavior_checkpoint_sha256"],
                "valid_fraction": group["valid_count"]/cfg.training.group_size,
                "reward_mean": float(np.mean([r["reward"] for r in group["results"]])),
                "learning_rate": optimizer.param_groups[0]["lr"], "group_seconds": seconds,
                "optimizer_updates_per_second": int(stats["updated"])/max(seconds, 1e-9),
                "transitions_per_second": stats["replay_transitions"]/max(seconds, 1e-9), **_memory(backend.device)}
            append_json(output / "metrics.jsonl", entry)
            monitor.add_scalars({"train/loss": stats["loss"], "train/policy_loss": stats["policy_loss"],
                "train/grad_norm": stats["gradient_norm"], "train/reference_kl": stats["reference_kl"],
                "train/behavior_kl": stats["behavior_kl"], "train/clip_fraction": stats["clip_fraction"],
                "train/normalized_surrogate_ess": stats["normalized_surrogate_ess"],
                "train/maximum_abs_log_ratio": stats["maximum_abs_log_ratio"],
                "train/learning_rate": entry["learning_rate"], "train/optimizer_updates": updates,
                "train/skipped": int(not stats["updated"]), "train/valid_fraction": entry["valid_fraction"],
                "throughput/optimizer_updates_per_second": entry["optimizer_updates_per_second"],
                "throughput/transitions_per_second": entry["transitions_per_second"], **_memory_scalars(entry)}, cursor)
            write_json(output / "pipeline_state.json", {"phase": "training", "groups_seen": cursor,
                "offline_optimizer_updates": updates, "target_updates": options.updates, "last_skip_reason": stats["skip_reason"]})
            if cursor % options.checkpoint_every == 0:
                save(f"update_{updates:06d}_seen_{cursor:06d}.pt")
            if stats["updated"] and options.evaluation_every and updates % options.evaluation_every == 0:
                result = _evaluate_triplets(cfg, eval_records, backend, reward, initial_policy,
                    output / "validation" / f"update_{updates:06d}", split="validation", step=updates, monitor=monitor)
                append_json(output / "validation_metrics.jsonl", result)
            if consecutive_skips >= options.max_consecutive_skips:
                raise RuntimeError("Consecutive skip/drift budget reached; preserve cache and collect fresh behavior data")
        save("final.pt")
        status = "complete" if updates == options.updates else "budget_slice_complete" if updates == target else "insufficient_usable_data"
        summary = {"status": status, "groups_seen": cursor, "offline_optimizer_updates": updates,
                   "total_optimizer_updates": previous_updates+updates, "target_updates": options.updates,
                   "elapsed_seconds": time.monotonic()-started, "music_quality_validated": False,
                   "final_checkpoint_sha256": file_sha256(output / "checkpoints/final.pt")}
        write_json(output / "summary.json", summary)
        write_json(output / "pipeline_state.json", {"phase": status, **summary})
        return summary
    except Exception as error:
        optimizer.zero_grad(set_to_none=True)
        write_json(output / "failure.json", {"error": str(error), "groups_seen": cursor,
                   "offline_optimizer_updates": updates, "target_updates": options.updates})
        write_json(output / "pipeline_state.json", {"phase": "failed", "groups_seen": cursor,
                   "offline_optimizer_updates": updates, "error": str(error)})
        try:
            save("failure.pt")
        except Exception as backup_error:
            write_json(output / "failure.json", {"error": str(error), "groups_seen": cursor,
                "offline_optimizer_updates": updates, "checkpoint_save_error": str(backup_error)})
        raise
    finally:
        monitor.close()


def train_offline(*args, **kwargs):
    """Also preserve initialization/replay failures before the training loop."""
    output = Path(args[5] if len(args) > 5 else kwargs["output"])
    existed = output.exists()
    try:
        return _train_offline(*args, **kwargs)
    except Exception as error:
        if not existed and output.is_dir() and not (output / "failure.json").exists():
            write_json(output / "failure.json", {"stage": "initialization", "error_type": type(error).__name__, "error": str(error)})
            write_json(output / "pipeline_state.json", {"phase": "failed", "stage": "initialization", "error": str(error)})
        raise


def evaluate_offline(initial_path, initial_sha, trained_path, trained_sha, data, output, *, overrides=None,
                     split="test", backend=None, reward=None):
    initial, cfg = checkpoint(initial_path, initial_sha)
    trained, trained_cfg = checkpoint(trained_path, trained_sha)
    if trained.get("offline_resume_boundary", {}).get("initial_checkpoint_sha256") != initial_sha:
        raise ValueError("Trained checkpoint is not a continuation of this initialization")
    if experiment_contract(cfg) != experiment_contract(trained_cfg):
        raise ValueError("Evaluation checkpoints have different experiment contracts")
    cfg = runtime_config(cfg, overrides)
    rows = read_prompts(data, split)
    boundary = trained["boundary"]
    if any(r["prompt_id"] in boundary["training_prompt_ids"] or caption_hash(r["caption"]) in boundary["training_caption_hashes"] or
           [r["source_dataset"], r["source_id"]] in boundary["training_sources"] for r in rows):
        raise ValueError("Evaluation overlaps historical/continued training")
    reward = reward or create_reward(cfg)
    backend = backend or create_backend(cfg)
    setup_policy(cfg, backend)
    if model_signature(backend.provenance()) != model_signature(initial["boundary"]["backend"]):
        raise ValueError("Evaluation base weights changed")
    if analysis_signature(getattr(getattr(reward, "_analyzer", None), "receipt", None)) != analysis_signature(initial["boundary"].get("analysis_receipt")):
        raise ValueError("Evaluation frozen analyzer changed")
    load_adapter_state(backend.decoder, trained["policy"])
    return _evaluate_triplets(cfg, rows, backend, reward, initial["policy"], output,
        split=split, step=trained["offline_optimizer_updates"])
