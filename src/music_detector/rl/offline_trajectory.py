"""Lossless frozen-policy trajectory records, not an offline-RL guarantee.

Store each latent state once. Deterministic solver steps have no Gaussian
likelihood: their zero placeholders MUST be masked out during training.
"""
from __future__ import annotations

import torch

from .flow import gaussian_log_prob, sample_transition, transition_mean
from .trainer import schedule


@torch.no_grad()
def full_rollout(backend, condition, cfg, seed: int, *, reference=False):
    generator = torch.Generator(device=backend.device).manual_seed(seed)
    x = torch.randn(backend.latent_shape(condition), device=backend.device,
                    dtype=torch.float32, generator=generator)
    states = [x.detach().cpu().clone()]
    times = schedule(cfg)
    logprobs, variances, masks = [], [], []
    for t, following in zip(times, times[1:]):
        eta = cfg.sampling.noise_level if (
            cfg.sampling.stochastic_t_min <= t <= cfg.sampling.stochastic_t_max) else 0.0
        velocity = backend.velocity(x, t, condition, reference=reference).float()
        following_x, _, variance, logprob = sample_transition(
            x, velocity, t, following-t, eta, generator=generator,
            reduction=cfg.sampling.logprob_reduction)
        if following_x.shape != x.shape or not torch.isfinite(following_x).all():
            raise RuntimeError("Nonfinite or malformed collected trajectory")
        stochastic = variance is not None and logprob is not None
        if stochastic and (variance.numel() != 1 or logprob.numel() != 1):
            raise ValueError("The lossless collector currently requires single-sample rollouts")
        masks.append(stochastic)
        variances.append(float(variance.item()) if stochastic else 0.0)
        logprobs.append(float(logprob.item()) if stochastic else 0.0)
        x = following_x
        states.append(x.detach().cpu().clone())
    record = {
        "schema_version": 1, "seed": seed,
        "states": torch.stack(states),
        "timesteps": torch.tensor(times[:-1], dtype=torch.float64),
        "dt": torch.tensor([b-a for a,b in zip(times,times[1:])], dtype=torch.float64),
        "variance": torch.tensor(variances, dtype=torch.float32),
        "old_logprobs": torch.tensor(logprobs, dtype=torch.float32),
        "likelihood_mask": torch.tensor(masks, dtype=torch.bool),
        "logprob_reduction": cfg.sampling.logprob_reduction,
    }
    validate_trajectory(record)
    return backend.decode(x), record


def validate_trajectory(record):
    states = record["states"]
    mask = record["likelihood_mask"]
    if states.dtype != torch.float32 or states.ndim != 4 or states.shape[1] != 1:
        raise ValueError("Expected lossless FP32 [steps+1,1,frames,channels] states")
    steps = states.shape[0]-1
    if mask.dtype != torch.bool or mask.shape != (steps,) or not mask.any():
        raise ValueError("Missing valid likelihood steps")
    if not torch.isfinite(states).all():
        raise ValueError("Nonfinite latent states")
    for name in ("timesteps", "dt", "variance", "old_logprobs"):
        value = record[name]
        if value.shape != (steps,) or not torch.isfinite(value).all():
            raise ValueError(f"Malformed trajectory field: {name}")
    if not (record["dt"] < 0).all() or not (record["variance"][mask] > 0).all():
        raise ValueError("Invalid reverse-time transition schedule")
    if not (record["variance"][~mask] == 0).all():
        raise ValueError("Deterministic likelihood placeholders must remain masked")


def transition_views(record):
    """GRPO-compatible adjacent-state views without duplicating disk storage."""
    validate_trajectory(record)
    mask = record["likelihood_mask"]
    return {"latents": record["states"][:-1][mask],
            "next_latents": record["states"][1:][mask],
            "timesteps": record["timesteps"][mask], "dt": record["dt"][mask],
            "variance": record["variance"][mask], "old_logprobs": record["old_logprobs"][mask]}


@torch.no_grad()
def verify_replay(backend, condition, cfg, record, *, maximum_steps=4, tolerance=1e-6):
    """Verify the frozen policy reproduces recorded likelihoods before scaling."""
    validate_trajectory(record)
    ids = torch.where(record["likelihood_mask"])[0].tolist()[:maximum_steps]
    differences = []
    for index in ids:
        x = record["states"][index].to(backend.device)
        action = record["states"][index+1].to(backend.device)
        t, dt = float(record["timesteps"][index]), float(record["dt"][index])
        v = backend.velocity(x, t, condition).float()
        mean = transition_mean(x, v, t, dt, cfg.sampling.noise_level)
        probability = gaussian_log_prob(action, mean,
            record["variance"][index].to(backend.device), cfg.sampling.logprob_reduction)
        differences.append(abs(float(probability.item())-float(record["old_logprobs"][index])))
    worst = max(differences)
    if worst > tolerance:
        raise RuntimeError(f"Frozen-policy replay logprob mismatch: {worst} > {tolerance}")
    return {"steps_verified": len(ids), "maximum_abs_logprob_difference": worst,
            "tolerance": tolerance}
