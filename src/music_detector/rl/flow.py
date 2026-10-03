"""Small, framework-independent Flow-GRPO math helpers.

The flow-matching path used by ACE-Step is parameterised as

``x_t = t * noise + (1 - t) * data`` and ``v_t = d x_t / d t``.

Rollouts run this path backwards (``dt < 0``).  For an interior time the
Flow-GRPO reverse-time SDE has

``sigma_t = eta * sqrt(t / (1 - t))``

and the Euler--Maruyama transition ``N(mu_t, sigma_t**2 * (-dt) I)`` where
``mu_t`` is implemented by :func:`transition_mean`.  At the two endpoints
the SDE is singular or has zero variance; we deliberately use a deterministic
Euler step there.  In particular, a zero variance is never silently clamped
to a small value just to manufacture a Gaussian log density.

``reduction="mean"`` is the dimension-normalised log-density convention used
by the official Flow-GRPO implementation (one value per batch item for a
batched tensor).  ``reduction="sum"`` is the joint event log density.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, float]


def _as_schedule(value: TensorLike, reference: torch.Tensor, *, name: str) -> torch.Tensor:
    """Convert a scalar/schedule to the reference dtype/device."""

    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference state must be a torch.Tensor")
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not (result.is_floating_point() or result.is_complex()):
        result = result.to(dtype=reference.dtype)
    if result.is_complex():
        raise TypeError(f"{name} must be real-valued")
    return result.to(device=reference.device, dtype=reference.dtype)


def _validate_state(x: torch.Tensor, velocity: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or not isinstance(velocity, torch.Tensor):
        raise TypeError("x and velocity must be torch.Tensor instances")
    if not (x.is_floating_point() and velocity.is_floating_point()):
        raise TypeError("x and velocity must use a floating-point dtype")
    if x.device != velocity.device:
        raise ValueError("x and velocity must be on the same device")
    # torch's broadcast rules are the useful contract here.  Probe them so a
    # malformed state fails with a clear error before doing SDE arithmetic.
    try:
        torch.broadcast_shapes(x.shape, velocity.shape)
    except RuntimeError as exc:
        raise ValueError("x and velocity shapes are not broadcastable") from exc


def _validate_time(t: torch.Tensor, dt: torch.Tensor, noise_level: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        raise ValueError("t must contain only finite values")
    if bool(((t < 0) | (t > 1)).any()):
        raise ValueError("t must lie in the closed interval [0, 1]")
    if not torch.isfinite(dt).all():
        raise ValueError("dt must contain only finite values")
    if bool((dt >= 0).any()):
        raise ValueError("reverse-time integration requires dt < 0")
    if not torch.isfinite(noise_level).all():
        raise ValueError("noise_level must contain only finite values")
    if bool((noise_level < 0).any()):
        raise ValueError("noise_level must be non-negative")


def _broadcast_schedule(value: torch.Tensor, state: torch.Tensor, *, name: str) -> torch.Tensor:
    """Broadcast a schedule over the state event dimensions.

    A scalar schedule and a leading-batch schedule are both accepted.  A
    schedule with shape ``(B,)`` is reshaped to ``(B, 1, ..., 1)`` when the
    state has more dimensions, matching the conventions used by diffusion
    schedulers.  Other shapes are delegated to PyTorch broadcasting.
    """

    if value.ndim == 0:
        return value
    if value.ndim == 1 and state.ndim > 1 and value.shape[0] == state.shape[0]:
        value = value.reshape((value.shape[0],) + (1,) * (state.ndim - 1))
    try:
        return torch.broadcast_to(value, torch.broadcast_shapes(value.shape, state.shape))
    except RuntimeError as exc:
        raise ValueError(f"{name} shape {tuple(value.shape)} cannot broadcast to state {tuple(state.shape)}") from exc


def transition_mean(
    x: torch.Tensor,
    velocity: torch.Tensor,
    t: TensorLike,
    dt: TensorLike,
    noise_level: TensorLike,
) -> torch.Tensor:
    """Return the reverse-time Flow-GRPO Euler drift mean.

    For ``0 < t < 1`` this is exactly

    ``x * (1 + sigma**2 / (2*t) * dt)``
    ``+ velocity * (1 + sigma**2 * (1-t) / (2*t)) * dt``.

    ``t == 0`` and ``t == 1`` are endpoint Euler steps ``x + velocity * dt``;
    the reverse SDE itself is not evaluated there because its variance or
    drift is singular.  ``dt`` must be strictly negative.
    """

    _validate_state(x, velocity)
    t_tensor = _as_schedule(t, x, name="t")
    dt_tensor = _as_schedule(dt, x, name="dt")
    noise_tensor = _as_schedule(noise_level, x, name="noise_level")
    # Schedule dimensions are expanded after validation so a [B] schedule
    # works for latent states [B, C, H, W].
    t_tensor = _broadcast_schedule(t_tensor, x, name="t")
    dt_tensor = _broadcast_schedule(dt_tensor, x, name="dt")
    noise_tensor = _broadcast_schedule(noise_tensor, x, name="noise_level")
    _validate_time(t_tensor, dt_tensor, noise_tensor)

    endpoint = (t_tensor == 0) | (t_tensor == 1)
    # torch.where evaluates both branches, so use safe interior values before
    # calculating ratios.  The result is then replaced by the endpoint Euler
    # value, avoiding NaN/Inf contamination from a singular branch.
    safe_t = torch.where(endpoint, torch.ones_like(t_tensor), t_tensor)
    safe_one_minus_t = torch.where(endpoint, torch.ones_like(t_tensor), 1 - t_tensor)
    sigma_sq = noise_tensor.square() * safe_t / safe_one_minus_t
    interior_mean = (
        x * (1 + sigma_sq / (2 * safe_t) * dt_tensor)
        + velocity * (1 + sigma_sq * safe_one_minus_t / (2 * safe_t)) * dt_tensor
    )
    endpoint_mean = x + velocity * dt_tensor
    return torch.where(endpoint, endpoint_mean, interior_mean)


def _event_reduce(log_density: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be either 'mean' or 'sum'")
    if log_density.ndim <= 1:
        return log_density.mean() if reduction == "mean" else log_density.sum()
    event_dims = tuple(range(1, log_density.ndim))
    return log_density.mean(dim=event_dims) if reduction == "mean" else log_density.sum(dim=event_dims)


def _validate_variance(variance: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    variance = variance if isinstance(variance, torch.Tensor) else torch.as_tensor(variance)
    variance = variance.to(device=reference.device, dtype=reference.dtype)
    if not variance.is_floating_point():
        variance = variance.to(dtype=reference.dtype)
    if not torch.isfinite(variance).all():
        raise ValueError("variance must contain only finite values")
    if bool((variance <= 0).any()):
        raise ValueError("variance must be strictly positive for a Gaussian density")
    return variance


def gaussian_log_prob(
    next_state: torch.Tensor,
    mean: torch.Tensor,
    variance: TensorLike,
    reduction: str = "mean",
) -> torch.Tensor:
    """Evaluate an isotropic/elementwise Gaussian transition log density.

    ``variance`` is the per-element variance (not standard deviation).  The
    strict positivity check is intentional: deterministic endpoint transitions
    must be excluded from policy log probabilities instead of represented by a
    fake near-zero Gaussian.
    """

    if not isinstance(next_state, torch.Tensor) or not isinstance(mean, torch.Tensor):
        raise TypeError("next_state and mean must be torch.Tensor instances")
    if not (next_state.is_floating_point() and mean.is_floating_point()):
        raise TypeError("next_state and mean must use a floating-point dtype")
    try:
        diff = next_state - mean
    except RuntimeError as exc:
        raise ValueError("next_state and mean shapes are not broadcastable") from exc
    var = _validate_variance(variance, diff)
    try:
        density = -0.5 * (diff.square() / var + torch.log(var) + math.log(2.0 * math.pi))
    except RuntimeError as exc:
        raise ValueError("variance shape is not broadcastable to next_state") from exc
    return _event_reduce(density, reduction)


def _randn_like(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(x)
    try:
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    except RuntimeError as exc:
        # A CPU generator can be paired with a CPU state only.  This explicit
        # error is easier to diagnose than the backend-specific randn error.
        raise ValueError("generator device is incompatible with the transition state") from exc


def sample_transition(
    x: torch.Tensor,
    velocity: torch.Tensor,
    t: TensorLike,
    dt: TensorLike,
    noise_level: TensorLike,
    generator: Optional[torch.Generator] = None,
    reduction: str = "mean",
    *,
    return_logprob: bool = True,
    return_log_prob: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]] | torch.Tensor:
    """Sample one reverse-time transition and return its rollout statistics.

    Returns ``(next_state, mean, variance, old_logprob)``.  For an interior
    stochastic step, ``variance`` is the transition variance and
    ``old_logprob`` is the detached Gaussian log density of the sampled state.
    For endpoint or ``noise_level == 0`` deterministic steps both are
    ``None``; callers must omit these transitions from policy log probabilities.

    The implementation accepts scalar ``t``/``dt`` values and leading-batch
    schedules.  A batch containing a mixture of stochastic and deterministic
    entries is returned with a variance tensor only when every entry has a
    valid positive variance; mixed batches should be split by the caller so
    endpoint entries cannot accidentally be scored as Gaussian actions.  Set
    ``return_logprob=False`` (or its spelling ``return_log_prob=False``) when
    only the sampled state is needed.
    """

    if return_log_prob is not None:
        return_logprob = bool(return_log_prob)
    _validate_state(x, velocity)
    t_tensor = _as_schedule(t, x, name="t")
    dt_tensor = _as_schedule(dt, x, name="dt")
    noise_tensor = _as_schedule(noise_level, x, name="noise_level")
    t_tensor = _broadcast_schedule(t_tensor, x, name="t")
    dt_tensor = _broadcast_schedule(dt_tensor, x, name="dt")
    noise_tensor = _broadcast_schedule(noise_tensor, x, name="noise_level")
    _validate_time(t_tensor, dt_tensor, noise_tensor)

    mean = transition_mean(x, velocity, t_tensor, dt_tensor, noise_tensor)
    endpoint = (t_tensor == 0) | (t_tensor == 1)
    sigma_sq = noise_tensor.square() * t_tensor / torch.where(
        endpoint, torch.ones_like(t_tensor), 1 - t_tensor
    )
    variance = sigma_sq * (-dt_tensor)
    stochastic = (~endpoint) & (variance > 0)
    if bool(stochastic.any()):
        noise = _randn_like(mean, generator)
        # Multiplication by sqrt(variance) gives the elementwise isotropic
        # Euler--Maruyama noise.  Deterministic entries are masked explicitly.
        next_state = mean + torch.sqrt(torch.clamp_min(variance, 0)) * noise * stochastic.to(mean.dtype)
    else:
        next_state = mean

    # For scalar/whole-batch calls this is the usual path.  A mixed schedule
    # cannot be represented by one Gaussian density without silently inventing
    # endpoint likelihoods, so expose no log-probability in that case.
    all_stochastic = bool(stochastic.all())
    if all_stochastic:
        old_logprob = gaussian_log_prob(next_state.detach(), mean.detach(), variance, reduction=reduction).detach()
        returned_variance: Optional[torch.Tensor] = variance
    else:
        old_logprob = None
        returned_variance = variance if bool(stochastic.any()) else None
    if not return_logprob:
        return next_state
    return next_state, mean, returned_variance, old_logprob


def group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
    clip: Optional[float] = 5,
) -> torch.Tensor:
    """Compute group-relative, zero-mean reward advantages.

    The last dimension is the group dimension; a one-dimensional reward tensor
    is treated as one group.  Population standard deviation (``unbiased=False``)
    matches GRPO's finite-group normalization.  ``clip=None`` disables the
    optional symmetric bound.
    """

    if not isinstance(rewards, torch.Tensor):
        rewards = torch.as_tensor(rewards)
    if not rewards.is_floating_point():
        rewards = rewards.float()
    if not math.isfinite(float(eps)) or eps < 0:
        raise ValueError("eps must be a finite non-negative number")
    if clip is not None and (not math.isfinite(float(clip)) or clip <= 0):
        raise ValueError("clip must be a positive finite number or None")
    if rewards.ndim == 0:
        # A one-sample group has no useful relative signal but is valid.
        centered = rewards - rewards
        return centered
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    advantages = (rewards - mean) / (std + eps)
    if clip is not None:
        advantages = advantages.clamp(min=-float(clip), max=float(clip))
    return advantages


def clipped_policy_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """Return the negative PPO/GRPO clipped surrogate objective."""

    if not isinstance(new_logprob, torch.Tensor) or not isinstance(old_logprob, torch.Tensor):
        raise TypeError("new_logprob and old_logprob must be torch.Tensor instances")
    if not isinstance(advantages, torch.Tensor):
        advantages = torch.as_tensor(advantages, device=new_logprob.device, dtype=new_logprob.dtype)
    if not math.isfinite(float(clip_range)) or not 0 <= float(clip_range) < 1:
        raise ValueError("clip_range must satisfy 0 <= clip_range < 1")
    try:
        ratio = torch.exp(new_logprob - old_logprob.detach())
        adv = advantages.to(device=ratio.device, dtype=ratio.dtype).detach()
        unclipped = ratio * adv
        clipped = ratio.clamp(1 - float(clip_range), 1 + float(clip_range)) * adv
        return -torch.minimum(unclipped, clipped).mean()
    except RuntimeError as exc:
        raise ValueError("new_logprob, old_logprob, and advantages must broadcast") from exc


def gaussian_kl(
    mean: torch.Tensor,
    reference_mean: torch.Tensor,
    variance: TensorLike,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL divergence between equal-variance Gaussian transitions.

    The reference policy is represented by ``reference_mean`` and shares the
    transition variance.  Thus ``KL(N(mean,var) || N(reference_mean,var))`` is
    ``(mean-reference_mean)**2 / (2*var)`` elementwise.
    """

    if not isinstance(mean, torch.Tensor) or not isinstance(reference_mean, torch.Tensor):
        raise TypeError("mean and reference_mean must be torch.Tensor instances")
    if not (mean.is_floating_point() and reference_mean.is_floating_point()):
        raise TypeError("mean and reference_mean must use a floating-point dtype")
    try:
        diff = mean - reference_mean
    except RuntimeError as exc:
        raise ValueError("mean and reference_mean shapes are not broadcastable") from exc
    var = _validate_variance(variance, diff)
    try:
        elementwise = diff.square() / (2 * var)
    except RuntimeError as exc:
        raise ValueError("variance shape is not broadcastable to mean") from exc
    return _event_reduce(elementwise, reduction)


__all__ = [
    "transition_mean",
    "gaussian_log_prob",
    "sample_transition",
    "group_advantages",
    "clipped_policy_loss",
    "gaussian_kl",
]
