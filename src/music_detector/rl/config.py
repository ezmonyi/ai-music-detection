"""Strict, serializable experiment settings (no implicit remote downloads)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "toy"
    device: str = "cpu"
    upstream_dir: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_name: str = "acestep-v15-sft"
    upstream_commit: str | None = None
    precision: str = "bfloat16"
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class PolicyConfig:
    mode: str = "lora"
    rank: int = 32
    alpha: float = 64.0
    targets: str = "all_linear"


@dataclass(frozen=True)
class SamplingConfig:
    duration_s: float = 30.0
    steps: int = 50
    time_shift: float = 1.0
    noise_level: float = 0.7
    stochastic_t_min: float = 0.2
    stochastic_t_max: float = 0.8
    train_timesteps: int = 4
    logprob_reduction: str = "mean"
    guidance_scale: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 100
    group_size: int = 4
    seed: int = 11
    learning_rate: float = 0.0001
    clip_range: float = 0.2
    kl_coefficient: float = 0.01
    advantage_clip: float = 5.0
    max_grad_norm: float = 1.0
    checkpoint_every: int = 10
    save_audio_every: int = 10
    all_invalid_policy: str = "abort"
    max_all_invalid_groups: int = 20
    max_consecutive_all_invalid_groups: int = 3


@dataclass(frozen=True)
class RewardConfig:
    kind: str = "toy"
    families: list[str] = field(default_factory=lambda: ["F", "SC"])
    bundle_sha256: str | None = None
    device: str = "cpu"
    guards: dict = field(default_factory=dict)
    # The released detector needs torch 2.8, whereas the ACE training runtime
    # uses torch 2.10. Never relax the frozen detector's version checks.
    analyzer_python: str | None = None
    analyzer_cache_dir: str | None = None
    analyzer_seed: int = 0
    analyzer_timeout_s: int = 600


@dataclass(frozen=True)
class MonitoringConfig:
    # Opt-in for older configurations; all real pilot templates enable it.
    tensorboard: bool = False
    evaluation_every: int = 0
    evaluation_prompts: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                          allow_nan=False).encode()).hexdigest()


def _construct(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a JSON object")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


def config_from_dict(value: dict) -> ExperimentConfig:
    if not isinstance(value, dict) or set(value) - {
            "schema_version", "model", "policy", "sampling", "training", "reward", "monitoring"}:
        raise ValueError("Unknown experiment fields or non-object config")
    cfg = ExperimentConfig(
        schema_version=value.get("schema_version", 1),
        **{name: _construct(cls, value.get(name, {})) for name, cls in (
            ("model", ModelConfig), ("policy", PolicyConfig),
            ("sampling", SamplingConfig), ("training", TrainingConfig),
            ("reward", RewardConfig), ("monitoring", MonitoringConfig))})
    validate_config(cfg)
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    return config_from_dict(json.loads(Path(path).read_text()))


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.schema_version != 1:
        raise ValueError("Unsupported experiment schema")
    if cfg.model.backend not in {"toy", "acestep"}:
        raise ValueError("backend must be toy or acestep")
    if not isinstance(cfg.model.device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", cfg.model.device):
        raise ValueError("Use explicit cpu or cuda device; no automatic fallback")
    if type(cfg.model.gradient_checkpointing) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    if type(cfg.monitoring.tensorboard) is not bool:
        raise ValueError("monitoring.tensorboard must be boolean")
    if type(cfg.monitoring.evaluation_every) is not int or cfg.monitoring.evaluation_every < 0:
        raise ValueError("evaluation_every must be a nonnegative integer; 0 disables periodic validation")
    if cfg.model.precision not in {"float32", "bfloat16"}:
        raise ValueError("Only float32 and bfloat16 are supported")
    if cfg.policy.mode not in {"lora", "full"}:
        raise ValueError("policy.mode must be lora or full (FM/DiT decoder only)")
    if cfg.policy.targets not in {"all_linear", "attention"}:
        raise ValueError("policy.targets must be all_linear or attention")
    if not isinstance(cfg.training.all_invalid_policy, str) or cfg.training.all_invalid_policy not in {"abort", "skip_bounded"}:
        raise ValueError("all_invalid_policy must be abort or skip_bounded")
    for name, number, minimum in (
        ("rank", cfg.policy.rank, 1), ("steps", cfg.sampling.steps, 3),
        ("train_timesteps", cfg.sampling.train_timesteps, 1),
        ("updates", cfg.training.updates, 1), ("group_size", cfg.training.group_size, 2),
        ("checkpoint_every", cfg.training.checkpoint_every, 1),
        ("save_audio_every", cfg.training.save_audio_every, 1),
        ("evaluation_prompts", cfg.monitoring.evaluation_prompts, 1),
        ("max_all_invalid_groups", cfg.training.max_all_invalid_groups, 1),
        ("max_consecutive_all_invalid_groups", cfg.training.max_consecutive_all_invalid_groups, 1),
    ):
        if type(number) is not int or number < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(cfg.training.seed) is not int or cfg.training.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, number in (
        ("alpha", cfg.policy.alpha), ("duration_s", cfg.sampling.duration_s),
        ("time_shift", cfg.sampling.time_shift), ("noise_level", cfg.sampling.noise_level),
        ("learning_rate", cfg.training.learning_rate), ("max_grad_norm", cfg.training.max_grad_norm),
        ("advantage_clip", cfg.training.advantage_clip),
    ):
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, number in (("stochastic_t_min", cfg.sampling.stochastic_t_min),
                         ("stochastic_t_max", cfg.sampling.stochastic_t_max),
                         ("clip_range", cfg.training.clip_range),
                         ("kl_coefficient", cfg.training.kl_coefficient),
                         ("guidance_scale", cfg.sampling.guidance_scale)):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
    if not 0 < cfg.sampling.stochastic_t_min < cfg.sampling.stochastic_t_max < 1:
        raise ValueError("The stochastic window must lie strictly inside (0, 1)")
    if cfg.sampling.logprob_reduction not in {"mean", "sum"}:
        raise ValueError("logprob_reduction must be mean or sum")
    if not 0 < cfg.training.clip_range < 1:
        raise ValueError("clip_range must lie in (0, 1)")
    if not math.isfinite(cfg.training.kl_coefficient) or cfg.training.kl_coefficient < 0:
        raise ValueError("kl_coefficient must be finite and nonnegative")
    if cfg.sampling.guidance_scale != 1.0:
        raise ValueError("Initial adapter supports guidance_scale=1 only; CFG must be implemented and tested before enabling")
    if cfg.reward.kind not in {"toy", "artifact"}:
        raise ValueError("reward.kind must be toy or artifact")
    if type(cfg.reward.analyzer_seed) is not int or cfg.reward.analyzer_seed < 0:
        raise ValueError("reward.analyzer_seed must be a nonnegative integer")
    if type(cfg.reward.analyzer_timeout_s) is not int or cfg.reward.analyzer_timeout_s < 1:
        raise ValueError("reward.analyzer_timeout_s must be a positive integer")
    if cfg.reward.analyzer_python is not None:
        if cfg.reward.kind != "artifact" or not isinstance(cfg.reward.analyzer_python, str) or not Path(cfg.reward.analyzer_python).is_absolute():
            raise ValueError("An isolated artifact analyzer requires an absolute Python executable path")
        if not isinstance(cfg.reward.analyzer_cache_dir, str) or not Path(cfg.reward.analyzer_cache_dir).is_absolute():
            raise ValueError("An isolated analyzer requires an absolute analysis cache path")
    elif cfg.reward.analyzer_cache_dir is not None:
        raise ValueError("analyzer_cache_dir requires analyzer_python")
    if (cfg.model.backend == "toy") != (cfg.reward.kind == "toy"):
        raise ValueError("Synthetic backend/reward must stay paired; synthetic runs are not music evidence")
    if cfg.model.backend == "acestep":
        if not cfg.model.device.startswith("cuda"):
            raise ValueError("Actual ACE-Step RL requires an explicit CUDA device; use toy for CPU testing")
        if cfg.sampling.duration_s != 30.0:
            raise ValueError("The frozen artifact reward currently requires the 30-second experiment contract")
        if not all((cfg.model.upstream_dir, cfg.model.checkpoint_dir, cfg.model.upstream_commit)):
            raise ValueError("ACE-Step requires explicit local upstream/checkpoint paths and pinned commit")
        if "sft" not in cfg.model.checkpoint_name or "turbo" in cfg.model.checkpoint_name:
            raise ValueError("This RL protocol targets the SFT checkpoint, not the historical Turbo corpus")
        if not isinstance(cfg.model.upstream_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", cfg.model.upstream_commit):
            raise ValueError("upstream_commit must be a full lowercase git SHA")
        if not isinstance(cfg.reward.bundle_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", cfg.reward.bundle_sha256):
            raise ValueError("Pin the frozen detector bundle SHA-256 for artifact reward")
