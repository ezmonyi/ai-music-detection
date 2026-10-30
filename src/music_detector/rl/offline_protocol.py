"""Portable identities for a new, explicitly off-policy experiment.

This does not relax the historical online trainer's exact-resume boundary.
Runtime paths are relocatable; model weights, sampler and reward are not.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import math

import torch

from .config import config_from_dict
from .data import caption_hash, file_sha256
from .rewards import RewardGuards


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def experiment_contract(cfg):
    """Only quantities which define the cached transitions and rewards."""
    model = cfg.to_dict()["model"]
    return {
        "model": {k: model[k] for k in ("backend", "checkpoint_name", "upstream_commit", "precision")},
        "policy": asdict(cfg.policy),
        "sampling": {k: v for k, v in asdict(cfg.sampling).items() if k != "train_timesteps"},
        "group_size": cfg.training.group_size,
        "reward": {"kind": cfg.reward.kind, "families": cfg.reward.families,
                   "bundle_sha256": cfg.reward.bundle_sha256,
                   "guards": asdict(RewardGuards(**cfg.reward.guards)),
                   "analyzer_seed": cfg.reward.analyzer_seed},
    }


def model_signature(provenance):
    keys = ("backend", "upstream_commit", "checkpoint_name", "artefact_hashes",
            "checkpoint_architecture", "vae_architecture", "sample_rate", "latent_hz", "latent_channels")
    return {k: provenance[k] for k in keys if k in provenance}


def analysis_signature(receipt):
    """Keep algorithm/weight/library identities, omit relocatable cache paths."""
    if receipt is None:
        return None
    def normalize(value):
        if isinstance(value, dict):
            return {k: (str(v).split(":")[0] if k == "device" else normalize(v))
                    for k, v in value.items() if k not in {
                        "cache_dir", "checkpoint_dir", "path", "python_executable", "executable"}}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    return normalize(receipt)


def policy_fingerprint(payload):
    """A stable tensor identity, independent of torch.save ZIP filenames."""
    result = hashlib.sha256(json.dumps(payload["config"], sort_keys=True).encode())
    tensors = payload.get("state_dict", payload.get("state"))
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Missing behavior policy tensors")
    for name, value in sorted(tensors.items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("Nonfinite or malformed policy tensor")
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def checkpoint(path, expected_sha256):
    path = Path(path)
    if len(expected_sha256) != 64 or file_sha256(path) != expected_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema_version") not in {1, 2} or not all(k in value for k in ("config", "boundary", "policy")):
        raise ValueError("Unsupported continuation checkpoint")
    cfg = config_from_dict(value["config"])
    if cfg.policy.mode != "lora":
        raise ValueError("Portable offline continuation currently supports the existing LoRA checkpoint only")
    policy_fingerprint(value["policy"])
    return value, cfg


def runtime_config(cfg, overrides=None):
    """Relocate runtime components without changing the experiment contract."""
    overrides = overrides or {}
    allowed = {"model": {"device", "upstream_dir", "checkpoint_dir", "gradient_checkpointing"},
               "reward": {"device", "analyzer_python", "analyzer_cache_dir", "analyzer_timeout_s"}}
    if set(overrides) - set(allowed):
        raise ValueError("Unknown runtime override section")
    value = cfg.to_dict()
    for section, fields in overrides.items():
        if not isinstance(fields, dict) or set(fields) - allowed[section]:
            raise ValueError("Runtime overrides may not change weights, precision, sampler, reward or guards")
        value[section].update(fields)
    result = config_from_dict(value)
    if experiment_contract(result) != experiment_contract(cfg):
        raise ValueError("Runtime override changed the experiment contract")
    return result


def prompt_keys(row):
    return row["prompt_id"], caption_hash(row["caption"]), (row["source_dataset"], row["source_id"])


def reject_overlap(records, excluded, *, description):
    sets = [set(keys[i] for keys in map(prompt_keys, excluded)) for i in range(3)]
    if any(any(keys[i] in sets[i] for i in range(3)) for keys in map(prompt_keys, records)):
        raise ValueError("Prompt/caption/source overlap with " + description)


def member(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Unsafe manifest member path")
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError("Unsafe manifest symlink")
    if path.is_symlink() or root not in path.resolve().parents or not path.is_file():
        raise ValueError("Unsafe or missing manifest member")
    return path


def verify_members(root, entries):
    seen = set()
    for entry in entries:
        name = entry["path"]
        if name in seen:
            raise ValueError("Duplicate manifest member")
        seen.add(name)
        path = member(root, name)
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError("Dataset member checksum mismatch: " + name)
    return seen


def validate_record(record, cfg):
    from .offline_trajectory import validate_trajectory
    from .trainer import schedule
    validate_trajectory(record)
    if (any(record[name].dtype != torch.float32 for name in ("variance", "old_logprobs")) or
            any(record[name].dtype != torch.float64 for name in ("timesteps", "dt"))):
        raise ValueError("Trajectory likelihood/schedule dtype differs from the lossless contract")
    times = torch.tensor(schedule(cfg), dtype=torch.float64)
    mask = (times[:-1] >= cfg.sampling.stochastic_t_min) & (times[:-1] <= cfg.sampling.stochastic_t_max)
    if record["states"].shape[0] != cfg.sampling.steps + 1:
        raise ValueError("Trajectory solver length mismatch")
    if not torch.equal(record["timesteps"].double(), times[:-1]) or not torch.equal(record["dt"].double(), times.diff()):
        raise ValueError("Trajectory schedule mismatch")
    if not torch.equal(record["likelihood_mask"], mask):
        raise ValueError("Trajectory stochastic mask mismatch")
    if record["logprob_reduction"] != cfg.sampling.logprob_reduction:
        raise ValueError("Log-density reduction mismatch")
    expected_variance = (cfg.sampling.noise_level**2 * times[:-1][mask] /
                         (1-times[:-1][mask]) * -times.diff()[mask]).float()
    if not torch.allclose(record["variance"][mask], expected_variance, rtol=2e-6, atol=1e-8):
        raise ValueError("Trajectory variance differs from the sampler")
    if cfg.model.backend == "acestep" and tuple(record["states"].shape[2:]) != (750, 64):
        raise ValueError("Expected native 30-second ACE-Step latent shape")


def finite_results(results, group_size):
    if len(results) != group_size or [r["sample"] for r in results] != list(range(group_size)):
        raise ValueError("Candidate group is incomplete or reordered")
    for result in results:
        if type(result["valid"]) is not bool or isinstance(result["reward"], bool) or not math.isfinite(result["reward"]):
            raise ValueError("Unscored or nonfinite reward result")
