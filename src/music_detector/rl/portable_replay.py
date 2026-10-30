"""Cross-machine likelihood replay of a closed collector probe, no optimizer."""
import json
from pathlib import Path

import torch

from .offline_io import write_json
from .offline_protocol import (checkpoint, experiment_contract, model_signature, policy_fingerprint,
                               runtime_config, validate_record)
from .offline_trajectory import verify_replay
from .policy import load_adapter_state
from .portable_delivery import inventory
from .trainer import create_backend, setup_policy


def replay_probe(root, checkpoint_path, checkpoint_sha256, output, *, overrides=None, maximum_groups=2):
    info = inventory(root, require_complete=False)
    if type(maximum_groups) is not int or not 1 <= maximum_groups <= info["closed_groups"]:
        raise ValueError("Replay requires the requested number of fully closed groups")
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Replay reports are immutable; choose a new output")
    source, source_cfg = checkpoint(checkpoint_path, checkpoint_sha256)
    cfg = runtime_config(source_cfg, overrides)
    manifest = json.loads((root/"COLLECTION_MANIFEST.json").read_text())
    policy = torch.load(root/"behavior_policy.pt", map_location="cpu", weights_only=True)
    if (manifest["checkpoint_sha256"] != checkpoint_sha256 or
            manifest["experiment_contract"] != experiment_contract(cfg) or
            policy_fingerprint(policy) != policy_fingerprint(source["policy"])):
        raise ValueError("Probe/checkpoint behavior identity mismatch")
    backend = create_backend(cfg)
    setup_policy(cfg, backend)
    load_adapter_state(backend.decoder, policy)
    backend.decoder.requires_grad_(False)
    if model_signature(backend.provenance()) != model_signature(source["boundary"]["backend"]):
        raise ValueError("Replay base-model identity differs from the collection source")
    checks = []
    for group in range(maximum_groups):
        folder = root/"groups"/f"g{group:06d}"
        condition = torch.load(folder/"condition.pt", map_location="cpu", weights_only=True)
        for sample in range(cfg.training.group_size):
            record = torch.load(folder/f"s{sample:02d}_trajectory.pt", map_location="cpu", weights_only=True)
            validate_record(record, cfg)
            checks.append({"group": group, "sample": sample,
                           **verify_replay(backend, condition, cfg, record)})
    result = {"status": "cached_behavior_replay_passed", "device": str(backend.device),
              "torch_version": str(torch.__version__), "groups_checked": maximum_groups,
              "checks": checks, "optimizer_updates": 0,
              "note": "Sampled first four stochastic transitions per candidate; not an exhaustive trajectory equality proof"}
    write_json(output, result)
    return result
