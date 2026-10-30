"""Plan and collect unscored portable trajectories on a dedicated RTX 5060.

Frozen native behavior policy, four sequential candidates, all-invalid outputs
retained. The frozen S+R+F analyzer scores immutable WAVs later on the cloud.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time

import torch

from .data import file_sha256
from .offline_io import cpu_condition, file_manifest, save_tensor, write_json
from .offline_protocol import (checkpoint, digest, experiment_contract, model_signature,
                               policy_fingerprint, prompt_keys, runtime_config, verify_members)
from .offline_trajectory import full_rollout, verify_replay
from .policy import export_adapter_state, load_adapter_state
from .trainer import create_backend, derive_seed, read_prompts, rollout, setup_policy, _memory, _save_audio


def plan_collection(train_data, validation_data, test_data, checkpoint_path, checkpoint_sha256,
                    output, *, count=1000, exclude_prompts=()):
    saved, cfg = checkpoint(checkpoint_path, checkpoint_sha256)
    rows = read_prompts(train_data, "train")
    held = read_prompts(validation_data, "validation") + read_prompts(test_data, "test")
    excluded = list(held)
    for path in exclude_prompts:
        value = json.loads(Path(path).read_text())
        excluded.extend(value if isinstance(value, list) else value["records"])
    blocked = [set(key[i] for key in map(prompt_keys, excluded)) for i in range(3)]
    # Exclude the checkpoint's entire historical train pool, not just inferred
    # group counts. This conservatively prevents accidentally reusing prompts.
    boundary = saved["boundary"]
    blocked[0].update(boundary.get("training_prompt_ids", []))
    blocked[1].update(boundary.get("training_caption_hashes", []))
    blocked[2].update(tuple(s) for s in boundary.get("training_sources", []))
    eligible = [r for r in rows if not any(k in blocked[i] for i, k in enumerate(prompt_keys(r)))]
    if type(count) is not int or count < 1 or len(eligible) < count:
        raise ValueError(f"Need {count} unique unused training prompts; only {len(eligible)} eligible. Supply more provenance-complete prompts")
    selected = eligible[:count]
    if any(r["duration_s"] != cfg.sampling.duration_s for r in selected):
        raise ValueError("Collection prompt duration differs from the source checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "PROMPTS.json", selected)
    write_json(output / "SOURCE_CONFIG.json", saved["config"])
    write_json(output / "SOURCE_BOUNDARY.json", boundary)
    plan = {"schema_version": 2, "kind": "portable_collection_plan", "prompt_groups": count,
            "group_size": cfg.training.group_size, "candidate_count": count*cfg.training.group_size,
            "checkpoint_sha256": checkpoint_sha256,
            "behavior_policy_fingerprint": policy_fingerprint(saved["policy"]),
            "source_groups_completed": saved["groups_completed"],
            "experiment_contract": experiment_contract(cfg),
            "validation_sha256": file_sha256(validation_data), "test_sha256": file_sha256(test_data),
            "prompts_sha256": file_sha256(output / "PROMPTS.json"),
            "source_config_sha256": file_sha256(output / "SOURCE_CONFIG.json"),
            "source_boundary_sha256": file_sha256(output / "SOURCE_BOUNDARY.json"),
            "optimizer_updates": 0, "scoring": "deferred_frozen_analyzer",
            "estimated_payload_bytes": count*150_000_000 if cfg.model.backend == "acestep" else None,
            "estimate_note": "Planning allowance, not measured RTX 5060 storage or throughput",
            "update_note": "Prompt groups are not guaranteed optimizer updates; invalid/zero-advantage/stale groups can be skipped"}
    write_json(output / "PLAN.json", plan)
    return plan


def collect_portable(plan_dir, checkpoint_path, checkpoint_sha256, output, *, overrides=None,
                     max_groups=None, resume=False, staged=True, disk_reserve_gib=2,
                     backend=None):
    plan_dir, output = Path(plan_dir), Path(output)
    plan = json.loads((plan_dir / "PLAN.json").read_text())
    saved, source_cfg = checkpoint(checkpoint_path, checkpoint_sha256)
    cfg = runtime_config(source_cfg, overrides)
    for name, key in [("PROMPTS.json", "prompts_sha256"), ("SOURCE_CONFIG.json", "source_config_sha256"),
                      ("SOURCE_BOUNDARY.json", "source_boundary_sha256")]:
        if file_sha256(plan_dir / name) != plan[key]:
            raise ValueError("Collection plan member changed")
    if (plan["checkpoint_sha256"] != checkpoint_sha256 or
            plan["behavior_policy_fingerprint"] != policy_fingerprint(saved["policy"]) or
            plan["experiment_contract"] != experiment_contract(cfg)):
        raise ValueError("Collection plan/behavior checkpoint identity mismatch")
    selected = json.loads((plan_dir / "PROMPTS.json").read_text())
    if len(selected) != plan["prompt_groups"]:
        raise ValueError("Collection plan group count mismatch")
    if max_groups is not None and (type(max_groups) is not int or max_groups < 1):
        raise ValueError("max_groups must be a positive prefix budget")
    stop = min(len(selected), max_groups) if max_groups else len(selected)
    header = {"schema_version": 2, "kind": "unscored_behavior_flow_grpo_cache",
              "checkpoint_sha256": checkpoint_sha256, "behavior_groups_completed": saved["groups_completed"],
              "behavior_policy_fingerprint": plan["behavior_policy_fingerprint"],
              "prompt_groups": len(selected), "group_size": cfg.training.group_size,
              "experiment_contract": experiment_contract(cfg), "plan_sha256": file_sha256(plan_dir / "PLAN.json"),
              "validation_data_sha256": plan["validation_sha256"], "test_data_sha256": plan["test_sha256"],
              "config_sha256": cfg.digest(), "trajectory_dtype": "float32", "solver_steps": cfg.sampling.steps,
              "deterministic_steps_masked": True, "optimizer_updates": 0,
              "offline_reuse_is_off_policy": True, "guards_changed": False,
              "reward_families": cfg.reward.families, "component_staging": staged}
    start = 0
    if output.exists():
        if not resume or (output / "FILES_SHA256.json").exists():
            raise FileExistsError("Use a new output or resume an incomplete cache; never overwrite completed data")
        if json.loads((output / "COLLECTION_MANIFEST.json").read_text()) != header:
            raise ValueError("Collection resume boundary changed")
        for folder in sorted((output / "groups").glob("g*")):
            group_path = folder / "group.json"
            if not group_path.exists():
                # Recoverable quarantine of this task's unfinished group only.
                quarantine = output / "incomplete" / (folder.name + "-" + str(time.time_ns()))
                quarantine.parent.mkdir(exist_ok=True)
                folder.rename(quarantine)
                continue
            group = json.loads(group_path.read_text())
            if folder.name != f"g{start:06d}" or group["prompt_id"] != selected[start]["prompt_id"] or not group["closed"]:
                raise ValueError("Collection resume requires an intact contiguous closed prefix")
            verify_members(folder, group["files"])
            start += 1
    else:
        output.mkdir(parents=True)
        (output / "groups").mkdir()
        write_json(output / "COLLECTION_MANIFEST.json", header)
        write_json(output / "CONFIG.json", cfg.to_dict())
        write_json(output / "PROMPTS.json", selected)
        write_json(output / "SOURCE_BOUNDARY.json", saved["boundary"])
        save_tensor(output / "behavior_policy.pt", saved["policy"])
    if start >= stop:
        raise ValueError("Requested collection prefix is already closed; increase max_groups or finish with resume")
    if (isinstance(disk_reserve_gib, bool) or not isinstance(disk_reserve_gib, (int, float)) or
            not math.isfinite(disk_reserve_gib) or disk_reserve_gib < 0):
        raise ValueError("disk_reserve_gib must be nonnegative")
    bytes_per_group = 150_000_000 if cfg.model.backend == "acestep" else 1_000_000
    needed = max(0, stop-start)*bytes_per_group + disk_reserve_gib*2**30
    if shutil.disk_usage(output).free < needed:
        raise RuntimeError(f"Insufficient local disk for this collection slice: need approximately {needed} bytes")
    if backend is None and cfg.model.backend == "acestep" and staged:
        from .staged_backend import StagedAceStepBackend
        backend = StagedAceStepBackend(cfg.model)
    backend = backend or create_backend(cfg)
    setup_policy(cfg, backend)
    load_adapter_state(backend.decoder, saved["policy"])
    backend.decoder.requires_grad_(False)
    if model_signature(backend.provenance()) != model_signature(saved["boundary"]["backend"]):
        raise ValueError("Local base model/VAE/text weights differ from the source checkpoint")
    # On resume this receipt must also be unchanged; do not silently mix kernels.
    runtime = {"torch_version": str(torch.__version__), "backend": backend.provenance(),
               "implementation_sha256": {n: file_sha256(Path(__file__).with_name(n)) for n in
                   ("portable_collect.py", "staged_backend.py", "offline_protocol.py", "offline_trajectory.py",
                    "flow.py", "policy.py", "trainer.py", "acestep_backend.py", "backends.py")}}
    if backend.device.type == "cuda":
        runtime.update(cuda_version=torch.version.cuda,
                       gpu_name=torch.cuda.get_device_name(backend.device),
                       gpu_capability=list(torch.cuda.get_device_capability(backend.device)))
    receipt = output / "COLLECTOR_RUNTIME.json"
    if receipt.exists() and json.loads(receipt.read_text()) != runtime:
        raise ValueError("Collector runtime changed during resume")
    write_json(receipt, runtime)
    started = time.monotonic()
    completed = start
    try:
        for index in range(start, stop):
            if shutil.disk_usage(output).free < disk_reserve_gib*2**30 + bytes_per_group:
                raise RuntimeError("Collection reached its disk reserve; preserve the closed prefix")
            row = selected[index]
            with torch.no_grad():
                condition = backend.condition(row)
            folder = output / "groups" / f"g{index:06d}"
            folder.mkdir(exist_ok=False)
            save_tensor(folder / "condition.pt", cpu_condition(condition))
            samples, checks = [], []
            for sample in range(cfg.training.group_size):
                seed = derive_seed(cfg.training.seed, row, index, sample)
                base, _ = rollout(backend, condition, cfg, seed, reference=True, collect=False)
                candidate, trajectory = full_rollout(backend, condition, cfg, seed)
                if index < 2:
                    checks.append(verify_replay(backend, condition, cfg, trajectory))
                save_tensor(folder / f"s{sample:02d}_trajectory.pt", trajectory)
                _save_audio(folder / f"s{sample:02d}_candidate.wav", candidate)
                _save_audio(folder / f"s{sample:02d}_base.wav", base)
                samples.append({"sample": sample, "seed": seed, "reward": None, "valid": None,
                                "diagnostics": {"status": "pending_cloud_scoring"}})
                del trajectory, candidate, base
            write_json(folder / "group.json", {"group": index, "prompt_id": row["prompt_id"],
                "results": samples, "replay_checks": checks, "files": file_manifest(folder),
                "closed": True, "scored": False})
            completed = index + 1
            state = {"phase": "collecting", "groups_completed": completed,
                     "target_groups": len(selected), "candidate_count": completed*cfg.training.group_size,
                     "optimizer_updates": 0, "elapsed_seconds": time.monotonic()-started, **_memory(backend.device)}
            write_json(output / "collection_state.json", state)
            print(json.dumps(state), flush=True)
        state["phase"] = "complete_unscored" if completed == len(selected) else "prefix_complete"
        write_json(output / "collection_state.json", state)
        if completed == len(selected):
            write_json(output / "SUMMARY.json", {**state, "music_quality_validated": False})
            entries = [e for e in file_manifest(output) if not e["path"].startswith("incomplete/")]
            write_json(output / "FILES_SHA256.json", {"files": entries})
        return state
    except Exception as error:
        write_json(output / "collection_state.json", {"phase": "failed", "groups_completed": completed,
                   "optimizer_updates": 0, "error": str(error)})
        raise
