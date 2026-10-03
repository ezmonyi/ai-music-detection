"""Checksum-verified legacy/new cache loading, frozen scoring and merge indexes."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from .config import config_from_dict
from .data import file_sha256
from .flow import group_advantages
from .offline_io import write_json
from .offline_protocol import (analysis_signature, checkpoint, experiment_contract, finite_results,
    member, model_signature, policy_fingerprint, prompt_keys, reject_overlap, runtime_config,
    validate_record, verify_members)
from .trainer import create_reward, read_prompts


def load_shard(root):
    root = Path(root).resolve()
    files = json.loads((root / "FILES_SHA256.json").read_text())["files"]
    names = verify_members(root, files)
    required = {"COLLECTION_MANIFEST.json", "CONFIG.json", "SOURCE_BOUNDARY.json", "PROMPTS.json", "behavior_policy.pt"}
    if not required <= names:
        raise ValueError("Cache checksum manifest does not cover its identity files")
    manifest = json.loads((root / "COLLECTION_MANIFEST.json").read_text())
    cfg = config_from_dict(json.loads((root / "CONFIG.json").read_text()))
    if manifest.get("config_sha256") != cfg.digest():
        raise ValueError("Cache configuration digest mismatch")
    if manifest.get("optimizer_updates") != 0 or manifest.get("trajectory_dtype") != "float32":
        raise ValueError("Expected a frozen full-FP32 behavior cache")
    if manifest["group_size"] != cfg.training.group_size or manifest["solver_steps"] != cfg.sampling.steps:
        raise ValueError("Cache shape/configuration mismatch")
    source = json.loads((root / "SOURCE_BOUNDARY.json").read_text())
    policy = torch.load(root / "behavior_policy.pt", map_location="cpu", weights_only=True)
    fingerprint = policy_fingerprint(policy)
    if "behavior_policy_fingerprint" in manifest and fingerprint != manifest["behavior_policy_fingerprint"]:
        raise ValueError("Behavior policy fingerprint mismatch")
    prompts = json.loads((root / "PROMPTS.json").read_text())
    folders = sorted((root / "groups").glob("g*"))
    if len(prompts) != manifest["prompt_groups"] or len(folders) != len(prompts):
        raise ValueError("Incomplete cache or prompt count mismatch")
    groups = []
    for index, (folder, row) in enumerate(zip(folders, prompts)):
        relative = str(folder.relative_to(root))
        if folder.name != f"g{index:06d}" or relative + "/group.json" not in names:
            raise ValueError("Incomplete or unordered cache group")
        group = json.loads((folder / "group.json").read_text())
        if not group["closed"] or group["prompt_id"] != row["prompt_id"] or row["split"] != "train":
            raise ValueError("Closed group/prompt identity mismatch")
        group_names = verify_members(folder, group["files"])
        expected = {"condition.pt"} | {f"s{s:02d}_{suffix}" for s in range(cfg.training.group_size)
                    for suffix in ("trajectory.pt", "candidate.wav", "base.wav")}
        if group_names != expected or any(relative + "/" + name not in names for name in expected):
            raise ValueError("Group manifest must cover all trajectories, conditions and both WAVs")
        if [r["sample"] for r in group["results"]] != list(range(cfg.training.group_size)):
            raise ValueError("Candidate group is incomplete or reordered")
        for sample in range(cfg.training.group_size):
            record = torch.load(folder / f"s{sample:02d}_trajectory.pt", map_location="cpu", weights_only=True)
            validate_record(record, cfg)
            if record["seed"] != group["results"][sample]["seed"]:
                raise ValueError("Trajectory seed differs from its group")
        groups.append({"path": relative, "record": row, "group": group,
                       "group_sha256": file_sha256(folder / "group.json")})
    keys = [prompt_keys(r) for r in prompts]
    if any(len({k[i] for k in keys}) != len(keys) for i in range(3)):
        raise ValueError("Duplicate prompt/caption/source inside cache")
    return {"root": root, "manifest": manifest, "config": cfg, "source": source,
            "policy": policy, "policy_fingerprint": fingerprint, "groups": groups,
            "files_sha256": file_sha256(root / "FILES_SHA256.json")}


def score_shard(root, output, *, overrides=None, resume=False, reward=None):
    """Append scores outside the immutable payload, never edit collection data."""
    shard = load_shard(root)
    cfg = runtime_config(shard["config"], overrides)
    reward = reward or create_reward(cfg)
    receipt = getattr(getattr(reward, "_analyzer", None), "receipt", None)
    expected = analysis_signature(shard["source"].get("analysis_receipt"))
    if analysis_signature(receipt) != expected:
        raise ValueError("Frozen cloud analyzer differs from the source checkpoint receipt")
    output = Path(output)
    header = {"schema_version": 2, "source_files_sha256": shard["files_sha256"],
              "experiment_contract": experiment_contract(cfg), "analysis_signature": expected}
    if output.exists():
        if not resume or (output / "SCORES.json").exists():
            raise FileExistsError("Use a new score directory or resume its incomplete prefix")
        if json.loads((output / "SCORE_BOUNDARY.json").read_text()) != header:
            raise ValueError("Score resume boundary changed")
    else:
        output.mkdir(parents=True)
        (output / "groups").mkdir()
        write_json(output / "SCORE_BOUNDARY.json", header)
    entries = []
    import soundfile as sf
    for group in shard["groups"]:
        destination = output / "groups" / (Path(group["path"]).name + ".json")
        if destination.exists():
            entry = json.loads(destination.read_text())
            if entry["source_group_sha256"] != group["group_sha256"]:
                raise ValueError("Scored group input changed")
        else:
            results = []
            for original in group["group"]["results"]:
                sample = original["sample"]
                folder = shard["root"] / group["path"]
                candidate, sr = sf.read(folder / f"s{sample:02d}_candidate.wav", dtype="float32", always_2d=True)
                base, base_sr = sf.read(folder / f"s{sample:02d}_base.wav", dtype="float32", always_2d=True)
                if sr != base_sr:
                    raise ValueError("Candidate/base sample rates disagree")
                result = reward.score(candidate, sr, base)
                results.append({"sample": sample, "seed": original["seed"], **asdict(result)})
            entry = {"path": group["path"], "source_group_sha256": group["group_sha256"], "results": results}
            write_json(destination, entry)
        finite_results(entry["results"], cfg.training.group_size)
        entries.append(entry)
        write_json(output / "score_state.json", {"phase": "scoring", "groups_completed": len(entries),
                   "target_groups": len(shard["groups"]), "optimizer_updates": 0})
        print(json.dumps({"scored_groups": len(entries), "target_groups": len(shard["groups"])}), flush=True)
    result = {**header, "groups": entries, "optimizer_updates": 0,
              "valid_candidates": sum(r["valid"] for e in entries for r in e["results"])}
    write_json(output / "SCORES.json", result)
    write_json(output / "SCORES_SHA256.json", {"sha256": file_sha256(output / "SCORES.json")})
    write_json(output / "score_state.json", {"phase": "complete", "groups_completed": len(entries),
               "target_groups": len(entries), "optimizer_updates": 0})
    return result


def merge_caches(roots, validation_data, test_data, checkpoint_path, checkpoint_sha256, output,
                 *, score_files=(), behavior_checkpoints=()):
    saved, cfg = checkpoint(checkpoint_path, checkpoint_sha256)
    expected_contract = experiment_contract(cfg)
    signatures = model_signature(saved["boundary"]["backend"])
    expected_analysis = analysis_signature(saved["boundary"].get("analysis_receipt"))
    held = read_prompts(validation_data, "validation") + read_prompts(test_data, "test")
    source_checkpoints = {checkpoint_sha256: saved}
    for path in behavior_checkpoints:
        sha = file_sha256(path)
        source_checkpoints[sha] = checkpoint(path, sha)[0]
    scores = {}
    for path in score_files:
        value = json.loads(Path(path).read_text())
        key = value["source_files_sha256"]
        if key in scores:
            raise ValueError("Multiple score versions supplied for one cache; choose explicitly")
        scores[key] = (Path(path).resolve(), value)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Merged indexes are immutable; use a new path")
    shards, groups, records = [], [], []
    for root in roots:
        shard = load_shard(root)
        if experiment_contract(shard["config"]) != expected_contract or model_signature(shard["source"]["backend"]) != signatures:
            raise ValueError("Cannot merge caches from different weights/samplers/rewards")
        behavior_sha = shard["manifest"]["checkpoint_sha256"]
        behavior = source_checkpoints.get(behavior_sha)
        if behavior is None:
            raise ValueError("Provide the original --behavior-checkpoint for " + behavior_sha)
        if (experiment_contract(config_from_dict(behavior["config"])) != expected_contract or
                model_signature(behavior["boundary"]["backend"]) != signatures or
                policy_fingerprint(behavior["policy"]) != shard["policy_fingerprint"]):
            raise ValueError("Behavior cache does not match its original checkpoint tensors")
        manifest = shard["manifest"]
        for field, path in [("validation_data_sha256", validation_data), ("test_data_sha256", test_data)]:
            if field in manifest and manifest[field] != file_sha256(path):
                raise ValueError("Cache held-out split differs from the supplied split")
        scored = scores.pop(shard["files_sha256"], None)
        scored_groups = {}
        if scored:
            score_path, score = scored
            if score["experiment_contract"] != expected_contract or score["analysis_signature"] != expected_analysis:
                raise ValueError("Score sidecar belongs to a different frozen analyzer")
            scored_groups = {g["path"]: g for g in score["groups"]}
            if len(scored_groups) != len(shard["groups"]):
                raise ValueError("Incomplete score sidecar")
        elif analysis_signature(shard["source"].get("analysis_receipt")) != expected_analysis:
            raise ValueError("Legacy cached scores use a different analyzer")
        shard_number = len(shards)
        shards.append({"root": os.path.relpath(shard["root"], output.parent),
                       "files_sha256": shard["files_sha256"], "behavior_checkpoint_sha256": behavior_sha,
                       "behavior_policy_fingerprint": shard["policy_fingerprint"],
                       "score_file": os.path.relpath(scored[0], output.parent) if scored else None,
                       "score_file_sha256": file_sha256(scored[0]) if scored else None})
        for group in shard["groups"]:
            entry = scored_groups.get(group["path"])
            if scored and (entry is None or entry["source_group_sha256"] != group["group_sha256"]):
                raise ValueError("Score/input group checksum mismatch")
            results = entry["results"] if entry else group["group"]["results"]
            finite_results(results, cfg.training.group_size)
            if [r["seed"] for r in results] != [r["seed"] for r in group["group"]["results"]]:
                raise ValueError("Score candidate seeds differ from the input group")
            reject_overlap([group["record"]], held, description="validation/test")
            reject_overlap([group["record"]], records, description="another merged training group")
            records.append(group["record"])
            advantages = group_advantages(torch.tensor([r["reward"] for r in results]), clip=cfg.training.advantage_clip)
            valid = sum(r["valid"] for r in results)
            groups.append({"shard": shard_number, "path": group["path"], "record": group["record"],
                           "group_sha256": group["group_sha256"], "valid_count": valid,
                           "results": results, "has_learning_signal": bool(valid and advantages.abs().max() > 1e-7)})
    if scores:
        raise ValueError("A score file was supplied without its matching cache")
    if not groups:
        raise ValueError("No training groups to merge")
    index = {"schema_version": 2, "kind": "offline_group_index", "experiment_contract": expected_contract,
             "model_signature": signatures, "analysis_signature": expected_analysis,
             "initial_checkpoint_sha256": checkpoint_sha256,
             "validation_sha256": file_sha256(validation_data), "test_sha256": file_sha256(test_data),
             "shards": shards, "groups": groups, "prompt_groups": len(groups),
             "candidates": len(groups)*cfg.training.group_size,
             "groups_with_learning_signal": sum(g["has_learning_signal"] for g in groups),
             "optimizer_updates": 0, "offline_reuse_is_off_policy": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, index)
    return index


def verify_index(path):
    """Recheck immutable source bytes on the training machine before GPU load."""
    path = Path(path).resolve()
    index = json.loads(path.read_text())
    if index.get("kind") != "offline_group_index" or index.get("schema_version") != 2:
        raise ValueError("Unsupported merged cache index")
    source_groups = {}
    for number, item in enumerate(index["shards"]):
        shard = load_shard(path.parent / item["root"])
        if shard["files_sha256"] != item["files_sha256"] or shard["policy_fingerprint"] != item["behavior_policy_fingerprint"]:
            raise ValueError("Merged cache changed after indexing")
        if (experiment_contract(shard["config"]) != index["experiment_contract"] or
                model_signature(shard["source"]["backend"]) != index["model_signature"]):
            raise ValueError("Indexed experiment contract changed")
        scored = {}
        if item["score_file"]:
            if file_sha256(path.parent / item["score_file"]) != item["score_file_sha256"]:
                raise ValueError("Frozen reward sidecar changed after indexing")
            score = json.loads((path.parent / item["score_file"]).read_text())
            if (score["source_files_sha256"] != shard["files_sha256"] or
                    score["experiment_contract"] != index["experiment_contract"] or
                    score["analysis_signature"] != index["analysis_signature"]):
                raise ValueError("Indexed score contract changed")
            scored = {g["path"]: g for g in score["groups"]}
        for group in shard["groups"]:
            results = scored[group["path"]]["results"] if scored else group["group"]["results"]
            finite_results(results, shard["config"].training.group_size)
            valid = sum(r["valid"] for r in results)
            advantage = group_advantages(torch.tensor([r["reward"] for r in results]), clip=shard["config"].training.advantage_clip)
            source_groups[(number, group["path"])] = {"group_sha256": group["group_sha256"],
                "record": group["record"], "results": results, "valid_count": valid,
                "has_learning_signal": bool(valid and advantage.abs().max() > 1e-7)}
    keys = set()
    for group in index["groups"]:
        key = (group["shard"], group["path"])
        original = source_groups.get(key)
        if key in keys or original is None or any(group[k] != v for k, v in original.items()):
            raise ValueError("Duplicate or changed indexed group")
        keys.add(key)
    if keys != set(source_groups):
        raise ValueError("Merged index dropped source groups")
    if index["prompt_groups"] != len(keys) or index["groups_with_learning_signal"] != sum(g["has_learning_signal"] for g in index["groups"]):
        raise ValueError("Index summary counts differ from verified groups")
    return index
