"""Real-model measurements for a separate, zero-update numerical study.

This file imports torch only when explicitly run on the provisioned GPU. It
never replaces cached likelihoods or edits any production training options.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

from .numerical_calibration import (
    audit_candidate, build_selection, digest, freeze_candidate, load_options,
    time_indices, write_once,
)


G100_SHA = "c28a4502247cb4736ec1f5b20322381192baef877d3a377188e7ee9a35b2ccd2"
G10_SHA = "909c003d33c9a150c6ac947969ccf139f4f4ae52d9a1d8ab5f8e36b1958047e4"
VALIDATION_SHA = "d6a0839cc2c69f5f8f83ece5e37fa31d7956d0f591773f8645d3296b325589ef"
TEST_SHA = "d9090e509eacca21ff64decc945fe31a197dd97d0edc47011b1e694cc52a1406"


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_study(options_path, plan_path, runtime_path, base, export, scratch, output):
    import torch
    from .offline_archive import hashes
    from .portable_delivery import _read_archive
    from .offline_io import cpu_condition, save_tensor
    from .offline_protocol import (
        checkpoint, experiment_contract, model_signature, policy_fingerprint,
        prompt_keys, reject_overlap, runtime_config, validate_record, verify_members,
    )
    from .offline_trajectory import full_rollout, verify_replay
    from .flow import gaussian_log_prob, transition_mean
    from .policy import load_adapter_state
    from .staged_backend import StagedAceStepBackend
    from .trainer import create_backend, setup_policy, _memory, _save_audio

    options = load_options(options_path)
    selection = json.loads(Path(plan_path).read_text())
    if selection != build_selection(options):
        raise ValueError("Pre-registered selection was changed")
    base, export, scratch, output = map(Path, (base, export, scratch, output))
    if output.exists() or scratch.exists():
        raise FileExistsError("Preserve an earlier study; choose fresh output and scratch directories")
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    started = time.monotonic()

    def state(phase, **details):
        value = {"utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
                 "elapsed_seconds": time.monotonic() - started, "formal_optimizer_updates": 0,
                 "production_protocol_activation": False, **details}
        temp = output / "study_state.writing"
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(output / "study_state.json")
        print(json.dumps(value), flush=True)

    def timeout(_signum, _frame):
        raise TimeoutError("Precommitted numerical-study timebox reached; no optimizer was run")

    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(options["gpu_timebox_seconds"])
    backend = None
    try:
        write_once(output / "OPTIONS.json", options)
        write_once(output / "SELECTION.json", selection)
        shutil.copy2(runtime_path, output / "RUNTIME.json")
        state("verifying_input_identities")
        if not torch.cuda.is_available() or "RTX 3090" not in torch.cuda.get_device_name():
            raise RuntimeError("This precommitted receiver study requires the selected RTX3090")
        if str(torch.__version__) != "2.10.0+cu128" or torch.version.cuda != "12.8":
            raise RuntimeError("Do not silently change the pinned numerical runtime")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Receiver must support native BF16")
        source100, original_cfg = checkpoint(base / "runs/a6000-srf-pilot-main-v2/checkpoints/group_000100.pt", G100_SHA)
        source10, cfg10 = checkpoint(base / "runs/a6000-srf-pilot-main-v2/checkpoints/group_000010.pt", G10_SHA)
        if experiment_contract(cfg10) != experiment_contract(original_cfg):
            raise ValueError("The two behavior versions have different mathematical contracts")
        overrides = json.loads(Path(runtime_path).read_text())
        cfg = runtime_config(original_cfg, overrides)
        policies = {"g100": source100["policy"], "g10": source10["policy"]}
        fingerprints = {key: policy_fingerprint(value) for key, value in policies.items()}
        if fingerprints["g100"] == fingerprints["g10"]:
            raise ValueError("Wrong-policy identity control needs distinct recorded policies")
        validation_path = base / "data/musiccaps_a6000_curated_500_20261003/validation.jsonl"
        test_path = base / "data/musiccaps_a6000_curated_500_20261003/test.jsonl"
        if _file_sha(validation_path) != VALIDATION_SHA or _file_sha(test_path) != TEST_SHA:
            raise ValueError("Heldout identities changed")
        heldout = [json.loads(line) for p in (validation_path, test_path) for line in p.read_text().splitlines() if line.strip()]
        index = json.loads((export / "EXPORT_INDEX.json").read_text())
        if not index["complete"] or index["exported_groups"] != 1000 or len(index["archives"]) != 42:
            raise ValueError("Wrong source export")
        restored = scratch / "portable"
        restored.mkdir()
        archives = ["metadata.tar"] + [f"groups_{block*25:06d}_{block*25+24:06d}.tar" for block in selection["portable_blocks"]]
        verified_archives = []
        for name in archives:
            state("verifying_selected_archive", archive=name)
            receipt = index["archives"][name]
            if hashes(export / name) != receipt["archive"]:
                raise ValueError("Source archive mismatch: " + name)
            _read_archive(export / name, receipt["files"], output=restored)
            verified_archives.append({"name": name, **receipt["archive"]})
        roots = {"portable": restored,
                 "legacy24": base / "data/acestep_grpo_offline_g10_24x4_20261003",
                 "legacy26": base / "data/acestep_grpo_offline_g10_extension_26x4_20261003"}
        rows_by_shard = {}
        for name, root in roots.items():
            rows_by_shard[name] = json.loads((root / "PROMPTS.json").read_text())
            behavior = "g100" if name == "portable" else "g10"
            actual = torch.load(root / "behavior_policy.pt", map_location="cpu", weights_only=True)
            if policy_fingerprint(actual) != fingerprints[behavior]:
                raise ValueError("Shard is bound to the wrong behavior policy")
        records = {}
        selected_rows = []
        for group in selection["groups"]:
            row = rows_by_shard[group["shard"]][group["group"]]
            if row.get("split") != "train":
                raise ValueError("Calibration must use training prompts only")
            reject_overlap([row], heldout, description="heldout validation/test")
            folder = roots[group["shard"]] / "groups" / f"g{group['group']:06d}"
            metadata = json.loads((folder / "group.json").read_text())
            if not metadata["closed"] or metadata["prompt_id"] != row["prompt_id"] or metadata["group"] != group["group"]:
                raise ValueError("Group identity changed")
            verify_members(folder, metadata["files"])
            records[group["key"]] = {"folder": folder, "prompt": row, "metadata": metadata}
            selected_rows.append({**group, "prompt_id": row["prompt_id"], "prompt_identity_sha256": digest(prompt_keys(row)),
                                  "group_metadata_sha256": _file_sha(folder / "group.json")})
        for split in ("calibration", "audit"):
            chosen = [records[g["key"]]["prompt"] for g in selection["groups"] if g["split"] == split]
            other = [records[g["key"]]["prompt"] for g in selection["groups"] if g["split"] != split]
            reject_overlap(chosen, other, description="opposite numerical-study split")
        write_once(output / "INPUT_VERIFICATION.json", {
            "source_export_index_sha256": _file_sha(export / "EXPORT_INDEX.json"),
            "verified_archives": verified_archives, "selected_groups": selected_rows,
            "source_checkpoint_sha256": {"g100": G100_SHA, "g10": G10_SHA},
            "policy_fingerprints": fingerprints, "validation_sha256": VALIDATION_SHA, "test_sha256": TEST_SHA,
            "selected_group_members_verified": True, "full_132GB_cache_verification_complete": False})

        def flags():
            return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic}

        def checked_backend(staged):
            instance = StagedAceStepBackend(cfg.model) if staged else create_backend(cfg)
            setup_policy(cfg, instance)
            instance.decoder.requires_grad_(False)
            if model_signature(instance.provenance()) != model_signature(source100["boundary"]["backend"]):
                raise ValueError("Model/VAE/text tree identity changed")
            return instance

        state("native_staged_collect_to_staged_replay")
        backend = checked_backend(True)
        native = []
        native_root = output / "native_controls"
        native_root.mkdir()
        for behavior in ("g100", "g10"):
            item = next(g for g in selection["groups"] if g["split"] == "calibration" and g["behavior"] == behavior)
            condition = torch.load(records[item["key"]]["folder"] / "condition.pt", map_location="cpu", weights_only=True)
            load_adapter_state(backend.decoder, policies[behavior])
            with torch.no_grad():
                audio, record = full_rollout(backend, condition, cfg, int(digest([options["seed"], behavior, "native"] )[:15], 16))
            save_tensor(native_root / f"{behavior}_trajectory.pt", record)
            save_tensor(native_root / f"{behavior}_condition.pt", cpu_condition(condition))
            _save_audio(native_root / f"{behavior}_candidate.wav", audio)
            entry = {"behavior": behavior, "source_train_group": item["key"], "collection_runtime": flags()}
            try:
                entry["same_staged_backend_replay"] = verify_replay(backend, condition, cfg, record,
                    maximum_steps=cfg.sampling.steps, tolerance=options["original_replay_tolerance"])
                entry["same_staged_backend_passed"] = True
            except RuntimeError as error:
                entry.update(same_staged_backend_passed=False, same_staged_backend_error=str(error))
            native.append(entry)
            del condition, audio, record
        del backend
        backend = None
        gc.collect()
        torch.cuda.empty_cache()

        state("native_staged_collect_to_original_backend_replay")
        backend = checked_backend(False)
        for entry in native:
            behavior = entry["behavior"]
            load_adapter_state(backend.decoder, policies[behavior])
            condition = torch.load(native_root / f"{behavior}_condition.pt", map_location="cpu", weights_only=True)
            record = torch.load(native_root / f"{behavior}_trajectory.pt", map_location="cpu", weights_only=True)
            try:
                entry["original_backend_replay"] = verify_replay(backend, condition, cfg, record,
                    maximum_steps=cfg.sampling.steps, tolerance=options["original_replay_tolerance"])
                entry["original_backend_passed"] = True
            except RuntimeError as error:
                entry.update(original_backend_passed=False, original_backend_error=str(error))
            entry["replay_runtime"] = flags()
        native_passed = all(e["same_staged_backend_passed"] and e["original_backend_passed"] for e in native)
        write_once(output / "NATIVE_CONTROLS.json", {"controls": native, "passed_original_tolerance": native_passed,
                    "optimizer_updates": 0, "native_trajectories": len(native), "not_training_groups": True})
        if not native_passed:
            result = {"status": "native_same_gpu_control_failed_investigate_implementation_before_cross_gpu_tolerance",
                      "candidate_absolute_logprob_budget": None, "production_protocol_activation": False,
                      "formal_optimizer_updates": 0, "audit_performed": False,
                      "native_controls": native, **_memory(backend.device)}
            write_once(output / "SUMMARY.json", result)
            state("study_complete_native_control_failed", **result)
            return result

        state("negative_control_checks")
        negatives = []
        for behavior in ("g100", "g10"):
            item = next(g for g in selection["groups"] if g["split"] == "calibration" and g["behavior"] == behavior)
            record = torch.load(records[item["key"]]["folder"] / "s00_trajectory.pt", map_location="cpu", weights_only=True)
            validate_record(record, cfg)
            altered = dict(record)
            altered["variance"] = record["variance"].clone() * 1.01
            rejected = False
            try:
                validate_record(altered, cfg)
            except ValueError:
                rejected = True
            wrong = "g10" if behavior == "g100" else "g100"
            negatives.append({"behavior": behavior, "wrong_behavior_fingerprint_rejected": fingerprints[wrong] != fingerprints[behavior],
                              "variance_corruption_rejected": rejected, "input_files_modified": False})
        write_once(output / "STRUCTURAL_NEGATIVE_CONTROLS.json", negatives)
        controls_passed = native_passed and all(e["wrong_behavior_fingerprint_rejected"] and e["variance_corruption_rejected"] for e in negatives)

        @torch.no_grad()
        def measure(split):
            measured = []
            path = output / f"{split}_measurements.jsonl"
            with path.open("x") as stream:
                for behavior in ("g100", "g10"):
                    load_adapter_state(backend.decoder, policies[behavior])
                    for item in selection["groups"]:
                        if item["split"] != split or item["behavior"] != behavior:
                            continue
                        folder = records[item["key"]]["folder"]
                        condition = torch.load(folder / "condition.pt", map_location="cpu", weights_only=True)
                        for sample in range(options["candidates_per_group"]):
                            record = torch.load(folder / f"s{sample:02d}_trajectory.pt", map_location="cpu", weights_only=True)
                            validate_record(record, cfg)
                            ids = time_indices(torch.where(record["likelihood_mask"])[0].tolist(), options["timesteps_per_candidate"])
                            for index in ids:
                                x = record["states"][index].to(backend.device)
                                action = record["states"][index + 1].to(backend.device)
                                variance = record["variance"][index].to(backend.device)
                                t, dt = float(record["timesteps"][index]), float(record["dt"][index])
                                velocity = backend.velocity(x, t, condition).float()
                                mean = transition_mean(x, velocity, t, dt, cfg.sampling.noise_level)
                                first = float(gaussian_log_prob(action, mean, variance, cfg.sampling.logprob_reduction))
                                repeated_velocity = backend.velocity(x, t, condition).float()
                                repeated_mean = transition_mean(x, repeated_velocity, t, dt, cfg.sampling.noise_level)
                                repeated = float(gaussian_log_prob(action, repeated_mean, variance, cfg.sampling.logprob_reduction))
                                row = {"key": item["key"], "behavior": behavior, "split": split,
                                       "sample": sample, "step": index, "time": t,
                                       "recorded_logprob": float(record["old_logprobs"][index]),
                                       "replayed_logprob": first, "repeated_logprob": repeated,
                                       "repeat_velocity_max_abs_error": float((velocity - repeated_velocity).abs().max())}
                                measured.append(row)
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                        state(f"measuring_{split}", group=item["key"], transitions=len(measured), **_memory(backend.device))
            return measured

        cal = measure("calibration")
        frozen = freeze_candidate(cal, selection, options, controls_passed=controls_passed)
        # Written before the first audit forward; the audit cannot select a new budget.
        write_once(output / "CANDIDATE_FROZEN.json", frozen)
        state("calibration_budget_frozen_before_audit", proposed_budget=frozen["candidate_absolute_logprob_budget"])
        audit = measure("audit")
        result = audit_candidate(audit, frozen, selection, options, controls_passed=controls_passed)
        budget = frozen["candidate_absolute_logprob_budget"]
        corruption_errors = [abs(r["replayed_logprob"] - (r["recorded_logprob"] + options["negative_logprob_shift"])) for r in cal[::32]]
        corrupt_rejected = all(e > (budget if budget is not None else max(options["candidate_absolute_logprob_budgets"])) for e in corruption_errors)
        result.update(corrupted_old_logprob_controls=len(corruption_errors), all_corrupted_old_logprobs_rejected=corrupt_rejected)
        if not corrupt_rejected:
            result.update(status="no_audited_budget_recommended", audit_passed=False)
        result.update(calibration=frozen["calibration"], native_controls_passed=native_passed,
                      structural_negative_controls_passed=controls_passed, **_memory(backend.device),
                      elapsed_seconds=time.monotonic() - started, numerical_flags_changed=False,
                      original_cached_probabilities_modified=False, full_132GB_cache_verification_complete=False)
        write_once(output / "SUMMARY.json", result)
        state("study_complete_requires_human_protocol_review", **result)
        return result
    except Exception as error:
        write_once(output / "FAILURE.json", {"error": str(error), "traceback": traceback.format_exc(),
                    "formal_optimizer_updates": 0, "production_protocol_activation": False})
        state("study_failed_preserved", error=str(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if backend is not None:
            del backend
        gc.collect()
        torch.cuda.empty_cache()
