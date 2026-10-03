"""RTX 5060 collection session: explicit probe, collect, export, verify and upload.

No training, rental, disk expansion, automatic downloads or proxy activation.
GPU execution targets Linux/WSL2; export/upload also work without PyTorch.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys

from .data import file_sha256
from .offline_archive import json_write
from .portable_delivery import export_batches, inventory, publish, restore_batches, task_lock


def load_session(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    fields = {"schema_version", "checkpoint", "checkpoint_sha256", "plan", "runtime", "output",
              "state_dir", "target_groups", "batch_groups", "disk_reserve_gib", "upload"}
    if set(value) != fields or value["schema_version"] != 1:
        raise ValueError("Session fields must match the reviewed example")
    for name in ("target_groups", "batch_groups"):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError(name + " must be a positive integer")
    reserve = value["disk_reserve_gib"]
    if isinstance(reserve, bool) or not isinstance(reserve, (float, int)) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("disk_reserve_gib must be finite and nonnegative")
    for name in ("checkpoint", "plan", "runtime", "output", "state_dir"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError("Missing session path: " + name)
        candidate = Path(value[name]).expanduser()
        value[name] = str((candidate if candidate.is_absolute() else path.parent/candidate).resolve())
    if set(value["upload"]) != {"remote_base", "verification", "rclone_config"}:
        raise ValueError("Unknown upload fields")
    if value["upload"]["rclone_config"]:
        candidate = Path(value["upload"]["rclone_config"]).expanduser()
        value["upload"]["rclone_config"] = str((candidate if candidate.is_absolute() else path.parent/candidate).resolve())
    output, state_dir = Path(value["output"]), Path(value["state_dir"])
    if output == state_dir or output in state_dir.parents:
        raise ValueError("Collector receipts must be outside the immutable cache")
    return value


def doctor(session, *, cuda_smoke=False):
    """Verify hardware/software prerequisites, not full-model fit or quality."""
    import torch
    from .offline_protocol import checkpoint, runtime_config
    saved, original = checkpoint(session["checkpoint"], session["checkpoint_sha256"])
    cfg = runtime_config(original, json.loads(Path(session["runtime"]).read_text()))
    errors, warnings = [], []
    required = {"torch": "2.10.0+cu128", "transformers": "4.57.6", "diffusers": "0.37.1"}
    versions = {}
    for name, expected in required.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != expected:
            errors.append(f"Collector pin {name}=={expected} required, found {versions[name]}")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        errors.append("Use Linux x86-64 or WSL2 with NVIDIA CUDA; native Windows/macOS collection is not validated")
    if sys.version_info[:2] != (3, 12):
        errors.append("Use the Python 3.12 collector environment")
    if cfg.model.backend != "acestep" or not cfg.model.device.startswith("cuda"):
        errors.append("Actual ACE-Step collection requires explicit CUDA, not a CPU/toy substitute")
    for path in (cfg.model.upstream_dir, cfg.model.checkpoint_dir):
        if not Path(path).is_dir():
            errors.append("Missing local source/weights directory: " + str(path))
    for name in (cfg.model.checkpoint_name, "vae", "Qwen3-Embedding-0.6B"):
        if not (Path(cfg.model.checkpoint_dir) / name).is_dir():
            errors.append("Missing frozen component: " + name)
    device = torch.device(cfg.model.device)
    hardware = {"cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda}
    if device.type != "cuda" or not torch.cuda.is_available():
        errors.append("CUDA unavailable; CPU fallback is forbidden")
    else:
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        hardware.update(name=properties.name, capability=[properties.major, properties.minor],
                        free_bytes=free, total_bytes=total, bf16_supported=torch.cuda.is_bf16_supported())
        if "5060" not in properties.name:
            errors.append("This collector session expects the dedicated RTX 5060; review a new hardware protocol explicitly")
        if not hardware["bf16_supported"]:
            errors.append("BF16 CUDA support is required")
        if cuda_smoke and not errors:
            generator = torch.Generator(device=device).manual_seed(0)
            a = torch.randn((64, 64), device=device, dtype=torch.bfloat16, generator=generator)
            result = a @ a.T
            torch.cuda.synchronize(device)
            hardware["small_bf16_matmul_passed"] = bool(torch.isfinite(result).all())
            if not hardware["small_bf16_matmul_passed"]:
                errors.append("CUDA BF16 smoke test failed")
    plan = json.loads((Path(session["plan"]) / "PLAN.json").read_text())
    if plan["checkpoint_sha256"] != session["checkpoint_sha256"] or plan["prompt_groups"] != session["target_groups"]:
        errors.append("Session/plan checkpoint or prompt budget differs")
    anchor = Path(session["output"])
    while not anchor.exists():
        anchor = anchor.parent
    disk = shutil.disk_usage(anchor)
    if disk.free < session["target_groups"]*150_000_000 + session["disk_reserve_gib"]*2**30:
        warnings.append("Free disk does not cover the entire estimated cache; each collection slice still fails closed on capacity")
    return {"status": "prerequisites_passed_model_fit_not_tested" if not errors else "prerequisites_failed",
            "errors": errors, "warnings": warnings, "hardware": hardware, "versions": versions,
            "source_groups_completed": saved["groups_completed"], "local_disk_free_bytes": disk.free,
            "target_prompt_groups": session["target_groups"], "target_candidates": session["target_groups"]*4,
            "estimated_cache_bytes": session["target_groups"]*150_000_000,
            "gpu_fit_verified": False, "optimizer_updates": 0}


def run_session(session_path, phase, *, execute=False):
    session = load_session(session_path)
    if phase not in {"doctor", "probe", "collect", "upload"}:
        raise ValueError("Unknown collector phase")
    if not execute:
        return {"status": "dry_run_no_network_or_gpu_or_writes", "phase": phase,
                "target_prompt_groups": session["target_groups"], "candidate_count": session["target_groups"]*4,
                "probe_groups": 2, "batch_groups": session["batch_groups"], "output": session["output"],
                "upload_configured": bool(session["upload"]["remote_base"]), "optimizer_updates": 0}
    if any("REPLACE" in session[name] for name in ("checkpoint", "plan", "runtime", "output", "state_dir")):
        raise ValueError("Replace example paths before executing")
    if phase == "upload":
        upload = session["upload"]
        return publish(session["output"], upload["remote_base"], Path(session["state_dir"])/"upload",
                       execute=True, verification=upload["verification"], rclone_config=upload["rclone_config"])
    report = doctor(session, cuda_smoke=True)
    if phase == "doctor":
        return report
    state_dir = Path(session["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    output = Path(session["output"])
    with task_lock(output.parent / (output.name + ".collector.lock")):
        json_write(state_dir / "DOCTOR.json", report)
        if report["errors"]:
            raise RuntimeError("5060 prerequisite check failed; see DOCTOR.json")
        from .portable_collect import collect_portable
        runtime = json.loads(Path(session["runtime"]).read_text())
        admission_path = state_dir / "GPU_ADMISSION.json"
        identity = {"checkpoint_sha256": session["checkpoint_sha256"],
                    "plan_sha256": file_sha256(Path(session["plan"])/"PLAN.json"),
                    "runtime_sha256": file_sha256(session["runtime"]),
                    "hardware_name": report["hardware"]["name"],
                    "versions": report["versions"]}
        if phase == "probe":
            if output.exists():
                raise FileExistsError("Probe uses a fresh cache. An interrupted probe needs explicit CLI resume and review, not overwrite")
            result = collect_portable(session["plan"], session["checkpoint"], session["checkpoint_sha256"],
                output, overrides=runtime, max_groups=min(2, session["target_groups"]),
                disk_reserve_gib=session["disk_reserve_gib"])
            info = inventory(output, require_complete=session["target_groups"] <= 2)
            reserved = result["cuda_peak_reserved_bytes"]
            admitted = reserved < 0.95*report["hardware"]["total_bytes"]
            receipt = {**identity, "status": "passed" if admitted else "insufficient_vram_headroom",
                       "full_native_groups": info["closed_groups"], "candidate_count": info["closed_groups"]*4,
                       "gpu_peak_reserved_bytes": reserved, "gpu_peak_allocated_bytes": result["cuda_peak_allocated_bytes"],
                       "elapsed_seconds": result["elapsed_seconds"], "local_verified_bytes": info["bytes"],
                       "collector_runtime_sha256": file_sha256(output/"COLLECTOR_RUNTIME.json"),
                       "same_device_replay_passed": True, "cross_gpu_replay_verified": False,
                       "optimizer_updates": 0}
            json_write(admission_path, receipt)
            if not admitted:
                raise RuntimeError("Full probe ran but VRAM headroom was below the admission margin; do not scale or change sampler")
            return receipt
        if not admission_path.exists():
            raise RuntimeError("Run the explicit two-group probe before full collection")
        admission = json.loads(admission_path.read_text())
        if admission.get("status") != "passed" or any(admission.get(k) != v for k, v in identity.items()):
            raise ValueError("GPU admission differs from the current hardware/plan/runtime")
        if file_sha256(output/"COLLECTOR_RUNTIME.json") != admission["collector_runtime_sha256"]:
            raise ValueError("Collector implementation/kernel receipt changed after probe")
        if (output/"FILES_SHA256.json").exists():
            info = inventory(output)
            return {"status": "already_complete_verified_no_restart", "groups_completed": info["closed_groups"],
                    "bytes": info["bytes"], "optimizer_updates": 0}
        if session["target_groups"] <= 2:
            raise RuntimeError("Small probe must be finalized before collection is reported complete")
        state = json.loads((output/"collection_state.json").read_text())
        completed = state["groups_completed"]
        while completed < session["target_groups"]:
            result = collect_portable(session["plan"], session["checkpoint"], session["checkpoint_sha256"],
                output, overrides=runtime, max_groups=min(session["target_groups"], completed+session["batch_groups"]),
                resume=True, disk_reserve_gib=session["disk_reserve_gib"])
            completed = result["groups_completed"]
            json_write(state_dir / "SESSION_STATE.json", {"phase": "collecting", **result})
        info = inventory(output)
        result = {"status": "complete_unscored_local_all_members_verified", "groups_completed": completed,
                  "bytes": info["bytes"], "candidate_count": completed*4, "audio_files": completed*8,
                  "upload_complete": False, "optimizer_updates": 0}
        json_write(state_dir / "SESSION_STATE.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="music-5060", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    session = sub.add_parser("session", help="Dry-run by default; GPU/network work requires --execute")
    session.add_argument("--config", required=True)
    session.add_argument("--phase", choices=("doctor", "probe", "collect", "upload"), required=True)
    session.add_argument("--execute", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--dataset", required=True)
    export = sub.add_parser("export", help="Explicit local TAR writes for MatPool web/client upload")
    export.add_argument("--dataset", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--first", type=int, default=0)
    export.add_argument("--last", type=int)
    export.add_argument("--groups-per-shard", type=int, default=25)
    restore = sub.add_parser("restore", help="Verify TARs then restore into a fresh directory")
    restore.add_argument("--source", required=True)
    restore.add_argument("--output", required=True)
    restore.add_argument("--allow-prefix", action="store_true", help="Probe replay only; not a completed training cache")
    upload = sub.add_parser("upload", help="Dry-run by default; remote must already be configured")
    upload.add_argument("--dataset", required=True)
    upload.add_argument("--remote", required=True)
    upload.add_argument("--receipts", required=True)
    upload.add_argument("--rclone-config")
    upload.add_argument("--verification", choices=("readback", "server_sha256"), default="readback")
    upload.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "session":
        result = run_session(args.config, args.phase, execute=args.execute)
    elif args.command == "verify":
        info = inventory(args.dataset)
        result = {k: info[k] for k in ("complete", "closed_groups", "target_groups", "bytes", "optimizer_updates")}
    elif args.command == "export":
        result = export_batches(args.dataset, args.output, first=args.first, last=args.last, groups_per_shard=args.groups_per_shard)
    elif args.command == "restore":
        result = restore_batches(args.source, args.output, allow_prefix=args.allow_prefix)
    else:
        result = publish(args.dataset, args.remote, args.receipts, execute=args.execute,
                         verification=args.verification, rclone_config=args.rclone_config)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get("status") == "prerequisites_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
