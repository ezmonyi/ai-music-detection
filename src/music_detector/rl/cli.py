"""Offline preparation, CPU smoke tests and explicit GPU online-RL commands."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .config import config_from_dict, load_config


def ablation_plan(config, output):
    """Write configurations, never launch jobs or rent GPUs."""
    from .trainer import write_json
    output = Path(output)
    if output.exists():
        raise FileExistsError("Ablation output must be a new directory")
    output.mkdir(parents=True)
    arms = [("lora", "all_linear", r) for r in (8, 32, 128)]
    arms += [("lora", "attention", 32), ("full", "all_linear", 32)]
    entries = []
    for mode, targets, rank in arms:
        # Starting grids, not tuned rates or promised performance.
        rates = (3e-5, 1e-4, 3e-4) if mode == "lora" else (1e-6, 3e-6, 1e-5)
        for seed in (11, 29, 47):
            for rate in rates:
                value = deepcopy(config.to_dict())
                value["policy"].update(mode=mode, targets=targets, rank=rank, alpha=2*rank)
                value["training"].update(seed=seed, learning_rate=rate)
                checked = config_from_dict(value)
                name = f"{mode}_{targets}_r{rank}_lr{rate:g}_s{seed}.json"
                write_json(output / name, checked.to_dict())
                entries.append({"config": name, "mode": mode, "targets": targets,
                                "rank": rank if mode == "lora" else None,
                                "seed": seed, "learning_rate": rate})
    write_json(output / "plan.json", {"status": "planned_not_run", "runs": entries,
               "fairness": "Same prompts, group seeds, rollout count, steps and group budget. Tune LR on validation only.",
               "additional_view": "Report GPU-hours and throughput separately; not a sample-matched comparison."})
    return {"planned_runs": len(entries), "output": str(output.resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="music-rl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "train", "evaluate", "ablation-plan"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True, type=Path)
        if name != "validate":
            sub.add_argument("--output", required=True, type=Path)
        if name in {"train", "evaluate"}:
            sub.add_argument("--data", required=True, type=Path)
        if name == "train":
            sub.add_argument("--resume", type=Path)
            sub.add_argument("--validation-data", type=Path,
                             help="Separate validation JSONL required for periodic evaluation; never the test split")
            sub.add_argument("--max-updates", type=int,
                             help="Stop after N more prompt groups; retain the full config for exact resume")
        if name == "evaluate":
            sub.add_argument("--checkpoint", required=True, type=Path)
            sub.add_argument("--split", choices=("validation", "test"), default="validation")
            sub.add_argument("--stochastic", action="store_true", help="Use training SDE; default is deployment-like ODE")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        result = {"config_sha256": config.digest(), "status": "schema_valid_only",
                  "note": "This does not load weights, profile memory, or establish GPU feasibility."}
    elif args.command == "ablation-plan":
        result = ablation_plan(config, args.output)
    elif args.command == "train":
        from .trainer import train
        result = train(config, args.data, args.output, resume=args.resume, max_updates=args.max_updates,
                       validation_data=args.validation_data)
    else:
        from .trainer import evaluate
        result = evaluate(config, args.data, args.checkpoint, args.output,
                          split=args.split, stochastic=args.stochastic)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
