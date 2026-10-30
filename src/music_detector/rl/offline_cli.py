"""Explicit local/cloud entry points; never rent, upload or start a GPU implicitly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _runtime(path):
    return json.loads(Path(path).read_text()) if path else None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="music-offline-rl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "collect", "replay", "score", "merge", "preflight", "train", "evaluate"):
        sub = commands.add_parser(name)
        if name not in {"score"}:
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--checkpoint-sha256", required=True)
        if name in {"plan", "merge", "preflight", "train"}:
            sub.add_argument("--validation-data", required=True)
            sub.add_argument("--test-data", required=True)
        if name != "preflight":
            sub.add_argument("--output", required=True)
        if name in {"collect", "replay", "score", "train", "evaluate"}:
            sub.add_argument("--runtime", help="JSON containing only permitted model/reward runtime relocations")
        if name == "plan":
            sub.add_argument("--train-data", required=True)
            sub.add_argument("--count", type=int, default=1000)
            sub.add_argument("--exclude-prompts", action="append", default=[])
        if name == "collect":
            sub.add_argument("--plan", required=True)
            sub.add_argument("--max-groups", type=int)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--no-staging", action="store_true", help="Dedicated large-GPU collection only")
            sub.add_argument("--disk-reserve-gib", type=float, default=2)
        if name == "score":
            sub.add_argument("--shard", required=True)
            sub.add_argument("--resume", action="store_true")
        if name == "replay":
            sub.add_argument("--shard", required=True)
            sub.add_argument("--max-groups", type=int, default=2)
        if name == "merge":
            sub.add_argument("--shard", required=True, action="append")
            sub.add_argument("--score", action="append", default=[])
            sub.add_argument("--behavior-checkpoint", action="append", default=[])
        if name in {"preflight", "train"}:
            sub.add_argument("--index", required=True)
            sub.add_argument("--options", required=True)
        if name == "train":
            sub.add_argument("--max-updates", type=int, help="Real optimizer-update slice, included in the full target")
            sub.add_argument("--resume", help="A previous offline checkpoint; use a fresh output directory")
        if name == "evaluate":
            sub.add_argument("--trained-checkpoint", required=True)
            sub.add_argument("--trained-checkpoint-sha256", required=True)
            sub.add_argument("--data", required=True)
            sub.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args(argv)
    if args.command == "plan":
        from .portable_collect import plan_collection
        result = plan_collection(args.train_data, args.validation_data, args.test_data,
            args.checkpoint, args.checkpoint_sha256, args.output, count=args.count, exclude_prompts=args.exclude_prompts)
    elif args.command == "collect":
        from .portable_collect import collect_portable
        result = collect_portable(args.plan, args.checkpoint, args.checkpoint_sha256, args.output,
            overrides=_runtime(args.runtime), max_groups=args.max_groups, resume=args.resume,
            staged=not args.no_staging, disk_reserve_gib=args.disk_reserve_gib)
    elif args.command == "replay":
        from .portable_replay import replay_probe
        value = replay_probe(args.shard, args.checkpoint, args.checkpoint_sha256, args.output,
                            overrides=_runtime(args.runtime), maximum_groups=args.max_groups)
        result = {k: v for k, v in value.items() if k != "checks"}
    elif args.command == "score":
        from .offline_cache import score_shard
        value = score_shard(args.shard, args.output, overrides=_runtime(args.runtime), resume=args.resume)
        result = {"scored_groups": len(value["groups"]), "valid_candidates": value["valid_candidates"], "optimizer_updates": 0}
    elif args.command == "merge":
        from .offline_cache import merge_caches
        value = merge_caches(args.shard, args.validation_data, args.test_data, args.checkpoint,
            args.checkpoint_sha256, args.output, score_files=args.score, behavior_checkpoints=args.behavior_checkpoint)
        result = {k: value[k] for k in ("prompt_groups", "candidates", "groups_with_learning_signal", "optimizer_updates")}
    elif args.command == "preflight":
        from .offline_train import load_options, preflight
        result = preflight(args.index, args.checkpoint, args.checkpoint_sha256, args.validation_data,
                           args.test_data, load_options(args.options))[0]
    elif args.command == "train":
        from .offline_train import load_options, train_offline
        result = train_offline(args.index, args.checkpoint, args.checkpoint_sha256, args.validation_data,
            args.test_data, args.output, load_options(args.options), overrides=_runtime(args.runtime),
            resume=args.resume, max_updates=args.max_updates)
    else:
        from .offline_train import evaluate_offline
        result = evaluate_offline(args.checkpoint, args.checkpoint_sha256, args.trained_checkpoint,
            args.trained_checkpoint_sha256, args.data, args.output, overrides=_runtime(args.runtime), split=args.split)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get("status") == "insufficient_usable_data" else 0


if __name__ == "__main__":
    raise SystemExit(main())
