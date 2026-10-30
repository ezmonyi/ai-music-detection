"""Explicit plan/analyse/run commands for the independent numerical study."""
import argparse
import json
from pathlib import Path

from .numerical_calibration import build_selection, load_options, summarize, write_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--options", required=True)
    plan.add_argument("--output", required=True)
    discovery = commands.add_parser("analyse-discovery")
    discovery.add_argument("--diagnosis", required=True)
    discovery.add_argument("--output", required=True)
    run = commands.add_parser("run")
    for flag in ("options", "plan", "runtime", "base", "export", "scratch", "output"):
        run.add_argument("--" + flag, required=True)
    run.add_argument("--execute", action="store_true", help="Explicitly run real CUDA measurements; never train")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = build_selection(load_options(args.options))
        write_once(args.output, result)
    elif args.command == "analyse-discovery":
        original = json.loads(Path(args.diagnosis).read_text())
        results = {}
        for check in original["checks"]:
            rows = results.setdefault(check["behavior"], [])
            rows.extend({"key": f"g{check['group']:06d}", "recorded_logprob": v["recorded"],
                         "replayed_logprob": v["replayed"], "repeated_logprob": None} for v in check["steps"])
        result = {"status": "discovery_only_not_independent_calibration", "formal_optimizer_updates": 0,
                  "production_protocol_activation": False,
                  "note": "Repeat errors are unavailable in this discovery file; no repeatability claim is made here.",
                  "behaviors": {key: summarize(rows) for key, rows in results.items()}}
        for value in result["behaviors"].values():
            value.pop("same_gpu_repeat_logprob_error_maximum")
        write_once(args.output, result)
    else:
        if not args.execute:
            parser.error("Real GPU study requires --execute; plan/analyse never load torch")
        from .numerical_calibration_gpu import run_study
        result = run_study(args.options, args.plan, args.runtime, args.base, args.export, args.scratch, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
