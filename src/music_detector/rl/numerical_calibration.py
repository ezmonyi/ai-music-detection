"""Pre-registered numerical study; this module never activates a training tolerance.

No torch dependency is needed to plan or analyse. The old production replay gate
and cached behavior denominators are deliberately outside this module's writes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_options(path):
    value = json.loads(Path(path).read_text())
    if (value.get("schema_version") != 1 or value.get("study") != "cross_gpu_numerical_calibration_v1"
            or value.get("original_replay_tolerance") != 1e-6
            or value.get("formal_optimizer_updates") != 0
            or value.get("production_protocol_activation") is not False
            or value.get("audit_can_refit_budget") is not False):
        raise ValueError("This study cannot activate production or change the original verdict")
    budgets = value["candidate_absolute_logprob_budgets"]
    if not 0 < value["maximum_surrogate_ratio_relative_error"] <= 1e-4:
        raise ValueError("This study version caps numerical surrogate-ratio uncertainty at 0.01 percent")
    cap = math.log1p(value["maximum_surrogate_ratio_relative_error"])
    if (not budgets or sorted(set(budgets)) != budgets or
            any(not math.isfinite(x) or x <= 0 or x > cap + 1e-15 for x in budgets)
            or value["calibration_safety_multiplier"] < 2):
        raise ValueError("Invalid precommitted numerical budgets")
    for field in ("groups_per_split_per_block", "groups_per_split_per_legacy_shard",
                  "candidates_per_group", "timesteps_per_candidate", "gpu_timebox_seconds"):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError("Invalid study count: " + field)
    if value["candidates_per_group"] != 4 or value["timesteps_per_candidate"] < 8:
        raise ValueError("Retain all four candidates and adequate stochastic-time coverage")
    return value


def _rank(values, seed):
    return sorted(values, key=lambda x: (digest([seed, x]), str(x)))


def build_selection(options):
    """Select before observing any new error; block0/pilot groups stay excluded."""
    seed = options["seed"]
    blocks = _rank([i for i in range(options["portable_archive_blocks"])
                    if i not in options["portable_excluded_blocks"]], seed + "/blocks")[:options["portable_blocks_to_sample"]]
    selected = []
    for block in blocks:
        groups = _rank(list(range(block * options["portable_groups_per_archive"],
                                  (block + 1) * options["portable_groups_per_archive"])), seed + f"/portable/{block}")
        n = options["groups_per_split_per_block"]
        if len(groups) < 2 * n:
            raise ValueError("Insufficient independent portable calibration/audit groups")
        for split, indices in (("calibration", groups[:n]), ("audit", groups[n:2*n])):
            selected.extend({"behavior": "g100", "shard": "portable", "group": i,
                             "block": block, "split": split, "key": f"portable/g{i:06d}"} for i in indices)
    for shard, count in options["legacy_shard_groups"].items():
        groups = _rank([i for i in range(count) if i not in options["legacy_excluded_groups"][shard]], seed + "/" + shard)
        n = options["groups_per_split_per_legacy_shard"]
        if len(groups) < 2 * n:
            raise ValueError("Insufficient independent legacy calibration/audit groups")
        for split, indices in (("calibration", groups[:n]), ("audit", groups[n:2*n])):
            selected.extend({"behavior": "g10", "shard": shard, "group": i,
                             "split": split, "key": f"{shard}/g{i:06d}"} for i in indices)
    selected.sort(key=lambda x: (x["split"], x["behavior"], x["key"]))
    if len({x["key"] for x in selected}) != len(selected):
        raise ValueError("Calibration/audit group overlap")
    return {"schema_version": 1, "kind": "pre_registered_numerical_study_selection",
            "options_sha256": digest(options), "portable_blocks": blocks, "groups": selected,
            "production_protocol_activation": False, "formal_optimizer_updates": 0}


def time_indices(eligible, count):
    eligible = list(eligible)
    if len(eligible) < count or count < 2:
        raise ValueError("Insufficient stochastic transitions")
    result = [eligible[round(i * (len(eligible) - 1) / (count - 1))] for i in range(count)]
    if len(set(result)) != count:
        raise ValueError("Repeated timestep in the study plan")
    return result


def quantile(values, q):
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError("Quantile needs nonempty values and q in [0,1]")
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(rows, original_tolerance=1e-6):
    if not rows:
        raise ValueError("No measured transitions")
    errors = []
    repeats = []
    group_errors = {}
    for row in rows:
        values = (row["recorded_logprob"], row["replayed_logprob"])
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values):
            raise ValueError("Nonfinite measured density is a failed study, not an omitted outlier")
        error = abs(values[1] - values[0])
        errors.append(error)
        repeat = row.get("repeated_logprob")
        if repeat is not None:
            if not isinstance(repeat, (int, float)) or isinstance(repeat, bool) or not math.isfinite(repeat):
                raise ValueError("Nonfinite repeat measurement")
            repeats.append(abs(repeat - values[1]))
        group_errors[row["key"]] = max(group_errors.get(row["key"], 0), error)
    return {"transitions": len(errors), "groups": len(group_errors),
            "absolute_logprob_error_minimum": min(errors), "absolute_logprob_error_median": statistics.median(errors),
            "absolute_logprob_error_p95": quantile(errors, .95), "absolute_logprob_error_p99": quantile(errors, .99),
            "absolute_logprob_error_maximum": max(errors),
            "maximum_surrogate_ratio_relative_error_bound": math.expm1(max(errors)),
            "same_gpu_repeat_logprob_error_maximum": max(repeats) if len(repeats) == len(errors) else None,
            "passes_original_tolerance": sum(e <= original_tolerance for e in errors),
            "group_maxima": group_errors}


def verify_coverage(rows, selection, options, split):
    expected = {g["key"]: g for g in selection["groups"] if g["split"] == split}
    by_group = {}
    seen = set()
    for row in rows:
        identity = (row["key"], row["sample"], row["step"])
        if (row.get("split") != split or row["key"] not in expected or identity in seen
                or row["behavior"] != expected[row["key"]]["behavior"]
                or type(row["sample"]) is not int or not 0 <= row["sample"] < options["candidates_per_group"]):
            raise ValueError("Duplicate, wrong-split or wrong-behavior measurement")
        seen.add(identity)
        by_group.setdefault(row["key"], {}).setdefault(row["sample"], []).append(row)
    if set(by_group) != set(expected):
        raise ValueError("Missing planned groups cannot become an apparent pass")
    for samples in by_group.values():
        if (set(samples) != set(range(options["candidates_per_group"])) or
                any(len(v) != options["timesteps_per_candidate"] for v in samples.values())):
            raise ValueError("Missing planned candidate/timestep measurements")


def freeze_candidate(calibration_rows, selection, options, *, controls_passed):
    verify_coverage(calibration_rows, selection, options, "calibration")
    stats = summarize(calibration_rows, options["original_replay_tolerance"])
    budget = None
    if (controls_passed and stats["same_gpu_repeat_logprob_error_maximum"] is not None
            and stats["same_gpu_repeat_logprob_error_maximum"] <= options["original_replay_tolerance"]):
        required = options["calibration_safety_multiplier"] * stats["absolute_logprob_error_maximum"]
        budget = next((x for x in options["candidate_absolute_logprob_budgets"] if x >= required), None)
    return {"schema_version": 1, "kind": "frozen_numerical_budget_proposal_not_training_approval",
            "selection_sha256": digest(selection), "options_sha256": digest(options),
            "calibration_rows_sha256": digest(calibration_rows), "controls_passed": bool(controls_passed),
            "calibration": stats, "candidate_absolute_logprob_budget": budget,
            "production_protocol_activation": False, "formal_optimizer_updates": 0}


def audit_candidate(audit_rows, frozen, selection, options, *, controls_passed):
    if (frozen["selection_sha256"] != digest(selection) or frozen["options_sha256"] != digest(options)
            or frozen["production_protocol_activation"] is not False):
        raise ValueError("Frozen proposal identity changed before audit")
    verify_coverage(audit_rows, selection, options, "audit")
    stats = summarize(audit_rows, options["original_replay_tolerance"])
    budget = frozen["candidate_absolute_logprob_budget"]
    passed = bool(controls_passed and frozen["controls_passed"] and budget is not None
                  and stats["absolute_logprob_error_maximum"] <= budget
                  and stats["same_gpu_repeat_logprob_error_maximum"] is not None
                  and stats["same_gpu_repeat_logprob_error_maximum"] <= options["original_replay_tolerance"])
    return {"status": "candidate_survived_independent_audit_requires_human_review" if passed else "no_audited_budget_recommended",
            "candidate_absolute_logprob_budget": budget, "audit_passed": passed, "audit": stats,
            "audit_refitted_budget": False, "production_protocol_activation": False,
            "formal_optimizer_updates": 0,
            "limitation": "Sampled train groups and transitions only; no guarantee for every transition in the full cache."}


def write_once(path, value):
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(serialized)
