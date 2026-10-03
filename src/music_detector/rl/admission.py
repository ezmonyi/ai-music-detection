"""Explicit group-level control flow; never changes a completion's validity.

Uniform rejected rewards contain no group-relative learning signal. The
opt-in bounded policy records such a group without replay/optimizer updates.
Finite budgets prevent an unavailable analyzer or collapsed policy from
silently consuming the entire run. Default behavior remains fail-fast.
"""
from __future__ import annotations


def initial_state() -> dict[str, int]:
    return {"all_invalid_groups": 0, "consecutive_all_invalid_groups": 0}


def validate_state(state: dict) -> None:
    if not isinstance(state, dict) or set(state) != set(initial_state()):
        raise ValueError("Malformed group admission checkpoint state")
    if any(type(v) is not int or v < 0 for v in state.values()):
        raise ValueError("Invalid group admission counters")
    if state["consecutive_all_invalid_groups"] > state["all_invalid_groups"]:
        raise ValueError("Inconsistent group admission counters")


def advance(valid_count: int, config, state: dict) -> dict[str, int]:
    validate_state(state)
    if type(valid_count) is not int or valid_count < 0:
        raise ValueError("valid_count must be a nonnegative integer")
    updated = dict(state)
    if valid_count:
        updated["consecutive_all_invalid_groups"] = 0
        return updated
    if config.all_invalid_policy == "abort":
        raise RuntimeError("All candidates failed reward admission; inspect rollouts.jsonl; no update performed")
    if config.all_invalid_policy != "skip_bounded":
        raise ValueError("Unknown group admission policy")
    updated["all_invalid_groups"] += 1
    updated["consecutive_all_invalid_groups"] += 1
    if (updated["all_invalid_groups"] > config.max_all_invalid_groups or
            updated["consecutive_all_invalid_groups"] > config.max_consecutive_all_invalid_groups):
        raise RuntimeError("All-invalid group safety budget exceeded; no update performed")
    return updated
