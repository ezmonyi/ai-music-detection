from dataclasses import replace

import pytest

from music_detector.rl.admission import advance, initial_state, validate_state
from music_detector.rl.config import TrainingConfig, config_from_dict


def test_default_policy_still_aborts():
    with pytest.raises(RuntimeError, match="All candidates failed"):
        advance(0, TrainingConfig(), initial_state())


def test_bounded_skip_is_auditable_and_resets_only_consecutive_counter():
    cfg = replace(TrainingConfig(), all_invalid_policy="skip_bounded")
    before = initial_state()
    rejected = advance(0, cfg, before)
    assert before == initial_state()
    assert rejected == {"all_invalid_groups": 1, "consecutive_all_invalid_groups": 1}
    assert advance(2, cfg, rejected) == {"all_invalid_groups": 1, "consecutive_all_invalid_groups": 0}


def test_both_safety_budgets_are_enforced():
    cfg = replace(TrainingConfig(), all_invalid_policy="skip_bounded",
                  max_all_invalid_groups=2, max_consecutive_all_invalid_groups=1)
    state = advance(0, cfg, initial_state())
    with pytest.raises(RuntimeError, match="safety budget"):
        advance(0, cfg, state)
    state = advance(0, cfg, advance(1, cfg, state))
    with pytest.raises(RuntimeError, match="safety budget"):
        advance(0, cfg, advance(1, cfg, state))


@pytest.mark.parametrize("state", [{}, {"all_invalid_groups": True, "consecutive_all_invalid_groups": 0},
                                  {"all_invalid_groups": 0, "consecutive_all_invalid_groups": 1}])
def test_checkpoint_counters_are_strict(state):
    with pytest.raises(ValueError):
        validate_state(state)


@pytest.mark.parametrize("field,value", [("all_invalid_policy", "ignore"),
                                          ("max_all_invalid_groups", True),
                                          ("max_consecutive_all_invalid_groups", 0)])
def test_invalid_policy_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        config_from_dict({"training": {field: value}})
