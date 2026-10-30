import math

import pytest
torch = pytest.importorskip("torch", reason="Install the optional rl extra")

from music_detector.rl.flow import (
    clipped_policy_loss,
    gaussian_kl,
    gaussian_log_prob,
    group_advantages,
    sample_transition,
    transition_mean,
)


def test_gaussian_log_prob_matches_normal_density_and_reductions():
    x = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=torch.float64)
    mean = torch.zeros_like(x)
    variance = torch.tensor(2.0, dtype=torch.float64)
    expected_elementwise = -0.5 * (x.square() / variance + math.log(2 * math.pi) + math.log(2.0))
    torch.testing.assert_close(gaussian_log_prob(x, mean, variance, "sum"), expected_elementwise.sum(dim=1))
    torch.testing.assert_close(gaussian_log_prob(x, mean, variance, "mean"), expected_elementwise.mean(dim=1))


def test_transition_mean_reverse_sde_formula_and_endpoints():
    x = torch.tensor([1.0, -2.0], dtype=torch.float64)
    velocity = torch.tensor([0.5, 0.25], dtype=torch.float64)
    t = torch.tensor(0.4, dtype=torch.float64)
    dt = torch.tensor(-0.1, dtype=torch.float64)
    eta = torch.tensor(0.7, dtype=torch.float64)
    sigma_sq = eta.square() * t / (1 - t)
    expected = x * (1 + sigma_sq / (2 * t) * dt) + velocity * (1 + sigma_sq * (1 - t) / (2 * t)) * dt
    torch.testing.assert_close(transition_mean(x, velocity, t, dt, eta), expected)
    torch.testing.assert_close(transition_mean(x, velocity, 0.0, dt, eta), x + velocity * dt)
    torch.testing.assert_close(transition_mean(x, velocity, 1.0, dt, eta), x + velocity * dt)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t": -0.01, "dt": -0.1},
        {"t": 1.01, "dt": -0.1},
        {"t": 0.5, "dt": 0.1},
        {"t": 0.5, "dt": 0.0},
    ],
)
def test_transition_rejects_invalid_time_domain(kwargs):
    with pytest.raises(ValueError):
        transition_mean(torch.ones(2), torch.ones(2), kwargs["t"], kwargs["dt"], 0.7)


def test_sample_transition_seed_and_deterministic_endpoint():
    x = torch.zeros(2, 3)
    velocity = torch.ones_like(x)
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    a = sample_transition(x, velocity, 0.5, -0.1, 0.7, generator=generator_a)
    b = sample_transition(x, velocity, 0.5, -0.1, 0.7, generator=generator_b)
    torch.testing.assert_close(a[0], b[0])
    assert a[2] is not None and a[3] is not None
    endpoint = sample_transition(x, velocity, 0.0, -0.1, 0.7)
    torch.testing.assert_close(endpoint[0], x - 0.1 * velocity)
    assert endpoint[2] is None and endpoint[3] is None


def test_zero_variance_is_not_clamped_into_gaussian():
    with pytest.raises(ValueError, match="strictly positive"):
        gaussian_log_prob(torch.zeros(2), torch.zeros(2), torch.zeros(2))


def test_group_advantages_and_clip():
    advantages = group_advantages(torch.tensor([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]]))
    torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.0))
    torch.testing.assert_close(advantages[1], torch.zeros(3))
    assert bool((group_advantages(torch.tensor([[0.0, 100.0]]), clip=0.5).abs() <= 0.5).all())


def test_clipped_loss_has_no_gradient_for_zero_advantages():
    new_logprob = torch.tensor([0.2, -0.1], requires_grad=True)
    loss = clipped_policy_loss(new_logprob, torch.zeros(2), torch.zeros(2), 0.2)
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(new_logprob.grad, torch.zeros_like(new_logprob))


def test_gaussian_kl_equal_variance():
    mean = torch.tensor([[1.0, 2.0]])
    reference = torch.zeros_like(mean)
    variance = torch.tensor(2.0)
    expected = mean.square() / (2 * variance)
    torch.testing.assert_close(gaussian_kl(mean, reference, variance, "sum"), expected.sum(dim=1))
