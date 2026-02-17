"""Tests for temperature schedule."""

import math
from rlbridge.training.config import TrainingConfig, compute_temperature


def test_constant_returns_fixed():
    """Constant schedule should return the fixed temperature value."""
    config = TrainingConfig(
        temperature=0.7,
        temperature_schedule='constant',
        num_iterations=100,
    )
    assert compute_temperature(config, 0) == 0.7
    assert compute_temperature(config, 50) == 0.7
    assert compute_temperature(config, 99) == 0.7


def test_linear_interpolates():
    """Linear schedule should interpolate from start to end."""
    config = TrainingConfig(
        temperature_start=1.0,
        temperature_end=0.2,
        temperature_schedule='linear',
        num_iterations=100,
    )
    # At iteration 0
    assert compute_temperature(config, 0) == 1.0
    # At last iteration
    assert abs(compute_temperature(config, 99) - 0.2) < 1e-6
    # Midpoint
    mid = compute_temperature(config, 49)
    assert 0.4 < mid < 0.7


def test_cosine_boundaries():
    """Cosine schedule should start at start and end at end."""
    config = TrainingConfig(
        temperature_start=1.5,
        temperature_end=0.3,
        temperature_schedule='cosine',
        num_iterations=100,
    )
    # At iteration 0: should be temperature_start
    assert abs(compute_temperature(config, 0) - 1.5) < 1e-6
    # At last iteration: should be temperature_end
    assert abs(compute_temperature(config, 99) - 0.3) < 1e-6


def test_cosine_midpoint():
    """Cosine schedule midpoint should be average of start and end."""
    config = TrainingConfig(
        temperature_start=1.0,
        temperature_end=0.0,
        temperature_schedule='cosine',
        num_iterations=101,  # so iteration 50 is exactly midpoint
    )
    mid = compute_temperature(config, 50)
    expected = 0.5  # cos(pi/2) = 0, so 0.5*(1+0) * (1-0) + 0 = 0.5
    assert abs(mid - expected) < 1e-6


def test_exponential_boundaries():
    """Exponential schedule should hit start and end values."""
    config = TrainingConfig(
        temperature_start=2.0,
        temperature_end=0.5,
        temperature_schedule='exponential',
        num_iterations=100,
    )
    assert abs(compute_temperature(config, 0) - 2.0) < 1e-6
    assert abs(compute_temperature(config, 99) - 0.5) < 1e-6


def test_linear_monotonically_decreasing():
    """Linear schedule with start > end should monotonically decrease."""
    config = TrainingConfig(
        temperature_start=2.0,
        temperature_end=0.1,
        temperature_schedule='linear',
        num_iterations=50,
    )
    prev = compute_temperature(config, 0)
    for i in range(1, 50):
        curr = compute_temperature(config, i)
        assert curr <= prev
        prev = curr
