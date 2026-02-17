"""Shared fixtures for rlbridge tests."""

import sys
import os

import pytest
import numpy as np

# Ensure project root and src/ are on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def rng():
    """Deterministic numpy RNG."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_deal(rng):
    """A single deterministic deal."""
    from rlbridge.engine.deal import Deal
    return Deal.random(rng)


@pytest.fixture
def sample_deals(rng):
    """A batch of 8 deterministic deals."""
    from rlbridge.engine.deal import Deal
    return [Deal.random(rng) for _ in range(8)]


@pytest.fixture
def model_config():
    """Small model config for fast tests."""
    from rlbridge.model.config import ModelConfig
    return ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, dropout=0.0)


@pytest.fixture
def training_config():
    """Minimal training config for fast tests."""
    from rlbridge.training.config import TrainingConfig
    return TrainingConfig(
        games_per_iteration=4,
        num_iterations=2,
        device='cpu',
    )
