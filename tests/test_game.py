"""Tests for full game execution with RandomAgent."""

import numpy as np
from rlbridge.engine.deal import Deal
from rlbridge.engine.game import Game
from rlbridge.engine.agents import RandomAgent
from rlbridge.engine.experience import GameResult


def test_random_game_completes(sample_deal):
    """A game with 4 random agents should complete without error."""
    rng = np.random.RandomState(42)
    agents = [RandomAgent(rng) for _ in range(4)]
    game = Game(agents, sample_deal)
    result = game.play()
    assert isinstance(result, GameResult)


def test_trajectory_non_empty(sample_deal):
    """A completed game should have a non-empty trajectory."""
    rng = np.random.RandomState(42)
    agents = [RandomAgent(rng) for _ in range(4)]
    game = Game(agents, sample_deal)
    result = game.play()
    assert len(result.trajectory) > 0


def test_multiple_random_games():
    """Multiple random games should all complete without error."""
    for seed in range(10):
        rng = np.random.RandomState(seed)
        deal = Deal.random(rng)
        agents = [RandomAgent(rng) for _ in range(4)]
        game = Game(agents, deal)
        result = game.play()
        assert isinstance(result, GameResult)
        assert result.final_state.is_terminal


def test_game_result_has_score(sample_deal):
    rng = np.random.RandomState(42)
    agents = [RandomAgent(rng) for _ in range(4)]
    game = Game(agents, sample_deal)
    result = game.play()
    assert isinstance(result.score_ns, int)


def test_game_result_has_deal(sample_deal):
    rng = np.random.RandomState(42)
    agents = [RandomAgent(rng) for _ in range(4)]
    game = Game(agents, sample_deal)
    result = game.play()
    assert result.deal is sample_deal
