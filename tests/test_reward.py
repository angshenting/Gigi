"""Tests for reward computation (mocking DDS)."""

from unittest.mock import patch, MagicMock
from rlbridge.training.reward import compute_reward, assign_rewards
from rlbridge.engine.experience import GameResult, ExperienceStep


def test_compute_reward_zero_diff():
    """Zero difference between score and PAR should give zero IMPs."""
    rns, rew = compute_reward(400, 400)
    assert rns == 0
    assert rew == 0


def test_compute_reward_positive():
    """Score > PAR should give positive reward for NS."""
    rns, rew = compute_reward(620, 400)
    assert rns > 0
    assert rew < 0
    assert rns == -rew


def test_compute_reward_negative():
    """Score < PAR should give negative reward for NS."""
    rns, rew = compute_reward(-100, 400)
    assert rns < 0
    assert rew > 0
    assert rns == -rew


def test_assign_rewards_ns_ew_opposite():
    """NS and EW players should get opposite rewards."""
    # Create minimal GameResult with trajectory
    steps = [
        ExperienceStep(player=0, observation={'phase': 'bidding'}, legal_actions=[0], action=0),
        ExperienceStep(player=1, observation={'phase': 'bidding'}, legal_actions=[0], action=0),
        ExperienceStep(player=2, observation={'phase': 'bidding'}, legal_actions=[0], action=0),
        ExperienceStep(player=3, observation={'phase': 'bidding'}, legal_actions=[0], action=0),
    ]
    result = GameResult(
        deal=None, final_state=None, trajectory=steps,
        score_ns=620, par_ns=400, contract='3NN', auction=()
    )
    returns = assign_rewards(result)
    assert len(returns) == 4
    # NS players (0, 2) should have same reward
    assert returns[0] == returns[2]
    # EW players (1, 3) should have same reward
    assert returns[1] == returns[3]
    # NS and EW opposite
    assert returns[0] == -returns[1]


def test_assign_rewards_no_par():
    """When par_ns is None, should use raw score normalization."""
    steps = [
        ExperienceStep(player=0, observation={'phase': 'bidding'}, legal_actions=[0], action=0),
    ]
    result = GameResult(
        deal=None, final_state=None, trajectory=steps,
        score_ns=200, par_ns=None, contract='1NN', auction=()
    )
    returns = assign_rewards(result)
    assert len(returns) == 1
    assert returns[0] == 200 / 100.0  # normalized
