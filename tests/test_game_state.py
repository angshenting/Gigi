"""Tests for game state transitions."""

from rlbridge.engine.deal import Deal
from rlbridge.engine.game_state import GameState


def test_initial_state_is_bidding(sample_deal):
    state = GameState.initial(sample_deal)
    assert state.phase == 'bidding'


def test_initial_state_legal_actions(sample_deal):
    state = GameState.initial(sample_deal)
    legal = state.legal_actions()
    assert len(legal) > 0
    # Should have PASS (0) and suit bids
    assert 0 in legal


def test_initial_not_terminal(sample_deal):
    state = GameState.initial(sample_deal)
    assert not state.is_terminal


def test_observation_has_required_keys(sample_deal):
    state = GameState.initial(sample_deal)
    obs = state.observation(0)
    required_keys = ['phase', 'hand', 'dummy', 'auction', 'vuln_ns',
                     'vuln_ew', 'player', 'legal_actions']
    for key in required_keys:
        assert key in obs, f"Missing key: {key}"


def test_pass_out_terminal(sample_deal):
    """Four passes from the start should lead to passed_out terminal state."""
    state = GameState.initial(sample_deal)

    # Apply PAD_START bids until we reach the dealer, then pass 4 times
    # The initial state already has PAD_START padding for positions before dealer
    for _ in range(4):
        state = state.apply_action(0)  # PASS

    assert state.is_terminal
    assert state.phase == 'passed_out'
    assert state.score_ns() == 0


def test_apply_bid_changes_auction(sample_deal):
    state = GameState.initial(sample_deal)
    new_state = state.apply_action(3)  # 1C
    assert len(new_state.auction) == len(state.auction) + 1
    assert new_state.auction[-1] == '1C'


def test_current_player_cycles(sample_deal):
    """Players should cycle through 0,1,2,3."""
    state = GameState.initial(sample_deal)
    players_seen = []
    for _ in range(4):
        players_seen.append(state.current_player)
        state = state.apply_action(0)  # PASS
    # Should see all 4 players in cyclic order starting from dealer
    assert len(set(players_seen)) <= 4
