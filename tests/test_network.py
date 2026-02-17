"""Tests for the BridgeModel forward pass."""

import torch
from rlbridge.engine.deal import Deal
from rlbridge.engine.game_state import GameState
from rlbridge.model.config import ModelConfig
from rlbridge.model.network import BridgeModel
from rlbridge.model.encoder import encode_observation, collate_observations


def _make_batch(model_config, n=4):
    """Create a small batch of encoded bidding observations."""
    import numpy as np
    rng = np.random.RandomState(42)
    encoded_list = []
    for _ in range(n):
        deal = Deal.random(rng)
        state = GameState.initial(deal)
        obs = state.observation(state.current_player)
        encoded_list.append(encode_observation(obs, model_config))
    return collate_observations(encoded_list, model_config)


def test_output_shapes(model_config):
    """Model should output correct shapes."""
    model = BridgeModel(model_config)
    model.eval()
    batch = _make_batch(model_config, n=4)

    with torch.no_grad():
        bid_logits, card_logits, value = model(batch)

    assert bid_logits.shape == (4, 38)
    assert card_logits.shape == (4, 52)
    assert value.shape == (4, 1)


def test_get_action_and_value(model_config):
    """get_action_and_value should return actions, log_probs, values."""
    model = BridgeModel(model_config)
    model.eval()
    batch = _make_batch(model_config, n=4)

    with torch.no_grad():
        result = model.get_action_and_value(batch, temperature=1.0)

    assert result['action'].shape == (4,)
    assert result['log_prob'].shape == (4,)
    assert result['value'].shape == (4,)
    assert result['entropy'].shape == (4,)


def test_masked_logits(model_config):
    """Illegal actions should have -inf logits."""
    model = BridgeModel(model_config)
    model.eval()
    batch = _make_batch(model_config, n=2)

    with torch.no_grad():
        bid_logits, card_logits, _ = model(batch)

    # During bidding, card_logits should be all -inf (no legal cards)
    assert (card_logits == float('-inf')).all()
