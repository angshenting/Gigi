"""Tests for observation encoding."""

import numpy as np
import torch
from rlbridge.engine.deal import Deal
from rlbridge.engine.game_state import GameState
from rlbridge.model.config import ModelConfig
from rlbridge.model.encoder import encode_observation, collate_observations


def test_bidding_observation_shapes(sample_deal):
    """Encoded bidding observation should have correct tensor shapes."""
    config = ModelConfig()
    state = GameState.initial(sample_deal)
    obs = state.observation(state.current_player)
    encoded = encode_observation(obs, config)

    assert encoded['hand'].shape == (52,)
    assert encoded['dummy'].shape == (52,)
    assert encoded['context'].shape == (8,)
    assert encoded['legal_mask_bids'].shape == (38,)
    assert encoded['legal_mask_cards'].shape == (52,)


def test_collate_produces_batched_tensors(sample_deals):
    """Collating multiple observations should produce properly batched tensors."""
    config = ModelConfig()
    encoded_list = []
    for deal in sample_deals:
        state = GameState.initial(deal)
        obs = state.observation(state.current_player)
        encoded_list.append(encode_observation(obs, config))

    batch = collate_observations(encoded_list, config)
    B = len(sample_deals)

    assert batch['hand'].shape == (B, 52)
    assert batch['dummy'].shape == (B, 52)
    assert batch['context'].shape == (B, 8)
    assert batch['legal_mask_bids'].shape == (B, 38)
    assert batch['legal_mask_cards'].shape == (B, 52)
    assert batch['seq_mask'].dtype == torch.bool


def test_legal_mask_bids_populated(sample_deal):
    """During bidding, legal_mask_bids should have at least one True entry."""
    config = ModelConfig()
    state = GameState.initial(sample_deal)
    obs = state.observation(state.current_player)
    encoded = encode_observation(obs, config)

    assert encoded['legal_mask_bids'].any()
    # No cards should be legal during bidding
    assert not encoded['legal_mask_cards'].any()
