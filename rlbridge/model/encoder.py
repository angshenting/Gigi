"""Observation to tensor encoding for the bridge transformer model.

Converts observation dicts from GameState into padded tensor batches.
"""

import numpy as np
import torch
from typing import Optional

from rlbridge.model.config import ModelConfig
from rlbridge.engine import bidding_phase


def encode_observation(obs: dict, config: ModelConfig = None) -> dict:
    """Encode a single observation dict into tensors.

    Args:
        obs: observation dict from GameState.observation()
        config: ModelConfig (uses defaults if None)

    Returns:
        dict of tensors (unbatched):
          hand: (52,) float
          dummy: (52,) float
          context: (8,) float
          action_tokens: (L,) long — bid IDs or card52 IDs
          action_types: (L,) long — 0=bid, 1=card
          player_ids: (L,) long — 0=me, 1=LHO, 2=partner, 3=RHO
          legal_mask_bids: (38,) bool
          legal_mask_cards: (52,) bool
    """
    if config is None:
        config = ModelConfig()

    player = obs['player']

    # Hand and dummy
    hand = torch.tensor(obs['hand'], dtype=torch.float32)
    dummy = torch.tensor(obs['dummy'], dtype=torch.float32)

    # Context: [vuln_us, vuln_them, dealer_rel_onehot(4), phase_onehot(2)]
    if player in (0, 2):  # NS
        vuln_us = float(obs['vuln_ns'])
        vuln_them = float(obs['vuln_ew'])
    else:  # EW
        vuln_us = float(obs['vuln_ew'])
        vuln_them = float(obs['vuln_ns'])

    dealer_rel = (obs['dealer'] - player) % 4
    dealer_onehot = [0.0] * 4
    dealer_onehot[dealer_rel] = 1.0

    phase_onehot = [0.0, 0.0]
    if obs['phase'] == 'bidding':
        phase_onehot[0] = 1.0
    else:
        phase_onehot[1] = 1.0

    context = torch.tensor(
        [vuln_us, vuln_them] + dealer_onehot + phase_onehot,
        dtype=torch.float32,
    )

    # Build action token sequence from auction + played cards
    action_tokens = []
    action_types = []
    player_ids = []

    # Auction tokens
    auction = obs['auction']
    for i, bid_str in enumerate(auction):
        if bid_str in ('PAD_START', 'PAD_END'):
            continue
        # Convert bid string to BEN ID for embedding
        try:
            from bidding.bidding import BID2ID
            ben_id = BID2ID[bid_str]
        except (ImportError, KeyError):
            ben_id = bidding_phase.bid_to_action(bid_str) + 2
        action_tokens.append(ben_id)
        action_types.append(0)  # bid

        # Relative player: who made this bid
        # The i-th entry in auction is made by player (i % 4)
        bidder = i % 4
        rel = (bidder - player) % 4
        player_ids.append(rel)

    # Card tokens from completed tricks and current trick
    trick_leaders = obs.get('trick_leaders', ())
    for trick_idx, trick in enumerate(obs['tricks']):
        leader = trick_leaders[trick_idx] if trick_idx < len(trick_leaders) else 0
        for card_pos, card in enumerate(trick):
            action_tokens.append(card)
            action_types.append(1)  # card
            card_player = (leader + card_pos) % 4
            rel = (card_player - player) % 4
            player_ids.append(rel)

    for card_pos, card in enumerate(obs['current_trick']):
        action_tokens.append(card)
        action_types.append(1)
        if obs['current_leader'] is not None:
            card_player = (obs['current_leader'] + card_pos) % 4
            rel = (card_player - player) % 4
            player_ids.append(rel)
        else:
            player_ids.append(0)

    # Convert to tensors
    if len(action_tokens) == 0:
        action_tokens_t = torch.zeros(0, dtype=torch.long)
        action_types_t = torch.zeros(0, dtype=torch.long)
        player_ids_t = torch.zeros(0, dtype=torch.long)
    else:
        action_tokens_t = torch.tensor(action_tokens, dtype=torch.long)
        action_types_t = torch.tensor(action_types, dtype=torch.long)
        player_ids_t = torch.tensor(player_ids, dtype=torch.long)

    # Legal action masks
    legal_mask_bids = torch.zeros(config.num_bid_actions, dtype=torch.bool)
    legal_mask_cards = torch.zeros(config.num_card_actions, dtype=torch.bool)

    if obs['phase'] == 'bidding':
        for a in obs['legal_actions']:
            if 0 <= a < config.num_bid_actions:
                legal_mask_bids[a] = True
    else:
        for a in obs['legal_actions']:
            if 0 <= a < config.num_card_actions:
                legal_mask_cards[a] = True

    return {
        'hand': hand,
        'dummy': dummy,
        'context': context,
        'action_tokens': action_tokens_t,
        'action_types': action_types_t,
        'player_ids': player_ids_t,
        'legal_mask_bids': legal_mask_bids,
        'legal_mask_cards': legal_mask_cards,
    }


def collate_observations(obs_list: list, config: ModelConfig = None) -> dict:
    """Collate a list of encoded observations into a padded batch.

    Args:
        obs_list: list of dicts from encode_observation()

    Returns:
        dict of batched tensors with padding
    """
    if config is None:
        config = ModelConfig()

    batch_size = len(obs_list)

    # Stack fixed-size tensors
    hands = torch.stack([o['hand'] for o in obs_list])
    dummies = torch.stack([o['dummy'] for o in obs_list])
    contexts = torch.stack([o['context'] for o in obs_list])
    legal_bids = torch.stack([o['legal_mask_bids'] for o in obs_list])
    legal_cards = torch.stack([o['legal_mask_cards'] for o in obs_list])

    # Pad variable-length sequences
    max_len = max(len(o['action_tokens']) for o in obs_list) if obs_list else 0
    max_len = max(max_len, 1)  # at least length 1

    action_tokens = torch.zeros(batch_size, max_len, dtype=torch.long)
    action_types = torch.zeros(batch_size, max_len, dtype=torch.long)
    player_ids_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    seq_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, o in enumerate(obs_list):
        L = len(o['action_tokens'])
        if L > 0:
            action_tokens[i, :L] = o['action_tokens']
            action_types[i, :L] = o['action_types']
            player_ids_batch[i, :L] = o['player_ids']
            seq_mask[i, :L] = True

    return {
        'hand': hands,
        'dummy': dummies,
        'context': contexts,
        'action_tokens': action_tokens,
        'action_types': action_types,
        'player_ids': player_ids_batch,
        'seq_mask': seq_mask,
        'legal_mask_bids': legal_bids,
        'legal_mask_cards': legal_cards,
    }
