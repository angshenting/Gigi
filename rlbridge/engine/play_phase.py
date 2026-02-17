"""Card play rules — follow suit enforcement and trick winner determination.

Card52 encoding: suit_i = card // 13, rank_i = card % 13
Suits: S=0, H=1, D=2, C=3
Ranks: A=0, K=1, Q=2, ..., 2=12 (lower rank_i = higher rank)

Strain conventions:
  - bidding.bidding: N=0, S=1, H=2, D=3, C=4
  - deck52.get_trick_winner_i: S=0, H=1, D=2, C=3, NT=4
  - Conversion: dds_strain = (bidding_strain - 1) % 5
"""

import numpy as np


def legal_cards(hand_binary: np.ndarray, current_trick: tuple, current_leader: int,
                player: int) -> list:
    """Return list of legal card52 indices the player can play.

    Enforces follow-suit: if cards of the led suit are held, only those may be played.
    If no cards of the led suit, any remaining card is legal.

    Args:
        hand_binary: (52,) float array, 1.0 for cards still held
        current_trick: tuple of card52 values played so far in this trick
        current_leader: player index who led this trick
        player: player index whose turn it is

    Returns:
        List of legal card52 indices
    """
    held = [i for i in range(52) if hand_binary[i] > 0.5]

    if len(current_trick) == 0:
        # Leading: any card is legal
        return held

    # Must follow suit of the lead card
    lead_card = current_trick[0]
    lead_suit = lead_card // 13

    suited = [c for c in held if c // 13 == lead_suit]
    if suited:
        return suited
    # Can't follow suit: play anything
    return held


def trick_winner(trick: tuple, strain_i: int) -> int:
    """Determine which position (0-3 relative to leader) won the trick.

    Args:
        trick: tuple of exactly 4 card52 values
        strain_i: bidding convention strain (N=0, S=1, H=2, D=3, C=4)

    Returns:
        Position 0-3 of the winner (relative to trick leader)
    """
    # Convert bidding strain to DDS/card suit convention
    # bidding: N=0, S=1, H=2, D=3, C=4
    # card52 suit: S=0, H=1, D=2, C=3, and NT means no trump suit
    if strain_i == 0:
        # No trump
        trump_suit = -1  # no trump suit
    else:
        # S=1->0, H=2->1, D=3->2, C=4->3
        trump_suit = strain_i - 1

    trick_suits = [card // 13 for card in trick]
    lead_suit = trick_suits[0]

    # Check if anyone played trump (when there is a trump suit)
    is_trumped = trump_suit >= 0 and any(s == trump_suit for s in trick_suits)

    if is_trumped:
        # Highest trump wins (lowest rank_i = highest rank)
        best_i = -1
        best_rank = 99
        for i in range(4):
            if trick_suits[i] == trump_suit:
                rank = trick[i] % 13
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
        return best_i
    else:
        # Highest card in lead suit wins
        best_i = 0
        best_rank = trick[0] % 13
        for i in range(1, 4):
            if trick_suits[i] == lead_suit:
                rank = trick[i] % 13
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
        return best_i


def is_trick_complete(current_trick: tuple) -> bool:
    """Check if exactly 4 cards have been played to the trick."""
    return len(current_trick) == 4
