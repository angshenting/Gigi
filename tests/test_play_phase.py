"""Tests for card play rules."""

import numpy as np
from rlbridge.engine.play_phase import legal_cards, trick_winner


def test_follow_suit_enforced():
    """If player has cards of the led suit, they must play one of those."""
    # Hand has spades (0-12) and a heart (13)
    hand = np.zeros(52, dtype=np.float32)
    hand[0] = 1.0   # SA
    hand[1] = 1.0   # SK
    hand[13] = 1.0  # HA

    # Spade led (card 5 = S8)
    current_trick = (5,)
    legal = legal_cards(hand, current_trick, current_leader=0, player=1)
    # Must follow spades: only SA (0) and SK (1)
    assert set(legal) == {0, 1}


def test_no_suit_play_any():
    """If player has no cards of the led suit, any card is legal."""
    # Hand has only hearts (13-25)
    hand = np.zeros(52, dtype=np.float32)
    hand[13] = 1.0  # HA
    hand[14] = 1.0  # HK
    hand[15] = 1.0  # HQ

    # Spade led (card 0 = SA)
    current_trick = (0,)
    legal = legal_cards(hand, current_trick, current_leader=0, player=1)
    # Can play any card
    assert set(legal) == {13, 14, 15}


def test_leading_all_legal():
    """When leading (empty trick), any card is legal."""
    hand = np.zeros(52, dtype=np.float32)
    hand[0] = 1.0
    hand[13] = 1.0
    hand[26] = 1.0

    current_trick = ()
    legal = legal_cards(hand, current_trick, current_leader=0, player=0)
    assert set(legal) == {0, 13, 26}


def test_trick_winner_nt_highest_in_lead_suit():
    """In NT (strain_i=0), highest card in lead suit wins."""
    # All spades: SA(0), SK(1), SQ(2), SJ(3) — lower rank_i = higher rank
    trick = (1, 3, 0, 2)  # SK, SJ, SA, SQ
    winner = trick_winner(trick, strain_i=0)  # NT
    # SA (card 0, rank_i=0) is highest, played by player at position 2
    assert winner == 2


def test_trick_winner_trump_beats_lead_suit():
    """Trump suit card should beat the lead suit."""
    # Lead: SA (0, spade), then HA (13, heart), then DA (26, diamond), then CA (39, club)
    trick = (0, 13, 26, 39)
    # strain_i=1 means spades is trump → spade already led, so highest spade wins
    # strain_i=2 means hearts is trump
    winner = trick_winner(trick, strain_i=2)  # hearts trump
    # HA (card 13) is the only heart/trump, at position 1
    assert winner == 1


def test_trick_winner_highest_trump_wins():
    """When multiple trump cards played, highest trump wins."""
    # All hearts: HA(13), HK(14), HQ(15), HJ(16)
    trick = (14, 13, 16, 15)  # HK, HA, HJ, HQ
    winner = trick_winner(trick, strain_i=2)  # hearts trump
    # HA (rank_i=0) at position 1
    assert winner == 1


def test_trick_winner_off_suit_no_trump_loses():
    """Off-suit non-trump cards lose to lead suit."""
    # Lead SA(0), then HA(13), DA(26), CA(39) — no trump
    trick = (0, 13, 26, 39)
    winner = trick_winner(trick, strain_i=0)  # NT
    # SA at position 0 wins (only lead suit card)
    assert winner == 0
