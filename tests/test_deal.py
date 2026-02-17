"""Tests for deal generation."""

import numpy as np
from rlbridge.engine.deal import Deal


def test_hand_sizes(sample_deal):
    """Each hand should have exactly 13 cards."""
    for i in range(4):
        assert len(sample_deal.hands[i]) == 13


def test_all_cards_unique(sample_deal):
    """All 52 cards should appear exactly once across all hands."""
    all_cards = []
    for i in range(4):
        all_cards.extend(sample_deal.hands[i])
    assert len(all_cards) == 52
    assert len(set(all_cards)) == 52


def test_hands_binary_matches_hands(sample_deal):
    """Binary representation should match the tuple representation."""
    for i in range(4):
        for card in range(52):
            if card in sample_deal.hands[i]:
                assert sample_deal.hands_binary[i, card] == 1.0
            else:
                assert sample_deal.hands_binary[i, card] == 0.0


def test_hands_binary_shape(sample_deal):
    assert sample_deal.hands_binary.shape == (4, 52)


def test_pbn_format(sample_deal):
    """PBN should start with 'N:' and have 4 hands separated by spaces."""
    pbn = sample_deal.hand_pbn()
    assert pbn.startswith('N:')
    hands = pbn[2:].split(' ')
    assert len(hands) == 4
    for hand in hands:
        suits = hand.split('.')
        assert len(suits) == 4


def test_deterministic_with_same_seed():
    """Same RNG seed should produce identical deals."""
    rng1 = np.random.RandomState(123)
    rng2 = np.random.RandomState(123)
    deal1 = Deal.random(rng1)
    deal2 = Deal.random(rng2)
    assert deal1.hands == deal2.hands
    assert deal1.dealer == deal2.dealer
    assert deal1.vuln_ns == deal2.vuln_ns
    assert deal1.vuln_ew == deal2.vuln_ew


def test_dealer_range(sample_deal):
    assert sample_deal.dealer in (0, 1, 2, 3)


def test_vuln_are_bool(sample_deal):
    assert isinstance(sample_deal.vuln_ns, bool)
    assert isinstance(sample_deal.vuln_ew, bool)
