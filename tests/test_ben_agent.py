"""Tests for BenAgent and its conversion utilities."""

import sys
import os
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from rlbridge.engine.ben_agent import (
    hand52_to_hand32,
    hand52_to_pbn,
    card52_to_card32,
    ben_bid_to_action_id,
)


# ---------------------------------------------------------------------------
# Conversion utility tests (no TF / BEN models needed)
# ---------------------------------------------------------------------------

class TestHand52ToHand32:
    def test_single_ace_of_spades(self):
        hand = np.zeros(52)
        hand[0] = 1  # SA (suit=0, rank=0 => A)
        h32 = hand52_to_hand32(hand)
        assert h32[0] == 1  # suit=0, slot=0 => A
        assert h32.sum() == 1

    def test_single_deuce_of_spades(self):
        """S2 = card52 index 12, maps to card32 suit=0 slot=7 (pip)."""
        hand = np.zeros(52)
        hand[12] = 1  # S2
        h32 = hand52_to_hand32(hand)
        assert h32[7] == 1  # pip slot
        assert h32.sum() == 1

    def test_multiple_pips_same_suit(self):
        """S7, S6, S5 all map to card32 slot 7 in suit 0."""
        hand = np.zeros(52)
        hand[6] = 1   # S7 (rank=6 => min(7,6)=6 => card32 index 6, NOT pip)
        hand[7] = 1   # S6 (rank=7 => min(7,7)=7 => pip)
        hand[8] = 1   # S5 (rank=8 => min(7,8)=7 => pip)
        h32 = hand52_to_hand32(hand)
        # S7 maps to slot 6 (rank 6 < 7, no clamping)
        # S6 maps to slot 7 (rank 7, clamped)
        # S5 maps to slot 7 (rank 8, clamped)
        assert h32[6] == 1  # S7
        assert h32[7] == 2  # S6+S5 in pip slot
        assert h32.sum() == 3

    def test_full_13_card_hand(self):
        """A hand with exactly 13 cards should produce 13 total in hand32."""
        hand = np.zeros(52)
        # Give spades: A K Q (indices 0, 1, 2)
        # Hearts: J T 9 (indices 13+3, 13+4, 13+5 = 16, 17, 18)
        # Diamonds: 8 7 6 5 (indices 26+5, 26+6, 26+7, 26+8 = 31, 32, 33, 34)
        # Clubs: 4 3 2 (indices 39+9, 39+10, 39+11 = 48, 49, 50)
        cards = [0, 1, 2, 16, 17, 18, 31, 32, 33, 34, 48, 49, 50]
        for c in cards:
            hand[c] = 1
        h32 = hand52_to_hand32(hand)
        assert h32.sum() == 13


class TestHand52ToPbn:
    def test_known_hand(self):
        """SA, SK, SQ => 'AKQ...' with empty suits shown as empty."""
        hand = np.zeros(52)
        hand[0] = 1  # SA
        hand[1] = 1  # SK
        hand[2] = 1  # SQ
        pbn = hand52_to_pbn(hand)
        suits = pbn.split('.')
        assert suits[0] == 'AKQ'
        assert suits[1] == ''
        assert suits[2] == ''
        assert suits[3] == ''

    def test_full_hand(self):
        """13 cards should produce 4 non-trivial suits."""
        hand = np.zeros(52)
        # Card52 encoding: suit*13 + rank where rank 0=A,1=K,...,12=2
        # Spades: A(0) K(1) Q(2)
        for c in [0, 1, 2]:
            hand[c] = 1
        # Hearts: J(13+3=16) T(13+4=17) 9(13+5=18)
        for c in [16, 17, 18]:
            hand[c] = 1
        # Diamonds: 9(26+5=31) 8(26+6=32) 7(26+7=33)
        for c in [31, 32, 33]:
            hand[c] = 1
        # Clubs: 6(39+8=47) 5(39+9=48) 4(39+10=49) 3(39+11=50)
        for c in [47, 48, 49, 50]:
            hand[c] = 1
        pbn = hand52_to_pbn(hand)
        suits = pbn.split('.')
        assert len(suits) == 4
        assert suits[0] == 'AKQ'
        assert suits[1] == 'JT9'
        assert suits[2] == '987'
        assert suits[3] == '6543'


class TestCard52ToCard32:
    def test_ace_of_spades(self):
        assert card52_to_card32(0) == 0  # S=0, rank=0 => 0*8+min(7,0)=0

    def test_deuce_of_spades(self):
        assert card52_to_card32(12) == 7  # S=0, rank=12 => 0*8+min(7,12)=7

    def test_ace_of_hearts(self):
        assert card52_to_card32(13) == 8  # H=1, rank=0 => 1*8+0=8

    def test_club_king(self):
        assert card52_to_card32(40) == 25  # C=3, rank=1 => 3*8+1=25

    def test_pip_clamping(self):
        # S5 = card52(8), rank=8 => 0*8+min(7,8)=7
        assert card52_to_card32(8) == 7
        # S4 = card52(9), rank=9 => 0*8+min(7,9)=7
        assert card52_to_card32(9) == 7


class TestBenBidToActionId:
    def test_pass(self):
        assert ben_bid_to_action_id('PASS') == 0

    def test_double(self):
        assert ben_bid_to_action_id('X') == 1

    def test_redouble(self):
        assert ben_bid_to_action_id('XX') == 2

    def test_one_club(self):
        assert ben_bid_to_action_id('1C') == 3

    def test_seven_nt(self):
        assert ben_bid_to_action_id('7N') == 37


# ---------------------------------------------------------------------------
# Integration tests requiring BEN models (skip if models not available)
# ---------------------------------------------------------------------------

def _ben_models_available():
    """Check if BEN TF2 models exist on disk and TensorFlow is installed."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    model_dir = os.path.join(PROJECT_ROOT, 'models', 'TF2models')
    bidder = os.path.join(
        model_dir, 'GIB-BBO-8730_2025-04-19-E30.keras'
    )
    return os.path.exists(bidder)


@pytest.fixture(scope='module')
def ben_models():
    """Load BEN models once for the whole test module."""
    if not _ben_models_available():
        pytest.skip("BEN TF2 models or TensorFlow not available")

    # Force TF to CPU to avoid GPU memory conflicts with PyTorch
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    from configparser import ConfigParser
    from nn.models_tf2 import Models

    conf = ConfigParser()
    conf.read(os.path.join(SRC_DIR, 'config', 'nn_only.conf'))
    return Models.from_conf(conf, base_path=PROJECT_ROOT)


@pytest.fixture
def sample_deal():
    from rlbridge.engine.deal import Deal
    rng = np.random.RandomState(42)
    return Deal.random(rng)


class TestBenAgentBidding:
    def test_returns_valid_action(self, ben_models, sample_deal):
        from rlbridge.engine.ben_agent import BenAgent
        from rlbridge.engine.game_state import GameState

        agent = BenAgent(ben_models, sample_deal)
        state = GameState.initial(sample_deal)

        # Advance to the first non-PAD_START bidder if needed
        while state.phase == 'bidding':
            player = state.current_player
            obs = state.observation(player)
            legal = state.legal_actions()

            action = agent.act(obs, legal)
            assert action in legal, f"BenAgent bid {action} not in {legal}"

            state = state.apply_action(action)

            # Safety: break after a reasonable number of bids
            if len(state.auction) > 50:
                break


class TestBenAgentCardPlay:
    def test_plays_valid_cards(self, ben_models, sample_deal):
        """Play a full game with BenAgent on all 4 seats and verify legality."""
        from rlbridge.engine.ben_agent import BenAgent
        from rlbridge.engine.game import Game

        agents = [BenAgent(ben_models, sample_deal) for _ in range(4)]
        result = Game(agents, sample_deal).play()

        # Game should complete without error
        assert result is not None
        assert result.score_ns is not None

    def test_multiple_deals(self, ben_models):
        """Play several deals to check robustness."""
        from rlbridge.engine.ben_agent import BenAgent
        from rlbridge.engine.game import Game
        from rlbridge.engine.deal import Deal

        rng = np.random.RandomState(123)
        for seed in range(5):
            deal = Deal.random(rng)
            agents = [BenAgent(ben_models, deal) for _ in range(4)]
            result = Game(agents, deal).play()
            assert result is not None, f"Game failed on seed {seed}"
