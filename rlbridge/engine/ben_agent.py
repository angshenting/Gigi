"""BEN neural network agent — wraps BEN's pre-trained NNs in our Agent interface.

Uses BEN's bidding model (via BotBid) and card-play models (BatchPlayer)
WITHOUT any search, sampling, DDS, PIMC, or BBA.  Pure NN-only play.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from rlbridge.engine.agents import Agent

# BEN imports (available after sys.path insert)
from bidding import bidding
import deck52


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def hand52_to_hand32(hand_binary_52):
    """Convert a (52,) binary hand to (32,) card32 binary.

    In the 32-card encoding each suit has 8 slots: A K Q J T 9 8 x.
    Pips (card52 ranks 7..12, i.e. 7 6 5 4 3 2) are summed into slot 7.
    """
    hand32 = np.zeros(32, dtype=np.float32)
    for c52 in range(52):
        if hand_binary_52[c52] > 0:
            c32 = deck52.card52to32(c52)
            hand32[c32] += hand_binary_52[c52]
    return hand32


def hand52_to_pbn(hand_binary_52):
    """Convert a (52,) binary hand to PBN string like 'AKQ.JT9.8765.432'."""
    return deck52.deal_to_str(hand_binary_52)


def card52_to_card32(c52):
    """Map a card52 index to a card32 index."""
    return deck52.card52to32(c52)


def ben_bid_to_action_id(bid_str):
    """Convert a BEN bid string to our action ID (0-37).

    BEN: PAD_START=0, PAD_END=1, PASS=2, X=3, XX=4, 1C=5 .. 7N=39
    Ours: PASS=0, X=1, XX=2, 1C=3 .. 7N=37
    """
    return bidding.BID2ID[bid_str] - 2


class _MockSampler:
    """Minimal sampler stub so BotBid constructor doesn't crash.

    BotBid.__init__ reads ``sampler.sample_boards_for_auction`` (line 33)
    and the no-rollout branch calls ``sampler.get_bidding_info``.
    With no_search_threshold=0 rollouts never fire and we never generate
    samples, so only dummy values are needed.
    """
    sample_boards_for_auction = 0
    no_samples_when_no_search = True
    sample_hands_auction = 0
    min_sample_hands_auction = 0
    bidding_threshold_sampling = 0
    bid_accept_threshold_bidding = 0

    def get_bidding_info(self, n_steps, auction, nesw_i, hand, vuln, models):
        """Return dummy HCP/shape predictions (never used for decisions)."""
        p_hcp = np.zeros((1, 3), dtype=np.float32)
        p_shp = np.zeros((1, 12), dtype=np.float32)
        return p_hcp, p_shp


# ---------------------------------------------------------------------------
# BenAgent
# ---------------------------------------------------------------------------

class BenAgent(Agent):
    """Agent that plays using BEN's pre-trained neural networks.

    Args:
        models: BEN ``Models`` object (from ``Models.from_conf()``)
        deal: Our ``Deal`` object (needed for dummy-hand access during play)
    """

    def __init__(self, models, deal):
        self.models = models
        self.deal = deal

        # Bidding state (lazy-init per seat on first bid call)
        self._bot_bid = None      # BotBid instance
        self._bid_seat = None     # which seat we're bidding for

        # Card-play state (lazy-init on first play call)
        self._player_model = None
        self._x_play = None       # (1, 13, 298) tensor
        self._trick_i = 0         # completed-trick count when we last wrote x_play
        self._play_initialized = False
        self._ben_pos = None      # position relative to declarer: 0=LHO 1=Dummy 2=RHO 3=Declarer
        self._strain_i = None     # 0=N 1=S 2=H 3=D 4=C (BEN convention)
        self._level = None
        self._declarer_abs = None
        self._my_abs_player = None
        self._playing_dummy = False

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def act(self, observation: dict, legal_actions: list) -> int:
        if observation['phase'] == 'bidding':
            return self._bid(observation, legal_actions)
        else:
            return self._play_card(observation, legal_actions)

    # ------------------------------------------------------------------
    # bidding
    # ------------------------------------------------------------------

    def _bid(self, observation, legal_actions):
        seat = observation['player']  # 0-3, N E S W

        # Lazy-create BotBid (one per seat; if agent is reused across
        # seats in the same game we re-create when seat changes)
        if self._bot_bid is None or self._bid_seat != seat:
            hand_str = hand52_to_pbn(observation['hand'])
            vuln = [observation['vuln_ns'], observation['vuln_ew']]
            self._bot_bid = _make_bot_bid(
                self.models, vuln, hand_str, seat, observation['dealer']
            )
            self._bid_seat = seat

        # Build the auction list that BotBid expects.
        # Our auction tuple contains 'PAD_START' entries for positions
        # before the dealer, plus actual bid strings.  BotBid.bid()
        # expects exactly this format (it validates len(auction)%4 == seat).
        auction = list(observation['auction'])

        resp = self._bot_bid.bid(auction)
        bid_str = resp.bid  # e.g. 'PASS', '1C', 'X', …
        action_id = ben_bid_to_action_id(bid_str)

        # Safety: clamp to legal_actions
        if action_id in legal_actions:
            return action_id
        # Fallback — shouldn't happen, but be safe
        return legal_actions[0]

    # ------------------------------------------------------------------
    # card play
    # ------------------------------------------------------------------

    def _play_card(self, observation, legal_actions):
        if not self._play_initialized:
            self._init_play(observation, legal_actions)

        # Figure out which completed-trick index we're up to
        n_completed = len(observation['tricks'])
        current_trick_cards = observation['current_trick']

        # The trick slot we write into is n_completed (for the current trick)
        trick_slot = n_completed

        # Rebuild x_play up to the current moment
        self._rebuild_x_play(observation, n_completed, current_trick_cards)

        # Query the NN: input shape (1, trick_slot+1, 298)
        x_in = self._x_play[:, :trick_slot + 1, :]
        x_in_f16 = x_in.astype(np.float16)
        softmax = self._player_model.next_cards_softmax(x_in_f16)  # (1, 32)

        # Map card32 probabilities to the best legal card52
        probs32 = softmax[0]  # shape (32,)
        best_action = None
        best_prob = -1.0

        for c52 in legal_actions:
            c32 = card52_to_card32(c52)
            if probs32[c32] > best_prob:
                best_prob = probs32[c32]
                best_action = c52

        if best_action is None:
            best_action = legal_actions[0]

        return best_action

    def _init_play(self, observation, legal_actions):
        """One-time setup when transitioning from bidding to card play."""
        contract = observation['contract']  # e.g. '3NE', '4HXS'
        self._declarer_abs = bidding.get_decl_i(contract)
        self._strain_i = bidding.get_strain_i(contract)
        self._level = int(contract[0])
        self._my_abs_player = observation['player']

        # Detect if we're playing dummy's cards.
        # When the game engine asks declarer to play dummy's cards, the
        # observation['hand'] still shows declarer's hand, but legal_actions
        # are from dummy's hand.  We detect this by checking if legal
        # actions are cards in dummy's hand (observation['dummy']).
        dummy_abs = (self._declarer_abs + 2) % 4
        dummy_hand = observation['dummy']
        own_hand = observation['hand']

        # Check if the legal actions come from dummy's hand
        self._playing_dummy = False
        if dummy_hand is not None and np.sum(dummy_hand) > 0:
            legal_in_dummy = all(dummy_hand[c] > 0 for c in legal_actions)
            legal_in_own = all(own_hand[c] > 0 for c in legal_actions)
            if legal_in_dummy and not legal_in_own:
                self._playing_dummy = True

        if self._playing_dummy:
            self._ben_pos = 1  # dummy model
        else:
            self._ben_pos = (self._my_abs_player - self._declarer_abs - 1) % 4

        # Select the right model: player_models[ben_pos] for NT, [ben_pos+4] for suit
        if self._strain_i == 0:  # NT
            self._player_model = self.models.player_models[self._ben_pos]
        else:
            self._player_model = self.models.player_models[self._ben_pos + 4]

        # Allocate x_play tensor: (1, 13, 298)
        self._x_play = np.zeros((1, 13, 298), dtype=np.float32)
        self._play_initialized = True

    def _rebuild_x_play(self, observation, n_completed, current_trick_cards):
        """Rebuild x_play tensor from observation state.

        Re-derives the full tensor each call from the observation so
        we don't need to maintain incremental state.
        """
        x = self._x_play
        x[:] = 0  # clear

        contract = observation['contract']
        declarer_abs = self._declarer_abs
        dummy_abs = (declarer_abs + 2) % 4
        ben_pos = self._ben_pos

        # Determine which hand we're playing with
        if self._playing_dummy:
            my_hand52 = observation['dummy'].copy()
        else:
            my_hand52 = observation['hand'].copy()

        # Dummy's initial hand (public hand) from the deal
        dummy_initial_hand52 = self.deal.hands_binary[dummy_abs].copy()

        # If WE are dummy, the "public hand" is declarer's hand
        if self._playing_dummy:
            public_initial_hand52 = self.deal.hands_binary[declarer_abs].copy()
        else:
            public_initial_hand52 = dummy_initial_hand52.copy()

        # We'll track remaining hands through tricks
        my_remaining = self.deal.hands_binary[
            dummy_abs if self._playing_dummy else self._my_abs_player
        ].copy()
        public_remaining = public_initial_hand52.copy()

        tricks = observation['tricks']
        trick_leaders = observation['trick_leaders']

        # Strain one-hot: N=0, S=1, H=2, D=3, C=4
        strain_one_hot = np.zeros(5, dtype=np.float32)
        strain_one_hot[self._strain_i] = 1.0

        # --- Trick 0: initial state ---
        my_hand32 = hand52_to_hand32(my_remaining)
        public_hand32 = hand52_to_hand32(public_remaining)

        x[0, 0, :32] = my_hand32
        x[0, 0, 32:64] = public_hand32
        # No last trick for trick 0 (slots 64:292 stay zero)
        x[0, 0, 292] = self._level
        x[0, 0, 293:298] = strain_one_hot

        # --- Fill completed tricks ---
        for t in range(n_completed):
            trick = tricks[t]
            leader = trick_leaders[t]

            # Figure out which card in this trick was ours
            my_card_in_trick = self._find_my_card_in_trick(
                trick, leader, my_remaining
            )
            if my_card_in_trick is not None:
                my_remaining[my_card_in_trick] = 0

            # Figure out public hand card
            pub_card_in_trick = self._find_public_card_in_trick(
                trick, leader, public_remaining
            )
            if pub_card_in_trick is not None:
                public_remaining[pub_card_in_trick] = 0

            # Write trick t's current-trick slots (cards played before us)
            self._write_current_trick_slots(x, t, trick, leader, ben_pos, declarer_abs, full_trick=True)

            # Prepare trick t+1 (if there's a next slot)
            if t + 1 < 13:
                next_my32 = hand52_to_hand32(my_remaining)
                next_pub32 = hand52_to_hand32(public_remaining)
                x[0, t + 1, :32] = next_my32
                x[0, t + 1, 32:64] = next_pub32

                # Last trick = trick t's 4 cards in card32, play order
                for ci, c52 in enumerate(trick):
                    c32 = card52_to_card32(c52)
                    x[0, t + 1, 64 + ci * 32 + c32] = 1.0

                # Last trick leader in BEN position
                leader_ben_pos = (leader - declarer_abs - 1) % 4
                x[0, t + 1, 288 + leader_ben_pos] = 1.0

                x[0, t + 1, 292] = self._level
                x[0, t + 1, 293:298] = strain_one_hot

        # --- Current (incomplete) trick ---
        if n_completed < 13:
            # The current trick slot is n_completed; its hand/public/level/strain
            # are already set (either from trick 0 init or from last completed trick prep)
            # Now set the current-trick card slots for cards already played this trick
            if current_trick_cards:
                leader = observation['current_leader']
                self._write_current_trick_slots(
                    x, n_completed, current_trick_cards, leader, ben_pos,
                    declarer_abs, full_trick=False
                )

    def _find_my_card_in_trick(self, trick, leader, my_remaining):
        """Find which card52 in a completed trick came from our hand."""
        for ci, c52 in enumerate(trick):
            player_abs = (leader + ci) % 4
            if self._playing_dummy:
                target = (self._declarer_abs + 2) % 4  # dummy
            else:
                target = self._my_abs_player
            if player_abs == target and my_remaining[c52] > 0:
                return c52
        return None

    def _find_public_card_in_trick(self, trick, leader, public_remaining):
        """Find which card52 in a completed trick came from the public hand."""
        for ci, c52 in enumerate(trick):
            player_abs = (leader + ci) % 4
            if self._playing_dummy:
                target = self._declarer_abs  # public = declarer when we're dummy
            else:
                target = (self._declarer_abs + 2) % 4  # public = dummy
            if player_abs == target and public_remaining[c52] > 0:
                return c52
        return None

    def _write_current_trick_slots(self, x, slot, trick_cards, leader,
                                    ben_pos, declarer_abs, full_trick):
        """Write the current-trick card slots (192:288) for a trick.

        ``trick_cards`` is a tuple of card52 values in play order.
        ``leader`` is the absolute player (0-3) who led.
        ``full_trick`` is True for completed tricks (4 cards), False for
        the in-progress trick.

        BEN positions relative to the player:
          192:224 = LHO card
          224:256 = partner card
          256:288 = RHO card
        We map absolute play positions to relative (LHO/partner/RHO) based
        on the BEN position system.
        """
        # Our absolute position
        if self._playing_dummy:
            my_abs = (declarer_abs + 2) % 4
        else:
            my_abs = self._my_abs_player

        for ci, c52 in enumerate(trick_cards):
            player_abs = (leader + ci) % 4
            if player_abs == my_abs:
                continue  # skip our own card (not in the input)

            # Relative position of this player to us
            rel = (player_abs - my_abs) % 4  # 1=LHO, 2=partner, 3=RHO

            c32 = card52_to_card32(c52)
            if rel == 1:    # LHO
                x[0, slot, 192 + c32] = 1.0
            elif rel == 2:  # partner
                x[0, slot, 224 + c32] = 1.0
            elif rel == 3:  # RHO
                x[0, slot, 256 + c32] = 1.0


def _make_bot_bid(models, vuln, hand_str, seat, dealer):
    """Create a BotBid instance configured for pure-NN bidding."""
    from botbidder import BotBid
    return BotBid(
        vuln=vuln,
        hand_str=hand_str,
        models=models,
        sampler=_MockSampler(),
        seat=seat,
        dealer=dealer,
        ddsolver=None,
        bba_is_controlling=False,
        verbose=False,
    )
