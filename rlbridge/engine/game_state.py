"""Immutable game state with state transitions for self-play bridge.

Phases:
  'bidding'      — auction in progress
  'opening_lead' — declarer's LHO leads (special: dummy not yet visible)
  'play'         — card play after opening lead
  'done'         — contract completed, can score
  'passed_out'   — four passes, no contract
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from scoring import score as ben_score

from rlbridge.engine.deal import Deal
from rlbridge.engine import bidding_phase
from rlbridge.engine import play_phase


@dataclass(frozen=True)
class GameState:
    deal: Deal
    auction: tuple                         # ('PAD_START', '1C', 'PASS', ...)
    contract: Optional[str]                # '3NE', '4HXS', None
    declarer: Optional[int]                # 0-3
    strain_i: Optional[int]                # bidding convention (N=0,S=1,H=2,D=3,C=4)
    dummy: Optional[int]                   # (declarer+2)%4
    tricks: tuple                          # completed tricks: ((card52,...), ...)
    current_trick: tuple                   # 0-3 card52 values
    trick_leaders: tuple                   # who led each completed trick
    current_leader: Optional[int]          # who leads current trick
    cards_played: tuple                    # per-player frozenset of card52 played

    def __eq__(self, other):
        if not isinstance(other, GameState):
            return NotImplemented
        return (self.deal == other.deal and
                self.auction == other.auction and
                self.contract == other.contract and
                self.tricks == other.tricks and
                self.current_trick == other.current_trick)

    def __hash__(self):
        return hash((id(self.deal), self.auction, self.contract,
                     self.tricks, self.current_trick))

    @staticmethod
    def initial(deal: Deal) -> 'GameState':
        """Create initial game state from a deal."""
        # Pad auction with PAD_START for positions before dealer
        pad = tuple('PAD_START' for _ in range(deal.dealer))
        return GameState(
            deal=deal,
            auction=pad,
            contract=None,
            declarer=None,
            strain_i=None,
            dummy=None,
            tricks=(),
            current_trick=(),
            trick_leaders=(),
            current_leader=None,
            cards_played=tuple(frozenset() for _ in range(4)),
        )

    @property
    def phase(self) -> str:
        """Current game phase."""
        if not bidding_phase.is_over(self.auction):
            return 'bidding'

        if self.contract is None:
            # Auction is over but no contract set — need to check
            result = bidding_phase.get_result(self.auction)
            if result is None:
                return 'passed_out'
            # This shouldn't happen if apply_action works correctly
            return 'bidding'

        total_cards = sum(len(cp) for cp in self.cards_played)
        if total_cards == 0:
            return 'opening_lead'
        if total_cards >= 52:
            return 'done'
        if len(self.tricks) == 13:
            return 'done'
        return 'play'

    @property
    def current_player(self) -> int:
        """Whose turn it is (0-3)."""
        p = self.phase
        if p == 'bidding':
            return len(self.auction) % 4
        if p == 'opening_lead':
            # Declarer's LHO leads
            return (self.declarer + 1) % 4
        if p == 'play':
            if len(self.current_trick) == 0:
                return self.current_leader
            # Next player after the last card in the trick
            n_played = len(self.current_trick)
            return (self.current_leader + n_played) % 4
        # done or passed_out
        return -1

    @property
    def is_terminal(self) -> bool:
        return self.phase in ('done', 'passed_out')

    @property
    def ns_tricks(self) -> int:
        """Number of tricks won by NS."""
        if self.declarer is None:
            return 0
        count = 0
        for i, trick in enumerate(self.tricks):
            leader = self.trick_leaders[i]
            winner_offset = play_phase.trick_winner(trick, self.strain_i)
            winner = (leader + winner_offset) % 4
            if winner in (0, 2):  # N or S
                count += 1
        return count

    @property
    def ew_tricks(self) -> int:
        return len(self.tricks) - self.ns_tricks

    def score_ns(self) -> int:
        """Compute NS score. Only valid when terminal."""
        if self.phase == 'passed_out':
            return 0
        if self.phase != 'done':
            raise ValueError("Cannot score non-terminal state")

        # Determine vulnerability for declarer
        if self.declarer in (0, 2):  # N or S
            is_vuln = self.deal.vuln_ns
        else:
            is_vuln = self.deal.vuln_ew

        # Determine tricks won by declaring side
        if self.declarer in (0, 2):
            decl_tricks = self.ns_tricks
        else:
            decl_tricks = self.ew_tricks

        # Build contract string for scoring: e.g. '3N', '4SX', '7NXX'
        # self.contract has format like '3NE', '4HXS' — strip the declarer letter
        contract_for_score = self.contract[:-1]  # remove declarer char

        raw_score = ben_score(contract_for_score, is_vuln, decl_tricks)

        # If EW is declaring, NS score is negative of the raw score
        if self.declarer in (0, 2):
            return raw_score
        else:
            return -raw_score

    def legal_actions(self) -> list:
        """Return legal action IDs for current player.

        Bidding phase: action IDs 0-37 (PASS=0, X=1, XX=2, 1C=3, ..., 7N=37)
        Play phase: card52 indices 0-51
        """
        p = self.phase
        if p == 'bidding':
            return bidding_phase.legal_bids(self.auction)
        if p in ('opening_lead', 'play'):
            player = self.current_player
            hand = self._remaining_hand(player)
            return play_phase.legal_cards(hand, self.current_trick,
                                          self.current_leader, player)
        return []

    def apply_action(self, action: int) -> 'GameState':
        """Apply an action and return a new GameState."""
        p = self.phase
        if p == 'bidding':
            return self._apply_bid(action)
        if p in ('opening_lead', 'play'):
            return self._apply_card(action)
        raise ValueError(f"Cannot apply action in phase '{p}'")

    def observation(self, player: int) -> dict:
        """Return imperfect-information observation for a player."""
        p = self.phase

        # Hand: player's remaining cards
        hand = self._remaining_hand(player)

        # Dummy visibility: visible after opening lead (except to dummy themselves)
        dummy_visible = False
        dummy_hand = np.zeros(52, dtype=np.float32)
        if self.dummy is not None and p == 'play':
            dummy_visible = True
            dummy_hand = self._remaining_hand(self.dummy)

        return {
            'phase': 'bidding' if p == 'bidding' else 'play',
            'hand': hand,
            'dummy': dummy_hand,
            'dummy_visible': dummy_visible,
            'auction': self.auction,
            'tricks': self.tricks,
            'trick_leaders': self.trick_leaders,
            'current_trick': self.current_trick,
            'current_leader': self.current_leader,
            'contract': self.contract,
            'vuln_ns': self.deal.vuln_ns,
            'vuln_ew': self.deal.vuln_ew,
            'player': player,
            'dealer': self.deal.dealer,
            'legal_actions': self.legal_actions(),
        }

    def _remaining_hand(self, player: int) -> np.ndarray:
        """Return (52,) binary array of remaining cards for player."""
        hand = self.deal.hands_binary[player].copy()
        for card in self.cards_played[player]:
            hand[card] = 0.0
        return hand

    def _apply_bid(self, action_id: int) -> 'GameState':
        """Apply a bidding action."""
        new_auction = bidding_phase.apply_bid(self.auction, action_id)

        # Check if auction is now over
        if bidding_phase.is_over(new_auction):
            result = bidding_phase.get_result(new_auction)
            if result is None:
                # Passed out
                return GameState(
                    deal=self.deal,
                    auction=new_auction,
                    contract=None,
                    declarer=None,
                    strain_i=None,
                    dummy=None,
                    tricks=(),
                    current_trick=(),
                    trick_leaders=(),
                    current_leader=None,
                    cards_played=self.cards_played,
                )
            contract, declarer_i, strain_i, dummy_i = result
            # Opening leader is declarer's LHO
            opening_leader = (declarer_i + 1) % 4
            return GameState(
                deal=self.deal,
                auction=new_auction,
                contract=contract,
                declarer=declarer_i,
                strain_i=strain_i,
                dummy=dummy_i,
                tricks=(),
                current_trick=(),
                trick_leaders=(),
                current_leader=opening_leader,
                cards_played=self.cards_played,
            )

        return GameState(
            deal=self.deal,
            auction=new_auction,
            contract=self.contract,
            declarer=self.declarer,
            strain_i=self.strain_i,
            dummy=self.dummy,
            tricks=self.tricks,
            current_trick=self.current_trick,
            trick_leaders=self.trick_leaders,
            current_leader=self.current_leader,
            cards_played=self.cards_played,
        )

    def _apply_card(self, card: int) -> 'GameState':
        """Apply a card play action."""
        player = self.current_player
        new_trick = self.current_trick + (card,)

        # Update cards_played
        new_cards_played = list(self.cards_played)
        new_cards_played[player] = self.cards_played[player] | frozenset([card])
        new_cards_played = tuple(new_cards_played)

        if play_phase.is_trick_complete(new_trick):
            # Determine winner
            winner_offset = play_phase.trick_winner(new_trick, self.strain_i)
            winner = (self.current_leader + winner_offset) % 4

            new_tricks = self.tricks + (new_trick,)
            new_trick_leaders = self.trick_leaders + (self.current_leader,)

            # Check if all 13 tricks are done
            if len(new_tricks) == 13:
                return GameState(
                    deal=self.deal,
                    auction=self.auction,
                    contract=self.contract,
                    declarer=self.declarer,
                    strain_i=self.strain_i,
                    dummy=self.dummy,
                    tricks=new_tricks,
                    current_trick=(),
                    trick_leaders=new_trick_leaders,
                    current_leader=None,
                    cards_played=new_cards_played,
                )

            # Start new trick, winner leads
            return GameState(
                deal=self.deal,
                auction=self.auction,
                contract=self.contract,
                declarer=self.declarer,
                strain_i=self.strain_i,
                dummy=self.dummy,
                tricks=new_tricks,
                current_trick=(),
                trick_leaders=new_trick_leaders,
                current_leader=winner,
                cards_played=new_cards_played,
            )

        # Trick not complete, continue
        return GameState(
            deal=self.deal,
            auction=self.auction,
            contract=self.contract,
            declarer=self.declarer,
            strain_i=self.strain_i,
            dummy=self.dummy,
            tricks=self.tricks,
            current_trick=new_trick,
            trick_leaders=self.trick_leaders,
            current_leader=self.current_leader,
            cards_played=new_cards_played,
        )
