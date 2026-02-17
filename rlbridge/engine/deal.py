import sys
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Add BEN src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@dataclass(frozen=True)
class Deal:
    """Immutable representation of a bridge deal."""
    hands: tuple             # ((card52,...), ...) for N, E, S, W
    hands_binary: np.ndarray  # (4, 52) binary indicator — not hashed
    dealer: int              # 0=N, 1=E, 2=S, 3=W
    vuln_ns: bool
    vuln_ew: bool

    def __eq__(self, other):
        if not isinstance(other, Deal):
            return NotImplemented
        return (self.hands == other.hands and
                self.dealer == other.dealer and
                self.vuln_ns == other.vuln_ns and
                self.vuln_ew == other.vuln_ew)

    def __hash__(self):
        return hash((self.hands, self.dealer, self.vuln_ns, self.vuln_ew))

    @staticmethod
    def random(rng: Optional[np.random.RandomState] = None) -> 'Deal':
        """Generate a random deal with random dealer and vulnerability."""
        if rng is None:
            rng = np.random.RandomState()

        all_cards = list(range(52))
        rng.shuffle(all_cards)

        hands_list = []
        hands_binary = np.zeros((4, 52), dtype=np.float32)
        for i in range(4):
            cards = tuple(sorted(all_cards[i * 13:(i + 1) * 13]))
            hands_list.append(cards)
            for c in cards:
                hands_binary[i, c] = 1.0

        hands = tuple(hands_list)
        dealer = rng.randint(4)
        vuln_ns = bool(rng.randint(2))
        vuln_ew = bool(rng.randint(2))

        return Deal(
            hands=hands,
            hands_binary=hands_binary,
            dealer=dealer,
            vuln_ns=vuln_ns,
            vuln_ew=vuln_ew,
        )

    def hand_pbn(self) -> str:
        """Convert to PBN format for DDS: 'N:S.H.D.C S.H.D.C S.H.D.C S.H.D.C'"""
        suits_str = 'SHDC'
        ranks_str = 'AKQJT98765432'
        hand_strings = []
        for player in range(4):
            suits = ['', '', '', '']
            for card in self.hands[player]:
                suit_i = card // 13
                rank_i = card % 13
                suits[suit_i] += ranks_str[rank_i]
            hand_strings.append('.'.join(suits))
        return 'N:' + ' '.join(hand_strings)

    def hand_cards(self, player: int) -> tuple:
        """Return card52 indices for a player."""
        return self.hands[player]

    def hand_str(self, player: int) -> str:
        """Human-readable hand string like 'AKQ.JT9.8765.432'"""
        ranks_str = 'AKQJT98765432'
        suits = ['', '', '', '']
        for card in self.hands[player]:
            suit_i = card // 13
            rank_i = card % 13
            suits[suit_i] += ranks_str[rank_i]
        return '.'.join(suits)
