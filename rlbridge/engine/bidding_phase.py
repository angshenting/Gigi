"""Bidding rules — thin wrapper over BEN's bidding.bidding module.

Action ID mapping (our convention):
  0=PASS, 1=X, 2=XX, 3=1C, 4=1D, ..., 37=7N

BEN convention (BID2ID):
  0=PAD_START, 1=PAD_END, 2=PASS, 3=X, 4=XX, 5=1C, ..., 39=7N

Conversion: ben_id = action_id + 2  (for PASS/X/XX/suit bids)
"""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from bidding import bidding


# Our action IDs: 0=PASS, 1=X, 2=XX, 3=1C .. 37=7N (38 total)
NUM_BID_ACTIONS = 38

# Build mapping tables
_ACTION_TO_BEN = {}  # our action_id -> BEN bid string
_BEN_TO_ACTION = {}  # BEN bid string -> our action_id

_ACTION_TO_BEN[0] = 'PASS'
_ACTION_TO_BEN[1] = 'X'
_ACTION_TO_BEN[2] = 'XX'
_BEN_TO_ACTION['PASS'] = 0
_BEN_TO_ACTION['X'] = 1
_BEN_TO_ACTION['XX'] = 2

for ben_id in range(5, 40):  # 1C=5 .. 7N=39
    bid_str = bidding.ID2BID[ben_id]
    action_id = ben_id - 2  # 1C=3 .. 7N=37
    _ACTION_TO_BEN[action_id] = bid_str
    _BEN_TO_ACTION[bid_str] = action_id


def action_to_bid(action_id: int) -> str:
    """Convert our action ID (0-37) to BEN bid string."""
    return _ACTION_TO_BEN[action_id]


def bid_to_action(bid_str: str) -> int:
    """Convert BEN bid string to our action ID (0-37)."""
    return _BEN_TO_ACTION[bid_str]


def legal_bids(auction: tuple) -> list:
    """Return list of legal action IDs (0-37) given current auction.

    auction: tuple of bid strings, e.g. ('PAD_START', '1C', 'PASS', ...)
    """
    # Strip PAD_START from the front for can_bid checks
    clean = [b for b in auction if b != 'PAD_START']

    actions = []
    for action_id in range(NUM_BID_ACTIONS):
        bid_str = _ACTION_TO_BEN[action_id]
        if bidding.can_bid(bid_str, clean):
            actions.append(action_id)
    return actions


def apply_bid(auction: tuple, action_id: int) -> tuple:
    """Apply a bid action and return the new auction tuple."""
    bid_str = _ACTION_TO_BEN[action_id]
    return auction + (bid_str,)


def is_over(auction: tuple) -> bool:
    """Check if the auction is complete."""
    clean = list(auction)
    return bidding.auction_over(clean)


def get_result(auction: tuple) -> Optional[tuple]:
    """Extract contract info from a completed auction.

    Returns (contract_str, declarer_i, strain_i, dummy_i) or None if passed out.
    contract_str: e.g. '3NE', '4HXS'
    declarer_i: 0-3 (N,E,S,W)
    strain_i: bidding convention (N=0,S=1,H=2,D=3,C=4)
    dummy_i: (declarer_i + 2) % 4
    """
    clean = [b for b in auction if b not in ('PAD_START', 'PAD_END')]

    contract = bidding.get_contract(clean)
    if contract is None or contract == 'PASS':
        return None

    strain_i = bidding.get_strain_i(contract)
    declarer_i = bidding.get_decl_i(contract)
    dummy_i = (declarer_i + 2) % 4

    return (contract, declarer_i, strain_i, dummy_i)
