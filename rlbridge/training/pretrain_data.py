"""PBN parser, auction/play replay, and PyTorch Dataset for supervised pre-training.

Parses PBN files (from Boards/BBA/, Boards/BBO/, etc.) into board dicts
with deal, auction, and optional play data.  Replays each game to generate
(observation, target_action_id, is_bid) examples for both bidding and card
play, and wraps them in a PyTorch Dataset compatible with SupervisedTrainer.
"""

import os
import re
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from rlbridge.engine.deal import Deal
from rlbridge.engine.game_state import GameState
from rlbridge.engine import bidding_phase
from rlbridge.model.config import ModelConfig
from rlbridge.model.encoder import encode_observation, collate_observations

logger = logging.getLogger(__name__)

# Card encoding: card52 = suit_i * 13 + rank_i
# suit_i: 0=S, 1=H, 2=D, 3=C
# rank_i: 0=A, 1=K, 2=Q, 3=J, 4=T, 5=9, ..., 12=2
_RANK_CHAR_TO_IDX = {
    'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4,
    '9': 5, '8': 6, '7': 7, '6': 8, '5': 9,
    '4': 10, '3': 11, '2': 12,
}

# PBN deal string order: S.H.D.C (suits separated by dots)
_PBN_SUIT_ORDER = [0, 1, 2, 3]  # S=0, H=1, D=2, C=3

# Player name to index
_PLAYER_TO_IDX = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

# PBN card notation: suit letter to card52 suit index
_SUIT_CHAR_TO_IDX = {'S': 0, 'H': 1, 'D': 2, 'C': 3}

# Vulnerability string normalization
_VULN_MAP = {
    'None': (False, False),
    'Love': (False, False),
    '-': (False, False),
    'NS': (True, False),
    'N/S': (True, False),
    'EW': (False, True),
    'E/W': (False, True),
    'All': (True, True),
    'Both': (True, True),
}

# PBN bid normalization to BEN bid strings
_BID_NORMALIZE = {
    'P': 'PASS',
    'Pass': 'PASS',
    'PASS': 'PASS',
    'Pas': 'PASS',
    'p': 'PASS',
    'pass': 'PASS',
    'D': 'X',
    'Dbl': 'X',
    'X': 'X',
    'R': 'XX',
    'Rdbl': 'XX',
    'XX': 'XX',
}


def pbn_card_to_card52(card_str: str) -> int:
    """Convert a PBN card string like 'CK' to a card52 index.

    Format: suit_char + rank_char, e.g. 'SA'=0, 'H2'=25, 'CK'=40.
    """
    if len(card_str) != 2:
        raise ValueError(f"Invalid card string: {card_str!r}")
    suit_i = _SUIT_CHAR_TO_IDX[card_str[0].upper()]
    rank_i = _RANK_CHAR_TO_IDX[card_str[1].upper()]
    return suit_i * 13 + rank_i


def _parse_play_section(play_starter: str, play_lines: list) -> Optional[list]:
    """Parse [Play] lines into a list of tricks.

    Args:
        play_starter: the player character from [Play "X"] tag (W, N, E, S)
        play_lines: raw text lines from the play section

    Returns:
        List of tricks, where each trick is a list of (player_idx, card52)
        tuples in play order (clockwise from the leader of that trick).
        Returns None if parsing fails.
    """
    if play_starter not in _PLAYER_TO_IDX:
        return None

    starter_idx = _PLAYER_TO_IDX[play_starter]
    tricks = []

    for line in play_lines:
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('%'):
            continue
        # Handle "*" annotations (BBO alert markers on cards)
        line = line.replace('*', '')
        tokens = line.split()
        if len(tokens) < 4:
            continue

        trick = []
        for col_i, token in enumerate(tokens[:4]):
            if token == '-':
                # Incomplete trick
                break
            try:
                card52 = pbn_card_to_card52(token)
            except (ValueError, KeyError):
                break
            player_idx = (starter_idx + col_i) % 4
            trick.append((player_idx, card52))

        if len(trick) == 4:
            tricks.append(trick)

    return tricks if tricks else None


def _parse_hand_string(hand_str: str) -> Tuple[int, ...]:
    """Convert a PBN hand string like 'KT.AJT82.763.K53' to card52 indices.

    Format: Spades.Hearts.Diamonds.Clubs
    Each suit is a string of rank characters (AKQJT98765432), empty for void.

    Returns:
        Sorted tuple of card52 indices.
    """
    suits = hand_str.split('.')
    if len(suits) != 4:
        raise ValueError(f"Expected 4 suits, got {len(suits)}: {hand_str!r}")

    cards = []
    for suit_i, suit_str in zip(_PBN_SUIT_ORDER, suits):
        for ch in suit_str:
            if ch in _RANK_CHAR_TO_IDX:
                rank_i = _RANK_CHAR_TO_IDX[ch]
                card52 = suit_i * 13 + rank_i
                cards.append(card52)
    return tuple(sorted(cards))


def _parse_deal_tag(deal_str: str) -> Tuple[Tuple[Tuple[int, ...], ...], int]:
    """Parse a PBN [Deal] value like 'N:S.H.D.C S.H.D.C S.H.D.C S.H.D.C'.

    The first letter indicates which player's hand is listed first, then
    hands proceed clockwise: first, LHO, partner, RHO.

    Returns:
        (hands, first_player_idx) where hands[0..3] are for N,E,S,W.
    """
    parts = deal_str.strip().split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid deal format: {deal_str!r}")

    first_player = _PLAYER_TO_IDX[parts[0].strip()]
    hand_strs = parts[1].strip().split()
    if len(hand_strs) != 4:
        raise ValueError(f"Expected 4 hands, got {len(hand_strs)}: {deal_str!r}")

    # Assign hands clockwise from first_player
    hands = [None, None, None, None]
    for i, hs in enumerate(hand_strs):
        player = (first_player + i) % 4
        hands[player] = _parse_hand_string(hs)

    return tuple(hands), first_player


def _make_deal(hands: tuple, dealer: int, vuln_ns: bool, vuln_ew: bool) -> Deal:
    """Construct a Deal from parsed PBN data."""
    hands_binary = np.zeros((4, 52), dtype=np.float32)
    for p in range(4):
        for c in hands[p]:
            hands_binary[p, c] = 1.0

    return Deal(
        hands=hands,
        hands_binary=hands_binary,
        dealer=dealer,
        vuln_ns=vuln_ns,
        vuln_ew=vuln_ew,
    )


def _normalize_bid(bid_str: str) -> Optional[str]:
    """Normalize a PBN bid string to BEN format.

    Returns None for unrecognized bids (alerts, notes, etc.).
    """
    # Strip annotation markers like =1=
    bid_str = re.sub(r'=\d+=', '', bid_str).strip()
    # Strip alert markers
    bid_str = bid_str.rstrip('*').strip()
    if not bid_str or bid_str == '-' or bid_str == 'AP':
        return None

    # Check simple mappings first
    if bid_str in _BID_NORMALIZE:
        return _BID_NORMALIZE[bid_str]

    # Contract bids: level + suit  (e.g. "1C", "3NT", "7N")
    m = re.match(r'^([1-7])(C|D|H|S|N|NT)$', bid_str, re.IGNORECASE)
    if m:
        level = m.group(1)
        suit = m.group(2).upper()
        if suit == 'NT':
            suit = 'N'
        return f'{level}{suit}'

    return None


def parse_pbn_file(path: str) -> List[dict]:
    """Parse a single PBN file and return list of board dicts.

    Each dict has keys: 'deal' (Deal), 'auction' (list of BEN bid strings),
    'play' (list of tricks or None).
    Boards without a valid [Deal] or [Auction] are skipped.
    """
    boards = []

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # State for current board
    tags = {}
    auction_lines = []
    in_auction = False
    play_lines = []
    in_play = False
    play_starter = None

    def _flush_board():
        """Try to build a board from accumulated tags."""
        if 'Deal' not in tags or 'Dealer' not in tags:
            return
        if not auction_lines and 'Auction' not in tags:
            return

        try:
            # Parse deal
            hands, _ = _parse_deal_tag(tags['Deal'])

            # Validate hands
            all_cards = set()
            for h in hands:
                if len(h) != 13:
                    return
                all_cards.update(h)
            if len(all_cards) != 52:
                return

            # Dealer
            dealer_str = tags['Dealer'].strip()
            if dealer_str not in _PLAYER_TO_IDX:
                return
            dealer = _PLAYER_TO_IDX[dealer_str]

            # Vulnerability
            vuln_str = tags.get('Vulnerable', 'None').strip()
            if vuln_str not in _VULN_MAP:
                return
            vuln_ns, vuln_ew = _VULN_MAP[vuln_str]

            # Parse auction
            auction_starter = tags.get('Auction', '').strip()
            if auction_starter and auction_starter not in _PLAYER_TO_IDX:
                return

            bids = []
            for line in auction_lines:
                # Split by whitespace
                tokens = line.split()
                for token in tokens:
                    bid = _normalize_bid(token)
                    if bid is not None:
                        bids.append(bid)
                    elif token.startswith('[') or token.startswith('%'):
                        break  # hit next tag or comment

            if not bids:
                return

            # Parse play data if available
            play_data = None
            if play_lines and play_starter:
                play_data = _parse_play_section(play_starter, play_lines)

            deal = _make_deal(hands, dealer, vuln_ns, vuln_ew)
            boards.append({'deal': deal, 'auction': bids, 'play': play_data})

        except (ValueError, KeyError, IndexError):
            pass  # skip malformed boards

    for line in lines:
        line = line.rstrip('\n\r')

        # Skip comments
        if line.startswith('%'):
            continue

        # Check for tag
        tag_match = re.match(r'^\[(\w+)\s+"(.*)"\]', line)
        if tag_match:
            tag_name = tag_match.group(1)
            tag_value = tag_match.group(2)

            # If we hit a new Event/Board tag, flush the previous board
            if tag_name in ('Event', 'Board'):
                if tags:
                    _flush_board()
                if tag_name == 'Event':
                    tags = {}
                    auction_lines = []
                    in_auction = False
                    play_lines = []
                    in_play = False
                    play_starter = None

            if tag_name == 'Auction':
                in_auction = True
                in_play = False
                tags['Auction'] = tag_value
                auction_lines = []
            elif tag_name == 'Play':
                in_auction = False
                in_play = True
                play_starter = tag_value.strip()
                play_lines = []
            elif tag_name in ('Note', 'OptimumResultTable',
                              'BidSystemNS', 'BidSystemEW', 'BCFlags'):
                in_auction = False
                in_play = False
            else:
                in_auction = False
                in_play = False
                tags[tag_name] = tag_value
            continue

        # If we're reading auction lines
        if in_auction and line.strip():
            # Skip lines that start with new tags
            if line.strip().startswith('['):
                in_auction = False
                continue
            auction_lines.append(line)

        # If we're reading play lines
        if in_play and line.strip():
            if line.strip().startswith('['):
                in_play = False
                continue
            play_lines.append(line)

    # Flush last board
    if tags:
        _flush_board()

    return boards


def load_all_pbn(directories: List[str]) -> List[dict]:
    """Load all PBN files from given directories.

    Returns list of board dicts with 'deal', 'auction', and 'play' keys.
    """
    all_boards = []
    for directory in directories:
        if not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith('.pbn'):
                filepath = os.path.join(directory, filename)
                try:
                    boards = parse_pbn_file(filepath)
                    all_boards.extend(boards)
                except Exception as e:
                    logger.warning(f"Error parsing {filepath}: {e}")
    logger.info(f"Loaded {len(all_boards)} boards from {len(directories)} directories")
    return all_boards


def generate_supervised_examples(
    boards: List[dict],
) -> List[Tuple[dict, int]]:
    """Replay auctions to generate (observation, target_action_id) pairs.

    For each board, creates a GameState and steps through the auction.
    At each step, records the current player's observation and the actual
    bid as the target action.

    Args:
        boards: list of dicts with 'deal' (Deal) and 'auction' (list[str])

    Returns:
        List of (observation_dict, target_action_id) tuples.
    """
    examples = []
    skipped = 0

    for board in boards:
        deal = board['deal']
        auction_bids = board['auction']

        try:
            state = GameState.initial(deal)
        except Exception:
            skipped += 1
            continue

        board_ok = True
        for bid_str in auction_bids:
            if state.phase != 'bidding':
                break

            # Convert bid string to action ID
            try:
                action_id = bidding_phase.bid_to_action(bid_str)
            except KeyError:
                board_ok = False
                break

            # Check legality
            legal = state.legal_actions()
            if action_id not in legal:
                board_ok = False
                break

            # Record example: observation for current player, target is this bid
            player = state.current_player
            obs = state.observation(player)
            examples.append((obs, action_id))

            # Apply the bid
            state = state.apply_action(action_id)

        if not board_ok:
            skipped += 1

    if skipped > 0:
        logger.info(f"Skipped {skipped}/{len(boards)} boards (illegal/malformed)")
    logger.info(f"Generated {len(examples)} supervised examples")
    return examples


def generate_full_game_examples(
    boards: List[dict],
) -> List[Tuple[dict, int, bool]]:
    """Replay full games to generate (obs, target, is_bid) triples.

    For each board, replays the auction (producing bidding examples) and,
    if play data is available, continues through card play (producing card
    play examples).

    Args:
        boards: list of dicts with 'deal', 'auction', and optional 'play' keys

    Returns:
        List of (observation_dict, target_action_id, is_bid) tuples.
    """
    examples = []
    skipped = 0
    bid_count = 0
    card_count = 0

    for board in boards:
        deal = board['deal']
        auction_bids = board['auction']
        play_data = board.get('play')

        try:
            state = GameState.initial(deal)
        except Exception:
            skipped += 1
            continue

        board_ok = True

        # --- Replay auction ---
        for bid_str in auction_bids:
            if state.phase != 'bidding':
                break

            try:
                action_id = bidding_phase.bid_to_action(bid_str)
            except KeyError:
                board_ok = False
                break

            legal = state.legal_actions()
            if action_id not in legal:
                board_ok = False
                break

            player = state.current_player
            obs = state.observation(player)
            examples.append((obs, action_id, True))
            bid_count += 1

            state = state.apply_action(action_id)

        if not board_ok:
            skipped += 1
            continue

        # --- Replay card play ---
        if play_data is None:
            continue

        if state.phase not in ('opening_lead', 'play'):
            # Auction ended in pass-out or didn't finish
            continue

        for trick_data in play_data:
            # trick_data is [(player_idx, card52), ...] in PBN column order.
            # We need to reorder to match the game state's play order:
            # the current_leader leads first, then clockwise.
            leader = state.current_leader
            if leader is None:
                break

            # Build a map from player_idx to card52 for this trick
            player_to_card = {}
            for player_idx, card52 in trick_data:
                player_to_card[player_idx] = card52

            # Play in game order: leader, leader+1, leader+2, leader+3
            trick_ok = True
            for offset in range(4):
                player = (leader + offset) % 4
                if player not in player_to_card:
                    trick_ok = False
                    break

                card52 = player_to_card[player]

                # Validate legality
                legal = state.legal_actions()
                if card52 not in legal:
                    trick_ok = False
                    break

                obs = state.observation(player)
                examples.append((obs, card52, False))
                card_count += 1

                state = state.apply_action(card52)

            if not trick_ok:
                break

    if skipped > 0:
        logger.info(f"Skipped {skipped}/{len(boards)} boards (illegal/malformed)")
    logger.info(
        f"Generated {len(examples)} examples: "
        f"{bid_count} bidding, {card_count} card play"
    )
    return examples


class PretrainDataset(Dataset):
    """PyTorch Dataset for supervised pre-training (bidding + card play).

    Encodes observations lazily to avoid high memory usage with large datasets.
    """

    def __init__(self, examples, model_config: ModelConfig = None):
        """
        Args:
            examples: list of (obs, target_action_id, is_bid) from
                      generate_full_game_examples(), or legacy 2-tuples from
                      generate_supervised_examples().
            model_config: ModelConfig for encode_observation()
        """
        self.model_config = model_config or ModelConfig()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if len(ex) == 3:
            obs, action_id, is_bid = ex
        else:
            obs, action_id = ex
            is_bid = True
        enc = encode_observation(obs, self.model_config)
        return enc, action_id, is_bid


# Backwards-compatible alias
BiddingDataset = PretrainDataset


def bidding_collate_fn(batch, model_config=None):
    """Custom collate function for BiddingDataset.

    Uses collate_observations() for proper padding of variable-length sequences.

    Args:
        batch: list of (encoded_obs, target_action_id, is_bid) tuples

    Returns:
        (batch_dict, targets_tensor, is_bid_tensor)
    """
    if model_config is None:
        model_config = ModelConfig()

    obs_list = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    is_bid = torch.tensor([item[2] for item in batch], dtype=torch.bool)

    batch_dict = collate_observations(obs_list, model_config)
    return batch_dict, targets, is_bid


def make_collate_fn(model_config: ModelConfig = None):
    """Create a collate function bound to a specific ModelConfig."""
    cfg = model_config or ModelConfig()

    def collate(batch):
        return bidding_collate_fn(batch, cfg)

    return collate
