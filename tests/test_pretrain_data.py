"""Tests for PBN parsing with card play data and full game example generation."""

import os
import sys
import tempfile

import pytest

# Ensure project root and src/ are on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from rlbridge.training.pretrain_data import (
    pbn_card_to_card52,
    _parse_play_section,
    parse_pbn_file,
    generate_full_game_examples,
    generate_supervised_examples,
)


class TestPbnCardToCard52:
    def test_spade_ace(self):
        assert pbn_card_to_card52('SA') == 0  # S=0, A=0

    def test_club_king(self):
        assert pbn_card_to_card52('CK') == 40  # C=3, K=1 -> 3*13+1

    def test_heart_two(self):
        assert pbn_card_to_card52('H2') == 25  # H=1, 2=12 -> 1*13+12

    def test_diamond_ten(self):
        assert pbn_card_to_card52('DT') == 30  # D=2, T=4 -> 2*13+4

    def test_case_insensitive(self):
        assert pbn_card_to_card52('sa') == pbn_card_to_card52('SA')

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            pbn_card_to_card52('SAK')

    def test_invalid_suit(self):
        with pytest.raises(KeyError):
            pbn_card_to_card52('XA')


class TestParsePlaySection:
    def test_basic_play(self):
        lines = [
            'H2\tH4\tHK\tD2',
            'D3\tDA\tDT\tD6',
        ]
        tricks = _parse_play_section('S', lines)
        assert tricks is not None
        assert len(tricks) == 2
        # First trick: S plays H2, W plays H4, N plays HK, E plays D2
        assert tricks[0][0] == (2, pbn_card_to_card52('H2'))  # S=2
        assert tricks[0][1] == (3, pbn_card_to_card52('H4'))  # W=3
        assert tricks[0][2] == (0, pbn_card_to_card52('HK'))  # N=0
        assert tricks[0][3] == (1, pbn_card_to_card52('D2'))  # E=1

    def test_west_starter(self):
        lines = ['CQ\tC3\tC6\tC2']
        tricks = _parse_play_section('W', lines)
        assert tricks is not None
        assert len(tricks) == 1
        assert tricks[0][0] == (3, pbn_card_to_card52('CQ'))  # W=3
        assert tricks[0][1] == (0, pbn_card_to_card52('C3'))  # N=0
        assert tricks[0][2] == (1, pbn_card_to_card52('C6'))  # E=1
        assert tricks[0][3] == (2, pbn_card_to_card52('C2'))  # S=2

    def test_incomplete_trick_skipped(self):
        lines = [
            'H2\tH4\tHK\tD2',
            'D3\tDA\t-',
        ]
        tricks = _parse_play_section('S', lines)
        assert tricks is not None
        assert len(tricks) == 1  # Only the complete trick

    def test_invalid_starter(self):
        assert _parse_play_section('X', ['H2\tH4\tHK\tD2']) is None

    def test_empty_lines(self):
        assert _parse_play_section('S', []) is None


# Minimal PBN content for a board with play data
_BBO_PBN_WITH_PLAY = """\
[Event "Test"]
[Board "1"]
[West "Bot"]
[North "Bot"]
[East "Bot"]
[South "Bot"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:AKQ2.AKQ2.AKQ.A2 JT98.JT98.JT9.KQ 7654.7654.876.J3 3.3.5432.T987654"]
[Declarer "N"]
[Contract "7N"]
[Auction "N"]
2N Pass 3N Pass
Pass Pass
[Play "E"]
SJ S3 SA S7
"""

# Minimal PBN without play data (like BBA files)
_BBA_PBN_NO_PLAY = """\
[Event "Test"]
[Board "1"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:AKQ2.AKQ2.AKQ.A2 JT98.JT98.JT9.KQ 7654.7654.876.J3 3.3.5432.T987654"]
[Auction "N"]
2N Pass 3N Pass
Pass Pass
"""


class TestParsePbnWithPlay:
    def test_play_data_present(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pbn', delete=False) as f:
            f.write(_BBO_PBN_WITH_PLAY)
            f.flush()
            try:
                boards = parse_pbn_file(f.name)
            finally:
                os.unlink(f.name)

        assert len(boards) >= 1
        board = boards[0]
        assert 'play' in board
        assert board['play'] is not None
        assert len(board['play']) >= 1
        # First trick should have 4 cards
        assert len(board['play'][0]) == 4

    def test_no_play_section(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pbn', delete=False) as f:
            f.write(_BBA_PBN_NO_PLAY)
            f.flush()
            try:
                boards = parse_pbn_file(f.name)
            finally:
                os.unlink(f.name)

        assert len(boards) >= 1
        board = boards[0]
        assert board.get('play') is None

    def test_parse_real_bbo_file(self):
        """Parse a real BBO file and verify play data is found."""
        bbo_dir = os.path.join(PROJECT_ROOT, 'Boards', 'BBO')
        if not os.path.isdir(bbo_dir):
            pytest.skip("Boards/BBO directory not found")
        pbn_files = [f for f in os.listdir(bbo_dir) if f.endswith('.pbn')]
        if not pbn_files:
            pytest.skip("No PBN files in Boards/BBO")

        filepath = os.path.join(bbo_dir, pbn_files[0])
        boards = parse_pbn_file(filepath)
        assert len(boards) > 0

        boards_with_play = [b for b in boards if b.get('play') is not None]
        assert len(boards_with_play) > 0, "Expected at least one board with play data"


class TestGenerateFullGameExamples:
    def test_bidding_and_card_play(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pbn', delete=False) as f:
            f.write(_BBO_PBN_WITH_PLAY)
            f.flush()
            try:
                boards = parse_pbn_file(f.name)
            finally:
                os.unlink(f.name)

        examples = generate_full_game_examples(boards)
        assert len(examples) > 0

        bid_examples = [(obs, act, ib) for obs, act, ib in examples if ib]
        card_examples = [(obs, act, ib) for obs, act, ib in examples if not ib]

        # Should have both bidding and card play examples
        assert len(bid_examples) > 0
        assert len(card_examples) > 0

        # Card play targets should be valid card52 indices
        for obs, card52, is_bid in card_examples:
            assert not is_bid
            assert 0 <= card52 < 52

        # Bidding targets should be valid action IDs (0-37)
        for obs, action_id, is_bid in bid_examples:
            assert is_bid
            assert 0 <= action_id < 38

    def test_boards_without_play(self):
        """Boards without play data should only produce is_bid=True examples."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pbn', delete=False) as f:
            f.write(_BBA_PBN_NO_PLAY)
            f.flush()
            try:
                boards = parse_pbn_file(f.name)
            finally:
                os.unlink(f.name)

        examples = generate_full_game_examples(boards)
        # All examples should be bidding
        for obs, action_id, is_bid in examples:
            assert is_bid is True

    def test_real_bbo_files(self):
        """End-to-end: parse real BBO files and generate examples."""
        bbo_dir = os.path.join(PROJECT_ROOT, 'Boards', 'BBO')
        if not os.path.isdir(bbo_dir):
            pytest.skip("Boards/BBO directory not found")

        from rlbridge.training.pretrain_data import load_all_pbn
        boards = load_all_pbn([bbo_dir])
        if not boards:
            pytest.skip("No boards loaded from BBO")

        examples = generate_full_game_examples(boards)
        bid_count = sum(1 for _, _, ib in examples if ib)
        card_count = len(examples) - bid_count

        assert bid_count > 0
        assert card_count > 0, "Expected card play examples from BBO data"
