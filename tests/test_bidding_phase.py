"""Tests for auction/bidding mechanics."""

from rlbridge.engine import bidding_phase


def test_action_bid_mapping():
    """PASS=0, X=1, XX=2, 1C=3, ..., 7N=37."""
    assert bidding_phase.action_to_bid(0) == 'PASS'
    assert bidding_phase.action_to_bid(1) == 'X'
    assert bidding_phase.action_to_bid(2) == 'XX'
    assert bidding_phase.action_to_bid(3) == '1C'
    assert bidding_phase.action_to_bid(37) == '7N'


def test_bid_action_mapping():
    assert bidding_phase.bid_to_action('PASS') == 0
    assert bidding_phase.bid_to_action('X') == 1
    assert bidding_phase.bid_to_action('1C') == 3
    assert bidding_phase.bid_to_action('7N') == 37


def test_roundtrip():
    """action_to_bid and bid_to_action should be inverses."""
    for action_id in range(38):
        bid = bidding_phase.action_to_bid(action_id)
        assert bidding_phase.bid_to_action(bid) == action_id


def test_initial_legal_bids():
    """At the start, all suit bids + PASS should be legal, no X/XX."""
    auction = ()  # dealer=0, North bids first
    legal = bidding_phase.legal_bids(auction)
    assert 0 in legal   # PASS
    assert 1 not in legal  # X not legal at start
    assert 2 not in legal  # XX not legal at start
    assert 3 in legal   # 1C
    assert 37 in legal  # 7N


def test_pass_out_detection():
    """Four PASSes should end the auction."""
    auction = ('PASS', 'PASS', 'PASS', 'PASS')
    assert bidding_phase.is_over(auction)
    assert bidding_phase.get_result(auction) is None


def test_simple_contract():
    """1C - PASS - PASS - PASS should produce a 1C contract."""
    auction = ('1C', 'PASS', 'PASS', 'PASS')
    assert bidding_phase.is_over(auction)
    result = bidding_phase.get_result(auction)
    assert result is not None
    contract, declarer_i, strain_i, dummy_i = result
    # Contract should contain '1C' and declarer direction
    assert '1C' in contract


def test_not_over_after_one_bid():
    """Auction with just one bid is not over."""
    auction = ('1C',)
    assert not bidding_phase.is_over(auction)


def test_apply_bid():
    """apply_bid should append the bid to the auction."""
    auction = ()
    new = bidding_phase.apply_bid(auction, 3)  # 1C
    assert new[-1] == '1C'
