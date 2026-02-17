"""Model configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    # Vocabulary sizes
    bid_vocab: int = 40       # PAD_START(0), PAD_END(1), PASS(2), X(3), XX(4), 1C(5)..7N(39)
    card_vocab: int = 52      # card52 encoding
    player_vocab: int = 4     # relative: 0=me, 1=LHO, 2=partner, 3=RHO

    # Input dimensions
    hand_dim: int = 52
    dummy_dim: int = 52
    context_dim: int = 8      # vuln_us, vuln_them, dealer_rel[4], phase[2]

    # Output dimensions
    num_bid_actions: int = 38  # PASS=0..7N=37
    num_card_actions: int = 52

    # Sequence limits
    max_bid_seq: int = 80     # max auction length
    max_card_seq: int = 52    # max cards played
    max_seq_len: int = 135    # 3 special + max_bid + max_card

    # Value head
    value_hidden: int = 1024
