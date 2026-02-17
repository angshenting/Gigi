from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ExperienceStep:
    """One decision point in a game."""
    player: int                  # 0-3 (N,E,S,W)
    observation: dict            # imperfect-info observation
    legal_actions: list          # legal action IDs
    action: int                  # chosen action
    action_log_prob: float = 0.0  # log probability (0.0 for non-NN agents)
    value_estimate: float = 0.0   # value estimate (0.0 for non-NN agents)


@dataclass
class GameResult:
    """Complete result of a single game."""
    deal: object                 # Deal
    final_state: object          # GameState
    trajectory: list             # list[ExperienceStep]
    score_ns: int                # final NS score
    par_ns: Optional[int] = None  # PAR score from DDS (filled later)
    contract: Optional[str] = None
    auction: tuple = ()
