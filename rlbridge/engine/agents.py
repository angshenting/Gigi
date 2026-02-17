"""Agent interface and basic implementations."""

from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    """Base class for bridge-playing agents."""

    @abstractmethod
    def act(self, observation: dict, legal_actions: list) -> int:
        """Choose an action given an observation and legal actions.

        Args:
            observation: imperfect-info observation dict from GameState
            legal_actions: list of legal action IDs

        Returns:
            Chosen action ID
        """
        ...

    def get_action_info(self) -> dict:
        """Return extra info about the last action (log_prob, value, etc.)."""
        return {'log_prob': 0.0, 'value': 0.0}


class RandomAgent(Agent):
    """Agent that plays uniformly at random from legal actions."""

    def __init__(self, rng: np.random.RandomState = None):
        self.rng = rng or np.random.RandomState()

    def act(self, observation: dict, legal_actions: list) -> int:
        return legal_actions[self.rng.randint(len(legal_actions))]
