"""Neural network agent wrapping BridgeModel for self-play."""

import torch
import numpy as np

from rlbridge.engine.agents import Agent
from rlbridge.model.config import ModelConfig
from rlbridge.model.encoder import encode_observation, collate_observations
from rlbridge.model.network import BridgeModel


class NNAgent(Agent):
    """Agent that uses a BridgeModel to select actions."""

    def __init__(self, model: BridgeModel, config: ModelConfig = None,
                 temperature: float = 1.0, device: str = 'cpu'):
        self.model = model
        self.config = config or ModelConfig()
        self.temperature = temperature
        self.device = device
        self._last_log_prob = 0.0
        self._last_value = 0.0

    def act(self, observation: dict, legal_actions: list) -> int:
        """Choose an action using the neural network."""
        encoded = encode_observation(observation, self.config)
        batch = collate_observations([encoded], self.config)

        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            result = self.model.get_action_and_value(batch, self.temperature)

        action = result['action'][0].item()
        self._last_log_prob = result['log_prob'][0].item()
        self._last_value = result['value'][0].item()

        # Verify the action is legal
        if action not in legal_actions:
            # Fallback: sample from legal actions using the model's logits
            bid_logits, card_logits, _ = self.model.forward(batch)
            if observation['phase'] == 'bidding':
                logits = bid_logits[0]
            else:
                logits = card_logits[0]

            # Mask to legal actions only
            mask = torch.full_like(logits, float('-inf'))
            for a in legal_actions:
                mask[a] = logits[a]

            dist = torch.distributions.Categorical(logits=mask / self.temperature)
            action = dist.sample().item()
            self._last_log_prob = dist.log_prob(
                torch.tensor(action, device=self.device)
            ).item()

        return action

    def get_action_info(self) -> dict:
        return {
            'log_prob': self._last_log_prob,
            'value': self._last_value,
        }
