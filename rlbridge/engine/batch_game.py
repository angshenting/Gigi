"""Batched game runner — runs N bridge games with batched NN inference."""

import torch
import logging

from rlbridge.engine.deal import Deal
from rlbridge.engine.game_state import GameState
from rlbridge.engine.experience import ExperienceStep, GameResult
from rlbridge.model.config import ModelConfig
from rlbridge.model.network import BridgeModel
from rlbridge.model.encoder import encode_observation, collate_observations

logger = logging.getLogger(__name__)


class BatchGameRunner:
    """Runs multiple bridge games simultaneously with batched model inference.

    Instead of creating NNAgent objects and running games sequentially,
    this collects observations from all active games and makes one batched
    forward pass per step.
    """

    def __init__(self, model: BridgeModel, model_config: ModelConfig,
                 temperature: float = 1.0, device: str = 'cpu'):
        self.model = model
        self.model_config = model_config
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def play_games(self, deals: list) -> list:
        """Play N games simultaneously with batched inference.

        Args:
            deals: list of Deal objects

        Returns:
            list of GameResult, one per deal
        """
        N = len(deals)
        if N == 0:
            return []

        # Initialize all games
        states = [GameState.initial(deal) for deal in deals]
        trajectories = [[] for _ in range(N)]
        active = [True] * N

        while any(active):
            # Collect observations from all active, non-terminal games
            batch_indices = []
            observations = []
            legal_actions_list = []
            players = []
            acting_agents = []

            for i in range(N):
                if not active[i]:
                    continue

                state = states[i]
                if state.is_terminal:
                    active[i] = False
                    continue

                player = state.current_player
                legal = state.legal_actions()

                if not legal:
                    logger.warning(f"Game {i}: no legal actions in phase "
                                   f"'{state.phase}' for player {player}")
                    active[i] = False
                    continue

                # Dummy delegation: during play, dummy's cards are played by declarer
                acting_agent_idx = player
                if (state.phase in ('opening_lead', 'play') and
                        state.dummy is not None and
                        player == state.dummy):
                    acting_agent_idx = state.declarer

                obs = state.observation(acting_agent_idx)

                batch_indices.append(i)
                observations.append(obs)
                legal_actions_list.append(legal)
                players.append(player)
                acting_agents.append(acting_agent_idx)

            if not batch_indices:
                break

            # Batch inference
            encoded = [encode_observation(obs, self.model_config)
                       for obs in observations]
            batch = collate_observations(encoded, self.model_config)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            result = self.model.get_action_and_value(batch, self.temperature)

            actions = result['action'].cpu()
            log_probs = result['log_prob'].cpu()
            values = result['value'].cpu()

            # Apply actions to each game
            for j, game_idx in enumerate(batch_indices):
                action = actions[j].item()
                legal = legal_actions_list[j]

                # Verify legality; fallback re-sample if illegal
                if action not in legal:
                    action = self._resample_legal(
                        batch, j, observations[j], legal
                    )

                log_prob = log_probs[j].item()
                value = values[j].item()

                step = ExperienceStep(
                    player=players[j],
                    observation=observations[j],
                    legal_actions=legal,
                    action=action,
                    action_log_prob=log_prob,
                    value_estimate=value,
                )
                trajectories[game_idx].append(step)

                states[game_idx] = states[game_idx].apply_action(action)

                if states[game_idx].is_terminal:
                    active[game_idx] = False

        # Build GameResults
        results = []
        for i in range(N):
            state = states[i]
            if state.phase == 'passed_out':
                score_ns = 0
            else:
                score_ns = state.score_ns()

            results.append(GameResult(
                deal=deals[i],
                final_state=state,
                trajectory=trajectories[i],
                score_ns=score_ns,
                par_ns=None,
                contract=state.contract,
                auction=state.auction,
            ))

        return results

    def _resample_legal(self, batch: dict, idx: int,
                        obs: dict, legal: list) -> int:
        """Re-sample from model logits restricted to legal actions."""
        bid_logits, card_logits, _ = self.model.forward(batch)

        if obs['phase'] == 'bidding':
            logits = bid_logits[idx]
        else:
            logits = card_logits[idx]

        mask = torch.full_like(logits, float('-inf'))
        for a in legal:
            mask[a] = logits[a]

        dist = torch.distributions.Categorical(logits=mask / self.temperature)
        return dist.sample().item()
