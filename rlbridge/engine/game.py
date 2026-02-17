"""Game orchestrator — runs a complete bridge game from deal to scoring."""

from typing import Optional

from rlbridge.engine.deal import Deal
from rlbridge.engine.game_state import GameState
from rlbridge.engine.agents import Agent
from rlbridge.engine.experience import ExperienceStep, GameResult


class Game:
    """Runs a complete bridge game with 4 agents."""

    def __init__(self, agents: list, deal: Optional[Deal] = None):
        """
        Args:
            agents: list of 4 Agent instances for N, E, S, W
            deal: optional Deal; if None, generates random
        """
        assert len(agents) == 4
        self.agents = agents
        self.deal = deal or Deal.random()

    def play(self) -> GameResult:
        """Play a complete game, collecting experience at every decision point.

        Returns:
            GameResult with full trajectory
        """
        state = GameState.initial(self.deal)
        trajectory = []

        while not state.is_terminal:
            player = state.current_player
            legal = state.legal_actions()

            if not legal:
                raise RuntimeError(
                    f"No legal actions in phase '{state.phase}' for player {player}"
                )

            # Get the acting agent
            # During play, dummy's cards are played by declarer
            acting_agent_idx = player
            if (state.phase in ('opening_lead', 'play') and
                    state.dummy is not None and
                    player == state.dummy):
                acting_agent_idx = state.declarer

            obs = state.observation(acting_agent_idx)
            action = self.agents[acting_agent_idx].act(obs, legal)
            info = self.agents[acting_agent_idx].get_action_info()

            step = ExperienceStep(
                player=player,
                observation=obs,
                legal_actions=legal,
                action=action,
                action_log_prob=info.get('log_prob', 0.0),
                value_estimate=info.get('value', 0.0),
            )
            trajectory.append(step)

            state = state.apply_action(action)

        # Compute final score
        score_ns = state.score_ns()

        return GameResult(
            deal=self.deal,
            final_state=state,
            trajectory=trajectory,
            score_ns=score_ns,
            par_ns=None,
            contract=state.contract,
            auction=state.auction,
        )
