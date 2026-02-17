"""Main self-play training loop."""

import os
import time
import logging

import torch
import numpy as np

from rlbridge.engine.deal import Deal
from rlbridge.engine.game import Game
from rlbridge.engine.batch_game import BatchGameRunner
from rlbridge.engine.agents import RandomAgent
from rlbridge.engine.experience import GameResult
from rlbridge.model.config import ModelConfig
from rlbridge.model.network import BridgeModel
from rlbridge.model.nn_agent import NNAgent
from rlbridge.training.config import TrainingConfig, compute_temperature
from rlbridge.training.reward import compute_par, compute_pars_batch, assign_rewards
from rlbridge.training.ppo import PPOTrainer

logger = logging.getLogger(__name__)


class SelfPlayTrainer:
    """Self-play RL training loop for bridge."""

    def __init__(self, model_config: ModelConfig = None,
                 training_config: TrainingConfig = None):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()

        self.model = BridgeModel(self.model_config)
        self.ppo = PPOTrainer(
            self.model, self.training_config, self.model_config
        )
        self.device = self.training_config.device

        self.iteration = 0
        self.best_eval_score = float('-inf')

    def run(self):
        """Main training loop."""
        logger.info("Starting self-play training")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        for iteration in range(self.training_config.num_iterations):
            self.iteration = iteration
            t0 = time.time()

            # 1. Self-play
            results = self._self_play()

            # 2. Compute PAR for each deal
            self._compute_pars(results)

            # 3. Process trajectories
            trajectories = self._process_trajectories(results)

            # 4. PPO update
            metrics = self.ppo.update(trajectories)

            elapsed = time.time() - t0

            # Logging
            avg_score = np.mean([r.score_ns for r in results])
            avg_par = np.mean([r.par_ns for r in results if r.par_ns is not None] or [0])
            avg_imp = 0.0
            if any(r.par_ns is not None for r in results):
                from rlbridge.training.reward import compute_reward
                imps = [compute_reward(r.score_ns, r.par_ns)[0]
                        for r in results if r.par_ns is not None]
                avg_imp = np.mean(imps) if imps else 0.0

            n_passed = sum(1 for r in results if r.contract is None)
            temperature = compute_temperature(self.training_config, iteration)

            logger.info(
                f"Iter {iteration:5d} | "
                f"games={len(results)} | "
                f"passed_out={n_passed} | "
                f"avg_score={avg_score:.0f} | "
                f"avg_par={avg_par:.0f} | "
                f"avg_imp={avg_imp:.1f} | "
                f"ploss={metrics.get('policy_loss', 0):.4f} | "
                f"vloss={metrics.get('value_loss', 0):.4f} | "
                f"ent={metrics.get('entropy', 0):.4f} | "
                f"temp={temperature:.3f} | "
                f"time={elapsed:.1f}s"
            )

            # 5. Evaluate periodically
            if (iteration + 1) % self.training_config.eval_interval == 0:
                self._evaluate(iteration)

            # 6. Checkpoint
            if (iteration + 1) % self.training_config.checkpoint_interval == 0:
                self._checkpoint(iteration)

    def _self_play(self) -> list:
        """Run self-play games with batched inference."""
        self.model.eval()
        temperature = compute_temperature(self.training_config, self.iteration)
        deals = [Deal.random() for _ in range(self.training_config.games_per_iteration)]
        runner = BatchGameRunner(self.model, self.model_config,
                                 temperature, self.device)
        results = runner.play_games(deals)
        self.model.train()
        return results

    def _compute_pars(self, results: list):
        """Compute PAR scores for all deals using batch DDS."""
        t0 = time.time()
        deals = [r.deal for r in results]
        pars = compute_pars_batch(deals)
        for result, par in zip(results, pars):
            result.par_ns = par
        logger.debug(f"PAR computation: {len(deals)} deals in {time.time() - t0:.2f}s")

    def _process_trajectories(self, results: list) -> list:
        """Convert game results into PPO training data."""
        all_trajectories = []

        for result in results:
            returns = assign_rewards(result)

            for step, ret in zip(result.trajectory, returns):
                advantage = ret - step.value_estimate
                is_bid = step.observation['phase'] == 'bidding'

                all_trajectories.append({
                    'observation': step.observation,
                    'action': step.action,
                    'old_log_prob': step.action_log_prob,
                    'return_': ret,
                    'advantage': advantage,
                    'is_bid': is_bid,
                })

        return all_trajectories

    def _evaluate(self, iteration: int):
        """Evaluate current model against random baseline."""
        self.model.eval()
        nn_scores = []
        random_scores = []

        rng = np.random.RandomState(42)

        for _ in range(self.training_config.eval_games):
            deal = Deal.random(rng)

            # NN plays NS, Random plays EW
            agents = [
                NNAgent(self.model, self.model_config,
                        temperature=0.5, device=self.device),  # N
                RandomAgent(rng),                                # E
                NNAgent(self.model, self.model_config,
                        temperature=0.5, device=self.device),  # S
                RandomAgent(rng),                                # W
            ]
            game = Game(agents, deal)
            try:
                result = game.play()
                nn_scores.append(result.score_ns)
            except Exception:
                pass

            # Random plays NS, NN plays EW
            agents2 = [
                RandomAgent(rng),                                # N
                NNAgent(self.model, self.model_config,
                        temperature=0.5, device=self.device),  # E
                RandomAgent(rng),                                # S
                NNAgent(self.model, self.model_config,
                        temperature=0.5, device=self.device),  # W
            ]
            game2 = Game(agents2, deal)
            try:
                result2 = game2.play()
                random_scores.append(result2.score_ns)
            except Exception:
                pass

        avg_nn = np.mean(nn_scores) if nn_scores else 0
        avg_rand = np.mean(random_scores) if random_scores else 0

        logger.info(
            f"EVAL iter {iteration}: "
            f"NN_as_NS={avg_nn:.0f} | Random_as_NS={avg_rand:.0f} | "
            f"advantage={avg_nn - avg_rand:.0f}"
        )

        self.model.train()

    def _checkpoint(self, iteration: int):
        """Save model checkpoint."""
        ckpt_dir = self.training_config.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        path = os.path.join(ckpt_dir, f'model_iter_{iteration:06d}.pt')
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
        }, path)
        logger.info(f"Checkpoint saved: {path}")
