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
from rlbridge.training.reward import compute_par, compute_pars_batch, compute_reward, assign_rewards
from rlbridge.training.ppo import PPOTrainer

logger = logging.getLogger(__name__)


class SelfPlayTrainer:
    """Self-play RL training loop for bridge."""

    def __init__(self, model_config: ModelConfig = None,
                 training_config: TrainingConfig = None,
                 ben_models=None):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()

        self.model = BridgeModel(self.model_config)
        self.ppo = PPOTrainer(
            self.model, self.training_config, self.model_config
        )
        self.device = self.training_config.device
        self.ben_models = ben_models

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
            t1 = time.time()
            results = self._self_play()
            t_play = time.time() - t1

            # 2. Compute PAR for each deal
            t1 = time.time()
            self._compute_pars(results)
            t_par = time.time() - t1

            # 3. Process trajectories
            t1 = time.time()
            trajectories = self._process_trajectories(results)
            t_traj = time.time() - t1

            # 4. PPO update
            t1 = time.time()
            metrics = self.ppo.update(trajectories)
            t_ppo = time.time() - t1

            elapsed = time.time() - t0

            # Logging
            avg_score = np.mean([r.score_ns for r in results])
            avg_par = np.mean([r.par_ns for r in results if r.par_ns is not None] or [0])
            avg_imp = 0.0
            if any(r.par_ns is not None for r in results):
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
                f"t_play={t_play:.1f}s | "
                f"t_par={t_par:.1f}s | "
                f"t_traj={t_traj:.1f}s | "
                f"t_ppo={t_ppo:.1f}s | "
                f"time={elapsed:.1f}s"
            )

            # 5. Evaluate periodically
            if (iteration + 1) % self.training_config.eval_interval == 0:
                self._evaluate(iteration)

            # 6. Checkpoint
            if (iteration + 1) % self.training_config.checkpoint_interval == 0:
                self._checkpoint(iteration)

    def _self_play(self) -> list:
        """Run self-play games (batched or vs BEN)."""
        if self.ben_models is not None:
            return self._self_play_vs_ben()

        self.model.eval()
        temperature = compute_temperature(self.training_config, self.iteration)
        deals = [Deal.random() for _ in range(self.training_config.games_per_iteration)]
        runner = BatchGameRunner(self.model, self.model_config,
                                 temperature, self.device)
        results = runner.play_games(deals)
        self.model.train()
        return results

    def _self_play_vs_ben(self) -> list:
        """Run games with NN as NS and BEN as EW (sequential)."""
        from rlbridge.engine.ben_agent import BenAgent

        self.model.eval()
        temperature = compute_temperature(self.training_config, self.iteration)
        n_games = self.training_config.games_per_iteration
        results = []

        for i in range(n_games):
            deal = Deal.random()
            agents = [
                NNAgent(self.model, self.model_config,
                        temperature=temperature, device=self.device),  # N
                BenAgent(self.ben_models, deal),                       # E
                NNAgent(self.model, self.model_config,
                        temperature=temperature, device=self.device),  # S
                BenAgent(self.ben_models, deal),                       # W
            ]
            try:
                result = Game(agents, deal).play()
                results.append(result)
            except Exception as e:
                logger.debug(f"BEN game {i} failed: {e}")

        if len(results) < n_games:
            logger.info(f"BEN self-play: {len(results)}/{n_games} games succeeded")

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
        filter_ns = self.ben_models is not None

        for result in results:
            returns = assign_rewards(result)

            for step, ret in zip(result.trajectory, returns):
                # Skip EW steps when training vs BEN — BenAgent returns
                # dummy log_prob=0.0, value=0.0 which would break PPO.
                if filter_ns and step.player not in (0, 2):
                    continue

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
        """Evaluate current model against random baseline using IMP scoring."""
        self.model.eval()
        t0 = time.time()

        rng = np.random.RandomState(42)
        n_games = self.training_config.eval_games

        # Generate all deals upfront and compute PAR in batch
        deals = [Deal.random(rng) for _ in range(n_games)]
        pars = compute_pars_batch(deals)

        nn_imps = []     # IMP score when NN plays NS
        rand_imps = []   # IMP score when Random plays NS

        for deal, par_ns in zip(deals, pars):
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
                imp, _ = compute_reward(result.score_ns, par_ns)
                nn_imps.append(imp)
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
                # Random is NS here; NN is EW so NN's IMP = -imp_ns
                imp_ns, _ = compute_reward(result2.score_ns, par_ns)
                rand_imps.append(imp_ns)
            except Exception:
                pass

        avg_nn = np.mean(nn_imps) if nn_imps else 0.0
        avg_rand = np.mean(rand_imps) if rand_imps else 0.0
        elapsed = time.time() - t0

        logger.info(
            f"EVAL iter {iteration}: "
            f"nn_imp={avg_nn:+.1f} rand_imp={avg_rand:+.1f} "
            f"advantage={avg_nn - avg_rand:+.1f} "
            f"({n_games} games, {elapsed:.1f}s)"
        )

        if self.ben_models is not None:
            self._evaluate_vs_ben(iteration)

        self.model.train()

    def _evaluate_vs_ben(self, iteration: int):
        """Evaluate current model against BEN using paired IMP comparison."""
        from rlbridge.engine.ben_agent import BenAgent
        from compare import get_imps

        t0 = time.time()
        rng = np.random.RandomState(42)
        n_games = self.training_config.eval_games

        deals = [Deal.random(rng) for _ in range(n_games)]
        imp_advantages = []

        for deal in deals:
            # Match 1: NN as NS, BEN as EW
            agents1 = [
                NNAgent(self.model, self.model_config,
                        temperature=0.5, device=self.device),  # N
                BenAgent(self.ben_models, deal),                # E
                NNAgent(self.model, self.model_config,
                        temperature=0.5, device=self.device),  # S
                BenAgent(self.ben_models, deal),                # W
            ]
            try:
                r1 = Game(agents1, deal).play()
                score_ns_1 = r1.score_ns
            except Exception:
                continue

            # Match 2: BEN as NS, NN as EW
            agents2 = [
                BenAgent(self.ben_models, deal),                # N
                NNAgent(self.model, self.model_config,
                        temperature=0.5, device=self.device),  # E
                BenAgent(self.ben_models, deal),                # S
                NNAgent(self.model, self.model_config,
                        temperature=0.5, device=self.device),  # W
            ]
            try:
                r2 = Game(agents2, deal).play()
                score_ns_2 = r2.score_ns
            except Exception:
                continue

            imp = get_imps(score_ns_1, score_ns_2)
            imp_advantages.append(imp)

        avg_imp = np.mean(imp_advantages) if imp_advantages else 0.0
        elapsed = time.time() - t0

        logger.info(
            f"EVAL-BEN iter {iteration}: "
            f"vs_ben={avg_imp:+.2f} IMP/deal "
            f"({len(imp_advantages)} paired deals, {elapsed:.1f}s)"
        )

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
