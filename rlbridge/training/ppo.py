"""PPO implementation for bridge self-play training."""

import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rlbridge.model.config import ModelConfig
from rlbridge.model.network import BridgeModel
from rlbridge.model.encoder import encode_observation, collate_observations
from rlbridge.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class PPOTrainer:
    """Proximal Policy Optimization trainer."""

    def __init__(self, model: BridgeModel, config: TrainingConfig,
                 model_config: ModelConfig = None):
        self.model = model
        self.config = config
        self.model_config = model_config or ModelConfig()
        self.device = config.device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, eps=1e-5
        )

        if config.lr_schedule == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_iterations,
            )
        else:
            self.scheduler = None

        self.model.to(self.device)

    def update(self, trajectories: list) -> dict:
        """Run PPO update on collected trajectories.

        Args:
            trajectories: list of dicts with keys:
                observation: dict (raw observation)
                action: int
                old_log_prob: float
                return_: float (computed return)
                advantage: float
                is_bid: bool

        Returns:
            dict of training metrics
        """
        if not trajectories:
            return {}

        t_prep_start = time.monotonic()

        # Encode all observations and pre-collate into one padded batch
        encoded = [encode_observation(t['observation'], self.model_config)
                    for t in trajectories]
        all_batch = collate_observations(encoded, self.model_config)
        all_batch = {k: v.to(self.device) for k, v in all_batch.items()}

        actions = torch.tensor([t['action'] for t in trajectories],
                               dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t['old_log_prob'] for t in trajectories],
                                     dtype=torch.float32, device=self.device)
        returns = torch.tensor([t['return_'] for t in trajectories],
                               dtype=torch.float32, device=self.device)
        advantages = torch.tensor([t['advantage'] for t in trajectories],
                                  dtype=torch.float32, device=self.device)
        is_bid = torch.tensor([t['is_bid'] for t in trajectories],
                              dtype=torch.bool, device=self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_prep = time.monotonic() - t_prep_start

        n = len(trajectories)
        total_metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
        }
        total_batches = 0
        total_steps_counted = 0
        t_fwd_total = 0.0
        t_bwd_total = 0.0
        epochs_completed = 0
        early_stopped = False

        for epoch in range(self.config.ppo_epochs):
            # Shuffle and create mini-batches
            indices = np.random.permutation(n)

            for start in range(0, n, self.config.batch_size):
                end = min(start + self.config.batch_size, n)
                batch_idx = indices[start:end]

                # Slice pre-collated batch by index
                batch = {k: v[batch_idx] for k, v in all_batch.items()}

                batch_actions = actions[batch_idx]
                batch_old_lp = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_is_bid = is_bid[batch_idx]

                # Forward pass
                t_fwd_start = time.monotonic()
                result = self.model.evaluate_actions(
                    batch, batch_actions, batch_is_bid
                )

                new_log_probs = result['log_prob']
                values = result['value']
                entropy = result['entropy']

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_lp)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                policy_loss = -torch.min(
                    ratio * batch_adv,
                    clipped_ratio * batch_adv,
                ).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                        self.config.value_coef * value_loss +
                        self.config.entropy_coef * entropy_loss)
                t_fwd_total += time.monotonic() - t_fwd_start

                # Backward + optimizer step
                t_bwd_start = time.monotonic()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                t_bwd_total += time.monotonic() - t_bwd_start

                # Metrics
                with torch.no_grad():
                    approx_kl = (batch_old_lp - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()

                bs = end - start
                total_metrics['policy_loss'] += policy_loss.item() * bs
                total_metrics['value_loss'] += value_loss.item() * bs
                total_metrics['entropy'] += entropy.mean().item() * bs
                total_metrics['total_loss'] += loss.item() * bs
                total_metrics['approx_kl'] += approx_kl * bs
                total_metrics['clip_fraction'] += clip_frac * bs
                total_batches += 1
                total_steps_counted += bs

                # KL early stopping
                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    logger.info(
                        "PPO early stopping at epoch %d/%d, batch %d: "
                        "approx_kl=%.4f > target_kl=%.4f",
                        epoch + 1, self.config.ppo_epochs,
                        total_batches, approx_kl, self.config.target_kl,
                    )
                    early_stopped = True
                    break

            epochs_completed = epoch + 1
            if early_stopped:
                break

        # Average metrics over actual steps processed
        for key in total_metrics:
            total_metrics[key] /= max(total_steps_counted, 1)

        if self.scheduler is not None:
            self.scheduler.step()

        total_metrics['lr'] = self.optimizer.param_groups[0]['lr']
        total_metrics['epochs_completed'] = epochs_completed

        # Log profiling summary
        t_total = t_prep + t_fwd_total + t_bwd_total
        logger.info(
            "PPO: %d steps, %d/%d epochs, %d batches | "
            "prep=%.1fs fwd=%.1fs bwd=%.1fs total=%.1fs%s",
            n, epochs_completed, self.config.ppo_epochs, total_batches,
            t_prep, t_fwd_total, t_bwd_total, t_total,
            " (early stopped)" if early_stopped else "",
        )

        return total_metrics
