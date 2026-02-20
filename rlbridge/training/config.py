"""Training configuration."""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


def _default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainingConfig:
    # Self-play
    games_per_iteration: int = 64
    num_iterations: int = 10000

    # PPO
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2
    batch_size: int = 256
    target_kl: Optional[float] = 0.02

    # Learning rate
    lr: float = 3e-4
    lr_schedule: str = 'cosine'  # 'cosine' or 'constant'
    warmup_steps: int = 100

    # Temperature
    temperature: float = 1.0
    temperature_start: float = 1.0
    temperature_end: float = 0.3
    temperature_schedule: str = 'constant'  # 'constant', 'linear', 'cosine', 'exponential'

    # Evaluation
    eval_interval: int = 50
    eval_games: int = 100

    # Checkpointing
    checkpoint_interval: int = 100
    checkpoint_dir: str = 'checkpoints'

    # Reward
    gamma: float = 1.0  # single terminal reward, no discounting

    # Device (auto-detects CUDA)
    device: str = field(default_factory=_default_device)


def compute_temperature(config: TrainingConfig, iteration: int) -> float:
    """Compute temperature for a given iteration based on the schedule.

    Args:
        config: TrainingConfig with schedule parameters
        iteration: current iteration (0-based)

    Returns:
        temperature value for this iteration
    """
    schedule = config.temperature_schedule

    if schedule == 'constant':
        return config.temperature

    t_start = config.temperature_start
    t_end = config.temperature_end
    total = max(config.num_iterations - 1, 1)
    progress = min(iteration / total, 1.0)

    if schedule == 'linear':
        return t_start + (t_end - t_start) * progress
    elif schedule == 'cosine':
        return t_end + (t_start - t_end) * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == 'exponential':
        # Exponential decay: t_start * (t_end/t_start)^progress
        if t_start <= 0:
            return t_end
        ratio = t_end / t_start
        if ratio <= 0:
            return t_end
        return t_start * (ratio ** progress)
    else:
        return config.temperature
