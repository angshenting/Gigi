"""Training configuration."""

from dataclasses import dataclass, field

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
    ppo_epochs: int = 4
    batch_size: int = 256

    # Learning rate
    lr: float = 3e-4
    lr_schedule: str = 'cosine'  # 'cosine' or 'constant'
    warmup_steps: int = 100

    # Temperature
    temperature: float = 1.0

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
