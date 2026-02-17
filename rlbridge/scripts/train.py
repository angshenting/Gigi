#!/usr/bin/env python3
"""Training entry point for RabbitBridge self-play RL."""

import argparse
import logging
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rlbridge.model.config import ModelConfig
from rlbridge.training.config import TrainingConfig
from rlbridge.training.trainer import SelfPlayTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='RabbitBridge Self-Play Training')

    # Model config
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training config
    parser.add_argument('--games-per-iter', type=int, default=64)
    parser.add_argument('--num-iterations', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--entropy-coef', type=float, default=0.01)

    # Evaluation & checkpointing
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--eval-games', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device (default: auto-detect CUDA)')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    )

    model_config = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    tc_kwargs = dict(
        games_per_iteration=args.games_per_iter,
        num_iterations=args.num_iterations,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        entropy_coef=args.entropy_coef,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    if args.device is not None:
        tc_kwargs['device'] = args.device
    training_config = TrainingConfig(**tc_kwargs)

    trainer = SelfPlayTrainer(model_config, training_config)

    if args.resume:
        import torch
        checkpoint = torch.load(args.resume, map_location=args.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.iteration = checkpoint.get('iteration', 0)
        logging.info(f"Resumed from {args.resume} at iteration {trainer.iteration}")

    trainer.run()


if __name__ == '__main__':
    main()
