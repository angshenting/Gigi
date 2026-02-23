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
    parser.add_argument('--ppo-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--target-kl', type=float, default=0.02,
                        help='KL divergence threshold for early stopping (0 to disable)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--temperature-start', type=float, default=1.0)
    parser.add_argument('--temperature-end', type=float, default=0.3)
    parser.add_argument('--temperature-schedule', type=str, default='constant',
                        choices=['constant', 'linear', 'cosine', 'exponential'])
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

    # BEN opponent
    parser.add_argument('--ben-opponent', action='store_true',
                        help='Train against BEN NNs as EW opponent')
    parser.add_argument('--ben-conf', type=str, default=None,
                        help='Path to BEN config (default: src/config/nn_only.conf)')

    # Supervised pre-training
    parser.add_argument('--pretrain-data', type=str, default=None,
                        help='Comma-separated paths to directories containing PBN files for pre-training')
    parser.add_argument('--pretrain-epochs', type=int, default=5,
                        help='Number of supervised pre-training epochs')
    parser.add_argument('--pretrain-lr', type=float, default=1e-4,
                        help='Learning rate for pre-training')
    parser.add_argument('--pretrain-batch-size', type=int, default=64,
                        help='Batch size for pre-training')

    # BEN distillation data
    parser.add_argument('--distill-data', type=str, default=None,
                        help='Path to pickle of (obs, action, is_bid) from distill_ben.py')

    # Reward mode
    parser.add_argument('--reward-mode', type=str, default='par',
                        choices=['par', 'ben'],
                        help='Reward signal: par (DDS PAR) or ben (BEN self-play reference)')

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
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        temperature_schedule=args.temperature_schedule,
        entropy_coef=args.entropy_coef,
        target_kl=args.target_kl if args.target_kl != 0 else None,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    if args.device is not None:
        tc_kwargs['device'] = args.device
    tc_kwargs['reward_mode'] = args.reward_mode
    training_config = TrainingConfig(**tc_kwargs)

    if args.reward_mode == 'ben' and not args.ben_opponent:
        parser = None  # already parsed, just raise
        raise SystemExit("ERROR: --reward-mode ben requires --ben-opponent")

    # Load BEN models if training against BEN
    ben_models = None
    if args.ben_opponent:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')  # TF on CPU only; doesn't affect PyTorch GPU

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        src_dir = os.path.join(repo_root, 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from configparser import ConfigParser
        from nn.models_tf2 import Models

        conf_path = args.ben_conf or os.path.join(src_dir, 'config', 'nn_only.conf')
        conf = ConfigParser()
        conf.read(conf_path)
        ben_models = Models.from_conf(conf, base_path=repo_root, verbose=True)
        logging.info(f"Loaded BEN models from {conf_path}")

    trainer = SelfPlayTrainer(model_config, training_config, ben_models=ben_models)

    if args.resume:
        import torch
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.iteration = checkpoint.get('iteration', 0)
        logging.info(f"Resumed from {args.resume} at iteration {trainer.iteration}")

    # Supervised pre-training (only if not resuming)
    if args.pretrain_data and not args.resume:
        from torch.utils.data import DataLoader
        from rlbridge.training.pretrain_data import (
            load_all_pbn, generate_full_game_examples,
            PretrainDataset, make_collate_fn,
        )
        from rlbridge.training.supervised import SupervisedTrainer

        logging.info("=== Supervised Pre-Training ===")
        data_dirs = [d.strip() for d in args.pretrain_data.split(',')]
        boards = load_all_pbn(data_dirs)
        logging.info(f"Loaded {len(boards)} boards from {data_dirs}")

        examples = generate_full_game_examples(boards)
        bid_count = sum(1 for _, _, ib in examples if ib)
        card_count = len(examples) - bid_count
        logging.info(f"Generated {len(examples)} examples: {bid_count} bidding, {card_count} card play")

        if examples:
            dataset = PretrainDataset(examples, model_config)
            collate_fn = make_collate_fn(model_config)
            dataloader = DataLoader(
                dataset,
                batch_size=args.pretrain_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
            )

            device = training_config.device
            sup_trainer = SupervisedTrainer(
                trainer.model, lr=args.pretrain_lr, device=device,
            )

            for epoch in range(args.pretrain_epochs):
                metrics = sup_trainer.train_epoch(dataloader)
                logging.info(
                    f"Pre-train epoch {epoch + 1}/{args.pretrain_epochs}: "
                    f"loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f}"
                )

            logging.info("=== Pre-Training Complete ===")
        else:
            logging.warning("No valid examples generated, skipping pre-training")

    # Distill pre-training from BEN self-play data (only if not resuming)
    if args.distill_data and not args.resume:
        import pickle
        from torch.utils.data import DataLoader
        from rlbridge.training.pretrain_data import PretrainDataset, make_collate_fn
        from rlbridge.training.supervised import SupervisedTrainer

        logging.info("=== Distill Pre-Training ===")
        with open(args.distill_data, 'rb') as f:
            examples = pickle.load(f)

        bid_count = sum(1 for _, _, ib in examples if ib)
        card_count = len(examples) - bid_count
        logging.info(f"Loaded {len(examples)} examples: {bid_count} bidding, {card_count} card play")

        if examples:
            dataset = PretrainDataset(examples, model_config)
            collate_fn = make_collate_fn(model_config)
            dataloader = DataLoader(
                dataset,
                batch_size=args.pretrain_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
            )

            device = training_config.device
            sup_trainer = SupervisedTrainer(
                trainer.model, lr=args.pretrain_lr, device=device,
            )

            for epoch in range(args.pretrain_epochs):
                metrics = sup_trainer.train_epoch(dataloader)
                logging.info(
                    f"Distill epoch {epoch + 1}/{args.pretrain_epochs}: "
                    f"loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f}"
                )

            logging.info("=== Distill Pre-Training Complete ===")
        else:
            logging.warning("No examples in distill data, skipping")

    trainer.run()


if __name__ == '__main__':
    main()
