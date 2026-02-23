#!/usr/bin/env python3
"""Generate supervised training data from BEN self-play.

Runs BEN at all 4 seats for N games and saves (observation, action, is_bid)
tuples as pickle -- the exact format PretrainDataset expects.

Usage:
    python rlbridge/scripts/distill_ben.py \
        --games 10000 --output ben_distill.pkl
"""

import argparse
import logging
import os
import pickle
import sys
import time

# Force TensorFlow to CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate supervised data from BEN self-play'
    )
    parser.add_argument('--games', type=int, default=10000,
                        help='Number of games to play (default: 10000)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle path')
    parser.add_argument('--ben-conf', type=str, default=None,
                        help='Path to BEN config (default: src/config/nn_only.conf)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deal generation')
    return parser.parse_args()


def main():
    args = parse_args()

    from configparser import ConfigParser
    from nn.models_tf2 import Models

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    src_dir = os.path.join(repo_root, 'src')

    conf_path = args.ben_conf or os.path.join(src_dir, 'config', 'nn_only.conf')
    conf = ConfigParser()
    conf.read(conf_path)
    ben_models = Models.from_conf(conf, base_path=repo_root, verbose=True)
    logger.info(f"Loaded BEN models from {conf_path}")

    from rlbridge.engine.deal import Deal
    from rlbridge.engine.game import Game
    from rlbridge.engine.ben_agent import BenAgent

    rng = np.random.RandomState(args.seed)
    examples = []
    n_success = 0
    n_fail = 0
    t_start = time.time()

    for i in range(args.games):
        deal = Deal.random(rng)
        agents = [BenAgent(ben_models, deal) for _ in range(4)]

        try:
            result = Game(agents, deal).play()
        except Exception as e:
            n_fail += 1
            logger.debug(f"Game {i} failed: {e}")
            continue

        n_success += 1
        for step in result.trajectory:
            obs = step.observation
            action = step.action
            is_bid = obs['phase'] == 'bidding'
            examples.append((obs, action, is_bid))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            bid_count = sum(1 for _, _, ib in examples if ib)
            card_count = len(examples) - bid_count
            logger.info(
                f"[{i + 1}/{args.games}] "
                f"success={n_success} fail={n_fail} "
                f"examples={len(examples)} (bid={bid_count} card={card_count}) "
                f"{elapsed:.0f}s"
            )

    bid_count = sum(1 for _, _, ib in examples if ib)
    card_count = len(examples) - bid_count
    elapsed = time.time() - t_start

    logger.info(
        f"Done: {n_success}/{args.games} games, "
        f"{len(examples)} examples (bid={bid_count} card={card_count}), "
        f"{elapsed:.0f}s"
    )

    with open(args.output, 'wb') as f:
        pickle.dump(examples, f)
    logger.info(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
