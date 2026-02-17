#!/usr/bin/env python3
"""Run self-play games and print results."""

import argparse
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from rlbridge.engine.deal import Deal
from rlbridge.engine.game import Game
from rlbridge.engine.agents import RandomAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Run bridge self-play games')
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (uses random agents if None)')
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # Create agents
    if args.model:
        import torch
        from rlbridge.model.config import ModelConfig
        from rlbridge.model.network import BridgeModel
        from rlbridge.model.nn_agent import NNAgent

        checkpoint = torch.load(args.model, map_location='cpu')
        config = checkpoint.get('model_config', ModelConfig())
        model = BridgeModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        agents = [NNAgent(model, config, temperature=0.5) for _ in range(4)]
    else:
        agents = [RandomAgent(rng) for _ in range(4)]

    scores = []
    contracts = []
    passed_out = 0
    errors = 0

    t0 = time.time()

    for i in range(args.num_games):
        deal = Deal.random(rng)
        game = Game(agents, deal)

        try:
            result = game.play()
            scores.append(result.score_ns)

            if result.contract is None:
                passed_out += 1
            else:
                contracts.append(result.contract)

            if args.verbose:
                print(f"Game {i+1}: contract={result.contract or 'PASS'} "
                      f"score_ns={result.score_ns} "
                      f"auction={'-'.join(a for a in result.auction if a != 'PAD_START')}")

        except Exception as e:
            errors += 1
            if args.verbose:
                print(f"Game {i+1}: ERROR - {e}")

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Results: {args.num_games} games in {elapsed:.1f}s "
          f"({args.num_games/elapsed:.0f} games/sec)")
    print(f"Errors: {errors}")
    print(f"Passed out: {passed_out}")

    if scores:
        print(f"NS Score - mean: {np.mean(scores):.0f}, "
              f"median: {np.median(scores):.0f}, "
              f"std: {np.std(scores):.0f}, "
              f"min: {np.min(scores)}, max: {np.max(scores)}")


if __name__ == '__main__':
    main()
