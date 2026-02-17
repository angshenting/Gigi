#!/usr/bin/env python3
"""Evaluation harness — compare model against random baseline."""

import argparse
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch

from rlbridge.engine.deal import Deal
from rlbridge.engine.game import Game
from rlbridge.engine.agents import RandomAgent
from rlbridge.model.config import ModelConfig
from rlbridge.model.network import BridgeModel
from rlbridge.model.nn_agent import NNAgent
from rlbridge.training.reward import compute_par, compute_reward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate bridge model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--compute-par', action='store_true',
                        help='Compute PAR scores via DDS')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # Load model
    checkpoint = torch.load(args.model, map_location=args.device)
    config = checkpoint.get('model_config', ModelConfig())
    model = BridgeModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    nn_as_ns_scores = []
    nn_as_ew_scores = []
    rand_vs_rand_scores = []
    par_scores = []
    imp_vs_par = []

    for i in range(args.num_games):
        deal = Deal.random(rng)

        # NN as NS vs Random as EW
        agents_nn_ns = [
            NNAgent(model, config, temperature=args.temperature,
                    device=args.device),  # N
            RandomAgent(rng),              # E
            NNAgent(model, config, temperature=args.temperature,
                    device=args.device),  # S
            RandomAgent(rng),              # W
        ]
        try:
            result_nn_ns = Game(agents_nn_ns, deal).play()
            nn_as_ns_scores.append(result_nn_ns.score_ns)
        except Exception:
            pass

        # Random as NS vs NN as EW
        agents_nn_ew = [
            RandomAgent(rng),              # N
            NNAgent(model, config, temperature=args.temperature,
                    device=args.device),  # E
            RandomAgent(rng),              # S
            NNAgent(model, config, temperature=args.temperature,
                    device=args.device),  # W
        ]
        try:
            result_nn_ew = Game(agents_nn_ew, deal).play()
            nn_as_ew_scores.append(result_nn_ew.score_ns)
        except Exception:
            pass

        # Random vs Random
        agents_rand = [RandomAgent(rng) for _ in range(4)]
        try:
            result_rand = Game(agents_rand, deal).play()
            rand_vs_rand_scores.append(result_rand.score_ns)
        except Exception:
            pass

        # PAR
        if args.compute_par:
            par = compute_par(deal)
            par_scores.append(par)
            if nn_as_ns_scores:
                r, _ = compute_reward(nn_as_ns_scores[-1], par)
                imp_vs_par.append(r)

        if (i + 1) % 50 == 0:
            logger.info(f"Completed {i+1}/{args.num_games} games")

    # Report
    print(f"\n{'='*60}")
    print(f"Evaluation: {args.num_games} deals")
    print(f"{'='*60}")

    if nn_as_ns_scores:
        print(f"NN as NS:     mean={np.mean(nn_as_ns_scores):>8.0f}  "
              f"std={np.std(nn_as_ns_scores):>8.0f}")
    if nn_as_ew_scores:
        print(f"NN as EW:     mean={np.mean(nn_as_ew_scores):>8.0f}  "
              f"std={np.std(nn_as_ew_scores):>8.0f}")
    if rand_vs_rand_scores:
        print(f"Random v Rand:mean={np.mean(rand_vs_rand_scores):>8.0f}  "
              f"std={np.std(rand_vs_rand_scores):>8.0f}")

    if nn_as_ns_scores and nn_as_ew_scores:
        # NN advantage: NN_as_NS score - NN_as_EW_NS score
        # Higher NN_as_NS and lower NN_as_EW (NS perspective) = NN is better
        nn_advantage = np.mean(nn_as_ns_scores) - np.mean(nn_as_ew_scores)
        print(f"\nNN advantage: {nn_advantage:.0f} (raw score diff)")

    if imp_vs_par:
        print(f"NN vs PAR:    {np.mean(imp_vs_par):.2f} IMPs/deal")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()
