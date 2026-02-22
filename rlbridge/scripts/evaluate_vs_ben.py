#!/usr/bin/env python3
"""Evaluate our RL model against BEN's pre-trained neural networks.

Usage:
    python rlbridge/scripts/evaluate_vs_ben.py \
        --checkpoint checkpoints_run7/model_iter_000999.pt \
        --games 100 --device cpu
"""

import argparse
import sys
import os
import logging
import time

# Force TensorFlow to CPU — avoids CUDA errors on unsupported GPU architectures
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import torch

from configparser import ConfigParser

from rlbridge.engine.deal import Deal
from rlbridge.engine.game import Game
from rlbridge.engine.agents import RandomAgent
from rlbridge.engine.ben_agent import BenAgent
from rlbridge.model.config import ModelConfig
from rlbridge.model.network import BridgeModel
from rlbridge.model.nn_agent import NNAgent
from rlbridge.training.reward import compute_pars_batch, compute_reward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RL model vs BEN neural networks'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to our model checkpoint')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of deals to play')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deal generation')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for our NN agent sampling')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for our model (cpu or cuda)')
    parser.add_argument('--ben-conf', type=str, default=None,
                        help='Path to BEN config (default: src/config/nn_only.conf)')
    parser.add_argument('--compute-par', action='store_true',
                        help='Compute PAR scores via DDS for IMP baselines')
    return parser.parse_args()


def load_ben_models(conf_path, base_path):
    """Load BEN's models from config."""
    from nn.models_tf2 import Models
    conf = ConfigParser()
    conf.read(conf_path)
    models = Models.from_conf(conf, base_path=base_path, verbose=True)
    return models


def load_our_model(checkpoint_path, device):
    """Load our RL model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', ModelConfig())
    model = BridgeModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def play_game(agents, deal):
    """Play one game, returning (score_ns, contract) or None on failure."""
    try:
        result = Game(agents, deal).play()
        return result.score_ns, result.contract or 'passed_out'
    except Exception as e:
        logger.warning(f"Game failed: {e}")
        return None


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # Paths
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
    repo_root = os.path.abspath(repo_root)
    src_dir = os.path.join(repo_root, 'src')

    if args.ben_conf:
        conf_path = args.ben_conf
    else:
        conf_path = os.path.join(src_dir, 'config', 'nn_only.conf')

    logger.info(f"Loading BEN models from {conf_path}")
    ben_models = load_ben_models(conf_path, base_path=repo_root)

    logger.info(f"Loading our model from {args.checkpoint}")
    our_model, our_config = load_our_model(args.checkpoint, args.device)

    # Generate all deals up front (so both sides see the same deals)
    deals = [Deal.random(rng) for _ in range(args.games)]

    # Compute PAR if requested
    par_scores = None
    if args.compute_par:
        logger.info("Computing PAR scores...")
        par_scores = compute_pars_batch(deals)

    # Results storage
    nn_ns_scores = []     # our NN as NS, BEN as EW
    nn_ew_scores = []     # BEN as NS, our NN as EW (store NS scores)
    ben_vs_rand = []      # BEN as NS, Random as EW
    nn_vs_rand = []       # our NN as NS, Random as EW
    nn_ns_contracts = []
    nn_ew_contracts = []

    t_start = time.time()

    for i, deal in enumerate(deals):
        # --- Match 1: Our NN as NS, BEN as EW ---
        agents_nn_ns = [
            NNAgent(our_model, our_config, temperature=args.temperature,
                    device=args.device),                    # N
            BenAgent(ben_models, deal),                     # E
            NNAgent(our_model, our_config, temperature=args.temperature,
                    device=args.device),                    # S
            BenAgent(ben_models, deal),                     # W
        ]
        r1 = play_game(agents_nn_ns, deal)
        if r1 is not None:
            nn_ns_scores.append(r1[0])
            nn_ns_contracts.append(r1[1])

        # --- Match 2: BEN as NS, Our NN as EW ---
        agents_ben_ns = [
            BenAgent(ben_models, deal),                     # N
            NNAgent(our_model, our_config, temperature=args.temperature,
                    device=args.device),                    # E
            BenAgent(ben_models, deal),                     # S
            NNAgent(our_model, our_config, temperature=args.temperature,
                    device=args.device),                    # W
        ]
        r2 = play_game(agents_ben_ns, deal)
        if r2 is not None:
            nn_ew_scores.append(r2[0])
            nn_ew_contracts.append(r2[1])

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            games_done = i + 1
            rate = elapsed / games_done
            logger.info(
                f"[{games_done}/{args.games}] "
                f"{elapsed:.0f}s elapsed, {rate:.1f}s/deal"
            )

    # ------------------------------------------------------------------
    # Compute IMP scores per deal (paired comparison)
    # ------------------------------------------------------------------
    # For each deal played both ways, the IMP advantage for our NN is:
    #   IMP(nn_as_ns_score vs ben_as_ns_score)
    # This is a standard "two-table" IMP comparison.

    from compare import get_imps

    imp_advantages = []
    n_paired = min(len(nn_ns_scores), len(nn_ew_scores))
    for j in range(n_paired):
        # nn_ns_scores[j] = NS score when our NN plays NS
        # nn_ew_scores[j] = NS score when BEN plays NS
        # Our NN's "total" is nn_ns_scores[j] - nn_ew_scores[j]
        imp = get_imps(nn_ns_scores[j], nn_ew_scores[j])
        imp_advantages.append(imp)

    # PAR-based IMPs
    imp_vs_par_nn = []
    imp_vs_par_ben = []
    if par_scores is not None:
        for j in range(min(len(nn_ns_scores), len(par_scores))):
            imp_nn, _ = compute_reward(nn_ns_scores[j], par_scores[j])
            imp_vs_par_nn.append(imp_nn)
        for j in range(min(len(nn_ew_scores), len(par_scores))):
            # nn_ew_scores[j] is NS score when BEN plays NS
            imp_ben, _ = compute_reward(nn_ew_scores[j], par_scores[j])
            imp_vs_par_ben.append(imp_ben)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Evaluation: Our NN vs BEN  ({args.games} deals, seed={args.seed})")
    print(f"{'=' * 70}")

    if nn_ns_scores:
        print(f"\n  Our NN as NS (BEN as EW):")
        print(f"    Raw NS score: mean={np.mean(nn_ns_scores):>8.0f}  "
              f"std={np.std(nn_ns_scores):>8.0f}  "
              f"({len(nn_ns_scores)} games)")

    if nn_ew_scores:
        print(f"\n  BEN as NS (Our NN as EW):")
        print(f"    Raw NS score: mean={np.mean(nn_ew_scores):>8.0f}  "
              f"std={np.std(nn_ew_scores):>8.0f}  "
              f"({len(nn_ew_scores)} games)")

    if imp_advantages:
        mean_imp = np.mean(imp_advantages)
        std_imp = np.std(imp_advantages)
        se_imp = std_imp / np.sqrt(len(imp_advantages))
        print(f"\n  ** IMP Advantage (Our NN - BEN, paired): **")
        print(f"    Mean: {mean_imp:>+.2f} IMP/deal")
        print(f"    Std:  {std_imp:>.2f}")
        print(f"    SE:   {se_imp:>.2f}")
        print(f"    95%CI: [{mean_imp - 1.96*se_imp:>+.2f}, "
              f"{mean_imp + 1.96*se_imp:>+.2f}]")
        print(f"    ({n_paired} paired deals)")

    if imp_vs_par_nn:
        print(f"\n  Our NN vs PAR: {np.mean(imp_vs_par_nn):>+.2f} IMP/deal")
    if imp_vs_par_ben:
        print(f"  BEN vs PAR:    {np.mean(imp_vs_par_ben):>+.2f} IMP/deal")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/max(1,args.games):.1f}s/deal)")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
