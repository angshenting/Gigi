"""DDS PAR reward computation — IMP vs PAR baseline."""

import sys
import os
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from compare import get_imps
from rlbridge.engine.deal import Deal
from rlbridge.engine.experience import GameResult

logger = logging.getLogger(__name__)

# Module-level DDSolver singleton (reused across compute_par calls)
_solver = None


def _get_solver():
    """Get or create the module-level DDSolver singleton."""
    global _solver
    if _solver is None:
        from ddsolver.ddsolver import DDSolver
        _solver = DDSolver(dds_mode=1, verbose=False)
    return _solver


def compute_par(deal: Deal) -> int:
    """Compute PAR score (NS perspective) for a deal using DDS.

    Returns:
        PAR score for NS, or 0 if DDS fails
    """
    try:
        solver = _get_solver()
        pbn = deal.hand_pbn()
        # Strip the "N:" prefix — calculatepar adds it internally
        hand = pbn[2:]
        vuln = [deal.vuln_ns, deal.vuln_ew]
        par = solver.calculatepar(hand, vuln, print_result=False)
        return par if par is not None else 0
    except Exception:
        return 0


def _compute_par_worker(args):
    """Worker function for multiprocessing PAR computation.

    Each worker process creates its own DDSolver (DDS is not thread-safe
    but separate processes are fine).
    """
    pbn_str, vuln_ns, vuln_ew = args
    try:
        from ddsolver.ddsolver import DDSolver
        solver = DDSolver(dds_mode=1, verbose=False)
        # Strip the "N:" prefix
        hand = pbn_str[2:]
        vuln = [vuln_ns, vuln_ew]
        par = solver.calculatepar(hand, vuln, print_result=False)
        return par if par is not None else 0
    except Exception:
        return 0


def compute_pars_batch(deals: list, max_workers: int = 1) -> list:
    """Compute PAR scores for multiple deals.

    Args:
        deals: list of Deal objects
        max_workers: 1 for sequential (reuses singleton solver),
                     >1 for parallel with ProcessPoolExecutor

    Returns:
        list of int PAR scores (NS perspective)
    """
    if not deals:
        return []

    if max_workers <= 1:
        # Sequential — reuse module-level solver
        return [compute_par(deal) for deal in deals]

    # Parallel — serialize to picklable tuples, use spawn to avoid
    # ctypes/DDS fork issues
    args = [(deal.hand_pbn(), deal.vuln_ns, deal.vuln_ew) for deal in deals]
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        results = list(executor.map(_compute_par_worker, args))
    return results


def compute_reward(score_ns: int, par_ns: int) -> tuple:
    """Compute IMP-based rewards for both sides.

    Args:
        score_ns: actual NS score
        par_ns: PAR NS score

    Returns:
        (reward_ns, reward_ew) — IMP difference from PAR
    """
    reward_ns = get_imps(score_ns, par_ns)
    reward_ew = -reward_ns
    return reward_ns, reward_ew


def assign_rewards(result: GameResult) -> list:
    """Assign terminal rewards to each step in the trajectory.

    Since gamma=1.0 and reward is terminal, each step gets the
    player's final reward as the return.

    Args:
        result: GameResult with par_ns filled in

    Returns:
        list of float returns, one per trajectory step
    """
    if result.par_ns is None:
        # No PAR available, use raw score as proxy
        reward_ns = float(result.score_ns) / 100.0  # normalize
        reward_ew = -reward_ns
    else:
        reward_ns, reward_ew = compute_reward(result.score_ns, result.par_ns)
        reward_ns = float(reward_ns)
        reward_ew = float(reward_ew)

    returns = []
    for step in result.trajectory:
        if step.player in (0, 2):  # NS
            returns.append(reward_ns)
        else:  # EW
            returns.append(reward_ew)

    return returns
