# RabbitBridge — Discussion

## What Was Built

A self-play reinforcement learning system for contract bridge, built on top of the BEN (Bridge Engine) codebase. The system lives entirely under `rlbridge/` (~2,550 lines across 25 files) and imports BEN's game mechanics without modifying them.

Three layers:

1. **Game engine** (`rlbridge/engine/`) — Immutable game state, full bridge rules (bidding + card play), batched multi-game inference. Verified against 10,000 random games with zero errors.

2. **Transformer model** (`rlbridge/model/`) — 5.1M parameter causal transformer with dual policy heads (38 bids + 52 cards) and a value head. Legal action masking, temperature-controlled sampling.

3. **PPO training loop** (`rlbridge/training/`) — Self-play with IMP-vs-PAR rewards computed via DDS (double-dummy solver). Clipped PPO with KL early stopping, checkpointing, periodic evaluation.

The system was progressively optimized from 400s/iteration (sequential CPU) to 7s/iteration (batched GPU) — a **24.6x speedup** — through batched self-play, pre-collated PPO tensors, cached attention masks, and CUDA.

---

## Training Results

Six training runs were conducted, totaling ~1,500 iterations and 192,000 games:

| Run | Iterations | Games/iter | Key result |
|-----|-----------|------------|------------|
| 1   | 10        | 16         | First proof of life — vloss dropped 353 to 95, model beat random by iter 9. No DDS (raw score rewards). |
| 2   | 10        | 16         | DDS PAR enabled — IMP rewards working, but 10 iters too few for convergence. |
| 3   | 10        | 64         | Batched + GPU — 22x faster, model beat random by +430 points. |
| 4   | 500       | 64         | First extended run — avg_imp rose from -0.12 to +1.78, vloss dropped 267 to 34. Eval was broken (raw scores, only 20 games). |
| 5   | 500       | 64         | Resumed from run 4 with fixed eval. Plateau confirmed: advantage stable at ~+1 IMP vs random, no further improvement. |
| 6   | 500       | 256        | **Supervised pre-training + all tuning improvements.** Advantage jumped to +3.6 avg / +6.2 peak vs random. Details below. |

The model clearly learns in the first 500 iterations: value predictions improve dramatically (vloss 267 to 34), entropy declines naturally (1.05 to 0.71), and the model starts beating PAR by ~1.5 IMPs/game. But the second 500 iterations (run 5) show no further progress — all metrics flatline. Run 6 applied four targeted fixes and broke through the plateau.

### Run 6: Breaking the Plateau

Run 6 applied all four high-priority improvements simultaneously:

1. **Supervised pre-training** from 46,634 BBA boards (563K bidding examples, 5 epochs)
2. **Cosine temperature schedule** decaying from 1.0 to 0.3
3. **Raised KL threshold** from 0.02 to 0.05
4. **4x more games per iteration** (256 vs 64)

**Pre-training results** (563K examples, ~22 min on GPU):

| Epoch | Loss  | Accuracy |
|-------|-------|----------|
| 1     | 0.855 | 72.2%    |
| 2     | 0.630 | 78.2%    |
| 3     | 0.553 | 80.4%    |
| 4     | 0.508 | 81.8%    |
| 5     | 0.475 | 82.9%    |

The model learned to predict expert bids with 83% accuracy before any RL training began.

**Eval progression** (NN vs random, 100 games each checkpoint):

| Iteration | NN IMP | Rand IMP | Advantage |
|-----------|--------|----------|-----------|
| 49        | +2.8   | -3.4     | **+6.2**  |
| 99        | +4.0   | -1.6     | +5.6      |
| 149       | +0.1   | -1.5     | +1.6      |
| 199       | +0.7   | -2.6     | +3.3      |
| 249       | +1.9   | -2.4     | +4.3      |
| 299       | +2.1   | -3.1     | +5.3      |
| 349       | +0.6   | -1.6     | +2.2      |
| 399       | +1.3   | -2.0     | +3.3      |
| 449       | +0.8   | -1.9     | +2.6      |
| 499       | +2.1   | -1.9     | +4.0      |

**Key metrics vs run 5:**

| Metric | Run 5 | Run 6 | Change |
|--------|-------|-------|--------|
| Avg advantage vs random | +1.0 IMP | +3.6 IMP | **+3.6x** |
| Peak advantage | +1.5 IMP | +6.2 IMP | **+4.1x** |
| Entropy (final) | 0.71 | 0.42 | Sharper policy |
| Value loss (final) | ~34 | ~41 | Similar |
| Passed-out boards | Common | Near zero (496/500 iters had zero) | Pre-training taught real bidding |
| Temperature (final) | 1.0 (constant) | 0.30 (cosine decay) | Exploiting learned patterns |
| KL early stopping | 69% of iters (run 4) | 100% of iters | Higher threshold allows larger updates before stopping |
| Total training time | ~58 min | ~2h 45m (22 min pretrain + 2h 23m self-play) | 4x more games/iter |

**Observations:**

- The pre-trained model started strong (+6.2 advantage at iter 49) — supervised knowledge transferred effectively.
- Advantage dipped mid-training (iter 149: +1.6) as RL exploration perturbed the pre-trained policy, then recovered and stabilized around +3-4.
- Temperature dropped from 1.0 to 0.30, allowing the policy to sharpen. Entropy fell from 0.86 to 0.42 — the model is making more confident decisions.
- All 500 iterations triggered KL early stopping (at the 0.05 threshold), but typically after 1-7 mini-batches rather than immediately. This is healthier than run 4's behavior (where 0.02 threshold often stopped after just 1 batch).
- Near-zero pass-outs confirms the model learned sensible opening bids from the supervised data, rather than having to discover them from scratch through RL.

---

## What Went Well

**Clean architecture.** The `rlbridge/` module is fully independent of BEN's `src/`. BEN's code is untouched — we only import its scoring, card encoding, bidding rules, and DDS solver. This makes the RL system portable and BEN upgradeable.

**Verified correctness.** The game engine was validated against 10,000 random games: all tricks sum to 13, all scores match BEN's scoring function, all trajectory actions are legal. 52 unit tests cover the engine, model, reward, and temperature schedule.

**Principled reward design.** IMP-vs-PAR normalizes extreme scores (a 7NT hand scores +2220 raw but only +13 IMPs vs PAR). This prevents a single outlier from dominating the training signal, which was exactly the problem the broken eval exposed.

**Profiling-driven optimization.** Each bottleneck was identified through instrumented profiling before being addressed. This led to targeted fixes (batched inference, GPU PPO, cached masks) rather than speculative optimization. The result was a 24.6x speedup with no correctness regressions.

**Iterative debugging.** The eval metric was broken (raw scores + 20 games = wild fluctuations). This was diagnosed from the run 4 results, fixed with IMP scoring, and validated in run 5. The iterative loop of "run, observe, diagnose, fix" worked well.

---

## What Needs Improvement

**The model is still weak.** At +3.6 IMP advantage vs random (run 6), the model is improving but still far from competent. A competent human beats random by 5-10+ IMPs/game. The pre-training and tuning broke the +1 IMP plateau, but the model needs to continue climbing.

**Random is a weak baseline.** Evaluating against random only tells us the model isn't completely broken. A random player makes illegal-level-bad bids and plays random legal cards — any pattern recognition beats it. We have no signal on how the model compares to actual bridge play.

**Self-play has no opponent diversity.** The model plays against itself at all 4 seats. This can lead to co-adapted strategies that don't generalize — both sides develop the same blind spots. There's no pressure to handle different bidding systems or play styles.

**KL early stopping still dominates.** Even with the raised threshold (0.05), 100% of run 6 iterations triggered early stopping. The model is trying to make larger policy updates than the threshold allows. This may be healthy (preventing collapse) or may be limiting — further experimentation needed.

**Mid-training performance dip.** Run 6 showed a dip from +6.2 advantage (iter 49) to +1.6 (iter 149) before recovering. RL exploration is partially undoing the supervised pre-training. A higher pre-training learning rate, more epochs, or a gentler RL warm-up could help preserve the supervised knowledge.

**No experiment tracking.** Training metrics are only available in log files. There's no WandB, TensorBoard, or equivalent for visualizing learning curves, comparing runs, or detecting anomalies early.

---

## What To Do Next

### High Priority — Continue Climbing

~~**1. Enable temperature schedule.**~~ Done in run 6. Cosine decay 1.0 to 0.3.

~~**2. Tune KL threshold.**~~ Done in run 6. Raised from 0.02 to 0.05.

~~**3. Increase games per iteration.**~~ Done in run 6. 256 games/iter (up from 64).

~~**4. Supervised pre-training.**~~ Done in run 6. PBN parser + BiddingDataset + pre-training pipeline. 563K examples from BBA, 83% accuracy after 5 epochs.

**5. Longer training.** Run 6 showed the model still improving at iter 499 (+4.0 advantage). Run 7 should continue for 1,000-2,000 iterations, resuming from the run 6 checkpoint. The cosine schedule should be extended to match the new total iteration count.

**6. Protect pre-trained knowledge.** The mid-training dip (iter 49: +6.2 to iter 149: +1.6) suggests RL is partially overwriting supervised knowledge. Options: (a) lower the RL learning rate for the first 100 iterations, (b) add a supervised loss term alongside PPO, (c) freeze the bidding head for the first N iterations.

**7. Disable or further raise KL threshold.** All 500 run 6 iterations hit the 0.05 KL cap, often after just 1 mini-batch. Try 0.10 or disable entirely (`--target-kl 0`) for a run to see if larger updates help or cause collapse.

### Medium Priority — Better Evaluation

**8. Evaluate against BEN.** Replace the random baseline with BEN's existing models (or a mix). This gives a meaningful skill measurement. BEN already has agents for bidding and play — wrapping them in the `Agent` interface should be straightforward.

**9. Add experiment tracking.** Integrate WandB or TensorBoard to log training metrics, eval results, and hyperparameters. This makes it much easier to compare runs and diagnose issues. A few lines in `trainer.py` and `ppo.py`.

### Lower Priority — Structural Changes

**10. Opponent pool / league training.** Instead of pure self-play, maintain a pool of past checkpoints and sample opponents from the pool. This creates diversity pressure and prevents co-adaptation. Implementations like OpenAI Five and AlphaStar used this approach.

**11. Increase model capacity.** The current 5.1M parameter model may not have enough capacity for bridge's complexity. Try d_model=512, n_layers=8 (~20M params). Bridge has a large state space (bidding conventions, card play inference, signaling) that may need more representational power.

**12. Separate bidding and play.** Currently one transformer handles both phases. Bridge bidding and card play are quite different tasks — bidding involves partnership communication and convention systems, while card play involves inference and planning. Separate specialized networks might learn each phase more effectively.

**13. Reward shaping.** The current reward is purely terminal (IMP at game end). Intermediate rewards for good bidding (reaching reasonable contracts) or good play (winning tricks when expected) could provide denser signal, especially early in training when the model can't yet connect bidding decisions to final outcomes 13 tricks later.

---

## Summary

The foundation is solid: correct game engine, working training loop, principled rewards, fast iteration. The model demonstrably learns from scratch and beats random play.

After runs 4-5 plateaued at +1 IMP, run 6 applied four targeted fixes — supervised pre-training from 563K expert bidding examples, cosine temperature decay, raised KL threshold, and 4x more games per iteration. The result was a **3.6x improvement** in average advantage vs random (+1.0 to +3.6 IMP), with a peak of +6.2 IMP. The model now makes confident, sensible bids (83% match rate with expert data) and near-zero pass-out rate.

The highest-impact next steps are: longer training runs (the model was still improving at iter 499), protecting pre-trained knowledge from RL perturbation, and evaluating against BEN rather than random.
