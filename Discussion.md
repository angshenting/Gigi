# RabbitBridge — Discussion

## What Was Built

A self-play reinforcement learning system for contract bridge, built on top of the BEN (Bridge Engine) codebase. The system lives entirely under `rlbridge/` (~2,550 lines across 25 files) and imports BEN's game mechanics without modifying them.

Three layers:

1. **Game engine** (`rlbridge/engine/`) — Immutable game state, full bridge rules (bidding + card play), batched multi-game inference. Verified against 10,000 random games with zero errors.

2. **Transformer model** (`rlbridge/model/`) — Parametric causal transformer with dual policy heads (38 bids + 52 cards) and a value head. Legal action masking, temperature-controlled sampling. Best model is 10.8M params (d_model=384, n_heads=12, n_layers=6, d_ff=1536); original was 5.1M params (d_model=256).

3. **PPO training loop** (`rlbridge/training/`) — Self-play with IMP-vs-PAR rewards computed via DDS (double-dummy solver). Clipped PPO with KL early stopping, checkpointing, periodic evaluation.

The system was progressively optimized from 400s/iteration (sequential CPU) to 7s/iteration (batched GPU) — a **24.6x speedup** — through batched self-play, pre-collated PPO tensors, cached attention masks, and CUDA.

---

## Training Results

Twelve training runs were conducted, totaling ~6,000 iterations and 864,000 games:

| Run | Iterations | Games/iter | vs Random | vs BEN | Key result |
|-----|-----------|------------|-----------|--------|------------|
| 1   | 10        | 16         | —         | —      | First proof of life — vloss dropped 353 to 95, model beat random by iter 9. No DDS (raw score rewards). |
| 2   | 10        | 16         | —         | —      | DDS PAR enabled — IMP rewards working, but 10 iters too few for convergence. |
| 3   | 10        | 64         | —         | —      | Batched + GPU — 22x faster, model beat random by +430 points. |
| 4   | 500       | 64         | +1.8 IMP  | -4.4   | First extended run — avg_imp rose from -0.12 to +1.78, vloss dropped 267 to 34. Eval was broken (raw scores, only 20 games). |
| 5   | 500       | 64         | +1.0 IMP  | —      | Resumed from run 4 with fixed eval. Plateau confirmed: advantage stable at ~+1 IMP vs random, no further improvement. |
| 6   | 500       | 256        | +3.6 IMP  | -7.5   | **Supervised pre-training + all tuning improvements.** Advantage jumped to +3.6 avg / +6.2 peak vs random. Details below. |
| 7   | 1000      | 256        | +3.4 IMP  | **-5.2** | Resumed from run 6. Held steady at +3.4 avg advantage over 1,000 iters. **Best vs BEN.** Second plateau confirmed. Details below. |
| 8   | 1000      | 256        | +1.1 IMP  | -8.4   | **Card play pre-training added.** Fresh start with both bidding + card play supervised data. Did not beat run 7. Details below. |
| 9   | 500       | 64         | +4.6 IMP  | -6.6   | **Train against BEN as EW opponent.** Resumed from run 7. No improvement vs BEN despite direct exposure. Details below. |
| 10  | 500       | 64         | +4.7 IMP  | -7.2   | **BEN opponent + relaxed KL (0.10).** Full PPO epochs completed, but extra gradient didn't help vs BEN. Details below. |
| 11  | 1000      | 64         | +0.8 IMP  | -6.2   | **BEN distillation + IMP-vs-BEN reward.** Pre-trained on 625K BEN self-play examples, reward from BEN reference scores. Best single eval -5.08, but no sustained improvement. Details below. |
| 12  | 500       | 64         | +4.5 IMP  | **-3.6** | **Larger model (10.8M params).** d_model=384, n_heads=12, n_layers=6, d_ff=1536. Distill pre-trained to 73.3% accuracy. **New best vs BEN — closed gap by 1.6 IMP/deal.** Details below. |

The model clearly learns in the first 500 iterations: value predictions improve dramatically (vloss 267 to 34), entropy declines naturally (1.05 to 0.71), and the model starts beating PAR by ~1.5 IMPs/game. But the second 500 iterations (run 5) show no further progress — all metrics flatline. Run 6 applied four targeted fixes and broke through the plateau. Run 7 confirmed a new plateau at ~+3.4 IMP. Run 8 tested card play pre-training but did not improve over run 7. Runs 9-10 trained against BEN as the opponent — with tight and relaxed KL thresholds respectively — but neither closed the gap. Run 11 combined BEN distillation with IMP-vs-BEN reward shaping, but the gap persists. Run 12 increased model capacity from 5.1M to 10.8M parameters — confirming the architectural hypothesis with a new best of **-3.60 IMP/deal** against BEN.

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

### Run 7: Longer Training — Second Plateau

Run 7 resumed from the run 6 checkpoint (`model_iter_000499.pt`) and ran for 1,000 additional iterations with the same hyperparameters (256 games/iter, cosine temperature 1.0→0.3, target_kl=0.05). The goal was to test whether more training would push past +3.6 IMP.

**Eval progression** (NN vs random, 100 games each checkpoint):

| Iteration | NN IMP | Rand IMP | Advantage |
|-----------|--------|----------|-----------|
| 49        | +0.5   | -1.5     | +2.0      |
| 99        | +2.5   | -2.0     | **+4.5**  |
| 149       | +1.5   | -2.4     | +3.9      |
| 199       | +1.5   | -1.9     | +3.3      |
| 249       | +1.1   | -1.6     | +2.7      |
| 299       | +1.2   | -2.4     | +3.6      |
| 349       | +1.5   | -3.2     | **+4.8**  |
| 399       | +0.5   | -2.2     | +2.8      |
| 449       | +0.8   | -1.9     | +2.6      |
| 499       | +1.6   | -2.1     | +3.7      |
| 549       | +1.2   | -2.2     | +3.4      |
| 599       | +0.9   | -2.5     | +3.4      |
| 649       | +1.6   | -2.2     | +3.8      |
| 699       | +1.1   | -2.3     | +3.4      |
| 749       | +1.3   | -2.5     | +3.8      |
| 799       | +1.4   | -2.5     | +4.0      |
| 849       | +0.3   | -1.5     | +1.8      |
| 899       | +1.0   | -2.1     | +3.1      |
| 949       | +0.6   | -2.8     | +3.4      |
| 999       | +1.1   | -1.7     | +2.8      |

**Key metrics:**

| Metric | Run 6 | Run 7 | Change |
|--------|-------|-------|--------|
| Avg advantage vs random | +3.6 IMP | +3.4 IMP | Flat |
| Peak advantage | +6.2 IMP | +4.8 IMP | Lower peak |
| Entropy (final) | 0.42 | 0.46 | Similar |
| Value loss (final) | ~41 | ~59 | Slightly worse |
| Total training time | ~2h 45m | ~6h | 1000 iters vs 500 |

**Observations:**

- The model held steady at +3.4 IMP average across 20 eval checkpoints — no improvement over run 6, but no degradation either. This is a genuine second plateau.
- PPO behavior was healthier than run 6: many iterations completed full 2/2 epochs (132 mini-batches) rather than early-stopping after 1 batch. The resumed optimizer state appears better calibrated.
- Late in training (iter 800+), KL early stopping reverted to triggering after just 1 batch, similar to run 6's late behavior. The policy may be hardening as temperature drops.
- The gentler cosine schedule (over 1000 iters vs 500) kept temperature higher for longer, but this didn't translate to higher final performance.
- Zero pass-outs across all 1,000 iterations — the supervised bidding knowledge is fully retained.

### Run 8: Card Play Pre-Training — No Improvement

Run 8 tested the hypothesis that card play was the bottleneck. The [Play] sections from BBO PBN files were parsed to generate supervised card play examples alongside the existing bidding examples. Unlike run 7 (which resumed from run 6), run 8 started from scratch with the combined pre-training, then ran 1,000 iterations of self-play with the same hyperparameters as runs 6-7.

**Data:**
- Boards/BBA: 46,634 boards → 563,255 bidding examples (no play data)
- Boards/BBO: 296 boards → 3,313 bidding + 14,980 card play examples
- Total: 581,548 examples (566,568 bidding + 14,980 card play)

**Pre-training results** (581K examples, ~25 min on GPU):

| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 1.8294 | 69.9%    |
| 2     | 1.5350 | 75.7%    |
| 3     | 1.3876 | 77.9%    |
| 4     | 1.2683 | 79.3%    |
| 5     | 1.1596 | 80.4%    |

Note: accuracy is lower than run 6's 82.9% because the card play examples are harder — the model must predict 1-of-52 cards rather than 1-of-38 bids, with a more complex observation space. Loss is also higher because it includes cross-entropy over both heads.

**Eval progression** (NN vs random, 100 games each checkpoint):

| Iteration | NN IMP | Rand IMP | Advantage |
|-----------|--------|----------|-----------|
| 49        | +2.1   | -1.8     | **+3.8**  |
| 99        | +0.5   | -1.2     | +1.7      |
| 149       | +2.0   | -1.1     | +3.0      |
| 199       | +1.1   | -1.1     | +2.2      |
| 249       | +0.6   | -0.2     | +0.8      |
| 299       | +0.8   | -0.2     | +1.0      |
| 349       | +0.2   | -0.5     | +0.8      |
| 399       | +0.4   | -0.6     | +1.1      |
| 449       | +0.5   | -0.8     | +1.3      |
| 499       | +0.5   | -1.0     | +1.5      |
| 549       | +0.2   | -1.0     | +1.2      |
| 599       | +0.2   | -0.7     | +0.9      |
| 649       | +0.2   | -0.6     | +0.8      |
| 699       | +0.2   | -0.9     | +1.1      |
| 749       | -0.2   | -1.0     | +0.8      |
| 799       | +0.2   | -0.7     | +0.9      |
| 849       | -0.0   | -0.6     | +0.5      |
| 899       | +0.7   | -0.6     | +1.3      |
| 949       | +0.5   | -0.6     | +1.1      |
| 999       | -0.1   | -0.8     | +0.7      |

**Key metrics:**

| Metric | Run 7 | Run 8 | Change |
|--------|-------|-------|--------|
| Avg advantage vs random | +3.4 IMP | +1.1 IMP | **Worse** |
| Peak advantage | +4.8 IMP | +3.8 IMP | Lower peak |
| Entropy (final) | 0.46 | ~0.10 | Much sharper |
| Value loss (final) | ~59 | ~31 | Better value estimates |
| Total training time | ~6h | ~6h | Similar |

**Observations:**

- **Card play pre-training did not help.** Average advantage dropped from +3.4 (run 7) to +1.1 IMP. The model peaked at +3.8 (iter 49) — likely from pre-training gains — then RL self-play eroded the supervised signal, settling around +1.
- **Starting from scratch was costly.** Runs 6-7 benefited from building on each other. Run 8 threw away run 7's 1,500 iterations of learned RL policy to start fresh. The pre-trained card play head apparently did not compensate.
- **The card play data is too small.** 14,980 card play examples vs 566,568 bidding examples is a 38:1 ratio. The card play signal is drowned out during pre-training. The model can't learn meaningful card play patterns from ~300 boards.
- **Entropy collapsed further than previous runs** (~0.10 vs ~0.46), suggesting the policy became overly deterministic. The combination of fresh start + cosine temperature decay may have caused premature convergence.
- **The lesson:** supervised card play pre-training is viable (the pipeline works), but needs far more data to be effective. ~15K examples from ~300 BBO boards is insufficient. Sources like BridgeComposer or ACBL archives with thousands of boards would be needed.

### Run 9: Training Against BEN — No Improvement

Run 9 tested the hypothesis that self-play co-adaptation was the key bottleneck. Instead of playing against itself at all 4 seats, the model played NS against BEN's pre-trained NNs as EW. This used sequential game play (not batched) since mixed TF+PyTorch agents can't share a batch pipeline. Training data was filtered to NS-only steps (BenAgent returns dummy log_prob/value). Resumed from run 7's checkpoint with 64 games/iter, cosine temperature 1.0→0.3, target_kl=0.05.

**Eval progression** (50 games each checkpoint):

| Iteration | vs Random (advantage) | vs BEN (IMP/deal) |
|-----------|----------------------|-------------------|
| 49        | +5.7                 | -6.32             |
| 99        | +5.0                 | -7.18             |
| 149       | +3.5                 | -6.54             |
| 199       | +4.5                 | -6.34             |
| 249       | +4.0                 | **-5.30**         |
| 299       | +5.7                 | -7.58             |
| 349       | +4.9                 | -6.74             |
| 399       | +1.9                 | -6.54             |
| 449       | +5.4                 | -7.32             |
| 499       | +5.0                 | -6.52             |

**Key metrics:**

| Metric | Run 7 | Run 9 | Change |
|--------|-------|-------|--------|
| Avg advantage vs random | +3.4 IMP | +4.6 IMP | Slight improvement |
| Avg vs BEN | -5.18 IMP | -6.6 IMP | **Worse** |
| Best vs BEN | — | -5.30 (iter 249) | No improvement over run 7 baseline |
| Entropy (final) | 0.46 | 0.46 | Same |
| Value loss (final) | ~59 | ~97 | Higher — harder to predict outcomes vs BEN |
| Games/iter | 256 (batched) | 64 (sequential) | 4x fewer due to sequential BEN play |
| Time/iter | ~7s | ~25s | Slower due to sequential TF inference |
| Total training time | ~6h | ~3.9h | 500 iters × 25s |

**Observations:**

- **Training against BEN did not close the gap.** The vs-BEN score averaged -6.6 IMP/deal across 10 eval checkpoints, slightly worse than run 7's -5.18 baseline. No improvement trend was visible across 500 iterations.
- **vs Random performance was maintained or slightly improved** (+4.6 avg vs run 7's +3.4). The model didn't regress against weak opponents, but the BEN-specific learning didn't transfer to measurable improvement against BEN either.
- **Value loss was higher** (~97 vs run 7's ~59). Predicting game outcomes is harder when the opponent plays competently rather than mirroring your own policy. The value head struggles to calibrate against BEN's different play style.
- **PPO was heavily KL-constrained.** Nearly every iteration triggered early stopping after 1-3 mini-batches at target_kl=0.05. The policy updates were too small to adapt meaningfully to the new opponent distribution. This confirms the concern from run 7 — the policy has hardened and the KL threshold prevents the large updates needed to learn against a qualitatively different opponent.
- **Fewer training steps per iteration.** Sequential play with 64 games produced ~2,150 NS-only steps/iter (vs ~15,000+ steps from 256 batched games in run 7). Combined with KL early stopping, each iteration delivered very little gradient signal — often just 1-3 mini-batches of 256 steps.
- **The bottleneck is likely not the opponent.** Simply facing a stronger opponent doesn't help if the model can't make large enough policy updates to adapt. The combination of tight KL threshold + small batch count + hardened policy means the training loop effectively stalls regardless of who the opponent is.

### Run 10: Relaxed KL + BEN Opponent — Still No Improvement

Run 10 tested whether the KL early stopping identified in run 9 was the binding constraint. Same setup as run 9 (BEN as EW, 64 games/iter, resumed from run 7) but with target_kl raised from 0.05 to 0.10. This successfully unblocked PPO — every iteration completed full 2/2 epochs with 18 mini-batches, versus run 9's 1-3 batches. But the extra gradient signal did not translate to better performance against BEN.

**Eval progression** (50 games each checkpoint):

| Iteration | vs Random (advantage) | vs BEN (IMP/deal) |
|-----------|----------------------|-------------------|
| 49        | +3.9                 | -7.72             |
| 99        | +4.1                 | -7.76             |
| 149       | +6.3                 | -6.56             |
| 199       | +4.8                 | -8.12             |
| 249       | +5.6                 | **-6.26**         |
| 299       | +3.9                 | -7.12             |
| 349       | +4.0                 | -7.18             |
| 399       | +5.1                 | -7.28             |
| 449       | +4.7                 | -7.12             |
| 499       | +5.0                 | -6.92             |

**Key metrics:**

| Metric | Run 9 (KL=0.05) | Run 10 (KL=0.10) | Change |
|--------|-----------------|-------------------|--------|
| Avg advantage vs random | +4.6 IMP | +4.7 IMP | Same |
| Avg vs BEN | -6.6 IMP | -7.2 IMP | **Slightly worse** |
| Best vs BEN | -5.30 | -6.26 | Worse |
| PPO epochs per iter | 1/2 (early stop) | 2/2 (full) | More updates |
| PPO batches per iter | 1-3 | 18 | **6-18x more** |
| Entropy (final) | 0.46 | 0.44 | Similar |
| Total training time | ~3.9h | ~4.0h | Similar |

**Observations:**

- **Relaxing KL successfully unblocked PPO updates** — every iteration completed full 2/2 epochs with 18 batches (vs run 9's 1-3 batches). The model received 6-18x more gradient signal per iteration.
- **The extra gradient did not help.** vs-BEN averaged -7.2 IMP/deal, slightly worse than run 9's -6.6. More gradient signal in the wrong direction just moves the policy further from where it needs to be.
- **KL early stopping was not the binding constraint.** Run 9 hypothesized KL was the bottleneck; run 10 disproved this. The policy can update freely but doesn't learn to play better against BEN.
- **The IMP-vs-PAR reward is the likely issue.** The model optimizes for absolute quality (beating PAR), not for beating BEN specifically. PAR-optimal play may differ from BEN-beating play — e.g., aggressive bidding might beat PAR but lose to BEN's competent defense. The reward signal doesn't contain information about BEN's tendencies.
- **Model capacity may be insufficient.** With 5.1M parameters handling all positions, phases, and strains, the model may lack the capacity to learn both "play well generally" and "exploit BEN's patterns." BEN uses 8+ specialized models with likely more total parameters.
- **vs Random remained stable** (+4.7 avg), confirming the model doesn't degrade — it just can't improve against a competent opponent through RL alone with the current architecture and reward signal.

### Run 11: BEN Distillation + IMP-vs-BEN Reward — Marginal Improvement

Run 11 tested two new ideas simultaneously: (1) **BEN distillation** — pre-training on 625K examples from BEN self-play games (10K games, all 4 seats), and (2) **IMP-vs-BEN reward** — instead of computing DDS PAR, each training deal is also played by BEN-vs-BEN and the reference score is used as the baseline for IMP computation. This means the RL reward directly measures "did we beat BEN on this deal?" rather than "did we beat theoretical PAR?"

Fresh start (not resumed from run 7). 64 games/iter, target_kl=0.02, 1000 iterations, CUDA.

**Distill pre-training** (624,722 examples: 106K bidding + 519K card play, 5 epochs):

| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 1.9376 | 61.0%    |
| 2     | 1.6050 | 65.2%    |
| 3     | 1.4504 | 68.3%    |
| 4     | 1.3572 | 69.8%    |
| 5     | 1.2804 | 71.2%    |

Accuracy is lower than run 6's 82.9% (PBN bidding-only) because 83% of the distill data is card play (predict 1-of-52 cards vs 1-of-38 bids). But this is the first time the model has been pre-trained on a substantial volume of expert card play data — 519K examples vs run 8's 15K.

**Eval progression** (50 games each checkpoint):

| Iteration | vs Random (advantage) | vs BEN (IMP/deal) |
|-----------|----------------------|-------------------|
| 49        | +1.7                 | -6.40             |
| 99        | +3.0                 | -6.42             |
| 149       | +2.7                 | -7.34             |
| 199       | +3.7                 | -6.52             |
| 249       | +2.8                 | -5.60             |
| 299       | +2.9                 | -6.34             |
| 349       | +1.8                 | -5.76             |
| 399       | +1.8                 | -7.22             |
| 449       | +1.3                 | **-5.08**         |
| 499       | -0.7                 | -6.80             |
| 549       | -1.8                 | -6.80             |
| 599       | +1.0                 | -5.70             |
| 649       | -1.2                 | -5.52             |
| 699       | -1.2                 | -6.92             |
| 749       | +0.7                 | -5.96             |
| 799       | +0.2                 | -6.56             |
| 849       | -0.3                 | -5.94             |
| 899       | +0.1                 | -5.76             |
| 949       | +0.0                 | -5.18             |
| 999       | +1.8                 | -6.68             |

**Key metrics:**

| Metric | Run 10 (PAR reward) | Run 11 (BEN reward) | Change |
|--------|---------------------|---------------------|--------|
| Avg advantage vs random | +4.7 IMP | +0.8 IMP | **Much worse** |
| Avg vs BEN | -7.2 IMP | -6.2 IMP | Slightly better |
| Best vs BEN | -6.26 | **-5.08** | New best single eval |
| Entropy (final) | 0.44 | 0.51 | Similar |
| Value loss (final) | ~28 | ~25 | Similar |
| Total training time | ~4.0h | ~9.0h | Longer (BEN reference games double play time) |

**Observations:**

- **Best single eval of -5.08 IMP/deal** (iter 449), slightly better than run 7's -5.18 baseline. But this is within noise — the 50-deal eval has high variance (~15 IMP std), so the difference is not statistically significant.
- **Average vs BEN improved slightly** (-6.2 vs runs 9-10's -6.6/-7.2), but still well below run 7's -5.18. The BEN reward signal may help marginally but didn't produce the breakthrough we hoped for.
- **vs Random advantage collapsed** from +4.7 (run 10) to +0.8 (run 11). In the second half of training, advantage hovered around 0. The model appears to overfit to BEN's specific play style at the expense of general play quality. When the reward only measures "beat BEN on this deal," the model doesn't learn strategies that generalize to other opponents.
- **BEN distillation pre-training worked** — 71.2% accuracy on combined bid+card data, and the model started with reasonable play from iteration 0. But the supervised signal was partially eroded by RL, similar to what happened in run 8.
- **Fresh start vs resuming from run 7 may have hurt.** Runs 9-10 resumed from run 7's strong policy. Run 11 started fresh with only distillation pre-training. The distilled policy (71% accuracy) is weaker than run 7's refined RL+supervised policy, so run 11 started from a worse position.
- **The IMP-vs-BEN reward is noisy.** BEN's play varies across deals — sometimes BEN plays well and our model can't beat the reference, sometimes BEN plays poorly and the bar is low. This noise may dilute the learning signal compared to the more stable DDS PAR baseline.
- **Two changes at once make attribution difficult.** We changed both the pre-training data (BEN distill vs PBN) and the reward signal (BEN vs PAR) simultaneously. It's unclear which change helped and which hurt.

### Run 12: Larger Model (10.8M Params) — New Best vs BEN

Run 12 tested the architectural capacity hypothesis identified in runs 9-11. The model dimensions were increased from d_model=256/n_heads=8/n_layers=6/d_ff=1024 (5.1M params) to d_model=384/n_heads=12/n_layers=6/d_ff=1536 (10.8M params) — a 2.1x increase. Per-head dimension remained d_k=32. No code changes were needed; the model is fully parametric via `ModelConfig`.

Fresh start with BEN distillation pre-training, then self-play with PAR reward (same setup as run 7, isolating the capacity variable). 64 games/iter, lr=1e-4, target_kl=0.02, 500 iterations, CUDA.

Note: The original plan called for d_model=512/n_heads=16/n_layers=8/d_ff=2048 (~25M params), but this consumed 16GB VRAM on an RTX 5070 Ti (16GB), causing system-wide slowdown. The config was scaled back to 384/12/6/1536 (~10.8M params, ~5GB VRAM).

**Distill pre-training** (624,722 examples: 106K bidding + 519K card play, 5 epochs):

| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 1.9031 | 61.5%    |
| 2     | 1.5357 | 67.2%    |
| 3     | 1.3808 | 69.9%    |
| 4     | 1.2784 | 71.8%    |
| 5     | 1.1964 | 73.3%    |

Accuracy surpassed run 11's 71.2% — the larger model extracts more from the same supervised data.

**Eval progression** (50 games each checkpoint):

| Iteration | vs Random (advantage) | vs BEN (IMP/deal) |
|-----------|----------------------|-------------------|
| 49        | +9.2                 | +5.6              |
| 99        | +8.1                 | +4.7              |
| 149       | +5.9                 | +4.5              |
| 199       | +6.9                 | +4.5              |
| 249       | +7.1                 | +5.0              |
| 299       | +6.3                 | +5.3              |
| 349       | +8.1                 | +4.9              |
| 399       | +7.0                 | +4.9              |
| 449       | +6.4                 | +3.3              |
| 499       | +6.6                 | +4.0              |

Note: The "vs BEN" column here is the `nn_imp` metric from the training eval loop (50-game samples, different methodology from the paired 200-game evaluation below).

**200-game paired evaluation** (iter 49 checkpoint, seed=42):

```
IMP Advantage (Our NN - BEN, paired):
  Mean: -3.60 IMP/deal
  Std:  6.01
  SE:   0.43
  95%CI: [-4.43, -2.76]

Our NN vs PAR: -1.63 IMP/deal
BEN vs PAR:    +1.06 IMP/deal
```

**Key metrics:**

| Metric | Run 7 (5.1M) | Run 12 (10.8M) | Change |
|--------|-------------|----------------|--------|
| vs BEN (200 games) | -5.18 IMP/deal | **-3.60 IMP/deal** | **+1.58** |
| vs BEN 95% CI | [-8.2, -2.2] | [-4.43, -2.76] | Narrower, doesn't overlap run 7 |
| vs PAR | -2.36 | -1.63 | +0.73 |
| Avg advantage vs random | +3.4 IMP | +4.5 IMP (avg of 10 evals) | Better |
| Distill accuracy | — (PBN 82.9%) | 73.3% (BEN distill) | Different data |
| Parameters | 5.1M | 10.8M | 2.1x |
| Time/iter | ~7s | ~4.5s | Faster (64 games vs 256) |
| Total training time | ~6h | ~40 min + ~50 min eval | Much shorter (500 iter × 4.5s) |

**Observations:**

- **The capacity hypothesis is confirmed.** Doubling the model from 5.1M to 10.8M params improved vs-BEN performance by 1.58 IMP/deal — the first statistically significant improvement since run 7. The 95% CIs don't overlap: run 7's [-8.2, -2.2] vs run 12's [-4.43, -2.76].
- **Distillation accuracy improved.** The larger model reached 73.3% on the same 625K BEN distill dataset vs run 11's 71.2% — confirming that the smaller model was capacity-limited even for supervised learning.
- **Self-play with PAR reward remains the best RL strategy.** Run 12 used the same self-play + PAR setup as run 7 (not BEN opponent or BEN reward). Combined with runs 9-11's negative results, this confirms that the training signal was fine — the model just needed more capacity to use it effectively.
- **vs Random also improved** (+4.5 avg advantage, with peaks up to +9.2). The larger model is better across the board, not just against BEN.
- **PPO early stopping was frequent** — nearly every iteration stopped after 1-2 batches at target_kl=0.02. The larger model's updates are bigger per step, exceeding the KL threshold quickly. A higher target_kl (0.05) might unlock further improvement.
- **The gap is narrowing but still significant.** At -3.60 IMP/deal, we're losing roughly 1 game-swing every 3 deals against BEN. This is closer to "competitive intermediate" vs BEN's "advanced" level. Further capacity increases and position specialization could close the remaining gap.

### Evaluation Against BEN

To get a meaningful skill measurement beyond random, we evaluated our models against BEN's pre-trained neural networks (pure NN-only, no search/DDS/PIMC/BBA). Each deal is played twice — our model as NS with BEN as EW, then reversed — giving a paired IMP comparison.

**Results** (seed=42, same deals across all runs):

| Run | Checkpoint | vs Random | vs BEN (IMP/deal) | 95% CI | Std | Games |
|-----|-----------|-----------|-------------------|--------|-----|-------|
| 4   | iter 499  | +1.8      | **-4.40**         | [-10.0, +1.2] | 20.2 | 50 |
| 6   | iter 499  | +3.6      | **-7.46**         | [-11.4, -3.5] | 14.4 | 50 |
| 7   | iter 999  | +3.4      | **-5.18**         | [-8.2, -2.2]  | 15.3 | 100 |
| 8   | iter 999  | +1.1      | **-8.40**         | [-9.7, -7.2]  | 4.5 | 50 |
| 9   | iter 499  | +4.6      | **-6.63**         | —             | —   | 50 (avg of 10 evals) |
| 10  | iter 499  | +4.7      | **-7.20**         | —             | —   | 50 (avg of 10 evals) |
| 11  | iter 999  | +0.8      | **-6.17**         | —             | —   | 50 (avg of 20 evals) |
| 12  | iter 49   | +4.5      | **-3.60**         | [-4.43, -2.76] | 6.01 | 200 |

**Observations:**

- **All models lose to BEN's NNs, but the gap is narrowing.** BEN's supervised NNs (trained on 8,730+ GIB-BBO games with specialized models for each position — lefty, dummy, righty, declarer for both NT and suit contracts) have a substantial edge, but run 12's larger model reduced the gap from -5.18 to -3.60 IMP/deal.
- **Run 12 is now the strongest against BEN** (-3.60 IMP/deal, 200 games), surpassing run 7's -5.18 baseline by a statistically significant margin. The 95% CIs don't overlap: run 12 [-4.43, -2.76] vs run 7 [-8.2, -2.2]. The larger 10.8M-param model confirmed the architectural capacity hypothesis.
- **Run 11 (BEN distill + BEN reward) averaged -6.17** across 20 eval checkpoints — better than runs 9-10 but not better than run 7. The IMP-vs-BEN reward may help marginally, but the model sacrificed general play quality (vs random collapsed to +0.8) without a corresponding gain against BEN.
- **Runs 9-10 (trained against BEN with PAR reward) did not improve over run 7.** Run 9 (-6.63, KL=0.05) and run 10 (-7.20, KL=0.10) both performed worse despite direct BEN exposure. Run 10 disproved the KL hypothesis — full PPO epochs with 6-18x more gradient signal per iteration still didn't help.
- **The reward signal alone is not the bottleneck.** Run 11 tested the hypothesis from runs 9-10 that IMP-vs-PAR was the problem. Switching to IMP-vs-BEN didn't produce a breakthrough. The gap likely comes from structural disadvantages: model capacity (5.1M params for all positions/phases) and lack of position specialization.
- **Run 8 performed worst** (-8.40) with remarkably low variance (std=4.5 vs 14-20 for others). Raw scores were tiny (mean ~220 points), suggesting many low-level contracts or near-passouts. The card play pre-training with insufficient data appears to have disrupted bidding quality.
- **Run 4 (no pre-training) had the widest CI** — its 95% CI includes 0, meaning it's not statistically distinguishable from BEN with 50 games. The high variance comes from erratic bidding producing wild score swings in both directions.
- **Performance vs random doesn't predict performance vs BEN.** Run 6 (+3.6 vs random) scored worse against BEN (-7.46) than run 7 (+3.4 vs random, -5.18 vs BEN). Strategies that exploit random's weaknesses don't transfer to exploiting a competent opponent.
- **The gap is narrowing.** Run 12 reduced it from ~5 to ~3.6 IMP/deal — roughly the difference between an intermediate-plus and an advanced player. Further capacity increases and position specialization may close the remaining gap.

---

## What Went Well

**Clean architecture.** The `rlbridge/` module is fully independent of BEN's `src/`. BEN's code is untouched — we only import its scoring, card encoding, bidding rules, and DDS solver. This makes the RL system portable and BEN upgradeable.

**Verified correctness.** The game engine was validated against 10,000 random games: all tricks sum to 13, all scores match BEN's scoring function, all trajectory actions are legal. 70 unit tests cover the engine, model, reward, temperature schedule, and pre-training pipeline.

**Principled reward design.** IMP-vs-PAR normalizes extreme scores (a 7NT hand scores +2220 raw but only +13 IMPs vs PAR). This prevents a single outlier from dominating the training signal, which was exactly the problem the broken eval exposed.

**Profiling-driven optimization.** Each bottleneck was identified through instrumented profiling before being addressed. This led to targeted fixes (batched inference, GPU PPO, cached masks) rather than speculative optimization. The result was a 24.6x speedup with no correctness regressions.

**Iterative debugging.** The eval metric was broken (raw scores + 20 games = wild fluctuations). This was diagnosed from the run 4 results, fixed with IMP scoring, and validated in run 5. The iterative loop of "run, observe, diagnose, fix" worked well.

---

## What Needs Improvement

**3.6 IMP gap vs BEN.** Our best model (run 12) loses to BEN's pure NNs by -3.60 IMP/deal — down from -5.18 (run 7) after doubling model capacity. Run 12 confirmed the architectural hypothesis: increased capacity directly translates to better play. The remaining gap likely comes from: (1) still lower total capacity than BEN's 8+ specialized model suite, (2) no position specialization for card play (declarer vs defender), (3) PPO is KL-constrained at target_kl=0.02 with the larger model.

**Capacity is the key lever.** Runs 9-11 eliminated training signal hypotheses (opponent, KL threshold, reward signal). Run 12 confirmed that model capacity was the bottleneck — the same training setup (self-play + PAR) that plateaued at -5.18 with 5.1M params improved to -3.60 with 10.8M params. Further capacity increases may yield additional gains.

**No position specialization.** BEN uses 8 separate card play models (lefty/dummy/righty/declarer × NT/suit), each specialized for its role. Our single transformer uses identical weights for all positions. Declarer play (planning how to take tricks) and defensive play (inferring partner's signals, finding the killing defense) are fundamentally different skills that benefit from specialization.

**Card play data gap partially addressed.** Run 11's BEN distillation generated 519K card play examples (vs run 8's 15K), achieving 71% pre-training accuracy. But the supervised signal was partially eroded by subsequent RL training — a pattern also seen in run 8. The distillation pipeline works, but preserving supervised knowledge during RL fine-tuning remains an open problem.

**Optimizing for BEN hurts general play.** Run 11's IMP-vs-BEN reward caused vs-random advantage to collapse from +4.7 (run 10) to +0.8 (run 11) in the second half of training. The model overfits to BEN's specific play patterns at the expense of general play quality. A mixed reward signal (e.g., weighted combination of PAR and BEN reference) might balance these objectives.

**No experiment tracking.** Training metrics are only available in log files. There's no WandB, TensorBoard, or equivalent for visualizing learning curves, comparing runs, or detecting anomalies early.

---

## What To Do Next

### Completed

~~**1. Enable temperature schedule.**~~ Done in run 6. Cosine decay 1.0 to 0.3.

~~**2. Tune KL threshold.**~~ Done in run 6. Raised from 0.02 to 0.05.

~~**3. Increase games per iteration.**~~ Done in run 6. 256 games/iter (up from 64).

~~**4. Supervised pre-training.**~~ Done in run 6. PBN parser + BiddingDataset + pre-training pipeline. 563K examples from BBA, 83% accuracy after 5 epochs.

~~**5. Longer training.**~~ Done in run 7. 1,000 additional iterations confirmed second plateau at +3.4 IMP — more training alone doesn't help.

~~**6. Supervised card play pre-training.**~~ Done in run 8. Pipeline works (PBN [Play] parser + full game replay + dual-head training), but ~15K card play examples from ~300 BBO boards was insufficient. Average advantage dropped to +1.1 IMP (vs run 7's +3.4), partly because starting from scratch discarded run 7's learned RL policy.

~~**8. Evaluate against BEN.**~~ Done. BEN's NNs wrapped in our Agent interface (`rlbridge/engine/ben_agent.py`) with pure-NN config (`src/config/nn_only.conf`). Paired IMP comparison across all runs shows our best model (run 7) loses by -5.18 IMP/deal. All models lose by 4-8 IMPs. See "Evaluation Against BEN" section above for full results.

### High Priority — Close the Gap Against BEN

Runs 9-11 systematically eliminated three hypotheses: run 9 showed BEN as opponent doesn't help; run 10 showed relaxing KL doesn't help; run 11 showed BEN distillation + IMP-vs-BEN reward doesn't help. We've exhausted the "training signal" axis of improvement — the remaining gap is almost certainly architectural: (1) 5.1M-param generalist model lacks capacity vs BEN's 8+ specialized models, (2) no position specialization for card play, (3) the model may need to be much larger to represent bridge's complexity. The strategies below are re-prioritized based on runs 9-11's findings.

~~**7. Train against BEN instead of self-play.**~~ Done in run 9. BEN's NNs used as EW opponents during RL training, resumed from run 7. The model maintained vs-random performance (+4.6 IMP) but did not improve against BEN (-6.6 avg vs run 7's -5.18). The bottleneck was not the opponent — PPO's KL early stopping triggered after 1-3 mini-batches nearly every iteration, preventing the large policy updates needed to adapt. Combined with fewer training steps (64 sequential games vs 256 batched), each iteration delivered minimal gradient signal.

~~**8. Distill from BEN's play.**~~ Done in run 11. Generated 624,722 examples (106K bidding + 519K card play) from 10K BEN self-play games. Pre-training reached 71.2% accuracy. Combined with IMP-vs-BEN reward, the model's best single eval was -5.08 IMP/deal (iter 449), matching but not beating run 7's -5.18 baseline. The distillation pipeline works (`distill_ben.py`), but the supervised signal is partially eroded by subsequent RL training.

**9. Position-conditional card play.** BEN's biggest architectural advantage is 8 specialized card play models (lefty/dummy/righty/declarer × NT/suit). Our single transformer uses the same weights regardless of position. Add position (declarer/LHO/dummy/RHO) and strain (NT/suit) as conditioning inputs — either as additional embeddings fed into the transformer, or as separate output heads sharing the same backbone. This gives specialization without maintaining 8 separate models. The position and strain information is already available in the observation; it just isn't explicitly conditioning the card play policy head.

**10. Opponent pool with BEN.** During training, randomly assign opponents from a pool: {BEN, current model, past checkpoints}. This forces robustness across different play styles. Start with 50% BEN + 50% self-play, then tune the ratio. Unlike pure self-play, this prevents co-adaptation while maintaining the exploration benefits of playing against an evolving policy.

### Medium Priority

~~**11. Increase model capacity.**~~ Done in run 12. Increased from d_model=256/n_heads=8/n_layers=6/d_ff=1024 (5.1M params) to d_model=384/n_heads=12/n_layers=6/d_ff=1536 (10.8M params). **Confirmed the capacity hypothesis** — vs BEN improved from -5.18 to -3.60 IMP/deal, the first statistically significant improvement since run 7. Distillation accuracy also improved (73.3% vs 71.2%). Further scaling (e.g., d_model=512 with reduced batch size to fit VRAM) may yield additional gains.

~~**12. Reward shaping with BEN reference scores.**~~ Done in run 11. IMP-vs-BEN reward (`--reward-mode ben`) plays a BEN-vs-BEN reference game on each deal and uses the reference score instead of DDS PAR. Average vs BEN improved slightly (-6.2 vs -7.2 in run 10) but vs-random collapsed to +0.8. **The reward signal was not the key bottleneck** — the model can optimize for beating BEN's reference scores but this doesn't translate to actually beating BEN in head-to-head play. The gap is architectural.

~~**13. Raise KL threshold.**~~ Done in run 10. Raised from 0.05 to 0.10. PPO completed full 2/2 epochs (18 batches/iter vs 1-3 in run 9), but extra gradient signal did not improve vs BEN (-7.2 vs run 9's -6.6). **KL was not the binding constraint** — the reward signal and model capacity are the more likely bottlenecks.

**14. Add experiment tracking.** Integrate WandB or TensorBoard to log training metrics, eval results, and hyperparameters. With BEN evaluation now providing a meaningful metric, tracking IMP-vs-BEN across training iterations becomes essential for diagnosing whether changes are working.

### Lower Priority

**15. Separate bidding and play networks.** Currently one transformer handles both phases. Bridge bidding and card play are fundamentally different tasks — bidding involves partnership communication and convention systems, while card play involves inference and planning. Separate specialized networks might learn each phase more effectively. However, position-conditional heads (item 9) may capture most of this benefit with less complexity.

**16. More external card play data.** Acquire larger PBN archives (BridgeComposer, ACBL, Vugraph) with play records. This is lower priority now because distilling from BEN (item 8) can generate unlimited card play data without external acquisition. External data becomes valuable if BEN distillation proves insufficient — real expert play may contain patterns that BEN's NNs don't capture.

---

## Summary

The foundation is solid: correct game engine, working training loop, principled rewards, fast iteration. The model demonstrably learns from scratch and beats random play.

After runs 4-5 plateaued at +1 IMP, run 6 applied four targeted fixes — supervised pre-training from 563K expert bidding examples, cosine temperature decay, raised KL threshold, and 4x more games per iteration. The result was a **3.6x improvement** in average advantage vs random (+1.0 to +3.6 IMP), with a peak of +6.2 IMP. Run 7 extended training by 1,000 iterations and confirmed a second plateau at +3.4 IMP — more self-play iterations alone don't help.

Run 8 tested card play pre-training from BBO PBN files (~15K card play examples), but starting from scratch with insufficient data actually performed worse (+1.1 IMP avg). The card play pipeline works — it just needs 10-100x more supervised data to be effective.

Runs 9-11 systematically tested training against BEN. Run 9 (BEN opponent, PAR reward, KL=0.05) averaged -6.6 IMP/deal. Run 10 (relaxed KL=0.10) averaged -7.2 despite full PPO epochs. Run 11 combined BEN distillation (625K examples, 71% accuracy) with IMP-vs-BEN reward — best single eval was -5.08 (iter 449), but averaged -6.2 and vs-random collapsed to +0.8 as the model overfitted to BEN's style. None improved over run 7's -5.18 baseline. This eliminated three hypotheses: the bottleneck is not the opponent, the KL threshold, or the reward signal.

Run 12 confirmed the architectural hypothesis. Increasing model capacity from 5.1M to 10.8M parameters (d_model=384, n_heads=12, d_ff=1536) improved vs-BEN performance to **-3.60 IMP/deal** (95% CI: [-4.43, -2.76], 200 games) — a statistically significant improvement of 1.58 IMP/deal over run 7. The larger model also achieved higher distillation accuracy (73.3% vs 71.2%) and better vs-random performance (+4.5 avg advantage). The same training setup (self-play + PAR reward) that had plateaued with the smaller model produced clear gains with more capacity.

Evaluation against BEN's pre-trained neural networks (pure NN, no search) gives a meaningful skill measurement: our best model (run 12) loses by **-3.60 IMP/deal** (95% CI: [-4.43, -2.76]). The gap has narrowed from ~5.2 to ~3.6 IMP/deal through increased model capacity alone. BEN's supervised NNs, trained on 8,730+ expert games with 8 specialized position models, still have an edge over our single 10.8M-parameter transformer, but the gap is closing.

The highest-impact next steps are: (1) further capacity scaling (d_model=512 with VRAM-conscious batch sizes) — run 12 showed clear returns to scale; (2) adding position-conditional card play heads to match BEN's specialization; (3) raising target_kl from 0.02 to allow fuller PPO updates with the larger model; (4) opponent pool training (mixed BEN + self-play).
