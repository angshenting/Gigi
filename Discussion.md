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

Nine training runs were conducted, totaling ~4,000 iterations and 736,000 games:

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

The model clearly learns in the first 500 iterations: value predictions improve dramatically (vloss 267 to 34), entropy declines naturally (1.05 to 0.71), and the model starts beating PAR by ~1.5 IMPs/game. But the second 500 iterations (run 5) show no further progress — all metrics flatline. Run 6 applied four targeted fixes and broke through the plateau. Run 7 confirmed a new plateau at ~+3.4 IMP. Run 8 tested card play pre-training but did not improve over run 7. Run 9 trained against BEN as the opponent but did not close the gap.

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

**Observations:**

- **All models lose to BEN's NNs by 4-8 IMPs/deal.** BEN's supervised NNs (trained on 8,730+ GIB-BBO games with specialized models for each position — lefty, dummy, righty, declarer for both NT and suit contracts) have a substantial edge over our single 5.1M-parameter transformer.
- **Run 7 is the strongest against BEN** (-5.18 IMP/deal), consistent with its status as the best model overall. The 1,000 extra iterations of RL self-play after supervised pre-training produced the most robust policy.
- **Run 9 (trained against BEN) did not improve over run 7** (-6.63 avg vs -5.18). Despite 500 iterations of direct BEN exposure, the model couldn't adapt — the combination of tight KL threshold, small step counts from sequential play, and a hardened policy prevented meaningful learning. Simply changing the opponent is insufficient without also addressing the training dynamics.
- **Run 8 performed worst** (-8.40) with remarkably low variance (std=4.5 vs 14-20 for others). Raw scores were tiny (mean ~220 points), suggesting many low-level contracts or near-passouts. The card play pre-training with insufficient data appears to have disrupted bidding quality.
- **Run 4 (no pre-training) had the widest CI** — its 95% CI includes 0, meaning it's not statistically distinguishable from BEN with 50 games. The high variance comes from erratic bidding producing wild score swings in both directions.
- **Performance vs random doesn't predict performance vs BEN.** Run 6 (+3.6 vs random) scored worse against BEN (-7.46) than run 7 (+3.4 vs random, -5.18 vs BEN). Strategies that exploit random's weaknesses don't transfer to exploiting a competent opponent.
- **The gap is significant but not enormous.** In competitive bridge, 5-8 IMPs/deal is roughly the difference between an intermediate and an advanced player. Our RL model plays recognizable bridge (not random), but BEN's supervised training on expert data gives it meaningfully better decisions in both bidding and card play.

---

## What Went Well

**Clean architecture.** The `rlbridge/` module is fully independent of BEN's `src/`. BEN's code is untouched — we only import its scoring, card encoding, bidding rules, and DDS solver. This makes the RL system portable and BEN upgradeable.

**Verified correctness.** The game engine was validated against 10,000 random games: all tricks sum to 13, all scores match BEN's scoring function, all trajectory actions are legal. 70 unit tests cover the engine, model, reward, temperature schedule, and pre-training pipeline.

**Principled reward design.** IMP-vs-PAR normalizes extreme scores (a 7NT hand scores +2220 raw but only +13 IMPs vs PAR). This prevents a single outlier from dominating the training signal, which was exactly the problem the broken eval exposed.

**Profiling-driven optimization.** Each bottleneck was identified through instrumented profiling before being addressed. This led to targeted fixes (batched inference, GPU PPO, cached masks) rather than speculative optimization. The result was a 24.6x speedup with no correctness regressions.

**Iterative debugging.** The eval metric was broken (raw scores + 20 games = wild fluctuations). This was diagnosed from the run 4 results, fixed with IMP scoring, and validated in run 5. The iterative loop of "run, observe, diagnose, fix" worked well.

---

## What Needs Improvement

**5 IMP gap vs BEN.** Our best model (run 7) loses to BEN's pure NNs by -5.18 IMP/deal. The gap comes from three structural disadvantages: (1) one generalist model vs BEN's 8+ specialized models, (2) self-play co-adaptation producing strategies that don't transfer to competent opponents, (3) card play learned entirely from self-play with no expert supervision. The BEN evaluation (see above) also revealed that performance vs random doesn't predict performance vs BEN — run 6 was better vs random but worse vs BEN than run 7.

**Self-play co-adaptation was not the key bottleneck.** Run 9 trained directly against BEN's NNs and showed no improvement vs BEN (-6.6 IMP vs run 7's -5.18). The real bottleneck appears to be the training dynamics: PPO's KL early stopping triggers after 1-3 mini-batches, the policy has hardened, and sequential play yields only ~2,150 steps/iter (vs ~15K batched). The model can't make large enough policy updates to adapt, regardless of opponent quality.

**No position specialization.** BEN uses 8 separate card play models (lefty/dummy/righty/declarer × NT/suit), each specialized for its role. Our single transformer uses identical weights for all positions. Declarer play (planning how to take tricks) and defensive play (inferring partner's signals, finding the killing defense) are fundamentally different skills that benefit from specialization.

**Card play data gap.** Our model learned card play entirely from RL self-play (run 7) or from 15K insufficient examples (run 8). BEN's card play models were trained on far more supervised expert data. Distilling from BEN's own play could generate unlimited card play training data without external acquisition.

**KL early stopping limits late-stage learning.** In runs 6, 7, and 9, late iterations triggered KL early stopping after just 1-3 mini-batches. The policy has hardened and small updates push past the KL threshold immediately. This effectively freezes the policy in later training. Run 9 confirmed this is the binding constraint — even with a qualitatively different opponent (BEN), the KL threshold prevented the large policy updates needed to adapt.

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

Runs 7-9 revealed that the gap isn't primarily about opponent quality or co-adaptation — run 9 trained directly against BEN and saw no improvement. The binding constraints appear to be: (1) KL early stopping freezes the policy before meaningful adaptation occurs, (2) one generalist 5.1M-param model vs BEN's suite of specialized models, (3) card play was learned entirely from self-play with no expert supervision. The strategies below are re-prioritized based on run 9's findings.

~~**7. Train against BEN instead of self-play.**~~ Done in run 9. BEN's NNs used as EW opponents during RL training, resumed from run 7. The model maintained vs-random performance (+4.6 IMP) but did not improve against BEN (-6.6 avg vs run 7's -5.18). The bottleneck was not the opponent — PPO's KL early stopping triggered after 1-3 mini-batches nearly every iteration, preventing the large policy updates needed to adapt. Combined with fewer training steps (64 sequential games vs 256 batched), each iteration delivered minimal gradient signal.

**8. Distill from BEN's play.** Have BEN play thousands of games against itself and record every decision — both bidding and card play. Use these as supervised training data for our model. This solves the card play data problem (unlimited examples vs. 15K from PBN) and provides expert-quality targets. The pre-training pipeline already supports both bidding and card play examples. Target: 10K+ games (130K+ card play examples, 30K+ bidding examples). This can be combined with item 7 — distill first, then fine-tune with RL against BEN.

**9. Position-conditional card play.** BEN's biggest architectural advantage is 8 specialized card play models (lefty/dummy/righty/declarer × NT/suit). Our single transformer uses the same weights regardless of position. Add position (declarer/LHO/dummy/RHO) and strain (NT/suit) as conditioning inputs — either as additional embeddings fed into the transformer, or as separate output heads sharing the same backbone. This gives specialization without maintaining 8 separate models. The position and strain information is already available in the observation; it just isn't explicitly conditioning the card play policy head.

**10. Opponent pool with BEN.** During training, randomly assign opponents from a pool: {BEN, current model, past checkpoints}. This forces robustness across different play styles. Start with 50% BEN + 50% self-play, then tune the ratio. Unlike pure self-play, this prevents co-adaptation while maintaining the exploration benefits of playing against an evolving policy.

### Medium Priority

**11. Increase model capacity.** The current 5.1M parameter model (d_model=256, 6 layers) may not have enough capacity for bridge's complexity. Try d_model=512, n_layers=8 (~20M params). BEN's collective model suite (bidding LSTM + 8 card play NNs + lead models) likely exceeds this in total parameters, and each component is specialized. A larger backbone could better support position-conditional heads (item 9).

**12. Reward shaping with BEN reference scores.** Instead of IMP-vs-PAR (which measures absolute quality), use IMP-vs-BEN as the training reward. For each deal, play it with BEN to get a reference score, then reward our model for beating BEN's result. This directly optimizes for closing the gap and provides a more informative gradient — the model learns not just "play well" but "play better than BEN on this deal."

**13. Disable or further raise KL threshold.** Runs 6, 7, and 9 all show late-stage KL early stopping after just 1-3 mini-batches. Run 9 confirmed this is the binding constraint — training against BEN couldn't improve because policy updates were too small. Try 0.10 or disable entirely (`--target-kl 0`). This is now the **highest priority** — no opponent change or architectural improvement will help if the training loop can't update the policy. Should be combined with re-running BEN opponent training (item 7) to test whether relaxed KL + BEN opponent together break the plateau.

**14. Add experiment tracking.** Integrate WandB or TensorBoard to log training metrics, eval results, and hyperparameters. With BEN evaluation now providing a meaningful metric, tracking IMP-vs-BEN across training iterations becomes essential for diagnosing whether changes are working.

### Lower Priority

**15. Separate bidding and play networks.** Currently one transformer handles both phases. Bridge bidding and card play are fundamentally different tasks — bidding involves partnership communication and convention systems, while card play involves inference and planning. Separate specialized networks might learn each phase more effectively. However, position-conditional heads (item 9) may capture most of this benefit with less complexity.

**16. More external card play data.** Acquire larger PBN archives (BridgeComposer, ACBL, Vugraph) with play records. This is lower priority now because distilling from BEN (item 8) can generate unlimited card play data without external acquisition. External data becomes valuable if BEN distillation proves insufficient — real expert play may contain patterns that BEN's NNs don't capture.

---

## Summary

The foundation is solid: correct game engine, working training loop, principled rewards, fast iteration. The model demonstrably learns from scratch and beats random play.

After runs 4-5 plateaued at +1 IMP, run 6 applied four targeted fixes — supervised pre-training from 563K expert bidding examples, cosine temperature decay, raised KL threshold, and 4x more games per iteration. The result was a **3.6x improvement** in average advantage vs random (+1.0 to +3.6 IMP), with a peak of +6.2 IMP. Run 7 extended training by 1,000 iterations and confirmed a second plateau at +3.4 IMP — more self-play iterations alone don't help.

Run 8 tested card play pre-training from BBO PBN files (~15K card play examples), but starting from scratch with insufficient data actually performed worse (+1.1 IMP avg). The card play pipeline works — it just needs 10-100x more supervised data to be effective.

Run 9 tested the hypothesis that self-play co-adaptation was the key bottleneck by training directly against BEN's NNs as the EW opponent. Despite 500 iterations of direct BEN exposure, the model did not improve against BEN (-6.6 IMP/deal avg, vs run 7's -5.18 baseline). The vs-random advantage was maintained (+4.6 IMP), but the model couldn't adapt to BEN because PPO's KL early stopping triggered after 1-3 mini-batches nearly every iteration, preventing meaningful policy updates.

Evaluation against BEN's pre-trained neural networks (pure NN, no search) gives a meaningful skill measurement: our best model (run 7) loses by **-5.18 IMP/deal** (95% CI: [-8.2, -2.2]). All models lose by 4-8 IMPs. BEN's supervised NNs, trained on 8,730+ expert games with 8 specialized position models, have a substantial edge over our single 5.1M-parameter transformer trained via RL self-play. The gap is significant but not enormous — roughly the difference between an intermediate and advanced player.

The highest-impact next steps are: (1) relaxing or disabling KL early stopping — run 9 confirmed this is the binding constraint that prevents adaptation regardless of opponent quality; (2) re-running BEN opponent training with relaxed KL to test whether the combination breaks the plateau; (3) distilling from BEN's play to generate unlimited supervised card play data; (4) adding position-conditional card play heads to match BEN's architectural specialization.
