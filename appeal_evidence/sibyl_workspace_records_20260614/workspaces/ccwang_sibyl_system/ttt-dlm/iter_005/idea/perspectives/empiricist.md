# Empiricist Perspective: Experiment-Driven Research Proposal for DLM Inference-Time Compute Scaling

**Agent**: sibyl-empiricist
**Date**: 2026-03-11
**Topic**: Masked Diffusion Language Model Inference-Time Compute Scaling (ReMask-Retry / TTT / TCR)
**Iteration**: 5 (post-pilot evidence round)

---

## Preamble: What the Data Actually Shows

As an experimentalist, I reject narrative and insist on measurement. After 3 pilot experiments and 18+ iterations, the empirical record is unambiguous:

| Experiment | SSL Metric | Task Metric | Verdict |
|-----------|-----------|-------------|---------|
| P1 (feasibility) | Loss -33% to -72% | Not measured | Mechanism works at SSL level |
| P2 (signal quality) | SNR peaks at r=0.6 | Not measured | Phase scheduling justified |
| P3 (quick eval) | Loss -52.7% | Accuracy -1.0pp (GSM8K-200) | **SSL-task decorrelation confirmed** |
| MetaState-GRU pilot | N/A | 43.75% vs vanilla 50.0% (n=16) | Paradigm-level red flag |

The most important number in this entire project is the **-1.0pp** in P3. The second most important is the **gate value of 0.007**. Together, they tell us: (a) the TTT mechanism learns at the SSL level, (b) the learned signal is not reaching the backbone, and (c) even if it did reach the backbone, we have zero evidence it would help task performance. These are three distinct problems, not one.

---

## Literature-Grounded Assessment

### TTT Auxiliary Loss Misalignment Is a Well-Documented Failure Mode

Liu et al. ("TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?", NeurIPS 2021) systematically documented that TTT fails when auxiliary loss gradients are not correlated with main task gradients. They showed that inappropriate pretext tasks cause feature distributions to shift away from the training domain, **actively degrading** performance. Our P3 result (SSL -52%, accuracy -1pp) is textbook TTT++ failure mode.

Meta-TTT (Tao et al., arXiv 2410.01709) further showed that without constraints on feature distribution during test-time adaptation, the encoder overfits to the SSL task and deteriorates main-task accuracy. They introduced feature alignment constraints -- a strategy DaL does not employ.

The recent "Illusion of Progress" paper (Sheng et al., arXiv 2506.24000) provides a comprehensive critical analysis of test-time adaptation methods, highlighting systematic methodological issues including unfair comparisons and inconsistent evaluation protocols that inflate reported gains.

### Verifier-Guided TTT Dramatically Outperforms Self-Supervised TTT

VDS-TTT (Moradi et al., arXiv 2505.19475) demonstrated up to 32.29% relative improvement by replacing self-supervised pseudo-labels with verifier-scored samples for test-time training data. This is direct evidence that **the quality of the supervision signal is the binding constraint**, not the update mechanism.

PonderTTT (Sim, arXiv 2601.00894) showed that adaptive compute allocation via reconstruction loss gating achieves 82-89% Oracle Recovery -- but on language modeling perplexity, not downstream task accuracy. The gap between SSL and task metrics persists.

### DLM-Specific Compute Scaling Has Better Alternatives

The DLM inference-time scaling landscape now includes multiple training-free methods with demonstrated task improvements:

1. **UnMaskFork** (Misaki & Akiba, arXiv 2602.04344): MCTS-based tree search over deterministic unmasking actions. Consistently outperforms baselines on coding benchmarks.
2. **TReASURe** (Yu et al., arXiv 2509.23146): Tree search with UnmaskBranch + ResubstituteScore. State-of-the-art in low-NFE regimes.
3. **Reward-Guided Stitching** (Miles et al., arXiv 2602.22871): Diffusion sampling + PRM step-level scoring + AR solver. +23.8% across 6 tasks with 1.8x latency reduction.
4. **MDPO + RCR** (He et al., arXiv 2508.13148): Running Confidence Remasking as plug-in. 9.6% on MATH500, 54.2% on Countdown over SOTA.
5. **DUS** (Luxembourg et al., arXiv 2506.19037): Dilated unmasking scheduler. Outperforms confidence-based planners across GSM8K, MATH500, HumanEval, MBPP, BBH, MMLU-Pro without modifying the denoiser.
6. **Where-to-Unmask** (Asano et al., arXiv 2602.09501): Ground-truth-guided Gt-Margin oracle significantly enhances reasoning accuracy.

**Critical comparison**: These methods have demonstrated **task-level improvements** on standard benchmarks. DaL has demonstrated **SSL-level improvement** only. The evidence gap is enormous.

### MesaNet Shows TTT Can Work -- Under Very Different Conditions

MesaNet (von Oswald et al., arXiv 2506.05233) demonstrates that optimal test-time training (solving in-context regression to optimality via conjugate gradient) outperforms approximate TTT methods (DeltaNet, Mamba, xLSTM) on language modeling and long-context tasks. However, MesaNet operates on autoregressive sequence modeling where the in-context objective is naturally aligned with the generative task. In DLM denoising, the in-context objective (MLM on revealed tokens) has no such natural alignment -- this is exactly the misalignment P3 exposed.

---

## Core Experimental Proposal: Diagnostic-First Design with Hard Kill Criteria

### Philosophy

I propose an experiment-first approach where every claim must be backed by a measurable experiment with pre-registered success/failure criteria. No hand-waving, no "directionally positive" interpretations. Binary GO/NO-GO at every gate.

### Proposal 1: D0c — The SSL-Task Alignment Diagnostic (HIGHEST PRIORITY)

**Hypothesis**: The Pearson correlation between SSL loss improvement and task accuracy improvement across DaL configurations is > 0.3.

**Why this is non-negotiable**: P3 showed SSL loss -52% with accuracy -1pp. If this decorrelation is structural, the entire DaL framework is unsalvageable regardless of gate repair, training compute, or architectural changes. D0c must run before any further training investment.

**Exact Protocol**:

1. Fix backbone: Dream-7B-Instruct, frozen
2. Fix TTT layer: MLP variant, d_ttt=448, layer 14
3. Generate 12 configurations by varying:
   - Gate init: sigmoid(-5), sigmoid(-2), sigmoid(0), sigmoid(2) [4 levels]
   - Meta_lr: 5e-5, 1e-4, 5e-4 [3 levels]
4. For each configuration:
   - Train 2000 meta-steps (K=4 unrolling) on OpenWebText-10K
   - Record final SSL loss on held-out validation set (50 sequences)
   - Evaluate GSM8K-50 accuracy (first 50 problems, 128 denoising steps)
   - Record gate value at training end
5. Compute:
   - Pearson r(SSL_loss_change, GSM8K_accuracy) across 12 configs
   - Pearson r(gate_value, GSM8K_accuracy) across 12 configs
   - Scatter plot with 95% CI

**Decision matrix (pre-registered)**:

| r(SSL, accuracy) | r(gate, accuracy) | Decision |
|:-:|:-:|:--|
| > 0.3 | > 0.3 | **PROCEED** to gate repair + full training |
| > 0.3 | < 0.3 | Gate is not the bottleneck; investigate injection mechanism |
| < 0.3 | > 0.3 | SSL loss is misaligned but gate helps; try alternative SSL objectives |
| < 0.3 | < 0.3 | **KILL DaL**. Full pivot to training-free methods |

**Confounders to control**:
- Same random seeds across all configs (seed=42)
- Same evaluation set for all configs (first 50 GSM8K)
- Training data fixed (same 10K OpenWebText subset)
- Evaluation temperature fixed at 0.0 (greedy)

**Estimated cost**: 12 configs x 2000 steps x ~3s/step = ~20 GPU-hours. With 4 GPUs: ~5 hours wall-clock.

**What would falsify the hypothesis**: r < 0.1 with p > 0.05 (correlation not significantly different from zero).

### Proposal 2: Controlled Update Rule Comparison (Conditional on D0c PASS)

**Hypothesis (H1)**: Under matched parameter budget and training FLOPs, TTT-MLP > MetaState-GRU > vanilla on at least 2 of 3 primary benchmarks.

**Why this is the core empirical contribution**: The literature contains no controlled comparison of update rules for DLM cross-step memory. MetaState uses GRU gates; DaL proposes gradient-based updates. Whether the update mechanism matters -- independently of architecture, data, and compute -- is an open question regardless of whether DaL "works."

**Exact Protocol**:

Phase A: Implementation validation (Day 2, 4h)
1. Implement 4 update rules in identical architecture (Mixer + Updater + Injector from MetaState):
   - GRU (MetaState baseline)
   - TTT-Linear (W -= lr * grad)
   - TTT-MLP (W = MLP(W, grad))
   - Momentum-TTT (W = (1-lambda)W - lr*(beta*m + grad))
2. Parameter budget: match within +-10% by adjusting hidden dimensions
3. Verify all 4 produce identical output shapes and pass gradient flow tests

Phase B: Training (Days 2-4, ~48 GPU-hours)
1. For each of 4 update rules x 3 random seeds:
   - 10K meta-training steps on OpenWebText-10K
   - K=4 unrolling, meta_lr=1e-4
   - Gate init: sigmoid(-2) (or best from D0c)
   - Checkpoint at 2K, 4K, 6K, 8K, 10K
   - GSM8K-50 quick eval at each checkpoint (trend monitoring)
2. Training abort: if accuracy clearly declining at 6K steps

Phase C: Evaluation (Day 5, ~16 GPU-hours)
1. Best checkpoint per variant on:
   - GSM8K-200 (reasoning)
   - HumanEval-164 (code generation)
   - ARC-Challenge-200 (knowledge + reasoning)
2. Vanilla Dream-7B baseline (same denoising steps, same seed)
3. Report: accuracy, 95% CI via paired bootstrap (n=1000), Cohen's d, Distinct-1/2/3

**Statistical plan**:
- Primary test: paired bootstrap, alpha=0.05
- Multiple comparisons: Holm-Bonferroni across 4 variants x 3 benchmarks
- Effect size: Cohen's d with interpretation (small/medium/large)
- Report all results including negative ones

**Confounders to control**:
- Same backbone, same insertion layer, same training data
- Parameter budget matched within +-10%
- Same denoising steps and schedule at evaluation time
- Same evaluation prompts and parsing logic
- 3 seeds minimum for final results

**What would falsify the hypothesis**:
- GRU >= TTT-MLP on all 3 benchmarks: update mechanism doesn't matter
- All variants <= vanilla: cross-step memory paradigm is harmful
- All variants = vanilla within noise: cross-step memory has no effect

### Proposal 3: FLOPs-Fair Compute Allocation Study

**Hypothesis (H2)**: At fixed FLOPs budget, DaL (fewer denoising steps + TTT updates) outperforms Dense Denoising (all FLOPs as denoising steps) on reasoning tasks.

**Why this matters regardless of DaL success**: This tests whether TTT-style computation provides orthogonal value to denoising iteration. If dense denoising is Pareto-optimal, no training-based inference-time method can improve on it -- and the entire field of DLM TTT is pointless.

**Exact Protocol**:

1. Calibrate FLOPs:
   - Measure wall-clock time for 1 denoising step (forward pass only)
   - Measure wall-clock time for 1 TTT step (forward + backward + weight update)
   - Compute FLOP ratio R = TTT_FLOPs / denoising_FLOPs (expected ~2.5)
2. Define matched FLOPs budgets equivalent to 64, 128, 256, 512 denoising steps
3. For each budget:
   - Dense Denoising: N denoising steps, no TTT
   - DaL: N/R fewer denoising steps + TTT updates at each step
   - DaL-Phase: TTT only in mask ratio [0.1, 0.7], dense outside
4. Evaluate GSM8K-200 for each configuration
5. Plot accuracy-vs-FLOPs Pareto frontier

**What would falsify the hypothesis**: Dense Denoising >= DaL at all FLOPs budgets, with DaL-Phase providing no advantage.

### Proposal 4: Alternative A -- Training-Free Ensemble Baseline (Parallel Track from Day 1)

**Hypothesis**: A systematic composition of training-free methods (A-CFG + Entropy-Adaptive Scheduling + Soft-Masking) achieves >=5% improvement over vanilla Dream-7B on GSM8K-200.

**Why this runs in parallel**: We cannot afford serial exploration with 4 GPUs. Alternative A has the strongest existing positive signal (A-CFG +12.5pp on GSM8K-16 in iter 4). It provides a credible fallback paper regardless of DaL outcomes.

**Exact Protocol**:

Day 1: Individual baselines (4 GPUs parallel)
- GPU 0: A-CFG on Dream-7B, GSM8K-200
- GPU 1: ReMDM on Dream-7B, GSM8K-200
- GPU 2: RCR on Dream-7B, GSM8K-200
- GPU 3: Vanilla Dream-7B, GSM8K-200 + ARC-C-200

Day 2: Best pairwise compositions
- Based on Day 1 results, compose the top 2 methods
- Evaluate on GSM8K-200 + HumanEval + ARC-C

**What would falsify the hypothesis**: No individual training-free method improves over vanilla at n=200 (iter 4's n=16 result was noise).

---

## Pilot Study Design: D0c Can Validate/Invalidate in <1 GPU-Hour

**Minimal viable D0c**: Instead of 12 configurations, run 4 configs on GPU 0:

| Config | Gate Init | Meta_lr | Training Steps |
|:------:|:---------:|:-------:|:--------------:|
| A | sigmoid(-5) | 1e-4 | 1000 |
| B | sigmoid(-2) | 1e-4 | 1000 |
| C | sigmoid(0) | 1e-4 | 1000 |
| D | sigmoid(0) | 5e-4 | 1000 |

Each config: 1000 steps x ~1.5s/step = 25 min training + 5 min eval (GSM8K-25).
Total: ~2 hours on 1 GPU (well under 1 GPU-hour per config).

If all 4 configs show accuracy <= vanilla, the correlation is trivially < 0.3 and we kill DaL immediately. If any config shows accuracy > vanilla, expand to full 12-config D0c.

---

## Confounders and Experimental Pitfalls

### Known Confounders in DLM Evaluation

1. **Denoising step count sensitivity**: DLM accuracy is highly sensitive to NFE. All comparisons MUST use identical step counts (or FLOPs-fair budgets). Prior iterations failed to control this.

2. **Answer parsing fragility**: GSM8K accuracy depends on extracting the final numerical answer. DLM-generated text may have different formatting than AR-generated text. Use robust parsing: extract last number after "####" or last number in text.

3. **Temperature effects**: At temperature > 0, DLM sampling introduces stochasticity. All pilot evaluations at temperature 0 (greedy/argmax at each denoising step). Final results with temperature sweep (0.0, 0.3, 0.7).

4. **Sequence length truncation**: Dream-7B has context length limits. GSM8K chain-of-thought can exceed limits. Fixed max generation length of 512 tokens for all methods.

5. **Pseudo-label contamination**: In DaL, revealed tokens at intermediate steps are the model's own predictions, not ground truth. This means the TTT layer is training on potentially incorrect tokens. The later the denoising step, the more accurate the pseudo-labels -- but early steps (high mask ratio) are training on noise. This is a fundamental confounder that phase scheduling partially addresses.

6. **Gate dynamics as confound**: If gate = 0.007 (as in P3), we are evaluating "vanilla + epsilon" rather than "vanilla + DaL". Any experimental comparison must verify gate > 0.10 at evaluation time, or the comparison is meaningless.

### Suspicious Improvement Thresholds

Per the Evidence Contract:
- **>30% improvement on a simple baseline**: Flag as potentially degenerate. Dream-7B vanilla is ~40% on GSM8K. Any method claiming >52% (i.e., +12pp) must be independently verified with diversity checks (Distinct-N) and manual sample inspection.
- **A-CFG at +12.5pp on n=16**: This is within the 30% threshold but n=16 has massive variance (95% CI approximately +/-15pp). Must verify at n>=100.

---

## Computational Cost Estimates

| Experiment | GPU-Hours | Wall-Clock (4 GPUs) | Priority |
|-----------|:---------:|:-------------------:|:--------:|
| D0c minimal (4 configs) | 8 | 2h | **P0** |
| D0c full (12 configs) | 20 | 5h | P0 (if minimal shows signal) |
| Alternative A baselines | 16 | 4h | **P0** (parallel) |
| Gate repair + training (10K steps) | 12 | 12h | P1 (if D0c passes) |
| Update rule ablation (4 variants x 3 seeds) | 48 | 16h | P2 (if P1 passes) |
| Full benchmark evaluation | 16 | 4h | P2 |
| FLOPs-fair compute study | 24 | 6h | P3 |
| **Total (if all gates pass)** | **~144** | **~50h** | |
| **Total (if D0c fails, pivot)** | **~24** | **~6h** | |

---

## Success Probability Estimates

| Scenario | P(success) | Reasoning |
|----------|:----------:|-----------|
| D0c r > 0.3 (SSL-task aligned) | 25% | P3 is strong counter-evidence; TTT++ literature says pretext task misalignment is structural, not engineering |
| D0c pass AND gate repair works | 18% | Conditional: 25% x 70% gate repair success |
| Full DaL > vanilla by >=2% | 15% | Conditional on D0c + gate + sufficient training. 18% x 80% training success |
| Update rule ablation publishable | 60% | Even with negative DaL result, controlled comparison is valuable |
| Alternative A > vanilla by >=3% | 45% | Training-free methods have stronger prior evidence; A-CFG at +12.5pp (n=16) is noisy but promising |
| **Any publishable paper** | **75%** | Diversified: DaL positive OR negative-result ablation OR Alternative A composition study |

---

## Negative Results and Unresolved Risks

### Explicitly Reported Negative Results

1. **P3 is a negative result**: SSL loss -52%, accuracy -1pp. This is the project's strongest empirical finding and must be reported prominently regardless of future outcomes.

2. **MetaState-GRU underperformance**: 43.75% vs vanilla 50.0% at n=16. While not statistically significant (massive CI), it challenges the entire paradigm.

3. **18+ iterations without positive task result**: The project has never produced a statistically significant task-level improvement over vanilla Dream-7B with any cross-step memory method. This pattern is itself a finding.

### Unresolved Risks

1. **Principal-agent misalignment may be unfixable**: If the MLM-on-revealed-tokens objective is structurally misaligned with task accuracy (as TTT++ predicts), no gate repair, training schedule, or architectural modification will save DaL. Only a fundamentally different supervision signal (e.g., verifier-guided, reward-based) could work -- but this changes the method into something else entirely.

2. **Pseudo-label feedback loop**: DaL trains on its own predictions. If early denoising steps produce poor predictions, the TTT layer learns from errors. This could create a positive feedback loop where the TTT layer becomes increasingly confident in incorrect patterns. No mitigation is proposed in the current design.

3. **Compute overhead may not justify gains**: Even if DaL achieves +2%, the 1.2x latency overhead (P3) means it is Pareto-dominated by simply running 1.2x more denoising steps. The FLOPs-fair study (Proposal 3) is essential to determine if DaL provides genuine orthogonal value.

4. **MetaState reproduction risk**: MetaState's reported gains may depend on training details not in the paper. If our simplified GRU reproduction also fails, we have no positive baseline for the paradigm.

5. **Sample size power analysis**: At n=200 with p=0.40 baseline and alpha=0.05, we have ~80% power to detect a +8pp effect but only ~35% power to detect a +3pp effect. The proposed 2% threshold may be below our detection power at n=200. Consider n=500 for definitive results (but this costs 2.5x more compute).

---

## Recommended Execution Order (Strict)

```
Hour 0-2:   D0c minimal (4 configs, GPU 0-1)
            Alternative A individual baselines (GPU 2-3)

Hour 2:     D0c DECISION GATE
            ├── r > 0.3 → Expand to full D0c (GPU 0-1)
            │              Continue Alternative A (GPU 2-3)
            ├── r in [0.1, 0.3] → Expand D0c with alt SSL objectives
            │                      Continue Alternative A
            └── r < 0.1 → KILL DaL. Full pivot to Alternative A.
                           Use all 4 GPUs for composition study.

Hour 6:     Full D0c + Alternative A Day 1 results
            GATE 2: D0c full r > 0.3 AND gate > 0.10?
            ├── YES → Start 10K training (Phase 1)
            └── NO  → Full pivot confirmed

Hour 18:    Phase 1 training complete
            GATE 3: GSM8K-200 accuracy >= vanilla + 2%?
            ├── YES → Phase 2: Update rule ablation
            └── NO  → Pivot to composition/negative-result paper

Hour 50:    All experiments complete. Write paper.
```

---

## Summary: What Can Actually Be Measured

| Claim | Measurable? | Measurement | Current Evidence |
|-------|:-----------:|-------------|:----------------:|
| TTT learns from denoising | YES | SSL loss decrease | **CONFIRMED (P1)** |
| SSL learning helps task accuracy | YES | D0c correlation | **UNKNOWN (P3 suggests NO)** |
| Gate repair enables signal injection | YES | Gate value > 0.10 | **UNKNOWN** |
| TTT-MLP > GRU > vanilla | YES | Benchmark accuracy | **UNKNOWN** |
| TTT is orthogonal to dense denoising | YES | FLOPs-fair Pareto | **UNKNOWN** |
| Training-free methods compose | YES | Benchmark accuracy | **PARTIALLY (A-CFG n=16)** |

The honest summary: we have confirmed exactly one thing (TTT mechanism works at SSL level). Everything else is unknown or negative. The D0c diagnostic will determine in ~2 hours whether the remaining unknowns are worth investigating.

---

## References

### TTT Failure Modes and Methodology
1. Liu et al. (2021). "TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?" NeurIPS 2021. [Link](https://proceedings.neurips.cc/paper/2021/file/b618c3210e934362ac261db280128c22-Paper.pdf)
2. Tao et al. (2024). "Meta-TTT: A Meta-learning Minimax Framework For Test-Time Training." arXiv 2410.01709
3. Sheng et al. (2025). "The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models." arXiv 2506.24000
4. Moradi et al. (2025). "VDS-TTT: Verifier-Driven Sample Selection for Test-Time Training." arXiv 2505.19475
5. Sim (2025). "PonderTTT: Adaptive Compute Allocation via Test-Time Training." arXiv 2601.00894
6. von Oswald et al. (2025). "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training." arXiv 2506.05233
7. Hu et al. (2025). "Test-time learning for large language models." arXiv 2505.20633
8. Bansal et al. (2025). "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs." arXiv 2512.13898

### DLM Inference-Time Scaling
9. Misaki & Akiba (2026). "UnMaskFork: Test-Time Scaling via Deterministic Action Branching." arXiv 2602.04344
10. Yu et al. (2025). "TReASURe: Tree Reward-Aligned Search for MDLMs." arXiv 2509.23146
11. Miles et al. (2026). "Reward-Guided Stitching for Diffusion Language Models." arXiv 2602.22871
12. He et al. (2025). "MDPO + RCR: Masked Diffusion Policy Optimization." arXiv 2508.13148
13. Luxembourg et al. (2025). "DUS: Dilated Unmasking Scheduler for MDLMs." arXiv 2506.19037
14. Asano et al. (2026). "Where-to-Unmask: Ground-Truth-Guided Unmasking Order." arXiv 2602.09501
15. Luo et al. (2026). "Self-Rewarding SMC for MDLMs." arXiv 2602.01849
16. Padole et al. (2025). "Improving Text Style Transfer using MDLMs with Inference-time Scaling." arXiv 2508.10995

### DLM Foundations
17. Sahoo et al. (2026). "Scaling Beyond Masked Diffusion Language Models." arXiv 2602.15014
18. Xia et al. (2026). "T*: Progressive Block Scaling for MDLMs." arXiv 2601.11214
19. Yang et al. (2025). "Taming MDLMs via Consistency Trajectory RL." arXiv 2509.23924
20. Huang & Mirzasoleiman (2026). "Tuning the Implicit Regularizer of MDLMs." arXiv 2601.22450
21. Shnaidman et al. (2025). "Activation Steering for MDLMs." arXiv 2512.24143

### TTT and Memory Mechanisms
22. MetaState (2026). arXiv 2603.01331
23. TTT (Sun et al., 2024). arXiv 2407.04620
24. Titans (2025). arXiv 2501.00663
25. SR-TTT (2026). arXiv 2603.06642
26. TTT-E2E (2025). arXiv 2512.23675
27. Hwang et al. (2026). "REFINE: Reinforced Fast Weights with Next-Sequence Prediction." arXiv 2602.16704

### Scaling and Evaluation
28. Isik et al. (2024). "Scaling Laws for Downstream Task Performance." arXiv 2402.04177
29. Huang et al. (2026). "Reasoning or Rationalization?" arXiv 2603.01190
30. Parashar et al. (2025). "Sys2Bench: Inference-Time Techniques for LLM Reasoning." arXiv 2502.12521

### Project Internal Evidence
- P1 (SSL feasibility): PASS -- SSL loss -33% to -72%
- P2 (Signal quality): PASS -- SNR peak at r=0.6, degradation at r>0.7
- P3 (Quick eval): FAIL -- SSL -52.7%, accuracy -1.0pp, gate=0.007
- MetaState-GRU pilot: 43.75% vs vanilla 50.0% (n=16)
