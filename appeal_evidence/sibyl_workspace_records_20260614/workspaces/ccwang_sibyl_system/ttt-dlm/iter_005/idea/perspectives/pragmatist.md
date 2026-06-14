# Pragmatist Perspective: Practical Research Proposals for DLM Inference-Time Scaling

**Agent**: sibyl-pragmatist
**Date**: 2026-03-11
**Topic**: Masked Diffusion Language Model Inference-Time Compute Scaling (ReMask-Retry / TTT / TCR)
**Iteration**: 5 (post-pilot evidence round)

---

## Preamble: Honest Assessment of Where We Stand

After 18+ iterations and 3 pilot experiments, the evidence tells a clear story:

1. **DaL (TTT insertion) is mechanistically feasible but practically blocked**: P1 confirms SSL loss drops 33-72%. P3 shows this doesn't translate to task accuracy (-1.0pp). Root cause: gate stuck at 0.007.
2. **MetaState GRU underperforms vanilla**: 43.75% vs 50.0% at n=16. While n=16 has enormous variance, this is a red flag for the entire "insert lightweight memory into frozen DLM" paradigm.
3. **Training-free methods have the strongest existing results**: A-CFG (+12.5pp on GSM8K-16 in iter 4), CoRe (+9.2% MBPP), RCR (consistent gains as plug-in).
4. **Compute budget is generous but time is limited**: 4x RTX PRO 6000 Blackwell (98GB each). We can run 7B models comfortably. The bottleneck is wall-clock time, not VRAM.

The pragmatist's mandate: **maximize the probability of producing a publishable paper within a 7-10 day timeline, using methods that are engineering-feasible on our 4-GPU setup.**

---

## Proposal 1: Ensemble Guidance Framework — Combining A-CFG + Soft-Masking + Entropy-Adaptive Scheduling (EGF)

### Angle: Improve Existing (Systematic Combination of Proven Training-Free Methods)

### Core Insight

The DLM inference-time scaling literature is fragmented: A-CFG (arXiv 2505.20199) improves conditional generation via dynamic low-confidence remasking, Soft-Masking (arXiv 2510.17206) preserves cross-step information by blending mask embeddings with top-k predictions, RCR (arXiv 2508.13148) tracks running confidence for principled remasking, and DUS (arXiv 2506.19037) uses dilated unmasking to minimize joint entropy. **Nobody has systematically studied how these methods compose.** Some are complementary (A-CFG operates on the guidance signal; SM operates on the embedding space; DUS operates on the unmasking schedule), while others may conflict (RCR and A-CFG both manipulate mask patterns).

The practical opportunity: **a unified benchmark + systematic composition study of training-free DLM TTS methods**, culminating in a best-practice ensemble that achieves new SOTA on reasoning benchmarks without any additional training.

### Hypothesis

**H_EGF**: A composition of A-CFG (guidance) + Entropy-Adaptive Scheduling (when to unmask) + Soft-Masking (how to handle retained masks) achieves >=5% improvement over vanilla Dream-7B on GSM8K, outperforming any individual method and establishing the composition as a unified framework.

### Method

```
Phase 1: Individual Baselines (Day 1, 4 GPUs parallel)
  GPU 0: A-CFG on Dream-7B (GSM8K-200, HumanEval, ARC-C)
  GPU 1: RCR on Dream-7B (same benchmarks)
  GPU 2: Soft-Masking (SM) on Dream-7B — use IBM's open-source finetuned checkpoint
  GPU 3: Vanilla Dream-7B + ReMDM baselines

Phase 2: Pairwise Composition (Day 2, 4 GPUs parallel)
  GPU 0: A-CFG + RCR (both manipulate masks → test for conflict)
  GPU 1: A-CFG + SM (guidance + embedding enrichment → should compose)
  GPU 2: SM + Entropy-Adaptive Schedule (embedding + when → should compose)
  GPU 3: A-CFG + SM + Entropy-Adaptive Schedule (triple stack)

Phase 3: Analysis + Ablation (Day 3-4)
  - Which pairs compose additively vs sub-additively?
  - Which method contributes most per-FLOP?
  - Pareto frontier: accuracy vs NFE for all combinations
```

Entropy-Adaptive Schedule: At each denoising step t, compute per-position entropy H(p(x_i | x_t)). Unmask positions with lowest entropy first (most confident). Allocate more denoising steps (finer granularity) when average entropy is in the "critical zone" (0.3-0.6 mask ratio, validated by P2).

### Why This Is Publishable

- **Systematic composition** of DLM TTS methods has not been done. Individual papers only compare against vanilla or one competitor.
- **Framework contribution**: The composition taxonomy (guidance-axis, embedding-axis, schedule-axis) provides organizing structure for the field.
- **Practical value**: A practitioner's recipe for "which methods to stack and in what order."
- **Negative results are equally publishable**: If methods conflict (sub-additive), that reveals important structural information.

### Failure Modes

- SM requires the IBM finetuned checkpoint — if unavailable, we implement a lightweight approximation (top-k embedding average, no training needed)
- A-CFG hyperparameter rho is sensitive — need grid search per benchmark
- Composition may be purely additive (boring) or sub-additive (less useful but still publishable)

### Computational Cost

- All training-free: zero training cost
- Evaluation: ~4 GPU-hours per method per benchmark × 3 benchmarks × 8 combinations = ~96 GPU-hours
- With 4 GPUs parallel: ~24 hours wall-clock for Phase 1+2
- **Success probability: 55%** (builds on proven individual methods; novel contribution is the composition study)
- **Time estimate: 4-5 days** including analysis and paper writing

### Key References

- A-CFG (arXiv 2505.20199). Adaptive Classifier-Free Guidance via Dynamic Low-Confidence Masking. [GitHub: pixeli99/A-CFG](https://github.com/pixeli99/A-CFG)
- Soft-Masking (arXiv 2510.17206). [GitHub: IBM/soft-masked-diffusion-language-models](https://github.com/IBM/soft-masked-diffusion-language-models)
- MDPO + RCR (arXiv 2508.13148). [GitHub: autonomousvision/mdpo](https://github.com/autonomousvision/mdpo)
- DUS (arXiv 2506.19037). Dilated Unmasking Scheduler for MDLMs.
- ReMDM (arXiv 2503.00307). [GitHub: kuleshov-group/remdm](https://github.com/kuleshov-group/remdm)

---

## Proposal 2: DaL Gate Repair + Minimal Viable Experiment (3-Day Sprint)

### Angle: Improve Existing (Engineering Fix for Identified Root Cause)

### Core Insight

P3 failed for a specific, diagnosable reason: **the residual gate stayed at sigmoid(-5) = 0.007**, meaning the TTT layer's learned representations were never injected into the backbone. The SSL loss dropped 52% (the TTT layer learned something), but the backbone never saw it. This is a <10 line code fix.

The pragmatist recognizes this is the highest-ROI single change we can make. If it works, DaL is back on track. If it doesn't work after proper gate initialization, then we have strong evidence that the SSL-task misalignment (not the gate) is the real problem, and we can pivot cleanly.

### Hypothesis

**H_gate_repair**: With gate initialization changed to sigmoid(-2) = 0.12 and gate_lr = 10x meta_lr, DaL-MLP achieves >=2% absolute improvement over vanilla Dream-7B on GSM8K-200, and the gate value reaches >= 0.10 by 5K training steps.

### Method

```
Day 1 (single GPU, ~8h):
  1. Code change: gate_init from -5 to -2, gate_lr = 10 * meta_lr
  2. Optional: gate warm-up loss L_gate = -lambda * log(sigmoid(gate_param))
  3. Meta-train 5K steps (K=4, AdamW meta_lr=1e-4, gate_lr=1e-3)
  4. Checkpoints at 1K, 2K, 3K, 4K, 5K with GSM8K-50 quick eval

Day 2 (single GPU, ~6h):
  1. Evaluate best checkpoint on GSM8K-200
  2. If accuracy >= vanilla + 2%: celebrate, proceed to Phase 1
  3. If gate > 0.10 but accuracy <= vanilla: run D0c (correlation test)
  4. If gate < 0.05: try sigmoid(0) init + warm-up loss

Day 3 (decision):
  - GO: Proceed with DaL Phase 1 (full ablation)
  - NO-GO: Clean pivot to Proposal 1 (EGF) or Proposal 3 (COBA)
```

### Why This Before Anything Else

- **Lowest hanging fruit**: 10 lines of code change, 8 hours of training
- **Diagnostic value**: Regardless of outcome, answers the critical question "is the gate the real problem or just a symptom?"
- **Sunk cost leverage**: We have the TTT layer, the training pipeline, the evaluation pipeline all working. We're one engineering fix away from a real test.

### Failure Modes

- Gate opens but accuracy still doesn't improve → SSL-task misalignment is structural
- Gate opens AND accuracy improves slightly (1-1.5%) → borderline, need more training
- Training instability with open gate → reduce gate_init to sigmoid(-1) = 0.27

### Computational Cost

- 5K training steps: ~8 GPU-hours
- GSM8K-200 eval: ~2 GPU-hours
- Total: ~10 GPU-hours (single GPU, ~10 hours wall-clock)
- **Success probability: 30%** (gate was clearly stuck; the fix is obvious; but SSL-task alignment may be the deeper issue)
- **Time estimate: 2-3 days**

### Key References

- DaL Proposal (this project, proposal.md): Gate initialization and training details
- TTT (arXiv 2407.04620): Original gate design
- Titans (arXiv 2501.00663): Momentum + weight decay for TTT layers

---

## Proposal 3: COBA — Compute-Optimal Budget Allocation for DLM Inference (Framework Paper)

### Angle: New Method (Framework + Empirical Characterization)

### Core Insight

The DLM inference-time scaling landscape has three orthogonal compute allocation axes:

```
D (Depth): More denoising steps (ReMDM, standard scaling)
W (Width): Multiple parallel trajectories (Best-of-N, SMC, tree search)
M (Memory): Cross-step information transfer (MetaState, DaL, soft-masking)
```

**No existing work systematically profiles how compute should be allocated across these axes.** ReMDM shows D-scaling works; TReASURe and Prism show W-scaling works; MetaState shows M-scaling (maybe) works. But nobody has answered: **given a fixed compute budget (measured in FLOPs, not NFE), what is the optimal D×W allocation for each task type?** This is the DLM analog of the "chain-of-thought vs. majority voting" trade-off in AR models.

This is inherently a training-free, evaluation-heavy study that can be executed entirely with existing codebases.

### Hypothesis

**H_COBA**: The optimal D×W allocation varies significantly by task type — reasoning tasks (GSM8K) favor D-scaling (more denoising steps), while code generation (HumanEval) favors W-scaling (more parallel samples). A simple regression model quality(D,W) = alpha*log(D) + beta*log(W) + gamma accurately predicts the optimal allocation point for each task.

### Method

```
Grid Search Protocol:
  D ∈ {32, 64, 128, 256, 512}  (denoising steps)
  W ∈ {1, 2, 4, 8, 16}         (parallel samples, Best-of-N or majority vote)
  Benchmarks: GSM8K, HumanEval, ARC-C, Countdown, MBPP

For each (D, W, Benchmark):
  1. Generate W samples with D denoising steps each
  2. Select best via majority vote (reasoning) or pass@1 (code)
  3. Record: accuracy, total FLOPs, wall-clock time

Analysis:
  1. Fit log-linear model per task type
  2. Compute Pareto frontier (accuracy vs FLOPs)
  3. Identify task-specific optimal allocation
  4. Compare with fixed D, W=1 (current default)
  5. Optional: add M-axis with A-CFG/RCR as the "memory" method
```

**Implementation**: Use Dream-7B + standard sampling (temperature, nucleus). No training needed. ReMDM for D>256 (remasking enables quality at high step counts). The key insight is that `W` parallel samples are *free* on 4 GPUs when W<=4.

### Why This Is A Strong Paper

1. **Practical utility**: Every DLM user faces this allocation decision. A recipe is immediately valuable.
2. **Unifies the literature**: Frames ReMDM, Best-of-N, Prism, SMC under one framework.
3. **Robust to negative results**: Even if COBA shows "just use more steps" is optimal, that's a valuable finding.
4. **Low risk**: Pure evaluation study with established models and codebases.
5. **Extensible**: The D×W framework naturally extends to D×W×M when memory methods mature.

### Failure Modes

- Grid search may be too coarse to find interesting structure
- Best-of-N selection (majority vote, pass@1) may be too noisy at small W
- The log-linear model may be too simple — task-specific nonlinearities may dominate

### Computational Cost

- 5 D values × 5 W values × 5 benchmarks × ~200 samples = 25,000 evaluations
- With W parallel samples: effectively 5 D values × 5 benchmarks × 200 = 5,000 generation runs
- At ~5s per generation (Dream-7B, 256 steps): ~25,000 seconds ≈ 7 GPU-hours per D-value
- Total: ~35 GPU-hours. With 4 GPUs: ~9 hours wall-clock for core grid, + ~16h for analysis
- **Success probability: 65%** (pure evaluation, low risk, clear contribution)
- **Time estimate: 5-6 days** including analysis and paper writing

### Key References

- ReMDM (arXiv 2503.00307). D-axis scaling.
- Prism (arXiv 2602.01842). W-axis scaling (hierarchical search).
- Self-Rewarding SMC (arXiv 2602.01849). W-axis scaling (particle sampling).
- Optimal Inference Schedules (arXiv 2511.04647). Theoretical D-scaling bounds.
- DUS (arXiv 2506.19037). Speed-quality frontier characterization.
- Parallelism and Generation Order (arXiv 2601.15593). AFP/Kendall-tau characterization of DLM behavior.

---

## Proposal 4: Confidence-Guided Progressive Refinement (CGPR) — Practical Cross-Step Information Transfer Without TTT

### Angle: New Method (Lightweight Alternative to MetaState/DaL)

### Core Insight

MetaState and DaL both try to maintain cross-step memory via trainable modules. But there's a much simpler way to transfer information across denoising steps that requires **zero training**: **use the model's own prediction confidence from step t as a soft prior for step t+1.**

This is essentially what Soft-Masking (arXiv 2510.17206) does for retained masks, but CGPR extends it to a full cross-step refinement strategy:

1. At step t, compute prediction probabilities p(x_i | x_t) for all masked positions
2. At step t+1, instead of hard binary masking, inject soft information: blend [MASK] embedding with top-k weighted prediction embeddings from step t
3. Weight the blending by the confidence from step t: high-confidence predictions get stronger injection
4. Unmask order: prioritize positions where confidence *increased most* between t-1 and t (momentum signal)

The critical difference from vanilla Soft-Masking: CGPR uses **cross-step confidence delta** (not absolute confidence) to guide the unmasking order and blending ratio. Positions where the model "changed its mind" the most are the ones that benefited most from additional context — they should get more compute.

### Hypothesis

**H_CGPR**: Confidence-Guided Progressive Refinement achieves >= 3% improvement over vanilla Dream-7B on GSM8K with zero training, by using cross-step confidence dynamics to guide both embedding enrichment and unmasking order.

### Method

```
At denoising step t:
  1. Run standard forward pass → logits L_t, probabilities P_t
  2. For each masked position i:
     a. Compute confidence: c_t(i) = max(P_t[i, :])
     b. Compute confidence delta: Δc(i) = c_t(i) - c_{t-1}(i)
     c. Compute top-k predictions with weights from P_t[i, :]
  3. Soft embedding injection for step t+1:
     e_i^{t+1} = (1 - alpha * c_t(i)) * e_[MASK] + alpha * c_t(i) * Σ_k P_t[i,k] * e_k
     where alpha ∈ [0, 1] is a single hyperparameter
  4. Unmasking order: unmask positions with highest c_t(i) first
     BUT: allocate extra steps to positions where |Δc(i)| is high (volatile positions need more refinement)

Key innovation: The confidence delta Δc guides adaptive compute allocation.
- Stable high confidence → unmask early (easy position)
- Increasing confidence → on track, unmask normally
- Volatile confidence (large |Δc|) → needs more steps, delay unmasking
- Consistently low confidence → may need remasking (integrate with RCR)
```

### Why This Works Where Simple Approaches Don't

- Vanilla Best-of-N: selects but doesn't learn from cross-step dynamics → ineffective (our iter history confirms)
- MetaState GRU: fixed update rule, needs training, adds parameters → brittle (43.75% vs vanilla 50.0%)
- DaL TTT: gradient-based updates, needs meta-training, gate problem → complex, unproven
- **CGPR**: zero parameters, zero training, uses information the model already computes. The key signal (confidence delta) is free — it's just the difference between two softmax outputs.

### Failure Modes

- Top-k embedding blending may introduce noise (averaged embeddings lose specificity)
- Confidence may not be well-calibrated in Dream-7B (high entropy distributions)
- The confidence delta signal may be too noisy to guide unmasking order reliably
- Alpha hyperparameter sensitivity

### Computational Cost

- Zero training cost
- Inference overhead: one extra softmax per step + embedding blend → ~1.05x overhead
- Pilot evaluation (GSM8K-200): ~3 GPU-hours
- Hyperparameter search (alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}): ~15 GPU-hours
- Full evaluation: ~20 GPU-hours
- **Success probability: 35%** (simple idea, strong motivation, but untested in DLM context)
- **Time estimate: 3-4 days**

### Key References

- Soft-Masking (arXiv 2510.17206). Embedding blending for retained masks.
- RCR (arXiv 2508.13148). Running confidence remasking.
- CoRe (arXiv 2602.04096). Context-robust remasking via perturbation.
- Where-to-Unmask (arXiv 2602.09501). Oracle unmasking order analysis.
- DUS (arXiv 2506.19037). Dilated unmasking for entropy-aware scheduling.

---

## Comparative Analysis and Recommendation

| Proposal | Novelty | Risk | Compute | Training | Time | Success P | Paper Quality |
|----------|---------|------|---------|----------|------|-----------|--------------|
| **1. EGF (Composition)** | Medium | Low | 96h GPU | None | 4-5d | **55%** | Medium-High |
| **2. DaL Gate Repair** | Low | Medium | 10h GPU | Yes | 2-3d | 30% | High (if works) |
| **3. COBA (Framework)** | Medium-High | Very Low | 35h GPU | None | 5-6d | **65%** | Medium-High |
| **4. CGPR (Confidence)** | High | Medium | 20h GPU | None | 3-4d | 35% | High (if works) |

### Pragmatist's Recommended Execution Plan

**The core principle: never have a GPU sitting idle, and always have a publishable fallback.**

```
Day 1 (All 4 GPUs):
  GPU 0: DaL Gate Repair (Proposal 2) — start training
  GPU 1: A-CFG baseline (Proposal 1, Phase 1)
  GPU 2: RCR + Vanilla baselines (Proposal 1, Phase 1)
  GPU 3: COBA grid search D-axis (Proposal 3) — D∈{32,64,128,256,512}, W=1

Day 2:
  GPU 0: DaL evaluation (if training done) or continue training
  GPU 1: SM + A-CFG composition (Proposal 1, Phase 2)
  GPU 2: CGPR pilot (Proposal 4) — alpha sweep on GSM8K-200
  GPU 3: COBA grid search W-axis — W∈{2,4,8,16}, D=128

Day 3 (Decision Gate):
  - DaL gate repair result available
  - Individual baselines available
  - COBA partial results available
  - CGPR pilot result available

  Decision:
  IF DaL >= vanilla + 2%:
    → Full commitment to DaL (Phase 1-2, 5 more days)
    → Paper: "Denoising-as-Learning" (DaL main story)
    → COBA + EGF as supplementary analysis

  ELIF CGPR >= vanilla + 3%:
    → Full commitment to CGPR (ablation + full benchmark, 4 more days)
    → Paper: "Confidence-Guided Progressive Refinement for DLM Inference"
    → EGF composition study as ablation

  ELIF EGF composition > best individual by >= 2%:
    → Full commitment to EGF (Phase 3 analysis, 3 more days)
    → Paper: "Ensemble Guidance: A Practitioner's Guide to DLM Inference-Time Scaling"
    → COBA as framework section

  ELSE:
    → Fall back to COBA + Diagnostic Study (Alternative D from proposal.md)
    → Paper: "Compute-Optimal Budget Allocation for Diffusion Language Models"
    → Include negative results from DaL/CGPR as case studies
    → This is the 65% success probability safety net

Days 4-7: Execute chosen track
Days 8-10: Paper writing + additional experiments
```

### Why This Plan Maximizes Expected Value

1. **No idle GPUs**: All 4 GPUs are productive from day 1.
2. **Parallel information gathering**: By day 3, we have 4 independent signals to guide the decision.
3. **Guaranteed publishable output**: COBA (65% success) is our safety net — it's a pure evaluation study with clear contribution regardless of other results.
4. **DaL gets its fair shot**: The gate repair is a 10-line fix. If it works, DaL is the highest-impact paper. We'd be foolish not to try.
5. **Training-free methods first**: 3 of 4 proposals require zero training. This front-loads low-risk progress.
6. **Incremental commits**: Each day produces usable data that either validates or invalidates a direction.

### Addressing Lessons from Evolution History

1. **"Contrarian was right about underestimating risks"**: Every proposal above includes explicit failure modes and decision triggers. The plan has a built-in pivot at day 3 — we don't wait until day 7 to discover something doesn't work.

2. **"First claims need verification"**: Proposal 1 (EGF) is explicitly positioned as "systematic composition study" — no novelty claims beyond the composition itself. Proposal 3 (COBA) is a framework paper — the contribution is the methodology, not a breakthrough result.

3. **"18 iterations of mostly negative results"**: The plan acknowledges this by making COBA the safety net. Even if DaL, CGPR, and EGF all fail, COBA produces a publishable characterization of *why* different compute allocation strategies work or don't work.

### Resource Budget Summary

| Resource | Available | Planned Usage |
|----------|-----------|---------------|
| GPU-hours (Days 1-3) | 288 (4 GPU × 72h) | ~180 (63%) |
| GPU-hours (Days 4-7) | 384 (4 GPU × 96h) | ~200 (52%) |
| Wall-clock time | 10 days | 7 days core + 3 days paper |
| Models needed | Dream-7B-Instruct | Already cached |
| External code | A-CFG, SM, RCR, ReMDM | All open-source with GitHub repos |
| Training data | GSM8K, OpenWebText | Already cached |
| Benchmarks | GSM8K, HumanEval, ARC-C, MBPP, Countdown | All cached |

---

## Appendix: Quick Reference — Open-Source Code Availability

| Method | Repository | Status | Effort to Integrate |
|--------|-----------|--------|-------------------|
| A-CFG | [pixeli99/A-CFG](https://github.com/pixeli99/A-CFG) | Released | Low (plug-in for Dream/LLaDA) |
| Soft-Masking | [IBM/soft-masked-diffusion-language-models](https://github.com/IBM/soft-masked-diffusion-language-models) | Released | Low (finetuned checkpoint available for Dream-7B) |
| RCR (MDPO) | [autonomousvision/mdpo](https://github.com/autonomousvision/mdpo) | Released | Low (inference-only plug-in) |
| ReMDM | [kuleshov-group/remdm](https://github.com/kuleshov-group/remdm) | Released | Low (sampler replacement) |
| Prism | [viiika/Prism](https://github.com/viiika/Prism) | Released | Medium (needs integration) |
| Dream-7B | [DreamLM/Dream](https://github.com/DreamLM/Dream) | Released | Already set up |
| dLLM Framework | [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) | Released | Medium (unified interface) |
| DUS | arXiv 2506.19037 | Paper only | Medium (re-implement from paper) |
| CJ-GRPO (EOSER+ASS) | [yjyddq/EOSER-ASS-RL](https://github.com/yjyddq/EOSER-ASS-RL) | Released | High (RL training required) |
| Eso-LMs | [s-sahoo.com/Eso-LMs](https://s-sahoo.com/Eso-LMs) | Released | High (different architecture) |
