# Methodology: Information-Guided Inference Scaling for Masked Diffusion Language Models

## Revision History
- **v3 (2026-03-11)**: Complete rewrite — dual-track IGGD + COBA primary, DaL conditional backup. Integrates post-pilot evidence (P1-P3, alt_a_pilot, baseline_remdm, P4 entropy) and 3 rounds of six-perspective debate.
- v2: DaL-primary with Alternative A fallback (superseded)
- v1: DaL-only (superseded)

## 1. Overview

This study investigates inference-time compute scaling for Masked Diffusion Language Models (MDLMs) through two co-equal research tracks and one conditional backup:

**Track A (IGGD)**: Information-Gain Guided Decoding — a training-free method that optimizes unmasking order by maximizing conditional mutual information. Pilot evidence: +5pp on GSM8K-100 (p=0.182) with zero overhead.

**Track C (COBA)**: Compute-Optimal Budget Allocation — systematic D x W grid search characterizing the compute-quality Pareto frontier for DLM inference. Independent publishable contribution.

**Track B (DaL, CONDITIONAL)**: Denoising-as-Learning — lightweight TTT layers in frozen DLM backbones. Proceeds ONLY if Phase 0 diagnostics pass (D0c: SSL-task alignment r>0.3, D0d: Delta_I > 0.1 bits/token).

### 1.1 Post-Pilot Evidence Summary (22+ iterations)

| Method | Type | Result | Iteration |
|--------|------|--------|-----------|
| Info-Gain Soft (tau=1.0) | Training-free (S-axis) | **+5pp** GSM8K-100 (p=0.182) | 5 |
| A-CFG | Training-free (S-axis) | +12.5pp GSM8K-16 (noisy, n=16) | 4 |
| ReMDM | Training-free (D-axis) | **-9pp** GSM8K-100, 1.8x overhead | 5 |
| DaL TTT-MLP | Training-based (M-axis) | -1.0pp GSM8K-200, gate=0.007 | 5 |
| MetaState-GRU | Training-based (M-axis) | -6.25pp GSM8K-16 (noisy) | 5 |
| Vanilla Dream-7B | Baseline | 34.0% GSM8K-100 | 5 |

**Structural conclusion**: Training-free methods providing extrinsic information succeed; training-based methods recycling endogenous information fail.

### 1.2 GPU Resources

- **4x NVIDIA RTX PRO 6000 Blackwell Server Edition** (98GB VRAM each)
- **Models**: Dream-7B-Instruct (7.6B, ~15GB bf16), LLaDA-8B-Instruct (8.0B, ~16GB bf16)
- **Batch size policy**: Maximize batch size to fully utilize 98GB VRAM (auto-detect via binary search)

## 2. Models & Baselines

### 2.1 Backbone Models

| Model | Parameters | Architecture | VRAM (bf16) | Notes |
|-------|-----------|-------------|-------------|-------|
| Dream-7B-Instruct | 7.6B | hidden=3584, layers=28, vocab=152064 | ~15GB | Primary backbone. Custom model class (`AutoModel`), bf16 attention mask required |
| LLaDA-8B-Instruct | 8.0B | hidden=4096, layers=32, vocab=126464 | ~16GB | Cross-backbone validation. Standard HuggingFace loading |

### 2.2 Methods Under Evaluation

**Track A — Training-Free Schedule Optimization:**

| Method | Description | Axes |
|--------|-------------|------|
| Vanilla | Standard random unmasking, 128 denoising steps | Baseline |
| IGGD-Soft | Information-gain soft selection (tau-controlled) | Schedule (S) |
| IGGD-Adaptive | Phase-aware tau schedule across mask ratios | Schedule (S) |
| A-CFG | Adaptive Classifier-Free Guidance (logit reweighting) | Guidance (G) |
| IGGD + A-CFG | Joint (tau, rho) composition | Schedule + Guidance |
| ReMDM | Principled remasking (training-free) | Depth (D) |

**Track C — COBA Framework:**

| Configuration | Description |
|--------------|-------------|
| D-scaling | D in {32, 64, 128, 256, 512}, W=1 |
| W-scaling | D=128, W in {1, 2, 4, 8, 16} |
| D x W grid | 5D x 5W = 25 configurations |
| Method overlay | Best IGGD config across D x W grid |

**Track B — Cross-Step Memory (Conditional):**

| Method | Description | Prerequisite |
|--------|-------------|-------------|
| DaL TTT-MLP | TTT-MLP fast weights, gate-repaired | D0c + D0d PASS |
| DaL TTT-Linear | TTT-Linear fast weights | D0c + D0d PASS |
| DaL Momentum | TTT-MLP + Titans-style momentum | D0c + D0d PASS |
| MetaState-GRU | GRU updater + cross-attention mixer | D0c + D0d PASS |

## 3. Benchmarks & Metrics

### 3.1 Evaluation Benchmarks

| Benchmark | Category | Metric | Pilot (n) | Full (n) | Selection |
|-----------|----------|--------|-----------|----------|-----------|
| GSM8K | Math Reasoning | Exact match accuracy | 100 | 1319 | — |
| HumanEval | Code Generation | pass@1 | 100 | 164 | — |
| ARC-Challenge | Commonsense | Accuracy | 100 | 1172 | — |
| MATH500 | Math Reasoning | Accuracy | — | 500 | Phase 2+ |
| MBPP | Code Generation | pass@1 | — | 500 | Phase 2+ |
| Countdown | Planning | Accuracy | — | 1000 | Phase 2+ |

**Evaluation note**: MDLMs use custom denoising pipelines; standard `lm_eval` harness cannot be used directly. All evaluation via custom scripts. For vanilla baselines on standard benchmarks, `lm_eval` with vLLM backend can accelerate if needed.

### 3.2 Metrics

**Primary**: Task benchmark accuracy (exact match for math/reasoning, pass@1 for code).

**Secondary**:
- Wall-clock time (seconds per sample)
- FLOPs (NOT NFE) — all comparisons FLOPs-fair
- Peak GPU memory (GB)
- Distinct-1/2/3 (degeneration check, every configuration)

**Diagnostic (Track A)**:
- Per-position conditional mutual information
- Entropy distribution across denoising steps
- Submodularity verification (diminishing returns check)

**Diagnostic (Track B, conditional)**:
- Gate value trajectory
- SSL-task Pearson correlation (D0c)
- Delta_I information-theoretic ceiling (D0d)
- Backbone-vs-TTT logit cosine similarity

### 3.3 Statistical Protocol

- **Paired bootstrap test**: n=1000 resamples, significance at alpha=0.05
- **Holm-Bonferroni correction**: For multi-condition comparisons (>=3 conditions)
- **Effect sizes**: Cohen's d alongside p-values
- **Pilot decisions (n=100)**: Relaxed to p<0.10 for directional GO/NO-GO
- **Signal confirmation (n=300)**: p<0.05 required for "confirmed"
- **Definitive (n=full)**: p<0.05 with 3 seeds (42, 123, 456), report mean +/- 95% CI

## 4. Track A: Information-Gain Guided Decoding (IGGD)

### 4.1 Core Algorithm

At each denoising step t with revealed set R_t and masked set M_t:

```
1. Forward pass -> logits L_t, probabilities P_t
2. For each masked position i in M_t:
   IG(i) ≈ H(X_i | X_{R_t})  (marginal entropy, computationally free)
3. Soft selection: sample k positions from:
   p(unmask i) ∝ exp(IG(i) / tau)
4. Unmask selected positions using backbone predictions
```

### 4.2 Adaptive Temperature Schedule

```
tau(r) = tau_base * schedule(r)
  r > 0.7:           tau_base * 2.0   (exploration)
  r in [0.3, 0.7]:   tau_base * 0.5   (exploitation in critical zone)
  r < 0.3:           tau_base * 1.0   (refinement)
```

### 4.3 Composition with A-CFG

IGGD selects which positions to unmask (schedule axis). A-CFG adjusts prediction logits for selected positions (guidance axis). Theoretically orthogonal. Joint search: {(tau, rho)} = {(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (2.0, 1.0)}.

### 4.4 Theoretical Grounding

- **Submodular optimization**: Information-gain f(S) is submodular. Greedy achieves (1-1/e) ≈ 0.63 approximation.
- **Turbo extrinsic information**: IGGD maximizes I(X_i; X_{M\{i}} | X_R) — extrinsic information transfer.
- **Free-energy minimization**: IGGD selects the action maximally reducing expected free energy.

## 5. Track C: COBA Framework

### 5.1 D x W Grid Search

```
D (denoising steps): {32, 64, 128, 256, 512}
W (parallel samples): {1, 2, 4, 8, 16}
Benchmarks: GSM8K-300, HumanEval-164, ARC-C-300
Selection: majority vote (reasoning), pass@1 (code)
```

### 5.2 Scaling Model

Fit regression: quality(D,W) = alpha*log(D) + beta*log(W) + gamma per task. Report R-squared and residuals. Plot Pareto frontiers at fixed FLOPs budgets.

### 5.3 Method Overlay

After establishing the D x W baseline grid, overlay best IGGD and A-CFG configurations to measure whether training-free methods shift the Pareto frontier.

## 6. Track B: Denoising-as-Learning (CONDITIONAL)

### 6.1 Phase 0 Prerequisites (Must PASS Before Any DaL Training)

**D0c (SSL-Task Alignment)**: Pearson r(SSL loss improvement, task accuracy improvement) across >=10 DaL configurations.
- r > 0.3 → PROCEED
- r [0.1, 0.3] → Try alternative SSL objectives (1 day budget)
- r < 0.1 → **KILL DaL**

**D0d (Information-Theoretic Ceiling)**: Linear probe predicting masked tokens from cross-step hidden states vs single-step states. Delta_I ≈ probe accuracy gap × log(vocab_size).
- Delta_I > 0.5 → Significant headroom
- Delta_I [0.1, 0.5] → Modest headroom (1-3% at best)
- Delta_I < 0.1 → **KILL all M-axis methods**

### 6.2 Architecture (Unchanged)

TTT layer inserted at backbone midpoint (Dream: layer 14/28). Gate init revised from sigmoid(-5)=0.007 to sigmoid(-2)=0.12. Independent gate_lr = 10x meta_lr.

### 6.3 Phase-Transition Scheduling

Active zone: mask ratio [0.1, 0.7]. Skip TTT at r > 0.7 (P2: SNR degrades from 0.009 at r=0.6 to 0.002 at r=0.9).

## 7. Experimental Phases

### Phase 0: Diagnostics + IGGD Verification (Day 1, All 4 GPUs)

| GPU | Task | Goal | Time |
|-----|------|------|------|
| 0 | IGGD n=300 verification (GSM8K) | Confirm +5pp signal at scale | 4-6h |
| 1 | Vanilla baselines (GSM8K-300, ARC-C-300, HumanEval-164) | Establish rigorous baselines | 4-6h |
| 2 | D0c (10+ DaL configs) + D0d (Delta_I probe) | SSL-task alignment + info ceiling | 6-8h |
| 3 | COBA D-axis grid (D={32,64,128,256,512}, W=1) | D-scaling characterization | 6-8h |

**Decision Gate 1 (Day 1 end)**:

| IGGD n=300 | D0c | D0d | Decision |
|:----------:|:---:|:---:|:---------|
| >=+3pp, p<0.05 | r>0.3 | >0.1 | Dual track: IGGD primary + DaL secondary |
| >=+3pp, p<0.05 | r<0.3 | Any | **KILL DaL**. IGGD + COBA primary |
| >=+3pp, p<0.05 | Any | <0.1 | IGGD + COBA + Diagnostic study |
| <+1pp | Any | Any | COBA framework + Diagnostic study |

### Phase 1: Track A Deepening + Track B Start (Days 2-4)

**Track A (IGGD, unconditional)**:

| ID | Experiment | Time |
|----|-----------|------|
| A1 | Adaptive tau sweep (4 schedules x GSM8K-100) | 4h |
| A2 | A-CFG + IGGD joint (tau, rho) search (4 configs x GSM8K-100) | 4h |
| A3 | Full benchmark: best IGGD on GSM8K-full + HumanEval + ARC-C | 8h |
| A4 | IGGD + Soft-Masking composition study | 4h |

**Track B (DaL, conditional on D0c + D0d PASS)**:

| ID | Experiment | Time |
|----|-----------|------|
| B1 | Gate repair (sigmoid(-2), gate_lr=10x) + 5K training | 8h |
| B2 | GSM8K-200 evaluation (GO/NO-GO gate) | 3h |

### Phase 2: Main Experiments (Days 5-8)

**Track A + C (continues regardless)**:

| ID | Experiment | Time |
|----|-----------|------|
| A5 | Per-step conditional MI measurement (mechanistic) | 4h |
| A6 | COBA W-axis + M-axis grid completion | 12h |
| A7 | Cross-backbone: IGGD on LLaDA-8B | 6h |
| A8 | Method composition: A-CFG + IGGD + RCR triple stack | 4h |

**Track B (if B2 passes: >=+2% at p<0.10)**:

| ID | Experiment | Time |
|----|-----------|------|
| B3 | Update rule ablation (GRU vs Linear vs MLP vs Momentum) | 24h |
| B4 | FLOPs-fair: Dense Denoising vs DaL at matched FLOPs | 8h |
| B5 | Cross-backbone: DaL on LLaDA-8B | 8h |

### Phase 3: Integration & Paper (Days 9-12)

| ID | Experiment | Time |
|----|-----------|------|
| C1 | Full D x W grid with method overlay (25 + method combos) | 12h |
| C2 | SOTA comparison table (all methods, all benchmarks, 3 seeds) | 16h |
| C3 | Qualitative analysis + figure generation | 4h |
| C4 | Latency & throughput benchmarking | 4h |

## 8. Expected Visualizations

### Tables
- **Table 1**: Main IGGD results — IGGD variants vs Vanilla vs A-CFG vs ReMDM on all benchmarks
- **Table 2**: Method composition — A-CFG × IGGD × RCR interaction matrix
- **Table 3**: COBA D × W grid — accuracy at each (D, W) point per task
- **Table 4**: Scaling model fit — regression coefficients alpha, beta, gamma per task, R-squared
- **Table 5**: D0c diagnostic — SSL loss vs task accuracy correlation across configs (publishable regardless)
- **Table 6** (conditional): DaL update rule comparison across benchmarks

### Figures
- **Figure 1**: Architecture diagram — IGGD algorithm flow (entropy computation → soft selection → unmasking)
- **Figure 2**: COBA Pareto frontier — accuracy vs FLOPs for each task (D and W axes labeled)
- **Figure 3**: IGGD per-step conditional MI — information gain measurement across denoising
- **Figure 4**: Adaptive tau schedule — tau vs mask ratio with accuracy overlay
- **Figure 5**: Method composition interaction — heatmap of (tau, rho) pairs
- **Figure 6**: Entropy distribution — bimodal structure at different mask ratios (extending P4 evidence)
- **Figure 7**: Extrinsic vs Endogenous taxonomy — visual classification of all methods with evidence arrows
- **Figure 8** (conditional): Gate value trajectory + D0c scatter plot
- **Figure 9** (conditional): DaL compute-accuracy tradeoff curves

## 9. Shared Resources

| Resource | Type | Path (remote) | Status |
|----------|------|---------------|--------|
| Dream-7B-Instruct | Checkpoint | shared/checkpoints/dream-7b-instruct | Downloaded, verified |
| LLaDA-8B-Instruct | Checkpoint | shared/checkpoints/llada-8b-instruct | Downloaded, verified |
| GSM8K | Dataset | shared/datasets/gsm8k | Downloaded |
| MATH500 | Dataset | shared/datasets/math500 | Downloaded |
| HumanEval | Dataset | shared/datasets/humaneval | Downloaded |
| MBPP | Dataset | shared/datasets/mbpp | Downloaded |
| ARC-Challenge | Dataset | shared/datasets/arc_challenge | Downloaded |
| Countdown | Dataset | shared/datasets/countdown | Downloaded (1K synthetic) |
| OpenWebText (10K) | Dataset | shared/datasets/openwebtext_10k | Downloaded (streaming subset) |

## 10. Implementation Notes

### 10.1 Evaluation
- MDLMs require custom denoising evaluation scripts (cannot use standard lm_eval directly)
- All custom evaluation in `exp/code/` — maintained per-task parsers for exact match and pass@1
- For vanilla baseline cross-checks, lm_eval with vLLM backend can optionally verify

### 10.2 Memory Management
- 98GB VRAM per GPU — ample for single-GPU inference with 7-8B models
- IGGD: zero overhead (entropy computed from existing logits)
- COBA W-scaling: W parallel samples fit in single GPU up to W=4 for 7B model; W>=8 uses sequential batching
- DaL K=4 unrolling + gradient checkpointing: ~30-40GB

### 10.3 Batch Size Strategy
- **Directive**: maximize batch size to fully utilize 98GB VRAM
- **Auto-detect**: binary search for max batch size before each run
- **Gradient accumulation**: for training tasks needing larger effective batch sizes

### 10.4 Reproducibility
- All experiments: seed 42 (primary), seeds 123/456 for variance estimation
- Package versions: PyTorch 2.10.0+cu128, Transformers 4.57.6, Datasets 4.7.0
- Conda env: sibyl_ttt-dlm (remote server)
- All configs saved alongside results

## 11. Risk Assessment

| Risk | P(risk) | Impact | Mitigation |
|------|---------|--------|------------|
| IGGD +5pp doesn't replicate at n=300 | 40% | High | COBA framework + diagnostic study still publishable |
| A-CFG + IGGD composition sub-additive | 30% | Medium | Sub-additivity is itself a publishable finding |
| D0c confirms SSL-task misalignment | 35% | Critical for DaL only | DaL killed; IGGD + COBA unaffected |
| Delta_I < 0.1 | 40% | Critical for DaL only | Powerful negative finding; diagnostic contribution |
| All methods < vanilla at n=300 | 10% | Critical | COBA + Diagnostic study (Alternative D) |
| COBA log-linear model poor fit (R²<0.5) | 25% | Medium | Use non-parametric Pareto analysis instead |

**Paper scenarios by probability**:

| Scenario | P | Paper |
|----------|:-:|-------|
| IGGD works + composition enhances | 40% | "Information-Gain Guided Decoding for DLMs" (NeurIPS) |
| IGGD works + DaL works | 10% | "Dual-Track Inference Scaling" (NeurIPS/ICML) |
| IGGD marginal + COBA strong | 30% | "Compute-Optimal Budget Allocation for DLMs" (ICML) |
| IGGD works + DaL fails (diagnostic) | 25% | "Extrinsic vs Endogenous: What Works for DLM Scaling" (NeurIPS) |
| All fail + diagnostic deep | 15% | "Why Inference-Time Scaling Remains Elusive for DLMs" (EMNLP) |

**Aggregate P(publishable paper) >= 80%**
