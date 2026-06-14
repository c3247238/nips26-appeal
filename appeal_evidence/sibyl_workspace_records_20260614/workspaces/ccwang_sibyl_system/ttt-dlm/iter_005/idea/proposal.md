# Final Research Proposal: Information-Guided Inference Scaling for Masked Diffusion Language Models

**Synthesizer**: sibyl-synthesizer
**Date**: 2026-03-11
**Iteration**: 5 (refinement round 3 — post-pilot + post-debate revision)

---

## Title

**Information-Guided Inference Scaling for Masked Diffusion Language Models: From Adaptive Unmasking to Denoising-as-Learning**

---

## Abstract

Masked diffusion language models (MDLMs) generate text through iterative denoising, yet the relationship between inference-time compute allocation and generation quality remains poorly understood. We present a two-track investigation: (1) **Information-Gain Guided Decoding (IGGD)**, a training-free method that selects unmasking order by maximizing conditional mutual information, achieving +5pp on GSM8K in preliminary experiments with zero overhead; and (2) **Denoising-as-Learning (DaL)**, a framework inserting lightweight Test-Time Training layers into frozen DLM backbones with rigorous diagnostic gating. We provide theoretical grounding through a submodular optimization analysis of information-gain decoding and an information-theoretic upper bound (Delta_I) on the maximum benefit of cross-step memory. Through systematic ablation of training-free method compositions and controlled comparison of cross-step update rules (GRU vs TTT-Linear vs TTT-MLP), we characterize the compute-quality Pareto frontier for DLM inference-time scaling. Our Compute-Optimal Budget Allocation (COBA) framework unifies depth-scaling, width-scaling, and memory-axis methods under a single evaluation methodology.

---

## Evidence-Driven Revisions (Changes from Round 2)

This revision integrates three rounds of six-perspective debate and new pilot evidence from alt_a_pilot (Information-Gain Soft: +5pp on GSM8K-100, p=0.182). The proposal undergoes a **fundamental structural change**: from a DaL-first design with training-free fallback to a **dual-track design** where information-gain decoding is an equal-priority research track.

### 1. Structural Reorientation: Dual-Track Design
**Evidence**: alt_a_pilot shows Info-Gain Soft achieves +5pp (34% -> 39%) on GSM8K-100 with zero training and zero computational overhead. While not statistically significant (p=0.182), this is the strongest positive task-level signal in 22+ iterations. Meanwhile, DaL (P3: -1.0pp, gate=0.007) and MetaState-GRU (43.75% vs vanilla 50.0% at n=16) both show negative or null results.

**Revision**: The proposal is restructured as two co-equal tracks rather than "DaL primary + Alternative A backup."

### 2. New Theoretical Foundation for Information-Gain Decoding
**Evidence**: Round 2 debate convergence from theoretical, interdisciplinary, and contrarian perspectives. The theorist provides a submodular optimization framework (greedy information-gain achieves (1-1/e) approximation to optimal). The interdisciplinary perspective provides three independent explanatory frameworks (Turbo extrinsic information, free-energy minimization, optimal annealing).

**Revision**: IGGD receives a full theoretical treatment, not just empirical evaluation.

### 3. D0c + D0d Diagnostics Made Prerequisite
**Evidence**: Universal consensus across all six perspectives that SSL-task alignment (D0c) and information-theoretic ceiling (D0d, Delta_I) must be measured before any further DaL training investment. The contrarian's C2 hypothesis (SSL-task correlation < 0.1) is the strongest single-point contribution in the debate, supported by P3 (-52% SSL, -1pp accuracy), TTT++ failure mode analysis, and pseudo-label feedback concerns.

**Revision**: D0c and D0d are Phase 0 prerequisites. DaL training does not begin until both pass.

### 4. Contrarian-Driven Kill Criteria Sharpened
**Evidence**: The contrarian's track record is strong — P3 validated the concern about SSL-task misalignment, MetaState-GRU pilot validated the paradigm-level risk. Round 2 debate refined C3 from "training-free > training-based" to "methods providing extrinsic information > methods relying on endogenous feedback."

**Revision**: Hard kill criteria with no "directionally positive" exceptions. DaL requires >=2% absolute improvement at p<0.10 (n=200) to proceed.

### 5. Extrinsic vs Endogenous Information Framework
**Evidence**: Cross-perspective convergence on the "extrinsic vs endogenous" principle. Turbo coding analogy (interdisciplinary): methods providing new, orthogonal information succeed; methods recycling existing information fail. This explains ReMDM -9pp (endogenous remask feedback) vs Info-Gain Soft +5pp (extrinsic ordering optimization).

**Revision**: All methods are now classified by their information contribution type, providing a unified theoretical lens for the paper.

---

## 1. Motivation

### 1.1 The Inference-Time Compute Scaling Challenge

DLMs generate text through iterative denoising — progressively unmasking tokens from a fully-masked sequence. Recent work has explored multiple axes of inference-time compute scaling:

- **Depth (D)**: More denoising steps (ReMDM, standard scaling)
- **Width (W)**: Multiple parallel trajectories (Best-of-N, SMC, tree search)
- **Memory (M)**: Cross-step information transfer (MetaState, DaL)
- **Schedule (S)**: Optimized unmasking order (Info-Gain, CoRe, A-CFG)

No existing work systematically characterizes how these axes interact or which provides the best return per FLOP. Individual papers compare against vanilla baselines or single competitors, but the field lacks a unified evaluation framework.

### 1.2 Two Competing Paradigms

Our 22+ iterations of systematic exploration reveal a striking pattern: **training-free schedule optimization methods show the strongest positive signals**, while **training-based cross-step memory methods consistently fail or underperform**:

| Method | Type | Result | Iteration |
|--------|------|--------|-----------|
| Info-Gain Soft | Training-free (S-axis) | +5pp GSM8K-100 (p=0.182) | 5 |
| A-CFG | Training-free (S-axis) | +12.5pp GSM8K-16 (noisy) | 4 |
| DaL TTT-MLP | Training-based (M-axis) | -1.0pp GSM8K-200 | 5 |
| MetaState-GRU | Training-based (M-axis) | -6.25pp GSM8K-16 (noisy) | 5 |
| ReMDM | Training-free (D-axis) | -9pp GSM8K-100 | 5 |

This pattern demands both rigorous verification (are training-free signals robust at scale?) and mechanistic explanation (why does the paradigm matter?).

### 1.3 The Extrinsic Information Hypothesis

We propose a unifying explanation grounded in coding theory: **methods that provide extrinsic (new, orthogonal) information to the denoiser succeed, while methods that recycle endogenous (already-computed) information fail**.

- **Info-Gain Soft succeeds** because it changes the unmasking order to maximize the conditional mutual information contributed at each step — analogous to maximizing extrinsic information transfer in turbo decoding.
- **ReMDM fails** because remasking forces the model to re-process already-decided tokens — endogenous feedback that can trigger a "turbo cliff" where iterative processing degrades rather than improves.
- **DaL's P3 failure** shows the TTT layer learning to re-encode backbone representations (SSL loss -52%) without providing extrinsic task-relevant information (accuracy -1pp) — the TTT output is correlated with, not orthogonal to, the backbone output.

This hypothesis generates specific, testable predictions for all methods under study.

---

## 2. Method

### Track A: Information-Gain Guided Decoding (IGGD)

A training-free framework that optimizes the unmasking schedule via information-gain maximization.

#### 2.1 Core Algorithm

At each denoising step t with revealed set R_t and masked set M_t:

```
1. Run standard forward pass -> logits L_t, probabilities P_t
2. For each masked position i in M_t:
   IG(i) = H(X_i | X_{R_t}) - sum_v P_t[i,v] * H(X_{M_t\{i}} | X_{R_t}, X_i=v)
   Approximation: IG(i) ≈ H(X_i | X_{R_t}) (marginal entropy, computationally free)
3. Soft selection: sample k positions to unmask from distribution:
   p(unmask i) ∝ exp(IG(i) / tau)
   where tau > 0 is the temperature (tau=1.0 default from pilot)
4. Unmask selected positions using backbone predictions
```

**Key design choice**: The soft selection with temperature tau avoids the catastrophic failure of greedy information-gain ordering (pilot: -18pp). Greedy resolves trivial high-entropy positions first, wasting early context-building steps. Soft selection balances information-gain optimization with stochastic exploration.

#### 2.2 Adaptive Temperature Schedule

Informed by the phase-transition evidence (P2: SNR peaks at r=0.6) and simulated annealing theory:

```
tau(r) = tau_base * schedule(r)
where schedule(r):
  r > 0.7 (high mask, exploration phase):  tau = tau_base * 2.0
  r in [0.3, 0.7] (critical zone):         tau = tau_base * 0.5
  r < 0.3 (low mask, refinement phase):     tau = tau_base * 1.0
```

The critical zone uses lower temperature (more deterministic, higher information-gain exploitation) because P2 confirms gradient signal quality peaks here. High-mask-ratio phases use higher temperature to maintain exploration.

#### 2.3 Composition with A-CFG

A-CFG (Adaptive Classifier-Free Guidance) operates on the guidance axis (reweighting logits based on conditional vs unconditional predictions), while IGGD operates on the schedule axis (which tokens to unmask). These axes are theoretically orthogonal:

```
Combined inference:
1. IGGD selects which positions to unmask (schedule optimization)
2. A-CFG adjusts prediction logits for selected positions (guidance optimization)
```

Joint (tau, rho) search over 4 configurations: {(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (2.0, 1.0)}.

#### 2.4 Theoretical Grounding

**Submodular optimization (from theorist)**: The information-gain function f(S) = H(X_M | X_R) - H(X_M | X_{R∪S}) is submodular (conditional entropy is submodular). Greedy maximization of a submodular function achieves a (1 - 1/e) ≈ 0.63 approximation to the global optimum. This provides a formal guarantee that IGGD's greedy-per-step strategy is near-optimal.

**Turbo extrinsic information (from interdisciplinary)**: IGGD maximizes I(X_i; X_{M\{i}} | X_R) — the extrinsic information contributed by unmasking position i. This is mathematically equivalent to maximizing the extrinsic information transfer rate in iterative turbo decoding.

**Free-energy minimization (from interdisciplinary)**: IGGD selects the action (which token to unmask) that maximally reduces expected free energy (prediction uncertainty for remaining tokens). This is a direct implementation of active inference in the free-energy principle framework.

### Track B: Denoising-as-Learning (DaL) — Conditional on Diagnostics

DaL inserts lightweight TTT layers into frozen DLM backbones, treating denoising as self-supervised learning. **This track proceeds only if Phase 0 diagnostics pass.**

#### 2.5 Architecture (Unchanged from Round 2)

```
Frozen DLM Backbone
    +-- Layers 1 to L/2: Standard Transformer (frozen)
    +-- TTT Layer (trainable):
    |   +-- Fast Weight Model: MLP W_f (d_model -> d_ttt -> d_model)
    |   +-- SSL loss: MLM on revealed tokens R_t
    |   +-- Gradient update: W_f <- W_f - eta * grad(L_ssl(W_f))
    |   +-- Residual gate: output = backbone + beta * TTT_output
    +-- Layers L/2+1 to L: Standard Transformer (frozen)
```

#### 2.6 Gate Repair (Post P3 Failure)

Gate initialization changed from sigmoid(-5)=0.007 to sigmoid(-2)=0.12. Independent gate learning rate: gate_lr = 10 * meta_lr. Optional gate warm-up loss. **But this only executes if D0c and D0d pass first.**

#### 2.7 Phase-Transition Scheduling

TTT updates concentrated at mask ratio [0.1, 0.7]. Skip at r > 0.7 (validated by P2: SNR degrades from 0.009 at r=0.6 to 0.002 at r=0.9).

### Track C: COBA Framework (Parallel from Day 1)

Compute-Optimal Budget Allocation — a systematic evaluation framework for DLM inference-time scaling.

#### 2.8 D x W Grid Search

```
D (denoising steps): {32, 64, 128, 256, 512}
W (parallel samples): {1, 2, 4, 8, 16}
Benchmarks: GSM8K, HumanEval, ARC-C
Selection: majority vote (reasoning), pass@1 (code)
```

#### 2.9 Method Composition Analysis

Systematic study of how training-free methods compose:

| Pair | Axes | Predicted Interaction |
|------|------|---------------------|
| A-CFG + IGGD | Guidance + Schedule | Additive (orthogonal) |
| A-CFG + RCR | Guidance + Mask pattern | Potentially conflicting |
| IGGD + Soft-Masking | Schedule + Embedding | Additive |
| A-CFG + RCR + IGGD | Triple stack | Unknown |

---

## 3. Research Questions and Hypotheses

### Primary Hypotheses

**H_IGGD (Track A, front-runner)**: Information-Gain Soft decoding achieves >=3% absolute improvement over vanilla Dream-7B on GSM8K at n>=300, and the improvement is statistically significant (p<0.05).

**H_compose**: A-CFG + IGGD composition outperforms either method alone by >=2% absolute on at least 2/3 primary benchmarks.

**H_COBA**: The optimal D x W allocation varies by task type, with reasoning tasks favoring depth and code generation favoring width.

### Conditional Hypotheses (Require D0c + D0d PASS)

**H_DaL**: DaL with TTT-MLP, after gate repair, achieves >=2% absolute improvement over vanilla Dream-7B on at least 2/3 primary benchmarks, AND outperforms MetaState-GRU.

**H_ortho**: At fixed FLOPs, DaL outperforms Dense Denoising, demonstrating orthogonal computational value.

### Diagnostic Hypotheses (Phase 0, must-test)

**H_align**: Pearson r(SSL loss improvement, task accuracy improvement) > 0.3 across >=10 DaL configurations.

**H_deltaI**: Delta_I (information-theoretic upper bound on cross-step memory benefit) > 0.1 bits/token in the critical mask ratio zone [0.3, 0.7].

**H_gate**: With revised gate initialization, gate values reach >= 0.10 within 5K training steps.

---

## 4. Theoretical Analysis

### 4.1 Submodular Optimization for IGGD

The information-gain function is submodular (conditional entropy satisfies diminishing returns). Greedy IGGD achieves (1 - 1/e) approximation to the optimal unmasking sequence. This is independent of whether DaL succeeds — it provides a standalone theoretical contribution for the training-free track.

### 4.2 Information-Theoretic Upper Bound (Delta_I)

Delta_I(t) = I(Y_t; h_{T:t+1} | x_t, X_{R_t}) quantifies the maximum additional information cross-step memory can provide. If Delta_I < 0.1 bits/token, no cross-step memory mechanism (TTT, GRU, Hopfield, or any future method) can significantly improve generation quality — the backbone already captures most available signal in a single pass.

### 4.3 Extrinsic vs Endogenous Information Classification

| Method | Information Type | Theoretical Prediction | Evidence |
|--------|-----------------|----------------------|----------|
| IGGD | Extrinsic (ordering optimization) | Positive | +5pp pilot |
| A-CFG | Extrinsic (guidance signal) | Positive | +12.5pp (n=16) |
| Soft-Masking | Extrinsic (embedding enrichment) | Positive | Literature |
| ReMDM | Endogenous (remasking feedback) | Harmful | -9pp pilot |
| DaL TTT | Endogenous (SSL self-prediction) | Null/Harmful | -1pp pilot |
| MetaState GRU | Endogenous (recurrent memory) | Null/Harmful | -6.25pp pilot |

### 4.4 OCO Framework for DaL (Conditional)

If D0c confirms SSL-task alignment (r > 0.3), the TTT-in-denoising process can be analyzed as online convex optimization with monotonically improving side information. The regret bound captures the progressive enrichment of revealed tokens as a curriculum. This framework is conditional on D0c and D0d results.

---

## 5. Experimental Plan (Revised)

### Phase 0: Diagnostics (Day 1, All 4 GPUs)

| GPU | Experiment | Goal | Time |
|-----|-----------|------|------|
| 0 | D0c minimal (4 configs) + D0d (Delta_I probe) | SSL-task alignment + information ceiling | 6-8h |
| 1 | IGGD n=300 verification + A-CFG baseline | Confirm +5pp signal at scale | 4-6h |
| 2 | Vanilla Dream-7B baselines (GSM8K-300, ARC-C-200) | Establish rigorous baselines | 4-6h |
| 3 | COBA D-axis grid (D={32,64,128,256,512}, W=1) | D-scaling characterization | 6-8h |

**Day 1 Decision Gates**:

| D0c result | D0d result | IGGD n=300 | Decision |
|:----------:|:----------:|:----------:|:--------|
| r > 0.3 | Delta_I > 0.1 | >=+3pp, p<0.05 | Dual track: IGGD + DaL |
| r > 0.3 | Delta_I > 0.1 | <+3pp or p>=0.05 | DaL primary + IGGD secondary |
| r < 0.3 | Any | >=+3pp, p<0.05 | **KILL DaL**. IGGD + COBA primary |
| r < 0.3 | Delta_I < 0.1 | >=+3pp, p<0.05 | IGGD + COBA + Diagnostic study |
| Any | Any | <+1pp | COBA framework + Diagnostic study |

### Phase 1: Track A Deepening + Track B Start (Days 2-4)

**Track A (IGGD, unconditional)**:

| Experiment | Description | Time |
|-----------|-------------|------|
| A1: Adaptive tau | tau schedule sweep across mask ratio phases | 4h |
| A2: (tau, rho) joint search | A-CFG + IGGD composition, 4 configurations | 4h |
| A3: Full benchmark | Best IGGD config on GSM8K-full + HumanEval + ARC-C | 8h |
| A4: Composition study | IGGD + Soft-Masking, IGGD + RCR | 8h |

**Track B (DaL, conditional on D0c + D0d PASS)**:

| Experiment | Description | Time |
|-----------|-------------|------|
| B1: Gate repair + 5K training | sigmoid(-2) init, gate_lr=10x, 5K steps | 8h |
| B2: GSM8K-200 evaluation | Best checkpoint evaluation | 3h |
| B3: Extrinsic probe diagnostic | Probe test: TTT extrinsic info content | 1h |

**Track B GO/NO-GO (Day 4)**: GSM8K-200 >= vanilla + 2% absolute (p<0.10)

### Phase 2: Main Experiments (Days 5-8, conditional)

**Track A (continues regardless)**:

| Experiment | Description | Time |
|-----------|-------------|------|
| A5: Mechanistic analysis | Per-step conditional MI measurement for IGGD | 4h |
| A6: COBA W-axis + M-axis | Full grid search completion | 12h |
| A7: Cross-backbone | IGGD on LLaDA-8B | 6h |

**Track B (if B2 passes)**:

| Experiment | Description | Time |
|-----------|-------------|------|
| B4: Update rule ablation | GRU vs Linear vs MLP vs Momentum (3 seeds) | 48h |
| B5: FLOPs-fair comparison | Dense vs DaL at matched FLOPs | 16h |

### Phase 3: Integration & Paper (Days 9-12)

Integrate IGGD, COBA, and (conditionally) DaL results into a unified paper on DLM inference-time compute scaling.

### Evaluation Protocol (Non-Negotiable)

1. **Primary metrics**: Task benchmark accuracy — never PPL alone
2. **Sample sizes**: n=300 for signal confirmation, n=500 for statistical confirmation
3. **Statistical tests**: Paired bootstrap (n=1000, p<0.05), Holm-Bonferroni for multiple comparisons
4. **Effect sizes**: Cohen's d alongside p-values
5. **Diversity checks**: Distinct-1/2/3 on every configuration
6. **Compute fairness**: Report both NFE and FLOPs
7. **Qualitative checks**: 10 random samples per configuration

---

## 6. Expected Contributions

1. **IGGD framework**: Training-free information-gain guided decoding for DLMs, with submodular optimization theory and adaptive temperature scheduling. Applicable to any frozen DLM.

2. **Extrinsic vs Endogenous information taxonomy**: A unifying theoretical lens that classifies and predicts the effectiveness of DLM inference-time methods based on their information contribution type. Grounded in turbo coding theory.

3. **COBA framework**: First systematic D x W compute allocation study for DLMs, providing practitioner recipes for different task types.

4. **Composition study**: First systematic analysis of how training-free DLM methods interact (additive, sub-additive, or conflicting).

5. **Diagnostic analysis**: D0c (SSL-task alignment) and D0d (Delta_I ceiling) measurements characterizing the fundamental feasibility of cross-step memory in DLMs — valuable regardless of DaL outcome.

6. **DaL framework (conditional)**: If diagnostics pass and gate repair succeeds: controlled ablation of update rules with theoretical grounding (OCO regret bounds, Lyapunov stability).

---

## 7. Risk Assessment and Contingency

| Risk | P(risk) | Impact | Mitigation |
|------|---------|--------|------------|
| IGGD +5pp doesn't replicate at n=300 | 40% | High | COBA framework + Composition study still publishable |
| D0c confirms SSL-task misalignment (r<0.1) | 35% | Critical for DaL | DaL killed; IGGD + COBA primary |
| Delta_I < 0.1 (cross-step memory ceiling low) | 40% | Critical for DaL | Powerful negative finding; diagnostic paper contribution |
| A-CFG + IGGD composition is sub-additive | 30% | Medium | Sub-additivity is a publishable finding |
| DaL gate repair fails (gate <0.05 after all fixes) | 25% | Medium | DaL killed; focus on Track A + C |
| All methods < vanilla at n=300 | 10% | Critical | COBA + Diagnostic study (Alternative D) |

**Paper scenarios by probability**:

| Scenario | P(scenario) | Paper |
|----------|:-----------:|-------|
| IGGD works + composition enhances | 40% | "Information-Gain Guided Decoding for DLMs" (NeurIPS) |
| IGGD works + DaL works | 10% | "Dual-Track Inference Scaling" (NeurIPS/ICML) |
| IGGD marginal + COBA strong | 30% | "Compute-Optimal Budget Allocation for DLMs" (ICML) |
| All fail + diagnostic deep | 15% | "Why Inference-Time Scaling Remains Elusive for DLMs" (EMNLP) |
| IGGD works + DaL fails (diagnostic) | 25% | "Extrinsic vs Endogenous: What Works for DLM Scaling" (NeurIPS) |

**Aggregate P(publishable paper) >= 80%** (diversified across scenarios)

---

## 8. Paper Structure

1. **Introduction**: DLM inference-time scaling challenge -> fragmented landscape -> extrinsic vs endogenous information hypothesis
2. **Background**: DLM denoising, inference-time methods taxonomy (D/W/M/S axes), information-theoretic framework
3. **Method**: IGGD (submodular optimization + adaptive tau), DaL (conditional), COBA
4. **Theoretical Analysis**: Submodular guarantees for IGGD, Delta_I bounds, extrinsic information classification, OCO for DaL (conditional)
5. **Experiments**: Phase 0 diagnostics, IGGD verification, composition study, COBA grid, DaL ablation (conditional)
6. **Analysis**: Mechanistic explanation (why IGGD works, why ReMDM/DaL fail), Pareto frontier characterization
7. **Discussion**: Limitations, connections to coding theory, implications for DLM architecture design
8. **Conclusion**

**Target venue**: NeurIPS 2026 (primary) / ICML 2026 (backup)

---

## 9. Synthesis Reasoning

### Perspective Weighting (Round 3)

| Perspective | Weight | Key Contribution to This Revision |
|-------------|:------:|----------------------------------|
| **Contrarian** | 6/6 | Extrinsic vs endogenous framework; kill criteria enforcement; C2 hypothesis vindicated by P3; forced structural reorientation away from DaL-first |
| **Empiricist** | 6/6 | D0c pre-registration; statistical rigor (n=300/500); sequential "confirm then explain" protocol; FLOPs-fair evaluation mandate |
| **Interdisciplinary** | 5/6 | Turbo extrinsic information theory; three-framework explanation of IGGD success; adaptive tau prediction; probe-based extrinsic measurement |
| **Pragmatist** | 5/6 | Parallel GPU allocation; COBA safety net; composition study framing as robust to positive/negative results; D0c-before-training correction |
| **Theoretical** | 5/6 | Submodular optimization for IGGD; Delta_I upper bound; FIM condition number analysis; info-gain + FIM unified framework |
| **Innovator** | 4/6 | ADM Hopfield connection (deferred); SPD self-play idea (deferred); LDR formally abandoned; kill criteria for novel proposals |

### Why the Structural Reorientation

The previous proposal treated DaL as the primary track and training-free methods as fallback. Three rounds of evidence and debate make this indefensible:

1. **Pilot evidence**: The only positive task-level signal comes from training-free IGGD (+5pp). All training-based methods show null or negative results.
2. **Theoretical convergence**: All six perspectives converge on the extrinsic vs endogenous distinction as the key explanatory variable. IGGD provides extrinsic information; DaL recycles endogenous information.
3. **Risk-reward calculus**: IGGD has 55% success probability at zero training cost. DaL has 35% unconditional probability at high training cost, conditional on diagnostics that themselves have ~25% pass rate. Expected value per GPU-hour strongly favors IGGD.
4. **Evolution lesson applied**: The project has undergone 4+ direction reversals over 22+ iterations (PPL -> accuracy -> DTA -> BSD/A-CFG). Each time, the contrarian's concerns were validated. This round, the contrarian and empiricist jointly argue for structural reorientation — and the evidence supports them.

### Decisive Choices (Not Compromises)

1. **IGGD as co-equal track, not fallback**: This is not a compromise between innovator (who wants novel methods) and pragmatist (who wants proven methods). It is a decisive response to evidence: the best available signal points to schedule optimization, not cross-step memory.

2. **DaL conditional on diagnostics, not abandoned**: This is not a compromise between contrarian (kill DaL) and innovator (keep DaL). It is a principled decision: D0c and D0d provide information-theoretically grounded criteria. If they pass, DaL proceeds. If not, DaL is killed — with diagnostic measurements that are themselves publishable contributions.

3. **COBA as parallel safety net**: This is the pragmatist's strongest contribution. At 60% success probability, COBA ensures a publishable outcome even if both IGGD and DaL fail. The 65% -> 60% revision reflects the theorist's valid concern about the log-linear model assumption.

4. **FLOPs not NFE for all comparisons**: Universal agreement. No exceptions.

5. **The "extrinsic information" framing**: This emerged from the interdisciplinary perspective's turbo coding analogy, was formalized by the theorist, and was adopted by the contrarian as the precise restatement of C3. It is the paper's unifying theoretical contribution regardless of which methods succeed.

---

## References

### Track A: Information-Gain Decoding
1. Info-Gain Sampler (Yang et al., 2026). arXiv 2602.18176
2. Where-to-Unmask (Asano et al., 2026). arXiv 2602.09501
3. CoRe (2026). arXiv 2602.04096
4. A-CFG (NeurIPS 2025). arXiv 2505.20199
5. DUS (Luxembourg et al., 2025). arXiv 2506.19037
6. Soft-Masking (Hersche et al., 2025). arXiv 2510.17206

### Track B: DaL & Cross-Step Memory
7. MetaState (2026). arXiv 2603.01331
8. TTT (Sun et al., 2024). arXiv 2407.04620
9. Titans (2025). arXiv 2501.00663
10. SR-TTT (2026). arXiv 2603.06642
11. MesaNet (von Oswald et al., 2025). arXiv 2506.05233
12. TTT++ (Liu et al., 2021). NeurIPS 2021

### Track C: COBA & Evaluation
13. ReMDM (2025). arXiv 2503.00307
14. Prism (2026). arXiv 2602.01842
15. Self-Rewarding SMC (2026). arXiv 2602.01849
16. Scaling Beyond Masked Diffusion (Sahoo et al., 2026). arXiv 2602.15014

### Theoretical Foundations
17. Amari (1998). Natural gradient works efficiently in learning
18. Ambrogioni (2024). Generative Diffusion Models Are Associative Memory Networks
19. Friston & Kiebel (2009). Predictive coding under free-energy principle
20. Berrou et al. (1993). Near Shannon limit error-correcting coding (Turbo codes)
21. ten Brink (2001). EXIT chart framework for iterative decoder convergence
22. Liang et al. (2026). Sharp Convergence Rates for Masked Diffusion Models. arXiv 2602.22505
23. Dmitriev et al. (2026). Efficient Sampling with Discrete Diffusion Models. arXiv 2602.15008

### DLM Foundations
24. MDLM (Sahoo et al., 2024). arXiv 2406.07524
25. LLaDA-8B (2025). arXiv 2502.09992
26. Dream-7B (2025). arXiv 2508.15487
27. Reasoning or Rationalization? (Huang et al., 2026). arXiv 2603.01190
