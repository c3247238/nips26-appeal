# Novelty Report: Cross-Domain Absorption Characterization with Rate-Distortion Diagnostics and Confound Decomposition

**Search date:** 2026-04-14 (updated 2026-04-14, synthesis round 2)
**Sources searched:** arXiv (via WebSearch), Google Scholar (via WebSearch), LessWrong/Alignment Forum, Neuronpedia/SAEBench, DeepMind blog, OpenReview, ResearchGate, Semantic Scholar, Tilde Research blog, Emergent Mind, HuggingFace Papers
**Proposal version:** cand_cross_domain_ratedistortion (front-runner, iteration 6, synthesis round 2)

---

## Executive Summary

The revised front-runner candidate (cand_cross_domain_ratedistortion) has **defensible novelty** (score: 7/10) across its three primary contributions. No prior work combines (1) cross-domain absorption measurement on Gemma 2 2B with (2) confound decomposition and (3) rate-distortion diagnostics grounded in the successive refinement theorem. The rate-distortion diagnostic remains the highest-novelty individual contribution (8/10) with no direct prior work found. The landscape has grown more competitive since the previous novelty check: a new progressive coding paper (Peter & Sogaard, May 2025) connects SAEs to successive refinement empirically but does NOT formalize absorption optimality via CMI, and a new Hierarchical SAE paper (Luo et al., Feb 2026) achieves strong absorption scores but does NOT perform cross-domain absorption rate measurement or confound decomposition. The impossibility theorem (dropped, score 5/10) remains correctly dropped.

**Overall novelty: HIGH** (front-runner >= 7, all active candidates assessed)

---

## Candidate 1: Cross-Domain Rate-Distortion (Front-Runner)

**Candidate ID:** cand_cross_domain_ratedistortion
**Novelty Score: 7/10** -- Novel with minor overlap; differences are clear and defensible

### Contribution 1: First Cross-Domain Absorption Characterization on Gemma 2 2B

**Verdict: Novel. No exact match found.**

All existing absorption studies use the first-letter spelling task as the sole evaluation domain:
- Chanin et al. (2024/2025, arXiv:2409.14507, NeurIPS 2025 Oral) define the absorption metric and demonstrate it on first-letter spelling using Gemma 2 2B.
- SAEBench (Karvonen et al., 2025, arXiv:2503.09532, ICML 2025) evaluates absorption but only on the first-letter task.
- OrtSAE (Korznikov et al., 2025, arXiv:2509.22033, ICLR 2026) evaluates absorption on the SAEBench first-letter benchmark.
- HSAE "From Atoms to Trees" (Luo et al., 2026, arXiv:2602.11881) evaluates absorption via the SAEBench benchmark (first-letter) but does NOT measure per-domain absorption rates for city-country, city-continent, city-language, or animal-class hierarchies.

**Partial overlap with RAVEL:** SAEBench includes the RAVEL benchmark testing city-country, city-continent, city-language disentanglement on Gemma 2 2B. However, RAVEL measures *feature disentanglement* (causal intervention accuracy), NOT *absorption rates* (probe-based detection of systematic parent-feature false negatives). These are fundamentally different metrics. The proposal must explicitly differentiate.

**Partial overlap with domain-specific SAEs:** "Resurrecting the Salmon" (O'Neill et al., 2025, arXiv:2508.09363) argues domain-specific SAEs reduce absorption but does NOT measure absorption rates cross-domain using the Chanin metric.

**New since last check:** Feature sensitivity measurement (Tian et al., 2025, arXiv:2509.23717, NeurIPS 2025 MI Workshop Spotlight) evaluates whether features reliably activate on similar texts, which is tangentially related to absorption (failure to activate) but uses a generation-based protocol, not probe-based absorption measurement.

**Risk assessment:** Reviewers may conflate RAVEL disentanglement with absorption measurement. The proposal MUST include a dedicated comparison explaining the difference.

### Contribution 2: Rate-Distortion Theory via Successive Refinement Theorem

**Verdict: Highly novel (8/10). No direct prior work applies the successive refinement theorem to SAE absorption.**

**Critical new paper:** Peter & Sogaard (2025, arXiv:2505.00190) "Empirical Evaluation of Progressive Coding for Sparse Autoencoders" is the closest new work. This paper:
- Connects SAEs to progressive/successive coding concepts
- Evaluates rate-distortion trade-offs between Matryoshka SAEs and pruned vanilla SAEs
- Mentions the successive refinement theorem (Equitz & Cover, 1991) as conceptual background
- Evaluates reconstruction loss and representational similarity

**However, Peter & Sogaard do NOT:**
- Formalize when absorption is information-theoretically optimal vs. suboptimal
- Define or compute I(X; f_parent | f_child) as an absorption diagnostic
- Derive a CMI threshold for absorption necessity
- Connect successive refinement to the Markov chain condition X -- f_child -- f_parent
- Predict which specific features will be absorbed based on CMI
- Address absorption at all (their paper is about efficient progressive coding, not about diagnosing or predicting absorption)

**Differentiation from Peter & Sogaard is clear:** Their paper asks "how to train SAEs that progressively refine representations" (engineering question). The proposal asks "when is absorption information-theoretically necessary vs. avoidable?" (diagnostic question). The successive refinement theorem appears in both as background, but the application is completely different.

**Other information-theoretic SAE work:**
- Bereska et al. (2025, arXiv:2512.13568) "Superposition as Lossy Compression" applies Shannon entropy to SAE activations to measure superposition degree. Does NOT address absorption, CMI, or successive refinement conditions. Different question entirely.
- Tilde Research blog "The Rate Distortion Dance of Sparse Autoencoders" discusses rate-distortion framing but not Markov chain conditions or CMI diagnostics.

**No prior work found that:**
- Applies the Equitz & Cover successive refinement theorem specifically to SAE absorption
- Defines CMI = I(X; f_parent | f_child) as an absorption diagnostic
- Derives a threshold for absorption optimality based on CMI and geometric constants
- Uses CMI to predict which features will be absorbed

### Contribution 3: Confound Decomposition with Multi-L0 Profiling

**Verdict: Moderate novelty (6/10). Extends prior work with meaningful additions.**

**Key prior work:**
- Chanin et al. (2025, arXiv:2505.11756) "Feature Hedging" decomposes SAE failures into hedging vs. absorption. Shows Matryoshka SAEs trade absorption for hedging. Proposes "balance matryoshka SAE." This directly covers the hedging component of the proposed decomposition.
- Chanin & Garriga-Alonso (2025) "Sparse but Wrong: Incorrect L0 Leads to Incorrect Features" shows incorrect L0 causes feature distortion, relevant to L0 confound analysis.
- SAEBench (2025) includes evaluation across L0 ranges.

**What the proposal adds beyond prior work:**
- Multi-L0 decomposition into THREE components (hierarchy-driven, hedging, reconstruction error) across multiple L0 values, showing how the composition changes
- Threshold sensitivity analysis (5x4 grid of cosine/magnitude thresholds) -- never published as a systematic sweep
- Rosenbaum matching and partial correlation analysis with L0 as covariate
- Cross-domain confound profiling (not just first-letter)

### Contribution 4: Lateral Inhibition Bifurcation Analysis

**Verdict: Novel (8/10). No prior work applies LCA bifurcation analysis to predict JumpReLU vs. L1 absorption dynamics.**

The search found no paper connecting:
- Locally Competitive Algorithm (LCA) dynamics to SAE absorption
- Transcritical bifurcation analysis to hard vs. soft threshold SAEs
- Prediction of bimodal vs. continuous absorption distributions based on activation function choice

**Related work:** Rozell et al. (2008) established LCA for sparse coding. WARP-LCA (2025) studies LCA efficiency. Neither addresses absorption prediction.

**Empirical support:** SAEBench data shows JumpReLU has worse absorption than ReLU at matched L0, consistent with the prediction, but no theoretical explanation exists in the literature.

### Contribution 5 (Secondary): Unsupervised Detection Pipeline

**Verdict: Moderate novelty, de-prioritized based on pilot evidence.**

The ITAC metric and conditional cosine + firing rate pipeline remain novel in concept, but pilot evidence (no significant ITAC separation) weakens the expected contribution. Correctly classified as secondary with pre-registered negative result framework.

---

## Candidate 2: Decomposition Reframing (Backup)

**Candidate ID:** cand_decomposition_reframing
**Novelty Score: 6/10** -- Partial overlap; needs repositioning

The hedging paper (Chanin et al., 2025) significantly reduces novelty. The "immune escape test" (zeroing child to see parent recovery) remains novel as a causal intervention diagnostic for absorption. The threshold sensitivity sweep provides incremental value.

**Recommendation:** MODIFY TO DIFFERENTIATE. Only pursue if front-runner fails. Emphasize causal patching angle.

---

## Candidate 3: Rate-Distortion Theory (Backup)

**Candidate ID:** cand_rate_distortion_theory
**Novelty Score: 8/10** -- Genuinely novel

The pure theory paper applying successive refinement + lateral inhibition to SAE absorption retains highest per-claim novelty. Peter & Sogaard (2025) does NOT overlap with the core theoretical claims (CMI threshold, Markov chain condition, bifurcation analysis).

**New concern:** Single-domain validation (first-letter only) limits generalizability. The front-runner addresses this by combining theory with cross-domain empirics.

**Recommendation:** PROCEED as pivot target if cross-domain empirics fail.

---

## New Papers Since Last Novelty Check

### Papers that STRENGTHEN the proposal's position

| Paper | How it helps |
|-------|-------------|
| Peter & Sogaard, 2025. Progressive Coding for SAEs. arXiv:2505.00190 | Establishes that progressive/successive coding is relevant to SAEs, creating intellectual context for the successive refinement framing. But does NOT address absorption -- leaving the diagnostic application wide open. |
| Luo et al., 2026. From Atoms to Trees (HSAE). arXiv:2602.11881 | Shows hierarchical SAEs reduce absorption, validating that hierarchy is central to the absorption problem. Does NOT measure per-domain rates or decompose confounds. |
| OrtSAE, 2025. arXiv:2509.22033 (ICLR 2026) | Architectural solution reducing absorption by 65%. Does NOT analyze when absorption is necessary. Makes the diagnostic question more relevant: should we eliminate ALL absorption? |
| KronSAE, 2025. arXiv:2505.22255 | Another architectural solution. Same as OrtSAE: addresses absorption as problem to fix, not as phenomenon to understand. |
| Tian et al., 2025. Feature Sensitivity. arXiv:2509.23717 | Measures feature reliability via generation-based testing. Complementary to probe-based absorption measurement. |

### Papers that CHALLENGE the proposal's position

| Paper | Challenge | Severity | Mitigation |
|-------|-----------|----------|------------|
| Peter & Sogaard, 2025. Progressive Coding. arXiv:2505.00190 | Connects SAEs to successive coding/rate-distortion, reducing novelty of the "first to apply rate-distortion to SAEs" claim | Moderate | The proposal's contribution is the DIAGNOSTIC application (CMI predicts absorption), not the general connection. Peter & Sogaard do not study absorption at all. Differentiate clearly in intro. |
| Chanin et al., 2025. Feature Hedging. arXiv:2505.11756 | Partially covers hedging-absorption decomposition | Moderate | Proposal extends to THREE components + multi-L0 profiling + threshold sensitivity. Cite and differentiate. |
| SAEBench, 2025. arXiv:2503.09532 | RAVEL uses same domain attributes (city/country/continent) | Low-Moderate | Different metric (disentanglement vs. absorption). Must differentiate clearly. |
| Song et al., 2025. Feature Consistency. arXiv:2505.20254 | Shows SAE features inconsistent across runs, challenging reliability of all SAE-based measurements | Low | Acknowledge as limitation. Use multiple random seeds if feasible. |

---

## Updated Prior Work Citation Requirements

### Must-cite papers (updated list)

| Paper | Relevance | Priority |
|-------|-----------|----------|
| Chanin et al., 2024/2025. A is for Absorption. arXiv:2409.14507 | Foundation paper | Essential |
| Karvonen et al., 2025. SAEBench. arXiv:2503.09532 | Benchmark with RAVEL overlap | Essential (differentiate) |
| Peter & Sogaard, 2025. Progressive Coding. arXiv:2505.00190 | Rate-distortion for SAEs | Essential (differentiate) |
| Chanin et al., 2025. Feature Hedging. arXiv:2505.11756 | Hedging-absorption decomposition | Essential (differentiate) |
| Luo et al., 2026. From Atoms to Trees (HSAE). arXiv:2602.11881 | Hierarchical SAE reduces absorption | Important |
| Costa et al., 2025. From Flat to Hierarchical. arXiv:2506.03093 | Conditional orthogonality | Important |
| Bussmann et al., 2025. Matryoshka SAEs. arXiv:2503.17547 | Architecture baseline | Important |
| Tang et al., 2025. Unified SDL Theory. arXiv:2512.05534 | Theoretical foundation | Important |
| Cui et al., 2025. Limits of SAEs. arXiv:2506.15963 | Impossibility results | Important |
| Bereska et al., 2025. Superposition as Lossy Compression. arXiv:2512.13568 | IT framework for SAEs | Important |
| Korznikov et al., 2025. OrtSAE. arXiv:2509.22033 | Architectural solution | Cite |
| DeepMind Safety, 2025. Negative Results for SAEs. Medium | Motivation | Cite |
| O'Neill et al., 2025. Resurrecting the Salmon. arXiv:2508.09363 | Domain-specific SAEs | Cite |
| Song et al., 2025. Feature Consistency. arXiv:2505.20254 | SAE reliability concern | Cite |
| KronSAE, 2025. arXiv:2505.22255 | Architectural solution | Cite |
| SynthSAEBench, 2026. arXiv:2602.14687 | Synthetic evaluation | Optional |

---

## Critical Differentiation Points for the Paper

### 1. Peter & Sogaard (Progressive Coding) vs. Our Rate-Distortion Diagnostic

This is the MOST IMPORTANT differentiation to get right. Both papers connect SAEs to successive refinement/rate-distortion. The key differences:

- **Peter & Sogaard ask:** "How can we train SAEs that efficiently code progressively?" (engineering/architecture question)
- **Our proposal asks:** "When is feature absorption information-theoretically necessary vs. avoidable?" (diagnostic/theoretical question)
- **Peter & Sogaard's method:** Compare Matryoshka vs. pruned SAEs on reconstruction loss
- **Our method:** Compute CMI = I(X; f_parent | f_child) to predict absorption for specific feature pairs
- **Peter & Sogaard do NOT:** Mention absorption, compute CMI, derive optimality conditions, or predict per-feature outcomes
- **Recommended framing:** "Peter & Sogaard (2025) empirically evaluate progressive coding for SAEs, establishing rate-distortion as a relevant framework. We complement their work by providing the first information-theoretic diagnostic for when absorption is optimal vs. suboptimal, grounded in the successive refinement theorem's Markov chain condition."

### 2. RAVEL (SAEBench) vs. Our Cross-Domain Absorption

- **RAVEL measures:** Whether SAE latent interventions correctly transfer attributes (e.g., change "Paris" from France to Japan)
- **We measure:** Whether parent features systematically fail to fire due to child features absorbing their information
- **Different failure modes:** RAVEL tests feature disentanglement; we test feature recall completeness

### 3. Feature Hedging (Chanin 2025) vs. Our Confound Decomposition

- **Hedging paper:** Decomposes into hedging vs. absorption (two components)
- **Our decomposition:** Three components (hierarchy-driven, hedging, reconstruction error) + multi-L0 profiling + threshold sensitivity analysis
- **Key addition:** The multi-L0 profile showing how the composition changes with sparsity is novel

---

## Risk Assessment Update

| Risk | Probability | Impact | Change from Last Round |
|------|------------|--------|------------------------|
| Peter & Sogaard perceived as prior work | 20% | High | NEW RISK. Mitigate with clear differentiation. |
| RAVEL conflation with absorption | 30% | Medium | Unchanged. |
| Cross-domain rates < 3% everywhere | 25% | Medium | Unchanged. |
| CMI-absorption correlation weak | 30% | Medium | Unchanged. |
| Controls not calibratable | 20% | High | Unchanged. |
| HSAE/OrtSAE make characterization feel incremental | 15% | Low | NEW RISK. These propose solutions; we characterize the problem. Complementary. |

---

## Summary Table

| Candidate | Score | Exact Matches | Partial Overlaps | Recommendation |
|-----------|-------|---------------|-------------------|----------------|
| cand_cross_domain_ratedistortion (front-runner) | 7 | 0 | 4 (RAVEL, Hedging, Progressive Coding, HSAE) | **Proceed** |
| cand_decomposition_reframing (backup) | 6 | 0 | 3 (Hedging, L0, SAEBench) | Modify to differentiate |
| cand_rate_distortion_theory (backup) | 8 | 0 | 1 (Progressive Coding -- weak overlap) | **Proceed** (pivot target) |
| cand_impossibility_theorem (dropped) | 5 | 0 | 4 | **Dropped** (correct) |
