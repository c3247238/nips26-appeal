# Novelty Report

**Date**: 2026-04-14 (Updated for Iteration 9)
**Reviewer**: sibyl-novelty-checker
**Workspace**: sae-absorption

---

## Executive Summary

The front-runner candidate (cand_metric_audit, score 8/10) has strong novelty across all three headline contributions. No prior work audits the Chanin absorption metric with shuffled-label controls, decomposes the hedging/competitive-exclusion confound quantitatively, or maps the full L0-absorption curve on JumpReLU SAEs. The closest prior work -- Feature Hedging (Chanin et al., 2025) and Sparse but Wrong (Chanin & Garriga-Alonso, 2025) -- defines the conceptual ingredients but does not perform the specific analyses proposed here.

The immunological candidate (cand_immunological, score 8/10) has the highest raw novelty due to its unprecedented cross-reactive absorption prediction, but carries execution risk regarding the distinction from feature hedging. The theory candidate (cand_theory, score 6/10) faces moderate overlap with the Unified Theory of SDL (Tang et al., 2025) and OrtSAE. The geometry candidate (cand_geometry, score 5/10) has significant overlap with OrtSAE and The Geometry of Concepts.

**Overall novelty**: HIGH (front-runner = 8, one backup = 8, all candidates >= 5).

---

## Candidate 1: cand_metric_audit (FRONT RUNNER)

**Title**: Auditing SAE Feature Absorption: Hedging Dominance, Rate-Distortion Diagnostics, and the L0 Phase Transition on JumpReLU SAEs

**Novelty Score**: 8/10 (Novel; differences from prior work are clear and well-supported by 8 iterations of evidence)

### Core Contribution Claims

1. **First metric audit of absorption on JumpReLU SAEs** -- shuffled-label controls exceed true labels in all 5 domains
2. **First quantitative confound decomposition** -- 98.6% hedging vs 1.4% hierarchy-driven at L0=22
3. **L0 phase transition discovery** -- monotonic decline from 42.9% to 0.8% with sharp transition in L0=40-80
4. **First cross-domain absorption measurements** -- five hierarchy domains with comprehensive control suite
5. **CMI diagnostic** -- conditional mutual information as exploratory predictor of absorption susceptibility
6. **Honest negative results** -- 4/7 hypotheses falsified, reported transparently

### Prior Art Analysis

#### Contribution 1: Metric Audit with Shuffled-Label Controls

**Status: NOVEL.** No published paper validates the Chanin absorption metric with shuffled-label controls on any SAE architecture.

Evidence from systematic search:
- "metric audit" + "absorption" + "sparse autoencoder": Zero results on arXiv, Google Scholar, web search
- "shuffled label" + "absorption" + "sparse autoencoder": Zero results
- "control failure" + "absorption metric": Zero results

Prior work context:
- Chanin et al. (2024, NeurIPS 2025 Oral) define and apply the metric but do not validate it with controls
- SAEBench (Karvonen et al., 2025, ICML 2025) applies the metric at face value across 200+ SAEs without validation
- All architecture mitigations (Matryoshka SAE, OrtSAE, ATM-SAE, KronSAE) report absorption scores using the unvalidated metric
- SynthSAEBench (Chanin & Garriga-Alonso, 2026) validates on synthetic data with ground truth, not on real LLM activations with shuffled controls

The finding that shuffled labels produce HIGHER absorption than true labels in all 5 domains (up to 4.7x on first-letter, 27.5x on animal-class) is completely new and directly challenges the interpretability of all prior absorption measurements on JumpReLU SAEs.

#### Contribution 2: Confound Decomposition (Hedging vs Competitive Exclusion)

**Status: NOVEL.** No prior work quantitatively separates the hedging and competitive exclusion components within standard absorption measurements.

Prior work context:
- Feature Hedging (Chanin et al., 2025, arXiv:2505.11756) DEFINES hedging theoretically and shows it exists. Does NOT quantify how much of measured "absorption" is actually hedging.
- Sparse but Wrong (Chanin & Garriga-Alonso, 2025, arXiv:2508.16560) shows incorrect L0 produces wrong features via hedging. Does NOT decompose absorption into hedging/competitive exclusion fractions.
- "confound decomposition" + "feature absorption" + SAE: Zero results on arXiv, Google Scholar
- "hedging dominance" + "absorption": Zero results outside this project

The finding that 98.6% of detected FNs at L0=22 are hedging (with F1=1.0 probes eliminating probe quality as a confound) is the first quantitative separation. This is enabled by the multi-L0 persistence analysis: checking whether tokens that are FN at L0=22 become TP at L0=176 (hedging) or remain FN (competitive exclusion).

#### Contribution 3: L0 Phase Transition

**Status: PARTIALLY NOVEL.** The qualitative insight that absorption increases with sparsity is known. The specific quantitative curve and phase transition characterization are new.

Prior work context:
- Chanin et al. (2024) report absorption rates across layers and SAE widths but not a systematic L0 sweep
- Sparse but Wrong (Chanin & Garriga-Alonso, 2025) shows L0 matters critically but studies feature quality (probing accuracy), not the absorption rate curve specifically
- SAEBench reports absorption across SAEs with different L0 values but as a ranking, not a parametric curve with phase transition analysis
- "L0 phase transition" + "absorption": Zero results

The specific findings -- 42.85% -> 37.49% -> 14.39% -> 0.84% across L0={22,41,82,176}, cross-layer stable with CV<10%, sharp transition in L0=40-80 -- constitute a new empirical contribution. The framing of L0 as the primary control parameter (not architecture) is a novel conclusion.

#### Contribution 4: Cross-Domain Measurements

**Status: NOVEL.** All existing absorption measurements use first-letter spelling.

Prior work context:
- SAEBench explicitly acknowledges: "not useful for evaluating domain-specific feature absorption"
- Resurrecting the Salmon (2025) identifies the same gap
- No paper measures the Chanin absorption metric on knowledge hierarchies (city-country, city-continent, city-language, animal-class)

The cross-domain measurements find all rates below shuffled controls, which is informative as evidence of universal metric failure rather than cross-domain absorption.

#### Contribution 5: CMI Diagnostic

**Status: NOVEL but EXPLORATORY.** No prior work applies conditional mutual information to predict absorption susceptibility.

Prior work context:
- The Rate Distortion Dance of SAEs (Tilde Research, 2025) connects SAEs to rate-distortion theory generally
- Rate-distortion auto-encoders (2014) use information-theoretic objectives for training
- No work connects successive refinement theory specifically to absorption

The CMI finding (Cohen's d=-0.924 but p=0.236 Bonferroni) is honestly reported as exploratory with all limitations. The sign reversal at d'>=20 and probe quality confound (rho=-0.67) are transparently disclosed.

#### Contribution 6: Honest Negative Results

**Status: DISTINCTIVE PRESENTATION CHOICE.** Reporting 4/7 falsified hypotheses with pre-registered targets vs actual is unusual and valued. Multiple reviews rate this as the paper's strongest aspect.

### Key Prior Work (must cite)

| Paper | Year/Venue | Relevance |
|-------|-----------|-----------|
| Chanin et al., "A is for Absorption" | 2024 / NeurIPS 2025 | Foundation: absorption definition, metric |
| Karvonen et al., SAEBench | 2025 / ICML 2025 | Benchmark: absorption at scale |
| Chanin et al., Feature Hedging | 2025 / arXiv | Complementary failure mode |
| Chanin & Garriga-Alonso, Sparse but Wrong | 2025 / arXiv | L0 as critical parameter |
| Bussmann et al., Matryoshka SAEs | 2025 / ICML 2025 | Architecture mitigation |
| Korznikov et al., OrtSAE | 2025 / arXiv | Architecture mitigation |
| Li et al., ATM-SAE | 2025 / arXiv | Architecture mitigation |
| Tang et al., Unified Theory of SDL | 2025 / arXiv | Theoretical framework |
| Chanin & Garriga-Alonso, SynthSAEBench | 2026 / arXiv | Synthetic benchmark |
| Huang et al., RAVEL | 2024 / ACL 2024 | Data source for cross-domain |
| Lieberum et al., Gemma Scope | 2024 / BlackBoxNLP | SAE infrastructure |
| KronSAE | 2025 / arXiv | Architecture mitigation |
| Resurrecting the Salmon | 2025 / arXiv | Identifies domain-specific gap |

### Recommendation: PROCEED

Novelty is strong across all headline contributions. The gap is well-documented by the field itself (SAEBench, Resurrecting the Salmon, Sparse but Wrong all identify aspects of what this work addresses). No exact match or near-exact match was found.

The main reviewer concern will not be novelty collision but rather: "Is an audit of a metric a sufficient contribution?" The answer -- supported by the data -- is yes: every architecture mitigation paper (Matryoshka, OrtSAE, ATM-SAE, KronSAE) reports absorption using the unvalidated metric. If the metric does not transfer to JumpReLU, all these claims require re-evaluation.

---

## Candidate 2: cand_geometry (BACKUP)

**Title**: Geometric Forensics of Feature Absorption: Unsupervised Detection via Decoder Weight Topology

**Novelty Score**: 5/10 (Partial overlap; needs repositioning)

### Core Contribution Claims

1. Unsupervised absorption detector using four geometric signals from decoder weights
2. Four-signal ensemble outperforms single signals
3. Geometric detector generalizes to non-spelling domains

### Prior Art Analysis

**Significant overlap with:**
- **OrtSAE (Korznikov et al., 2025)**: Already demonstrates decoder cosine similarity as a strong absorption predictor. Density plots function as a one-signal unsupervised detector.
- **The Geometry of Concepts (Li et al., 2025)**: Already computes decoder Gram matrix, analyzes cosine clustering.
- **KronSAE (2025)**: Shows structured decoder geometry predicts absorption.
- **Quasi-Orthogonality evaluation (arXiv:2503.24277)**: Proposes input-driven metrics beyond decoder cosine similarities.

**Novel elements**: Activation-decoder misalignment (ADM) and residual absorption score (RAS) appear novel. The multi-signal ensemble has not been tested. Cross-domain validation would be new.

### Recommendation: MODIFY TO DIFFERENTIATE

Must demonstrate substantial improvement over OrtSAE's single-signal cosine similarity baseline. Consider repositioning as a component of cand_metric_audit rather than a standalone paper.

---

## Candidate 3: cand_immunological (BACKUP)

**Title**: Immunological Imprinting in Sparse Autoencoders: Cross-Reactive Absorption and Frequency-Ratio Scaling Laws

**Novelty Score**: 8/10 (Genuinely novel; no close prior work)

### Core Contribution Claims

1. Cross-reactive absorption exists (non-hierarchical, co-occurrence-driven suppression)
2. Frequency-ratio scaling law: P(absorption) ~ log(freq_i/freq_j)
3. Training-order imprinting effects
4. Differential mitigation effectiveness across architectures

### Prior Art Analysis

**Status: GENUINELY NOVEL across all claims.**

The immunological imprinting analogy and cross-reactive absorption prediction have zero precedent in the SAE/ML interpretability literature. Web search for "immunological imprinting" + "sparse autoencoder" returns only immunology papers.

**Critical collision to monitor**: Feature Hedging (Chanin et al., 2025) is the closest work. If cross-reactive absorption in wide SAEs is empirically indistinguishable from hedging, novelty collapses to "hedging with a biological analogy."

### Recommendation: PROCEED (with hedging-distinction caveat)

The cross-reactive absorption prediction is unique and testable. The immunological framing is intellectually compelling. Main risk is experimental: requires infrastructure beyond what cand_metric_audit needs, and the hedging distinction must be demonstrated.

---

## Candidate 4: cand_theory (BACKUP)

**Title**: Quantitative Theory of Feature Absorption: Coherence, Frequency, and the Absorption Phase Boundary

**Novelty Score**: 6/10 (Partially novel; faces theoretical competition)

### Core Contribution Claims

1. Quantitative formula: P(absorption) >= Phi(lambda * mu_H * sqrt(rho) / sigma - Phi^-1(1 - mu_H^2))
2. Critical width formula: M_c >= K(1 + D*mu_avg^2)/(1-epsilon)
3. Coherence-absorption correlation validation
4. Absorption-hedging tradeoff formalization

### Prior Art Analysis

**Moderate overlap with:**
- **Unified Theory of SDL (Tang et al., 2025)**: More general theoretical framework for absorption via non-identifiability. May subsume the proposed mean-field approximation.
- **OrtSAE (2025)**: Empirically demonstrates coherence-absorption link. The proposed formula may add limited value beyond this observation.
- **Classical dictionary identifiability (Spielman et al., 2012)**: The critical width formula builds on this. Appropriate foundation, not a collision.

**Novel elements**: The specific absorption probability formula and critical width prediction have not been published. The connection to classical identifiability through a SAE-specific formula is new.

### Recommendation: MODIFY TO DIFFERENTIATE

The unified theory of SDL is a stronger theoretical competitor. The proposed work's value depends on predictive accuracy of the formula. If the sigmoid model achieves R^2 > 0.5, the narrow-but-predictive contribution differentiates from Tang et al.'s broad-but-explanatory framework. The critical width prediction is the most distinctive element.

---

## Cross-Candidate Analysis

### Novelty Ranking

1. **cand_metric_audit** (8/10): Fills the most urgent gap, backed by 8 iterations of evidence. Best novelty-to-risk ratio.
2. **cand_immunological** (8/10): Highest raw novelty (unprecedented prediction). Highest execution risk.
3. **cand_theory** (6/10): Moderate overlap with Unified Theory of SDL. Depends on formula accuracy.
4. **cand_geometry** (5/10): Significant overlap with OrtSAE and Geometry of Concepts.

### Strategic Assessment

The front-runner selection of **cand_metric_audit** is strongly justified:

1. **Gap is field-acknowledged**: SAEBench, Resurrecting the Salmon, and Sparse but Wrong all point to the need for metric validation and L0 characterization.
2. **Evidence is strong**: 8 iterations, 2 confirmed hypotheses, 4 falsified hypotheses, 3 pending experiments.
3. **Impact is broad**: If the metric does not transfer, all absorption-related claims on JumpReLU SAEs (made by Matryoshka, OrtSAE, ATM-SAE, KronSAE, and SAEBench) require re-evaluation.
4. **Execution is tractable**: 3 GPU-hours of blocking experiments with clear gate structure.

### Recommendation for the Synthesizer

- **Proceed with cand_metric_audit** -- novelty is strong, gap is real, execution is feasible.
- **Reserve cand_immunological** for follow-up -- highest raw novelty but requires cand_metric_audit's cross-domain infrastructure as a foundation.
- **Integrate cand_theory's CMI diagnostic** into cand_metric_audit as Contribution 5 (already done in current proposal).
- **Deprioritize cand_geometry** unless substantial improvement over OrtSAE baseline is demonstrated.
- **Key differentiator to emphasize**: This work AUDITS the metric; all prior work APPLIES the metric. The universal control failure finding is the headline -- it reframes the entire absorption mitigation literature.
