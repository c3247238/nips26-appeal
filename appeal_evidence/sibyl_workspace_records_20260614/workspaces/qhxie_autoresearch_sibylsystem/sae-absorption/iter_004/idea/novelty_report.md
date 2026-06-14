# Novelty Report: The Competitive Geometry of Feature Absorption in Sparse Autoencoders

**Date:** April 14, 2026
**Novelty Checker:** sibyl-novelty-checker
**Workspace:** /home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current

---

## Executive Summary

The front-runner proposal (cand_a) integrates four main contributions: (1) Lotka-Volterra competition coefficient as unsupervised absorption detector, (2) corpus PMI as primary predictor, (3) first systematic downstream causal chain test, and (4) distributed absorption metric to explain the width paradox. After thorough search across arXiv, web, and the LessWrong/Alignment Forum community, the core novelty claims hold — with one important caveat: the theoretical explanation of absorption as corpus co-occurrence has already been qualitatively established by Chanin et al. (2024). The *quantitative prediction* framing (PMI as a regression predictor) and the *unsupervised detection* using the Lotka-Volterra formalism are genuinely novel. The downstream causal chain test is also novel in the systematic sense.

**Overall novelty: high** (all main contributions score 7–9)

---

## Prior Art Landscape

### Core Prior Work (Must Cite)

1. **Chanin et al. (2024) — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"**
   - arXiv:2409.14507, NeurIPS 2025 Oral
   - **Relevance:** THE foundational paper. Establishes the first-letter task ground truth, the absorption metric, and the qualitative corpus co-occurrence mechanism. Provides the Gemma Scope SAE evaluation suite. The proposal treats this as primary ground truth — correct.
   - **Collision class:** related_work (ground truth source, not a collision on the novel contributions)
   - **Critical note:** Chanin et al. already establish that absorption is *caused* by hierarchical corpus co-occurrence in a qualitative sense. H2 (PMI predictor) must be clearly positioned as the *quantitative* prediction test, not the discovery of the causal mechanism itself.

2. **Karvonen et al. (2025) — SAEBench: A Comprehensive Benchmark for Sparse Autoencoders**
   - arXiv:2503.09532, ICML 2025
   - **Relevance:** Provides 200+ SAEs, absorption scores, downstream task metrics (RAVEL, SCR, TPP, sparse probing). The proposal's Component 3 directly uses this data. SAEBench paper does NOT run a systematic Pearson correlation between absorption scores and downstream metrics with pre-specified minimum effect size and Bonferroni correction. SAEBench notes that "gains on proxy metrics do not reliably translate" but this is a qualitative statement, not the controlled correlation analysis the proposal plans.
   - **Collision class:** related_work / partial_overlap on Component 3 (downstream analysis)
   - **Critical note:** SAEBench does include some cross-metric comparisons but does not pre-register the analysis or test for the specific absorption–downstream correlation across all 200+ SAEs. The proposal's pre-registered systematic test remains novel.

3. **DeepMind Safety Research (2025) — "Negative Results for Sparse Autoencoders On Downstream Tasks and Deprioritising SAE Research"**
   - Medium blog post, March 26, 2025
   - **Relevance:** Reports that 1-sparse SAE probes underperform dense linear probes on harmful intent detection OOD. Directly supports H3. Does NOT provide a systematic correlation across SAEs at varying absorption levels.
   - **Collision class:** related_work (motivates H3; does not pre-empt the systematic test)

4. **LessWrong — "Looking for Feature Absorption Automatically"**
   - URL: lesswrong.com/posts/z7iyek97dAeQMxdSd/looking-for-feature-absorption-automatically
   - **Relevance:** Proposes detecting absorption by finding latents with similar *causal effects on downstream layers* but non-co-occurrence on the same token. Reports a NEGATIVE RESULT: the method does not work in practice for general features. The method is based on ablation effect vector similarity, NOT decoder cosine similarity + co-activation statistics (the LV approach). This confirms the proposal's claimed differentiation.
   - **Collision class:** partial_overlap (same goal: unsupervised absorption detection; different method; negative result)
   - **Differentiation:** The proposal's LV coefficient uses activation frequency + pairwise co-activation rate + decoder cosine pre-filter, which is mechanistically distinct from ablation effect similarity. The negative result on ablation-effect approach actually *supports* novelty of the LV approach.

5. **Chanin, Dulka, Garriga-Alonso (2025) — "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders"**
   - arXiv:2505.11756
   - **Relevance:** Studies feature *hedging* (correlated features merged in narrow SAEs) as distinct from feature absorption. Establishes that correlated features cause SAE pathologies. This is a different phenomenon from absorption (hedging is caused by reconstruction loss; absorption by sparsity penalty). Some overlap with the corpus co-occurrence framing.
   - **Collision class:** related_work
   - **Note:** The width relationship is interesting — hedging is worse in *narrow* SAEs, while absorption is worse in *wider* SAEs. The proposal's H4 (distributed absorption increases with width) is not in conflict.

6. **Li, Ren (2025) — "Time-Aware Feature Selection: Adaptive Temporal Masking (ATM) for Stable SAE Training"**
   - arXiv:2510.08855, ICLR 2025 Workshop
   - **Relevance:** ATM achieves dramatic absorption reduction (0.0068 vs TopK 0.1402). Involves masking mechanism that disrupts co-occurrence patterns during training. The paper mentions ATM as a masked regularization solution — this is likely what the proposal cites as "arXiv:2604.06495." However, 2604.06495 was not indexed in web searches; it may be a different, newer paper. The proposal's claim that "masked regularization works but no paper shows corpus statistics predict absorption beforehand" remains valid given ATM's focus on training dynamics, not corpus statistics.
   - **Collision class:** related_work (architecture paper, not a novelty collision on corpus PMI prediction)

7. **Tang et al. (2025) — "A Unified Theory of Sparse Dictionary Learning: Piecewise Biconvexity and Spurious Minima"**
   - arXiv:2512.05534
   - **Relevance:** First unified theoretical framework for SAEs as piecewise biconvex optimization. Explains feature absorption and dead neurons theoretically. Proposes "feature anchoring" as a remedy. This is the "Tang et al. (2024) biconvex framework" cited in cand_b.
   - **Note:** The theory is identifiability-based, not competition-based. The LV competition coefficient approach is a different theoretical framing.
   - **Collision class:** related_work (theoretical grounding from different angle)

8. **Korznikov et al. (2025) — "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features"**
   - arXiv:2509.22033
   - **Relevance:** Orthogonality penalty on decoder reduces absorption by 65%. The proposal's H6 (exploratory) predicts OrtSAE works by reducing σ_ij (niche overlap), which is directly testable if OrtSAE checkpoints are available. OrtSAE does not provide a mechanistic explanation in LV terms.
   - **Collision class:** related_work

9. **Li et al. (2024) — "The Geometry of Concepts: Sparse Autoencoder Feature Structure"**
   - arXiv:2410.19750
   - **Relevance:** Maps SAE feature co-occurrence to geometric structure; co-occurring features are spatially co-located. Uses spectral clustering on co-occurrence statistics. Does NOT apply LV competitive exclusion or use co-occurrence to *predict* absorption rates.
   - **Collision class:** related_work

10. **Wu et al. (2025) — "Interpreting and Steering LLMs with Mutual Information-Based Explanations on Sparse Autoencoders"**
    - arXiv:2502.15576
    - **Relevance:** Uses mutual information between SAE features and tokens for feature explanation (not absorption detection). The MI is for *explanation quality*, not absorption prediction.
    - **Collision class:** related_work

---

## Per-Candidate Novelty Analysis

### Candidate A (Front-runner): Competitive Geometry of Feature Absorption

**Novelty Score: 8**

#### Contribution 1: LV Competition Coefficient as Unsupervised Absorption Detector (H1)

**Assessment: NOVEL (score: 9)**

No paper applies Lotka-Volterra competitive exclusion formalism to SAE features. The LessWrong "Looking for Feature Absorption Automatically" post is the closest prior work on unsupervised detection, and it reports a NEGATIVE RESULT on a different method (ablation effect similarity). The LV coefficient α_ij = σ_ij · (f_j/f_i) computes differently: it uses decoder cosine similarity as a pre-filter plus activation co-occurrence rate normalized by feature frequencies. This is distinct from both the Chanin metric (requires probe directions) and the LessWrong approach (requires downstream ablation effects).

**Risk:** The SAEBench absorption metric already uses decoder cosine similarity as part of its latent-finding step, and the Masked Cosine Similarity (MCS) metric in Matryoshka SAEs uses co-activation statistics. The LV coefficient combines these in a specific ratio form derived from ecological theory, which is the novel theoretical grounding. However, the individual components (cosine similarity filter + co-activation rate) are not new.

**Recommendation:** Proceed. Explicitly acknowledge that the components exist; emphasize the LV-theoretic derivation of the specific ratio form and the sharp threshold prediction (sigmoid vs. linear AIC test) as the contribution. If F1 fails, the sharpness test itself is a publishable diagnostic.

#### Contribution 2: Corpus PMI as Primary Predictor of Absorption Patterns (H2)

**Assessment: NOVEL WITH CAVEAT (score: 7)**

The qualitative mechanism (corpus co-occurrence causes absorption) is established by Chanin et al. (2024). The specific quantitative claim — that token-pair PMI from the training corpus predicts *which* specific letter features are absorbed across 30+ SAEs in a regression — has NOT been done. The closest is ATM (2510.08855) which shows corpus-disrupting masking reduces absorption, but does not extract PMI values and run regression against absorption rates.

**Critical caveat:** The proposal must clearly position H2 as extending Chanin et al.'s qualitative mechanism to a *quantitative prediction model*. The regression framing (partial R² ≥ 0.10 after controlling for SAE config) is specific and novel.

**Recommendation:** Proceed. Reframe the claim as: "We provide the first quantitative test of the corpus co-occurrence hypothesis, showing that token-pair PMI predicts absorption rates independently of SAE hyperparameters."

#### Contribution 3: First Systematic Downstream Causal Chain Test (H3)

**Assessment: NOVEL (score: 8)**

SAEBench (2503.09532) provides the raw data but does NOT run the systematic Pearson/Spearman correlation analysis between absorption scores and downstream task performance across 200+ SAEs with pre-specified minimum effect size (|r| > 0.3) and Bonferroni correction. SAEBench notes qualitatively that "gains on proxy metrics do not reliably translate" but provides no absorption-specific correlation test.

DeepMind's finding (harmful intent probe) is specific to one architecture and not correlated with absorption rate variation across SAEs.

**Recommendation:** Proceed. The pre-registered systematic test with pre-specified effect size is the key claim. Emphasize that this is the *first* absorption-specific correlation analysis across the full SAEBench suite.

#### Contribution 4: Distributed Absorption Score DAS(k=3) and Width Paradox Resolution (H4)

**Assessment: NOVEL (score: 8)**

No paper proposes a distributed absorption metric DAS(P, k=3) measuring information loss conditioned on 3 children simultaneously. The width paradox (wider SAEs show higher single-child absorption) is documented in SAEBench but unexplained. Feature fragmentation literature (2509.02565) discusses multi-latent phenomena but not in the specific framework of LV competitive exclusion or the DAS metric.

**Risk:** The width paradox is confirmed as a known fact (SAEBench). The distributed absorption hypothesis is new but is a corollary of the LV theory — if LV theory fails (H1 fails), the DAS metric loses its theoretical grounding. However, DAS as a descriptive measurement tool remains valid even if the LV mechanism fails.

**Recommendation:** Proceed. Note that H4 is contingent on H1's success for full theoretical weight, but the measurement itself (DAS(k=3) vs k=1 across widths) is an independent empirical contribution.

---

### Candidate B: The Absorption-Sparsity Phase Transition (Information-Theoretic)

**Novelty Score: 7**

A(P,C) = I(f_P; f_C)/H(f_P) as an absorption predictor overlaps with Wu et al. (2502.15576) which uses MI for SAE feature explanations. However, using it specifically as an *absorption* predictor (rather than explanation quality) is novel. The phase transition framing overlaps with Tang et al. (2512.05534) which derives identifiability theory for SAEs and notes absorption as an outcome. Tang et al. use biconvex theory, not an explicit (ρ, L0, q) phase diagram.

**Risk:** Tang et al. (2512.05534) is a close competitor — it provides a formal theory explaining absorption. The phase transition claim in cand_b must clearly go beyond Tang et al.'s identifiability framing.

**Recommendation:** Maintain as backup for H1 failure. Differentiate from Tang et al. by emphasizing the explicit (ρ, L0, q) phase boundary and empirical validation on Gemma Scope.

---

### Candidate C: Absorption Scaling Laws and Iso-Absorption Curves

**Novelty Score: 6**

This is the weakest candidate. SAEBench already provides some scaling information. Chanin et al. (2024) Fig 7b shows qualitative trends. The quantitative regression and iso-absorption curves are a modest incremental contribution. Cross-domain validation (country names, given names) adds novelty.

**Key risk:** Reviewers will note that SAEBench already does systematic comparison across 200+ SAEs at varying widths, L0 values, architectures. The regression formula may not add substantially beyond what SAEBench presents.

**Recommendation:** Use only as fallback if both H1 and H2 fail. Not sufficient as a standalone paper without strong positive results on the regression.

---

### Candidate D: Downstream Causal Audit

**Novelty Score: 7**

The controlled L0-matching approach is methodologically sound and genuinely novel. However, this is subsumed by the front-runner's Component 3. If Component 3 is executed, cand_d's main contribution is already included.

**Recommendation:** Keep Component 3 of the front-runner as the implementation; do not treat cand_d as a separate pivot unless the front-runner's Component 3 yields inconsistent results requiring methodological re-framing.

---

## Key Risks and Required Repositioning

### Risk 1: H2 framing — "novel causal claim" vs. "quantitative extension"

The corpus co-occurrence mechanism is already established qualitatively by Chanin et al. The contribution must be clearly framed as "first *quantitative* prediction test" not "first discovery of the corpus mechanism." Reviewers at NeurIPS/ICLR will know the Chanin paper.

**Action required:** Revise proposal language to say "We are the first to test whether corpus PMI *quantitatively* predicts absorption rates across 30+ SAEs in a controlled regression" rather than "reframing absorption as a data-driven phenomenon."

### Risk 2: The 2604.06495 citation

The proposal cites "masked regularization (arXiv:2604.06495)" but this paper did not appear in web searches. The closest match is ATM (arXiv:2510.08855) from October 2025. Either:
- 2604.06495 is a very recent (April 2026) paper that is not yet widely indexed
- There is a citation error in the proposal

**Action required:** Verify arXiv:2604.06495 before finalizing the paper. If it exists and shows corpus statistics *predict* absorption beforehand, this could be a partial overlap with H2. If it only shows masked training reduces absorption without measuring PMI predictivity, H2 remains novel.

### Risk 3: SAEBench correlation analysis may already exist

The proposal plans to compute Pearson correlations between absorption scores and downstream tasks across 200+ SAEs. SAEBench authors have access to this data and may publish follow-up analyses. The pre-registration and Bonferroni correction framing is the key differentiation.

**Action required:** Run the correlation analysis immediately in the pilot — if SAEBench v4 (June 2025) already includes this analysis, the contribution is pre-empted. Check the SAEBench paper's supplementary materials.

### Risk 4: The "Looking for Feature Absorption Automatically" LessWrong post

The post confirms a negative result for the ablation-effect cosine similarity method. The LV method's differentiation is clear. However, if the LV method also fails (F1 < 0.50), reviewers may compare the two negative results and question whether unsupervised detection is fundamentally hard.

**Action required:** If H1 fails, pivot to cand_b (information-theoretic). Do not concede that unsupervised detection is impossible — the LessWrong negative result is on a different metric.

---

## Recommendations Summary

| Candidate | Novelty Score | Recommendation |
|---|---|---|
| cand_a (front-runner) | 8 | Proceed — all four contributions are novel; reframe H2 as quantitative extension |
| cand_b (info-theoretic) | 7 | Keep as backup; differentiate from Tang et al. (2512.05534) |
| cand_c (scaling laws) | 6 | Fallback only; weak without H1/H2 success |
| cand_d (causal audit) | 7 | Subsumed by front-runner Component 3 |

**Overall novelty assessment: HIGH**

The proposal identifies four genuinely novel contributions and is well-positioned against the existing literature. The main danger is framing — the corpus co-occurrence mechanism must be presented as quantitative extension rather than discovery, and the LV formalism must be distinguished from the simpler co-activation + cosine approaches that already exist in the literature.

---

## Complete Prior Art Reference List

| Paper | arXiv / Source | Relation to Proposal |
|---|---|---|
| Chanin et al. (2024), "A is for Absorption" | 2409.14507 | Ground truth source; establishes corpus co-occurrence mechanism qualitatively |
| Karvonen et al. (2025), SAEBench | 2503.09532 | Data source for Component 3; partial overlap on downstream evaluation |
| DeepMind (2025), Negative Results blog | Medium | Motivates H3; does not pre-empt systematic test |
| LessWrong, "Looking for Feature Absorption Automatically" | lesswrong.com | Negative result on different unsupervised method; supports LV novelty |
| Chanin et al. (2025), Feature Hedging | 2505.11756 | Related phenomenon (hedging ≠ absorption); related_work |
| Li & Ren (2025), ATM SAE | 2510.08855 | Masked training approach; possible match for proposal's 2604.06495 cite |
| Tang et al. (2025), Piecewise Biconvexity | 2512.05534 | Theoretical framing from different angle; backup cand_b differentiation required |
| Korznikov et al. (2025), OrtSAE | 2509.22033 | OrtSAE result; proposal's H6 exploratory prediction |
| Li et al. (2024), Geometry of Concepts | 2410.19750 | Co-occurrence geometry; related_work |
| Wu et al. (2025), MI-Based Explanations | 2502.15576 | MI for explanation, not absorption; related_work |
| Bussmann et al. (2025), Matryoshka SAEs | alignment forum | Architecture approach; related_work |
| Chanin et al. (2025), Toy Models LessWrong | lesswrong.com | Mechanism confirmation; related_work |
| Gao et al. (2025), Scaling SAEs | ICLR 2025 | Scaling context; related_work |
| Li et al. (2024), Gemma Scope | 2408.05147 | Primary SAE suite used; infrastructure |
| SynthSAEBench (2026) | 2602.14687 | Synthetic benchmark; related_work |
