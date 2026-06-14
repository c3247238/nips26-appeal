# Comparativist Analysis: SAE Feature Absorption — EDA as Geometric Fingerprint

**Agent:** sibyl-comparativist  
**Date:** 2026-04-13  
**Workspace:** sae-absorption  
**Experiment state:** iter_002 pilot complete; audit blocked writing on 3 critical flags

---

## 1. Baseline Landscape

The results must be evaluated against the existing literature on (a) absorption detection metrics, and (b) theoretical accounts of absorption. The table below summarizes directly comparable published numbers.

### Table 1: Absorption Detection Performance — Published vs. This Work

| Method | Model / Layer | Metric | Published Value | This Work | Delta |
|---|---|---|---|---|---|
| Chanin et al. absorption rate (sae-spelling) | Gemma Scope 16k, multiple layers | Absorption rate | 15–35% across SAEs | ~96% saturation across GPT-2 Small all layers | Not comparable; different model and saturation regime |
| ATM SAE (Li et al., 2025) | Gemma-2-2B | Absorption score (lower=better) | 0.0068 | N/A (no training) | — |
| TopK SAE (Gao et al., 2024) | Gemma-2-2B | Absorption score | 0.1402 | N/A | — |
| JumpReLU SAE (Rajamanoharan et al., 2024) | Gemma-2-2B | Absorption score | 0.0114 | N/A | — |
| OrtSAE (Korznikov et al., 2025) | Various | Absorption reduction | 65% vs. TopK baseline | N/A (no training) | — |
| Feature Sensitivity (Tian et al., 2025) | Gemma Scope | AUROC (sensitivity vs. absorption) | Not directly AUROC for absorption | Baseline comparison: N/A | — |
| **EDA — this work** | GPT-2 Small, Layer 6 | AUROC (letter features vs. non-letter) | No prior published number for EDA | **0.681** (Cohen's d = 0.70) | First result |
| **ASI — this work** | GPT-2 Small, Layer 6 | AUROC | No prior published number for ASI | **0.422** (below random null 0.497) | FAILED |
| Freq ratio alone | GPT-2 Small, Layer 6 | AUROC | No prior published | 0.613 | — |
| cos² alone | GPT-2 Small, Layer 6 | AUROC | No prior published | 0.206 | FAILED |

**Key observation:** No prior paper has published a feature-level AUROC for probe-free absorption detection. This means EDA's AUROC of 0.681 cannot be directly compared against a published baseline on the same task. The iter_001 reference value for EDA was 0.629, so iter_002 represents a +0.052 improvement within the same experimental setup — but this is an internal comparison, not external validation.

**Critical caveat:** The absorption labels used in this work are proxy labels (SAE features with cos > 0.32 alignment to letter probes), not the ground-truth Chanin et al. labels obtained via integrated-gradients ablation. The label quality directly impacts AUROC validity.

---

## 2. Contribution Margin

### 2.1 EDA as Probe-Free Detector (AUROC = 0.681)

- **Versus random baseline (AUROC = 0.500):** Delta = +0.181. DeLong test vs. null: p < 10⁻⁷.
- **Versus freq_ratio (AUROC = 0.613):** Delta = +0.068. This is a moderate improvement.
- **Versus cos² alone (AUROC = 0.206):** EDA is dramatically better, confirming that the encoder direction (not just decoder geometry) carries the signal.
- **Versus ASI (AUROC = 0.422):** EDA is better by +0.259; ASI is worse than random.

**Contribution margin classification:** EDA vs. random = **MODERATE** (+0.181 AUROC). This is not a large margin. An AUROC of 0.681 on a heavily imbalanced task (0.29% base rate, n_pos=71 out of 24,576) is meaningful but not dramatic. Precision@500 = 0.008 = 0.28% — barely above the 0.29% base rate.

**Important framing issue:** The result lacks an external baseline because no prior probe-free method exists. The contribution is the existence of a working signal, not a large magnitude improvement over an established baseline.

### 2.2 Rate-Distortion Threshold (lambda > sin²(theta)) — FALSIFIED

- **Predicted direction (H1):** Absorbed features should have higher decoder-decoder cosine similarity with parents.
- **Observed direction:** Absorbed (letter) features have LOWER cos² with non-letter features (pos_mean = 0.044 vs. neg_mean = 0.086; Cohen's d = -0.48; AUROC = 0.318, well below random).
- **Contribution margin:** NEGATIVE. The theory makes a wrong directional prediction about post-convergence decoder geometry. The revised explanation (decoder drift post-absorption) is a plausible reconciliation but is untested and should be clearly flagged as a conjecture, not a proven result.

### 2.3 Cross-Domain Absorption (H2) — NOT SUPPORTED

- C2 pilot shows absorption_rate = 0.0 for all non-spelling hierarchies (entity type, geographic, grammatical).
- No signal above shuffled-label null on any non-spelling domain.
- **Contribution margin:** Zero for cross-domain claim. The original proposal's framing ("first cross-domain characterization") cannot be made. The negative result — that first-letter absorption does not generalize to semantic hierarchies — is itself informative but is a narrow and partial result.

### 2.4 Phase Transition (H4) — NOT SUPPORTED

- LRT p = 0.456; BIC difference = -3.22 favoring sigmoid over linear in B2 sparsity analysis only marginally and not significantly.
- Absorption rate is uniformly near-saturation (~96%) across all 11 tested configurations.
- **Contribution margin:** No evidence for a phase transition. The saturation regime means the prediction cannot be tested: if all SAEs are past the transition point, there is no observable transition with this dataset.

---

## 3. Concurrent Work Scan

Based on literature review and web search (April 2026):

### 3.1 Direct Competitors for Probe-Free Absorption Detection

**Feature Sensitivity (Tian et al., arXiv:2509.23717, 2025):** Frames absorption as a special case of poor feature sensitivity. Develops scalable sensitivity evaluation methods but does not compute AUROC as a detection metric. Focuses on a different geometric measure. No direct AUROC comparison is possible. The EDA metric is distinct from sensitivity: sensitivity measures how reliably a feature fires on semantically similar inputs, while EDA measures encoder-decoder alignment. These are complementary metrics with different failure modes.

**KronSAE (arXiv:2505.22255, 2025):** Introduces three absorption-related metrics (mean absorption fraction, full-absorption score, feature splits) but these are training-architecture metrics, not probe-free detectors applied post-hoc to a trained SAE.

**SAEBench absorption metric (Karvonen et al., 2025):** Uses integrated-gradients ablation to measure absorption rate — this is a supervised method requiring known probe directions. EDA targets a different sub-problem: identifying which features are absorbed without specifying what they are absorbed into.

**Conclusion:** No concurrent paper publishes an unsupervised, probe-free AUROC-based method for identifying absorbed features in a trained SAE. EDA is the first such reported result. However, the AUROC is moderate (0.681), and the labels are proxy labels, so the practical value is limited.

### 3.2 Direct Competitors for Absorption Theory

**arXiv:2512.05534 (unified SDL theory, 2024):** Proves that absorption solutions exist as spurious minima in the piecewise biconvex optimization landscape. This is the closest theoretical competitor. Critically, it does NOT derive the specific geometric threshold lambda > sin²(theta) or Corollary 1 (frequency cancels). The current work's Proposition 1 is genuinely novel relative to this paper.

**arXiv:2506.15963 (On the Limits of SAEs, Cui et al., 2025):** Provides closed-form analysis of when SAEs fail to recover ground truth features. Uses a different theoretical framework (statistical recovery guarantees, not rate-distortion). Does not derive absorption thresholds.

**arXiv:2502.16681 (2026):** Reports that SAE probes underperform logistic regression baselines in each evaluation regime — supports the view that absorption is widespread and fundamental, consistent with the saturation finding.

**Tilde Research blog (informal):** Mentions rate-distortion framing informally without formal derivation. We cite this as a precursor; Proposition 1 formalizes it.

**Conclusion on theory:** Proposition 1 (lambda > sin²(theta)) and Corollary 1 (frequency cancels) are genuinely novel formal results relative to published work as of April 2026. However, the empirical falsification of the predicted direction undermines the claim that the theorem predicts absorption in practice.

### 3.3 No Concurrent Work Found for EDA Mechanism

Web search returns no paper using EDA (Encoder-Decoder Alignment/Dissociation) as defined in this work as a fingerprint of absorption. The closest adjacent result is the "Select-and-Project" paper (arXiv:2509.10809) which notes that encoder and decoder features can diverge, but this is for a different purpose (control) rather than absorption detection.

---

## 4. Novelty Verdict

**The ONE thing this work does that no prior work does:**

> This work provides a mechanistic explanation and empirical detection signal for feature absorption based on the *intra-feature* encoder-decoder angle (EDA), and formally proves the rate-distortion optimality condition (lambda > sin²(theta)) with the novel corollary that co-occurrence frequency cancels from the threshold.

This is defensible as a genuine first. However, it is split between a theoretical contribution (Proposition 1) and an empirical detection contribution (EDA AUROC = 0.681 on proxy labels, single model, single layer). Neither half alone would clear the novelty bar for a top-tier venue; together they form a moderate contribution.

**What the work CANNOT claim as novel:**

- Cross-domain characterization of absorption (C2 null result: not observed).
- Probe-free detection with high accuracy (EDA AUROC = 0.681 on proxy labels ≠ validated ground truth).
- Phase transition / hysteresis (H4 not supported; saturation regime throughout).
- Absorption Susceptibility Index (H3 falsified: ASI AUROC = 0.422, anti-correlated with absorption).

---

## 5. Venue Recommendation

### Assessment Criteria

| Criterion | Assessment |
|---|---|
| Core theoretical result | Proposition 1 is formal and novel. Corollary 1 (frequency cancels) is counterintuitive and testable. But empirical validation is mixed (direction falsified; EDA reconciliation is a conjecture). |
| Core empirical result | EDA AUROC = 0.681 on proxy labels, single model (GPT-2 Small), single layer (L6). Not validated on ground truth labels from Chanin et al.; not validated on Gemma Scope. |
| Falsified hypotheses | H1, H3, H4 explicitly falsified or unsupported. Papers that report falsifications are publishable but require honest framing. |
| Comparison scope | Only GPT-2 Small tested. SAEBench / Gemma Scope results absent. This is a significant weakness vs. the field standard. |
| Concurrency risk | No direct concurrent paper on EDA-based detection. Theory is ahead of the literature. |

### Recommended Venue

**Mid-tier or specialized workshop — NOT top-tier (NeurIPS/ICML/ICLR) in current state.**

Specifically:
- **EMNLP / BlackboxNLP workshop (highest realistic target):** The mechanistic interpretability community overlaps with NLP venues. CE-Bench (arXiv:2509.00691) was accepted at BlackboxNLP 2025. EDA + Proposition 1 + honest falsification framing is plausible for this venue.
- **ICLR 2027 (with revisions):** If EDA is validated on Gemma Scope ground-truth labels and Proposition 1 empirical validation is completed (actual theta measurement for absorbed pairs), this could reach ICLR. Comparable papers at ICLR include TopK SAE (Gao et al., arXiv:2406.04093, ICLR 2025) which had substantially more thorough empirical validation.
- **Workshop on Mechanistic Interpretability (e.g., MI@ICML or Alignment Forum track):** Most appropriate for the current results given their depth of theory but narrowness of empirical scope.

**Why NOT NeurIPS/ICML in current form:**
- The primary empirical result (EDA) uses proxy labels, single model, single layer.
- Three of four core hypotheses falsified or unsupported.
- No validation on Gemma Scope (the field standard for SAE evaluation).
- Absorption rate saturation (~96%) means most of the experimental variance is unexploitable.

**Comparable paper at intended venue (BlackboxNLP/EMNLP):** CE-Bench (Gulko et al., 2025) — lightweight evaluation on structured story pairs, limited scope but rigorous methodology, accepted at BlackboxNLP. The EDA result is similar in scope: a methodological contribution with limited but rigorous validation.

---

## 6. Strengthening Plan

Three specific additions that would maximally strengthen the paper's positioning:

### 6.1 (HIGH PRIORITY) Validate EDA on Gemma Scope with Chanin et al. Ground-Truth Labels

The current EDA result (AUROC = 0.681) uses proxy labels (decoder alignment threshold), not the ground-truth integrated-gradients absorption labels from Chanin et al. (arXiv:2409.14507). The Chanin et al. code (sae-spelling) is publicly available (MIT license) and can be run on Gemma Scope SAEs. This would:
- Convert proxy labels to ground-truth labels on a model/SAE combination that the field benchmarks.
- Enable direct comparison against the SAEBench absorption metric.
- Establish EDA on the canonical evaluation target (Gemma 2 2B).

Expected time: 2–4 hours on a GPU. Expected outcome: either validates EDA (strongly strengthens the contribution) or reveals EDA is a proxy-label artifact (changes the paper story).

### 6.2 (MEDIUM PRIORITY) Measure Actual theta for Confirmed Absorbed Pairs

The unresolved tension in the theory — absorbed features have LOWER decoder-decoder cosine (unexpected direction) — can be resolved by measuring theta for feature pairs that are confirmed absorbed via integrated-gradients ablation. This tests whether Proposition 1 describes training-time geometry (small theta at absorption onset) or is simply not the mechanistic cause of the observed absorption in trained SAEs.

Expected time: 2–3 hours. This directly addresses the "direction falsified" audit flag.

### 6.3 (LOWER PRIORITY BUT HIGH NARRATIVE IMPACT) Report the Negative Results Systematically

H2 (cross-domain) null result and H4 (phase transition) non-support are potentially publishable as negative results if framed correctly. The null result for cross-domain absorption (entity type, geographic, grammatical hierarchies show zero absorption) is itself informative: it suggests that the first-letter task is unusually susceptible to absorption and may not generalize to safety-relevant semantic hierarchies. This reframes the "failed" H2 as a scoping result: the scope of absorption may be narrower than assumed in the field. This requires honest reporting and appropriate null control analysis (which was already partially implemented in C2 pilot).

---

## 7. Risk Assessment: Is the Contribution Sufficient for Any Publication?

### What Survives Honest Scrutiny

| Result | Status | Publication Value |
|---|---|---|
| Proposition 1 (lambda > sin²(theta)) + Corollary 1 (frequency cancels) | PROVEN (comparison of two candidate solutions) | Moderate novelty; limited practical validation |
| EDA AUROC = 0.681 at GPT-2 Small L6 | Empirical, proxy labels only | First result of this type; limited scope |
| H1 direction falsification (absorbed features have LOWER cos² with parents) | Empirical, statistically significant | Interesting negative result |
| H2 null (cross-domain absorption absent) | Empirical, but C2 full analysis not complete | Informative scope limitation |
| H3 falsification (ASI anti-correlated with absorption) | Empirical, statistically significant | Identifies what DOES NOT work |
| H4 non-support (no phase transition visible) | Empirical | Saturation interpretation required |

### Conservative Assessment

The combination of (1) a formal theorem with a counterintuitive corollary and (2) EDA as the first probe-free absorption detector with an empirically grounded mechanistic explanation is a publishable unit — but at a modest venue. The key risk is that EDA may not survive validation on ground-truth labels. Until that validation is done, any submission should be treated as preliminary.

**Minimum viable paper (current data):** A short paper or workshop contribution presenting Proposition 1 + Corollary 1 as a theoretical contribution, EDA as an empirical finding (with honest caveats about proxy labels and single model), and the falsification of ASI and the RD direction as informative negative results. This is a 6–8 page workshop paper.

**Stronger paper (after 6.1 + 6.2):** If EDA validates on ground-truth Gemma Scope labels and the theta measurement clarifies the theory-data tension, this becomes a full EMNLP or ICLR paper.

---

## Sources

- [A is for Absorption (Chanin et al., 2024)](https://arxiv.org/abs/2409.14507)
- [SAEBench (Karvonen et al., 2025)](https://arxiv.org/abs/2503.09532)
- [Unified SDL Theory / Piecewise Biconvexity (2024)](https://arxiv.org/abs/2512.05534)
- [On the Limits of SAEs (Cui et al., 2025)](https://arxiv.org/abs/2506.15963)
- [Measuring SAE Feature Sensitivity (Tian et al., 2025)](https://arxiv.org/abs/2509.23717)
- [Matryoshka SAE (Bussmann et al., 2025)](https://arxiv.org/abs/2503.17547)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [ATM SAE (Li et al., 2025)](https://arxiv.org/abs/2510.08855)
- [SAEBench Neuronpedia interactive](https://www.neuronpedia.org/sae-bench/info)
- [TopK SAE / Scaling SAEs (Gao et al., ICLR 2025)](https://arxiv.org/abs/2406.04093)
- [CE-Bench (Gulko et al., BlackboxNLP 2025)](https://arxiv.org/abs/2509.00691)
- [SynthSAEBench (2026)](https://arxiv.org/pdf/2602.14687)
- [Negative Results for SAEs - DeepMind (2025)](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
- [Feature Hedging (Chanin et al., 2025)](https://arxiv.org/abs/2505.11756)
- [Rethinking MI with Domain-Specific SAEs (arXiv:2508.09363)](https://arxiv.org/html/2508.09363v1)
- [Select-and-Project SAE (arXiv:2509.10809)](https://arxiv.org/abs/2509.10809)
- [SAE probes underperform baselines (arXiv:2502.16681)](https://arxiv.org/html/2502.16681v1)
