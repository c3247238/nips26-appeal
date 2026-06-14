# Research Proposal (Iteration 2 — Synthesized from Six Perspectives)

## Title

**Unsupervised Absorption Fingerprinting and the Amortization Gap: A Unified Framework for Diagnosing and Separating the Root Causes of SAE Feature Absorption**

---

## Abstract

Feature absorption—the systematic failure of parent features to fire when child features absorb their activation budget—has been operationally characterized but mechanistically debated. This proposal unifies the strongest threads from six independent perspectives to produce a decisive, training-free research program built on three interlocking contributions:

**(1) Absorption Fingerprint Score (AFS):** A novel unsupervised, training-free absorption detector derived from the directed co-occurrence network of SAE latent activations. Drawing on the ecological competitive exclusion principle, we formalize the Absorption Risk Score (ARS) as a combination of Jaccard niche overlap, activation asymmetry, and decoder cosine similarity. ARS predicts absorbed latents without requiring any pre-specified probe directions, addressing the most critical unsolved gap in the absorption literature (Gap 7). Novelty: no prior work uses co-occurrence graph topology for post-hoc unsupervised absorption detection.

**(2) Amortization Gap Controlled Dictionary Experiment:** A decisive experiment separating two competing causal explanations for absorption—the sparsity optimization landscape (Tang et al./Shu et al., 2512.05534) vs. the encoder amortization gap (O'Neill et al., 2411.13117). We fix the pre-trained decoder dictionary and compare absorption rates under standard feedforward encoding vs. Orthogonal Matching Pursuit (OMP), directly adjudicating this mechanistic debate. The previous iteration's most important finding—that ~75% of absorbed latents are "early absorption" (decoder-absent, dictionary coverage failure)—motivates this test as the most impactful next experiment.

**(3) Cross-Hierarchy Absorption Characterization with Negative Controls:** A systematic measurement of absorption rates across four hierarchy types (first-letter, entity-type, syntactic, and negative-control non-hierarchical pairs with matched frequency ratios), testing whether the canonical first-letter metric generalizes to semantically richer hierarchies. This confirms or challenges existing architectural comparison results built on the first-letter task.

Together, these three contributions provide: (a) the first unsupervised absorption detection tool, (b) the first controlled mechanistic experiment separating encoder vs. dictionary causes of absorption, and (c) the first rigorous cross-hierarchy absorption characterization with proper negative controls. All work is training-free, using existing Gemma Scope SAEs loaded via SAELens.

---

## Motivation

Feature absorption has been characterized as a universal, consequential failure mode in SAE-based mechanistic interpretability (Chanin et al., NeurIPS 2025). DeepMind's deprioritization of SAE research for safety-relevant feature detection (2025) is the highest-stakes consequence. The field now has:
- A well-established measurement tool (sae-spelling, Chanin et al.) and benchmark (SAEBench)
- Multiple architectural mitigation proposals (OrtSAE, Matryoshka SAE, ATM SAE, Masked Regularization)
- A theoretical optimization landscape characterization (Tang et al., piecewise biconvex, 2512.05534)
- An empirical prior iteration finding that ~75% of absorbed latents are "early type" (decoder-absent dictionary coverage failures)

**What remains missing:**

**Gap 7 (Critical — addressed by Contribution 1):** No method exists for detecting absorbed features without pre-specified probe directions. Every current approach (Chanin metric, SAEBench absorption task) requires knowing which features to look for. An unsupervised detector would dramatically expand absorption analysis to all features in any SAE.

**Gap Mechanism (Critical — addressed by Contribution 2):** The field has two competing mechanistic explanations for absorption:
- Tang et al. (2512.05534): Absorption is a stable spurious minimum of the piecewise biconvex SDL loss. The dictionary geometry itself encodes absorption; encoder changes alone cannot fix it.
- O'Neill et al. (2411.13117): The amortization gap—the provable gap between feedforward encoder performance and optimal sparse inference—causes absorption. Iterative encoding methods should largely eliminate it.

These are not equally testable by analyzing existing SAEs; they require a controlled experiment that has never been done.

**Gap 2 (High priority — addressed by Contribution 3):** Absorption rates have only been measured on first-letter spelling tasks. All architectural comparisons in SAEBench rest on this single, artificially symmetric hierarchy. Whether OrtSAE's 65% absorption reduction or Matryoshka SAE's improvement extends to semantic hierarchies is entirely unknown.

**Additional post-experimental insight from Iteration 1:** The previous iteration's EDA detector achieves AUROC = 0.776 in favorable regimes but fails for the ~75% "early absorption" majority (decoder-absent cases where EDA has no signal). This motivates an alternative detection strategy—co-occurrence network topology—that is theoretically motivated to work even for early absorption (where the absorbing latent's high co-occurrence with the parent is observable regardless of decoder geometry).

---

## Research Questions

1. **RQ1 (Detection):** Does the Absorption Risk Score (ARS), derived from directed co-occurrence network topology, predict absorbed latents without requiring labeled probe data? What AUROC can it achieve on the Chanin et al. canonical validation set?

2. **RQ2 (Mechanism):** What fraction of feature absorption survives when the decoder dictionary is fixed and encoding method is switched from feedforward to OMP? Does this fraction change between early-absorbed and late-absorbed latents?

3. **RQ3 (Generalization):** Do absorption rates in entity-type hierarchies (ANIMAL ⊃ DOG) and syntactic hierarchies (VERB ⊃ IRREGULAR-PAST-TENSE) exceed absorption rates in negative-control non-hierarchical pairs with matched frequency ratios? Does frequency imbalance (parent-child frequency ratio ρ) predict absorption severity across hierarchy types?

4. **RQ4 (Theory):** Is there an empirically observable phase transition in absorption rate as a function of the parent-child frequency ratio ρ, consistent with the Critical Frequency Ratio Theorem derived by the theoretical perspective?

5. **RQ5 (Safety implications):** What fraction of the SAE probe vs. dense linear probe performance gap on harmful intent detection is attribution-attributable to absorption specifically?

---

## Hypotheses

**H1 (ARS Unsupervised Detection):**
The Absorption Risk Score ARS(i,j) = O(i,j) × A(i,j) × cos²(d_i, d_j) achieves AUROC > 0.70 against Chanin et al. canonical absorption labels on Gemma Scope 16k at layer 12. High-ARS pairs will have significantly higher ARS than random non-absorbed pairs. The directed co-occurrence asymmetry (A(i,j) = P(j fires | i fires) / P(i fires | j fires)) will be the highest-information single component when decomposed.

*Falsification criterion:* ARS AUROC < 0.55 on Gemma Scope 16k layer 12 (no better than chance); AND rank-ordering of high-ARS pairs does not show even directional enrichment for absorbed features.

**H2 (Amortization Gap Dominant Cause):**
If the amortization gap is the dominant cause of early absorption: OMP absorption rate < 50% of feedforward absorption rate on the same decoder dictionary at matched L0.
If the sparsity landscape is the dominant cause: OMP absorption rates are similar to feedforward (difference < 20%).
We pre-commit to reporting whichever direction is found; both outcomes are publishable.

*Falsification criterion for amortization gap dominance:* OMP absorption rate >= 80% of feedforward rate with same decoder.

**H3 (Cross-Hierarchy Generalization):**
Absorption rate for entity-type hierarchies (ANIMAL ⊃ specific animals) is > 10 percentage points higher than negative-control non-hierarchical pairs with matched frequency ratio (paired t-test, p < 0.05, Bonferroni-corrected). Frequency ratio ρ = p_child/p_parent is the strongest predictor of absorption rate across hierarchy types (Spearman ρ > 0.40).

*Falsification criterion:* Entity-type absorption rate is not significantly different from matched negative-control (p > 0.1 Bonferroni), AND ρ does not correlate with absorption across hierarchy types (Spearman ρ < 0.2).

**H4 (Absorption Phase Transition):**
The absorption rate curve as a function of ρ shows a step-like increase near a threshold ρ* that is predictable from the Critical Frequency Ratio Theorem: ρ*(k, p₁, λ) ∝ 1 - λk / (p₁ · const_geometry · (1-ε²)). Letters with ρ > ρ* will show systematically higher absorption than letters with ρ < ρ*.

*Falsification criterion:* No inflection point in absorption rate vs. ρ curve (flat relationship, R² < 0.15 for step-function fit).

**H5 (Safety Attribution):**
At least 20% of the cases where the SAE probe fails but the dense linear probe succeeds (false-negative rate) can be attributed to absorption, confirmed by: identifying the absorbing latent for each false-negative token and verifying that manually adding the absorbed parent latent converts classification from false-negative to true-positive.

*Falsification criterion:* Absorption-attributable fraction < 10% AND random latent addition produces equivalent improvement (confirming non-specific effect).

---

## Novelty Assessment

### Contribution 1: ARS Unsupervised Absorption Detector

**Novelty claim:** The ARS metric is the first unsupervised, training-free method for predicting SAE feature absorption from co-occurrence network topology, without requiring pre-specified probe directions or known feature hierarchies.

**Evidence no prior work has done this:**
- arXiv search "co-occurrence graph absorption SAE unsupervised detection" returns 0 relevant results (April 2026)
- The Geometry of Concepts (arXiv:2410.19750) studies co-occurrence structure but does not connect it to absorption detection
- OrtSAE uses decoder cosine similarity (not co-occurrence statistics) to reduce absorption structurally
- Masked Regularization (arXiv:2604.06495) uses token masking to disrupt co-occurrence during training, not to detect absorption post-hoc
- SAEBench absorption metric requires probe labels; Tian et al. sensitivity metric does not provide unsupervised detection
- ARS combines three factors (Jaccard overlap, activation asymmetry, decoder cosine similarity) in a product motivated by ecological competitive exclusion theory — this product form has not been proposed before

**Differentiation from EDA (previous iteration):** EDA measures encoder-decoder angular divergence and fails for ~75% early absorption cases (decoder-absent). ARS operates on co-occurrence statistics and activation patterns, which are observable regardless of whether the SAE has a decoder direction for the parent feature. ARS is theoretically complementary and practically superior for the early-absorption majority.

### Contribution 2: Amortization Gap Controlled Dictionary Experiment

**Novelty claim:** No prior paper has fixed the pre-trained decoder dictionary while varying encoding method to directly test the amortization gap vs. sparsity landscape explanations of absorption.

**Evidence no prior work has done this:**
- MP-SAE (Costa et al., arXiv:2506.03093): uses iterative encoding but RETRAINS the decoder jointly — does not isolate encoder effects
- O'Neill et al. (2411.13117): proves the amortization gap exists theoretically but does not run the controlled dictionary experiment
- Tang et al. (2512.05534): argues for sparsity landscape as the cause but does not hold the dictionary fixed to test this

**Impact:** Both possible outcomes are highly citable — if OMP eliminates absorption, the field should immediately switch to iterative encoders for safety-sensitive applications; if OMP does not eliminate absorption, the sparsity landscape explanation is confirmed and architectural changes (Matryoshka, OrtSAE) are the correct lever.

### Contribution 3: Cross-Hierarchy Absorption Characterization

**Novelty claim:** No prior paper measures SAE feature absorption in entity-type or syntactic hierarchies with proper negative controls (non-hierarchical pairs matched on frequency ratio). The contrarian perspective's key insight is confirmed: the first-letter task uses an artificially symmetric hierarchy, and its absorption rates may not predict rates in the asymmetric hierarchies relevant to safety applications.

**Evidence no prior work has done this:**
- All published absorption measurements use the first-letter spelling task (Chanin et al., 2024; SAEBench, 2025)
- "Resurrecting the Salmon" (arXiv:2508.09363) explicitly states first-letter metric is "not useful for evaluating domain-specific feature absorption"
- No paper includes matched-frequency-ratio negative controls for the absorption metric

---

## Revisions from Prior Iteration (Iter 001)

**What changed:**

1. **EDA repositioned as a solved sub-problem.** The previous iteration established EDA as a regime-specific weight-only detector (AUROC = 0.776 at L12-16k). This is now treated as a confirmed contribution whose results should be published; the current iteration does not re-run EDA experiments but instead builds on its findings.

2. **ARS proposed as EDA's complement for early absorption.** The 75% early-absorption dominance finding (H4 from iter_001) directly motivates ARS: EDA has no signal for early absorption (decoder-absent), but co-occurrence topology can detect it because the absorbing child's high co-occurrence with the parent is observable regardless of decoder geometry.

3. **Amortization gap experiment elevated to primary.** Iter_001's early-dominance finding creates an immediate question: what drives early absorption? The controlled dictionary experiment (Backup A from iter_001) is now elevated to a primary contribution because it can definitively answer whether early absorption is driven by the encoder (amortization gap), the training objective (sparsity landscape), or the dictionary structure.

4. **Cross-hierarchy characterization refined with pre-specified negative controls.** Iter_001's cross-domain (RAVEL) result was confounded by wrong-model probes. The current proposal specifies proper controls and a clear methodology that can be executed in one iteration.

5. **Phase transition theory integrated.** The theoretical perspective's Critical Frequency Ratio Theorem provides a testable empirical prediction that is tested within the cross-hierarchy experiment (H4) using the 26-letter natural sweep of ρ values.

6. **Safety attribution analysis added** as a concrete downstream consequence study (H5), building on the contrarian's challenge that absorption may not be the primary driver of probe gaps.

---

## Method

All experiments are training-free, using pre-trained Gemma Scope SAEs (Gemma-2-2B, residual stream) loaded via SAELens. No gradient computation is required.

### Contribution 1: ARS Unsupervised Absorption Detector

**Step 1: Build directed co-occurrence graph**
- Load Gemma Scope 16k SAE at layer 12 via SAELens
- Pass 200k tokens from OpenWebText through the model; record all SAE latent activations above threshold
- For each ordered pair (i,j): compute c_{i→j} = P(j fires | i fires) and f_i = activation frequency of latent i
- Compute the directed absorption propensity: abs_prop_{i→j} = c_{i→j} × (f_j / f_i) [high when i fires rarely relative to j and tends to co-fire with j]

**Step 2: Compute Absorption Risk Score (ARS)**
- O(i,j) = |activations_i ∩ activations_j| / |activations_i ∪ activations_j| (Jaccard overlap of activation sets)
- A(i,j) = max(f_i/f_j, f_j/f_i) (activation asymmetry; high when one fires much more than other)
- cos²(d_i, d_j) = squared decoder direction cosine similarity
- ARS(i,j) = O(i,j) × A(i,j) × cos²(d_i, d_j)
- The Absorption Fingerprint Score (AFS) for latent i = max_{j: f_j > 3×f_i} [ARS(i,j)]

**Step 3: Validate against Chanin et al. canonical labels**
- Run sae-spelling absorption metric on 5 letters (A, E, I, O, T) for Gemma Scope 16k layer 12
- This produces ground-truth absorbed vs. non-absorbed latents for those letters
- Compute AUROC, AUPR for predicting which latents are absorption-positive using AFS
- Decompose ARS components: compare AUROC of O alone, A alone, cos² alone vs. full ARS product

**Step 4: Cross-width and cross-architecture comparison**
- Repeat ARS computation for Gemma Scope 1k, 4k, 65k SAEs at layer 12
- Test whether ARS statistics (mean, 95th percentile) correlate with SAEBench absorption scores
- Check whether ARS predicts early-absorbed latents (from iter_001 taxonomy) better than late-absorbed latents

### Contribution 2: Amortization Gap Controlled Dictionary Experiment

**Step 1: Load pre-trained decoder dictionary**
- Load Gemma Scope L12-16k SAE weights (W_dec: [16k, 2304])
- This is the fixed dictionary D that will be used across all encoding methods

**Step 2: Three encoding conditions**
- **Condition A (Feedforward):** Standard SAE encoder (W_enc @ x + b_enc → ReLU/TopK)
- **Condition B (OMP):** Orthogonal Matching Pursuit on D at matched K (same average L0 as Condition A)
- **Condition C (2-pass):** Feedforward encoding + one residual correction pass (a = encoder(x); r = x - D @ a; a += encoder(r) with step size 0.5)

**Step 3: Measure absorption rates under all three conditions**
- Pass 10,000 tokens from OpenWebText through the model
- For each encoding condition: extract latent activations on all tokens
- Adapt the Chanin et al. absorption metric to measure absorption under each encoding condition (same probe directions, same canonical threshold)
- Report absorption rate per letter × encoding condition matrix

**Step 4: Analysis**
- Paired Wilcoxon signed-rank test: Condition A vs. B, A vs. C, B vs. C on per-letter absorption rates
- Compute "amortization gap reduction ratio": (AR_feedforward - AR_OMP) / AR_feedforward
- Stratify by absorption subtype (early vs. late, using iter_001 taxonomy): report whether amortization gap primarily affects early or late absorption

### Contribution 3: Cross-Hierarchy Absorption Characterization

**Step 1: Construct probe datasets**
- **Hierarchy 1 (Alphabetical, baseline):** First-letter task per Chanin et al. — use sae-spelling directly
- **Hierarchy 2 (Entity-type):** ANIMAL ⊃ {dog, cat, bird, fish, horse}. Construct 200 templated sentences per class from OpenWebText distribution. Train LR probe on Gemma-2-2B layer 12 residual stream (must achieve > 80% accuracy)
- **Hierarchy 3 (Syntactic):** PAST-TENSE ⊃ IRREGULAR-PAST-TENSE. Use Penn Treebank / Universal Dependencies annotated data. Train LR probe (must achieve > 75% accuracy)
- **Hierarchy 4 (Negative control):** Two co-occurring but non-hierarchical features. Use "mentions weather" AND "mentions time of day" — these co-occur in weather reports but neither logically implies the other. Select pairs matched on frequency ratio ρ with Hierarchy 2. (WordNet check: neither appears on the other's hypernym path)

**Step 2: Compute frequency ratios**
- For each hierarchy, compute ρ = p_child/p_parent from OpenWebText token counts
- Also compute conditional activation probability P(parent fires | child fires) from SAE activation statistics

**Step 3: Run absorption measurement**
- Apply sae-spelling k-sparse probing + integrated-gradients pipeline to each hierarchy type
- Use Gemma Scope 16k at layer 12 for all hierarchies (controlled comparison)
- Report absorption rate per hierarchy with 3 probe training seeds; report mean ± std

**Step 4: Statistical analysis**
- Paired t-test: Hierarchy 2 vs. Hierarchy 4 (entity-type vs. matched negative control)
- Spearman correlation: absorption rate vs. ρ across all positive hierarchy types
- Phase transition test: plot absorption rate vs. ρ for 26 letters (natural ρ sweep); fit step-function and check R² vs. linear fit (H4 test)

---

## Experimental Plan

| Priority | Experiment | Resources | Time | Validates |
|----------|-----------|-----------|------|---------|
| 1 — PILOT | ARS computation on Gemma Scope 16k L12; top-10 high-ARS pairs vs. Chanin labels on 3 letters | 1 GPU, 10 min | 10 min | H1 pilot — go/no-go for ARS |
| 1 — PILOT | Feedforward vs. OMP on Gemma Scope L12-16k, 1000 tokens, 3 letters | 1 GPU, 15 min | 15 min | H2 pilot — go/no-go for amortization gap |
| 2 — MAIN | Full ARS validation: 5 letters × 4 widths (1k/4k/16k/65k) × layer 12 | 1 GPU, ~3 hr total (≤1 hr/width) | 3 hr | H1 full AUROC validation |
| 2 — MAIN | Amortization gap: feedforward vs. OMP vs. 2-pass, 10k tokens, 26 letters, L12-16k | 1 GPU, ~2 hr | 2 hr | H2 definitive answer |
| 2 — MAIN | Cross-hierarchy: entity-type + syntactic + negative control, 3 seeds each, L12-16k | 1 GPU, ~3 hr | 3 hr | H3 generalization test |
| 3 — SUPPLEMENTARY | ρ sweep: plot absorption rate vs. ρ for 26 letters, fit step-function (phase transition test) | CPU, ~30 min | 30 min | H4 phase transition |
| 3 — SUPPLEMENTARY | Safety attribution: AdvBench + ToxiGen SAE probe vs. linear probe gap attribution | 1 GPU, ~2 hr | 2 hr | H5 safety implication |
| 4 — CROSS-ARCHITECTURE | ARS on 5 SAEBench architectures (TopK, Gated, JumpReLU, Matryoshka, BatchTopK) at 16k | 1 GPU, ~2 hr | 2 hr | ARS as cheap SAEBench proxy |

**Total estimated GPU-hours:** 12-16 GPU-hours (all forward passes, no training)
**Pilot decision gate:** If ARS pilot AUROC < 0.55 (Contribution 1) AND OMP reduces absorption < 5% (Contribution 2) — pivot to Contribution 3 as primary (cross-hierarchy characterization is independently motivated and does not depend on ARS or amortization gap results)

---

## Baselines

1. **Chanin et al. first-letter absorption rates** (15–35% on Gemma Scope 16k/65k) — exact replication required before extension
2. **Random latent ranking** — null baseline for ARS AUROC (should give 0.50)
3. **Decoder cosine similarity alone** — single-component ARS baseline (tests whether the product form adds information)
4. **Dense linear probe performance** — reference for safety attribution analysis (from DeepMind 2025 blog)
5. **SAEBench pre-computed absorption scores** — for correlation with ARS statistics

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ARS AUROC < 0.60 (no signal for early absorption via topology) | 35% | High | Pivot to ARS as supplementary; focus paper on amortization gap + cross-hierarchy; the directed asymmetry A(i,j) component may be retained even if full ARS product underperforms |
| OMP shows no reduction in absorption (sparsity landscape dominates) | 40% | Medium | Publishable negative result confirming Tang et al.; recommend hierarchically-aware training objectives as the only viable fix |
| Entity-type probe accuracy < 80% at all layers | 30% | Medium | Use syntactic hierarchy as primary (more cleanly linearly represented); report entity-type as preliminary |
| Negative control shows same absorption rates as positive hierarchies | 20% | High | Investigate if the metric is hierarchy-specific at all; reframe as metric validation paper |
| sae-spelling code incompatible with current SAELens | 20% | Low | Pin to compatible version; 2-4 hr engineering cost |
| Gemma-2-2B model access blocked | 15% | Medium | Use Llama-3.1-3B with publicly available SAEs as fallback; acknowledge as proxy |

---

## Resource Estimate

- **Models:** Gemma-2-2B (via Gemma Scope, non-commercial research license); GPT-2-small (EleutherAI SAEs, MIT) for replication
- **GPU:** ~12-16 A100-hours total (all forward passes, no training)
- **Storage:** ~20 GB for activation caches (reusable across experiments)
- **Key code:** sae-spelling (Chanin et al., MIT), SAELens (MIT), Gemma Scope (HuggingFace)
- **Datasets:** OpenWebText (training corpus), Penn Treebank (syntactic), WordNet (entity-type hierarchy), AdvBench + ToxiGen (safety)

---

## Expected Contributions

### Primary

**C1 (Novel):** Absorption Fingerprint Score (ARS) — first unsupervised absorption detector based on co-occurrence network topology. If AUROC > 0.70, enables community-wide absorption auditing without probe labeling.

**C2 (Novel):** Amortization Gap vs. Sparsity Landscape controlled experiment — first direct mechanistic evidence separating encoder from dictionary causes of absorption. Either outcome redirects the mitigation research program.

**C3 (Novel):** Cross-hierarchy absorption characterization with matched negative controls — first systematic test of whether absorption generalizes beyond the first-letter task, and first measurement of frequency ratio as a cross-hierarchy predictor.

### Secondary

**C4 (Supporting):** Empirical phase transition test (H4) — tests the Critical Frequency Ratio Theorem prediction using the natural ρ sweep across 26 letters.

**C5 (Supporting):** Safety attribution analysis — quantifies what fraction of the SAE-vs-linear-probe gap is absorption-attributable, providing evidence-based motivation for absorption mitigation in safety contexts.

---

## Venue Target

**Initial target:** NeurIPS 2026 (main track) if all three primary contributions are confirmed (ARS AUROC > 0.70, amortization gap result is decisive, cross-hierarchy generalization confirmed). Strong story combining theory (C2 mechanism), detection (C1 method), and generalization (C3 scope).

**Fallback:** EMNLP 2026 / NeurIPS 2026 MI Workshop if only 2/3 contributions are confirmed; the amortization gap experiment alone is a workshop-level contribution with main-track potential depending on effect size.

---

## Evidence-Driven Revisions from Iteration 1

The following changes from the previous proposal are grounded in empirical findings from iter_001:

1. **EDA contribution preserved but not repeated.** EDA (AUROC = 0.776 at L12-16k) is a completed contribution; we do not re-run these experiments but instead build on the regime-specific characterization finding.

2. **ARS motivated by EDA's blindspot.** The 75% early-absorption finding (iter_001 H4) is the empirical driver for why a co-occurrence topology approach is needed: EDA fails for the majority case (early absorption, decoder-absent), but ARS can detect the absorbing relationship via co-occurrence even without decoder signal.

3. **Amortization gap experiment elevated.** Iter_001 Backup A identified this experiment as high-priority but did not run it. With the 75% early-dominance finding now confirmed, the amortization gap experiment answers "what causes early absorption?" — the field's most important open question.

4. **RAVEL replaced by better-controlled cross-hierarchy study.** Iter_001's RAVEL cross-domain result was confounded by wrong-model probes. The current proposal uses proper Gemma-2-2B probes and adds matched negative controls, making the result conclusive.

5. **Phase transition theory added.** The theoretical perspective's Critical Frequency Ratio Theorem provides a specific testable prediction (step-function in absorption rate vs. ρ) that can be tested as a supplementary experiment within the cross-hierarchy study.
