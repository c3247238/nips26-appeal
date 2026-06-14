# Research Proposal: When Sparsity Eats Its Young — Feature Absorption as Rate-Distortion Optimal Behavior, Cross-Domain Characterization, and Probe-Free Detection in Sparse Autoencoders

## Title

**When Sparsity Eats Its Young: Feature Absorption as Rate-Distortion Optimal Behavior, Cross-Domain Characterization, and Probe-Free Detection in Sparse Autoencoders**

---

## Abstract

Feature absorption — the systematic failure of SAE latents to fire on hierarchically related inputs — has been identified as the most fundamental reliability failure in sparse autoencoders. Yet despite its recognized importance, three critical gaps remain: (1) no theory explains *why* absorption is as severe as it is for any given SAE configuration; (2) absorption has been measured only on a single artificial proxy task (first-letter spelling); and (3) no method detects absorbed features without pre-specified probe directions. We address all three gaps in a unified, primarily training-free research program. Our central contribution is a formal proof that feature absorption is the **rate-distortion optimal behavior** under flat sparsity penalties when features are hierarchically structured: the SAE is not failing at optimization — it is succeeding at optimizing the wrong objective. This reframing yields a quantitative absorption threshold (a function of decoder cosine similarity and sparsity penalty strength) that makes falsifiable, testable predictions — including the counterintuitive finding that co-occurrence frequency cancels from the threshold, so absorption risk is determined entirely by decoder geometry, not by how often features co-occur. We validate this theory empirically across multiple semantic hierarchy types (first-letter spelling, entity type, geographic, grammatical) on Gemma 2 2B using Gemma Scope pre-trained SAEs, establishing the first systematic cross-domain characterization of absorption, with rigorous null controls. In parallel, we develop and validate an **Absorption Susceptibility Index (ASI)** — a probe-free, training-free metric computed from SAE weights and activation statistics that predicts which features are at risk of absorption. Finally, we demonstrate that absorption exhibits **phase-transition dynamics** with potential hysteresis, implying that post-hoc sparsity reduction may not reliably reverse established absorption. These contributions collectively explain, predict, detect, and characterize feature absorption at a depth not previously achieved.

---

## Motivation

Feature absorption is arguably the most serious failure mode of SAEs for mechanistic interpretability. It creates a dangerous false confidence: an SAE appears to have learned a clean monosemantic latent for a concept, but that latent systematically fails to fire on 15–35% of inputs where the concept is present (Chanin et al., 2024). DeepMind's safety team found that this failure extends to safety-relevant applications: SAE probes fail catastrophically at detecting harmful intent while dense linear probes succeed even out-of-distribution. Despite this importance, the current state of the field has three critical blind spots:

**Blind spot 1: No causal theory of absorption severity.** Chanin et al. (2024) showed that absorption exists in every tested SAE and provided an informal argument: absorption "saves one L0 per parent-child pair." This is actually a rate-distortion statement in disguise, but it has never been formalized. Without a theory, practitioners cannot predict how severe absorption will be for a new SAE configuration or which architectural choices will reduce it.

**Blind spot 2: Single-task empiricism.** The entire quantitative understanding of absorption rests on one measurement task: whether SAE features for first-letter membership ("starts with E") absorb specific token features ("elephant", "eight", etc.). SAEBench explicitly acknowledges: "SAEBench evaluates feature absorption by using features for 'word starts with X', which is not useful for evaluating domain-specific feature absorption." We have no systematic evidence for whether absorption affects the semantically rich hierarchies (city → country, entity type → specific entity) that matter for safety applications.

**Blind spot 3: Supervised-probe dependency.** Every absorption metric requires the researcher to specify in advance which features to look for. This prohibits any broad-spectrum absorption survey. If a model has absorbed a safety-relevant feature that the researcher did not know to look for, no current method would detect it.

The rate-distortion framework provides the conceptual key to unlocking all three blind spots simultaneously. By formalizing "absorption saves L0" as a rate-distortion optimization, we derive (1) a closed-form absorption threshold that makes quantitative predictions; (2) a probe-free Absorption Susceptibility Index computable from decoder geometry; and (3) a phase-transition account of absorption onset that explains why simply reducing sparsity may not reverse absorption that has already occurred.

---

## Research Questions

1. **RQ1 (Theory):** Is feature absorption the unique rate-distortion optimal behavior under flat sparsity penalties for hierarchically structured features? What is the closed-form absorption threshold as a function of measurable SAE properties?

2. **RQ2 (Cross-domain):** Does absorption occur at comparable rates across semantically distinct feature hierarchies (spelling, entity type, geographic, grammatical)? What hierarchy properties predict absorption severity?

3. **RQ3 (Probe-free detection):** Does the Absorption Susceptibility Index (ASI) — derived from decoder geometry and activation statistics alone — predict which feature pairs will exhibit absorption with sufficient precision and recall to be practically useful (AUROC ≥ 0.70)?

4. **RQ4 (Dynamics):** Is absorption onset a phase transition in the sparsity parameter? Is it reversible (continuous crossover) or does it exhibit hysteresis (first-order transition with metastable absorbed states)?

---

## Hypotheses

**H1 (Rate-Distortion Optimality):** For any parent feature p and child feature c, the SAE loss landscape has a solution where c's decoder absorbs p's information that achieves strictly lower loss than the non-absorption solution when:
```
lambda > sin^2(theta_{p,c})
```
where theta is the angle between parent and child decoder directions. Crucially, the co-occurrence frequency p_co cancels from this expression, so absorption risk is determined entirely by decoder geometry and sparsity penalty, not by frequency. This is a falsifiable prediction.

*Falsification criterion:* If the theoretical threshold fails to predict which feature pairs exhibit absorption with AUROC < 0.65 on held-out SAE configurations, H1 is rejected.

**H2 (Cross-domain generality):** Absorption rates on entity-type hierarchies (e.g., city → country from RAVEL) and grammatical hierarchies are statistically distinguishable from shuffled-label controls (permutation test p < 0.01, Bonferroni-corrected) across at least 3 of 4 tested hierarchy types.

*Falsification criterion:* If absorption rates on ALL non-spelling hierarchies are indistinguishable from shuffled-label controls, H2 is rejected.

**H3 (Probe-free detection):** The Absorption Susceptibility Index:
```
ASI(p, c) = cos^2(theta_{p,c}) × (freq_p / freq_c)
```
computed from SAE decoder weights and activation frequencies alone achieves AUROC ≥ 0.70 against the Chanin et al. ground-truth absorption labels on the first-letter spelling task.

*Falsification criterion:* AUROC < 0.60 constitutes rejection.

**H4 (Phase transition):** Absorption rate as a function of effective sparsity (1/L0) exhibits a rapid transition identifiable as a sigmoid-like crossover. If first-order, the absorbed state persists when sparsity is subsequently reduced (hysteresis detectable within 60-minute fine-tuning budget).

*Falsification criterion:* If absorption rate increases monotonically and smoothly with 1/L0 with linear fit explaining ≥ 90% of variance, H4's phase-transition framing adds no explanatory value.

---

## Expected Contributions

1. **Formal proof that absorption is rate-distortion optimal** under flat L0/L1 sparsity penalties for hierarchically structured features. Resolves the field's conceptual confusion: absorption is not a failure of SAE training but the success of optimizing an objective that implicitly rewards hierarchical feature merging.

2. **Closed-form absorption threshold** (lambda > sin^2(theta_{p,c})) that makes quantitative predictions across SAE configurations — the first principled guide for understanding absorption severity from measurable SAE properties. Key counterintuitive prediction: co-occurrence frequency cancels, so rare and common feature pairs with the same decoder angle are equally at risk.

3. **Absorption Impossibility Theorem:** For any SAE with flat sparsity penalty, there exists a critical hierarchy depth h* = O(1/sqrt(lambda)) such that all hierarchies of depth ≥ h* will exhibit absorption regardless of SAE width or training duration.

4. **First cross-domain absorption characterization** with rigorous null controls: systematic measurement of absorption across first-letter spelling, entity-type (RAVEL), geographic (city→country→continent), and grammatical (POS) hierarchies on Gemma 2 2B using Gemma Scope SAEs. All comparisons include shuffled-label permutation baselines, bootstrap confidence intervals, and Bonferroni correction.

5. **Absorption Susceptibility Index (ASI):** A training-free, probe-free metric for identifying feature pairs at risk of absorption, validated against the Chanin et al. ground truth. This is the first probe-free absorption predictor.

6. **Phase transition characterization:** First empirical evidence for whether absorption onset is smooth (crossover) or sharp (first-order), and whether it is reversible. If hysteresis is confirmed, this changes the practical approach: architectural changes during training are necessary; post-hoc sparsity tuning cannot fix established absorption.

7. **Unified account of architectural mitigations:** The rate-distortion framework provides a single theoretical explanation for why Matryoshka SAEs (hierarchical codebook), OrtSAE (increased decoder angle between features), and ATM SAE (non-uniform sparsity) all reduce absorption through different mechanisms that each modify the absorption threshold.

---

## Method

All experiments are primarily training-free analysis on pre-trained SAEs. Primary models: Gemma 2 2B (via Gemma Scope SAEs, multiple widths and L0 settings) and GPT-2 Small (via SAELens pre-trained SAEs, as fallback and for reproducibility). Every individual experiment targets ≤ 1 hour wall-clock time.

### Phase A: Theoretical Framework and Absorption Threshold Derivation

**A.1 — Formal rate-distortion analysis (no GPU required):**
Extend the piecewise biconvex framework from arXiv:2512.05534 with an explicit rate term. The SAE loss is a Lagrangian: L = E[||x - Df(Ex + b)||^2] + lambda × E[||f(Ex + b)||_0]. For a hierarchically related pair (parent p, child c) with co-occurrence probability p_co:

- Compare two solutions: (i) both features active (L0 cost 2, distortion minimal), vs. (ii) only c active with c's decoder absorbing p's direction (L0 cost 1, distortion = ||d_p - proj_{d_c} d_p||^2 × p_co = sin^2(theta_{p,c}) × p_co).
- Absorption is preferred when: lambda × p_co > sin^2(theta_{p,c}) × p_co, i.e., **lambda > sin^2(theta_{p,c})**.
- **Key insight:** The co-occurrence frequency p_co cancels, so the absorption threshold depends only on the decoder angle and sparsity penalty. This is the central falsifiable prediction.

**A.2 — Absorption Impossibility Theorem:**
For a complete b-ary feature hierarchy of depth h where each parent-child pair has angle theta, the critical depth h* where absorption becomes unavoidable for any fixed lambda scales as h* = O(1/sqrt(lambda)). The exact bound is h* = min{k : 1 - prod_{i=1}^{k} cos^2(theta_i) > k × lambda}. All hierarchies with depth exceeding h* will exhibit absorption regardless of SAE width, training duration, or initialization.

### Phase B: Empirical Validation of the Absorption Threshold

**B.1 — Decoder geometry analysis (30 min):**
For absorbed feature pairs identified by the Chanin et al. sae-spelling metric on Gemma Scope (layer 12, widths 16k and 65k), compute the decoder cosine similarity between the absorber and absorbee. Test H1: absorbed pairs should have significantly smaller decoder angle (higher cosine similarity) than non-absorbed pairs with similar co-occurrence rates. Report Wilcoxon rank-sum test with effect size (Cohen's d).

**B.2 — Threshold AUROC validation (30 min):**
Compute AUROC for predicting absorption from the threshold classifier (cos^2(theta_{p,c}) > 1 - lambda). Use leave-one-SAE-out cross-validation across Gemma Scope L0 settings. Compare against baselines: random ranking, cosine-only ranking, frequency-only ranking.

**B.3 — Cross-architecture comparison (30 min):**
Compare absorption rates across Gemma Scope architectures (TopK, JumpReLU) at matched L0 values. The theory predicts that JumpReLU's higher effective lambda should raise absorption rates.

### Phase C: Cross-Domain Absorption Characterization (Rigorous Controls)

**C.1 — Probe training with quality gate (45 min):**
Train logistic regression probes on Gemma 2 2B residual stream activations (layer 12) for four hierarchy types:
- First-letter spelling (baseline, from sae-spelling repo)
- Entity type: city→country using RAVEL dataset (3000+ cities, already in SAEBench)
- Geographic: city→continent from RAVEL
- Grammatical: noun/verb POS → specific subtypes using Penn Treebank-tagged data

Require F1 ≥ 0.80 before computing absorption. Verify RAVEL entities appear in SAE training corpus (≥10 occurrences in 100M token sample).

**C.2 — Cross-domain absorption measurement with null controls (60 min):**
Adapt the sae-spelling absorption calculator to use entity-attribute probes. Measure absorption rate per hierarchy type, per SAE width (16k, 65k). Run:
- Shuffled-label null controls (100 permutations per hierarchy) to establish noise floor
- Bootstrap 95% CI on absorption rate (1000 resamples)
- Bonferroni correction for 8 primary comparisons (4 hierarchies × 2 widths)
- Random-SAE baseline: measure "absorption" with a randomly initialized untrained SAE

**C.3 — Hierarchy property analysis (30 min):**
For each hierarchy type, compute mean decoder cosine similarity between parent and child latents and mean co-occurrence rate (parent freq / child freq). Test H2: hierarchies with higher decoder cosine similarity should exhibit higher absorption. Report Spearman correlation across hierarchy types.

**C.4 — Control: L0 misspecification deconfounding (30 min):**
Run the "Sparse but Wrong" L0 proxy metric (Chanin & Garriga-Alonso, 2025) on all Gemma Scope SAEs used. Report absorption rates separately for SAEs with correct vs. incorrect L0 to control for feature hedging masquerading as absorption.

### Phase D: Absorption Susceptibility Index (Probe-Free Detection)

**D.1 — ASI computation (15 min):**
For all feature pairs in one SAE (Gemma Scope 16k, layer 12), compute:
```
ASI(p, c) = cos^2(theta_{p,c}) × (freq_p / freq_c)
```
Pre-filter to pairs with co-activation frequency > 0.01 to reduce the O(d_sae^2) comparison to tractable scale (~10k candidate pairs for a 16k SAE). Compare against ablated versions: (a) cosine similarity only; (b) frequency ratio only.

**D.2 — Validation against ground truth (30 min):**
Compare ASI-ranked pairs against the Chanin et al. absorption labels on the first-letter task. Compute AUROC, AUPRC, precision@k (for k=100, 500). Run permutation null (shuffle ASI scores 1000 times) to establish null AUPRC distribution.

**D.3 — Cross-domain ASI validation (30 min):**
Test whether ASI rank-correlates with cross-domain absorption rates (Spearman rho > 0.4). Compare against the feature sensitivity metric of Tian et al. (2025) as an alternative probe-free predictor.

### Phase E: Phase Transition and Dynamics

**E.1 — Sparsity sweep phase detection (45 min):**
Using the full suite of Gemma Scope SAEs for layer 12 (multiple L0 values), measure absorption rate as a function of effective sparsity. Fit sigmoid vs. linear functional forms using likelihood ratio test. Report R^2 for both models.

**E.2 — Hysteresis test (60 min):**
Starting from a high-absorption SAE (GPT-2 Small at L0=25 or similar via SAELens), fine-tune for 500 steps with reduced sparsity (L0=75 or similar). Measure: (a) absorption rate before fine-tuning; (b) absorption rate after fine-tuning; (c) absorption rate of a from-scratch SAE at the same target L0. If (b) >> (c), hysteresis is confirmed.

**E.3 — Phase diagram (30 min):**
Map absorption rate across (L0, width) space using Gemma Scope suite. Identify whether the critical L0 shifts with width as predicted by the theory (wider SAEs tolerate more sparsity before absorption dominates).

### Phase F: Unified Account of Architectural Mitigations

**F.1 — Theoretical analysis (no GPU required):**
Analyze each leading mitigation through the absorption threshold lens:
- **Matryoshka SAE:** Inner dictionaries give parent features a dedicated "slot," effectively increasing sin^2(theta) for inner levels
- **OrtSAE:** Orthogonality penalty directly increases decoder angles, pushing pairs below the absorption threshold
- **ATM SAE:** Per-latent importance scoring creates non-uniform effective lambda, reducing absorption without architecture change
- **Masked regularization:** Prevents the absorbed solution from forming during training, consistent with hysteresis

**F.2 — Empirical verification (30 min):**
For Matryoshka vs. TopK on the same model/layer (available via SAEBench), compute: (a) mean decoder cosine similarity between parent-child pairs; (b) absorption rate; (c) ASI distribution. Test whether Matryoshka's lower absorption is accompanied by larger decoder angles.

---

## Experimental Plan

| Experiment | Phase | Tool | Target | Falsification Criterion | Estimated Time |
|---|---|---|---|---|---|
| Decoder geometry analysis | B | SAELens | Gemma Scope 16k/65k | Non-significant angle difference for absorbed pairs | 30 min |
| Threshold AUROC validation | B | sae-spelling, SAELens | Gemma Scope layer 12 | AUROC < 0.65 | 30 min |
| Cross-architecture comparison | B | SAEBench | Gemma Scope TopK vs. JumpReLU | Architecture ranking contradicts theory | 30 min |
| Probe training (4 hierarchy types) | C | sae-spelling, RAVEL | Gemma 2 2B layer 12 | F1 < 0.80 for ≥ 3 hierarchies | 45 min |
| Cross-domain absorption + null controls | C | sae-spelling, permutation test | Gemma Scope 16k & 65k | Shuffled null indistinguishable from real (p > 0.05) | 60 min |
| L0 misspecification control | C | "Sparse but Wrong" metric | Gemma Scope suite | N/A (confound control) | 30 min |
| Hierarchy property correlation | C | SAELens | Gemma Scope | Spearman rho < 0.3 for cosine vs. absorption rate | 30 min |
| ASI computation and validation | D | SAELens | Gemma Scope 16k, L12 | AUROC < 0.60, permutation null not exceeded | 45 min |
| ASI cross-domain validation | D | SAELens | Gemma Scope | Cross-domain Spearman rho < 0.3 | 30 min |
| Phase transition detection | E | SAEBench | Gemma Scope suite | Linear model R^2 ≥ 0.90 (no sigmoid improvement) | 45 min |
| Hysteresis test | E | SAELens | GPT-2 Small | Fine-tuned rate drops to match scratch-trained baseline | 60 min |
| Phase diagram mapping | E | SAEBench | Gemma Scope | No systematic shift of critical L0 with width | 30 min |
| Mitigation theoretical analysis | F | Theory | N/A | Mitigations unexplainable via threshold formula | — |
| Mitigation empirical verification | F | SAEBench | Matryoshka vs. TopK | No decoder angle difference for hierarchical pairs | 30 min |

**Total estimated compute:** ~9–11 GPU-hours (single A100 or equivalent). All experiments except Phase E.2 (hysteresis test) are analysis-only on pre-trained SAEs. Phase E.2 adds ~1 hour of fine-tuning on GPT-2 Small.

**Pilot (15 min):** Load GPT-2 Small SAE (layer 6, width 24576 from SAELens). Compute ASI for all feature pairs (filter to co-activation > 0.01 → ~10k pairs). Identify top-100 pairs by ASI. Cross-reference against Neuronpedia labels to manually verify these pairs look like absorption candidates. Run the first-letter absorption rate calculation. Validates the core computational pipeline and gives early signal on ASI predictive value.

---

## Resource Estimate

- **GPU:** Single A100 (40GB). Gemma 2 2B fits in ~10GB; SAEs add ~200MB each. GPT-2 Small for pilot and hysteresis.
- **GPU-hours:** ~9–11 hours total.
- **Models:** Gemma 2 2B (pre-trained, HuggingFace), GPT-2 Small (pre-trained via TransformerLens).
- **SAEs:** Gemma Scope pre-trained SAEs (16k, 65k widths, layer 12 primary). GPT-2 Small SAEs from SAELens.
- **Datasets:** RAVEL (hij/ravel on HuggingFace), sae-spelling first-letter task data, OpenWebText subset for frequency computation, Penn Treebank POS-tagged data.
- **Code:** sae-spelling (MIT), SAELens (MIT), SAEBench (Apache 2.0), Gemma Scope (Gemma ToS).
- **Storage:** ~20GB for cached model activations and SAE weights.

---

## Risk Assessment and Mitigations

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Rate-distortion threshold has AUROC < 0.65 | High | Low-Medium | Theory remains valid as existence proof; qualitative predictions (absorbed pairs have higher cosine similarity) are still publishable; reframe as "necessary condition" rather than "sufficient prediction" |
| Entity-attribute probes fail quality gate (F1 < 0.80) | Medium | Medium | Start with 8 candidate hierarchies; require only 3 of 4 to pass; pilot screens in 15 min |
| Cross-domain absorption indistinguishable from null | Medium | Medium | Report as a positive null result: "absorption is specific to syntactic/spelling hierarchies"; this refines the scope and is equally publishable |
| ASI AUROC < 0.60 (probe-free detection fails) | High | Medium | Drop as primary contribution; retain as supplementary; focus on cross-domain + theory |
| Hysteresis experiment inconclusive | Medium | Medium | Report absence of hysteresis — consistent with continuous crossover rather than first-order transition; still characterizes phase dynamics |
| "Sparse but Wrong" confound dominates | Medium | Medium | L0 diagnostic control in Phase C.4 directly addresses this; if L0 explains most variance, that is itself an important finding |
| Multiple testing inflation | Low | Low | Pre-registered Bonferroni correction applied; bootstrap CIs reported |

---

## Novelty Assessment

Four distinct contributions are novel relative to the literature as of April 2026:

1. **Formal proof that absorption is rate-distortion optimal** with closed-form geometric threshold lambda > sin^2(theta_{p,c}). Prior art: Chanin et al. informally note "saves one L0"; Tilde Research blog mentions rate-distortion framing; arXiv:2512.05534 proves absorption solutions exist as spurious minima. None derives a closed-form threshold or proves that co-occurrence frequency cancels. **Novelty: HIGH.**

2. **First cross-domain absorption characterization** across multiple semantic hierarchy types with rigorous null controls. SAEBench explicitly acknowledges this gap. No paper has measured absorption on entity-type, geographic, or grammatical hierarchies. **Novelty: VERY HIGH.**

3. **Probe-free Absorption Susceptibility Index (ASI)** computed from decoder geometry and activation statistics alone. Prior art: Bricken et al. use decoder cosine similarity to classify latent types (different metric, different application). No probe-free absorption predictor exists. **Novelty: HIGH.**

4. **Phase transition and hysteresis characterization** of absorption onset. No prior work has framed absorption onset as a phase transition or tested whether the absorbed state is metastable. **Novelty: VERY HIGH.**

**Potential collision flags** (all resolved):
- Tilde Research blog: informal observation, not a formal theorem; we cite it as precursor
- arXiv:2512.05534: proves spurious minima exist, does not derive quantitative geometric threshold
- Bricken et al.: per-feature latent-type classifier, not pairwise probe-free absorption predictor

---

## Revisions from Prior Feedback

This synthesis integrates six perspectives (Innovator, Pragmatist, Theoretical, Contrarian, Interdisciplinary, Empiricist) into a coherent, decisive proposal. Key integration decisions:

1. **Theory framing:** The Innovator and Theoretical perspectives agreed on the rate-distortion threshold as the core formal contribution. The Theoretical perspective strengthened the proof structure (local-vs-global optimality addressed via biconvex decomposition; impossibility theorem derived with exact iterated-projection formula). The Pragmatist reinforced feasibility.

2. **Cross-domain as primary empirical contribution:** Pragmatist and Empiricist both independently selected cross-domain characterization as the highest-value empirical experiment with clearest code path. This is now the primary empirical contribution.

3. **Contrarian controls incorporated:** The Contrarian raised legitimate concerns about L0 misspecification confounding absorption measurements. Phase C.4 directly addresses this with the "Sparse but Wrong" L0 diagnostic. The Contrarian's seed-dependence concern is noted but deprioritized: using canonical Gemma Scope SAEs (single releases, not multi-seed) limits this confound without additional multi-seed training.

4. **Empiricist's statistical rigor adopted:** Shuffled-label permutation null controls, bootstrap confidence intervals, Bonferroni correction across 8 comparisons, probe quality gate (F1 ≥ 0.80), and a random-SAE baseline are all incorporated. These are non-negotiable methodological standards.

5. **Interdisciplinary insights as interpretive framing:** The immunodominance analogy (Ferretti-Kardar quasispecies dynamics) and the Lotka-Volterra coexistence condition both map onto the rate-distortion threshold when formalized. The immunodominance score (affinity ratio × residual variance ratio) reduces to ASI under simplification. These analogies are retained as interpretive framing in the discussion section, not as independent primary contributions.

6. **Phase transition dynamics:** Incorporated from Interdisciplinary and Theoretical perspectives as a secondary empirical contribution. The rigorous part (existence of a crossover) follows from Theorem 1 applied to a distribution of angles; the hysteresis prediction is a conjecture validated by the fine-tuning experiment.

---

## Backup Ideas for Pivot

See `alternatives.md` for full descriptions of backup candidates.

---

## Appendix: Key Equations

**Absorption Threshold (from rate-distortion analysis):**
Absorption of parent p into child c is energetically preferred when:
```
lambda > sin^2(theta_{p,c})
```
where lambda is the sparsity penalty (or 1/L0 for TopK SAEs) and theta_{p,c} is the angle between the decoder directions of parent and child features. The co-occurrence frequency p_co cancels from this expression.

**Absorption Susceptibility Index:**
```
ASI(p, c) = cos^2(theta_{p,c}) × (freq_p / freq_c)
```
High ASI indicates high absorption risk. Computable from SAE weights and activation frequency counts without any probe training.

**Critical Hierarchy Depth for Inevitable Absorption:**
```
h* = min{k : 1 - prod_{i=1}^{k} cos^2(theta_i) > k × lambda}
```
For uniform angles theta_i = theta: h* ~ lambda / sin^2(theta). All hierarchies with depth ≥ h* will have at least one absorbed feature pair, regardless of SAE width or training duration.

**Lotka-Volterra Coexistence Condition (from Interdisciplinary perspective):**
Parent and child features coexist (no absorption) iff:
```
alpha_{pc} × alpha_{cp} < 1
where alpha_{ij} = cos^2(d_i, d_j) × overlap(activation_patterns_i, activation_patterns_j)
```
This provides a complementary analytic perspective on the same threshold, grounded in ecological competitive exclusion theory.
