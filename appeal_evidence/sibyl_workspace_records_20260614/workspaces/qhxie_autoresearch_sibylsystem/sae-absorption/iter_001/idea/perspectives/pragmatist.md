# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **sae-spelling** (Chanin et al., 2024) -- Canonical absorption metric implementation. MIT license. Poetry-managed. Key modules: `feature_absorption_calculator`, `feature_attribution`, `feature_ablation`, `k_sparse_probing`. The absorption calculator pipeline: k-sparse probing -> false-negative identification -> integrated-gradients ablation -> cosine similarity check (>0.025) with dominance criterion (>=1.0 gap). Directly reusable for first-letter task; adapting to new hierarchies requires replacing the probing dataset, not the metric logic. **Critical implementation detail**: the metric cannot capture absorption past layer 17 in Gemma 2 2B since ablation relies on starting letter information still being at the source token position. This constrains our layer range for scaling analysis. https://github.com/lasr-spelling/sae-spelling

2. **SAELens** (Bloom, Chanin et al.) -- Standard SAE library. MIT. Supports `SAE.from_pretrained(release, sae_id)` for Gemma Scope loading. Encoder/decoder weights directly accessible as `sae.W_enc` (shape: d_model x d_sae), `sae.W_dec` (shape: d_sae x d_model), `sae.b_enc`, `sae.b_dec`. Loading is straightforward:
   ```python
   from sae_lens import SAE
   sae, cfg_dict, sparsity = SAE.from_pretrained(
       release="gemma-scope-2b-pt-res-canonical",
       sae_id="layer_12/width_16k/canonical",
       device="cuda"
   )
   ```
   EDA computation is literally 5 lines of PyTorch: load SAE, compute `F.cosine_similarity(sae.W_enc.T, sae.W_dec, dim=-1)`. No activation data needed. Gemma 2 2B has `d_model=2304`. https://github.com/decoderesearch/SAELens

3. **Gemma Scope** (DeepMind) -- 400+ pre-trained JumpReLU SAEs on Gemma 2 (2B, 9B, 27B), all layers, widths 1k-1M. HuggingFace-hosted. Loading one SAE takes ~30 seconds. Canonical release naming: `release="gemma-scope-2b-pt-res-canonical"`, `sae_id="layer_12/width_16k/canonical"`. Canonical SAEs have average L0 closest to 100. Eliminates all training cost -- this is our primary evaluation target. https://huggingface.co/google/gemma-scope

4. **SAEBench** (Karvonen et al., 2025) -- 8-metric evaluation suite with pre-computed absorption scores for 200+ SAEs. Apache 2.0. **Key scaling finding**: SCR and Feature Absorption metrics show *inverse scaling* (worse with larger dictionaries) for all previously tested non-Matryoshka architectures. Matryoshka SAEs are the sole exception, maintaining or improving absorption with scale. Pre-computed results at neuronpedia.org/sae-bench mean some data points are free. **Critical nuance from SAEBench**: dead features mask absorption -- ReLU SAEs appeared to have low absorption in original results but had high dead feature rates; improved training revealed higher absorption. Dead feature rate is a confound that must be controlled. https://github.com/adamkarvonen/SAEBench

5. **RAVEL** (Huang et al., ACL 2024) -- Entity-attribute disentanglement benchmark. Provides pre-validated probe datasets for cities (Country, Continent, Language attributes) and Nobel Prize winners (Country of Birth, Field, Gender). Already integrated into SAEBench. 5 entity types with 400-800 instances each, 4-6 attributes per entity type. **Critical caveat from SAEBench**: RAVEL disentanglement unexpectedly prefers higher L0 (>400), and attribute pairs "country-language" and "latitude-longitude" are inherently entangled even by the best methods. For absorption measurement, city-continent is the safest hierarchy. https://github.com/explanare/ravel

6. **sae-probes** (Kantamneni & Engels, ICML 2025) -- 113 binary classification datasets on Gemma-2-9B. Shows SAE probes underperform logistic regression baselines across all 4 test regimes. Important calibration: any cross-domain probe we build must beat this baseline or we are measuring probe failure, not absorption. https://github.com/sae-probes/sae-probes

7. **"Empirical Insights into Feature Geometry" (LessWrong, Jan 2025)** -- Directly computes encoder-decoder cosine similarity as a "self-similarity score." Finds potential explanations for lack of geometric relationships in semantically related SAE features could be contextual noise. Average cosine similarity hovers near random distributions. **Critical reality check for EDA**: if encoder-decoder cosine similarity has a narrow distribution, EDA might lack discriminative power.

8. **Stable and Steerable SAEs with Weight Regularization (March 2026)** -- Shows encoder-decoder alignment across SAEs with tied initialization and unit-norm decoders can be substantially improved by weight regularization. L2 regularization yields highest shared-feature fraction. Shared features appear qualitatively more interpretable. **Relevant**: if weight regularization improves encoder-decoder alignment, it should also reduce EDA and potentially reduce absorption. This is a testable prediction.

9. **Matryoshka SAE** (Bussmann et al., ICML 2025) -- Best SAEBench results on absorption. At L0=40, absorption rate 0.05 vs. 0.49 for BatchTopK. MIT. Code available. **Key trade-off**: inner levels suffer from feature hedging (Chanin & Dulka, 2025). arXiv:2503.17547

10. **OrtSAE** (Korznikov et al., 2025) -- Orthogonality penalty reduces absorption 65%. Addresses inter-latent cosine similarity (between decoder columns), NOT encoder-decoder alignment within a single latent. Important distinction from EDA. arXiv:2509.22033

11. **Feature Hedging** (Chanin & Dulka, 2025) -- Complementary failure to absorption. In absorption, parent gets suppressed when child fires. Feature absorption is an effective L0 reduction strategy: absorbing parent saves +1 L0 per parent-child pair. **Critical insight**: any solution removing absorption will likely have worse variance explained against L0. This is a fundamental constraint, not a bug. arXiv:2505.11756

12. **DeepMind Negative Results (March 2025)** -- SAEs underperformed simple linear probes on safety-relevant downstream tasks. DeepMind deprioritized fundamental SAE research. Feature absorption is cited as a key culprit for why 1-sparse SAE probes fail catastrophically at detecting harmful intent. This provides strong motivation for our work: quantifying the absorption-safety link.

### Landscape Summary

**What actually works in practice**: The `sae-spelling + SAELens + Gemma Scope` stack gives a complete, battle-tested pipeline. The first-letter task has known ground truth. SAEBench has pre-computed scores for 200+ SAEs. RAVEL provides pre-validated entity-attribute probing datasets. All critical libraries are MIT/Apache licensed with active maintenance by the same group of researchers (Chanin, Bloom, Karvonen are co-authors across multiple papers).

**Where the engineering risks hide**:

1. **EDA noise floor problem**: The LessWrong feature geometry post and the Stable SAEs paper both suggest encoder-decoder cosine similarity varies substantially across SAE training configurations. In JumpReLU SAEs (which Gemma Scope uses), the encoder and decoder are more loosely coupled than in standard L1 SAEs because JumpReLU applies a hard threshold. This could mean ALL latents have moderate EDA, making absorbed latents hard to distinguish. The AUROC >= 0.75 target is optimistic without first characterizing the baseline EDA distribution.

2. **Cross-domain probe quality**: Kantamneni & Engels (ICML 2025) show that SAE probes do not consistently beat logistic regression on the raw residual stream. If the underlying features are not cleanly linearly separable, our absorption metric becomes meaningless (we measure "probe failure" not "absorption"). The RAVEL dataset for cities is the safest bet because it has pre-validated linear separability. But RAVEL was validated on Llama2-7B, not Gemma 2 2B -- probes MUST be re-validated.

3. **Absorption metric layer constraint**: The sae-spelling metric cannot operate past layer 17 in Gemma 2 2B because attention has already moved starting letter information from the source token to the final token position. This constrains our scaling analysis to layers 0-17. SAEBench's extended implementation may partially address this but needs verification.

4. **Dead feature confound**: SAEBench discovered that dead features mask absorption measurements. Any absorption comparison across architectures or widths must control for dead feature rate. Gemma Scope canonical SAEs use auxiliary loss to minimize dead features, but the rate still varies by config.

5. **The "it's just cosine similarity" risk**: EDA = 1 - cos(encoder_row, decoder_column) is trivially computable. Reviewers may perceive this as a thin contribution unless the empirical validation is strong and the combined detector (D-EDA with residual decomposition) adds genuine discriminative value.

---

## Phase 2: Initial Candidates

### Candidate A: Scaling Analysis + Partial Correlation (The "Do the Boring Thing Right" Approach)

- **Hypothesis**: Absorption rates follow a predictable scaling relationship with SAE width and L0, but the relationship is confounded: after controlling for L0, wider SAEs show *lower* (not higher) absorption rates, reversing the marginal positive correlation observed by Chanin et al.

- **Implementation sketch**: (1) Load all Gemma Scope 2B residual stream canonical SAEs via SAELens (~20-30 configs). (2) Run sae-spelling `feature_absorption_calculator` on each for the first-letter task. (3) Cross-validate against SAEBench pre-computed scores where available. (4) Record: absorption_rate, false_negative_rate, L0, width, layer, num_dead_features, average_feature_density. (5) Fit parametric models in log-space. Compute partial Spearman correlation of absorption with width controlling for L0.

- **Simplest version**: 10 SAEs (2 widths x 5 layers), compute absorption via sae-spelling, fit partial correlation. 3 GPU-hours.

- **Time estimate**: Loading + absorption metric on 20 SAEs: ~5 GPU-hours. Analysis: 30 min CPU. Total: ~5.5 GPU-hours.

- **Reusable components**: SAELens (direct), sae-spelling (direct), SAEBench pre-computed scores (free data points), Gemma Scope (no training).

### Candidate B: Cross-Domain Absorption via RAVEL Entity-Attribute Probes

- **Hypothesis**: Feature absorption occurs in entity-attribute hierarchies (city-country, city-continent) at rates measurable by the adapted Chanin et al. metric, and the cross-domain absorption severity is predicted by hierarchy frequency imbalance (parent/child co-occurrence ratio).

- **Implementation sketch**: (1) Use RAVEL pre-validated city-attribute dataset (3000+ cities with Country, Continent, Language). (2) Train LR probes for parent (continent) and child (specific city) features on Gemma 2 2B residual stream at target layers. (3) Adapt sae-spelling absorption metric: replace first-letter probing dataset with RAVEL entity data; keep integrated-gradients ablation unchanged. (4) Measure absorption on Gemma Scope 16k and 65k SAEs at layers 6, 12, 16 (6 SAEs). (5) Compare to first-letter baseline on same SAEs. (6) Compute cross-domain Spearman correlation.

- **Simplest version**: RAVEL city-continent hierarchy only. 1 probe. 1 SAE. Compare to first-letter on same SAE. ~30 min.

- **Time estimate**: Probe training: ~15 min per hierarchy. Absorption metric per SAE per hierarchy: ~15 min. For 3 hierarchies x 6 SAEs: ~5 GPU-hours. Total: ~6 GPU-hours.

- **Reusable components**: RAVEL dataset (pre-validated, no dataset construction needed), sae-spelling metric (adapt probing dataset only), SAELens, Gemma Scope.

### Candidate C: EDA Weight-Based Detector + Two-Subtype Taxonomy

- **Hypothesis**: Encoder-decoder misalignment (EDA) is elevated for absorbed latents and achieves AUROC >= 0.65 against ground-truth absorption labels. Absorbed latents partition into mechanistically distinct subtypes: late absorption (parent decoder direction exists but encoder is suppressed, high EDA) and early absorption (parent direction never learned, low EDA).

- **Implementation sketch**: (1) For all SAEs from Candidate A, compute EDA(j) = 1 - cos(W_enc[:,j], W_dec[j,:]) for all latents j. (2) Using absorption labels from sae-spelling, compute AUROC of EDA vs. ground-truth absorption. (3) If AUROC >= 0.65: proceed with D-EDA (directional decomposition of encoder residual). (4) Classify absorbed latents into early vs. late by checking max cosine similarity between parent probe direction and all decoder columns. (5) Compare EDA distributions between subtypes.

- **Simplest version**: 1 SAE (layer 12, 16k). Compute EDA distribution. Run sae-spelling for labels. Mann-Whitney U test. AUROC. 15 minutes total.

- **Time estimate**: EDA computation across all Gemma Scope 2B SAEs: ~30 min (CPU only). Taxonomy analysis: ~1 hour (piggybacks on Candidate A data). Total: ~1.5 hours.

- **Reusable components**: Entirely derived from Candidate A outputs plus cosine similarity computation.

---

## Phase 3: Self-Critique

### Against Candidate A: Scaling Analysis + Partial Correlation

- **Implementation reality check**: EDA computation is trivially implementable. The sae-spelling codebase is working and maintained. SAEBench has pre-computed data points. The engineering risk is near zero. I confirmed via SAELens documentation that `sae.W_enc` and `sae.W_dec` are directly accessible PyTorch tensors. The absorption metric from sae-spelling is the community standard.

- **Reproducibility attack**: The absorption metric has well-documented hyperparameters (k-sparse probing with k=15, integrated-gradients with 100 steps, cosine threshold 0.025, magnitude gap 1.0). Using canonical Gemma Scope SAEs pins the SAE choice. The main uncertainty is whether these thresholds are robust across SAE widths -- the threshold sensitivity sweep in Phase 0 addresses this. Highly reproducible.

- **Baseline sanity check**: The null hypothesis is that absorption rate is random noise uncorrelated with SAE configuration (R^2 < 0.1). The existing observation from Chanin et al. that wider/sparser SAEs have more absorption already refutes this at the marginal level. The novel contribution is the partial correlation analysis. If partial rho(width, absorption | L0) reverses sign from the marginal correlation, this is a concrete actionable finding that no prior paper has reported.

- **Scope attack**: Scaling analysis alone without EDA is a solid but incremental contribution. However, the partial correlation disentangling width from L0 is genuinely novel. SAEBench reports inverse scaling but does not decompose the confound. The contribution becomes strong if combined with Candidate B (cross-domain generalization).

- **Verdict**: **STRONG**. Near-zero implementation risk. Guaranteed to produce publishable data. The partial correlation analysis is the key novelty.

### Against Candidate B: Cross-Domain Absorption via RAVEL

- **Implementation reality check**: RAVEL's pre-validated dataset dramatically reduces engineering overhead compared to constructing probing datasets from scratch. The main adaptation is reformatting RAVEL prompts to match sae-spelling's template format. The integrated-gradients ablation code is hierarchy-agnostic. Feasibility is HIGH.

- **Reproducibility attack**: The biggest risk is probe quality on Gemma 2 2B. RAVEL was validated on Llama2-7B. City-continent probes are likely robust (geographic knowledge is well-represented in 2B+ models), but city-language probes are empirically harder (SAEBench found country-language inherently entangled). We must require >= 85% probe accuracy and exclude failing hierarchies. Risk: LOW-MEDIUM.

- **Baseline sanity check**: The null hypothesis is "absorption rates are identical across all feature hierarchies." If rates are within 5 percentage points of first-letter, the contribution is "confirming universality" (useful). If rates differ significantly, the contribution is "identifying what drives severity" (more exciting). Either outcome is publishable.

- **Scope attack**: RAVEL provides 3 clean hierarchies for cities (continent, country, language). Combined with first-letter baseline, that is 4 hierarchies -- borderline for "systematic cross-domain characterization" but defensible given RAVEL is the only pre-validated entity-attribute dataset.

- **Verdict**: **MODERATE-STRONG**. High practical value. Using RAVEL makes it feasible within the time budget. Strengthens significantly when combined with Candidate A.

### Against Candidate C: EDA Weight-Based Detector + Taxonomy

- **Implementation reality check**: EDA computation is computationally trivial (one cosine similarity operation per latent). D-EDA decomposition (projecting encoder residual onto other decoder columns) adds modest complexity. The main risk is that EDA lacks discriminative power -- the signal-to-noise ratio may be poor if all JumpReLU SAE latents have moderate encoder-decoder misalignment.

- **Reproducibility attack**: EDA is deterministic (computed from weights alone). The two-subtype taxonomy has a sample size problem: Chanin et al. report absorption in 15-35% of 26 letters tested, giving 4-9 absorbed letters per SAE. With 6 SAEs total, that is 24-54 absorbed latents. Splitting into subtypes gives ~12-27 per group -- marginal for robust statistical conclusions.

- **Baseline sanity check**: Random EDA scores should give AUROC ~0.50. If EDA achieves 0.60-0.65, it is better than random but not impressive. The D-EDA directional decomposition is where the real signal might be: if the encoder residual aligns with specific decoder columns that have high cosine similarity with the parent probe direction, that is strong absorption evidence. But this requires the same probe directions that EDA was supposed to replace.

- **Scope attack**: As a standalone, EDA+taxonomy is thin (especially if AUROC is mediocre). As a component of a larger paper with Candidates A+B, it adds depth and mechanistic insight. The key question is whether it crosses the threshold from "we computed cosine similarity" to "we discovered a diagnostic tool."

- **Verdict**: **MODERATE**. Conceptually interesting but pilot-dependent. The 15-min pilot experiment directly tests viability before committing compute. If EDA AUROC < 0.60, pivot to scaling+cross-domain as primary contributions.

---

## Phase 4: Refinement

### Architecture Decision: Three-Tier Contribution Structure

Based on self-critique, I propose restructuring the paper with three tiers of contribution ordered by implementation confidence:

**Tier 1 (Core, guaranteed): Scaling Analysis + Partial Correlation**
- Measure absorption across 20-30 Gemma Scope SAEs on first-letter task
- Fit scaling model: absorption_rate ~ width + L0 + layer + dead_feature_rate
- Partial correlation analysis disentangling width from L0
- Use SAEBench pre-computed scores where available
- This is virtually guaranteed to produce publishable results

**Tier 2 (High-confidence): Cross-Domain via RAVEL**
- Measure absorption on RAVEL city-attribute hierarchies (Continent, Country, Language)
- Compare absorption rates to first-letter baseline on same SAEs
- Test cross-domain correlation
- Test whether hierarchy frequency imbalance predicts absorption severity
- RAVEL pre-validated probes minimize engineering risk

**Tier 3 (Exploratory, pilot-dependent): EDA Metric + Taxonomy**
- Compute EDA for all latents across all SAEs
- Validate against ground-truth absorption labels from Tier 1
- If AUROC >= 0.65: promote to primary contribution with D-EDA enhancement
- If AUROC < 0.65: report as negative result or supplementary material
- Two-subtype taxonomy included as secondary analysis with explicit sample-size caveats

### Key Engineering Refinements

**Refinement 1: Scope RAVEL to proven hierarchies only.**
Drop any hierarchy where RAVEL reports inherent entanglement (country-language). Focus on: city-continent (safest), city-country (medium), first-letter (baseline). Three hierarchies + baseline = 4, which is sufficient.

**Refinement 2: Control for dead feature rate.**
SAEBench discovered that dead features mask absorption. Every absorption measurement must report dead_feature_rate alongside absorption_rate. Include dead_feature_rate as a covariate in the scaling model.

**Refinement 3: Threshold sensitivity sweep as Phase 0.**
Before all experiments, sweep cosine threshold {0.005, 0.01, 0.025, 0.05, 0.10} and magnitude gap {0.5, 1.0, 1.5, 2.0}. If absorption rate varies >30% across reasonable thresholds, this is a methodological finding worth reporting.

**Refinement 4: De-risk EDA via 15-min pilot.**
Run EDA pilot on 1 SAE before committing to the full EDA story. Decision rule:
- AUROC >= 0.70: EDA is promising, proceed with D-EDA and full validation
- AUROC 0.60-0.70: EDA works but is noisy; combine with D-EDA and report with caveats
- AUROC < 0.60: EDA alone is insufficient; pivot to Tier 1+2 as primary, EDA as negative result

**Refinement 5: Honest compute budget.**
Breaking down carefully:
- Loading Gemma 2 2B + SAEs: ~5 min per config x 20 configs = ~1.5 hours (with caching)
- Absorption metric (integrated-gradients ablation): ~15 min per SAE per hierarchy
- For 1 hierarchy x 20 SAEs = 20 runs x 15 min = 5 hours
- For 3 RAVEL hierarchies x 6 SAEs = 18 runs x 15 min = 4.5 hours
- EDA computation: trivial (<30 min total, CPU)
- Probe training: ~30 min per hierarchy x 3 = 1.5 hours
- Phase 0 threshold sweep: ~1 hour

Total realistic estimate: **~14 GPU-hours**. With 2 GPUs: ~7 hours wall-clock. Individual experiments fit the 1-hour budget.

### Minimal Pilot Experiment (< 15 min)

1. Load Gemma Scope Gemma 2 2B, layer 12, 16k width via SAELens
2. Compute EDA = 1 - cos(W_enc.T, W_dec) for all 16k latents
3. Plot EDA distribution -- is it bimodal or unimodal? What is the variance?
4. Run sae-spelling absorption metric on first-letter task (~10 min)
5. Compare EDA of absorbed vs. non-absorbed latents: Mann-Whitney U, AUROC

**Decision rule**:
- AUROC >= 0.70: EDA is promising, proceed with full validation
- AUROC 0.60-0.70: EDA works but needs combined detector; proceed cautiously
- AUROC < 0.60: EDA alone is insufficient; pivot to scaling + cross-domain as primary

---

## Phase 5: Final Proposal

### Title

Anatomy of Feature Absorption in Sparse Autoencoders: Scaling Laws, Cross-Domain Generalization, and Weight-Based Detection

### Hypothesis

Feature absorption in SAEs is (1) predictable from SAE hyperparameters via scaling relationships with a confounded width-L0 interaction, (2) generalizable beyond the first-letter spelling task to entity-attribute feature hierarchies, and (3) partially detectable from SAE weights alone via encoder-decoder misalignment.

**Falsifiable predictions**:
1. log(absorption_rate) is log-linear in width and L0 (R^2 >= 0.50 in partial correlation model including dead_feature_rate as covariate)
2. Partial Spearman rho(width, absorption | L0, dead_feature_rate) < 0, reversing the marginal positive correlation
3. Absorption occurs in RAVEL city-continent hierarchy at rates measurable by the adapted metric, with cross-domain Spearman rho >= 0.35 with first-letter rates on the same SAEs
4. EDA achieves AUROC >= 0.65 against Chanin et al. supervised absorption labels (pilot-dependent; if fails, demoted to supplementary material)
5. The absorption metric's canonical thresholds (cosine > 0.025, gap >= 1.0) are robust: absorption rate varies < 30% across reasonable threshold sweeps

### Motivation

Feature absorption undermines the reliability of SAE-based mechanistic interpretability. DeepMind publicly deprioritized SAE research in 2025 after finding that 1-sparse SAE probes fail catastrophically at detecting harmful intent while dense probes succeed. Three practical questions remain unanswered:

1. **How do I choose SAE hyperparameters to minimize absorption?** No quantitative model predicts absorption severity as a function of width, L0, and layer. SAEBench reports inverse scaling but does not decompose the width-L0 confound. Practitioners make SAE configuration choices blind to absorption consequences.

2. **Is absorption a spelling artifact?** Every published measurement uses the first-letter task. If absorption is specific to this narrow hierarchy, the field's focus on absorption mitigation may be misdirected.

3. **Can I detect absorption without knowing what to look for?** The supervised metric requires pre-specified probe directions, creating a chicken-and-egg problem. An unsupervised detector would enable large-scale absorption auditing.

### Method

**Phase 0: Metric Validation (1 hour, 1 GPU -- prerequisite)**

Before building on the Chanin et al. metric, validate it is reliable:
1. Threshold sensitivity sweep: cosine {0.005, 0.01, 0.025, 0.05, 0.10}, magnitude gap {0.5, 1.0, 1.5, 2.0} on 2 SAEs (layer 12, width 16k and 65k)
2. Random direction baseline: 100 random unit vectors as "probe directions"; measure absorption rate (expect < 5%)
3. Dead feature analysis: report and control for dead feature rate across all SAE configs
4. **Pass criterion**: absorption rate varies < 30% across reasonable threshold range AND random baseline < 5%
5. **If fails**: pivot to reporting metric limitations as primary contribution

**Phase 1: Comprehensive Scaling Characterization (5 GPU-hours)**

- Load Gemma Scope residual stream SAEs for Gemma 2 2B: layers {3, 6, 9, 12, 15}, widths {16k, 65k}, multiple L0 variants per config. ~20-30 SAEs total. Layer range capped at 17 due to metric constraint.
- Run sae-spelling `feature_absorption_calculator` on each SAE for the canonical first-letter task.
- Cross-validate against SAEBench pre-computed absorption scores where available.
- Record for each SAE: absorption_rate, false_negative_rate, L0, width, layer, num_dead_features, average_feature_density.
- Fit parametric models in log-space. Report R^2, AIC/BIC for model selection.
- Key analysis: partial Spearman correlation of absorption_rate with width, controlling for L0 and dead_feature_rate. Test sign reversal hypothesis.

**Phase 2: Cross-Domain Generalization via RAVEL (5 GPU-hours)**

- Use RAVEL pre-validated entity-attribute dataset for cities.
- Define 3 feature hierarchies with clear parent-child structure:
  - **City-Continent**: Parent = continent membership (Europe/Asia/etc.), Child = specific city
  - **City-Country**: Parent = country membership, Child = specific city
  - **City-Language**: Parent = primary language, Child = specific city (only if probe accuracy >= 85% on Gemma 2 2B; exclude otherwise)
- Train LR probes on Gemma 2 2B residual stream at layer 12. Require >= 85% probe accuracy or exclude the hierarchy.
- Adapt sae-spelling absorption metric: replace first-letter probing dataset with RAVEL city data; keep integrated-gradients ablation unchanged.
- Measure absorption on Gemma Scope 16k and 65k SAEs at layers 6, 12, 16 (6 SAEs total).
- Compare absorption rates to first-letter on same SAEs. Test cross-domain Spearman correlation.
- Controls: random probe direction control (expect < 5%), shuffled hierarchy control.

**Phase 3: EDA Metric Analysis (1 hour, CPU + pilot GPU)**

- Pilot: 15 min on 1 SAE to test EDA viability (see pilot experiment above).
- If pilot passes: for each SAE from Phase 1, compute EDA(j) = 1 - cos(w_{e,j}, d_j) for all latents j.
- Plot EDA distributions stratified by absorption status from Phase 1 ground truth.
- Compute AUROC, precision@k, Spearman rho.
- If AUROC >= 0.65: proceed with D-EDA enhancement.
  - D-EDA: decompose encoder residual r_j = w_{e,j} - alpha_j * d_j, project onto decoder matrix, check if top-K projections align with known absorbing latents.
  - Two-subtype taxonomy: classify absorbed latents into early (max decoder cosine < 0.3 with parent probe) vs. late (>= 0.3). Report as exploratory with sample-size caveats.
- If AUROC < 0.65: report as negative result. Characterize EDA distribution as contribution to understanding SAE weight geometry.

**Phase 4: Practical Guidance**

- Translate scaling law results into actionable hyperparameter recommendations.
- If partial correlation confirms L0 confound: recommend practitioners target specific L0 ranges rather than simply using wider SAEs.
- Release adapted absorption metric code for RAVEL hierarchies as SAEBench extension.

### Simplest Version

Run sae-spelling absorption metric on 10 Gemma Scope SAEs (2 widths x 5 layers), compute partial correlation of absorption with width controlling for L0, and compute EDA on the same 10 SAEs. If partial correlation shows a meaningful relationship and EDA AUROC >= 0.65, this is a workshop-quality paper in 1 day of GPU time.

### Baselines

1. **Null model**: Absorption rate is random noise uncorrelated with SAE configuration. R^2 < 0.1 for scaling model.
2. **Width-only model**: absorption_rate ~ width. No L0 control. This is the implicit model from Chanin et al. and SAEBench's observation that wider SAEs have more absorption.
3. **Random EDA baseline**: Replace EDA with random scores from the same distribution. AUROC should be ~0.50.
4. **Decoder cosine similarity baseline**: Use max pairwise decoder cosine similarity as absorption predictor instead of EDA. OrtSAE uses this signal; it should be weaker than EDA for individual latent diagnosis.
5. **Random probe direction baseline**: Apply absorption metric with random unit vector probes. Absorption rate should be < 5%.

### Experimental Plan

| Experiment | Scope | GPU-hours | Wall-clock | Validates |
|-----------|-------|-----------|------------|-----------|
| **Phase 0**: Threshold sweep + baselines | 2 SAEs, 20 threshold combos | 1 | 1 hour | Metric robustness |
| **Pilot**: EDA + absorption on 1 SAE | Layer 12, 16k | 0.3 | 15 min | EDA viability |
| **Phase 1**: Scaling on 20 SAEs | 5 layers x 2 widths + L0 variants | 5 | 3 hours | Scaling laws |
| **Phase 2a**: RAVEL probes train + validate | 3 hierarchies | 1 | 1 hour | Probe quality |
| **Phase 2b**: Cross-domain absorption | 6 SAEs x 3 hierarchies | 4.5 | 3 hours | Generalization |
| **Phase 3**: EDA full validation | 20 SAEs (CPU) | 0.5 | 30 min | EDA metric |
| **Analysis + visualization** | CPU only | 0 | 2 hours | All |
| **Total** | | **~12.5** | **~10.5 hours** | |

Note: Phases 1 and 2b parallelize across 2 GPUs to ~5.5 hours wall-clock. Every individual experiment fits within the 1-hour budget.

### Resource Estimate

- **GPU**: Single A100/H100, Gemma 2 2B (~5GB VRAM). Total ~12.5 GPU-hours.
- **Models**: Gemma 2 2B (5GB), Gemma Scope SAEs (~200MB each, ~20 SAEs = 4GB)
- **Data**: RAVEL cities dataset (HuggingFace: `hij/ravel`), first-letter spelling task (constructed in sae-spelling)
- **Storage**: ~40GB for SAE weights + cached activations
- **Software**: SAELens (pip), sae-spelling (poetry), RAVEL, TransformerLens, standard scientific Python
- **Wall-clock**: ~10.5 hours sequential, ~5.5 hours with 2 GPUs

### Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| EDA AUROC < 0.65 (metric does not work) | Medium | **Medium-High** | Demote to supplementary; scaling + cross-domain carry the paper |
| EDA distribution is unimodal/narrow | Medium | Medium | Pilot tests this in 15 min; pivot before committing GPU-hours |
| RAVEL probes < 85% accuracy on Gemma 2 2B | High | Low | Gemma 2 2B represents geographic knowledge well. Fallback: use city-continent only |
| No clean scaling law (R^2 < 0.3) | Medium | Low | Partial correlation analysis is valuable even without clean functional form |
| Dead feature rate confounds absorption | Medium | **Medium** | Include dead_feature_rate as covariate; SAEBench already flagged this |
| Absorption metric threshold sensitivity | Medium | Medium | Phase 0 sweep addresses this directly; report as finding either way |
| Two-subtype taxonomy lacks statistical power | Low | **High** | Already scoped as exploratory with explicit sample-size caveat |
| Layer 17 cutoff limits scaling analysis | Low | Certain | Use SAEBench extended implementation for higher layers if available |
| sae-spelling dependency version conflict | Low | Low | Maintained by overlapping author group; pin versions; test in pilot |

### Novelty Claim

1. **First systematic scaling analysis of feature absorption with confound control** (Tier 1): No prior work fits a quantitative scaling model to absorption rates with partial correlation analysis disentangling width from L0. SAEBench reports inverse scaling but does not decompose the confound. The dead feature rate control is also new.

2. **First cross-domain absorption characterization** (Tier 2): Every published absorption measurement uses only the first-letter spelling task. Extending to RAVEL entity-attribute hierarchies tests the generalizability assumption that underlies all absorption mitigation research.

3. **First unsupervised absorption detection attempt** (Tier 3, pilot-dependent): If EDA works, it enables large-scale absorption auditing without known probe directions. If it does not, the negative result is still informative and directs future work toward activation-based methods.

4. **Metric robustness analysis** (Phase 0): The threshold sensitivity sweep establishes whether the canonical metric's ad-hoc thresholds are reliable. This methodological contribution is useful to the community regardless of other results.

5. **Actionable practitioner guidance**: Scaling laws and partial correlations translate directly to hyperparameter recommendations. This is the kind of contribution that gets cited in practice.

### Key Dependencies and Confirmed Availability

| Dependency | Status | License | Confirmed |
|-----------|--------|---------|-----------|
| SAELens | Active, pip-installable | MIT | Yes |
| sae-spelling | Public, Poetry-managed | MIT | Yes |
| Gemma Scope SAEs | HuggingFace, free download | Gemma ToS | Yes |
| SAEBench pre-computed scores | Public, Neuronpedia | Apache 2.0 | Yes |
| RAVEL dataset | Public, HuggingFace `hij/ravel` | MIT | Yes |
| Gemma 2 2B | HuggingFace | Gemma ToS | Yes |
| TransformerLens | Active | MIT | Yes |

### Critical Differences from Prior Pragmatist Draft

1. **Dead feature confound explicitly addressed**: Added dead_feature_rate as a covariate in the scaling model and noted SAEBench's discovery of this confound.

2. **Layer constraint acknowledged**: sae-spelling metric cannot operate past layer 17; this constrains the scaling analysis and is now explicitly noted.

3. **RAVEL hierarchy filtering**: Dropped city-language from default plan since SAEBench found country-language inherently entangled. Include only if probe accuracy >= 85%.

4. **Phase 0 metric validation added**: Threshold sensitivity sweep and random baseline are prerequisites before any absorption measurement. This was missing from the prior draft.

5. **Compute budget updated**: 12.5 GPU-hours, not 5.5 or 9.5. Individual experiments still fit the 1-hour budget. Honest accounting.

6. **EDA remains pilot-dependent**: The 15-min pilot with explicit AUROC decision gates is preserved as the critical de-risking step. No commitment to EDA as a primary contribution until the pilot passes.

7. **Three-tier structure formalized**: Tier 1 (scaling, guaranteed) > Tier 2 (cross-domain, high-confidence) > Tier 3 (EDA, pilot-dependent). This ensures the paper has a strong core even if the most novel contribution (EDA) underperforms.
