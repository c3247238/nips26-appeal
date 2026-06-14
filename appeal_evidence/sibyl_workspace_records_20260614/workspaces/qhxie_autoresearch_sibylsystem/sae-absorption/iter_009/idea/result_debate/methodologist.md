# Methodologist Analysis: The Absorption Tax

**Agent**: Methodologist
**Iteration**: 9 (FULL mode)
**Date**: 2026-04-16

---

## 1. Baseline Fairness Audit

### 1.1 Random Direction Baseline
The methodology includes a random direction baseline for absorption measurement, which is the correct control. The consolidation states "Random baseline near-zero," confirming that the absorption measurement pipeline does not produce spurious positives from noise. **PASS.**

### 1.2 Shuffled Hierarchy Control
Planned in the methodology but not explicitly reported in the consolidation summary. The task_plan specifies "shuffled hierarchy control" and "probe-only baseline," but the results files do not show dedicated results for these controls. **FLAG: Shuffled hierarchy control results missing or not separately reported.** This matters because it would distinguish absorption driven by genuine hierarchy structure from absorption driven by probe artifacts.

### 1.3 Cross-Domain Probe Asymmetry (CRITICAL)
The most significant baseline fairness issue across this entire paper:

- **First-letter probes**: F1 = 1.0 at position -6 (sae_spelling-style), F1 = 0.96 at position -2 (sklearn). Perfect or near-perfect.
- **City-continent probes**: F1 = 0.87 (weighted, L24). Passes relaxed gate.
- **City-language probes**: F1 = 0.82 (weighted, L24). Passes relaxed gate.
- **City-country probes**: F1 = 0.73 (weighted, L24). **Below both gates.** Balanced accuracy = 0.56 (80 classes).

The probes used for absorption measurement are **dramatically asymmetric** across hierarchies. Absorption rate is defined as the fraction of probe-correct instances where the SAE reconstruction leads to a false negative. A lower-quality probe has fewer "probe correct" instances in the denominator, and the instances it does get correct may be biased toward "easy" examples where the model's representation is cleanest.

**City-country's 45.1% absorption rate is measured with a probe that has F1=0.73 and balanced accuracy of 0.56.** This means the probe only reliably identifies about half the countries, and the "probe correct" subset is heavily biased toward well-represented countries (United States: 176/176, India: 47/49, China: 96/100). The claimed "highest absorption" may partly reflect the probe's failure to capture the full distribution. Evidence: United States has 0% absorption (176 probe correct, 0 FN) while Albania has 100% (6 probe correct, 6 FN) -- the probe only gets the "easy" countries right, and absorption on these unequal slices may not be comparable.

**Severity**: HIGH. This confound does not invalidate the primary finding (cross-domain variation exists) but it significantly weakens the quantitative ranking claim. The paper should not claim city-country > first-letter absorption without extensive discussion of this asymmetry.

### 1.4 First-Letter Token Position Asymmetry
First-letter absorption uses token position -6 with F1=1.0 probes (sae_spelling pipeline). Cross-domain absorption uses token position -2 with imperfect probes. These are **different experimental setups measuring different things at different token positions.** The consolidation notes "Different measurement methodology (sae_spelling vs full pipeline)" for the first-letter iter_008 vs iter_009 discrepancy (34.5% vs 27.1%), confirming that methodology matters.

**Severity**: MEDIUM. The paper compares absorption rates across hierarchies measured with fundamentally different probe pipelines and token positions. Permutation tests comparing cross-domain rates to first-letter rates (consolidation Table: permutation p-values) treat these as comparable measurements when they are not.

### 1.5 Architecture Comparison Fairness
Architecture comparison tests JumpReLU 16k/65k, BatchTopK 16k, and Matryoshka 32k. These have different dictionary sizes (16k vs 32k vs 65k) and potentially different L0 values. The methodology acknowledges the need for "matched L0" comparison, but the results do not report whether L0 was matched. At L12, 4 architectures are tested; at L24, only 3 (BatchTopK absent). This is a fair limitation acknowledged in the results.

**Severity**: LOW. The null result (no architecture effect, p=0.75) is robust to this concern -- if there were a real architecture effect, unmatched L0 would make it harder to detect, not create false positives.

---

## 2. Metric-Claim Alignment

### 2.1 Claim: "Cross-domain absorption variation" -- Metric: Absorption rate
The absorption rate is defined as (number of false negatives where SAE reconstruction causes probe misclassification) / (number of probe-correct instances). This directly measures what is claimed. **ALIGNED**, with the caveat that probe quality variation across hierarchies introduces measurement noise (see Section 1.3).

### 2.2 Claim: "Absorption is always pathological" -- Metric: Logit change from parent direction ablation
H8 tests whether ablating the parent direction component from the child latent's decoder causes logit changes. Mean |logit change| = 3.98 nats vs control 0.004 nats (1000x ratio). This metric measures whether the child feature's decoder carries parent-relevant information that affects model output.

**Potential gap**: The metric measures whether removing parent direction from child decoder changes logits -- but this is expected by construction if the child feature absorbed parent-relevant information into its decoder. A high logit change confirms that the decoder encodes parent information, but **does not necessarily mean the model was "using" this information for downstream computation** in the original (unmodified) forward pass. The logit change measures decoder perturbation impact, not computational dependency.

However, the 1000x ratio and 0% benign rate across three thresholds (0.05, 0.1, 0.2) with n=1471 instances is overwhelming. Even if the metric slightly overcounts pathological instances, the conclusion (most absorption is not benign) is likely robust. **Partially aligned -- metric may overstate pathological severity but direction is credible.**

### 2.3 Claim: "Causal evidence for competitive exclusion" -- Metric: Probe recovery after child zeroing
Activation patching zeroes child features and measures parent probe recovery. For city-continent: primary recovery 62%, all-absorbers recovery 80%, control 5.2%. Wilcoxon p < 1e-20, Cohen's d = 1.50. This is strong causal evidence that child features causally suppress parent probe predictions. **ALIGNED.**

### 2.4 Claim: "Hedging decomposition varies by hierarchy" -- Metric: Strict vs compensatory classification
Chi-square(3) = 91.51, p = 1.04e-19. However, the classification scheme is defined relative to L0 changes: "strict" means the parent feature fires at higher L0, "compensatory" means any feature resolves the FN at higher L0. This is somewhat circular -- if the SAE is already under-representing features at the operating L0, increasing L0 mechanically resolves FNs. **Partially aligned -- the variation across hierarchies is real, but the absolute strict/compensatory percentages may not have a clean causal interpretation.**

### 2.5 Claims vs metrics summary table

| Claim | Metric | Alignment | Gap |
|-------|--------|-----------|-----|
| Cross-domain variation | Absorption rate across hierarchies | Aligned | Probe quality confound |
| First-letter is atypical | Permutation tests vs FL | Weak | Different pipelines/positions |
| All absorption is pathological | Logit change from decoder ablation | Partial | Measures decoder content, not computational use |
| Causal competitive exclusion | Probe recovery after zeroing | Strong | Cross-domain patching fixes prior bug |
| Hedging varies by hierarchy | Chi-square on strict/compensatory | Aligned | Strict/compensatory definition somewhat circular |
| Architecture invariance | Kruskal-Wallis across architectures | Aligned | Unmatched L0 |

---

## 3. Validity Threats Checklist

- [x] **Data leakage**: No leakage detected. Train/test splits are seed-fixed (42). First-letter uses separate train words (1033) and test words (500). RAVEL uses 80/20 splits. Probes are trained on residual stream activations, not SAE reconstructions.

- [ ] **Contamination**: Not applicable in the traditional sense (no training). However, Gemma 2B's pretraining data likely contains information about cities, countries, continents, and languages. This means the model's internal representations already encode these hierarchies, which is the premise of the work. No contamination concern.

- [x] **Selection bias -- probe quality gate**: The relaxed quality gate (F1 >= 0.80) was applied, and city-country (F1 = 0.73) was included "with caveat." The strict gate (F1 > 0.90) would have excluded all RAVEL hierarchies except possibly city-continent. The relaxed gate lets city-country in despite being below threshold. The per-class absorption rates for city-country show extreme values (0% for US, 100% for 17 countries) that correlate with probe quality per-class. **FLAG: Selection bias in which instances contribute to city-country absorption measurement.**

- [x] **Overfitting to evaluation**: The primary findings (cross-domain variation) are tested across 4 hierarchies, 2 SAE widths, and 4 layers. The p-values are extremely significant (p < 1e-50). The finding is not specific to one configuration. **LOW RISK.**

- [x] **Cross-domain activation patching bug and fix**: The pilot had a critical bug (zeroing features with highest POSITIVE cosine instead of most NEGATIVE contribution). This was fixed in the FULL run, yielding city-continent recovery of 62% vs control 5%. The methodology fix is documented. However, the consolidation summary (written before the fix) still reports "Cross-domain patching FAILED" and "reverse direction." **FLAG: Consolidation summary is outdated -- the FULL activation patching results (with fix) show strong positive results (p < 1e-20, d = 1.50) that contradict the consolidation's "FAILED" verdict for cross-domain patching.**

- [x] **Multiple testing**: Bonferroni correction is applied for 6 pairwise comparisons. ANOVA-like tests (Kruskal-Wallis) are used before post-hoc tests. **Appropriate.**

---

## 4. Ablation Gap Analysis

### Components claimed as contributions vs ablation evidence:

| Component | Ablation exists? | Notes |
|-----------|------------------|-------|
| Cross-domain measurement extension | Yes | Multiple hierarchies serve as mutual controls |
| Probe-based absorption detection | Partial | Random direction baseline present; shuffled hierarchy absent |
| Activation patching causal test | Yes | Random latent zeroing control |
| Benign/pathological diagnostic | Yes | Random direction control (100% benign) |
| Hedging decomposition | Partial | Multi-L0 analysis for first-letter; single-L0 for RAVEL |
| Architecture comparison | Yes | Multiple architectures, statistical tests |
| Rate-distortion predictors | Yes | Leave-one-out CV, individual predictor tests |

### Missing ablations:

1. **Probe quality impact on absorption**: No systematic ablation showing how absorption rate changes as probe quality degrades. City-country (F1=0.73) shows 45% absorption while city-language (F1=0.82) shows 12% -- is this hierarchy-driven or probe-quality-driven? An ablation that artificially degrades probe quality on first-letter (where F1=1.0) and measures absorption change would be definitive.

2. **Token position ablation**: First-letter uses position -6; cross-domain uses position -2. No experiment tests the same hierarchy at both positions to quantify the position effect.

3. **Context template ablation**: The first-letter pipeline uses 3 prompts per word; cross-domain uses RAVEL entity templates. No experiment tests whether different prompt templates for the same hierarchy change absorption rates.

---

## 5. Reproducibility Score: 3.5 / 5

| Criterion | Score | Notes |
|-----------|-------|-------|
| Random seeds fixed | 5/5 | seed=42 throughout |
| Hyperparameters specified | 4/5 | Probe C values, L0 thresholds documented; some details (batch size, exact prompt templates) not in results |
| Code/data available | 3/5 | Code in `exp/code/`, uses public SAEs (Gemma Scope) and datasets (RAVEL). But no single script reproduces end-to-end |
| Hardware documented | 4/5 | GPU specified (RTX PRO 6000 Blackwell 95GB), model (Gemma 2B), software versions (SAELens >= 6.39) |
| 10% reproducibility likelihood | 3/5 | Main finding (cross-domain variation) likely reproducible. Exact numbers depend on probe quality, which depends on training randomness. city-country results especially sensitive. |

**Key reproducibility risk**: The probe training step is the critical bottleneck. Probes at F1=0.73 (city-country) are unstable -- small changes in training data or regularization could shift the quality gate assessment and substantially change downstream absorption rates. The paper should report probe training variability (e.g., std across multiple seed runs).

---

## 6. Top-3 Methodology Recommendations (Effort-to-Credibility Ratio)

### Recommendation 1: Probe-Quality Sensitivity Analysis (HIGH PRIORITY)
**What**: Train probes with 5 different random seeds. Report absorption rate mean and std across probe seeds for each hierarchy. Alternatively, artificially degrade first-letter probe quality to F1~0.85 and F1~0.73 (by adding label noise or using less data) and measure how absorption rate changes.

**Why**: This is the single biggest threat to the paper's central claim. If absorption rate is a function of probe quality rather than hierarchy structure, the paper's primary contribution is an artifact. Currently, city-country (lowest probe quality) shows highest absorption. This correlation must be addressed.

**What it would change**: If probe degradation increases absorption: the cross-domain variation finding needs major caveats. If probe degradation does NOT increase absorption: the paper's finding is greatly strengthened.

**Effort**: 2-4 GPU-hours. Reuses existing probe training code with different seeds/noise levels.

### Recommendation 2: Update Consolidation with Fixed Cross-Domain Patching Results (MEDIUM PRIORITY)
**What**: The consolidation summary (phase5) reports cross-domain patching as "FAILED" (0.05% recovery, reverse direction), but the actual FULL-run results (phase2/activation_patching_crossdomain.json) show strong positive results after the methodology bug fix: city-continent primary recovery 62% (p < 1e-20, d = 1.50), city-language primary recovery 34% (p < 1e-18, d = 0.75). The consolidation is stale.

**Why**: This is a critical inconsistency that could mislead the writing agent. The H7 verdict should be upgraded from "SUPPORTED_FIRSTLETTER_ONLY" to "SUPPORTED_CROSS_DOMAIN" based on the corrected results. The paper now has strong causal evidence across both first-letter AND semantic hierarchies, which substantially strengthens the secondary contribution.

**What it would change**: Upgrades the causal mechanism claim from "works for first-letter only (different mechanism for semantic)" to "works across domains with varying effect sizes (d=1.50 for city-continent, d=0.75 for city-language)." This is a major improvement to the paper.

**Effort**: 30 minutes (CPU only, update JSON/summaries).

### Recommendation 3: Report City-Country Results Separately with Explicit Caveat (MEDIUM PRIORITY)
**What**: Present city-country absorption results in a clearly marked "below quality gate" section. Report probe per-class breakdown showing that absorption rate is inversely correlated with per-class probe quality. Use only city-continent and city-language (both above relaxed gate) for the primary cross-domain comparison.

**Why**: Including city-country (F1=0.73, balanced accuracy 0.56) in the primary analysis without adequate caveats will draw reviewer criticism. The 80-class probe has many classes with <10 examples (Chad: 5 total, 1 probe correct, 100% absorption). These per-class rates are unreliable.

**What it would change**: The main finding (cross-domain variation, p < 1e-65) remains significant with 3 hierarchies. Removing city-country from the primary comparison makes the result more credible. City-country can be an informative supplementary analysis.

**Effort**: 1 hour (restructuring analysis, no new experiments needed).

---

## 7. Additional Observations

### 7.1 Rate-Distortion Predictor Direction Reversal
Individual predictor correlations are OPPOSITE direction in the FULL run (n=262) vs pilot (n=20): cos_sim rho went from positive to -0.108, co_occur from positive to -0.173. This is a textbook example of pilot instability with small n. The paper correctly reports this as a negative result. The lesson for the methodology: **never trust correlational results from n=20 pilot studies.** The bootstrap CIs should have flagged this (if they spanned zero in the pilot, which they likely did).

### 7.2 Benign/Pathological Test on Single Hierarchy
H8 was tested only on city-continent (50 entities, 1471 instances). The falsification is convincing for this hierarchy, but the paper would be stronger with even a small verification on first-letter or city-language to confirm generality.

### 7.3 First-Letter Absorption Rate Discrepancy Across Iterations
First-letter absorption at L24_16k: iter_008 reports 34.5%, iter_009 (full pipeline) reports 27.1%, and the permutation test table uses 42.9% as the first-letter rate. These discrepancies need reconciliation. The 42.9% figure likely comes from a different denominator or measurement approach. This inconsistency would confuse readers.

### 7.4 Cross-Domain Patching: Oceania Anomaly
In activation patching results, Oceania shows drastically lower recovery (4.2% primary) compared to other continents (53-94%). With 29 entities and 1121 absorbed contexts, this is a well-powered observation. This suggests absorption in Oceania cities operates through a different mechanism or the probe/absorber identification fails for this class. The paper should discuss this per-class heterogeneity rather than only reporting aggregate recovery rates.

---

## Summary Verdict

The experimental methodology is generally sound with proper controls, statistical tests, and honestly reported negative results. The **primary threat** to the paper's main conclusion is the correlation between probe quality and measured absorption rate across hierarchies. This must be addressed with a sensitivity analysis. The **secondary issue** is that the consolidation summary is outdated regarding cross-domain activation patching (which actually succeeded after a bug fix), and the writing agent should use the corrected FULL-run results. With these two issues addressed, the paper's methodology would be at the quality level expected for NeurIPS/ICLR submission.

**Overall methodology rating**: 7/10 (would be 9/10 with probe sensitivity analysis and corrected consolidation).
