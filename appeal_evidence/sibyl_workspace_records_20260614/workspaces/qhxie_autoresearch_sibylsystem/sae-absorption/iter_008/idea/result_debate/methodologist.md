# Methodologist Audit: SAE Absorption Cross-Domain Characterization (Iteration 9)

## Executive Summary

The experimental methodology demonstrates genuine rigor in several areas -- honest negative result reporting, threshold sensitivity analysis, and proper statistical testing for activation patching. However, three critical threats to the primary conclusions remain: (1) probe quality asymmetry between first-letter (F1=0.97) and cross-domain hierarchies (F1=0.79-0.84) is a **severe confound** that likely drives the central cross-domain comparison, (2) the activation patching experiment uses subword-fragment tokens that are not representative of natural language, and (3) the architecture comparison is statistically underpowered and conducted at a layer where cross-domain probes are far below quality gate. The paper's primary claim -- that absorption varies significantly across hierarchy types -- rests on comparisons where the measurement instrument (probe) has dramatically different error rates across conditions.

---

## 1. Baseline Fairness Audit

### 1.1 First-Letter vs. Cross-Domain: Asymmetric Probe Quality (CRITICAL)

**The most serious methodological threat in this study.** The central contribution compares absorption rates across hierarchy types, but the probes used to measure absorption have very different quality levels:

| Hierarchy | Best Probe F1 | Quality Gate |
|-----------|--------------|-------------|
| first-letter | 0.9711 | PASS (strict) |
| city-continent | 0.8428 | Below strict |
| city-country | 0.7895 | Below relaxed |
| city-language | 0.8234 | Below strict |

The threshold sensitivity analysis itself shows rho=-0.756 between probe F1 and false negative rate (p<0.001). This means **lower probe quality mechanistically inflates absorption measurements**. The cross-domain absorption rates (13.6-35.8%) are measured with probes that have 3-4x higher error rates than the first-letter probe (2.9% error for first-letter vs. 15.7-21.1% for RAVEL hierarchies).

**Specific concern:** At L24, first-letter absorption is 34.5% (16k) and 25.5% (65k). City-country is 18.5% and 12.7% respectively -- *lower* than first-letter. City-language is 13.6% -- also lower. Only city-continent is comparable (35.8%). The 4/6 significant pairwise comparisons all show cross-domain rates *below* first-letter. This is the opposite of the pilot finding at L12, where semantic hierarchies showed higher rates.

**The question this raises:** Is the observed absorption variation genuinely caused by hierarchy structure, or is it an artifact of differential probe error rates? The data is consistent with both interpretations:
- True interpretation: Different hierarchy types have different absorption susceptibility
- Confound interpretation: Lower-quality probes miss true positives differently through SAE vs. raw residual stream, creating spurious absorption measurements

The paper should present a **probe-error-adjusted analysis**: for each hierarchy, compute the expected false negative rate attributable purely to probe error (by measuring probe FN rate on raw residual stream), subtract this from the SAE-induced FN rate, and compare the residual. Without this, the central finding is vulnerable to the probe quality confound.

### 1.2 Activation Patching Baseline Fairness

The activation patching experiment (Phase 0.1) is better controlled:
- Child-zeroed vs. magnitude-matched random control (15 control features per word)
- Same probe applied to both conditions
- Recovery rate difference measured within-word (paired design)

**However**, the probe used has F1=0.883 (below strict 0.90 gate). This means ~12% of the recovery measurement could reflect probe noise rather than genuine signal recovery. The strong effect size (d=1.33) and significance (p=0.000218) provide substantial margin, but the probe quality should be explicitly tested as a moderator.

### 1.3 Hedging Classification Baseline

The multi-L0 hedging analysis (L0=22 to 176) is methodologically sound for first-letter. **However**, the cross-domain hedging comparison is not fair:
- First-letter uses multi-L0 analysis at layer 12 (three L0 levels: 22, 82, 176)
- City-language uses single-L0 analysis at layer 24 ("canonical" L0 only)
- The two analyses use different layers, different L0 ranges, and different SAE configurations

The hedging comparison table (city-language strict=66.7% vs. first-letter strict=7.9%) compares incomparable analyses. This is stated in the limitations but the comparison is still prominently featured in the paper claims.

---

## 2. Metric-Claim Alignment

### 2.1 Central Claim: "Absorption varies significantly across hierarchy types"

**Metric used:** ANOVA on absorption rates across hierarchies (p=0.005)
**Assessment:** The metric captures variation, but does not disentangle probe-quality-driven variation from genuine absorption variation. The claim would be strengthened by:
- Controlling for probe quality in the ANOVA (include probe F1 as covariate -- ANCOVA)
- Showing that within-hierarchy absorption variation (across SAE configs) is smaller than between-hierarchy variation when probe quality is held constant

### 2.2 Claim: "First causal evidence for competitive exclusion"

**Metric used:** Recovery rate after child feature zeroing vs. control
**Assessment:** This metric is well-aligned with the claim. Recovery rate directly measures whether removing the child feature allows the parent concept to be recovered. The control (magnitude-matched random feature zeroing) is appropriate.

**Gap:** The claim says "competitive exclusion" but the mechanism tested is "child suppresses parent." The activation patching does not distinguish between:
(a) Child feature actively suppresses parent (competitive exclusion)
(b) Child feature absorbs the activation that would otherwise be captured by parent (absorption)
(c) SAE simply encodes a different representation when child is present (interference)

The evidence supports absorption but does not isolate the mechanism to competitive exclusion specifically.

### 2.3 Claim: "Tightened hedging reveals near-tautology"

**Metric used:** Strict vs. loose hedging percentage
**Assessment:** Well-aligned. The finding that 86.2% of false negatives resolve via non-parent features (compensatory) directly supports the claim that loose hedging classification is overly permissive.

### 2.4 Claim: "Layer-dependent absorption (15x variation)"

**Metric used:** Absorption rate at L6/L12/L18/L24
**Assessment:** The 15x variation (2.2% to 34.5% for first-letter) is measured with F1=1.0 probes from sae_spelling at the same position. This is the most cleanly measured finding in the paper because the probe quality is uniformly excellent across layers. The layer-dependence finding is credible.

---

## 3. Validity Threats Checklist

### 3.1 Data Leakage
- [ ] **RAVEL dataset split:** 80/20 stratified split, seed=42. No explicit mention of preventing city-name leakage between probe training and absorption measurement. Since the same cities are used for both probe training and absorption measurement, the probe's test-set F1 may overestimate generalization.
- [x] **First-letter probes:** sae_spelling pipeline produces F1=1.0, which is appropriate because first-letter is a deterministic property.

### 3.2 Contamination
- [x] **Not applicable:** No model training occurs. All experiments are inference-only on pre-trained SAEs and models.

### 3.3 Selection Bias
- [x] **Activation patching word selection:** 7 pilot core words + 18 discovered via integrated-gradients-inspired search. The discovery method (decoder-probe cosine x mean activation on absorbed tokens) is biased toward finding words where absorption IS present. The 6/25 words with zero absorption were largely the pilot core words.
  - **Concern:** The 58.8% mean absorption rate is inflated by this selection bias. The paper should report both the overall corpus absorption rate and the per-word rate among selected words, making the selection process explicit.
- [ ] **RAVEL city subset:** 200 cities in pilot mode. The subsetting strategy is not documented in the results. If cities are randomly sampled, this is fine. If selected by frequency or availability, bias could exist.

### 3.4 Overfitting to Evaluation
- [x] **Layer selection:** All RAVEL hierarchies converge on L24 as the best layer. This is selected to maximize probe quality (which is reasonable), but it means all cross-domain comparisons are at a single layer. The generalizability of findings to other layers is unknown.
- [x] **SAE selection:** Gemma Scope canonical SAEs are the default in the field. Using pre-trained SAEs avoids hyperparameter optimization bias.

### 3.5 Token Population Concerns (Activation Patching)
- **Many discovered words are subword fragments or non-English tokens:** "xfa", "udy", "wner", "uzu", "uki", "unton", "wikk", "zorgt", "conmigo", "antigos", "menjadikan", "yaitu", "membuka", "recoge", "washingtonpost"
  - Only 7/25 words are common English words (the pilot core)
  - Among the pilot core: "eight" (raw acc=0.0), "liked/lower/offer/other/under" (raw acc=0.96-1.0, no absorption), "often" (12 absorbed)
  - The discovery procedure finds absorption in *uncommon tokens* where the model is uncertain about the first letter
  - **This creates a selection effect:** absorption is demonstrated primarily on tokens where the model is already performing poorly on first-letter identification (low raw accuracy). It is unclear whether this generalizes to well-represented tokens.
  - 4/7 pilot core words show ZERO absorption, and 1 shows raw_acc=0.0 (model cannot identify first letter at all). Only "often" shows genuine absorption among common English words.

---

## 4. Ablation Gap Analysis

### Proposed Components and Ablation Coverage

| Component | Ablation Exists? | Assessment |
|-----------|-----------------|-----------|
| Cross-domain probe training | Yes (4 layers x 4 hierarchies) | Comprehensive |
| SAE width effect (16k vs 65k) | Yes (paired measurements) | Good |
| SAE layer effect (L6/12/18/24) | Yes (first-letter across 8 configs) | Excellent |
| Architecture effect (JumpReLU/BatchTopK/Matryoshka) | Yes but underpowered | ANOVA p=0.87, n too small |
| Probe quality as confounder | **NO** | CRITICAL MISSING ABLATION |
| Token position effect | Partial (v1 used -2, v2 used -1) | Not systematically ablated |
| Context count effect (100 vs 200) | **NO** | Not tested |
| Integrated-gradients attribution steps | **NO** | Fixed at 10 steps, sensitivity unknown |

### Critical Missing Ablation: Probe Quality Control

The single most important ablation is missing. To disentangle probe quality from hierarchy type:
1. **Degrade the first-letter probe** to F1 ~0.84 (matching cross-domain quality) by subsampling training data or adding noise, then re-measure first-letter absorption. If absorption increases to match cross-domain rates, the finding is a probe artifact.
2. Alternatively, **train MLP probes** for RAVEL hierarchies (potentially achieving higher F1) and re-measure cross-domain absorption. If absorption changes proportionally to F1 change, probe quality dominates.

### Missing Ablation: SAE-Independent False Negative Rate

The absorption measurement relies on comparing probe predictions on raw residual stream vs. SAE-reconstructed stream. The paper does not report the "probe-only baseline" (FN rate without SAE) separately for each hierarchy at each layer. The methodology section mentions this control but it is not in the results.

---

## 5. Reproducibility Score: 3/5

| Criterion | Score | Notes |
|-----------|-------|-------|
| Random seeds fixed | 1/1 | seed=42 throughout |
| All hyperparameters specified | 0.5/1 | Probe C values reported; activation patching batch size not always specified; integrated-gradients steps=10 stated but sensitivity not tested |
| Code/data available | 0.5/1 | All code in exp/code/; RAVEL is public; SAEs from HuggingFace; but intermediate cached activations not available |
| Hardware requirements documented | 0.5/1 | GPU specified (RTX PRO 6000 95GB) but timing anomalies suggest some results may be cached or pre-computed (e.g., "activation patching full" completed in 2.3 minutes for 25 words x 200 contexts -- this seems too fast for 5000+ forward passes through Gemma 2B + SAE) |
| Reproducible within 10%? | 0.5/1 | First-letter results should reproduce (F1=1.0 probe is deterministic). Cross-domain results depend heavily on probe training which is only 0.5/1 reproducible due to class imbalance and small effective sample sizes for rare categories |

**Timing anomaly flag:** The consolidation summary reports "total_wall_clock_min: 5" for all tasks, and individual phases completed in 1-2 minutes each. The task plan estimated 60 minutes for activation patching alone (20+ words x 200 contexts x forward passes). Either the implementation is remarkably efficient, or some results are loaded from cache rather than freshly computed. The paper should clarify this.

---

## 6. Top-3 Methodology Recommendations (Effort-to-Credibility Ratio)

### Recommendation 1: Probe-Quality-Controlled Cross-Domain Comparison (HIGH IMPACT, MEDIUM EFFORT)

**What:** Degrade the first-letter probe to match RAVEL probe quality (~F1=0.84) by subsampling training data (e.g., use only 200 training examples instead of 1040). Re-measure first-letter absorption at L24 with this degraded probe. Also compute the "probe-only baseline" (FN rate on raw residual stream without SAE) for each hierarchy.

**Why:** This single experiment would resolve the most critical ambiguity in the paper. If degraded-probe first-letter absorption remains at ~34.5%, the cross-domain variation is genuine. If it increases dramatically, the central finding is a probe artifact.

**What changes if it fails:** If absorption at degraded probe quality is 50%+ (matching or exceeding cross-domain rates), the paper's primary contribution collapses. The narrative would shift from "hierarchy type determines absorption" to "probe quality determines measured absorption."

### Recommendation 2: Report Probe-Only FN Rate as Explicit Baseline (HIGH IMPACT, LOW EFFORT)

**What:** For every hierarchy x layer x SAE combination, report three numbers: (a) probe accuracy on raw residual stream, (b) probe accuracy after SAE reconstruction, (c) the difference (SAE-induced FN). Currently only (b) is systematically reported.

**Why:** This separates "probe error" from "SAE-induced absorption." The current absorption rate conflates both. Even if the probe is perfect on raw activations, the SAE reconstruction error may cause probe failures that are not absorption.

**What changes if it shows:** If the raw-stream probe already has 10-15% FN rate for cross-domain hierarchies (which F1=0.84 implies), then the "absorption" rate of 13-36% would include substantial probe error, not just genuine SAE absorption.

### Recommendation 3: Activation Patching on Natural Tokens (MEDIUM IMPACT, MEDIUM EFFORT)

**What:** Restrict or separately report activation patching results for common English words only (pilot core minus "eight"). Filter discovered words to tokens that appear in standard English text (frequency rank < 50k). Currently 15/18 discovered words are subword fragments, non-English words, or rare tokens.

**Why:** The paper claims to provide "causal evidence for absorption" but the evidence comes primarily from tokens where the model already struggles with first-letter identification. For the causal claim to have impact, it should apply to representative tokens, not adversarially-selected edge cases.

**What changes if it shows:** If common English words show minimal absorption (as 4/7 pilot words suggest), the practical impact of absorption is limited to rare/non-standard tokens. This would not invalidate the finding but would substantially narrow its significance for safety-relevant feature detection.

---

## Additional Observations

### Layer Mismatch in Cross-Domain Comparisons

The architecture comparison uses layer 12 (the only layer with all architectures), but the cross-domain absorption measurement uses layer 24 (best probe quality). This means:
- Architecture rankings are measured at a layer where cross-domain probes are very poor (F1=0.62-0.79 at L12)
- The primary absorption comparison is at a layer where no architecture comparison exists
- These two results cannot be directly related

### Honest Negative Results

The transparent reporting of GAS failure (rho=0.12), CMI failure (rho=0.044), and Absorption Tax failure (rho=-0.20) is genuinely commendable. These negative results strengthen the paper's credibility significantly.

### Sample Size Concerns in Hedging Analysis

The cross-domain hedging analysis has n=0 false negatives for city-continent and n=3 for city-language. The city-language hedging decomposition (66.7% strict / 33.3% compensatory) is based on classifying exactly 3 tokens. Bootstrap CIs of [0%, 100%] confirm this is uninformative. The comparison of city-language hedging (66.7%) vs. first-letter hedging (7.9%) should not be presented as a finding.

### Consolidation Timing Anomaly

The consolidation summary reports the entire iteration 9 completing 4 tasks in ~5 minutes wall-clock time. The task plan estimated 5.5-8 hours. Either many tasks were reused from prior iterations (which should be stated explicitly), or timing data is inaccurate. This discrepancy should be addressed for reproducibility transparency.

---

## Summary Assessment

The methodology has solid foundations: seed-fixed experiments, proper statistical tests for the activation patching result, honest negative result reporting, and structural threshold sensitivity analysis. However, the central cross-domain comparison suffers from a severe **probe quality confound** that is acknowledged but not controlled for. Until a probe-quality-matched comparison is conducted (Recommendation 1), the paper's primary contribution -- that absorption varies by hierarchy type -- cannot be distinguished from the alternative explanation that absorption varies by probe quality. The activation patching result (H7) is the strongest finding and should be more prominently positioned, while the cross-domain comparison (H1) needs the quality-controlled ablation before it can be claimed with high confidence.
