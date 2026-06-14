# Result Debate Synthesis: Competitive Geometry of Feature Absorption in SAEs

**Synthesizer**: Result Debate Synthesizer Agent (sibyl-heavy)
**Date**: 2026-04-14
**Iteration**: 004
**Input**: 6 perspectives (optimist, skeptic, strategist, methodologist, comparativist, revisionist)

---

## 1. Consensus Map

The following conclusions are shared across all 6 perspectives with high confidence:

### 1.1 H1 (LV Detector) Is Empirically Falsified on GPT-2 Small
- **All 6 agree**: Test F1 = 0.128 (target: 0.65), ROC-AUC = 0.148 (below random), cosine baseline outperforms (F1 = 0.165). The sharp-threshold prediction fails (linear AIC < sigmoid AIC). Cross-architecture generalization produces F1 = 0.0.
- **Nuance**: The methodologist and revisionist note the model substitution (GPT-2 instead of Gemma-2-2B) creates ambiguity -- the failure may be model-specific rather than theory-terminal. However, even the optimist acknowledges F1 = 0.128 is far below the 0.65 criterion.
- **Verdict**: H1 is falsified on the tested model. The LV detector in its current form has no practical value.

### 1.2 H2 (Corpus PMI) Is Cleanly Falsified
- **All 6 agree**: PMI coefficient is negative (-0.006, wrong sign), non-significant (p = 0.593), partial R^2 = 0.0006 (criterion: 0.10). No perspective offers a credible defense.
- **Verdict**: Corpus co-occurrence statistics do not predict absorption. This is the cleanest null result in the study.

### 1.3 H3 Falsification (Downstream Correlation) Is the Study's Strongest Finding
- **All 6 agree**: Absorption correlates with downstream SAE quality at r = -0.595 (sparse probing), -0.431 (SCR), -0.454 (RAVEL TPP). These survive Bonferroni correction. Partial correlations increase after controlling for width/layer/architecture (up to r = -0.677). The matched-pair RAVEL comparison yields Cohen's d = 2.13 (p = 0.006).
- **5 of 6 agree** this is the paper's headline contribution and the most practically important result.
- **Verdict**: The first systematic, statistically controlled evidence that SAE absorption predicts downstream task degradation. This is the paper's real contribution.

### 1.4 The Paper Needs a Major Framing Pivot
- **All 6 agree**: The paper cannot proceed with "Lotka-Volterra competitive exclusion" as its theoretical centerpiece. The strategist, comparativist, and revisionist all independently recommend pivoting to an empirical survey/downstream-impact framing. The optimist agrees the LV detector failed but frames it as a secondary concern behind the H3 victory.
- **Verdict**: Restructure around the downstream correlation finding as primary contribution, with the absorption taxonomy and scaling survey as supporting contributions, and the LV/PMI results as honest negative results.

### 1.5 Gemma-2-2B Replication Is Essential
- **All 6 agree**: The model substitution (GPT-2 Small instead of Gemma-2-2B) is the single largest methodological limitation. The C3A result (the strongest finding) does use Gemma-2-2B data from SAEBench, but all other components are GPT-2 only.
- **Verdict**: Gemma-2-2B replication is the highest-priority next step.

---

## 2. Conflict Resolution

### Conflict A: Is the C3A Downstream Correlation Genuine or Confounded by Width?

- **Optimist position**: The partial correlations controlling for log(width), layer, and architecture are *stronger* than raw correlations (r = -0.661 vs -0.595 for sparse probing). Width is a suppressor variable, not a confound. The matched-pair comparison confirms the relationship with p = 0.006.
- **Skeptic position**: The matched pairs are not actually matched -- all high-absorption SAEs are 1M-width, all low-absorption SAEs are 16k/65k. L0 (average active features) is not controlled and could be the true common cause. The partial correlation with only 54 data points and highly correlated predictors may be unreliable.
- **Methodologist position**: The correlation is real but the causal language ("downstream impact") is overstated for observational data. Adding L0 as a covariate is critical.

**Synthesis judgment**: The skeptic raises a legitimate concern about L0 as an uncontrolled confound, but the weight of evidence favors the correlation being genuine for the following reasons:
1. Three independent downstream metrics (sparse probing, SCR, RAVEL) all show the same direction, reducing the probability of a spurious result.
2. Partial correlations *increase* after controlling for width/layer/arch, which is the opposite of what would happen if width were driving a spurious correlation.
3. The matched-pair Cohen's d = 2.13 is enormous -- even if partially confounded by width, the effect size suggests a real underlying relationship.
4. However, the skeptic is correct that L0 must be controlled before claiming causation. The within-width stratified analysis is essential.

**Action**: Run partial correlation additionally controlling for L0. If partial r remains > 0.40 for sparse probing and SCR, the confound objection is substantially weakened. Also compute within-width correlations (especially within the 65k width class where absorption variation is most natural).

### Conflict B: Is the Absorption Taxonomy (92.3%) a Genuine Discovery or a Methodological Artifact?

- **Optimist position**: The 92.3% comprehensive rate (driven by 88.5% Type II) shows the Chanin metric captures only the extreme tail. "True" absorption is a continuous spectrum of signal degradation.
- **Skeptic position**: The Type II rate is inflated because parent features were identified by selectivity heuristic (not sae-spelling ground truth) and the comparison baseline (global mean-when-active) is flawed. The same metric would flag polysemantic features that fire at varying magnitudes in different contexts -- this has nothing to do with absorption.
- **Methodologist position**: The 0.5 magnitude-ratio threshold is arbitrary with no sensitivity ablation. The result is "CRITICAL: likely inflated" per the experiment's own audit.

**Synthesis judgment**: The skeptic and methodologist are correct that the 92.3% figure is not credible in its current form. The key issue is parent-feature misidentification: if the heuristically-identified parents are not the true first-letter parent features, the magnitude deficit is meaningless. However, the optimist is correct that the *concept* of a multi-type taxonomy (Type I full absorption, Type II partial suppression, Type III distributed) is valuable and likely to survive recalibration.

**Action**: Re-run the taxonomy using sae-spelling's actual parent feature IDs. Ablate the magnitude-ratio threshold at 0.3, 0.4, 0.5, 0.6, 0.7. A properly calibrated taxonomy with Type II rate between 40-60% would be credible and still substantially exceed the Chanin Type I rate. The 92.3% figure must not appear in the paper.

### Conflict C: Is the LV Theory "Dead" or "Untested on the Right Model"?

- **Skeptic position**: The LV framework is empirically falsified -- AUC < 0.5 means negative informational value. The theory is dead.
- **Optimist position**: The LV pipeline works computationally (37.5M pairs, zero NaN/Inf) and shows precision = 0.50-0.60 at tight thresholds. It is not dead, just narrower than hoped.
- **Methodologist position**: The model substitution (GPT-2 instead of Gemma-2) makes the failure ambiguous. The theory should be tested on its intended model before being declared dead.
- **Revisionist position**: The fundamental analogy (ecology to SAE feature space) is flawed. The LV framework conflates frequency differences with causal competitive pressure.

**Synthesis judgment**: The revisionist's structural critique is the most persuasive. The problem is not just empirical failure on the wrong model -- the alpha_ij formulation conflates base-rate frequency imbalance with genuine competitive displacement. Even if Gemma-2 results improve marginally, the underlying analogy is mechanistically unsound. However, the methodologist is correct that a clean negative result on Gemma-2 would strengthen the paper's narrative (testing and honestly reporting a falsified hypothesis).

**Action**: If Gemma-2 access is obtained, run the LV detector as a completeness check. Report the result regardless of outcome. But do not invest significant effort in "saving" the LV framework -- the paper's framing should not depend on it.

### Conflict D: Should the Paper Target a Top-Tier Venue or a Workshop?

- **Optimist position**: The H3 falsification alone constitutes a "strong NeurIPS-track contribution."
- **Comparativist position**: Three of four hypotheses failed. Recommend workshop paper at NeurIPS 2026 or mid-tier venue.
- **Strategist position**: PROCEED with a major framing pivot. The downstream correlation + taxonomy + scaling survey form a coherent, publishable paper.

**Synthesis judgment**: The current results, with the recommended additional analyses, could support a main-conference submission if and only if: (1) the L0 confound is addressed and the downstream correlation holds, (2) the taxonomy is properly calibrated, and (3) ideally some Gemma-2 replication is available. Without these, a workshop paper is more appropriate. With them, the paper offers the first systematic evidence that absorption matters for downstream safety -- a timely contribution given DeepMind's SAE deprioritization.

**Action**: Pursue the recommended additional analyses (L0 control, taxonomy fix, Gemma-2 replication). If all three succeed, target NeurIPS 2026 main conference. If the L0 control weakens the C3A result substantially, pivot to workshop submission.

---

## 3. Result Quality Score: 5.5 / 10

**Justification**:

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Hypothesis testing rigor | 7/10 | Pre-registered thresholds, Bonferroni correction, multiple metrics. Model substitution is the main deduction. |
| Effect sizes | 8/10 | H3 correlations (r = -0.60 to -0.68) and matched-pair Cohen's d = 2.13 are genuinely large. |
| Hypothesis success rate | 3/10 | 2 of 5 hypotheses clearly falsified (H1, H2), 1 partially falsified (H4), 1 supported with caveats (H5). H3 "falsified" in the positive direction is the bright spot. |
| Methodological soundness | 4/10 | Model substitution, inflated taxonomy, missing L0 control, underpowered safety probe (n=3). |
| Novelty of findings | 7/10 | The downstream correlation is genuinely novel. The layer gradient and width scaling are novel empirical contributions. The negative results (LV, PMI) are informative. |
| Completeness | 5/10 | Gemma-2 replication missing. Taxonomy needs recalibration. Safety probe underpowered. |

**Overall**: The study has one excellent finding (C3A downstream correlation), two informative negative results (LV, PMI), one novel empirical contribution (30-SAE scaling survey), and one conceptually valuable but methodologically flawed contribution (taxonomy). The overall quality is dragged down by the model substitution and the gulf between the ambitious theoretical proposal and what the data actually supports.

---

## 4. Key Findings

1. **Absorption predicts downstream SAE quality**: Across 54 Gemma-2-2B SAEs, absorption score correlates at r = -0.595 to -0.677 (after confound controls) with sparse probing F1, SCR, and RAVEL performance. This is the first systematic quantification of this relationship with proper statistical controls, and it directly addresses the question of whether absorption matters beyond a metric.

2. **The Lotka-Volterra theoretical framework is empirically falsified**: The LV competition coefficient alpha_ij fails as an unsupervised absorption detector (F1 = 0.128, worse than cosine baseline), the predicted sharp threshold transition does not exist, and the framework does not generalize across SAE architectures. The ecological analogy does not transfer to SAE feature dynamics.

3. **Corpus co-occurrence does not predict absorption**: PMI has zero predictive power (partial R^2 = 0.0006, wrong sign). Absorption appears to be architecture/objective-driven, not data-driven. This is an informative null that updates the field's understanding.

4. **Absorption follows systematic layer and width scaling laws**: Absorption peaks at early-to-mid layers (L3-L5) and drops to near-zero by L10-L11. It increases with SAE width at fixed layer. This 30-SAE survey across 31 configurations is the most comprehensive absorption measurement to date.

5. **The true scope of absorption likely exceeds reported rates, but the precise magnitude is uncertain**: The three-type taxonomy (Type I/II/III) suggests comprehensive absorption is substantially higher than Chanin's 15-35% Type I rate, but the specific 92.3% figure is methodologically inflated and must be recalibrated before publication.

---

## 5. Methodology Gaps

### Critical (must fix before submission)

1. **L0 confound in C3A**: Add L0 as an explicit covariate in the partial correlation analysis. Without this, the skeptic's objection (L0 as common cause of both high absorption and low downstream performance) cannot be dismissed. Effort: 0 GPU-hours (analysis only).

2. **Taxonomy recalibration**: Replace the selectivity-heuristic parent features with sae-spelling ground-truth parent feature IDs. Ablate the magnitude-ratio threshold. The 92.3% figure is not publishable. Effort: 1-2 GPU-hours.

3. **Gemma-2-2B replication**: At minimum, replicate the C2D taxonomy and C1B LV detector on Gemma Scope SAEs. The C3A result already uses Gemma-2 data, but the asymmetry (H1/H2/H4/H5 on GPT-2, H3 on Gemma-2) weakens the paper. Effort: 8 GPU-hours.

### Important (strongly recommended)

4. **Properly powered safety probe (C3C)**: Current n=3, confounded by layer. Need n >= 10 SAEs at the same layer (e.g., layer 8) with varying absorption. Effort: 2 GPU-hours.

5. **Within-width stratified correlations**: Compute absorption-downstream correlations within each width class (especially 65k). If the correlation holds within width, the confound objection is eliminated. Effort: 0 GPU-hours (analysis only).

6. **Component ablation of alpha_ij**: No experiment separates sigma_ij (co-activation rate) from the frequency ratio (f_j/f_i). The proper ablation would test each component independently. Effort: 1 GPU-hour.

### Desirable (would strengthen paper)

7. **Causal intervention via activation patching**: Verify the absorption-downstream correlation is causal by ablating absorbed features and measuring downstream performance change. Effort: 4 GPU-hours.

8. **Add ATM SAE and Matryoshka SAE to C3A**: Expand the architectural diversity of the downstream correlation analysis. Effort: 2 GPU-hours.

---

## 6. Competitive Position

### What This Work Does That No Prior Work Does

The **single novel contribution** that survives scrutiny is the systematic, statistically controlled quantification that SAE absorption score predicts downstream task degradation. No prior work -- not Chanin et al. (2024), not SAEBench (Karvonen et al., 2025), not the DeepMind blog post -- provides this analysis with Bonferroni correction, partial correlations, and matched-pair effect sizes.

### Position Relative to Concurrent Work

| Concurrent Work | Our Position |
|----------------|-------------|
| Chanin et al. (NeurIPS 2025): Defined absorption, 15-35% rate | We extend with taxonomy and downstream validation |
| SAEBench (ICML 2025): Absorption as one of 8 metrics, no correlation analysis | We provide the correlation analysis they did not |
| OrtSAE, ATM SAE, Matryoshka SAE: Absorption reduction methods | We provide the empirical justification for why reduction matters |
| Masked Regularization (2026): Co-occurrence disruption reduces absorption | Our PMI null result complements their interventional finding |
| DeepMind deprioritization (2025): Qualitative claim that SAEs underperform dense probes | We quantify *why*: absorption degrades downstream SAE quality |
| Sanity Checks for SAEs (2026): SAEs recover only 9% of true features | Our work is complementary -- absorption is one mechanism of this failure |

### Venue Assessment

With the recommended additional analyses (L0 control, taxonomy fix, Gemma-2 replication), the paper is competitive for **NeurIPS 2026 main conference**. Without them, it is appropriate for a **NeurIPS 2026 workshop** or **AAAI/EMNLP main conference**. The downstream correlation finding is timely and fills a specific gap in the SAE evaluation literature.

---

## 7. Hypothesis Update

### Hypotheses That Survived (with caveats)

| Hypothesis | Status | Caveat |
|-----------|--------|--------|
| H3 (falsified positively): Absorption predicts downstream quality | SURVIVED | L0 confound must be controlled; matched-pair width imbalance must be addressed |
| H5: Comprehensive absorption rate > Chanin Type I rate | SURVIVED (concept) | Specific rate (92.3%) is inflated; true rate likely 40-70% after recalibration |

### Hypotheses That Need Revision

| Hypothesis | Revision |
|-----------|----------|
| H1: LV competition coefficient detects absorption | Revised: alpha_ij does not predict absorption. The ecological competition analogy does not transfer. Absorption is better understood as a sparsity-reconstruction tradeoff symptom than competitive exclusion. |
| H2: Corpus PMI drives absorption patterns | Revised: Absorption is architecture/objective-driven, not data-driven. Corpus statistics explain ~0% of absorption variance. |
| H4: DAS(k=3) increases monotonically with width | Revised: The width-absorption relationship exists (absorption generally increases with width at fixed layer) but the distributed competitive exclusion mechanism (DAS) does not produce the predicted monotonic pattern. L0 variation across width confounds the analysis. |

### New Hypotheses Generated by the Data

1. **NH1 (from Revisionist)**: Absorption is primarily mediated by SAE sparsity pressure (L0), not competitive exclusion or corpus statistics. Test by varying L0 penalty at fixed width/layer.
2. **NH2 (from Revisionist)**: The C3A downstream correlation is driven by the 1M-width extreme and vanishes within narrower width classes. Test by computing within-width correlations.
3. **NH3 (from Optimist)**: The unlearning metric's decorrelation from absorption (r = -0.175) reveals the mechanistic boundary of absorption's impact -- it affects feature identification tasks but not knowledge circuit tasks. This is a publishable taxonomic observation.

---

## 8. Action Plan

### Overall Recommendation: **PROCEED** with major framing pivot

The study has one strong positive finding (downstream correlation), two informative negative results (LV detector, PMI), and valuable empirical contributions (scaling survey, taxonomy concept). This justifies proceeding to paper writing with a restructured framing.

### Immediate Actions (Priority 1 -- before writing)

| Action | Effort | Expected Impact | Blocker? |
|--------|--------|-----------------|----------|
| Add L0 as covariate in C3A partial correlations | 0 GPU-hrs | Addresses the single biggest threat to the headline finding | No |
| Compute within-width stratified correlations for C3A | 0 GPU-hrs | Eliminates or confirms width confound objection | No |
| Re-run taxonomy (C2D) with sae-spelling ground-truth parent features | 1-2 GPU-hrs | Makes the taxonomy publishable (current 92.3% is not credible) | No |

### Near-Term Actions (Priority 2 -- parallel with writing)

| Action | Effort | Expected Impact | Blocker? |
|--------|--------|-----------------|----------|
| Obtain Gemma-2-2B access and replicate C2D + C1B | 8 GPU-hrs | Cross-model validation; essential for NeurIPS-tier submission | HuggingFace gated access |
| Properly powered safety probe: 10+ SAEs at layer 8 | 2 GPU-hrs | Replaces underpowered C3C (n=3) with credible result | No |
| Begin paper draft with pivoted framing | 0 GPU-hrs | Writing should not wait for all experiments | No |

### Paper Structure (Recommended)

1. **Introduction**: "Absorption is pervasive but its downstream consequences are unvalidated. We provide the first systematic evidence that it matters."
2. **Absorption Taxonomy**: Type I/II/III classification showing comprehensive absorption exceeds the Chanin metric (with properly calibrated rates).
3. **Scaling Survey**: Layer gradient (absorption peaks at L3-5, zero by L10-11) and width scaling (absorption increases with width at fixed layer).
4. **Downstream Impact Validation**: C3A correlation analysis (the headline finding) with partial correlations and matched-pair comparison.
5. **Negative Results**: LV detector and PMI prediction reported honestly as informative nulls. Brief, in the discussion section or appendix.
6. **Implications**: For SAE development (optimize for low absorption), safety applications (absorption is a concrete mechanism for SAE failure on safety tasks), and future work (causal intervention, Gemma-2 replication).

### What NOT to Do

- Do NOT invest further GPU-hours in "saving" the LV detector. The framework is mechanistically flawed (per revisionist), not just empirically challenged.
- Do NOT report the 92.3% taxonomy figure without recalibration. It will invite immediate reviewer rejection.
- Do NOT claim "causal" downstream impact based on correlational evidence alone. Use language like "absorption is a strong predictor of" rather than "absorption causes."
- Do NOT suppress the negative results (LV, PMI). Reporting honest falsifications strengthens credibility.
