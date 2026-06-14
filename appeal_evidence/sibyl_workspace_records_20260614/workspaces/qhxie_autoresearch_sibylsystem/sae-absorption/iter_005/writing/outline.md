# Paper Outline

## Title

**Disentangling Feature Absorption: Confound Resolution, Cross-Domain Probing, and Scaling Surfaces for SAE Quality Assessment**

---

## 1. Introduction

- Feature absorption -- where an SAE latent silently fails to fire because a more specific co-occurring latent subsumes its role -- is the most documented failure mode of sparse autoencoders for mechanistic interpretability. Chanin et al. (2024) measure 15--35% absorption rates on the first-letter spelling task across hundreds of Gemma Scope SAEs, and every tested SAE architecture exhibits the phenomenon.
- Three blocking weaknesses limit the field's understanding: (a) the absorption-quality correlation (r = -0.595 across 54 Gemma Scope SAEs) has never been tested with L0 as a covariate, leaving open the possibility that absorption is merely a proxy for sparsity; (b) all absorption measurements use a single evaluation task (first-letter spelling), a deterministic hierarchy that may systematically inflate rates; (c) no study maps absorption jointly across SAE width and L0, preventing practitioners from selecting hyperparameters that minimize absorption.
- This paper addresses all three weaknesses. We apply epidemiological causal inference methods (partial correlation with L0 control, Baron-Kenny mediation, Rosenbaum sensitivity analysis) to 48 Gemma Scope SAEs, attempt the first absorption measurement on knowledge hierarchies (city-country, continent, language), and construct a 420-SAE absorption scaling surface with formal interaction testing.
- **Key result preview**: After controlling for L0, the absorption-quality association strengthens for sparse probing (partial r from -0.664 to -0.746, a suppression effect) and remains significant for 3/4 quality metrics. The 420-SAE scaling surface reveals a highly significant width-L0 interaction (GAM p = 3.1e-15), with absorption concentrated in the high-width, low-L0 regime. Cross-domain measurement exposes a critical limitation of the dominance-based absorption metric when applied beyond spelling.

> **Ref before appearance**: See Figure 1 for the partial correlation results and Figure 2 for the mediation path diagram.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders for Mechanistic Interpretability
- SAEs decompose polysemantic LLM activations into an overcomplete sparse basis (Bricken et al., 2023; Templeton et al., 2024). Core architectures: L1 SAE, TopK (Gao et al., 2024), JumpReLU (Rajamanoharan et al., 2024), Gated SAE.
- Standard evaluation via reconstruction loss, L0, and downstream metrics (SAEBench; Karvonen et al., 2025). Proxy metrics do not reliably predict downstream performance.

### 2.2 Feature Absorption: Definition and Prior Measurements
- Definition from Chanin et al. (2024): absorption occurs when a parent feature (e.g., "starts with S") fails to fire on tokens where a more specific child feature (e.g., "September") activates instead, because the SAE's sparsity objective favors the child's single-latent encoding.
- Toy model proof: absorption is loss-optimal when sparsity penalty exceeds sin^2(theta), where theta is the decoder angle between parent and child directions.
- Empirical rates: 15--35% on first-letter spelling across Gemma Scope, Llama 3.2, Qwen2 SAEs.

### 2.3 Mitigation Approaches
- Matryoshka SAE (Bussmann et al., 2025): nested dictionaries reduce absorption but trade it for hedging at inner levels.
- OrtSAE (Korznikov et al., 2025): orthogonality penalty reduces absorption by 65%.
- ATM (Li et al., 2025): adaptive temporal masking achieves absorption score 0.0068 vs. TopK 0.1402.
- Masked regularization (Narayanaswamy et al., 2026): disrupts co-occurrence patterns during training.

### 2.4 The Unresolved Confound Problem
- Chanin et al. report correlations between absorption and downstream quality but do not control for L0. All high-absorption SAEs in their dataset are 1M width with low L0; all low-absorption SAEs are 16k/65k with high L0.
- Chanin & Garriga-Alonso (2025) show incorrect L0 causes feature hedging, a distinct failure mode that overlaps observationally with absorption.
- No prior work applies formal causal inference methods to SAE evaluation.

---

## 3. Method

### 3.1 Confound Resolution via Epidemiological Causal Inference
- **Dataset**: 48 Gemma Scope SAEs from SAEBench with absorption scores, four quality metrics (sparse probing F1, SCR, RAVEL TPP, unlearning), width, layer, and L0.
- **Step 1 -- L0 as covariate**: Partial correlations between absorption and each quality metric controlling for log(width), layer, and log(L0). Go/no-go criterion: at least one metric retains |partial r| > 0.2.
- **Step 2 -- Width-stratified analysis**: Spearman correlations within each width stratum (16k, 65k, 1M) with BCa bootstrap 95% CIs.
- **Step 3 -- Baron-Kenny mediation**: Path model L0 -> Absorption -> Quality with Sobel test and bootstrap CIs on the indirect effect.
- **Step 4 -- Rosenbaum sensitivity analysis**: Propensity score and Mahalanobis matching between high- and low-absorption SAEs; Gamma sensitivity parameter quantifying robustness to hidden confounders.

### 3.2 Cross-Domain Absorption Measurement
- **Model and SAEs**: GPT-2 Small (124M) with gpt2-small-res-jb SAE (24k features) at layers 5, 8, 11.
- **Probe training**: Logistic regression probes for five attribute types: Country binary (US vs. non-US), Language binary (English vs. non-English), Continent (6-class), Country top-10, Language top-10. Probes trained on residual stream at hook_resid_pre to match SAE input.
- **Absorption metric adaptation**: k-sparse probing for feature splits; false-negative identification; dominance-based absorption detection (threshold sweep: selectivity in {2, 3, 5, 10}, dominance in {1.0, 2.0}).
- **Controls**: Shuffled-hierarchy control (randomized city-attribute mappings), random probe direction control.

### 3.3 Absorption Scaling Surface
- **Dataset**: 420 SAEs from SAEBench (Gemma 2 2B) spanning width 2,304--1,048,576 and L0 9.3--8,277.
- **GAM fit**: absorption ~ s(log_width) + s(log_L0) + ti(log_width, log_L0) + layer, testing the tensor interaction term for significance.
- **Phase boundary detection**: Gradient magnitude of the GAM surface; ridge identification for candidate phase transitions.
- **Model comparison**: Linear (R^2), additive GAM, interaction GAM to quantify the interaction contribution.

### 3.4 Taxonomy Correction
- Frequency-matched comparison tokens for letters with n_comparison_tokens = 0 in the original Chanin et al. classification.
- Recomputed Type II absorption classification with corrected baselines.

> **Ref before appearance**: Figure 3 illustrates the cross-domain absorption rates with shuffled controls. Table 1 reports per-metric partial correlations before and after L0 control.

---

## 4. Results

### 4.1 Confound Resolution: Absorption Survives L0 Control (H1)

- **Go/no-go**: 3/4 quality metrics retain |partial r| > 0.2 after adding log(L0) as covariate.
- **Suppression effect**: Sparse probing F1 partial correlation strengthened from r = -0.664 to r = -0.746 (p = 1.2e-9) when L0 is added. L0 was partially masking absorption's true effect -- a classical suppression variable.
- **SCR and TPP**: SCR partial r = -0.570 (p = 6.6e-5); TPP partial r = -0.331 (p = 0.022). Unlearning: r = -0.123 (n.s.).
- **Mediation**: SCR shows full Baron-Kenny mediation (direct effect c' = -0.003, n.s.; indirect effect = 0.025, 95% CI [0.007, 0.048]). TPP shows partial mediation (proportion mediated = 0.54).
- **Rosenbaum sensitivity**: Under Mahalanobis matching (17 pairs), Gamma = 2.65 for TPP and Gamma = 1.85 for sparse probing -- moderate to strong robustness to unmeasured confounders.
- **Within-width caveat**: Within-width matching strategies (median split: 23 pairs; tertile: 16 pairs) detect no significant quality differences, suggesting the absorption-quality association is driven by cross-width variation.

> **Tables**: Table 1 (partial correlations), Table 2 (mediation results), Table 3 (Rosenbaum sensitivity).
> **Figures**: Figure 1 (bar chart comparing partial r with/without L0), Figure 2 (mediation path diagram for SCR).

### 4.2 Cross-Domain Absorption: Metric Limitation Exposed (H2)

- **Probe quality**: Binary probes (Country US, Language English) achieve 86--92% accuracy. Multiclass probes (Continent, Country top-10, Language top-10) range from 59--69%, below the 85% quality gate for some configurations.
- **Dominance-based absorption rates**: 51--85% across all domains and layers, with highest rates at layers 5 and 8. Country binary US at layer 11 shows the lowest rate (11.3%).
- **Critical finding**: Shuffled controls show 100% absorption rate. The dominance-based metric does not discriminate real from shuffled hierarchies, because the dominant features (particularly "super-absorber" Feature 8213) are polysemantic city/location features unrelated to specific probe directions.
- **Cosine-calibrated metric**: 0% absorption across all thresholds, indicating no SAE features align with knowledge-hierarchy probe directions on GPT-2 Small.
- **Interpretation**: The discrepancy between dominance-based (50--85%) and cosine-calibrated (0%) metrics reveals that GPT-2 Small's SAE does not encode knowledge-hierarchy directions as dedicated features. With 98% dead features, the SAE lacks the capacity for fine-grained geographic representations.

> **Figures**: Figure 3 (cross-domain absorption rates with shuffled control line), Figure 4 (layer-wise breakdown).
> **Table**: Table 4 (per-domain, per-layer absorption rates with probe accuracy and FN rates).

### 4.3 Absorption Scaling Surface: Significant Interaction (H3)

- **Sample**: 420 SAEs across 9 release families, 3 layers, widths 2,304--1,048,576, L0 9.3--8,277.
- **Model comparison**: Linear R^2 = 0.488; additive GAM R^2 = 0.620; interaction GAM R^2 = 0.693. The interaction term is highly significant (p = 3.1e-15).
- **Scaling structure**: Absorption increases with width (rho = +0.35 to +0.42 per layer) and decreases with L0 (rho = -0.46 to -0.49 per layer). The interaction means these effects are not independent: the width effect is strongest at low L0, and the L0 effect is strongest at high width.
- **Phase boundary**: Gradient analysis identifies a transition zone at log2(L0) in [2.7, 3.8] (L0 ~ 6.5--14), where absorption rises sharply. At L0 > 14, absorption is low regardless of width. At L0 < 7, absorption increases dramatically from 16k to 1M width.
- **Linear model coefficients**: log_width: +0.054 (wider = more absorption); log_L0: -0.014 (higher L0 = less absorption); layer: +0.003 (deeper = slightly more).

> **Figures**: Figure 5 (2D contour plot of absorption surface), Figure 6 (gradient magnitude surface with phase boundary overlay).
> **Table**: Table 5 (GAM model comparison: R^2, AIC, interaction p-value).

### 4.4 Taxonomy Correction Validates Original Rate (H5)

- Frequency-matched correction applied to 8/26 letters with n_comparison_tokens = 0.
- Zero letters changed classification. Corrected combined rate: 92.3% (identical to original).
- The high Type II rate reflects genuine feature selectivity rather than measurement artifact.

> **Figure**: Figure 7 (taxonomy comparison: original vs. corrected, with per-letter absorption bar chart).

---

## 5. Discussion

### 5.1 The Causal Status of Absorption
- The suppression effect for sparse probing is the most surprising finding: L0 was partially masking absorption's true effect on feature quality. This reverses the prior concern that absorption might be an epiphenomenon of L0.
- Full mediation for SCR establishes that absorption is the primary pathway through which L0 affects spurious correlation removal quality. For TPP, absorption mediates 54% of L0's effect.
- The within-width matching null is an important caveat: the evidence for a causal chain is strongest in cross-width comparisons. Whether absorption causes quality degradation within a fixed width remains unresolved with the current sample size (n = 15--18 per stratum).

### 5.2 Why the Dominance Metric Fails on Knowledge Hierarchies
- The 100% shuffled control rate reveals that the standard absorption metric (Chanin et al., 2024) measures feature dominance concentration, not probe-direction-specific absorption, when applied to knowledge features on small models.
- Root cause: GPT-2 Small's 24k SAE has 98% dead features on city prompts. The surviving 298 features include polysemantic "super-absorbers" (e.g., Feature 8213) that dominate at all probe-miss positions regardless of attribute type.
- This is a methodological contribution: future cross-domain absorption studies must use cosine-calibrated metrics or larger models where knowledge features are linearly separable in SAE decoder space.

### 5.3 Practical Implications of the Scaling Surface
- The phase boundary at L0 ~ 7--14 provides an actionable threshold: practitioners selecting SAEs for interpretability should prefer L0 > 14 to avoid the high-absorption regime.
- The significant width-L0 interaction implies that scaling width alone (the standard strategy for improving SAE quality) increases absorption risk unless L0 is simultaneously increased. This trades off against the findings of Chanin et al. (2025) that too-high L0 leads to feature hedging.
- The three-regime picture (hedging at low width, absorption at high width/low L0, recovery at high width/high L0) provides a conceptual map for SAE hyperparameter selection.

### 5.4 Limitations
- Sample size: 48 SAEs for Phase 1, all from Gemma 2 2B. Cross-architecture validation needed.
- Cross-domain experiment limited to GPT-2 Small due to Gemma 2B HuggingFace access restrictions. Larger models may show different cross-domain absorption patterns.
- Observational design: mediation analysis assumes correctly specified causal ordering (L0 -> Absorption -> Quality). Reverse causation or common causes cannot be ruled out.
- The 420-SAE scaling surface is dominated by standard architecture SAEs (360/420); architecture subgroup analysis has limited power.

---

## 6. Conclusion

- Absorption retains a strong, independent association with SAE quality after controlling for the L0 confound. For sparse probing, L0 was a suppression variable: controlling for it strengthened the absorption-quality link (partial r = -0.746). Mediation analysis establishes absorption as the primary pathway for 2/4 quality metrics.
- The dominance-based absorption metric does not transfer to knowledge hierarchies on GPT-2 Small, exposing a critical methodological limitation. Future work should use larger models with dedicated knowledge features and cosine-calibrated detection.
- The 420-SAE absorption scaling surface reveals a highly significant width-L0 interaction (p = 3.1e-15), with a phase boundary near L0 ~ 7--14. Practitioners should target L0 > 14 and be aware that increasing width at low L0 amplifies absorption.
- Epidemiological causal inference methods (mediation analysis, Rosenbaum sensitivity bounds) provide rigorous tools for establishing causal claims from observational SAE data, and we encourage their adoption in future SAE evaluation studies.

---

## Figure & Table Plan

### Figure 1: Absorption-Quality Partial Correlations Before vs. After L0 Control (Section: Results 4.1)
- **Purpose**: Show that 3/4 quality metrics retain meaningful absorption-quality association after controlling for L0, with the key suppression effect on sparse probing.
- **Type**: bar_chart (grouped bars: without L0 control vs. with L0 control, per metric)
- **Content**: Partial Pearson r for 4 metrics (Sparse Probing F1, SCR, RAVEL TPP, Unlearning) with 95% CIs. Dashed |r|=0.2 threshold line. Arrow annotation for suppression effect.
- **Key takeaway**: L0 is not a confound -- it is a suppression variable. Absorption's link to quality is genuine and, for sparse probing, even stronger than previously measured.
- **Generation**: code (matplotlib)
- **Data source**: `P1_confound_go_nogo.json`
- **Status**: GENERATED (fig1_partial_correlations_l0.png)

### Figure 2: Mediation Path Diagram (L0 -> Absorption -> Quality) (Section: Results 4.1)
- **Purpose**: Visualize the causal mediation model with standardized coefficients, showing full mediation for SCR.
- **Type**: flow_chart (path diagram with a, b, c' coefficients)
- **Content**: Three-box path diagram: log(L0) -> Absorption (a = -0.543***) -> Quality Metric (b = -0.457***). Direct path c' = -0.003 (n.s.). Indirect effect box: 0.025, 95% CI [0.007, 0.048], "Full mediation (SCR)".
- **Key takeaway**: Absorption fully mediates L0's effect on SCR. The direct effect of L0 on quality vanishes once absorption is in the model.
- **Generation**: code (matplotlib)
- **Data source**: `P1_mediation.json`
- **Status**: GENERATED (fig2_mediation_path.png)

### Figure 3: Cross-Domain Absorption Rates with Shuffled Controls (Section: Results 4.2)
- **Purpose**: Show dominance-based absorption rates across knowledge domains alongside the 100% shuffled control, revealing the metric limitation.
- **Type**: bar_chart (grouped bars per domain with shuffled control dashed line)
- **Content**: Mean absorption rate per domain (Country, Language, Continent) across all layers, with error bars (std). Dashed red line at shuffled control rate = 1.0.
- **Key takeaway**: The dominance-based metric shows 50-85% absorption but the shuffled control at 100% demonstrates the metric captures background feature concentration, not genuine probe-direction absorption.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `P2_crossdomain_comparison.json`, `P2_controls.json`
- **Status**: GENERATED (fig3_crossdomain_absorption.png)

### Figure 4: Cross-Domain Absorption Rates by Layer (Section: Results 4.2)
- **Purpose**: Show how absorption rates vary across layers for each probe type, revealing layer 11 as the lowest-absorption layer.
- **Type**: bar_chart (grouped bars per layer, colored by probe type)
- **Content**: Absorption rate per probe type (5 types) at layers 5, 8, 11. Layer 11 shows consistently lower rates, especially for Country binary US (11.3%).
- **Key takeaway**: Absorption rates decrease at later layers, consistent with GPT-2 Small encoding geographic knowledge more explicitly at deeper layers where SAE features better align with the representations.
- **Generation**: code (matplotlib)
- **Data source**: `P2_crossdomain_comparison.json`
- **Status**: GENERATED (fig7_crossdomain_by_layer.png)

### Figure 5: Absorption Scaling Surface Contour Plot (Section: Results 4.3)
- **Purpose**: Visualize the 2D absorption surface in (log2(width), log2(L0)) space, showing the interaction structure and phase regions.
- **Type**: heatmap / contour_plot (2D)
- **Content**: Left panel: raw scatter of 420 SAEs colored by absorption score. Right panel: GAM-predicted absorption contour surface at layer 12 with labeled phase regions (low-absorption, transition, high-absorption).
- **Key takeaway**: Absorption depends on the joint structure of width and L0 -- neither alone predicts absorption rate. The high-absorption regime occupies the high-width, low-L0 corner.
- **Generation**: code (matplotlib)
- **Data source**: `P3_scaling_surface.json`
- **Status**: GENERATED (P3_absorption_contour.png) -- needs upgrade to publication quality for final paper

### Figure 6: Gradient Magnitude Surface with Phase Boundary (Section: Results 4.3)
- **Purpose**: Identify the phase boundary where absorption transitions sharply from low to high.
- **Type**: heatmap with overlay (gradient magnitude + ridge markers)
- **Content**: Gradient magnitude of GAM surface plotted as filled contours. Cyan dots mark the detected phase boundary ridge at log2(L0) in [2.7, 3.8]. White dots show SAE data points.
- **Key takeaway**: A sharp transition zone exists at L0 ~ 7--14 where absorption rises rapidly, consistent with the theoretical prediction of a hedging-to-absorption phase transition.
- **Generation**: code (matplotlib)
- **Data source**: `P3_scaling_surface.json`
- **Status**: GENERATED (P3_gradient_surface.png) -- needs upgrade to publication quality

### Figure 7: Taxonomy Correction Comparison (Section: Results 4.4)
- **Purpose**: Show that frequency-matched correction does not change the original 92.3% combined absorption rate, validating the original taxonomy.
- **Type**: comparison_table (stacked bar left panel) + bar_chart (per-letter right panel)
- **Content**: Left: stacked bar of Type I/II/III/None counts, original vs. corrected. Right: per-letter absorption rate bar chart on GPT-2 Small with literature mean reference line.
- **Key takeaway**: The original taxonomy is robust -- the high Type II rate reflects genuine feature selectivity, not measurement artifact.
- **Generation**: code (matplotlib)
- **Data source**: `P4_taxonomy_correction.json`
- **Status**: GENERATED (fig6_taxonomy_correction.png)

### Figure 8: Rosenbaum Sensitivity Analysis (Section: Results 4.1)
- **Purpose**: Quantify how robust the absorption-quality association is to unmeasured confounders under two matching strategies.
- **Type**: bar_chart (grouped bars per matching strategy and quality metric)
- **Content**: Gamma values for sparse probing F1, SCR, and RAVEL TPP under propensity score matching (6 pairs) and Mahalanobis matching (17 pairs). Dashed lines at Gamma = 1.5 (moderate) and Gamma = 2.0 (strong).
- **Key takeaway**: RAVEL TPP achieves Gamma = 2.65 under Mahalanobis matching -- the result can withstand a 2.65:1 hidden confounder odds ratio.
- **Generation**: code (matplotlib)
- **Data source**: `P1_rosenbaum.json`
- **Status**: GENERATED (fig5_rosenbaum_sensitivity.png)

### Table 1: Partial Correlations Before and After L0 Control (Section: Results 4.1)
- **Purpose**: Core result table for confound resolution.
- **Content**: Rows: 4 quality metrics. Columns: bivariate r, partial r (no L0), partial r (with L0), delta, p-value (with L0). Bold the strongest effect (sparse probing r = -0.746).
- **Data source**: `P1_confound_go_nogo.json`

### Table 2: Mediation Analysis Results (Section: Results 4.1)
- **Purpose**: Report mediation type, indirect effect, Sobel p, and proportion mediated for each quality metric.
- **Content**: Rows: 4 quality metrics. Columns: mediation type (full/partial/none), indirect effect, Sobel z, Sobel p, proportion mediated.
- **Data source**: `P1_mediation.json`

### Table 3: Rosenbaum Sensitivity Analysis by Matching Strategy (Section: Results 4.1)
- **Purpose**: Compare five matching strategies (exact width, within-width median, propensity, Mahalanobis, tertile) on n pairs, significant metrics, and max Gamma.
- **Content**: Rows: 5 strategies. Columns: n pairs, sparse probing p, TPP p, max Gamma.
- **Data source**: `P1_rosenbaum.json`

### Table 4: Cross-Domain Absorption Rates (Section: Results 4.2)
- **Purpose**: Report per-domain, per-layer absorption rates with probe accuracy and false-negative rates.
- **Content**: Rows: 5 probe types x 3 layers = 15 rows. Columns: probe type, layer, probe accuracy, n split features, FN rate, absorption rate (dom>=1.0), absorption rate (dom>=2.0).
- **Data source**: `P2_crossdomain_comparison.json`

### Table 5: Scaling Surface Model Comparison (Section: Results 4.3)
- **Purpose**: Compare linear, additive GAM, and interaction GAM on R^2, AIC, and interaction significance.
- **Content**: 3 rows (models) x 3 columns (R^2, AIC, interaction p-value). Bold interaction GAM.
- **Data source**: `P3_scaling_surface.json`

### Table 6: Bradford Hill Criteria Assessment (Section: Discussion 5.1)
- **Purpose**: Systematic evaluation of the absorption-quality causal claim using Bradford Hill epidemiological criteria.
- **Content**: 9 rows (criteria) x 3 columns (criterion, assessment [strong/moderate/weak], key evidence). Summary: 3 strong, 5 moderate, 0 weak.
- **Data source**: `P1_synthesis.json`

---

## Transition Logic

1. **Introduction -> Background**: The introduction previews three contributions; the background provides the technical context for each (absorption definition, confound problem, single-task limitation).

2. **Background -> Method**: Section 2.4 ends by noting that no prior work applies causal inference methods to SAE evaluation. Section 3.1 directly addresses this gap with epidemiological methods.

3. **Method -> Results**: The method section establishes the go/no-go criterion (|partial r| > 0.2) and pre-registered thresholds. Results open by reporting the go/no-go test outcome and proceeding through each hypothesis in order.

4. **Results 4.1 -> 4.2**: "Having established that absorption is a genuine quality predictor on the first-letter task, we now test whether the phenomenon extends to knowledge hierarchies."

5. **Results 4.2 -> 4.3**: "The cross-domain experiment reveals a metric limitation rather than a domain limitation. We turn to the 420-SAE scaling surface for the most statistically powerful analysis."

6. **Results 4.3 -> 4.4**: "Before discussing implications, we complete the audit by validating the original taxonomy rate."

7. **Results -> Discussion**: Results present findings neutrally; Discussion interprets the suppression effect (5.1), explains the metric failure mechanism (5.2), and derives practical guidelines (5.3).

8. **Discussion -> Conclusion**: Discussion acknowledges limitations; Conclusion distills actionable takeaways for the SAE practitioner and identifies the most pressing follow-up (cross-domain with larger models, within-width causal evidence).
