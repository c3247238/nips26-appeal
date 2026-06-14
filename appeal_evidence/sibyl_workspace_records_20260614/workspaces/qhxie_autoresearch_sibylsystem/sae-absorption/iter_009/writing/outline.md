# Paper Outline: Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders

## Title

**Cross-Domain Characterization of Feature Absorption in Sparse Autoencoders**

Alternative: *The Absorption Tax: Hierarchy-Dependent Feature Failure in SAEs*

---

## Section 1: Introduction (1.5 pages)

### Key arguments
- Feature absorption -- the systematic failure of parent features to fire when child features are active -- threatens SAE-based mechanistic interpretability. Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10--40%.
- All published absorption measurements use a single proxy: first-letter spelling in GPT-2 Small (Chanin et al., 2024). The field's understanding rests on this narrow empirical base.
- We present the first systematic cross-domain absorption characterization, extending measurement from spelling to entity-attribute hierarchies (city-country, city-continent, city-language) on Gemma 2 2B with Gemma Scope SAEs.

### Evidence highlights for intro
- Absorption rates vary 4x across hierarchies at L24 (11.6% city-language to 45.1% city-country, Kruskal-Wallis p=7.4e-66).
- Activation patching causally confirms competitive exclusion for first-letter (32.5% recovery vs 1.5% control, p=0.000218, d=1.33), but the mechanism does not transfer cross-domain.
- Absorption is 100% pathological: mean |logit change| = 3.98, ~1000x above control (0.004).

### Transition to Section 2
Introduce the gap: no formal definition covers multiple hierarchy types; prior work is first-letter only.

### Visual element
- **Figure 1 (teaser)**: A panel figure showing (left) the absorption measurement concept (parent probe correct on raw activations, wrong on SAE activations while child feature fires) and (right) a bar chart of absorption rates across the four hierarchies at L24_16k with 95% CI error bars. Purpose: immediately convey the core finding and the measurement framework.

---

## Section 2: Background and Related Work (1.5 pages)

### Key arguments
- Sparse autoencoders decompose neural activations into interpretable features (Cunningham et al., 2023; Bricken et al., 2023). Monosemanticity depends on features firing reliably for their semantic concepts.
- Feature absorption (Chanin et al., 2024): when a parent feature (e.g., "starts with S") fails to fire because a child feature (e.g., "the word Saturday") captures the input. Proven in a two-layer toy model; measured at 15--35% on first-letter spelling.
- SAE architectures: JumpReLU (Rajamanoharan et al., 2024a), BatchTopK (Rajamanoharan et al., 2024b), Matryoshka (Bussmann et al., 2024). Prior architecture comparisons for absorption are first-letter only.
- RAVEL dataset (Huang et al., 2024): entity-attribute evaluation framework with validated probes for city properties (country, continent, language). Provides natural feature hierarchies.
- SAEBench (Karvonen et al., 2025) and Gemma Scope (Lieberum et al., 2024) provide standardized SAEs for reproducible evaluation.

### Transition to Section 3
Current methods measure absorption on one task type; extending to entity-attribute hierarchies requires adapted probes and measurement pipelines.

---

## Section 3: Methodology (2 pages)

### 3.1 Feature hierarchies
- **First-letter spelling**: 26 letters, ~500 words, binary probes, 100% parent-child co-occurrence. Positive control.
- **City-continent**: 6 classes, ~1567 entities from RAVEL, moderate class balance.
- **City-language**: 23 classes, ~1229 entities, intermediate balance.
- **City-country**: 80 classes, ~1405 entities, highly imbalanced (USA: 176, Chad: 5).

### 3.2 Probe training and quality gates
- Logistic regression probes at layers 6, 12, 18, 24 on Gemma 2 2B residual stream activations.
- Quality gate: F1 > 0.90 (strict) or >= 0.80 (relaxed with documented caveat).
- Achieved: first-letter F1 = 0.97 at L24 (strict pass); city-continent F1 = 0.87 (relaxed pass); city-language F1 = 0.82 (relaxed pass); city-country F1 = 0.73 (below gate, included with caveat).

### 3.3 Absorption measurement pipeline
- Adapted from Chanin et al. (2024): encode activations through SAE, apply probe to raw and SAE-reconstructed activations, classify false negatives (probe correct on raw, wrong on SAE) as absorption.
- Integrated-gradients attribution identifies parent-child pairs.
- Controls: random direction baseline, shuffled hierarchy control.

### 3.4 Activation patching protocol
- Zero identified child feature in SAE latent space; propagate through remaining layers; measure parent probe recovery.
- Control: zero random non-child feature matched by activation magnitude.
- Statistics: Wilcoxon signed-rank test, bootstrap CI, Cohen's d.

### 3.5 Benign vs. pathological diagnostic
- Ablate parent probe direction from child feature's decoder vector; measure downstream logit change for parent-relevant tokens.
- Benign: |logit change| <= 0.1; Pathological: |logit change| > 0.1. Multiple thresholds (0.05, 0.1, 0.2) for robustness.

### 3.6 Hedging decomposition
- Three-way classification of false negatives: strict absorbed (main parent feature absent), compensatory (main parent feature fires but probe wrong), persistent (residual).
- First-letter multi-L0 analysis (L0=22 to L0=176) from iter_008 for tightened hedging.

### Visual elements
- **Table 1**: Probe quality (F1) across 4 hierarchies x 4 layers, with gate status indicated. Purpose: establish reliability bounds on all downstream results.

### Transition to Section 4
With validated probes (and documented limitations), we measure absorption across hierarchies.

---

## Section 4: Cross-Domain Absorption Results (2 pages)

### Key arguments
- **H1 confirmed**: Absorption rates differ significantly across hierarchies (Kruskal-Wallis p=7.4e-66 at L24_16k). Rates: city-country 45.1%, first-letter 42.9%, city-continent 31.4%, city-language 11.6%.
- **H2' refuted**: No simple semantic-vs-syntactic ordering. City-country exceeds first-letter; city-language is far below it. Absorption is hierarchy-specific, not category-specific.
- **Layer dependence is dramatic**: First-letter absorption ranges from 2.4% (L6) to 42.9% (L24) -- a 15x increase. Absorption concentrates at the final prediction layer.
- **Width effect**: Wider SAEs (65k vs 16k) generally reduce absorption (city-country: 45.1% at 16k vs 32.9% at 65k).

### Evidence
- Pairwise permutation tests: city-language vs first-letter significant after Bonferroni correction (p=0.003); city-continent and city-country not significantly different from first-letter at L24_16k.
- Per-class variation is large: within city-country, USA shows 0% absorption while Albania, Algeria, Argentina show 100%. Within city-continent, Europe shows 90.2% while Africa and South America show ~4%.

### Visual elements
- **Figure 2**: Layer-dependent absorption profile. A line plot with 4 lines (one per hierarchy) across layers 6, 12, 18, 24 for the 16k SAE. X-axis: layer. Y-axis: absorption rate. Shows the dramatic concentration at L24 for first-letter and the cross-domain variation pattern.
  - **Purpose**: Communicate the novel layer-dependence finding and cross-domain variation simultaneously.
  - **Type**: line_plot
  - **Data source**: consolidation_summary.json absorption_rate_table + phase1_absorption_crossdomain.json
  - **Key takeaway**: Absorption concentrates at the final prediction layer, with magnitude varying by hierarchy.
  - **Generation**: matplotlib

- **Table 2**: Cross-domain absorption rates at L24 (4 hierarchies x 2 widths). Columns: hierarchy, SAE config, absorption rate, 95% CI, probe F1, N entities, N false negatives.
  - **Purpose**: The paper's central results table.
  - **Type**: data_table
  - **Data source**: phase1_absorption_crossdomain.json + phase1_absorption_firstletter.json
  - **Key takeaway**: Rates span 7.7% to 45.1% across hierarchy-SAE combinations.

- **Figure 3**: Per-class absorption heatmap for city-continent (6 continents x 2 SAE widths). Color intensity = absorption rate. Annotated with n per cell.
  - **Purpose**: Show within-hierarchy variance (Europe 90% vs Africa 4%).
  - **Type**: heatmap
  - **Data source**: phase1_absorption_crossdomain_summary.md per-class tables
  - **Key takeaway**: Absorption is concentrated in specific subclasses, not uniform across a hierarchy.
  - **Generation**: matplotlib/seaborn

### Transition to Section 5
Cross-domain variation is established; now we investigate the causal mechanism and whether absorption represents harmful information loss.

---

## Section 5: Mechanism Analysis (2 pages)

### 5.1 Causal confirmation via activation patching (first-letter)
- 25 words, 200 contexts each. Child-zeroed recovery: 32.5% vs control 1.5%. Wilcoxon p=0.000218, Cohen's d=1.33 (large effect).
- 16 of 19 words with absorption show positive recovery. This is the first interventional (not correlational) evidence for competitive exclusion in SAEs.

### 5.2 Cross-domain patching fails (city-continent)
- 93 entities, 50 contexts each, 3751 FN instances. Child-zeroed recovery: 0.05% vs control 14.5%. Effect is reversed (d=-0.91).
- Interpretation: continent information is distributed across multiple features, not concentrated in a single child. This suggests two distinct absorption mechanisms -- concentrated (first-letter) vs distributed (semantic hierarchies).

### 5.3 Absorption is always pathological (H8 falsified)
- 50 entities, 1471 FN instances. 0% benign at all thresholds (0.05, 0.1, 0.2).
- Mean |logit change| from parent direction ablation: 3.98, ~1000x above control (0.004). t=-365.3, p<10^{-100}.
- The contrarian hypothesis that absorption might faithfully reflect computational redundancy is decisively falsified: absorption always degrades model output.

### 5.4 Hedging decomposition
- Compensatory hedging dominates across all hierarchies at L24 (77-100% of FNs). The main parent feature fires in most cases, but the SAE reconstruction distorts the representation enough to change the probe's prediction.
- Strict absorbed fraction varies: 0% for first-letter, 6.2% for city-continent, 22.6% for city-language, 3.7% for city-country. Chi-square p<0.0001.
- Multi-L0 first-letter analysis: strict hedging 7.9%, compensatory 86.2%, persistent 5.9%. The widely cited 98.6% loose hedging figure is near-tautological.

### Visual elements
- **Figure 4**: Activation patching results -- paired box plots showing child-zeroed vs control recovery rates for first-letter (left panel, positive result) and city-continent (right panel, negative result). Each point = one word/entity. Horizontal line at y=0.
  - **Purpose**: Contrast the strong causal evidence for first-letter with the failure to generalize cross-domain.
  - **Type**: box_plot (paired)
  - **Data source**: iter_008 activation patching (first-letter), phase2_activation_patching_crossdomain.json (city-continent)
  - **Key takeaway**: Competitive exclusion confirmed for first-letter but not semantic hierarchies.
  - **Generation**: matplotlib

- **Figure 5**: Logit change distribution histogram for benign/pathological diagnostic. Main panel: distribution of |logit change| when parent direction is ablated (mean=3.98). Inset: control distribution (mean=0.004). Vertical dashed lines at thresholds (0.05, 0.1, 0.2). All 1471 instances exceed all thresholds.
  - **Purpose**: Visually demonstrate the 1000x gap between parent ablation impact and control.
  - **Type**: histogram with inset
  - **Data source**: phase2/benign_pathological.json
  - **Key takeaway**: Absorption is 100% pathological -- never benign.
  - **Generation**: matplotlib

- **Table 3**: Hedging decomposition across hierarchies at L24. Columns: hierarchy, N FN, strict absorbed %, compensatory %, persistent %, chi-square test result.
  - **Purpose**: Show that compensatory FNs dominate and the pattern varies by hierarchy.
  - **Type**: data_table
  - **Data source**: phase1/hedging_crossdomain.json

### Transition to Section 6
With the mechanism characterized, we test whether architecture choice can mitigate absorption.

---

## Section 6: Architecture Analysis (1 page)

### Key arguments
- **H6 partially supported**: No significant architecture effect on absorption (ANOVA p=0.53 at L12, p=0.50 at L24). Hierarchy type is the dominant factor (p=0.005 at L12, p=0.041 at L24).
- Tested architectures: JumpReLU 16k, JumpReLU 65k, BatchTopK 16k, Matryoshka 32k.
- This is a "negative-as-positive" finding: absorption is a fundamental phenomenon of sparse decomposition, not an architecture-specific artifact. Switching architectures will not solve the absorption problem.

### Caveats
- RAVEL probes below strict gate at L12 (where all 4 architectures are compared).
- Width mismatch: Matryoshka 32k vs others 16k/65k.
- Only 2 layers tested across all architectures.

### Visual elements
- **Figure 6**: Grouped bar chart -- absorption rate by architecture (x-axis: 4 architectures, grouped by hierarchy). Error bars from bootstrap CI. Horizontal reference line at mean. ANOVA p-values annotated.
  - **Purpose**: Show that bars within each hierarchy cluster overlap, while clusters differ.
  - **Type**: bar_chart (grouped)
  - **Data source**: phase1/architecture_comparison.json
  - **Key takeaway**: Hierarchy type, not architecture, determines absorption rate.
  - **Generation**: matplotlib

### Transition to Section 7
Architecture provides no solution; we discuss what these findings mean for SAE-based interpretability.

---

## Section 7: Discussion (1.5 pages)

### Key arguments
- **Correlational vs causal methods**: Five correlational/statistical approaches failed (GAS rho=0.12, CMI rho=0.04, Absorption Tax ranking rho=-0.20, rate-distortion model R^2=0.088, competition coefficients non-significant). Only activation patching succeeds -- and only for first-letter. This motivates a shift from correlational to causal methods in SAE analysis.
- **Concentrated vs distributed absorption**: First-letter shows single-feature competitive exclusion (child suppresses parent). Semantic hierarchies may involve multi-feature distributed absorption where no single child feature captures the parent information. The mechanism depends on the task's computational structure.
- **Implications for SAE reliability**: Absorption is always pathological (1000x logit change ratio). For safety applications, where feature reliability is critical, absorption rates of 11--45% at the final prediction layer are concerning. Current probing-based detection requires supervised labels, limiting scalability.
- **Probe quality limitation**: RAVEL probes achieve F1=0.73--0.87, meaning RAVEL absorption rates are upper bounds. The first-letter task (F1=0.97) serves as the gold-standard anchor. Better probe training (e.g., with richer prompt templates, larger entity sets) could sharpen cross-domain estimates.
- **Layer dependence and its implications**: Absorption concentrating at L24 (the final prediction layer, 15x higher than earlier layers) suggests it arises from the model's task-specific computation, not from generic feature representation. This connects absorption to the model's use of features for downstream prediction.

### Visual element
- **Table 4**: Summary of all 9 hypotheses with verdict, key metric, confidence, and paper section. Three columns of hypothesis status: SUPPORTED (green), REFUTED/FALSIFIED (red), NOT_SUPPORTED (gray).
  - **Purpose**: Give readers a single-glance summary of which hypotheses survived testing.
  - **Type**: summary_table
  - **Data source**: consolidation_summary.json hypothesis_verdicts
  - **Key takeaway**: 2/9 hypotheses fully supported, 2 falsified, 5 not supported -- honest reporting.
  - **Generation**: LaTeX table

---

## Section 8: Conclusion (0.5 pages)

### Key contributions (enumerated)
1. **First cross-domain absorption characterization**: Absorption rates vary 4x across hierarchies (11.6--45.1% at L24, Kruskal-Wallis p=7.4e-66). No simple semantic/syntactic ordering.
2. **Causal evidence for competitive exclusion**: Activation patching confirms the mechanism for first-letter (32.5% recovery vs 1.5%, p=0.000218) but reveals it does not generalize to semantic hierarchies.
3. **Absorption is always pathological**: 0% benign at all thresholds; mean |logit change|=3.98, ~1000x above control. The hypothesis that absorption might be benign computational redundancy is falsified.
4. **Hierarchy dominates architecture**: No significant architecture effect (p=0.50--0.53); hierarchy type is the sole significant predictor (p=0.005--0.041).
5. **Comprehensive negative results**: 5 correlational approaches fail; honest reporting motivates a shift to causal methods.

### Limitations
- Single model (Gemma 2 2B). Generalization to other architectures/scales untested.
- RAVEL probes below strict quality gate; absorption rates are upper bounds.
- Cross-domain activation patching negative result limits causal claims to first-letter.
- Architecture comparison limited by width mismatch and probe quality at L12.

### Future work
- Multi-feature distributed absorption detection methods.
- Better probes for entity-attribute hierarchies (contrastive learning, richer templates).
- Extension to larger models (Gemma 2 9B, Llama 3) and more hierarchy types.
- Unsupervised absorption detection that accounts for encoder dynamics, not just decoder geometry.

---

## Appendices

### Appendix A: Threshold Sensitivity Analysis
- 5x4 grid (cosine threshold x magnitude gap). CV=0.077. FN count constant (87/576) across all 20 cells.
- Conclusion: absorption is structural, not threshold-dependent.
- **Table A1**: Threshold sensitivity heatmap.

### Appendix B: GAS Detector Failure (NR1)
- rho=0.116, AUROC=0.571, bootstrap CI includes zero. 25x scale-up from pilot confirmed signal absent.
- Failure analysis: GAS measures decoder geometry; absorption is driven by encoder dynamics.
- **Figure A1**: GAS vs absorption scatter plot with regression line and CI band.

### Appendix C: CMI Null Result (NR2)
- rho=0.044, p=0.83 at L0=22 (where all probes F1=1.0, eliminating probe quality confound).
- **Table A2**: CMI correlation results with bootstrap CI.

### Appendix D: Absorption Tax Quantitative Failure (NR3)
- T(G) ranking rho=-0.20, concordance 50% (chance).
- T(G) values by hierarchy and layer-SAE configuration.
- **Table A3**: T(G) values for all hierarchy-SAE combinations.

### Appendix E: Rate-Distortion Predictors (NR4)
- Model rho=0.250 (significant at n=262 but below 0.3 target), R^2=0.088.
- Individual predictors in OPPOSITE direction to hypothesis (co_occur rho=-0.17, r_parent rho=-0.20).
- **Figure A2**: Three-panel scatter of absorption vs each predictor, with per-hierarchy coloring.
- **Table A4**: Full predictor correlation table with bootstrap CIs.

### Appendix F: Cross-Domain Activation Patching Details (NR5)
- Per-entity recovery rates for city-continent (93 entities).
- Per-class breakdown: Europe (83 entities, 0% recovery), Asia (4 entities, 0% recovery), etc.
- **Table A5**: Full per-class patching results.

### Appendix G: Probe Training Details
- Training protocol, hyperparameters, per-layer F1 for all 16 hierarchy-layer combinations.
- **Table A6**: Complete probe quality matrix (4 hierarchies x 4 layers).

---

## Figure & Table Plan

### Figure 1: Teaser -- Absorption Concept and Cross-Domain Rates (Section: Introduction)
- **Purpose**: Establish the measurement framework and present the paper's headline result in a single figure.
- **Type**: panel_figure (left: schematic diagram, right: bar_chart)
- **Content**: (Left) Schematic of absorption measurement: raw activation -> probe correct; SAE activation -> probe wrong + child feature fires. (Right) Grouped bar chart of absorption rates at L24_16k for 4 hierarchies with 95% CI.
- **Key takeaway**: Absorption exists across semantic domains at substantially different rates.
- **Generation**: tikz (schematic) + matplotlib (bar chart), composed in LaTeX
- **Data source**: consolidation_summary.json absorption_rate_table

### Figure 2: Layer-Dependent Absorption Profile (Section: Results)
- **Purpose**: Show that absorption concentrates at the final prediction layer and varies by hierarchy.
- **Type**: line_plot
- **Content**: 4 lines (first-letter, city-continent, city-language, city-country) across layers 6, 12, 18, 24 for 16k SAEs. Shaded 95% CI bands where available.
- **Key takeaway**: 15x increase from L6 to L24 for first-letter; cross-domain variation at every layer.
- **Generation**: matplotlib
- **Data source**: phase1 absorption results across all layers

### Figure 3: Per-Class Absorption Heatmap (Section: Results)
- **Purpose**: Reveal within-hierarchy variance (Europe 90% vs Africa 4% in city-continent).
- **Type**: heatmap
- **Content**: 6 continent classes x 2 SAE widths. Cell color = absorption rate, annotated with n.
- **Key takeaway**: Absorption is concentrated in specific subclasses, not uniform.
- **Generation**: seaborn heatmap
- **Data source**: phase1_absorption_crossdomain_summary.md per-class tables

### Figure 4: Activation Patching -- First-Letter vs Cross-Domain (Section: Mechanism)
- **Purpose**: Contrast the strong causal evidence for first-letter with the cross-domain failure.
- **Type**: paired_box_plot (two panels)
- **Content**: (Left) First-letter: 19 words with absorption, child-zeroed vs control recovery. (Right) City-continent: 93 entities, child-zeroed vs control. Horizontal line at 0.
- **Key takeaway**: Competitive exclusion is confirmed for first-letter (d=1.33) but absent for semantic hierarchies (d=-0.91).
- **Generation**: matplotlib
- **Data source**: iter_008 activation patching + phase2_activation_patching_crossdomain.json

### Figure 5: Pathological Absorption -- Logit Change Distribution (Section: Mechanism)
- **Purpose**: Demonstrate the 1000x gap between parent direction ablation and random control.
- **Type**: histogram with inset
- **Content**: Main: distribution of |logit change| for parent ablation (n=1471, mean=3.98). Inset: control distribution (mean=0.004). Vertical lines at 0.05, 0.1, 0.2 thresholds.
- **Key takeaway**: 100% of absorption instances are pathological -- the benign hypothesis is falsified.
- **Generation**: matplotlib
- **Data source**: phase2/benign_pathological.json

### Figure 6: Architecture Comparison (Section: Architecture)
- **Purpose**: Show that hierarchy type dominates architecture choice for absorption rates.
- **Type**: grouped_bar_chart
- **Content**: 4 architecture groups, each with 4 bars (one per hierarchy). ANOVA p-values annotated.
- **Key takeaway**: Bars within each hierarchy cluster overlap; clusters differ (hierarchy >> architecture).
- **Generation**: matplotlib
- **Data source**: phase1/architecture_comparison.json

### Figure A1: GAS vs Absorption Scatter (Appendix B)
- **Purpose**: Document the GAS detector failure.
- **Type**: scatter_plot with regression line
- **Content**: 25 points (one per letter), x=GAS score, y=absorption rate. CI band around regression.
- **Key takeaway**: rho=0.116, no meaningful correlation.
- **Generation**: matplotlib
- **Data source**: iter_008/exp/results/phase2/gas_full.json

### Figure A2: Rate-Distortion Predictor Scatter (Appendix E)
- **Purpose**: Show that individual predictors are weakly or oppositely correlated with absorption.
- **Type**: three-panel scatter (cos_sim, co_occur, r_parent)
- **Content**: 262 points colored by hierarchy. Regression lines with CI.
- **Key takeaway**: Best predictor rho=-0.20 (wrong direction); model R^2=0.088.
- **Generation**: matplotlib
- **Data source**: phase3/rate_distortion_predictors.json

### Table 1: Probe Quality (Section: Methodology)
- **Purpose**: Establish measurement reliability for all downstream results.
- **Content**: 4 hierarchies x 4 layers, F1 scores, gate status (strict/relaxed/below).
- **Key takeaway**: Only first-letter at L24 passes strict gate; RAVEL probes require caveats.
- **Data source**: phase1/probe_training.json

### Table 2: Cross-Domain Absorption Rates (Section: Results)
- **Purpose**: Central results table.
- **Content**: 4 hierarchies x 2 widths at L24, absorption rate, 95% CI, N, N FN, probe F1.
- **Key takeaway**: 11.6--45.1% absorption rates, statistically significant variation (p=7.4e-66).
- **Data source**: consolidation_summary.json + phase1_absorption_crossdomain.json

### Table 3: Hedging Decomposition (Section: Mechanism)
- **Content**: 4 hierarchies x 2 widths at L24, strict %, compensatory %, persistent %, chi-square.
- **Key takeaway**: Compensatory FNs dominate (77-100%); pattern varies by hierarchy.
- **Data source**: phase1/hedging_crossdomain.json

### Table 4: Hypothesis Verdict Summary (Section: Discussion)
- **Content**: 9 hypotheses, verdict, key metric, confidence, section reference.
- **Key takeaway**: 2 supported, 2 falsified, 5 not supported -- honest, comprehensive reporting.
- **Data source**: consolidation_summary.json hypothesis_verdicts

### Table A1: Threshold Sensitivity Heatmap (Appendix A)
- **Content**: 4 magnitude gaps x 5 cosine thresholds, absorption rates.
- **Data source**: phase0/threshold_sensitivity_report.json

### Table A2: CMI Correlation (Appendix C)
- **Content**: rho, p, CI, conditions (L0=22, all probes F1=1.0).
- **Data source**: iter_008/exp/results/phase0/cmi_l0_22.json

### Table A3: Absorption Tax T(G) Values (Appendix D)
- **Content**: T(G) by hierarchy and SAE configuration.
- **Data source**: phase3/absorption_tax.json

### Table A4: Rate-Distortion Predictor Correlations (Appendix E)
- **Content**: Each predictor, rho, p, bootstrap CI, per-hierarchy rho.
- **Data source**: phase3/rate_distortion_predictors.json

### Table A5: Cross-Domain Activation Patching Per-Class (Appendix F)
- **Content**: Per-class entity counts, FN instances, child recovery, control recovery.
- **Data source**: phase2_activation_patching_crossdomain.json

### Table A6: Complete Probe Quality Matrix (Appendix G)
- **Content**: 4 hierarchies x 4 layers, F1, accuracy, N classes, N samples.
- **Data source**: phase1/probe_training.json

---

## Narrative Flow Summary

1. **Introduction**: Absorption threatens SAE reliability; all evidence is from one task. We extend to 4 hierarchies and find 4x variation.
2. **Background**: SAEs, absorption definition, architectures, RAVEL, prior work.
3. **Methodology**: Probes, measurement pipeline, patching protocol, benign/pathological diagnostic, hedging decomposition. Honest about probe quality limits.
4. **Cross-Domain Results**: H1 confirmed (variation is real, p=7.4e-66). H2' refuted (no semantic > syntactic ordering). Layer dependence is dramatic (15x). Width helps. Per-class variance is large.
5. **Mechanism**: Patching confirms causality for first-letter (p=0.000218) but fails cross-domain (concentrated vs distributed absorption). Absorption is 100% pathological (1000x effect). Hedging decomposition shows compensatory dominance.
6. **Architecture**: Hierarchy >> architecture (p=0.50 vs p=0.005). Architecture choice does not solve absorption.
7. **Discussion**: Correlational methods fail (5/5); causal methods succeed (partially). Implications for safety. Probe quality as fundamental limitation.
8. **Conclusion**: Five contributions, honest limitations, future directions.
