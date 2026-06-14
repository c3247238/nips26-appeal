# Paper Outline: The Absorption Tax -- How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains

## Title

**The Absorption Tax: How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains**

---

## Section 1: Introduction (1.5 pages)

### Key arguments
- Feature absorption -- the systematic failure of parent SAE features to fire when child features are active -- threatens SAE-based mechanistic interpretability. Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10--40%.
- All published absorption measurements use a single proxy: first-letter spelling (Chanin et al., 2024). The field's understanding rests on this narrow empirical base -- yet safety-relevant features live in knowledge/reasoning space, not spelling space.
- We present the first systematic cross-domain absorption characterization, extending measurement from spelling to entity-attribute hierarchies (city-country, city-continent, city-language) using RAVEL on Gemma 2 2B with Gemma Scope SAEs.

### Evidence highlights for intro
- Absorption rates show a 4.1x descriptive range across hierarchies at L24 (11.6% city-language to 45.1% city-country). Within-RAVEL variation is statistically significant (Kruskal-Wallis p=7.4e-66).
- A probe degradation ablation (R^2=0.777, perfect monotonic rho=-1.0) decomposes this variation: city-continent absorption is fully explained by probe quality (delta=+0.6pp from curve), while city-language is a genuine hierarchy-specific outlier (delta=-21.3pp below curve).
- Activation patching causally confirms competitive exclusion across ALL hierarchy types: first-letter d=1.33, city-continent d=1.50, city-language d=0.75 (all p<1e-3). This is the first interventional evidence for a universal absorption mechanism.

### Three contributions stated
1. First cross-domain absorption characterization with quantitative probe quality confound decomposition (Section 4).
2. Universal causal mechanism via activation patching confirmed across all tested hierarchy types (Section 5).
3. Decoder information entanglement analysis with acknowledged circularity limitation (Section 5.3).

### Transition to Section 2
Introduce the gap: no formal characterization covers multiple hierarchy types; prior work is first-letter only.

### Visual element
- **Figure 1 (teaser)**: Two-panel figure. (Left) Schematic of the absorption measurement: raw activations pass parent probe correctly, SAE-reconstructed activations fail while child feature fires. (Right) Bar chart of absorption rates across 4 hierarchies at L24_16k with bootstrap 95% CI error bars, RAVEL points overlaid on the probe degradation curve. Purpose: immediately convey the core finding and the measurement framework.

---

## Section 2: Background and Related Work (1.5 pages)

### Key arguments
- Sparse autoencoders decompose neural activations into interpretable features (Cunningham et al., 2023; Bricken et al., 2023). Monosemanticity depends on features firing reliably for their semantic concepts.
- Feature absorption (Chanin et al., 2024): when a parent feature fails to fire because a child feature captures the input. Proven in a two-layer toy model; measured at 15--35% on first-letter spelling in GPT-2 Small.
- SAE architectures: JumpReLU (Rajamanoharan et al., 2024a), BatchTopK (Rajamanoharan et al., 2024b), Matryoshka (Bussmann et al., 2024). Prior architecture comparisons for absorption are first-letter only.
- RAVEL (Huang et al., 2024): entity-attribute evaluation framework with validated probes for city properties (country, continent, language). Provides natural feature hierarchies with known ground truth.
- Gemma Scope (Lieberum et al., 2024) and SAEBench (Karvonen et al., 2025) provide standardized SAEs for reproducible evaluation.
- Anthropic's circuit tracing in Claude 3.5 Haiku demonstrates that when features are reliable, they enable powerful mechanistic understanding -- making absorption a critical obstacle to broader adoption.

### Transition to Section 3
Current methods measure absorption on one task type; extending to entity-attribute hierarchies requires adapted probes and measurement pipelines.

---

## Section 3: Methodology (2 pages)

### 3.1 Feature hierarchies
- **First-letter spelling**: 26 letters, ~2,345 test words, binary probes, 100% parent-child co-occurrence by construction. Serves as positive control with near-perfect probes (F1=1.0).
- **City-continent**: 6 continent classes, ~173 entities from RAVEL, moderate class balance. Probe F1=0.87 at L24.
- **City-language**: ~20 language classes, ~201 entities from RAVEL, intermediate balance. Probe F1=0.82 at L24.
- **City-country**: ~80 country classes, ~1,405 entities, highly imbalanced (USA: 176, Chad: 5). Probe F1=0.73 at L24 (below strict gate; included with documented caveat).

### 3.2 Probe training and quality gates
- Logistic regression probes at layers 6, 12, 18, 24 on Gemma 2 2B residual stream activations.
- Quality gate: F1 > 0.90 (strict) or >= 0.80 (relaxed with documented caveat).
- Per-token aggregation method documented (each token counted independently, not averaged per word).
- Token position asymmetry noted as limitation: first-letter at position -6, RAVEL at position -2.

### 3.3 Absorption measurement pipeline
- Adapted from Chanin et al. (2024): encode activations through SAE, apply probe to raw and SAE-reconstructed activations, classify false negatives (probe correct on raw, wrong on SAE) as absorption candidates.
- Contribution-based child feature identification via integrated-gradients attribution.
- Controls: random direction baseline, shuffled hierarchy control.
- Threshold for parent-child cosine similarity: 0.30 (validated robust across 0.20--0.40 range; Appendix E).

### 3.4 Activation patching protocol
- Zero identified child feature in SAE latent space; propagate through remaining layers; measure parent probe recovery.
- Control: zero random non-child feature matched by activation magnitude.
- Statistics: Wilcoxon signed-rank test, bootstrap 95% CI (10,000 resamples), Cohen's d.

### 3.5 Decoder information entanglement diagnostic
- Ablate parent probe direction from child feature's decoder vector; measure downstream logit change for parent-relevant tokens.
- Thresholds: 0.05, 0.1, 0.2 (multiple thresholds for robustness).
- Control: random direction of same norm.
- **Circularity caveat**: This diagnostic shares the probe direction with the FN classification. It measures decoder geometry, not computational redundancy. What a genuine test would require: activation-level ablation (z_parent=0) or path patching through separate circuits.

### 3.6 Hedging decomposition
- Three-way classification of false negatives: strict absorbed (main parent feature absent), compensatory (main parent feature fires but probe wrong), persistent (residual probe boundary error).
- Multi-L0 first-letter analysis (L0=22 to L0=176) for tightened hedging.
- Strict vs. loose hedging classification compared.

### 3.7 Probe degradation ablation (NEW)
- Inject weight noise into trained first-letter probes to degrade F1 to 7 target levels (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0).
- Re-run full absorption measurement pipeline at each degraded level.
- Average across 3 noise seeds per level for stability.
- Compare first-letter absorption at matched F1 to RAVEL absorption rates, decomposing probe quality confound from genuine hierarchy effects.

### Visual elements
- **Table 1**: Probe quality (F1) across 4 hierarchies x 4 layers, with gate status (strict pass / relaxed pass / below gate).

### Transition to Section 4
With validated probes (and documented limitations), we measure absorption across hierarchies.

---

## Section 4: Cross-Domain Absorption Results (2.5 pages)

### 4.1 Cross-domain variation is real
- Absorption rates at L24_16k: first-letter 27.1%, city-continent 31.4%, city-country 45.1%, city-language 11.6%.
- Descriptive range: 4.1x (11.6% to 45.1%).
- Within-RAVEL Kruskal-Wallis p=7.4e-66 (3 RAVEL hierarchies only).
- Pairwise significance: city-language vs. first-letter p_Bonf=0.003 (significant); city-continent vs. first-letter p_Bonf=1.0 (not significant); city-country vs. first-letter p_Bonf=1.0 (not significant).
- All within-RAVEL pairwise comparisons significant after Bonferroni.

### 4.2 Layer-dependent absorption
- First-letter: 2.4% (L6) to 27.1% (L24) -- dramatic concentration at the final prediction layer.
- Layer-hierarchy interaction: all hierarchies show L24 > earlier layers, but the magnitudes differ.
- Layer 24 is the dominant contributor; earlier layers show 2--9% absorption.
- F1 >= 0.96 at all layers tested, ensuring probe quality is not the driver of layer effects.

### 4.3 Width effect
- Wider SAEs (65k vs 16k) generally reduce absorption (city-country: 45.1% at 16k vs 32.9% at 65k).
- Consistent across hierarchies but magnitude varies.

### 4.4 Per-class variation
- Within city-continent: Europe 90.2% absorption vs. Africa 4%, South America ~4%.
- Within city-country: USA 0% vs. Albania, Algeria, Argentina at 100%.
- Large within-hierarchy variance suggests absorption is driven by specific parent-child pair properties, not hierarchy-wide factors.

### 4.5 First-letter is NOT the worst case
- At L24, first-letter (27.1%) shows LOWER absorption than city-country (45.1%) and comparable to city-continent (31.4%).
- This inverts the prior assumption that the first-letter spelling task represents a worst-case scenario for absorption.
- The received wisdom from Chanin et al. (2024) does not generalize.

### 4.6 Probe degradation ablation resolves the confound (NEW, key result)
- Perfect monotonic relationship: as probe F1 degrades from 1.0 to 0.70, absorption increases from 21.6% to 36.1% (Spearman rho=-1.0).
- Linear fit: slope=-0.398, R^2=0.777, p=0.0087. Quadratic fit: R^2=0.942.
- RAVEL comparison against the probe degradation curve:
  - City-continent (F1=0.87, absorption 31.4%): matches curve within 0.6pp. Variation FULLY EXPLAINED by probe quality.
  - City-country (F1=0.73, absorption 45.1%): within 8.5pp of curve. Mostly explained, with modest residual.
  - City-language (F1=0.82, absorption 11.6%): 21.3pp BELOW curve prediction (32.9%). GENUINE OUTLIER. Probe quality alone cannot explain this -- city-language has hierarchy-specific absorption suppression.
- Conclusion: probe quality is a MAJOR confound explaining most cross-domain variation, but city-language reveals genuine hierarchy-specific effects exist.

### Evidence
- 7 F1 levels tested, 3 noise seeds per level, 11,725 tokens per level.
- Internal control: F1=1.0 matches baseline (with documented per-token vs. per-word aggregation note).
- Quadratic fit (R^2=0.942) suggests nonlinear probe-absorption relationship.

### Visual elements
- **Figure 2**: Layer-dependent absorption profile. Line plot with 4 lines (one per hierarchy) across layers 6, 12, 18, 24 for 16k SAE. Shaded 95% CI bands.
  - **Purpose**: Show absorption concentrates at L24 and varies by hierarchy.
  - **Type**: line_plot
  - **Content**: 4 hierarchy lines, x=layer, y=absorption rate
  - **Key takeaway**: 15x increase from L6 to L24; cross-domain variation at every layer.
  - **Generation**: matplotlib
  - **Data source**: cross-domain absorption results across all layers

- **Table 2**: Cross-domain absorption rates at L24. Columns: hierarchy, SAE config, absorption rate, 95% CI, probe F1, N entities, N false negatives. Two rows per hierarchy (16k and 65k).
  - **Purpose**: The paper's central results table.
  - **Type**: data_table
  - **Key takeaway**: 4.1x descriptive range; variation significant within RAVEL (p=7.4e-66).

- **Figure 3**: Per-class absorption heatmap for city-continent (6 continents x 2 SAE widths). Color intensity = absorption rate, annotated with n per cell.
  - **Purpose**: Show within-hierarchy variance (Europe 90% vs Africa 4%).
  - **Type**: heatmap
  - **Key takeaway**: Absorption concentrates in specific subclasses.
  - **Generation**: seaborn heatmap

- **Figure 7 (NEW)**: Probe degradation curve. X-axis: probe F1. Y-axis: absorption rate. 7 first-letter data points with regression line (R^2=0.777). RAVEL hierarchy points overlaid at their actual F1 levels. City-continent near curve; city-language far below.
  - **Purpose**: Decompose probe quality confound from genuine hierarchy effects.
  - **Type**: scatter_plot with regression line
  - **Content**: 7 first-letter degradation points + 3 RAVEL reference points
  - **Key takeaway**: Probe quality is a major confound, but city-language is a genuine outlier (delta=-21.3pp).
  - **Generation**: matplotlib
  - **Data source**: phase1/probe_degradation.json

- **Table 5 (NEW)**: Probe degradation ablation results. Columns: target F1, actual F1, absorption rate, 95% CI, delta from RAVEL at matched F1.
  - **Purpose**: Quantitative decomposition of variation sources.
  - **Type**: data_table

### Transition to Section 5
Cross-domain variation is established and partially decomposed; now we investigate the causal mechanism underlying absorption.

---

## Section 5: Mechanism Analysis (2 pages)

### 5.1 Causal confirmation via activation patching (first-letter)
- 25 words, 200 contexts each. Child-zeroed recovery: 32.5% vs control 1.5%.
- Wilcoxon signed-rank p=0.000218. Cohen's d=1.33 (large effect). Bootstrap 95% CI for difference: [0.213, 0.421].
- 16 of 19 words with absorption show positive recovery.
- This is the first interventional (not correlational) evidence for competitive exclusion in SAEs.

### 5.2 Cross-domain patching confirms universal mechanism (CORRECTED from iter_009)
- **City-continent**: 128 entities, 4,902 absorbed contexts. Primary child-zeroed recovery: 61.9% vs control 5.2%. Cohen's d=1.50 (large). p<1e-20.
- **City-language**: 201 entities, 7,814 absorbed contexts. Primary child-zeroed recovery: 34.2% vs control 6.8%. Cohen's d=0.75 (medium). p<1e-18.
- **Universal mechanism**: Competitive exclusion operates across ALL tested hierarchy types. The concentrated-vs-distributed dichotomy from earlier iterations (based on a buggy pilot, d=-0.91) is retracted.
- Recovery magnitude varies by hierarchy: first-letter 32.5%, city-language 34.2%, city-continent 61.9%. This suggests hierarchy-dependent recovery efficiency, not different mechanisms.

### 5.3 Decoder information entanglement (reframed from "benign vs. pathological")
- **City-continent**: mean |logit change| = 3.98 nats, N=1,464 instances. 100% exceed all thresholds (0.05, 0.1, 0.2). Control: 0.12 nats.
- **First-letter**: mean |logit change| = 6.16 nats, N=158 instances. 100% exceed all thresholds. Control: 0.012 nats.
- Cross-hierarchy consistency: both hierarchies show strong direction specificity; ratio = 1.55x.
- **Circularity acknowledgment**: The diagnostic ablates the same direction used to classify FNs. It measures decoder geometry, not computational redundancy. A genuine test would require activation-level ablation (z_parent=0) or path patching through separate circuits. The result establishes that child decoders carry large-magnitude parent information, which is informative about absorption mechanics.

### 5.4 Hedging decomposition
- Compensatory hedging dominates across all hierarchies at L24: city-continent 90.9%, city-language 77%, city-country ~95%.
- Strict absorbed fraction varies: first-letter 7.9% (multi-L0), city-continent 9.1%, city-language 22.6%.
- The widely cited loose hedging classification (92.6%) is near-tautological; strict classification yields 0--22.6%.
- This is a reusable methodological contribution: future absorption studies should report strict and compensatory fractions separately.

### Visual elements
- **Figure 4**: Cross-domain activation patching results. Paired dot plot showing recovery rates for 3 hierarchies (first-letter, city-continent, city-language). Child-zeroed vs control for each. All three show significant positive recovery.
  - **Purpose**: Demonstrate universal competitive exclusion across hierarchy types.
  - **Type**: paired_dot_plot (three panels)
  - **Content**: Recovery rates per hierarchy, child-zeroed vs control, with effect sizes annotated.
  - **Key takeaway**: Competitive exclusion is universal (d=0.75--1.50), not first-letter-specific.
  - **Generation**: matplotlib
  - **Data source**: corrected full-mode activation patching data (iter_009 bugfix + iter_010)

- **Figure 5**: Decoder information entanglement histogram. Main panel: |logit change| distribution for both first-letter (N=158) and city-continent (N=1,464). Inset: control distribution. Vertical lines at thresholds.
  - **Purpose**: Show cross-hierarchy consistency of decoder entanglement.
  - **Type**: overlaid histograms with inset
  - **Content**: Two distributions + control, threshold markers
  - **Key takeaway**: Child decoders carry large-magnitude parent information in both hierarchy types (3.98--6.16 nats vs 0.01--0.12 control).
  - **Generation**: matplotlib
  - **Data source**: phase2/decoder_magnitude_firstletter.json + iter_009 benign_pathological.json

- **Table 3**: Hedging decomposition across hierarchies at L24. Columns: hierarchy, N FN, strict absorbed %, compensatory %, persistent %, significance test.
  - **Purpose**: Show compensatory dominance and between-hierarchy variation in strict fraction.
  - **Type**: data_table

### Transition to Section 6
With the universal mechanism characterized, we test whether architecture choice can mitigate absorption.

---

## Section 6: Architecture Analysis (1 page)

### Key arguments
- No significant architecture effect on absorption (ANOVA p=0.53 at L12, p=0.50 at L24). Hierarchy type is the dominant factor (p=0.005 at L12, p=0.041 at L24).
- Tested architectures: JumpReLU 16k, JumpReLU 65k, BatchTopK 16k, Matryoshka 32k.
- Reframed as "effect not detected with limited statistical power" (12 observations total). The comparison is underpowered and should not be interpreted as evidence of no effect.
- Width mismatch (Matryoshka 32k vs others 16k) is a confound.

### Visual elements
- **Figure 6**: Grouped bar chart of absorption rates by architecture, grouped by hierarchy. Error bars from bootstrap CI. ANOVA p-values annotated. "Underpowered" caveat in caption.
  - **Purpose**: Show hierarchy >> architecture in determining absorption rates.
  - **Type**: grouped_bar_chart
  - **Key takeaway**: Architecture clusters overlap within each hierarchy; hierarchies differ.
  - **Generation**: matplotlib
  - **Data source**: architecture comparison data

### Transition to Section 7
Architecture provides no detected mitigation; we discuss implications.

---

## Section 7: Discussion (1.5 pages)

### 7.1 Causal methods succeed where correlational methods fail
- Four correlational/statistical approaches failed as absorption predictors:
  - GAS: rho=0.116, AUROC=0.571 (Appendix B)
  - CMI: rho=0.044, p=0.84 (Appendix C)
  - Absorption Tax T(G): ranking rho=-0.20, concordance 50% (Appendix D)
  - Rate-distortion three-factor model: rho=0.286, R^2=0.104, all individual predictors in WRONG direction (Appendix F)
- Only activation patching (interventional/causal) successfully characterizes absorption. This quadruple negative establishes a clear methodological boundary: absorption is a causal phenomenon requiring causal methods.

### 7.2 Universal competitive exclusion with hierarchy-dependent recovery
- Activation patching confirms the same mechanism operates across all tested hierarchies (d=0.75--1.50).
- Recovery magnitude varies: city-continent 61.9% > city-language 34.2% > first-letter 32.5%. This may reflect hierarchy-dependent information distribution rather than distinct mechanisms.
- The concentrated-vs-distributed dichotomy (from iter_009) is retracted based on corrected data.

### 7.3 Probe quality as a major confound
- The probe degradation ablation (R^2=0.777) shows probe F1 is a MAJOR confound in cross-domain absorption measurement.
- City-continent variation is fully explained by probe quality. City-country is mostly explained.
- City-language is a genuine outlier: 21.3pp below the curve prediction. This hierarchy-specific suppression effect warrants investigation (possibly related to the many-to-many structure of city-language mappings).
- Recommendation: future cross-domain absorption studies MUST include probe degradation controls.

### 7.4 Implications for SAE reliability
- Absorption rates of 11--45% at the final prediction layer are concerning for safety applications.
- Current detection requires supervised probes, limiting scalability.
- The decoder information entanglement finding (3.98--6.16 nats, with circularity caveat) suggests absorbed information is not redundant -- it carries weight in downstream computation.

### 7.5 Layer dependence
- Absorption concentrating at L24 (15x higher than earlier layers) suggests it arises from the model's task-specific computation at the final prediction layer, not from generic feature representation.

### Visual elements
- **Table 4**: Summary of all hypothesis verdicts. Columns: hypothesis, verdict, key metric, confidence, paper section.
  - **Purpose**: Single-glance summary of which hypotheses survived testing.
  - **Type**: summary_table
  - **Content**: H1 (SUPPORTED_WITH_NUANCE), H2' (REFUTED), H3 (PARTIALLY_SUPPORTED), H4 (DEFINITIVE_NEGATIVE), H5 (NOT_SUPPORTED), H6 (PARTIALLY_SUPPORTED, low confidence), H7 (SUPPORTED), H7-crossdomain (SUPPORTED), H8 (CONSISTENT_CROSS_HIERARCHY), H9 (NOT_SUPPORTED), H10 (MIXED)
  - **Key takeaway**: Honest reporting -- multiple negative results alongside positive findings.
  - **Generation**: LaTeX table

---

## Section 8: Conclusion (0.5 pages)

### Key contributions (enumerated)
1. **First cross-domain absorption characterization with probe quality decomposition**: Absorption rates show 4.1x descriptive range (11.6--45.1%) across hierarchies. Probe degradation ablation (R^2=0.777) decomposes variation into probe quality effects and genuine hierarchy effects. City-language identified as a genuine outlier.
2. **Universal causal mechanism via activation patching**: Competitive exclusion confirmed across ALL tested hierarchy types (d=0.75--1.50, all p<1e-3). The mechanism is universal, not first-letter-specific.
3. **Decoder information entanglement**: Child decoders carry large-magnitude parent information (3.98--6.16 nats vs 0.01--0.12 control) consistently across hierarchies. Circularity limitation acknowledged.
4. **Hierarchy dominates architecture**: No significant architecture effect detected (p=0.50--0.53); hierarchy type is the sole significant predictor (p=0.005--0.041). Caveat: underpowered comparison.
5. **Quadruple negative for correlational predictors**: GAS, CMI, Absorption Tax, and rate-distortion models all fail. Honest reporting motivates a methodological shift to causal approaches.

### Limitations
- Single model (Gemma 2 2B). Generalization to other architectures/scales untested.
- RAVEL probes below strict quality gate; absorption rates include probe quality confound (partially decomposed by H10).
- Circularity in decoder information entanglement diagnostic.
- Architecture comparison underpowered (12 observations).
- Token position asymmetry (first-letter pos=-6, RAVEL pos=-2) not controlled.

### Future work
- Better probes for entity-attribute hierarchies (contrastive learning, richer prompt templates).
- Extension to larger models (Gemma 2 9B, Llama 3) and more hierarchy types.
- Unsupervised absorption detection accounting for encoder dynamics, not just decoder geometry.
- Investigation of city-language suppression mechanism.
- Genuine computational-redundancy test (activation-level ablation, path patching).

---

## Appendices

### Appendix A: Experimental Setup and Reproducibility
- Seeds, library versions, SAE identifiers, dataset specifications.
- Per-token aggregation method documentation.
- Token position asymmetry details.

### Appendix B: GAS Detector Failure (NR1)
- rho=0.116, AUROC=0.571, bootstrap CI [-0.333, 0.536]. 25x scale-up from pilot confirmed signal absent.
- Failure analysis: GAS measures decoder geometry; absorption is driven by encoder competitive exclusion.
- **Figure A1**: GAS vs absorption scatter plot (25 points) with regression line and CI band.

### Appendix C: CMI Null Result (NR2)
- rho=0.044, p=0.835 at L0=22 (where all 25 probes F1=1.0, eliminating probe quality confound).
- Clean null: bootstrap CI [-0.41, 0.47] firmly includes zero.
- **Table A2**: CMI correlation results with bootstrap CI.

### Appendix D: Absorption Tax Quantitative Failure (NR3)
- T(G) ranking rho=-0.20, concordance 50% (chance level).
- Qualitative concept retained; quantitative predictions fail.
- **Table A3**: T(G) values for all hierarchy-SAE combinations.

### Appendix E: Threshold Sensitivity Analysis
- 5 thresholds (0.20--0.40) x 2 SAE configs. CV=0.077. late>early ordering holds in 10/10 cells.
- Data-driven threshold (p95 random cosine = 0.044--0.049) confirms all tested thresholds well above chance.
- **Table A1**: Threshold sensitivity heatmap with subtype counts and Kruskal-Wallis p-values.

### Appendix F: Rate-Distortion Three-Factor Model (NR4)
- 131 pairs (iter_010) / 262 pairs (iter_009): model rho=0.250--0.286, R^2<10.4%. All individual predictors in WRONG direction.
- Direction reversal from pilot (n=20) to full (n=262) demonstrates small-sample instability.
- **Figure A2**: Three-panel scatter (cos_sim, co_occur, r_parent vs absorption, colored by hierarchy).
- **Table A4**: Full predictor correlation table with bootstrap CIs.

### Appendix G: Probe Training Details
- Training protocol, hyperparameters, per-layer F1 for all 16 hierarchy-layer combinations.
- **Table A6**: Complete probe quality matrix (4 hierarchies x 4 layers).

### Appendix H: Activation Patching Methodology Correction
- Documentation of pilot bug (d=-0.91) and FULL-mode correction (d=1.50).
- Per-class breakdown for cross-domain patching.

---

## Figure & Table Plan

### Figure 1: Teaser -- Absorption Concept and Cross-Domain Rates (Section: Introduction)
- **Purpose**: Establish the measurement framework and present the headline result.
- **Type**: panel_figure (left: schematic, right: bar_chart)
- **Content**: (Left) Schematic: raw activation -> probe correct; SAE activation -> probe wrong + child feature fires. (Right) Bar chart of absorption rates at L24_16k for 4 hierarchies with 95% CI.
- **Key takeaway**: Absorption exists across semantic domains at substantially different rates.
- **Generation**: tikz (schematic) + matplotlib (bar chart), composed in LaTeX
- **Data source**: consolidation_summary.json

### Figure 2: Layer-Dependent Absorption Profile (Section: Results 4.2)
- **Purpose**: Show absorption concentrates at L24 and varies by hierarchy.
- **Type**: line_plot
- **Content**: 4 lines (one per hierarchy) across layers 6, 12, 18, 24 for 16k SAEs. Shaded 95% CI bands.
- **Key takeaway**: 15x variation from L6 to L24; cross-domain variation at every layer.
- **Generation**: matplotlib
- **Data source**: cross-domain absorption results across all layers

### Figure 3: Per-Class Absorption Heatmap (Section: Results 4.4)
- **Purpose**: Reveal within-hierarchy variance (Europe 90% vs Africa 4%).
- **Type**: heatmap
- **Content**: 6 continent classes x 2 SAE widths. Cell color = absorption rate, annotated with n.
- **Key takeaway**: Absorption is concentrated in specific subclasses, not uniform.
- **Generation**: seaborn heatmap
- **Data source**: per-class tables from cross-domain results

### Figure 4: Cross-Domain Activation Patching (Section: Mechanism 5.2)
- **Purpose**: Demonstrate universal competitive exclusion across hierarchy types.
- **Type**: paired_dot_plot (three panels)
- **Content**: Recovery rates for first-letter (d=1.33), city-continent (d=1.50), city-language (d=0.75). Child-zeroed vs control.
- **Key takeaway**: Competitive exclusion confirmed universally, with hierarchy-dependent recovery magnitude.
- **Generation**: matplotlib
- **Data source**: corrected full-mode activation patching data

### Figure 5: Decoder Information Entanglement Distribution (Section: Mechanism 5.3)
- **Purpose**: Show cross-hierarchy consistency of decoder entanglement and its magnitude.
- **Type**: overlaid_histograms with inset
- **Content**: |logit change| distributions for first-letter (N=158, mean=6.16) and city-continent (N=1,464, mean=3.98). Control inset (mean ~0.01--0.12). Threshold lines at 0.05, 0.1, 0.2.
- **Key takeaway**: Child decoders carry large-magnitude parent information in both hierarchies (3.98--6.16 nats vs 0.01--0.12 control).
- **Generation**: matplotlib
- **Data source**: phase2/decoder_magnitude_firstletter.json + iter_009 benign_pathological data

### Figure 6: Architecture Comparison (Section: Architecture 6)
- **Purpose**: Show hierarchy >> architecture for absorption rates.
- **Type**: grouped_bar_chart
- **Content**: 4 architectures x 4 hierarchies. ANOVA p-values annotated. "Underpowered" noted in caption.
- **Key takeaway**: Architecture bars overlap within hierarchy clusters; clusters differ.
- **Generation**: matplotlib
- **Data source**: architecture comparison data

### Figure 7: Probe Degradation Ablation Curve (Section: Results 4.6) -- NEW, KEY FIGURE
- **Purpose**: Decompose probe quality confound from genuine hierarchy effects.
- **Type**: scatter_plot with regression line
- **Content**: 7 first-letter points (F1 vs absorption) with linear regression (R^2=0.777) and quadratic fit (R^2=0.942). 3 RAVEL points overlaid at their F1 levels. City-continent near curve; city-language far below; city-country slightly above.
- **Key takeaway**: Probe quality explains most variation; city-language is a genuine outlier (-21.3pp).
- **Generation**: matplotlib
- **Data source**: phase1/probe_degradation.json

### Figure A1: GAS vs Absorption Scatter (Appendix B)
- **Purpose**: Document GAS detector failure.
- **Type**: scatter_plot with regression line
- **Content**: 25 points (one per letter), x=GAS, y=absorption rate. CI band.
- **Key takeaway**: rho=0.116, no meaningful correlation.
- **Generation**: matplotlib
- **Data source**: gas_full.json

### Figure A2: Rate-Distortion Predictor Scatter (Appendix F)
- **Purpose**: Show individual predictors are weakly or oppositely correlated with absorption.
- **Type**: three-panel scatter (cos_sim, co_occur, r_parent)
- **Content**: 131 points colored by hierarchy. Regression lines with CI.
- **Key takeaway**: All predictors in wrong direction; model R^2=0.104.
- **Generation**: matplotlib
- **Data source**: phase2/rate_distortion_pooled.json

### Table 1: Probe Quality Matrix (Section: Methodology 3.2)
- **Purpose**: Establish measurement reliability for all downstream results.
- **Content**: 4 hierarchies x 4 layers, F1 scores, gate status (strict/relaxed/below).

### Table 2: Cross-Domain Absorption Rates (Section: Results 4.1)
- **Purpose**: Central results table.
- **Content**: 4 hierarchies x 2 widths at L24, absorption rate, 95% CI, N, N FN, probe F1.

### Table 3: Hedging Decomposition (Section: Mechanism 5.4)
- **Content**: Hierarchies at L24, N FN, strict %, compensatory %, persistent %, chi-square.

### Table 4: Hypothesis Verdict Summary (Section: Discussion 7)
- **Content**: All hypotheses with verdict, key metric, confidence, section.

### Table 5: Probe Degradation Ablation (Section: Results 4.6) -- NEW
- **Content**: 7 F1 levels, actual F1, absorption rate, CI, delta from RAVEL at matched F1.

### Table A1: Threshold Sensitivity (Appendix E)
- **Content**: Subtype proportions across 5 thresholds x 2 SAE configs.

### Table A2: CMI Correlation (Appendix C)
- **Content**: rho, p, CI at L0=22 (all probes F1=1.0).

### Table A3: Absorption Tax T(G) (Appendix D)
- **Content**: T(G) by hierarchy and SAE configuration.

### Table A4: Rate-Distortion Predictors (Appendix F)
- **Content**: Each predictor, rho, p, bootstrap CI, per-hierarchy rho.

### Table A5: Decoder Information Entanglement Cross-Hierarchy (Section 5.3 or Appendix)
- **Content**: First-letter vs city-continent: mean |logit change|, N, control |logit change|, statistical tests.

### Table A6: Complete Probe Quality Matrix (Appendix G)
- **Content**: 4 hierarchies x 4 layers, F1, accuracy, N classes, N samples.

---

## Narrative Flow Summary

1. **Introduction**: Absorption threatens SAE reliability; all evidence is from one task. We extend to 4 hierarchies, find 4.1x descriptive variation, decompose probe confound, and confirm a universal causal mechanism.
2. **Background**: SAEs, absorption definition, architectures, RAVEL, prior work.
3. **Methodology**: Probes, measurement pipeline, patching protocol, decoder entanglement diagnostic (with circularity caveat), hedging decomposition, probe degradation ablation.
4. **Cross-Domain Results**: Variation is real (p=7.4e-66) but LARGELY confounded by probe quality (R^2=0.777). City-language is a genuine outlier. First-letter is not worst case. Layer dependence is dramatic (15x).
5. **Mechanism**: Activation patching confirms universal competitive exclusion (d=0.75--1.50). Decoder entanglement consistent cross-hierarchy (3.98--6.16 nats). Compensatory hedging dominates.
6. **Architecture**: Hierarchy >> architecture. No detected architecture effect (underpowered).
7. **Discussion**: Quadruple negative for correlational predictors. Probe quality as major confound. City-language anomaly. Implications for safety. Layer dependence.
8. **Conclusion**: Five contributions with honest limitations. Probe degradation ablation as new methodological standard.
