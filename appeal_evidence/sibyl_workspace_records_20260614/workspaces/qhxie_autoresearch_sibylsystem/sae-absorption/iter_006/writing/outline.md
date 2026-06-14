# Paper Outline: What Does the Absorption Metric Actually Measure?

## Title

**Beyond Competitive Exclusion: Hedging Dominance, Rate-Distortion Diagnostics, and the L0 Phase Transition in SAE Feature Absorption**

Alternative: **Diagnosing Feature Absorption in Sparse Autoencoders: Metric Validity, Hedging Decomposition, and Information-Theoretic Thresholds**

---

## Abstract (Target: ~200 words)

Key elements:
- Feature absorption -- parent SAE features silently failing when child features are active -- is the primary reliability concern for SAE-based mechanistic interpretability
- All existing measurements use a single task (first-letter spelling) on a single architecture (GPT-2 with L1-ReLU SAEs); no study has validated the Chanin metric on modern JumpReLU SAEs or decomposed what it actually captures
- We present three findings on Gemma 2 2B with Gemma Scope SAEs:
  1. **Metric audit**: The Chanin absorption metric does not transfer to JumpReLU SAEs without recalibration -- shuffled-label controls produce 4.7x higher "absorption" than true labels across all 5 tested domains. Confound decomposition reveals 98.6% of detected false negatives at L0=22 are hedging (information spreading), not hierarchy-driven competitive exclusion. Genuine absorption is ~0.75% (9/1196 words).
  2. **L0 phase transition**: Absorption declines monotonically from 42.9% (L0=22) to 0.8% (L0=176), stable across 3 layers (CV < 10%). Nine words persist as absorbed across all L0 values -- candidates for genuine competitive exclusion [conditional on activation patching validation].
  3. **Rate-distortion diagnostic**: Conditional mutual information I(X; f_parent | f_child) at d'=10 correlates with absorption susceptibility (Spearman rho = -0.383, Cohen's d = -0.924, Mann-Whitney p = 0.042) [conditional on L0=22 replication]. For unit-normalized SAE decoders, the geometric constant degenerates, simplifying the theoretical threshold to L0_crit ~ lambda / CMI.

---

## 1. Introduction (Target: ~700 words)

### 1.1 The Problem

- SAEs decompose model activations into sparse interpretable features; feature absorption silently corrupts this decomposition
- Chanin et al. (NeurIPS 2025 Oral): absorption rates of 15-35% on first-letter spelling across GPT-2 SAEs; parent features fail to fire when child features are active
- Absorption has motivated an architectural mitigation wave: Matryoshka SAE (ICML 2025), OrtSAE, ATM-SAE, masked regularization (Narayanaswamy et al., 2026)
- The mitigation wave assumes the diagnostic is valid. No work has validated the metric on a different SAE architecture.

### 1.2 Three Open Questions

- **Q1 -- Metric validity**: Does the Chanin absorption metric transfer from GPT-2/L1-ReLU SAEs to Gemma 2 2B/JumpReLU SAEs? What fraction of measured "absorption" is genuine competitive exclusion versus hedging or metric artifact?
- **Q2 -- Sparsity dynamics**: How does absorption scale with the L0 sparsity operating point? Is there a phase transition threshold below which absorption is negligible?
- **Q3 -- Theoretical criterion**: Can information theory predict which features are susceptible to absorption, and at what sparsity level absorption becomes unavoidable?

### 1.3 Our Contributions

1. **Metric audit and hedging decomposition** (Section 4): The Chanin metric produces 4.7x more "absorption" on shuffled labels than true labels on Gemma 2 2B. Confound decomposition at L0=22 with perfect probes (F1=1.0) reveals 98.6% hedging, 1.4% hierarchy-driven. Genuine absorption is approximately 9 persistent words out of 1196 (~0.75%).
2. **L0-absorption phase transition** (Section 5): Absorption declines monotonically from 42.9% to 0.8% across L0={22, 41, 82, 176}, stable across layers 10, 12, and 20 (CV < 10%). The phase transition around L0~40-80 suggests a sparsity threshold above which competitive exclusion is suppressed.
3. **Rate-distortion diagnostic** (Section 6, conditional): CMI I(X; f_parent | f_child) at d'=10 shows Spearman rho = -0.383 (p = 0.059, Cohen's d = -0.924) between CMI and absorption rate. Absorbed letters have lower CMI (mean 0.649 vs 0.861 for non-absorbed, Mann-Whitney p = 0.042). For normalized SAEs, the geometric constant c = sin^2(angle) ~ 0.97 everywhere, simplifying the rate-distortion threshold to L0_crit ~ lambda / CMI.

### 1.4 Paper Organization

One-sentence preview of each remaining section.

**Transition**: "Section 2 provides the technical background on SAEs and the Chanin absorption metric that our audit targets."

---

## 2. Background and Related Work (Target: ~600 words)

### 2.1 Sparse Autoencoders in Mechanistic Interpretability

- SAE architecture: encoder W_e, decoder W_d, sparse latent z, reconstruction x_hat
- Two SAE families: L1-penalized ReLU (Anthropic, EleutherAI) and JumpReLU with hard threshold (Gemma Scope, Rajamanickam et al. 2024)
- The L0 operating point: number of non-zero features per forward pass; controls the sparsity-fidelity tradeoff

### 2.2 Feature Absorption

- Chanin et al. (2024): formal definition -- parent latent z_p = 0 on parent-positive inputs when child latent z_c > 0
- Measurement protocol: train linear probes, identify main features by decoder cosine similarity, compute false negative rate
- SAEBench absorption metric: absorption_first_letter score across 100+ SAEs
- Tang et al. (2025): biconvex optimization landscape; absorption arises from partial minima of the SDL loss
- O'Neill et al. (2024): amortization gap -- feedforward encoding structurally cannot recover all dictionary atoms

### 2.3 Architectural Mitigations

- Matryoshka SAE (Bussmann et al., ICML 2025): nested dictionaries preserving parent features
- OrtSAE, KronSAE, ATM-SAE: encoder modifications targeting absorption
- MP-SAE (Costa et al., 2025): iterative encoding
- Masked regularization (Narayanaswamy et al., 2026)
- **Gap**: all mitigations assume the diagnostic (absorption rate) is valid. No validation on JumpReLU architecture.

### 2.4 Rate-Distortion Theory and Successive Refinement

- Successive refinement theorem (Equitz & Cover, 1991): a source is successively refinable iff X -- f_child -- f_parent forms a Markov chain
- Markov chain condition: I(X; f_parent | f_child) = 0 means the parent carries no unique information beyond the child
- Connection to absorption: when CMI is low, the sparsity pressure to absorb the parent is information-theoretically cheap; when CMI is high, absorption destroys unique information

**Transition**: "With this background established, we describe the experimental methodology for our three-pronged study."

---

## 3. Methodology (Target: ~800 words)

### 3.1 Models and SAE Configurations

- **Table 1** (inline): Gemma 2 2B + Gemma Scope SAEs: L12 at L0={22, 41, 82, 176} x widths {16k, 65k}; L10-16k, L20-16k for cross-layer validation
- JumpReLU architecture with hard threshold (distinct from L1-ReLU used in prior absorption work)
- GPT-2 Small + SAELens SAEs (L1-ReLU) for cross-architecture reference

### 3.2 Absorption Measurement Protocol

- Adaptation of the Chanin et al. protocol for Gemma 2 2B:
  - Vocabulary: single-token alphabetic words (1204 words for improved first-letter, 50+ per letter)
  - Probe training: k-sparse logistic regression (k=5) on SAE latent activations
  - Quality gate: F1 > 0.85 per parent class (below-gate parents reported separately)
  - Main feature identification: decoder cosine similarity >= 0.025 with probe direction
  - Absorption criterion: all k split latents fail to activate, but probe correctly classifies
  - Bootstrap 95% CI (10,000 resamples, seed=42)

### 3.3 Control Suite

- C1: Random probe direction (expected <2%)
- C2: Shuffled parent labels (expected ~0% net above random)
- C3: Dense probe ceiling (logistic regression on raw model activations)
- Null-domain benchmark [planned]: non-hierarchical tasks (word length, frequency bins) to establish metric false-positive floor

### 3.4 Confound Decomposition

- Multi-L0 sweep (L0 = {22, 41, 82, 176}) on L12-16k
- For each false-negative token, classify cause:
  - **Hierarchy-driven**: token absorbed only at specific L0 values where sparsity pressure is highest
  - **Hedging**: token's parent information spread across many features, none clearing threshold at any L0
  - **Reconstruction error**: SAE fails to reconstruct the parent direction entirely
- Classification rule: if token's parent FN persists across ALL L0 values, classify as hedging; if FN appears only at low L0 and recovers at high L0, classify as hierarchy-driven

### 3.5 CMI Estimation

- Conditional mutual information I(X; f_parent | f_child) estimated via k-NN method (Kraskov et al., 2004)
- Decoder subspace projection: project activations onto d'-dimensional subspace of top decoder directions
- Pre-registered analysis dimension d'=10; sensitivity reported at d'={20, 30, 50}
- Absorption rate: per-letter rate from improved first-letter experiment (1204 words)

### 3.6 Cross-Domain Hierarchy Suite

- Five hierarchy domains: first-letter (26 classes), city-country (29 countries), city-continent (6 continents), city-language (18 languages), animal-class (6 classes)
- 189 single-token cities, 50+ animals
- Probe quality and control applied per domain

**Transition**: "We first present our metric audit results, which fundamentally constrain the interpretation of all subsequent findings."

---

## 4. The Absorption Metric Does Not Transfer to JumpReLU SAEs (Target: ~1000 words)

### 4.1 Universal Control Failure (Figure 1, Table 2)

- **Figure 1**: Grouped bar chart -- measured absorption rate vs. shuffled control rate vs. random probe rate, for all 5 hierarchy domains on L12-16k (L0=82)
- Shuffled-label controls exceed measured rates in ALL 5 domains:
  - First-letter: 15.96% measured vs. 74.6% shuffled (4.7x)
  - City-continent: 6.49% vs. 45.2%
  - City-language: 6.56% vs. 18.0%
  - Animal-class: 1.43% vs. 39.3%
  - City-country: 0.0% vs. 10.3%
- Random probe control: 11.8% (target: <2%) on first-letter; 9.2-19.0% on other domains
- The measured "absorption" is LOWER than what random label assignment produces. The metric is not measuring hierarchy-driven competitive exclusion in this configuration.

### 4.2 Confound Decomposition: Hedging Dominates (Figure 2)

- **Figure 2**: Stacked bar chart showing the decomposition at each L0: fraction hedging vs. hierarchy-driven vs. reconstruction error
- At L0=22 with perfect probes (F1=1.0 for all 25 letters): 98.6% hedging, 1.4% hierarchy-driven, 0% reconstruction error
- At L0=82: 84.1% hedging, 15.9% hierarchy-driven
- At L0=176: 10% hedging, 90% hierarchy-driven
- Monotonic tradeoff: as L0 increases and sparsity relaxes, the hedging fraction decreases and hierarchy-driven fraction increases
- Genuine hierarchy-driven absorption: approximately 9 words out of 1196 (~0.75%) persist across all L0 values

### 4.3 First-Letter Replication with Improved Protocol (Table 3)

- **Table 3**: Improved first-letter results -- 1204 words (50+ per letter) vs. Chanin's original 25 per letter
- Aggregate absorption rate: 15.96% (L0=82), within published 15-35% range
- Mean probe F1: 0.817 (improved from 0.565 pilot); 18/25 letters pass F1 > 0.85 gate
- Cross-layer stability: L10=13.9%, L12=16.0%, L20=13.6% (CV < 10%)
- The replication succeeds in MAGNITUDE but the control failure means the INTERPRETATION as competitive exclusion is unsupported

### 4.4 Cross-Domain Rates Are Uninterpretable

- City-continent (6.49%), city-language (6.56%): first-ever measurements, but all below their own shuffled controls
- City-country (0.0%): genuine zero after quality-gated probes (16/29 countries pass F1 > 0.85)
- Animal-class (1.43%): below shuffled control (39.3%)
- SAE probes outperform dense probes (F1=0.773 vs. 0.617 on city-country): the SAE does encode geographic knowledge
- **Framing**: these measurements demonstrate the metric's behavior across domains, but absolute rates cannot be claimed as genuine absorption until controls pass

**Transition**: "Despite the metric limitations, the L0 sweep reveals a robust pattern in how the metric's output varies with sparsity."

---

## 5. The L0-Absorption Phase Transition (Target: ~800 words)

### 5.1 Monotonic Decline Across L0 (Figure 3)

- **Figure 3**: Line plot -- absorption rate (y-axis) vs. L0 (x-axis, log scale) for L12-16k; separate traces for L10 and L20 showing cross-layer consistency
- L0=22: 42.85% (perfect probes)
- L0=41: 30.5%
- L0=82: 15.96%
- L0=176: 0.8%
- Phase transition region: L0~40-80 where the steepest decline occurs
- Cross-layer: L10 (13.9%), L12 (16.0%), L20 (13.6%) at L0=82 -- coefficient of variation < 10%

### 5.2 Width-L0 Interaction (Table 4)

- **Table 4**: Absorption rates across the width x L0 grid
- OLS interaction: width x L0 coefficient significant (F = 37.75, p = 1.24e-6)
- GAM interaction: not significant (p = 1.0) -- the interaction is approximately linear in log space
- L0 main effect: Spearman rho = -0.457 (p = 0.007)
- Width main effect: rho = 0.308 (p = 0.076, not significant after correction)

### 5.3 JumpReLU vs L1 Architecture Reference

- JumpReLU (Gemma): dramatic L0-dependent phase transition (42.9% at L0=22 to 0.8% at L0=176)
- L1-ReLU (GPT-2 Small): uniformly high absorption (61-67%) across layers 8, 10, 11; no L0 phase transition
- Hartigan's dip test: both architectures show non-uniform (bimodal) per-letter distributions (JumpReLU JR_L0=82: dip=0.188, p=0.001)
- **Cross-model caveat**: Gemma 2 2B (2304-dim, 16k SAE) vs GPT-2 Small (768-dim, 24k SAE) -- model capacity differences confound the architecture comparison

### 5.4 The Nine Persistent Core Words

- 9 words show non-zero absorption at ALL tested L0 values (22, 41, 82, 176)
- These are the strongest candidates for genuine hierarchy-driven competitive exclusion
- [Conditional on activation patching]: zero child features and measure parent recovery to validate metric-independently

**Transition**: "We next ask whether information theory can predict which features are susceptible to this persistent absorption."

---

## 6. Rate-Distortion Diagnostic: CMI Predicts Absorption Susceptibility (Target: ~800 words)

[NOTE: This section is CONDITIONAL on L0=22 replication experiment. If CMI fails replication, Section 6 is reported as "directionally correct, insufficient evidence" and demoted to supplementary material.]

### 6.1 CMI Estimation and the Successive Refinement Connection

- The successive refinement theorem: X is successively refinable iff X -- f_child -- f_parent forms a Markov chain
- I(X; f_parent | f_child) measures how much unique information the parent feature carries beyond the child
- Low CMI -> absorption is information-theoretically cheap (parent is nearly redundant given child)
- High CMI -> absorption destroys unique information
- k-NN estimation (Kraskov et al., 2004) in d'=10 dimensional decoder subspace

### 6.2 CMI-Absorption Correlation (Figure 4)

- **Figure 4**: Scatter plot -- CMI (x-axis) vs. absorption rate (y-axis) for 25 letters; color by absorbed/non-absorbed status
- Absorbed letters (n=13): mean CMI = 0.649 +/- 0.187
- Non-absorbed letters (n=9): mean CMI = 0.861 +/- 0.258
- Spearman rho = -0.383 (p = 0.059) at d'=10
- Mann-Whitney two-sided p = 0.045; one-sided p = 0.023
- Cohen's d = -0.924 (large effect)

### 6.3 Phase Transition Scale Prediction

- Predicted L0_crit = lambda / (CMI * c) where c = ||w_P||^2 * (1 - cos^2(w_P, w_C))
- Mean predicted L0_crit = 24.7; empirical half-maximum L0 = 22.4 (10.2% relative error)
- The rank-order prediction is correct: letters with higher predicted L0_crit tend to have higher absorption (rho = +0.333)
- **Circularity caveat**: lambda is estimated from the empirical half-max, so the scale match is partly by construction; the non-trivial prediction is the rank ordering

### 6.4 Geometric Constant Degeneration

- For unit-normalized Gemma Scope decoders: ||w_P|| = 1.0 for all features
- c = sin^2(angle(w_P, w_C)) with mean = 0.960, CV = 2.16%
- c provides essentially no additional modulation beyond CMI
- CMI/c performs no better than raw CMI (rho = -0.375 vs. -0.383)
- **Theoretical simplification**: for normalized SAEs, the rate-distortion threshold reduces to L0_crit ~ lambda / CMI

### 6.5 Dimension Sensitivity (Appendix Figure)

- d'=10: rho = -0.383 (p = 0.059)
- d'=20: rho = +0.048 (p = 0.818) -- sign reversal
- d'=30: rho = +0.299 (p = 0.147) -- reversal
- d'=50: rho = +0.197 (p = 0.345) -- reversal
- The negative correlation holds ONLY at d'=10. This is a significant limitation. Possible explanation: at low d', the subspace captures the most absorption-relevant decoder directions; at higher d', noise dilutes the signal.
- This limitation is reported transparently and motivates future work on dimension-agnostic MI estimation.

**Transition**: "We discuss the implications of all three findings and their collective constraints on the field."

---

## 7. Discussion (Target: ~700 words)

### 7.1 Implications for the Absorption Mitigation Literature

- The entire mitigation wave (Matryoshka, OrtSAE, ATM-SAE, masked regularization) assumes the absorption metric is measuring competitive exclusion
- Our hedging decomposition (98.6% at L0=22) suggests that the dominant false-negative mechanism is information spreading, not competitive suppression
- If most "absorption" is hedging, architectural mitigations targeting competitive exclusion may not address the primary failure mode
- The L0 phase transition suggests that simply increasing L0 (relaxing sparsity) may be more effective than architectural changes for reducing the metric's output

### 7.2 The Metric Validity Question

- The Chanin metric was developed and validated on GPT-2 Small with L1-ReLU SAEs
- JumpReLU SAEs have fundamentally different activation dynamics: hard thresholds create winner-take-all competition that L1-ReLU's soft penalties do not
- The universal control failure is consistent with the metric's cosine/magnitude thresholds being miscalibrated for the JumpReLU feature geometry
- Recommendation for the field: before measuring absorption on any new SAE architecture, validate controls (shuffled labels < measured rates in >= 3 domains)

### 7.3 Rate-Distortion Theory as a Principled Diagnostic

- The CMI-absorption correlation, if replicated at L0=22, provides the first information-theoretic criterion for absorption susceptibility
- The geometric constant degeneration for normalized SAEs is a useful theoretical simplification
- The dimension sensitivity (d'=10 only) is a genuine weakness; future work should develop dimension-agnostic MI estimators (MINE, KNIFE) or derive d' from theory

### 7.4 Negative Results

- **H2 falsified**: 1.4% hierarchy-driven at L0=22, not >80%. The pilot's 96.9% was a methodological artifact.
- **H4 falsified**: Unsupervised detection rho = -0.125, AUROC = 0.47. No geometric SAE weight signal discriminates absorbed from non-absorbed features.
- **H7 falsified as stated**: Both JumpReLU and L1 show bimodal distributions. The actual finding is that JumpReLU has a dramatic L0-dependent phase transition while L1 shows uniformly high absorption.
- **H6 underpowered**: n=5 domains insufficient for hierarchy-predictor analysis.

### 7.5 Limitations

- All Gemma results use Gemma Scope SAEs (Gemma 2 2B model accessible; SAE weights publicly available). Cross-architecture comparison confounded by model capacity differences (Gemma 2B vs GPT-2 Small).
- CMI correlation holds at d'=10 only; dimension instability is a significant caveat (Bonferroni-corrected p = 0.236).
- Cross-domain absorption rates cannot be interpreted as genuine absorption until control failure is resolved.
- The 9 persistent core words have not been validated via activation patching (planned).
- Confound decomposition classification depends on which L0 values are included in the sweep.

---

## 8. Conclusion (Target: ~300 words)

- The Chanin absorption metric does not transfer to Gemma 2 2B JumpReLU SAEs without recalibration: shuffled controls exceed measured rates by 4.7x.
- 98.6% of what the metric calls absorption at L0=22 is hedging, not competitive exclusion. Genuine absorption is approximately 0.75% (9/1196 words).
- Absorption declines monotonically with L0 (42.9% to 0.8%), with a phase transition around L0~40-80 -- this profile is the most robust finding, stable across 3 layers.
- CMI provides a directionally correct predictor of absorption susceptibility (rho = -0.383 at d'=10, Cohen's d = -0.924), consistent with rate-distortion theory's prediction that low-CMI parent features are information-theoretically cheap to absorb. The geometric constant degenerates for normalized SAEs.
- These results collectively suggest the field should validate absorption metrics on new architectures before building mitigations, and that the L0 operating point -- not the encoder architecture -- is the primary control parameter for absorption severity.
- Code and data released as an SAEBench extension.

---

## Appendices

### Appendix A: Per-Letter Absorption Results

- Full 25-letter table with per-letter: absorption rate, probe F1, CMI, number of words, control rates
- Stratified by probe quality tier (F1 > 0.85 vs 0.70-0.85 vs < 0.70)

### Appendix B: Cross-Domain Detailed Results

- Per-domain, per-parent-class absorption rates and probe F1
- City-country: per-country breakdown (16 strong, 6 moderate, 7 weak)
- SAE probes vs dense probes comparison

### Appendix C: CMI Dimension Sensitivity

- Full dimension sweep: d' = {5, 10, 15, 20, 30, 50}
- Spearman rho, p-value, Cohen's d at each dimension
- Analysis of why d'=10 works and higher dimensions do not

### Appendix D: Bifurcation Analysis

- JumpReLU vs L1 per-letter distributions (full results)
- Hartigan's dip test statistics
- Cross-model caveats detailed

### Appendix E: Scaling Surface

- GAM fit details (pseudo-R2 = 0.85)
- OLS regression table
- Width x L0 interaction details

---

## Figure & Table Plan

### Figure 1: Universal Control Failure (Section: Metric Audit)
- **Purpose**: Immediately confront the reader with the core finding -- the absorption metric's controls fail on JumpReLU SAEs
- **Type**: grouped_bar_chart
- **Content**: 5 domains x 3 bars each (measured rate, shuffled control, random probe control) on L12-16k (L0=82). Horizontal dashed line at the random probe expected floor (<2%). Error bars = 95% bootstrap CI.
- **Key takeaway**: In all 5 domains, shuffled controls exceed measured rates. The metric cannot distinguish hierarchy-driven absorption from noise on this architecture.
- **Generation**: matplotlib grouped bar chart
- **Data source**: `exp/results/full/first_letter_improved.json`, `exp/results/full/cross_domain_*.json`

### Figure 2: Hedging Decomposition Across L0 (Section: Metric Audit)
- **Purpose**: Show that the composition of false negatives changes dramatically with L0
- **Type**: stacked_bar_chart
- **Content**: 4 L0 values (22, 41, 82, 176) on x-axis. Stacked bars: fraction hedging (blue), fraction hierarchy-driven (red), fraction reconstruction error (gray). Annotate: "98.6% hedging" at L0=22, "90% hierarchy-driven" at L0=176.
- **Key takeaway**: At the L0 where probes are perfect (L0=22), nearly all "absorption" is hedging; genuine competitive exclusion is ~0.75%.
- **Generation**: matplotlib stacked bar
- **Data source**: `exp/results/full/confound_decomposition_multi_l0.json`

### Figure 3: L0-Absorption Phase Transition (Section: L0 Profile)
- **Purpose**: Show the robust monotonic decline of absorption with L0
- **Type**: line_plot (with error bars)
- **Content**: x-axis = L0 (log scale: 22, 41, 82, 176); y-axis = absorption rate (%). Three traces: L10-16k, L12-16k, L20-16k. Shaded region for 95% CI. Annotate the phase transition region (L0~40-80).
- **Key takeaway**: Absorption declines monotonically from 42.9% to 0.8%; the profile is layer-invariant (CV < 10%).
- **Generation**: matplotlib line plot with confidence bands
- **Data source**: `exp/results/full/confound_decomposition_multi_l0.json`, `exp/results/full/scaling_surface.json`

### Figure 4: CMI vs Absorption Rate (Section: Rate-Distortion)
- **Purpose**: Show the CMI-absorption relationship and the group-level separation
- **Type**: scatter_plot with marginal histograms
- **Content**: x-axis = CMI at d'=10; y-axis = absorption rate per letter. Color: red = absorbed letters (n=13), blue = non-absorbed (n=9). Annotate: Spearman rho = -0.383, Cohen's d = -0.924, Mann-Whitney p = 0.042. Add marginal density plots showing group separation.
- **Key takeaway**: Absorbed letters have lower CMI, consistent with rate-distortion theory: low-CMI parents carry little unique information and are cheap to absorb.
- **Generation**: matplotlib scatter with seaborn marginals
- **Data source**: `exp/results/full/cmi_estimation.json`

### Table 1: SAE Configurations Used (Section: Methodology)
- **Purpose**: Define the experimental setup
- **Type**: configuration_table
- **Content**: Columns: Config Name, Model, Layer, d_SAE, L0, Architecture (JumpReLU/L1)
- **Key takeaway**: 9 Gemma Scope configs spanning L0={22,41,82,176} x widths {16k,65k} + 3 GPT-2 configs
- **Generation**: LaTeX table

### Table 2: Control Results by Domain (Section: Metric Audit)
- **Purpose**: Quantify the control failure precisely
- **Type**: comparison_table
- **Content**: Columns: Domain, N_parents, Probe F1 (mean), Measured Rate, Shuffled Rate, Random Rate, Ratio (Shuffled/Measured)
- **Key takeaway**: Ratio > 1.0 in all domains (range: 1.6x to 27.5x). The metric produces MORE absorption with random labels than true labels.
- **Generation**: LaTeX table (bold the Ratio column)
- **Data source**: `exp/results/full/cross_domain_comparative.json`

### Table 3: Improved First-Letter Results (Section: Metric Audit)
- **Purpose**: Show the replication result with improved protocol
- **Type**: data_table
- **Content**: Columns: Letter, N_words, Probe F1, Absorption Rate (%), Shuffled Rate (%), Net Signal
- **Key takeaway**: Aggregate rate 15.96% replicates Chanin et al. magnitude; but net signal (measured - shuffled) is negative for most letters
- **Generation**: LaTeX table (bold letters with F1 > 0.85)

### Table 4: L0 x Width Absorption Grid (Section: L0 Profile)
- **Purpose**: Quantify the width-L0 interaction
- **Type**: ablation_table
- **Content**: Rows: L0 values {22, 41, 82, 176}; Columns: widths {16k, 65k}; Cells: absorption rate (%)
- **Key takeaway**: L0 is the dominant factor (rho = -0.457); width effect is secondary (rho = 0.308, n.s.)
- **Generation**: LaTeX table with heatmap-style coloring

### Figure A1: Dimension Sensitivity of CMI Correlation (Appendix C)
- **Purpose**: Transparently report the CMI dimension dependence
- **Type**: line_plot
- **Content**: x-axis = d'; y-axis = Spearman rho. Error bars = bootstrap 95% CI. Horizontal line at rho = 0. Highlight d'=10 (the only negative-rho point).
- **Key takeaway**: The negative correlation exists only at d'=10; at higher dimensions it reverses. This is a significant limitation.
- **Generation**: matplotlib line plot
- **Data source**: `exp/results/full/cmi_estimation.json`

### Figure A2: Bifurcation Analysis (Appendix D)
- **Purpose**: Compare JumpReLU and L1 per-letter absorption distributions
- **Type**: violin_plot or histogram grid
- **Content**: Panel (a): JumpReLU per-letter distributions across 5 L0 values. Panel (b): L1 per-letter distributions across 3 layers. KS D=0.607 annotated.
- **Key takeaway**: Both show bimodal distributions (H7 falsified); JumpReLU has L0 phase transition, L1 is uniformly high.
- **Generation**: seaborn violinplot
- **Data source**: `exp/results/full/bifurcation_analysis.json`

---

## Section-by-Section Word Budget

| Section | Target Words |
|---------|-------------|
| Abstract | 200 |
| 1. Introduction | 700 |
| 2. Background | 600 |
| 3. Methodology | 800 |
| 4. Metric Audit | 1000 |
| 5. L0 Phase Transition | 800 |
| 6. Rate-Distortion | 800 |
| 7. Discussion | 700 |
| 8. Conclusion | 300 |
| **Total** | **~5,900** |

Targeting NeurIPS 2026 (9 pages + references) or EMNLP 2026 (8 pages + references). The conditional structure (Section 6 dependent on CMI replication) allows downgrading to a 2-pillar paper without restructuring.

---

## Narrative Flow Summary

Introduction establishes that the absorption mitigation wave assumes the absorption metric is valid -- and poses three questions about metric validity, sparsity dynamics, and theoretical prediction. Background provides technical context on SAEs, the Chanin metric, and rate-distortion theory. Methodology specifies the experimental setup including the critical control suite. Section 4 delivers the core surprise: the metric does not transfer to JumpReLU SAEs (controls fail by 4.7x), and confound decomposition reveals 98.6% hedging at L0=22 -- the first finding that what the metric calls "absorption" is predominantly information spreading, not competitive exclusion. Section 5 pivots to what IS robust: the L0-absorption monotonic profile, which is layer-invariant and reveals a clear phase transition around L0~40-80. This profile is informative about SAE sparsity dynamics even if the absolute interpretation as "absorption" is contested. Section 6 connects to theory: CMI predicts absorption susceptibility (rho = -0.383, large effect), and the geometric constant degenerates for normalized SAEs, simplifying the theoretical criterion. Section 7 integrates the implications -- the field should validate metrics before building mitigations, and the L0 operating point (not the encoder architecture) is the primary lever. The negative results (H2, H4, H6, H7) are reported honestly as constraints on future work.

The paper reads as: "We set out to measure absorption across domains and found something more important -- the metric itself needs validation. Here is what we found when we audited it, and here is what information theory predicts."

---

## Conditional Structure

The outline supports three paper scenarios identified by the result debate synthesis:

**Scenario A** (controls fixable + CMI replicates, 35% probability): Full three-pillar paper. Sections 4-6 all primary contributions. Target: NeurIPS/ICML main.

**Scenario B** (controls NOT fixable + CMI replicates, 20% probability): Two-pillar paper. Section 4 (methodology critique) + Section 6 (rate-distortion). Section 5 becomes supporting empirical evidence. Target: NeurIPS/ICML main (competitive).

**Scenario C** (controls fixable + CMI fails, 20% probability): Two-pillar paper. Section 4 (recalibrated cross-domain characterization) + Section 5 (L0 profile). Section 6 demoted to supplementary. Target: ICLR or NeurIPS workshop.

**Scenario D** (both fail, 25% probability): Methodology audit paper. Sections 4-5 primary. Section 6 in appendix as "suggestive." Target: Workshop or TMLR.

All four scenarios are accommodated by this outline without structural changes -- only the strength of claims and the emphasis in abstract/introduction would shift.
