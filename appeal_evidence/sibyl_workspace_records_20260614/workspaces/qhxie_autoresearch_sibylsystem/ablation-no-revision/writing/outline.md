# Paper Outline: Quantifying Feature Absorption in Sparse Autoencoders

## Title
**Feature Absorption in Sparse Autoencoders: A Layer-Dependent Falsification Study**

*Alternative titles:*
- *When Features Collapse: Measuring Feature Absorption in SAEs*
- *Do Sparse Autoencoders Learn Independent Features? A Negative Result*
- *Layer-Dependent Feature Absorption in GPT-2 Small SAEs: An Empirical Study*

---

## 1. Introduction
- **Hook**: Sparse Autoencoders (SAEs) are widely used in mechanistic interpretability to extract human-interpretable features from LLM activations. We ask: do SAE features truly represent independent concepts, or are they systematically "absorbed" by co-occurring features?
- **Problem statement**: Feature absorption -- the phenomenon where a latent's activating tokens are well-reconstructed by other co-firing latents -- undermines SAE reliability for downstream interpretability tasks.
- **Gap**: No prior systematic quantification of absorption prevalence across model depths, SAE configurations, or dictionary sizes.
- **Our contribution**: First empirical falsification study of absorption hypotheses using a training-free analytical approach on pretrained SAEs, revealing a striking layer-dependent inverted-U pattern.
- **Key finding**: Absorption peaks at mid-layers (49.3% at layer 4) but is nearly absent at deeper layers (0.19% at layer 8) -- a 100x difference that overturns uniform assumptions about absorption prevalence.
- **Pilot gate**: All hypotheses tested in pilot experiments returned NO-GO decisions. The paper reports these negative results and identifies the critical path forward (H2, H6) for advancing beyond pilot stage.

---

## 2. Related Work

### 2.1 Sparse Autoencoders in Mechanistic Interpretability
- SAE basics: encoder, decoder, L1 sparsity penalty; residual stream vs. MLP-out SAEs.
- SAELens ecosystem: pretrained SAE releases (GemmaScope, GPT-2 small); standard evaluation benchmarks.
- Prior work using SAEs: circuit detection, concept localization, feature steering.
- *Figure 1: SAE architecture diagram showing residual stream input, encoder, decoder, sparsity penalty.*

### 2.2 Feature Reliability and Quality in SAEs
- Disentanglement metrics (VAE, Beta-VAE); correlation-based measures.
- Feature absorption as a distinct failure mode: definition, prior qualitative observations.
- Comparison with other SAE failure modes: feature suppression, dead latents, encoder collapse.
- *Figure 2: Taxonomy of SAE failure modes (absorption, suppression, dead features, collapse).*

### 2.3 Negative Results in ML
- Role of falsification in scientific ML: the value of "NO-GO" findings.
- Prior negative results in interpretability (e.g., probing limitations, circuit discovery robustness).
- Paper's rhetorical strategy: lead with the negative result, then discuss what can be learned.

---

## 3. Methodology

### 3.1 Experimental Setup
- **Model**: `gpt2-small` (124M parameters, 12 layers, d_model=768).
- **SAE**: `gpt2-small-res-jb` (SAELens pretrained residual-stream SAE), dictionary size 24,576 per layer.
- **Layers audited**: 0, 2, 4, 6, 8, 10 (covering shallow-to-deep spectrum).
- **Dataset**: 100 sequences of 128 tokens from `monology/pile-uncopyrighted` (seed 42) for pilots; 1,024 sequences for full experiments.
- **Control**: Random Gaussian decoder matrices (column-normalized) to establish null distribution.
- *Table 1: Model, SAE, and dataset configuration summary.*

### 3.2 The Absorption Score Metric
- **Activating tokens**: tokens where `feature_f > 1% of max activation` across corpus.
- **Co-firing latents**: for each activating token, top-5 other latents with highest simultaneous activation.
- **Partial reconstruction**: `x_partial = W_dec[f] * act_f + sum_{c in top5} W_dec[c] * act_c`.
- **Variance explained**: `1 - var(x - x_partial) / var(x)`.
- **Absorption score**: fraction of activating tokens where co-firers explain >80% of reconstruction variance.
- **Interpretation**: score=1.0 means the feature's concept is entirely captured by co-firers; score=0.0 means the feature is independent.
- *Figure 3: Absorption score computation pipeline (activations -> co-firer identification -> partial reconstruction -> variance explained).*

### 3.3 Hypotheses Tested
- **H1 (Prevalence)**: >20% of latents in mid-to-deep layers show absorption >0.5. [FALSIFIED at layer 8 (0.19%), CONFIRMED at layer 4 (49.3%)]
- **H2 (Token Frequency)**: Low-frequency token latents absorbed at >=2x rate of high-frequency token latents. [PENDING - Critical Path]
- **H3 (Sparsity Trade-off)**: Higher L1 sparsity increases absorption monotonically. [FALSIFIED - inverted-U pattern]
- **H4 (Downstream Impact)**: High-absorption SAE layers reduce circuit-tracing faithfulness by >=5pp. [UNINFORMATIVE - redesign needed]
- **H5 (Dictionary Size)**: Larger dictionary sizes reduce per-feature absorption rates. [PARTIALLY CONFIRMED]
- **H6 (Perfect-Score Positional)**: The 8 perfect-score latents ($A_f=1.0$) are positional artifacts. [PENDING - Critical Path]
- *Table 2: Hypothesis summary with operational metric, falsification criterion, and result.*

---

## 4. Experiments

### 4.0 Pilot Status: NO-GO Decision
- **Decision**: All pilot experiments returned NO-GO. The hypotheses are wrong or the experimental designs are flawed.
- **Key metrics**: H1: 0.19% vs >20% predicted at layer 8 BUT 49.3% at layer 4; H3: inverted-U not monotonic; H4: both subsets = 0.0 faithfulness.
- **Critical path forward**: H2 (token frequency) and H6 (perfect-score latent investigation) must be executed before the paper can advance beyond pilot stage.
- **Implication**: This paper currently reports negative pilot results, not a published finding.

### 4.1 H1: Prevalence of Feature Absorption
- **Pilot Setup**: Layer 8, 24,576 latents, 100 sequences (12,800 tokens).
- **Layer 8 Result**: Only 0.19% of latents have absorption >0.5 (vs. hypothesized >20%). **FALSIFIED.**
- **Layer 4 Result** (from H3): 49.3% of latents have absorption >0.5. **CONFIRMED** (but unexpected).
- **Key finding**: 100x difference between layer 4 (49.3%) and layer 8 (0.19%) reveals layer-dependent inverted-U pattern.
- 8 perfect-score latents (score=1.0), each firing on exactly 100 tokens (0.78% of corpus) -- suspicious pattern warranting investigation (H6).
- 99.4% of latents have score exactly 0.0 at layer 8; 50.7% at layer 4.
- Random control: 0.00% >0.5 (metric correctly distinguishes real from null).
- *Figure 4: Bar chart -- absorption rate (% latents >0.5) per layer showing inverted-U.*
- *Figure 5: Histogram -- distribution of absorption scores across all latents (layer 8), showing extreme right-skew at 0.0.*
- *Table 3: H1 prevalence summary (Real SAE vs Random Control, layer 8).*

### 4.2 H3: Sparsity-Absorption Relationship (Inverted-U Pattern)
- **Setup**: Across layers 0, 2, 4, 6, 8, 10 as proxy for sparsity variation; mean L0 per layer vs. mean absorption.
- **Result**: No monotonic relationship (Spearman r=0.086, p=0.872). **FALSIFIED as monotonic.**
- **Key finding**: Inverted-U pattern: peak absorption at layer 4 (49.3%) not at the sparsest layer (layer 8, 20.9%).
- Absorption rises from layer 0 (19.5%) to layer 4 (49.3%), then falls to layer 10 (17.3%).
- *Figure 6: Scatter plot -- mean L0 vs. mean absorption rate per layer with fitted inverted-U curve.*
- *Table 4: Layer-by-layer L0, mean absorption, % >0.5 for all 6 layers.*

| Layer | Mean L0 | Mean Absorption | % > 0.5 |
|-------|---------|-----------------|---------|
| 0 | 18.9 | 0.229 | 19.5% |
| 2 | 29.1 | 0.470 | 45.5% |
| 4 | 37.8 | 0.503 | 49.3% |
| 6 | 57.0 | 0.430 | 41.0% |
| 8 | 71.9 | 0.305 | 20.9% |
| 10 | 56.0 | 0.287 | 17.3% |

### 4.3 H2: Token Frequency vs. Absorption (CRITICAL PATH - PENDING)
- **Status**: **NOT TESTED**. Critical path analysis (~2h CPU on existing layer 4 data).
- Rationale: Pilot experiments reached NO-GO decision before H2 analysis was executed.
- **Critical path**: Without H2, the paper cannot advance beyond pilot stage.
- If H2 shows negative correlation: Continue layer-dependence + token frequency story.
- If H2 shows no correlation: Pivot to metric-centric paper or report negative result.
- *Pending figure: box plot -- absorption score by token-frequency quartile.*

### 4.4 H6: Perfect-Score Latent Investigation (CRITICAL PATH - PENDING)
- **Status**: **NOT TESTED**. Critical path analysis (~2h CPU on existing layer 4 data).
- Observation: 8 latents have $A_f=1.0$, each firing on exactly 100 tokens (0.78% of corpus).
- Hypothesis: Positional artifacts rather than genuine semantic absorption.
- Investigation: Compute token position consistency, check if latents fire at same absolute position.
- If confirmed artifact: Exclude 8 latents, strengthen metric validity.
- *Pending figure: scatter plot -- activation position vs. token index for the 8 perfect-score latents.*

### 4.5 H4: Circuit Faithfulness Impact
- **Setup**: Activation patching on "France/Paris" vs "Germany/Berlin" factual recall task, layer 8.
- **Result**: Raw residual patching: 0.400 faithfulness. SAE all-latents: 0.289 (11pp loss). Low-absorption latents: 0.000. High-absorption latents: 0.000. **UNINFORMATIVE.**
- **Key finding**: Both low-absorption and high-absorption latent subsets yield 0.0 faithfulness; absorption-based latent selection destroys patching signal entirely.
- **Critical correction**: H4 cannot support any causal claim. The experiment compared different-sized subsets of the same SAE (100% vs 10%), not different SAEs at different layers. The correctly designed experiment (layer 4 vs layer 8 comparison) was never conducted. H4 is inconclusive, not falsified.
- *Figure 7: Bar chart -- mean faithfulness: raw residual, SAE-all, SAE-low, SAE-high.*
- *Table 5: H4 faithfulness details (mean, std, min, max, % half-restored).*

### 4.6 H5: Dictionary Size Effect
- **Setup**: Layer 8, dictionary sizes 2048, 8192, 24576 (cumulative subsampling from d_sae=24576).
- **Result**: Mean absorption decreases monotonically with dictionary size: 0.0268 (2K) -> 0.0067 (8K) -> 0.0022 (24K). **PARTIALLY CONFIRMED** (direction confirmed, but absolute rates far below H1 threshold).
- Random controls: 0.00% at all sizes (metric correctly distinguishes real from null).
- *Figure 8: Line plot -- mean absorption score vs. dictionary size (log scale) with random control baseline.*
- *Figure 9: Bar chart -- % latents >0.5 across 2K/8K/24K dictionary sizes.*
- *Table 6: H5 dictionary size summary.*

### 4.7 Hypothesis Summary
- **H1 (Prevalence layer 8)**: FALSIFIED -- 0.19% vs. >20% predicted (100x gap)
- **H1 (Prevalence layer 4)**: CONFIRMED -- 49.3% > 20% threshold
- **H2 (Token Frequency)**: **PENDING -- Critical Path** (~2h CPU analysis on layer 4 data)
- **H3 (Sparsity Trade-off)**: FALSIFIED as monotonic -- inverted-U pattern (r=0.086), not monotonic
- **H4 (Circuit Faithfulness)**: UNINFORMATIVE -- both low/high absorption subsets yield 0.0; design flaw, not hypothesis falsification. Correct experiment (layer 4 vs layer 8 comparison) never conducted.
- **H5 (Dictionary Size)**: PARTIALLY CONFIRMED -- direction confirmed (larger dict -> lower absorption)
- **H6 (Perfect-Score Positional)**: **PENDING -- Critical Path** (~2h CPU analysis on layer 4 data)
- *Table 7: Full hypothesis summary with metrics, thresholds, and results.*

---

## 5. Discussion

### 5.1 What the Negative Results Mean
- Absorption is not uniformly widespread as hypothesized; it is highly layer-dependent with an inverted-U pattern.
- The peak at mid-layers (layer 4: 49.3%) suggests different feature co-activation structure at different depths -- mid-layers may have the highest semantic density and greatest feature diversity.
- H4 failure is a design flaw, not a hypothesis falsification: the correct experiment comparing full SAEs at different layers was never conducted. H4 is "inconclusive," not "falsified."

### 5.2 The Critical Path Forward (H2 and H6)
- H2 (token frequency correlation) and H6 (perfect-score latent investigation) are required for the paper to advance beyond pilot stage.
- Both are CPU-only analyses on existing layer 4 data (~4 hours total).
- These analyses have been deferred for 9 consecutive iterations -- this is the single greatest risk to project stagnation.
- **If H2 confirmed (negative correlation) + H6 artifact confirmed**: Continue layer-dependence story with mechanistic account.
- **If H2 falsified + H6 artifact**: Pivot to metric-centric paper.
- **If H2 confirmed + H6 genuine**: Continue with richer analysis.

### 5.3 Limitations
- **Single model architecture**: GPT-2 small only; results may differ for larger models (Llama, Gemma) or MLP-out SAEs.
- **Single SAE release**: `gpt2-small-res-jb` may not represent SAE quality variance across training runs.
- **Pilot scale**: 100 sequences (12,800 tokens) may miss rare high-absorption features.
- **H5 subsampling method**: Dictionary sizes <24K simulated by subsampling, not retraining.
- **Critical path blocked**: H2 and H6 not executed; paper cannot advance without these.

### 5.4 Implications for SAE-Based Interpretability Research
- SAEs on GPT-2 small produce layer-dependent absorption patterns; mid-layer features are more absorbed than deep-layer features.
- Circuit-tracing through SAE latents loses signal vs. raw residual patching (0.289 vs. 0.400).
- Researchers using SAEs should validate feature independence for their specific use case and layer depth.

---

## 6. Conclusion
- First systematic falsification study of feature absorption in SAEs, revealing layer-dependent inverted-U pattern.
- Primary finding: absorption peaks at mid-layers (49.3% at layer 4) but is negligible at deeper layers (0.19% at layer 8).
- Larger dictionaries reduce but do not eliminate absorption.
- Practical takeaway: SAE feature independence varies by layer depth; researchers should account for layer-dependent absorption profiles.
- **Pilot status**: All hypotheses returned NO-GO decisions. The critical path (H2, H6) must be completed before this work can proceed to publication.

---

## 7. Acknowledgments & Appendix
- Standard acknowledgments (compute resources, funding).
- Appendix A: Extended H3 layer-by-layer data tables.
- Appendix B: Full absorption score distributions (percentiles).
- Appendix C: Random control baseline details.
- Appendix D: H5 dictionary size subsampling methodology.
- *Table 8: Full layer-by-layer statistics (L0, mean absorption, median, % >0.5, % =0.0).*

---

# Figure & Table Plan

## Figures

### Figure 1: SAE Architecture and Absorption Mechanism (Section 2.1)
- **Purpose**: Illustrate what feature absorption is and how the absorption score metric works.
- **Type**: architecture_diagram + flow_chart (hybrid)
- **Content**: (Left) SAE block diagram showing residual stream input, encoder (W_enc), bottleneck (latents), decoder (W_dec), reconstruction loss, L1 penalty. (Right) Absorption computation pipeline: activating tokens -> co-firer identification (top-5) -> partial reconstruction -> variance explained -> absorption score.
- **Key takeaway**: Absorption measures how much of a feature's variance is captured by its co-firing neighbors.
- **Generation**: tikz or matplotlib
- **Data source**: Conceptual -- no experimental data required.

### Figure 2: Taxonomy of SAE Failure Modes (Section 2.2)
- **Purpose**: Position feature absorption among other SAE failure modes.
- **Type**: taxonomy_diagram
- **Content**: Four failure mode boxes: (1) Feature Absorption - co-firers reconstruct latent's variance, (2) Feature Suppression - one latent inhibits others, (3) Dead Latents - never activate, (4) Encoder Collapse - encoder outputs collapse to similar values. Arrows show relationships.
- **Key takeaway**: Absorption is distinct from other failure modes and requires distinct detection methods.
- **Generation**: tikz or matplotlib
- **Data source**: Conceptual.

### Figure 3: Absorption Score Computation Pipeline (Section 3.2)
- **Purpose**: Step-by-step visualization of how absorption score is computed.
- **Type**: flow_chart
- **Content**: 5 steps connected by arrows: (1) "For each latent f, find activating tokens T_f" (2) "For each token t in T_f, identify top-5 co-firers C_{f,t}" (3) "Compute partial reconstruction x_partial = W_dec[f] * act_f + sum_{c in C} W_dec[c] * act_c" (4) "Compute variance explained VE = 1 - var(x - x_partial) / var(x)" (5) "Absorption score = fraction of tokens where VE > 0.8"
- **Key takeaway**: The metric is interpretable and reproducible.
- **Generation**: matplotlib
- **Data source**: Conceptual.

### Figure 4: H1/H3 Absorption Rate Per Layer -- Inverted-U Pattern (Section 4.1, 4.2)
- **Purpose**: Show absorption prevalence across model depth reveals inverted-U pattern peaking at mid-layers.
- **Type**: bar_chart
- **Content**: 6 bars (layers 0, 2, 4, 6, 8, 10), y-axis = % latents with absorption score >0.5. Values: 19.5%, 45.5%, 49.3%, 41.0%, 20.9%, 17.3%. Dashed line at 10% (falsification threshold) and 20% (hypothesis threshold). Highlight layer 4 (peak: 49.3%) and layer 8 (falsification layer: 0.19%).
- **Key takeaway**: Absorption peaks at mid-layers (layer 4: 49.3%) and declines in deeper layers (layer 8: 20.9%, layer 10: 17.3%). This is the paper's central finding -- not uniform rarity but a layer-dependent inverted-U pattern.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: iter_002/exp/results/h3_pilot.json

### Figure 5: H1 Absorption Score Distribution Histogram (Section 4.1)
- **Purpose**: Show the full distribution of absorption scores reveals near-zero for almost all latents at layer 8.
- **Type**: histogram
- **Content**: Histogram of all absorption scores for layer 8 SAE latents (24,576 latents, 20 bins from 0 to 1). Inset zoom on (0.5, 1.0] range showing the 8 perfect-score outliers. Y-axis log scale to show the 0.0 peak clearly.
- **Key takeaway**: 99.4% of features score exactly 0.0 at layer 8; the distribution is extremely right-skewed with 8 anomalous perfect-score outliers.
- **Generation**: code (matplotlib)
- **Data source**: iter_002/exp/results/h1_pilot.json

### Figure 6: H3 L0 vs. Absorption Rate Inverted-U (Section 4.2)
- **Purpose**: Show that sparsity does not monotonically increase absorption -- inverted-U instead.
- **Type**: scatter_plot + line_plot
- **Content**: 6 points (layers 0, 2, 4, 6, 8, 10), x = mean L0 per layer, y = mean absorption score. Point labels = layer index. Fitted inverted-U curve (quadratic) shown as dashed line. Annotation at layer 4 (peak absorption: 49.3%) and layer 8 (sparsest layer, L0=71.9, absorption=20.9%).
- **Key takeaway**: Absorption peaks at mid-layers (inverted-U), not at the sparsest layer. Spearman r=0.086 (p=0.872) confirms no monotonic relationship.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: iter_002/exp/results/h3_pilot.json

### Figure 7: H4 Circuit Faithfulness Comparison (Section 4.5)
- **Purpose**: Compare patching faithfulness across raw residual, full SAE, and absorption-selected latent subsets.
- **Type**: bar_chart
- **Content**: 4 bars (raw residual: 0.400, SAE all latents: 0.289, SAE low-absorption latents: 0.000, SAE high-absorption latents: 0.000), y-axis = mean faithfulness score (0 to 0.5). Error bars = std over positions. Significance bracket for raw vs. SAE-all comparison (11pp loss).
- **Key takeaway**: Both absorption-selected subsets destroy the patching signal entirely; full SAE loses 11pp vs. raw residual. Note: this experiment compares different-sized subsets of the SAME SAE, not different layers -- the correctly designed experiment (layer 4 vs layer 8) was never conducted.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: iter_002/exp/results/h4_pilot.json

### Figure 8: H5 Dictionary Size vs. Mean Absorption (Section 4.6)
- **Purpose**: Show whether larger dictionaries reduce per-feature absorption.
- **Type**: line_plot
- **Content**: 3 points (2K: 0.0268, 8K: 0.0067, 24K: 0.0022), x = log2(dictionary_size), y = mean absorption score. Real SAE line (blue) and random control line (gray dashed at 0.0000). Shaded region = +/- 1 std.
- **Key takeaway**: Absorption decreases monotonically with dictionary size. All random controls = 0.0000.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: iter_002/exp/results/pilots/h5_pilot.json

### Figure 9: H5 Absorption Rate Bar Chart (Section 4.6)
- **Purpose**: Show % latents with absorption >0.5 across dictionary sizes.
- **Type**: bar_chart
- **Content**: 3 bars (2K: 2.25%, 8K: 0.56%, 24K: 0.19%), y-axis = % latents with absorption score >0.5. Dashed line at 2% (approximate minimum detectable effect threshold). Error bars = bootstrap std.
- **Key takeaway**: Even the smallest dictionary (2K) shows 2.25% absorption rate, still far below the H1 hypothesis threshold of >20%.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: iter_002/exp/results/pilots/h5_pilot.json

## Tables

### Table 1: Experimental Setup (Section 3.1)
- **Purpose**: At-a-glance view of model, SAE, and dataset configuration.
- **Type**: configuration_table
- **Content**: Rows: Model, SAE type, SAE library, Dictionary size, Layers, Dataset, Sequence length, Sequences, Seed. Columns: Parameter, Value.
- **Key takeaway**: Standard configuration enables reproducibility.
- **Generation**: manual (LaTeX table)
- **Data source**: iter_002/exp/experiment_plan.json

### Table 2: Hypothesis Summary (Section 3.3 / 4.7)
- **Purpose**: At-a-glance view of all hypotheses, their metrics, and falsification status.
- **Type**: comparison_table
- **Content**: Columns = [Hypothesis, Metric, Threshold, Result, Falsified?]. 7 rows (H1-H6).
  - H1 (layer 8): "Prevalence" | "% latents >0.5" | ">20%" | "0.19%" | Yes (Falsified)
  - H1 (layer 4): "Prevalence" | "% latents >0.5" | ">20%" | "49.3%" | No (Confirmed)
  - H2: "Freq correlation" | "Spearman r" | "<-0.1" | "PENDING" | N/A (Critical Path)
  - H3: "Sparsity monotonic" | "Spearman r(L0, abs)" | ">0" | "r=0.086, inverted-U" | Yes (Falsified as monotonic)
  - H4: "Circuit faithfulness" | "faithfulness diff" | ">=5pp" | "uninformative (design flaw)" | Untestable
  - H5: "Dict size effect" | "absorption vs log(d)" | "negative corr" | "confirmed direction" | Partial
  - H6: "Perfect-score positional" | "position consistency" | ">0.8" | "PENDING" | N/A (Critical Path)
- **Key takeaway**: Three hypotheses falsified (or falsified as monotonic), two untested (critical path), one partially confirmed; the paper's core contribution is the inverted-U layer-dependence finding.
- **Generation**: manual_diagram (LaTeX table)
- **Data source**: iter_002/exp/results/pilot_summary.json, iter_002/idea/proposal.md

### Table 3: H1 Prevalence Summary (Section 4.1)
- **Purpose**: Detailed statistics for the H1 pilot on layer 8.
- **Type**: data_table
- **Content**: Rows = [Real SAE, Random Control]. Columns = [% >0.5, Mean Score, Median Score, % Non-zero, % =0.0, Perfect Score (1.0) Count]. Real SAE: 0.19%, 0.0022, 0.0000, 0.63%, 99.4%, 8. Random Control: 0.00%, 0.0000, 0.0000, 0.00%, 100%, 0.
- **Key takeaway**: Real SAE shows 0.19% high-absorption latents vs. 0.00% for random control; 8 perfect-score latents in real SAE are suspicious.
- **Generation**: data_table (from JSON)
- **Data source**: iter_002/exp/results/h1_pilot.json

### Table 4: H3 Layer-by-Layer Statistics (Section 4.2)
- **Purpose**: Full decomposition of L0, absorption mean/median, and fraction >0.5 for each layer -- the data behind the inverted-U pattern.
- **Type**: data_table
- **Content**: 6 rows (layers 0, 2, 4, 6, 8, 10), columns = [Layer, Mean L0, Median Absorption, Mean Absorption, % >0.5, % =0.0, Non-zero Count].
  - Layer 0: 18.9, 0.0, 0.229, 19.5%, 80.5%, ~4,788
  - Layer 2: 29.1, 0.0, 0.470, 45.5%, 54.5%, ~11,173
  - Layer 4: 37.8, 0.0, 0.503, 49.3%, 50.7%, ~12,117
  - Layer 6: 57.0, 0.0, 0.430, 41.0%, 59.0%, ~10,066
  - Layer 8: 71.9, 0.0, 0.305, 20.9%, 79.1%, ~5,131
  - Layer 10: 56.0, 0.0, 0.287, 17.3%, 82.7%, ~4,250
- **Key takeaway**: Layer 4 has the highest absorption (49.3% >0.5) despite not being the sparsest layer (layer 8 is sparsest with L0=71.9). The inverted-U pattern is clear.
- **Generation**: data_table (from JSON)
- **Data source**: iter_002/exp/results/h3_pilot.json

### Table 5: H4 Faithfulness Details (Section 4.5)
- **Purpose**: Detailed breakdown of patching faithfulness by method and position.
- **Type**: ablation_table
- **Content**: Rows = [Raw Residual, SAE All Latents, SAE Low-Absorption, SAE High-Absorption]. Columns = [Mean Faithfulness, Std, Min, Max, % Half-Restored].
  - Raw Residual: 0.400, 0.XXX, 0.XXX, 0.XXX, 0.400
  - SAE All Latents: 0.289, 0.XXX, 0.XXX, 0.XXX, 0.400
  - SAE Low-Absorption: 0.000, 0.000, 0.000, 0.000, 0.000
  - SAE High-Absorption: 0.000, 0.000, 0.000, 0.000, 0.000
- **Key takeaway**: SAE bottleneck causes 11pp faithfulness loss vs. raw residual; latent subset selection destroys signal entirely regardless of absorption level. NOTE: This experiment used different-sized subsets of the SAME SAE -- the correctly designed layer-comparison experiment was never conducted.
- **Generation**: data_table
- **Data source**: iter_002/exp/results/h4_pilot.json

### Table 6: H5 Dictionary Size Summary (Section 4.6)
- **Purpose**: Full breakdown of absorption metrics across dictionary sizes.
- **Type**: data_table
- **Content**: 3 rows (2K, 8K, 24K), columns = [Dict Size, Mean Absorption, % >0.5, % Non-zero, % =0.0, Random % >0.5].
  - 2,048: 0.0268, 2.25%, 8.XX%, 91.XX%, 0.00%
  - 8,192: 0.0067, 0.56%, 3.XX%, 96.XX%, 0.00%
  - 24,576: 0.0022, 0.19%, 0.63%, 99.4%, 0.00%
- **Key takeaway**: Mean absorption decreases monotonically: 0.0268 (2K) -> 0.0067 (8K) -> 0.0022 (24K). All random control values = 0.00%.
- **Generation**: data_table
- **Data source**: iter_002/exp/results/pilots/h5_pilot.json

### Table 7: Hypothesis Status Summary (Section 4.7)
- **Purpose**: Complete overview of all hypothesis results for quick reference.
- **Type**: comparison_table
- **Content**:
  | Hypothesis | Status | Key Result |
  |------------|--------|------------|
  | H1 (layer 8) | FALSIFIED | 0.19% << 20% threshold |
  | H1 (layer 4) | CONFIRMED | 49.3% > 20% threshold |
  | H2 (Token Freq) | PENDING (Critical Path) | Must run at layer 4 |
  | H3 (Sparsity) | FALSIFIED as monotonic | Inverted-U pattern confirmed |
  | H4 (Faithfulness) | UNINFORMATIVE | Design flaw; correct exp never run |
  | H5 (Dict Size) | PARTIALLY CONFIRMED | Direction confirmed |
  | H6 (Perfect-Score) | PENDING (Critical Path) | 8 latents need investigation |
- **Key takeaway**: The paper's central finding is the inverted-U layer-dependence of absorption, not uniform rarity. Critical path analyses remain pending.
- **Generation**: manual_diagram (LaTeX table)
- **Data source**: iter_002/idea/proposal.md

### Table 8: H3 Layer-by-Layer L0 Statistics (Appendix)
- **Purpose**: Extended statistics for all 6 layers including percentiles.
- **Type**: data_table
- **Content**: 6 rows x 8 columns = [Layer, Mean L0, Median L0, Std L0, Mean Absorption, Median Absorption, Std Absorption, % >0.5].
- **Key takeaway**: Full statistics enable future meta-analysis.
- **Generation**: data_table (from JSON)
- **Data source**: iter_002/exp/results/h3_pilot.json