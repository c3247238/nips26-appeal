# Paper Outline: Feature Absorption in Sparse Autoencoders

**Working title:** Feature Absorption in Sparse Autoencoders is a Sparsity Landscape Problem: Encoder Weight Norm as a Weight-Only Detection Heuristic

**Venue target:** NeurIPS 2026 Mechanistic Interpretability Workshop (primary); NeurIPS 2026 main track (stretch)

**Status:** Based on confirmed results from iter_003 experiments (A1, A2, A3, B1, B2, C2, F1)

---

## Narrative Arc

The paper makes three interlocking claims:

1. The dominant causal explanation for SAE feature absorption is the sparsity landscape (partial minimum during training), **not** the encoder amortization gap — established by OMP oracle experiment (C2).
2. Encoder weight norm (`||W_enc_j||`) is the best weight-only proxy for detecting absorbed features — AUROC=0.757 (GPT-2 L6, Standard/L1) and 0.837 (TopK-32k), significantly exceeding EDA (AUROC=0.650) — established by experiments A1, A3.
3. Co-occurrence structure (O_jaccard) provides independent, complementary detection signal (AUROC=0.721, AUPRC=0.075), enabling a dual-signal audit approach — established by B1, B2.
4. Supporting: dictionary width partially but incompletely remediates absorption — 12/18 (67%) absorbed features have counterparts in wider SAE, 6/18 (33%) do not — established by F1.

**Lead with H2 falsification** (OMP oracle) as the primary contribution; encoder_norm detection as the enabling methodology; width recovery as a practical implication.

---

## Section Outline

### 1. Abstract (250 words)

- State the mechanistic question: is absorption caused by encoder amortization gap or sparsity landscape?
- State the experimental approach: OMP oracle with fixed decoder dictionary at matched sparsity
- State the decisive finding: OMP achieves 0% absorption reduction vs. feedforward encoder, falsifying amortization gap dominance
- State the enabling detection result: encoder weight norm predicts absorbed latents (AUROC=0.757–0.837), outperforming EDA
- State the practical implication: 33% of absorbed features require training-objective changes, not just wider dictionaries

### 2. Introduction (1–1.5 pages)

Key content:
- Feature absorption: operational definition (parent features fail to fire when child features absorb activation budget), measurement tool (Chanin et al. IG pipeline), prevalence (15–35% first-letter task on Gemma Scope 16k/65k)
- Motivation: SAE-based mechanistic interpretability is disrupted by absorption; DeepMind safety deprioritization as highest-stakes consequence
- Two competing mechanistic explanations with opposite implications for practitioners:
  - O'Neill et al. (2411.13117): amortization gap — iterative encoders should fix absorption
  - Tang et al. (2512.05534): sparsity landscape (partial minimum) — only training-objective or dictionary changes can fix it
- The controlled experiment that resolves this debate: fix pre-trained decoder, compare feedforward vs. OMP encoding at matched L0
- Preview of results: OMP = 0% reduction (sparsity landscape supported); encoder_norm AUROC=0.757+ as weight-only detector; width expansion partial but insufficient
- Figure 1 (teaser): Key result panel — absorption rate under feedforward vs. OMP (bar chart) + encoder_norm ROC curve

Transition: "Before presenting the mechanistic test, we introduce the encoder weight norm detector that makes absorption auditing scalable."

### 3. Background and Related Work (0.75 pages)

Key content:
- SAE training objective and sparsity penalty (L1 or TopK)
- Feature absorption operational definition and Chanin IG measurement pipeline
- Prior mitigation proposals: OrtSAE, Matryoshka SAE, ATM SAE, Masked Regularization
- Amortization gap theory (O'Neill et al.): feedforward encoder approximates, but never matches, optimal sparse inference
- Partial minimum theory (Tang et al.): absorption is a stable spurious minimum of the piecewise biconvex SDL loss
- EDA detector (iter_001/002): AUROC=0.650 baseline; limitation for early-absorption (decoder-absent)
- Table: summary of mechanistic hypotheses and their implications for mitigation

Transition: "Against this backdrop, we design the first controlled experiment separating these two mechanistic accounts."

### 4. Method (1.5–2 pages)

#### 4.1 Encoder Weight Norm Detector

Key content:
- Definition: `enc_norm(j) = ||W_enc_j||` where `W_enc_j` is the j-th row of the encoder weight matrix
- Mechanistic motivation: gradient competition during training inflates encoder weights of features that must compete with absorbing children; absorbed features receive elevated gradient signal when absorbing children interfere with their reconstruction target
- Computation: weight-only (no activation data required), O(d_SAE) per SAE
- Comparison to EDA: EDA measures angular divergence between encoder and decoder directions (requires both); encoder_norm requires only encoder weights
- Figure 2: Histogram of encoder_norm for absorbed vs. non-absorbed latents at GPT-2 L6; mean absorbed=3.26 vs. overall mean=2.58; Cohen's d=0.971

#### 4.2 Jaccard Co-occurrence Detector

Key content:
- Definition: `O_jaccard(j)` = maximum Jaccard overlap between feature j's activation set and any higher-frequency feature's activation set
- Complementarity: Spearman ρ(encoder_norm, O_jaccard) = 0.044 — near-zero, confirms independent information
- Practical use: O_jaccard has dramatically higher AUPRC (0.075 vs. 0.004) due to different score distribution; use for ranking top-k candidates

#### 4.3 Amortization Gap Controlled Experiment

Key content:
- Setup: GPT-2-small L6 SAE; fixed pre-trained decoder W_dec (24k × 768); vary encoding method at matched L0
- Encoding conditions:
  - A (Feedforward): standard ReLU encoder `z = ReLU(W_enc @ x + b_enc)`, mean L0=55.7
  - B (OMP oracle): Orthogonal Matching Pursuit at K=53 (matched mean L0); reconstruction error 0.219 < feedforward 0.242
- Absorption measurement: Chanin IG main-feature activation check on letters {a, e, s}; 30 tokens per letter
- Pre-committed falsification criterion (from proposal): OMP >= 80% of feedforward absorption rate → amortization gap falsified
- Figure 3: Architecture diagram — fixed decoder vs. varied encoder; absorption measurement pipeline

Transition: "We now report the outcome of all three contributions."

### 5. Experiments (2.5–3 pages)

#### 5.1 Experimental Setup

Key content:
- Model: GPT-2-small (117M), SAEs from Bloom et al. (gpt2-small-res-jb, Standard/L1) and EleutherAI (gpt2-small-resid-post-v5-32k, TopK-32k) via SAELens
- Gold labels: Chanin IG FeatureAbsorptionCalculator at GPT-2 L6; n_pos=18, n_neg=24,558 (Standard); n_pos=77, n_neg=32,691 (TopK-32k)
- Dataset: OpenWebText (10k tokens for co-occurrence; 30 tokens per letter for absorption)
- Metrics: AUROC (primary), AUPRC, Precision@k, DeLong test for pairwise AUROC comparison

#### 5.2 Encoder Norm Replication and Cross-Architecture Validation (Tasks A1, A3)

Key content:
- Table 1: Detector AUROC Comparison — rows: Random, activation_freq, EDA, encoder_norm (ours), O_jaccard (ours); columns: GPT-2 L6 Standard (n_pos=18), GPT-2 L6 TopK-32k (n_pos=77)
- L6 Standard: encoder_norm AUROC=0.757 [0.655, 0.849] > EDA AUROC=0.650 [0.526, 0.774]; DeLong z=3.046, p=0.0012
- L6 TopK-32k: encoder_norm AUROC=0.837 [0.807, 0.870], Cohen's d=1.235
- Hook confound: Standard uses resid_pre, TopK uses resid_post; claim is "encoder_norm works on both" not "AUROC values are directly comparable"
- Layer analysis (A2): encoder_norm signal peaks at L6 (absorbed/non-absorbed ratio=1.267); weaker at L2 (0.877), L4 (1.055), L8 (0.891), L10 (0.933)
- Figure 4: ROC curves for all detectors on L6 Standard and TopK-32k (two-panel plot)

#### 5.3 Co-occurrence Structure as Independent Signal (Tasks B1, B2)

Key content:
- O_jaccard AUROC=0.721 [0.604, 0.843]; AUPRC=0.075; Precision@50=0.10
- Spearman ρ(encoder_norm, O_jaccard)=0.044 — confirms independence
- ARS_v2 (encoder_norm × A_cooccur) AUROC=0.586 — does NOT improve over encoder_norm alone; DeLong z=-2.455, p=0.993
- Figure 5: Ablation bar chart — AUROC for each component and combination; highlights encoder_norm and O_jaccard as the two valid signals

#### 5.4 Amortization Gap Oracle Experiment (Tasks C1, C2)

Key content:
- Feedforward AR=0.978; OMP AR=0.978; omp_reduction_ratio=0.0 for all three letters (a, e, s)
- OMP technical validity: reconstruction error OMP=0.219 < feedforward=0.242 (oracle is genuinely more optimal)
- Pre-committed criterion met: OMP >= 80% of feedforward rate (100% of feedforward rate); amortization gap hypothesis falsified in strong form
- Interpretation: absorption is set at training time by the dictionary structure; the encoder approximation quality does not determine whether absorption occurs
- Figure 6: Box/bar plot — absorption rate per letter under feedforward vs. OMP (paired; y-axis absorption rate; x-axis condition)

#### 5.5 Dictionary Width Recovery Analysis (Task F1)

Key content:
- 12/18 (67%) absorbed features in 24k SAE have direction-aligned counterpart in 32k SAE (cos_sim > 0.80)
- mean_best_cosine=0.791, median=0.815
- 6/18 (33%) have no close match — genuine semantic coverage gap not addressable by capacity alone
- enc_norm of recovered vs. not-recovered features: recovered mean=3.29 vs. not-recovered mean=3.21 (no significant difference)
- Table 2: Width recovery summary — n_recovered, frac_recovered, mean/median cosine, inference for practitioners

Transition: "Together these results support a unified picture of absorption as a training-time sparsity landscape phenomenon."

### 6. Discussion (1–1.5 pages)

Key content:
- Mechanistic synthesis: OMP 0% reduction + Tang et al. theory + F1 partial recovery = absorption is embedded in dictionary geometry at training time, not in encoder inference quality
- Gradient competition interpretation of encoder_norm: during training, absorbed features receive elevated gradient signal from loss terms where absorbing children interfere; this drives ||W_enc_j|| upward
- Implications for mitigation: iterative encoders (MP-SAE, OMP at inference) will NOT fix absorption; architectural changes (Matryoshka, OrtSAE) or training-objective changes needed
- Implications for detection: encoder_norm is a practical pre-screening tool; <1 second for 65k SAE; enables audit-at-scale before activation data collection
- Limitations:
  - n_pos=18 for L6 gold labels (bootstrap CI ±0.09)
  - Hook confound in cross-architecture comparison (resid_pre vs. resid_post)
  - Single-task domain (first-letter spelling); cross-hierarchy measurement failed due to probe methodology (future work)
  - OMP falsification assumes fixed K; alternative: LASSO at very small λ
- Table 3: Implications for mitigation approaches — what each result says for practitioners (which fixes help, which don't)

### 7. Related Work (0.5 pages)

Key content:
- Absorption measurement: Chanin et al. (2024), SAEBench (Karvonen et al., 2025)
- Absorption theory: Tang et al. (2512.05534), O'Neill et al. (2411.13117)
- Architectural mitigations: OrtSAE, Matryoshka SAE, ATM SAE, Masked Regularization (arXiv:2604.06495)
- Co-occurrence and geometry: Geometry of Concepts (arXiv:2410.19750)
- Prior detection: EDA (iter_001/002 finding); "Resurrecting the Salmon" (arXiv:2508.09363)
- Positioning: our paper is the first empirical mechanistic test separating encoder vs. dictionary causes; encoder_norm complements Chanin IG by being weight-only

### 8. Conclusion (0.25 pages)

Key content:
- Summary: absorption is a sparsity landscape problem; OMP oracle is decisive evidence; encoder_norm is the best weight-only detector
- Call to action: SAE training objectives must be modified; iterative encoder improvements are insufficient
- Future work: cross-hierarchy absorption measurement; functional recovery validation; safety attribution analysis

---

## Figure & Table Plan

### Figure 1: Teaser — Key Results Panel (Section: Introduction)

- **Purpose**: Communicate the two primary results in a single glanceable figure before methods are introduced
- **Type**: two-panel composite (bar_chart + line_plot)
- **Content**:
  - Left panel: absorption rate under feedforward vs. OMP for letters {a, e, s}; horizontal line at "feedforward baseline"; OMP bars identical to feedforward bars (0% reduction)
  - Right panel: ROC curve for encoder_norm vs. EDA vs. random at GPT-2 L6; show AUROC=0.757 vs. 0.650 vs. 0.50
- **Key takeaway**: OMP does not reduce absorption; encoder_norm is the best detector
- **Generation**: matplotlib (bar_chart + ROC curve); consistent color scheme
- **Data source**: `exp/results/full/C2_amortization_gap_full.json` (absorption rates); `exp/results/full/A3_encoder_norm_cross_arch.json` (AUROC values)

### Figure 2: Encoder Norm Geometric Explanation (Section: Method)

- **Purpose**: Illustrate that absorbed features have systematically higher encoder weight norms
- **Type**: dual histogram with overlapping distributions + vertical mean lines
- **Content**:
  - Histogram of encoder_norm for absorbed latents (n=18) vs. random sample of non-absorbed latents (n=500) at GPT-2 L6
  - Vertical lines at absorbed mean=3.26 and overall mean=2.58
  - Inset: theoretical diagram showing gradient competition mechanism (absorbed feature j receives gradient from residual where absorbing child fires instead)
  - Cohen's d=0.971 annotated
- **Key takeaway**: absorbed features have higher encoder norms; this is detectable from weights alone
- **Generation**: matplotlib (histograms); seaborn KDE overlay
- **Data source**: `exp/results/full/A2_encoder_norm_theory.json` (per-feature encoder_norm values)

### Figure 3: Amortization Gap Experiment Architecture (Section: Method)

- **Purpose**: Visually explain the controlled experiment design (fixed decoder, varied encoder)
- **Type**: architecture diagram / flow chart
- **Content**:
  - Left branch: Feedforward encoder → sparse codes → decoder (fixed) → reconstruction
  - Right branch: OMP oracle → sparse codes → same decoder (fixed) → reconstruction
  - Red box: "Absorption measured here" (whether parent feature fires)
  - Annotation: "K=53 (matched mean L0)" and "reconstruction error: OMP=0.219 < FF=0.242"
- **Key takeaway**: the only thing that changes between conditions is the encoder; decoder is held constant
- **Generation**: tikz or matplotlib patches
- **Data source**: methodology description

### Figure 4: ROC Curves — All Detectors (Section: Experiments 5.2)

- **Purpose**: Show AUROC comparison across all detectors on both architectures
- **Type**: line_plot (ROC curves), two-panel
- **Content**:
  - Left panel (GPT-2 L6 Standard/L1, n_pos=18): encoder_norm, O_jaccard, EDA, activation_freq, random
  - Right panel (GPT-2 L6 TopK-32k, n_pos=77): encoder_norm, EDA, random
  - Shaded CI bands for each curve (bootstrap CI from JSON)
  - AUC values annotated in legend
- **Key takeaway**: encoder_norm consistently best; O_jaccard competitive; EDA is a weaker baseline
- **Generation**: matplotlib; consistent color palette across panels
- **Data source**: `exp/results/full/A3_encoder_norm_cross_arch.json`, `exp/results/full/B2_ars_v2_validation.json`

### Figure 5: ARS Component Ablation (Section: Experiments 5.3)

- **Purpose**: Show which components of the absorption score contribute vs. dilute signal
- **Type**: horizontal bar_chart (AUROC per method)
- **Content**:
  - Bars for: random (0.50), ARS_original (0.528), ARS_full (0.528), ARS_v2 (0.586), A_cooccur (0.584), EDA (0.650), O_jaccard (0.721), encoder_norm (0.757)
  - Sorted ascending; color-coded (grey=baselines, blue=co-occurrence, orange=encoder geometry, red=EDA)
  - Error bars from bootstrap CI
  - Annotations: "H_ARS failed — product formulation hurts" at ARS_v2
- **Key takeaway**: encoding geometry (encoder_norm) and co-occurrence (O_jaccard) are the two valid signals; product combinations dilute signal
- **Generation**: matplotlib (horizontal barh)
- **Data source**: `exp/results/full/B2_ars_v2_validation.json`

### Figure 6: Amortization Gap Oracle — Per-Letter Results (Section: Experiments 5.4)

- **Purpose**: Show the decisive 0% absorption reduction under OMP encoding
- **Type**: grouped bar_chart or dot plot, with jitter
- **Content**:
  - X-axis: letters {a, e, s}; Y-axis: absorption rate (0–1)
  - Two bars per letter: Feedforward (grey) and OMP (orange)
  - All pairs are identical (AR=0.967, 1.0, 0.967 for both conditions)
  - Horizontal dashed line at feedforward mean=0.978
  - Caption note: "Pre-committed falsification criterion: OMP >= 0.80 × feedforward. Observed: OMP = 1.000 × feedforward."
- **Key takeaway**: OMP (optimal sparse encoder) does not reduce absorption at all; amortization gap is not the cause
- **Generation**: matplotlib
- **Data source**: `exp/results/full/C2_amortization_gap_full.json`

### Table 1: Detector AUROC Comparison (Section: Experiments 5.2)

- **Purpose**: Primary results comparison across detectors and architectures
- **Type**: comparison_table
- **Content**:
  - Columns: Method | Computation | GPT-2 L6 Standard AUROC [95% CI] | AUPRC | GPT-2 L6 TopK AUROC [95% CI] | AUPRC
  - Rows: Random, activation_freq_inv, EDA (baseline), encoder_norm (ours), O_jaccard (ours), ARS_v2 (ours)
  - Bold the best result per column
  - Footnote: "Standard uses resid_pre hook; TopK uses resid_post hook — AUROC magnitudes not directly comparable across architectures (see Section 5.2)"
- **Data source**: `exp/results/full/A3_encoder_norm_cross_arch.json`, `exp/results/full/B2_ars_v2_validation.json`

### Table 2: Amortization Gap Experiment Results (Section: Experiments 5.4)

- **Purpose**: Quantitative summary of OMP oracle experiment
- **Type**: ablation_table
- **Content**:
  - Columns: Encoding Method | Mean L0 | Recon Error | AR(a) | AR(e) | AR(s) | Mean AR | Reduction vs. FF
  - Rows: Feedforward (A), OMP oracle (B)
  - Bold "Reduction vs. FF" column showing 0.0% for all letters
- **Data source**: `exp/results/full/C2_amortization_gap_full.json`

### Table 3: Width Recovery Summary (Section: Experiments 5.5)

- **Purpose**: Show partial remediation of absorption via dictionary width expansion
- **Type**: comparison_table
- **Content**:
  - Columns: SAE Width | N Absorbed | N Recovered (cos_sim>0.80) | Frac Recovered | Mean Best Cosine | Implication
  - Row 1: 24k → 32k | 18 | 12 | 67% | 0.791 | "Width expansion sufficient for 2/3"
  - Row 2: 24k → 32k (not recovered) | 18 | 6 | 33% | <0.76 | "Training-objective change needed for 1/3"
  - Inset note: "Features not recovered have no close decoder direction in wider SAE, suggesting genuine semantic coverage gap"
- **Data source**: `exp/results/full/F1_successive_refinement.json`

---

## Transition Logic

1. **Introduction → Background**: "To understand why OMP is the right test, we first review the two mechanistic theories and the prior detection work."
2. **Background → Method**: "We present encoder weight norm as the detection tool enabling this controlled mechanistic analysis, then describe the OMP oracle experiment design."
3. **Method (4.2) → Method (4.3)**: "Having established encoder_norm as a weight-only detector, we turn to the mechanistic question it enables us to address: is absorption caused by the encoder approximation or by the training-time dictionary structure?"
4. **Experiments → Discussion**: "The convergence of OMP 0% reduction, encoder_norm peak at training-layer L6, and 33% width-irrecoverable absorbed features points to a coherent picture of absorption as a training-time phenomenon."
5. **Discussion → Conclusion**: "These results redirect the SAE mitigation research program: iterative encoders will not fix absorption; dictionary coverage and training objectives must change."

---

## Word Budget (approximate)

| Section | Target Words |
|---|---|
| Abstract | 250 |
| Introduction | 600 |
| Background | 400 |
| Method | 900 |
| Experiments | 1,400 |
| Discussion | 700 |
| Related Work | 300 |
| Conclusion | 150 |
| **Total** | **~4,700** |

(Typical NeurIPS workshop paper: 4–8 pages; main track: 9 pages excl. references)
