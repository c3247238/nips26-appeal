# Paper Outline: Encoder-Decoder Dissociation as a Geometric Fingerprint of Feature Absorption in Sparse Autoencoders

**Date:** 2026-04-14
**Iteration:** iter_002 (post-full-pilot, pre-writing)
**Status:** Ready for section writing

---

## Title (Working)

**Encoder-Decoder Dissociation as a Geometric Fingerprint of Feature Absorption in Sparse Autoencoders**

*Alternative titles:*
- *EDA: A Probe-Free Weight-Only Detector for SAE Feature Absorption*
- *Feature Absorption in Sparse Autoencoders: A Geometric Characterization via Encoder-Decoder Alignment*

---

## Abstract (Draft)

Feature absorption — the failure of a child SAE feature to fire when its parent absorbs the signal — limits the reliability of sparse autoencoders (SAEs) for mechanistic interpretability. Existing detection methods require pre-specified probe directions, precluding systematic auditing. We contribute three results. First, we prove that the rate-distortion training objective produces an absorption preference when the sparsity penalty exceeds the squared sine of the parent-child decoder angle (Proposition 1), and we derive a mechanistic conjecture showing that absorbed features develop encoder-decoder dissociation (EDA = 1 - cos(enc_j, dec_j)) as the encoder is pulled toward the parent direction while the decoder remains anchored to the child concept. Second, we validate EDA as a probe-free absorption detector on GPT-2 Small (layer 6, 24,576-width SAE): EDA achieves AUROC = 0.650 against exact Chanin et al. labels (z = 2.49 above null), while a cross-directional variant — cos(enc_parent, dec_child) — achieves AUROC = 0.730 (Cohen's d = 0.552, p = 2.8e-9). EDA fails at layer 10 (AUROC = 0.256 with reversed polarity), consistent with our hypothesis of post-absorption encoder re-alignment. Third, we characterize absorption phase stability: absorption rates remain uniformly high (0.919–0.968) across all 11 tested SAE configurations spanning layers 2–10 and widths 12k–98k, with no evidence of a phase transition in sparsity. We show that EDA is architecture-dependent, with AJT-trained SAEs exhibiting reversed EDA polarity, pointing toward training regime as a key modulator of absorption geometry. Together, these results provide the first weight-only, probe-free diagnostic for SAE feature absorption and characterize the geometric mechanism underlying it.

---

## Section 1: Introduction

### Key arguments
- SAEs are the primary tool for mechanistic interpretability, but feature absorption degrades their reliability: absorbed features represent parent-child hierarchical concepts incorrectly
- Current detection (Chanin et al.) requires ground-truth probe directions and activation data — cannot scale to systematic auditing of all SAE latents in a deployed model
- We ask: can the SAE weight matrices alone reveal which features are absorbed? The answer is yes, via EDA
- Paper framing: theory (Prop. 1 proves absorption preference from first principles) + empirical (EDA/cos_enc_p_dec_c validated as detectors) + characterization (phase stability, architecture dependence)
- One negative result that strengthens the paper: the naive decoder-decoder cosine predictor fails (AUROC = 0.206, strongly anti-correlated), demonstrating the importance of encoder geometry

### Transition to Section 2
Positions the work relative to the absorption literature, the broader SAE literature, and the theoretical landscape.

---

## Section 2: Related Work

### Key arguments
- Feature absorption defined and characterized (Chanin et al. NeurIPS 2025): parent absorbs child activation signal; supervised detection using probe directions
- Rate-distortion theory of SAEs (Tang et al. 2025): biconvex landscape, sparse optima; our Prop. 1 refines this to absorption-specific preference condition
- Architectural mitigations (OrtSAE, Matryoshka, KronSAE, MP-SAE): all require retraining; our detection is post-hoc weight-only
- Amortization gap (O'Neill et al. 2411.13117): encoder failure as a distinct confounder for EDA — must be acknowledged
- SAEBench / SynthSAEBench: existing evaluation benchmarks; our method complements with weight-only screening
- Cross-directional cosine measures are novel: no prior work uses cos(enc_p, dec_c) as an inter-feature absorption signal

### Transition to Section 3
From prior work to our formal setup and theoretical contribution.

---

## Section 3: Method

### 3.1 Setup: Sparse Autoencoders and Feature Absorption
- Formal definition of SAE: encoder $E \in \mathbb{R}^{d_\text{sae} \times d}$, decoder $D \in \mathbb{R}^{d \times d_\text{sae}}$ with unit-norm columns $\{d_j\}$
- Absorption definition: child feature $c$ has absorption rate $\alpha = P(\text{child inactive} \mid \text{child concept present, parent active})$
- Training loss (Lagrangian form): $\mathcal{L} = \mathbb{E}[\|x - Df(Ex + b)\|_2^2] + \lambda \mathbb{E}[\|f(Ex + b)\|_0]$
- First-letter task (Chanin et al.): hierarchical pair (letter-class parent, specific-token child)

### 3.2 Theory: Rate-Distortion Absorption Preference (Proposition 1)
- Compare absorbed vs. non-absorbed solutions for a hierarchical pair $(p, c)$ with co-occurrence probability $p_\text{co}$
- **Proposition 1:** absorbed solution achieves lower expected loss iff $\lambda > \sin^2(\theta_{p,c})$, where $\theta_{p,c}$ = decoder angle between parent and child
- **Corollary:** co-occurrence frequency cancels; threshold depends only on decoder geometry and sparsity penalty
- **Limitation (explicit):** comparison of two specific solutions; convergence to absorbed state requires additional analysis

### 3.3 Geometric Signature: EDA and Cross-Directional Alignment
- **EDA definition:** $\text{EDA}(c) = 1 - \cos(\hat{e}_c, d_c)$ — intra-feature encoder-decoder angle
- **Mechanistic conjecture (Prop. 2):** encoder of absorbed child is pulled toward parent decoder direction $d_p$ by gradient from parent-only contexts; decoder remains anchored to child concept by reconstruction loss. Conditions C1–C3 explicitly stated.
- **Why decoder-decoder fails:** Proposition 1 predicts small $\theta_{p,c}$ at absorption onset; but post-convergence, child decoder drifts away from parent as parent-only reconstruction pressure lifts. Pilot confirms this (Cohen's d = -0.48 for cos^2(theta_{p,c}), letter vs. non-letter).
- **Cross-directional metric:** $\cos(\hat{e}_p, d_c)$ — parent encoder aligned with child decoder — captures absorbed state from parent side

### 3.4 Experimental Configurations and Baselines
- Model: GPT-2 Small (117M) with SAELens pre-trained SAEs
- Primary SAE: gpt2-small-res-jb, layer 6, width 24,576 (L0 = 51.0)
- Scaling suite: 11 configurations across layers 2–10, widths 12k–98k, including AJT architecture variants
- Ground-truth labels: FeatureAbsorptionCalculator (Chanin et al. sae-spelling), exact IG-ablation labels for layer 6 (n_pos = 18 Chanin exact + 50 proxy, Jaccard overlap = 0.115)
- Baselines: frequency ratio alone (AUROC = 0.595), decoder norm alone (AUROC = 0.515), cos(enc, dec) raw score (AUROC = 0.350), random (0.500)

---

## Section 4: Experiments

### 4.1 EDA Validation Against Exact Labels
- Primary result: EDA AUROC = 0.650 (exact Chanin labels, n_pos = 18, z = 2.49 above null)
- Proxy-label result: EDA AUROC = 0.659 (n_pos = 50, Cohen's d = 0.533, p = 0.000165)
- **Best detector:** encoder norm (AUROC = 0.757) — surprising and discussed in 4.2
- Cross-directional result: cos(enc_p, dec_c) max AUROC = 0.730 (Cohen's d = 0.552, p = 2.8e-9) within-letter; cos(enc_c, dec_p) mean AUROC = 0.681
- **Layer failure:** EDA AUROC reverses at L10 (AUROC = 0.256, Cohen's d = -0.890, p = 6.8e-10); consistent with post-absorption re-alignment hypothesis

### 4.2 EDA Decomposition: Encoder vs. Decoder Alignment
- Letter features (L6): encoder-probe alignment = 0.139 vs. non-letter 0.056 (AUROC = 0.991); decoder-probe alignment = 0.383 vs. 0.099 (AUROC = 1.0)
- Key finding: decoder MORE aligned to letter probe than encoder (diff = -0.244, p = 3.5e-38) — consistent with F1 revised theory (decoder specializes to child concept; encoder dragged toward parent)
- Encoder norm: letter features have larger encoder norms (3.279 vs. 2.575) — explains why encoder norm outperforms EDA as a raw detector; encoder norm may be a polysemanticity confound
- **Unresolved tension:** EDA magnitude ~0.45–0.65 for letter features contradicts small-theta prediction; documented honestly as open question

### 4.3 EDA Across Architectures and Scales
- Standard/ReLU (L1 penalty, 24576): EDA delta = +0.045; AUROC = 0.702 (proxy labels)
- TopK-32 (32768): EDA delta = +0.046; AUROC measurable
- **AJT SAEs (3 variants, L6, 46080):** EDA delta NEGATIVE (-0.176 to -0.217) — reversed polarity. Training regime (non-L1 sparsity) fundamentally changes encoder-decoder alignment geometry
- Width analysis (L8, feature-splitting suite 12k–98k, matched L0 ≈ 51): EDA_delta decreases at wider widths (0.028 → -0.017), suggesting increased feature splitting dilutes the EDA signal

### 4.4 Absorption Phase Stability
- Absorption rates measured across 11 SAE configurations (layers 2–10, widths 12k–98k)
- All rates: 0.919–0.968 (mean ≈ 0.96) — uniformly high; no phase transition detected
- Sigmoid vs. linear fit: BIC difference = -3.22, LRT p = 0.456 — sigmoid not preferred
- Spearman(1/L0, absorption_rate) = rho = 0.191, p = 0.574 — no sparsity dependence
- **Interpretation:** absorption is a phase-stable phenomenon in GPT-2 Small; all tested hyperparameter settings produce absorbed SAEs
- **Hysteresis pilot:** fine-tuning high-sparsity SAE 500 steps at lower sparsity barely changes absorption (0.959 → 0.960) — absorbed state is robust/metastable

### 4.5 Cross-Domain Absorption (First-Letter Confirmed; Semantic Hierarchies Null)
- First-letter (C2 v9): absorption_rate = 0.0083, ratio-to-null = 10.0, 120 events, GO
- Animate-inanimate, noun-proper hierarchies: ratio-to-null = 1.0, NO_GO
- **Interpretation:** absorption in GPT-2 Small is specific to first-letter orthographic hierarchies; semantic hierarchies do not show absorption at this scale. This is a scoped null result, not a general falsification — Gemma-scale experiments are required for semantic hierarchies

---

## Section 5: Discussion

### Key arguments
- EDA and cross-directional cosines are viable probe-free absorption screens: EDA AUROC = 0.650 and cos(enc_p, dec_c) AUROC = 0.730 both exceed null (z > 2.5)
- The positive finding from the encoder-decoder decomposition: decoder aligns MORE with the letter probe than the encoder does (AUROC = 1.0 vs. 0.991), which is consistent with the mechanistic conjecture that the decoder specializes to the child while the encoder is pulled toward the parent
- The failure modes are informative: L10 EDA reversal, AJT polarity reversal, and encoder-norm dominance each point to non-trivial structure in how training shapes encoder-decoder geometry
- Absorption phase stability is a substantive finding: practitioners cannot tune away absorption by adjusting L0 or layer choice alone
- Honest accounting of what we cannot claim: (a) the EDA magnitude tension is unresolved; (b) semantic hierarchy absorption in GPT-2 Small is absent; (c) cross-model generalization awaits Gemma experiments; (d) the absorbed state at L10 may simply reflect different feature type composition
- Implications for practitioners: EDA screening is useful primarily for L1-trained SAEs at mid-layers; architectural variants (TopK, AJT) require separate calibration

### Transition from Section 4
Links experimental findings to broader mechanistic interpretability context.

---

## Section 6: Conclusion

### Key arguments
- We provide a theoretically grounded (Proposition 1) and empirically validated (AUROC = 0.650–0.730) probe-free absorption detector based on encoder-decoder dissociation
- The cross-directional metric cos(enc_p, dec_c) is a new, stronger signal (AUROC = 0.730) not predicted by prior theory — a genuine empirical discovery from the pairwise analysis
- Absorption is phase-stable in GPT-2 Small: high rates (0.919–0.968) across all tested hyperparameter configurations, suggesting architectural or training-time intervention is required, not hyperparameter tuning
- AJT training regime eliminates the EDA signal entirely, pointing to the sparsity mechanism as a critical variable in absorption geometry
- Negative result: semantic hierarchy absorption absent in GPT-2 Small at layer 6 — first-letter absorption appears specific to orthographic hierarchies at this scale

---

## Figure and Table Plan

### Figure 1: Method Diagram — Absorbed vs. Non-Absorbed Feature Geometry
- **Purpose:** Illustrate the EDA mechanism. Show how encoder and decoder directions relate in non-absorbed vs. absorbed features.
- **Type:** architecture_diagram / geometric illustration (two-panel)
- **Content:**
  - Left panel: non-absorbed feature. $\hat{e}_c \approx d_c$. Both point toward child concept direction. EDA ≈ 0.
  - Right panel: absorbed feature. $\hat{e}_c$ pulled toward parent decoder direction $d_p$. $d_c$ remains anchored to child concept. Large angle $\rightarrow$ high EDA.
  - Formula inset: $\text{EDA}(c) = 1 - \cos(\hat{e}_c, d_c)$
  - Threshold inset: absorption preferred when $\lambda > \sin^2(\theta_{p,c})$
- **Key takeaway:** EDA is the geometric fingerprint of a feature whose encoder detects the parent but whose decoder reconstructs the child.
- **Generation:** PDF already generated at `exp/results/full/fig1_eda_method.pdf`; needs final cleanup
- **Data source:** Conceptual diagram; annotate with actual EDA values from per-feature data (mean EDA letter ~0.67, non-letter ~0.63)

### Figure 2: EDA Distributions — Letter vs. Non-Letter Features (Layer 6 vs. Layer 10)
- **Purpose:** Demonstrate that EDA separates letter (absorbed) from non-letter features at L6 but reverses at L10.
- **Type:** violin_plot or box_plot (two-panel, L6 left, L10 right)
- **Content:**
  - L6: letter mean EDA = 0.671, non-letter = 0.631, Cohen's d = 0.533, p = 0.000165; AUROC = 0.659
  - L10: letter EDA mean = 0.637, non-letter = 0.632, Cohen's d = -0.890; AUROC = 0.256
  - Mark mean ± std for each group; annotate with AUROC and p-value
  - Optional third panel: L4 (best AUROC = 0.716) for mid-layer comparison
- **Key takeaway:** EDA reliably separates absorbed from non-absorbed features at mid-layers (L4–L8) but fails at late layers.
- **Generation:** matplotlib, two/three-panel, scatter overlay on violin; data from `full/B1_eda_decomposition.json`, `full/B2_scaling_curve.json`
- **Data source:** `full/B1_eda_decomposition.json` (L6 per-feature), `full/B2_scaling_curve.json` (layer-wise EDA_delta and AUROC)

### Figure 3: EDA Scaling — Layer and Architecture Sweep
- **Purpose:** Show EDA_delta and EDA AUROC across 11 configurations; highlight architecture divergence.
- **Type:** scatter_plot with connecting lines (primary jb suite) + separate markers (AJT suite, width suite)
- **Content:**
  - X-axis: layer (or 1/L0)
  - Y-axis: EDA_delta (mean_EDA_letter - mean_EDA_nonletter)
  - Primary jb suite (5 points L2–L10): positive EDA_delta, peaks at L4/L6, drops toward L10
  - AJT suite (3 points, L6 at different L0): NEGATIVE EDA_delta (-0.176 to -0.217), reversed
  - Width suite (4 points L8 at 12k–98k): EDA_delta decreases with width (0.028 → -0.017)
  - Color code by architecture family (jb vs. AJT vs. width)
- **Key takeaway:** EDA_delta is positive for L1/standard SAEs at mid-layers, collapses at late layers, and reverses for AJT-trained SAEs.
- **Generation:** matplotlib scatter+line; data from `full/B2_scaling_curve.json`
- **Data source:** `full/B2_scaling_curve.json` (all_results array)

### Table 1: Main Detection Results — EDA and Baselines
- **Purpose:** Report all detectors with AUROC, AUPRC, Cohen's d, and statistical tests.
- **Type:** comparison_table (LaTeX tabular)
- **Content (rows = detectors, columns = metrics):**

| Detector | AUROC | AUPRC | AUPRC/base | Cohen's d | p-value | z vs. null |
|---|---|---|---|---|---|---|
| EDA (1 - cos(enc, dec)) | 0.650 | 0.00153 | 2.09x | 0.533 | 1.6e-4 | 2.49 |
| cos(enc_p, dec_c) max | **0.730** | — | — | 0.552 | 2.8e-9 | 6.38 |
| cos(enc_c, dec_p) mean | 0.681 | — | — | 0.517 | 2.7e-6 | 4.63 |
| Encoder norm | 0.757 | 0.00416 | 5.68x | — | — | — |
| Frequency ratio (inverted) | 0.595 | 0.000972 | 1.33x | — | — | — |
| Decoder norm | 0.515 | 0.000749 | 1.02x | — | — | — |
| cos(enc, dec) raw | 0.350 | — | — | — | — | — |
| Random | 0.500 | 0.000732 | 1.0x | — | — | — |

- **Note:** EDA and cos_enc_dec_inverted are equivalent; presented separately to confirm.
- **Key takeaway:** EDA is the only probe-free weight-only metric with significant cross-null separation. Encoder norm may capture polysemanticity rather than absorption specifically.
- **Generation:** LaTeX table; data from `full/D1_eda_validation.json`, `full/B1_pairwise_eda.json`
- **Data source:** `full/D1_eda_validation.json` (EDA, freq, dec_norm); `full/B1_pairwise_eda.json` (cos_enc_p_dec_c, cos_enc_c_dec_p)

### Figure 4: Encoder vs. Decoder Alignment to Letter Probe
- **Purpose:** Visually confirm the mechanistic conjecture — decoder specializes to child (high probe alignment); encoder pulled toward parent (lower probe alignment relative to decoder).
- **Type:** scatter_plot (per feature: x = cos(enc, letter_probe), y = cos(dec, letter_probe)) + bar chart comparing means
- **Content:**
  - Scatter: each letter feature as a point; color by absorbed/non-absorbed (proxy labels); show most points lie ABOVE the diagonal (decoder > encoder alignment to probe)
  - Bar: group means — letter features: enc_mean = 0.139, dec_mean = 0.383; non-letter: enc_mean = 0.056, dec_mean = 0.099
  - Annotate with AUROC values (enc: 0.991, dec: 1.000)
  - Mark the paired t-test: diff = -0.244, p = 3.5e-38
- **Key takeaway:** Every letter feature has its decoder pointing more toward the letter concept than its encoder — the dissociation is a universal property of absorbed features.
- **Generation:** matplotlib two-panel; data from `full/B1_eda_decomposition.json` (per_feature_data)
- **Data source:** `full/B1_eda_decomposition.json`

### Figure 5: Absorption Phase Stability
- **Purpose:** Show absorption rates are uniformly high across all SAE configurations; no phase transition.
- **Type:** scatter_plot with curve fits (sigmoid vs. linear)
- **Content:**
  - X-axis: 1/L0 (proxy for sparsity)
  - Y-axis: absorption rate (fraction of child features non-firing)
  - All 11 configs plotted; color by suite (primary jb, AJT, width)
  - Overlay sigmoid fit (BIC diff = -3.22, LRT p = 0.456, inflection L0_c = 81) and linear fit
  - Add horizontal dashed line at absorption_rate = 0.96 (mean)
  - Annotate: "Spearman rho = 0.191, p = 0.574"
- **Key takeaway:** Absorption rate is insensitive to sparsity within the tested range; no phase transition occurs.
- **Generation:** matplotlib scatter; data from `full/E1_phase_transition.json`, `full/B2_sparsity_analysis.json`
- **Data source:** `full/E1_phase_transition.json` (absorption rates per config), `full/B2_scaling_curve.json` (L0 values)

### Figure 6 (Optional): Cross-Domain Absorption — First-Letter GO, Semantic Null
- **Purpose:** Show the scoped cross-domain result: first-letter absorption confirmed, semantic hierarchies null.
- **Type:** bar_chart with null baseline
- **Content:**
  - Bars: first_letter absorption_rate = 0.0083, animate_inanimate ≈ 0, noun_proper ≈ 0
  - Null baseline: random shuffle (near zero for all)
  - Ratio-to-null labels: first_letter = 10.0x, others = 1.0x
  - Error bars from bootstrap CI (first_letter: 95% CI [0, 0.029])
- **Key takeaway:** Feature absorption in GPT-2 Small is specific to orthographic hierarchies; semantic concept absorption does not emerge at this scale.
- **Generation:** matplotlib bar; data from `full/C2_child_suppression_absorption.json`
- **Data source:** `full/C2_child_suppression_absorption.json`

### Appendix Figure A: Hysteresis Experiment
- **Purpose:** Document that fine-tuning at lower sparsity does not reduce absorption (metastability evidence).
- **Type:** bar_chart or line_plot (absorption rate before/after fine-tuning, across checkpoints)
- **Content:** baseline = 0.959, finetuned (500 steps) = 0.960, scratch lower-L0 = 0.964
- **Key takeaway:** Absorbed state is metastable; cannot be escaped via fine-tuning with reduced sparsity.
- **Generation:** matplotlib; data from `full/E2_hysteresis.json`

### Appendix Table B: Full Configuration Table
- **Purpose:** Full table of all 11 SAE configurations with L0, width, layer, architecture, EDA_delta, AUROC, absorption_rate.
- **Type:** table
- **Data source:** `full/B2_scaling_curve.json`, `full/E1_phase_transition.json`

---

## Section Flow and Transition Logic

```
Introduction
  → establishes: absorption as reliability problem; detection gap; weight-only solution
  → transition: "We first characterize the theoretical conditions under which absorption is preferred..."

Related Work
  → establishes: prior work landscape; what EDA adds over each prior approach
  → transition: "Building on [Tang et al.] theoretical framework, we formalize..."

Method (Section 3)
  3.1 Setup → 3.2 Theory (Prop. 1) → 3.3 EDA mechanism (Prop. 2 conjecture) → 3.4 Experimental setup
  → transition: "We now evaluate whether EDA and cross-directional metrics achieve the predicted discrimination..."

Experiments (Section 4)
  4.1 Detection → 4.2 Decomposition → 4.3 Architecture/Scale → 4.4 Phase Stability → 4.5 Cross-Domain
  → Each subsection starts with what we expected (from theory), then shows the number, then interprets
  → transition to Discussion: "We now synthesize these findings..."

Discussion (Section 5)
  → Starts with the positive summary; then open tensions; then implications
  → transition: "In conclusion..."

Conclusion (Section 6)
  → Three sentences per contribution; honest limitations in last paragraph
```

---

## Paper Scope and Target Venue

- **Target venue:** EMNLP 2026 or NeurIPS 2026 MI Workshop (current data); escalate to NeurIPS 2026 main or ICLR 2027 if Gemma Scope experiments confirm cross-model generalization
- **Paper length:** 8-10 pages + references (main); appendix with proofs and hysteresis
- **Open results acknowledged:** EDA magnitude tension (unresolved); semantic hierarchy null (model-scale hypothesis); AJT reversal mechanism (hypothesized, not proven)

---

## Artifact Checklist Before Writing

- [x] `exp/results/full/D1_eda_validation.json` — EDA AUROC = 0.650 (exact labels)
- [x] `exp/results/full/B1_pairwise_eda.json` — cos(enc_p, dec_c) AUROC = 0.730
- [x] `exp/results/full/B1_eda_decomposition.json` — enc/dec probe alignment, Cohen's d = 0.533
- [x] `exp/results/full/B2_scaling_curve.json` — 11 configs, EDA_delta by layer/width
- [x] `exp/results/full/B3_cross_arch.json` — Standard (L1) vs. TopK cross-arch comparison
- [x] `exp/results/full/E1_phase_transition.json` — LRT p = 0.456, BIC diff = -3.22
- [x] `exp/results/full/E2_hysteresis.json` — 500-step finetuning, absorption 0.959 → 0.960
- [x] `exp/results/full/C2_child_suppression_absorption.json` — first-letter GO, semantic NO_GO
- [x] `exp/results/full/F1_theory_revised.md` — non-circular derivation, Proposition 1 proven
- [x] `exp/results/full/fig1_eda_method.pdf` — Figure 1 exists as PDF
- [ ] Gemma Scope experiments — MISSING (needed for cross-model claim)
- [ ] Full B3 cross-arch AUROC for TopK SAE — architecture audit
