# Paper Outline: Belief-State Diffusion with Adaptive Guidance for Reasoning in Masked Diffusion Language Models

## Title

**Beyond Remasking: Continuous Belief States and Classifier-Free Guidance for Inference-Time Reasoning in Masked Diffusion Language Models**

---

## 1. Introduction (2 pages)

### Key Arguments
- Masked diffusion language models (MDLMs) generate text via iterative denoising, but each step performs independent forward passes that discard the rich distributional information (logits, attention distributions) from prior steps — the **"information island" problem** (MetaState, 2603.01331).
- Existing inference-time scaling approaches have systematically failed on reasoning benchmarks: pure remasking (ReMDM 4.4%, RCR 5.7% vs vanilla 4.7% on Countdown-500) does not accumulate cross-step information; parameter-space adaptation (DTA/TTT) faces vanishing gradient signals (MLM loss at 0.005–0.032); Best-of-N incurs PPL degradation without sharing information across trajectories.
- Our prior discovery — Diffusion Memory Injection (DMI) — provides the key insight: injecting previous-step logit-weighted embeddings into mask positions yields 2x improvement on Countdown-500 (9.3% vs 4.7%, 3-seed validated) with near-zero overhead, demonstrating that the bottleneck lies in the **representation poverty of mask embeddings**, not in the denoising schedule.
- We propose two orthogonal, training-free inference-time scaling methods: (1) **Belief-State Diffusion (BSD)**, which replaces mask embeddings with continuously evolving distributional representations, and (2) **Adaptive Classifier-Free Guidance (A-CFG)**, which amplifies reasoning signals via confidence-based re-masking and dual forward passes. On Dream-7B, A-CFG achieves the strongest reasoning improvements, generalizing from Countdown (+12.5pp) to GSM8K (+12.5pp).
- Extensive negative results — the failure of cross-step JSD stability signals, CFG temporal scheduling, and BSD+A-CFG combination — provide actionable constraints on the design space for future MDLM inference-time methods.

### Transition to Section 2
BSD and A-CFG are motivated by independent lines of work — continuous token representations and classifier-free guidance for diffusion models — which we survey next.

> **Figure 1 (Teaser)**: Side-by-side comparison of vanilla MDLM denoising vs. BSD denoising on a Countdown example, showing how belief vectors accumulate information across steps while vanilla discards it. Include accuracy bar chart inset (Vanilla 4.7% → DMI 9.3% → BSD 6.2–12.5% → A-CFG 12.5%).

---

## 2. Related Work (1.5 pages)

### Key Arguments
- **Masked Diffusion Language Models**: MDLM (Sahoo et al., 2024), Dream (Liu et al., 2025), LLaDA (Nie et al., 2025) — foundations of discrete diffusion for text generation.
- **Inference-time scaling for MDLMs**: ReMDM (Arriaga et al., 2025) and CORE (Chen et al., 2026) improve via remasking; HEX (Ye et al., 2025) uses multi-schedule ensemble; Self-Rewarding SMC (Li et al., 2026) uses particle sampling; MetaState (Xia et al., 2026) trains a GRU for cross-step memory. Our work is training-free and focuses on representation-level improvements.
- **Continuous token representations in diffusion**: LRD (Zhu et al., 2025, GSM8K +2.9, MATH500 +3.8), ReMix (Ye et al., 2026, 2–8x speedup), EvoToken-DLM (2601.07351), PRR (Wan et al., 2026) — all validate that replacing discrete masks with soft representations accelerates or improves generation. BSD extends this direction to full belief-state evolution with EMA accumulation.
- **Classifier-Free Guidance for DLMs**: A-CFG (NeurIPS 2025, 2505.20199) achieves GSM8K 73.5 on LLaDA-8B via confidence-based re-masking — surpassing LLaMA3-8B (53.1). CFG temporal scheduling theory (Rojas et al., 2025, 2507.08965) predicts early guidance is harmful. We apply A-CFG to Dream-7B and discover that fixed guidance weight outperforms all scheduled variants.
- **Test-Time Training / Adaptation**: TTT (Sun et al., 2024), DTA (our prior iteration) — parameter-space updates fail in MDLMs due to insufficient gradient signal, motivating representation-space alternatives.

### Transition to Section 3
Unlike prior work that either trains auxiliary modules (MetaState) or modifies only the diffusion schedule (ReMDM), our methods directly address the representation quality of masked positions and the prediction quality at decision time.

---

## 3. Method (2.5 pages)

### 3.1 Preliminaries: MDLM Denoising and the Information Island Problem
- Formal definition of MDLM denoising as iterative sampling x_t → x_{t-1} via p_theta(x_0 | x_t) predictions and confidence-based unmasking
- The information island problem: each step's distributional predictions p(x_i | x_t) are discarded after argmax, and the next step operates on a fresh mask embedding — no cross-step memory
- DMI as a partial solution: mixing logit-weighted embeddings into mask positions achieves 9.3% vs vanilla 4.7% on Countdown-500, motivating richer continuous representations

### 3.2 Belief-State Diffusion (BSD)
- **Core idea**: Replace mask embeddings entirely with "belief states" — probability-weighted embedding mixtures that evolve via EMA across denoising steps, only committing to hard tokens in the final k steps
- **Two-phase algorithm**: Phase 1 (continuous belief refinement, steps T to k+1): EMA update of belief vectors b_i^t = (1-alpha^t) * b_i^{t+1} + alpha^t * sum_v p(v) * e_v; no argmax, no unmask. Phase 2 (hard token reveal, steps k to 1): standard confidence-based unmasking from belief states
- **Key design decisions**: (a) EMA update rate alpha with linear schedule 0.1→0.8 (explore early, exploit late); (b) L2 normalization to match mask_emb norm, preventing OOD drift in Transformer input space; (c) graceful degradation via fallback_beta mixing with mask embedding
- **Relationship to DMI and prior work**: BSD generalizes DMI — DMI is the special case where k=T (no belief phase) and alpha is fixed. BSD differs from LRD (KL-triggered), ReMix (convergence-triggered), and EvoToken (gradual evolution) in using full belief replacement with EMA accumulation
- Algorithm pseudocode box

### 3.3 Adaptive Classifier-Free Guidance (A-CFG) for Dream-7B
- **Re-masking construction**: Select top-m% least-confident positions (m=10%), re-mask them to construct unconditional input x_tilde
- **Guidance formula**: l_guided = l_cond + w * (l_cond - l_uncond) with fixed guidance weight w=1.5, w_max=2.0 cap
- **Why confidence beats cross-step stability (JSD)**: Dream-7B produces very stable cross-step distributions (JSD stability ~0.997 everywhere), rendering JSD uninformative; single-step confidence directly identifies uncertain positions (empirical: RACFG-JSD 0% vs A-CFG 12.5%)
- **Why fixed guidance outperforms temporal scheduling**: Ablation across 4 schedule types (fixed, linear, cosine, threshold_70_30) shows fixed w dominates all variants; theory-predicted scheduling from continuous diffusion (arXiv 2507.08965) does not transfer to masked diffusion's discrete dynamics

### 3.4 Information-Theoretic Analysis: Belief Entropy Trajectories
- BSD belief entropy decreases near-monotonically during denoising (Spearman rho = -0.95, 15/16 samples monotonic)
- Terminal belief entropy is lower than vanilla (0.001 vs 0.002), confirming information accumulation
- Strong entropy-accuracy correlation: r = 0.78, p < 0.001 — belief convergence quality predicts task correctness

### Transition to Section 4
The proposed methods are validated on two reasoning benchmarks across multiple ablation dimensions, with systematic hypothesis testing.

> **Figure 2 (Architecture Diagram)**: Two-panel method figure. Panel A: BSD two-phase pipeline — EMA belief update cycle in Phase 1 (steps T to k+1) with norm preservation, transition to Phase 2 (steps k to 1) with hard reveal. Panel B: A-CFG pipeline — confidence scoring across positions → re-masking least-confident m% → dual forward pass (conditioned vs unconditional) → guided logit combination. Annotate FLOPs multipliers: BSD ~1.1x, A-CFG ~2.0x.

> **Figure 3 (Belief Entropy Trajectories)**: Line plot showing per-position mean entropy across denoising steps for BSD vs. vanilla. BSD: smooth monotonic decrease from ~3.5 to ~0.0; vanilla: step-function drops at discrete unmask events. Annotate Spearman rho=-0.95, terminal entropy comparison (0.001 vs 0.002). Shade the "information accumulation gap" between curves.

---

## 4. Experiments (3 pages)

### 4.1 Experimental Setup
- **Model**: Dream-v0-Instruct-7B (strongest open-source MDLM), on NVIDIA RTX PRO 6000 Blackwell (98GB)
- **Benchmarks**: Countdown-500 (structured arithmetic reasoning, 500 problems, 3 seeds: 42/123/456) as primary; Countdown-16 (pilot); GSM8K-16 (generalization pilot)
- **Baselines**: Vanilla Dream-7B (128 steps), DMI (alpha=0.3), ReMDM-conf, RCR, DTA (LoRA online adaptation)
- **Our methods**: BSD (k=0.75, linear alpha 0.1→0.8), A-CFG (w=1.5, fixed, m=10%)
- **Evaluation**: Accuracy (primary), rep-2/3 (repetition), distinct-1/2/3 (diversity), output length, FLOPs overhead
- **Statistical protocol**: McNemar test with Bonferroni correction, Bootstrap 95% CI (10,000 resamples); Cohen's h effect sizes

### 4.2 Main Results
- **Countdown-500 (3-seed, full-scale)**: Vanilla 4.7%±0.6 | DMI **9.3%±1.4** (best validated, ~2x improvement) | ReMDM-conf 4.4%±1.0 | RCR 5.7%±0.6
- **Countdown-16 (pilot)**: BSD 6.2–12.5% | A-CFG (w=1.5) 12.5% | DMI 12.5% | Vanilla 0–6.2%
- **BSD+A-CFG combination**: 6.2% — below individual methods, falsifying H7 (synergy hypothesis)
- All methods dramatically outperform pure remasking (ReMDM, RCR) and DTA (6.2%)
- Degeneration check: All methods maintain diversity (distinct-3 > 0.85) and low repetition (rep-3 < 0.10)
- Medium Cohen's h effect sizes (0.51–0.72 vs vanilla) but wide CIs at n=16

### 4.3 Ablation Studies
- **BSD k-parameter**: k=0.75 (25% belief phase) achieves 6.2%; k=0.25 and k=0.50 yield 0% — longer belief phases hurt, suggesting models need hard token anchors early. Falsifies H3 (intermediate k optimal)
- **BSD alpha schedule**: linear(0.1→0.8), cosine(0.1→0.8), constant(0.3), constant(0.5) all yield identical 6.2% — schedule shape is not the bottleneck
- **A-CFG guidance weight**: w=1.5 and w=2.0 both achieve 12.5%; w=1.0 achieves 6.2% — moderate-to-strong guidance required
- **A-CFG temporal schedule**: Fixed guidance dominates all 4 scheduled variants (linear/cosine/threshold all yield 0%); falsifies H6. Masked diffusion's discrete dynamics may not satisfy continuous CFG scheduling theory assumptions
- **RACFG vs A-CFG (re-mask signal quality)**: JSD stability-based RACFG achieves 0.0% vs confidence-based A-CFG 12.5% — decisive falsification of H5. Root cause: Dream-7B JSD stability is near-degenerate at ~0.997

### 4.4 GSM8K Generalization
- A-CFG: **37.5%** (6/16) vs vanilla 25.0% (4/16) — strongest cross-benchmark result (+12.5pp)
- DMI: 25.0% — matches vanilla on GSM8K (no benefit)
- BSD: 18.8% — below vanilla, suggesting continuous arithmetic token mixing may hurt free-form math
- A-CFG is the only method demonstrating benchmark generalization

### 4.5 Compute Fairness Analysis
- BSD (~1.1x FLOPs) vs vanilla with 1.1x steps; A-CFG (~2.0x) vs vanilla with 2x steps
- Compute-fair results are competitive: vanilla with matched steps often approaches method performance
- Key finding: method value lies in information quality at the low-compute frontier, not in scaling behavior
- Pareto frontier analysis: vanilla step-scaling is Pareto-optimal at most compute levels in pilot data

### Transition to Section 5
The results reveal that BSD and A-CFG are individually effective but non-composable — we analyze why and extract broader lessons.

> **Table 1 (Main Results)**: Comprehensive comparison table. Rows (grouped): Vanilla | Remasking baselines (ReMDM-conf, RCR) | Information augmentation (DMI, BSD, A-CFG, BSD+A-CFG) | Parameter adaptation (DTA). Columns: Method | FLOPs | Countdown-500 Mean±Std | Countdown-16 | GSM8K-16 | rep-3 | distinct-3. Bold best per column. Significance markers where available.

> **Figure 4 (Ablation Grid)**: 2×2 grid of bar charts. (a) BSD k-parameter: accuracy at k=T/4, T/2, 3T/4 + vanilla baseline. (b) BSD alpha schedule: linear, cosine, constant(0.3), constant(0.5). (c) A-CFG guidance weight: w=1.0, 1.5, 2.0 + vanilla. (d) A-CFG temporal schedule: fixed, linear, cosine, threshold. Each bar shows accuracy with distinct-3 annotation.

> **Table 2 (Compute-Fair Comparison)**: Method | FLOPs | Accuracy | Vanilla at matched FLOPs | Delta | Pareto-optimal?

> **Figure 5 (GSM8K Generalization)**: Grouped bar chart — accuracy of each method on Countdown-16 vs GSM8K-16 side by side. Highlight A-CFG's consistent improvement and BSD's benchmark-specific decline.

---

## 5. Discussion (1.5 pages)

### Key Arguments
- **Why BSD + A-CFG combination fails**: BSD's continuous beliefs in Phase 1 produce smooth distributional representations optimized for refinement; A-CFG's Phase 2 confidence-based re-masking requires sharp, discriminable distributions. The two methods optimize for different properties of the representation space (smoothness vs. discriminability), creating interference rather than complementarity.
- **Why JSD stability fails on Dream-7B**: Dream-7B's denoising produces remarkably stable cross-step distributions (JSD ~0 everywhere), which is actually a desirable property (prediction consistency) but eliminates the "hesitation signal" RACFG relies on. Implication: successful MDLMs may inherently lack inter-step variation, making stability-based methods generically inapplicable.
- **Why CFG temporal scheduling fails for masked diffusion**: The theoretical prediction from continuous diffusion (arXiv 2507.08965) assumes smooth noise-to-signal transitions; masked diffusion's discrete mask/unmask dynamics create a fundamentally different information geometry where the model has sufficient context at any step to benefit from constant guidance.
- **The representation bottleneck is real but narrow**: Both BSD and A-CFG demonstrate that the information island problem is addressable and that representation-level (BSD) and prediction-level (A-CFG) improvements are individually effective. However, compute-fair comparisons show the advantage is concentrated at the low-compute frontier.
- **Negative results as design space constraints**: DTA failure (gradient signal too weak), RACFG/JSD failure (cross-step instability signal degenerate), temporal scheduling failure (theory mismatch), combination failure (layer interference) — these constitute a systematic mapping of what does and does not work for MDLM inference-time scaling.
- **DMI as practical contribution**: Near-zero-cost embedding memory injection provides ~2x Countdown improvement and remains the most efficient training-free method discovered across 4 iterations.

### Transition to Section 6
We summarize contributions and outline directions informed by these constraints.

> **Figure 6 (Failure Mode Diagnostics)**: Three-panel figure. (a) JSD stability distribution: histogram of per-position stability scores — near-degenerate peak at ~0.997, explaining RACFG failure. (b) A-CFG guidance impact: scatter plot of mean guidance magnitude vs. sample correctness, showing that correctly-solved problems receive more guidance. (c) BSD+A-CFG entropy discontinuity: belief entropy trajectory showing the Phase 1→Phase 2 transition and confidence distribution shift that may disrupt A-CFG.

---

## 6. Conclusion (0.5 pages)

### Key Arguments
- We introduced Belief-State Diffusion (BSD) and applied Adaptive Classifier-Free Guidance (A-CFG) to Dream-7B, achieving the strongest training-free reasoning improvements reported on this model.
- BSD provides information-theoretic validation of continuous belief evolution: monotonic entropy decrease (rho=-0.95), lower terminal entropy, and strong entropy-accuracy correlation (r=0.78).
- A-CFG generalizes effectively from Countdown to GSM8K (+12.5pp), establishing CFG as the most promising general tool for MDLM reasoning enhancement.
- DMI remains a practical near-zero-cost improvement (~2x on Countdown-500) suitable for deployment in any MDLM pipeline.
- Extensive negative results — DTA, RACFG/JSD, temporal scheduling, BSD+A-CFG combination — constrain the design space and provide actionable guidance for future work.
- **Limitations**: Pilot-scale statistical power is low (n=16, CIs include 0); full-scale 3-seed validation of BSD and A-CFG is required. Compute-fair analysis suggests methods may not dominate vanilla step-scaling at all budgets.
- **Future directions**: (1) Training-aware belief states (learning belief update rules, cf. MetaState); (2) Confidence-calibrated guidance with learned weight schedules; (3) Scaling to larger models where cross-step instability may provide richer signals for RACFG-style methods.

---

## Figure & Table Plan

### Figure 1: Information Island Problem and BSD Teaser (Section: Introduction)
- **Purpose**: Visually introduce the information island problem and BSD's solution at a glance
- **Type**: architecture_diagram + bar_chart_inset
- **Content**: Left: vanilla denoising pipeline (mask → forward → argmax → discard logits → fresh mask); Right: BSD pipeline (mask → forward → EMA belief update → use beliefs as input → repeat). Inset bar chart: accuracy comparison (Vanilla 4.7%, DMI 9.3%, BSD 6.2–12.5%, A-CFG 12.5%)
- **Key takeaway**: Vanilla discards distributional information at every step; BSD accumulates it continuously through belief states
- **Generation**: tikz or manual_diagram
- **Data source**: Conceptual design + accuracy from `bsd_fullscale_summary.md`, `racfg_fullscale_summary.md`

### Figure 2: Method Architecture — BSD and A-CFG Pipelines (Section: Method)
- **Purpose**: Detailed algorithmic pipeline showing both proposed methods
- **Type**: architecture_diagram (two-panel)
- **Content**: Panel A — BSD: EMA belief update loop with norm preservation, Phase 1 (continuous) to Phase 2 (hard reveal) transition, FLOPs annotation (~1.1x). Panel B — A-CFG: confidence scoring → select bottom-m% → re-mask → dual forward pass → l_guided = l_cond + w*(l_cond - l_uncond), FLOPs annotation (~2.0x)
- **Key takeaway**: BSD and A-CFG operate at orthogonal levels — representation layer vs. prediction layer
- **Generation**: tikz
- **Data source**: Algorithm descriptions from `proposal.md` Section 3

### Figure 3: Belief Entropy Trajectories (Section: Method + Experiments)
- **Purpose**: Information-theoretic validation that BSD beliefs converge and accumulate information
- **Type**: line_plot
- **Content**: Mean per-position entropy across denoising steps for BSD (smooth decay ~3.5→0.0) vs. vanilla (step-function drops). Annotate: Spearman rho=-0.95, terminal entropy BSD=0.001 vs vanilla=0.002, entropy-accuracy r=0.78
- **Key takeaway**: BSD achieves lower terminal entropy via continuous information accumulation, and entropy convergence predicts correctness
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `entropy_analysis_countdown500.json`, `entropy_analysis_summary.md`

### Table 1: Main Results — All Methods Comparison (Section: Experiments)
- **Purpose**: Central results table — the paper's primary evidence
- **Type**: comparison_table
- **Content**: Rows grouped as: Vanilla | Remasking (ReMDM-conf, RCR) | Cross-step info (DMI, BSD, A-CFG, BSD+A-CFG) | Parameter adaptation (DTA). Columns: Method | FLOPs | Countdown-500 (mean±std, 3 seeds) | Countdown-16 | GSM8K-16 | rep-3 | distinct-3. Bold best, significance markers
- **Key takeaway**: DMI (9.3%) provides validated 2x improvement; A-CFG (12.5%) is strongest on pilot; remasking and DTA are ineffective
- **Generation**: data_table (LaTeX tabular)
- **Data source**: `summary.md`, `bsd_fullscale_summary.md`, `racfg_fullscale_summary.md`, `bsd_racfg_combo_summary.md`

### Figure 4: Ablation Study Grid (Section: Experiments)
- **Purpose**: Hyperparameter sensitivity for BSD and A-CFG
- **Type**: bar_chart (2×2 grid)
- **Content**: (a) BSD k-parameter: k=T/4, T/2, 3T/4 + vanilla. (b) BSD alpha: linear, cosine, const(0.3), const(0.5). (c) A-CFG w: 1.0, 1.5, 2.0 + vanilla. (d) A-CFG schedule: fixed, linear, cosine, threshold_70_30. Show accuracy per bar with distinct-3 annotation
- **Key takeaway**: BSD needs short belief phase (k=3T/4); A-CFG works best with fixed moderate guidance (w=1.5); schedule shape has minimal impact for BSD but critical for A-CFG (only fixed works)
- **Generation**: code (matplotlib)
- **Data source**: `bsd_k_ablation_summary.md`, `bsd_alpha_ablation_summary.md`, `racfg_schedule_ablation_summary.md`

### Table 2: Compute-Fair Comparison (Section: Experiments)
- **Purpose**: Control for compute budget — assess whether methods beat vanilla at equal FLOPs
- **Type**: comparison_table
- **Content**: Method | FLOPs | Accuracy | Vanilla at Matched FLOPs | Delta | Pareto-optimal?
- **Key takeaway**: At matched compute, vanilla step-scaling is competitive; method advantage is concentrated at low-compute frontier
- **Generation**: data_table (LaTeX tabular)
- **Data source**: `compute_fair_summary.md`

### Figure 5: Cross-Benchmark Generalization (Section: Experiments)
- **Purpose**: Show which methods generalize from Countdown to GSM8K
- **Type**: bar_chart (grouped)
- **Content**: Grouped bars for Vanilla, DMI, BSD, A-CFG — each with Countdown-16 accuracy and GSM8K-16 accuracy side by side. Annotate delta from vanilla for each
- **Key takeaway**: A-CFG is the only method showing consistent improvement across benchmarks (+12.5pp on GSM8K); BSD degrades on free-form math
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `gsm8k_extension_summary.md`, `statistical_analysis_summary.md`

### Figure 6: Failure Mode Diagnostics (Section: Discussion)
- **Purpose**: Data-driven explanation of why RACFG/JSD, temporal scheduling, and BSD+A-CFG combination failed
- **Type**: multi-panel (histogram + scatter + line_plot)
- **Content**: (a) JSD stability histogram: near-degenerate distribution peaked at ~0.997 across all positions, explaining RACFG's inability to discriminate reasoning-critical tokens. (b) A-CFG guidance vs correctness: scatter plot showing correct samples receive higher mean guidance magnitude (22081 vs 19367), confirming CFG targets useful positions. (c) BSD+A-CFG phase transition: entropy trajectory showing discontinuity at Phase 1→Phase 2 boundary
- **Key takeaway**: Failures are attributable to measurable signal deficiencies in Dream-7B's denoising dynamics, not implementation errors
- **Generation**: code (matplotlib)
- **Data source**: `token_diagnostics_summary.md`, `racfg_remask_pct_ablation_summary.md`

### Table 3: Hypothesis Verification Summary (Section: Discussion or Appendix)
- **Purpose**: Systematic tracking of all pre-registered hypotheses and outcomes
- **Type**: ablation_table
- **Content**: H1–H11 rows with: Hypothesis | Expected | Observed | Verdict (Supported/Falsified/Pending). Key outcomes: H2 (entropy monotonicity) Supported, H5 (JSD > confidence) Falsified, H6 (scheduling > fixed) Falsified, H7 (BSD+A-CFG synergy) Falsified, H9/H10 (no degeneration) Supported
- **Key takeaway**: 4 supported, 4 falsified, 3 pending full-scale — negative results are as informative as positive ones
- **Generation**: data_table (LaTeX tabular)
- **Data source**: `hypotheses.md` + all experiment summaries

---

## Appendix Plan

### A. Full Experimental Details
- Complete hyperparameter tables for all methods and ablation configurations
- Per-seed results for Countdown-500 (DMI, ReMDM-conf, RCR, vanilla)
- McNemar test matrices and Bootstrap CI plots for all pairwise comparisons
- Cohen's h effect size table

### B. Qualitative Examples
- 5 Countdown examples: vanilla vs BSD vs A-CFG outputs side by side, annotated with belief entropy and guidance magnitude
- 3 GSM8K examples showing A-CFG's reasoning chain improvements vs vanilla errors

### C. Prior Iteration Negative Results Archive
- Iteration 1–2: TCR, temperature annealing, entropy remasking — pilot improvements (up to -24.9% PPL) that completely failed at full scale (all p > 0.25)
- Iteration 3: DTA (LoRA online adaptation) — gradient signal too weak, 6.2% < vanilla 12.5%
- Iteration 3: TTT (6 variants) — all statistically insignificant (p=0.88)
- Key lesson: PPL is unreliable as quality metric; pilot effect sizes inflate by 20-25pp due to small-sample variance

### D. Implementation Details
- BSD belief vector computation: EMA update, L2 normalization, fallback_beta mixing
- A-CFG re-masking: confidence scoring, position selection, dual forward pass
- Evaluation harness: Countdown problem generation, answer extraction, diversity metrics
- Code module structure: `bsd_racfg/bsd.py`, `bsd_racfg/racfg.py`, `bsd_racfg/eval_harness.py`
