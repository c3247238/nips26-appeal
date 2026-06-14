# 3. The Absorption Score

We first define a metric to quantify feature absorption in pretrained SAEs, then validate it against random and edge-case controls.

## 3.1 Definition

Given a pretrained SAE with decoder weight matrix $W_{\text{dec}} \in \mathbb{R}^{d \times d_{\text{sae}}}$ and a corpus of $N_{\text{seq}}$ sequences, we compute the absorption score $A_f \in [0, 1]$ for each latent feature $f$ as follows.

**Step 1 — Activating tokens.** For each latent $f$, identify the set of activating tokens:
$$T_f = \{t : \text{act}_f(t) > 0.01 \cdot \max_{t' \in \text{corpus}} \text{act}_f(t')\}$$
These are tokens where $f$ fires above the 1% activation threshold relative to its maximum on the corpus.

**Step 2 — Co-firing latents.** For each token $t \in T_f$, identify the top-5 other latents simultaneously active on that token:
$$\text{top5}(f, t) = \underset{c \neq f}{\text{argtop-5}} \, \text{act}_c(t)$$

**Step 3 — Partial reconstruction.** Compute a partial reconstruction of the original residual-stream activation $x_t \in \mathbb{R}^d$ using only feature $f$ and its top-5 co-firers:
$$x_t^{\text{partial}} = W_{\text{dec}}[f] \cdot \text{act}_f(t) + \sum_{c \in \text{top5}(f, t)} W_{\text{dec}}[c] \cdot \text{act}_c(t)$$

**Step 4 — Reconstruction variance explained (RVE).** Measure what fraction of the token's residual-stream variance is attributable to the co-firing features:
$$\text{RVE}_f(t) = 1 - \frac{\text{var}(x_t - x_t^{\text{partial}})}{\text{var}(x_t)}$$

**Step 5 — Absorption score.** The absorption score is the fraction of activating tokens where top-5 co-firers explain more than 80% of reconstruction variance:
$$A_f = \frac{1}{|T_f|} \sum_{t \in T_f} \mathbb{1}[\text{RVE}_f(t) > 0.80]$$

An absorption score $A_f = 1$ means the feature's variance is fully recoverable from its co-firers alone on every activating token; the feature contributes no independent directional signal. $A_f = 0$ means the feature is fully independent.

## 3.2 Design Rationale

**Top-5 co-firers.** Five is the smallest co-firer count that balances sensitivity against noise. Using fewer co-firers (top-3) dilutes the absorption signal by missing secondary absorbers; using more (top-10) introduces high-rank co-occurrences unrelated to absorption. Empirically, top-5 captures absorption effects in pilot experiments without diluting the metric with incidental co-activations.

**80% RVE threshold.** We calibrate this threshold against random dictionary controls (Section 3.3). A random Gaussian decoder yields exactly 0% of latents above $A_f = 0.5$ by construction, confirming that the threshold cleanly separates learned structure from noise.

**Why not Pearson correlation?** Pearson correlation between $f$ and its co-firers measures linear association, not reconstruction quality. A feature may be linearly correlated with its co-firers without their directional contributions being sufficient to reconstruct $f$'s residual-stream variance. The RVE-based metric explicitly measures reconstruction quality, making it sensitive to whether the absorbed feature's directional signal is recoverable from co-firers alone.

## 3.3 Validation

We validate the absorption score on two control cases that produce known, interpretable outcomes.

**Random dictionary control.** We generate SAE decoders with the same dimensionalities as the real SAE but with random Gaussian columns normalized per column. Because these random directions have no structure, co-firers cannot reconstruct a target feature's variance any better than random chance. Absorption scores on these controls yield 0.00% of latents with $A_f > 0.5$ by design, providing a null baseline.

**Always-on features.** Features that activate on more than 90% of corpus tokens are excluded from analysis. These bias-like features co-fire with every other feature trivially and would inflate absorption scores artificially. In practice, fewer than 1% of latents are always-on and excluded.

**Threshold sensitivity.** We vary the RVE threshold (0.70, 0.80, 0.90) and co-firer count (top-3, top-5, top-10) across all pilot experiments. Rankings of latents by absorption score are stable across these variations, confirming that the metric detects a genuine structural signal rather than a threshold artifact.

---

# 4. Experimental Setup

## 4.1 Models and SAEs

We use GPT-2 small (124M parameters, 12 layers, $d_{\text{model}} = 768$) as our base model. SAEs are loaded from SAELens (`sae_lens >= 0.5.0`), specifically the `gpt2-small-res-jb` release, which provides residual-stream SAEs for all 12 layers. We audit layers $\ell \in \{0, 2, 4, 6, 8, 10\}$ — spanning shallow to deep — to characterize depth dependence of absorption.

The primary dictionary size is $d_{\text{sae}} = 24{,}576$ (the full release). For the dictionary-size experiment (H5), sub-dictionaries of $d_{\text{sae}} = 2{,}048$ and $8{,}192$ are simulated by cumulatively subsampling latents from the full SAE, prioritizing absorbable latents to provide an upper-bound estimate of absorption at smaller sizes.

## 4.2 Dataset

Our analysis corpus consists of $N_{\text{seq}} = 100$ sequences of $T = 128$ tokens each (pilot experiment, seed 42), drawn from `monology/pile-uncopyrighted` — the standard SAELens evaluation dataset. Token frequencies for the H2 analysis are computed over the full Pile validation split.

## 4.3 Hypotheses Tested

We pre-registered five hypotheses and falsification criteria before running any experiments:

| ID | Hypothesis | Falsification Criterion |
|----|------------|--------------------------|
| H1 | $>20\%$ of mid-layer latents have $A_f > 0.5$ | $<10\%$ prevalence across layers 4-10 |
| H2 | Low-frequency latents absorbed at $\geq 2\times$ rate of high-frequency | Spearman $r_s \geq 0$ OR ratio $<2\times$ |
| H3 | Higher sparsity (L1) monotonically increases absorption | Non-monotonic trend in L0 vs. $A_f$ |
| H4 | High-absorption patching reduces faithfulness by $\geq 5$pp vs. low-absorption | Diff $<5$pp |
| H5 | Larger dictionary size reduces absorption | Positive or non-monotonic correlation with $d_{\text{sae}}$ |

---

# 5. Activation Patching Protocol

For H4 (downstream impact on circuit discovery), we use activation patching on a factual recall task to measure whether absorption level predicts causal importance.

**Task and prompts.** The clean prompt is `"The capital of France is"` (target: `" Paris"`). The corrupted prompt is `"The capital of Germany is"` (target: `" Berlin"`). The logit difference between clean and corrupted runs is $\Delta_{\text{logit}} = \log p_{\text{clean}}(y) - \log p_{\text{corrupt}}(y)$.

**Patching conditions.** We patch four conditions at layer 8:

1. **Raw residual**: direct patching on the residual stream (no SAE bottleneck)
2. **SAE all latents**: patching through the full SAE decoder using all $d_{\text{sae}} = 24{,}576$ latents
3. **SAE low-absorption**: patching using only the bottom 10% of latents ranked corpus-wide by $A_f$ score
4. **SAE high-absorption**: patching using only the top 10% of latents ranked corpus-wide by $A_f$ score

**Faithfulness metric.** We compute the fraction of the clean-to-corrupted logit difference restored by patching:
$$\text{faithfulness} = \frac{\Delta_{\text{patch}}}{\Delta_{\text{logit}}}$$

**Key comparison.** High-absorption latents should, if absorption degrades causal analysis, produce faithfulness at least 5 percentage points lower than low-absorption latents. A diff $< 5$pp falsifies this hypothesis.

---

# 6. Computational Setup

All pilot experiments were executed on a single NVIDIA GPU. The largest memory footprint is the full $d_{\text{sae}} = 24{,}576$ SAE at layer 8 with cached activations for 100 sequences ($\approx$12,800 tokens $\times$ 768 dimensions), which fits comfortably on a single consumer GPU.

**Pilot runtimes (single GPU):**

| Experiment | Target Hypothesis | Runtime |
|------------|-------------------|---------|
| H1 | Absorption prevalence at layer 8 | $\approx$ 8 min |
| H3 | Sparsity-absorption relationship across 6 layers | $\approx$ 25 min |
| H4 | Circuit faithfulness comparison | $\approx$ 22 s |
| H5 | Dictionary size effect | $\approx$ 152 s |

Total pilot compute budget was under 2 GPU-hours.

<!-- FIGURES
- Figure: gen_pipeline.pdf — Architecture diagram of the experimental pipeline: corpus tokenization, model forward pass with SAE hook, per-latent absorption scoring, and downstream patching evaluation.
- Table 2: L0 and Absorption by Layer — inline table in Section 4 (Experimental Setup) showing mean L0, mean absorption, and % $A_f > 0.5$ for each of the six audited layers.
- None
-->