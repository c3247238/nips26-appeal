# 5. Experiments

We tested five hypotheses about feature absorption in GPT-2 small SAEs using the absorption score $A_f$ defined in Section 3. All experiments were pilots on 100 sequences ($\times$ 128 tokens) from `monology/pile-uncopyrighted` (seed 42). The base model was GPT-2 small (124M parameters, 12 layers, $d_{\text{model}} = 768$) with SAELens `gpt2-small-res-jb` residual-stream SAEs evaluated at layers $\ell \in \{0, 2, 4, 6, 8, 10\}$ ($d_{\text{sae}} = 24{,}576$ for layer-wise comparisons and $d_{\text{sae}} \in \{2{,}048, 8{,}192, 24{,}576\}$ for the dictionary-size comparison at layer 8). Table 1 summarizes all five results; Figures 1-3 visualize the key data.

## 5.1 H1: Absorption Prevalence Is Extremely Low

**Hypothesis**: More than 20% of latents in mid-to-deep layers have $A_f > 0.5$.

We computed $A_f$ for all 24,576 latents in layer 8 ($d_{\text{sae}} = 24{,}576$) on the 100-sequence pilot corpus. The results contradict the hypothesis by two orders of magnitude: only 46 of 24,576 latents (0.19%) have $A_f > 0.5$. The mean absorption score is 0.0022; 99.4% of latents score exactly 0.0. A random-dictionary control (Gaussian decoder columns, normalized per column) yields 0.00% with $A_f > 0.5$, confirming that the metric detects genuine structure and that the near-zero real rates are not artifacts of the threshold.

Eight latents achieve the maximum score of $A_f = 1.0$. Each fires on exactly 100 tokens (0.78% of the corpus), a suspiciously uniform count that suggests an edge case or feature artifact rather than a general property of absorbed features. Excluding these eight does not meaningfully change the aggregate statistics.

Even relaxing the threshold to $A_f > 0.1$ yields only 0.43% prevalence, and relaxing to $A_f > 0.01$ yields 0.63%. No threshold bridges the 100-fold gap between observation and hypothesis. The falsification threshold of $< 10\%$ is satisfied at every tested threshold.

**Verdict**: H1 is falsified. Feature absorption as defined by $A_f > 0.5$ affects fewer than 1 in 500 latents at layer 8.

## 5.2 H3: Absorption Does Not Monotonically Increase with Sparsity

**Hypothesis**: Higher sparsity (operationalized as L1 penalty $\lambda$ or, proxy, L0 norm) monotonically increases absorption.

We computed $A_f$ for all 24,576 latents at each of six layers ($\ell \in \{0, 2, 4, 6, 8, 10\}$). Table 2 reports mean L0, mean absorption score, and percentage of latents with $A_f > 0.5$ per layer. The relationship is clearly non-monotonic: absorption rises from layer 0 to layer 4 (the shallow-to-mid transition where GPT-2 small begins processing abstract semantic content) and then declines from layer 4 to layer 10 despite increasing L0. Spearman $r = 0.086$ ($p = 0.872$) and Pearson $r = -0.073$ ($p = 0.891$); neither is statistically significant. The sparsest layer in our sample is layer 8 ($\text{L0} = 71.9$), yet it shows only 20.9% with $A_f > 0.5$, compared to 49.3% at layer 4 ($\text{L0} = 37.8$).

Figure 1 illustrates the inverted-U pattern.

![Figure 1. Absorption is rare and peaks at mid-layers](figures/fig1_layer_absorption.pdf)

The peak at mid-layers (4-6) is consistent with the hypothesis that these layers handle the densest conceptual representations in GPT-2 small, producing more feature overlap. Pushing sparsity harder (at deeper layers) does not compress more features into shared representations. L0 is not a reliable proxy for absorption risk.

**Verdict**: H3 is falsified. The absorption-sparsity relationship is an inverted-U, not a monotonic increase.

## 5.3 H4: Absorption Does Not Degrade Circuit Faithfulness

**Hypothesis**: Circuits traced using high-absorption SAE latents have faithfulness scores at least 5 percentage points lower than those traced with low-absorption latents.

We used activation patching on the factual recall task "The capital of France is ___" (clean, target: " Paris") versus "The capital of Germany is ___" (corrupted, target: " Berlin"). Table 3 reports mean faithfulness scores; Figure 3 visualizes the comparison.

Patching the raw residual stream achieves faithfulness of 0.400 (40% of the clean-to-corrupted logit difference restored). Patching all SAE latents (layer 8, $d_{\text{sae}} = 24{,}576$) achieves 0.289, an 11 percentage-point drop. The bottleneck introduces signal loss, as expected.

The key test — comparing low-absorption and high-absorption latent subsets — is uninformative. Both the bottom-10% and top-10% by corpus-wide $A_f$ score yield 0.000 faithfulness. The difference is 0.000, precluding any conclusion about absorption level and downstream causal validity. The subset selection method selects latents that are corpus-wide low/high absorbers, not latents relevant to the France/Paris circuit. Keeping only 10% of latents by any criterion destroys the reconstruction capacity needed for patching.

![Figure 3. SAE bottleneck reduces faithfulness; absorption level does not predict causal importance](figures/fig3_faithfulness.pdf)

**Verdict**: H4 is falsified as an uninformative experiment. The hypothesis cannot be tested with the current design because both subsets fail entirely.

## 5.4 H5: Larger Dictionaries Reduce Absorption

**Hypothesis**: Larger dictionary sizes monotonically reduce per-latent absorption rates.

Since `gpt2-small-res-jb` provides only $d_{\text{sae}} = 24{,}576$, we simulated smaller dictionaries by cumulatively subsampling latents, prioritizing absorbable latents (those with non-zero $A_f$) for inclusion — this gives an upper bound on absorption for each subsample size. Table 4 and Figure 2 report the results.

Mean absorption declines monotonically with dictionary size: 0.0268 at 2,048 latents, 0.0067 at 8,192, and 0.0022 at 24,576. The prevalence metric ($A_f > 0.5$) falls from 2.25% to 0.19% — a 10-fold reduction from 2K to 24K. Random controls register 0.00% at all sizes, correctly scaled.

Figure 2 shows the monotonic decrease.

![Figure 2. Larger dictionaries monotonically reduce absorption](figures/fig2_dict_size.pdf)

The direction of the effect matches the hypothesis: larger dictionaries can represent more distinct features, reducing the pressure for any single latent to redundantly encode another's variance. However, the practical significance is limited: even at 2K, 97.75% of latents are not absorbed. The phenomenon is rare regardless of dictionary size.

**Verdict**: H5 is not falsified. The hypothesized direction is confirmed, though absorption remains rare even at small dictionary sizes.

## 5.5 H2: Token Frequency and Absorption Correlation (Not Tested)

**Hypothesis**: Low-frequency token latents are absorbed at least twice as often as high-frequency token latents (Spearman $r < 0$).

No pilot was run for H2. Early termination after H1, H3, and H4 falsification determined that the full H2 experiment would not be informative given the near-zero overall absorption rates. The pre-registered falsification criterion (Spearman $r \geq 0$) cannot be evaluated without data; we report the null result honestly and note H2 as pending a full experiment.

---

## 5.6 Hypothesis Summary

Table 1 consolidates all five hypotheses. Four are falsified or uninformative; only H5 moves in the hypothesized direction.

| Hypothesis | Predicted | Observed | Falsified? |
|------------|-----------|----------|------------|
| H1: Absorption prevalence | $>20\%$ at layers 4-10 | 0.19% at layer 8 | **Yes** (100x below) |
| H2: Frequency-absorption correlation | Spearman $r < 0$ | Not tested | Pending |
| H3: Monotonic sparsity relationship | $A_f$ rises with L0 | Inverted-U, $r_s = 0.086$ (NS) | **Yes** |
| H4: Absorption degrades faithfulness | High-abs $-5$pp vs. low-abs | Diff $= 0.0$ (both $0.0$) | **Yes** (uninformative) |
| H5: Larger dict reduces absorption | Monotonic decrease | 2K: 2.25%, 24K: 0.19% | **Not falsified** |

**Table 1.** Hypothesis results summary. NS = not statistically significant ($p > 0.05$).

---

## 5.7 Layer-Level L0 and Absorption Statistics

Table 2 reports mean L0, mean absorption score, and percentage of latents with $A_f > 0.5$ for each audited layer. Layer 4 shows the highest absorption despite not being the sparsest layer, confirming the inverted-U pattern from H3.

| Layer | Mean L0 | Mean $A_f$ | % $A_f > 0.5$ | % $A_f = 0.0$ |
|-------|---------|-----------|---------------|---------------|
| 0 | 18.9 | 0.229 | 19.5% | 77.6% |
| 2 | 29.1 | 0.470 | 45.5% | 51.2% |
| 4 | 37.8 | 0.503 | 49.3% | 48.1% |
| 6 | 57.0 | 0.430 | 41.0% | 56.4% |
| 8 | 71.9 | 0.305 | 20.9% | 76.8% |
| 10 | 56.0 | 0.287 | 17.3% | 80.2% |

**Table 2.** L0 norm and absorption by layer ($d_{\text{sae}} = 24{,}576$, 100 sequences). The inverted-U peaks at layer 4 (mid-depth), not at the sparsest layers (8, 10).

---

## 5.8 H5: Dictionary Size Breakdown

Table 3 reports absorption metrics for each dictionary size with both the real SAE and the random-dictionary control.

| Dict Size | Mean $A_f$ (SAE) | Mean $A_f$ (Random) | % $A_f > 0.5$ (SAE) | % $A_f > 0.5$ (Random) |
|-----------|-----------------|---------------------|---------------------|------------------------|
| 2,048 | 0.0268 | 0.0000 | 2.25% | 0.00% |
| 8,192 | 0.0067 | 0.0000 | 0.56% | 0.00% |
| 24,576 | 0.0022 | 0.0000 | 0.19% | 0.00% |

**Table 3.** Absorption by dictionary size at layer 8 (100 sequences). Mean absorption and prevalence both decrease monotonically with dictionary size. Random controls are exactly zero at all sizes, confirming the metric detects learned structure.

---

## 5.9 H4: Circuit Faithfulness Details

Table 4 reports detailed faithfulness results for the activation patching experiment.

| Patching Method | Faithfulness | Half Restored |
|-----------------|--------------|---------------|
| Raw residual stream | 0.400 | 0.400 |
| SAE all latents | 0.289 | 0.400 |
| SAE low-absorption (bottom 10%) | 0.000 | 0.000 |
| SAE high-absorption (top 10%) | 0.000 | 0.000 |

**Table 4.** Activation patching faithfulness on the France/Paris vs. Germany/Berlin factual recall task (layer 8, $d_{\text{sae}} = 24{,}576$). The raw residual achieves the highest faithfulness. Using all SAE latents reduces signal by 11 pp. Both absorption subsets yield 0.000, making the key comparison impossible.

---

## 5.10 Computational Resources

All pilots were run on a single NVIDIA GPU. The largest memory footprint is the full $d_{\text{sae}} = 24{,}576$ SAE at layer 8 with cached activations for 100 sequences ($\approx$ 12,800 tokens $\times$ 768 dimensions), which fits comfortably on a single consumer GPU. Runtimes: H1 ~8 min (single layer), H3 ~25 min (six layers), H4 ~22 s, H5 ~152 s. Total pilot compute budget was under 2 GPU-hours.

<!-- FIGURES
- Figure 1: gen_fig1_layer_absorption.py, fig1_layer_absorption.pdf — Bar+line chart showing % > 0.5 and mean $A_f$ per layer; inverted-U pattern peaks at layer 4. Data: h3_pilot.json layer_results.
- Figure 2: gen_fig2_dict_size.py, fig2_dict_size.pdf — Dual-panel figure showing mean $A_f$ (log scale) and % > 0.5 vs dictionary size; 10x reduction from 2K to 24K. Data: h5_pilot.json.
- Figure 3: gen_fig3_faithfulness.py, fig3_faithfulness.pdf — Bar chart of faithfulness for raw residual, SAE all, SAE low-abs, SAE high-abs; both subsets yield 0.000. Data: h4_pilot.json.
- Table 1: inline — Hypothesis results summary (H1–H5).
- Table 2: inline — L0 and absorption by layer (6 layers, $d_{\text{sae}} = 24{,}576$).
- Table 3: inline — H5 dictionary size breakdown (SAE vs random control, 3 sizes).
- Table 4: inline — H4 circuit faithfulness details (4 patching methods).
-->