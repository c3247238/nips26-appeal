# 4 Experiments

We evaluate EDA and cross-directional metrics on the first-letter task using GPT-2 Small
(117M parameters) with SAELens pre-trained sparse autoencoders (SAEs).
All experiments use the gpt2-small-res-jb release as the primary suite, with layer 6
($d_\text{sae} = 24{,}576$, $L_0 \approx 51$) as the main evaluation point.
Ground-truth absorption labels come from FeatureAbsorptionCalculator
(Chanin et al. 2024, sae_spelling; henceforth exact labels), which applies integrated-gradients
ablation to identify which SAE features fail to fire when the corresponding first-letter concept
is present in the input.
The exact label set contains $n_\text{pos} = 18$ absorbed features out of 24,576
(base rate $= 7.3 \times 10^{-4}$).

**Baselines.** We compare EDA against four weight-only or activation-statistic alternatives:
activation frequency (inverted), decoder norm, encoder norm, and the raw cosine similarity
$\cos(\hat{e}_j, d_j)$ (which is $1 - \text{EDA}$).
Statistical significance of AUROC uses a permutation null ($n = 100$ permutations of
absorption labels), reported as $z_\text{null}$.

## 4.1 EDA Validation Against Exact Labels

Figure 2 shows EDA distributions for letter versus non-letter features at layer 6 and layer 10.
Table 1 reports all detector metrics.

Against exact Chanin labels ($n_\text{pos} = 18$), EDA achieves $\text{AUROC} = 0.650$,
$z_\text{null} = 2.49$ (above the 2-SD significance threshold), and
$\text{AUPRC} = 0.00153$ (2.09$\times$ the base-rate baseline).
Proxy labels (letter features with high decoder-probe alignment; $n_\text{pos} = 50$,
Jaccard overlap with exact labels $= 0.115$) yield $\text{AUROC} = 0.659$ and
Cohen's $d = 0.533$ ($p = 1.6 \times 10^{-4}$), confirming the signal is stable across
label definitions.

The strongest per-feature signal comes from the cross-directional metric
$\cos(\hat{e}_p, d_c)$, defined as the cosine similarity between the parent encoder
direction and the child decoder direction.
This metric achieves $\text{AUROC} = 0.730$ (Cohen's $d = 0.552$, $p = 2.8 \times 10^{-9}$,
$z_\text{null} = 6.38$), and captures absorbed features from the parent side of the
hierarchical pair, complementing EDA's child-side measurement.
The child-side cross-directional metric $\cos(\hat{e}_c, d_p)$ achieves
$\text{AUROC} = 0.681$ (Cohen's $d = 0.517$, $p = 2.7 \times 10^{-6}$, $z_\text{null} = 4.63$).

**The encoder norm finding.** Encoder norm yields the highest AUROC among individual
weight-only features ($\text{AUROC} = 0.757$, $\text{AUPRC}/\text{base} = 5.68\times$).
A DeLong test comparing EDA to encoder norm gives $p = 0.153$, so the difference is not
statistically significant at $\alpha = 0.05$.
The letter-feature mean encoder norm is $3.279$ versus $2.575$ for non-letter features.
Because encoder norm is not mechanistically predicted by the EDA theory and may reflect
polysemanticity rather than absorption specifically, we treat it as a confounded baseline
and examine its source in Section 4.2.

**Decoder baselines fail.** Decoder norm is near-random ($\text{AUROC} = 0.515$), and
the raw cosine $\cos(\hat{e}_c, d_c)$ achieves only $\text{AUROC} = 0.350$ — strongly
anti-correlated with absorption labels, confirming that simply checking encoder-decoder
alignment in the aligned direction is the wrong signal.
Activation frequency inverted achieves $\text{AUROC} = 0.595$, consistent with the
observation that absorbed features tend to fire less frequently.

**Layer 10 reversal.** At layer 10, EDA computed on the same gpt2-small-res-jb suite gives
$\text{AUROC} = 0.256$ (Cohen's $d = -0.890$, $p = 6.8 \times 10^{-10}$) — the EDA polarity
reverses, meaning letter features at L10 have *lower* EDA than non-letter features.
This finding is consistent with the mechanistic conjecture (Proposition 2): post-absorption
re-alignment of the encoder toward the child concept could reduce EDA at late layers once the
absorption stabilizes. We report this as an open empirical finding; the exact L10 mechanism
is not resolved by our current data.

![EDA distributions at L6 and L10, showing polarity reversal](figures/fig2_eda_distributions.pdf)

---

**Table 1: Main Detection Results — EDA and Baselines (GPT-2 Small, Layer 6, $d_\text{sae} = 24{,}576$)**

| Detector | AUROC | AUPRC/base | Cohen's $d$ | $p$-value | $z_\text{null}$ |
|---|---|---|---|---|---|
| $\cos(\hat{e}_p, d_c)$ max | **0.730** | — | 0.552 | $2.8 \times 10^{-9}$ | 6.38 |
| Encoder norm | 0.757 | 5.68$\times$ | — | — | — |
| $\cos(\hat{e}_c, d_p)$ mean | 0.681 | — | 0.517 | $2.7 \times 10^{-6}$ | 4.63 |
| EDA: $1 - \cos(\hat{e}_c, d_c)$ | 0.650 | 2.09$\times$ | 0.533 | $1.6 \times 10^{-4}$ | 2.49 |
| Freq. ratio (inverted) | 0.595 | 1.33$\times$ | — | — | — |
| Decoder norm | 0.515 | 1.02$\times$ | — | — | — |
| $\cos(\hat{e}_c, d_c)$ raw | 0.350 | — | — | — | — |
| Random baseline | 0.500 | 1.00$\times$ | — | — | — |

*EDA and $\cos(\hat{e}_c, d_c)$ raw are mathematical inverses; identical performance confirms the implementation. Encoder norm AUROC is not significantly higher than EDA (DeLong $p = 0.153$). Cross-directional metrics use proxy labels ($n_\text{pos} = 50$); EDA validation uses exact labels ($n_\text{pos} = 18$). $z_\text{null}$ is the $z$-score above permutation null AUROC.*

---

## 4.2 EDA Decomposition: Encoder vs. Decoder Alignment

Figure 4 decomposes EDA into its encoder and decoder components to test the mechanistic
conjecture underlying Proposition 2.

For letter features at layer 6 ($n = 50$), the decoder aligns more strongly with the
first-letter probe than the encoder:
decoder-probe cosine mean $= 0.383$ versus encoder-probe cosine mean $= 0.139$
(paired $t$-test: $t = -38.3$, $p = 3.5 \times 10^{-38}$, diff $= -0.244$).
As predictors of letter-feature membership, the decoder achieves AUROC $= 1.000$ and the
encoder achieves $\text{AUROC} = 0.991$ — both near-perfect, but the decoder is strictly
stronger.
For non-letter features, the gap collapses: decoder mean $= 0.099$, encoder mean $= 0.056$
(diff $= -0.043$).

This pattern is consistent with the mechanistic conjecture: the decoder remains anchored to
the child concept direction (letter identity), while the encoder is partially pulled toward the
parent direction during training, reducing encoder-probe alignment relative to decoder-probe
alignment.

**Encoder norm as confound.** Letter features have systematically larger encoder norms
($3.279 \pm 0.544$) than non-letter features ($2.575 \pm 0.707$).
This inflated norm is consistent with a feature whose encoder direction is under competing
gradient pressure — both the child reconstruction signal and the parent activation signal
act on $\hat{e}_c$ — but also with polysemanticity.
Because encoder norm does not distinguish between these two explanations, it cannot be
interpreted as a direct absorption signal without additional evidence.
We note this as an unresolved tension: the mechanistic story predicts encoder norm should
be elevated in absorbed features, but the same prediction follows from polysemanticity.

**EDA magnitude tension.** A distinct unresolved tension: the observed mean EDA for letter
features is $0.671$ (implying a $\sim 48°$ angle between $\hat{e}_c$ and $d_c$),
whereas Proposition 1 predicts that absorption onset occurs at small $\theta_{p,c}$
(parent-child decoder angle), which should correspond to small EDA in a newly absorbed feature.
Large observed EDA is consistent with long-term evolution of the encoder direction post-absorption
but is not directly predicted by the theory. We report this as an open question.

![Encoder and decoder probe alignment for letter vs. non-letter features (L6)](figures/fig4_enc_dec_alignment.pdf)

## 4.3 EDA Across Architectures and Scales

Figure 3 plots EDA$_\Delta$ (letter minus non-letter mean EDA) across 11 SAE configurations.

**Standard/L1 suite.** For the five gpt2-small-res-jb configurations (layers 2–10, width 24,576),
EDA$_\Delta$ is positive at all five layers and peaks at L4 ($+0.045$, AUROC $= 0.716$) and L6
($+0.050$, AUROC $= 0.702$), declining toward L10 ($+0.005$, AUROC $= 0.505$).
The L10 near-zero EDA$_\Delta$ — despite the strongly reversed AUROC seen in the pairwise analysis
— reflects the fact that the full layer 10 letter-feature population is heterogeneous.

**TopK SAE comparison.** A TopK SAE ($k = 32$, width 32,768, layer 6) shows lower mean letter-feature
EDA ($0.476$) compared to the L1-penalized SAE at the same layer ($0.676$), with the fraction of
letter features exceeding $\text{EDA} > 0.5$ dropping from 100\% to 25\%.
Both architectures show statistically significant positive EDA$_\Delta$ (Wilcoxon $p = 3.3 \times 10^{-4}$
for TopK), but the signal is weaker for TopK, consistent with the prediction that exact-sparsity
constraints alter the gradient landscape during training.

**AJT training regime reversal.** Three AJT-trained SAEs at layer 6 (width 46,080, L0 ranging
34.5–81.0) exhibit strongly negative EDA$_\Delta$: $-0.204$, $-0.177$, and $-0.217$,
with AUROC values of $0.154$, $0.354$, and $0.158$.
Letter features in AJT SAEs have *lower* EDA than non-letter features, the opposite of the L1-SAE
pattern.
This polarity reversal — present across all three AJT variants regardless of their different
sparsity levels — suggests that the AJT training regime fundamentally alters the geometry of
encoder-decoder dissociation for absorbed features.
The mechanism is not established by the current data; we conjecture that AJT's non-L1 sparsity
formulation changes the gradient signal that pulls encoder directions during training.

**Width analysis.** The feature-splitting suite at layer 8 (widths 12k, 24k, 49k, 98k,
matched $L_0 \approx 51$) shows decreasing EDA$_\Delta$ as width increases:
$+0.028$ at 12k, $+0.041$ at 24k, $+0.012$ at 49k, and $-0.017$ at 98k.
As SAE width grows, feature splitting produces additional features that cover fine-grained
child concepts, and the EDA signal dilutes because not all newly split child features will have
the same absorption geometry.

![EDA$_\Delta$ and AUROC across 11 SAE configurations, by layer, architecture family, and width](figures/fig3_eda_scaling.pdf)

## 4.4 Absorption Phase Stability

Figure 5 shows absorption rates across all 11 SAE configurations as a function of $1/L_0$.

All 11 configurations produce uniformly high absorption rates: the range is $0.876$–$0.978$,
with mean $0.950$.
The standard/L1 jb suite alone spans $0.938$–$0.967$ across layers 2–10.
AJT variants, despite their reversed EDA polarity, show absorption rates of $0.919$, $0.876$,
and $0.978$, all substantially above zero.

Testing whether absorption rate follows a sigmoid-shaped transition in $1/L_0$
(Hypothesis H4): the likelihood-ratio test comparing sigmoid versus linear fit gives
LRT $p = 0.456$ and BIC difference $= -3.22$ (negative means sigmoid is not preferred),
with the sigmoid inflection estimated at $L_0 \approx 81$ — outside the observed range.
Spearman rank correlation between $1/L_0$ and absorption rate across all 11 configurations:
$\rho = -0.482$, $p = 0.133$ (not significant).
Within the primary jb suite only: $\rho = -0.100$, $p = 0.873$.
These results indicate that no phase transition in sparsity is detectable within the tested
$L_0$ range of 18–81.

**Hysteresis experiment.** To test whether the absorbed state is metastable,
we fine-tuned a high-sparsity SAE (gpt2-small-res-jb, layer 2, baseline $L_0 = 33.7$,
absorption rate $= 0.959$) for 500 steps with a 5$\times$-reduced sparsity coefficient
($\lambda_\text{finetune} = 0.2\lambda_\text{original}$), reaching $L_0 = 68.6$.
After 500 fine-tuning steps, absorption rate $= 0.960$ — effectively unchanged
(fraction of baseline $= 1.001$).
A from-scratch SAE trained at the lower sparsity level achieves absorption rate $= 0.964$
($L_0 = 84.2$), also high.
The checkpoint trajectory shows no decrease in absorption at any step: $0.959 \to 0.959 \to
0.960 \to 0.960 \to 0.960$ at steps 100, 200, 300, 400, 500.
These data are consistent with the absorbed state being metastable: reducing sparsity at
fine-tuning time does not escape the absorbed configuration.
We note that the interpretation is constrained by the saturation: because all tested
$L_0$ values produce very high absorption, classical hysteresis (where a second stable
non-absorbed state exists at low sparsity) cannot be tested in this regime.

## 4.5 Cross-Domain Absorption

To determine whether feature absorption is specific to orthographic hierarchies or extends to
semantic concept hierarchies, we measure absorption rates on three hierarchy types:
(1) first-letter, (2) animate-inanimate, and (3) noun-proper noun.

**First-letter: GO.** The first-letter hierarchy yields absorption rate $= 0.0083$
(120 IG-ablation events across 8 letters: a, g, h, i, j, l, m, q),
ratio-to-null $= 10.0$, 95\% bootstrap CI $[0, 0.029]$.
This confirms that the sae_spelling methodology detects genuine absorption, with one letter
(h) showing absorption rate $= 0.067$ and the remaining seven letters at 0.

**Semantic hierarchies: NO\_GO.** Both animate-inanimate and noun-proper noun hierarchies
produce absorption rate $\approx 0$ ($9.999 \times 10^{-9}$, indistinguishable from the
shuffled-label null), ratio-to-null $= 1.0$.
Under the animate-inanimate hierarchy, the five parent latents (features 3152, 18287, 18617,
16169, 3278) do not produce measurable child suppression over 20 test words.
The noun-proper noun hierarchy shows the same null pattern.

**Interpretation.** Feature absorption in GPT-2 Small layer 6 is specific to orthographic
(first-letter) hierarchies at the tested scale; semantic concept absorption does not emerge.
This is a scoped null result: it does not rule out semantic absorption in larger models or
at different layers, and our experimental design cannot test all possible semantic hierarchies.
The parent feature identification method for semantic hierarchies (top-5 probe-aligned latents)
may also underestimate the relevant feature set.
Experiments on Gemma Scope SAEs, which operate at larger scale and may develop richer semantic
feature hierarchies, are necessary to determine whether this null result generalizes.

---

**Summary of experimental findings.** EDA achieves statistically significant detection of
absorbed features (AUROC $= 0.650$, $z_\text{null} = 2.49$) using only SAE weight matrices,
with no probe training required. The cross-directional metric $\cos(\hat{e}_p, d_c)$ is
stronger (AUROC $= 0.730$). Absorption rates are uniformly high (0.876–0.978) across all
tested configurations, no phase transition is detected, and the absorbed state is metastable
to fine-tuning. The EDA signal is architecture-dependent: L1-penalized SAEs show positive
EDA$_\Delta$; AJT-trained SAEs show reversed polarity; TopK SAEs show reduced signal.
Absorption is specific to the first-letter orthographic hierarchy in GPT-2 Small; semantic
hierarchies produce null results at this scale.

<!-- FIGURES
- Figure 2: gen_fig2_eda_distributions.py, fig2_eda_distributions.pdf — EDA violin distributions for letter vs. non-letter features at L6 (AUROC=0.659, Cohen's d=0.533) and L10 (AUROC=0.256, Cohen's d=-0.890, reversed polarity)
- Figure 3: gen_fig3_eda_scaling.py, fig3_eda_scaling.pdf — EDA_delta and EDA AUROC across 11 configurations: primary jb suite (L2-L10), AJT suite (reversed), width sweep (L8 12k-98k)
- Figure 4: gen_fig4_enc_dec_alignment.py, fig4_enc_dec_alignment.pdf — Encoder vs. decoder probe alignment scatter and bar chart for L6 letter features (dec AUROC=1.000 > enc AUROC=0.991, diff=-0.244)
- Figure 5: gen_fig5_phase_stability.py, fig5_phase_stability.pdf — Absorption rate vs. 1/L0 across 11 configs with sigmoid/linear fit overlay; LRT p=0.456, BIC diff=-3.22
- Table 1: inline — Main detection results: all detectors with AUROC, AUPRC/base, Cohen's d, p-value, z_null
-->
