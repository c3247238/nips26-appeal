# 5 Discussion

## 5.1 What EDA Establishes and What It Does Not

EDA achieves AUROC $= 0.650$ ($z_\text{null} = 2.49$) against exact FeatureAbsorptionCalculator
labels, and the cross-directional metric $\cos(\hat{e}_p, d_c)$ achieves AUROC $= 0.730$
(Cohen's $d = 0.552$, $p = 2.8 \times 10^{-9}$), both statistically significant.
These results establish that SAE weight matrices alone carry absorption signal, without probes
and without activation data.

The practical scope of this claim is precise. The positive result holds for L1-penalized
(res-jb) SAEs at mid-layers (L4–L8) of GPT-2 Small, on the first-letter orthographic hierarchy.
EDA does not achieve AUROC $\geq 0.7$ everywhere: L2 ($= 0.596$), L10 ($= 0.505$), and all
three AJT configurations ($0.154$–$0.354$) fall short. The correct reading is that EDA is
a reliable signal specifically in the regime where L1-penalized training has had time to
establish absorption but before late-layer encoder re-alignment erases the dissociation.

Practitioners evaluating a deployed SAE can use EDA as a first-pass screen on the condition
that they (a) know the SAE was trained with L1 sparsity and (b) are examining mid-network
layers. Features ranked in the top EDA percentile are enriched for absorbed features
(AUPRC $= 2.09\times$ base rate at L6), providing a tractable candidate set for downstream
IG-based verification — the $n_+ = 18$ exact positives, which require IG ablation to confirm,
can be targeted from a much smaller candidate pool rather than an exhaustive sweep over all
24,576 features.

## 5.2 The Encoder-Decoder Decomposition: What the Data Supports

The most direct mechanistic evidence comes from Figure 4. For 50 letter features at layer 6,
decoder-probe cosine alignment averages $0.383$ versus encoder-probe cosine alignment of $0.139$
(paired $t$, diff $= -0.244$, $p = 3.5 \times 10^{-38}$). The decoder points toward the
letter-identity direction; the encoder does not, even though both vectors nominally belong to
the same feature. This differential alignment is precisely what Proposition 2 (mechanistic
conjecture) predicts: reconstruction loss anchors the decoder to the child concept, while
gradient pressure from parent-only contexts pulls the encoder toward the parent direction.

The pattern disappears for non-letter features: encoder and decoder probe alignment are
$0.056$ and $0.099$, a much smaller gap (diff $= -0.043$). The decoder $>$ encoder alignment
asymmetry is therefore specific to absorbed-candidate features, not a universal property of
the SAE geometry.

What the data does *not* support directly is that this dissociation proves Proposition 2's
causal mechanism. The observation is consistent with Proposition 2 but also consistent with
alternative explanations, including amortization gap effects (O'Neill et al. 2024), where the
encoder fails to specialize to a feature the decoder has already committed to. Distinguishing
causal absorption from amortization gap would require activation experiments beyond the
weight-only analysis presented here.

**EDA magnitude tension.** Proposition 1 predicts absorption is preferred when $\lambda >
\sin^2(\theta_{p,c})$, which implies small parent-child decoder angle at absorption onset.
Small $\theta_{p,c}$ should correspond to small EDA in a newly absorbed feature — yet the
observed letter-feature mean EDA is $0.671$ (implying roughly a $48°$ encoder-decoder angle).
One reconciliation is temporal: the theoretical threshold describes the *onset* of the
absorbed solution's energy advantage, while observed EDA reflects the *post-convergence* state
after substantial encoder drift. If the absorbed encoder continues drifting toward the parent
direction for the duration of training, large EDA can accumulate even when the initial
transition occurred at small $\theta_{p,c}$. This explanation is plausible but unverified;
we report it as an unresolved tension.

## 5.3 Informative Failure Modes

The three failure modes — L10 EDA reversal, AJT polarity reversal, and encoder norm
dominance — collectively provide more mechanistic insight than the positive results alone.

**L10 reversal.** At layer 10, letter features have *lower* EDA than non-letter features
(Cohen's $d = -0.890$, $p = 6.8 \times 10^{-10}$, AUROC $= 0.256$). One interpretation:
late-layer features of GPT-2 Small may include large numbers of non-absorbed letter-adjacent
features (e.g., positional or syntactic features that are highly encoder-decoder aligned),
diluting the L6 absorption signal in the opposite direction. A second interpretation, consistent
with Proposition 2: at late layers, the encoder of an absorbed feature may re-align toward the
child concept as training converges, reducing EDA even as the functional absorption persists.
Both interpretations predict that layer-specific calibration is required; neither is ruled out
by current data. The key practical implication is that EDA thresholds trained on mid-layer
SAEs cannot be transferred to late-layer SAEs without recalibration.

**AJT polarity reversal.** All three AJT-trained SAEs (res_sce-ajt, res_scl-ajt, res_sle-ajt)
show strongly negative EDA$_\Delta$ ($-0.204$, $-0.177$, $-0.217$), meaning absorbed-candidate
features have *lower* EDA in AJT SAEs than the non-absorbed background. This finding rules out
EDA as a universal absorption indicator: the signal direction depends on the training regime.
AJT training uses a non-L1 sparsity formulation; the difference in gradient structure may prevent
or reverse the encoder drift described in Proposition 2. Concrete investigation of AJT's
gradient dynamics is needed to determine the mechanism.

**Encoder norm dominance.** Encoder norm achieves AUROC $= 0.757$ — higher than EDA's $0.650$
but not statistically significantly so (DeLong $p = 0.153$). The mean encoder norm is $3.279$
for letter features versus $2.575$ for non-letter features. Two explanations are compatible:
(1) absorbed features experience competing gradient pressures, increasing the norm of $e_c$
as it balances child reconstruction and parent detection signals; (2) polysemantic features
with multiple strong associations have elevated encoder norms regardless of absorption.
Because encoder norm is not derived from the EDA theory and cannot be interpreted mechanistically
without distinguishing these two sources, it is treated as a confounded baseline rather than a
primary detector. The coincidence between elevated encoder norm and letter-feature membership
nevertheless warrants follow-up experiments that hold polysemanticity constant.

## 5.4 Phase Stability and the Implications for Remediation

Absorption rates are 0.876–0.978 across all 11 tested configurations (layers 2–10, widths
12k–98k, L1-penalized and AJT training regimes, $L_0$ range 18–81). No sigmoid-shaped
transition is detectable (LRT $p = 0.456$, BIC difference $= -3.22$), and no sparsity
dependence is measurable (Spearman $\rho = -0.482$, $p = 0.133$ across all 11; $\rho = -0.100$,
$p = 0.873$ within the primary jb suite).

The hysteresis experiment reinforces this picture: fine-tuning a high-absorption SAE
($\alpha = 0.959$) for 500 steps at one-fifth the original sparsity coefficient
(achieving $L_0 = 68.6$) does not reduce absorption ($\alpha = 0.960$, fraction of baseline
$= 1.001$). The checkpoint trajectory shows absorption stable at $0.959$–$0.960$ throughout.

Two practical implications follow from absorption phase stability. First, practitioners cannot
reduce absorption in a deployed SAE by adjusting L0 via activation-function threshold changes
or top-$k$ recalibration: the sparsity coefficient change required to escape absorption is
beyond anything achievable without retraining from scratch. Second, because all tested
hyperparameter configurations produce highly absorbed SAEs, the structural source of absorption
must reside in the training objective itself (consistent with Proposition 1) rather than in
any particular hyperparameter choice. Architectural interventions that modify the absorption
threshold — orthogonality penalties (OrtSAE), hierarchical codebooks (Matryoshka SAE), or
per-latent importance weighting — act on the objective directly, which is why they reduce
absorption while hyperparameter tuning does not.

The saturation of absorption rates throughout the tested range also means that the classical
hysteresis experiment (comparing the absorbed state at high sparsity to a non-absorbed stable
state at low sparsity) cannot be executed in this regime: no non-absorbed stable state is
accessible via parameter changes within the tested L0 range. Testing for hysteresis would
require either an architectural change that creates a non-absorbed equilibrium or experiments
on much wider SAEs at much lower sparsity, which we leave for future work.

## 5.5 Cross-Domain Scope: Orthographic vs. Semantic Absorption

First-letter absorption (ratio-to-null $= 10.0$, 120 events, GO) and null semantic results
(animate-inanimate, noun-proper; ratio-to-null $= 1.0$, NO\_GO) together establish that
feature absorption in GPT-2 Small is specific to orthographic hierarchies at the tested scale.

This null result for semantic hierarchies should not be over-interpreted. The absence of
detectable absorption under animate-inanimate and noun-proper noun hierarchies in GPT-2 Small
is consistent with three distinct explanations: (1) semantic concept hierarchies do not produce
sufficient co-occurrence of parent and child features in this model to trigger the absorption
threshold; (2) the parent feature identification method (top-5 probe-aligned latents) misses the
true parent features in semantically complex hierarchies; (3) semantic absorption exists but the
IG-ablation methodology requires richer context prompts than the fixed ICL format used here.
All three explanations predict that experiments on Gemma Scope SAEs (Gemma 2 2B or larger, where
semantic hierarchies should be more richly represented) could find semantic absorption even if
GPT-2 Small does not exhibit it.

The scope of the present result is therefore: orthographic (first-letter) absorption is
confirmed; semantic absorption is neither confirmed nor definitively ruled out. Gemma Scope
experiments are required before drawing conclusions about the generality of absorption across
hierarchy types.

## 5.6 Connections to Architectural Mitigations

The rate-distortion framework (Proposition 1) provides a unified account of why architectural
mitigations work: any intervention that increases $\sin^2(\theta_{p,c})$ for parent-child pairs
raises the sparsity threshold below which absorption is preferred.

OrtSAE imposes an orthogonality penalty on decoder columns, directly increasing the angle
$\theta_{p,c}$ between any two decoder directions, including parent-child pairs; by Proposition 1,
this raises the absorption onset threshold. Matryoshka SAE assigns parent features to an inner
dictionary and child features to an outer dictionary, so parent and child decoders operate in
structurally different subspaces; the effective $\theta_{p,c}$ for cross-level pairs is larger
by construction. ATM SAE uses per-latent importance weighting, creating non-uniform effective
$\lambda$ across features; child features with reduced effective sparsity penalty fall below the
absorption threshold even when their decoder angle is small.

These connections are theoretical predictions, not verified by our experiments, but the
predictions are falsifiable: each mitigation should be accompanied by an increase in the
mean decoder angle $\theta_{p,c}$ for absorbed pairs, measurable from weights alone. The
EDA framework provides the measurement tool.

## 5.7 Limitations and Open Questions

**Label sparsity.** The exact FeatureAbsorptionCalculator labels contain $n_+ = 18$ absorbed
features out of 24,576 at layer 6. All AUROC comparisons involving exact labels operate in a
regime where small changes in the label set can substantially alter the apparent metric. The
proxy label expansion to $n_+ = 50$ stabilizes results (AUROC $= 0.659$ vs. $0.650$), and the
Jaccard overlap of $0.115$ between exact and proxy sets suggests they measure overlapping but
not identical aspects of absorption. A larger exact label set, obtainable by running
FeatureAbsorptionCalculator across more letters and at lower activation thresholds, would
strengthen statistical power substantially.

**Single model and task.** All results are obtained on GPT-2 Small (117M parameters), which
is at the smallest end of the model scale relevant to mechanistic interpretability work.
Gemma 2 2B has richer semantic features, deeper feature hierarchies, and substantially larger
SAEs; whether the EDA signal, the cross-directional metric, and the AJT polarity reversal
all replicate at this scale remains open. Cross-model generalization is the most important
single extension to this work.

**Amortization gap confound.** EDA measures intra-feature encoder-decoder misalignment.
High EDA could arise from absorption (encoder pulled toward parent direction) or from
amortization gap (encoder fails to specialize to a feature the decoder has committed to).
Distinguishing these requires activation-data experiments comparing encoder firing patterns to
decoder direction projections; the weight-only analysis in this paper cannot separate them.

**Proxy label quality.** Proxy labels (letter features above decoder-probe threshold) have
Jaccard overlap $= 0.115$ with exact Chanin labels. The low overlap means that most proxy-labeled
features are not confirmed absorbed by IG-ablation. The AUROC results using proxy labels should
be interpreted with this in mind: they measure a related but broader property than the exact
absorption labels.

**AJT mechanism unknown.** The three AJT SAEs produce a clear and reproducible polarity
reversal (EDA$_\Delta < 0$, AUROC $< 0.2$), but the mechanism is not established. Whether
the AJT sparsity formulation prevents encoder drift, reverses it, or creates a qualitatively
different geometry for absorbed features is unknown. This is a concrete open question with
practical implications: if AJT training were combined with post-hoc EDA screening, a new
calibrated detector (possibly using inverted EDA) might still work, but this requires
verification.

---

The central results are stable to these limitations: EDA provides a statistically significant
weight-only absorption signal for L1-penalized SAEs at mid-layers ($z_\text{null} = 2.49$,
AUROC $= 0.650$), the cross-directional metric $\cos(\hat{e}_p, d_c)$ is stronger still
(AUROC $= 0.730$), absorption is phase-stable across the tested hyperparameter range
(rates 0.876–0.978, no phase transition, hysteresis not escaped by fine-tuning), and the
encoder-decoder decomposition provides direct geometric evidence consistent with the
mechanistic conjecture.

<!-- FIGURES
- None (Discussion section contains no new figures; references figures from earlier sections)
-->
