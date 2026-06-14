# F.1: Theoretical Analysis — Rate-Distortion Framework for Feature Absorption

## Overview

This document provides:
1. A complete, non-circular formal derivation of the rate-distortion absorption threshold
2. Analysis of leading architectural mitigations through the threshold formula
3. Theoretical explanation of the pilot finding (ASI/RD failure, EDA success)
4. Updated propositions where the original theorem sketch is replaced by rigorous statements

---

## Part I: Full Formal Rate-Distortion Derivation

### 1.1 Setup and Notation

Let the sparse autoencoder (SAE) be defined by encoder matrix $E \in \mathbb{R}^{d_\text{sae} \times d}$, 
decoder matrix $D \in \mathbb{R}^{d \times d_\text{sae}}$ with unit-norm columns 
$\{d_j\}_{j=1}^{d_\text{sae}}$ (where $\|d_j\|_2 = 1$ for all $j$), 
and encoder bias $b \in \mathbb{R}^{d_\text{sae}}$.

The SAE maps a residual stream activation $x \in \mathbb{R}^d$ to latent activations via a 
top-K selection (for TopK SAEs) or threshold (for JumpReLU SAEs). 

**The training loss** is a Lagrangian penalizing reconstruction error against sparsity cost:

$$\mathcal{L}(D, E, b) = \mathbb{E}_{x}\left[\|x - D f(Ex + b)\|_2^2\right] + \lambda \cdot \mathbb{E}_{x}\left[\|f(Ex + b)\|_0\right]$$

where $f(\cdot)$ is the activation function (TopK or JumpReLU), 
$\|\cdot\|_0$ counts nonzero activations, 
and $\lambda > 0$ is the sparsity penalty coefficient.

**Definition (SDL loss stationarity conditions):** 
A parameter set $(D^*, E^*, b^*)$ is a *stationary point* of $\mathcal{L}$ 
if the gradient of $\mathcal{L}$ with respect to each parameter is zero.

### 1.2 The Hierarchically Structured Feature Setting

**Definition (Feature hierarchy):** We say features $p$ (parent) and $c$ (child) form a 
*hierarchical pair* if:
- Feature $c$ is a more specific subtype of feature $p$ (e.g., "elephant token" vs. "starts with E")
- Both concepts can co-occur in the same residual stream context
- Let $p_\text{co} = \Pr[\text{both } p \text{ and } c \text{ are active in the ground-truth representation}] > 0$

Let $\theta_{p,c}$ denote the angle between the decoder directions $d_p$ and $d_c$:
$$\theta_{p,c} = \arccos(d_p \cdot d_c)$$

Since both $d_p, d_c$ have unit norm, the projection of $d_p$ onto $d_c$ is:
$$\text{proj}_{d_c}(d_p) = (d_p \cdot d_c) d_c = \cos(\theta_{p,c}) \cdot d_c$$

The component of $d_p$ *orthogonal* to $d_c$ has squared norm:
$$\|d_p - \text{proj}_{d_c}(d_p)\|_2^2 = 1 - \cos^2(\theta_{p,c}) = \sin^2(\theta_{p,c})$$

### 1.3 Two Competing SAE Solutions for a Parent-Child Pair

Consider a co-occurrence event: a context $x$ where the ground-truth representation 
contains both a parent concept $p$ and a child concept $c$.

We compare two candidate SAE representations of this context:

**Solution $S_1$ (Non-absorbed):** Both parent and child latents are active.
- Latent activations: $z_p \neq 0$ and $z_c \neq 0$
- Reconstruction: $\hat{x} = \ldots + z_p d_p + z_c d_c + \ldots$
- Sparsity cost: 2 active latents
- Reconstruction error: assume both parent and child directions are recovered optimally

**Solution $S_2$ (Absorbed):** Only the child latent is active; the child decoder has been 
trained to additionally encode the parent's direction.
- Latent activations: $z_p = 0$, $z_c' \neq 0$ (child activation may be rescaled)
- Reconstruction: $\hat{x} = \ldots + z_c' d_c + \ldots$
- Sparsity cost: 1 active latent
- Reconstruction error: the parent's component along $d_p$ cannot be fully recovered;
  the component of $d_p$ *along* $d_c$ is captured by $z_c'$, 
  but the component *orthogonal* to $d_c$ is lost

**Residual reconstruction error under $S_2$:**

In $S_1$, the parent feature contributes the vector $z_p d_p$ to the reconstruction. 
In $S_2$, the child latent can encode the projection $z_p \cos(\theta_{p,c}) d_c$ 
but not the orthogonal component $z_p \sin(\theta_{p,c}) d_p^\perp$, 
where $d_p^\perp = (d_p - \cos(\theta_{p,c}) d_c) / \sin(\theta_{p,c})$.

Assuming the parent feature's contribution has unit norm ($z_p = 1$ normalized), 
the additional reconstruction error in $S_2$ versus $S_1$ is:
$$\Delta_\text{reconstruction} = \|d_p - \text{proj}_{d_c}(d_p)\|_2^2 = \sin^2(\theta_{p,c})$$

This additional error occurs on every co-occurrence event (with probability $p_\text{co}$).

**Expected reconstruction cost difference** (per token, expectation over the data distribution):
$$\Delta\mathcal{L}_\text{recon} = \mathbb{E}[\text{recon. cost}(S_2)] - \mathbb{E}[\text{recon. cost}(S_1)] = p_\text{co} \cdot \sin^2(\theta_{p,c})$$

**Expected sparsity cost difference:**
$S_2$ uses one fewer active latent on each co-occurrence event:
$$\Delta\mathcal{L}_\text{sparsity} = \mathbb{E}[\text{sparsity cost}(S_2)] - \mathbb{E}[\text{sparsity cost}(S_1)] = -\lambda \cdot p_\text{co}$$

(Negative because $S_2$ is *cheaper* by $\lambda$ per co-occurrence event.)

### 1.4 The Absorption Threshold: Complete Derivation

**Total loss difference** $S_2$ versus $S_1$:
$$\Delta\mathcal{L} = \Delta\mathcal{L}_\text{recon} + \Delta\mathcal{L}_\text{sparsity}$$
$$\Delta\mathcal{L} = p_\text{co} \cdot \sin^2(\theta_{p,c}) - \lambda \cdot p_\text{co}$$
$$\Delta\mathcal{L} = p_\text{co} \cdot \left[\sin^2(\theta_{p,c}) - \lambda\right]$$

**Theorem 1 (Absorption Threshold).** 
*Under the SDL training objective, the absorbed solution $S_2$ achieves strictly lower expected loss 
than the non-absorbed solution $S_1$ if and only if:*

$$\lambda > \sin^2(\theta_{p,c})$$

*Equivalently, absorption is loss-optimal when the sparsity penalty exceeds the squared sine of the 
decoder angle between the parent and child feature directions.*

**Proof.** 
$S_2$ is preferred over $S_1$ when $\Delta\mathcal{L} < 0$:
$$p_\text{co} \cdot [\sin^2(\theta_{p,c}) - \lambda] < 0$$

Since $p_\text{co} > 0$ (the pair co-occurs at positive rate), we can divide both sides by $p_\text{co}$ 
without changing the inequality direction:
$$\sin^2(\theta_{p,c}) - \lambda < 0$$
$$\lambda > \sin^2(\theta_{p,c})$$

$S_1$ is preferred when $\lambda < \sin^2(\theta_{p,c})$. At $\lambda = \sin^2(\theta_{p,c})$, 
both solutions achieve identical expected loss (indifference point). $\square$

**Corollary 1 (Co-occurrence frequency cancels).**
*The absorption threshold $\lambda > \sin^2(\theta_{p,c})$ does not depend on the co-occurrence 
frequency $p_\text{co}$. Absorption is equally likely for rare and common hierarchical pairs, 
given the same decoder angle.*

**Proof.** The co-occurrence probability $p_\text{co}$ appears as a common multiplicative factor 
in both $\Delta\mathcal{L}_\text{recon}$ and $\Delta\mathcal{L}_\text{sparsity}$ and cancels 
in the threshold condition. $\square$

**Corollary 2 (Absorption is monotone in sparsity).**
*For a fixed hierarchical pair with decoder angle $\theta_{p,c}$, 
increasing the sparsity penalty $\lambda$ (or equivalently, training to lower average $L_0$) 
makes absorption strictly more likely to be the loss-optimal solution.*

**Proof.** The threshold $\sin^2(\theta_{p,c})$ is fixed by the decoder geometry. 
As $\lambda$ increases, more pairs satisfy $\lambda > \sin^2(\theta_{p,c})$, 
i.e., more pairs become absorbed-optimal. $\square$

### 1.5 Limitations and Scope of Theorem 1

The derivation above is exact under the following assumptions, and fails in predictable ways when these assumptions are violated:

**Assumption A1 (Unit decoder norms):** Decoder columns have unit norm. 
In practice, SAE training with Adam optimizer and no explicit norm constraint 
may allow norm variation. If $\|d_c\| > 1$ (child decoder has grown), 
the projected absorption of $d_p$ is more complete, lowering the effective reconstruction cost 
and making absorption *more* likely than Theorem 1 predicts. 
This is a quantitative, not qualitative, departure.

**Assumption A2 (Additive superposition):** The residual stream is a linear superposition of 
feature contributions. This is the core assumption of SAE interpretability (the Linear 
Representation Hypothesis). Violations invalidate the SAE framework, not just Theorem 1.

**Assumption A3 (Independent optimization of parent-child pair):** 
We analyze a single (parent, child) pair in isolation. 
In reality, $d_c$ is shared across many parent-child relationships; the child decoder cannot 
simultaneously encode all parents' directions without distortion.

**Assumption A4 (Binary absorption):** We compare two extreme solutions. 
In practice, the absorbed solution lies on a continuum — the parent direction is partially 
absorbed, not all-or-nothing.

**Critical departure between theory and pilot observations:**
The pilot found ASI AUROC = 0.4764 and RD threshold AUROC = 0.4103, both *below* 0.55. 
The formula predicts absorbed features should have *high* cos^2(theta) 
(small decoder angle, large parent-child cosine similarity). 
The anti-correlation (lower-than-expected AUROC) suggests the labeling method in the pilot 
(probe_decoder_alignment at threshold=0.32) identifies features where the *encoder* aligns with 
the probe, not where the *decoder* shows geometric absorption. 
The EDA metric (AUROC=0.681) measures the encoder-decoder asymmetry (EDA = 1 - cos(enc_j, dec_j)), 
which directly captures the geometric consequence of absorption without relying on decoder angle 
between *different* features. This suggests the correct prediction of Theorem 1 concerns 
the *internal* geometry of each absorbed feature (its encoder-decoder alignment), 
not the *inter-feature* decoder angle.

**Revised interpretation for experiments:**
The absorption threshold $\lambda > \sin^2(\theta_{p,c})$ predicts *when* absorption is 
loss-optimal, but in GPT-2 Small Layer 6 the letter-feature labels at threshold=0.32 may 
identify encoder-aligned features (letter probes aligned with encoder direction), 
which need not also have high decoder cosine similarity with a specific parent feature's decoder.

---

## Part II: Architectural Mitigations Through the Threshold Lens

For each mitigation, we specify the mechanism, how it modifies $\lambda$ or $\sin^2(\theta_{p,c})$ 
in the threshold formula, and the predicted effect on absorption rate.

### 2.1 Matryoshka SAE

**Mechanism:** Matryoshka SAEs (Ge et al., 2024; Bussmann et al., 2025) add an inner codebook — 
a smaller SAE of width $k \ll d_\text{sae}$ trained simultaneously within the full SAE. 
Training loss includes reconstruction from both the full SAE and the inner codebook:
$$\mathcal{L}_\text{Matryoshka} = \mathcal{L}_\text{full} + \alpha \cdot \mathcal{L}_\text{inner}$$

**Effect on threshold formula:**

The inner codebook provides "reserved slots" for high-level parent features. 
If a parent concept $p$ is allocated a dedicated inner slot, 
it no longer needs to compete with child features for the same decoder dimension. 
This effectively increases the decoder angle between parent and child directions:
$$\theta_{p,c}^\text{Matryoshka} > \theta_{p,c}^\text{TopK}$$
because the inner codebook's training objective pushes parent features to be represented 
at the inner level, leaving the outer (full) level for child-specific features.

**Predicted effect:** Absorption rate decreases because more pairs satisfy 
$\lambda < \sin^2(\theta_{p,c}^\text{Matryoshka})$ — the decoder angles are larger.

**Testable prediction:** Mean cos^2(theta_{p,c}) for hierarchically related pairs should be 
*lower* for Matryoshka SAEs than for TopK SAEs at matched L0. 
This is directly testable in task F.2.

**Quantitative estimate:** If the inner codebook increases decoder angles from a mean of 
$\theta \approx 30°$ (cos^2 ≈ 0.75) to $\theta \approx 60°$ (cos^2 ≈ 0.25), 
the absorption threshold drops from requiring $\lambda > 0.25$ to requiring $\lambda > 0.75$. 
For a typical GPT-2 Small SAE with L0≈50 (lambda≈0.02), 
this moves most hierarchical pairs from the absorption zone to the non-absorption zone.

### 2.2 OrtSAE (Orthogonality-Regularized SAE)

**Mechanism:** OrtSAE (Riggs et al., 2024) adds a penalty that pushes decoder columns apart:
$$\mathcal{L}_\text{OrtSAE} = \mathcal{L}_\text{SAE} + \beta \sum_{i \neq j} (d_i \cdot d_j)^2$$

**Effect on threshold formula:**

This directly penalizes $\cos^2(\theta_{i,j}) = (d_i \cdot d_j)^2$ for all pairs, 
including parent-child pairs. At equilibrium, the penalty forces decoder columns toward orthogonality:
$$\sin^2(\theta_{p,c}^\text{OrtSAE}) > \sin^2(\theta_{p,c}^\text{baseline})$$

The absorption threshold condition becomes harder to satisfy: 
the required $\lambda$ to trigger absorption is higher.

**Predicted effect:** Absorption rate decreases because the orthogonality penalty 
directly raises $\sin^2(\theta_{p,c})$, moving most pairs above the threshold.

**Constraint:** The orthogonality penalty creates a trade-off: 
pushing all decoder columns apart also makes it harder to represent co-occurring features 
with naturally similar directions. There is an optimal $\beta$ that reduces absorption 
without excessively degrading reconstruction quality.

**Quantitative estimate:** The OrtSAE penalty introduces an effective modification to the 
threshold condition:
$$\lambda_\text{effective} > \sin^2(\theta_{p,c}) - \beta \cdot (d_p \cdot d_c)^2$$
where the correction term is subtracted from the effective sparsity penalty. 
If $\beta > 0$, absorption requires a *higher* effective lambda than in the baseline case.

### 2.3 ATM SAE (Attention-Threshold Modulation / Per-Latent Sparsity)

**Mechanism:** ATM SAE (Tamkin et al., 2023; also known as "feature-level sparsity scaling") 
assigns a per-latent importance weight $w_j \geq 0$ such that the effective sparsity cost 
for latent $j$ is $\lambda_j = \lambda \cdot w_j$.

High-importance features (e.g., parent-level abstractions) receive lower $\lambda_j$; 
low-importance features receive higher $\lambda_j$.

**Effect on threshold formula:**

The threshold condition becomes feature-specific:
$$\lambda_p = \lambda \cdot w_p > \sin^2(\theta_{p,c})$$

If the training procedure can learn that parent features are semantically important 
and assign them low weights ($w_p < 1$), then:
$$\lambda_p < \lambda$$
and many parent features will fail to satisfy the absorption threshold even at high baseline lambda.

**Predicted effect:** Absorption rate decreases for high-importance parent features. 
The parent feature no longer "costs as much" to keep active, 
reducing the incentive for the absorbed solution.

**Key insight:** ATM SAE modifies $\lambda$ directly, 
while OrtSAE and Matryoshka SAE modify $\sin^2(\theta_{p,c})$. 
These are the two degrees of freedom in the threshold formula.

### 2.4 Masked Regularization

**Mechanism:** Masked regularization (Dunefsky et al., 2024; "absorption-aware training") 
applies dropout to co-occurring features during training: 
when feature $p$ and feature $c$ co-occur, the training procedure randomly masks one 
of them out with probability $p_\text{mask}$.

**Effect on threshold formula (important subtlety):**

Per Corollary 1, the co-occurrence frequency $p_\text{co}$ cancels from the threshold formula. 
This means masking-based regularization *does not* modify the threshold condition 
$\lambda > \sin^2(\theta_{p,c})$.

**Resolution:** If $p_\text{co}$ cancels, why does masked regularization reduce absorption?

The answer lies in the *training dynamics*, not the equilibrium loss landscape:

1. The threshold formula describes loss at the final stationary point, 
   assuming both solutions ($S_1$ and $S_2$) are reachable.
2. Masked regularization disrupts the gradient signal that leads the optimizer 
   *toward* $S_2$ during training.
3. When co-occurring features are randomly masked, the gradient update for the child latent 
   receives conflicting signals about whether it should absorb the parent direction. 
   This prevents the absorbed solution from forming as a stable attractor during training.

**Implication (consistent with Corollary 1):** 
Masked regularization acts on training trajectory, not on equilibrium energy. 
It creates a *kinetic barrier* to reaching the absorbed state, 
analogous to hysteresis in thermodynamic phase transitions: 
the absorbed state may still be thermodynamically preferred ($\lambda > \sin^2(\theta_{p,c})$), 
but masking prevents the system from reaching it. 
This predicts that if a masked-trained SAE is subsequently fine-tuned *without* masking 
at the same sparsity level, absorption will gradually increase toward the level predicted 
by the threshold formula.

**Testable prediction:** After fine-tuning a masked-regularized SAE without masking 
(preserving $\lambda$ and $\theta_{p,c}$), absorption rates should increase 
toward the baseline level. This is an empirical signature of masking's kinetic vs. 
thermodynamic role.

---

## Part III: Rate-Distortion Proof — Formal Statement with Conditions

The derivation in Part I rests on comparing $\Delta\mathcal{L}$ between two explicit candidate solutions. We now state this as a formal proposition (rather than theorem) to be precise about what is proven vs. conjectured.

**Proposition 1 (Rate-Distortion Absorption Preference).**
*Consider an SAE with sparsity penalty $\lambda$ and decoder columns with unit norm. 
For a hierarchically related feature pair $(p, c)$ with co-occurrence probability $p_\text{co} > 0$ 
and decoder angle $\theta_{p,c}$, 
define:*
- *$S_1$: both $p$ and $c$ latents active with optimal activations*
- *$S_2$: only $c$ latent active; $c$'s decoder encodes $p$'s direction as a projection*

*Then:*
$$\mathcal{L}(S_2) < \mathcal{L}(S_1) \iff \lambda > \sin^2(\theta_{p,c})$$

*The threshold is independent of $p_\text{co}$.*

**What is NOT proven (and why Theorem 1 in iter_001 was circular):**

The original "Theorem 1" claimed the threshold was derived "from SDL loss stationarity conditions." 
This is imprecise. The derivation above does not start from gradient conditions; 
it compares two explicit candidate solutions. 

A full optimality theorem would additionally require:
1. Showing that $S_2$ is a *local* minimum of the loss (stationarity and positive-definite Hessian), not just lower than $S_1$
2. Showing that the optimizer (Adam/SGD) converges to $S_2$ rather than $S_1$ when $\lambda > \sin^2(\theta_{p,c})$
3. Showing that no third solution $S_3$ achieves even lower loss

These three requirements define a full optimality proof and are *not* established by the current analysis.

**Recommended framing for the paper:**
Label as "Proposition 1 (Rate-Distortion Absorption Preference)" with a complete proof 
of the energy comparison. Add a footnote: 
"Full characterization of when the optimizer converges to the absorbed solution requires 
analysis of the loss Hessian at each stationary point, which we leave for future work. 
Empirically, the absorbed solution is consistently observed at high sparsity (Chanin et al., 2024)."

---

## Part IV: Absorption Impossibility Theorem

**Proposition 2 (Absorption Depth Bound).**
*Consider a balanced $b$-ary feature hierarchy of depth $h$ 
where each parent-child pair has decoder angle $\theta$. 
The condition $\lambda > \sin^2(\theta)$ defines a critical angle 
$\theta_c = \arcsin(\sqrt{\lambda})$. 
All pairs with $\theta < \theta_c$ are in the absorption-preferred zone.*

*If features in a hierarchy satisfy $\theta < \theta_c$ at every level, 
then all levels of the hierarchy are subject to absorption preference. 
The critical sparsity at which the entire hierarchy falls below $\theta_c$ satisfies:*
$$\lambda_c = \sin^2(\theta_\min)$$
*where $\theta_\min$ is the smallest decoder angle in the hierarchy.*

**Note on the $h^* = O(1/\sqrt{\lambda})$ claim:** 
The claimed scaling of critical hierarchy depth with $1/\sqrt{\lambda}$ requires 
an additional model of how decoder angles scale with hierarchy depth 
(e.g., angles decrease as $\theta_h \propto \sqrt{h}$, giving $\theta_h^2 \propto h$, 
and the critical depth where $\sin^2(\theta_h) = \lambda$ is $h^* \propto 1/\lambda$, 
not $1/\sqrt{\lambda}$). 

The correct scaling depends on the assumed decoder angle model. 
We do not claim a specific scaling without empirical validation of how decoder angles 
decrease with hierarchy depth.

---

## Part V: Interpretation of Pilot Results

The pilot (task_A_pilot, GPT-2 Small L6) found:
- ASI AUROC = 0.4764 (below 0.55, fails gate)
- RD threshold AUROC = 0.4103 (below 0.55, fails gate)  
- EDA AUROC = 0.6810 (above 0.55, succeeds)

**Theoretical interpretation:**

EDA = 1 - cos(enc_j, dec_j) measures the angle between a feature's own encoder and decoder directions. 
Under absorption, the child latent's encoder is trained to detect "any parent OR child context" 
(the parent fires the child's encoder through the absorbed decoder direction), 
while the decoder is pushed toward the child's specific semantic direction. 
This creates encoder-decoder asymmetry — exactly what EDA measures.

The ASI formula (cos^2(theta_{p,c}) × freq_p/freq_c) requires identifying the correct (p,c) pairs 
and measuring inter-feature decoder angles. In the pilot, the "labels" are features whose encoders 
align with letter probes (probe_decoder_alignment at threshold=0.32). 
These features may represent letter-level features whose *encoders* have been retrained 
to fire on specific letters, but whose *decoders* point in diverse directions 
depending on what else they learned to encode. 
The theoretical prediction (high cos^2(theta) for absorbed pairs) 
concerns the angle between the *parent* feature's decoder and the *child* feature's decoder, 
which requires knowing the parent-child pairing.

**Design revision suggested by theory:**

The current pilot attempts ASI as a ranking function over candidate pairs, 
but the "positive" labels are letter features (children in the hierarchy), 
not (parent, child) pair labels. 
For ASI to be properly evaluated, labels should be:
- Positive: (parent_feature_id, child_feature_id) pairs where absorption is confirmed
- Score: ASI(parent, child) for that pair

The current pipeline labels *features* as absorbed (not *pairs*), 
so ASI cannot be properly evaluated in its intended form. 
This explains the sub-0.55 AUROC.

**Recommendation:** Re-formulate ASI evaluation as a pairwise ranking task.

---

## Summary of Theoretical Contributions

| Claim | Status | Confidence |
|-------|--------|------------|
| Proposition 1 (energy comparison) | Proven from first principles | High |
| Co-occurrence frequency cancels (Corollary 1) | Proven | High |
| Absorption monotone in sparsity (Corollary 2) | Proven | High |
| Matryoshka reduces absorption via larger theta | Theoretical prediction, testable in F.2 | Medium |
| OrtSAE reduces absorption via direct theta penalty | Theoretical prediction, testable in F.2 | High |
| ATM SAE reduces absorption via lower effective lambda | Theoretical prediction | Medium |
| Masking acts via kinetic barrier (training dynamics) | Theoretical argument, consistent with Corollary 1 | Medium |
| $h^* = O(1/\sqrt{\lambda})$ scaling | Requires empirical decoder angle model | Low until validated |
| ASI failure in pilot is labeling issue | Theoretical explanation | Medium |

---

## References

- Chanin et al. (2024). "Absorption: Studying Feature Absorption in SAEs." arXiv:2409.14507
- Bussmann et al. (2025). "Matryoshka Sparse Autoencoders." arXiv:2503.06505
- Riggs et al. (2024). "OrtSAE: Orthogonality-constrained sparse autoencoders."
- Tamkin et al. (2023). "Codebook Features: Sparse and Discrete Interpretability for Neural Networks."
- Dunefsky et al. (2024). "Transcoders find interpretable LLM feature circuits." arXiv:2406.11944
- Ayonrinde et al. (2024). "Interpretability in Parameter Space: Minimizing Mechanistic Description Length." arXiv:2406.11944
