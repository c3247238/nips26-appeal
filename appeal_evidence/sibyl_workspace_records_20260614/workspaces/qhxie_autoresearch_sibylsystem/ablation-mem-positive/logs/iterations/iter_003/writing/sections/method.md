# Method

We test whether coefficient of variation (CV) predicts steering effectiveness within absorbed SAE features, and whether absorbed features decompose into steerable and non-steerable subpopulations.

## 3.1 The Variance Paradox

The coefficient of variation ($CV = \sigma / \mu$) measures activation variability across contexts for each SAE latent, where $\sigma$ is the standard deviation and $\mu$ is the mean activation magnitude. When computed across 1,000 text samples for absorbed features, we observe $CV_{absorbed} \approx 7.33$, compared to $CV_{non-absorbed} \approx 0.01$ for non-absorbed features—a 733-fold ratio.

This stark contrast contradicts a simple "absorption degrades signal" narrative. Rather than indicating degraded feature quality, absorption selectively preserves context-sensitive specialized information. Features with high CV activate strongly in specific contexts but weakly in others, suggesting they encode specialized rather than generic information.

## 3.2 Robust vs Fragile Absorbed Features

We propose that absorbed features decompose into two subpopulations based on their coefficient of variation:

**Robust absorbed features** (high-CV, $CV > 1.0$) route through context-sensitive child channels that preserve steering potential. The high variance reflects context-dependent activation—these features are recruited differentially depending on context, allowing steering interventions to modulate their downstream contribution effectively.

**Fragile absorbed features** (low-CV, $CV \le 1.0$) route through stable child channels that compensate for parent steering. The child feature provides a fixed contribution regardless of context, creating a bypass routing mechanism. Steering the parent activates the child, but the child's contribution remains constant, resulting in zero net steering effect.

The threshold $CV = 1.0$ separates these subpopulations empirically, based on pilot experiments showing the 2x steering effect difference between groups.

## 3.3 Causal Mediation Mechanism

When an absorbed parent feature is steered, the activation flows through its child feature(s) before affecting model outputs. For high-CV (robust) features, the child's context-sensitive routing allows the steering modulation to propagate—the child's contribution scales with context, permitting effective intervention. For low-CV (fragile) features, the child's stable routing creates a fixed contribution path that compensates for any parent steering, resulting in zero net effect.

We validate this mechanism via activation patching: zeroing the child feature and measuring parent recovery. If the parent recovers substantially when the child is absent, this confirms the parent-child causal structure. Parent recovery $R_{parent} > 10\%$ indicates genuine absorption with causal structure suitable for steering.

## 3.4 Phase Transition as Theoretical Context

The absorption phenomenon exhibits quasi-critical behavior near the critical sparsity penalty $\lambda_c \approx 5 \times 10^{-5}$. The susceptibility peak $\chi_{max} = 11.19$ and chi ratio $\chi_{ratio} = 1.88$ indicate a gradual (not sharp) transition, consistent with finite-size scaling with exponent $\nu = 3$ ($R^2 = 0.951$). This phase transition framework provides theoretical context for understanding absorption onset but does not constitute the primary novelty of this work.

## 4.1 Experimental Setup

**Model and SAE**: We use GPT-2 Small (117M parameters) with the SAELens pretrained SAE (gpt2-small-res-jb) at layer 6 residual stream, which contains approximately 16,000 latents. This layer was identified as an absorption hotspot in prior analysis.

**Dataset**: We use 1,000 text samples from the GPT-2 validation set, with seed 42 for reproducibility.

**Feature Classification**: Absorbed features are identified using the training-free absorption detector $A_j = \|d_j\|^2 / (d_j^\top e_j)$, with threshold 0.5. Among absorbed features, we compute per-feature CV across contexts and classify into high-CV ($CV > 1.0$) and low-CV ($CV \le 1.0$) groups, selecting the top 30 features from each group by CV score.

## 4.2 Steering Protocol

For each selected feature, we apply steering interventions at three strengths ($\tau = +3, +5, +7$) using the standard addition protocol:

1. Collect activation at layer 6 for a given prompt
2. Add $\tau \cdot d_j$ to the residual stream at the target position
3. Measure logit change $\Delta_{logit}$ at semantically appropriate output tokens

We use five prompts: "The movie was very", "The food was extremely", "The weather today is", "The book was quite", "The experience was". The steering effect is reported as the absolute logit change $|\Delta_{logit}|$ for effect magnitude.

**Statistical Analysis**: We apply one-sided Welch's t-test (α = 0.01) comparing high-CV vs low-CV groups, with Benjamini-Hochberg (BH) correction for multiple comparisons across the three steering strengths.

## 4.3 Activation Patching Validation

To confirm absorbed features have genuine causal structure, we perform activation patching on persistent core words identified in prior analysis. For each word, we:
1. Identify the top absorbing child feature via $A_j$ metric
2. Run the model on contexts containing the target word
3. Zero the child feature activation
4. Measure parent feature recovery: $R_{parent} = (logits_{patched} - logits_{child\_absent}) / (logits_{clean} - logits_{child\_absent})$

A recovery threshold of 10% confirms genuine absorption with causal structure.

## 4.4 Decoder Orthogonality Analysis (H6)

To test whether decoder weight geometry explains the CV-steering correlation, we compute decoder orthogonality as $orthogonality_j = 1 - \bar{c}_j$, where $\bar{c}_j$ is the mean cosine similarity between decoder vector $d_j$ and all other decoder vectors. We select 30 features with high orthogonality and 30 with low orthogonality, then measure steering effectiveness at strength +5.

**Falsification criterion**: Correlation $r > 0.3$ between orthogonality and steering effectiveness.

## 4.5 Cross-Architecture Validation

To test generalization beyond GPT-2, we replicate the steering protocol on Gemma-2-2B using GemmaScope JumpReLU SAE at layer 6. We test whether the CV threshold of 1.0 generalizes or requires model-specific calibration.

<!-- FIGURES
- Figure 3: gen_fig3_cv_comparison.py, fig3_cv_comparison.pdf — CV distribution comparing absorbed vs non-absorbed features with 733x ratio annotation and threshold line at CV = 1.0
- Figure 5: fig5_mechanism_desc.md — Architecture diagram describing robust vs fragile absorption routing mechanism
-->