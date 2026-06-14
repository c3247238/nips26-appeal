# Method Section Critique: Denoising-Time Adaptation (DTA)

**Section**: 3. Method (Sections 3.1–3.4)
**Score**: 7/10

---

## Overall Assessment

The method section is well-structured and technically detailed, presenting DTA with clear algorithmic exposition, a variational interpretation, a systematic ablation framework (information augmentation spectrum), and a discussion of orthogonality with remasking. The E-step/M-step formulation is intuitive and the design decisions are well-motivated by pilot experiments. However, several issues weaken the section's rigor, clarity, and completeness.

---

## Issues

### Critical

**C1. Proposition 2 (Information Accumulation) lacks a proof or formal argument.**
- Location: Section 3.2, lines 110–115
- Severity: **Critical**
- The proposition claims monotonic increase of mutual information $I(\Delta\theta^{(t+1)}; \mathbf{x}_0) \geq I(\Delta\theta^{(t)}; \mathbf{x}_0)$, but provides only an "Intuition" paragraph rather than a proof sketch. The intuition ("each update integrates information from a strictly larger context") is hand-wavy. Mutual information monotonicity does not follow trivially from the fact that more tokens are revealed — the gradient update could in principle overwrite previously accumulated information, especially with the decay factor $\gamma < 1$. The decay factor explicitly *down-weights* older information, which could violate monotonicity.
- **Suggestion**: Either (a) provide a formal proof under clearly stated assumptions (e.g., assuming the masked LM loss is a sufficient statistic for the revealed tokens), or (b) downgrade the statement to a conjecture/empirical observation supported by the confidence-increase trajectory (0.969 → 0.995). Currently, calling it a "Proposition" overstates its rigor.

**C2. The ELBO definition in Proposition 1 conflates point estimates with distributions.**
- Location: Section 3.2, lines 104–106
- Severity: **Critical**
- The ELBO uses $q(\Delta\theta) = \delta(\Delta\theta^{(t)})$ (a point mass), making the KL divergence $D_{\text{KL}}(q(\Delta\theta) \| p(\Delta\theta))$ either infinite (if $p$ is continuous) or ill-defined. A Dirac delta has infinite KL divergence with respect to any absolutely continuous distribution. The weight decay regularization gives a *penalty* term $\frac{\lambda}{2}\|\Delta\theta\|^2$ that is finite, but calling it a KL divergence requires either a Laplace approximation argument or a different variational family.
- **Suggestion**: Either (a) explicitly define the ELBO using the L2 penalty directly (dropping the KL notation), or (b) use a Gaussian variational family $q(\Delta\theta) = \mathcal{N}(\Delta\theta^{(t)}, \sigma^2 I)$ and take the zero-variance limit with appropriate justification. This is a standard issue in variational point-estimate methods; it should be addressed.

### Major

**M1. Algorithm 1 uses inconsistent update rule vs. the text.**
- Location: Algorithm 1 line 11 vs. Section 3.1 equation for $\Delta\theta$ update
- Severity: **Major**
- Algorithm 1 line 11 writes: `Δθ ← γ · Δθ − η · AdamW_step(∇L)`. But the text (line 22) writes: $\Delta\theta \leftarrow \gamma \cdot \Delta\theta - \eta \cdot \nabla_{\Delta\theta} \mathcal{L}_{\text{DTA}}$, which is plain gradient descent with decay, not AdamW. AdamW has its own adaptive learning rate and weight decay mechanism. The interplay between the cumulative decay $\gamma$ and AdamW's internal weight decay ($\lambda = 0.01$) is not discussed — they serve overlapping purposes and could interact adversely.
- **Suggestion**: (a) Clarify whether the formal equation or the algorithm is authoritative. (b) Discuss the interaction between $\gamma$-decay and AdamW weight decay. (c) Consider whether both are necessary — perhaps one subsumes the other.

**M2. SCP compute overhead claim is inconsistent.**
- Location: Section 3.3, line 159 vs. Spectrum Summary Table (line 177)
- Severity: **Major**
- The text says SCP costs "one forward pass per revealed token per probing step" and quotes $\sim$7$\times$. However, the outline (Section 3.3 in outline) initially estimated $\sim$2$\times$ ("1 extra forward pass/step"). Leave-one-out probing for all revealed positions requires $|\mathcal{R}_t|$ additional forward passes per step, which could be much more than 7$\times$ at later denoising steps when most tokens are revealed. The $\sim$7$\times$ figure appears to be an empirical measurement, but the mechanism description suggests it should be much higher. Either the batching strategy makes this efficient, or the probing is done on a subset — this needs clarification.
- **Suggestion**: Explain how 7$\times$ is achieved. Is probing done on all revealed positions or a sample? Is it batched? Provide the calculation.

**M3. The mask-and-predict loss description is incomplete regarding gradient flow.**
- Location: Section 3.1, lines 17–23
- Severity: **Major**
- The M-step masks positions from $\mathcal{R}_{t-1}$ (revealed tokens) and predicts them. But $x_i^*$ — the target — was itself generated by the model in a previous E-step. This creates a potential issue: the model is being trained to reproduce its own previous predictions, which is a form of self-distillation rather than ground-truth supervision. The section should explicitly acknowledge that the "ground truth" for the M-step is the model's own E-step output, and discuss why this is still a useful learning signal (e.g., it forces the model to reconstruct tokens from bidirectional context, which is different from the original prediction that used unidirectional-at-that-step context).
- **Suggestion**: Add a sentence clarifying that $x_i^*$ is not externally supervised but rather the model's own committed prediction, and explain why this self-supervised signal is still informative (the mask-and-predict requires integrating full bidirectional context rather than the partial context available when the token was first committed).

**M4. Missing discussion of prompt tokens in the M-step.**
- Location: Section 3.1
- Severity: **Major**
- The description focuses on the generation area (fully masked initially), but says nothing about how prompt tokens are handled in the M-step. Are prompt tokens included in $\mathcal{R}_{t-1}$ and potentially masked? If so, this gives the M-step access to ground-truth tokens (from the prompt), which is a qualitatively different signal than re-predicting generated tokens. If not, this should be stated. This distinction matters for the theoretical interpretation.
- **Suggestion**: Explicitly state whether prompt tokens are included in the M-step's maskable set. If yes, discuss the implications.

### Minor

**m1. The term "origin sampling" is used without definition.**
- Location: Section 3.1, line 15
- Severity: **Minor**
- The text says "the `origin` sampling algorithm (Dream; Gong et al., 2025)" but does not define what `origin` means. While a citation is provided, the reader should have at least a one-sentence description (e.g., "the default sampling strategy of Dream that reveals tokens by confidence ranking without remasking").
- **Suggestion**: Add a brief parenthetical definition.

**m2. LoRA alpha ($\alpha$) is introduced without explanation.**
- Location: Section 3.1, line 62
- Severity: **Minor**
- The equation $h' = h + \frac{\alpha}{r} BA h$ introduces the LoRA scaling factor $\alpha$, but its value is never specified. Standard LoRA practice sets $\alpha = r$ (so the scaling is 1), but other choices are common. Since the paper uses $r = 4$, the reader needs to know whether $\alpha = 4$ or some other value.
- **Suggestion**: State the value of $\alpha$ (e.g., $\alpha = r = 4$).

**m3. "Approximately doubling the per-step cost" vs. "$\sim$4$\times$ wall-clock overhead" is confusing.**
- Location: Section 3.1, line 75
- Severity: **Minor**
- The text says "each DTA update requires one additional backward pass... approximately doubling the per-step cost" but then states "$\sim$4$\times$ wall-clock overhead compared to vanilla denoising (15.9s vs. 3.7s)." If each step costs $\sim$2$\times$, the total should also be $\sim$2$\times$ (not $\sim$4$\times$). The discrepancy likely comes from optimizer state updates, memory overhead, or gradient clipping costs, but this is not explained.
- **Suggestion**: Reconcile the two claims. Explain what accounts for the additional 2$\times$ beyond the backward pass.

**m4. Section 3.3 Level 2 (SCP) description has a logical gap.**
- Location: Section 3.3, lines 153–157
- Severity: **Minor**
- SCP remasks contradictory positions "allowing the next denoising step to re-predict them." But this means SCP modifies the token sequence — it is a token-space intervention, like remasking. The text in Section 3.4 implies that only DTA operates in parameter space while remasking operates in token space. SCP's relationship to the remasking category should be clarified to avoid confusion.
- **Suggestion**: Add a sentence noting that SCP is indeed a token-space method (like remasking) but with a more targeted detection mechanism.

**m5. Notation inconsistency: $\Delta\theta^{(t)}$ superscript direction.**
- Location: Section 3.2, lines 89–90 vs. lines 112–113
- Severity: **Minor**
- In the EM setup (line 90), the M-step updates $\Delta\theta^{(t)} \to \Delta\theta^{(t+1)}$, using ascending superscripts as $t$ decreases (denoising goes from $T$ to 1). But Proposition 2 (line 113) also uses $I(\Delta\theta^{(t+1)}; \mathbf{x}_0) \geq I(\Delta\theta^{(t)}; \mathbf{x}_0)$. This notation overloads $t$ — in the algorithm, $t$ counts down (from $T$ to 1), but the superscript counts up. This should be clarified.
- **Suggestion**: Use a separate index for the update count (e.g., $k = T - t$) or explicitly state that the superscript on $\Delta\theta$ counts updates, not denoising steps.

**m6. "Contrast with Autoregressive TTT" subsection placement.**
- Location: Section 3.2, lines 117–127
- Severity: **Minor**
- This subsection is placed within the variational interpretation section (3.2), but it discusses a conceptual contrast that is orthogonal to the variational framework. It would fit more naturally as a separate subsection (3.2.1) or as part of related work (Section 2.3 already discusses AR TTT). Its current placement disrupts the flow of the variational argument.
- **Suggestion**: Consider moving to the end of Section 3.1 (after design decisions) or keeping it but adding a transition sentence explaining why the contrast is relevant to the variational interpretation.

---

## Visual Element Review

**Figure 1 (DTA Algorithm Overview)**: Referenced in the HTML comment at lines 195–197 but NOT referenced in the running text. The outline mandates this figure for Section 3.1. A description file exists (`fig1_dta_overview_desc.md`) with a detailed specification, but the actual TikZ/diagram has not been generated.
- **Issue**: The figure must be referenced in the text (e.g., after Algorithm 1, "see Figure 1 for a visual overview") and actually generated.

**Figure 2 (Information Augmentation Spectrum)**: Also referenced only in the HTML comment (lines 195–198), not in the running text. A description file exists (`fig2_info_spectrum_desc.md`) with a detailed specification but no generated diagram.
- **Issue**: Should be referenced in Section 3.3, ideally after the spectrum summary table.

**Spectrum Summary Table (line 172–178)**: Present and effective. Provides a clear at-a-glance comparison of the four levels. However, the "Compute" column uses "$\sim$1$\times$" for DMI, which is inconsistent with the text stating "$\sim$1.05$\times$" (line 148). Minor but worth reconciling.

**Algorithm 1 (lines 29–53)**: Well-formatted pseudocode. Clear and readable. However, it uses a code block rather than a formal algorithm environment (algorithm2e or algorithmicx), which may look out of place in the final paper. This is a formatting concern for LaTeX conversion.

**Missing visual**: The method section would benefit from a small diagram or inset showing the LoRA injection points within the Transformer architecture (which layers, which projection matrices). The Figure 1 description includes this as an inset, but it is not yet rendered.

---

## Strengths

1. The E-step/M-step formulation is clean and intuitive, making the algorithm easy to understand.
2. Design decisions are well-motivated by pilot experiments with specific numbers (e.g., warmup preventing noisy early gradients, self-consistency loss producing near-zero gradients).
3. The information augmentation spectrum (Section 3.3) is a strong conceptual contribution — it transforms what could be a single-method paper into a systematic study of cross-step information transfer.
4. The combination with remasking (Section 3.4) is clearly argued as orthogonal and complementary.
5. Computational overhead is honestly reported with specific wall-clock measurements.

---

## Score Justification: 7/10

The method section is solid in exposition and algorithmic clarity, with a creative spectrum ablation framework. The main weaknesses are: (a) the variational interpretation has formal gaps (KL divergence with point mass, unproven Proposition 2) that undermine its theoretical contribution; (b) several inconsistencies between the formal equations and Algorithm 1 (gradient descent vs. AdamW); (c) two planned figures are not yet rendered or referenced in the text. Addressing the two critical issues (C1, C2) and the major issues (M1, M3, M4) would raise the score to 8–9.
