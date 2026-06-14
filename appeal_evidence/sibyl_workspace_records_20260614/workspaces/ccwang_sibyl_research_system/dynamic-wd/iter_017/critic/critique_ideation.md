# Ideation Critique -- Iteration 017

## Core Idea Assessment

EqWD is a clean, well-motivated idea: use the gradient-to-weight ratio's deviation from its EMA equilibrium to modulate weight decay per-layer. The theoretical grounding in Defazio's equilibrium result is sound, and the connection to the alignment-based framework of Sun et al. is a natural extension.

## Strengths

1. **The idea is simple and implementable.** Two norms per layer per step, one EMA, one deviation calculation. This is the kind of method that practitioners will actually adopt if results are convincing.

2. **The theoretical motivation is principled.** Building on Defazio's proven convergence result (r_t -> lambda/gamma) and using deviations from this equilibrium as a signal for when modulation matters most is a well-motivated design choice.

3. **The scope reduction from "unified framework" to "focused method" was correct.** The previous iterations' attempt to unify all dynamic WD methods into a PID framework was overambitious and empirically unsupported. EqWD as a standalone method is a better contribution.

4. **CAWD as a negative ablation is well-conceived.** Isolating the signal choice (ratio deviation vs. cosine alignment) using the same framework is a textbook ablation design.

## Concerns

### Major

1. **The phi_l(t) >= 1 design choice is questionable.** EqWD only INCREASES weight decay relative to baseline. It never reduces WD when the optimization is near equilibrium (it recovers baseline WD). This means EqWD cannot correct for cases where lambda_base is too high. A symmetric variant (phi_l(t) = 1 +/- beta * dev) that could also reduce WD during stable phases might be more principled. The current design introduces a systematic upward bias in effective WD that confounds the "adaptive modulation" claim with "more regularization."

2. **The EMA equilibrium is inherently lagged.** With alpha=0.9, the EMA has a time constant of 10 steps. When the learning rate changes (e.g., step decay at epoch 30), the EMA takes ~30 steps to converge to the new quasi-static equilibrium. During this transition, the deviation is artificially large because the EMA is tracking the OLD equilibrium. This means EqWD's modulation during LR transitions is not responding to genuine instability but rather to the EMA's inability to track the new equilibrium quickly enough. The paper presents this as a feature ("Schedule-aligned modulation, Section 4.4") but it is arguably an artifact.

3. **Why deviation from EMA and not deviation from lambda/gamma?** The paper argues that the theoretical r* = lambda/gamma doesn't hold in practice due to BN, augmentation, etc. But if these factors shift the equilibrium, they also shift the EMA equilibrium. The EMA tracks a time-averaged r_t, not the true equilibrium. In smooth training regimes (like CIFAR-100), the EMA closely tracks r_t, making the deviation nearly zero regardless of whether the optimization is actually in equilibrium. This explains why EqWD underperforms on CIFAR-100 with default beta.

### Minor

4. **Scale invariance claim needs qualification.** The paper claims the normalized deviation ensures "scale invariance across layers." This is scale invariance in the ratio space (relative deviations are comparable), but the EFFECT of the modulation (multiplying lambda_base) has the same absolute impact regardless of the layer's ratio scale. A more precise claim would be that the modulation SIGNAL is scale-invariant, but the modulation EFFECT is not.

5. **The method adds complexity for modest gains.** +0.38% on ImageNet over FixedWD is within reasonable tuning range. If budget-matched FixedWD achieves similar accuracy, the practical motivation for EqWD becomes primarily the convenience of not needing to tune lambda -- a valid but less exciting contribution.

## Assessment of Alternatives

The paper wisely does not overclaim. The idea would be stronger if:
- The budget-matched experiment showed adaptive modulation genuinely helps beyond regularization strength
- A symmetric variant (allowing WD reduction) was tested
- Results extended to AdamW + Transformers
