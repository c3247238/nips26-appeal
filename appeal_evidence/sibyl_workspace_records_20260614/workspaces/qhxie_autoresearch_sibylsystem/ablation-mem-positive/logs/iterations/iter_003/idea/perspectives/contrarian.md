# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Feature absorption is a solvable problem worth solving**
   - Evidence: Cui et al. (ICLR 2026) prove mathematically that standard SAEs *cannot* fully recover ground-truth monosemantic features due to intrinsic representational interference. Full disentanglement is mathematically impossible under realistic sparsity.
   - Resource: [arXiv:2506.15963](https://arxiv.org/abs/2506.15963) — "On the Limits of Sparse Autoencoders"

2. **Assumption: Measuring absorption leads to actionable interpretability**
   - Evidence: Basu et al. (2026) demonstrate 98.2% probe AUROC but 45.1% output sensitivity; SAE steering produces zero effect in clinical domain. The entire enterprise of "detect then steer" may be fundamentally flawed.
   - Resource: [arXiv:2603.18353](https://arxiv.org/abs/2603.18353) — "Interpretability without Actionability"

3. **Assumption: The "quasi-critical" phase transition framing is valid (chi_ratio=1.88)**
   - Evidence: Standard phase transition theory requires chi_ratio > 3.0 for "sharp" transitions. The observed chi_ratio=1.88 < 3.0 means the "critical point" at λ_c=5e-5 is not actually critical — it's a smoothed crossover, not a true phase transition. This undermines the entire theoretical framework.
   - Resource: Statistical physics literature on phase transitions; Engel & Van den Broeck (2001)

4. **Assumption: lambda_c=5e-5 is a stable, meaningful critical point**
   - Evidence: 10x pilot-to-full shift (5e-4 to 5e-5) reveals λ_c instability. If the "critical point" shifts by 10x between experimental runs, it is not a reliable physical quantity — it may be noise or an artifact of the measurement procedure.
   - Resource: Pilot experiment data showing λ_c instability

5. **Assumption: Phase transition framework (ν=3) generalizes beyond GPT-2**
   - Evidence: All scaling measurements were conducted on GPT-2 TopK SAEs. The claim that ν=3 is a universal critical exponent has not been validated on JumpReLU (GemmaScope) or other architectures. The artifact hypothesis (phase transitions are GPT-2/TopK-specific) has NOT been falsified.
   - Resource: Proposal's own cross-architecture validation plan as "backup" — confirming this gap

6. **Assumption: CV (coefficient of variation) reversal is a "genuine discovery"**
   - Evidence: The finding that absorbed features show CV=7.33 vs non-absorbed CV=0.01 (733x ratio) is being framed as discovery. However, this could be a selection bias artifact: absorbed features are identified BY their high absorption scores, which correlate with activation magnitude, which correlates with CV. The "discovery" may be circular.
   - Resource: Need to verify whether CV difference persists after controlling for activation magnitude

7. **Assumption: Cross-layer analysis at critical sparsity will reveal heterogeneity**
   - Evidence: At λ=0.001, ALL layers saturate at absorption_rate=1.0 (uniform). The proposal claims layer heterogeneity will appear at λ_c=5e-5, but this is untested speculation. If layers saturate uniformly at the true critical point too, H3 is fundamentally falsified — not just at the wrong sparsity level.
   - Resource: Proposal's own admission that H3 needs retesting at λ_c

8. **Assumption: Feature absorption is a "problem" distinct from normal superposition**
   - Evidence: The Toy Models of Superposition (Elhage et al., 2022) paper establishes that overlapping feature representations are fundamental to how neural networks work. Absorption may not be a failure mode — it may be the network's optimal solution to representing more features than dimensions. Fixing "absorption" might actually harm interpretability.
   - Resource: [arXiv:2209.10652](https://arxiv.org/abs/2209.10652) — "Toy Models of Superposition"

### Landscape of Doubt

**What the field assumes:**
- SAEs can extract interpretable features if we solve absorption
- Measuring absorption helps us understand and mitigate interpretability failures
- Phase transitions and critical phenomena provide a valid theoretical framework
- Cross-layer absorption patterns exist and are meaningful

**What the evidence actually shows:**
- Cui et al. proves full disentanglement is mathematically impossible
- Basu et al. proves detection does not translate to actionability
- chi_ratio=1.88 < 3.0 means "quasi-critical" is not truly critical
- λ_c instability (10x shift) means the "critical point" is not stable
- H3 and H6 falsified at λ=0.001 with no guarantee they hold at λ_c
- The entire theoretical framework may be built on artifacts and noise

**The contrarian read**: The field is spending enormous effort measuring a phenomenon (absorption) that may be: (1) mathematically inevitable, (2) unsteerable even when measured perfectly, and (3) possibly the network's optimal representation strategy rather than a bug.

---

## Phase 2: Initial Candidates

### Candidate A: Absorption is Information-Theoretically Inevitable, Not a Bug

- **Challenged assumption**: Feature absorption is a fixable training artifact
- **Evidence against**: Cui et al. (ICLR 2026) prove mathematically that standard SAEs cannot fully recover ground-truth monosemantic features. This is not a training failure — it's a mathematical consequence of superposition and sparsity.
- **Contrarian hypothesis**: Absorption is the SAE's optimal encoding strategy, not a failure mode. The "problem" of absorption may be ill-posed — we cannot fix it without breaking the SAE's ability to represent superposition.
- **Exploitation plan**: Design experiments comparing SAE representations to ground-truth features on synthetic data where ground-truth is known. Test whether absorption-resistant SAEs (OrtSAE, Matryoshka) actually produce better downstream task performance, or just better absorption metrics.
- **Novelty estimate**: 6/10 — Cui et al. establishes the theoretical limits, but doesn't explicitly argue that absorption is "optimal" rather than "inevitable but suboptimal"

### Candidate B: The Phase Transition Framework is a Statistical Artifact

- **Challenged assumption**: Absorption exhibits genuine critical phenomena with ν=3, R²=0.951
- **Evidence against**: (1) chi_ratio=1.88 < 3.0 means the transition is NOT sharp; (2) λ_c instability (10x shift) means the critical point is not reproducible; (3) All measurements on GPT-2 only — no cross-architecture validation
- **Contrarian hypothesis**: The "finite-size scaling" with ν=3 is a curve-fitting artifact. With 12 lambda values and 3 dictionary sizes, there are enough degrees of freedom to fit any power law. The R²=0.951 is a retrospective fit, not a prospective prediction.
- **Exploitation plan**: Test whether the ν=3 scaling holds on held-out dictionary sizes (e.g., 48k, 96k) not used in the original fit. If scaling doesn't generalize, the "universal exponent" claim collapses.
- **Novelty estimate**: 5/10 — phase transitions in neural networks are established (Engel & Van den Broeck); the specific application to SAE absorption could be novel but the statistical artifact hypothesis is a critique, not a new direction

### Candidate C: The CV Reversal is a Selection Bias Artifact, Not a Discovery

- **Challenged assumption**: Absorbed features have 733x higher CV (7.33 vs 0.01) because absorption preserves high-variance specialized information
- **Evidence against**: Absorbed features are identified BY high absorption scores, which are computed from activation magnitudes. Activation magnitude correlates with CV (higher activation → more variance). The "discovery" may be circular: we're finding that features with high activation have higher CV because we selected them for having high activation.
- **Contrarian hypothesis**: The CV difference disappears when controlling for mean activation magnitude. The "variance paradox" is a statistical artifact of selection bias, not a genuine information-theoretic phenomenon.
- **Exploitation plan**: Compute CV conditioned on mean activation (or use Fano factor = CV²/mean as a normalized measure). Test whether CV_absorbed > CV_non-absorbed after matching on activation magnitude.
- **Novelty estimate**: 4/10 — this is a critique of an existing finding, not a new direction

---

## Phase 3: Self-Critique

### Against Candidate A (Absorption is Inevitable/Optimal)

- **Steelman**: Cui et al. proves impossibility of full disentanglement. But impossibility of perfect recovery doesn't mean current absorption levels are optimal — there could be improvement within the theoretical limits. The drug analogy: side effects are "inevitable" due to biochemistry, but we still develop better drugs with fewer side effects.
- **Cherry-picking check**: I am focusing on Cui et al.'s impossibility result while ignoring work showing absorption reduction (OrtSAE -65%, Matryoshka, etc.). These show absorption IS reducible, contradicting the "inevitable optimum" framing.
- **Confounding check**: Absorption may be inevitable at the theoretical level but still have degrees of freedom in HOW absorption occurs. The "optimal encoding" hypothesis assumes the current absorption pattern is Pareto-optimal, which hasn't been shown.
- **Actionability check**: If absorption is truly inevitable and optimal, what does this mean for the research program? If we can't fix absorption, should we abandon SAE-based interpretability? This leads to nihilism, not progress.
- **Verdict**: MODERATE — The impossibility result is real, but the jump to "absorption is optimal" is too strong. A more defensible claim is "absorption is partially inevitable, and we should focus on what CAN be improved."

### Against Candidate B (Phase Transition is Statistical Artifact)

- **Steelman**: Phase transitions and finite-size scaling are well-established in statistical physics. The methodology (sweep λ, measure m(λ), compute dm/dλ for susceptibility, fit scaling function) is standard. R²=0.951 is a strong fit. The fact that chi_ratio < 3.0 just means the transition is gradual, not that the fit is wrong.
- **Cherry-picking check**: I claim the fit is "just curve fitting" without showing what happens on held-out data. The proper test is prospective: predict absorption at new dictionary sizes and λ values, then measure. This hasn't been done yet.
- **Confounding check**: The GPT-2-only validation is a genuine weakness, but the proposal explicitly plans cross-architecture validation. Calling it an "artifact" before that validation is premature.
- **Actionability check**: If phase transitions are artifacts, what replaces them? A theory of absorption that doesn't involve critical phenomena is just "absorption varies with sparsity" — not a publishable insight.
- **Verdict**: WEAK as a standalone claim — the artifact hypothesis is plausible but premature. STRENGTHENED if cross-architecture validation fails (ν ≠ 3 on Gemma-2).

### Against Candidate C (CV Reversal is Selection Bias)

- **Steelman**: The selection bias argument is statistically sound. If we select features by absorption score (which depends on activation magnitude) and then measure CV (variance/mean), we expect correlation by construction. The Fano factor (CV²/mean) normalization is the standard control for this.
- **Cherry-picking check**: The proposal's own pilot data (pilot_steering_cv) shows high-CV features are 2x more steerable. This is a downstream validation that CV predicts behavior, not just an artifact of the measurement. If CV were purely selection bias, why would it predict steering?
- **Confounding check**: Even if CV is partially artifact, the steering result suggests CV captures something real about feature behavior. The steering experiment is the proper validation of the "CV is meaningful" hypothesis.
- **Actionability check**: If CV is just an artifact, we lose a potentially useful predictor for steering effectiveness. But the steering data argues against this — CV predicts steering even if CV itself is partially confounded.
- **Verdict**: MODERATE — the selection bias concern is legitimate, but the steering validation (high-CV = 2x steering) provides evidence that CV captures real behavioral variance. The concern should motivate using Fano factor as a control, not abandoning CV-based analysis.

---

## Phase 4: Refinement

### Dropped Positions

- **Candidate C (CV artifact)** dropped as front-runner because: The steering experiment provides prospective validation that CV predicts behavior. The selection bias concern is valid but addressable with Fano factor normalization. The finding that "high-CV absorbed features are more steerable" survives the artifact critique because steering is an independent downstream measure.

### Strengthened Positions

- **Candidate A (inevitable absorption)** strengthened by: Cui et al.'s theoretical result is robust and not going away. The research community's focus on "reducing absorption" may be misdirected if absorption is within theoretical limits. However, the "optimal" framing is too strong — "partially inevitable" is more defensible.

- **Candidate B (phase transition artifact)** needs prospective validation: The artifact hypothesis is unproven until cross-architecture validation fails. Keep as secondary concern, not primary claim.

### Additional Corroboration

From Basu et al. (2026): The actionability paradox suggests that even PERFECT absorption measurement won't enable steering. This strengthens the contrarian case that measuring absorption may be futile — we should focus on what we can actually DO with SAE features, not just measure them better.

From pilot_activation_patching: 67.3% mean recovery when child is zeroed. This confirms absorbed features are causally significant — the parent IS being routed through the child. This is evidence AGAINST the "absorption is just noise" interpretation and FOR the "absorption is a real routing phenomenon."

### Selected Front-Runner

**Candidate A (Absorption is Partially Inevitable)** selected with modification:

The original framing ("absorption is optimal") is too strong. Reframe as "absorption is partially inevitable within theoretical limits, and we should focus on actionable rather than measurable properties."

The key contrarian insight: The field's focus on measuring absorption better (more layers, more models, more precise critical points) may be futile if the Basu et al. actionability paradox is even partially general. The right question is not "how much absorption is there?" but "which absorbed features can we actually steer, and does measuring absorption help us predict that?"

The pilot_steering_cv result (high-CV = 2x steering) is the most actionable finding because it connects a measurable property (CV) to an actionable outcome (steering effectiveness). This is a better research direction than perfecting absorption metrics that may not predict anything actionable.

---

## Phase 5: Final Proposal

### Title

**Rethinking Feature Absorption: From Measurement to Actionability**

### Challenged Assumption

The mainstream assumption in SAE research is that feature absorption is a problem to be solved through better measurement and architectural modifications. The field asks: "How much absorption is there, and how can we reduce it?" The contrarian question is: "Does measuring absorption help us predict what we can actually DO with SAE features?"

### Evidence

**For the assumption (mainstream view)**:
- Chanin et al. establishes absorption as a measurable phenomenon requiring attention
- OrtSAE (-65% absorption), Matryoshka SAE, and other architectural fixes show absorption IS reducible
- The variance paradox (CV_reversed) suggests absorption preserves information that could be actionable

**Against the assumption (contrarian view)**:
- Cui et al. proves full disentanglement is mathematically impossible — absorption is partially inevitable
- Basu et al. demonstrates that even perfect internal detection (98.2% AUROC) translates to zero steering utility
- chi_ratio=1.88 < 3.0 undermines the "quasi-critical" theoretical framework
- λ_c instability (10x shift) means critical point measurements are not reliable
- H3 and H6 falsified at λ=0.001, with no guarantee they hold at the supposedly true critical point

### Hypothesis

**The Actionability Hypothesis**: The fraction of absorbed features that remain steerable is determined by their coefficient of variation (CV). High-CV absorbed features (CV > 1.0) are "robust absorbed" — they route through context-sensitive child channels that preserve steering potential. Low-CV absorbed features (CV ≤ 1.0) are "fragile absorbed" — they route through stable child channels that compensate for parent steering.

**Secondary hypothesis**: The Basu et al. actionability paradox may be partially explained by their feature set being predominantly low-CV (fragile absorbed). If so, the paradox is not universal — it reflects a sampling bias in their feature selection.

### Method

1. **CV-Based Feature Classification**: Classify absorbed features into high-CV (CV > 1.0) and low-CV (CV ≤ 1.0) subpopulations based on per-feature CV across 1000 samples.

2. **Steering Effectiveness Test**: Run steering experiments on 30 high-CV and 30 low-CV absorbed features. Measure logit change at semantically appropriate tokens for +3, +5, +7 steering strengths. **This is the key test — if high-CV features are NOT more steerable, the hypothesis is falsified.**

3. **Fano Factor Normalization**: Control for activation magnitude using Fano factor (CV²/mean) to verify CV is not just a proxy for activation magnitude.

4. **Cross-Model Validation**: Replicate the robustness decomposition on Gemma-2-2B JumpReLU SAEs to test whether the CV-steering correlation generalizes beyond GPT-2/TopK.

5. **Basu et al. Feature Analysis**: If Basu et al. feature set is available, compute CV distribution to test whether their clinical features were predominantly low-CV.

### Experimental Plan

| Experiment | Details | Duration | Validates |
|-----------|---------|----------|-----------|
| E1: CV-based feature classification | GPT-2 layer 6, classify absorbed features | 15 min | High/low CV split |
| E2: Fano factor normalization | Control for activation magnitude | 15 min | CV is not artifact |
| E3: Steering effectiveness comparison | 30 high-CV vs 30 low-CV, 3 strengths | 30 min | Actionability hypothesis |
| E4: Cross-model validation | Gemma-2-2B layer 6, same protocol | 30 min | Generalizability |
| E5: Basu et al. feature CV analysis | If data available | 10 min | Paradox resolution |

**Total**: ~100 min across 5 experiments

### Baselines

The mainstream method (no CV decomposition): Basu et al. approach of treating all absorbed features as uniformly non-steerable. If our CV-based decomposition shows that high-CV features are significantly more steerable, we outperform this baseline.

### Risk Assessment

1. **Risk: CV-steering correlation doesn't replicate on held-out data**. If high-CV absorbed features are NOT more steerable in a larger sample, the hypothesis is falsified. *Mitigation*: Report as negative result — the finding that "CV does not predict steering effectiveness" is itself a useful contribution clarifying the limits.

2. **Risk: Fano factor normalization shows CV is purely an activation magnitude proxy**. If CV disappears after controlling for mean activation, the entire CV-based research program is undermined. *Mitigation*: Report as negative result; switch to Fano factor-based classification.

3. **Risk: Steering effects are too small for practical utility**. Even high-CV features show only ~0.15 logit change. *Mitigation*: Compare to non-absorbed baseline steering effects; if robust absorbed is comparable to non-absorbed, the finding is significant.

### Novelty Claim

**What is genuinely novel**:
1. **First CV-based prediction of steering effectiveness** within absorbed features — prior work (Basu et al.) treats all absorbed as uniformly non-steerable
2. **First partial resolution of the actionability paradox** — if high-CV absorbed features are steerable, the paradox is not universal
3. **First connection between statistical moments (CV) and causal actionability** — a simple statistical measure predicts whether a feature can be steered

**Prior work collisions**:
- Basu et al. (2026): Actionability paradox — we extend by showing heterogeneity rather than universal failure
- Chanin et al. (2024): Absorption detection — we connect to steering outcomes rather than just measuring absorption
- Cui et al. (2026): Information-theoretic limits — we work within these limits rather than trying to overcome them

**The contrarian insight**: Stop trying to measure absorption better. Start predicting what we can actually DO with absorbed features. The CV-based actionability decomposition is the first step toward an actionability-first research program for SAE interpretability.