# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption**: Feature absorption is a fixable training artifact that can be reduced through architectural modifications (Matryoshka SAE, OrtSAE, ATM, Masked Regularization).
   - Evidence challenging it: Cui et al. (ICLR 2026) prove mathematically that standard SAEs **cannot** fully recover ground-truth monosemantic features due to intrinsic representational interference. This is not a training bug — it's a mathematical inevitability under realistic sparsity constraints. The "fixes" address symptoms, not the root cause.

2. **Assumption**: Quantifying absorption leads to actionable interpretability insights. SAE features with high detection AUROC can be used to steer model behavior.
   - Evidence challenging it: Basu et al. (2026) demonstrate **98.2% AUROC but zero output change** via SAE steering. The field has spent 2+ years developing absorption metrics, but this critical negative result suggests the entire research program may be practically useless — we can detect features perfectly but cannot modify model behavior through them.
   - This is the most dangerous assumption in the field: mistaking measurement validity for actionable utility.

3. **Assumption**: The phase transition framework (H1: critical sparsity threshold, H2: finite-size scaling) represents a fundamental discovery about SAE training dynamics.
   - Evidence challenging it: The critical lambda varies across models (GPT-2: 5e-5), and the framework failed for H3 (layer criticality — absorption saturates uniformly at ~1.0 for all layers, contradicting the "temperature" analogy) and H6 (graph topology fails as order parameter). The phase transition may be an artifact of the specific GPT-2/TopK configuration rather than a universal phenomenon.

4. **Assumption**: Absorbed features are uniformly "degraded" versions of parent features that lose information.
   - Evidence challenging it: The **reversed CV finding** (CV_absorbed=7.33 vs CV_non_absorbed=0.01) suggests absorbed features are actually MORE variable, not less. The innovator and theoretical perspectives interpret this as "context-sensitive information preserved." But an alternative interpretation: high CV could indicate **noise amplification from suppression** — when parent activation is suppressed, the residual through child channels becomes noisy, inflating variance.

5. **Assumption**: The actionability paradox is domain-specific (Basu et al. studied clinical domain).
   - Evidence challenging it: The fundamental mechanism (residual stream compensation) is architectural, not domain-specific. If absorbed features route through child latents to residual stream, and residual stream dynamics determine output, then steering the parent via SAE activates the child, which contributes to residual exactly as it would without steering. The output change is zero by construction, not by domain.

### Landscape of Doubt

The SAE field operates under an **"interpretability illusion"** — we believe measuring internal features helps us understand and control neural networks, but Basu et al. prove this control is an illusion. The absorption literature (Chanin et al., 2024) documents the problem in detail, but all proposed solutions require retraining SAEs, and Cui et al. prove retraining cannot solve the fundamental limitation.

The project's own experimental evidence reveals additional cracks:
- H3 failure: "Layer 6 as critical point" was NOT supported — absorption saturates uniformly
- H6 failure: Graph topology does NOT serve as order parameter for absorption phase transition
- H4 reversal: The hypothesis that "absorbed features have lower CV" is disproven; they have MUCH higher CV

These failures suggest the theoretical framework may be partially wrong, not just incomplete.

---

## Phase 2: Initial Candidates

### Candidate A: The "Nothing Works" Hypothesis — Absorbed Features Are Unsteerable by Design

- **Challenged assumption**: That better absorption metrics or architectural modifications will lead to actionable interpretability tools.
- **Evidence against**: Basu et al. (2026) show 98.2% AUROC → 0% output change. This is not a calibration problem — it's a structural inevitability. When you activate a parent feature via SAE, you also activate the absorbing child feature(s), whose contribution to residual stream is what the model actually uses. The parent's "detected" presence is epiphenomenal.
- **Contrarian hypothesis**: SAE steering is fundamentally broken for absorbed features because the residual stream compensation is architectural, not learnable. The only path to actionable interpretability is NOT through better SAE training but through direct residual stream intervention that bypasses SAE entirely.
- **Exploitation plan**: Demonstrate that residual stream steering (without SAE) produces output changes where SAE steering produces none. This is a negative result — showing what doesn't work — but could be a strong ICLR submission if framed as "What Breaks SAE-Based Interpretability."
- **Novelty estimate**: 6/10 — Basu et al. already planted the seed. The incremental contribution would be systematic characterization of WHEN the actionability paradox holds (all absorbed features? all layers? all models?) and WHY (mediation structure).

### Candidate B: Phase Transitions Are a GPT-2/TopK Artifact

- **Challenged assumption**: The critical sparsity threshold (H1) and finite-size scaling (H2) represent universal SAE behavior.
- **Evidence against**: The critical lambda was discovered on GPT-2 layer 6 with TopK SAEs. H3 and H6 failures suggest the "temperature" and "order parameter" analogies are wrong. If the phase transition framework were universal, why did H3 and H6 fail?
- **Contrarian hypothesis**: Phase transition behavior is an artifact of the specific architecture (TopK SAEs enforce exact k-sparsity, creating sharp transitions). JumpReLU SAEs (used in GemmaScope) have continuous thresholds and may not exhibit sharp phase transitions. The apparent critical behavior is a measurement artifact of TopK's discrete sparsity enforcement.
- **Exploitation plan**: Run sparsity sweeps on Gemma-2-2B SAEs (JumpReLU) using the same methodology. If no critical threshold is found (absorption varies smoothly with lambda), this validates the artifact hypothesis. If a critical threshold exists with different lambda_c, this suggests architectural dependence rather than universality.
- **Novelty estimate**: 5/10 — Cross-architecture comparison is valuable but the "artifact hypothesis" is less publishable than "universal phase transition."

### Candidate C: The CV Reversal Is Noise, Not Signal

- **Challenged assumption**: The reversed CV finding (7.33 vs 0.01) indicates meaningful difference between absorbed and non-absorbed features.
- **Evidence against**: CV_non_absorbed = 0.01 is suspiciously low — near-zero variance suggests these are "always-on" features that activate identically across all contexts. The CV comparison may be comparing apples (context-sensitive features) to oranges (constitutively active features), not absorbed vs non-absorbed.
- **Contrarian hypothesis**: The CV difference is a statistical artifact of heterogeneous feature populations, not a meaningful signal about absorption. Absorbed features may simply be a random subset of features that happen to be context-sensitive, and the absorption classification correlates with context-sensitivity only because both are more likely in certain layers.
- **Exploitation plan**: Control for layer and activation frequency. Compare CV only within features of similar overall activation magnitude. If the CV difference disappears when controlling for these confounders, the CV reversal is an artifact.
- **Novelty estimate**: 4/10 — Statistical artifact explanation is less interesting than novel signal, but worth validating to avoid building theory on shaky foundations.

---

## Phase 3: Self-Critique

### Against Candidate A: "Nothing Works"

- **Steelman**: Basu et al. is a single paper in clinical domain. The actionability paradox may be solvable through better steering protocols (stronger interventions, different steering vectors, residual stream normalization). Perhaps absorbed features CAN be steered if we bypass SAE activation and directly modify residual stream.
- **Cherry-picking check**: The field has many positive SAE steering results (Cunningham et al., 2023; Templeton et al., 2024). Cherry-picking Basu et al. as the definitive result ignores the majority of work showing SAE steering IS possible.
- **Confounding check**: Basu et al. studied clinical domain (medical code generation). Maybe LLMs for code generation have different residual stream dynamics that prevent SAE steering. The paradox may not generalize to other domains.
- **Actionability check**: Even if absorbed features are unsteerable, non-absorbed features should remain steerable. The practical implication: use absorption metrics to identify which features are safe to steer, avoiding wasted effort on absorbed (unsteerable) features.
- **Verdict**: MODERATE — The steelman is strong (positive steering results exist), but the structural mechanism Basu proposes is compelling. The practical implication (filter by absorption score before steering) is actionable and doesn't require solving absorption.

### Against Candidate B: Artifact Hypothesis

- **Steelman**: H1 and H2 are strongly supported (chi_peak=11.19, R²=0.95). The phase transition framework is the project's strongest empirical finding. Dismissing it as artifact requires strong evidence, not just speculation about architecture differences.
- **Cherry-picking check**: H3 and H6 failures are cherry-picked from the full hypothesis set — 4/6 hypotheses passed. The failures may be due to incorrect hypothesis formulation (H3: wrong "temperature" analogy) rather than framework invalidity.
- **Confounding check**: GPT-2 is a small model (85M params). Phase transitions may be more pronounced in smaller models with discrete TopK SAEs, while larger models with different architectures show smoother behavior. This is an architectural dependency, not an artifact.
- **Actionability check**: If phase transitions are architecture-specific, the practical implication is: choose lambda < lambda_c for your specific architecture before training SAEs. This is actionable guidance even if not universal.
- **Verdict**: STRONG — The artifact hypothesis is plausible but premature. The empirical evidence for phase transitions is strong. Better to validate by testing on JumpReLU SAEs (Gemma) than to dismiss the finding.

### Against Candidate C: Noise Hypothesis

- **Steelman**: CV_non_absorbed=0.01 is genuinely suspicious. The hypothesis that absorbed features are more "variable" is interesting, but the alternative explanation (always-on features have near-zero variance) explains the observation without requiring absorption theory.
- **Cherry-picking check**: The CV comparison uses absorption classification from the project's own metric, which may have errors. If some "absorbed" features are misclassified, the CV difference is confounded by classification noise.
- **Confounding check**: Activating features with high mean activation (always-on) would naturally have low CV (denominator is mean, numerator is standard deviation — large mean → small CV even with moderate variance). This is a mathematical artifact of CV as a metric, not a feature property.
- **Actionability check**: If CV difference is noise, it has no practical implications — you cannot use CV to predict steering effectiveness.
- **Verdict**: WEAK — The noise hypothesis is methodologically sound (CV is problematic for high-mean features) but offers no actionable insight. Better to reformulate the comparison (e.g., use Fano factor = variance/mean instead of CV) than to dismiss the finding.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Noise Hypothesis)**: WEAK — Conceded the methodological point but offers no path to actionable findings. The reformulation (Fano factor) is an incremental improvement that doesn't advance the research program.

### Strengthened Ideas

- **Candidate A (Nothing Works)**: Strengthened by the Basu et al. structural argument. The residual stream compensation mechanism is theoretically sound and explains why absorption metrics don't translate to steering utility. However, the cherry-picking concern is real — positive steering results exist. The resolution: distinguish absorbed vs non-absorbed features. Perhaps non-absorbed features ARE steerable, and only absorbed features exhibit the paradox.

- **Candidate B (Artifact Hypothesis)**: Refined to focus on VALIDATION rather than dismissal. The correct approach: test GPT-2 phase transition findings on Gemma-2-2B JumpReLU SAEs. If the critical threshold appears at different lambda_c with similar susceptibility peak, the framework generalizes with architectural correction. If no critical threshold appears, the artifact hypothesis is confirmed.

### Selected Front-Runner

**Candidate A (Nothing Works) with Gemma validation**

Rationale:
1. Basu et al.'s negative result is the most important finding in the field (raising questions about the entire research program's utility)
2. The project's phase transition findings (H1/H2) are strong but may not generalize beyond GPT-2/TopK
3. The practical path forward: distinguish steerable (non-absorbed) from unsteerable (absorbed) features using absorption metrics, then apply steering only to non-absorbed features
4. The Gemma validation (Candidate B) becomes supporting evidence rather than competing hypothesis: we validate phase transitions on JumpReLU, and separately validate that non-absorbed features are indeed steerable

The final proposal: **Absorption Metrics Don't Predict Steering Effectiveness: A Negative Result and Its Implications for SAE-Based Interpretability**

---

## Phase 5: Final Proposal

### Title

**Rethinking SAE Feature Absorption: Why Detection Metrics Don't Predict Steering Utility**

### Challenged Assumption

The SAE field assumes that features with high absorption scores are "problematic" (non-monosemantic) and features with low absorption scores are "good" (monosemantic, actionable). This assumption underlies the entire research program: if we can measure absorption, we can identify which features to trust for interpretability.

### Evidence

**For the assumption**: Chanin et al. (2024) show absorbed features have systematic false negatives in probing tasks. The Innovator and Theoretical perspectives argue that high-CV absorbed features may retain steering potential through specialized context sensitivity. The theoretical prediction is that non-absorbed features should be more reliably steerable.

**Against the assumption**: Basu et al. (2026) demonstrate 98.2% AUROC but zero output change via SAE steering. Critically, they do not filter by absorption status — their negative result applies to features detected with high AUROC, which presumably includes both absorbed and non-absorbed features. This suggests the actionability paradox is NOT specific to absorbed features but may be a general property of SAE steering.

### Hypothesis

The actionability paradox (probe AUROC ≠ steering effectiveness) applies to ALL SAE features, not just absorbed ones, because:
1. SAE steering activates features through the encoder, which modifies both the target feature AND correlated features (due to superposition)
2. The correlated features' contributions to residual stream may cancel or compensate for the target feature's effect
3. This residual stream compensation is architectural, not learnable — it occurs regardless of absorption status

**Secondary hypothesis**: If any subset of features is steerable, they are features whose decoder weights are maximally orthogonal to all other features' decoders (minimizing residual stream interference).

### Method

**Step 1: Systematic Steering Test**
- Select 50 absorbed features (high absorption score) and 50 non-absorbed features (low absorption score) from GPT-2 layer 6 SAE
- For each feature, measure steering effectiveness: activation strength ±3, ±5, ±7
- Steering target: measure logit change at semantically appropriate tokens
- Compare steering effectiveness between absorbed vs non-absorbed groups

**Step 2: Decoder Orthogonality Analysis**
- Compute decoder weight cosine similarity matrix for all selected features
- Test correlation: features with low mean cosine similarity to other features should show higher steering effectiveness
- This tests the "orthogonal = steerable" hypothesis

**Step 3: Cross-Architecture Validation**
- Replicate on Gemma-2-2B layer 6 SAE to test generalization
- Use SAEBench's probe projection metric for absorption classification (works across all layers)

### Experimental Plan

| Experiment | Details | Duration |
|------------|---------|----------|
| E1: Steering comparison | 50 absorbed vs 50 non-absorbed features, 3 steering strengths each | 45 min |
| E2: Orthogonality analysis | Cosine similarity computation for selected features | 15 min |
| E3: Gemma replication | Repeat E1-E2 on Gemma-2-2B SAE | 45 min |

**Falsification criteria**:
- If non-absorbed features show significantly higher steering effectiveness (p < 0.01, Cohen's d > 0.5), the absorption metric IS predictive of steering utility
- If no significant difference between absorbed and non-absorbed, the actionability paradox is universal and absorption metrics are diagnostically useless for steering
- If orthogonality correlates with steering effectiveness (r > 0.3), this provides an alternative predictor

### Baselines

1. **Basu et al. (2026)**: 98.2% AUROC → 45.1% output sensitivity, zero steering effect in clinical domain. This is the baseline negative result we expect to either confirm or refute.
2. **Chanin et al. (2024)**: Absorption metric as classification of monosemantic vs non-monosemantic features. We test whether this classification predicts steering effectiveness.
3. **This project's H4/H5**: CV reversal and co-occurrence analysis suggesting absorbed features have different statistical properties. We test whether these properties predict steering.

### Risk Assessment

**If the hypothesis is wrong (absorption metrics ARE predictive)**: This is the expected positive result — we would report that non-absorbed features are reliably steerable and absorption metrics guide interpretability tool development.

**If the hypothesis is confirmed (actionability paradox is universal)**: This is a strong negative result — the entire research program of "measure absorption to identify trustworthy features" is invalidated. The practical implication: SAE steering doesn't work regardless of absorption status, and the field needs to pivot to non-SAE interpretability methods.

**Risk of confounded measurement**: Steering effectiveness is sensitive to steering strength, prompt context, and tokenization. Mitigation: use standardized prompts and multiple steering strengths, report full distribution not just mean.

### Novelty Claim

1. **First systematic test** of whether absorption metrics predict steering effectiveness in non-clinical domain (GPT-2, Gemma-2)

2. **First connection** between decoder orthogonality and steering effectiveness — provides mechanistic hypothesis for why some features may be steerable

3. **Direct replication and extension** of Basu et al.'s actionability paradox — confirms or refutes whether the paradox generalizes beyond clinical domain

4. **Actionable negative result**: If confirmed, provides clear guidance that the field's focus on absorption metrics is misguided and alternative interpretability approaches are needed

---

## Key Contrarian Insights

1. **The emperor has no clothes**: The SAE field has spent 2+ years developing absorption metrics, but Basu et al. suggest these metrics don't predict what we actually care about (steering utility). This doesn't mean the metrics are wrong — they measure what they claim to measure. But what they measure (internal feature detection) may be irrelevant to the task (model control).

2. **The phase transition framework may be an artifact**: H1/H2 are strong on GPT-2/TopK, but H3/H6 failures suggest the theoretical analogies are wrong. The critical threshold may be architecture-specific, not universal.

3. **The CV reversal demands explanation**: The theoretical and innovator perspectives interpret the CV_reversed finding as "absorption preserves context-sensitive information." But the contrarian explanation (noise amplification from suppression) is equally valid without requiring new theory.

4. **Actionability paradox may be universal, not specific to absorbed features**: If confirmed, this means the entire research program of "measure features, then steer" is broken, not just for absorbed features but for all SAE-based interpretability.

The contrarian contribution is not to dismiss the field's achievements but to identify the critical assumptions that may be wrong, forcing rigorous validation before the research program proceeds further down a potentially dead end.