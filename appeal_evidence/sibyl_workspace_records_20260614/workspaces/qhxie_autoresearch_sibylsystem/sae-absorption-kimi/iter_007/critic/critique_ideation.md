# Ideation Critique: Iteration 009 Proposal

## Overall Assessment

The research direction is sound and addresses a genuine gap in the literature. The contrarian framing (does absorption actually matter?) is intellectually honest and potentially high-impact. However, the proposal overpromises on what can be delivered in a single iteration.

## Strengths

1. **Honest negative results framing**: The proposal explicitly plans for null results and frames them as valuable. This is methodologically mature.

2. **L0-matching protocol**: The attempt to control for sparsity when comparing architectures is methodologically sound and addresses a real gap.

3. **Ground-truth synthetic data**: Using SynthSAEBench eliminates probe-based confounds, a genuine improvement over prior work.

4. **Pre-registration**: The analysis plan is written before data collection, reducing p-hacking risk.

## Weaknesses

### 1. Scope Creep (MAJOR)

The proposal lists 4 research questions (RQ1-RQ4) with 8 hypotheses (H1a-c, H2a-c, H3a-b, H4a-b). In practice, only RQ1 and RQ2 were addressed, and RQ2 only partially (missing Manipulation A). RQ3 and RQ4 are mentioned in the paper but not substantively analyzed.

**Recommendation**: Focus on RQ1 (L0 confound) and RQ2 (causality) only. Defer RQ3-RQ4 to future work.

### 2. Hypothesis H2c is Untestable (MAJOR)

H2c claims: "The correlation is causal, not merely correlational: artificially inducing absorption (via targeted sparsity increase) degrades downstream performance."

The dose-response design (Manipulation B) varies lambda, which affects BOTH sparsity AND absorption. It cannot isolate absorption as the causal factor. If only Manipulation B shows a correlation, the effect is sparsity-driven, not absorption-specific—as the methodology itself acknowledges.

**Recommendation**: Remove H2c or redesign to isolate absorption independently of sparsity. One approach: fix sparsity (e.g., L0=50) and manipulate absorption via architectural changes.

### 3. Theoretical Contribution Overclaimed (MAJOR)

The mutual coherence predictor (H3a-b) is framed as "the first theoretically grounded absorption predictor" bridging compressed sensing to SAEs. However:
- The compressed sensing theory assumes linear measurements; SAEs are nonlinear
- The threshold mu < 1/(2k-1) is for exact recovery in linear systems, not feature absorption in autoencoders
- No theoretical justification is given for why this threshold should apply to SAEs

**Recommendation**: Frame H3 as an exploratory hypothesis, not a theoretical contribution. Acknowledge the linearity mismatch.

### 4. Task Generalization is Underdeveloped (MINOR)

RQ4 proposes testing first-letter vs. semantic absorption, but the semantic tasks (syntactic, factual, safety) are not defined in sufficient detail. The pilot result for RQ4 (`pilot_rq4_semantic_absorption_results.json`) shows rates "below shuffled-label baselines," suggesting the detection method failed.

**Recommendation**: Defer RQ4 until a reliable semantic absorption detection method is developed.

### 5. Pilot-First Design Not Fully Executed (MINOR)

The proposal promises a 15-minute pilot before scaling, but the pilot and full experiments appear to have run simultaneously or the pilot was not used for go/no-go decisions.

**Recommendation**: Actually use pilot results for go/no-go decisions. If pilot shows MCC is flat (as it did), reconsider the downstream metric before running full experiments.

## Risk Assessment Accuracy

The proposal's risk assessment was partially accurate:
- "No causal link found" (Medium likelihood): **Occurred**, but for the wrong reason (metric insensitivity, not genuine null effect)
- "Mutual coherence theory doesn't empirically hold" (Medium likelihood): **Not tested** (no correlation computed)
- "Semantic detection unreliable" (Medium likelihood): **Occurred** (rates below baseline)
- "Training time exceeds budget" (Low likelihood): **Did not occur** (training was suspiciously fast)

## Pivot Recommendation

Given the issues identified, the project should:

1. **Fix the metrics**: Debug MCC, explained variance, and dead latent reporting
2. **Run true L0-matched comparison**: Include lambda=0.02 Baseline per pilot data
3. **Add statistical tests**: Report p-values and effect sizes
4. **Then decide**: If L0-matched comparison shows architectural differences persist, the contribution is stronger. If differences vanish, the contrarian framing gains support.

Do NOT pivot to Alternative D (compositional semantics) until the metric issues are resolved. A null result on a broken metric is not evidence for the contrarian view.
