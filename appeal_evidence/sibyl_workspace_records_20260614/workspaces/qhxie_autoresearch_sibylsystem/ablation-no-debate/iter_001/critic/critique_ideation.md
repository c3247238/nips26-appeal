# Ideation Critique: Feature Absorption Research

## Overall Assessment

The ideation phase produced a genuinely novel measurement methodology (multi-child proportional ablation) and identified a high-novelty direction (safety-critical feature absorption, H_Safe). However, the competitive exclusion framing (H2) was speculative and unsupported by the data, and the causal steering validation (H3) was technically infeasible on synthetic activations.

---

## Critical Issues

### 1. Competitive Exclusion Framing Was Never Plausible (CRITICAL)

**Location**: hypotheses.md H2, proposal.md Section 4.3

**Problem**: The competitive exclusion hypothesis (rho < -0.3) was imported from ecological theory without any theoretical bridge to SAE optimization. The proposal even notes: "Features do not 'compete' in the same biological sense; rather, they are jointly optimized to minimize reconstruction loss." Yet H2 was pre-registered and the paper frames its falsification as a contribution.

**Critique**: Pre-registering a hypothesis borrowed from ecology without theoretical justification is methodologically questionable. The "falsification" is not informative because the hypothesis had no theoretical basis in the SAE domain.

**Fix**: Frame H2 as an exploratory analysis, not a pre-registered hypothesis. State: "We tested whether frequency-absorption patterns resemble competitive exclusion dynamics; the null result is uninformative due to lack of theoretical grounding."

---

### 2. H_Safe Never Executed Despite Being Highest Novelty (CRITICAL)

**Location**: proposal.md Part C, hypotheses.md H_Safe

**Problem**: H_Safe was rated 9/10 novelty with "no prior work examines whether safety-critical features are disproportionately absorbed." The proposal explicitly prioritizes it as "highest novelty priority" and "no training needed." Yet it was never executed.

**Critique**: The paper claims novelty in safety-critical analysis but provides zero evidence. This is a significant gap between promised and delivered contributions.

**Fix**: Either (a) execute H_Safe before publication, or (b) remove safety claims and reframe as future work.

---

### 3. H_Mech (Geometric vs Learned) Never Executed (MAJOR)

**Location**: proposal.md Part B, hypotheses.md H_Mech

**Problem**: The 2x2 factorial decomposition was planned as a key contribution distinguishing geometric from learned components. The pilot data already suggested geometric dominance (shuffled/permuted ~0.48 vs trained 0.50). But the factorial experiment was never run.

**Critique**: The paper makes claims about "absorption is primarily geometric" without the planned experiment to support them. The conclusion is inferred from baseline comparisons, not from the designed factorial.

**Fix**: Either run H_Mech or weaken claims to "pilot evidence suggests geometric dominance; factorial decomposition planned."

---

### 4. Theoretical Framework Is Thin (MAJOR)

**Location**: proposal.md, paper.md Background

**Problem**: The paper borrows ecological competitive exclusion and niche geometry concepts without establishing their validity for SAE feature dynamics. The Tang et al. (2025) citation provides theoretical grounding for local minima = absorption, but the paper does not connect this to the competitive exclusion framing.

**Critique**: The theoretical framework is a patchwork of analogies (ecology, geometry) rather than a coherent theory. The paper would be stronger with a single theoretical lens.

**Fix**: Drop competitive exclusion framing entirely. Focus on Tang et al.'s geometric local minima theory as the theoretical foundation.

---

## What Works Well

1. **Multi-child proportional ablation**: Genuinely novel measurement fix for saturation
2. **Honest negative result documentation**: H2 failure reported without spin
3. **Novelty assessment**: The novelty report correctly identified H_Safe as highest-value direction
4. **Pivot planning**: alternatives.md provides clear decision tree for adapting to results

---

## Score: 5/10

Strong methodology idea undermined by speculative theoretical framing and failure to execute highest-novelty experiments.
