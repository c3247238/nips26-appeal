# Skeptic Report: Feature Absorption in SAEs (Iteration 7)

**Date**: 2026-04-29
**Iteration**: 7 (post-pilot synthesis with new experiments)
**Skeptic Analysis**: Maximum skepticism review

---

## Executive Summary

New experiments have addressed some prior concerns (H3 steering fixed, multi-seed validation added, H_Mech factorial conducted) but critical statistical issues remain. The core finding -- that trained SAEs show higher absorption than random baselines -- is robust (d=8.94), but the interpretation of "absorption as encoder-driven learning" rests on a confounded experimental design. H_Safe is unsupported by validated safety annotations.

---

## 1. Statistical Risk Inventory

### Risk 1: Deterministic Trained SAE Absorption (ZERO Variance Persists)

**Citation**: multiseed_pilot.json shows all 3 seeds (42, 43, 44) yield exactly 0.500 absorption for trained SAEs. std = 0.0, variance = 0.0.

**Why this is unreliable**: With n=3 seeds and zero variance, you cannot compute a meaningful confidence interval. The t-test with t=82.78 and p=0.0 is mathematically meaningless -- p-values require variance in both groups. The pilot explicitly notes `overall_pass: false` and `non_zero_variance: false`. This is not a statistical test; it is a tautology: "trained SAE absorption equals 0.5" vs "random SAE absorption does not equal 0.5."

**Concern from prior iteration**: The skeptic recommended "introduce stochastic noise in hierarchy generation." This was NOT done. The multi-seed validation still uses the same fixed synthetic hierarchy across seeds.

**Severity**: **Fatal Flaw** -- The core claim of H1 rests on a statistical artifact. A single deterministic value cannot demonstrate a "significant difference" in the traditional sense.

---

### Risk 2: H_Mech Factorial Uses Same Trained Encoder for All Conditions

**Citation**: h_mech_factorial.json shows:
- Condition A (Rand Enc, Rand Dec): 0.299
- Condition B (Train Enc, Rand Dec): 0.490
- Condition C (Rand Enc, Train Dec): 0.299
- Condition D (Train Enc, Train Dec): 0.484

**Why this is unreliable**: The paper claims decoder contribution is "0.0" based on A = C = 0.299. However, Condition C uses the SAME trained encoder from Condition B (just labeled "Rand Enc" for the comparison). This means the encoder still contains learned hierarchical structure. You cannot isolate decoder geometry when the encoder is already trained.

**Internal inconsistency**: B (0.490) > D (0.484), suggesting the decoder ACTUALLY reduces absorption when trained alongside the encoder. This contradicts the conclusion that "decoder has zero effect."

**The decomposition math**: geometric=0.299, learned=0.185, total=0.484. But 0.299 + 0.185 = 0.484 = D. This means the "geometric contribution" is actually what Condition A measures with a trained-but-re-labeled encoder. The experiment design is circular.

**Severity**: **Serious Concern** -- The interpretation that "absorption is encoder-driven, not decoder geometry" is not supported by the experimental design.

---

### Risk 3: H_Safe Uses Synthetic Feature Labels (Not Real Safety Annotations)

**Citation**: h_safe_pilot.json uses features 500-519 as "safety" and 100-119 as "non-safety" with no validation that these indices correspond to actual safety-relevant features.

**Why this is unreliable**: The paper claims to study "safety-critical feature absorption" but selects features by index, not by verified safety relevance. The 90.7% absorption rate for BOTH groups is suspicious -- it suggests either (a) ALL features are equally absorbed, or (b) the measurement methodology saturates at ~90% regardless of feature type.

**From prior iteration**: The skeptic recommended using Gemma Scope SAEs with Neuronpedia-verified safety annotations. The pilot_summary.md explicitly states "H_Safe pilot used synthetic SAE (Gemma Scope not available)." This means the safety claim is entirely unsupported by the actual experimental execution.

**Severity**: **Serious Concern** -- H_Safe addresses a genuine safety concern but uses unvalidated proxies.

---

### Risk 4: H3 Steering Sensitivity Conflates Activation Magnitude

**Citation**: h3_fix_pilot.json shows:
- Absorbed mean sensitivity: 0.055
- Non-absorbed mean sensitivity: 0.034
- Ratio: 1.62x

**Why this is unreliable**: The experiment does not control for activation magnitude. Features with higher mean activations naturally show larger absolute changes for the same relative steering coefficient. The steering_verification shows delta norms of 0.013-0.152, but does not normalize by baseline activation level.

**Remediation attempted**: The prior skeptic recommended "control for activation magnitude." The steering_verification section shows raw delta norms but does not report relative (percentage) sensitivity. The sensitivity_ratio=1.62 is computed on absolute values.

**Severity**: **Serious Concern** -- Causal link between absorption and steering sensitivity is not established; magnitude confound persists.

---

## 2. Alternative Explanations

### Alternative for H_Mech (Encoder-Driven, Not Decoder Geometry)

**Original claim**: Absorption is determined by encoder alignment, decoder geometry contributes nothing.

**Alternative 1**: The H_Mech experiment conflates "trained encoder" with "encoder that aligns with hierarchy." When you compare A vs B, you are comparing "untrained encoder (no hierarchy capture)" vs "trained encoder (hierarchy captured)." But Condition C (labeled "Rand Enc, Train Dec") actually uses the same trained encoder from B, so the comparison is meaningless.

**Alternative 2**: The decoder DOES contribute to absorption, but in the opposite direction. B (0.490) > D (0.484) suggests training the decoder REDUCES absorption slightly when the encoder is also trained. The decoder's role may be orthogonal to absorption -- it minimizes reconstruction error, which incidentally affects how parent-child relationships are encoded.

### Alternative for Deterministic 0.5 Absorption

**Original claim**: Absorption is a learned property of trained SAEs.

**Alternative**: The 0.5 absorption rate reflects a mathematical constraint of the synthetic hierarchy construction. The hierarchy has 2 children and 4 grandchildren with fixed cosine similarities (0.67 between parent-children, 1.0 between child-grandchild). With L0=32 and the specific geometry used, 0.5 may be the equilibrium absorption rate regardless of training.

### Alternative for H3 Steering Sensitivity

**Original claim**: Absorbed features are 1.62x more sensitive to steering toward parent directions.

**Alternative**: The sensitivity difference reflects raw activation magnitude differences between absorbed (stronger) vs non-absorbed (weaker) features, not absorption-specific effects.

---

## 3. Proxy Metric Audit

### Gap 1: Absorption Rate vs Actual Feature Hierarchy

**Claimed**: Multi-child proportional ablation measures "absorption" (child features substituting for parent features).

**Measured**: `parent_activation_after / parent_activation_before` -- a reconstruction residual.

**Gap**: There is no verification that the "children" are actually child nodes in a semantic hierarchy -- they are just the top-5 correlated features by cosine similarity. The paper acknowledges MCC ~0.21 (chance-level) in prior iterations, suggesting the overlap method does not reliably recover ground-truth hierarchies.

### Gap 2: Gemma Scope SAEs Not Actually Used

The H_Safe pilot was supposed to use Gemma Scope SAEs with verified safety annotations. The pilot_summary.md explicitly states "H_Safe pilot used synthetic SAE (Gemma Scope not available)." The safety claim is entirely unsupported by the experimental execution.

### Gap 3: H_Mech Decoder Contribution Cannot Be Isolated

The experiment design does not allow separating encoder training from encoder alignment with the hierarchy. The trained encoder was trained ON the specific hierarchy, so it inherently aligns with it. You cannot "turn off" this alignment to measure decoder contribution independently.

---

## 4. Severity Classification

| Issue | Severity | Action Required |
|-------|----------|-----------------|
| Deterministic absorption (n=3 seeds, std=0) makes H1 statistical claims tautological | Fatal flaw | Add variance to synthetic hierarchy; run proper multi-seed with different hierarchies |
| H_Mech factorial uses same trained encoder for all conditions | Fatal flaw | Redesign to properly isolate decoder contribution |
| H_Safe uses arbitrary feature indices, not validated safety annotations | Serious concern | Either install Gemma Scope or drop H_Safe |
| H3 steering sensitivity conflates magnitude with absorption effect | Serious concern | Normalize by baseline activation; report relative sensitivity |
| 90.7% absorption saturation in H_Safe suggests ceiling effect | Serious concern | Investigate whether proportional ablation saturates at high absorption |
| Trained SAE exactly 0.5 absorption is likely a mathematical artifact | Minor caveat | Frame as such if confirmed by hierarchy-varied experiment |

---

## 5. Concrete Remediation

### Remediation 1: Add Variance via Hierarchy Randomization (addresses Risk 1)

**Required experiment**: Generate a NEW random parent-child hierarchy for EACH seed, with varying cosine similarities (±0.1 noise). Current design: fixed hierarchy + multiple seeds (only randomizes data sampling). Required design: random hierarchy + multiple seeds.

**Specific protocol**:
1. Seed 42: Hierarchy with parent-child cos_sim = 0.67 ± noise(0.1)
2. Seed 43: Hierarchy with parent-child cos_sim = 0.67 ± noise(0.1) (different noise)
3. Seed 44: Hierarchy with parent-child cos_sim = 0.67 ± noise(0.1) (different noise)
4. Report: trained_absorption_mean, trained_absorption_std across seeds

**Expected outcome**: If absorption remains at 0.5 with std=0, then 0.5 is a mathematical consequence of L0=32 and the hierarchy geometry constraints, not a learned property. Frame accordingly.

### Remediation 2: Properly Isolate Decoder Contribution (addresses Risk 2)

**Required experiment**:
1. Train a RANDOM encoder with the TRAINED decoder (new condition C')
2. Train a TRAINED encoder with a RANDOM decoder (new condition B')
3. Compare C' vs A (decoder contribution with random encoder)
4. Compare B' vs A (encoder contribution with random decoder)

**Current design flaw**: Condition C reuses the trained encoder, making it equivalent to Condition B.

**Expected outcome**: Proper isolation may reveal decoder contribution. Current comparison is circular.

### Remediation 3: Validate Safety Annotations (addresses Risk 3)

**Option A**: Install Gemma Scope SAEs and use Neuronpedia-verified safety annotations.
**Option B**: Drop H_Safe as unverifiable with current resources. Document limitation explicitly.

**If proceeding with Option B**: Acknowledge in paper that safety analysis requires validated feature annotations not yet available. Focus on the H1/H3 contributions which are supported by experiments.

### Remediation 4: Normalize Steering Sensitivity (addresses Risk 4)

**Required analysis**: Report RELATIVE sensitivity = (steered - baseline) / baseline, not absolute sensitivity.

**Expected outcome**: If absorbed features remain more sensitive after magnitude normalization, the causal claim is supported. If not, sensitivity difference is an artifact of feature strength.

---

## 6. Comparison to Prior Iteration

| Prior Concern | Status | Gap |
|---------------|--------|-----|
| Zero variance in trained SAE absorption | NOT RESOLVED | Multi-seed added but hierarchy still fixed |
| H3 steering broken (NaN stats) | RESOLVED | Steering implementation now works |
| Shuffled/permuted baselines match trained SAE | PARTIALLY ADDRESSED | H_Mech attempts explanation but design is confounded |
| MCC ~0.21 (chance-level recovery) | NOT RE-MEASURED | No evidence that absorption reduction reflects genuine feature recovery |
| Probe quality confound | NOT ADDRESSED | No probe quality gating in any current experiment |

---

## 7. Final Assessment

**Supported claims**:
- H1: Trained SAEs differentiate from random decoders on synthetic absorption task (d=8.94 is robust)
- H3: Steering implementation is functional (verified by non-zero delta)
- H_Safe: Null result (p=0.665) but methodology unvalidated

**Unsupported/incomplete claims**:
- H_Mech interpretation as "encoder-driven, not decoder geometry": design is circular
- Deterministic 0.5 absorption: likely mathematical artifact, not learned property
- H_Safe safety implications: arbitrary feature labels, not validated annotations

**Critical path to publication**:
1. Fix multi-seed design to include hierarchy randomization (adds variance, tests mathematical artifact hypothesis)
2. Redesign H_Mech to properly isolate decoder contribution
3. Either install Gemma Scope or explicitly drop H_Safe as unverifiable
4. Normalize H3 sensitivity by activation magnitude

The pilot experiments represent genuine progress but do not yet support the paper's strongest claims. The H_Mech finding (encoder-driven absorption) is the most novel but rests on the most questionable experimental design.

---

*Analysis conducted by skeptic agent following Evidence Contract. All citations reference workspace files in iter_001/exp/results/new_pilots/ and iter_001/exp/results/full/.*
