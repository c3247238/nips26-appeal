# Skeptic Analysis: Result Debate

## Statistical Risk Inventory

### Risk 1: The H_Mech "Pass Rate" Is a Moving Target
**Specific concern**: The original pass criteria (B ~ D, |diff| < 0.25) failed on 14/15 runs (6.7% pass rate). The criteria were then revised to "encoder effect > 0.5, decoder effect < 0.1" which yields 100% pass. This is post-hoc criterion adjustment.

**Why it's unreliable**: The original criteria were derived from the pilot (where B ~ D held at L0=32, seed 42). In the full experiment, B vs D diverges significantly at every L0 level and seed (e.g., seed 42, L0=20: B overlap = 0.804, D overlap = 0.467; t = 59.6, p ~ 0). The revised criteria discard the most diagnostic comparison (B ~ D) and replace it with a one-sided threshold that cannot fail. This is not validation -- it is redefinition.

**Evidence**: `h_mech_factorial.json`, aggregate pass_rate_original = 0.067, pass_rate_revised = 1.0. Every single run shows b_vs_d p ~ 0 (highly significant difference), yet the paper reports this as "encoder-driven absorption CONFIRMED."

### Risk 2: The "Perfect" Generalization Is a Ceiling Effect
**Specific concern**: Held-out generalization reports Pearson r = 0.998 across seed means (n=5), with train/test differences < 1% for 4/5 seeds.

**Why it's unreliable**: The generalization test uses the SAME SAE trained on 80% of synthetic data and tested on 20%. The test data is drawn from the exact same generative process. The "perfect" correlation reflects that absorption is a deterministic geometric property of the fixed decoder directions -- not that the SAE has learned a generalizable concept. If you change the hierarchy geometry (e.g., different parent-child cosine), absorption changes (as shown in the hierarchy strength ablation). The generalization test does NOT test generalization to new geometry, only to new samples from the same geometry.

**Evidence**: `heldout_generalization.json` shows seed 45 has parent_child1_cosine = 0.248, parent_child2_cosine = 0.288 -- these are the ACTUAL realized cosines in the synthetic data, not the intended 0.67. The low realized cosine explains the low absorption (0.283). The SAE is not generalizing; it is faithfully reproducing the geometric structure of the specific training distribution.

### Risk 3: The H3 Steering Experiment Has Degenerate Statistics
**Specific concern**: H3 steering reports ratios that flip direction across alpha values (0.43x at alpha=2.0, 1.54x at alpha=0.5), with massive standard deviations (e.g., absorbed std = 1.42 at alpha=2.0 vs mean = 0.22).

**Why it's unreliable**: The coefficient of variation (std/mean) exceeds 6 at alpha=2.0 for absorbed features. This indicates the steering response is dominated by a few extreme outliers, not a systematic effect. The "primary" alpha=2.0 was selected post-hoc (the pilot used different alphas). At alpha=0.5, the ratio is 1.54x (p=0.17); at alpha=2.0, it is 0.43x (p=0.02). This is not a dose-response relationship -- it is noise. The experiment has no pre-registered primary alpha, and any alpha could be cherry-picked.

**Evidence**: `h3_steering.json`, seed 42, random_input_parent_dir_steer: alpha=0.5 ratio=1.54 (p=0.17), alpha=1.0 ratio=1.36 (p=0.34), alpha=2.0 ratio=0.43 (p=0.02), alpha=5.0 ratio=0.93 (p=0.81). The effect direction reverses twice.

---

## Alternative Explanations

### For H_Mech (encoder-driven absorption)
**Alternative**: The encoder effect is not "driving" absorption in a causal sense -- it is merely selecting which pre-existing decoder directions get activated. The random decoder + trained encoder condition (B) shows high absorption overlap (~0.80) because the trained encoder learns to route activation through the decoder directions that happen to have high parent-child geometric overlap. The decoder was never trained to create this overlap; it is a property of the random initialization + the specific synthetic hierarchy geometry (fixed cosine=0.67). If you changed the hierarchy geometry, the "encoder effect" would change because the available decoder directions would have different overlap properties.

**Evidence**: Condition A (random/random) already shows non-zero absorption_cosine (~0.44-0.49) and non-zero absorption_correlation (~0.16-0.28). The random decoder directions are not orthogonal to the hierarchy structure -- they have random projections that, by chance, align with the fixed synthetic geometry. The trained encoder amplifies these chance alignments.

### For Multi-seed Validation (trained > random)
**Alternative**: The "random SAE" baseline uses a randomly initialized encoder AND decoder. But the H_Mech factorial shows that random decoder + trained encoder produces absorption rates comparable to fully trained SAEs. Therefore, the multi-seed comparison (trained vs. random) conflates two different baselines: (1) random encoder + random decoder (very low absorption), and (2) trained encoder + random decoder (very high absorption). The comparison is not "trained vs. untrained" but "encoder-trained vs. nothing-trained."

**Evidence**: `multiseed_validation.json` trained_mean = 0.477, random_mean = 0.033. But `h_mech_factorial.json` shows condition B (trained encoder, random decoder) has overlap ~0.80 -- far above the "trained" mean of 0.477. The multi-seed "trained" SAE uses a different metric (Jaccard) than the factorial (overlap), but the directional pattern is clear: encoder training is sufficient for high absorption.

### For Hierarchy Strength Ablation
**Alternative**: The monotonic increase in absorption with cosine similarity (0.5 -> 0.67 -> 0.8) is a tautology, not a discovery. The absorption metric is DEFINED as the overlap between parent and child activation patterns. If parent and child vectors are more similar (higher cosine), their activation patterns MUST overlap more by construction. The SAE is not "learning" to absorb more strongly -- the metric itself scales with the input similarity.

**Evidence**: `ablation_hierarchy_strength.json` shows absorption means of 0.416, 0.501, 0.544 for cosines 0.5, 0.67, 0.8. But the random baseline (condition A in factorial) also shows absorption_cosine scaling with the fixed geometry. The "effect" may be entirely in the data generation, not the SAE.

### For L0 Sparsity Ablation
**Alternative**: The finding that lower L0 -> higher absorption is interpreted as "fewer active features must represent more concepts." But an equally valid interpretation is that lower L0 increases the variance of the absorption metric because fewer active features means each feature's inclusion/exclusion has larger impact on Jaccard overlap. The higher mean at L0=20 may reflect higher variance + floor effects, not a true mechanistic difference.

**Evidence**: `ablation_l0_sparsity.json` shows L0=20 std = 0.028, L0=50 std = 0.039. The variance is actually LOWER at L0=20, which partially contradicts this alternative. However, the within-seed variance at L0=20 ranges from 0.031 to 0.081, while L0=50 ranges from 0.049 to 0.089 -- the highest-variance seeds at L0=20 (seeds 45, 46) have absorption means of 0.561 and 0.545, pulling the mean up.

---

## Proxy Metric Audit

### Absorption Overlap vs. Absorption Cosine
The factorial experiment reports three metrics (cosine, overlap, correlation) that disagree in magnitude and sometimes in direction. For example, seed 42 L0=20: condition B has absorption_cosine = 0.062 but absorption_overlap = 0.804. The paper focuses on "overlap" for the factorial but "cosine" or "Jaccard" for other experiments. This inconsistency means the "absorption" construct is operationally undefined -- different metrics capture different geometric properties.

**Gap**: What we measure (decoder-direction overlap) is not what we claim (hierarchical feature absorption in the SAE). The metric assumes that if parent and child decoder directions overlap, the parent feature is "absorbed" by children. But this conflates geometric overlap with functional absorption (whether the parent feature is uninterpretable or unusable for steering).

### Safety Feature Matching
The H_Safe experiment matches safety and non-safety features by "mean activation magnitude." But safety features were selected by their differential activation on safety vs. neutral prompts, while non-safety features were selected by overall magnitude. This matching is asymmetric: safety features are selected for a specific functional property, while non-safety features are selected for a generic statistical property. The matching does not control for feature interpretability, sparsity, or layer position.

**Gap**: What we measure (absorption rate on matched features) does not test whether safety features are disproportionately absorbed. It tests whether 20 features selected for one property have the same absorption rate as 20 features selected for a different property, after matching on a third property. This is not a fair comparison.

---

## Severity Classification

| Issue | Severity | Rationale |
|-------|----------|-----------|
| H_Mech post-hoc criterion revision | **Fatal flaw** | The original criteria were the only test that could falsify the hypothesis. Revising them after seeing the data invalidates the confirmation. |
| Absorption metric inconsistency (cosine vs. overlap vs. Jaccard) | **Serious concern** | The core construct is measured differently across experiments, making cross-experiment claims unreliable. |
| Held-out "generalization" tests same geometry | **Serious concern** | The perfect correlation is a ceiling effect, not evidence of generalization. Claims about robustness are overstated. |
| H3 steering alpha cherry-picking | **Serious concern** | No pre-registered primary alpha; effect direction reverses with alpha; high variance. |
| H_Safe asymmetric feature matching | **Minor caveat** | The null result is still informative, but the comparison is not as controlled as claimed. |
| L0 ablation variance confound | **Minor caveat** | The monotonic decrease is real, but the interpretation ("fewer features = more absorption") is one of several plausible explanations. |

---

## Concrete Remediation

### For H_Mech Fatal Flaw
**Experiment**: Re-run the factorial with a PRE-REGISTERED criterion: "The encoder effect must be at least 5x larger than the decoder effect, measured by the ratio of (B-A) to (C-A), averaged across all seeds and L0 levels." This ratio-based criterion is scale-invariant and cannot be gamed by metric choice.

**Expected outcome**: If the geometric hypothesis is wrong and training genuinely drives absorption, the ratio should be < 5x in at least some conditions. If the ratio is consistently > 5x, the encoder-dominance claim is robust.

### For Metric Inconsistency
**Experiment**: Compute all three metrics (cosine, overlap, correlation) for EVERY experiment and report their inter-correlation. If they disagree systematically, the paper must acknowledge that "absorption" is a family of related geometric properties, not a single phenomenon.

**Expected outcome**: A correlation matrix showing r > 0.8 across metrics would justify treating them as equivalent. r < 0.5 would require the paper to commit to one metric and justify it.

### For Generalization Ceiling Effect
**Experiment**: Train SAEs on one hierarchy geometry (cosine=0.5) and test absorption on a DIFFERENT geometry (cosine=0.8) without retraining. Measure whether absorption rates shift toward the test geometry or remain at the training geometry.

**Expected outcome**: If the SAE has genuinely learned hierarchical structure, absorption should be stable across geometries. If absorption is purely a geometric readout of decoder-parent alignment, it will shift to match the test geometry (because the decoder directions are fixed).

### For H3 Steering
**Experiment**: Pre-register alpha=1.0 as the primary analysis (mid-range, no cherry-picking). Report the ratio and p-value at alpha=1.0 for ALL steering directions (parent, child, orthogonal). Use a non-parametric test (Mann-Whitney) due to the extreme variance. Report effect size (rank-biserial) rather than raw ratio.

**Expected outcome**: If absorbed features are genuinely more steerable, the effect should be consistent across alphas and directions. If the effect is noise, it will not replicate at the pre-registered alpha.
