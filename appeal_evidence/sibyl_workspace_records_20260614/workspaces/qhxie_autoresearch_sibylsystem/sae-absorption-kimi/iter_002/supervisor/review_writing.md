# Supervisor Review: Construct Validity of SAEBench Feature Absorption Metric

**Overall Score: 5.5 / 10** (Borderline Reject)
**Verdict: CONTINUE** (requires substantial revision)

---

## Executive Summary

This paper proposes the first construct-validity study of the SAEBench feature absorption metric, asking whether first-letter absorption scores predict semantic-hierarchy absorption. The question is timely and important---the absorption metric shapes architectural development across the SAE field, and its validity beyond first-letter tasks has never been tested. However, the execution suffers from critical methodological flaws that undermine its central claims.

The strongest result is the hierarchy specificity failure (H2): non-hierarchy correlated features produce significantly higher absorption scores than semantic hierarchies (paired t-test: t = -4.748, p = 0.0032). This is a genuine negative result worth reporting. But the paper's framing of the construct-validity question (H1) and the Random-SAE control contains logical errors that a rigorous reviewer would flag as fatal.

The score of 5.5 reflects: a valuable research question (novelty: 7) undermined by fundamental methodological issues (soundness: 5), an underpowered and confounded experimental design (experiments: 5), and incomplete reproducibility documentation (reproducibility: 5).

---

## Dimension Scores

### 1. Novelty & Significance: 7/10

**Strengths:**
- The core question---does first-letter absorption generalize to semantic hierarchies?---is genuinely novel. No prior work has systematically tested this.
- The stakes are high: if the metric lacks construct validity, a large body of follow-up work (Matryoshka SAEs, OrtSAE, HSAE) may have optimized for the wrong target.
- The negative results (hierarchy specificity failure) are valuable and actionable for the community.

**Weaknesses:**
- Korznikov et al. (2026) already showed that random SAE baselines match trained SAEs on standard metrics, which partially anticipates the Random-SAE finding. The paper does not engage with this prior work in the main text.
- The contribution is narrower than claimed: because the absorption formula collapses to k-sparse loss on semantic tasks (see Critical Issue #2), the paper is really testing "does first-letter k-sparse loss predict semantic-hierarchy k-sparse loss" rather than validating the full absorption metric.

### 2. Technical Soundness: 5/10

**Critical Issues:**

**Issue 1: Random-SAE Contradiction (CRITICAL)**
The paper contains an internal contradiction about the Random-SAE construction:
- Section 3.1 / Table 1: "Permuted encoder directions from Standard" (also: "The Random-SAE control permutes the encoder matrix W_enc")
- Section 4.5: "the Random SAE only permutes the decoder directions, leaving encoder activations unchanged"

These are mutually exclusive. The raw data resolves the ambiguity: Standard and Random have bit-for-bit identical absorption scores across all hierarchies (building=0.5333, container=0.5000, tool=0.2333, etc.). This exact identity is mathematically guaranteed if the Random SAE permutes decoder directions (encoder outputs unchanged, so absorption formula produces identical results). If the Random SAE permuted encoder directions, identical scores would be inexplicable.

The paper does not acknowledge this contradiction. Instead, it presents the identical scores as a surprising empirical finding ("degeneracy") when they are in fact expected behavior.

**Issue 2: Perfect Probe Ceiling Effect (CRITICAL)**
For all semantic hierarchies and all SAEs, resid_acc = sae_acc = 1.0. This means the absorption formula:

```
A_full = max(0, (resid_acc - sae_acc)/resid_acc, (resid_acc - k_sparse_acc)/resid_acc)
```

simplifies to:

```
A_full = 1 - k_sparse_acc
```

The metric is no longer measuring "how much parent information is lost in the SAE encoding" (the theoretical construct) but rather "how well can a logistic probe classify from the top-10 latents." The SAE encoding itself (sae_acc) never degrades performance---it always matches the residual stream. This is acknowledged in Section 3.4 but not grappled with in the conclusions.

This collapse has profound implications: the paper cannot claim to be validating the SAEBench absorption metric on semantic tasks, because the metric's first term (resid_acc - sae_acc) is always zero. It is validating k-sparse probing, not absorption.

**Issue 3: Overclaim About Random-SAE "Degeneracy" (MAJOR)**
The paper claims the Random-SAE result "indicates that the semantic-hierarchy adaptation of the SAEBench absorption metric does not measure learned SAE structure" and calls it "confounded by geometric artifacts unrelated to learned SAE structure." But if the Random SAE permutes decoder directions (as the data confirms), identical encoder outputs produce identical absorption scores by construction. This is expected behavior, not a degeneracy. The paper conflates "decoder permutation does not change encoder outputs" with "the metric is degenerate."

The correct interpretation is: the metric depends on encoder geometry, not decoder geometry. This is actually a positive result---it shows the metric is sensitive to the encoder's learned structure. If the paper wants to claim metric degeneracy, it needs an encoder-permuted Random SAE.

### 3. Experimental Rigor: 5/10

**Issue 4: Severely Underpowered Sample (MAJOR)**
The primary correlation (H1) uses n=7 trained SAEs. The bootstrap 95% CI spans 1.37 correlation units: [-0.389, 0.981]. This interval is so wide that it includes essentially every plausible correlation value. The paper acknowledges this but still draws substantive conclusions from an "inconclusive" result. The distinction matters: "inconclusive" implies the test was adequate but the data were ambiguous; "insufficient" implies the test itself was inadequate. The latter is more accurate.

The pre-registered threshold (r > 0.6) was chosen without power analysis. With n=7, the power to detect r=0.6 at alpha=0.05 is approximately 0.35---the test was underpowered by design.

**Issue 5: Hierarchy Specificity Confounded by Task Structure (MAJOR)**
The hierarchy specificity test compares multi-class hierarchies (parent vs. 2-3 children) to binary non-hierarchy pairs (word A vs. word B). These are not structurally equivalent. Binary tasks with similar template structures may be harder for k-sparse probes than multi-class tasks, inflating apparent "absorption." The paper does not address this confound.

Additionally, some hierarchy choices are questionable. "animal" with children "pet" and "male" is not a clean hypernym relationship in WordNet. "Male" is not a type of animal in the same sense as "pet" is.

**Issue 6: GPT-2 Ceiling Effect (MAJOR)**
The GPT-2 replication shows near-zero scores, which the paper interprets as "model-specific behavior." But the raw data reveals that GPT-2 k_sparse_acc values are near 1.0 for almost all hierarchies (Standard SAE: all k_sparse_acc = 1.0), producing a ceiling effect. The probe task is too easy on GPT-2, compressing the dynamic range. This is a methodological limitation, not a model difference.

**Issue 7: Selective Reporting (MAJOR)**
The paper does not report the first-letter vs. non-hierarchy correlation (r = 0.218, CI: [-0.915, 0.852]). This is a missed opportunity: if first-letter absorption also fails to correlate with non-hierarchy absorption, it would strengthen the claim that the two tasks measure different phenomena.

**Issue 8: No Multiple Comparison Correction (MAJOR)**
The study performs ~9 statistical tests without correction. At alpha=0.05, ~0.45 false positives are expected. The hierarchy specificity p-value (0.0032) survives Bonferroni correction (0.05/9 = 0.0056), but this is not reported. Reporting corrected p-values would actually strengthen the claim.

**Issue 9: No k-Sensitivity Analysis (MAJOR)**
The absorption formula depends on the arbitrary choice of k=10 for k-sparse probing. The paper does not test whether different k values change the hierarchy specificity result. This is a key sensitivity check that is missing.

### 4. Reproducibility: 5/10

**Strengths:**
- Raw result files are available in JSON format with per-hierarchy, per-pair breakdowns.
- The SAEBench evaluator configuration is documented.
- The statistical analysis code produces bootstrap CIs.

**Weaknesses:**
- Random seeds for probe training are not reported.
- Exact sentence templates are not provided.
- No code repository URL is mentioned.
- The Random-SAE construction is described inconsistently, making it impossible to reproduce the control condition without inspecting the source code.

---

## What Would Raise the Score

**To 6.5-7.0 (Weak Accept to Borderline Accept):**
1. Resolve the Random-SAE contradiction by verifying the actual implementation and reframing the finding as encoder-geometry dependence rather than metric degeneracy.
2. Acknowledge the perfect-probe ceiling effect explicitly and reframe the contribution as "k-sparse probing on semantic hierarchies" rather than "absorption validation."
3. Report the first-letter vs. non-hierarchy correlation.
4. Apply multiple comparison correction and report corrected p-values.

**To 7.5-8.0 (Borderline Accept to Accept):**
5. Add a sensitivity analysis for k (k=5, 10, 20).
6. Add more SAEs (n>=12) or use a non-parametric approach that doesn't rely on correlation.
7. Fix the hierarchy specificity confound by using structurally equivalent controls (binary hierarchies or multi-class non-hierarchies).
8. Add a true encoder-permuted Random SAE control.
9. Address the questionable hierarchy choices (e.g., "animal: pet, male").

---

## Risks

1. **Desk rejection risk:** The Random-SAE internal contradiction, if unresolved, could lead to desk rejection at a rigorous venue. Reviewers will spot this immediately.
2. **Contribution scope risk:** The perfect-probe ceiling effect means the semantic-hierarchy adaptation cannot measure true SAE encoding loss. The paper's contribution is narrower than claimed---it is about k-sparse probing, not absorption validation.
3. **Power risk:** The small sample size (n=7) makes the H1 result uninformative. Reviewers will demand more SAEs or a different analysis.
4. **GPT-2 interpretation risk:** The ceiling effect undermines the "model-specific behavior" interpretation. Reviewers may see this as a methodological flaw.

---

## Bottom Line

This paper addresses a genuinely important and timely question. The hierarchy specificity failure (H2) is a solid negative result that the community should know about. However, the paper's central claims about construct validity (H1) and the Random-SAE control are undermined by logical errors and methodological flaws. The perfect-probe ceiling effect means the paper is not actually testing absorption---it is testing k-sparse probing. The Random-SAE "degeneracy" is expected behavior from decoder permutation, not a surprising finding.

With substantial revision---resolving the Random-SAE contradiction, reframing the contribution around k-sparse probing, adding sensitivity analyses, and fixing the hierarchy specificity confound---this could become a 6.5-7.0 paper. To reach 7.5+, it would need more SAEs, better controls, and a true encoder-permuted Random SAE. In its current form, it is a 5.5: valuable idea, flawed execution.
