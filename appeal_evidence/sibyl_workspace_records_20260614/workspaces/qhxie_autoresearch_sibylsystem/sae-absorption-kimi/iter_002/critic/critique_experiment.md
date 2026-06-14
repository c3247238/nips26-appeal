# Experiment Critique: Construct Validity of SAEBench Absorption Metric

## Critical Issues

### 1. Random-SAE Control: Internal Contradiction and Possible Bug (CRITICAL)

The paper contains a direct contradiction about how the Random-SAE control is constructed:

- **Section 3.1** (Method): "The Random-SAE control permutes the **encoder matrix W_enc** of the Standard SAE"
- **Section 4.5** (Results): "the Random SAE **only permutes the decoder directions**, leaving encoder activations unchanged"

These are mutually exclusive claims. The raw data resolves the ambiguity: Standard and Random SAEs have **bit-for-bit identical per-hierarchy absorption scores** across all 10 hierarchies. This is only possible if the encoder outputs are identical, which occurs when decoder directions are permuted (not encoder directions).

**If Section 3.1 is correct** (encoder permutation), then identical scores are inexplicable and suggest a bug in the randomization code. Encoder permutation would change the latent activations z = ReLU(x @ W_enc + b_enc), producing different absorption scores.

**If Section 4.5 is correct** (decoder permutation), then identical scores are expected by construction. The absorption formula depends on encoder outputs (z), not decoder outputs (z @ W_dec). Permuting W_dec does not change z, so absorption scores remain identical.

**The paper interprets identical scores as evidence of metric "degeneracy"**---but this interpretation is only valid if the Random SAE was supposed to change the metric. If decoder permutation was the intended control, identical scores are the expected outcome, not a degeneracy.

**Recommendation:** Verify the actual implementation. If decoder permutation was used, reframe the Random-SAE finding as showing the metric depends on encoder geometry (which is actually a positive result for metric validity). If encoder permutation was intended but decoder permutation was accidentally used, rerun the experiment with correct encoder permutation.

---

### 2. Perfect Probe Accuracy Collapses the Absorption Metric (CRITICAL)

For ALL semantic hierarchies and ALL SAEs, the raw data shows:
- resid_acc = 1.0 (perfect base-model classification)
- sae_acc = 1.0 (perfect SAE-latent classification)
- k_sparse_acc varies (0.466 to 0.975)

This means the absorption formula:
```
A_full = max(0, (resid_acc - sae_acc)/resid_acc, (resid_acc - k_sparse_acc)/resid_acc)
```
always simplifies to:
```
A_full = 1.0 - k_sparse_acc
```

The metric is measuring **exclusively** k-sparse probing degradation. It captures zero information about SAE encoding quality because sae_acc never drops below resid_acc. This is a fundamental design flaw in the semantic-hierarchy adaptation.

**Why this matters:** The original SAEBench first-letter metric was designed so that sae_acc < resid_acc in many cases, capturing genuine information loss from the SAE encoding. The semantic-hierarchy adaptation never exhibits this property, making it a different metric entirely.

**Recommendation:** Acknowledge this explicitly. The semantic-hierarchy "absorption" is actually "k-sparse probing difficulty," not absorption in the Chanin/SAEBench sense. Consider redesigning with harder hierarchies where sae_acc < resid_acc, or use a different metric formulation.

---

### 3. Severely Underpowered Sample Size (MAJOR)

The primary construct-validity test uses n=7 trained SAEs. The bootstrap 95% CI spans 1.37 correlation units ([-0.389, 0.981]). For context:
- A CI width of 0.6 (still very wide) would require ~15-20 SAEs
- The current CI includes "strong negative correlation" through "near-perfect positive correlation"
- No meaningful statistical inference is possible with this precision

The paper acknowledges the limitation but still frames the result as "inconclusive" rather than "uninformative." The distinction matters: "inconclusive" implies the test had adequate power but the data were ambiguous; "uninformative" implies the test was inadequately powered from the start.

**Recommendation:** Reframe as "the evidence base is insufficient to evaluate construct validity." Do not present r=0.463 as a meaningful point estimate when the CI spans the entire possible range.

---

### 4. Hierarchy Specificity Test is Structurally Confounded (MAJOR)

The hierarchy specificity test compares:
- **Hierarchy condition:** Multi-class (parent vs. 2-3 children), e.g., "building" vs. {house, school, library}
- **Non-hierarchy condition:** Binary (word A vs. word B), e.g., "big" vs. "large"

These are not structurally equivalent tasks. Binary classification with similar template structures may be intrinsically harder for k-sparse probes than multi-class classification, independent of hierarchy structure. The observed difference (non-hierarchy > hierarchy) may reflect task difficulty, not metric invalidity.

Additionally, non-hierarchy pairs like "big--large" and "stone--rock" are near-synonyms with highly overlapping distributional patterns. The k-sparse probe may struggle to distinguish them because they activate nearly identical SAE latents---which is actually the correct behavior for a well-trained SAE (similar inputs should activate similar features).

**Recommendation:** Design structurally equivalent controls. For example, use binary parent-child pairs ("animal" vs. "dog") for the hierarchy condition, or use multi-class non-hierarchy controls ("big" vs. "fast" vs. "happy"). Without structural equivalence, the hierarchy specificity test is confounded.

---

### 5. GPT-2 Replication: Ceiling Effect or Model Difference? (MAJOR)

GPT-2 shows near-zero absorption scores (Standard: 0.000, TopK: 0.003) compared to Pythia-160M (Standard: 0.352, TopK: 0.250). The paper interprets this as "model-specific behavior."

However, the raw GPT-2 data (from gpt2_replication_results.json) shows k_sparse_acc values that are all near 1.0, suggesting a ceiling effect: the probe task is too easy on GPT-2, compressing the dynamic range to near-zero. This is not necessarily a model difference in absorption behavior---it may be a probe-task design flaw.

**Recommendation:** Report raw k_sparse_acc values for GPT-2. If they are all near 1.0, acknowledge the ceiling effect. Consider using harder hierarchies or more challenging probe tasks for GPT-2.

---

### 6. Missing Correlation: First-Letter vs. Non-Hierarchy (MAJOR)

The statistical_analysis_summary.json contains a first-letter vs. non-hierarchy correlation (r = 0.218, CI: [-0.915, 0.852]) that is never reported or discussed in the paper. This is selective reporting.

If first-letter absorption also fails to correlate with non-hierarchy absorption, it would strengthen the claim that first-letter absorption does not generalize to any non-first-letter task. The omission weakens the paper's argument.

**Recommendation:** Report this correlation and discuss its implications.

---

### 7. No Multiple Comparison Correction (MAJOR)

The study performs approximately 9 statistical tests:
- 2 Pearson correlations (with/without Random SAE)
- 1 first-letter vs. non-hierarchy correlation (unreported)
- 3 paired t-tests (hierarchy vs. non-hierarchy at 3 tau_fs values)
- 3 tau_fs robustness correlations

At alpha=0.05, ~0.45 false positives are expected. The hierarchy specificity p-value (0.0032) survives Bonferroni correction (0.05/9 = 0.0056), which would actually strengthen the claim. The construct-validity result does not rely on a single p-value, so correction is less critical there.

**Recommendation:** Apply Bonferroni or Benjamini-Hochberg correction and report corrected p-values.

---

### 8. Synthetic Template Artifacts (MAJOR)

The sentence templates ("The {concept} is a place with rich history.") may introduce spurious lexical correlations. For example:
- "building," "house," "school," "library" all appear in similar contexts
- "big" and "large" are synonyms with nearly identical distributional patterns

The k-sparse probe may rely on shallow template-level cues rather than deep semantic hierarchy structure. This would inflate absorption for synonym pairs (which share distributional patterns) relative to hierarchy pairs (which may have more distinct patterns).

**Recommendation:** Test with natural corpus sentences as an ablation. Report whether absorption scores differ between template-generated and natural data.

---

### 9. Fixed k=10 Penalizes Complex Hierarchies (MAJOR)

The k-sparse probe uses a fixed k=10 for all hierarchies, regardless of the number of children:
- "building" has 3 children (house, school, library)
- "device" has 2 children (fan, key)

A parent with more children requires representing more distinctions, potentially needing more latent dimensions. Fixed k systematically penalizes hierarchies with more children.

**Recommendation:** Use adaptive k proportional to hierarchy complexity, or report per-hierarchy k values.

---

## Minor Issues

1. **Random seeds not reported** for probe training (only SAEBench evaluator seed=42 is reported)
2. **Exact sentence templates not provided** in the paper or appendix
3. **No code repository URL** is provided
4. **"Random-SAE" vs. "Random SAE"** hyphenation inconsistency

## What Works Well

1. **Transparent raw data reporting:** All per-hierarchy and per-pair scores are available in JSON files
2. **Bootstrap CIs:** Appropriate for small-n correlation
3. **Pre-registered hypotheses:** Clear falsification criteria
4. **Negative result reporting:** The hierarchy specificity failure is reported honestly
5. **Multiple controls:** Random-SAE, non-hierarchy, and GPT-2 replication provide triangulation
