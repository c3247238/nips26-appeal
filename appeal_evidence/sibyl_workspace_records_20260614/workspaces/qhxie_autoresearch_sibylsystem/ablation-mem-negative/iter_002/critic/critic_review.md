# Critic Review: The Limits of Unsupervised Absorption Detection in Sparse Autoencoders

**Reviewer:** sibyl-critic (Academic Critic Agent)
**Date:** 2026-04-28
**Paper:** "The Limits of Unsupervised Absorption Detection in Sparse Autoencoders: A Systematic Analysis"
**Status:** Iteration 2, writing phase

---

## Executive Summary

This paper presents a negative result: the Unsupervised Absorption Detector (UAD), which uses co-occurrence clustering to detect feature absorption in Sparse Autoencoders (SAEs), performs no better than random selection. The authors argue this is a conceptual rather than implementational failure---co-occurrence detects correlation, while absorption requires detecting suppression. The paper is well-structured and the negative result is honestly reported, which is commendable. However, there are serious methodological, statistical, and conceptual issues that substantially weaken the paper's claims. The review below identifies these issues across five dimensions.

**Overall Assessment:** The paper has a clear central message and honest reporting, but suffers from insufficient ground truth, questionable experimental execution, overstated theoretical claims, and a mismatch between the proposed methodology and what was actually implemented. Significant revisions are needed before this work is submission-ready.

---

## 1. Technical Correctness

### 1.1 The Ground Truth Problem (CRITICAL)

The entire paper rests on a ground truth of **exactly 2 known absorbed pairs** (feature 18486 paired with each of two child features from the first-letter spelling tasks). This is catastrophically insufficient for any meaningful statistical inference.

- With only 2 true positives, the F1 score is dominated by the false positive rate. A method that detects 541 pairs and gets 2 correct achieves F1 = 0.007---but so would any method that happens to include those 2 pairs in a large enough detection set.
- The "perfect recall" of 1.0 is misleading: it simply means both known pairs were detected somewhere in the 3,702 same-cluster pairs. This is not evidence of detection capability; it is evidence that the known feature (18486) appeared in the top 500 active features and was clustered with other features.
- The paper claims UAD is "indistinguishable from random"---but with n=2 true positives, **no method could achieve statistically significant F1 > 0.01** without near-perfect precision. The experiment is underpowered by design.

**Recommendation:** Either (a) expand the ground truth to at least 20-50 known absorption pairs, or (b) reframe the paper as a pilot study with appropriately scoped claims. The current framing as "systematic empirical analysis at scale" is not supported by the ground truth size.

### 1.2 The "Random Baseline" is Misleadingly Framed

The random baseline selects 541 pairs from 3,702 same-cluster pairs and achieves F1 = 0.0075. The paper frames this as "UAD performs no better than random." However:

- The random baseline is selecting from the **same-cluster pairs**, not from all possible pairs in the 500-feature set (which would be C(500,2) = 124,750 pairs). If the baseline selected from all possible pairs, the expected number of true positives would be 2 * (541 / 124,750) ≈ 0.009, yielding F1 ≈ 0.00003.
- By restricting the random baseline to same-cluster pairs, the baseline inherits the clustering's (weak) signal. UAD is being compared against a baseline that already benefits from the clustering step. This is not a fair null model.
- A proper random baseline should select 541 pairs uniformly from all C(500,2) possible feature pairs.

**Recommendation:** Recompute the random baseline against all possible pairs, not just same-cluster pairs. Alternatively, report both baselines and discuss the difference transparently.

### 1.3 The Suppression Signal Definition is Circular

The paper defines the suppression signal as:

$$\Delta_{\text{supp}}(c, p) = \mathbb{E}[z_c \mid z_p = 0] - \mathbb{E}[z_c \mid z_p > 0]$$

This definition requires knowing which features are parent-child pairs **a priori** to compute the conditional expectations. The paper acknowledges this ("computing it requires knowing which parent-child pairs to test"), but then uses this circular definition to argue that absorption detection is inherently causal. The argument is valid but the definition does not help operationalize detection.

**Recommendation:** Either (a) propose a practical unsupervised estimator for this quantity, or (b) acknowledge more explicitly that this definition is descriptive, not operational, and that the paper's negative result applies specifically to co-occurrence clustering, not to all possible unsupervised methods.

### 1.4 The DFDA Evaluation is Under-Specified

Section 4.8 claims DFDA improves per-pair residual MSE by 21.2%. However:

- The improvement is from 5.2e-6 to 4.1e-6---absolute improvements in the 1e-6 range. The relative improvement looks large because the baseline is already tiny.
- The paper does not report whether this improvement is statistically significant (no confidence interval, no repeated trials).
- The "per-pair" qualifier is important but buried. The improvement is on a single known pair (feature 18486 with letters c, i, o, p, u), not averaged across multiple pairs.
- The DFDA MLP uses 97 parameters and trains on 10,000 tokens for 100 epochs. With such a small model and limited data, the result may not generalize.

**Recommendation:** Report absolute MSE values prominently, add statistical testing (bootstrap CI), and clarify that the 21.2% is from a single pair.

---

## 2. Experimental Rigor

### 2.1 The Experiments Were Not Actually Run as Described (CRITICAL)

A review of the experimental code (`run_iter2.py`) and results reveals a serious discrepancy between what the methodology document promises and what was actually executed:

**Methodology promises (from `methodology.md`):**
- E1: UAD with random baseline (100 random trials, permutation test)
- E2: Ablations (A1-A5: no clustering, no phi, no dead filter, single-link, k-means)
- E3: Cross-layer validation (layers 4, 8, 10 with individual reporting)
- E4: False positive analysis (categorization of FP pairs)
- E5: Statistical testing (bootstrap CI, permutation test with 100 permutations)
- E6: DFDA with parent-positive evaluation
- E7: Alternative correlation metrics (Pearson, MI, Jaccard, PMI)

**What was actually executed:**
- All experiments appear to run the **same UAD code** with identical parameters (top 500 features, 50 clusters, layer 8) and produce **identical results** (F1 = 0.007, precision = 0.37%, 2 true positives, 541 detected pairs, 3,702 same-cluster pairs).
- The DONE marker files for E3, E4, E5, E7, baseline_random, baseline_random_residual, analysis_main, and analysis_dfda all contain the **exact same data** as E1.
- The ablation results file (`e2_uad_ablations_results.json`) only reports same-cluster pair counts for full, k-means, and all-24K variants---no F1, precision, or recall for any ablation.
- The cross-layer experiment (E3) does not report per-layer results; it reports the same layer-8 numbers.
- The false positive analysis (E4) does not categorize false positives; it reports the same aggregate numbers.
- The statistical testing (E5) does not report bootstrap CI or permutation p-values; it reports the same aggregate numbers.
- The correlation metrics experiment (E7) does not compare Pearson, MI, Jaccard, or PMI; it reports the same aggregate numbers.

**This is a fatal methodological flaw.** The paper claims to have run 8 distinct experiments with multiple ablations, baselines, and statistical tests, but the evidence suggests all tasks ran the same code and produced identical outputs. The ablation table (Table 3) reports pair counts that do not appear in the results files. The cross-layer validation (Section 4.5) claims "all layers produce F1 ≈ 0.007" but there is no evidence that layers other than 8 were evaluated.

**Recommendation:** This must be addressed immediately. Either (a) re-run all experiments properly with distinct code paths for each variant, or (b) remove all claims about experiments that were not actually performed. The current state is not acceptable for a research paper.

### 2.2 The Corpus is Artificially Small and Unrepresentative

The code uses only 500 prompts (5 sentences repeated 100 times) for co-occurrence matrix construction. This is not a corpus; it is a toy example. The prompts are:
- "The quick brown fox jumps over the lazy dog."
- "In 1492, Columbus sailed the ocean blue."
- "Machine learning is transforming artificial intelligence."
- "The capital of France is Paris, known for the Eiffel Tower."
- "Photosynthesis converts light energy into chemical energy."

These 5 sentences repeated 100 times will produce highly artificial co-occurrence patterns. The first-letter spelling task features (the ground truth) require tokens starting with specific letters. The probability that these 5 sentences contain tokens starting with c, i, o, p, u in contexts where feature 18486 would activate is vanishingly small.

**Recommendation:** Use a proper corpus sample (e.g., 10,000 tokens from OpenWebText, as stated in the methodology). The current corpus is inadequate for any meaningful analysis.

### 2.3 Single Seed, No Replication

The paper acknowledges this as a limitation (Section 5.6, item 5), but it is more severe than admitted. With only 2 true positives and a stochastic clustering algorithm, the results are highly sensitive to:
- The specific 500 features selected (top by mean activation)
- The random initialization in clustering (though Ward linkage is deterministic, the feature subset selection is not)
- The corpus sample

**Recommendation:** Run at least 5 independent replications with different corpus samples and seeds. Report variance across replications.

### 2.4 The "Collision Detection" Step is Not Defined

Section 3.3 Step 4 mentions "collision detection" where "features in the same cluster with high mutual activation are flagged." However, the code implements this as a simple top-10% co-occurrence threshold within same-cluster pairs. The paper does not justify the 10% threshold, nor does it explore threshold sensitivity. A precision-recall curve (promised in the methodology as Figure 1) is not produced.

**Recommendation:** Add a threshold sensitivity analysis or justify the 10% choice. Produce the promised precision-recall curve.

---

## 3. Logical Argumentation

### 3.1 The Core Argument is Sound but Overstated

The central claim---that co-occurrence clustering detects correlation, not suppression---is conceptually correct and well-argued in Section 5.2. The distinction between:
- Correlation: A and B activate together because they are semantically related
- Suppression: A activates and prevents B from activating

is a genuine and important insight. The argument that suppression reduces co-occurrence (making absorbed features *less* likely to cluster) is particularly sharp.

However, the paper overstates this conceptual point by claiming it "rules out" all unsupervised detection approaches:

> "Our work rules out the third option [unsupervised detection] as a viable path." (Section 2.4)
> "absorption detection is inherently a causal inference task" (Abstract)
> "absorption detection is a causal inference problem, not a clustering problem" (Conclusion)

These are stronger than the evidence supports. The paper only tests **one** unsupervised approach (co-occurrence clustering with Ward linkage). It does not test:
- Intervention-based methods (ablating features and measuring activation changes)
- Residual-based methods (analyzing reconstruction errors)
- Gradient-based methods (using attribution to detect suppressive relationships)
- Causal discovery algorithms (e.g., PC algorithm, NOTEARS) applied to feature activation graphs

**Recommendation:** Soften the claims. State that co-occurrence clustering specifically fails, and that the conceptual distinction suggests other unsupervised approaches would need to model suppression explicitly. Do not claim to have ruled out all unsupervised detection.

### 3.2 The Scaling Analysis is Extrapolation Without Evidence

Table 2 extrapolates precision to the full 24K dictionary as "~0.001%" based on the assumption that the same 2 true positives would be present. This is:
- Unfalsifiable (no experiment was run on the full 24K features with collision detection)
- Misleadingly precise ("~0.001%" implies a measurement; it is a back-of-the-envelope calculation)
- Not informative (the precision collapse is obvious from the O(d^2) growth of same-cluster pairs)

**Recommendation:** Remove the extrapolation or move it to a footnote. The O(d^2) argument in Section 5.1 is sufficient without the pseudo-quantitative table row.

### 3.3 The "Mitigation is Easier Than Detection" Claim is Unsupported

Section 5.5 claims an "asymmetry: mitigation is easier than detection." But DFDA was only tested on a single known pair with a tiny MLP. This is not evidence that mitigation is "easier" in any general sense---it is evidence that a simple model can overfit to a single example. The claim requires:
- Testing DFDA on multiple known pairs
- Showing that DFDA works without knowing which pairs are absorbed ("blind mitigation")
- Comparing the difficulty of detection vs. mitigation in a controlled way

None of this is present.

**Recommendation:** Remove or substantially weaken this claim. State only that DFDA showed improvement on a single known pair when the pair was provided.

---

## 4. Paper Structure and Clarity

### 4.1 Strengths

- The paper has a clear narrative arc: problem → hypothesis → experiments → negative result → conceptual explanation → implications.
- The RQ-contribution mapping in the Introduction is clean and easy to follow.
- Section 5.2 (Correlation vs. Suppression) is the strongest part of the paper---conceptually clear and well-explained.
- The limitations section (5.6) is honest and appropriately scoped.

### 4.2 Weaknesses

**Abstract overclaims:** The abstract states "UAD performs no better than random for absorption detection at scale" and "Our work provides the first systematic empirical analysis of unsupervised absorption detection at scale." Given the ground truth of 2 pairs and the artificial corpus, "at scale" is not justified.

**Table inconsistency:** The E1 table reports 3,702 same-cluster pairs, while the ablation table reports 7,608 for the "same configuration (Ward linkage, 500 features)." The footnote explaining this discrepancy (Section 4.4) is confusing---readers will not understand why two different numbers are presented for the same configuration without clear labels.

**Missing figures:** The methodology promises 6 figures (precision-recall curve, grouped bar chart, cross-layer bars, FP categorization, bootstrap distribution, DFDA recovery chart). None appear in the paper. A paper with only tables and no figures is visually impoverished and harder to follow.

**Citation of Chen et al. [2025] is questionable:** The paper cites "Chen et al., 2025" as proposing co-occurrence clustering for absorption detection. This appears to be a forward citation to unpublished or anticipated work. If this is a real paper, the citation should include a venue or arXiv ID. If it is a placeholder, it must be replaced with a real citation or removed.

**Terminology drift:** The paper uses "absorption" throughout, but the ground truth labels are from "Chanin et al.'s parent-child hierarchy protocol." The relationship between parent-child hierarchies and absorption is assumed but not demonstrated. Are all parent-child pairs absorbed? Are all absorbed pairs parent-child? This is not clarified.

---

## 5. Contribution Clarity

### 5.1 What the Paper Actually Contributes

1. **A negative result on co-occurrence clustering for absorption detection** (on a single model, single layer, with 2 ground-truth pairs). This is a valid but limited contribution.
2. **A conceptual distinction between correlation and suppression** (Section 5.2). This is the strongest contribution---it is general, well-argued, and potentially influential.
3. **A tiny DFDA experiment on one pair** (21.2% MSE improvement, unvalidated). This is a preliminary result at best.

### 5.2 What the Paper Claims to Contribute

The abstract and introduction frame three contributions:
1. "First empirical demonstration that UAD performs no better than random at scale"
2. "Formalize the distinction between correlation and suppression"
3. "Demonstrate that inference-time mitigation (DFDA) remains feasible"

Contribution 1 is overstated (not "at scale"). Contribution 2 is solid. Contribution 3 is overstated (single pair, unvalidated).

### 5.3 Recommendations for Reframing

Consider reframing the paper around the **conceptual insight** (correlation vs. suppression) rather than the **empirical negative result**. The empirical result is too limited to stand alone, but the conceptual insight is general and important. A possible reframing:

> "Why Co-Occurrence Clustering Cannot Detect Feature Absorption: A Conceptual and Empirical Analysis"

This would:
- Lead with the conceptual argument (Section 5.2)
- Use the empirical results as supporting evidence rather than the main contribution
- Reduce the emphasis on "scale" and "systematic"
- Allow the paper to be honest about the limited experimental scope while still making a valuable contribution

---

## 6. Specific Line-by-Line Issues

| Location | Issue | Severity |
|----------|-------|----------|
| Abstract | "at scale" with 2 ground-truth pairs | High |
| Abstract | "first systematic empirical analysis" | High |
| Sec 3.2 | Suppression signal definition is circular | Medium |
| Sec 3.3 | "collision detection" step not formally defined | Medium |
| Sec 4.2 | Random baseline restricted to same-cluster pairs | High |
| Sec 4.3 | Table 2 extrapolation unfalsifiable | Medium |
| Sec 4.4 | Table 3 pair count inconsistency (3,702 vs 7,608) | Medium |
| Sec 4.5 | Cross-layer results not evidenced in data files | Critical |
| Sec 4.7 | p = 0.87 not computed from actual permutation test | Critical |
| Sec 4.8 | 21.2% from single pair, no CI | Medium |
| Sec 5.1 | "rules out" unsupervised detection too strong | High |
| Sec 5.5 | "mitigation is easier than detection" unsupported | Medium |
| Sec 5.6 | Ground truth size limitation understated | High |
| Throughout | Chen et al. [2025] citation validity | Medium |
| Throughout | Missing all 6 promised figures | High |

---

## 7. Actionable Recommendations (Priority-Ordered)

### Critical (Must Fix)

1. **Address the experimental execution gap.** Either re-run all experiments properly (distinct code for each variant) or remove claims about experiments that were not performed. The current state where all experiments produce identical outputs is not acceptable.

2. **Expand the ground truth.** With only 2 true positives, no meaningful statistical inference is possible. Identify at least 10-20 additional known absorption pairs, or reframe the paper as a conceptual analysis with limited empirical support.

3. **Use a real corpus.** Replace the 5-sentence repeated corpus with 10,000+ tokens from OpenWebText or similar. The current corpus is inadequate.

### High Priority

4. **Fix the random baseline.** Compare against all possible pairs, not just same-cluster pairs. Report both baselines transparently.

5. **Soften the "rules out unsupervised detection" claims.** The paper only tests one method. State that co-occurrence clustering fails, not that all unsupervised detection is impossible.

6. **Remove or reframe "at scale" claims.** With 2 ground-truth pairs and 500 features, this is not "at scale."

7. **Add the missing figures.** At minimum, produce the precision-recall curve and the cross-layer comparison.

### Medium Priority

8. **Clarify the Table 3 inconsistency.** Use clear labels distinguishing "same-cluster pairs" from "detected pairs after thresholding."

9. **Validate the Chen et al. [2025] citation.** Replace with a real citation or remove.

10. **Add statistical testing to DFDA.** Bootstrap CI, multiple seeds, multiple pairs.

11. **Clarify the relationship between parent-child hierarchies and absorption.** Are they equivalent? Is one a subset of the other?

---

## 8. Final Verdict

**Score: 5/10** (Borderline reject / Major revision)

**Strengths:**
- Honest reporting of negative results (scientific integrity)
- Clear conceptual distinction between correlation and suppression
- Well-structured narrative with clean RQ-contribution mapping
- Good limitations disclosure

**Weaknesses:**
- Experimental execution does not match methodology claims (critical)
- Ground truth of 2 pairs is insufficient for any statistical inference (critical)
- Corpus is artificially small and unrepresentative (critical)
- Claims are systematically overstated relative to evidence (high)
- Missing all promised figures (high)
- Random baseline is improperly framed (high)

**Path to acceptance:** The paper needs either (a) substantially more rigorous experiments with expanded ground truth and proper execution, or (b) a reframing around the conceptual insight with the empirical results demoted to supporting evidence. The current version is not submission-ready.
