# Writing Critique (Iteration 3 — Post-DTA Full-Scale Results)

**Reviewer**: Critic Agent
**Date**: 2026-03-10
**Scope**: All writing sections (`writing/sections/*.md`, `writing/outline.md`)

---

## Overall Assessment: 4.5/10

The paper is competently written at a sentence level, with clear algorithmic exposition and a well-structured information augmentation spectrum. However, it suffers from a **fatal framing crisis**: the paper was written around a DTA success narrative, but the full-scale data (`full_scale_summary.json`) proves DTA is ineffective. The entire paper needs restructuring before it can be submitted anywhere.

---

## CRITICAL Issues

### W1. Title and Narrative Centering DTA Despite Full-Scale Failure

**Severity**: CRITICAL (grounds for desk rejection)

The title is "Denoising-Time Adaptation: Turning Diffusion Iterations into Test-Time Learning." Full-scale Countdown-500 results:

| Method | Mean Accuracy |
|--------|--------------|
| **DMI** | **9.3%** |
| SCP | 9.1% |
| RCR | 5.7% |
| **DTA** | **4.8%** |
| Vanilla | 4.7% |
| ReMDM-conf | 4.4% |
| **DTA+ReMDM** | **3.6%** |

DTA (4.8%) is statistically indistinguishable from vanilla (4.7%). DTA+ReMDM (3.6%) is *actively harmful*. The paper's title character is a failed method.

The abstract states DTA "significantly outperforms" remasking methods---this is false. The introduction lists DTA as Contribution 1. The conclusion leads with "DTA establishes a novel connection." All of this is misleading given that the only validated contribution is DMI.

**Required action**: Either (a) retitle around the information augmentation spectrum with DMI as the primary validated contribution and DTA as an instructive negative result, or (b) wait for full-scale MBPP/GSM8K data to see if DTA works on any benchmark at scale.

### W2. Experiments Section Contains Stale "Pending" Entries

**Severity**: CRITICAL

Table 1 (Section 4.2) still shows DTA and DTA+ReMDM as "pending." But `full_scale_summary.json` contains completed results for all methods including DTA (4.8%) and DTA+ReMDM (3.6%). Presenting known results as "pending" is deceptive. SCP also has full results (9.1%) but is shown as "~8.4%*" based on an interim estimate.

**Required action**: Update Table 1 immediately with all completed data. Rewrite the surrounding narrative to honestly report that DTA shows zero improvement on Countdown-500.

### W3. Abstract Contains Multiple False Claims

**Severity**: CRITICAL

The Chinese abstract in `proposal.md` and the English abstract in the introduction both claim DTA "significantly outperforms" remasking methods. This is false for the only benchmark with full-scale data. The abstract must lead with DMI as the validated positive finding. DTA should be described as "theoretically motivated but showing task-dependent and limited empirical gains at full scale."

---

## HIGH Issues

### W4. Pilot-Scale Claims Dominate Cross-Benchmark Discussion

The paper repeatedly highlights "DTA +12.5pp on MBPP" (Section 4.3, Section 6.2). This is from N=16 pilot data. The paper's own Lesson 2 (Section 6.4) documents that pilot results "systematically overestimate effect sizes" with "24 percentage point" discrepancies. The paper simultaneously warns about pilot unreliability and builds entire subsections around pilot DTA claims. A reviewer will identify this hypocrisy immediately.

Section 6.2 ("Why DTA excels on code generation") should not exist based on N=16 data. The GSM8K pilot showing ReMDM-conf > DTA is equally unreliable. All cross-benchmark interpretive claims must be explicitly marked as unvalidated hypotheses.

### W5. Introduction's Key Findings List Is Misleading

Finding 3 states: "DTA shows task-dependent effectiveness: DTA achieves its strongest gains on code generation (MBPP pilot: +12.5pp)." This buries the crucial qualifiers ("pilot", N=16) in parentheses while presenting the result as a "key finding" of the paper. Meanwhile, the finding that DTA = vanilla on the only full-scale benchmark is not listed as a key finding at all.

Honest key findings should be:
1. DMI achieves ~2x improvement at near-zero cost (VALIDATED, N=500x3)
2. Pure remasking is insufficient (VALIDATED)
3. DTA shows no improvement on Countdown-500 despite theoretical motivation (VALIDATED)
4. DTA shows preliminary positive signal on MBPP code generation (UNVALIDATED, N=16)

### W6. VDTA Theory-Practice Disconnect Never Addressed

Section 3.2 proves ELBO monotonicity and information accumulation. DTA shows 4.8% on Countdown vs 4.7% vanilla. The paper never confronts this contradiction. If DTA truly improves the ELBO and accumulates information, why does accuracy not improve? The discussion must address this directly. Possible explanations: (a) the ELBO improvement is real but irrelevant to task accuracy; (b) the self-supervised MLM loss is fundamentally misaligned with arithmetic correctness; (c) the regularization conditions (strong convexity) don't hold in practice.

### W7. Related Work Positioning Table Is Misleading

The positioning table (Section 2) claims DTA uniquely satisfies all five desiderata. But a method satisfying all criteria while showing no empirical improvement is not superior to methods that work. The table needs either (a) an "Empirical validation" column, or (b) a footnote acknowledging that DTA's theoretical properties have not translated into accuracy gains on the tested benchmarks.

---

## MEDIUM Issues

### W8. Conclusion Centers DTA Despite Evidence

The conclusion leads with DTA's "novel connection between DLM denoising and test-time training." Given the results, the honest conclusion should lead with: (1) the information augmentation spectrum as a conceptual/analytical framework, (2) DMI as the validated practical contribution, (3) DTA as a theoretically motivated approach whose effectiveness appears limited to code generation tasks (pending full-scale validation). DTA's negative Countdown result is itself a contribution---it reveals that parameter-space adaptation via MLM self-supervision cannot overcome the mismatch between local token co-occurrence and global constraint satisfaction.

### W9. Computational Overhead Inconsistencies

The method section states DTA overhead is "~4x" vanilla. Table 1 in Section 4.2 shows DTA at 23.0s vs vanilla 3.7s = 6.2x. Section 3.1 says "~2-3x standard denoising." The outline says "~4x." SCP is listed as "~7x" in Section 3.3 but "~2x" in the outline. These inconsistencies will erode reviewer trust.

### W10. GSM8K Partial Full-Scale Data Not Discussed

`full_scale_summary.json` contains partial GSM8K results: vanilla 29.6% (1300 done), DTA 29.0% (900 done), DTA+ReMDM 24.5% (400 done), ReMDM-conf 21.3% (1300 done). The paper does not mention these partial full-scale results at all. The GSM8K data shows DTA is at best equivalent to vanilla and ReMDM-conf is actively harmful---consistent with the Countdown pattern. This should be reported.

### W11. Figures Still Missing

The outline defines 8 figures/tables but the paper sections reference them as "<!-- FIGURES" HTML comments. No actual figures exist. A paper with this many quantitative results and zero visualizations is unacceptable for any venue.

---

## Summary: Top 5 Required Actions

1. **CRITICAL**: Update all tables with completed full-scale results. Remove all "pending" entries.
2. **CRITICAL**: Retitle and reframe the paper. DMI is the validated contribution; DTA is the theoretical framework; DTA's negative result on reasoning tasks is itself a finding.
3. **CRITICAL**: Rewrite the abstract to honestly reflect the actual results.
4. **HIGH**: Downgrade all pilot-scale DTA claims to hypothesis-level with explicit unreliability warnings.
5. **HIGH**: Address the VDTA theory-vs-practice disconnect in the discussion.
