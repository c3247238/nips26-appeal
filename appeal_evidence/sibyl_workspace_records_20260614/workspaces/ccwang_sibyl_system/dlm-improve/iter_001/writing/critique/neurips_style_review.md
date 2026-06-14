# NeurIPS-Style Review for `dlm-improve`

## Overall Recommendation

- Recommendation: `Weak Reject`
- Score: `4/10`
- Confidence: `3/5`

## Summary

This paper studies whether confidence and entropy signals in masked diffusion language models (DLMs) are actually reliable during denoising, and proposes a simple inference method, CARD, that reallocates compute from extra denoising steps to entropy-guided revision. The paper is clear, well motivated, and has one genuinely interesting angle: the distinction between self-consistency calibration and oracle calibration during the denoising process.

The strongest part of the paper is the diagnostic story. The method itself is simple and sensible, but in my view it is more incremental than decisive. The current empirical support is not yet broad enough for a strong NeurIPS acceptance case.

## Strengths

- The calibration diagnosis is the strongest contribution. The paper distinguishes self-consistency calibration from oracle calibration and uses this to argue that DLM miscalibration is a process artifact rather than an intrinsic model deficiency.
- The compute-normalized baseline is well chosen. The DNB-84 comparison is necessary and materially strengthens the claim that targeted revision is a better use of compute than simply adding more denoising steps.
- The method is simple, easy to understand, and likely easy to reproduce.
- The paper is generally well written and presents a coherent narrative from diagnosis to method design.

## Weaknesses

- The empirical scope is narrow. Meaningful gains are shown mainly on GSM8K, while MMLU shows no meaningful change. The evaluation uses a single model and a single seed.
- The self-consistency calibration protocol uses agreement with the final committed token rather than agreement with ground truth. This is a valid process diagnostic, but it is not the same as standard correctness calibration, and the paper should be more careful in how broadly it frames the resulting conclusion.
- The method novelty is limited. The core recipe, uncertainty-guided remasking followed by short revision, is close in spirit to other recent DLM inference-time correction or remasking methods.
- The paper does not directly compare against stronger concurrent remasking or revision baselines, only against standard denoising, a compute-matched denoising baseline, and random revision.
- The statistical evidence is still weak by conference standards. The paper reports a single-seed binomial significance test, but does not measure seed-to-seed variability.

## Detailed Comments

### 1. Main technical contribution is diagnostic, not algorithmic

The paper's most compelling contribution is the calibration analysis. The method section feels more like a practical consequence of the analysis than a sufficiently strong standalone algorithmic novelty. I would encourage the authors to lean into that framing more explicitly if they revise.

### 2. Calibration definition needs clearer positioning

The self-consistency metric evaluates whether an intermediate token prediction matches the final committed token, not whether it matches the reference answer. This is a useful trajectory-level diagnostic, but it should not be presented too casually as direct evidence of task-level miscalibration without qualification.

### 3. Limited evidence of generality

The method helps on GSM8K, but does not move MMLU. The discussion gives a plausible explanation, but from a NeurIPS review perspective this still leaves the method looking domain-specific. Additional tasks, especially code or structured generation, would help clarify whether the effect is general or mainly a math-reasoning phenomenon.

### 4. Stronger baselines are missing

Since the paper positions itself relative to confidence-based and revision-based DLM inference methods, the lack of direct head-to-head comparison against stronger contemporaneous revision or remasking methods weakens the empirical case.

### 5. Effect-size shrinkage is concerning

The pilot-to-full reduction in gain is substantial. That does not invalidate the result, but it does increase concern that the observed effect may be unstable or smaller than initially suggested.

## Questions for the Authors

- How stable is the GSM8K gain across random seeds?
- Does the effect persist on code generation or other structured generation tasks?
- Can the authors compare directly against stronger concurrent revision or remasking baselines?
- How different are the conclusions if calibration is measured against ground truth rather than final committed tokens?

## Final Verdict

I see real value in the paper, especially in the calibration diagnosis. However, in its current form I do not think the method contribution is sufficiently novel or sufficiently validated for NeurIPS acceptance. I would score this as a weak reject rather than a hard reject because there is a promising and potentially publishable core here, especially if the paper is reframed around the diagnostic contribution and supported by broader evaluation.

---

# Separate Assessment: Originality and Novelty of the Idea

## Bottom-Line Assessment

- Diagnostic idea originality: `7/10`
- Diagnostic idea novelty: `7/10`
- Method idea originality: `4/10`
- Method idea novelty: `3/10`
- Overall paper originality: `5/10`
- Overall paper novelty: `4/10`

## Diagnostic Idea

The paper's more original idea is not CARD itself, but the measurement framing:

- explicitly separating self-consistency calibration from oracle calibration in DLM denoising,
- using that distinction to argue that apparent overconfidence is largely induced by the iterative denoising process,
- and showing that entropy ranking may remain useful even when absolute calibration is poor.

This part feels meaningfully fresher than the method contribution and is the main reason the work still feels research-worthy.

## Method Idea

The method idea is much less original. In essence, it is:

1. run a standard draft,
2. score tokens by entropy,
3. re-mask the highest-entropy positions,
4. run a short revision pass.

That recipe is reasonable and useful, but it sits very close to the existing family of inference-time remasking, revision, fragile-token detection, or lookahead correction methods in the DLM space. The specific instantiation here is cleaner and simpler than some alternatives, but the core idea itself does not feel highly novel.

## Originality Judgment

If I separate the paper into "diagnosis" and "method":

- The diagnosis is fairly original.
- The method is mostly an incremental synthesis or simplification.

So the paper's originality is moderate overall, not low, but the originality mostly comes from the empirical/analytical framing rather than from the proposed decoding algorithm.

## Novelty Judgment

As a NeurIPS submission, the novelty bar is high. Under that bar:

- the calibration analysis is novel enough to be interesting,
- the proposed decoding method alone is probably not novel enough to carry the paper,
- and the paper would be stronger if it positioned CARD more clearly as a diagnostic-derived baseline or consequence rather than as a major standalone decoding advance.

## One-Sentence Takeaway

The paper's strongest novelty is the calibration diagnosis of DLM denoising; the proposed revision method is useful but only modestly original relative to the surrounding literature on confidence-guided remasking and inference-time revision.
