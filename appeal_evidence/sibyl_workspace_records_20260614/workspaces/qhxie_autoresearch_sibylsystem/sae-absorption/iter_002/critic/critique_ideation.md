# Ideation Critique — SAE Absorption Paper

## Originality Assessment

### Proposition 1 (Rate-Distortion Threshold)

**Verdict: Valuable but overstated as a primary contribution.**

The derivation is correct and the co-occurrence cancellation corollary is genuinely insightful. However, the proof technique — comparing the expected loss of two candidate solutions — is the simplest possible energy argument and was essentially already present in Chanin et al.'s informal statement ("absorption saves one L0"). Tang et al. (arXiv:2512.05534) prove that absorption solutions are spurious minima of the SAE biconvex loss without deriving the threshold, so Proposition 1 does add the quantitative geometric condition. But the novelty of the corollary (p_co cancels) depends on no prior work having noticed it. The paper should verify this by checking whether the Tang et al. energy analysis implies the same threshold implicitly.

**What Proposition 1 does NOT provide (and must not claim):**
- Proof that gradient descent reaches S2 (requires convergence analysis)
- Proof that S2 is a local minimum (requires Hessian analysis)
- Explanation of why absorbed features show EDA (that is Proposition 2, which is a conjecture)

**Risk:** If a reviewer identifies that Tang et al. 2512.05534 already implies this threshold (even informally), the paper loses this contribution. The paper should add a sentence: "Unlike Tang et al., who prove S2 is a spurious minimum without quantifying when it is preferred, Proposition 1 derives the exact geometric condition under which S2 achieves lower expected loss than S1."

---

### EDA as a Probe-Free Detector

**Verdict: Genuinely novel empirical finding, but the theoretical motivation is post-hoc.**

EDA was discovered empirically in the pilot (AUROC=0.681 from pilot A) after the originally proposed ASI metric failed (AUROC=0.476). Proposition 2 was then constructed as a mechanistic explanation for the observed EDA signal. This is valid science — empirical discovery followed by theoretical account — but the paper's narrative presents Proposition 2 as a prediction that motivates the experiment, when the actual sequence was reversed.

**Specific novelty concern:** The amortization gap (O'Neill et al. 2024) already documents that SAE encoders and decoders are systematically misaligned, with the encoder underestimating activations due to L1 bias. EDA measures the same directional misalignment that amortization gap theory predicts will be present in all L1-penalized SAEs, not just absorbed ones. The paper acknowledges this confound but the claim "EDA identifies absorbed features" requires evidence that absorbed features have *more* EDA than the amortization gap baseline. The paper does not directly compute the amortization gap for the 18 exact-label positives vs. negatives.

**Suggestion:** Compute the amortization gap metric from O'Neill et al. for the 18 exact-label positives and the non-absorbed background. If EDA and amortization gap are correlated, check whether the EDA AUROC for absorption persists after regressing out amortization gap.

---

### Cross-Directional Metric cos(ê_p, d_c)

**Verdict: The strongest empirical finding in the paper, genuinely novel, but underexplored.**

AUROC = 0.730 with z=6.38 above null is the paper's most robust result. It is not predicted by EDA theory (which is an intra-feature metric), and the paper correctly presents it as "an empirical discovery not anticipated by the original EDA theory." This is a significant and properly claimed novelty.

**Underexplored:**
1. The metric requires knowing which features are parent candidates. The paper uses 43 letter features as the parent pool. For deployed SAEs where parents are unknown, a scalable version would need to operate over all feature pairs — an O(d_sae^2) sweep or an approximate nearest-neighbor search. The paper does not describe how practitioners would use this metric without a known parent set.
2. The analogous metric cos(ê_c, d_p) achieves AUROC = 0.681 (also significant). The paper reports both but does not discuss why one is stronger, which would illuminate the absorption geometry further.
3. The metric is validated only on the first-letter hierarchy at GPT-2 Small Layer 6. Whether it generalizes to semantic hierarchies or larger models is entirely unknown.

---

### Phase Stability and Hysteresis

**Verdict: The null result is well-executed but the hysteresis concept was inappropriate for this regime.**

The finding that absorption rates are uniformly high (0.876–0.978) across all configurations is a valuable characterization result — it tells practitioners that adjusting L0 does not help and only architectural intervention does. However, the hysteresis test was never well-suited to this regime: you cannot test whether a transition is reversible if you cannot find a non-absorbed state to reverse to. The pre-registration of H4 (phase transition with potential hysteresis) was overly ambitious given that the sparsity range accessible via pre-trained SAEs does not include a non-absorbed stable state.

**Verdict on the "Absorption Impossibility Theorem" from the original proposal:** The h* = O(1/sqrt(lambda)) claim has been correctly downgraded to a Proposition 2 Depth Bound in F1_theory_analysis.md, with the note that the scaling requires an assumed decoder angle model. This should be stated explicitly in the paper — the impossibility theorem from the proposal is not proven.

---

## Missed Opportunities

1. **EDA on known-absorbed features from SAEBench**: SAEBench (karvonen2025saebench) evaluates 200+ SAEs on the Chanin absorption metric. If the paper had computed EDA on any of these SAEs and correlated it with the SAEBench absorption score, this would provide much stronger validation of EDA as a cross-SAE predictor. The current validation is limited to one SAE (gpt2-small-res-jb L6).

2. **Neuronpedia annotations for the 18 positives**: The paper could have checked what Neuronpedia labels say for the 18 exact-label absorbed features — are they described as letter features? polysemantic features? This would test whether the 18 positives are coherent and strengthen the ground truth.

3. **Why only 8 letters in C2?**: The cross-domain experiment uses only 8 letters (a, g, h, i, j, l, m, q) out of 26. The selection criterion is not stated. 7 of these 8 show absorption_rate = 0; only 'h' shows 0.067. With a different letter selection, the first-letter result might look very different. The paper should state why these 8 letters were selected and whether the same letters were used in the original Chanin et al. validation.
