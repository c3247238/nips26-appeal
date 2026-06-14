# Experiment Critique

## Executive Summary

The experimental execution is thorough in scope (11 completed tasks, all components delivered) but suffers from three structural problems that undermine the paper's strongest claims: (1) a width confound in the downstream correlation analysis (H3), (2) an invalid parent-identification procedure in the taxonomy, and (3) a model mismatch between detection (GPT-2) and downstream (Gemma) experiments that prevents composing results into a single narrative.

---

## H1: LV Detector -- Well-Executed Negative Result

**Strengths:**
- Clean calibration/test split (13/13 letters) with multiple threshold values
- Appropriate cosine-similarity baseline for comparison
- Sharpness test (sigmoid vs. linear AIC) is a rigorous falsification strategy
- Candid reporting: test F1 = 0.128, ROC-AUC = 0.148

**Weaknesses:**
- The ROC-AUC = 0.148 (below 0.5) means the detector anti-predicts absorption. This is more informative than F1 and deserves prominent discussion. The first alpha_ij bin [0, 0.1] has absorption rate 0.848, which means the quantity is inversely related to absorption at low values -- a fundamental problem with the formulation, not just a threshold issue.
- The decoder cosine pre-filter (cos > 0.15) imposes a hard recall ceiling of 34%, meaning 66% of absorbed pairs are excluded before classification begins. The F1 = 0.128 is partly a pre-filter artifact, not purely an alpha_ij failure. The oracle F1 within the filtered set is never reported.
- Cross-architecture validation conflates hook-point change (resid_pre vs resid_post) with architecture change. These are different activation spaces; F1 = 0.0 on v5-128k could be caused by the hook-point mismatch alone.

**Verdict:** The negative result is genuine and well-supported. The LV competitive exclusion analogy does not work as a binary detector. The paper should more clearly distinguish between pre-filter failure (coverage issue) and alpha_ij failure (discrimination issue).

---

## H2: PMI Regression -- Clean Null Result

**Strengths:**
- Large sample (806 observations, 31 configurations, 26 letters)
- HC3 robust standard errors
- Clear separation of partial R^2 for PMI term (0.0006)
- Per-layer ablation confirms instability

**Weaknesses:**
- No clustering of standard errors at the letter level. The 806 observations are 31 configs x 26 letters, with letters repeated across configs. This violates the independence assumption. Layer coefficient (p < 0.001) might survive clustering; L0 coefficient (p = 0.012) might not.
- The Durbin-Watson statistic of 1.335 indicates positive autocorrelation in residuals, suggesting model misspecification. The Jarque-Bera test (p = 0.00) confirms non-normality. The skewness of 5.186 means the distribution is heavily right-tailed -- most absorption rates are near zero with a long right tail. OLS on this data is questionable; a beta regression or zero-inflated model would be more appropriate.
- PMI aggregation uses "median_top10" but the choice of this aggregation (vs. mean, max, or per-token PMI) is not justified. Different aggregations could yield different conclusions.

**Verdict:** The null result is robust. PMI does not predict absorption. The dominant predictors (layer, L0) are consistent with an optimization-driven account.

---

## H3: Downstream Correlation -- Strong But Confounded

**Critical concern: Width confound in matched comparison.**
The 5 high-absorption SAEs (absorption 0.81-0.90) are ALL 1M-width with L0 16-58. The 5 low-absorption SAEs (absorption 0.003-0.038) are ALL 16k/65k with L0 137-297. This is not a matched comparison -- it is a width comparison disguised as an absorption comparison. The Cohen's d = 2.13 reflects the width difference, not an absorption effect.

**Partial correlation caveat.**
The partial correlations control for log(width), layer, and arch_class. However:
- log(width) is continuous, treating 16k and 1M as differing by a factor. The actual difference is 64x, and 1M SAEs may be qualitatively different (more dead features, different optimization dynamics). A linear partial may not remove this.
- The SCR partial r jumps from -0.431 to -0.677 -- a suppressor effect. Which covariate produces this? If it is width, the causal interpretation is further undermined.
- At n=54 (n=49 for SCR, n=40 for unlearning), confidence intervals are wide. The 95% CI for sparse probing r is [-0.744, -0.389], spanning a 36 percentage-point range.

**Positive aspects:**
- Real SAEBench data (not simulated)
- Bonferroni correction applied appropriately
- Pre-registered threshold (|r| > 0.3) was exceeded
- Three of four tasks significant
- The falsification of H3 (authors predicted |r| < 0.2, found |r| > 0.4) is honest and publishable

**Verdict:** The correlation is real but the causal interpretation is fragile. Within-width-tier stratified analysis is essential before claiming absorption predicts downstream quality. The matched comparison is invalid as presented.

---

## Taxonomy: Exploratory but Misleading

**Critical concern: Type II identification is invalid.**
The taxonomy experiment's own evidence_quality.suspicious_flags state:
> "CRITICAL: Type II rate of 88.5% (23/26 letters) is likely inflated. The parent features identified by our selectivity heuristic are NOT the sae-spelling ground truth first-letter parent features."

The n_comparison_tokens = 0 for most letters means the "expected magnitude" baseline is a global fallback, not a letter-specific comparison. The magnitude_ratio < 0.5 threshold triggers systematically because the identified "parent" features fire for reasons unrelated to the first-letter function.

**Consequence:** Only Type I (1/26 = 3.8%) and Type III (0/26 = 0.0%) are validated. The 92.3% headline is an artifact. The Chanin any-absorption rate of 80.8% (21/26 letters) -- computed with proper ground truth -- is the most defensible comprehensive figure.

**Positive aspects:**
- The taxonomy concept (Type I/II/III) is well-defined and potentially useful
- Cross-width analysis (24k/49k/98k) shows stability
- The paper is honest about the limitation (Section 7.4, Table 1 caveat)

**Verdict:** The taxonomy is a valid conceptual contribution but the 92.3% rate should not be presented as a finding. It should be presented as an upper bound with strong caveats. The abstract must be revised.

---

## H4: Width Paradox -- Inconclusive

- DAS(k=3) does NOT increase monotonically: only 42.3% of letters show positive slope (target was 80%)
- DAS(k=1) is non-monotone but meets the 60% target (57.7% non-positive)
- High per-letter variance (letter N: DAS(k=1) drops from 0.571 to 0.100; letter X jumps from 0.0 to 1.0)
- The small word-activation sample (n=40 per letter, not the 10,000 claimed in methodology) raises concerns about DAS reliability

**Verdict:** H4 is not supported. The width paradox remains unexplained. This is a minor negative result that does not add much to the paper.

---

## Safety Probe: Should Be Relegated to Appendix

- n=3 SAEs, n=100 prompts, 3+ confounds (layer, width, architecture, hook)
- Probe gap does NOT increase with absorption (0.118, 0.148, 0.051)
- Dense probe saturates at AUC ~1.0, providing no discriminative signal
- Zero statistical power for the intended hypothesis

**Verdict:** This experiment should not appear as a main result. Move to appendix as "pilot attempt."
