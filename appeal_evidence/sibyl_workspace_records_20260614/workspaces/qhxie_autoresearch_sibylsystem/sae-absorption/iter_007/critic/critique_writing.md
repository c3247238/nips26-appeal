# Writing Critique -- Iteration 9

## Overall Assessment

The paper has improved substantially from iteration 6. The title is now appropriate ("Auditing Feature Absorption Metrics on JumpReLU SAEs"), the abstract is within length limits (237 words), the data integrity issues from partition mismatches are resolved (84/85 claims verified), and the three-finding structure is clean. The writing review scores it 8/10. Negative results reporting remains exemplary and is now even stronger with 8 of 10 hypotheses falsified.

However, the writing has not fully absorbed the implications of the three new experiments. The paper body was updated with results (Tables 4, 5; Section 6.3) but the NARRATIVE framing in several sections still reflects the pre-experiment state. Six issues remain.

---

## Critical Issues

### 1. Section 6 Narrative Mismatch (Severity: Critical)

Section 6 is titled "Exploratory CMI-Absorption Association" -- correct. But Section 6.1 opens with: "The successive refinement theorem predicts that features with low conditional mutual information should be more susceptible to absorption." The word "predicts" implies the theory has empirical traction. The reader expects confirmation. Section 6.3 then delivers the null result (rho=+0.044, p=0.835).

This up-then-down structure wastes reader attention and risks the perception that the authors are burying bad news. In a paper whose strongest asset is honest negative reporting, this narrative structure is inconsistent with the paper's identity.

**Fix**: Rewrite Section 6.1 to set expectations: "We hypothesized that CMI could predict absorption susceptibility, motivated by the successive refinement theorem. We report two experiments: a marginally significant result at L0=82 that does not survive probe quality controls, and a pre-registered null at L0=22."

### 2. Abstract Overclaims Patching Result (Severity: Critical)

Abstract: "Activation patching on 8 persistent core words finds 0/8 parent recovery after child zeroing, ruling out competitive exclusion."

"Ruling out" with n=8 and 95% CI [0%, 36.9%] is overclaiming. The data are consistent with competitive exclusion rates up to 37%. The JumpReLU hard threshold confound further weakens the interpretive strength.

**Fix**: "finding no evidence of competitive exclusion (0/8 recovery; 95% CI: [0%, 36.9%])."

---

## Major Issues

### 3. Hypotheses Not Enumerated in Paper Body (Severity: Major)

Section 7.4 references "five pre-registered hypotheses" (H2, H4, H5, H6, H7). Section 8 says "Four of seven pre-registered hypotheses are falsified." But the reader cannot verify: the paper body never lists H1-H7.

The hypotheses.md document has a clear table with 10 hypotheses, predictions, and outcomes. This should be in the paper.

**Fix**: Add a hypothesis summary table in Section 3 or expand Section 7.4. Each entry: hypothesis statement, pre-registered threshold, observed value, status.

### 4. 93.8% Non-Hedging Result Under-Discussed (Severity: Major)

Section 4.4 reports that 615/656 FN tokens at L0=22 have none of the 5 parent-associated latents firing at L0=176 -- yet these tokens are NOT false negatives at L0=176. The paper reports the 6.2% strict hedging rate and the 93.8% non-hedging rate but does not explain how the non-hedging tokens resolve.

The most natural explanation -- different latents handle first-letter classification at different L0 values -- has implications for the confound decomposition framework that the paper does not address.

**Fix**: Add a paragraph explaining this finding and discussing whether the probe latents are L0-specific.

### 5. No Method Diagram (Severity: Major)

Five pages of methodology (Section 3) without a visual. The measurement protocol has 7+ steps that a reviewer unfamiliar with the Chanin protocol will struggle to follow. The outline planned a schematic; it was never realized.

**Fix**: Add a pipeline diagram showing: input tokens -> SAE encoding -> probe training -> candidate identification -> FN classification -> four controls -> confound decomposition -> activation patching.

### 6. Table Numbering Out of Sequence (Severity: Major)

Table 7 (Section 5.1) precedes Table 6 (Section 6.2) in the text. Table 8 (Section 3.7) has the highest number but appears earliest.

**Fix**: Renumber all tables sequentially.

---

## Minor Issues

### 7. Section 7.2 Not Updated for Patching

Section 7.2 (Two Interpretations) was written when patching was pending. The 0/8 result partially favors Interpretation B (genuine low absorption) over Interpretation A (pure miscalibration), but the section does not incorporate this.

**Fix**: Add a sentence noting patching evidence, with the JumpReLU threshold caveat.

### 8. "Stark" in Section 4.4

"The discrepancy between permissive (98.6%) and strict (6.2%) hedging is stark." Replace with the specific magnitude: "spans 92.4 percentage points."

### 9. Placeholder URL

Section 8: "[URL]" must be filled or changed to "[URL upon acceptance]."

### 10. Abstract Length

237 words -- acceptable but dense. First sentence is 98 words, which taxes working memory.

### 11. One Stale notation.md Value

notation.md lists C1 (Random probe) expected absorption as "~9.2%" but the paper reports 11.8%.

---

## Exemplary Practices (Preserve)

1. **Table 6 (CMI progressive weakening)**: Model of statistical transparency. Raw -> partial -> restricted -> replicated. Each row adds one control. The "Interpretation" column is a reader service.

2. **Section 4.2 (Candidate Explosion)**: Best paragraph-level writing. Numbers-first: random vector -> 23% candidates -> true vs shuffled indistinguishable -> candidate step vacuous.

3. **Section 7.2 (Two Interpretations)**: Honest, calibrated. "Both interpretations agree on the empirical facts and the practical consequence."

4. **Negative result reporting**: 8/10 hypotheses falsified, each with pre-registered target vs. observed value. Consistently the paper's strongest methodological feature across ALL reviews.

5. **Table 5 (Activation Patching)**: Clean per-word format with three methods and controls.

6. **Section 5.1 reconciliation note**: Transparent about the 14.39%/15.96% difference with explanation. This is how methodological variations should be handled.

SCORE: 7.5 (8.0 after critical and major fixes applied)
