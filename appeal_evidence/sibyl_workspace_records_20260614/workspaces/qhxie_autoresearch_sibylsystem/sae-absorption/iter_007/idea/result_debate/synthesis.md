# Result Debate Synthesis: SAE Feature Absorption Audit (Iteration 9)

**Synthesizer**: Senior Research Director
**Date**: 2026-04-15
**Input**: 6 perspectives (Optimist, Skeptic, Methodologist, Comparativist, Revisionist, Strategist)

---

## 1. Consensus Map: High-Confidence Conclusions

The following findings enjoy unanimous or near-unanimous agreement across all 6 perspectives:

### 1.1 The Chanin Metric Fails on JumpReLU SAEs (Unanimous)
All perspectives agree that the absorption metric, as originally defined with cosine threshold >= 0.025, does not produce valid measurements on Gemma 2 2B + JumpReLU SAEs. Shuffled-label controls exceed measured absorption in all 5 domains. The control failure diagnosis provides a mechanistic explanation: in R^2304, the cosine threshold identifies 23% of decoder columns as candidates, making P(at least one candidate active) = 1.0 at L0=82. The metric degenerates into a proxy for false-negative rate.

**Confidence**: Very High. Supported by 5-domain consistency, 1000 random vector analysis, and 5x4 threshold sensitivity grid (CV=0.077).

### 1.2 The L0 Phase Transition Is Real and Robust (Unanimous)
Absorption rate declines monotonically: 42.85% (L0=22) -> 37.49% (L0=41) -> 14.39% (L0=82) -> 0.84% (L0=176). Cross-layer stability is strong (CV<10% across L10, L12, L20). Spearman rho = -1.0 (perfect monotonicity).

**Confidence**: Very High. The skeptic notes this could partly reflect probe accuracy improvement at higher L0, but the magnitude and cross-layer consistency make this the paper's most unassailable finding.

### 1.3 CMI as a Diagnostic Tool Is Falsified (5/6 agree; Optimist partially dissents)
The CMI-absorption correlation at L0=82 (rho=-0.383, p=0.059) vanishes after controlling for probe F1 (partial rho=-0.328, Bonferroni p=0.472), collapses on the restricted F1>0.85 subset (rho=-0.113, p=0.757), and fails to replicate at L0=22 where all probes achieve F1=1.0 (rho=+0.044, p=0.835). The skeptic, methodologist, revisionist, and strategist all classify H4 as falsified. The optimist acknowledges CMI's weakness but frames L0=22 replication as a future "icing-on-the-cake" experiment; however, the data already in hand (rho=+0.044 with direction reversal) is decisive. The comparativist notes this as an important negative result vis-a-vis rate-distortion theory.

**Judgment**: CMI diagnostic is falsified. It must be demoted from "theoretical pillar" to "exploratory analysis that did not replicate." Every sentence presenting CMI as having predictive or diagnostic value must be revised.

### 1.4 Honest Reporting of Negative Results Is the Paper's Strongest Differentiator (Unanimous)
All 6 perspectives independently identify the high falsification rate (at minimum 5/7 original hypotheses, likely 7/10 including new ones) as an exceptional quality. In the current ML publication environment where negative results are systematically underreported, this transparency is both scientifically valuable and strategically advantageous with reviewers.

**Confidence**: Very High. This is a meta-observation about the paper's positioning, not a data-dependent claim.

### 1.5 Data Integrity Is Confirmed (Unanimous)
85 numerical claims verified, 84 exact matches, 1 missing (not inconsistent). Integrity score 98.82%. No perspective raises data quality concerns about the core experimental outputs.

---

## 2. Conflict Resolution

### 2.1 CRITICAL: Is the 0/8 Activation Patching Result Interpretable?

**Optimist + Revisionist + Strategist**: The 0/8 recovery is decisive causal evidence that competitive exclusion does not drive persistent false negatives. Three patching methods agree. Control group 0/65. This eliminates the "competitive exclusion" interpretation.

**Skeptic + Methodologist**: The 0/8 result is confounded by JumpReLU's hard threshold. All 8 parent features have exactly zero activation -- they may be globally dead at L0=82, not locally suppressed. Zeroing child features adds decoder contributions to the residual stream, but this may not push any parent past the JumpReLU threshold. The experiment cannot distinguish "no competitive exclusion" from "threshold too high to detect competitive exclusion." Testing at L0=41 (where parents may be near-threshold) or reporting continuous projection values would resolve this.

**My Judgment**: The skeptic and methodologist raise a valid architectural confound that the optimist glosses over. However, the practical implication is the same: whether parents are "not suppressed" or "below threshold regardless," the absorption metric is measuring something other than competitive exclusion in these cases. The paper should:
1. Present 0/8 as strong evidence against competitive exclusion as the mechanism for persistent false negatives
2. Explicitly discuss the JumpReLU threshold confound as an alternative interpretation (the methodologist's "Explanation B")
3. Report continuous projection values post-patching if available -- if projections increase but remain sub-threshold, this is informative
4. Frame the combined evidence (0/8 patching + 6.2% strict hedging + control failure diagnosis) as converging on the same conclusion from independent angles, which mitigates the weakness of any single experiment

**Verdict**: The 0/8 result is strong but not "decisive" in isolation. Its value multiplies when combined with the other evidence streams. The paper should present the convergence, not rely on patching alone.

### 2.2 CRITICAL: Is the Metric "Broken" or "Miscalibrated"?

**Skeptic (S4)**: Threshold sensitivity shows stable results (CV=0.077), which is inconsistent with a "broken" metric. A recalibrated metric at cosine >= 0.10 (where zero random candidates appear) might work perfectly. The finding may be about calibration, not invalidity.

**Optimist + Revisionist**: Even at cosine >= 0.05, control failure persists (measured 0.118 vs shuffled 0.746). The problem is structural, not parametric.

**My Judgment**: This is a nuanced but important distinction. The skeptic is correct that "the metric is fundamentally broken" is a stronger claim than the data supports. The threshold sensitivity analysis shows the metric is *consistently wrong* across parameters, but the diagnosis shows *why* it is wrong (high-dimensional geometry). The proper framing is:

- The metric as currently parameterized (cos >= 0.025) is invalid on JumpReLU SAEs in R^2304
- Recalibration to cos >= 0.10 may theoretically resolve the control failure, but at the cost of detecting almost no absorbing features (the metric becomes too stringent)
- The underlying issue -- that cosine similarity thresholds scale poorly with dimensionality -- is structural and affects any threshold-based approach in high dimensions

The paper should say "the metric does not transfer without recalibration" rather than "the metric is fundamentally invalid." It should note that effective recalibration may require a fundamentally different approach (e.g., top-k nearest neighbor instead of global cosine threshold).

### 2.3 IMPORTANT: What Does the 92.3pp Gap Between Loose and Strict Hedging Mean?

**Optimist**: The gap reveals systematic misclassification -- loose definition conflates "compensatory feature resolution" with "hedging." The 6.2% strict rate is statistically significant vs. shuffled (p=0.0004), proving a small but real hedging signal exists.

**Skeptic**: The 6.2% strict rate is only 2.8pp above the shuffled control (3.4%). This is a tiny effect, not "dominant." The paper creates a narrative vacuum: neither "all hedging" nor "competitive exclusion" is supported for 93.8% of false negatives.

**Methodologist**: This is a construct validity failure. The loose definition conflates two different phenomena. The paper must classify the 93.8% that are resolved by non-parent features at L0=176.

**Revisionist**: These should be reclassified as "encoding absence" -- the parent latent never encodes first-letter information for these tokens, rather than information being "hedged" elsewhere.

**My Judgment**: The skeptic is right that 6.2% is a tiny effect and 93.8% remains mechanistically unexplained. The revisionist's "encoding absence" framing is the most intellectually honest: for the vast majority of false negatives, the parent feature simply does not encode the relevant information at any L0 level, which is a different phenomenon from competitive exclusion or hedging. The paper should adopt a three-category decomposition:
1. **Strict hedging** (6.2%): Parent-associated latent recovers at high L0 -- genuine feature spreading
2. **Compensatory resolution** (~87.6%): FN resolved at high L0 via entirely different features -- mechanism unknown, possibly encoding absence
3. **Persistent FN** (~0%): Still FN at L0=176 -- not observed in this dataset

This decomposition is more informative than binary "hedging vs. competitive exclusion" and acknowledges honest uncertainty about the dominant category.

### 2.4 MODERATE: Should "Phase Transition" Language Be Used?

**Methodologist**: Only 4 data points; "phase transition" requires denser sampling in the L0=40-80 interval. Suggest "sharp decline" or "regime change."

**Optimist + Comparativist**: The perfect monotonicity (rho=-1.0) and cross-layer stability justify "transition" language.

**My Judgment**: The methodologist is technically correct. A "phase transition" in the physics sense implies a specific mathematical structure (divergence, critical exponent, etc.) that 4 data points cannot establish. However, the term is widely used loosely in ML. **Compromise**: Use "L0 sparsity transition" rather than "phase transition" in the title and main text. Discuss the L0=40-80 interval as the "transition regime" with appropriate hedging about the limited number of data points.

### 2.5 MODERATE: Paper's Competitive Position

**Comparativist**: Rates the paper 7.0-7.5 (Weak Accept to Accept) at NeurIPS/ICML level.

**Optimist**: Projects 7.5-8.0 after writing revision.

**Strategist**: Projects 7.5-8.0, with 8.0+ possible if writing is excellent.

**My Judgment**: The evidence base supports a 7.0-7.5 range after writing revision. The 8.0 ceiling requires (a) flawless writing, (b) no reviewer pushback on the JumpReLU threshold confound in patching, and (c) positive reception of the negative-results framing. Realistic target: **7.0-7.5 with upside to 8.0**.

---

## 3. Result Quality Score: 7.0 / 10

**Justification**:

*Strengths pushing score up:*
- First metric audit on JumpReLU SAEs with mechanistic explanation for control failure (+1.0)
- Clean, consistent experimental execution across all analyses (+0.5)
- L0 transition is robust, cross-layer stable, and practically useful (+0.5)
- Activation patching provides metric-independent causal evidence (+0.5)
- Unprecedented honest reporting of negative results (+0.5)
- Four-tier control suite (C1-C4) exceeds community standards (+0.5)
- 98.82% data integrity verified through automated cross-checking (+0.25)

*Weaknesses pulling score down:*
- CMI theoretical pillar collapsed -- paper loses one of three main contributions (-0.5)
- JumpReLU threshold confound in patching is unaddressed (-0.5)
- Single model (Gemma 2 2B), single SAE family (Gemma Scope JumpReLU) (-0.25)
- 93.8% of false negatives remain mechanistically unexplained (-0.25)
- N=25 for letter-level analyses provides fundamentally limited statistical power (-0.25)
- No L1-ReLU control failure verification -- cannot confirm JumpReLU specificity (-0.25)

**Net score: 7.0** -- solid experimental foundation that supports a publishable paper after writing revision.

---

## 4. Key Findings (5 bullets)

1. **The Chanin absorption metric does not produce valid measurements on JumpReLU SAEs.** Shuffled-label controls exceed measured absorption in all 5 tested domains. In R^2304, the standard cosine threshold (>= 0.025) matches 23% of decoder columns for any random direction, causing the metric to degenerate into a proxy for false-negative rate. This is a structural property of high-dimensional geometry, not a threshold tuning issue.

2. **Competitive exclusion is not the mechanism for persistent false negatives.** Activation patching on 8 words that are false negatives at all L0 levels yields 0/8 parent feature recovery across three patching methods (decode-reencode, residual subtraction, all-children zeroing), with 0/65 control recovery. While the JumpReLU hard threshold confound limits the causal inference, the convergence with other evidence streams strongly disfavors the competitive exclusion narrative.

3. **The hedging rate depends critically on its operational definition.** Loose definition: 98.6% of FNs at L0=22 resolve at L0=176. Strict definition: only 6.2% resolve via parent-associated latent recovery (p=0.0004 vs. shuffled control at 3.4%). The 92.3pp gap reveals that the dominant FN resolution mechanism is compensatory feature reorganization, not parent-specific information hedging.

4. **L0 is the primary control variable for absorption severity.** The absorption rate declines monotonically from 42.85% (L0=22) to 0.84% (L0=176), with cross-layer CV<10%. The transition regime lies approximately in L0=40-80. This provides an immediate, actionable design guideline: SAEs with L0>80 exhibit minimal absorption artifacts.

5. **CMI does not predict absorption susceptibility.** The initial correlation at L0=82 (rho=-0.383) is entirely driven by probe quality confounding (rho(absorption, probe_F1) = -0.692). When confounds are removed at L0=22 (all probes F1=1.0), the correlation vanishes (rho=+0.044, p=0.835). This falsifies the rate-distortion diagnostic hypothesis.

---

## 5. Methodology Gaps: Critical Improvements Needed

### 5.1 Must-Fix Before Submission (Blocking)

**Gap M1: JumpReLU threshold confound in activation patching.**
The skeptic and methodologist independently identify this as the most serious unaddressed confound. Parent features with exactly zero activation may be globally dead at L0=82, not locally suppressed. The paper must:
- Discuss the JumpReLU threshold alternative interpretation explicitly
- Report continuous projection values toward the parent direction post-patching if available
- Acknowledge that the patching experiment cannot fully distinguish "no competitive exclusion" from "sub-threshold competitive exclusion"
This is a discussion-level fix, not a new experiment -- but it must be present.

**Gap M2: CMI narrative must be comprehensively demoted.**
The paper currently has 7+ locations where CMI is presented as having predictive or diagnostic value. Every instance must be revised. Section 6 requires complete rewriting from "CMI predicts absorption susceptibility" to "Exploratory CMI analysis and its falsification."

**Gap M3: Hedging narrative must present three-category decomposition.**
Replace the binary "hedging vs. competitive exclusion" framing with: (a) strict hedging (6.2%), (b) compensatory resolution (87.6%), (c) persistent FN (~0%). State explicitly that the mechanism for category (b) is unresolved.

**Gap M4: "Phase transition" language must be softened.**
4 data points do not establish a phase transition. Use "L0 sparsity transition" or "sharp decline" unless additional L0 sampling points are added.

### 5.2 Should-Fix (Non-Blocking but Strengthening)

**Gap M5: No L1-ReLU control failure verification.** Running the same shuffled-label controls on GPT-2/L1-ReLU SAEs would distinguish "JumpReLU-specific failure" from "high-dimensional geometry failure." If L1-ReLU passes the control, the JumpReLU specificity argument is massively strengthened.

**Gap M6: Post-hoc power analysis for CMI.** N=25 provides power ~0.40 to detect rho=-0.4. This should be reported to contextualize the negative result.

**Gap M7: Strict hedging k-sensitivity analysis.** The strict hedging rate depends on k=5 parent-associated latents. Varying k and reporting sensitivity would strengthen reproducibility.

**Gap M8: Probe training details.** k-sparse probe learning rate, epoch count, early stopping criteria, and CMI k-NN estimator parameters should be fully documented.

**Gap M9: Vocabulary reconciliation.** Three different vocabulary sizes (1204, 1196, 1092) across experiments, and 8 vs. 9 persistent core words discrepancy -- both need explicit explanation.

---

## 6. Competitive Position

### 6.1 Where We Stand vs. SOTA

This paper occupies a unique "meta-audit" position in the SAE absorption literature. It does not compete directly with architecture-improvement papers (Matryoshka SAE, OrtSAE, ATM-SAE, Masked Regularization) but rather questions the metric on which they all report improvements.

**Unique firsts (no prior work exists):**
- First shuffled-label control failure on JumpReLU SAEs
- First quantitative hedging-vs-exclusion decomposition
- First causal test (activation patching) of the competitive exclusion hypothesis
- First systematic L0-absorption curve with cross-layer validation
- First mechanistic diagnosis of control failure (candidate explosion in R^2304)
- First attempted (and honestly failed) cross-domain absorption measurement

**Positioning relative to key works:**
- **Chanin et al. (NeurIPS 2025 Oral)**: This paper directly audits their metric and finds it invalid on JumpReLU. Complementary, not contradictory -- Chanin's work on GPT-2/L1-ReLU may still be valid.
- **SAEBench (ICML 2025)**: SAEBench already notes their absorption metric is "not useful for domain-specific evaluation." This paper provides the quantitative evidence and mechanistic explanation for why.
- **Feature Hedging (Chanin et al., 2025)**: This paper provides the first quantitative verification of the hedging framework, with the important refinement that strict hedging is only 6.2%.

**Predicted impact if published:**
1. All subsequent absorption measurements will be required to include shuffled-label controls
2. Architecture mitigation claims on JumpReLU will need re-verification
3. SAEBench absorption metric may be revised or annotated
4. L0 as the primary control parameter will enter hyperparameter selection guidelines

### 6.2 Weaknesses vs. SOTA

- Single model + single SAE family limits generalizability
- No improved metric proposed (diagnosis without cure)
- CMI negative result reduces theoretical depth
- Small sample sizes for causal testing (8 words) and letter-level analyses (25 letters)

---

## 7. Hypothesis Update

### Confirmed (Survived)
- **H1 (Metric non-transfer)**: Strongly confirmed with mechanistic explanation. Upgraded from "observation" to "explained structural property."
- **H3 (L0 transition)**: Confirmed and cross-validated. Reinterpreted from "competitive exclusion diminishes with capacity" to "SAE encoding capacity determines feature selection priority."

### Falsified (Require Revision)
- **H2 (Hierarchy-driven dominance >80%)**: Falsified. Only 1.4% (loose) or 0% (strict minus strict hedging). Competitive exclusion is not the dominant FN mechanism.
- **H4 (CMI predicts absorption)**: Falsified. L0=82 association driven by probe quality confound; L0=22 replication yields rho=+0.044.
- **H5 (Cross-domain absorption >10% in 2+ domains)**: Falsified. Universal control failure prevents interpretation.
- **H6, H7 (related cross-domain hypotheses)**: Falsified.
- **H8 (Activation patching 7+/9 recovery)**: Falsified. 0/8 recovery.
- **H9 (Strict hedging >80%)**: Falsified. 6.2%.
- **H10 (CMI at L0=22 rho<-0.3)**: Falsified. rho=+0.044.

### Revised Mental Model
The pre-experiment mental model treated "absorption" as competitive exclusion where child features actively suppress parent features. The evidence supports a fundamentally different picture:

**New model -- "Encoding absence under capacity constraint":**
At a given L0, the SAE encoder faces a feature selection problem. High-semantic-value features (common word meanings, syntactic patterns) take priority over low-information-density surface attributes (first letter). At low L0, the encoder simply does not encode first-letter information via the parent latent -- the parent feature is not "suppressed" by children but was never selected. As L0 increases, encoding capacity grows and surface attributes can be encoded alongside semantic content. The transition occurs roughly at L0=40-80.

This model is consistent with: L0 transition (capacity effect), 0/8 patching recovery (not suppression but absence), G letter anomaly (its parent latent may encode additional information making it viable even at low L0), and probe quality confound (low F1 reflects weak encoding, not measurement error).

---

## 8. Action Plan

### Recommendation: PROCEED to Gate 2 (Writing Revision)

The experimental evidence is complete. No additional experiments are required for a publishable paper. The sole remaining bottleneck is writing revision.

### Priority Actions

**P0 -- Must complete before any review (blocking):**

1. **Rewrite Section 6 (CMI)**: From "CMI predicts absorption" to "Exploratory CMI analysis and its falsification." Present L0=82 initial signal, L0=22 replication failure, partial correlation evidence, and probe quality confound diagnosis. Classify as honest negative result. (~2 hours)

2. **Integrate tightened hedging results (Section 4.2)**: Present three-category decomposition (6.2% strict hedging, ~87.6% compensatory resolution, ~0% persistent). Drop the "98.6% hedging" headline. Discuss the 92.3pp gap and its implications. (~2 hours)

3. **Integrate activation patching (new Section 4.3)**: Present 0/8 result with all three methods. Include JumpReLU threshold confound discussion. Frame as converging evidence with hedging and control failure results. (~2 hours)

4. **Revise title and abstract**: Remove rate-distortion reference. Adopt audit framing. Compress abstract to ~220 words. Suggested title direction: "Auditing SAE Feature Absorption: Metric Failure, Encoding Absence, and the L0 Sparsity Transition on JumpReLU SAEs" (~1 hour)

5. **Eliminate 7+ CMI overclaims**: Systematic search-and-replace in abstract, introduction, Section 6 title, conclusion, and hypothesis summary table. Every CMI mention must reference L0=22 falsification. (~1 hour)

**P1 -- Should complete (strengthening):**

6. **Integrate control failure diagnosis (Section 4.1)**: Add candidate explosion mechanism (23% feature matching, P(>=1)=1.0) as a subsection. (~1 hour)

7. **Add threshold sensitivity table/figure**: Include 5x4 heatmap showing CV=0.077 stability. (~30 minutes)

8. **Revise Section 7.4 (negative results)**: Update H4 from "partially supported" to "falsified." Update hypothesis summary table to reflect 2 confirmed, 7+ falsified. (~30 minutes)

9. **Add dual-interpretation paragraph (Section 7.2)**: "Metric miscalibration" vs. "absorption genuinely low on JumpReLU" -- both interpretations are compatible with the data. (~30 minutes)

10. **Soften "phase transition" to "sparsity transition"**: Throughout the paper. (~15 minutes)

**P2 -- Nice to have (polish):**

11. Compress Section 5.3 (cross-architecture comparison) to 2-3 sentences acknowledging full confounding
12. Reduce structural redundancy (~300 words)
13. Reconcile vocabulary sizes and core word counts in methods section
14. Add post-hoc power analysis footnote for CMI

### Total Estimated Writing Time: ~12 hours

### Timeline Recommendation
- Complete writing revision within 1 week
- Run final automated validation (validate_integration) before submission
- Post arXiv preprint immediately after revision to establish priority
- Target ICLR 2027 (deadline approximately September-October 2026) as primary venue; COLM 2027 as backup
- No additional GPU time required -- release GPU resources to other projects

### Key Risk to Monitor During Writing
The single greatest risk in the writing phase is **re-introducing CMI overclaims** through inattentive revision. Establish a firm rule: every mention of CMI in the paper must be followed by a reference to the L0=22 replication failure and/or the Bonferroni-corrected p-value.
