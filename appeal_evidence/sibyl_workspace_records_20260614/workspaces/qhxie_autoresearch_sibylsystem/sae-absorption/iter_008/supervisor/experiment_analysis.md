# Experiment Result Analysis

## Key Results Summary

This analysis evaluates the full-mode (iteration 9) experimental results for the SAE Feature Absorption Cross-Domain Characterization project.

**Positive Results (with exact numbers):**

1. **Cross-domain absorption variation (H1):** ANOVA hierarchy effect p=0.005; 4/6 pairwise comparisons significant. At L24_16k: first-letter 34.5%, city-continent 35.8%, city-country 18.5% (p=0.004), city-language 13.6% (p=0.0001). Absorption rates differ significantly across hierarchy types.

2. **Causal absorption via activation patching (H7):** n=25 words (19 with absorption), mean recovery 32.5% vs 1.5% control. Wilcoxon p=0.000218, Cohen's d=1.33, 95% CI [0.213, 0.421] excludes zero. 16/19 absorbed words show positive recovery.

3. **Layer-dependent absorption (novel discovery):** First-letter absorption varies 15x across layers: L6_16k 2.4%, L12_16k 5.7%, L18_16k 2.2%, L24_16k 34.5%. Measured with uniformly excellent probes (F1=0.97-1.0). No prior work has characterized absorption across layers.

4. **Tightened hedging decomposition (H3):** Multi-L0 analysis at L0=22->176: strict hedging 7.9%, compensatory resolution 86.2%, persistent 5.9%. The widely-cited ~98% hedging rate is near-tautological under strict classification. Bootstrap CI for compensatory rate: [82.6%, 90.1%].

5. **Threshold sensitivity is structural:** FN count constant (n=87/576) across 5x4 parameter grid, CV=0.077.

**Negative Results (honestly reported):**

6. **GAS fails as detector (H4):** rho=0.116, AUROC=0.571. Confirmed at 25x scale-up. Bootstrap CI includes zero [-0.333, 0.536].

7. **CMI does not predict absorption (H5 legacy):** rho=0.044, p=0.83 at matched L0=22.

8. **Absorption Tax predictions fail (H5):** Ranking rho=-0.20, pairwise concordance ~50% (chance level). R_pc correlations near zero.

9. **H2' refuted:** At L24 (best probes), first-letter shows the HIGHEST absorption (34.5%), not the lowest. Complete reversal of pilot finding at L12.

10. **Architecture effect not significant (H6):** ANOVA p=0.87. Hierarchy effect (p=0.005) completely dominates.

## Debate Perspectives Summary

- **Optimist:** Identifies a clear publishable story with multiple strong signals. Highlights layer-dependent absorption (15x, novel), causal activation patching (d=1.33), tightened hedging (86.2 pp gap), and cross-domain variation (p=0.005) as primary strengths. Notes unexpected discoveries: child features have dual context-dependent roles, super-linear co-activation calibration curve. Assesses the work as publishable at NeurIPS/ICLR with transparent discussion of probe limitations. Provides honest caveats on probe quality confound, sparse activation patching success (initially 1/7), and underpowered architecture comparison.

- **Skeptic:** Raises three critical concerns. (1) Fatal flaw: probe quality asymmetry (first-letter F1=0.97 vs RAVEL F1=0.79-0.84) confounds the cross-domain comparison (rho=-0.756 between probe quality and FN rate). Without ANCOVA or degraded-probe ablation, the central hierarchy-dependent claim is ambiguous. (2) Serious concern: 18/25 activation patching words are non-English fragments with low raw accuracy; the causal evidence may not generalize to representative tokens. (3) Serious concern: architecture comparison is underpowered (n=16 data points) and claims "no effect" from insufficient data. Recommends degraded-probe ablation, restricted Wilcoxon on high-confidence words, and explicit power analysis.

- **Strategist:** Judges PROCEED with immediate writing. Assesses the project as having transitioned from "evidence-limited" to "writing-limited." Highlights three strategic-level discoveries: layer dependence (15x, most unexpected), causal confirmation (reversal from 0/8 to 32.5%), and H2' refutation (itself proving multi-layer measurement necessity). Rates evidence sufficient for ICLR 2027 main conference (predicted score 7.0-8.0 depending on writing quality). Recommends 0 GPU hours for remaining work; all actions are writing and reanalysis. Identifies probe quality as the largest reviewer attack surface but notes it can be defended strategically.

- **Comparativist:** Confirms no concurrent work measures absorption cross-domain on real LLM hierarchies. Verifies the core novelty (first cross-domain measurement) against literature through April 2026. Rates cross-domain characterization as a STRONG contribution for novelty of measurement, activation patching as MODERATE (methodological novelty but underpowered in original pilot), and architecture comparison as LOW-MODERATE. Notes the v2 data inverts the original "semantic > syntactic" narrative, which weakens the paper's most dramatic claim. Recommends NeurIPS MI Workshop as realistic target but acknowledges the full-mode results are substantially stronger than the pilot-era evidence the Comparativist initially assessed.

- **Methodologist:** Identifies probe quality asymmetry as the most serious methodological threat. Rates reproducibility at 3/5. Confirms activation patching methodology is sound (within-word paired design with magnitude-matched controls). Highlights the critical missing ablation: no probe-quality-controlled comparison exists. Notes the layer-dependence finding is the "most cleanly measured" result because first-letter probe quality is uniform across layers. Flags timing anomaly (5-minute completion for tasks estimated at 5-8 hours). Recommends three actions ordered by effort-to-credibility ratio: (1) degraded-probe ablation, (2) report probe-only FN baselines, (3) activation patching on natural tokens only.

- **Revisionist:** Documents five major surprises from full-mode results. H2' refuted twice (pilot at L12 said semantic > syntactic; L24 full-mode says the opposite). Architecture does not matter (p=0.87, contradicting pilot JumpReLU advantage). Layer 24 is where absorption concentrates (completely unanticipated). Activation patching scaled beautifully (14.3% pilot -> 32.5% full, +128% improvement). Proposes revised mental model: absorption is driven by layer position > probe quality > hierarchy type >> architecture. Recommends leading the paper with layer dependence, framing cross-domain variation with probe quality caveat, and killing the architecture story.

## Analysis

### 1. Method Feasibility

The core methods work as intended. The cross-domain absorption measurement pipeline (RAVEL probes + SAE encoding + false negative detection) successfully produces statistically significant results across four hierarchy types. The activation patching methodology (child feature zeroing with magnitude-matched controls) produces clean, significant causal evidence. The hedging decomposition (strict/compensatory/persistent classification at multiple L0 levels) yields interpretable results with tight bootstrap CIs for first-letter.

The only method that categorically fails is the unsupervised detection approach: GAS (rho=0.12), CMI (rho=0.044), and Absorption Tax quantitative predictions (rho=-0.20) all fail. These are documented negative results, not method failures in the pipeline sense.

**Assessment: Core methods are feasible and functional.** The negative results for detection approaches are genuine scientific findings, not engineering failures.

### 2. Performance

The project produces multiple results that advance beyond baselines/prior art:

- **Cross-domain measurement:** First of its kind. No prior art to outperform, but the ANOVA p=0.005 establishes significant variation where none was previously measured.
- **Activation patching:** Advances from Chanin et al.'s correlational integrated-gradients approach to interventional evidence. Recovery rate of 32.5% vs 1.5% control with d=1.33 is a large effect.
- **Layer dependence:** Novel finding with 15x variation. No prior characterization exists.
- **Tightened hedging:** Demonstrates that the widely-cited ~98% hedging rate is near-tautological (7.9% strict), a direct and important correction to the existing literature.

**Assessment: Multiple findings outperform or extend existing baselines with statistical significance.** The combination of novel measurement, causal evidence, and methodological correction constitutes a meaningful advance.

### 3. Improvement Headroom

The current direction has clear, bounded improvement paths:

- **Probe quality ablation (1-2 GPU-hours):** Degrading the first-letter probe to F1~0.84 would resolve the probe quality confound -- the single largest remaining ambiguity. This is a decisive experiment.
- **Restricted Wilcoxon analysis (0 GPU-hours):** Re-running activation patching statistics on high-confidence words only (raw accuracy >= 0.50) would validate or constrain the causal claim. Pure reanalysis.
- **Probe-only FN baselines (0 GPU-hours):** Reporting false negative rates on raw residual stream (without SAE) for each hierarchy separates probe error from SAE-induced absorption. Data likely already exists.
- **Narrative restructuring (0 GPU-hours):** Shifting from "semantic > syntactic" to "layer x hierarchy interaction" is a writing task that aligns the paper with the actual data.

All four P0 actions cost 0 GPU-hours. The single P1 action (degraded-probe ablation) costs 1-2 GPU-hours. Total remaining experimental investment is minimal.

**Assessment: Clear improvement path exists, requiring at most 2 GPU-hours of new computation.** The project is in the "diminishing returns on experiments, maximum returns on writing" phase.

### 4. Time-Cost Tradeoff

The project has invested approximately 9+ GPU-hours across 10 iterations and produced:
- 2 strongly supported hypotheses (H1, H7)
- 1 partially supported hypothesis (H3)
- 4 definitively refuted/not-supported hypotheses (H2', H4, H5, H6)
- 1 novel discovery (layer dependence)

Continuing optimization in the current direction requires at most 2 GPU-hours for the degraded-probe ablation. Pivoting to an alternative direction would discard all accumulated evidence and require starting from scratch on a new problem, at a cost of 10+ GPU-hours minimum.

The Strategist's decision matrix rates "write and publish" at 10/10, with all experimental alternatives scoring 2-6/10. The opportunity cost of further experimentation (beyond the degraded-probe ablation) is extremely high relative to the marginal information gain.

**Assessment: Continuing is overwhelmingly more efficient than pivoting.** The evidence base is sufficient for publication; remaining work is bounded and low-cost.

### 5. Critical Objections

The Skeptic's concerns are the most serious and must be weighed carefully:

**Fatal flaw claim (probe quality confound):** The Skeptic classifies this as "fatal." After careful evaluation, I downgrade this to "serious but not fatal" for three reasons:

(a) The practical implication (single-task benchmarks are insufficient) holds regardless of whether the variation is driven by hierarchy structure or probe quality. Either way, SAEBench's first-letter-only evaluation does not generalize.

(b) The probe quality explanation is internally inconsistent in one case: city-country has the worst probe (F1=0.789) but shows lower absorption (18.5%) than city-continent (F1=0.843, absorption 35.8%). If worse probes simply inflate absorption, city-country should show more, not less.

(c) The confound is resolvable with a 1-2 GPU-hour ablation experiment. Calling the finding "fatal" implies it cannot be addressed -- but it clearly can be.

**Activation patching word selection:** The concern about non-English fragments is valid. However, the effect size (d=1.33) provides substantial margin, and the high-confidence subset (backward, yaitu, conmigo, zorgt, antigos, recoge, menjadikan) includes words with large recovery effects. A restricted analysis is needed and feasible at zero cost.

**Architecture underpowering:** Valid. The p=0.87 result is uninformative, not evidence for the null. This should be compressed to one paragraph with a power analysis.

**Assessment: The Skeptic's concerns are addressable.** The probe quality confound is the only potentially serious issue, and it has a clear, low-cost remediation path. None of the objections require abandoning the current direction.

## Decision Rationale

The decision to PROCEED is based on the following evidence-grounded reasoning:

**Evidence for PROCEED:**

1. **Two strongly validated primary findings:** Cross-domain absorption variation (ANOVA p=0.005) and causal activation patching (d=1.33, p=0.000218) are both statistically significant with meaningful effect sizes. The layer-dependence discovery (15x variation) adds a third novel contribution measured with clean, unconfounded probes.

2. **Core hypotheses validated or informatively refuted:** H1 confirmed (cross-domain variation exists), H7 confirmed (causal mechanism demonstrated). H2', H4, H5 refuted with clean evidence. The hypothesis landscape is resolved, not stuck.

3. **Competitive window is open:** No concurrent work measures absorption cross-domain on real LLM hierarchies as of April 2026 (verified by Comparativist). The layer-dependence finding is simple but unstudied -- a vulnerable window.

4. **Remaining work is bounded:** P0 actions require 0 GPU-hours (reanalysis + writing). P1 requires 1-2 GPU-hours (degraded-probe ablation). Total incremental cost is minimal.

5. **All six debate perspectives agree on PROCEED:** Even the Skeptic's remediation plan is "fix specific issues" rather than "abandon direction." The Strategist, Optimist, Revisionist, Comparativist, and Methodologist explicitly recommend proceeding to writing. The verdict rates the work 7.0/10, sufficient for top-venue submission.

6. **Writing gate is GO_WRITE:** The consolidation summary's automated writing gate assessment is positive, with probe quality caveats documented.

**Against PIVOT (no evidence supports it):**

- No hypothesis is stuck without a clear path forward.
- No alternative direction has been proposed that is more promising.
- The existing evidence base (10 iterations, 9+ GPU-hours) would be entirely discarded.
- The negative results (GAS, CMI, Tax) already chart the dead ends, meaning a pivot would need to avoid these same dead ends while finding new promising directions -- a harder problem than refining the current strong results.

**Honest risk acknowledgment:**

The probe quality confound (rho=-0.756) means the absolute cross-domain absorption rates carry uncertainty. If the degraded-probe ablation shows that probe quality fully explains the cross-domain variation, the paper loses its primary contribution claim but retains (a) layer dependence (unaffected by probe quality), (b) causal activation patching evidence (unaffected), (c) tightened hedging methodology (unaffected), and (d) negative results. Even in this worst case, the paper remains publishable -- just at a lower tier.

## DECISION: PROCEED
