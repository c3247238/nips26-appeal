# Critique: Experiments

## Summary Assessment
The Experiments section (Section 5) presents a well-organized, four-part analysis covering single-method Pareto curves, pairwise composition, three-way composition, and IGSD ablation. The data-driven narrative is generally strong: claims are backed by specific numbers, the composition taxonomy (near-orthogonal / task-dependent / destructive interference) is clearly illustrated, and the honest handling of negative results (M1+M3 interference, d2Cache failure, KL profile refuting H6) is commendable. However, the section has significant issues with baseline inconsistencies across experiments, missing or incomplete cross-verification against raw data, and several claims that lack the statistical rigor the paper's own protocol demands.

## Score: 6/10
**Justification**: The section scores well on structure, narrative clarity, and honesty about negative results. It loses points for: (a) using different baselines across subsections without transparent reconciliation, (b) reporting pairwise results from 100-sample single-seed pilots alongside three-way results from 3-seed validation without flagging the unequal statistical power, (c) missing a planned figure (Figure 7), and (d) several numerical inconsistencies between the text and raw JSON artifacts. Reaching 7 requires fixing the baseline confusion, adding confidence intervals to pairwise results, and restoring Figure 7.

## Critical Issues

### Issue 1: Inconsistent baseline TPS across experiments
- **Location**: Section 5.1, opening paragraph and throughout all subsections
- **Quote**: "LLaDA-8B-Instruct baseline ($T$=64 steps, bf16 greedy decoding; GSM8K accuracy 71.2% +/- 1.5%, 33.8 TPS"
- **Problem**: The section states the baseline GSM8K TPS is 33.8, and the full-scale M1 experiment (`m1_pareto_full.json`) uses a baseline of 31.013 TPS (1319 samples). However, all pairwise experiments (`m1_igsd_full.json`) and three-way experiments (`threeway_pareto_full.json`) use a pilot baseline of 58.505 TPS (100-sample subset). The method section states "GSM8K: 71.2% +/- 1.5% accuracy, 33.8 TPS." Yet the M1 full-scale data computes speedup from 31.013 TPS, and all composition experiments compute speedup from 58.505 TPS. This means the **speedup numbers reported in different subsections are computed against different denominators**: M1's 1.16x is against 31.013 TPS while IGSD's 2.81x and all pairwise/three-way speedups are against 58.505 TPS. The discrepancy is never disclosed or reconciled, and it undermines cross-subsection comparisons (e.g., comparing M1's 1.16x speedup from Table 3 with the pairwise composition speedups in Table 4).
- **Fix**: (1) Disclose which baseline each subsection uses, explaining why (full-scale vs. pilot TPS differ due to prompt length distribution effects in subsampling). (2) Re-compute all speedups relative to a single consistent baseline, or (3) add a reconciliation paragraph explaining the baseline discrepancy and its impact on cross-table comparisons. Given that the pairwise/three-way Ortho metric normalizes by individual QAS and is thus internally consistent, the Ortho results are likely valid regardless, but absolute speedup values in Table 3 vs. Tables 4-5 are not directly comparable.

### Issue 2: Pairwise results lack multi-seed validation
- **Location**: Section 5.2, all three pairwise analyses
- **Quote**: "100-sample GSM8K and 100-sample MATH500 pilots (seed 42)"
- **Problem**: The section reports pairwise orthogonality from single-seed, 100-sample pilot experiments, yet the three-way results in Section 5.3 use 3-seed validation with stability criteria (QAS CV < 10%). The paper's own experimental protocol (Section 4.5) specifies "Pilot experiments use seed 42; full-scale experiments use seeds {42, 123, 456}." The pairwise results are pilot-scale by this definition, but the section presents them with the same confidence as the three-way results. The M1+IGSD combined Ortho of 0.96 and M1+M3 Ortho of 0.41 could shift materially with additional seeds -- the discussion section itself acknowledges that iter_001's M1+M3 Ortho=1.34 was "an artifact of small-sample variance."
- **Fix**: Either (a) run 3-seed validation for the most important pairwise configs (at minimum M1+IGSD best and M1+M3 best) and report mean +/- std, or (b) add explicit caveats on every pairwise Ortho number stating it is from a single-seed 100-sample pilot, and flag this as a limitation directly in Section 5.2 rather than only in 7.6.

### Issue 3: MATH500 Ortho values are unreliable but drive key verdicts
- **Location**: Table 4, M1+IGSD row
- **Quote**: "MATH500 Ortho=0.64" (Table 4), combined Ortho "0.96"
- **Problem**: The MATH500 baseline is 0.32 exact match on 100 samples (32 correct), meaning a 3-sample fluctuation changes accuracy by nearly 1 percentage point and Ortho by a large margin. The combined metric weights MATH500 at 30%, so MATH500 Ortho=0.64 drags the combined Ortho from 0.99 (GSM8K-only) to 0.96. While the combined is still "near-orthogonal," the MATH500 number itself may be noise. Worse, the M3+IGSD verdict of "task-dependent" (Ortho_combined=0.84, with MATH500 Ortho=0.76) rests on an even thinner statistical foundation. The paper does not report confidence intervals for any pairwise Ortho values.
- **Fix**: (1) Report bootstrap or analytical confidence intervals for all Ortho values, especially those from 100-sample pilots. (2) Flag MATH500 Ortho explicitly as having wide uncertainty bands. (3) Consider whether the "task-dependent" verdict for M3+IGSD is robust to the expected variance on 100 MATH500 samples.

## Major Issues

### Issue 4: Missing Figure 7 (KL divergence profile)
- **Location**: Section 5.4, "KL divergence profile" paragraph
- **Quote**: "Measured across 100 GSM8K samples, the per-step KL(p_t || p_{t-1}) profile is monotonically decreasing..."
- **Problem**: The outline (Section 5.4 entry) specifies Figure 7 as the per-step KL divergence profile visualization. The data file `igsd_kl_profiles_raw.json` exists. However, the experiments section never references Figure 7 -- the KL profile is described in text only, with no visual element. This is the only planned figure from the outline that is missing from the section. The claim that the KL profile is "monotonically decreasing with high variance" and "spiky behavior with occasional values exceeding 0.5" in later steps is hard for readers to evaluate without the visualization.
- **Fix**: Add `![Figure 7: Per-Step KL Divergence Profile](figures/fig7_kl_profile.pdf)` and reference it in the KL divergence paragraph. This is the primary evidence for refuting hypothesis H6 and explaining tau insensitivity -- it deserves a figure.

### Issue 5: Confidence gate ablation shows zero effect but the section understates this
- **Location**: Section 5.4, "Confidence gate ablation" paragraph
- **Quote**: "tau=0.0 achieves 49.5% accuracy on GSM8K versus 49.5% for tau=0.9, a negligible 0.0 pp difference."
- **Problem**: The text says "The confidence gate adds at most marginal quality gain." In fact, the data shows **exactly 0.0 pp difference** -- not "at most marginal" but "no measurable effect whatsoever." This is a stronger and more surprising finding than the hedged language suggests. The confidence partitioning mechanism is a core component of IGSD (the "I" in Information-Geometric), and the ablation shows it contributes nothing at T_draft=32. The section should explicitly state what this means for the IGSD algorithm: at T_draft=32, IGSD reduces to naive step reduction with no benefit from the confidence gate.
- **Fix**: Rewrite to directly state: "At T_draft=32, tau=0.0 and tau=0.9 produce identical GSM8K accuracy (49.5%), demonstrating that confidence partitioning provides zero measurable benefit at this operating point. IGSD at T_draft=32 is functionally equivalent to naive step reduction." Then discuss why: the monotonic KL profile means later-step refinement adds little regardless of which tokens are refined.

### Issue 6: Three-way composition does not beat pairwise, but the section buries this finding
- **Location**: Section 5.3, final paragraph
- **Quote**: "Three-way compositions do not extend the pairwise Pareto frontier on GSM8K."
- **Problem**: This is arguably the most important finding of the three-way analysis, but it appears only in the last paragraph as a passing observation. The best pairwise QAS (M1+IGSD, td=16: QAS=1.34 from the JSON data, reported as 1.34 in the text) exceeds the best three-way QAS (1.07). In other words, adding M1 to the three-way mix at T_draft=32 does not compensate for using the more conservative draft setting. The section presents five three-way configs in Table 5 before arriving at this conclusion, which may leave readers expecting that three-way compositions offer something new. The logical structure should lead with this finding or at least flag it prominently.
- **Fix**: Open Section 5.3 with the key finding: "Three-way composition does not extend the pairwise Pareto frontier established by M1+IGSD." Then present the five configs as supporting evidence, noting that the three-way study's value is in confirming near-orthogonal Ortho at the three-way level and in definitively ruling out M3 guidance as a composition layer.

### Issue 7: Table 3 M3 rows omit MATH500 data
- **Location**: Table 3
- **Problem**: Table 3 is titled "Single-Method Pareto Results on GSM8K" but the outline (Table 3 specification) says the table should include "Method x {Speedup, AccRet, QAS, CHR/AcceptRate} on GSM8K + MATH500." MATH500 results are entirely absent from Table 3. For the M1 and IGSD rows this may be a space decision, but the missing MATH500 data makes it impossible for readers to verify the combined metric claims (0.7 * GSM8K + 0.3 * MATH500) directly from the table.
- **Fix**: Either (a) add MATH500 columns to Table 3, or (b) split into two tables (GSM8K and MATH500), or (c) add a companion table in the appendix and reference it. At minimum, MATH500 AccRet values should appear since they feed into all combined Ortho calculations.

### Issue 8: M3 "full" evaluation is actually on 100-sample subsets
- **Location**: Section 5.1, M3 paragraph
- **Quote**: "The 2-seed mean (seeds 42, 123) on a 100-sample GSM8K subset confirms stability."
- **Problem**: The M3 results in Table 3 are presented alongside M1 results that use 1319 GSM8K samples (full scale). The text does not clearly flag that M3 speedup (1.68x) and AccRet (103.9%) are from 100 samples, not from the 1319-sample evaluation used for M1. A reader comparing M1 (eta=0.5, 1.16x) to M3 (gw=0.3, 1.68x) may not realize these numbers come from different sample sizes. The IGSD Pareto results are also from 200 GSM8K + 100 MATH500 pilot samples, not full scale.
- **Fix**: Add a "Scale" or "N" column to Table 3 indicating the sample count for each row, or add a table footnote clarifying which rows are full-scale (1319 GSM8K) versus pilot-scale (100--200 GSM8K).

## Minor Issues

- **Section 5.1, M1 paragraph**: "Projected speedups from published cache hit rates would be 2.27--2.47x" -- the text says "projected from published cache hit rates" but the CHR values are measured in *this* paper, not published elsewhere. Reword to "projected from the measured cache hit rates."
- **Section 5.1, M3 paragraph**: Claims M3 speedup is 1.68x "across all guidance weights." The raw data shows speedups of 1.678x, 1.676x, 1.676x, 1.676x for gw=0.3, 0.5, 0.7, 1.0. These round to 1.68x, but it would be more precise to say "approximately 1.68x" rather than implying exact identity.
- **Table 3**: Bold formatting is used inconsistently. "1.50x" (M1 eta=2.0) and "2.81x" (IGSD tau=0.7, td=16) are bolded for being the fastest per method, but "103.9" (M3 gw=0.3 AccRet) is also bolded despite M3 gw=0.7 achieving the same value. The QAS column bolds 1.64 and 1.69/1.71 but the basis for which values to bold is unclear.
- **Table 4**: The "Best Config" for M1+IGSD lists "td=16" as shorthand for T_draft=16, but earlier text uses "$T_{\text{draft}}$=16". Use consistent notation throughout tables.
- **Table 5**: "Balanced-B" and "Max-Speed" have identical numbers (speedup 1.71, AccRet 62.7%, QAS 1.07, Ortho 1.02, QAS CV 8.2%). These differ only in eta (0.5 vs 1.0). The section text does not discuss Balanced-B at all. Either explain why two distinct recipes produce identical results or consolidate them.
- **Section 5.2, M1+M3 paragraph**: "TPS from 58.5 (baseline) to 50.3 (0.86x)" -- 50.3 / 58.5 = 0.860, confirmed. But 58.5 TPS is the pilot baseline, not the 33.8 TPS stated in the section opening. This reinforces Critical Issue 1.
- **Section 5.3**: "The highest two-way QAS (M1+IGSD, td=16: QAS=1.34)" -- this QAS=1.34 is from the combined metric. It would be clearer to specify "combined-metric QAS=1.34" to distinguish from per-benchmark QAS.
- **Section 5.4**: "T_draft=48 gives QAS=0.90 at 1.22x speedup and 73.3% accuracy retention" -- but the outline says "T_draft=48 gives QAS=1.05 (1.44x, 72.7%)." These numbers differ substantially (QAS 0.90 vs 1.05, speedup 1.22x vs 1.44x). The section appears to use the Pareto data from the original IGSD sweep while the outline uses ablation-specific data. This inconsistency between outline and section should be resolved -- verify which data source is authoritative.
- **Section 5.4, tau sweep**: "tau=0.7 gives QAS=1.17; tau=0.85 gives QAS=1.17; tau=0.9 gives QAS=1.16" -- accuracy retention "ranges from 66.4% (tau=0.7) to 67.8% (tau=0.85 and 0.9)." The claim that "IGSD is insensitive to tau in the [0.7, 0.9] range" is supported, but the 1.4 pp accuracy difference between tau=0.7 and tau=0.9 is unreported variance territory at 200 samples. A sentence acknowledging this would strengthen the claim.
- **Figure 3 placement**: Figure 3 appears *after* the M3 discussion but *before* Table 3. Standard practice is to reference the figure and then present it. The current ordering works but Figure 3 and Table 3 should be more tightly coupled with explicit "see Figure 3 and Table 3" callouts in the M1/IGSD/M3 paragraphs.

## Visual Element Assessment
- [x] Figures/tables match outline plan (Figures 3, 4, 5, 6 and Tables 3, 4, 5 present)
- [ ] **Figure 7 (KL divergence profile) is missing** -- planned in outline, data exists, not referenced or included
- [x] All present visuals referenced before appearance
- [x] Table captions present (as bold text above tables)
- [ ] Missing visual: A breakdown of M3 data on MATH500 would strengthen the "task-dependent" discussion
- [ ] Table 3 could benefit from a MATH500 companion or additional columns

## What Works Well
1. **Honest treatment of negative results.** The M1+M3 interference analysis (Section 5.2) is exemplary: it identifies the root cause (speed penalty from guide model loading), quantifies it precisely (12% overhead, 0.86x TPS), and reconciles the discrepancy with the iter_001 pilot finding. The explicit correction of the earlier synergy claim (Ortho=1.34 was an artifact) builds reader trust.
2. **IGSD ablation structure.** Section 5.4 systematically varies one parameter at a time (T_draft sweep at fixed tau, tau sweep at fixed T_draft, then confidence gate ablation), producing clean causal attribution. The finding that IGSD at T_draft=32 is Pareto-optimal is well-supported by the dual evidence of QAS comparisons and the monotonic KL profile.
3. **Three-way validation rigor.** The 3-seed protocol for three-way compositions with the QAS CV < 10% stability criterion is a strong methodological choice that should be highlighted more prominently, as it distinguishes the three-way results from the less rigorous pairwise pilots.
