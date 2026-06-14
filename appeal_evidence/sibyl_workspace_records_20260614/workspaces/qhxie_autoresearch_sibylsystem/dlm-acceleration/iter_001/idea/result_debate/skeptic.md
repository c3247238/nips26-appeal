# Skeptic Analysis: ComposeAccel Results (Round 2)

## Context

This analysis follows the Round 1 verdict (score 6/10) and the revised proposal. The proposal has been updated to reposition as an analysis paper, acknowledge SSD/SSMD as prior work, rename IGSD to CD-SSD, correct the abstract, and identify mandatory follow-up experiments. This round re-examines the data with maximum skepticism, checking whether the revisions adequately addressed prior concerns and surfacing new issues revealed by deeper quantitative scrutiny.

---

## Step 1: Statistical Risk Inventory

### Risk 1 (PERSISTS, SEVERITY INCREASED): The Headline Synergy Number Has No Confidence Interval

The central claim — M1+IGSD Ortho=1.385, "super-multiplicative synergy" — still rests on:
- **200 GSM8K samples** (15.2% of the 1319-sample full benchmark)
- **164 HumanEval samples** where all methods score pass@1=0.0 (degenerate — contributes only to speedup, not quality)
- **2 seeds** (42, 123)

Per-seed Ortho values: seed 42 = 1.292, seed 123 = 1.478. The range is 0.186, which is 13.4% of the mean. With n=2, no standard error, no confidence interval, no significance test is possible. The proposal acknowledges this and lists "full-scale M1+IGSD (3 seeds, all 4 benchmarks)" as Priority 1 at ~8 GPU-hours. **This experiment has not been run.** Until it is, the synergy claim is a point estimate from a convenience sample.

Deeper concern: the pairwise "combined speedup" is the mean of GSM8K speedup and HumanEval speedup. Since HumanEval accuracy is 0% for all methods, the HumanEval component of the combined metric reflects only throughput differences on output sequences where the model produces garbage. Seed 42 GSM8K speedup for M1+IGSD is 6.66x; seed 123 is 6.69x — consistent. But HumanEval speedup is 3.26x for both seeds, which is lower because HumanEval generates longer output sequences (avg ~120 tokens vs ~50 for GSM8K). The "combined" 5.13x is an arithmetic artifact of mixing two incommensurable quantities (useful speed on GSM8K and garbage-throughput speed on HumanEval).

**Specific concern (new)**: In `full_pairwise_ortho.json`, the GSM8K accuracy for M1+IGSD is 0.39 (seed 42) and 0.445 (seed 123). The full baseline GSM8K accuracy is 0.712 (3-seed mean). So M1+IGSD retains only 54.8%–62.5% of GSM8K accuracy. The "super-multiplicative synergy" in speedup comes at 37-45% accuracy destruction on the only reliable benchmark. **Ortho measures speedup composition only; it says nothing about whether the result is usable.**

**Severity: Fatal flaw (unchanged).** The paper's headline metric is still unvalidated at full scale.

### Risk 2 (PERSISTS, PARTIALLY ADDRESSED): Degenerate Coding Benchmarks Contaminate All Aggregate Metrics

The proposal now states "Coding metrics are reported but not used as primary evaluation criteria" and acknowledges HumanEval 2.4% and MBPP 0.0% baselines. However, the summary tables in `summary.md` still report "Combined Acc Retention" and "Combined QAS" that average across all 4 benchmarks, including the degenerate ones. For example:

- IGSD Combined AccRet = 70.3% (Table in summary.md line 109). But the per-benchmark breakdown shows: GSM8K 63.7%, MATH500 88.5%, HumanEval 0.0%, MBPP 100% (by convention). This average is meaningless — it mixes a real signal (GSM8K, MATH500) with two fake signals (HumanEval always=0, MBPP always=1.0).

- M1+IGSD avg_acc_ret = 0.322 (from `full_pairwise_ortho.json`). This is the mean of GSM8K acc_ret (~0.59) and HumanEval acc_ret (0.0). The "32.2% accuracy retention" headline understates performance on GSM8K while including a completely uninformative zero.

The proposal's revision says "Primary claims are based on GSM8K and MATH500" but the actual data artifacts and summary tables do not reflect this. All JSON files still compute "combined" metrics over 4 benchmarks.

**Severity: Serious concern.** The tables need to be recomputed with reasoning-only aggregation before any publication-ready claim can be stated.

### Risk 3 (PERSISTS, NOW ACKNOWLEDGED BUT UNRESOLVED): tau=0.0 Paradox Threatens the Method

The ablation data (`igsd_ablation.json`) shows:
- IGSD-full (tau=0.9, T=16): avg_speedup = 2.664x, avg_acc_ret = 0.359, QAS = 0.956
- IGSD-no-partition (tau=0.0): avg_speedup = 5.558x, avg_acc_ret = 0.324, QAS = 1.801

The proposal explicitly calls this "Critical Pre-Submission Experiment" (Priority 1) and proposes comparing tau=0.0 against naive T=16 uniform denoising. **This experiment has not been run.**

Deeper scrutiny reveals an alarming detail: the ablation's IGSD-full (tau=0.9, T=16) numbers **disagree with the full Pareto results**. The ablation reports QAS=0.956 and combined_speedup=2.664x, while the Pareto (`igsd_pareto_full.json`) reports QAS=1.194 and combined_speedup=3.399x for the same tau=0.9, T=16 configuration. The `data_consistency_notes` in `failure_mode_atlas_full.json` explains: "Different benchmark compositions account for the difference." The ablation used 200 GSM8K + 164 HumanEval (2 seeds), while the Pareto used full GSM8K 1319 + MATH500 500 + HumanEval 164 + MBPP 257 (3 seeds).

This means the QAS for "IGSD-full" varies from 0.956 to 1.194 depending on which benchmarks and how many seeds you include — a 25% swing. The tau=0.0 advantage of +88.5% is measured against the ablation's QAS=0.956, not the full Pareto's 1.194. Against 1.194, the advantage shrinks to +50.8%. While still significant, this illustrates how unstable the metrics are when benchmark composition changes. **The tau=0.0 vs. tau=0.9 comparison is being made on a benchmark mix (200 GSM8K + 164 HumanEval) that is 50% degenerate data.**

**Severity: Fatal flaw (unchanged).** The core method's signature component appears net-negative, and the magnitude of the effect depends on which (partially degenerate) benchmark mix is used.

---

## Step 2: Alternative Explanations

### Claim: M1+IGSD achieves super-multiplicative synergy (Ortho=1.385)

**Alternative 1 (refined): The "synergy" is an accounting identity, not a mechanism.**

Consider the causal chain:
1. IGSD runs T_draft=16 steps, producing output with ~52% high-confidence tokens.
2. The REFINE phase runs T_full=64 steps on the remaining ~48% of tokens, with frozen tokens held constant.
3. M1 (EntropyCache) caches KV entries. Frozen tokens have perfectly predictable entropy (zero change), so cache hit rate approaches 100% for those positions.

The paper explains this as "frozen-token KV amplification mechanism." But consider: the speedup of IGSD alone is 3.40x. The speedup of M1 alone is 1.38x. The combined speedup is 5.13x. The expected independent product is 4.69x. The "super-multiplicative" excess is 5.13/4.69 = 1.094 — only 9.4% above independence. This is within the measurement uncertainty given n=2 seeds and the 13.4% spread in per-seed Ortho values.

**Reframing**: The Ortho=1.385 metric means the combined speedup is 38.5% above the product of individual speedups. But M1 alone already achieves only 1.38x (slower than published EntropyCache by 10x, as acknowledged). If M1 were implemented at published performance (15x+), the product M1 x IGSD would be 15 x 3.4 = 51x — and the actual combined speedup would almost certainly not reach 51x. The "super-multiplicativity" may be an artifact of M1's underperformance: when M1 alone barely works (1.38x), any improvement from context-dependent caching appears as a multiplicative bonus.

**Alternative 2 (new): Selection bias in the 200-sample GSM8K subset.**

The pairwise experiments were run on 200 out of 1319 GSM8K problems. The sampling method is not documented in any results file. If these 200 problems were the first 200 (by index), they may have different difficulty distribution than the full benchmark. Problems at the beginning of GSM8K tend to be simpler, with shorter solutions and more repetitive reasoning patterns — exactly the conditions where KV-caching would be most effective. Without documentation of the sampling procedure (random? stratified? sequential?), we cannot rule out selection bias.

### Claim: The composability landscape is binary (synergy vs. interference, no middle ground)

**Alternative: Insufficient sampling of the landscape.** With M2 eliminated as NO_GO, only 3 viable method pairs were tested: M1+IGSD, M3+IGSD, M1+M3. Three data points cannot establish that a landscape is "binary." The true landscape might include intermediate Ortho values for method combinations not tested (e.g., d2Cache+IGSD, Elastic-Cache+IGSD, SPA-Cache+IGSD, SSD+M1). Calling the landscape "binary" based on 3 observations is a dramatic overstatement. More precisely: **of 3 tested pairs, 1 synergizes and 2 interfere.** That is the fact. "Binary landscape" is an inference from an n=3 sample.

### Claim: M3 improves reasoning accuracy by +3.9% on GSM8K

**Alternative (strengthened): This is within sampling noise of n=100.**

From `m3_pareto_full.json`: M3 with guidance_weight=0.3, GSM8K results:
- Seed 42: exact_match = 0.73, n=100
- Seed 123: exact_match = 0.75, n=100

Full baseline GSM8K (3-seed mean): exact_match = 0.712 (1319 samples per seed).

The 95% confidence interval for a binomial proportion at p=0.73, n=100 is approximately [0.64, 0.81]. The baseline 0.712 falls comfortably within this interval. The "improvement" of 0.73 - 0.712 = 0.018 (seed 42) to 0.75 - 0.712 = 0.038 (seed 123) on n=100 samples is **not statistically significant** by any standard test. The claim that "M3 guidance improves reasoning accuracy" is not supported.

Furthermore, the M3 Pareto was run on only **100 GSM8K samples** and **50 MATH500 samples** per seed, with only **2 seeds** (42, 123). The MATH500 "accuracy retention of 243.9%" comes from baseline 0.1107 vs. M3 achieving ~0.27 on 50 samples — where even 1 additional correct answer changes accuracy by 2 percentage points. This is severely underpowered.

---

## Step 3: Proxy Metric Audit

### QAS = Speedup x Accuracy_Retention

**Problem 1 (persists): QAS is Pareto-insensitive to catastrophic accuracy loss.**

The best QAS for IGSD-no-partition (tau=0.0) is 1.801, achieved with 32.4% accuracy retention. In absolute terms: GSM8K accuracy drops from 71.2% to ~23%. No deployment scenario would accept a method that changes a model from "reasonably good at math" to "fails 3 out of 4 problems." Yet QAS reports this as the best configuration.

The proposal's QAS formula includes a "0.5x penalty for >5% GSM8K drop" for IGSD at tau=0.9 (penalized QAS = 3.40 x 0.703 x 0.5 = 1.194). But this penalty is: (a) ad hoc — why 0.5? Why not 0.1 or 0.0? (b) inconsistently applied — M1 at t=2.0 has 45% GSM8K accuracy drop (summary.md line 167: gsm8k_acc_drop = 0.450) but no penalty is applied (QAS=0.836). M1's penalty-free QAS actually makes M1 look worse than penalized IGSD, obscuring that M1 retains 55% accuracy vs. IGSD's 35%.

**Problem 2 (new): Ortho and QAS measure orthogonal qualities (no pun intended).**

Ortho = Speedup(A+B) / (Speedup(A) x Speedup(B)). It measures *speedup composition only*.
QAS = Speedup x AccRet. It measures *quality-adjusted single-method performance*.

Neither metric captures *quality-adjusted composition*. Consider a "Quality-Ortho" = QAS(A+B) / max(QAS(A), QAS(B)):
- M1+IGSD: QAS(A+B)=1.654, max(QAS(M1), QAS(IGSD))=1.194. Quality-Ortho = 1.385. Looks good.
- But QAS(A+B)=1.654 is achieved at 32.2% accuracy retention. Compare to IGSD alone at 3.40x speedup with 70.3% retention (QAS=1.194). The composition doubles speed but halves quality. Is that "synergy"?

**Problem 3 (new): The Ortho calculation uses inconsistent baseline TPS.**

From `full_pairwise_ortho.json`:
- M1+IGSD seed 42: gsm_speedup = 6.662x. This is computed as avg_tps / baseline_tps.
- M1+IGSD avg_tps on GSM8K = 206.6 tok/s (seed 42).
- Baseline GSM8K avg_tps for seed 42 = 25.35 tok/s (from full_baseline).

But the M1 standalone GSM8K speedup at t=2.0 (from m1_pareto_full.json, seed 42) = 1.496x, based on avg_tps = 46.37 vs. baseline avg_tps = 31.01. The baseline TPS differs: 25.35 vs. 31.01 across experiments. This 22% discrepancy in baseline TPS between the pairwise experiment and the single-method experiment means the Ortho calculation is comparing speedup ratios computed against different denominators. If the pairwise experiment's lower baseline TPS (25.35) inflates the M1+IGSD speedup, the resulting Ortho would be artificially high.

The root cause is likely batch size, warmup, or GPU thermal state differences between experimental sessions. But it means: **the Ortho score is not a clean ratio because the individual speedup measurements come from different experimental sessions with different baseline conditions.**

**Severity: Serious concern (new).** The Ortho calculation requires consistent baseline measurements across all compared configurations.

---

## Step 4: Severity Classification

| # | Issue | Severity | Status vs. Round 1 |
|---|-------|----------|---------------------|
| 1 | Pairwise synergy rests on 200 samples / 2 seeds — mandatory P1 experiment not yet run | **Fatal flaw** | PERSISTS (acknowledged, experiment pending) |
| 2 | tau=0.0 paradox — core method component appears net-negative; decisive experiment not run | **Fatal flaw** | PERSISTS (acknowledged, experiment pending) |
| 3 | Baseline TPS inconsistency across experimental sessions (25.35 vs 31.01 for GSM8K seed 42) inflates Ortho | **Serious concern** | NEW |
| 4 | "Binary composability landscape" claim from n=3 pairs — dramatic overstatement | **Serious concern** | NEW |
| 5 | Coding benchmarks (HumanEval 2.4%, MBPP 0.0%) still contaminate all "combined" metrics in data artifacts | **Serious concern** | PERSISTS (acknowledged, not yet fixed in data files) |
| 6 | M3 accuracy improvement on GSM8K (n=100, 2 seeds) is not statistically significant | **Serious concern** | PERSISTS (flagged for full-scale replication) |
| 7 | M1 implementation is 10x slower than published EntropyCache — biases the entire composability analysis | **Serious concern** | PERSISTS (acknowledged, unexplained) |
| 8 | QAS penalty (0.5x for >5% GSM8K drop) is ad hoc and inconsistently applied | **Serious concern** | NEW (partially) |
| 9 | "Super-multiplicative" excess is only 9.4% above independence, within measurement noise of n=2 | **Serious concern** | NEW (deeper quantification of existing concern) |
| 10 | M2 simplified implementation — NO_GO verdict may not apply to real Saber | **Minor caveat** | PERSISTS |
| 11 | Single model (LLaDA-8B) — Dream-7B unavailable; generalizability untested | **Minor caveat** | PERSISTS |
| 12 | Batch size = 1 throughout; real deployment uses batched inference | **Minor caveat** | PERSISTS |

---

## Step 5: Concrete Remediation

### For Fatal Flaw #1: Full-Scale Pairwise Validation

**Experiment**: M1+IGSD on full GSM8K (1319) + full MATH500 (500), 3 seeds [42, 123, 456]. Use IGSD tau=0.9, T_draft=16; M1 entropy_threshold=2.0.

**Protocol requirement**: Run baseline on the same session/GPU thermal state as the pairwise experiment to eliminate TPS denominator inconsistency. Record baseline TPS within the same script run (before/after warmup), not from a separate experimental session.

**Success criterion**: Ortho mean >= 1.0 with std < 0.15 across 3 seeds. If std > 0.15, the claim must be hedged to "directionally super-multiplicative but not conclusively demonstrated."

**Expected outcome**: Ortho will likely decrease from 1.385 toward 1.1-1.2 when measured on the harder full distribution (the 200-sample subset may over-represent easy problems). It should remain above 1.0 based on the mechanistic argument, but the magnitude of super-multiplicativity will likely shrink.

### For Fatal Flaw #2: tau=0.0 Resolution

**Experiment A**: Run naive T=16 uniform denoising (no IGSD machinery — just LLaDA with num_steps=16 instead of 64). Compare accuracy and throughput against IGSD tau=0.0 on the same 200 GSM8K + 164 HumanEval with seeds [42, 123].

**Experiment B**: Run M1 + naive-T16, M1 + IGSD(tau=0.0), and M1 + IGSD(tau=0.9). This clarifies whether the M1 synergy requires IGSD's partition mechanism or just step reduction.

**Decision tree**:
- If naive-T16 accuracy approximately equals IGSD-tau=0.0 accuracy (within 3pp on GSM8K): IGSD's acceptance gate adds no value. Report as negative finding. Reframe the paper: the "method" is simply reduced-step denoising, and the contribution is the composability analysis.
- If naive-T16 accuracy is substantially worse than IGSD-tau=0.0 (>5pp on GSM8K): IGSD's coarse-draft mechanism provides residual quality benefit even without partition. The method is salvageable.
- If M1 + naive-T16 achieves Ortho >= 1.2: the synergy is a general property of step reduction + KV-caching, not specific to IGSD. Stronger generalization claim.

### For Serious Concern #3: Baseline TPS Consistency

**Action**: In every experimental script, record an in-session baseline measurement (at least 50 samples with the same batch size, seed, and warmup protocol) immediately before applying acceleration methods. Report speedup relative to this in-session baseline, not a cross-session reference. Re-examine whether the 22% TPS discrepancy (seed 42: 25.35 in full_baseline vs. 31.01 in pairwise reference) changes the Ortho calculation.

**Quick check**: Recompute M1+IGSD Ortho using the full_baseline per-seed TPS as the denominator instead of pairwise-session TPS. If Ortho drops below 1.0, the synergy claim is invalidated. If it remains above 1.2, the claim survives.

### For Serious Concern #4: "Binary Landscape" Overclaim

**Action**: Soften the language. Replace "MDM acceleration composability is binary" with "Among the three viable pairs tested, composability exhibited a stark pattern: one strong synergy and two clear cases of interference, with no intermediate regime observed." The paper should explicitly note that only 3 of the originally planned 6+ pairwise combinations were testable (due to M2 elimination and model unavailability), and that the landscape characterization is preliminary.

If the SSD+M1 experiment (Priority 4 in the proposal) shows intermediate Ortho (0.6-0.9), the "binary" claim is falsified. Account for this possibility in the writing.

### For Serious Concern #6: M3 Accuracy Improvement Validation

**Experiment**: Run M3 (gw=0.3) on full GSM8K (1319 samples) with 3 seeds. Compute paired test (McNemar's test or bootstrap) against baseline.

**Decision**: If the improvement is not significant at p<0.05, do not claim "M3 improves reasoning accuracy." Instead report: "M3 maintains baseline accuracy while providing modest speedup on reasoning tasks."

### For Serious Concern #9: Quantifying Super-Multiplicative Significance

**Analysis**: With the full-scale data from remediation #1 (3 seeds), compute: (a) Ortho for each seed; (b) mean and standard deviation; (c) one-sample t-test or bootstrap CI for H0: Ortho=1.0. If the 95% CI for Ortho includes 1.0, the claim must be downgraded from "super-multiplicative" to "approximately multiplicative (consistent with independence)."

---

## Summary Judgment (Round 2)

The proposal has improved significantly since Round 1. The abstract has been corrected (no more "20-30x" claims), the paper is being repositioned as an analysis contribution, SSD/SSMD are acknowledged as concurrent work, and the critical experiments are correctly identified. The team is being honest about limitations.

However, **the two fatal flaws from Round 1 remain unresolved because the decisive experiments have not been run**:
1. The pairwise synergy number needs full-scale validation with consistent baselines.
2. The tau=0.0 paradox needs resolution against a naive T=16 baseline.

Additionally, Round 2 analysis reveals:
3. A baseline TPS inconsistency (22% between sessions) that could inflate the Ortho calculation.
4. The "super-multiplicative" excess is only 9.4% above independence — possibly within noise.
5. The "binary landscape" claim is based on 3 data points, which is insufficient for a structural conclusion.

**Bottom line**: The composability framework and failure-mode atlas remain genuinely valuable contributions. The findings about M2 incompatibility and the M1+IGSD interaction are interesting and directionally believable. But the quantitative headline ("super-multiplicative synergy, Ortho=1.385") is currently standing on a statistical foundation that cannot support it. The mandatory experiments identified in the proposal's own action plan are exactly the right ones. Until they are completed, any paper claims about the specific Ortho value or the "binary landscape" are premature.

**Recommendation for the writing phase**: Frame all quantitative synergy claims as preliminary estimates pending full-scale validation. Lead with the qualitative insight (frozen-token KV amplification mechanism) and the failure-mode atlas (which are robust to sample-size concerns). Reserve strong quantitative claims for after the P1-P3 experiments are completed and analyzed.
