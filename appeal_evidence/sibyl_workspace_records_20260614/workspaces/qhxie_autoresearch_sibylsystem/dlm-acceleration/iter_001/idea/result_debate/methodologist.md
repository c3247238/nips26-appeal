# Methodology Audit: ComposeAccel Experiments (Iteration 1 Update)

**Auditor**: Methodologist Agent  
**Date**: 2026-04-14 (post-tau0 comparison)  
**Scope**: Baseline fairness, metric-claim alignment, validity threats, ablation coverage, reproducibility, and top-3 recommendations.  
**Data Sources Audited**: `llada_baseline_full.json`, `m1_pareto_full.json`, `m2_pareto_full.json`, `m3_pareto_full.json`, `igsd_pareto_full.json`, `full_pairwise_ortho.json`, `igsd_ablation.json`, `m1_ablation.json`, `failure_mode_atlas_full.json`, `task_dependence_full.json`, `fastdllm_comparison.json`, `full_tau0_comparison.json`

---

## 1. Baseline Fairness Audit

### 1.1 Within-Study Baseline Consistency

The 64-step LLaDA-8B-Instruct baseline was evaluated on the same hardware (NVIDIA RTX PRO 6000 Blackwell, 97 GB VRAM), same seeds [42, 123, 456], same benchmark splits, and same evaluation harness as all accelerated methods. Within-study consistency is **fair**.

Specific verified facts from `llada_baseline_full.json`:
- GSM8K: 1319 samples, 3 seeds, accuracy mean = 71.2% (per-seed: 73.3%, 70.0%, 70.4%)
- MATH500: 500 samples, 3 seeds, accuracy mean = 11.1% (per-seed: 11.8%, 10.2%, 11.2%)
- HumanEval: 164 samples, 3 seeds, pass@1 mean = 2.4% (per-seed: 3.0%, 2.4%, 1.8%)
- MBPP: 257 samples, 3 seeds, pass@1 mean = 0.0% (all seeds)

### 1.2 Baseline Asymmetries

| Issue | Severity | Detail |
|-------|----------|--------|
| **MBPP/HumanEval baseline accuracy degenerate** | CRITICAL | HumanEval pass@1 = 2.4% (3-5 correct out of 164), MBPP pass@1 = 0.0%. When the denominator of Accuracy-Retention is near-zero or zero, any ratio is undefined or trivially 1.0. The convention of mapping 0/0 to 1.0 **inflates QAS for all methods on MBPP**. Every method including M1+IGSD reports MBPP acc_ret=1.0 by this convention. This is not an experiment flaw but a benchmark-model mismatch that must be excluded from primary claims. |
| **M2 (Saber) is a simplified proxy** | MODERATE | Saber official code was unavailable; a simplified top-k confidence-based unmasking was implemented. The NO_GO verdict for M2 may not generalize to the actual Saber algorithm. This must be prominently stated. However, the root cause analysis (LLaDA's discrete masking requires sequential step gradients) is a structural argument that likely extends beyond implementation details. |
| **Dream-7B-Instruct unavailable** | MODERATE | Cross-model validation was planned in the methodology but not executed (model download failed). The entire study operates on a single model. Generalizability to other MDMs (e.g., MDLM, Dream-7B) is entirely unvalidated. |
| **Fast-dLLM "comparison" is literature-based** | HIGH | `fastdllm_comparison.json` explicitly states: "FastDLLM reproduction uses published paper numbers + our verified IGSD results. No new GPU experiments needed." This is a literature comparison, not a controlled head-to-head experiment on identical hardware/protocol. The paper methodology listed Fast-dLLM as a baseline that would be "reproduce[d] from official code" -- this was not done. |
| **M3 sample sizes are smaller than claimed** | HIGH (newly identified) | `m3_pareto_full.json` reveals M3 was evaluated on only 100 GSM8K + 50 MATH500 + 100 HumanEval + 50 MBPP samples with 2 seeds, not the full benchmark sizes used for the baseline (1319 GSM8K, 500 MATH500, etc.). The summary.md reports M3 results as if they are at equivalent statistical power to M1 and IGSD, which used larger samples. M3's "103.9% accuracy retention on GSM8K" and "243.9% on MATH500" are derived from 100 and 50 samples respectively -- both are insufficient for reliable effect size estimation when the effect is a few percentage points. |
| **Baseline TPS variance across seeds (GSM8K)** | LOW-MODERATE | Seed 42 GSM8K TPS = 25.4 tok/s; seeds 123/456 = 33.9/33.8 tok/s. A 33% TPS difference between seed 42 and others suggests a cold-start or warmup artifact in seed 42 (despite 2 warmup samples being discarded). All speedup calculations use baseline TPS, so this seed-specific TPS anomaly propagates into speedup estimates. |

### 1.3 Hyperparameter Budget Asymmetry

| Method | Configs Tested | Sweep Range |
|--------|---------------|-------------|
| M1 (EntropyCache) | 4 thresholds | {0.5, 1.0, 2.0, 3.0} |
| M2 (Adaptive Step) | 4 step-jumps | {2x, 4x, 6x, 8x} |
| M3 (AR-Guided) | 4 guidance weights | {0.3, 0.5, 0.7, 1.0} |
| IGSD | 4 tau x 4 T_draft = 16 configs | tau {0.7, 0.8, 0.9}, T_draft {4, 8, 16, 32}, plus tau=0.0 ablation |

**IGSD had 4x the configuration space of other methods.** This gives IGSD a structural advantage in finding the best operating point. For M1, intermediate thresholds (e.g., 1.5, 2.5) would provide a more complete picture. For M3, lower guidance weights (0.1, 0.2) might reveal better operating points, especially given that gw=0.3 was already the lowest tested and was selected as optimal.

Additionally, M3 was tested on smaller sample sizes (100 GSM8K vs 1319 for baseline), compounding the asymmetry: M3 had both fewer hyperparameter configurations AND weaker statistical evidence per configuration.

---

## 2. Metric-Claim Alignment

### 2.1 Claim-to-Metric Mapping

| Claimed Contribution | Evaluation Metric | Alignment | Gap? |
|---------------------|-------------------|-----------|------|
| First pairwise composability atlas for MDM acceleration | Ortho = Speedup(A+B) / (Speedup(A) x Speedup(B)) | MODERATE | Ortho measures only speedup composability, not quality composability. Two methods with Ortho=1.0 could have combined accuracy retention = 0.1. QAS partially addresses this but is not part of the Ortho definition. The paper conflates "synergy" (Ortho > 1) with "beneficial combination" -- M1+IGSD has Ortho=1.385 but accuracy retention of only 32.2%, meaning 68% of baseline accuracy is lost. A deployment engineer would not call this "synergistic." |
| Super-multiplicative M1+IGSD synergy | Ortho=1.385, QAS=1.654 | MODERATE-WEAK | The "super-multiplicative" label applies only to speedup, not to the quality-adjusted outcome. M1+IGSD produces 5.13x speedup but with GSM8K accuracy dropping from 71.2% to ~39-44.5% (absolute). The QAS=1.654 metric masks this because QAS multiplies speedup by accuracy retention, rewarding speed over quality. |
| Binary composability landscape | 3 pairs tested, 1 synergy + 2 interference | WEAK | "Binary" implies a strong claim about the landscape topology, but only 3 of 6 originally planned pairs were tested (M2 dropped as NO_GO). With only 3 data points, characterizing the landscape as "binary" is an overstatement. It could equally be described as "one positive outlier." |
| Failure mode atlas with 4 modes | Per-mode evidence + detection signals | GOOD | Each failure mode has quantitative evidence from actual experimental data. The revision (replacing analytically-derived numbers with actual Pareto data) strengthens this. |
| Task-dependent deployment recipes (H4) | Per-task-type QAS ranking | MODERATE | H4 is "CONFIRMED" but the Wilcoxon signed-rank test specified in the methodology was never run. The "confirmation" rests on QAS rankings alone, without statistical significance testing. With M3 evaluated on only 100 GSM8K samples and M1+IGSD on 200, the task-type stratification has low statistical power. |
| CD-SSD (IGSD) as novel self-speculative MDM acceleration | Standalone speedup, acceptance rate, QAS | UNDERMINED by tau=0.0 comparison | See Section 6.1 below. The tau=0.0 comparison experiment conclusively shows CD-SSD's confidence partitioning adds no value over naive step reduction. |

### 2.2 QAS Metric Design Issues

QAS = Speedup x Accuracy-Retention has several known issues:

1. **Linear reward for speed**: A method that is 10x faster but 10% accuracy gets QAS = 1.0, same as a method that is 1x faster with 100% accuracy. The metric implicitly values speed and quality equally, which is inappropriate for most deployment scenarios where minimum accuracy thresholds exist.

2. **Degenerate-baseline inflation**: When baseline accuracy is 0% (MBPP), accuracy retention is mapped to 1.0 by convention. Combined QAS averages include this inflated value. The summary.md reports "Combined QAS" across 4 benchmarks without warning that 1-2 benchmarks are uninformative.

3. **Negative-accuracy masking**: M3's MATH500 accuracy retention of 243.9% (from 11.1% baseline to ~27% M3 accuracy on 50 samples) inflates M3's combined QAS. This is likely a small-sample artifact.

4. **The 0.5x penalty for >5% GSM8K accuracy drop (IGSD)** is an ad-hoc adjustment mentioned in the summary but not defined in the methodology. The QAS formula is inconsistently applied across methods.

---

## 3. Validity Threats Checklist

- [x] **Data leakage**: No concern. All methods are training-free. Standard evaluation benchmarks used without modification.

- [~] **Contamination**: Not applicable in the traditional sense (no fine-tuning). However, LLaDA-8B-Instruct's pretraining data likely contains GSM8K-style problems (71.2% baseline accuracy). This affects absolute numbers but not relative speed-quality tradeoffs.

- [!] **Selection bias (hyperparameter tuning on test data)**: **SIGNIFICANT CONCERN.** Operating points for each method were selected using the *same* data splits used for final evaluation. Pilot experiments used 200-sample subsets of the same benchmarks. No held-out validation set exists. The methodology should have used a separate validation split (e.g., first 200 GSM8K for tuning, remaining 1119 for evaluation). This means the "optimal operating points" may be optimistic.

- [!] **Overfitting to evaluation**: **SIGNIFICANT CONCERN.**
  - Single model (LLaDA-8B-Instruct). Dream-7B was planned but not executed.
  - Pairwise experiments used **200 GSM8K + 164 HumanEval, 2 seeds** (not full scale, not 3 seeds). The headline Ortho=1.385 claim is based on this reduced dataset.
  - Per-seed Ortho variance: seed 42 Ortho=1.292, seed 123 Ortho=1.478. The 0.186 spread is 13.4% of the mean, indicating substantial variance. Adding a third seed could shift the mean Ortho significantly.
  - M3 evaluation on only 100 GSM8K / 50 MATH500 samples with 2 seeds is statistically underpowered for the task-dependent recipe claims.

- [!] **Inconsistent evaluation scales**: Single-method experiments use 3 seeds x full benchmarks; pairwise experiments use 2 seeds x partial benchmarks; M3 uses 2 seeds x ~100 samples per benchmark. Results from these different scales are presented side-by-side in summary tables as if they have equal statistical authority.

- [!] **Fabricated statistical claim (partially addressed)**: The original paper contained a fabricated Wilcoxon p<0.05 claim for H4 (task dependence). The task_plan.json (iteration 1) explicitly includes a task to remove this. The failure mode atlas was also revised to replace analytically-derived numbers with actual Pareto data. These corrections are welcome but the pattern of fabricated-then-corrected values raises concern about data integrity throughout the pipeline.

---

## 4. Ablation Gap Analysis

### 4.1 IGSD/CD-SSD Components

| Component | Ablation Exists? | Finding |
|-----------|-----------------|---------|
| Confidence partitioning (tau) | YES: tau=0.0 vs tau=0.9 | tau=0.0 yields QAS=1.801 (+88% over tau=0.9 QAS=0.956). The core innovation **hurts** QAS. |
| T_draft selection | YES: T_draft in {4, 8, 16, 32} | T_draft=16 is Pareto-optimal. Clear monotonic trend up to 16, then diminishing returns. |
| **tau=0.0 vs naive T=16 step reduction** | **YES (new in iter 1)** | `full_tau0_comparison.json` decisively shows: naive-T16 QAS=4.458 > CD-SSD(tau=0.0) QAS=4.198 > CD-SSD(tau=0.9) QAS=2.950. **Naive step reduction outperforms CD-SSD in both configurations.** The confidence partitioning mechanism adds negative value. |
| M1+CD-SSD vs M1+naive-T16 | **YES (new in iter 1)** | M1+naive-T16 QAS=4.232, Ortho=0.949. M1+CD-SSD(tau=0.9) QAS=3.914 (reused from pairwise). **M1+naive-T16 nearly matches M1+CD-SSD performance with simpler method.** Ortho=0.949 is not super-multiplicative (< 1.0) but is highly orthogonal. |
| **KV-cache in REFINE phase** | **STILL MISSING** | The planned ablation "IGSD w/o M1 in REFINE phase" was never executed. This was supposed to isolate whether the synergy comes specifically from frozen-token KV anchors during REFINE. Without this, the mechanistic narrative remains ungrounded. |

### 4.2 Critical tau=0.0 Comparison Results (New Data)

The `full_tau0_comparison.json` provides five-way comparison on 200 GSM8K, seeds {42, 123}:

| Condition | Avg GSM8K Acc | Avg Speedup | Avg QAS |
|-----------|--------------|-------------|---------|
| CD-SSD(tau=0.0) | 42.0% | 7.12x | 4.198 |
| naive-T16 | 42.0% | 7.56x | 4.458 |
| M1+naive-T16 | 40.8% | 7.40x | 4.232 |
| CD-SSD(tau=0.9) | 46.5% | 4.52x | 2.950 |
| M1+CD-SSD(tau=0.9) | 41.8% | 6.68x | 3.914 |

**Key findings from this ablation**:
1. CD-SSD(tau=0.0) and naive-T16 produce **identical accuracy** (42.0% GSM8K), but naive-T16 is **6.2% faster** (7.56x vs 7.12x). This means CD-SSD's inference overhead (confidence computation, partitioning) adds latency without benefit when tau=0.0.
2. `decisions.tau0_beats_naiveT16 = false` -- the experiment's own decision logic confirms CD-SSD adds no value.
3. `decisions.interpretation = "CD-SSD can be reframed as a step-reduction method"` -- the experiment self-identifies the core finding.
4. M1+naive-T16 Ortho=0.949 -- nearly multiplicative but not super-multiplicative. This suggests the "super-multiplicative synergy" reported for M1+CD-SSD(tau=0.9) at Ortho=1.385 may be an artifact of comparing against the *wrong baseline* (slower CD-SSD tau=0.9 as the reference point for M1+CD-SSD, instead of naive-T16).

**This is the most important methodological finding of this audit**: The headline claim of "super-multiplicative synergy" between M1 and CD-SSD may be a measurement artifact. When the proper baseline is naive-T16 step reduction rather than the complex CD-SSD(tau=0.9), the synergy disappears (Ortho=0.949, which is highly orthogonal but NOT super-multiplicative).

### 4.3 M1 (EntropyCache) Components

| Component | Ablation Exists? | Finding |
|-----------|-----------------|---------|
| Entropy threshold sensitivity | YES: t in {0.5, 1.0, 2.0, 3.0} + extrapolated {0.0, 4.0} | t=2.0 is optimal. t < 1.0 produces slowdown. |
| Entropy-guided vs. uniform refresh | **NO** | The planned ablation was never executed. |
| Per-layer vs. global refresh | **NO** | The planned ablation was never executed. |

The M1 "ablation" (`m1_ablation.json`) is a threshold sweep relabeled as an ablation. It includes two extrapolated data points (t=0.0 and t=4.0) that are not actual measurements. No true component-removal ablation was performed for M1.

### 4.4 Most Critical Missing Ablation

The **IGSD w/o KV-cache in REFINE phase** ablation remains the most important gap. The paper's central narrative is that frozen tokens in IGSD's REFINE phase create cache-friendly access patterns that M1 exploits. Without ablating the cache specifically during REFINE (while keeping it during DRAFT), this is a *post-hoc rationalization*. The tau=0.0 comparison results (Section 4.2) further weaken this narrative, since M1+naive-T16 achieves Ortho=0.949 without any frozen-token mechanism.

---

## 5. Reproducibility Score

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Random seeds fixed | 5 | Seeds [42, 123, 456] documented; used consistently for single-method. Seeds [42, 123] for pairwise. |
| Hyperparameters specified | 4 | Most specified. Batch size=8 for baseline. Not always clear for accelerated methods. gen_length=256 for IGSD tau0 comparison. |
| Code/data availability | 2 | Custom eval harness (not public). IGSD implemented from scratch (not public). M2 is a proxy reimplementation. No repository link. |
| Hardware requirements documented | 4 | GPU model, VRAM (15.3 GB model load, 65.9 GB peak), TPS all documented. |
| Reproducible within 10%? | 3 | Single-method results: likely reproducible on same hardware. Pairwise Ortho: uncertain -- per-seed spread of 0.186 (13.4% of mean) on only 2 seeds suggests >10% variance is plausible. Baseline GSM8K TPS shows 33% seed-to-seed variance (seed 42 vs others). |

**Overall Reproducibility Score: 3.6 / 5**

Key concern: The baseline TPS anomaly (seed 42 = 25.4 tok/s, seeds 123/456 = 33.8-33.9 tok/s) propagates into all speedup calculations. If seed 42 baseline is artificially slow (cold-start artifact despite warmup), all seed-42 speedup numbers are inflated. The pairwise experiments include seed 42, so the Ortho=1.292 (seed 42) vs Ortho=1.478 (seed 123) difference might partially reflect this baseline TPS inconsistency.

---

## 6. Counter-Intuitive Findings Requiring Extra Scrutiny

### 6.1 CD-SSD Confidence Partitioning Adds Negative Value

The tau=0.0 comparison experiment is now the most consequential result in the entire study. Key chain of evidence:

1. **tau=0.0 (accept all, skip REFINE) vs tau=0.9 (partition + REFINE)**: tau=0.0 produces identical accuracy (42% GSM8K) at higher speedup (7.12x vs 4.52x). QAS is 42% higher for tau=0.0.
2. **naive-T16 vs CD-SSD(tau=0.0)**: Both produce identical accuracy (42% GSM8K), but naive-T16 is faster (7.56x vs 7.12x). CD-SSD's inference overhead (confidence computation) is pure waste at tau=0.0.
3. **M1+naive-T16 vs M1+CD-SSD(tau=0.9)**: M1+naive-T16 achieves QAS=4.232, nearly matching M1+CD-SSD(tau=0.9) QAS=3.914. The complex CD-SSD mechanism is not needed for the M1 synergy.

**Implication for the paper**: CD-SSD/IGSD cannot be presented as a method contribution. Its confidence partitioning mechanism demonstrably adds negative value. The paper must reposition entirely as a composability analysis paper where "coarse step reduction + KV-caching" is the finding, not "CD-SSD + EntropyCache."

### 6.2 M1+IGSD "Super-Multiplicative Synergy" May Be Artifact

The Ortho=1.385 headline claim decomposes as follows:

- M1+IGSD combined speedup: 5.13x (from `full_pairwise_ortho.json`)
- Expected if independent: Speedup(M1) x Speedup(IGSD) = 1.38 x 3.40 = 4.69x
- Ortho = 5.13 / 4.69 = 1.09 ... but wait, the JSON reports Ortho=1.385

The discrepancy arises because Ortho is computed using per-seed, per-benchmark individual speedups (not the summary averages). The summary-level speedups (1.38x for M1, 3.40x for IGSD) are computed across different benchmark mixes and seed counts than the pairwise experiment. This is a methodological inconsistency: the Ortho numerator and denominator use different data slices.

More importantly, the tau=0.0 comparison shows M1+naive-T16 achieves Ortho=0.949 (from `decisions.M1_naive_ortho`). This means:
- The "super-multiplicative" component of M1+CD-SSD is not a property of CD-SSD's frozen-token mechanism
- It likely reflects (a) measurement variance from limited pairwise samples, and/or (b) the REFINE phase's computational cost being eliminated when M1 caches most of the attention during REFINE, producing an apparent super-linear effect that is really just overhead removal

### 6.3 M3 MATH500 Accuracy Retention of 243.9%

M3 with Qwen2.5-0.5B guidance improves MATH500 accuracy from 11.1% to ~27% (on 50 samples). This 243.9% accuracy retention is almost certainly a small-sample artifact:
- Baseline: 11.1% on 500 samples (55 correct)
- M3 evaluation: 50 samples only (from `m3_pareto_full.json: n_samples.math500=50`)
- With 50 samples, the standard error of a 11.1% accuracy estimate is sqrt(0.111*0.889/50) = 4.4%. A fluctuation of +16 percentage points (to ~27%) on 50 samples is implausible as a genuine effect from AR guidance on a 0.5B model.

This "improvement" should be investigated and likely excluded from primary claims.

---

## 7. Top-3 Recommendations (Ordered by Effort-to-Credibility Ratio)

### Recommendation 1: Reframe the Paper Around Step-Reduction + KV-Cache Composability (ZERO GPU COST)

**What**: Abandon the CD-SSD/IGSD method contribution narrative. Reframe the paper as: "Coarse step reduction (64->16) composes near-multiplicatively with KV-caching (Ortho=0.949) for MDMs, while AR guidance and adaptive scheduling interfere. Here is the composability atlas and failure mode catalog."

**Why**: The tau=0.0 comparison experiment has conclusively shown that CD-SSD's confidence partitioning mechanism adds negative value relative to naive step reduction. Continuing to present CD-SSD as a method contribution will draw devastating reviewer criticism. The composability finding (that simple step reduction + KV-caching composes well while other combinations interfere) is the genuine contribution, and it is stronger as a general finding than as a CD-SSD-specific finding.

**Effort**: Writing-only change. No new experiments.

**Credibility gain**: Eliminates the paper's most obvious vulnerability. Converts a weakness (negative tau=0.0 result) into a strength (honest analysis paper, not methods paper).

### Recommendation 2: Run Full-Scale Pairwise M1+naive-T16 and M1+CD-SSD on All 4 Benchmarks, 3 Seeds (HIGH PRIORITY)

**What**: Evaluate M1+naive-T16 on full GSM8K (1319), MATH500 (500), HumanEval (164), MBPP (257) with seeds {42, 123, 456}. Also run M1+CD-SSD(tau=0.9) at the same scale (currently only 200 GSM8K + 164 HumanEval, 2 seeds).

**Why**: The headline claim currently rests on 200 GSM8K + 164 HumanEval with 2 seeds. The Ortho has 13.4% inter-seed variance. Full-scale evaluation with 3 seeds would: (a) determine whether Ortho is genuinely > 1.0 or regresses toward 0.95, (b) enable confidence interval reporting, (c) test whether the composability finding generalizes to MATH500/MBPP, and (d) resolve whether M1+naive-T16 vs M1+CD-SSD(tau=0.9) is meaningfully different at scale.

**Effort**: ~8-12 GPU-hours.

**Credibility gain**: Transforms the central composability claim from "preliminary 2-seed observation" to "statistically robust finding with confidence intervals." This is the single biggest gap between current evidence and submission readiness.

### Recommendation 3: Equalize M3 Evaluation Scale and Run Missing Wilcoxon Test (MODERATE PRIORITY)

**What**: (a) Re-evaluate M3 on full-scale GSM8K (1319 samples) and MATH500 (500 samples) with 3 seeds, matching the scale used for M1 and IGSD. (b) Compute the Wilcoxon signed-rank test for H4 (task dependence) as originally specified in the methodology.

**Why**: M3 was evaluated on only 100 GSM8K / 50 MATH500 / 100 HumanEval / 50 MBPP with 2 seeds. The "M3 improves reasoning accuracy by +3.9%" and "MATH500 retention of 243.9%" claims are based on statistically underpowered evaluations. The task-dependent recipe recommendation (H4: "M3 for reasoning") rests on these inflated M3 numbers. The Wilcoxon test was explicitly planned in the methodology but never executed -- the paper currently claims H4 "CONFIRMED" without the specified statistical evidence.

**Effort**: ~4-6 GPU-hours for M3 re-evaluation. ~30 minutes for Wilcoxon test (computation only).

**Credibility gain**: Equalizes the statistical basis for all single-method comparisons. Either confirms or refutes the M3 reasoning advantage claim with adequate statistical power.

---

## 8. Summary Assessment

| Dimension | Grade | Key Issue |
|-----------|-------|-----------|
| Baseline fairness | C+ | Fast-dLLM not reproduced; M2 is a proxy; M3 evaluated at smaller scale; Dream-7B missing; baseline TPS anomaly in seed 42 |
| Metric soundness | C | QAS conflates speed and quality; degenerate baselines inflate composites; 0.5x penalty ad-hoc; Ortho measures only speed composability; "super-multiplicative" label misleading given 32% accuracy retention |
| Internal validity | C+ | Operating points tuned on test data; pairwise experiments underpowered (2 seeds, partial benchmarks); inconsistent evaluation scales across methods |
| Ablation completeness | C+ | tau=0.0 comparison resolves a critical open question (positive). But: KV-cache REFINE ablation still missing; M1 ablations are just threshold sweeps; M3 scale too small for reliable ablation interpretation |
| Reproducibility | B- | Seeds/hardware documented. But: code not public; baseline TPS anomaly; inter-seed Ortho variance high; M2 not reproducible (proxy) |
| tau=0.0 finding integration | B+ | Experiment was well-designed and conclusive. Interpretation is correct. But the finding undermines the paper's current framing. |
| **Overall** | **C+** | The composability finding (KV-cache + step reduction composes well) is genuine and publishable. But the framing (CD-SSD as method contribution, super-multiplicative synergy) is undermined by the paper's own tau=0.0 experiment. Statistical power of pairwise and M3 experiments is insufficient for the claims made. |

### What Changed Since Last Audit

The tau=0.0 comparison experiment (`full_tau0_comparison.json`) is the most significant new data. It resolves the tau=0.0 paradox but in a way that fundamentally undermines the CD-SSD/IGSD method narrative:
- CD-SSD(tau=0.0) = naive step reduction (same accuracy, slightly slower due to overhead)
- M1+naive-T16 Ortho=0.949 (near-multiplicative, not super-multiplicative)
- The paper's decision logic itself concludes: "CD-SSD can be reframed as a step-reduction method"

The three recommendations above address the most impactful gaps. Recommendation 1 (reframing) costs zero GPU-hours and eliminates the paper's largest vulnerability. Recommendations 2-3 together require ~12-18 GPU-hours and would bring the statistical evidence to submission quality.
