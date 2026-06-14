# Result Debate Synthesis: Iteration 4 (BSD + RACFG + A-CFG)

**Synthesized by**: Result Debate Synthesizer
**Date**: 2026-03-10
**Perspectives synthesized**: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist

---

## 1. Consensus Map: Where All 6 Perspectives Agree

The following conclusions are **high-confidence** — endorsed (explicitly or implicitly) by all 6 analysts:

1. **RACFG (JSD cross-step stability) is definitively falsified on Dream-7B.** JSD stability ~0.997 means cross-step distribution changes are negligible, rendering stability-based guidance inoperative. This is a model-architecture finding, not a hyperparameter issue (all 6 remask_pct x ema_lambda combinations yield 0%). All perspectives agree this is a clear negative result with diagnostic value.

2. **H5 (JSD > confidence-based remasking) is falsified.** RACFG 0% vs A-CFG 12.5% on Countdown-16. No analyst disputes this.

3. **H7 (BSD + A-CFG synergy) is not supported.** Combo 6.2% < max(BSD, A-CFG) 12.5%. The sub-additive result invalidates the "orthogonal complementarity" thesis. (Skeptic and methodologist note this is n=16 and the difference is 1 sample; optimist suggests hyperparameter mismatch. But the directional signal is negative and no perspective recommends pursuing the combination as a priority.)

4. **n=16 pilot data has extremely low statistical power.** All perspectives acknowledge this explicitly. The project has accumulated 4+ pilot-to-full-scale reversal cases (entropy PPL, TCR PPL, anneal PPL, ReMDM-conf GSM8K). No verdict based solely on n=16 should be treated as definitive.

5. **Full-scale validation (Countdown-500 x 3 seeds) is the single most important next step.** Every perspective identifies this as P0. Without it, no publishable claims can be made about BSD or A-CFG.

6. **DMI remains the only method validated at full scale** (9.3% vs vanilla 4.7%, Countdown-500 x 3 seeds). This is the project's most reliable positive result, though the revisionist raises valid questions about causal mechanism.

7. **The original proposal narrative (BSD + RACFG three-layer architecture) must be abandoned.** RACFG failed completely, BSD+A-CFG showed no synergy. The paper needs a fundamentally restructured narrative.

---

## 2. Conflict Resolution: Where They Disagree

### 2.1 How to interpret A-CFG's GSM8K pilot result (37.5% vs vanilla 25.0%)

| Perspective | Position |
|-------------|----------|
| **Optimist** | This is "the biggest absolute improvement in 4 iterations" — a breakthrough signal worth aggressive follow-up |
| **Skeptic** | p > 0.6 on McNemar, 95% CI = [-12.5%, +37.5%], statistically meaningless at n=16 |
| **Strategist** | Directionally strong + literature backing (A-CFG NeurIPS 2025 on LLaDA) justifies full-scale investment |
| **Comparativist** | A-CFG is an existing published method; reproducing it on Dream-7B is not a novel contribution |
| **Revisionist** | A-CFG on Dream-7B hasn't even been verified yet in their timeline; they urge caution |

**Judgment**: The skeptic is correct that n=16 provides no statistical confidence. However, A-CFG is not a speculative method — it has independent validation at NeurIPS 2025 with strong results on LLaDA-8B (GSM8K 73.5%) and Dream-7B (GSM8K 77.9%). The prior probability of A-CFG working on Dream-7B is substantially higher than for our novel methods (BSD, RACFG). The pilot signal is consistent with this prior. **Full-scale validation is clearly warranted** (cost ~18 GPU-hours, well within budget), but we must not make strong claims until that data exists. The comparativist is right that A-CFG reproduction alone is not a novel contribution — the novelty must come from additional analysis or combination.

### 2.2 Is BSD a viable method?

| Perspective | Position |
|-------------|----------|
| **Optimist** | Excellent text quality (rep-2 -50%), elegant entropy framework, publication-worthy analysis |
| **Skeptic** | 6.2% accuracy = tied with vanilla 256-step; entropy decrease is a tautology of EMA+temperature annealing |
| **Methodologist** | BSD config chosen from n=16 ablation with zero discriminative power (4 configs all at 6.2%) |
| **Comparativist** | Core idea overlaps with Soft-Masked (NeurIPS 2025), ReMix, LRD — novelty is narrow |
| **Revisionist** | OOD risk underestimated; DMI's causal mechanism should be validated before building BSD on top of it |

**Judgment**: The skeptic raises a valid point about entropy decrease being partially tautological (EMA + temperature annealing mathematically guarantee convergence). However, the optimist is right that entropy-accuracy correlation (r=0.78, p=0.0003) and the text quality improvement (rep-2/3 reduction ~50%) are real signals worth investigating at scale. The comparativist's overlap concern is serious — BSD's differentiation from Soft-Masked DLMs, ReMix, and LRD must be clearly articulated. **BSD is a secondary priority after A-CFG validation**, and its role is likely as an analytical contribution (belief entropy framework) rather than a standalone accuracy-improving method.

### 2.3 Compute-fair comparison: Is vanilla Pareto-optimal?

| Perspective | Position |
|-------------|----------|
| **Skeptic** | Pilot compute-fair data shows vanilla dominates at all FLOP levels — this "directly negates the value proposition of all inference-time scaling methods" |
| **Optimist** | A-CFG (2x FLOPs, 12.5%) > vanilla 256-step (2x FLOPs, 6.2%) on Countdown-16 |
| **Strategist** | Critical risk (20% probability) but n=16 variance is too high to conclude |

**Judgment**: The data is contradictory because of extreme n=16 noise. In some runs vanilla 2x = 12.5%, in others = 6.2%. A-CFG in some runs = 0%, in others = 12.5%. **No reliable Pareto conclusion can be drawn from n=16 data.** However, DMI at full scale (9.3% at 1.05x FLOPs vs vanilla 4.7% at 1.0x) clearly beats the Pareto frontier — you cannot get 9.3% by adding 5% more vanilla steps. The compute-fair question must be resolved at full scale.

### 2.4 Publication target and novelty assessment

| Perspective | Optimist | Comparativist |
|-------------|----------|---------------|
| Top venue probability | 45% | 10-15% |
| Core contribution | A-CFG on Dream-7B + DMI + systematic study | Mostly reproduction + negative results |
| Paper type | Methods paper | Diagnostic/analysis paper |

**Judgment**: The comparativist's assessment is more grounded. A-CFG is an existing NeurIPS 2025 method; reproducing it is confirmation, not contribution. BSD overlaps with multiple published works. DMI 9.3% is real but modest. The honest publication ceiling, assuming A-CFG validates at full scale, is **EMNLP/ACL main or NeurIPS as a systematic study paper**. The strongest framing emphasizes the **diagnostic value**: why some approaches work (representation injection) while others fail (parameter adaptation, cross-step signals), with compute-efficiency analysis. If A-CFG fails at full scale, the paper becomes a negative-results contribution at Findings level.

---

## 3. Result Quality Score

**Score: 4.5 / 10**

**Justification**:

- (+2.0) DMI full-scale validation is solid (500 x 3 seeds, 9.3% vs 4.7%, the only reliable positive result)
- (+1.5) RACFG falsification is clean and informative (JSD stability ~0.997, all configs 0%)
- (+1.0) A-CFG pilot signal is directionally positive and literature-backed
- (+0.5) BSD entropy analysis shows genuine information-theoretic patterns
- (+0.5) Systematic coverage: 7+ methods, multiple benchmarks, compute-fair comparisons attempted
- (-0.5) All iter 4 new methods (BSD, RACFG, A-CFG) only evaluated at n=16 pilot — no full-scale data
- (-0.5) 0/42 statistical comparisons reach significance (even uncorrected)
- (-0.5) Vanilla baseline inconsistency across runs (0% to 18.8% on same 16 samples)
- (-0.5) BSD core concept overlaps with 3+ published works (Soft-Masked, ReMix, LRD)
- (-0.5) Original proposal's three pillars (BSD, RACFG, combination) all failed their hypotheses

The score reflects that **the project has generated valuable diagnostic data but lacks the statistical evidence to support positive claims about its novel methods**. The gap between pilot ambitions and full-scale validation is the dominant weakness.

---

## 4. Key Findings

1. **A-CFG (classifier-free guidance with confidence-based remasking) is the most promising training-free method for Dream-7B**, showing +12.5pp on GSM8K-16 (37.5% vs 25.0%) and 12.5% on Countdown-16 (vs vanilla 0-6.2%). This is consistent with its NeurIPS 2025 publication results on LLaDA-8B. Full-scale validation is critical.

2. **Cross-step distribution signals are uninformative in Dream-7B's denoising process.** JSD stability ~0.997 across steps means the model makes nearly identical predictions at each step. This explains why TCR (iter 3), RACFG (iter 4), and all cross-step dependent methods fail on this architecture. This is an important empirical finding about DLM denoising dynamics.

3. **DMI (Diffusion Memory Injection) remains the best-validated method** at 9.3% accuracy on Countdown-500 (3 seeds) with near-zero compute overhead (~1.05x FLOPs). However, its causal mechanism is unclear — it may function primarily through implicit temperature regulation rather than genuine information transfer.

4. **The project has established a robust meta-finding: n=16 pilot results are unreliable for DLM evaluation.** Four independent reversal cases demonstrate that pilot improvements of 12-25pp can completely vanish at full scale. This methodological insight has independent value for the DLM community.

5. **Representation-level interventions (DMI, BSD) consistently outperform parameter-level interventions (DTA) and cross-step signal methods (TCR, RACFG).** This "shallow beats deep" pattern across 4 iterations is a genuine empirical contribution, though the ceiling remains modest (~9% at full scale).

---

## 5. Methodology Gaps (from Methodologist + Skeptic)

### Critical (must fix before submission)

1. **Full-scale evaluation completely missing for all iter 4 methods.** BSD, A-CFG, RACFG, and combinations have only n=16 pilot data. The task_plan specified Countdown-500 x 3 seeds as core evaluation — this was never executed.

2. **Baseline inconsistency across runs.** DMI shows 0% in some pilot runs and 12.5% in others on the same 16 samples. Vanilla ranges from 0% to 18.8%. This suggests either (a) random seed not properly fixed, (b) evaluation prompt set varies between runs, or (c) CUDA non-determinism. Root cause must be identified and eliminated.

3. **Vanilla baseline gap with original paper.** Our Countdown vanilla 4.7% vs Dream paper's 16.0% (3.4x gap). Our GSM8K ~29.6% vs paper's 77.2% (2.6x gap). Without explaining this discrepancy (likely zero-shot vs 8-shot, generation length, answer extraction), reviewers will question all results.

### Important (significantly improves paper quality)

4. **External baselines entirely missing.** Proposal listed MetaState, CORE, HEX, Self-Rewarding SMC as comparisons — none implemented. At minimum, Best-of-N (N=2-4) and LookUM (2-3 paths) should be included as compute-matched baselines.

5. **Hyperparameter selection on n=16 is unreliable.** BSD k ablation (0%, 0%, 6.2%) and alpha schedule ablation (all 6.2%) have zero discriminative power. Best config is effectively random selection.

6. **DMI causal mechanism unvalidated.** Three plausible alternative explanations (implicit temperature, format correction, random perturbation) have not been tested with control experiments.

---

## 6. Competitive Position (from Comparativist)

### Training-free methods landscape (2025-2026)

| Method | Venue | Key Result | vs Our Best |
|--------|-------|------------|-------------|
| A-CFG | NeurIPS 2025 | LLaDA GSM8K 73.5%, Dream GSM8K 77.9% | We reproduce on Dream (37.5% pilot, different eval setup) |
| Soft-Masked DLM | NeurIPS 2025 | Dream-7B code generation | Overlaps with BSD/DMI core idea |
| LookUM | arXiv 2511 | 2-3 paths matches RL post-training | Stronger training-free baseline we haven't compared |
| ReMix | arXiv 2602 | Continuous mixing state, 2-8x speedup | Conceptually similar to BSD |
| LRD | arXiv 2510 | GSM8K +2.9 via belief refinement | Directly comparable to BSD |
| POKE | arXiv 2602 | Path likelihood optimization | Training-free, verified on Countdown |

**Honest assessment**: BSD's novelty space is narrow given 3+ overlapping published works. A-CFG is a reproduction. DMI (our most validated method) overlaps with Soft-Masked DLM. Our strongest differentiation is the **systematic diagnostic study** across 20+ method variants over 4 iterations, plus the RACFG failure analysis revealing Dream-7B's cross-step stability property.

**Best positioning**: Empirical systems paper analyzing why different inference-time scaling strategies succeed or fail on DLMs, rather than a methods paper claiming novel algorithms.

---

## 7. Hypothesis Update (from Revisionist)

### Hypotheses that survived

| Hypothesis | Status | Confidence |
|------------|--------|------------|
| H2: BSD belief entropy monotonically decreases | **Supported** (15/16 monotonic, rho=-0.95) | Medium — partially tautological due to EMA+temperature design |
| H8: A-CFG generalizes to GSM8K | **Directionally positive** (37.5% vs 25.0% pilot) | Low — n=16 only, needs full-scale |

### Hypotheses requiring revision

| Hypothesis | Original | Revised Understanding |
|------------|----------|----------------------|
| H1: BSD > DMI | BSD should reach 14%+ | BSD ~6.2% at pilot, likely comparable to or weaker than DMI. Revise to: BSD's value is analytical (entropy framework, text quality) not accuracy |
| H4: RACFG > vanilla | RACFG should reach 15%+ | RACFG is impossible on Dream-7B due to cross-step stability. Revise to: A-CFG is the viable guidance method |
| H6: Temporal scheduling > fixed | Scheduled CFG should outperform | All scheduling variants = 0% vs fixed = 12.5%. Fixed weight is strictly better on Dream-7B |

### Hypotheses definitively falsified

| Hypothesis | Evidence | Implication |
|------------|----------|-------------|
| H5: JSD > confidence | RACFG 0% vs A-CFG 12.5% | Cross-step signals are dead on Dream-7B |
| H7: BSD+RACFG synergy | Combo 6.2% < max 12.5% | Orthogonal complementarity thesis fails |
| H3: Intermediate k optimal | k=3T/4 best (not T/4-T/2) | If BSD works, it's via maximal belief accumulation |

### New hypotheses proposed by revisionist (worth testing)

- **NH1**: DMI improvement comes primarily from implicit softmax temperature regulation (testable with temperature-matched vanilla control)
- **NH4**: Correction Precision < 50% is the ceiling condition for all remasking methods (testable with existing data aggregation)
- **NH5**: Minimum N=200 x 2 seeds needed to detect 3pp effects on Dream-7B Countdown (testable with post-hoc power analysis)

---

## 8. Action Plan

### Verdict: **PROCEED** with narrative restructuring and A-CFG full-scale validation

The project should not pivot (DMI provides a solid safety net), but the original BSD+RACFG narrative is dead. The path forward centers on A-CFG validation and diagnostic analysis.

### Priority 0 — Immediate (next 12-18 hours)

1. **A-CFG Countdown-500 x 3 seeds** (~8 GPU-hours)
   - This is the single experiment that determines the paper's ceiling
   - If A-CFG >= 10%: paper upgrades to systematic study + A-CFG confirmation on Dream-7B
   - If A-CFG < 7%: paper downgrades to diagnostic/negative-results study

2. **A-CFG GSM8K-1319 x 1 seed** (~5 GPU-hours)
   - GSM8K pilot 37.5% is the strongest signal; verify it
   - If >= 33% at full scale: strong cross-task generalization claim

3. **Fix baseline inconsistency**
   - Identify why vanilla/DMI results vary across runs on same 16 samples
   - Standardize evaluation prompt set and ensure seed control

### Priority 1 — Short-term (24-48 hours, conditional on P0 results)

4. **DMI + A-CFG combination pilot** (n=100, ~2 GPU-hours)
   - DMI (embedding-level, 1.05x) + A-CFG (prediction-level, 2x) operate at different layers
   - More promising combination than BSD+A-CFG since DMI is validated

5. **Compute-fair comparison at full scale**
   - Resolve whether vanilla step-doubling dominates all methods
   - Critical for paper credibility

6. **Baseline alignment with Dream paper**
   - Run 8-shot evaluation to reproduce Dream paper's 16.0% Countdown / 77.2% GSM8K
   - Without this, absolute numbers are uninterpretable to reviewers

### Priority 2 — Medium-term (if P0/P1 yield positive results)

7. **Best-of-N (N=2,4) baseline implementation**
   - Simplest inference-time scaling baseline that reviewers will expect

8. **BSD Countdown-500 x 3 seeds** (only if BSD pilot shows promise at n=100 first)
   - BSD's role is likely analytical, not accuracy-focused

9. **MBPP evaluation for A-CFG and DMI**
   - Third benchmark strengthens paper significantly

### Not recommended

- **Further BSD+A-CFG combination work** — pilot clearly negative, n=16 but directionally clear
- **Any RACFG variant** — Dream-7B's cross-step stability makes this entire approach unviable
- **n=16 pilot for any new method** — 4+ reversal cases prove this has no screening power
- **New method development** before full-scale data is collected

### Pivot trigger conditions

1. A-CFG Countdown-500 seed 42 < 6% accuracy → diagnose implementation; if confirmed, pivot to DMI-centered diagnostic paper
2. Full-scale statistical tests show DMI vs vanilla is also non-significant (p > 0.1) → urgent data quality audit
3. All full-scale results show vanilla step-doubling is Pareto-optimal → pivot to negative-results + methodology paper

### Paper strategy (scenario-dependent)

| Scenario | Probability | Target Venue | Core Narrative |
|----------|------------|--------------|----------------|
| A: A-CFG Countdown >= 10% + GSM8K >= 33% | ~35% | EMNLP/NeurIPS | A-CFG confirmation on Dream-7B + DMI + systematic diagnostic study |
| B: A-CFG works only on GSM8K | ~25% | ACL/EMNLP | Task-dependent nature of DLM inference-time scaling |
| C: A-CFG fails at full scale, DMI holds | ~25% | EMNLP/Findings | DMI as zero-overhead method + comprehensive negative results |
| D: Everything fails at full scale | ~15% | ACL Findings | "Why training-free inference-time scaling remains elusive for DLMs" |

### Confidence assessment

- Paper publishable (any venue): **85%** (DMI safety net + rich diagnostic data)
- Paper at main conference: **30-40%** (requires A-CFG full-scale success)
- Needs major pivot: **<15%** (only if DMI also fails statistical tests)
- Estimated time to paper position lock: **12-18 hours** (pending A-CFG Countdown seed 42)

---

*Synthesis completed 2026-03-10. This assessment supersedes individual perspective reports and should be used as the basis for immediate action planning.*
