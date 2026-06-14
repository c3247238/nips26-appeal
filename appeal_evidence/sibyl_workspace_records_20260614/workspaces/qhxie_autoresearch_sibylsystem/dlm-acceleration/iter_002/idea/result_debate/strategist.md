# Strategic Analysis: ComposeAccel Iteration 2 Results

## Executive Summary

**Verdict: PROCEED -- with significant narrative reframing.**

The experimental evidence is comprehensive and the dataset is publication-worthy, but the story is not what the original proposal expected. The "systematic composition yields multiplicative speedup" narrative is dead. The actual story -- "composition of training-free DLM acceleration methods reveals pervasive interference, and the field's headline speedup numbers are misleading" -- is arguably more publishable. The key strategic question is whether we can frame this as a strong positive contribution rather than a negative result.

---

## 1. Signal Strength Assessment

| Result | Signal | Delta | Justification |
|--------|--------|-------|---------------|
| M1 (EntropyCache) best speedup: 1.16x at eta=0.5, 94.5% acc_ret | **Weak** | +16% TPS, -5.5pp GSM8K | Barely measurable speedup from software-level caching; no kernel-level acceleration realized |
| IGSD best QAS config (tau=0.7, T_draft=16): 2.81x, 58% acc_ret | **Moderate** | +181% TPS, -15pp GSM8K | High speedup but severe accuracy collapse; the KL-guided mechanism works for speed but not quality |
| M3 (AR-guided, gw=0.3): 1.68x, 103.9% acc_ret on GSM8K | **Strong** | +68% TPS, +3.9pp GSM8K | Quality IMPROVES while gaining speed. Robust across gw=0.3-0.7. This is the standout individual method |
| M1+IGSD composition: ortho=0.96 (GSM8K), 0.64 (MATH500) | **Strong** | Near-orthogonal on reasoning | Validates H3 partially: M1 and IGSD operate on different axes and barely interfere on GSM8K |
| M1+M3 composition: ortho=0.41-0.52, speedup <1x | **Strong (negative)** | Severe interference, SLOWDOWN | M3's per-step guidance overhead + M1's cache checking = net slowdown. TPS drops below baseline. This kills H2 |
| M3+IGSD composition: ortho=0.61-0.84 | **Moderate** | Interference except at aggressive IGSD settings | M3's guidance operates on IGSD's noisier draft trajectory, degrading both quality and speed signals |
| Three-way (M1+IGSD+M3): varied ortho | **Moderate** | Results depend heavily on M3 gw setting; gw=0.0 performs best | Adding M3 to M1+IGSD hurts more than helps; the "quality insurance" hypothesis (H2) is falsified |
| Dream-7B generalization: transfer_ratio ~1.86 | **Strong** | Recipes transfer with amplification | All 5 top configs show higher QAS on Dream than LLaDA. Hyperparameters generalize. |
| AR comparison (Qwen7B): 96% acc, 70.9 TPS (b=1) vs LLaDA 71.2% acc, 33.8 TPS | **Strong** | AR dominates on accuracy; competitive on speed at b=1 | LLaDA-8B is simply not competitive with Qwen2.5-7B on these benchmarks. At batch=8, AR wins by 15x on throughput |

### Critical Signal: The AR Comparison Is Devastating

Qwen2.5-7B-Instruct achieves **96% GSM8K accuracy at 70.9 TPS** (batch=1), versus LLaDA-8B's **71.2% at 33.8 TPS** even before acceleration. At batch=8, Qwen reaches **471 TPS**. Even the best composed DLM acceleration (M1+IGSD at ~2.75x = ~93 TPS) does not close this gap. This is the elephant in the room for the paper's practical claims.

---

## 2. Opportunity Cost Analysis

| Direction | GPU Hours | Info Gain per GPU-hr | Risk |
|-----------|-----------|---------------------|------|
| Additional three-way Pareto points (more configs) | 4-6 hrs | Low -- diminishing returns, pattern is clear | Low |
| Order-correction ablation (entropy priority inversion) | 3-4 hrs | **High** -- directly tests whether order confound explains M2/IGSD collapse | Medium |
| d2Cache kernel-level integration (real M1 speedup) | 8-12 hrs | **Medium** -- would convert M1 from 1.16x to potentially 3-5x, changing composition ratios | High (integration risk) |
| Deeper IGSD analysis (KL profiles, calibration study) | 2-3 hrs | Medium -- the ablation data already exists | Low |
| Additional Dream-7B compositions | 2-3 hrs | Low -- transfer is already validated | Low |
| Larger-scale GSM8K eval (full 1319 samples, 3 seeds, all compositions) | 10-15 hrs | Medium -- reduces error bars, strengthens statistical claims | Low |
| Writing and analysis | 0 GPU hrs | **Critical** -- the data is complete enough to write | N/A |

---

## 3. Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome |
|-----------|----------------|----------|------|-----------------|
| **Write the paper now** | Strong data already | 0 | Low | Publication-ready results with honest negative findings |
| Order-correction ablation | High potential | 3-4 hrs | Medium | Could explain WHY compositions fail; adds mechanistic insight |
| Kernel-level d2Cache | Medium | 8-12 hrs | High | Could rescue M1 speedup claims but may not change composition story |
| More three-way configs | Diminishing | 4-6 hrs | Low | Marginal improvement to Pareto frontier characterization |
| Scale up for statistics | Medium | 10-15 hrs | Low | Tighter error bars, same conclusions |

---

## 4. PIVOT vs PROCEED Verdict

**PROCEED.** Justification:

1. **The dataset is comprehensive.** 15 completed experiment groups covering: full baselines (3 seeds, 4 benchmarks), individual Pareto sweeps (M1 x3, IGSD x9, M3 x4), all three pairwise compositions, three-way composition (5 configs x 3 seeds), Dream-7B cross-model validation, AR baseline comparison, batch sensitivity, IGSD ablation, and cache hit rate refinement. This is a complete factorial study.

2. **The findings are publishable and interesting.** The headline finding is NOT "composition works" but rather "composition of training-free DLM acceleration methods reveals three distinct interference regimes: (a) near-orthogonal speed axes (M1+IGSD, ortho~0.96), (b) destructive interference when quality-preserving and speed methods operate on the same denoising trajectory (M1+M3, ortho~0.41-0.52), and (c) partial interference when step reduction degrades guidance context (M3+IGSD, ortho~0.61-0.84)."

3. **The AR comparison provides essential honesty.** Including the Qwen-7B benchmark alongside DLM results distinguishes this paper from the stream of DLM acceleration papers that avoid direct AR comparison.

4. **Cross-model generalization is validated.** Dream-7B shows consistent patterns with amplified QAS, confirming that composition behavior is architecture-general, not LLaDA-specific.

5. **IGSD, while not a breakthrough, is a clean algorithmic contribution.** The 50-line KL-divergence step scheduler with documented Pareto frontier is a useful component even if it does not achieve lossless acceleration.

---

## 5. Recommended Next Steps (Priority Order)

### Step 1 (Highest Priority): Begin paper writing immediately

Estimated time: 0 GPU hours. This is where the maximum research impact per hour lies.

The narrative should be restructured around **three key claims**:
1. **Interference taxonomy**: First systematic quantification of how training-free DLM acceleration methods compose across three orthogonal axes. The orthogonality metric (QAS composition ratio) is a methodological contribution.
2. **Practical recipes**: Despite interference, M1+IGSD at conservative settings achieves 1.65-1.75x speedup on reasoning tasks. M3 alone at gw=0.3 is the best single method (1.68x with accuracy improvement). These are actionable for practitioners.
3. **Honest DLM positioning**: Accelerated DLMs still lag behind optimized AR baselines, but the gap narrows at batch=1 for latency-sensitive applications. The paper should be framed as "what is achievable today" rather than "DLMs are better."

### Step 2 (If time permits before submission): Order-correction ablation

Run entropy-based priority inversion on GSM8K 200 samples to test whether unmasking order accounts for the M2/IGSD accuracy collapse. This experiment directly addresses the contrarian critique and could elevate the paper from "empirical survey" to "mechanistic insight."

Estimated cost: 3-4 GPU hours. Expected information gain: high, as it would explain the root cause of the key negative finding.

### Step 3 (Defer to revision): Full statistical rigour

If reviewers request it, scale up to full 1319 GSM8K samples with 3 seeds for all composition configs. The current pilot-scale (100-200 samples, 1-3 seeds) data is directionally correct but may face statistical power critiques.

---

## 6. Resource Allocation Recommendation

| Resource | Allocation |
|----------|-----------|
| GPU time (remaining) | 80% writing/analysis, 20% order-correction ablation |
| Research effort | Focus on paper narrative and figure design |
| Risk mitigation | Do NOT attempt kernel-level d2Cache integration -- too high risk for remaining timeline |

---

## 7. Key Risks Going Forward

1. **Reviewer complaint: "Just an empirical survey, no new method."** Mitigation: Position IGSD as a novel contribution; emphasize the orthogonality metric framework; highlight the interference taxonomy as a conceptual contribution.

2. **Reviewer complaint: "LLaDA-8B baseline is too weak (71% GSM8K vs 96% for AR)."** Mitigation: Acknowledge this directly in the paper; argue that the composition analysis methodology transfers to future stronger DLMs; note that Dream-7B validation confirms generalization.

3. **Reviewer complaint: "Sample sizes too small for statistical claims."** Mitigation: Three-way configs have 3-seed evaluation; can add error bars. Some pairwise configs are single-seed -- flag these as pilot-scale in the paper.

4. **Speed competition: someone publishes a composition study first.** Mitigation: This iteration's 15-experiment dataset and honest AR comparison differentiate from any surface-level composition paper. Fast execution to submission is critical.

---

## Summary Recommendation

**PROCEED to paper writing.** The data tells a clear, honest, and interesting story: training-free DLM acceleration methods do NOT compose as the field implicitly assumes. This is a service contribution to the community. Execute the order-correction ablation as a bonus experiment if the timeline permits, but do not gate the paper on it.
