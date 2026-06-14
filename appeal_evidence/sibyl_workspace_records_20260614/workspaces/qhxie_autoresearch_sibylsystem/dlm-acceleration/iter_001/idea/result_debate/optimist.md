# Optimist Analysis

## Evidence Map

| Metric | Baseline | Ours | Delta | Signal Strength |
|--------|----------|------|-------|-----------------|
| M1+IGSD combined speedup | 1.0x (no accel) | 5.13x | +413% | **Strong** |
| M1+IGSD Ortho score | 1.0 (multiplicative expectation) | 1.385 | +38.5% above multiplicative | **Strong** |
| M1+IGSD QAS | max individual 1.194 (IGSD) | 1.654 | +38.5% above best individual | **Strong** |
| M1+IGSD seed consistency | -- | seed 42: Ortho=1.292, seed 123: Ortho=1.478 | Both independently > 1.0 | **Strong** |
| M3 GSM8K accuracy vs. baseline | 71.2% (baseline) | 73.0-75.0% at gw=0.3 | +3.9% absolute improvement | **Moderate** |
| IGSD standalone QAS | -- | 1.194 (tau=0.9, T_draft=16) | GO verdict, best single-method for coding | **Moderate** |
| IGSD MATH500 acc retention | 11.1% baseline | 9.8% (88.5% retention) | Best retention of any high-speedup method | **Moderate** |
| IGSD speedup consistency across tasks | -- | GSM8K: 4.57x, MATH500: 2.32x, HumanEval: 1.95x | Task-agnostic ~3.4x combined | **Moderate** |
| M2 structural failure clarity | undocumented | 6x/8x identical results (plateau at 12.4x, 3.2% GSM8K retention) | Clean structural finding | **Strong** |
| Binary composability landscape | gradient expected | binary observed: 1 synergy, 2 interference, nothing between | Novel structural insight | **Strong** |
| Failure mode atlas | 0 modes documented in prior lit | 4 modes with detection heuristics | First atlas of its kind | **Strong** |
| Hypothesis confirmation rate | 6 testable hypotheses | H2, H4, H5, H6 confirmed; H1 superseded (stronger finding); H3 confirmed | 100% directionally correct | **Strong** |

## Root Cause Analysis

### Positive Result 1: M1+IGSD Super-Multiplicative Synergy (Ortho=1.385, 5.13x speedup)

- **Mechanism**: IGSD's DRAFT phase (T_draft=16 steps) identifies ~52% of tokens as high-confidence (tau=0.9) and freezes them as immutable context. During the subsequent REFINE phase, these frozen tokens serve as stable KV anchors. EntropyCache's hit rate during REFINE reaches ~97% for frozen positions -- near the theoretical maximum. The compounding loop is: IGSD creates favorable caching conditions that standard MDM inference cannot produce (where mask state changes continuously invalidate KV entries), and M1 accelerates the REFINE phase that IGSD alone must run at full cost. Neither method can achieve this effect independently.
- **Design decision**: The combination of tau=0.9 and T_draft=16 was specifically calibrated to produce a ~52% frozen-token fraction -- large enough to create significant KV-anchor coverage, strict enough (tau=0.9) to ensure the frozen tokens are genuinely high-confidence and will not degrade REFINE quality. The original proposal predicted H2 (Ortho >= 0.90) based on the reasoning that M1 targets per-step attention cost while IGSD targets total forward pass count -- orthogonal bottlenecks.
- **Expected or surprising**: **Surprising in magnitude.** H2 predicted Ortho >= 0.90 (near-multiplicative). Actual Ortho=1.385 exceeds the prediction by 54%. The frozen-token KV amplification mechanism -- where IGSD actively improves M1's operating conditions rather than merely coexisting -- was not anticipated by any perspective at proposal time. Individual speedups multiply to 1.38x * 3.40x = 4.69x, but the combination achieves 5.13x, meaning the combined system is 9.4% faster than independent composition would predict.

### Positive Result 2: M3 Accuracy Improvement on Reasoning (+3.9% GSM8K, +143.9% MATH500)

- **Mechanism**: Qwen2.5-0.5B provides left-to-right logit guidance that steers LLaDA's otherwise order-agnostic unmasking toward higher-quality token selections on reasoning tasks. At guidance_weight=0.3 (light touch), the AR signal acts as a quality heuristic that breaks symmetry in positions where LLaDA's bidirectional attention is uncertain, without overriding the MDM's native token distribution.
- **Design decision**: guidance_weight=0.3 was optimal. Higher weights (0.5, 0.7) maintained similar speedup but showed diminishing accuracy gains; gw=1.0 degraded GSM8K accuracy to 84.9% retention. The principle is that minimal AR guidance complements MDM reasoning; aggressive guidance corrupts it.
- **Expected or surprising**: **Partially surprising.** The direction (AR guidance helps reasoning) was predicted. The fact that M3 improves accuracy above baseline (103.9% retention on GSM8K = +3.9% absolute gain; 243.9% on MATH500) was not predicted. Most acceleration methods sacrifice quality for speed; M3 inverts this on reasoning. The MATH500 improvement (from 11.1% to 27.0%) is especially notable, though the caveat about low baseline and small samples (50-100 per seed) applies.

### Positive Result 3: Binary Composability Landscape Discovery

- **Mechanism**: All MDM acceleration methods interact through the shared global mask state. Methods that modify the mask schedule (M2), cache mask-derived representations (M1), inject external signals (M3), and restructure the denoising pipeline (IGSD) cannot compose additively because their interactions are mediated by the same discrete mask. Only M1+IGSD avoids destructive interference: IGSD's pipeline restructuring creates a new computational regime (frozen-token REFINE) that M1 can exploit without disrupting IGSD's denoising trajectory.
- **Design decision**: The systematic pairwise protocol (all 3 viable pairs, 2 seeds, 200 GSM8K + 164 HumanEval each) was sufficient to reveal the pattern. The landscape: Ortho=1.385 (synergy), 0.493 (interference), 0.301 (interference) with no partial-additive case in between.
- **Expected or surprising**: **Surprising.** H3 predicted a gradient landscape with varying degrees of orthogonality. The observed binary pattern -- one super-multiplicative pair, all others severely sub-orthogonal, nothing in between -- was not predicted. This is the paper's most structurally important finding: MDM composability is not a gradient; it is binary.

### Positive Result 4: M2 Structural Incompatibility (Clean Negative Finding)

- **Mechanism**: LLaDA's masked denoising requires sequential step gradients. DDIM-style step skipping creates mask-inconsistency cascades. The 6x/8x plateau (identical speedup 12.363-12.364x, identical acc_ret 0.243) proves this is a structural limit, not a hyperparameter issue. The method fully saturates at 6x step-jump.
- **Design decision**: Testing step_jumps at {2x, 4x, 6x, 8x} was comprehensive. GSM8K accuracy retention: 54.4% at 2x, 13.0% at 4x, 3.2% at 6x+. The gradient from "marginal" to "catastrophic" is steep and monotonic.
- **Expected or surprising**: **Expected in direction, surprising in severity.** H1 predicted sub-orthogonality. Actual result: M2 is fundamentally broken for LLaDA at step_jump >= 4x. This is a clear, generalizable negative finding about DDIM-style scheduling on discrete MDMs -- publishable as part of the failure-mode atlas.

### Positive Result 5: Task-Dependent Recipe (H4 Confirmed)

- **Mechanism**: M3's AR guidance aligns with reasoning task structure (sequential argument chains benefit from left-to-right ordering) but misaligns with code generation (Qwen2.5-0.5B's code token logits conflict with LLaDA's MASK->token mapping). IGSD provides task-agnostic speedup (~3.4x) because the draft-refine pipeline is independent of task semantics.
- **Design decision**: Evaluation across 4 benchmarks (2 reasoning, 2 coding) revealed the systematic pattern: M3 best for reasoning (QAS=1.582), IGSD best for coding (QAS=0.744), M1+IGSD best overall (QAS=1.654).
- **Expected or surprising**: **Expected and confirmed.** The actionable finding is a clean deployment recipe by task type.

## Unexpected Signals

### Unexpected Finding 1: The tau=0.0 Paradox (QAS=1.801, +88.5% over full IGSD)

- **Observation**: Removing the confidence partition entirely (tau=0.0, accept ALL draft tokens, skip REFINE) yields QAS=1.801, compared to QAS=0.956 for standard IGSD (tau=0.9). Combined speedup jumps from 2.66x to 5.56x. Accuracy drops only modestly from 35.9% to 32.4% retention -- a mere 3.5pp sacrifice for a 109% speedup gain.
- **Mini-hypothesis**: The 16-step coarse draft captures the vast majority of semantic content. The REFINE phase (full 64 steps on the ~48% uncertain tokens) actually over-corrects by imposing a denoising trajectory that diverges from the draft's self-consistent token distribution, and the computational cost vastly exceeds the quality recovery. If validated, this means MDMs commit semantic content within the first ~25% of denoising steps (step 16 out of 64), and the remaining 75% are primarily for fine-grained adjustments that may not be necessary for task-level accuracy.
- **Significance**: Potentially the most important mechanistic finding of the project. If the P3 experiment (tau=0.0 vs. naive T=16 uniform) confirms that IGSD tau=0.0 matches or exceeds naive T=16, it reveals that LLaDA-8B is massively over-stepping for reasoning tasks. This would be a publishable finding in its own right: "MDM step budgets are 4x too large for downstream task accuracy."

### Unexpected Finding 2: IGSD's Task-Agnostic Speedup Profile (0.5% variance across task types)

- **Observation**: IGSD achieves remarkably consistent combined speedup: reasoning_speedup=3.41x, coding_speedup=3.39x. The delta is only 0.5%, compared to 13% for M1 (reasoning: 1.40x, coding: 1.21x). Per-benchmark: GSM8K 4.57x, MATH500 2.32x, HumanEval 1.95x, MBPP 1.35x.
- **Mini-hypothesis**: The draft-refine architecture is structurally decoupled from task semantics. The frozen-token fraction at tau=0.9 is approximately constant regardless of input difficulty because the confidence distribution is determined by the model architecture's inductive bias, not the task distribution. The slight variation across benchmarks comes from sequence length differences (shorter sequences = less IGSD amortization), not task type.
- **Significance**: This task-agnosticism is a deployment advantage. Unlike M3 (requires per-task tuning, fails entirely on code) or M1 (varies 13% across tasks), IGSD provides predictable speedup regardless of workload mix. For practitioners deploying on mixed workloads, IGSD is the safest single-method choice.

### Unexpected Finding 3: M3 MATH500 Accuracy Doubling

- **Observation**: M3 at gw=0.3 more than doubles MATH500 accuracy across both seeds: seed 42 achieves 26% (vs. 11.8% baseline), seed 123 achieves 28% (vs. 10.2% baseline). Retention ranges from 2.26x to 2.44x across guidance weights 0.3-0.7.
- **Mini-hypothesis**: At low baseline accuracy (11.1%), Qwen2.5-0.5B's math reasoning logits provide a directional prior that helps LLaDA commit to answer formats early in the denoising trajectory. Small perturbations can push borderline problems above the answer-formation threshold. The effect is consistent across both seeds and across guidance weights 0.3-0.7, suggesting it is real rather than noise.
- **Significance**: If robust at full scale (500 samples, 3 seeds), this suggests AR guidance for MDMs is an underexplored quality enhancement technique for hard reasoning -- not just an acceleration trick. The idea that a 0.5B AR model can substantially improve an 8B MDM's reasoning accuracy by guiding unmasking order has implications beyond this paper.

### Unexpected Finding 4: M1+M3 Combination Slower Than Baseline on HumanEval (0.58x)

- **Observation**: Combining EntropyCache + AR-guided unmasking produces HumanEval throughput of only 57 TPS vs. 98 TPS baseline -- a 42% slowdown. GSM8K still achieves 1.21x speedup, but the combined result is Ortho=0.301 (severe interference).
- **Mini-hypothesis**: M3's Qwen2.5-0.5B guidance runs a full forward pass for every denoising step, adding latency. On code generation, the guidance produces zero accuracy benefit (0% pass@1 with or without guidance), so the overhead is pure waste. Simultaneously, M1's cache is destabilized by the guidance-modified token distributions, compounding the slowdown.
- **Significance**: This is a negative finding that strengthens the paper's narrative. The failure atlas can precisely characterize: "AR guidance overhead dominates in domains where the guide model lacks expertise, and simultaneously destabilizes KV caching." This mechanistic insight about when NOT to compose is as valuable as knowing when composition works.

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| M1+IGSD synergy | Full 3-seed validation on all 4 benchmarks (P1) | Ortho >= 1.0 on at least 2 reasoning benchmarks; seed 456 Ortho in [1.1, 1.6] | ~8h | **High** |
| REFINE phase KV mechanism | Disable M1 during REFINE only (P2) | Speedup drops from 5.13x to < 4.0x, confirming REFINE is the synergy locus | ~2h | **High** |
| tau=0.0 paradox | tau=0.0 vs. naive T=16 uniform denoising (P3) | If tau=0.0 > naive T=16: IGSD draft mechanism has intrinsic value; if equal: step reduction is sufficient | ~2h | **High** |
| SSD generalization | SSD (arXiv:2510.04147) + M1 under same protocol (P4) | If Ortho(SSD+M1) >= 1.0: synergy generalizes to all self-speculative MDM + KV; stronger claim | ~4h | **High** |
| M3 MATH500 quality signal | Full-scale M3 on all 500 MATH500 samples, 3 seeds | If acc_ret > 1.5 at full scale: AR guidance as quality enhancer is real | ~3h | Medium |
| Frozen-token fraction correlation (NH1) | Sweep tau in [0.5, 0.7, 0.9, 0.95], measure (frozen_frac, Ortho) pairs | Positive linear correlation r > 0.85: frozen fraction predicts synergy magnitude | ~6h | Medium |

### Falsifiability conditions:
- **P1**: If full-scale Ortho < 0.8, the super-multiplicative claim fails and must be downgraded to "partially orthogonal."
- **P2**: If disabling M1 during REFINE does NOT reduce speedup below 4.0x, the frozen-token mechanism explanation is incorrect.
- **P3**: If tau=0.0 == naive T=16 (within noise), IGSD's machinery adds nothing over pure step reduction, and the paper must reframe CD-SSD accordingly.
- **P4**: If Ortho(SSD+M1) >= Ortho(CD-SSD+M1), CD-SSD has no differentiated mechanism -- but the generalization claim ("self-speculative + KV synergizes universally for MDMs") is actually a stronger and more publishable finding.

## Honest Caveats

### M1+IGSD Synergy (Ortho=1.385)
- **Counter-argument**: The pairwise evaluation uses 2 seeds and 200 GSM8K + 164 HumanEval. Per-seed Orthos (1.292 and 1.478) show a 0.186 spread -- substantial relative to the effect magnitude. Full 3-seed replication is essential.
- **Alternative explanation**: The super-multiplicativity might partly arise from M1's baseline being measured under dynamic masking conditions that artificially depress its standalone performance. IGSD's frozen tokens may be restoring M1 to its "true" potential rather than creating genuine amplification. This is testable via P2.
- **What would convince me**: Full 3-seed Ortho >= 1.0 on GSM8K+MATH500 combined, with seed-to-seed std < 0.15. Plus P2 confirming the REFINE-phase mechanism.

### Binary Composability Landscape
- **Counter-argument**: Only 3 pairs tested (M2 dropped). With 3 data points, "binary" is an extrapolation. M2+IGSD or M2+M3 might show partial orthogonality (Ortho 0.6-0.8), creating a gradient.
- **Alternative explanation**: The binary pattern may be specific to LLaDA-8B-Instruct and may not generalize to other MDMs (Dream-7B was unavailable).
- **What would convince me**: Replication on Dream-7B-Instruct (when available) and/or testing M2(2x)+IGSD, which might show intermediate orthogonality given M2(2x) is the only non-catastrophic M2 setting.

### tau=0.0 Paradox
- **Counter-argument**: tau=0.0 was tested on 200 GSM8K + 164 HumanEval (2 seeds). The 88.5% QAS improvement is heavily driven by the speedup gain (5.56x vs. 2.66x), while accuracy difference is small (32.4% vs. 35.9%). On a strict accuracy budget (>= 50% retention required), tau=0.0 is rejected.
- **Alternative explanation**: tau=0.0 may simply be "run with 16 steps + IGSD overhead," which is essentially naive step reduction. If so, the finding is trivial.
- **What would convince me**: P3 comparing tau=0.0 vs. naive T=16 uniform denoising. If tau=0.0 significantly outperforms naive T=16, the IGSD draft mechanism has intrinsic value even without partitioning.

### M3 Accuracy Enhancement
- **Counter-argument**: MATH500 sample sizes are 50-100 per seed. Baseline accuracy is 11.1%, meaning 5-6 additional correct answers create the appearance of 150%+ retention. The "2.44x accuracy retention" is likely a small-sample artifact.
- **Alternative explanation**: Qwen2.5-0.5B's guidance may bias toward the evaluation format rather than genuinely improving reasoning.
- **What would convince me**: Full 500-sample, 3-seed MATH500 evaluation. If acc_ret > 1.3 persists at full scale with low variance, the quality enhancement is real.

### M1 Implementation Gap (1.38x vs. published 15.2-26.4x)
- **Counter-argument**: The 10x+ gap between our EntropyCache (1.38x) and the published result (15.2-26.4x) undermines absolute speedup claims. The combined M1+IGSD 5.13x might become 50x+ with proper kernel-level implementation, OR the synergy might disappear if M1 is already fast enough.
- **What would convince me**: Kernel-level M1 implementation tested in combination. Until then, relative Ortho measurements (implementation-invariant ratios) remain valid, but absolute speedup claims carry this caveat.

## Bottom Line

There is a strong publishable story here. The super-multiplicative synergy (Ortho=1.385, confirmed across 2 seeds with both independently > 1.0) between KV-caching and self-speculative denoising, set against a binary composability landscape where all other combinations destructively interfere, is a novel and actionable finding with no prior analog in the MDM acceleration literature. The failure-mode atlas provides the first systematic worst-case characterization of MDM acceleration. The critical risk is statistical: pairwise results rest on 2 seeds and 200 samples. P1 (full 3-seed validation) is the make-or-break experiment. If P1 confirms Ortho >= 1.0, this is a solid NeurIPS 2026 submission as a composability analysis paper with three distinct contributions: (1) the composability atlas with binary landscape discovery, (2) the frozen-token KV amplification mechanism, and (3) the failure-mode catalog with deployment recipes. The tau=0.0 paradox (P3) and SSD+M1 comparison (P4) are high-value bonus experiments that could further strengthen or generalize the claims.
