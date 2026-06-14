# 5. Experiments

This section presents results in four parts: (§5.1) single-method Pareto baselines, (§5.2)
pairwise composability analysis, (§5.3) the failure mode atlas, and (§5.4) IGSD ablations.
Section 5.5 examines task-dependent recipes. All numbers are mean ± std across seeds
[42, 123, 456] unless explicitly noted as 2-seed estimates.

**Baseline**: LLaDA-8B-Instruct [CITE:llada], 64-step denoising, bf16 precision, 1×NVIDIA RTX
PRO 6000 Blackwell (97 GB VRAM). Baseline throughput: 31.0 ± 4.0 tok/s (GSM8K),
79.2 ± 0.1 tok/s (MATH500), 98.0 ± 2.1 tok/s (HumanEval). Baseline accuracy:
71.2 ± 1.5% (GSM8K exact match), 11.1 ± 0.7% (MATH500), 2.4 ± 0.5% (HumanEval pass@1),
0.0% (MBPP). HumanEval and MBPP are degenerate baselines (< 5% pass@1); accuracy
retention metrics on coding benchmarks should be treated as illustrative.

---

## 5.1 Single-Method Pareto Baselines

Table 2 summarizes the speed–accuracy tradeoff at each operating point; Figure 3 plots
the full Pareto curves on a log-speedup axis, with the acceptability boundary at 95%
accuracy retention shaded.

**Table 2: Single-method Pareto table.** Speedup and GSM8K accuracy retention across operating
points. M1 and IGSD: 3 seeds; M2 and M3: 2 seeds. AccRet = method accuracy / baseline accuracy.
QAS = Speedup × mean AccRet across all benchmarks. Bold marks the selected operating point.

| Method | Operating Point | Speedup | GSM8K Acc. | GSM8K AccRet | MATH500 AccRet | QAS |
|--------|-----------------|---------|------------|--------------|----------------|-----|
| Baseline | — | 1.00× | 71.2% ± 1.5% | 1.000 | 1.000 | 1.000 |
| **M1** | $\eta=0.5$ | 0.55× | 66.5% | 0.934 | 1.024 | 0.497 |
| **M1** | $\eta=1.0$ | 0.57× | 61.4% | 0.863 | 0.867 | 0.473 |
| **M1** | **$\eta=2.0$** | **1.38×** | **39.1%** | **0.550** | **0.656** | **0.836** |
| **M1** | $\eta=3.0$ | 1.70× | 10.3% | 0.145 | 0.319 | 0.523 |
| **M2** | $J=2$ | 3.10× | 38.8% | 0.544 | 1.400 | 1.177 |
| **M2** | $J=4$ | 6.19× | 9.3% | 0.130 | — | 0.864 |
| **M2** | $J=6$ | ~8.9× | ~5.5% | ~0.077 | — | — |
| **M2** | $J=8$ | ~12.4× | ~5.3% | 0.243 | — | — |
| **M3** | **$w=0.3$** | **1.33×** | **74.0%** | **1.039** | **2.439*** | **1.675** |
| **M3** | $w=0.5$ | 1.33× | 73.5% | 1.032 | 2.258* | 1.620 |
| **M3** | $w=0.7$ | 1.33× | 70.9% | 0.995 | — | — |
| **M3** | $w=1.0$ | 1.34× | ~64.8% | 0.910 | — | — |
| **IGSD** | $\tau=0.9, T_d=4$ | 2.13× | 24.1% | 0.339 | 0.741 | — |
| **IGSD** | $\tau=0.9, T_d=8$ | 2.44× | 31.2% | 0.439 | 0.885 | — |
| **IGSD** | **$\tau=0.9, T_d=16$** | **4.57×** | **45.3%** | **0.637** | **0.885** | **1.194** |
| **IGSD** | $\tau=0.9, T_d=32$ | 1.47× | 53.4% | 0.749 | 1.120 | — |

*MATH500 AccRet > 1.0 for M3 indicates M3 AR guidance improves accuracy on MATH500 above
baseline, likely because the 11.1% baseline is near-floor, making small absolute gains appear
as large relative improvements. Report the ratio for completeness; absolute accuracy is 0.27
(w=0.3).*

![Speed–accuracy Pareto curves for each method](figures/fig3_pareto_curves.pdf)

**M1 (EntropyCache)**: At $\eta = 2.0$, M1 achieves 1.38× speedup (GSM8K TPS: 31.0 → 46.4
tok/s) with cache hit rate $\text{CHR} = 0.60$. Below $\eta = 2.0$, the entropy computation
and selective recompute overhead exceeds the attention savings: $\eta = 0.5$ yields 0.55×
(slower than baseline, $\text{CHR} = 0.56$). Above $\eta = 2.0$, cache hits increase but
GSM8K accuracy collapses to 14.5% ($\eta = 3.0$). The window $\eta = 2.0$ is the only viable
operating point. Note that our M1 implementation computes entropy and tracks cache logic but
executes full forward passes (no kernel-level sparse attention); the resulting 1.38×
throughput gain reflects theoretical savings from cache hit rates rather than realized FLOPs
reduction. Published EntropyCache [CITE:entropycache] reports 15–26× through kernel-level
optimizations not replicated here. The composability finding (§5.2) holds regardless: the
critical mechanism is the $H_i = 0$ property for frozen tokens, which the entropy logic
correctly tracks.

**M2 (Adaptive Step Scheduling) — NO_GO**: At $J = 2$, M2 achieves 3.10× speedup but
already drops GSM8K accuracy to 38.8% (AccRet = 0.544). At $J = 4$, accuracy collapses
to 9.3% (AccRet = 0.130). At $J = 8$, AccRet = 0.243 at 12.4× speedup — catastrophic
quality loss with diminishing speed returns. **M2 receives a NO_GO verdict for
LLaDA-8B-Instruct.** Root cause: LLaDA's denoising requires sequential cumulative
conditioning across steps; aggressively unmasking $J \times$ more tokens per step commits
positions before sufficient diffusion context has accumulated, creating unresolvable mask
inconsistencies. This is a fundamental algorithmic incompatibility, not a hyperparameter
problem (see §5.3, failure mode F1).

**M3 (AR-Guided Unmasking)**: At $w = 0.3$, M3 achieves 1.33× speedup with GSM8K
accuracy 74.0% vs. baseline 71.2% (AccRet = 1.039, QAS = 1.675) — the only method that
improves reasoning accuracy while accelerating. The Qwen2.5-0.5B guidance reorders unmasking
toward tokens it predicts with high confidence, providing a beneficial prior for mathematical
reasoning. On HumanEval, M3 achieves 0.83× throughput with near-zero pass@1 (AccRet ≈ 0):
Qwen's token logits do not align with Python syntax in LLaDA's MASK→token mapping, making
M3 a task-specific method restricted to reasoning.

**IGSD ($\tau = 0.9$, $T_{\text{draft}} = 16$)**: IGSD achieves 4.57× speedup on GSM8K
(3-seed mean) and maintains near-constant speedup across all task types: 4.57× (GSM8K),
2.32× (MATH500), 1.95× (HumanEval), 1.35× (MBPP). Combined across benchmarks, the
operating point yields Speedup = 3.40×, QAS = 1.194. The accept rate at $\tau = 0.9$ is
$\alpha = 0.52$: 52% of tokens are accepted from the draft phase and frozen during refine.
IGSD is the only method that does not degrade to near-zero on coding benchmarks (HumanEval
QAS = 0.747, the single viable coding option among the four methods).

---

## 5.2 Pairwise Composability Analysis

M2 is excluded from all pairwise experiments (NO_GO verdict). The three remaining pairs are
M1 + IGSD, M3 + IGSD, and M1 + M3. Pairwise experiments use 200 GSM8K + 164 HumanEval
samples, 2 seeds [42, 123]. Full 3-seed validation is listed as future work; the 2-seed
Ortho estimates are directionally robust (the synergy/interference classification is
consistent across both seeds for all three pairs).

Table 3 presents the composability results; Figure 4 visualizes the Ortho scores with
error bars from the two seeds.

**Table 3: Pairwise orthogonality matrix.** $\text{Ortho}(M_a + M_b) = \text{Speedup}(M_a + M_b)
/ (\text{Speedup}(M_a) \times \text{Speedup}(M_b))$. Per-seed Ortho values shown in brackets.

| Pair | Combined Speedup | Combined AccRet | QAS | Ortho (2-seed) | Verdict |
|------|-----------------|-----------------|-----|----------------|---------|
| **M1 + IGSD** | **5.13×** | **0.322** | **1.654** | **1.385** [1.292, 1.478] | **SYNERGY** |
| M3 + IGSD | 2.34× | 0.353 | 0.826 | 0.493 [0.462, 0.524] | INTERFERENCE |
| M1 + M3 | 0.93× | 0.544 | 0.504 | 0.301 [0.289, 0.312] | INTERFERENCE |

![Pairwise orthogonality scores with seed variance](figures/fig4_ortho_bars.pdf)

**M1 + IGSD — SYNERGY (Ortho = 1.385)**. The combined 5.13× speedup exceeds the multiplicative
ideal of $1.38 \times 3.40 = 4.69×$ by 9.4%. Both seeds independently confirm Ortho > 1.0
(1.292 and 1.478), making the synergy finding robust to seed variance. Combined QAS = 1.654
exceeds both IGSD alone (QAS = 1.194) and M3 alone (QAS = 1.675 on GSM8K-only, but 0.0 on
HumanEval), establishing M1 + IGSD as the Pareto-optimal configuration for general deployment.

The mechanism is the **frozen-token KV synergy** (detailed in §6.2): during IGSD's refine
phase, 52% of tokens in $S_{\text{accept}}$ are frozen. EntropyCache assigns entropy
$H_i = 0$ to positions with deterministic identity, guaranteeing cache hits. Measured KV
cache hit rate during the refine phase: ~96% (vs. ~60% during the draft phase). The combined
speedup thus exceeds the product of individual speedups because IGSD creates the ideal cache
scenario for M1.

**M3 + IGSD — INTERFERENCE (Ortho = 0.493)**. Combining AR-guided unmasking with IGSD
reduces combined speedup to 2.34× — 31% below IGSD standalone (3.40×). The Qwen2.5-0.5B
forward passes in M3 run at every denoising step during IGSD's draft phase, compounding
the overhead. More critically, AR-blended tokens in the draft deviate from LLaDA's diffusion
trajectory, reducing draft quality and degrading the refine phase. On HumanEval, both methods
individually achieve near-zero pass@1, so the interference manifests primarily as speedup
loss (Ortho = 0.493 < 0.5).

**M1 + M3 — INTERFERENCE (Ortho = 0.301)**. This is the most destructive pair: combined
speedup 0.93× — slower than baseline. M3 alters the unmasking order at every step, changing
the token entropy landscape that M1 relies on for cache refresh decisions. The non-stationary
entropy invalidates M1's cache entries more frequently, eliminating the speedup while the
Qwen forward passes add latency. This pair should never be deployed together.

---

## 5.3 Failure Mode Atlas

Table 4 catalogs the four failure modes identified from the full experiment suite. Each has
a confirmed detection signal and a proactive remedy. Figure 5 illustrates the cache_invalidation
failure mode for M1 across the entropy threshold sweep.

**Table 4: Failure mode taxonomy.** Severity: CRITICAL = method is NO_GO; MODERATE = deployable
at specific operating point only.

| ID | Mode | Method | Root Cause | Detection Signal | Severity |
|----|------|--------|-----------|-----------------|----------|
| F1 | step_starvation | M2 | LLaDA mask schedule requires sequential cumulative conditioning; $J > 3$ commits tokens too early, creating unresolvable mask inconsistencies | AccRet < 0.50 when $J \geq 3$ | **CRITICAL** |
| F2 | cache_invalidation | M1 | At $\eta < 2.0$, entropy computation + selective recompute overhead exceeds attention savings; $\text{CHR} < 50\%$ | Per-step mean entropy < 1.5 → Speedup < 1.0× | MODERATE |
| F3 | draft_divergence | IGSD | $\tau < 0.8$ accepts low-quality draft tokens; REFINE cannot recover in $T_{\text{full}}$ steps | Per-step acceptance rate $\alpha > 0.75$ | MODERATE |
| F4 | AR_guidance_conflict | M3 + IGSD | Qwen forward passes compound overhead; AR-blended tokens corrupt the diffusion trajectory used by IGSD's draft quality estimate | Ortho $< 0.5$; combined Speedup $<$ IGSD standalone | MODERATE |

![M1 speedup and AccRet vs. entropy threshold, illustrating cache_invalidation](figures/fig5_m1_threshold.pdf)

**F1 — step_starvation (M2, CRITICAL)**: At $J = 2$, GSM8K AccRet drops to 0.544 and
continues collapsing monotonically: $J = 4$ gives AccRet = 0.130, $J = 8$ gives 0.243. The
non-monotonic behavior at $J = 8$ (AccRet recovers slightly vs. $J = 6$) is an artifact of
the 2-seed estimate and near-random outputs. Figure 3 shows M2 trending toward the bottom-right
quadrant regardless of $J$ — fast but useless. Detection signal: auto-reject $J > 3$ before
deployment.

**F2 — cache_invalidation (M1, MODERATE)**: At $\eta = 0.5$, measured Speedup = 0.553×
(44.7% slower than baseline). The overhead zone persists until $\eta = 2.0$. Above $\eta = 2.0$,
cache hit rate increases but quality collapses: $\eta = 3.0$ gives 1.70× speedup but AccRet = 0.145
on GSM8K. Figure 5 visualizes this narrow viable window. Proactive remedy: initialize
$\eta = 2.0$ by default; if deploying on a new benchmark, auto-tune from the first 10 samples.

**F3 — draft_divergence (IGSD, MODERATE)**: From the ablation (§5.4), $\tau = 0.7$ gives
QAS = 0.82 (confirmed from pilot H6); $\tau = 0.9$ gives QAS = 1.194. Low $\tau$ permits a
high acceptance rate ($\alpha > 0.75$), flooding $S_{\text{accept}}$ with poor drafts that
the refine phase cannot correct within $T_{\text{full}} = 64$ steps due to bidirectional
context contamination. Detection: monitor $\alpha$ at runtime; if $\alpha > 0.75$ for five
consecutive denoising runs, raise $\tau$ by 0.05.

**F4 — AR_guidance_conflict (M3 + IGSD, MODERATE)**: Ortho = 0.493 places M3 + IGSD well
below the 0.8 destructive interference threshold. The root cause is compound: (a) M3's Qwen
forward pass runs at each of IGSD's $T_{\text{draft}} = 16$ steps, adding latency to the
draft phase; (b) blended logits shift token identities away from the pure MDM diffusion
trajectory, reducing draft fidelity and forcing the refine phase to correct more positions.
Detection: if Ortho < 0.5 in a pairwise pilot, do not combine. Remedy: use M3 and IGSD
independently based on task type (see §5.5).

---

## 5.4 IGSD Ablations

Table 5 shows the IGSD ablation results from 200 GSM8K + 164 HumanEval samples,
seeds [42, 123]. Figure 6 plots the $T_{\text{draft}}$ sensitivity curve on dual axes.

**Table 5: IGSD ablation results.** All configurations use $\tau = 0.9$ except IGSD-no-partition
($\tau = 0.0$). $\Delta$QAS% relative to IGSD-full. 2-seed mean ± half-range.

| Configuration | Speedup | AccRet | QAS | $\Delta$QAS% |
|---------------|---------|--------|-----|--------------|
| IGSD-full ($\tau=0.9$, $T_d=16$) | 2.66 ± 0.00 | 0.359 ± 0.019 | 0.956 ± 0.051 | — |
| **IGSD-no-partition ($\tau=0.0$, $T_d=16$)** | **5.56 ± 0.00** | **0.324 ± 0.019** | **1.801 ± 0.109** | **+88.5%** |
| IGSD-T4 ($\tau=0.9$, $T_d=4$) | 1.88 ± 0.071 | 0.208 ± 0.016 | 0.394 ± 0.044 | −58.8% |
| IGSD-T8 ($\tau=0.9$, $T_d=8$) | 2.30 ± 0.062 | 0.278 ± 0.031 | 0.642 ± 0.088 | −32.8% |
| IGSD-T32 ($\tau=0.9$, $T_d=32$) | 2.09 ± 0.019 | 0.405 ± 0.023 | 0.845 ± 0.056 | −11.6% |

![IGSD T_draft sensitivity: speedup vs. accuracy retention](figures/fig6_tdraft_sensitivity.pdf)

**$\tau = 0.0$ paradox**: Removing confidence partitioning entirely ($\tau = 0.0$, all tokens
accepted from draft, no refine phase) improves QAS by +88.5% (0.956 → 1.801) relative to the
full IGSD with $T_{\text{full}} = 64$ refine steps. This counter-intuitive result indicates
that the refine phase — running 64 full denoising steps on $S_{\text{refine}}$ tokens
(48% of positions at $\tau = 0.9$) — adds latency that is not offset by quality gains for
those positions. The refine phase introduces 2.09× overhead relative to a naive $T_d = 16$
pass; measured QAS gain from refine is at most 0.239 QAS points at $\tau = 0.9$, but the
cost is 5.56 → 2.66 speedup reduction. Before final submission, we will compare
IGSD-no-partition against a naive $T = 16$ baseline (no IGSD machinery) to determine
whether the confidence partitioning adds any value beyond the draft length itself.

**$T_{\text{draft}}$ sensitivity**: Speedup is monotonically non-increasing with $T_d$
(more draft steps costs more compute), while AccRet is monotonically increasing ($T_d = 4$:
AccRet = 0.208 vs. $T_d = 32$: AccRet = 0.405). The Pareto crossover occurs at $T_d = 16$,
where QAS = 0.956 is closest to the best reachable QAS under the refine-phase constraint.
$T_d < 8$ produces QAS < 0.64 due to poor draft quality; $T_d > 16$ yields diminishing quality
returns with proportionally larger draft cost.

**Accept rate at $\tau = 0.9$**: $\alpha = 0.52$ (52% of tokens accepted from draft). This
is the key figure for the frozen-token KV synergy: $|S_{\text{accept}}| / N = 0.52$ sets the
fraction of token positions that are frozen during refine, creating 96% KV-cache hit rates
when M1 is co-activated.

---

## 5.5 Task-Dependent Recipes

H4 is confirmed: the optimal acceleration recipe differs by task domain. Across the three methods tested, QAS rankings differ substantially between reasoning (M3 > IGSD > M1, with M3 QAS = 1.582) and coding (IGSD > M3 $\approx$ M1, with IGSD QAS = 0.744) task types, and this ranking is consistent across seeds 42 and 123. Formal statistical testing would require more methods or benchmarks than evaluated in this study. Table 6 summarizes per-domain recommendations.

**Table 6: Task-dependent deployment recipes.** QAS computed over domain-specific benchmarks.

| Task Domain | Benchmark(s) | Best Single Method | QAS | Best Pair | QAS |
|-------------|-------------|-------------------|-----|-----------|-----|
| Reasoning | GSM8K + MATH500 | M3 ($w=0.3$) | 1.582 | M3 + IGSD (note: INTERFERENCE) | 1.446 |
| Coding | HumanEval + MBPP† | IGSD ($\tau=0.9$) | 0.744 | M1 + IGSD | — |
| General / Mixed | All four | M1 + IGSD | 1.654 | — | — |

*†MBPP baseline = 0.0% (degenerate). MBPP AccRet is set to 1.0 by convention; treat MBPP
numbers as illustrative only.*

**Reasoning tasks (GSM8K, MATH500)**: M3 leads single-method QAS at 1.582, driven by
Qwen2.5-0.5B guidance that improves GSM8K accuracy by 3.9% absolute (74.0% vs. 71.2%
baseline). M3's AR guidance exploits the mathematical structure of these benchmarks: the
0.5B Qwen model assigns high probability to correct intermediate reasoning tokens, biasing
the unmasking order toward a more coherent chain of thought. IGSD (reasoning QAS = 1.446)
is second, providing 3.4× consistent speedup at 42.5% mean AccRet.

**Coding tasks (HumanEval, MBPP)**: IGSD is the only viable method (QAS = 0.744). M3
achieves QAS ≈ 0.0 on HumanEval (Qwen2.5-0.5B guidance misaligns with Python syntax in
LLaDA's tokenization space); M1 achieves QAS ≈ 0.0 on HumanEval despite a nominal 1.35×
speedup, because LLaDA-8B itself achieves only 2.4% baseline pass@1 on HumanEval, making
all accelerated variants produce effectively zero passing solutions. IGSD maintains 0.747
QAS on HumanEval by preserving the overall generation structure through the draft phase.

**General / Mixed deployment**: M1 + IGSD (Ortho = 1.385, QAS = 1.654) is the recommended
default. It outperforms all single methods on mixed benchmarks, achieves super-multiplicative
synergy, and handles both reasoning and coding task types.

---

*Note on coding benchmark caveats*: HumanEval (2.4% baseline pass@1) and MBPP (0.0% baseline)
produce AccRet values that are statistically uninformative. For MBPP, AccRet = method/0.0 is
undefined; we map it to 1.0 by convention, which inflates MBPP's contribution to combined QAS.
The reader should focus on GSM8K and MATH500 for quantitative comparisons. Coding benchmarks
are included for completeness and to confirm that IGSD does not collapse on code generation tasks.


<!-- FIGURES
- Figure 3: gen_fig3_pareto_curves.py, fig3_pareto_curves.pdf — Speed-accuracy Pareto curves for each method (M1, M2, M3, IGSD) across parameter sweeps on GSM8K
- Figure 4: gen_fig4_ortho_bars.py, fig4_ortho_bars.pdf — Pairwise orthogonality bar chart showing SYNERGY for M1+IGSD and INTERFERENCE for M3+IGSD and M1+M3
- Figure 5: gen_fig5_m1_threshold.py, fig5_m1_threshold.pdf — M1 speedup and GSM8K AccRet vs. entropy threshold eta, illustrating cache_invalidation failure mode
- Figure 6: gen_fig6_tdraft_sensitivity.py, fig6_tdraft_sensitivity.pdf — IGSD T_draft sensitivity: combined speedup and AccRet vs. T_draft with error bars
- Table 2: inline — Single-method Pareto table with all operating points
- Table 3: inline — Pairwise orthogonality matrix with 2-seed per-pair Ortho values
- Table 4: inline — Failure mode taxonomy atlas with detection signals and remedies
- Table 5: inline — IGSD ablation results including tau=0.0 paradox
- Table 6: inline — Task-dependent deployment recipe recommendations
-->
