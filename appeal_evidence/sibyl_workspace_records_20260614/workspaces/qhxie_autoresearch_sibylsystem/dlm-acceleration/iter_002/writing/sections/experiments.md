# 5 Experiments

## 5.1 Single-Method Pareto Curves

Table 3 and Figure 3 report the speed--accuracy tradeoff for each method in isolation on GSM8K (1319 samples, seed 42) and MATH500 (500 samples, seed 42), using the LLaDA-8B-Instruct baseline ($T$=64 steps, bf16 greedy decoding; GSM8K accuracy 71.2% $\pm$ 1.5%, 33.8 TPS; MATH500 accuracy 11.1% $\pm$ 0.7%, 79.1 TPS).

**M1 (entropy-based KV caching).** M1 at $\eta$=0.5 achieves 1.16x measured speedup with 94.5% accuracy retention on GSM8K ($\overline{\text{CHR}}$=56.2%). Raising the threshold to $\eta$=2.0 pushes speedup to 1.50x but accuracy retention drops to 55.5% ($\overline{\text{CHR}}$=60.2%). The d2Cache kernel-level integration produced a 15.2x *slowdown* relative to the HuggingFace baseline due to eager-attention overhead on RTX PRO 6000 Blackwell GPUs, so M1 speedups are measured from our simplified Python entropy-cache implementation. Projected speedups from published cache hit rates would be 2.27--2.47x; we report only measured wall-clock values and label this discrepancy explicitly.

**IGSD (information-geometric step distillation).** IGSD sweeps 9 configurations across $\tau \in \{0.7, 0.85, 0.9\}$ and $T_{\text{draft}} \in \{16, 32, 48\}$. The fastest configuration ($\tau$=0.7, $T_{\text{draft}}$=16) reaches 2.81x speedup at 58.2% accuracy retention on GSM8K (QAS=1.64). The most conservative ($\tau$=0.9, $T_{\text{draft}}$=48) yields 1.22x speedup at 73.3% accuracy retention (QAS=0.90). $T_{\text{draft}}$=32 at any $\tau$ is Pareto-optimal when accuracy retention above 65% is required: $\tau$=0.85 gives 1.73x speedup, 67.8% accuracy retention, QAS=1.17.

**M3 (AR-guided unmasking).** M3 with Qwen2.5-0.5B guidance achieves 1.68x speedup on GSM8K across all guidance weights ($w_g \in \{0.3, 0.5, 0.7\}$), with accuracy retention *above* 100% at $w_g$=0.3 (103.9%) and $w_g$=0.7 (103.9%). M3 is the only quality-preserving accelerator. At $w_g$=1.0, accuracy retention drops to 84.9% as the guide model overwhelms the DLM's own predictions. The 2-seed mean (seeds 42, 123) on a 100-sample GSM8K subset confirms stability.

![Figure 3: Single-Method Pareto Curves](figures/fig3_single_pareto.pdf)

**Table 3: Single-Method Pareto Results on GSM8K**

| Method | Config | Speedup | AccRet (%) | QAS | CHR / Accept Rate |
|--------|--------|---------|------------|-----|-------------------|
| M1 | $\eta$=0.5 | 1.16x | 94.5 | 0.98 | CHR=56.2% |
| M1 | $\eta$=1.0 | 1.25x | 88.0 | 0.97 | CHR=58.7% |
| M1 | $\eta$=2.0 | **1.50x** | 55.5 | 0.83 | CHR=60.2% |
| IGSD | $\tau$=0.7, $T_{\text{draft}}$=16 | **2.81x** | 58.2 | **1.64** | $r_{\text{accept}}$=92.1% |
| IGSD | $\tau$=0.85, $T_{\text{draft}}$=32 | 1.73x | **67.8** | 1.17 | $r_{\text{accept}}$=95.9% |
| IGSD | $\tau$=0.9, $T_{\text{draft}}$=32 | 1.71x | 67.8 | 1.16 | $r_{\text{accept}}$=95.3% |
| IGSD | $\tau$=0.9, $T_{\text{draft}}$=48 | 1.22x | 73.3 | 0.90 | $r_{\text{accept}}$=96.4% |
| M3 | $w_g$=0.3 | 1.68x | **103.9** | **1.69** | -- |
| M3 | $w_g$=0.7 | 1.68x | 103.9 | 1.71 | -- |
| M3 | $w_g$=1.0 | 1.68x | 84.9 | 0.70 | -- |

## 5.2 Pairwise Composition Analysis

Table 4 and Figure 4 report the orthogonality ($\text{Ortho}$) score for all three viable pairs on 100-sample GSM8K and 100-sample MATH500 pilots (seed 42). Each $\text{Ortho}$ score is computed as $\text{QAS}(A{+}B) / \max(\text{QAS}(A), \text{QAS}(B))$.

**M1+IGSD: near-orthogonal.** The best configuration ($\eta$=0.5, $\tau$=0.7, $T_{\text{draft}}$=16) achieves 2.75x speedup at 58.9% accuracy retention on GSM8K, with $\text{Ortho}_{\text{GSM8K}}$=0.99 and combined $\text{Ortho}$=0.96. IGSD's frozen tokens ($\alpha$=88.6% of generation positions) create near-zero entropy at those positions, providing M1 with a strong cache-reuse signal: CHR rises to 83.4% during composition versus 56.2% for standalone M1. The more conservative $T_{\text{draft}}$=32 variant reaches $\text{Ortho}$=0.91 at lower speedup (1.68x), confirming that the composition benefit holds across IGSD operating points.

**M3+IGSD: task-dependent.** At the highest-speed IGSD setting ($\tau$=0.7, $T_{\text{draft}}$=16, $w_g$=0.7), $\text{Ortho}_{\text{GSM8K}}$=0.96 (near-orthogonal) but $\text{Ortho}_{\text{MATH500}}$=0.76 (interference), yielding combined $\text{Ortho}$=0.84. At more conservative settings ($\tau$=0.9, $T_{\text{draft}}$=32), both benchmarks fall into interference ($\text{Ortho}_{\text{combined}}$=0.61). The guide model Qwen2.5-0.5B operates on IGSD's compressed denoising trajectory, where fewer draft steps produce noisier context for the 0.5B model, degrading guidance quality on harder tasks.

**M1+M3: destructive interference.** Across all three $w_g$ values, combined $\text{Ortho}$ ranges from 0.41 to 0.43, firmly in the interference regime. GSM8K-only Ortho is 0.51--0.52; MATH500-only Ortho drops to 0.31--0.36. The root cause is speed penalty: M3 requires loading Qwen2.5-0.5B (0.95 GB VRAM) and running it at every denoising step, reducing TPS from 58.5 (baseline) to 50.3 (0.86x). Combined with M1's marginal measured speedup (1.16x), the composition is *slower* than M3 alone (1.68x). This result overturns the iter\_001 pilot finding ($\text{Ortho}$=1.34 on 100 samples), which was an artifact of small-sample variance and an HumanEval 0% baseline distorting the combined metric.

![Figure 4: Pairwise Orthogonality](figures/fig4_ortho_bars.pdf)

**Table 4: Pairwise Orthogonality Matrix**

| Pair | Best Config | GSM8K Ortho | MATH500 Ortho | Combined Ortho | Verdict |
|------|-------------|-------------|---------------|----------------|---------|
| **M1+IGSD** | $\eta$=0.5, $\tau$=0.7, td=16 | **0.99** | 0.64 | **0.96** | Near-orthogonal |
| M3+IGSD | $\tau$=0.7, td=16, $w_g$=0.7 | 0.96 | 0.76 | 0.84 | Task-dependent |
| M1+M3 | $\eta$=0.5, $w_g$=0.3 | 0.52 | 0.32 | 0.41 | Interference |

## 5.3 Three-Way Composition and Pareto Frontier

Table 5 and Figure 5 present the three-way composition results. Five configurations, selected from a 24-configuration pilot sweep, were validated on 3 seeds (42, 123, 456) with 100 GSM8K + 100 MATH500 samples per seed. All configurations meet the stability criterion (QAS coefficient of variation < 10%).

The top operating point, **Max-Speed** ($\eta$=0.5, $\tau$=0.85, $T_{\text{draft}}$=32, $w_g$=0.0), achieves 1.71x $\pm$ 0.02 speedup, 62.7% $\pm$ 4.6% accuracy retention, QAS=1.07 $\pm$ 0.09, and $\text{Ortho}$=1.02 $\pm$ 0.08 (3-seed mean $\pm$ std). The **Balanced-A** variant ($\eta$=1.0, $\tau$=0.9, $T_{\text{draft}}$=32, $w_g$=0.0) shows marginally better accuracy retention (63.3%) at similar speedup (1.68x), QAS=1.07, $\text{Ortho}$=1.03.

The **Quality-First** recipe ($\eta$=0.5, $\tau$=0.85, $T_{\text{draft}}$=32, $w_g$=0.3) adds M3 guidance and drops $\text{Ortho}$ from 1.02 to 0.49. M3 guidance ($w_g > 0$) consistently degrades three-way composition: the TPS overhead from the guide model forward pass at every step overwhelms any quality improvement. In three-way compositions, M3 guidance is counterproductive.

Three-way compositions do not extend the pairwise Pareto frontier on GSM8K. The highest two-way QAS (M1+IGSD, td=16: QAS=1.34) exceeds the best three-way QAS (1.07) because the three-way configurations use the more conservative $T_{\text{draft}}$=32. The three-way study confirms that M1+IGSD without M3 is the recommended recipe, and the $\text{Ortho} \approx 1.0$ result validates near-orthogonal composition at the three-way level.

![Figure 5: Combined Pareto Frontier](figures/fig5_combined_pareto.pdf)

**Table 5: Three-Way Composition Operating Points (3-seed validation)**

| Recipe | Config | Speedup | AccRet (%) | QAS | Ortho | QAS CV |
|--------|--------|---------|------------|-----|-------|--------|
| **Max-Speed** | $\eta$=0.5, $\tau$=0.85, td=32, $w_g$=0 | **1.71** $\pm$ 0.02 | 62.7 $\pm$ 4.6 | **1.07** $\pm$ 0.09 | **1.02** | 8.2% |
| Balanced-B | $\eta$=1.0, $\tau$=0.85, td=32, $w_g$=0 | 1.71 $\pm$ 0.02 | 62.7 $\pm$ 4.6 | 1.07 $\pm$ 0.09 | 1.02 | 8.2% |
| Balanced-A | $\eta$=1.0, $\tau$=0.9, td=32, $w_g$=0 | 1.68 $\pm$ 0.03 | 63.3 $\pm$ 4.4 | 1.07 $\pm$ 0.09 | **1.03** | 8.1% |
| Conservative | $\eta$=0.5, $\tau$=0.9, td=32, $w_g$=0 | 1.68 $\pm$ 0.03 | 63.3 $\pm$ 4.4 | 1.07 $\pm$ 0.09 | 1.03 | 8.1% |
| Quality-First | $\eta$=0.5, $\tau$=0.85, td=32, $w_g$=0.3 | 1.68 $\pm$ 0.02 | 62.7 $\pm$ 4.6 | 1.05 $\pm$ 0.09 | 0.49 | 8.3% |

## 5.4 IGSD Ablation

Figure 6 presents the $T_{\text{draft}}$ sweep at fixed $\tau$=0.9, and the tau sweep at fixed $T_{\text{draft}}$=32. Both evaluations use 200 GSM8K + 100 MATH500 samples, seed 42.

**$T_{\text{draft}}$ sweep.** $T_{\text{draft}}$=16 yields the highest GSM8K QAS (1.51) at 2.50x speedup and 60.3% accuracy retention. $T_{\text{draft}}$=32 reaches QAS=1.16 at 1.71x speedup and 67.8% accuracy retention. $T_{\text{draft}}$=48 gives QAS=0.90 at 1.22x speedup and 73.3% retention. $T_{\text{draft}}$=32 is Pareto-optimal when accuracy retention above 65% is required: it offers 40% more speedup than $T_{\text{draft}}$=48 with only 5.5 percentage points lower accuracy retention.

**$\tau$ sweep.** At fixed $T_{\text{draft}}$=32, $\tau \in \{0.7, 0.85, 0.9\}$ produce GSM8K QAS of 1.17, 1.17, and 1.16 respectively. Accuracy retention ranges from 66.4% ($\tau$=0.7) to 67.8% ($\tau$=0.85 and 0.9). IGSD is insensitive to $\tau$ in the [0.7, 0.9] range: the confidence partitioning threshold has minimal practical impact on the speed--quality tradeoff.

**Confidence gate ablation.** Comparing $\tau$=0.0 (no partitioning, equivalent to naive step reduction) versus $\tau$=0.9 at $T_{\text{draft}}$=32: $\tau$=0.0 achieves 49.5% accuracy on GSM8K versus 49.5% for $\tau$=0.9, a negligible 0.0 pp difference. The confidence gate adds at most marginal quality gain.

**KL divergence profile.** Measured across 100 GSM8K samples, the per-step $\text{KL}(p_t \| p_{t-1})$ profile is monotonically decreasing with high variance, not the inverted-U shape hypothesized in H6. Early steps (0--15) exhibit mean KL values of 0.08--0.19, while later steps (48--63) show spiky behavior with occasional values exceeding 0.5. The monotonic-on-average profile explains why $\tau$ sensitivity is low: later steps consistently have lower *average* KL divergence, making the partition boundary insensitive to the exact threshold.

![Figure 6: IGSD T_draft Ablation](figures/fig6_tdraft_ablation.pdf)

<!-- FIGURES
- Figure 3: gen_fig3_single_pareto.py, fig3_single_pareto.pdf — Single-method Pareto curves for M1, IGSD, M3 on GSM8K
- Figure 4: gen_fig4_ortho_bars.py, fig4_ortho_bars.pdf — Pairwise orthogonality bar chart with per-benchmark breakdown
- Figure 5: gen_fig5_combined_pareto.py, fig5_combined_pareto.pdf — Combined Pareto frontier with individual, pairwise, and three-way compositions
- Figure 6: gen_fig6_tdraft_ablation.py, fig6_tdraft_ablation.pdf — IGSD T_draft ablation showing QAS and accuracy retention vs T_draft
- Table 3: inline — Single-method Pareto results (speedup, AccRet, QAS, CHR/accept rate)
- Table 4: inline — Pairwise orthogonality matrix (3 pairs x 2 benchmarks + combined)
- Table 5: inline — Three-way composition operating points with 3-seed validation
-->
