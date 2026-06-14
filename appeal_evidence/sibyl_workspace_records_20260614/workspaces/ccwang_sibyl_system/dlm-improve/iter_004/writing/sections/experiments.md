# Experiments

## Experimental Setup

We evaluate on the full GSM8K test set (`n=1319`) under a unified runtime contract inherited from the retained full-scale iteration-4 bundle. All methods use a shared 64-step draft trajectory, and the current comparison is executed with an eager backend, compile disabled, and flash attention disabled. We compare four methods:

- `cand_espd`: the entropy-routed candidate
- `ESPD-FixedFrontier`: the matched fixed-frontier sham
- `CARD-84`: the shared entropy-centered control
- `RAND-84`: the shared random control

The primary outcome is accuracy. We additionally report quality at equal compute, speed at an equal-quality band, average NFE, active-frontier ratio, and paired repair/harm decomposition. The purpose of this metric set is to separate "small but structured" gains from visually impressive but poorly attributed benchmark movement.

## Main Full-Scale Results

Table 1 summarizes the main full-scale result. The candidate is not the absolute fastest method, but it achieves the highest quality among the four methods in the current bundle while preserving a better quality-speed trade-off than the matched sham.

| Method | Accuracy | Correct | Equal-quality speed (tok/s) | Avg. NFE | Frontier ratio | Effective batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **cand_espd** | **0.4041** | **533** | 124.42 | 67.93 | 0.1211 | 54 |
| ESPD-FixedFrontier | 0.3988 | 526 | 105.73 | 68.00 | 0.1211 | 52 |
| CARD-84 | 0.3965 | 523 | 126.08 | 68.00 | 0.1000 | 57 |
| RAND-84 | 0.3980 | 525 | **128.00** | 67.00 | 0.1000 | 57 |

**Table 1.** Main full-scale GSM8K results under the current runtime contract. The candidate yields the highest accuracy in the bundle, but its contribution is best interpreted as a bounded trade-off improvement rather than an absolute speed win.

As shown by Figure 2, the candidate occupies a different point in the quality-speed plane from the fixed-frontier sham: it is both more accurate and materially faster within the candidate-versus-sham comparison, even though it does not dominate the shared controls in absolute speed.

**Figure 2 (fig2_quality_speed.pdf).** Quality-speed positioning under a unified runtime contract. The candidate is not the absolute fastest method, but it dominates the matched fixed-frontier sham and slightly improves quality over the shared controls.

## Paired Repair/Harm Analysis

The aggregate accuracy gap is small, so the structure of the gain matters. Table 2 reports the paired repair/harm decomposition. Against both shared controls, the candidate yields a modestly positive net repair count. Against the candidate, the sham is net negative.

| Comparison | Fixed | Harmed | Unchanged correct | Unchanged wrong | Net repaired |
| --- | ---: | ---: | ---: | ---: | ---: |
| cand_espd vs RAND-84 | 73 | 65 | 460 | 721 | +8 |
| cand_espd vs CARD-84 | 52 | 42 | 481 | 744 | +10 |
| ESPD-FixedFrontier vs cand_espd | 57 | 62 | 469 | 731 | -5 |

**Table 2.** Paired repair/harm decomposition. The gains are small, but they are not purely an artifact of aggregate averaging; the candidate maintains a modestly positive repair-harm balance.

## Stopping Behavior

Figure 3 visualizes the stopping behavior of the candidate and the sham. The candidate has a strongly bimodal stopping pattern (`702` samples stop after the first frontier-only step, `613` run to the third step, and only `4` stop after the second), which is consistent with non-uniform compute allocation. The sham shows a different distribution even under the same nominal frontier budget family.

**Figure 3 (fig3_stopping_hist.pdf).** Stopping behavior of the candidate and the fixed-frontier sham. The candidate's sharply bimodal pattern supports a routing interpretation, although the absence of a threshold ablation means the mechanism is not yet fully isolated.

## Runtime Lineage

Because the gains are small, runtime lineage must be read together with the main result. Figure 4 shows effective batch size, peak VRAM, and wall-clock latency across the candidate, sham, and shared controls. The candidate uses a smaller effective batch than the shared controls (`54` vs `57`) but a larger batch than the sham (`52`), and it exhibits a much lower peak VRAM footprint than the sham. These differences reinforce the bounded nature of the claim: the current result is credible, but the execution envelope is not yet fully matched or fully optimized.

**Figure 4 (fig4_runtime_lineage.pdf).** Runtime lineage audit across the candidate, sham, and shared controls. The figure makes the execution envelope explicit rather than hiding it in prose.

## Summary

The full-scale experiment supports a narrow conclusion. `cand_espd` is a credible positive line under the current GSM8K contract, and the strongest evidence comes from the candidate-versus-sham comparison rather than from raw superiority over all controls. This result is therefore best read as a bounded positive trade-off, not as a broad benchmark-level win.

<!-- FIGURES
- Figure 2: gen_fig2_quality_speed.py, fig2_quality_speed.pdf — Quality-speed positioning under the unified runtime contract
- Figure 3: gen_fig3_stopping_hist.py, fig3_stopping_hist.pdf — Stopping behavior of the candidate and the sham
- Figure 4: gen_fig4_runtime_lineage.py, fig4_runtime_lineage.pdf — Runtime lineage audit across methods
- Table 1: inline — Main full-scale GSM8K results
- Table 2: inline — Paired repair/harm decomposition
-->
