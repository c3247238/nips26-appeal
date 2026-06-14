# Codex 独立评审 - review

**评审时间**: 2026-03-10
**模型**: Codex (GPT-5.4 high)

## 评审意见

**Overall score: 4/10 (weak reject)**

The paper identifies a real issue in masked diffusion LMs and reports one genuinely interesting result: `DMI` improves Countdown-500 from `4.7% +/- 0.6` to `9.3% +/- 1.4` at about `1.05x` FLOPs, which is a meaningful relative gain at very low cost. But for a NeurIPS/ICML main-track standard, the paper overclaims on both novelty and evidence. The most ambitious ideas, `BSD` and the `A-CFG` transfer, are supported only by `n=16` pilot runs, where the reported gains correspond to just `1-2` additional solved examples.

### 1. Novelty
The novelty is mixed and, as of March 10, 2026, weaker than the paper seems to suggest. `A-CFG` is not novel here: it was already introduced in *Adaptive Classifier-Free Guidance via Dynamic Low-Confidence Masking* on May 26, 2025, so this paper is mainly an application to Dream-7B. `BSD` is directionally similar to a cluster of recent work that tries to preserve or reuse intermediate uncertainty instead of hard remasking: *Soft-Masked Diffusion Language Models* (October 20, 2025), *EvoToken* (January 12, 2026), *ReMix* (February 26, 2026), and especially *MetaState* (March 2, 2026), which explicitly names the same "Information Island" problem. The distinctive part of `BSD/DMI` is that they are training-free, inference-only, and lightweight. That is a useful angle, but it is an incremental contribution, not a new paradigm.

### 2. Methodology Risks
The biggest methodological risk is post hoc selection. The paper explores many variants, many fail, and the surviving settings are quite brittle: `BSD` only works at `k_frac=0.75`, fails at `<=0.50`, underperforms vanilla on GSM8K, and the supposedly orthogonal combination `BSD+A-CFG` is worse than both components. That pattern raises concern that the final methods may be tuned to a narrow slice of tasks or even to the evaluation set. There is also a serious train-test mismatch in `BSD`: replacing mask embeddings with probability-weighted mixtures is off-distribution for a model trained on hard mask tokens, which likely explains the brittleness. Finally, the "belief state" framing feels stronger than the method warrants; an EMA-smoothed embedding mixture is a heuristic memory mechanism, not a principled posterior state estimator.

### 3. Statistical Concerns
The paper's headline inferences about `BSD` and `A-CFG` are not justified by `n=16`. On Countdown, `A-CFG` and `DMI` each improve from `0/16` to `2/16`; on GSM8K, `A-CFG` improves from `4/16` to `6/16`. Using only these counts, the differences are nowhere near convincing: `2/16` vs `0/16` gives an unpaired Fisher exact `p ~ 0.48`, and `6/16` vs `4/16` gives `p ~ 0.70`. The confidence intervals are also huge. So terms like "generalization," "amplifies reasoning," or "supported hypothesis" are too strong for these pilots. Even for `DMI`, the `p<0.05` claim on Countdown-500 is underspecified: if significance is computed over only `3` seed means, that is weak; the paper should report paired item-level statistics, not just seed-level averages.

### 4. Overlooked Perspectives
The paper underplays simpler alternative explanations. `DMI` may work not because it creates a better "belief state," but because it effectively slows commitment and regularizes premature reveals; that should be compared against simpler controls like lowering reveal rate, increasing steps, or mixing only top-1/top-k embeddings without EMA. The entropy analysis is also weak evidence: decreasing entropy and correlation with correctness can just reflect increased confidence or overconfidence, not better internal state tracking. I would also want stronger external baselines from the current MDLM literature, especially training-free ones such as `ReMDM`, `CORE`, and `ReMix`, plus at least one second backbone beyond Dream-7B to show the effect is not architecture-specific.

### 5. Practical Impact
`DMI` is the only method I can imagine practitioners adopting soon. It is training-free, near-zero overhead, and the only method with full-scale validation. Even so, the absolute accuracy is still only `9.3%`, so this is more a promising knob than a deployable breakthrough. `A-CFG` is much less attractive in practice because it costs about `2x` FLOPs, and on the paper's own compute-fair pilot it merely ties vanilla at `256` steps on Countdown (`12.5%` vs `12.5%`). `BSD` looks too unstable to recommend.

### 6. Strengths
The paper does several things well. It is unusually honest about negative results, which is valuable. It makes a real effort at compute-aware comparison rather than only headline accuracy. It identifies a plausible MDLM bottleneck and tests it through multiple interventions instead of one cherry-picked trick. And `DMI` is a neat practical finding: a small, cheap inference-time change nearly doubles performance on the one benchmark where it is properly validated.

### 7. Concrete Improvement Suggestions
Reframe the paper around `DMI` as the main contribution and clearly label `BSD` and `A-CFG` as exploratory. Run full-scale evaluations for `BSD` and `A-CFG` on at least one more reasoning benchmark, not just `n=16`. Compare directly against `ReMDM`, `CORE`, and `ReMix` under matched compute on Dream-7B. Move all hyperparameter selection to a held-out dev set and keep the test set untouched. Report paired per-example significance tests, confidence intervals, and a correction for multiple comparisons. Add calibration metrics such as ECE/Brier/NLL, not just entropy. Finally, test on a second MDLM backbone to establish whether this is a Dream-7B-specific artifact or a broader inference-time phenomenon.

**Bottom line**
This is a worthwhile exploratory paper with one credible and useful result (`DMI`), but in its current form it is not strong enough for a top-tier acceptance. The conceptual novelty is now crowded by very recent MDLM work, and the strongest claims rest on evidence that is far too small.

**Sources checked for the novelty landscape:**
- [A-CFG](https://arxiv.org/pdf/2505.20199v1)
- [ReMDM](https://arxiv.org/pdf/2503.00307v4)
- [Dream 7B](https://arxiv.org/pdf/2508.15487v1)
- [Soft-Masked Diffusion Language Models](https://arxiv.org/pdf/2510.17206v2)
- [EvoToken](https://arxiv.org/pdf/2601.07351v2)
- [CORE](https://arxiv.org/pdf/2602.04096v2)
- [LR-DLLM](https://arxiv.org/pdf/2602.07546v1)
- [ReMix](https://arxiv.org/pdf/2602.22868v1)
- [MetaState](https://arxiv.org/pdf/2603.01331v1)

## 评分

4/10
