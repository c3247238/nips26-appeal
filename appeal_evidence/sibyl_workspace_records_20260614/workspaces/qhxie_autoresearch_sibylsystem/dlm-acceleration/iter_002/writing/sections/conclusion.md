# 8. Conclusion

ComposeAccel provides the first controlled factorial study of training-free acceleration composition for diffusion language models. Three method families -- entropy-based KV caching (M1), information-geometric step distillation (IGSD), and AR-guided unmasking (M3) -- were evaluated individually, in all three pairwise combinations, and in three-way compositions on LLaDA-8B-Instruct across GSM8K (1319 samples) and MATH500 (500 samples), with 3-seed validation on the top five configurations.

The central empirical finding is that composition outcomes are predictable from the interaction mechanism between methods:

- **M1+IGSD composes near-orthogonally.** Combined Ortho = 0.96, with 2.75x speedup at 58.9% accuracy retention on GSM8K. IGSD's draft-partition-refine pipeline creates frozen tokens ($\alpha = 0.886$) with near-zero entropy, which M1 exploits at 83% cache hit rate. The two methods target non-overlapping computational bottlenecks -- step count and per-step KV reuse -- and their overhead profiles do not conflict.

- **M3+IGSD is task-dependent.** GSM8K Ortho = 0.96; MATH500 Ortho = 0.76; combined Ortho = 0.84. AR guidance from Qwen2.5-0.5B improves accuracy on GSM8K reasoning but provides no benefit on MATH500, where the guide model's own accuracy is insufficient. The composition is viable for reasoning-focused deployments but not for general use.

- **M1+M3 interferes destructively.** Combined Ortho = 0.41--0.43 across all configurations. M3 adds ~12% wall-clock overhead per step from the Qwen2.5-0.5B forward pass. Since M1's measured speedup without kernel-level integration is only 1.16x, the M3 overhead negates M1's gains entirely. The composition is slower than M3 alone.

Three-way composition confirms the pairwise structure. The Pareto-optimal operating point -- M1 ($\eta = 0.5$) + IGSD ($\tau = 0.85$, $T_{\text{draft}} = 32$) with M3 off -- achieves 1.71x speedup, QAS = 1.07, and Ortho = 1.02 (3-seed mean, QAS CV < 10%). Activating M3 guidance ($w_g = 0.3$) drops Ortho to 0.49 due to per-step TPS overhead, confirming that M3's value is as a standalone quality-preserving accelerator, not as a composition layer.

Cross-model validation on Dream-7B-Instruct shows that composition patterns transfer with amplified effects: the Max-Speed recipe achieves QAS = 2.18 on Dream-7B GSM8K versus 1.07 on LLaDA, with 4 of 5 recipes showing consistent synergy patterns (average transfer ratio = 1.86). The amplification reflects Dream-7B's lower baseline accuracy (36% vs. 71.2% GSM8K), which inflates accuracy retention under IGSD's draft mechanism.

An honest AR comparison establishes the remaining gap. Qwen2.5-7B at batch=1 achieves 96% GSM8K accuracy at 71 TPS (QAS = 3.08 relative to the LLaDA baseline), versus 1.07 for the best composed DLM. At batch=8, the AR advantage widens to QAS = 20.5. Training-free composition narrows but does not close the DLM-AR speed-quality gap. The value of this study is in mapping the composition design space, not in claiming DLM speed parity with AR inference.

**Limitations.** All experiments use LLaDA-8B-Instruct and Dream-7B-Instruct; generalization to MDLM, SEDD, or models above 10B parameters is untested. M1 speedup is projected from measured cache hit rates because d2Cache kernel integration produced 15.2x framework overhead on Blackwell GPUs with eager attention; the Ortho metric is dimensionless and valid regardless, but absolute combined speedup may differ with kernel-level caching. MATH500 baseline accuracy (11.1%) limits statistical power on that benchmark, and HumanEval (2.4% baseline) and MBPP (0% baseline) were excluded as uninformative.

**Released artifacts.** The IGSD implementation (50 lines), all acceleration recipes with validated hyperparameters, and the composability benchmark suite are released to support future composition studies as the DLM acceleration landscape continues to expand.

<!-- FIGURES
- None
-->
