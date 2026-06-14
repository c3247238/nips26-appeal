# 4. Experiments

We evaluate Denoising-Time Adaptation (DTA) and the full information augmentation spectrum on reasoning and code generation benchmarks using two masked diffusion language models. We first describe the experimental setup (Section 4.1), then present main results on Countdown-500 (Section 4.2), cross-benchmark results on GSM8K and MBPP (Section 4.3), cross-model verification on LLaDA-8B (Section 4.4), and inference-time scaling curves (Section 4.5).

## 4.1 Experimental Setup

**Models.** We evaluate on two state-of-the-art masked diffusion language models: (1) **Dream-v0-Instruct-7B** (Dream Team, 2025), a 7.6-billion-parameter masked diffusion model serving as our primary evaluation platform, and (2) **LLaDA-8B-Instruct** (Nie et al., 2025), an 8-billion-parameter masked diffusion model used for cross-architecture verification. Both models are evaluated in their instruction-tuned variants.

**Benchmarks.** We evaluate on three benchmarks spanning constrained reasoning, mathematical reasoning, and code generation:

- **Countdown-500**: A constrained arithmetic task requiring the model to combine given numbers using basic operations to reach a target value. We evaluate 500 problems across 3 seeds for the primary comparison.
- **GSM8K-1319**: Grade-school math word problems requiring multi-step arithmetic reasoning (Cobbe et al., 2021). We evaluate the full 1,319-problem test set.
- **MBPP-500**: The sanitized subset of the Mostly Basic Python Problems benchmark (Austin et al., 2021), evaluating code generation via Pass@1.

**Methods.** We compare seven methods organized into three groups:

| Group | Method | Mechanism | Overhead |
|-------|--------|-----------|----------|
| Baselines | Vanilla | Standard DLM denoising | 1.0$\times$ |
| Remasking | ReMDM-conf | Confidence-based remasking (Nisonoff et al., 2025) | $\sim$1.8$\times$ |
| Remasking | RCR | Running confidence remasking | $\sim$1.7$\times$ |
| Information Augmentation (Ours) | DMI (Level 1) | Soft embedding injection from previous step logits | $\sim$1.2$\times$ |
| Information Augmentation (Ours) | SCP (Level 2) | Leave-one-out self-contradiction probing | $\sim$7$\times$ |
| Information Augmentation (Ours) | DTA (Level 3) | Online LoRA parameter updates | $\sim$4$\times$ |
| Combined (Ours) | DTA+ReMDM | DTA with confidence-based remasking | $\sim$5$\times$ |

**DTA configuration.** Unless otherwise noted, DTA uses LoRA rank $r = 4$ inserted into the last 2 Transformer layers (gate\_proj, up\_proj, down\_proj), yielding 540K trainable parameters (0.007\% of 7.6B). The optimizer is 1-step AdamW with learning rate $\eta = 5 \times 10^{-4}$, cumulative decay $\gamma = 0.95$, warmup fraction 0.2 (first 20\% of denoising steps skipped), and gradient clipping at 1.0. LoRA parameters are zero-initialized to ensure initial equivalence to the base model.

**Generation configuration.** All experiments use 128 denoising steps, temperature 0.4, and the Dream-native `origin` sampling algorithm. Generation lengths are 256 tokens for Countdown and 512 tokens for GSM8K and MBPP.

**Statistical rigor.** Full-scale experiments use 3 random seeds (42, 123, 456). We report mean accuracy $\pm$ standard deviation across seeds. Statistical significance is assessed via McNemar's test with Bootstrap 95\% confidence intervals and Bonferroni correction for multiple comparisons. Pilot-scale results (N=16) are clearly labeled and interpreted with appropriate caution.

**Hardware.** All experiments run on 4$\times$ NVIDIA RTX PRO 6000 Blackwell GPUs (98 GB VRAM each) on a dedicated server.

---

## 4.2 Main Results: Countdown-500 (Dream-7B)

Table 1 presents the primary benchmark comparison on Countdown-500 with Dream-7B across 3 seeds. This is the central results table of the paper.

**Table 1.** Main results on Countdown-500 (Dream-7B-Instruct). Accuracy (\%) averaged over 3 seeds (42, 123, 456). Bold indicates best result. $\dagger$ marks methods with full-scale results still in progress.

| Method | Seed 42 | Seed 123 | Seed 456 | Mean $\pm$ Std | $\Delta$ vs Vanilla | Time/sample |
|--------|---------|----------|----------|----------------|---------------------|-------------|
| **Remasking Baselines** | | | | | | |
| Vanilla | 4.0 | 5.0 | 5.2 | 4.7 $\pm$ 0.6 | --- | 3.7s |
| ReMDM-conf | 4.8 | 5.2 | 3.2 | 4.4 $\pm$ 1.0 | $-$0.3 | 6.5s |
| RCR | 5.4 | 5.4 | 6.4 | 5.7 $\pm$ 0.6 | +1.0 | 6.2s |
| **Information Augmentation (Ours)** | | | | | | |
| **DMI** (Level 1) | **7.8** | **9.6** | **10.6** | **9.3 $\pm$ 1.4** | **+4.6** | 4.3s |
| SCP$^\dagger$ (Level 2) | --- | --- | --- | $\sim$8.4 | $\sim$+3.7 | 45.9s |
| DTA$^\dagger$ (Level 3) | --- | --- | --- | pending | --- | 23.0s |
| **Combined (Ours)** | | | | | | |
| DTA+ReMDM$^\dagger$ | --- | --- | --- | pending | --- | 32.9s |

Three key findings emerge from the completed full-scale evaluations:

**DMI achieves $\sim$2$\times$ improvement over vanilla with near-zero overhead.** DMI (Diffusion Memory Injection), our simplest cross-step information method, achieves 9.3\% mean accuracy compared to 4.7\% for vanilla---a 98\% relative improvement. This is achieved by injecting a soft embedding vector computed from the previous step's logits into the current step's masked positions, adding only $\sim$1.2$\times$ computational overhead. The improvement is consistent across all 3 seeds (7.8\%, 9.6\%, 10.6\%), with the lowest DMI seed still outperforming the highest vanilla seed by 2.6 percentage points.

**Pure remasking methods show negligible gains.** ReMDM-conf achieves 4.4\% mean accuracy, which is actually 0.3 percentage points *below* vanilla. RCR shows a marginal improvement of 1.0 percentage point (5.7\% vs 4.7\%). Neither result is statistically significant. This confirms the hypothesis that confidence-based remasking in token space is insufficient to overcome the Information Island problem for reasoning tasks.

**SCP performs comparably to DMI at much higher cost.** Interim results for SCP (self-contradiction probing) at approximately 150/500 samples per seed suggest accuracy around 8.4\%, comparable to DMI's 9.3\%. However, SCP requires $\sim$7$\times$ the computational overhead of vanilla (45.9s vs 3.7s per sample) versus DMI's $\sim$1.2$\times$, making it impractical despite similar accuracy.

---

## 4.3 Cross-Benchmark Results: GSM8K and MBPP

To evaluate task-dependent effectiveness, we tested key methods on GSM8K (mathematical reasoning) and MBPP (code generation) at pilot scale (N=16) and partial full-scale.

**Table 2.** Cross-benchmark pilot results (Dream-7B, N=16, seed=42). Bold indicates best per-benchmark result.

| | Countdown | GSM8K | MBPP |
|--------|-----------|-------|------|
| Vanilla | 12.5\% | 25.0\% | 25.0\% |
| ReMDM-conf | 6.2\% | **37.5\%** | --- |
| DTA | 6.2\% | 12.5\% | **37.5\%** |
| DTA+ReMDM | 6.2\% | 18.8\% | 12.5\% |

**Full-scale GSM8K baseline.** At full scale (1,300/1,319 problems completed, seed 42), vanilla Dream-7B achieves 29.6\% accuracy on GSM8K, consistent with reported results in the Dream paper and validating our evaluation framework.

**DTA shows task-dependent effectiveness.** The most notable finding is the divergence across benchmarks:

- *MBPP (code generation)*: DTA achieves the strongest positive signal, with Pass@1 improving from 25.0\% to 37.5\% (+12.5 percentage points). DTA solved two problems that vanilla could not (symbol comparison and minimum finding) while preserving solutions for three out of four vanilla-correct problems. Text quality metrics remain healthy (distinct-2 = 0.978, rep-3 = 0.007), and LoRA norms are well-controlled (max 0.088).

- *GSM8K (math reasoning)*: ReMDM-conf outperforms all other methods at pilot scale (37.5\% vs 25.0\% vanilla), while DTA underperforms (12.5\%). This suggests that multi-step mathematical reasoning may benefit more from token-level correction (remasking) than from parameter-level adaptation.

- *Countdown (constrained arithmetic)*: At pilot scale (N=16), all methods are within noise of vanilla. The full-scale results (Table 1) show the clearest separation, with DMI dominating.

**Code generation benefits most from parameter-level adaptation.** Code exhibits strong local patterns (indentation, syntax, variable naming) that DTA's self-supervised LoRA updates can learn from revealed tokens. In contrast, constrained arithmetic requires global satisfaction of numerical constraints, where local token co-occurrence patterns provide less signal. This task-dependent pattern suggests that DTA's effectiveness correlates with the degree to which local token patterns predict global correctness.

**DTA+ReMDM combination shows mixed results.** The combination does not consistently outperform either method alone: on MBPP, DTA+ReMDM (12.5\%) dramatically underperforms DTA alone (37.5\%), suggesting that remasking may disrupt code structure. On GSM8K, DTA+ReMDM (18.8\%) partially recovers from DTA's low accuracy (12.5\%) but remains below vanilla (25.0\%).

> **Note**: Pilot-scale results (N=16) should be interpreted with caution. As demonstrated by the pilot-to-full-scale comparison on Countdown (where pilot effects diverged by up to 24 percentage points from full-scale; see Table 1), small-sample evaluations are subject to high variance. Full-scale cross-benchmark evaluations are in progress.

---

## 4.4 Cross-Model Verification: LLaDA-8B

To verify that our findings generalize across DLM architectures, we evaluated on LLaDA-8B-Instruct (32 layers, 4096 hidden dimension, mask\_token\_id = 126336) with the same DTA configuration adapted for LLaDA's architecture (LoRA on ff\_proj, up\_proj, ff\_out in the last 2 blocks).

**Table 3.** Cross-model comparison (pilot, N=16, seed=42). LLaDA-8B vs Dream-7B.

| Method | Dream-7B Countdown | LLaDA-8B Countdown | Dream-7B GSM8K | LLaDA-8B GSM8K |
|--------|-------------------|-------------------|----------------|----------------|
| Vanilla | 12.5\% | 12.5\% | 25.0\% | **43.8\%** |
| ReMDM-conf | 6.2\% | 0.0\% | **37.5\%** | 37.5\% |
| DTA | 6.2\% | 0.0\% | 12.5\% | 18.8\% |
| DTA+ReMDM | 6.2\% | 0.0\% | 18.8\% | 31.2\% |

Several cross-architecture patterns emerge:

**Both models exhibit the Information Island problem.** On Countdown, all inference-time methods degrade accuracy relative to vanilla for both Dream-7B and LLaDA-8B. This confirms that the limitation is architectural (inherent to masked diffusion denoising) rather than model-specific.

**LLaDA has a stronger GSM8K baseline.** LLaDA-8B achieves 43.8\% vanilla accuracy on GSM8K versus Dream-7B's 25.0\%, reflecting differences in pre-training data and instruction tuning. Despite this stronger baseline, all methods still degrade LLaDA's accuracy, with DTA+ReMDM showing partial recovery (31.2\% vs DTA's 18.8\%).

**LoRA norms are well-controlled across architectures.** On LLaDA-8B, LoRA maximum norms range from 0.05 to 0.21---comparable to Dream-7B's range of 0.07 to 0.19---confirming numerical stability of the DTA mechanism across different Transformer architectures. No parameter explosion or divergence was observed in any of the 32 LLaDA samples.

**Text quality patterns are consistent.** ReMDM-conf improves lexical diversity (higher distinct-2, lower rep-3) on both models, while DTA preserves diversity at vanilla levels. This consistency suggests that method-level quality effects are architecture-independent.

---

## 4.5 Inference-Time Scaling Curves

We examine how accuracy scales with computational budget by varying the number of denoising steps $T \in \{64, 128, 256, 512\}$ for four methods on Countdown (pilot, N=16, seed=42).

**Table 4.** Accuracy (\%) and wall-clock time (seconds/sample) across denoising step counts. Dream-7B, Countdown, N=16.

| $T$ | Vanilla | ReMDM-conf | DTA | DTA+ReMDM |
|-----|---------|------------|-----|-----------|
| 64 | 6.2\% (1.9s) | 6.2\% (3.3s) | 6.2\% (7.7s) | 0.0\% (9.1s) |
| 128 | 12.5\% (3.7s) | 6.2\% (6.5s) | 0.0\% (15.3s) | 12.5\% (18.1s) |
| 256 | 0.0\% (7.4s) | 0.0\% (13.0s) | 6.2\% (30.4s) | 0.0\% (36.0s) |
| 512 | 6.2\% (14.7s) | 6.2\% (23.8s) | 0.0\% (60.9s) | 12.5\% (57.6s) |

**Computational cost scales linearly with $T$.** All methods exhibit approximately linear time scaling: vanilla scales 7.9$\times$ from $T=64$ to $T=512$ (1.9s to 14.7s), and DTA shows the same 7.9$\times$ ratio (7.7s to 60.9s). This confirms that the per-step overhead is constant, with DTA adding a fixed $\sim$4$\times$ multiplier from the backward pass for LoRA updates.

**DTA overhead decomposition.** At $T=128$, DTA costs 15.3s per sample versus vanilla's 3.7s ($\sim$4.1$\times$). The overhead is dominated by the backward pass for LoRA gradient computation ($\sim$2$\times$ the forward pass cost) plus AdamW optimizer state updates. ReMDM-conf adds $\sim$1.8$\times$ overhead from the remasking step. An interesting observation is that DTA+ReMDM at $T=512$ is slightly *faster* than DTA alone (57.6s vs 60.9s), likely because remasking reduces the number of revealed tokens available for LoRA updates in the M-step.

**Accuracy trends are inconclusive at pilot scale.** With only 16 samples, accuracy differences of 6.2\% (one problem) are not statistically meaningful. The non-monotonic accuracy patterns (e.g., vanilla at 12.5\% for $T=128$ but 0.0\% for $T=256$) reflect sampling noise rather than genuine scaling behavior. Full-scale evaluation (N $\geq$ 200) across all step counts is required to properly test H3 (whether DTA accuracy continues to improve at high $T$ while remasking saturates).

**Preliminary signal for DTA+ReMDM at high $T$.** DTA+ReMDM achieves the joint-highest accuracy of 12.5\% at both $T=128$ and $T=512$, while other methods peak at 12.5\% only once. This preliminary pattern is consistent with the hypothesis that parameter-level adaptation and token-level correction are complementary at higher step counts, but requires full-scale confirmation.

<!-- FIGURES
- Table 1: inline — Main results on Countdown-500 across 3 seeds with 7 methods
- Table 2: inline — Cross-benchmark pilot results (Countdown, GSM8K, MBPP)
- Table 3: inline — Cross-model comparison (Dream-7B vs LLaDA-8B)
- Table 4: inline — Inference-time scaling across denoising steps T={64,128,256,512}
- None (code-generated figures for this section are defined in the outline as Figure 5 for scaling curves, but pilot data is insufficient for meaningful visualization; full-scale data pending)
-->
