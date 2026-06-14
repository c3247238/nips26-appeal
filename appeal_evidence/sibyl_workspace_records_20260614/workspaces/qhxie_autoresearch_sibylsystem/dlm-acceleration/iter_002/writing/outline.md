# Paper Outline: ComposeAccel (Iteration 2)

## Title

**ComposeAccel: Systematic Composition and Orthogonality Analysis of Training-Free Acceleration Methods for Diffusion Language Models**

---

## 1. Abstract

Key elements (all numbers from iter_002 verified JSON artifacts):

- Diffusion Language Models (DLMs) such as LLaDA-8B and Dream-7B generate text via iterative denoising but remain 2--15x slower than autoregressive (AR) models. Over 20 training-free acceleration methods target three axes: KV cache approximation, adaptive step scheduling, and guided/speculative decoding.
- No prior work measures how these methods compose. We conduct ComposeAccel, the first controlled factorial study of training-free DLM acceleration composition.
- Three individual methods: M1 (entropy-based KV caching, 1.16x measured / 2.27x projected speedup), M3 (AR-guided unmasking, 1.65x speedup at 103.9% accuracy retention on GSM8K), IGSD (information-geometric step distillation, up to 2.81x speedup). M2 (naive step scheduling) excluded: structural NO_GO.
- Pairwise composition results: M1+IGSD achieves near-orthogonal composition (Ortho=0.96 on GSM8K, 2.75x speedup); M3+IGSD shows task-dependent behavior (near-orthogonal on GSM8K Ortho=0.96, interference on MATH500); M1+M3 shows destructive interference (Ortho=0.41--0.52) driven by speed penalty from dual model loading.
- Three-way composition (M1+IGSD+M3_off): 1.71x speedup, QAS=1.07, Ortho=1.02 (3-seed mean, stable). Adding M3 guidance (gw=0.3) drops Ortho to 0.49 due to TPS overhead.
- Cross-model validation on Dream-7B-Instruct confirms composition patterns transfer with amplified effects (Dream M1+IGSD+M3_off QAS=2.18 on GSM8K vs. LLaDA QAS=1.07).
- AR baseline comparison: Qwen2.5-7B at batch=1 achieves 96% GSM8K accuracy at 71 TPS (vs. LLaDA 71.2% at 34 TPS), establishing a 3.08x QAS gap that composed DLM acceleration narrows but does not close.
- IGSD ablation: T_draft=32 is Pareto-optimal (QAS=1.16 vs. T_draft=16 QAS=1.51 with 27% lower accuracy); confidence partitioning (tau>0) adds marginal quality gain.

---

## 2. Introduction (1.5 pages)

### 2.1 The Fragmentation Problem
- DLM inference acceleration has produced 20+ independent methods in Q1 2026 alone, targeting three computational axes. Each paper evaluates in isolation under different protocols, models, and benchmarks.
- Practitioners face a concrete deployment question: which combination maximizes throughput at acceptable quality? No published answer exists.
- *Ref: Figure 1 (teaser) -- speed-quality landscape showing individual methods and compositions.*

### 2.2 The Composition Gap
1. **Unknown composability**: Methods targeting different bottlenecks should compose multiplicatively, but DLM's global mask-state coupling creates hidden interaction risks.
2. **No failure characterization**: Published methods report average-case results; practitioners cannot predict when compositions fail catastrophically.
3. **No honest AR comparison**: DLM acceleration papers omit comparison with properly optimized AR inference, making practical value unclear.

### 2.3 Contributions (numbered)
1. **First systematic composition study** of three training-free DLM acceleration families across three axes, with formal orthogonality metric (Ortho) and corrected Quality-Adjusted Speedup (QAS).
2. **Composition taxonomy**: M1+IGSD is near-orthogonal (Ortho=0.96); M3+IGSD is task-dependent; M1+M3 shows destructive interference from overhead accumulation.
3. **IGSD (Information-Geometric Step Distillation)**: A 50-line training-free step scheduler using inter-step logit KL divergence. T_draft=32 is Pareto-optimal.
4. **Cross-model validation**: Top 5 recipes validated on Dream-7B-Instruct show amplified composition effects, confirming transferability.
5. **Honest AR comparison**: Best composed DLM acceleration reaches QAS=1.07 vs. AR baseline QAS=3.08 (Qwen2.5-7B, batch=1), quantifying the remaining gap.
6. **Practical recipes**: Task-specific recommended combinations with validated hyperparameters.

### 2.4 Scope Statement
- Analysis paper, not a methods paper. IGSD is a composability study vehicle.
- Primary model: LLaDA-8B-Instruct; generalization: Dream-7B-Instruct.
- Primary benchmarks: GSM8K (1319, full), MATH500 (500); HumanEval reported in appendix (2.4% baseline, uninformative). MBPP dropped (0% baseline).
- M1 reports measured cache hit rate (56--99%) and projected speedup (2.27--2.47x) because d2Cache kernel integration was 15.2x slower than HF baseline due to eager attention overhead.
- M2 excluded (NO_GO); reported as negative result.

*Transition: Section 3 surveys related work; Section 4 defines methods and metrics; Section 5 reports results.*

---

## 3. Background and Related Work (1.5 pages)

### 3.1 Masked Diffusion Language Models
- LLaDA forward/reverse process: T=64 denoising steps, bidirectional attention, block-based semi-autoregressive generation.
- Inference bottleneck: each step requires full O(N^2) forward pass; global mask state changes prevent standard KV reuse.

### 3.2 Training-Free Acceleration Families
- **KV-Cache Approximation (M1)**: EntropyCache, Fast-dLLM (ICLR 2026), dKV-Cache, Elastic-Cache, Window-Diffusion. Reuse KV for stable positions across steps; differ in refresh criterion.
- **Adaptive Step Scheduling (M2)**: Saber, PRR. Reduce total steps by unmasking more tokens per step. Risk: discrete masking violates DDIM-style schedule assumptions.
- **AR-Guided Unmasking (M3)**: FlashDLM. Lightweight AR model biases unmasking order toward high-confidence tokens.
- **Speculative Denoising**: SSD, SSMD, S2D2, DualDiffusion. Our IGSD is concurrent with these; differentiated by coarse-step draft mechanism and composability focus.

### 3.3 The Composability Gap in Literature
- All methods evaluated in isolation. No pairwise orthogonality measurement across families.
- Closest work: TORS (arXiv:2603.00763) notes text-to-image diffusion methods are "developed independently, leaving compatibility unexplored" -- we address the same gap for language DLMs.
- Kolbeinsson et al. (2024) measure composable LLM interventions (compression, editing) -- different intervention types, not inference acceleration.

*Transition: Section 4 defines our framework.*

---

## 4. Methods (2.5 pages)

### 4.1 Composability Framework
- **Quality-Adjusted Speedup (QAS)**: `QAS = Speedup * AccuracyRetention`, where `AccRet = Acc(method) / Acc(baseline)`. No penalty factor.
- **Orthogonality Metric (Ortho)**: `Ortho(A+B) = QAS(A+B) / max(QAS(A), QAS(B))`. Ortho > 1.0 = synergy; 0.8--1.0 = near-orthogonal; < 0.8 = interference.
- **Combined metric**: `0.7 * GSM8K + 0.3 * MATH500` for all QAS, Ortho, and Pareto computations.
- *Ref: Table 2 (metric definitions and interpretation thresholds).*

### 4.2 M1: Entropy-Based KV Caching
- Entropy threshold eta determines which positions reuse cached KV.
- Cache hit rate (CHR) measured at 56--99% across eta={0.5, 1.0, 2.0}.
- d2Cache kernel integration attempted but produced 15.2x framework overhead; M1 speedup reported as projected from measured CHR.
- *Ref: Figure 2 (IGSD architecture + M1/M3 integration points).*

### 4.3 IGSD: Information-Geometric Step Distillation
- Draft phase: run T_draft steps (16, 32, or 48 out of 64).
- Partition: tokens with logit confidence above tau threshold are frozen (S_accept); remaining tokens enter refine phase.
- Refine phase: continue from step T_draft to T_full=64 on non-frozen tokens only.
- KL-divergence between consecutive steps guides the partition threshold.
- Algorithm pseudocode (5 lines).

### 4.4 M3: AR-Guided Unmasking
- Qwen2.5-0.5B generates guidance logits at each denoising step.
- Guidance weight gw={0.3, 0.5, 0.7} interpolates between DLM and AR logits for masked positions.
- Quality-preserving: GSM8K accuracy retention 103--104% across gw values.

### 4.5 Experimental Setup
- **Models**: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (generalization).
- **Hardware**: 2x NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM each).
- **Benchmarks**: GSM8K (1319 samples, exact match), MATH500 (500 samples, exact match), HumanEval (164, pass@1, appendix only).
- **Seeds**: 42, 123, 456 for full experiments; seed=42 for pilots.
- **Baseline**: LLaDA-8B 64-step denoising, bf16, greedy. GSM8K=71.2% +/- 1.5%, 34 TPS. MATH500=11.1% +/- 0.7%.

*Transition: Section 5 presents results in four parts.*

---

## 5. Results (3.5 pages)

### 5.1 Single-Method Pareto Curves
- M1 (eta=0.5): 1.16x speedup, 94.5% AccRet on GSM8K. CHR=56%. Projected speedup 2.27x if kernel-level cache implemented.
- M1 (eta=2.0): 1.37x speedup, 67.2% AccRet. CHR=99%. Higher threshold trades accuracy for speed aggressively.
- IGSD (tau=0.9, T_draft=32): 1.71x speedup, 67.8% AccRet, QAS=1.16 on GSM8K. Pareto-optimal among IGSD configs.
- IGSD (tau=0.7, T_draft=16): 2.81x speedup, 58.2% AccRet, QAS=1.64 on GSM8K. Maximum speed at large accuracy cost.
- M3 (gw=0.3): 1.65x speedup, 102.5% AccRet on GSM8K, QAS=1.69. M3 at gw=0.7: 1.65x, 103.9% AccRet, QAS=1.71. M3 alone is quality-preserving.
- *Ref: Table 3 (single-method Pareto table: Method x {Speedup, AccRet, QAS, CHR/AcceptRate} on GSM8K + MATH500).*
- *Ref: Figure 3 (Pareto curves: speed vs. accuracy for each method individually).*

### 5.2 Pairwise Composition Analysis
- **M1+IGSD** (best config: eta=0.5, tau=0.7, T_draft=16): GSM8K Ortho=0.99, 2.75x speedup, 58.9% AccRet. MATH500 Ortho=0.64. Combined Ortho=0.96. **Near-orthogonal**.
  - Mechanism: IGSD's frozen tokens create low-entropy KV entries that M1 exploits at 83% CHR.
- **M3+IGSD** (best: tau=0.7, T_draft=16, gw=0.7): GSM8K Ortho=0.96, 2.72x speedup. MATH500 Ortho=0.76. Combined Ortho=0.84. **Task-dependent** (near-orthogonal on GSM8K, interference on MATH500).
  - Mechanism: AR guidance on IGSD's compressed trajectory provides marginal quality improvement on GSM8K reasoning but MATH500 is too difficult for the guide model.
- **M1+M3** (all configs): GSM8K Ortho=0.51--0.52, Combined Ortho=0.41--0.43. **Destructive interference**.
  - Root cause: M3 adds Qwen2.5-0.5B forward pass overhead (~12% per step), reducing TPS to 50 (0.86x baseline). Combined with M1's marginal measured speedup (1.16x), the composition is slower than M3 alone.
  - iter_001 discrepancy resolved: pilot Ortho=1.34 was an artifact of 100-sample variance; iter_002 full-scale confirms interference.
- *Ref: Table 4 (pairwise Ortho matrix: 3 pairs x 2 benchmarks + combined, with confidence intervals from 3-seed validation where available).*
- *Ref: Figure 4 (Ortho bar chart: pairwise comparison with per-benchmark breakdown).*

### 5.3 Three-Way Composition and Pareto Frontier
- Top 5 configs evaluated on 3 seeds each. All stable (QAS CV < 10%).
- **Max-Speed** (M1 eta=0.5 + IGSD tau=0.85 td=32 + M3 off): 1.71x speedup, 62.7% AccRet, QAS=1.07, Ortho=1.02. The Pareto-optimal operating point.
- **Balanced-A** (M1 eta=1.0 + IGSD tau=0.9 td=32 + M3 off): 1.68x speedup, 63.3% AccRet, QAS=1.07, Ortho=1.03. Marginally better accuracy.
- **Quality-First** (M1 eta=0.5 + IGSD tau=0.85 td=32 + M3 gw=0.3): 1.68x speedup, 62.7% AccRet, QAS=1.05, Ortho=0.49. Adding M3 guidance hurts: overhead dominates any quality gain.
- Key finding: M3 guidance (gw>0) consistently reduces Ortho from ~1.0 to ~0.5 in three-way compositions. M3 is beneficial only as a standalone method, not as a composition layer.
- *Ref: Table 5 (three-way Pareto table: top 5 configs with speedup, AccRet, QAS, Ortho, per seed).*
- *Ref: Figure 5 (speed-quality Pareto frontier: individual methods, pairwise, and three-way compositions on same axes).*

### 5.4 IGSD Ablation
- **T_draft sweep** (fixed tau=0.9): T_draft=16 gives QAS=1.51 (2.50x, 60.3% AccRet); T_draft=32 gives QAS=1.16 (1.71x, 67.8%); T_draft=48 gives QAS=1.05 (1.44x, 72.7%). T_draft=32 is Pareto-optimal when accuracy preservation > 65% is required.
- **tau sweep** (fixed T_draft=32): tau=0.7 gives QAS=1.17; tau=0.85 gives QAS=1.17; tau=0.9 gives QAS=1.16. Minimal sensitivity to tau in [0.7, 0.9] range.
- **Confidence gate ablation** (tau=0.0 vs tau=0.9 at T_draft=32): tau=0.0 (no partitioning) is equivalent to naive step reduction; tau=0.9 adds 1--2 pp accuracy on GSM8K. Marginal improvement.
- **KL divergence profile**: Measured on 100 GSM8K samples. Monotonically decreasing (NOT inverted-U as hypothesized). H6 not confirmed. The monotonic profile explains why tau sensitivity is low: later steps always have lower KL, making the partition boundary insensitive to tau.
- *Ref: Figure 6 (T_draft ablation: QAS and accuracy vs. T_draft).*
- *Ref: Figure 7 (per-step KL divergence profile, averaged over 100 samples).*

---

## 6. Cross-Model and AR Comparison (1.5 pages)

### 6.1 Dream-7B-Instruct Validation
- Dream-7B baseline: GSM8K=36.0% (vs. LLaDA 71.2%), 64.5 TPS.
- Max-Speed recipe on Dream: GSM8K AccRet=125%, QAS=2.18 (vs. LLaDA QAS=1.07). Composition amplifies on Dream due to lower baseline accuracy and higher relative IGSD speedup.
- Composition patterns transfer: M1+IGSD without M3 remains the best recipe on Dream-7B as well.
- *Ref: Table 6 (cross-model comparison: LLaDA vs Dream, top 5 recipes).*

### 6.2 AR Baseline Comparison
- **Qwen2.5-7B greedy (batch=1)**: GSM8K 96%, 71 TPS, QAS=3.08 (vs. LLaDA baseline).
- **Qwen2.5-7B greedy (batch=8)**: GSM8K 96%, 471 TPS, QAS=20.5.
- **Qwen2.5-7B + speculative decoding (batch=1)**: GSM8K 97%, 48 TPS, QAS=2.12. Speculative decoding with HF-native implementation is slower than greedy due to overhead.
- **Speculative decoding at batch=8**: HF does not support assisted generation at batch>1.
- Honest assessment: Best composed DLM (QAS=1.07) falls 2.9x short of AR greedy at batch=1. DLMs close no speed gap vs. AR models; composition study value is in understanding the design space, not in claiming DLM speed parity.
- *Ref: Table 7 (AR vs DLM comparison at batch={1, 8}).*

### 6.3 Batch Size Sensitivity
- M1+IGSD at batch=1: 1.64x speedup, 96 TPS. At batch=4: accuracy improves to 50% (from 45%) but TPS drops to 56. At batch=8: accuracy 52%, TPS=34.
- Larger batch sizes reduce per-sample IGSD speedup because confidence profiles are averaged across samples in a batch, reducing accept rate specificity.
- *Ref: Figure 8 (batch size sensitivity: TPS and accuracy vs. batch size for key compositions).*

---

## 7. Discussion (1 page)

### 7.1 Why M1+IGSD Composes but M1+M3 Does Not
- M1+IGSD synergy mechanism: IGSD's draft-partition-refine pipeline creates two distinct phases. In the refine phase, frozen tokens (S_accept, alpha=88.6% of generation tokens) have near-zero entropy, giving M1 a high-quality signal for cache reuse. The 83.4% CHR during composition is not significantly different from standalone M1 (56--93% depending on eta), but the reduced step count amplifies the per-step savings.
- M1+M3 interference mechanism: M3 requires loading Qwen2.5-0.5B (0.95 GB VRAM) and running it at every denoising step. This adds ~12% wall-clock overhead per step. Since M1's measured speedup (1.16x) is marginal, the M3 overhead negates M1's gains entirely.

### 7.2 Implications for DLM Acceleration Design
- Method composition is NOT free: overhead stacks subadditively. Practitioners should prioritize single high-impact methods (M3 alone for quality, IGSD alone for speed) over multi-method stacking.
- KV caching requires kernel-level integration to deliver published speedups (15--26x). Without it, measured speedup (1.16x) is insufficient to absorb composition overhead.
- The "quality insurance" role of M3 is better served as a standalone post-processing step, not as a composition layer.

### 7.3 Limitations
- Single model family (LLaDA + Dream). Generalization to MDLM, SEDD, or larger models untested.
- M1 speedup is projected (from CHR), not directly measured with kernel-level implementation. Ortho metric is dimensionless and valid regardless, but absolute combined speedup may differ.
- MATH500 baseline is weak (11.1%), limiting statistical power on that benchmark.
- HumanEval/MBPP baselines degenerate (2.4%/0%), effectively excluded.
- d2Cache integration failure is specific to Blackwell GPUs and eager attention; may succeed on different hardware/framework.

---

## 8. Conclusion (0.5 pages)

- ComposeAccel provides the first systematic composition study of training-free DLM acceleration methods.
- M1+IGSD is the only near-orthogonal pair (Ortho=0.96); all compositions involving M3 guidance suffer overhead interference.
- Three-way composition at M1+IGSD achieves 1.71x speedup at QAS=1.07, confirmed stable across 3 seeds and transferable to Dream-7B.
- DLM acceleration does not yet close the gap with optimized AR inference (QAS 1.07 vs. 3.08).
- Released: acceleration recipes, IGSD implementation (50 lines), and composability benchmark suite for future work.

---

## Figure & Table Plan

### Figure 1: Speed-Quality Landscape Teaser (Section: Introduction)
- **Purpose**: Immediately show readers the composition design space and where individual/composed methods land.
- **Type**: scatter plot with Pareto frontier overlay
- **Content**: X-axis = speedup (relative to baseline), Y-axis = accuracy retention (%). Points for M1, IGSD, M3 (individual) and M1+IGSD, M3+IGSD, M1+M3 (pairwise) plus three-way compositions. Color by method axis (KV cache = blue, step scheduling = orange, AR guidance = green, compositions = red). Pareto frontier line connecting non-dominated points.
- **Key takeaway**: Composition does NOT always improve the Pareto frontier; M1+M3 falls below both individual methods.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `iter_002/exp/results/full_m1/`, `igsd_pareto/`, `full_m3/`, `pairwise/`, `threeway/`

### Figure 2: ComposeAccel Architecture Diagram (Section: Method)
- **Purpose**: Show the IGSD draft-partition-refine pipeline and how M1 KV caching and M3 AR guidance integrate.
- **Type**: architecture_diagram / flow_chart
- **Content**: Left: standard 64-step denoising (baseline). Right: IGSD splits into draft (T_draft steps) -> partition (confidence gate at tau) -> refine (remaining steps on non-frozen tokens). M1 cache reuse shown as blue arrow across steps; M3 AR guidance shown as green arrow from Qwen2.5 to mask logits.
- **Key takeaway**: IGSD creates a natural boundary at T_draft where M1 cache efficiency peaks due to frozen tokens.
- **Generation**: tikz or manual_diagram
- **Data source**: Method description, algorithm pseudocode

### Figure 3: Single-Method Pareto Curves (Section: Results 5.1)
- **Purpose**: Show the speed-accuracy tradeoff for each method individually before composition.
- **Type**: line_plot with error bands
- **Content**: Three subplots (M1, IGSD, M3). X-axis = speedup, Y-axis = accuracy retention. Each point = one hyperparameter configuration. Error bars from 3-seed experiments where available.
- **Key takeaway**: M3 is uniquely quality-preserving (AccRet > 100%); IGSD trades accuracy for speed on a continuum; M1 has a narrow operating range.
- **Generation**: code (matplotlib)
- **Data source**: `iter_002/exp/results/full_m1/m1_pareto_full.json`, `igsd_pareto/igsd_pareto_corrected.json`, `full_m3/m3_pareto_full.json`

### Figure 4: Pairwise Orthogonality Bar Chart (Section: Results 5.2)
- **Purpose**: Compare Ortho scores across all three pairs, broken down by benchmark.
- **Type**: grouped bar_chart
- **Content**: 3 groups (M1+IGSD, M3+IGSD, M1+M3). Within each: GSM8K Ortho, MATH500 Ortho, Combined Ortho. Horizontal lines at Ortho=1.0 (synergy threshold) and Ortho=0.8 (near-orthogonal threshold).
- **Key takeaway**: M1+IGSD is near-orthogonal; M1+M3 shows clear interference across both benchmarks.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `iter_002/exp/results/pairwise/m1_igsd_full.json`, `m3_igsd_full.json`, `m1_m3_full.json`

### Figure 5: Combined Pareto Frontier (Section: Results 5.3)
- **Purpose**: Show the full Pareto frontier including individual, pairwise, and three-way compositions.
- **Type**: scatter with Pareto frontier lines
- **Content**: All evaluated configurations plotted on speedup vs. accuracy retention. Pareto frontier highlighted. Color by composition order (single = circles, pairwise = squares, three-way = diamonds). Key operating points labeled (Max-Speed, Balanced-A, Quality-First).
- **Key takeaway**: Three-way compositions do not significantly extend the pairwise Pareto frontier; the main gains come from pairwise M1+IGSD.
- **Generation**: code (matplotlib)
- **Data source**: `iter_002/exp/results/threeway/threeway_pareto_full.json`

### Figure 6: IGSD T_draft Ablation (Section: Results 5.4)
- **Purpose**: Show the effect of draft step count on speed-quality tradeoff.
- **Type**: line_plot (dual y-axis)
- **Content**: X-axis = T_draft {16, 32, 48}. Left Y-axis = QAS. Right Y-axis = accuracy retention (%). Two lines: GSM8K and MATH500.
- **Key takeaway**: T_draft=32 is Pareto-optimal; T_draft=16 maximizes speed at severe accuracy cost.
- **Generation**: code (matplotlib)
- **Data source**: `iter_002/exp/results/ablation/igsd_ablation_refined.json` (part1_tdraft_sweep)

### Figure 7: Per-Step KL Divergence Profile (Section: Results 5.4)
- **Purpose**: Validate (or refute) the inverted-U KL profile hypothesis and explain tau insensitivity.
- **Type**: line_plot with shaded std band
- **Content**: X-axis = denoising step (0 to 63). Y-axis = mean token-level KL(p_t || p_{t-1}). Shaded region = +/- 1 std across 100 samples. Horizontal lines at tau={0.7, 0.85, 0.9} for reference.
- **Key takeaway**: KL profile is monotonically decreasing, NOT inverted-U. Later steps always have lower KL, explaining why tau sensitivity is minimal.
- **Generation**: code (matplotlib)
- **Data source**: `iter_002/exp/results/ablation/igsd_kl_profiles_raw.json`

### Figure 8: Batch Size Sensitivity (Section: Cross-Model/AR 6.3)
- **Purpose**: Show how composition effects change with batch size.
- **Type**: grouped bar_chart (dual y-axis)
- **Content**: X-axis = batch size {1, 4, 8}. Bars: TPS (left y-axis). Line overlay: accuracy (right y-axis). Two series: M1+IGSD and M1+IGSD+M3.
- **Key takeaway**: Larger batches reduce per-sample IGSD speedup; accuracy slightly improves due to batch averaging effects.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `iter_002/exp/results/batch_sensitivity/batch_sensitivity.json`

### Table 1: Related Work Speed Comparison (Section: Introduction/Related Work)
- **Purpose**: Position ComposeAccel within the DLM acceleration landscape.
- **Content**: Method | Axis | Model | Reported Speedup | Eval Protocol | Composition Tested?
- Rows for Fast-dLLM, EntropyCache, Saber, FlashDLM, SSD, SSMD, S2D2, DualDiffusion, KLASS.

### Table 2: Metric Definitions (Section: Methods 4.1)
- **Purpose**: Formally define QAS, Ortho, combined metric, and interpretation thresholds.

### Table 3: Single-Method Pareto Table (Section: Results 5.1)
- **Content**: Method | Config | Speedup | AccRet | QAS | CHR/AcceptRate | Benchmark
- Rows for M1 x3 configs, IGSD x9 configs, M3 x3 configs. Bold Pareto-optimal.

### Table 4: Pairwise Orthogonality Matrix (Section: Results 5.2)
- **Content**: Pair | GSM8K Ortho | MATH500 Ortho | Combined Ortho | Verdict
- 3 rows. Bold near-orthogonal pairs.

### Table 5: Three-Way Pareto Operating Points (Section: Results 5.3)
- **Content**: Recipe | Config | Speedup (mean +/- std) | AccRet | QAS | Ortho | Verdict
- 5 rows for top configs. Per-seed breakdown in appendix.

### Table 6: Cross-Model Comparison (Section: Cross-Model 6.1)
- **Content**: Recipe | LLaDA GSM8K QAS | Dream GSM8K QAS | LLaDA MATH500 QAS | Dream MATH500 QAS
- 5 rows for top configs.

### Table 7: AR vs DLM Comparison (Section: AR Comparison 6.2)
- **Content**: System | Batch | GSM8K Acc | GSM8K TPS | QAS | MATH500 Acc | MATH500 TPS
- Rows: LLaDA baseline, LLaDA best-composed, Qwen2.5-7B greedy b1, Qwen2.5-7B greedy b8, Qwen2.5-7B spec b1.

---

## Visual Storytelling Flow Summary

| Section | Visual Element | Purpose |
|---------|---------------|---------|
| Introduction | Figure 1 (teaser scatter) | Hook: show the messy composition landscape |
| Methods | Figure 2 (architecture) | Explain IGSD + integration points |
| Results 5.1 | Table 3 + Figure 3 | Individual method baselines |
| Results 5.2 | Table 4 + Figure 4 | Pairwise composition results |
| Results 5.3 | Table 5 + Figure 5 | Three-way Pareto frontier |
| Results 5.4 | Figure 6 + Figure 7 | IGSD ablation and KL profile |
| Cross-Model | Table 6 | Dream-7B validation |
| AR Comparison | Table 7 | Honest positioning |
| Batch Size | Figure 8 | Production relevance |
