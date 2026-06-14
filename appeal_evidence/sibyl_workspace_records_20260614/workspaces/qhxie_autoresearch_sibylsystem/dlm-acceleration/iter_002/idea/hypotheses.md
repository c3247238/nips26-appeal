# Testable Hypotheses with Expected Outcomes

## Primary Hypotheses (Front-Runner: ComposeAccel)

### H1: Composition Subadditivity on Reasoning Tasks

**Statement**: The composition ratio (combined_speedup / product_of_individual_speedups) for three-way combinations of M1 (KV cache) + IGSD (step scheduling) + M3 (AR guidance) is < 0.5 on GSM8K but > 0.7 on MBPP/HumanEval.

**Expected outcome**: Reasoning tasks have stronger inter-token dependencies, causing KV cache errors to amplify when combined with aggressive step scheduling. Code generation has more local structure, allowing methods to compose more cleanly.

**Measurement**: Composition ratio = S_combined / (S_M1 * S_IGSD * S_M3) for each benchmark.

**Falsification**: Disproved if composition ratio > 0.7 on GSM8K at < 2% accuracy loss.

**Prior evidence**: iter_001 M1+IGSD pilot achieved 8.88x speedup; individual speedups were ~0.6x (M1) and ~2.7x (IGSD), giving product ~1.6x. The 8.88x >> 1.6x suggests super-additivity in speedup, but with 48% accuracy loss. The quality-controlled composition ratio requires re-measurement under accuracy constraints.

---

### H2: Quality-First Composition Dominates

**Statement**: M1 + M3 (KV cache + AR guidance) at conservative settings achieves a strictly better Pareto frontier than any single-axis aggressive acceleration (e.g., M2 at 4x step jump, or IGSD at aggressive tau).

**Expected outcome**: M1+M3 achieves 2-3x speedup at > 97% accuracy retention, while M2-4x achieves 4x speedup at < 15% accuracy retention. The M1+M3 quality-adjusted speedup (QAS = speedup * accuracy_retention) dominates.

**Measurement**: Plot Pareto curves of accuracy vs. speedup for single methods and compositions. M1+M3 should be on or above the Pareto frontier.

**Falsification**: Disproved if any single method achieves higher QAS than M1+M3 by > 10%.

**Prior evidence**: iter_001 M1+M3 achieved speedup=2.25x at acc_ret=0.997, QAS=2.24. M2-2x achieved speedup=3.78x at acc_ret=0.544, QAS=2.06. M1+M3 already dominates on QAS.

---

### H3: IGSD as Composable Step Scheduler

**Statement**: IGSD with KL threshold tau achieves 1.5-2.5x step reduction when composed with M1 (KV caching), and the combined speedup is within 80% of the product of individual speedups (composition ratio > 0.8).

**Expected outcome**: IGSD reduces the number of forward passes; M1 reduces the cost of each forward pass. These are orthogonal axes (step count vs. per-step cost), so composition should be near-multiplicative.

**Measurement**: S_M1_IGSD / (S_M1 * S_IGSD) for each benchmark.

**Falsification**: Disproved if composition ratio < 0.5 consistently, indicating that KL-based step decisions are disrupted by cached (approximate) attention.

**Prior evidence**: iter_001 M1+IGSD achieved 8.88x (pilot, 100 samples). With proper kernel-level M1 implementation, we expect 3-5x (M1) * 1.5-2.0x (IGSD) = 4.5-10x combined.

---

### H4: Task-Dependent Optimal Recipes

**Statement**: The Pareto-optimal method combination differs by task category:
- Reasoning (GSM8K, MATH500): M1 (conservative) + M3 (gw=0.3-0.7) -- quality preservation priority
- Code (HumanEval, MBPP): M1 (moderate) + IGSD (tau=0.85) -- speed priority acceptable
- Knowledge (MMLU): M1 (aggressive) alone -- simple factual retrieval tolerates aggressive caching

**Expected outcome**: Different benchmarks have different accuracy-speedup tradeoff curves, leading to different optimal operating points.

**Measurement**: For each benchmark, identify the configuration that maximizes QAS. Compare across benchmarks.

**Falsification**: Disproved if the same configuration is Pareto-optimal across all task types (no task dependence).

**Prior evidence**: iter_001 shows M3 improves MATH500 from 0.111 to 0.260 (seed 42, gw=0.3) but fails entirely on HumanEval (pass@1=0 due to tokenizer mismatch). Task dependence is already evident.

---

## Secondary Hypotheses (IGSD-specific)

### H5: KL Divergence as Sufficient Signal for Step Skipping

**Statement**: Per-token KL divergence between consecutive denoising step logits is a sufficient signal for deciding whether to skip a step. A threshold-based skipper achieves equivalent quality to 64-step uniform decoding using at most 40 effective forward passes (37.5% reduction) on GSM8K.

**Expected outcome**: At moderate KL thresholds (tau ~0.05-0.1), 40-50% of steps can be skipped at the late denoising phase (steps 40-64) where tokens are nearly converged. Early/middle phases require most steps.

**Measurement**: Effective step count at iso-accuracy (within 1% of 64-step baseline) across thresholds.

**Falsification**: Disproved if the best threshold requires > 55 effective steps to match 64-step quality.

**Prior evidence**: iter_001 IGSD accept_rate@tau=0.85 was 0.637 overall (0.589 on GSM8K, 0.829 on HumanEval), confirming that substantial temporal consistency exists.

---

### H6: IGSD Phase Characterization

**Statement**: The inter-step KL divergence follows a characteristic "inverted-U" profile across the denoising trajectory: low at early steps (tokens mostly masked, distributions flat), peaking at intermediate steps (active distributional change), and low again at late steps (tokens converged).

**Expected outcome**: The KL profile should show a clear peak around steps 15-35 (out of 64), matching the "critical regime" identified by the model scheduling literature (arXiv 2604.02340). IGSD skip rate should be highest at steps 1-10 and 50-64.

**Measurement**: Plot mean KL divergence vs. step number across 100 GSM8K samples.

**Falsification**: Disproved if the KL profile is monotonically decreasing (no middle peak) or flat.

**Prior evidence**: iter_001 IGSD showed higher accept rates on HumanEval (0.829) than GSM8K (0.589), consistent with code having more predictable token trajectories.

---

## Backup Hypotheses (for pivot scenarios)

### H7: Per-Token Convergence Hierarchy (Alternative 1)

**Statement**: Different token categories converge at measurably different rates during DLM denoising. Specifically: function words (determiners, prepositions) converge within steps 1-5; common content words within steps 5-15; domain-specific terms within steps 15-40; and novel or context-dependent tokens within steps 40-64.

**Expected outcome**: Measuring per-token KL divergence across denoising steps reveals a clear separation of convergence speeds by token type.

**Falsification**: Disproved if convergence rates show no systematic dependence on token type (all tokens converge at roughly the same rate).

---

### H8: Unmasking Order Confound (Alternative 2)

**Statement**: Changing the unmasking order from default confidence-based to entropy-based priority inversion (moderate-confidence tokens first) improves GSM8K accuracy by > 10pp while adding < 5% latency overhead.

**Expected outcome**: The Flexibility Trap (arXiv 2601.15165) shows that confidence-based order systematically avoids reasoning-critical tokens. Our entropy inversion prioritizes tokens at the decision boundary, providing more context for subsequent hard tokens.

**Falsification**: Disproved if entropy inversion produces < 3pp improvement on GSM8K over default order.

---

### H9: V-Cycle Outperforms Fragments (Alternative 3)

**Statement**: The full V-cycle denoising step (smooth + restrict + correct + prolongate + post-smooth) achieves higher QAS than either smoothing-only (KV cache reuse) or correction-only (token-selective attention) at the same compute budget.

**Expected outcome**: The V-cycle captures the benefit of BOTH cheap global context maintenance (smoothing) and expensive focused dependency resolution (correction), which are complementary rather than redundant.

**Falsification**: Disproved if ablating either the smoothing or correction phase produces < 5% QAS degradation (i.e., the V-cycle structure is not load-bearing).
