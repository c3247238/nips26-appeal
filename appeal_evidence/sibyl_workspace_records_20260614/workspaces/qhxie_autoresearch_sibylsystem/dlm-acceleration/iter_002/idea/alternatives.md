# Backup Research Ideas for Pivot

## Alternative 1: Token-Level Convergence Theory with Empirical Validation

**Source**: Theoretical perspective (Candidate B -- Token Convergence Rate Characterization)

### Summary

Develop per-token convergence rate bounds for masked diffusion language models, proving that the optimal per-token denoising budget is proportional to the square root of the conditional mutual information I(X_i; X_{-i}). Use this theory to (a) formally justify why entropy-based methods (EntropyCache, KLASS) work, (b) rank proxy signals (KL, cosine similarity, tensor variation, entropy) by theoretical optimality, and (c) derive principled per-token scheduling that outperforms heuristic approaches.

### Why This Is a Strong Backup

- **Deep theoretical novelty**: All prior DLM convergence theory (Li & Cai 2025; Chen et al. 2025; Lavenant & Zanella 2025) is sequence-level. Per-token bounds are genuinely new.
- **Unifying framework**: Explains WHY existing heuristic methods work, which is a contribution independent of any new algorithm.
- **Empirical predictions**: Token convergence hierarchy (function words fast, novel terms slow), proxy signal ranking, oracle vs. heuristic scheduling gap.
- **Composable with front-runner**: The theoretical insights directly inform which proxy signals to use in the composition study.

### Key Hypotheses

1. Per-token convergence rate: E[D_KL(p_T(X_i | context) || p(X_i | X_{-i}))] <= I(X_i; X_{-i}) / T_i
2. Optimal budget allocation: T_i* = B * sqrt(I(X_i; X_{-i})) / sum_j sqrt(I(X_j; X_{-j}))
3. Entropy is a computable proxy upper bound for the optimal convergence budget

### Pivot Trigger

Pivot to this alternative if:
- The composition study (front-runner) produces only incremental results (all compositions subadditive with < 20% gain)
- The empirical data from the composition study strongly supports per-token convergence differences that demand theoretical explanation
- A competing composition study is published before ours

### Resource Estimate

- Phase 1 (convergence measurement): 1 hour on LLaDA-8B
- Phase 2 (proxy signal comparison): 1 hour
- Phase 3 (oracle scheduling): 2 hours
- Phase 4 (theory writeup): 4 hours
- Total: ~8 hours experimental + substantial theory work

---

## Alternative 2: Order-First Acceleration -- Fixing What to Accelerate Before How to Accelerate It

**Source**: Contrarian perspective (Candidate B -- Rethinking Unmasking Order)

### Summary

The dominant DLM acceleration paradigm optimizes the SPEED of a fundamentally flawed decoding strategy. Confidence-based unmasking systematically avoids high-entropy reasoning tokens, causing accuracy collapse on GSM8K (22% -> 60.7% with LogicDiff's order correction alone). We propose a two-stage "order-first" pipeline: (1) correct the unmasking order via entropy-based priority inversion (training-free) or lightweight dependency-aware scheduling, then (2) apply standard acceleration methods on top. The key hypothesis: acceleration methods compose more additively under a corrected order because dependency-aware scheduling reduces the inter-method error interaction term.

### Why This Is a Strong Backup

- **Addresses the root cause**: Our iter_001 data shows that M2 (step scheduling) destroys accuracy -- the contrarian perspective argues this is because we are accelerating a broken baseline.
- **Constructive contrarian result**: Not just "DLM acceleration does not work" but "here is how to fix it."
- **Composable**: The order correction is a preprocessing step that stacks with any acceleration method.
- **Training-free variant possible**: Entropy-based priority inversion does not require a trained classifier.

### Key Hypotheses

1. Entropy-based priority inversion (moderate-confidence tokens first) outperforms default confidence-based order on reasoning tasks by > 10pp
2. Acceleration methods compose more additively under corrected order (composition ratio > 0.7) than default order (< 0.5)
3. The order correction adds < 5% overhead per step

### Pivot Trigger

Pivot to this alternative if:
- The composition study (front-runner) reveals that all compositions are severely subadditive on reasoning tasks
- The diagnostic analysis identifies unmasking order as the dominant confound
- The contrarian's prediction (decoding order accounts for > 30% of accuracy variation) is confirmed

### Resource Estimate

- Phase 1 (implement entropy priority inversion): 2 hours
- Phase 2 (single-method + corrected order): 3 hours
- Phase 3 (composition under corrected order): 6 hours
- Total: ~11 hours

---

## Alternative 3: Multigrid V-Cycle Denoising for Diffusion Language Models

**Source**: Interdisciplinary perspective (Candidate A -- Multigrid V-Cycle)

### Summary

Transplant the multigrid V-cycle from numerical analysis to DLM denoising. Each denoising step is decomposed into: (1) SMOOTH: cheap cached-KV pass on all tokens, (2) RESTRICT: identify uncertain tokens, (3) COARSE-GRID SOLVE: full attention on uncertain tokens only, (4) PROLONGATE: inject corrections back, (5) POST-SMOOTH: reconcile. The key insight: the V-cycle's separation of cheap global smoothing from expensive focused correction matches the heterogeneous convergence rates of tokens during denoising.

### Why This Is a Strong Backup

- **Genuinely novel cross-disciplinary transplant**: No prior work implements the V-cycle structure for DLM denoising (verified by search).
- **Composable**: V-cycle determines intra-step computation; step scheduling (IGSD) determines inter-step budget. They compose naturally.
- **Testable predictions**: Smoothing should improve local token quality; correction should improve global coherence. The V-cycle should outperform DyLLM (correction-only) and EntropyCache (smoothing-only).
- **Strong theoretical backing**: Multigrid convergence theory guarantees improvement under smoothing + correction conditions.

### Key Hypotheses

1. V-cycle denoising achieves better speed-quality Pareto than either smoothing-only (KV cache) or correction-only (token selection) approaches
2. The post-smoothing step is necessary for reasoning tasks (> 3pp accuracy improvement) but optional for simpler tasks
3. The uncertain set |U| decreases from ~80% at early steps to ~10% at late steps, enabling progressive speedup

### Pivot Trigger

Pivot to this alternative if:
- The composition study reveals that KV caching (smoothing) and token selection (correction) are highly complementary -- motivating their principled integration via V-cycle
- The theoretical analysis (Alternative 1) confirms heterogeneous per-token convergence rates that map to the multigrid frequency decomposition

### Resource Estimate

- Phase 1 (implement V-cycle on LLaDA): 4 hours
- Phase 2 (ablation: smooth vs. correct vs. V-cycle): 3 hours
- Phase 3 (compose with IGSD step scheduling): 3 hours
- Total: ~10 hours
