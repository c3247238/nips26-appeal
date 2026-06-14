# Experiment Result Analysis

## Key Results Summary

### Cross-Architecture Comparison (5 seeds, 1024 features, 2M samples)

| Variant | Absorption Rate (mean ± std) | L0 Sparsity (mean) | Dead Latents |
|---------|------------------------------|-------------------|--------------|
| Random Control | 0.495 ± 0.035 | 1030 | — |
| Baseline ReLU | 0.254 ± 0.047 | 964 | 0 |
| Gated SAE | 0.257 ± 0.052 | 962 | — |
| Orthogonality SAE | 0.247 ± 0.048 | 550 | — |
| TopK (k=50) | 0.056 ± 0.021 | 50 | 1677 |
| Matryoshka | 0.057 ± 0.023 | 50 | 1151 |

### Dose-Response Study (RQ2): Absorption vs. Lambda_L1

| Lambda_L1 | Absorption Range (5 seeds) | L0 Range | MCC Range |
|-----------|---------------------------|----------|-----------|
| 5e-05 | 0.146 – 0.238 | 801 – 1130 | 0.220 – 0.222 |
| 0.0002 | 0.141 – 0.256 | 748 – 1111 | 0.220 – 0.221 |
| 0.0005 | 0.147 – 0.258 | 733 – 1126 | 0.219 – 0.221 |
| 0.001 | 0.153 – 0.289 | 757 – 1125 | 0.218 – 0.219 |
| 0.002 | 0.176 – 0.319 | 795 – 1102 | 0.217 – 0.219 |

**Key observation**: MCC remains flat (~0.218–0.222) across all lambda values despite absorption varying by 2x. This falsifies the hypothesis that absorption causally degrades feature recovery.

### Downstream Metrics (Feature Recovery MCC)

| Variant | MCC (mean ± std) |
|---------|-----------------|
| Baseline ReLU | 0.216 ± 0.0004 |
| TopK | 0.213 ± 0.0013 |
| Matryoshka | 0.220 ± 0.0004 |

MCC shows virtually no variation across architectures (range: 0.213–0.220), despite absorption rates differing by 4.5x.

---

## Debate Perspectives Summary

- **Optimist**: The results show clear discriminative power (Random 0.495 vs TopK/Matryoshka ~0.056). TopK and Matryoshka consistently show low absorption across 5 seeds. The OrtSAE null result is an important negative finding. The paper answers whether absorption deserves community attention.

- **Skeptic**: L0 mismatch is fatal — TopK/Matryoshka L0=50 vs Baseline L0=964 is apples-to-oranges. MCC may be saturated at ~0.22. Training times of 2–3 seconds for 2M samples suggest possible convergence issues. Dead latent percentages appear buggy (TopK reports 1677 dead out of 2048 latents, but the display may be incorrect). No p-values or confidence intervals reported.

- **Strategist**: The critical next step is L0-matched experiments. If L0 matching eliminates architecture differences, reposition the paper around "L0 confound" as a methodological contribution. If differences persist, the core architecture claim is validated. Both outcomes lead to a defensible paper. Estimated time: ~15 min pilot + ~1 hour full.

- **Comparativist**: This is the first systematic L0-matched comparison and the first causal test of absorption-downstream links. The contribution is novel regardless of outcome. Position vs. SOTA: weak on scale (synthetic data only) but first on L0-matched methodology.

- **Methodologist**: Component isolation is clean, but L0 matching is mandatory before any architecture comparison. Missing: convergence diagnostics, proper statistics (Welch's t-test, Cohen's d, CI), and adequate sample size. Current credibility of "TopK absorbs less": LOW (L0 confounded). Credibility of "OrtSAE has no effect": MEDIUM.

- **Revisionist**: H1c (TopK > Baseline) is contradicted by confounded data. H2a (absorption negatively correlates with MCC) is falsified — no correlation observed. H1b (OrtSAE == Baseline) is supported. The core claim must shift from "architecture effects" to "L0 is the dominant driver of absorption."

---

## Analysis

### 1. Method Feasibility

The core experimental pipeline works. Data integrity mechanisms (MD5 hashing, feature count validation) successfully prevented the iter_006 data replication bug. The absorption metric discriminates extreme cases (Random 0.495 vs TopK 0.056). However, three issues undermine confidence:

- **L0 confound**: The most critical issue. TopK and Matryoshka achieve L0=50 by architectural design (k-selection), while Baseline achieves L0=964 with default lambda. This is not a fair comparison.
- **Convergence unverified**: Training completes in 13–24 seconds for 2M samples. Loss curves were not logged.
- **Dead latent display bug**: TopK reports 1677 dead latents (82% of 2048), Matryoshka reports 1151 (56%). These are raw counts, not percentages, and the display format is inconsistent.

### 2. Performance

- **OrtSAE null result is solid**: 0.247 vs Baseline 0.254, with similar L0 (~550 vs ~964). Even without perfect L0 matching, the orthogonal penalty shows no meaningful absorption reduction. This directly challenges the OrtSAE paper's ~65% reduction claim.
- **Architecture comparison is confounded**: TopK/Matryoshka show 4.5x lower absorption than Baseline, but also 19x lower L0. The dose-response data shows absorption varies with lambda_L1, confirming sparsity is a major driver.
- **MCC is flat across all conditions**: This is either (a) a genuine finding that absorption does not predict downstream performance, or (b) a metric sensitivity issue. The revisionist and optimist treat it as (a); the skeptic and methodologist treat it as (b).

### 3. Improvement Headroom

There is a clear and feasible path to improve the results:

1. **L0-matched Baseline experiment** (~15 min pilot): Train Baseline with tuned lambda_L1 to achieve L0=50. Compare absorption to TopK/Matryoshka. This single experiment unlocks all RQ1 conclusions.
2. **Statistical rigor** (~20 min): Add Welch's t-test, Cohen's d with pooled std, 95% CI, and Bonferroni correction.
3. **Convergence diagnostics** (~10 min): Log loss curves and final loss values.
4. **Fix dead% display** (~5 min): Convert raw counts to percentages.

These improvements are all tractable within a single iteration. The score would rise from 4/10 to 6–7/10.

### 4. Time-Cost Tradeoff

- **Continuing**: ~1–2 hours of experiments and analysis to resolve L0 matching and add statistics. Both outcomes (L0 explains everything vs. architecture effects persist) yield publishable contributions.
- **Pivoting**: Would require discarding the existing experimental infrastructure and starting fresh with one of the backup ideas (temporal dynamics, cross-layer propagation, coherence-constrained architecture, or compositional semantics). None of these have existing pilot data.

The time investment to continue is small relative to the sunk cost and the clarity of the remaining experiments.

### 5. Critical Objections

The skeptic's concerns are serious but addressable:

- **L0 confound**: Addressable via L0-matched pilot (feasible, ~15 min).
- **MCC saturation**: Addressable via sensitivity analysis (per-feature distributions, alternative metrics).
- **Convergence issues**: Addressable via loss curve logging.
- **Statistical incompleteness**: Addressable via pre-registered analysis plan.

None of these are fatal to the project. They are standard next steps for an ongoing experimental program.

---

## Decision Rationale

The project should **PROCEED** with the current direction for the following reasons:

1. **The critical test is feasible and fast**: An L0-matched pilot experiment (Baseline L0=50, 1 seed) takes approximately 15 minutes. This single experiment resolves the central confound that invalidates current architecture claims.

2. **Both outcomes are publishable**:
   - If L0 matching eliminates architecture differences: The paper's primary contribution becomes the "L0 confound" methodological warning — a genuinely novel and important contribution to a field that has been making architecture claims without controlling sparsity.
   - If architecture differences persist after L0 matching: The original architecture comparison claim is validated under rigorous conditions, making it much stronger than prior work.

3. **Solid negative results already exist**: The OrtSAE null result (0.247 vs 0.254) does not depend on L0 matching and directly challenges a published claim. The flat MCC finding — whether interpreted as "absorption doesn't harm downstream performance" or "metric needs refinement" — is a novel observation.

4. **The backup ideas are contingency plans, not urgent pivots**: Alternatives A–D are well-defined but require starting from scratch. They are appropriate responses to specific experimental outcomes (e.g., Alternative D if RQ2 is null), not preemptive escapes from methodological gaps.

5. **The cost of continuing is low, the cost of pivoting is high**: ~1–2 hours of focused experiments resolves the central issue. Pivoting would discard the existing ground-truth synthetic data framework, 5-seed statistical design, and dose-response data.

---

## DECISION: PROCEED
