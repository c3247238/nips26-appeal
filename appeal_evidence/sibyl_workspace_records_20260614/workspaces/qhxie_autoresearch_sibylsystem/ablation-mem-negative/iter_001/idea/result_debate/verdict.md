# Verdict: SAE Feature Absorption -- Iteration 001 Result Assessment

## Overall Score: 5/10

---

## One-Sentence Conclusion

The project has a **genuine novel contribution (unsupervised absorption detection, UAD)** but the current experimental package is **not publication-ready** due to proxy metrics, insufficient statistical power, and four inconclusive primary hypotheses.

---

## What Worked

| Result | Evidence | Confidence |
|--------|----------|------------|
| **UAD (H5-E)** | F1=0.704, perfect recall (1.0), precision=0.543 on GPT-2 Small | HIGH -- exceeds 60% threshold, first unsupervised method |
| **DFDA (H6-E)** | 11.14% avg MSE improvement, 3/4 pairs positive, 388 params | MEDIUM -- small sample, high variance |
| **Framework feasibility** | All 6 experiments completed successfully, pipeline works | HIGH -- technical infrastructure is sound |

## What Did Not Work

| Result | Evidence | Interpretation |
|--------|----------|----------------|
| **H2 (causal link)** | r=0.103, p=0.87 | INCONCLUSIVE -- likely underpowered (n=5) |
| **H3 (sparsity trend)** | r=-0.10, p=0.87 | INCONCLUSIVE -- likely underpowered (n=5) |
| **H4 (layer pattern)** | r=0.088, p=0.87 | INCONCLUSIVE -- likely underpowered (n=6) |
| **H1 (architecture variation)** | Collision rate differs but is NOT true absorption | MISLABELLED -- real observation, wrong interpretation |

## Critical Risks

1. **UAD may not generalize** -- only tested on GPT-2 Small, single seed, single layer
2. **Core metric is a proxy** -- "collision rate" != Chanin et al. absorption
3. **Dead feature ratio is catastrophic** -- 94-99% dead features in trained SAEs
4. **No statistical rigor** -- single seed, no CIs, no power analysis

---

## Competitive Position

- **Leading** in unsupervised absorption detection (UAD) -- no prior work eliminates supervised probe directions
- **Following** in cross-architecture comparison, causal assessment, and mitigation methods
- **Unique angle**: End-to-end detect-then-fix pipeline without supervision

---

## Recommendation: REFINE

### Immediate Actions (Next 1-2 Iterations)

1. **Validate UAD on Gemma-2B or Pythia-2.8B** -- if F1 < 0.6, PIVOT
2. **Run multi-seed replication** (3 seeds) for all key results
3. **Implement true absorption detection** per Chanin et al. protocol
4. **Reframe paper scope** around UAD + DFDA only; move H1-H4 to supplementary/future work

### Go/No-Go Gate

| Condition | Threshold | Decision |
|-----------|-----------|----------|
| UAD cross-model F1 | >= 0.6 on >= 2 models | PROCEED to paper |
| UAD cross-model F1 | < 0.6 on >= 2 models | PIVOT |

---

## Bottom Line

**Do not write the paper yet.** The UAD result is real and novel, but it needs cross-model validation and the overall package needs methodological fixes. Invest 1-2 more iterations in refinement. If UAD generalizes, this becomes a solid workshop paper with potential for full conference expansion.
