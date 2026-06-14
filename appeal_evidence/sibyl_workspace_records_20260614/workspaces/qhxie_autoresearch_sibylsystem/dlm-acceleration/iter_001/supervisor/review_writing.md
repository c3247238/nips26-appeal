# ComposeAccel — Supervisor Review

**Score: 6.0 / 10** | **Verdict: CONTINUE**

**Dimension Scores:**
- Novelty: 7/10
- Soundness: 5/10
- Experiments: 6/10
- Reproducibility: 5/10

---

## Executive Summary

ComposeAccel presents the first systematic composability study of training-free MDM acceleration methods. The core finding — that M1+CD-SSD achieves super-multiplicative synergy (Ortho=1.385) while all other tested pairs destructively interfere — is real and reproducible from raw JSON artifacts. The paper has materially improved since the prior review iteration: the fabricated Wilcoxon p-value has been removed, the tau=0.0 paradox has been honestly addressed (CD-SSD(tau=0.0) ≈ naive T=16), the QAS inconsistency has been fixed (standard formula now used: QAS=3.40×0.703=2.39 for CD-SSD), and CD-SSD is correctly repositioned as concurrent with SSD and SSMD rather than claiming to be "first."

The CHR_refine=0.940 measurement is correctly reported and verified in the raw data (igsd_p2_tau09_td16_s123.json: avg_kv_hit_rate_refine=0.9403; igsd_p2_tau09_td16_s456.json: same). The synergy mechanism is coherent and causally supported.

However, this review identifies a NEW critical error: **the paper repeatedly claims alpha=0.52 (52% tokens accepted at tau=0.9, T_draft=16) but the actual measured avg_accept_rate in the full experiment data is 0.881 (88.1% on GSM8K)**. This figure appears in 7+ locations including the abstract-level contribution bullet, Section 2.2, Section 3.4, Section 4.2, and Section 6.1. The synergy mechanism still holds qualitatively, but the quantitative description is wrong by 36 percentage points. Three prior major issues also persist.

---

## Research Contribution Assessment

### 1. Novelty and Significance (Score: 7/10)

**What is novel:** The composability framework for MDM inference-time methods is the first of its kind. No prior work has measured pairwise orthogonality of MDM acceleration methods across families (KV-caching, speculative decoding, AR guidance). The binary composability discovery (one synergistic pair, all others interference) is a structural finding that did not exist in the literature. The failure-mode atlas with four characterized patterns and proactive detection signals has no prior analog in the DLM acceleration literature.

**What weakens novelty:** CD-SSD is approximate and achieves lower standalone quality than the concurrent lossless SSD (63.7% GSM8K AccRet vs SSD's lossless 2.11-3.46x). The paper correctly frames CD-SSD as a composability vehicle rather than a standalone contribution. The Ortho metric is a natural product-rule extension; the paper should not claim it "did not exist before this work."

**Assessment:** The composability framework and binary landscape discovery are significant contributions at the 7/10 level. The deployment guidance (failure mode atlas, task-dependent recipes) adds applied value.

### 2. Technical Soundness (Score: 5/10)

**Critical unsoundness:** The alpha=0.52 claim is factually wrong based on raw experimental data. The full experiment script (igsd_p2_tau09_td16_s123.py) computes accept_rate = n_accept / n_total where n_total = all generation positions. At T_draft=16, 88% of positions have confidence >= 0.9 — not 52%. The mechanistic argument in Section 4.2 that "at alpha=0.52, over half the position-step pairs incur zero KV recomputation" understates the actual frozen-token fraction by 36 points.

**What is sound:** The CHR_refine=0.940 measurement is correctly reported (verified against igsd_p2_tau09_td16_s123.json). The Ortho=1.385 synergy finding is real. The tau=0.0 paradox analysis is honest. The M2 NO_GO verdict is supported by data.

**What is post-hoc:** The trajectory-preserving vs trajectory-modifying classification (Section 4.1) explains interference patterns from their outcome rather than predicting them a priori. No per-step entropy measurements comparing M3-on vs M3-off validate the trajectory-modification claim.

### 3. Experimental Rigor (Score: 6/10)

**Cross-validation of paper claims against raw data:**

| Claim | Raw Data | Status |
|-------|----------|--------|
| M1 eta=2.0: combined speedup=1.38x | m1_pareto_full.json: 1.380 | MATCH |
| M1 eta=2.0: GSM8K AccRet=0.550 | Raw: acc_retention_mean=0.5496 | NEAR MATCH |
| CD-SSD tau=0.9 T16: combined speedup=3.40x | igsd_pareto_full.json: 3.399 | MATCH |
| CD-SSD GSM8K AccRet=0.637 | Raw: 0.6366 | MATCH |
| CD-SSD QAS=2.39 | 3.40 × 0.703 = 2.39 | MATCH |
| M1+CD-SSD Ortho=1.385 | full_pairwise_ortho.json: 1.3852 | MATCH |
| CHR_refine=0.940 (seeds 123, 456) | igsd_p2_tau09_td16_s123.json: 0.9403; s456: 0.9403 | MATCH |
| **alpha=0.52 at tau=0.9, T_draft=16** | **igsd_pareto_full.json: avg_accept_rate=0.881** | **MISMATCH** |
| tau=0.0 == naive T=16 accuracy | full_tau0_comparison.json: AccRet=0.5897 both | MATCH |
| M2 J=2: combined speedup=3.10x | m2_pareto_full.json: 3.095 | MATCH |
| M2 J=4: GSM8K AccRet=0.130 | Raw: 0.130 | MATCH |

**Scope limitations:** Pairwise Ortho is 2-seed, 15% benchmark scale. Single-model evaluation only.

### 4. Reproducibility (Score: 5/10)

Most key numbers trace to specific raw experiment files. However: alpha=0.52 cannot be traced to any raw data file; Appendices A, C, D are placeholders; Dream-7B-Instruct unavailable; SSD+M1 composability untested.

---

## Issues

### Critical

**C1: alpha=0.52 factual error** — Paper states 52% tokens accepted at tau=0.9, T_draft=16. Actual measured avg_accept_rate=0.881 (GSM8K), 0.830 (combined). Appears in 7+ locations including the central mechanistic claim in Section 4.2. The synergy mechanism is qualitatively correct but the quantitative description is wrong. The CHR_refine=0.940 is consistent with alpha=0.88 (not 0.52) — if 88% of tokens are frozen from the start of refine, the near-100% hit rate at end of refine makes the 0.940 average fully coherent.

### Major

**M1: Pairwise Ortho at 2-seed, 15% scale** — Per-seed Ortho [1.292, 1.478] with lower bound barely clearing 1.0. Binary composability claim needs hedging.

**M2: M1 implementation gap** — 1.38x vs published 15-26x. Absolute numbers in Table 3 uninterpretable alongside published baselines.

**M3: Single-model evaluation** — Binary composability claimed as structural MDM property, validated on one model only.

**M4: M2 NO_GO without backtracking** — Simplified Saber; verdict may not transfer to actual Saber. Needs more visible qualification.

**M5: Trajectory-modification claim for M3 lacks measurement evidence** — Post-hoc rationalization without per-step entropy distributions.

### Minor

**m1: Degenerate coding benchmarks** included in combined QAS — add reasoning-only QAS column.

**m2: Composability metric novelty overstated** — "did not exist before this work" should be "first application to MDM inference-time acceleration."

**m3: Appendices A, C, D are placeholders** — per-seed variance data and qualitative examples missing.

---

## Score Path

- **Current: 6.0** — Core finding real, multiple prior issues fixed, but new critical alpha error and persistent major issues
- **After fixing C1 + M4 + tone-down M1 binary language: ~7.0** — Internally consistent, honest paper with real finding
- **After 3-seed pairwise + M3 entropy measurement + Appendix A: ~7.5** — Defensible primary claim with appropriate statistical backing
- **After SSD+M1 composability + second MDM + batch-size experiment: ~8.0+** — Paper with strong generalizable claims
