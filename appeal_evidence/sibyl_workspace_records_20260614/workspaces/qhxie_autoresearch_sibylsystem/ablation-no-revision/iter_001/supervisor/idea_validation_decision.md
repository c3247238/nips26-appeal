# Idea Validation Decision

## Pilot Evidence Summary

### cand_a (Activation Energy Theory - Routing Failure Analysis)

| Metric | Pilot Value | Threshold | Status |
|--------|-------------|-----------|--------|
| H1 (Arrhenius R²) | 0.936 (pilot n=30), 0.924 (full n=50) | >0.85 | **CONFIRMED** |
| H2 (Spearman Ea-difficulty) | 0.578, p=0.0008 | >0.4 | **CONFIRMED** |
| H3 (Single-pass accuracy, low-Ea) | 68.4% | >75% | **FALSIFIED** |
| Multi-sample (k=16) accuracy | 83.3% | - | Observed |
| Asymptotic ceiling (P_inf) | 81.8% | - | Estimated |

### Key Finding
The core application hypothesis H3 (Ea-based routing to predict single-pass threshold) was **falsified**: low-Ea problems achieved only 68.4% accuracy, below the 75% threshold. Ea captures difficulty (H2 confirmed) but NOT solveability.

### Novelty Analysis
- **Yang et al. 2025 (2508.16456)**: Exact collision on exponential saturation formula. H1-H2 are replications, not novel.
- **ACAR 2026 (2602.21231)**: Found "agreement-but-wrong" ceiling of 8pp, entropy showed weak correlation. **H5 at HIGH RISK**.
- **Li 2026 (2601.00828)**: Error Depth Hypothesis supports H4 (calculation errors 62% in low-depth failures). **H4 is promising but untested**.

### cand_b (Entropy-Based Routing)
- **Status**: Not piloted. ACAR 2026 preemptively found entropy routing ineffective.
- **Novelty**: Low — entropy as confidence measure is standard practice.

### cand_c (Error Depth Targeted Training - EDTT)
- **Status**: Not piloted. Theoretical support from Li 2026 (calculation errors 62% in low-depth failures).
- **Novelty**: High — first error-depth-targeted training combining GRPO with depth-aware rewards.
- **GPU Availability**: PyTorch 2.11.0 supports RTX PRO 6000 Blackwell (sm_120), enabling training.

### cand_d (H1-H2 Validation Paper)
- **Status**: Fallback only. Acknowledges Yang et al. collision explicitly.
- **Novelty**: Low — validation paper with explicit prior art acknowledgment.

---

## Decision Matrix

### cand_a

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | H1+H2 solid, H3 falsified |
| Hypothesis survival | 0.25 | 2 | H3 (main application) falsified; H5 HIGH RISK per ACAR |
| Path to full result | 0.20 | 2 | Novelty challenged by Yang et al.; ACAR preempts H5 |
| Novelty (from report) | 0.15 | 4 | H3 negative result + H4 diagnostic are novel; H1-H2 are not |
| Resource efficiency | 0.10 | 3 | GPU available, but continued Ea routing is diminishing returns |
| **Weighted Score** | | **2.70** | |

### cand_b

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | Not tested; ACAR found entropy ineffective |
| Hypothesis survival | 0.25 | 2 | Core idea preempted by ACAR |
| Path to full result | 0.20 | 2 | Well-known method, limited novelty |
| Novelty (from report) | 0.15 | 2 | Standard entropy routing |
| Resource efficiency | 0.10 | 3 | Inference only |
| **Weighted Score** | | **2.10** | |

### cand_c

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No pilot yet; theoretical support from Li 2026 |
| Hypothesis survival | 0.25 | 3 | New hypotheses, untested but theoretically grounded |
| Path to full result | 0.20 | 4 | Novel direction, GPU available, clear methodology |
| Novelty (from report) | 0.15 | 5 | First error-depth-targeted training |
| Resource efficiency | 0.10 | 3 | Training takes time but GPU is available |
| **Weighted Score** | | **3.20** | |

### cand_d

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | H1+H2 confirmed |
| Hypothesis survival | 0.25 | 4 | No falsification risk (validation paper) |
| Path to full result | 0.20 | 3 | Clear but limited impact |
| Novelty (from report) | 0.15 | 2 | Low — acknowledges Yang et al. collision |
| Resource efficiency | 0.10 | 4 | Low cost (analysis paper) |
| **Weighted Score** | | **3.20** | |

---

## Decision Rationale

**Why PIVOT from cand_a?**

1. **Main hypothesis H3 is falsified**: Ea-based routing does NOT predict single-pass solveability (68.4% < 75%). This is the core application hypothesis of cand_a.

2. **H5 is preempted by ACAR (2026)**: The novelty report explicitly flags that ACAR found entropy routing shows weak correlation with ground truth. The backup routing signal for cand_a has been challenged by concurrent work.

3. **Yang et al. collision undermines H1-H2 novelty**: The exponential saturation formula is already published. H1-H2 cannot be claimed as novel contributions.

4. **Sunk cost is irrelevant**: Prior effort on Ea-based routing does not justify continued investment in a falsified direction.

**Why select cand_c (EDTT) as next direction?**

1. **Genuinely novel**: First error-depth-targeted training combining GRPO with depth-aware rewards.

2. **Theoretical support**: Li (2026) Error Depth Hypothesis supports the core mechanism (calculation errors 62% in low-depth failures).

3. **GPU is now available**: PyTorch 2.11.0 supports RTX PRO 6000 Blackwell (sm_120). Training experiments that failed in Round 2 can now proceed.

4. **Addresses root cause**: Instead of diagnosing a failed routing method, EDTT targets the underlying error patterns that cause routing failure.

**Why not cand_b?**
- Entropy-based routing is standard practice, not novel.
- ACAR explicitly found entropy ineffective.
- Lower novelty and lower upside than cand_c.

**Why not cand_d (fallback)?**
- cand_d is a safe fallback but has limited impact.
- cand_c has higher upside if the pilot succeeds.
- cand_d remains available if cand_c fails.

---

## Next Actions

### Immediate (This Iteration)
1. Abandon Ea-based routing analysis (cand_a)
2. Design EDTT pilot experiment: classify errors by depth (L1/L2/L3 per Li 2026 taxonomy)
3. Implement depth-aware GRPO rewards targeting L3 (hardest-to-correct) errors
4. Run pilot: EDTT vs standard GRPO on Qwen2.5-Math-7B

### Pilot Criteria for EDTT
- H_C1: EDTT reduces L3 error rate more than standard GRPO
- H_C2: Error depth targeting improves over generic training by >5%

### If EDTT Pilot Succeeds
- Full training run on n=200 MATH problems
- Compare with RASC and SwS baselines
- Write paper as first error-depth-targeted training for LLM reasoning

### If EDTT Pilot Fails
- Fallback: cand_d (H1-H2 validation paper with H3 negative result acknowledgment)
- Acknowledge Yang et al. collision explicitly
- Focus on empirical validation + per-level accuracy analysis on Qwen2.5-Math-7B

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.72
DECISION: PIVOT
