# Idea Validation Decision

## Pilot Evidence Summary

### cand_sensitivity_floor (front-runner) - Layer 4 Pilot
| Metric | Predicted | Actual | Result |
|--------|-----------|--------|--------|
| Q2+Q4 fraction | < 10% | 100% (50/50) | **FAIL - FALSIFIED** |
| U-shape coefficient a | > 0 | 0.82 | PASS |

**Quadrant distribution (N=50, layer 4)**:
- Q1 (high abs + low sens): 0 (0%)
- Q2 (high abs + high sens): 48 (96%)
- Q3 (low abs + low sens): 0 (0%)
- Q4 (low abs + high sens): 2 (4%)

The Sensitivity Floor predicted Q2+Q4 < 10% based on layer 8 data (iter_008: Q2+Q4 = 0/43). Layer 4 shows the OPPOSITE pattern: 96% of features are Q2 (absorbed + high sensitivity). This is a definitive falsification of H-SF1 at layer 4.

**Contradictory findings across layers**:
- Layer 8 (iter_008): Q3 dominates at 65%, Q2+Q4 = 0/43
- Layer 4 (iter_011 pilot): Q2 dominates at 96%, Q2+Q4 = 50/50

### cand_sensitivity_first - Not Directly Tested in This Pilot
- Steering protocol returned all zeros (broken protocol)
- Layer 8 showed Q3 dominance (65%) consistent with sensitivity-first
- Layer 4 showed Q2 dominance (96%) -- inconsistent with pure sensitivity-first
- Needs pilot debugging before it can be evaluated

### cand_geometric_density - Not Tested
- Remains viable as a common-cause explanation for r=0.59
- No pilot data yet

### cand_layer_uas_saturation - Not Tested
- Layer-specificity is now the critical question given contradictory layer 4 vs layer 8 results
- Could resolve whether Sensitivity Floor is layer-specific or universally falsified

### cand_steering_diffusion - Not Tested (Protocol Broken)
- Steering protocol returned all zeros at beta in {5, 10, 20}
- Pilot H-STEER pass criterion (at least 1 non-zero effect) was not met

---

## Decision Matrix

| Criterion | Weight | cand_sensitivity_floor | cand_sensitivity_first | cand_geometric_density | cand_layer_uas_saturation | cand_steering_diffusion |
|-----------|--------|------------------------|------------------------|------------------------|---------------------------|------------------------|
| Pilot signal strength | 0.30 | 1 (H-SF1 falsified) | 3 (untested, Q3 at L8 consistent) | 2 (untested) | 2 (untested) | 1 (protocol broken) |
| Hypothesis survival | 0.25 | 1 (main H-SF1 false) | 3 (Q3 dominance at L8) | 3 (density mediation viable) | 3 (layer diff testable) | 2 (mechanism untested) |
| Path to full result | 0.20 | 2 (need recalibration) | 3 (steering validation) | 3 (direct measurement) | 4 (clear measurement path) | 2 (protocol must fix first) |
| Novelty | 0.15 | 5 (8/10, novel) | 4 (6/10) | 4 (7/10) | 3 (6/10) | 3 (6/10) |
| Resource efficiency | 0.10 | 3 (recalibration needed) | 4 (15 min pilot) | 3 (15 min pilot) | 4 (20 min pilot) | 2 (must fix protocol) |
| **Weighted Score** | | **2.0** | **3.35** | **2.85** | **3.05** | **1.95** |

---

## Decision Rationale

**PIVOT** is triggered for two reasons:
1. cand_sensitivity_floor's main hypothesis (H-SF1: Q2+Q4 < 10%) is definitively falsified at layer 4 with 100% Q2+Q4. The Sensitivity Floor predicted structural emptiness; it found structural abundance of Q2.
2. No candidate scores >= 3.5 (ADVANCE threshold). The highest is cand_sensitivity_first at 3.35, but its core hypothesis (Q3 = random baseline) has not been directly tested in this pilot.

**Key evidence triggering pivot**:
- H-SF1 falsified: Q2+Q4 = 100% at layer 4 vs predicted < 10%
- Contradictory layer patterns: Q2 dominates at layer 4 (96%), Q3 dominates at layer 8 (65%)
- The Sensitivity Floor cannot explain why layer 4 features are predominantly high-sensitivity (Q2)

**What the pivot should explore**:
1. **Layer-specificity as the dominant factor** -- layer 4 and layer 8 show fundamentally different absorption-sensitivity relationships. Any viable theory must explain this.
2. **Sensitivity-first remains viable** at layer 8 (Q3 = 65%), but layer 4 contradicts pure sensitivity-first. The discrepancy suggests absorption and sensitivity interact differently at different layers.
3. **Geometric density** could explain layer differences if density varies systematically across layers.
4. **Steering protocol is broken** -- must be debugged before cand_sensitivity_first or cand_steering_diffusion can be evaluated.

**Sanity checks**:
- [x] Compared ALL candidates, not just front-runner
- [x] Penalized cand_sensitivity_floor for failing its own falsification criterion (H-SF1)
- [x] Not swayed by sunk cost (iter_008 investment is irrelevant to this decision)
- [x] Not defaulting blindly -- PIVOT is triggered by definitive falsification + no candidate scoring >= 3.5

---

## Next Actions

1. **Recalibrate absorption metric**: The simplified proxy (max/mean ratio) may not match the Chanin first-letter protocol. Recalibration is needed before any candidate can be properly evaluated.
2. **Run layer-wise UAS mapping** (cand_layer_uas_saturation): The contradictory layer 4 vs layer 8 results make layer-specificity the most urgent measurement. 20 min on CPU.
3. **Debug steering protocol** (for cand_sensitivity_first): iter_010 found all steering effects = 0.0. This must be fixed before sensitivity-first can be tested.
4. **Consider geometric density as unifying explanation**: If dense regions vary by layer, this could explain why Q2 dominates at layer 4 but Q3 dominates at layer 8.

SELECTED_CANDIDATE: none
CONFIDENCE: 0.72
DECISION: PIVOT
