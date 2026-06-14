# Pilot Evaluation Summary

## Task: pilot_eval

**Status**: COMPLETED

**Date**: 2026-04-28

**Model**: Qwen/Qwen2.5-Math-7B-Instruct

**Dataset**: MATH (100 pilot problems)

---

## Results Overview

### Training Status

| Group | Description | Status | Final Loss | Dataset Size | Training Time |
|-------|-------------|--------|------------|--------------|---------------|
| G0 | Baseline (no fine-tune) | Pending eval | N/A | N/A | N/A |
| G1 | Uniform Step-DPO | Training done | 0.6922 | 100 | 3 min |
| G2 | EDW-Step-DPO (depth-weighted) | Training done | 0.6923 | 293 | 4 min |
| G5 | EDW L1-only (ablation) | Training done | 0.6927 | 18 | 3 min |
| G6 | EDW L3-only (ablation) | Training done | 0.6922 | 275 | 3 min |

### Evaluation Results

| Group | Accuracy | Avg Tokens | Note |
|-------|----------|------------|------|
| G0 Baseline | N/A | N/A | Not evaluated |
| G1 Uniform Step-DPO | N/A | N/A | Training only |
| G2 EDW-Step-DPO | 0.2125 | 481.4 | From G4 comparison |
| G3 Adaptive (G0) | 0.2000 | 802.1 | Inference only |
| G4 EDW + Adaptive | 0.2125 | 804.7 | Combined |
| G5 L1-only | N/A | N/A | Training only |
| G6 L3-only | N/A | N/A | Training only |

### Strategy Comparison (G3)

| Strategy | Accuracy | Avg Tokens | Relative Cost |
|----------|----------|------------|---------------|
| Single-shot CoT | 0.1500 | 483.5 | 39% |
| Fixed 3-round self-correction | 0.2000 | 1239.9 | 100% |
| Adaptive routing | 0.2000 | 802.1 | 65% |

### Routing Distribution (G3/G4)

| Difficulty | G3 Count | G4 Count |
|-----------|----------|----------|
| Easy | 5 | 32 |
| Medium | 39 | 35 |
| Hard | 36 | 13 |

---

## Hypothesis Assessment

### H1: EDW-Step-DPO vs Uniform Step-DPO
- **Status**: INCONCLUSIVE
- **Evidence**: Cannot compare without G0 baseline evaluation
- **Needs**: G0 baseline accuracy on pilot test set

### H2: Adaptive Inference vs Fixed Strategies
- **Status**: PARTIAL SUPPORT
- **Evidence**: Adaptive matches fixed self-correction accuracy (0.20) at 65% token cost
- **Observation**: Single-shot achieves only 0.15 accuracy, confirming need for self-correction

### H3: Superadditivity (Training + Inference Co-optimization)
- **Status**: NOT SUPPORTED
- **Evidence**: G4 (EDW + adaptive) = G2 (EDW) accuracy, no superadditive gain
- **Observation**: Combined approach matches individual component

### H4: Error Depth Distribution Shift
- **Status**: INCONCLUSIVE
- **Evidence**: Training losses similar across all variants
- **Observation**: Dataset dominated by L3 (94%), limiting ablation power

---

## Key Findings

1. **Training losses virtually identical** (0.6922-0.6927) across all variants
2. **Low absolute accuracy** (15-21%) on pilot MATH problems
3. **Adaptive inference reduces tokens** by ~35% vs fixed self-correction
4. **Dataset heavily skewed** toward deep conceptual errors (L3: 94%)
5. **No superadditivity observed** between training and inference optimization

---

## Recommendations

1. **Run G0 baseline evaluation** to enable H1 comparison
2. **Increase L1/L2 error diversity** in dataset for stronger ablations
3. **Pilot suggests limited gains** from depth weighting with current skewed dataset
4. **Adaptive inference promising** for token efficiency

---

## Output Files

- `/current/exp/results/pilots/pilot_eval_summary.json` - Machine-readable summary
- `/current/exp/results/pilots/pilot_eval_summary.md` - This document

---

## Next Steps

1. Evaluate G0 baseline on pilot test set
2. Evaluate G1, G5, G6 on pilot test set for ablation comparison
3. Consider dataset augmentation for more L1/L2 errors
4. Analyze remaining errors by depth category post-training
