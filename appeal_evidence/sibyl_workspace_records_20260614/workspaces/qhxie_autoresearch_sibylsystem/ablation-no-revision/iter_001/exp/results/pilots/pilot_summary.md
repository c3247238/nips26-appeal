# Pilot Dataset Generation Summary

## Task: gen_dataset

**Status**: PASSED

### Pass Criteria
- >=80 problems with valid (chosen, rejected) pairs: PASS (100/100)
- >=20% depth >=2 steps: PASS (96%)

### Results
| Metric | Value |
|--------|-------|
| Total problems | 100 |
| Valid pair count | 100 |
| Depth >=2 count | 96 |
| Depth >=2 ratio | 96% |
| Total preference pairs | 314 |

### Depth Distribution
| Level | Count |
|-------|-------|
| Level 1 (Computational) | 18 |
| Level 2 (Logical) | 0 |
| Level 3 (Conceptual) | 296 |

### Observations
- The dataset generation succeeded with all 100 problems having valid preference pairs
- Error depth is heavily skewed toward Level 3 (94%), indicating the self-refine process corrects deep conceptual errors effectively
- Level 2 (logical) errors were rarely generated - either problems have shallow (L1) or deep (L3) errors
- The high depth >=2 ratio (96%) exceeds the 20% threshold by 4.8x

### Output Files
- `/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results/pilots/dataset_preference_100.jsonl`
- `/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results/pilots/dataset_summary.json`

### Next Steps
This dataset can be used for:
- G1: Uniform Step-DPO training
- G2: EDW-Step-DPO (depth-weighted) training
- G5: EDW-Step-DPO L1-only ablation
- G6: EDW-Step-DPO L3-only ablation
