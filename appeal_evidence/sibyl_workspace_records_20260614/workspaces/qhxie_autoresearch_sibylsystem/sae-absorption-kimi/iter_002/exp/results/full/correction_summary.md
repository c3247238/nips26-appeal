# Experiment Correction Summary

## Date: 2026-04-23

## Bug Fixed: Random SAE Randomization

### Problem
The Random SAE control produced **exactly identical** absorption scores to the Standard SAE across all 10 semantic hierarchies and all 10 non-hierarchy control pairs. This was statistically impossible and indicated a failure in the `make_random_sae()` function.

### Root Cause
The original `make_random_sae()` function used `copy.deepcopy(base_sae)` followed by reassigning `torch.nn.Parameter` objects:
```python
random_sae.W_enc = torch.nn.Parameter(new_enc)
```

While this approach works in isolated testing, the Parameter reassignment may not reliably update the module's internal state in all PyTorch versions or when the module has complex internal structures.

### Fix
Changed to **in-place mutation** with verification:
```python
random_sae.W_enc.data.copy_(new_enc)
random_sae.b_enc.data.copy_(new_b)
random_sae.W_dec.data.copy_(new_dec)
```

Added checksum verification to ensure weights actually changed:
```python
if abs(new_enc_sum - orig_enc_sum) < 1e-6:
    raise RuntimeError("Random SAE encoder weights did not change after randomization!")
```

### Verification
After the fix, the Random SAE produces different scores:
- **Semantic-hierarchy absorption**: 0.1750 (was 0.3517, same as Standard)
- **Non-hierarchy control absorption**: 0.2329 (was 0.4161, same as Standard)
- **First-letter absorption**: 0.0298 (unchanged, already correct)

The Random SAE is now correctly lower than Standard on both custom tasks.

## Updated Results

### H1: Construct Validity (First-Letter vs Semantic-Hierarchy)
- **Pearson r** (7 SAEs, excluding Random): 0.463
- **Bootstrap 95% CI**: [-0.389, 0.981]
- **Assessment**: INCONCLUSIVE (r < 0.6 threshold, CI spans zero)

### H2: Hierarchy Specificity (Semantic vs Non-Hierarchy Control)
- **Paired t-test**: t = -4.748, p = 0.0032
- **Mean semantic-hierarchy absorption**: 0.235
- **Mean non-hierarchy control absorption**: 0.331
- **Assessment**: REJECTED (semantic < control, opposite of prediction)

### H3: Robustness Across tau_fs
- tau_fs = 0.01: r = 0.468
- tau_fs = 0.03: r = 0.463
- tau_fs = 0.05: r = 0.471
- **Assessment**: INCONCLUSIVE (r < 0.5 threshold)

## Implications

The Random SAE bug was a **data-handling issue**, not a fundamental flaw in the experimental design. After fixing it:

1. The Random SAE correctly shows lower absorption than Standard, confirming the metric responds to learned structure.
2. H1 remains inconclusive - the correlation between first-letter and semantic-hierarchy absorption is moderate (r = 0.46) but below the pre-registered 0.6 threshold.
3. H2 remains rejected - semantic-hierarchy absorption is significantly **lower** than non-hierarchy control, suggesting the metric may not be hierarchy-specific in the way hypothesized.
4. The small sample size (n = 7 SAEs) limits statistical power for correlation analysis.

## Files Modified
- `iter_002/exp/scripts/semantic_hierarchy_pythia.py` - Fixed `make_random_sae()`
- `iter_002/exp/scripts/nonhierarchy_control_pythia.py` - Fixed `make_random_sae()`
- `iter_002/exp/results/full/semantic_hierarchy_pythia_results.json` - Updated with correct Random SAE scores
- `iter_002/exp/results/full/nonhierarchy_control_pythia_results.json` - Updated with correct Random SAE scores
- `iter_002/exp/results/full/statistical_analysis_summary.json` - Recomputed with updated results
- `iter_002/exp/results/full/statistical_analysis_summary.md` - Recomputed with updated results
- `iter_002/exp/results/full/fig*.png` - Regenerated figures
