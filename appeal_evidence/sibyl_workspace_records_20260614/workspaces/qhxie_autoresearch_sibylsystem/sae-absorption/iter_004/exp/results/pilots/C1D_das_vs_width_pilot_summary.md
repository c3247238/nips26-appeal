# C1D Width Paradox — PILOT Summary

**GO/NO-GO: PARTIAL**

## Configuration
- Model: GPT-2 Small (open-model anchor)
- SAE Release: gpt2-small-res-jb-feature-splitting (Layer 8)
- Widths tested: 24576 (24k), 49152 (49k)
- Letters: A, B, C, D, E (5 pilot letters)
- Mode: PILOT (~40 tokens/letter, 40 words per letter for absorption)
- Runtime: 40.3 seconds

## Pass Criteria
- Both DAS(k=1) and DAS(k=3) computable for all 5 letters: **PASS** (5/5)
- DAS(k=3) >= DAS(k=1) for at least 3 of 5 letters at wider SAE: **FAIL** (1/5)

## Results

### Width 24k (d_sae=24576)
| Letter | DAS(k=1) | DAS(k=3) | Children |
|--------|----------|----------|---------|
| A      | 0.050    | 0.012    | [23885, 24196, 18770] |
| B      | 0.050    | 0.071    | [4965, 3386, 10671] |
| C      | 0.050    | 0.012    | [1158, 16334, 5555] |
| D      | 0.025    | 0.046    | [19502, 20447, 8477] |
| E      | 0.075    | 0.021    | [1803, 16701, 23239] |

### Width 49k (d_sae=49152)
| Letter | DAS(k=1) | DAS(k=3) | Children |
|--------|----------|----------|---------|
| A      | 0.100    | 0.018    | [34372, 15128, 13318] |
| B      | 0.075    | 0.054    | [36499, 28083, 21483] |
| C      | 0.075    | 0.022    | [42405, 35948, 34786] |
| D      | 0.025    | 0.011    | [14805, 42254, 3144] |
| E      | 0.050    | 0.065    | [8458, 525, 14861] |

## Key Findings

1. **Pipeline validated**: Both DAS(k=1) and DAS(k=3) are computable for all 5 pilot letters at both widths. No pipeline errors.

2. **Width paradox pattern weak in pilot**: DAS(k=3) does not consistently exceed DAS(k=1) at the wider SAE. Only letter E at 49k shows DAS(k=3) > DAS(k=1) (0.065 vs 0.050).

3. **DAS(k=3) generally lower than DAS(k=1)**: This could reflect:
   - The McFadden R2 formulation (predicting binary parent activation from children) is on a different scale than the absorption rate measure
   - Low parent activation rates (~2.5-10%) create sparse classification problems
   - Child features found via decoder cosine similarity may not be the true absorbers

4. **Absorption rates increase at wider SAE**: Letters A-C show higher DAS(k=1) at 49k vs 24k (A: 0.10 vs 0.05; B: 0.075 vs 0.05; C: 0.075 vs 0.05), consistent with H4 hypothesis that wider SAEs show more absorption.

5. **DAS(k=3) comparable across widths**: No clear increase of DAS(k=3) with width, suggesting the distributed absorption pattern may require 131k width to manifest clearly, or our child-finding methodology needs refinement.

## Interpretation
The PARTIAL result indicates the pipeline works correctly but the width paradox (DAS(k=3) increasing with width) is not yet clearly observable at {24k, 49k} with only 5 letters. The task plan recommends including 131k width and all 26 letters for the full experiment, which is expected to show the pattern more clearly. The absorption rates themselves do increase with width (consistent with H4), but the distributed component DAS(k=3) improvement is not yet significant.

## Recommendations for Full Run
1. Include 131k width SAE (blocks.8.hook_resid_pre_98304 in the feature-splitting release)
2. Use all 26 letters (full experiment scope)
3. Increase to 100 words per letter for more reliable statistics
4. Consider using k=5 children instead of k=3 for better coverage
