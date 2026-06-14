# Critique: Experiments

## Summary Assessment
The Experiments section is well-structured with clear hypothesis-driven organization and follows the evidence contract by reporting both confirmations and failures with specific numbers. The section correctly identifies H_Mech as "conditionally confirmed" rather than fully confirmed, honestly acknowledging that the decoder contribution is configuration-dependent. However, there are significant issues with table numbering ambiguity, missing standard deviations for GPT-2 Small safety results, and rounding in H_Comp that masks real variance differences.

## Score: 6/10
**Justification**: The section demonstrates strong experimental rigor and appropriate handling of negative/null results. However, critical table numbering ambiguity (experiments.md Table 1 vs outline Table 1), missing standard deviations for GPT-2 Small safety results, and the rounding of H_Comp std values to uniform 0.08 (when actual values range from 0.131 to 0.397) significantly weaken the section's credibility. These are not cosmetic issues -- missing std for safety results affects the reader's ability to assess the null result's reliability.

## Critical Issues

### Issue 1: Table Numbering Ambiguity
- **Location**: Line 21, "Table 1 reports absorption rates by condition"
- **Quote**: "Table 1 reports absorption rates by condition. Condition D (trained/trained) serves as the baseline from full joint training."
- **Problem**: The outline defines "Table 1" as the Hypothesis Summary table (Section 4, line 311-324), but in experiments.md the first table is the H_Mech condition results. The reader cannot determine which table is being referenced when the text says "Table 1" in the Summary section (line 175). This creates confusion in cross-referencing.
- **Fix**: Use descriptive labels instead of numbers when referencing tables in the body text. For example: "The H_Mech condition results (Table A) report absorption rates..." or "The hypothesis summary (Table B) summarizes..."

### Issue 2: Missing Standard Deviations for GPT-2 Small Safety Results
- **Location**: Lines 158-161, H_Safe results table
- **Quote**: "| Safety | 233.13 | — |"
- **Problem**: The GPT-2 Small H_Safe results lack standard deviations. According to held_out_validation.json, safety has std=47.302 and non-safety has std=46.5. Presenting means without std misrepresents the data quality and makes it impossible for readers to assess the null result's reliability.
- **Fix**: Add standard deviation columns to the GPT-2 Small table:
  | Safety | 233.13 | 47.3 |
  | Non-Safety | 221.70 | 46.5 |

### Issue 3: Rounding H_Comp Standard Deviations to Mask Variance Differences
- **Location**: Lines 67-74, H_Comp results table
- **Quote**: All rows show std=0.131 or similar rounded values
- **Problem**: The actual data from h_comp_6levels_3seeds.json shows std values ranging from 0.131 to 0.397 across cosine levels:
  - cos_0.5: std=0.131
  - cos_0.6: std=0.163
  - cos_0.7: std=0.262
  - cos_0.8: std=0.341
  - cos_0.9: std=0.397
  - cos_0.95: std=0.242
  
  Rounding all to 0.08 in the presentation is factually incorrect and masks the important finding that variance increases at middle cosine levels. This undermines the section's own conclusion that "different seeds produce different rank orderings."
- **Fix**: Use actual std values from the JSON data. The variance pattern (lowest at cos=0.5, highest at cos=0.9) is itself an interesting finding that supports the non-monotonic interpretation.

## Major Issues

### Issue 4: H_Pareto Table Contains Placeholder Std Values
- **Location**: Lines 111-116, H_Pareto results table
- **Quote**: "| 16 | **0.0** | 0.08 | 0.1054 | 0.0008 |"
- **Problem**: The std=0.08 for absorption appears to be a placeholder. The h_pareto_4l0_3seeds.json shows the actual std varies by seed and level, and the reported value doesn't match the source data structure. More importantly, the std=0.0008 for sensitivity looks like a typographical error -- it implies extraordinary precision that doesn't match the measurement methodology.
- **Fix**: Verify std values against h_pareto_4l0_3seeds.json and h_pareto_full.json. If 0.0008 is correct, note that this indicates sensitivity was nearly identical across seeds/levels, which is itself a notable finding.

### Issue 5: Cross-Model Validation Lacks Detail
- **Location**: Lines 47-49, "Cross-Model Validation"
- **Quote**: "We validated on GPT-2 Small SAE using held-out data splits. The encoder-driven pattern replicated with delta B-D = 0.0 and delta C-A = 0.0, confirming generalization to real pretrained SAEs."
- **Problem**: The held_out_validation.json shows that GPT-2 Small has condition_A mean=34.39 and condition_B mean=218.52 -- these are vastly different from the synthetic MLP results (A=0.184, B=0.055). The "encoder-driven pattern replicated" claim oversimplifies. The deltas being zero means B=D and C=A in the GPT-2 Small, but the actual absorption magnitudes are 100x higher. This needs explanation -- the pattern (encoder sufficiency) replicates but the absolute magnitudes do not.
- **Fix**: Clarify that the encoder-driven mechanism (B≈D) replicates, but note the magnitude difference: "The encoder-driven pattern (B≈D) replicated, though absolute absorption magnitudes in GPT-2 Small (~200) far exceed those in synthetic SAEs (~0.1), suggesting the mechanism but not the scale is preserved."

### Issue 6: H_Pareto Section Unclear on Why L0 Variation Produced No Signal
- **Location**: Lines 124-129, H_Pareto interpretation
- **Quote**: "Possible explanations: 1. Multi-child proportional ablation with k = 5 saturates at zero absorption for this hierarchy depth (3-level)"
- **Problem**: The explanation attributes saturation to hierarchy depth but provides no evidence. If k=5 causes saturation, why did H_Comp (which also uses k=5) produce non-zero results? The reader cannot evaluate this hypothesis.
- **Fix**: Either remove speculation without evidence, or cross-reference H_Comp's successful non-zero results to show the saturation hypothesis is contradicted by other experiments.

## Minor Issues

- **Line 5**: "contradicting the pilot result" -- clarify that the pilot was seed=42 only and the full experiment shows configuration-dependent effects. The current phrasing could imply the full experiment refutes the pilot entirely.
- **Figure references**: Figures 2-5 are referenced but no indication of whether they are generated/included. The FIGURES block at the end documents generation plans but doesn't indicate current status.
- **Table 1 vs Hypothesis Summary**: In the Hypothesis Summary section (lines 175-185), the "Table 1" reference is ambiguous. The outline defines Table 1 as the Hypothesis Summary, but experiments.md uses Table 1 for the H_Mech conditions. Clarify: "Our central results table (Table X in the outline, reproduced here as Table 1)..."

## Visual Element Assessment

- [ ] Figures/tables match outline plan (PARTIAL -- table numbering is ambiguous)
- [ ] All visuals referenced before appearance (YES -- Figure 2, 3, 4, 5 all referenced before they appear in the section)
- [ ] Captions are self-explanatory (YES -- each table has clear labels)
- [ ] No text-heavy sections that need visual support (YES -- tables break up text appropriately)
- [ ] H_Comp results would benefit from visualizing the variance pattern (cos 0.5 std=0.131 vs cos 0.9 std=0.397)

## What Works Well

1. **Appropriate handling of null/negative results** (lines 165-169): The H_Safe interpretation correctly frames the null result as positive for safety analysis, avoiding the trap of presenting it as a failure. The phrasing "valid negative result with positive implications" is exemplary.

2. **Transparent reporting of failed hypotheses** (lines 81-82): "Monotonic fit: FAILED" with specific R²=0.04 is exactly what the evidence contract requires. No smoothing or hedge language.

3. **Cross-seed robustness documentation** (lines 191-201): The multi-seed table showing all |B-D| values below threshold (0.009 to 0.089) is strong evidence for the encoder sufficiency finding. Showing seed-level variation builds credibility.

4. **Clear separation of pilot vs full experiment** (line 45): "The stochastic hierarchy exposes decoder contributions that deterministic hierarchies in pilot experiments masked" explicitly connects the pilot's C≈A≈0.0 result to the full experiment's C≈A failure, helping readers understand why the narrative evolved.

5. **Correct use of "conditionally confirmed"** (line 85): The experiments section correctly avoids claiming full confirmation for H_Mech when only B≈D holds. This precision maintains scientific credibility.