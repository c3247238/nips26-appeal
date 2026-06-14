# Critique: Method

## Summary Assessment
The Method section is well-structured and provides thorough documentation of the experimental design across four hypotheses (H_Mech, H_Comp, H_Pareto, H_Safe). The factorial design is clearly explained, hypotheses are properly stated with pass criteria, and results are reported with appropriate statistical detail. However, there are critical consistency issues between the Method section description and the actual experimental configuration described elsewhere, and some visual elements referenced in the text are missing or incorrectly specified.

## Score: 6/10
**Justification**: The section has strong technical content but loses points on cross-section consistency (the Training Protocol table specifications don't match the actual experiments run), missing visual elements (Figure 1 is referenced but only conceptually described, not actually generated), and terminology drift from the glossary (using "SAE expansion factor" instead of proper feature hierarchy terminology). The section would benefit from tighter integration with the experimental results in the Experiments section.

## Critical Issues

### Issue 1: H_Pareto Results Contradict Section Description
- **Location**: H_Pareto Results subsection (lines 141-151)
- **Quote**: "Absorption collapses to zero across all L0 levels while sensitivity remains stable (0.1054 ± 0.0008)"
- **Problem**: The Method section reports these results but provides no explanation for WHY absorption collapses to zero. The H_Comp section (line 118) states the phase-transition framing is "not supported" but the H_Pareto subsection doesn't even attempt to explain the degenerate result.
- **Fix**: Add a "Potential explanations" paragraph to H_Pareto Results discussing: (1) k=5 multi-child ablation may saturate at zero for 3-level hierarchies, (2) L0 variation may not produce absorption signal in this setup, (3) sensitivity metric (steering coefficient variance) may not be appropriate for this hierarchy depth. This matches what appears in the outline.md (Section 3.3) and discussion.md (Section 5.3).

### Issue 2: Missing Figure 1 in Method Section
- **Location**: Overview subsection, line 5
- **Quote**: References "synthetic hierarchy generation, the 2x2 factorial design for isolating absorption mechanisms, hierarchy strength dependence experiments, sensitivity-absorption frontier measurements, and safety-critical feature analysis" but no figure reference
- **Problem**: The outline specifies "Figure 1: conceptual_diagram — conceptual illustration of absorption phenomenon" should appear in the Background section, but the Method section overview mentions none of the visual elements. The FIGURES block at the end only references Figures 2-4 for H_Mech/H_Comp/H_Pareto but not a pipeline diagram for the method itself.
- **Fix**: Either (1) add a reference to a conceptual architecture diagram in the Method overview, or (2) clarify that Method figures (architecture/pipeline) are deferred to a combined figures section.

## Major Issues

### Issue 3: Training Protocol Table Inconsistency
- **Location**: Training Protocol subsection (lines 54-60)
- **Quote**: "**Training Protocol**: Model: 2-layer MLP, d_model = 128, d_hidden = 512; SAE: d_sae = 512, TopK activation (k=32), L1 coefficient λ = 10^−4; Training: 50,000 samples, batch size 256; Seeds: 42, 123, 456, 789, 1024"
- **Problem**: The Experimental Configuration table (lines 197-203) shows different seeds for different experiments:
  - H_Mech: 5 seeds (matches Training Protocol)
  - H_Comp: 3 seeds (not 5)
  - H_Pareto: 3 seeds (not 5)
  - H_Safe (Gemma): 1 seed
  - H_Safe (GPT-2): 3 seeds
  The Training Protocol only describes H_Mech's configuration. Other experiments have different seed counts.
- **Fix**: Update the Training Protocol to clarify it specifically describes H_Mech's configuration, or restructure as an "H_Mech Training Protocol" subsection with separate subsections for other experiments.

### Issue 4: Notation Inconsistency with notation.md
- **Location**: Throughout the section
- **Quote**: "d_sae = 512 (SAE expansion factor 4x from d_model = 128)"
- **Problem**: notation.md defines d_sae as feature dimensionality (n), not "expansion factor." The notation "SAE expansion factor 4x" is non-standard and conflicts with notation.md's definition of d_sae as the latent dimension.
- **Fix**: Use notation.md-compliant notation: "d_model = 128 input dimension, d_sae = 512 SAE latent dimension (4x expansion)" — or simply report the dimensionalities without the ambiguous "expansion factor" phrasing.

### Issue 5: H_Safe Dataset Description Missing Detail
- **Location**: Safety-Critical Feature Analysis subsection (lines 159-163)
- **Quote**: "Dataset: Gemma Scope, gemma-2b residual SAE, layer 12, d=16384; Safety features: 5 (pilot) / 20 (validation) features annotated on Neuronpedia (deception, jailbreak, harm); Control features: Matched non-safety features by activation frequency"
- **Problem**: The section doesn't specify which Gemma Scope SAE variant is used beyond "gemma-2b residual SAE, layer 12." The notation.md specifies "Gemma Scope residual SAE: gemma-2b, layer 12, d=16384" but doesn't specify the exact SAE variant name (e.g., "gemma-2b-resid-pre-layer-12-scoped"). Also missing: how many GPT-2 Small features were used in validation (implied 20 per group from outline, not stated in method).
- **Fix**: Add explicit mention: "Gemma Scope variant: gemma-2b-resid-pre-layer-12-scoped" and clarify "20 safety + 20 non-safety features for GPT-2 Small validation."

### Issue 6: Missing Model Architecture Details for Real SAEs
- **Location**: H_Safe subsection
- **Problem**: For H_Safe, the method uses pretrained Gemma Scope and GPT-2 Small SAEs but provides no details on: (1) how multi-child proportional absorption was measured in real SAEs (the k=5 ablation may not apply to pretrained SAEs with different hierarchy structures), (2) what the "parent" features are for real SAEs (real SAEs don't have explicit parent-child hierarchies).
- **Fix**: Add a paragraph explaining: (1) For Gemma Scope/GPT-2 Small, absorption is measured by ablating features that co-activate with the target feature (identifying them via correlation analysis), (2) The k=5 was adapted based on co-activation threshold, not fixed hierarchy depth.

## Minor Issues

- **Line 58**: "L1 coefficient λ = 10^−4" — should use proper LaTeX: $\lambda = 10^{-4}$
- **Line 59**: "50,000 samples" — use "50,000" with comma consistent with other large numbers, or "5e4" for scientific notation consistency
- **Line 62**: "100 hierarchy samples per condition" — "hierarchy samples" is ambiguous; "synthetic hierarchy instances" or "hierarchy instantiations" would be clearer
- **Line 64**: "For parent feature p and k=5 child features" — k=5 should be formatted as $k=5$ for consistency
- **Line 80**: Table 1 shows "N" column but values are all 5; clarify this is number of seeds, not sample count
- **Line 84**: "Condition C exhibits extreme variance (std = 17.13, range 0-43.84)" — range should be formatted as [0, 43.84] or "0.0–43.84" for consistency
- **Line 196-203**: The Experimental Configuration table should include a "Hypothesis" column for clarity, matching the outline's structure

## Visual Element Assessment

- [ ] Figures/tables match outline plan — **PARTIAL**: Figure 1 (conceptual diagram) is mentioned in outline but not present in Method section
- [ ] All visuals referenced before appearance — **NO**: Figure 2 (H_Mech bar chart) is referenced at line 72 before appearing; the FIGURES block at the end only lists generation scripts, not actual figure placement in text
- [x] Captions are self-explanatory — N/A (no inline figures in Method)
- [ ] No text-heavy sections that need visual support — **FAIL**: The H_Mech subsection would benefit from a small flowchart showing the 2x2 factorial design inline, not just in the table
- [x] Architecture/pipeline diagram — **MISSING**: No architecture diagram for the synthetic hierarchy generation process or the overall experimental pipeline
- [x] Results presented with both tables AND charts — **PARTIAL**: Only inline tables; no actual generated figures in this section

**Specific issues**:
- FIGURES block (lines 207-216) lists generation scripts but NOT where figures should appear in the text
- No reference to where Table 1 is introduced (should be before line 72)
- Figure 1 (conceptual_diagram) from outline is missing entirely from Method

## What Works Well

1. **Clear hypothesis structure** (lines 36-52): The 2x2 factorial design is clearly explained with interpretation table and explicit pass criteria. The H_Mech hypothesis framing is exemplary.

2. **Multi-child proportional absorption formula** (lines 64-68): The measurement method is precisely defined with clear notation, and the note about values > 1.0 indicating suppression is valuable.

3. **Comprehensive results reporting** (lines 70-86): H_Mech results include full statistics (mean, std, min, max) for all 4 conditions with explicit pass/fail assessment for both checks. This is exactly what the evidence contract requires.

4. **Honest negative result reporting** (lines 115-118): The H_Comp regression statistics (R² = 0.04, p = 0.703) are reported with appropriate context about what the failed pass criterion means.

5. **Statistical test selection rationale** (lines 171-172): Mann-Whitney U test is properly justified for non-parametric distribution comparison with safety vs non-safety groups.