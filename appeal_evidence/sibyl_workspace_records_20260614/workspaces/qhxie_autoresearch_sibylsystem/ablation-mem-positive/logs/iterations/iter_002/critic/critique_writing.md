# Writing Critique

## Overall Assessment

The paper demonstrates strong evidence-first writing with specific numbers leading most claims. The honest reporting of H2/H3/H6 failures and reframing as exploratory findings is exemplary—it builds reviewer trust. The structure is clear and the related work is comprehensive. However, several writing issues remain: HARKing language in H2 framing, 'training-invariant' overclaim, and chi_ratio below threshold not prominently acknowledged. The full_cross_layer_critical results are missing entirely.

**Score: 6/10** (would be 7-8 with critical fixes)

---

## Critical Issues

### 1. H2 Framing Still Contains HARKing (Unresolved Despite Prior Feedback)

**Location**: Section 4.2, "H2 Analysis" paragraph

The text says: "layer 8 achieves rho = 0.705 > 0.6, satisfying the threshold." This frames a post-hoc observation as meeting a pre-registered threshold. The original H2 threshold (rho > 0.6) applied to the mean across layers, not a single cherry-picked layer. With 3 layers tested, the probability of one reaching p < 0.05 by chance is approximately 14.3%.

This was flagged in prior critique but has not been fixed. A reviewer will immediately flag this as HARKing.

**Fix**: Replace with: "Layer 8 shows rho = 0.705, but with only 3 layers tested, the probability of one reaching p < 0.05 by chance is approximately 14.3%. This observation requires validation on additional layers before it can be treated as a confirmed pattern."

### 2. "Training-Invariant" Overclaim Persists in Subtle Form

**Location**: Section 6.1, Conclusion paragraph 4

The text states: "decoder normalization emerges from training dynamics rather than architectural design alone." This causal claim is unsupported—no training dynamics were analyzed. All SAEs are pretrained. "Emerges from" is causal language that implies a mechanism that was never studied.

**Fix**: Replace "emerges from training dynamics" with "is consistent with training dynamics contributing to normalization, though architectural effects cannot be ruled out with only two SAE families."

### 3. Chi_ratio Below Threshold Not Prominently Acknowledged

**Location**: Abstract and throughout

The Abstract opens with "quasi-critical threshold behavior" which is appropriate, but the body of the paper uses "phase transition" language that implies sharpness. The chi_ratio=1.88 is below the "sharp transition" threshold of 3.0. A physics-savvy reviewer will notice this and question whether "phase transition" is the right framing.

A reader scanning the Abstract would not realize that chi_ratio < 3.0 means the transition is gradual, not sharp.

**Fix**: In the Abstract, add "(chi_ratio = 1.88 < 3.0, indicating gradual transition)" after the susceptibility peak claim. This makes the gradual nature explicit upfront.

---

## Major Issues

### 4. Full_cross_layer_critical Results Missing

**Location**: Entire paper

The paper covers layers 5, 8, 10 on GPT-2 and layers 5, 10, 15 on Gemma, but the proposal mentions measuring at λ_c=5e-5 across layers 0, 3, 6, 9, 11. The full_cross_layer_critical experiment was completed (experiment_state.json shows status: completed) but is not mentioned anywhere in the paper.

If the results show uniform saturation at λ_c, this is a negative result that should be reported. If variation was found, this would strengthen the paper.

**Fix**: Add a section or subsection reporting the full_cross_layer_critical results. If negative, frame as informative null result.

### 5. Missing Confidence Intervals in Text

**Location**: Table 2 and throughout

Table 2 reports 95% CIs for correlations, but the text does not prominently discuss these CI widths. For rho=0.705 at layer 8, the 95% CI is [0.12, 0.93]—extremely wide. The text should explicitly discuss this uncertainty.

**Fix**: In Section 4.2, add: "The 95% CI for layer 8 correlation is [0.12, 0.93], reflecting the limited sample size (n=10 categories). This wide interval means the true effect size could be anywhere from weak to very strong."

### 6. Section 5.6 Lacks Concrete Thresholds

**Location**: Section 5.6, "Implications for SAE Evaluation Benchmarks"

The four recommendations (two-tier reporting, ablation as sensitivity check, layer-aware zones, norm verification) are sensible but lack concrete implementation guidance. "Shallow," "mid," and "deep" layers are mentioned but specific relative depth thresholds are not provided.

**Fix**: Add specific thresholds: "shallow (relative depth < 0.5), mid (0.5-0.8), deep (> 0.8)" with justification: "Our layer 8 finding (relative depth 0.67) represents the mid-layer peak where A_j correlation is strongest."

### 7. CV-Steering Mechanism Not Explained

**Location**: Section 4 and Discussion

The paper reports that high-CV features show 2x larger steering effect (pilot_steering_cv), but does not explain why. The mechanism is crucial for understanding whether this finding is generalizable.

**Fix**: Add a paragraph in Section 5.3 (or new subsection) discussing possible mechanisms: "High-CV features may be more steerable because [reason 1], [reason 2], or [reason 3]. Distinguishing these requires [experiment design]."

---

## Minor Issues

### 8. "This suggests that" Overused

The phrase appears 8+ times throughout the paper. Vary with "indicates," "is consistent with," "implies," "points to."

### 9. Figure Numbering Gap

Figures are numbered 1, 3, 4, 5, 6, 7 (Figure 2 omitted). Consider renumbering to close the gap, or explain why Figure 2 was omitted.

### 10. Table 3 Footnote Could Be More Prominent

The footnote explaining the misleading 91.9% difference appears below the table. Consider adding a dagger symbol directly on the cell value to draw scanning readers' attention.

---

## What Works Well

1. **Honest hypothesis framing**: The transparent reporting of H2/H3/H6 failures and reframing as exploratory is the paper's strongest writing feature. It builds reviewer trust rather than undermining it.

2. **Specific numbers lead every claim**: The Abstract opens with exact statistics (98.2%, 7.7%, rho=0.705). Every paragraph in the Results section leads with a number. This evidence-first approach is reviewer-resonant.

3. **Comprehensive related work**: Section 2 covers SAE architectures, absorption pathologies, evaluation benchmarks, and architectural solutions thoroughly. The positioning in Section 2.5 effectively sets up the paper's contributions.

4. **Clear section structure**: The progression from E3v2 (probe scalability) to E6v2 (A_j validation) to E7 (cross-architecture) is logical and builds cumulatively.

5. **Honest negative result framing**: "H2 failed as originally stated" is direct and honorable. The reframing of the layer-dependent pattern as exploratory rather than confirmatory is appropriate.

---

## Priority Fixes for Next Revision

1. **Fix HARKing in H2 framing** (Critical)
2. **Weaken 'training-invariant' to 'consistent with' language** (Critical)
3. **Add chi_ratio=1.88 caveat to Abstract** (Critical)
4. **Integrate full_cross_layer_critical results or document as negative** (Major)
5. **Add CI width discussion in text** (Major)
6. **Add concrete thresholds to Section 5.6** (Medium)