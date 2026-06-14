# Critique: Method

## Summary Assessment
The Method section presents a coherent theoretical framework connecting coefficient of variation (CV) to steering actionability, followed by detailed experimental protocols. However, the section suffers from inconsistent section numbering that jumps from "3.x" to "4.x" mid-document, creating confusion about the paper structure. Additionally, several visual elements referenced in the outline (Figure 3: CV distribution, Figure 5: mechanism diagram) are missing from the body text, and a key mechanistic claim is presented as established fact rather than hypothesis to be tested.

## Score: 6/10
**Justification**: The method is technically sound and well-reasoned, but the structural disorganization (inconsistent section numbers, missing visual elements) and overclaimed mechanistic language prevent it from reaching a higher score. The section reads as two halves that were never properly integrated.

## Critical Issues

### Issue 1: Section Numbering Discontinuity
- **Location**: Lines 5-6 and 31
- **Quote**: "## 3.1 The Variance Paradox ... ## 4.1 Experimental Setup"
- **Problem**: The section uses both "3.x" and "4.x" numbering within a single document. The outline shows Section 3 as "Theoretical Framework" and Section 4 as "Experiments," but the method.md document mixes these numbering systems. This creates confusion about whether Phase Transition belongs to Theory or Method, and whether Experimental Setup is part of a larger experiments section or standalone. A reader checking the outline against the draft would find mismatched section numbers.
- **Fix**: Either (a) split into two files: method.md for theory (3.X content), experiments.md for protocol (4.X content), or (b) re-number consistently within method.md (e.g., 2.1-2.4 for theory, 3.1-3.5 for experiments). The current draft's hybrid numbering suggests the authors copied section numbers from an outline without adjusting for actual chapter structure.

### Issue 2: Mechanistic Language Overclaims Before Validation
- **Location**: Section 3.3, lines 22-25
- **Quote**: "When an absorbed parent feature is steered, the activation flows through its child feature(s) before affecting model outputs. For high-CV (robust) features, the child's context-sensitive routing allows the steering modulation to propagate..."
- **Problem**: This presents the causal mediation mechanism as established fact ("the activation flows through"), but the mechanism has not yet been experimentally validated. Section 4.3 describes activation patching experiments to validate the mechanism, but Section 3.3 introduces it as a conclusion. This reverses the logical structure—hypothesis should precede validation.
- **Fix**: Rephrase Section 3.3 to frame the mechanism as a hypothesis: "We propose that when an absorbed parent feature is steered, activation flows through its child feature(s)... We test this via activation patching..." This preserves the theoretical framework while correctly framing the mechanism as a prediction to be tested.

### Issue 3: Missing Visual Elements Not Referenced in Text
- **Location**: HTML comments at end of file (lines 71-74)
- **Quote**: "<!-- FIGURES: Figure 3: gen_fig3_cv_comparison.py, fig3_cv_comparison.pdf — CV distribution... Figure 5: fig5_mechanism_desc.md — Architecture diagram... -->"
- **Problem**: Two figures are planned (Figure 3: CV distribution, Figure 5: mechanism diagram) but neither is referenced in the body text. Figure 3 illustrates the key "variance paradox" claim (733x CV ratio) but appears only in HTML comments, not as a proper figure reference with caption. A reader would not know this figure exists or what it contains.
- **Fix**: Add proper figure references within the body text:
  - After "733-fold ratio" in Section 3.1: "Figure 3 illustrates this distribution..."
  - After "bypass routing mechanism" in Section 3.2: "Figure 5 shows the proposed routing architecture..."

## Major Issues

### Issue 4: "Context-Sensitive" Routing Not Operationally Defined
- **Location**: Section 3.2, lines 14-17
- **Quote**: "Robust absorbed features (high-CV, CV > 1.0) route through context-sensitive child channels that preserve steering potential."
- **Problem**: The phrase "context-sensitive child channels" is central to the mechanism but is not operationally defined. The glossary defines it as "a routing mechanism where high-CV features activate strongly in specific contexts but weakly in others," but this restates the CV correlation without explaining the causal mechanism. What mathematical property distinguishes context-sensitive from stable routing?
- **Fix**: Add operationalization: "Context-sensitive routing means the child feature's contribution to the output is modulated by input context. Formally, if $c_j(x)$ is the child routing coefficient for input $x$, high-CV features exhibit $Var_x[c_j(x)] > 0$, while low-CV features exhibit $Var_x[c_j(x)] \approx 0$." This connects the statistical definition (CV) to the mechanistic claim.

### Issue 5: Activation Patching Protocol Missing Control Condition
- **Location**: Section 4.3, lines 52-59
- **Quote**: "Zero the child feature activation. Measure parent feature recovery: $R_{parent} = (logits_{patched} - logits_{child\_absent}) / (logits_{clean} - logits_{child\_absent})$"
- **Problem**: The activation patching protocol measures parent recovery when the child is zeroed, but does not include a control condition where an unrelated (non-child) feature is zeroed. Without a control, it is unclear whether the recovery is specific to the parent-child relationship or reflects general effects of feature ablation on model behavior.
- **Fix**: Add a control condition: "As a control, we also zero a randomly selected non-absorbed feature and measure logit change. If parent recovery is substantially larger than control recovery, this confirms the parent-child relationship is specific."

### Issue 6: Decoder Orthogonality Falsification Criterion Not Justified
- **Location**: Section 4.4, lines 65
- **Quote**: "Falsification criterion: Correlation $r > 0.3$ between orthogonality and steering effectiveness."
- **Problem**: The threshold $r > 0.3$ is stated but not justified. Why 0.3 and not 0.2 or 0.5? Additionally, the sign expectation is unclear—is a correlation of -0.4 considered supporting or contradicting H6?
- **Fix**: Add justification: "We set $r > 0.3$ as the falsification threshold based on medium effect size conventions in SAE steering studies (Karvonen et al., 2025). We use $|r|$ because H6 predicts a positive relationship, but any significant correlation would indicate orthogonality-steering relationship."

### Issue 7: Cross-Architecture Validation Has No Quantitative Details
- **Location**: Section 4.5, lines 68-69
- **Quote**: "We test whether the CV threshold of 1.0 generalizes or requires model-specific calibration."
- **Problem**: The cross-architecture validation section provides no quantitative details—no steering effects, no effect ratios, no comparison to GPT-2 results. The experiments.md notes "Cross-architecture results require detailed integration" but the method section should at minimum report what was observed even if not fully analyzed.
- **Fix**: Report preliminary findings: "Preliminary analysis suggests the CV-steering relationship [does/does not] replicate on Gemma-2-2B, with [observed effect ratio] compared to [GPT-2 ratio of 1.47]." If no results are available, remove this subsection and note it as future work.

## Minor Issues

- **Section 3.4**: "quasi-critical behavior" is not defined in the method section body. Add brief definition: "quasi-critical behavior refers to a phase transition broadened by finite-size effects, exhibiting gradual rather than sharp onset."
- **Section 4.1**: "top 30 features from each group by CV score" — does not specify tie-breaking when more than 30 features share similar CV scores. Suggest: "top 30 features from each group by CV score (ties broken randomly)."
- **Section 4.2, item 2**: "Add $\tau \cdot d_j$" uses $\cdot$ for scalar multiplication but notation.md specifies $\times$. Use consistent notation: "Add $\tau \times d_j$ to the residual stream."
- **Section 4.4**: Mentions "H6" in the heading but H6 is a hypothesis identifier, not the method protocol. Consider rephrasing as "Decoder Orthogonality Analysis" without the H6 tag.

## Visual Element Assessment

- [ ] Figures/tables match outline plan — PARTIAL: Figure 3 and Figure 5 are mentioned in HTML comments but not referenced in body text
- [ ] All visuals referenced before appearance — FAIL: Figure 3 and Figure 5 are not referenced in the method text
- [ ] Captions are self-explanatory — CANNOT ASSESS: No captions appear in the draft
- [x] No text-heavy sections that need visual support — PASS: The method section is appropriately textual
- [ ] Figure numbers consistent with outline — PARTIAL: Outline mentions Figure 3 and Figure 5 but method draft has no figure references

## What Works Well

### Strength 1: Precise CV Operationalization (Section 3.1)
The coefficient of variation formula "$CV = \sigma / \mu$" is clearly defined with explicit definitions of both $\sigma$ and $\mu$. The comparison to $CV_{non-absorbed} \approx 0.01$ provides a concrete reference point. The interpretation that "absorption selectively preserves context-sensitive specialized information" is a strong framing.

### Strength 2: Clear Steering Protocol (Section 4.2)
The step-by-step steering protocol is detailed enough for replication:
- "Add $\tau \cdot d_j$ to the residual stream at the target position" (explicit intervention)
- Five specific prompts listed verbatim (reproducibility)
- Metric defined as absolute logit change (unambiguous outcome)
- Statistical test specified with exact parameters (Welch's t-test, $\alpha = 0.01$, BH correction)

### Strength 3: Honest Acknowledgment of Pilot-Derived Threshold (Section 3.2)
"The threshold $CV = 1.0$ separates these subpopulations empirically, based on pilot experiments showing the 2x steering effect difference between groups." This acknowledges the threshold was derived from data rather than theory, allowing readers to assess confirmatory vs. exploratory status.
