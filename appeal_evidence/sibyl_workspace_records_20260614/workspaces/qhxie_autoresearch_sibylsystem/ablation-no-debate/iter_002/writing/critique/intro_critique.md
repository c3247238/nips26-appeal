# Critique: Introduction

## Summary Assessment
The Introduction is well-structured with a clear problem statement, well-motivated research gap, and specific research questions. The logical progression from problem → gap → questions → contributions is effective. However, the contributions section reads like an abstract rather than a roadmap, and the transition from theoretical framing to the five empirical questions could be more explicitly tied to the experimental sections where each is answered.

## Score: 7/10
**Justification**: Strong motivation and clear structure. Deducted points for: (1) the contributions list duplicating abstract content rather than framing the paper's narrative arc, (2) missing explicit section-by-section mapping from research questions to experimental sections, (3) the final contribution sentence is a conclusion rather than a contribution.

## Critical Issues

### Issue 1: Contributions Section Reads as Abstract, Not Roadmap
- **Location**: Section 1.4, lines 33-47
- **Problem**: Items 1-6 are result statements (e.g., "encoder effect E_enc = 0.843 ± 0.082") rather than contribution statements. A contributions section should describe what the paper adds to knowledge, not repeat findings. Compare to the structure of Bricken et al. (2023) or Chanin et al. (2024) whose contributions explicitly state novel methods or frameworks introduced.
- **Fix**: Reframe contributions as methodological or conceptual advances. For example: "We introduce a 2x2 factorial decomposition method for isolating encoder vs. decoder contributions to absorption" (methodological); "We identify encoder geometry as the dominant driver of absorption, redirecting mitigation efforts from decoder-side to encoder-side strategies" (conceptual).

### Issue 2: No Explicit Section Mapping for Research Questions
- **Location**: Section 1.3, lines 18-29
- **Problem**: Five research questions are stated but not mapped to the sections where they are answered. The reader must infer that RQ1 maps to Section 5.4/6.1, RQ2 to Section 6.2, etc. This is especially problematic because the Method section (5.1-5.7) and Results section structure may not be obvious from the question text alone.
- **Fix**: Add a brief sentence after each research question noting which section addresses it. Example: "RQ1 is addressed through the 2x2 factorial design described in Section 5.4 and results in Section 6.1."

### Issue 3: The Research Gap Section Is Too Short
- **Location**: Section 1.2, lines 12-15 (less than 10 lines total)
- **Problem**: Section 1.2 "The Mechanistic Gap" is the intellectual core of the introduction but is the shortest subsection. It correctly identifies the gap but does not explain *why* prior approaches missed the decomposition, what makes this decomposition hard, or what the implications are of finally answering it.
- **Fix**: Expand to 2-3 paragraphs: (1) explain why joint optimization makes decomposition non-obvious, (2) note what the 2x2 design reveals that prior analytical approaches cannot, (3) briefly state what changes in the mitigation landscape if the answer is "encoder."

### Issue 4: Missing Connection to Prior Negative Results
- **Location**: Section 1.1, lines 6-9
- **Problem**: The problem statement mentions that absorption "undermines" the promise of SAEs but does not acknowledge that some prior work (e.g., Bricken et al.) still achieved valuable interpretability despite absorption. This makes the problem statement slightly overstated.
- **Fix**: Add a sentence acknowledging that absorption does not preclude interpretability but limits reliability: "While absorption does not prevent feature discovery entirely, it creates systematic blind spots where parent concepts go undetected."

## Minor Issues

- **Notation**: The paper equation in 1.1 lacks a bias term for the decoder ($b_{dec}$). The standard SAE equation is $x \approx W_{dec} \, \sigma(W_{enc}x + b_{enc}) + b_{dec}$. Add $b_{dec}$.
- **Citation format**: Some citations use "et al., 2024/2025" format. Pick one year consistently. The Chanin et al. citation in 1.1 says "(2024)" but the references may say 2025.
- **Acronym usage**: "SAE" is used before being defined (it appears in the title but the expansion "Sparse Autoencoder" first appears in 1.1). Add expansion in the title or first sentence.
