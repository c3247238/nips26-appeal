# Critique: Methodology

## Summary Assessment
The Methodology section provides a clear algorithmic description of UAD and DFDA with appropriate mathematical notation. The definitions are precise and the validation protocol table is useful. However, the section has a critical notation inconsistency (activation matrix A described as real-valued but used as binary), lacks a clear explanation of how parent vs. child is assigned within a same-cluster pair, and the DFDA parameter count claim needs verification.

## Score: 6.5/10
**Justification**: Technically competent with good notation discipline, but hampered by the A/binary inconsistency and unclear parent-child assignment logic. To reach 8, fix the notation and add the parent-child disambiguation rule. To reach 9+, add a complexity analysis (time/space) and a formal statement of the absorption signature detection criterion.

---

## Critical Issues

### Issue 1: Activation Matrix A Is Described as Real but Used as Binary
- **Location**: Section 3.3, Step 1 and Step 2
- **Quote**: "Extract feature activation matrix $A \in \mathbb{R}^{N \times d_{\text{SAE}}}$... Compute feature co-occurrence matrix $C = A^T A$, where $C_{ij}$ counts how often features $i$ and $j$ activate together."
- **Problem**: The notation says $A \in \mathbb{R}^{N \times d_{\text{SAE}}}$ (real-valued activations), but the co-occurrence count description implies binary activations ("counts how often... activate together"). The notation table defines $A_{ni} = \phi_i(x_n)$ as the activation value, but the co-occurrence formula $C = A^T A$ would compute weighted sums, not counts. The glossary defines $C_{ij} = \sum_n \mathbb{1}[A_{ni} > 0] \cdot \mathbb{1}[A_{nj} > 0]$, which uses indicator functions.
- **Fix**: Either (a) define $A$ as binary: $A_{ni} = \mathbb{1}[\phi_i(x_n) > 0]$, or (b) use the indicator-based formula from the glossary: $C_{ij} = \sum_n \mathbb{1}[A_{ni} > 0] \cdot \mathbb{1}[A_{nj} > 0]$. The current text contradicts the glossary.

---

## Major Issues

### Issue 2: Parent-Child Assignment Rule Is Not Specified
- **Location**: Section 3.3, Step 5
- **Quote**: "All same-cluster feature pairs form the candidate set $\mathcal{P}_{\text{cand}}$."
- **Problem**: The section identifies same-cluster pairs as candidates but never explains how parent vs. child is assigned within each pair. Is it by activation frequency (lower = parent)? By feature index? By some other criterion? This is a critical algorithmic detail missing from the description.
- **Fix**: Add a step or sub-step: "For each same-cluster pair $(i, j)$, the feature with lower marginal activation frequency is designated the parent, and the feature with higher marginal activation frequency is designated the child. This follows the absorption definition: absorbed parents fire rarely independently but frequently when children fire."

### Issue 3: DFDA Parameter Count May Be Incorrect
- **Location**: Section 3.4
- **Quote**: "where $W_1 \in \mathbb{R}^{64 \times 1}$, $W_2 \in \mathbb{R}^{1 \times 64}$, $b_1 \in \mathbb{R}^{64}$, $b_2 \in \mathbb{R}$. Total: 193 parameters per pair."
- **Problem**: Let's verify: $W_1$ has $64 \times 1 = 64$ params. $W_2$ has $1 \times 64 = 64$ params. $b_1$ has 64 params. $b_2$ has 1 param. Total = $64 + 64 + 64 + 1 = 193$. The math checks out. However, the notation table says $W_1 \in \mathbb{R}^{64 \times 1}$ and $W_2 \in \mathbb{R}^{1 \times 64}$, which is consistent with the text. No issue here -- the count is correct.
- **Fix**: None needed. The parameter count is correct.

### Issue 4: "Top 500 Features" Selection Criterion Is Underspecified
- **Location**: Section 3.3, Hyperparameters
- **Quote**: "We select the top 500 most active features (by total activation count) from $d_{\text{SAE}} = 24{,}576$ total"
- **Problem**: "Total activation count" could mean sum of raw activations, sum of binary activations, or count of non-zero activations. The choice affects which features are selected. Also, the rationale "Most active features from 24,576 total" is tautological, not a rationale.
- **Fix**: Specify the exact metric: "by total activation count (sum of $\phi_i(x_n)$ over all $n$)" or "by activation frequency (number of examples where $\phi_i(x_n) > 0$)." Add a real rationale: "Focusing on the most active features ensures sufficient co-occurrence statistics for reliable phi coefficient estimation while filtering dead features that never activate."

### Issue 5: Phi Coefficient Formula Not Provided
- **Location**: Section 3.3, Step 3
- **Quote**: "Normalize $C$ to phi coefficient correlation matrix $R$... The phi coefficient $\phi_{ij}$ measures association between binary variables"
- **Problem**: The text describes what phi coefficient does but does not give the formula. A methodology section should be self-contained enough that a reader could reimplement the method.
- **Fix**: Add the formula:
  $$\phi_{ij} = \frac{C_{ij}N - C_i C_j}{\sqrt{C_i(N - C_i)C_j(N - C_j)}}$$
  where $C_i = \sum_j C_{ij}$ is the marginal count for feature $i$, or provide a citation to the standard definition.

---

## Minor Issues

- **Section 3.1**: "Feature Absorption [Chanin et al., 2024]" -- the citation is placed after the term, which is fine, but consider whether this should be a formal definition block (e.g., "Definition 1 (Feature Absorption)") for easier reference.
- **Section 3.2**: "SAELens >= 2.0, TransformerLens >= 2.0" -- version requirements in a methodology section are unusual unless the method depends on specific API features. Consider moving to a reproducibility appendix or footnote.
- **Section 3.3**: "Figure 1 illustrates the UAD pipeline" -- the figure is referenced but the text does not walk through it. Add a sentence describing what the figure shows: "Figure 1 shows the six-step pipeline from activation extraction through hierarchical clustering to candidate pair identification."
- **Section 3.4**: "At inference: add the predicted residual to the parent feature's SAE output" -- clarify whether this modifies the SAE's internal state or only the output interpretation. Does $z_p^{\text{comp}}$ feed back into reconstruction?
- **Table 4**: The hyperparameter table is labeled "Table 4" but the outline does not mention a Table 4. This conflicts with the outline's table numbering (Tables 1-3). Renumber to avoid confusion.
- **Validation Protocol table**: The "Success Criteria" column uses ">=" symbols which are fine, but "PARTIAL_PASS" is used as a status in the outline, not a success criterion. The table should only contain criteria, not outcomes.
- **Missing**: No time or space complexity analysis. UAD on 24,576 features with 1,000 samples: what is the asymptotic complexity? This is useful for scalability claims.
- **Missing**: No discussion of why Ward linkage was chosen over other linkages (single, complete, average). A one-sentence justification would strengthen the methodology.

---

## Visual Element Assessment

- [x] **Figures/tables match outline plan**: Figure 1 is referenced as planned. Table 4 (hyperparameters) is not in the outline plan.
- [ ] **All visuals referenced before appearance**: Figure 1 is referenced ("Figure 1 illustrates") but the actual figure appears after the reference, which is correct. However, the figure path `figures/fig1.pdf` cannot be verified.
- [ ] **Captions are self-explanatory**: No separate caption is provided for Figure 1; only an alt-text in the markdown image link.
- [ ] **No text-heavy sections that need visual support**: The algorithm description in Section 3.3 is text-heavy and would benefit from the flow diagram (Figure 1) being present and well-captioned.

---

## What Works Well

1. **Clear Input/Output specification (Sections 3.3, 3.4)**: The "Input / Output / Algorithm" structure makes both UAD and DFDA easy to understand at a glance. This is good technical writing.

2. **Explicit known limitation for DFDA (Section 3.4)**: Stating the metric limitation in the methodology section (not just experiments) shows intellectual honesty and prevents readers from forming false expectations.

3. **Validation protocol table (Section 3.5)**: The staged validation with explicit success criteria is excellent experimental design documentation. This should be standard in methodology sections but rarely is.
