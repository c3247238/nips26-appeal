# Figure 1: UAD Detection Pipeline

## Description

A horizontal flow diagram showing the six steps of the UAD algorithm, left to right.

**Step 1: Extract Activations**
- Box labeled "Corpus (OpenWebText)" with arrow pointing to "SAE"
- Output: Matrix A (n_examples x d_SAE)

**Step 2: Compute Co-Occurrence**
- Box labeled "A^T A" (matrix multiplication)
- Output: Co-occurrence matrix C (d_SAE x d_SAE)

**Step 3: Phi Coefficient Normalization**
- Box labeled "phi coefficient"
- Output: Correlation matrix R (d_SAE x d_SAE)

**Step 4: Hierarchical Clustering**
- Box labeled "HAC (Ward, 50 clusters)"
- Output: Cluster assignments

**Step 5: Same-Cluster Pairs**
- Box labeled "Candidate pairs"
- Output: P_cand (suspected absorbed pairs)

**Step 6: Validate**
- Box labeled "Chanin labels"
- Output: Precision, Recall, F1

**Key annotation (below the flow):**
"No ground truth required for Steps 1-5. Validation (Step 6) uses supervised labels only for evaluation, not for detection."

**Style:** Clean boxes with rounded corners, arrows between steps, blue gradient for processing steps, green for outputs, red for validation. Use sans-serif font (Helvetica or Arial). Total width: 0.95\textwidth.
