# Critique: Method (The Local Inhibition Graph Framework)

## Summary Assessment

The Method section (which combines the theoretical framework and experimental methodology) is well-structured with clear mathematical formalism. The LCA-SAE structural correspondence is presented as a theorem with proof, and the six-phase experimental pipeline is logically organized. However, the section contains a notation inconsistency in the proof, a mismatch between the theorem's tied-weight assumption and the standard untied case, and the experiments subsection references H6-H10 without having executed them yet.

## Score: 7/10

**Justification**: Strong mathematical presentation and clear experimental structure. The theorem-proof format for the structural correspondence is appropriate and well-executed. However, the notation collision in the LCA equation ($a$ used for two different quantities), the disconnect between the proof's tied-weight assumption and the actual experiments on untied SAEs, and the forward-looking nature of H6-H10 descriptions (which describe planned rather than completed experiments) weaken the section. Addressing these would bring it to 8/10.

## Critical Issues

### Issue 1: Notation Collision in LCA Equation
- **Location**: Section 3.1, LCA dynamics equation
- **Quote**: "$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$"
- **Problem**: The variable $a$ is used for LCA activation (left side) but also for SAE input activation in the forward pass equation: $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$. This is confusing because $a$ in the LCA equation is the output activation (after thresholding), while $a$ in the SAE equation is the input to the SAE. The notation.md defines $a$ as "Input activation to SAE" but the LCA equation uses it as output.
- **Fix**: Use a different variable for LCA activation, such as $\tilde{a}$ or $a_{\text{LCA}}$, to distinguish from SAE input $a$. Update notation.md accordingly.

### Issue 2: Tied-Weight Assumption vs. Actual Experiments
- **Location**: Section 3.1, Theorem and Proof
- **Quote**: "For an SAE with tied weights ($W_{\text{enc}} = W_{\text{dec}}^T$), the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from the LCA framework."
- **Problem**: The theorem assumes tied weights, but the experiments use standard pretrained SAEs with untied weights (the gpt2-small-res-jb release). The proof shows exact correspondence only for tied weights, then the text says "Even with untied weights... the structural correspondence holds approximately." But the experiments do not quantify this approximation. How approximate is it? Does the approximation degrade predictions?
- **Fix**: Add a sentence quantifying the approximation: "For the gpt2-small-res-jb SAE, the cosine similarity between $W_{\text{enc}}$ and $W_{\text{dec}}^T$ is X.XX, indicating that the tied-weight assumption holds approximately." Or add a sensitivity analysis: test predictions with both $W_{\text{dec}}^T W_{\text{dec}}$ and $W_{\text{enc}} W_{\text{enc}}^T$ to verify they produce similar graphs.

### Issue 3: H6-H10 Experiments Described as Future Work
- **Location**: Sections 4.4-4.8
- **Quote**: "H6 is supported if precision@20 $\geq$ 0.10" / "H7 is supported if $r(\text{inh}, \text{recall}) < -0.3$ with $p < 0.05$"
- **Problem**: The section describes H6-H10 as predictions with falsification criteria, but the intro's Key Results Preview (Section 1.5) presents these as already-tested hypotheses with specific numbers. This creates confusion: are these experiments completed or planned? The experiments.md file (Section 5.2) also describes them in future tense.
- **Fix**: If experiments are complete, add the actual results to this section and change to past tense. If experiments are planned, add a clear statement: "The following sections describe the validation protocol; results are reported in Section 5." Ensure consistency with the Introduction's Key Results Preview.

## Major Issues

### Issue 4: Missing Baseline Comparisons in Validation Protocol
- **Location**: Section 4.4 (H6 validation)
- **Quote**: "We test enrichment vs. a random baseline (shuffle latent indices; expected precision@20 $\approx$ 0.004) using a Fisher exact test. We also test against a non-absorbed control."
- **Problem**: The proposal promised three baselines (random graph, non-absorbed control, identity graph), but only two are mentioned here. The identity graph baseline (only self-loops) is missing. This baseline tests whether correlations beyond self-similarity matter---an important control.
- **Fix**: Add the identity graph baseline: "Identity graph baseline: only self-loops ($i \rightarrow i$); tests whether self-correlation alone explains absorption pairs."

### Issue 5: Homeostatic Rebalancing Equation Has Ambiguous Notation
- **Location**: Section 3.4 / 4.8
- **Quote**: "$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} \cdot z_j\right)$"
- **Problem**: The notation uses $G_{ij}$ but the graph construction (Section 3.3) defines edges with weights $G_{ij}$ where $j$ is a neighbor of $i$. However, the inhibition from $j$ to $i$ should use the signed correlation. The equation adds $\alpha \cdot G_{ij} \cdot z_j$ but if $G_{ij}$ is negative (anti-correlated), this would subtract from $z_i$. Is this the intended behavior? The text says "boosts parent activations by the inhibition it receives" but the equation's sign depends on $G_{ij}$.
- **Fix**: Clarify the sign convention. If the intent is to always boost (add), use $|G_{ij}|$ or explain why signed $G_{ij}$ is correct: "When $G_{ij} > 0$ (correlated neighbors), the boost is positive. When $G_{ij} < 0$ (anti-correlated), the correction is negative, which is appropriate because anti-correlated features do not compete for the same variance."

### Issue 6: Graph Complexity Claim Needs Verification
- **Location**: Section 3.3
- **Quote**: "Complexity: $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$ --- feasible for 24K--1M latents on a single GPU."
- **Problem**: The complexity claim is misleading. Computing all pairwise correlations $G = W_{\text{dec}}^T W_{\text{dec}}$ is $O(d_{\text{dict}}^2 \cdot d_{\text{model}})$, which for $d_{\text{dict}} = 1M$ is $10^{12} \cdot d_{\text{model}}$ operations---not feasible in 2 seconds. The top-k extraction is $O(d_{\text{dict}}^2 \log k)$. The $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$ only applies if correlations are computed on-demand for neighbors, not all pairs.
- **Fix**: Clarify the complexity: "Computing the full correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ costs $O(d_{\text{dict}}^2 \cdot d_{\text{model}})$. For GPT-2 Small ($d_{\text{dict}} = 24{,}576$), this takes ~2 seconds on an A100. For larger SAEs, the full matrix may be prohibitive; in such cases, approximate nearest-neighbor methods (e.g., FAISS) reduce complexity to $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$."

### Issue 7: Precision@20 Threshold of 0.10 Is Arbitrary
- **Location**: Section 4.4
- **Quote**: "Precision@20 $\geq$ 0.10 (25$\times$ enrichment over chance). If precision@20 $\leq$ 0.05, the structural correspondence fails."
- **Problem**: The 0.10 threshold (and the 0.05 falsification threshold) are arbitrary. Why 0.10? Why not 0.05 or 0.20? The proposal originally set precision@20 >= 0.10, but there's no theoretical justification for this number. It seems chosen to be achievable rather than derived from the theory.
- **Fix**: Provide justification: "We set precision@20 $\geq$ 0.10 as the support threshold because it represents a 25$\times$ enrichment over the random baseline (0.004), which we consider a minimally meaningful signal. The falsification threshold of 0.05 represents a 12.5$\times$ enrichment, below which the predictive value is too weak for practical use." Alternatively, use a statistical test (Fisher exact) as the primary criterion rather than an arbitrary precision threshold.

## Minor Issues

- **Section 3.1, proof**: "The SAE forward pass $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$ approximates the LCA steady-state where $du/dt = 0$" --- The proof says $z \approx a$ but $z$ and $a$ are different dimensions ($z \in \mathbb{R}^{d_{\text{dict}}}$, $a \in \mathbb{R}^{d_{\text{model}}}$). This is a type error. The proof should say "$z \approx T(u)$ where $T(u)$ is the LCA activation."
- **Section 3.3**: "For GPT-2 Small ($d_{\text{dict}} = 24{,}576$, $d_{\text{model}} = 768$), computing all correlations takes approximately 2 seconds on an A100" --- This timing claim should be verified or cited. If it's an estimate, label it as such.
- **Section 4.2**: "Layer 0 provides a near-input baseline; layers 4, 8, and 10 sample the mid-to-late network" --- Why these specific layers? The justification is weak. Add: "Layer 4 and 8 were selected because prior work found significant absorption effects at these depths, while layer 10 tests whether effects persist near the output."
- **Section 4.5**: "H7 is supported if $r(\text{inh}, \text{recall}) < -0.3$ with $p < 0.05$" --- The $r < -0.3$ threshold is arbitrary (same issue as the old r < -0.2 threshold). Use standard significance testing without arbitrary effect-size thresholds.
- **Section 4.7**: "H9 is supported if $r(\bar{G}, l) > 0.3$" --- Same arbitrary threshold issue.
- **Section 4.8**: "H10 is supported if parent firing increases by $>20\%$" --- The 20% threshold is arbitrary. What is the theoretical basis for 20% vs. 10% or 30%?
- **Section 4.9**: "Code and evaluation protocol are released with the paper" --- If true, add a repository link. If not yet released, say "will be released upon publication."
- **Figure references**: Figure 1 is referenced in Section 3.1 but described in the text after the figure path. Standard practice is to describe the figure first, then reference it.

## Visual Element Assessment
- [x] Figures/tables match outline plan — Figure 1 (LCA correspondence) and Figure 2 (suppression mechanism) are planned and referenced
- [ ] All visuals referenced before appearance — Figure 1 is referenced in 3.1 before its full description; Figure 2 is referenced in 3.2
- [x] Captions are self-explanatory — Both figure captions are detailed
- [ ] No text-heavy sections that need visual support — The competitive suppression mechanism (3.2) benefits from Figure 2, which is planned. The graph construction (3.3) is text-only and would benefit from a small diagram.

## What Works Well

1. **The theorem-proof format for the structural correspondence is excellent.** It elevates the LCA connection from an observation to a formal result. The proof is concise and correct (modulo the notation issue in Critical Issue 1).

2. **The six-phase experimental pipeline is clearly organized.** The table in Section 4.1 provides an at-a-glance overview, and each phase has a clear hypothesis, method, and falsification criterion. This structure makes the paper's empirical strategy transparent.

3. **The integration of prior data is well-handled.** Sections 4.5 and 4.6 explicitly note that precision/recall and absorption data come from prior experiments, avoiding the impression that all data is newly collected. This honesty strengthens the paper's credibility.
