# Critique: Methodology

## Summary Assessment
The Methodology section is thorough and well-structured. The synthetic hierarchy generation, SAE training protocol, absorption metric, and factorial design are all clearly described. The section benefits from precise operationalization of abstract concepts (absorption rate, encoder effect, decoder effect). However, several critical gaps reduce reproducibility and clarity: the factorial diagram is referenced but its source is unclear, the steering protocol is vague about key implementation details, and the safety-critical feature analysis lacks sufficient methodological specificity.

## Score: 6/10
**Justification**: Strong experimental design with clear factorial structure. Deducted points for: (1) Figure 7 caption references a PDF that may not be included in the submission, (2) steering protocol lacks key details (how is parent direction determined? what does "reconstruction quality" mean as an ablation metric?), (3) safety feature selection procedure is under-specified, (4) the held-out generalization description (80/20 split) needs more detail on whether this is by hierarchy instance or by sample.

## Critical Issues

### Issue 1: Figure 7 Factorial Diagram Is Referenced But May Not Exist in Final Paper
- **Location**: Section 5.4, lines 37-39; FIGURES block at end of section
- **Problem**: The text references "Figure 7 illustrates the design" showing the 2x2 decomposition. The FIGURES block lists "figure_7_factorial_design.pdf" and "figure_7_factorial_design_desc.md". The description file likely contains the LaTeX/tikzpicture code or a placeholder description. In a final submission, this figure must render correctly. The current reference in the text ("Figure 7") should be verified against the actual figure numbering in the paper.
- **Fix**: (1) Verify the figure file is included and renders correctly. (2) Ensure the figure caption clearly explains what each of the four conditions (A, B, C, D) represents and what the arrows in the diagram represent. (3) Consider whether the figure needs a legend explaining the absorption pathway arrows.

### Issue 2: Steering Protocol Is Under-Specified
- **Location**: Section 5.5, lines 45-51
- **Problem**: Three key details are missing: (1) How is the "parent direction" $v_{parent}$ determined? Is it the mean activation direction of tokens where the parent is active? The encoder's learned direction? Something else? (2) The metric "change in reconstruction quality when the feature is ablated" -- ablation means zeroing the feature activation or zeroing the decoder column? Before and after compared to what baseline? (3) The paper reports $s_{ratio} = s_{abs} / s_{non}$ at $\alpha = 2.0$, but how is $\alpha$ scaled -- is it the same across all features or feature-specific?
- **Fix**: Add explicit detail on $v_{parent}$ determination (e.g., "the mean of residual stream activations across all parent-active tokens, computed before SAE training"). Define ablation explicitly: "zeroing the corresponding column of $W_{dec}$." Specify that $\alpha$ is scaled relative to the feature's mean activation magnitude.

### Issue 3: Safety Feature Selection Procedure Lacks Reproducibility
- **Location**: Section 5.6, lines 55-59
- **Problem**: "Querying Neuronpedia for annotations related to deception, jailbreaking, harm, or manipulation" is not a reproducible selection procedure. Neuronpedia annotations are community-contributed and change over time. The paper needs to either: (a) cite a specific snapshot or list of feature IDs, or (b) describe the selection criteria more precisely (e.g., features with safety-related natural language explanations in the top-5 most activating tokens).
- **Fix**: Either provide a table of the 20 specific safety feature IDs (e.g., from the Gemma Scope release) or describe a more precise algorithmic criterion. Also note how the "matched non-safety control" is selected -- "matched by activation frequency distribution" needs a specific metric (KL divergence? correlation? Kolmogorov-Smirnov statistic?).

### Issue 4: Held-Out Generalization Details Insufficient
- **Location**: Section 5.7, "Held-out generalization" paragraph, lines 69-70
- **Problem**: "Split synthetic data 80/20 by hierarchy instance (not by sample)" -- but 80/20 of what? 80% of hierarchies for training, 20% held out? Or 80% of samples within each hierarchy? The latter would not test generalization to new hierarchies. Also unclear: how many hierarchies total per seed (5 as stated in 5.1)? If 5 total and 80/20 by instance, that means ~4 train and ~1 test per seed, which is very small for generalization testing.
- **Fix**: Clarify: "We generate 5 hierarchies per seed, hold out 1 hierarchy per seed for testing (20%), and train on the remaining 4 (80%)." Also note that with n=1 held-out hierarchy per seed, the correlation estimate has only 5 data points -- add a statistical caveat about the wide confidence intervals this implies.

## Minor Issues

- **Equation formatting**: The absorption equation (Section 5.3) uses set notation which is clear, but the inline version says "absorption rate $\alpha_{abs}$" -- verify the mathematical typesetting is consistent with the rest of the paper's conventions.
- **Condition A description**: Section 5.4 says Condition A is "random initialization baseline" but this is also the starting point for conditions B, C, D before training. Verify this is clearly understood as the *fully random* case where neither encoder nor decoder changes from initialization.
- **L0 sparsity range**: Section 5.2 says $L_0 \in \{20, 32, 50\}$ but the results show $L_0 = 20, 32, 50$ in the ablation. Confirm all three levels are used in both training and ablation, not just as training hyperparameters.
