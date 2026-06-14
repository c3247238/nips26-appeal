# Critique: Methodology

## Summary Assessment
The Methodology section clearly describes the four-phase pipeline with appropriate mathematical formalism. It is largely reproducible and well-structured. However, it contains several inconsistencies with the outline and other sections, a questionable statistical claim about power analysis, and missing controls that were promised in the proposal.

## Score: 6/10
**Justification**: Good structure and mathematical clarity, but inconsistencies in absorption thresholds, a flawed power analysis claim, missing controls, and a questionable CV calculation drag the score down. Fixing these would bring it to 7-8.

---

## Critical Issues

### Issue 1: Absorption Classification Thresholds Are Inconsistent
- **Location**: Section 4.3
- **Quote**: "HIGH: A(f) > 0.10; MEDIUM: 0.05 <= A(f) <= 0.10; LOW: A(f) < 0.05"
- **Problem**: These thresholds differ from the outline which states: "HIGH (>50%), MEDIUM (10-50%), LOW (<10%)". The Method uses 10% as the HIGH threshold while the outline uses 50%. The proposal also uses >50% for HIGH. This is a major inconsistency that affects interpretation of results.
- **Fix**: Standardize across all sections. The Method's 10% threshold is more appropriate given the empirical distribution (max absorption is 0.242), but this must be consistent with the outline and proposal. Update the outline and proposal to match the Method, or vice versa.

### Issue 2: Power Analysis Claim Is Incorrect
- **Location**: Section 4.6, final paragraph
- **Quote**: "This sample size provides 80% power to detect a Pearson correlation of |r| >= 0.50 at alpha = 0.05 (two-tailed)."
- **Problem**: For n=26, the power to detect r=0.50 at alpha=0.05 is approximately 64-68%, not 80%. Using G*Power or standard power tables: for a two-tailed test with n=26, to achieve 80% power for r=0.50, you would need n~29-30. The claim overstates the study's statistical power.
- **Fix**: Correct to "This sample size provides approximately 65% power to detect a Pearson correlation of |r| >= 0.50 at alpha = 0.05 (two-tailed)." Alternatively, recalculate what effect size n=26 provides 80% power for (approximately |r| >= 0.55).

### Issue 3: Missing Random Feature Baseline
- **Location**: Section 4.4, Controls paragraph
- **Quote**: "We do not include a random feature baseline because the primary comparison is between HIGH and LOW absorption features within the same model and layer, which controls for layer-specific activation statistics."
- **Problem**: The proposal explicitly promised a random feature baseline: "Include random feature baseline (randomly selected SAE latents)" with expected outcome "Near-zero steering effectiveness, random probing F1". The Method's justification for omitting it is weak -- within-layer comparison doesn't validate that the steering effect is specific to the feature direction rather than any decoder direction.
- **Fix**: Either (a) add the random feature baseline and report results, or (b) provide a stronger justification for omitting it, acknowledging this as a limitation.

---

## Major Issues

### Issue 4: CV Calculation Is Questionable
- **Location**: Section 4.6
- **Quote**: "The coefficient of variation CV = sigma / mu quantifies consistency: CV < 0.5 indicates consistent degradation coefficients."
- **Problem**: The CV is reported as negative (-1.079 for H1, -0.932 for H2) in the Results section. CV = sigma / |mu| should be positive by definition (standard deviation is always non-negative). The negative values arise because the mean slope is negative for H2 and near-zero for H1. Using CV with opposite-sign slopes is methodologically problematic -- CV is designed for ratio-scale positive data, not signed coefficients.
- **Fix**: Replace CV with an alternative consistency measure. Options: (a) report the absolute difference in slopes |beta_4 - beta_8|, (b) use a formal test for slope homogeneity, or (c) simply note that slopes have opposite signs, which is sufficient to reject consistency. The CV < 0.5 criterion should be dropped entirely.

### Issue 5: Steering Success Metric Is Underdefined
- **Location**: Section 4.4
- **Quote**: "S(f, s) = (1/N) sum_{i=1}^N 1[P_f(t_i) > P_0(t_i)]"
- **Problem**: The metric checks whether the probability of the target token increases, but doesn't quantify by how much. A feature that increases probability by 0.1% counts the same as one that increases it by 50%. The proposal mentioned "% increase in target-letter token probability vs. unsteered baseline" which suggests a relative measure, but the Method uses a binary threshold.
- **Fix**: Clarify whether the primary metric is binary success rate or mean probability lift (Delta P). If both are computed, report both. The binary metric may lack sensitivity to detect subtle degradation.

### Issue 6: Probing Description Lacks Key Details
- **Location**: Section 4.5
- **Problem**: The sparse probing description omits several critical details: (a) What classifier is used? (logistic regression? linear SVM?), (b) How is k-sparsity enforced? (L0 regularization? top-k selection?), (c) What is the train/test split?, (d) How is the "full-activation" probe trained -- is it just k=n_latent?
- **Fix**: Add: "We use logistic regression with L1 regularization for k-sparse probes, selecting the k features with largest absolute weights. The full-activation probe uses all 24,576 latents with L2-regularized logistic regression. We use 80/20 train/test split with stratification."

### Issue 7: Layer 0 and 10 Results Are Underutilized
- **Location**: Section 4.6
- **Quote**: "For each layer where both absorption and task data are available (l in {4, 8})"
- **Problem**: Absorption data was collected for layers 0 and 10, but steering/probing only for layers 4 and 8. The reason is not explained. Layer 0 (near-input) and layer 10 (late) could provide valuable context about whether the absorption-task relationship varies with layer depth.
- **Fix**: Explain why layers 0 and 10 were excluded from steering/probing: "Steering and probing were conducted on layers 4 and 8 as representative mid-network layers where feature abstraction is substantial but not yet dominated by output-specific representations. Layer 0 (near-input) lacks sufficient feature abstraction for meaningful first-letter features, and layer 10 approaches the output layer where steering effects may be confounded by the unembedding."

---

## Minor Issues

- **Section 4.1**: "illustrated in Figure 1" -- Figure 1 is referenced but the PDF path suggests it may not be generated yet. Verify figure generation status.
- **Section 4.2**: "The original plan targeted Gemma-2-2B, but gated HuggingFace access prevented loading" -- this is important context but belongs in Limitations, not Method. Here it distracts from the methodology description.
- **Section 4.3**: "We classify features into three categories" but only HIGH and LOW are used in results. The MEDIUM category (0.05-0.10) has no features in the data, making it an empty bin. Remove MEDIUM or note that no features fell into this category.
- **Section 4.4**: "Prompts are drawn from a curated vocabulary list" -- what list? How curated? Provide source or describe criteria.
- **Section 4.5**: "We also compute F1_full using all 24,576 latents" -- but Results section 5.3 only mentions this in passing without numbers. Either report the numbers or remove the claim.
- **Section 4.6**: "Falsification criteria. H1 and H2 are not supported if r > -0.2 or p > 0.05" -- the r > -0.2 criterion is arbitrary and non-standard. See Critical Issue 2 in intro critique.
- **Section 4.7**: "Code and evaluation protocol are released with the paper" -- if true, add a link or repository reference. If not yet released, remove or note "will be released".

---

## Visual Element Assessment
- [x] Figures/tables match outline plan -- Figure 1 (pipeline) is described and referenced
- [ ] All visuals referenced before appearance -- Figure 1 is referenced in 4.1 before its description, but the figure description comes after the reference. Standard practice is to reference after describing.
- [x] Captions are self-explanatory -- Figure 1 caption is detailed
- [ ] No text-heavy sections that need visual support -- Section 4.3 (absorption detection) would benefit from a small diagram showing parent/child feature relationships.

---

## What Works Well
1. **Clear four-phase structure**: The pipeline is easy to follow and each phase feeds logically into the next.
2. **Mathematical notation is consistent**: All symbols match notation.md and are used correctly throughout.
3. **Reproducibility details**: Software versions, random seed, and exact model specifications are provided, enabling replication.

---

## Revision Notes (Post-Fix)

The following critical issues from this critique have been addressed in the revised sections:

- Power analysis corrected: 80% → ~65% power for n=26, |r|>=0.50
- Model size standardized: 124M parameters (85M non-embedding) throughout
- Falsification criteria fixed: removed arbitrary r<-0.2 threshold, now uses standard p<0.05
- H3 CV definition fixed: CV = sigma/|mu| (was sigma/mu, producing impossible negative values)
