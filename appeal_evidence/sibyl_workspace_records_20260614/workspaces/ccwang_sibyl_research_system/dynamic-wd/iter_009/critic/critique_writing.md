# Writing Critique

## Overall Assessment

The paper is well-written at the sentence level—clear, precise, and well-organized. The problem framing (fragmented WD literature, no unified framework, no controlled comparison) is effective. However, there are structural and rhetorical issues that undermine credibility.

## Critical Issues

### 1. Title Makes Unsupported Causal Claim

The title says "Why Dynamic Weight Decay Methods Are Equivalent Under AdamW." The paper shows THAT they are equivalent on CIFAR-scale, not WHY. The mechanistic hypothesis (Section 6.1) is plausible but unproven—no theoretical derivation, no causal intervention (e.g., ablating AdamW's second-moment estimation to break invariance). Change "Why" to "When" or remove the causal framing entirely.

### 2. Abstract Misrepresents Experimental Coverage

The abstract says "2 architectures" and "2 datasets" implying a fully crossed design. In reality, VGG-16-BN is only tested with SGD on CIFAR-10 (21 of 105 runs). The 84 ResNet-20 runs carry the paper's core claims. This partial coverage should be transparent in the abstract.

### 3. Overclaiming in Conclusion

The conclusion states "Practitioners using AdamW can safely rely on constant weight decay—the simplest strategy already captures the available benefit." This is a universal recommendation based on CIFAR-scale experiments with 270K and 15M parameter models. The paper has no evidence for ImageNet, LLM, or even moderate-scale (ViT-B/16) settings. The recommendation should be scoped: "On CIFAR-scale benchmarks with BN architectures..."

### 4. Section 5.7 Statistical Impropriety

Drawing conclusions from the difference between two non-significant correlations (|Delta rho| = 0.424) is not valid without a formal test. This was flagged in previous iterations. Either add a proper statistical test or remove the directional claim.

### 5. Figure Naming Inconsistency

Figure references in text (Figures 2-8) do not consistently match file names. `theorem2_validation.png` references abandoned Theorem 2 framing. This suggests the paper was assembled from components of an earlier, more theoretical version without full cleanup.

## Minor Issues

- Section 2.2 lists "Chen et al. (2026a)" and "Chen et al. (2026b)" without clarifying these are different papers. Add first names or titles in the citation.
- The phrase "degenerate case of no weight decay at all" in the abstract is confusing—"degenerate" suggests something pathological, but no_wd is a valid ablation baseline.
- Table 1 includes AdamWN and AlphaDecay as special cases but these are not tested experimentally. This creates an expectations gap.
- The paper lacks a "Notation" paragraph or table. Symbols (phi, delta, R_t, etc.) are introduced inline but a reader encountering them out of order will struggle.

## Strengths

- Clean experimental methodology with fair hyperparameter protocol
- Honest limitation section (6.4) that identifies most gaps
- Good use of TOST equivalence testing alongside traditional hypothesis testing
- BEM analysis (Section 5.4) is compelling—the 10x budget variation with <0.5% accuracy change is a striking result
- Weight norm analysis (Section 5.6) provides clean mechanistic evidence for the AdamW vs SGD dichotomy
