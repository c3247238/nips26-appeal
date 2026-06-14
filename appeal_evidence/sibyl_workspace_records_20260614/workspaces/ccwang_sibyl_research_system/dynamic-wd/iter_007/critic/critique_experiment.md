# Experiment Critique

## Critical Issues

### 1. Certified Band Visualization is Unreadable

The certified band figure (Figure 8) is the paper's most important theoretical visualization -- and it is completely unreadable. The plot shows what appears to be random noise: high-frequency oscillating colored lines filling the entire y-axis range (0 to 0.0006). No band narrowing is visible. No individual method trajectory is distinguishable. The resolution is poor and the figure appears to have been generated at low DPI or reduced in size. This figure must be completely redesigned before the paper can be submitted.

### 2. Missing ImageNet Experiments

The project spec requires ImageNet experiments. CWD, AlphaDecay, and SWD all report improvements at scales larger than CIFAR. Testing the Phi Invariance Conjecture only on CIFAR-10/100 is insufficient -- the community already informally knows that WD strategy barely matters on CIFAR with AdamW. ImageNet with ResNet-50 would be the minimum credible scale for this paper's claims.

### 3. BEM Computation Bug

BEM for half_lambda reports 0.000 in both Table 6 and the diagnostic heatmap. By the paper's own definition, half_lambda (phi=0.5) should have BEM=0.5. This is either a code bug or a definition mismatch. The entire budget equivalence analysis section (5.4) is potentially invalidated because "10x variation in effective WD budget" may actually be less if BEM is miscalculated.

### 4. Method Set Inconsistency Across Figures

Figure 3 (multi-architecture comparison) shows PMP-WD in the ResNet-20 panel but not in the VGG-16-BN panel. The VGG panel shows Half-lambda and Random Mask instead. The spread comparison (0.49pp vs 0.16pp) is computed over different method sets, making the direct comparison invalid.

## Major Issues

### 5. Incomplete VGG-16-BN Evaluation

VGG-16-BN is tested only under SGD on CIFAR-10 (21 experiments). Without AdamW runs on VGG-16-BN, the paper cannot claim "2 architectures" in general -- VGG data is architecture-specific and optimizer-specific. Without CIFAR-100 runs, the generalization claim is weaker.

### 6. No Architecture Without Batch Normalization

Both ResNet-20 and VGG-16-BN use batch normalization. Section 6.2 hypothesizes that BN enables Phi Invariance under SGD, but there are no experiments on architectures without BN (plain ResNets, Transformers with LayerNorm only) to test this hypothesis. The NoBN experiments mentioned in lessons_learned.md appear to have been conducted but are not in the paper.

### 7. Statistical Power Limitations

Three seeds provide 80% power only for effects >= 0.7%. For a paper arguing "no difference exists," this is problematic. The TOST equivalence tests confirm equivalence for only 6/12 comparisons at delta=1.0%. The other 6 comparisons are inconclusive -- neither significantly different nor provably equivalent.

### 8. CIFAR-100 Data Provenance

The lessons_learned.md reveals that 5 of 6 methods use iter_003 data while PMP-WD uses iter_006 data from a different code version. Different iterations may have different augmentation pipelines or initialization seeds. This is not disclosed in the paper. Although PMP-WD doesn't appear in the tables, if any iter mixing exists for the 7 reported methods, it must be disclosed.

## Minor Issues

- Cosine_schedule's low variance (0.07 vs 0.25-0.32) is reported without testing whether this variance reduction is statistically significant (Levene's test or Brown-Forsythe).
- The paper reports "200 epochs with batch size 128" but does not specify warmup schedule, gradient clipping, or other training details that could interact with WD.
- SWD's gradient-norm sensitivity function h() is never explicitly defined.
