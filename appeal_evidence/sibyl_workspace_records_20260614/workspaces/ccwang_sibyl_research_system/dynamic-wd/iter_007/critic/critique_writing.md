# Writing Critique

## Overall Assessment

The paper is well-organized with clear section structure and readable prose. The statistical reporting is exemplary -- paired t-tests, Bonferroni correction, TOST equivalence testing, Cohen's d, and explicit power analysis set a high standard for null-result studies. However, several serious writing issues undermine credibility.

## Critical Issues

### 1. Abstract Misrepresents Experimental Design

The abstract states "7 methods, 2 optimizers, 2 architectures, 2 datasets, and 3 seeds" implying a full factorial design (336 experiments). The actual design is partial: VGG-16-BN only runs SGD on CIFAR-10 (21 runs). The 105 count is correct but the enumeration is misleading. Fix: explicitly decompose the 105 into its two factorial blocks.

### 2. Paper-Figure Method Set Mismatch

Tables 2-5 describe 7 methods: {constant, cwd_hard, swd, cosine_schedule, random_mask, half_lambda, no_wd}. Figures 3 and 9 include **PMP-WD**, which is never defined in Section 4.1, never appears in Table 1, and has no phi expression. This is a glaring inconsistency that signals either (a) figures were generated from a different data source than the tables, or (b) PMP-WD was removed from the text but not the figures. Either way, a reviewer will notice immediately.

### 3. Section 5.7 Text Contradicts Its Own Figure

The text claims "the band narrows rapidly during training, converging to a tight interval by epoch 50" and "this narrowing explains Phi Invariance." The certified band figure shows chaotic oscillating lines with no visible narrowing. The visual evidence directly contradicts the textual claim. This is the most dangerous writing issue -- it looks like the authors did not check their own figure.

### 4. Dual Identity Crisis

The paper oscillates between "unified framework contribution" (Sections 1, 3) and "null result paper" (Sections 5, 6). The introduction promises four contributions but the core finding is negative. Recommendation: lead with the empirical finding, present the framework as infrastructure that enabled the fair comparison.

## Major Issues

### 5. Theorem 2 Overclaiming

Section 5.8 says the cumulative measure shows "moderate negative correlation" (rho=-0.379, p=0.121). At p=0.121, this is not statistically significant. The phrase "directionally consistent with the theoretical improvement" is spin language. Honest phrasing: "a non-significant trend suggestive of, but insufficient to confirm, the theoretical prediction."

### 6. Missing Proofs

A paper with Theorem 1, Theorem 2, and a Lyapunov stability analysis that provides zero proofs -- not even in an appendix -- will be desk-rejected at NeurIPS/ICML. This is the 4th consecutive iteration where this has been flagged and not addressed.

### 7. Conjecture vs. Theorem Confusion

The paper numbers Conjecture 1 (Section 6.1) but earlier mentions "Theorem 1" and "Theorem 2" in the context of Lyapunov stability and cumulative alignment bounds. The reader cannot tell which claims are proven, which are conjectured, and which are empirical observations. Clear labeling is needed.

## Minor Issues

- Figure file names don't match figure numbers (fig3_bem_vs_accuracy.png is Figure 4, fig4_diagnostic_heatmap.png is Figure 5).
- The cosine_schedule low-variance observation (sigma=0.07) is highlighted without a significance test for variance differences.
- References to "PMP-WD" appear in figures but not in the bibliography or method descriptions.
- Table 5 column header says "Accuracy (%)" without specifying dataset (CIFAR-10 is only in the table caption).
