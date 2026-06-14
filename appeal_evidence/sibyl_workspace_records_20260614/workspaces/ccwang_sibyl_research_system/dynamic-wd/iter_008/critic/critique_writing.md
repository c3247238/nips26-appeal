# Writing Critique

## Overall Assessment: 6/10

The writing is technically competent but undermined by serious internal inconsistencies between text, tables, and figures. The paper reads well in isolation, but cross-checking reveals contradictions that would be caught by any careful reviewer.

## Critical Writing Issues

### 1. Figure-Text Method Set Mismatch (Fatal)

Three figures (Figures 3, 8, 9) show a method "PMP-WD" that does not exist anywhere in the paper text, Table 1, or Section 4.1. Meanwhile, half_lambda and random_mask---two methods discussed extensively in the paper---are absent from these figures. This is the single most damaging writing problem: it signals to reviewers that the paper was assembled from mismatched components without quality control.

**Impact**: A reviewer who checks Figure 3 against Table 2 will immediately see different method sets and lose trust in all other claims.

### 2. Spread Contradiction (0.49pp vs 0.25pp vs 0.92pp)

Figure 3 panel (a) annotates "Spread: 0.49pp" for ResNet-20/CIFAR-10. Table 2 shows a 0.25pp spread (90.13 - 89.88). Section 5.2 states "The CIFAR-10 spread is 0.92%." Three different numbers for what should be the same quantity. The 0.92pp number is actually from SGD (Table 4), misattributed to the AdamW discussion context. The 0.49pp in Figure 3 appears to come from a different method subset (6 methods including PMP-WD instead of 7 methods from Table 2).

### 3. Undefined Symbols in Section 5.8

Section 5.8 introduces delta_t, bar_delta_T, and delta_T^sup without any prior definition. These symbols are not in Section 3 (Framework), not in Section 4 (Setup), and not defined inline. The reader encounters a correlation analysis of quantities they have never seen defined. This section reads as if it was transplanted from a different version of the paper that had a "Theorem 2" defining these terms.

### 4. Abstract-Body Scope Disconnect

Abstract claims: "7 methods, 2 optimizers, 2 architectures, 2 datasets." Body shows: 7 methods, 2 optimizers, but VGG-16-BN only under SGD with 1 dataset. The abstract implies all factor combinations were tested; the body reveals an incomplete factorial design. The Conjecture statement in the abstract also omits the "sufficiently overparameterized" and "with batch normalization" qualifiers present in the body.

## Major Writing Issues

### 5. Figure Numbering Chaos

The paper's internal figure numbering (Figures 1-9) does not match the filenames (fig1_, fig2_, fig3_, fig4_, fig5_, fig6_). Figure 3 in the paper is multi_arch_comparison.png, but fig3_bem_vs_accuracy.png is Figure 4 in the paper. Section 5.2 references Figure 6 before Figures 3-5 have been introduced. This creates forward references that break reading flow.

### 6. SWD Incomplete Specification

Table 1 defines SWD's phi as h(||g_t||) * 1, but h(.) is never given a concrete form. A framework paper that claims to "recover" existing methods must provide enough information for the reader to verify the recovery. The SWD entry is an IOU, not a definition.

### 7. Build Manifest as Paper Content

Lines 432-448 contain a "Figures and Tables" index listing filenames and descriptions. This is internal documentation, not paper content. It must be removed before submission.

## Minor Writing Issues

- Section 1.1, paragraph 3: "A critical problem pervades this rapidly growing literature"---"rapidly growing" is mild filler.
- Section 5.8: N=18 data points used for correlations without explaining derivation.
- Proposition 1 (Composition) is trivially obvious and inflates the theoretical contribution. Downgrade to inline remark.
- "Best test accuracy" vs "final test accuracy" never specified (Section 4).

## What Works Well

1. **Statistical reporting is exemplary.** Paired t-tests, Bonferroni correction, Cohen's d, TOST equivalence testing, explicit p-values. This is above community norms and is the paper's competitive advantage.
2. **Table 1 (method catalog)** is excellent---clean, informative, well-structured.
3. **Section 5.6 (weight norm analysis)** is the strongest section. The AdamW 1.2% vs SGD 97% contrast is compelling and mechanistically clear.
4. **Abstract precision.** Specific numbers (105 experiments, 7 methods, <0.5% variation) make claims evaluable from the abstract alone.
5. **Limitations section is honest.** Six limitations explicitly acknowledged, covering scale, architecture, optimizer, power, hyperparameters, and regime.
