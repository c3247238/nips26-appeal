# Writing Critique: Feature Absorption Paper

## Overall Assessment

The paper is well-structured and clearly written, with honest negative result reporting as its strongest aspect. However, critical writing errors -- including a typo that contradicts the core methodological argument and overclaiming from broken experiments -- severely undermine credibility.

---

## Critical Issues

### 1. Typo Contradicting Core Methodology (CRITICAL)

**Location**: Section 5.1, line ~350

**Text**: "Single-child ablation, which ablates the top-k children simultaneously"

**Problem**: This describes MULTI-child ablation, not single-child. The entire paper's argument is that single-child ablation saturates while multi-child ablation differentiates conditions. A typo that conflates the two undermines the reader's confidence in the authors' understanding of their own method.

**Fix**: Change to "Multi-child ablation, which ablates the top-k children simultaneously"

---

### 2. Overclaiming from Broken H3 Experiment (CRITICAL)

**Location**: Abstract, Section 4.4, Section 5.3

**Problem**: The paper draws causal/epistemic conclusions from a steering experiment where baseline equals steered mean to full double precision (37.44512171672082). The t-statistic is NaN. This is not a "negative result" -- it is a broken experiment. Yet the abstract states:

> "Steering experiments reveal that absorbed features do not respond to parent-direction interventions, suggesting absorption may reflect learned geometric structure rather than active causal interference."

And Section 5.3 concludes:

> "The complete absence of improvement suggests that absorption may instead be epistemic"

**Fix**: Remove ALL causal/epistemic claims. State: "Steering implementation failed to produce measurable changes; causal status of absorption remains undetermined."

---

### 3. Abstract Density and Overclaiming

**Location**: Abstract

**Problem**: The abstract is dense with subordinate clauses and contains unsupported claims:
- "Causal validation via steering" (contribution #2) -- H3 is broken
- "Competitive exclusion falsification" (contribution #3) -- H2 is based on 1.5% non-zero outliers

**Fix**: Rewrite abstract to reflect actual achievements. Remove contribution #2 and #3 from the abstract. Focus on H1 methodology and honest negative result reporting.

---

### 4. Figure Order Confusion

**Location**: Section 3.4, Section 4.2.2

**Problem**: Figure 5 (method architecture) is referenced in Section 3.4 BEFORE Figure 1 appears in Section 4.2.2. This creates reader confusion.

**Fix**: Add a forward-reference paragraph at the end of Methodology: "The following sections present experimental results in Figures 1-4. Figure 1 shows [X], Figure 2 shows [Y]..."

---

### 5. Banned Patterns

**Location**: Section 1 (line 374 equivalent), Section 5.4

**Problem**: "a growing body of work" appears twice. This is vague filler that should be replaced with specific citations.

**Fix**: Replace with specific paper citations.

---

### 6. Section Order

**Location**: Section 5.5 (Limitations) appears after Section 5.6 (Future Directions)

**Problem**: Unconventional ordering. Limitations should precede future directions.

**Fix**: Swap section order.

---

## Minor Issues

1. **Notation inconsistency**: `parent_dir` (underscore) in equations vs `parent_direction` in notation.md
2. **Alpha range inconsistency**: Methodology says {0.0, 0.1, 0.2} but notation.md says {0.05, 0.1, 0.15, 0.2, 0.25}
3. **Missing proportional variance definition**: Used in Table 3 but not formally defined in Methodology
4. **H3 power understatement**: "only 7 absorbed features were tested, limiting statistical power" understates the problem -- power is ~18% for d=0.5

---

## What Works Well

1. **Claim-evidence alignment**: Every quantitative claim matches source data exactly
2. **Hypothesis structure**: Clear falsification criteria and unambiguous results
3. **Transparent negative result reporting**: H2 and H3 failures are reported without spin
4. **Methodological clarity**: Saturation problem and multi-child solution are explained well

---

## Score: 5/10

The writing would score 7-8/10 on mechanics alone, but the critical typo and overclaiming from broken data drop it to 5/10. A paper that draws causal conclusions from NaN t-statistics cannot be considered rigorous regardless of how well-written the prose is.
