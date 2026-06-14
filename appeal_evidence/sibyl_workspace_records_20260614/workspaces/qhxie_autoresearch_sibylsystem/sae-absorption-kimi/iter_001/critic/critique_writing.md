# Writing Critique

## Overall Assessment

The manuscript is structurally sound and the prose is generally clear, but it suffers from three major writing flaws that would draw negative reviewer attention at a top venue: (1) a missing forward reference for Figure 4, (2) unlabeled inline tables instead of proper LaTeX tables, and (3) terminology drift that undermines precision. The paper also overreaches with causal language in an observational study. Most critically, the paper draft has NOT been updated to reflect the newly completed E2 full experiments, creating severe internal contradictions.

## Critical Issues (New Since Last Round)

### H2 Framing Is Now Factually Wrong
The abstract and introduction still claim the "first-letter benchmark may be degenerate" based on the E1/E3 simplified proxy. However, the completed E2 full GPT-2 and Pythia experiments show the *official* sae-spelling metric has robust variance (GPT-2: 0.04-0.67; Pythia: 0.007-0.579). The paper MUST distinguish between:
- The degenerate **simplified proxy** used in E1 and the E3 pilot
- The valid **official sae-spelling metric** used in E2 full

Failure to make this distinction creates a severe internal contradiction and undermines credibility.

## Major Issues

### 1. Missing Forward Reference for Figure 4
**Location:** Section 7.1 (Discussion)  
**Problem:** Figure 4 is introduced for the first time in the Discussion, with no prior mention in Experiment 2 where the underlying regression results are presented.  
**Fix:** Add a forward reference at the end of Experiment 2: "Figure 4 visualizes the standardized regression coefficients and partial-correlation scatter for these relationships."

### 2. Inline Tables Lack Labels and Captions
**Location:** Sections 4, 5, and 6  
**Problem:** Tables 1-3 are rendered as inline markdown tables without LaTeX `\label{}` or `\caption{}`. The outline explicitly planned labeled LaTeX tables for a NeurIPS-style submission.  
**Fix:** Convert all three tables into proper `table` environments with concise captions. Ensure each is referenced in the text before it appears.

### 3. Causal Language Overreaches the Design
**Location:** Abstract, Section 5 title, Discussion  
**Problem:** The paper repeatedly uses "causal cost," "unique causal effect," and "causal disentanglement" to describe Experiment 2, which is an observational meta-analysis. Reviewers at causally-minded venues will flag this.  
**Fix:** Replace "causal" with "predictive" or "associational" in titles and abstract. Keep the caveats, but do not lead with causal claims.

### 4. Terminology Inconsistencies
**Observed drift:**
- "feature-splitting" (hyphen) vs. "feature_splitting" (underscore)
- "CE recovered" vs. "CE loss recovered" vs. "$\text{CE}_{\text{recovered}}$"
- "JumpReLU" vs. "JumpRelu"
- "TopK_MLP" and "TopK_Attn" appear in E3 but are not defined in the glossary

**Fix:** Do a global alignment pass using `glossary.md` and `notation.md` as canonical sources.

## Minor Issues

1. **Filler phrase:** "It is important to note that..." appears in E1. Rewrite as a direct statement.
2. **Repetitive transitions:** "These results support H1/H2/H3" is formulaic. Vary the phrasing.
3. **Abstract scope ambiguity:** The abstract says E1 evaluates "Standard, TopK, and feature-splitting families" without noting the GPT-2 Small limitation.
4. **Superlative hedging:** "The largest-scale evidence to date" edges close to unverifiable hype. Prefer "the largest-scale meta-analysis of which we are aware."
