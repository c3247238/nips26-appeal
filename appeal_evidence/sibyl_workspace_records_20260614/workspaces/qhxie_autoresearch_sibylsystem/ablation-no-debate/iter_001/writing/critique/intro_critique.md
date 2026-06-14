# Critique: Introduction

## Summary Assessment

The Introduction successfully establishes the research problem, measurement crisis, and contributions with strong evidence-based writing. The three research questions are clearly articulated, and the contribution claims are well-grounded in specific numbers. The main weakness is the framing of the safety question (RQ3) as a contribution when it was not actually tested, creating a promise-reward gap for readers.

## Score: 8/10
**Justification**: The writing quality is high -- concrete, specific, evidence-driven with no banned patterns. Logical flow from problem through measurement crisis to research questions is strong. The score would reach 9+ with resolution of the dangling safety contribution and explicit visual references where helpful.

## Critical Issues

### Issue 1: Dangling Safety Contribution (RQ3 Never Tested)
- **Location**: Lines 14-15 and Contribution #3 (lines 27-28)
- **Quote**: "Do safety-critical features exhibit elevated absorption?...Safety-critical features: Evidence on whether SAE-based safety analysis is unreliable for the most important cases"
- **Problem**: The paper poses three research questions but only tests two (H1, H2, H3). H_Safe was not implemented. Contribution #3 promises "evidence on whether safety-critical features are disproportionately absorbed" but this evidence does not exist in the paper. The Discussion (Section 7.4) explicitly acknowledges: "Our H_Safe hypothesis was not implemented in this study due to time constraints."
- **Fix**: Remove safety-critical features from the list of research questions (or mark it as future work), or remove Contribution #3. Alternatively, reframe as a motivation-only question that remains open. The intro currently implies this will be answered; it is not.

## Major Issues

### Issue 1: "Recovered" vs "Identified" Features
- **Location**: Line 3
- **Quote**: "...researchers have recovered thousands of monosemantic features..."
- **Problem**: "Recovered" implies the features pre-existed and were extracted. SAEs learn/discover features from data; they do not recover pre-existing ground-truth features (unless on synthetic data). This is imprecise language for a technical paper.
- **Fix**: Change to "identified" or "discovered" or "learned." The Bricken et al. (2023) reference uses "extracted" which is also slightly misleading; "identified" is safer.

### Issue 2: Competitive Exclusion Undefined
- **Location**: Line 19, Contribution #3
- **Quote**: "...contradicting the competitive exclusion hypothesis..."
- **Problem**: Competitive exclusion (ecological principle) is mentioned as a hypothesis but never defined in the Introduction. Readers unfamiliar with the ecological analogy will not understand what is being falsified. The Discussion (7.2) eventually defines it, but the Introduction uses the term as if the reader already knows it.
- **Fix**: Add one sentence after line 19 defining competitive exclusion: "Ecological competitive exclusion predicts that features competing for inclusion in the sparse representation should show inversely correlated frequency and absorption."

## Minor Issues

- **Line 5**: "child features (children)" and "parent features (parents)" -- parentheses with synonyms are redundant. Either use the term or the parenthetical, not both.
- **Line 17**: "absorption rates of 0.50 compared to 0.059" -- the absorption rate for trained SAEs (0.50) is higher than random (0.059). But in the ablation framework, lower residual = more absorption. The text says "trained SAEs exhibit absorption rates of 0.50" which is correct per the paper's definition, but the framing could confuse readers who expect higher = worse. Consider clarifying: "meaning the parent is still 50% active after ablating children."
- **Line 27-28**: "Safety-critical features: Evidence on whether SAE-based safety analysis is unreliable" -- this is framed as a contribution but no evidence exists. See Critical Issue #1.

## Visual Element Assessment
- [ ] Figures/tables match outline plan -- The outline says no figures in Intro (line 32: "<!-- FIGURES - None -->"). This is correct and consistent.
- [ ] All visuals referenced before appearance -- N/A (no visuals in intro)
- [ ] Captions are self-explanatory -- N/A
- [ ] No text-heavy sections that need visual support -- Acceptable for a methodologically-focused intro. A simple synthetic hierarchy diagram could aid intuition, but the outline plan correctly lists no intro figures.

## Cross-Section Consistency

### Checked Items:
1. **Method preview accuracy**: The intro's description of multi-child proportional ablation (lines 16-19) accurately previews the method section. The key insight (ablating top-k children tests collective substitution) matches Section 5.4.

2. **Result numbers consistency**: All numbers cited in the intro match the experiments section exactly:
   - Absorption 0.50 vs 0.059 (intro line 17) matches H1 Table 1
   - Cohen's d = 8.94, p < 10^-133 (line 17) matches H1 Table 2
   - Spearman rho = 0.17, p < 10^-7 (line 19) matches H2 Table 3 (6.47e-08 rounds to ~10^-7)
   - Steering improvement 0.0 (line 19) matches H3 Table 4

3. **Terminology consistency**: All terms match glossary.md and method.md:
   - "multi-child proportional ablation" -- consistent
   - "feature absorption" -- consistent
   - "SAE" -- consistent
   - "absorption rate" -- consistent (decimal format per glossary)

4. **Notation consistency**: The intro does not introduce formal notation, which is appropriate for a high-level overview. The method section contains all formal definitions.

5. **Citation format**: "Bricken et al., 2023; Templeton et al., 2024; Chanin et al., 2024; Korznikov et al., 2026" -- consistent with References section and with method/background sections.

## What Works Well

1. **Lines 5-8**: The measurement crisis paragraph is exemplary. It clearly explains the problem (single-feature ablation saturates at 1.0 for both SAE and baseline), why this matters (raises question of genuine vs. artifact), and sets up the need for the paper's contribution. Specific, evidence-driven, no filler.

2. **Lines 9-16**: The three research questions are crisp, specific, and directly tied to the measurement crisis. RQ1 addresses the measurement problem, RQ2 addresses causation, RQ3 addresses safety implications. The structure signals a well-thought-out research agenda.

3. **Lines 21-29**: The four contributions are well-structured and falsifiable. Contribution #4 ("Negative results") is particularly valuable framing that previews the paper's scientific honesty. The language "enabling the field to move past unsuccessful approaches" is specific and meaningful.

4. **Evidence density**: Every claim in the introduction is backed by specific numbers or citations. No vague superlatives. This is a model for empirical paper introductions.
