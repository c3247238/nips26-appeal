# Critique: Related Work

## Summary Assessment
The Related Work section provides adequate coverage of prior literature on SAEs, feature absorption, and related topics. It successfully positions the paper's contribution (encoder alignment hypothesis with factorial decomposition) relative to prior work. However, the section has structural issues—it concludes with language more appropriate for an intro/contribution statement, and the prose contains several instances of vague or unsubstantiated claims that weaken its scholarly tone.

## Score: 6/10
**Justification**: The section covers the essential background and positions the work correctly. It loses points for: (1) the conclusion paragraph which oversteps the bounds of a literature review, (2) several claims without supporting evidence, and (3) missing citations for some claims (e.g., the "70% monosemantic" figure attribution is unclear). Reaching the next level requires tighter prose discipline and proper attribution throughout.

## Critical Issues

### Issue 1: Conclusion paragraph crosses literature review boundaries
- **Location**: Lines 47-49 ("Summary" subsection)
- **Quote**: "Our work differs from prior approaches in one critical way: we perform factorial decomposition to isolate encoder versus decoder contributions. Rather than comparing baseline SAEs or varying single parameters, we independently randomize encoder and decoder weights to attribute absorption to its true cause."
- **Problem**: This is an intro/method contribution statement, not a literature review conclusion. A Related Work section should describe and contextualize prior work, not argue for the paper's merits. This undermines the section's scholarly neutrality.
- **Fix**: Replace with a paragraph summarizing the state of the field after the described works—what questions remain unanswered, what gaps exist. For example: "Together, these works establish that absorption is widespread and problematic, but the mechanistic source remains contested. This paper resolves that debate through factorial decomposition."

### Issue 2: Unsubstantiated quantitative claim
- **Location**: Line 5
- **Quote**: "with 70% of features judged genuinely monosemantic by human evaluators"
- **Problem**: This specific statistic is presented without citation. The preceding sentence attributes work to "Anthropic's foundational work (2023)" but 70% is a specific number that requires a direct citation. Readers cannot verify this claim.
- **Fix**: Add citation to the specific Anthropic technical report or paper that contains this human evaluation result. If this is from the original SAE paper, it should read "(Bricken et al., 2023)" or similar with full reference in the bibliography.

## Major Issues

### Issue 3: Vague language in encoder alignment hypothesis description
- **Location**: Lines 25-26
- **Quote**: "During training, the encoder learns to align child feature directions with parent features when they co-activate, creating hierarchical representations where children can substitute for parents."
- **Problem**: The phrase "align child feature directions with parent features" is described as an empirical finding but appears in the Related Work section as if it were established fact. This is actually the paper's own contribution. The language conflates prior literature's claims with the paper's novel findings.
- **Fix**: Qualify this as a hypothesis from this work rather than established knowledge. Change to: "We hypothesize that encoder learning may align child feature directions with parent features when they co-activate, creating hierarchical representations where children can substitute for parents."

### Issue 4: Theoretical Foundations subsection lacks depth
- **Location**: Lines 43-46
- **Quote**: "Tang et al. (2025) provided theoretical grounding for sparse dictionary learning in mechanistic interpretability. Their analysis suggests that hierarchical feature structures emerge naturally from training on data with inherent compositional structure."
- **Problem**: This is extremely brief for a theoretical foundations subsection. The claim that "hierarchical feature structures emerge naturally" is stated without any explanation of the mechanism or evidence. A reader cannot assess whether this supports or complicates the paper's approach.
- **Fix**: Either expand to explain the key theoretical result (1-2 sentences on what Tang et al. actually proved) or consider removing this subsection if the connection is tangential. If retained, add what this theory predicts for absorption.

### Issue 5: Missing alternative explanation discussion
- **Location**: "Alternative Explanations" subsection (lines 18-26)
- **Problem**: Only two alternative hypotheses are discussed (decoder geometry, sparsity optimization). However, the literature may contain other proposed explanations (e.g., training dynamics, feature co-occurrence statistics). The section should acknowledge if no other explanations exist in the literature, not just present two.
- **Fix**: Add a sentence clarifying that these two represent the full set of proposed explanations, or acknowledge if additional hypotheses exist but are less prominent.

## Minor Issues

- **Line 11**: "$x$ is the original activation" — define dimensions (e.g., $x \in \mathbb{R}^d$) for precision
- **Line 31**: The formula uses "$\mathbb{E}[a_p \mid \text{no ablation}]$" but denominator should be the baseline activation, not conditional. The metric definition in method.md (line 136) uses the same formula and may have the same issue—verify this is correct.
- **Line 39**: "Basu et al. (2026)" — the citation format is inconsistent with others (uses "2026" for a future date). Confirm this is the correct year or if it should be "under review" / "preprint".
- **Line 48**: "independently randomize" — this describes methodology, not a finding. Move to method section.
- **General**: No mention of whether prior work tested interventions for absorption (mitigation strategies). This would strengthen the gap identification.

## Visual Element Assessment
- [ ] This section does not require figures/tables per the outline plan
- [ ] N/A for visual communication assessment

## What Works Well

1. **Clear taxonomic structure** (lines 18-26): The "Alternative Explanations" subsection effectively organizes the competing hypotheses into a clean three-way comparison (decoder geometry, sparsity optimization, encoder alignment). This makes the paper's positioning clear.

2. **Appropriate metric attribution** (lines 28-33): The multi-child proportional absorption metric is correctly attributed to Chanin et al. (2024) with a proper mathematical definition, and the formula is well-presented.

3. **Relevant safety literature inclusion** (lines 40-42): Including Basu et al. (2026) on the interpretability-actionability tension is appropriate for a paper with safety implications, providing necessary context for why absorption matters.