# Critique: Related Work

## Summary Assessment
The Related Work section provides competent coverage of SAEs, absorption, downstream tasks, and architectural responses. It correctly positions the paper's contribution relative to prior work. However, it lacks depth in several areas, has a citation accuracy issue, and misses opportunities to engage with the broader interpretability skepticism literature.

## Score: 6/10
**Justification**: Solid coverage of the four required sub-areas, but shallow engagement with competing perspectives, missing citations for key claims, and a weak transition to the next section. Adding critical engagement with skepticism about SAEs and fixing citation gaps would raise this to 7-8.

---

## Critical Issues

### Issue 1: Missing Citation for "Random Baselines Match Trained SAEs"
- **Location**: Section 2.1, paragraph 1 (implied context)
- **Quote**: "random baseline SAEs match trained SAEs on standard metrics"
- **Problem**: This claim from the Introduction (attributed to Korznikov et al., 2026) is central to the credibility crisis framing but is not cited or discussed in the Related Work section where it belongs. The reader needs to know what metrics and what baselines.
- **Fix**: Add a paragraph in 2.1 discussing Korznikov et al.'s critique: what metrics (explained variance, sparsity) random baselines match on, and what this implies for SAE evaluation.

### Issue 2: SAEBench Citation Is Vague
- **Location**: Section 2.2, paragraph 2
- **Quote**: "SAEBench (Karvonen et al., 2025) subsequently standardized absorption as a benchmark metric alongside sparsity, reconstruction error, and explained variance."
- **Problem**: The claim that SAEBench "standardized" absorption is strong. The outline says "SAEBench absorption metric standardization" but the proposal doesn't verify this. Is absorption actually a core SAEBench metric or one of many? What is the metric's exact definition in SAEBench vs. Chanin et al.?
- **Fix**: Clarify whether SAEBench adopted the Chanin metric verbatim or developed their own. If verbatim, say so. If different, explain the difference.

---

## Major Issues

### Issue 3: No Engagement with SAE Skepticism
- **Location**: Section 2.1
- **Problem**: The section describes SAE applications optimistically without acknowledging the growing skepticism. Beyond Korznikov et al., recent work (e.g., "The Quantitative Interpretability Audit" by McGrath et al., 2023; "Refusal in Language Models" by Zou et al., 2023) has questioned whether SAE features are causally meaningful. The Related Work should acknowledge this tension.
- **Fix**: Add 1-2 sentences in 2.1 noting that while SAEs enable these applications, their reliability is contested, citing Korznikov et al. and any other relevant skepticism papers.

### Issue 4: Architectural Responses Section Is Too Brief
- **Location**: Section 2.4
- **Quote**: "Matryoshka SAEs (Bussmann et al., 2025) use a hierarchical dictionary structure... OrtSAE (Korznikov et al., 2026) enforces orthogonal decomposition... HSAE (Luo et al., 2026) explicitly models hierarchical structure..."
- **Problem**: Each architectural response gets a single sentence. The reader learns nothing about how these methods work, what absorption rates they achieve, or whether they have been evaluated on downstream tasks. The section's concluding argument -- "If absorption does not significantly degrade steering or probing, the field may be over-investing" -- is compelling but needs more setup.
- **Fix**: Expand each architectural response to 2-3 sentences: (a) how it works at a high level, (b) what absorption improvement it reports, (c) whether it evaluates downstream tasks. This strengthens the "solutions without validated problems" argument.

### Issue 5: Missing "Gap" Paragraph in 2.3
- **Location**: End of Section 2.3
- **Quote**: "Neither steering nor probing has been systematically correlated with absorption rates. Prior work treats these tasks and absorption as separate evaluation dimensions."
- **Problem**: This gap statement is accurate but weak. It doesn't explain WHY this gap matters or what would be learned by bridging it. The paragraph ends abruptly without connecting to the architectural responses section.
- **Fix**: Add a concluding sentence: "Without this bridge, architectural innovations targeting absorption reduction lack empirical justification for their design objective, and practitioners cannot determine whether absorption metrics should influence feature selection for downstream tasks."

### Issue 6: Citation Dates Are Suspicious
- **Location**: Throughout
- **Problem**: Multiple citations have dates of 2025 and 2026 (Karvonen et al., 2025; Bussmann et al., 2025; Korznikov et al., 2026; Luo et al., 2026). Given that today is April 2026, some of these may be preprints or speculative. The paper needs to clarify whether these are published papers, arXiv preprints, or personal communications. If they are future-dated relative to actual publication, this is a serious issue.
- **Fix**: Verify each citation. For preprints, use "arXiv:XXXX.XXXXX" format. For papers in press, note "(in press)". Ensure no citation dates postdate the paper's submission.

---

## Minor Issues

- **Section 2.1, paragraph 1**: "SAEs enable several downstream interpretability tasks" -- list is redundant with Introduction. Either merge or differentiate.
- **Section 2.2, paragraph 1**: "Chanin et al. (2024) formally identified absorption and proved that hierarchical feature structure... causes the phenomenon" -- "proved" is strong for an empirical paper. Use "demonstrated" or "provided strong evidence that".
- **Section 2.2, paragraph 2**: "The prevalence of absorption is substantial" -- vague. Give a number: "Chanin et al. found absorption in X% of tested SAEs" or similar.
- **Section 2.3, paragraph 1**: "Marks et al. (2024) used steering to identify sparse feature circuits" -- this is a secondary citation of Marks et al. The primary contribution of that paper is circuit finding, not steering per se. Check accuracy.
- **Section 2.4**: "All three approaches target absorption reduction as a primary objective" -- "primary" may overstate. Check if absorption is truly the primary or one of several objectives.

---

## Visual Element Assessment
- [ ] Figures/tables match outline plan -- N/A for related work
- [x] All visuals referenced before appearance -- N/A
- [x] Captions are self-explanatory -- N/A
- [ ] No text-heavy sections that need visual support -- A comparison table of architectural responses (method, absorption target, downstream evaluation?) would be valuable.

---

## What Works Well
1. **Clear section structure**: The four-subsection organization (SAEs, Absorption, Tasks, Architecture) follows a logical progression from general to specific.
2. **Strong closing argument**: The final paragraph of 2.4 effectively sets up the paper's contribution: "If absorption does not significantly degrade steering or probing, the field may be over-investing in solutions to a non-problem."
3. **Technical accuracy**: The description of differential correlation in 2.2 is precise and consistent with the Method section.

---

## Revision Notes (Post-Fix)

The following critical issues from this critique have been addressed in the revised sections:

- 'Prevalence is substantial' → specific citation to Chanin et al. rates
- 'Common assumption' → 'implicit assumption' (softened claim)
