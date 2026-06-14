# Critique: Related Work

## Summary Assessment

The Related Work section provides a competent survey of the SAE literature, feature absorption research, benchmark validation, and construct validity theory. It is well-organized into thematic subsections and effectively positions the paper at the intersection of three literatures. However, it contains a critical factual error (Random-SAE score of 0.352), makes an unsupported claim about the Random SAE matching trained SAEs, and has several cross-section inconsistencies with the Introduction and Method sections. The section also misses opportunities to connect to the proposal's Goodhart's Law framing and omits key recent work on SAE utility skepticism.

## Score: 5/10

**Justification**: The section earns points for comprehensive coverage of the absorption literature, clear thematic organization, and effective positioning. It loses ground due to the critical Random-SAE data error (carried from iter_002/003), unsupported claims about metric degeneracy, a missing citation to Kantamneni et al. (2025) in the benchmarks subsection, and failure to connect to the proposal's central Goodhart's Law argument. To reach 7/10, fix the data error, add missing citations, and strengthen the benchmark-validation argument. To reach 9/10, integrate the Goodhart's Law framing, add the missing utility-skepticism literature, and reconcile terminology with the glossary.

---

## Critical Issues

### Issue 1: Factual Error — Random-SAE Score Is 0.175, Not 0.352

- **Location**: Section 2.4, paragraph 2
- **Quote**: "Our Random-SAE finding---that a control with no learned structure achieves comparable semantic-hierarchy absorption---suggests the metric may be measuring geometric properties of the probe-task interaction rather than a meaningful structural property of the SAE."
- **Problem**: The claim about "comparable semantic-hierarchy absorption" is calibrated to the iter_002/003 value where Random = 0.352 (identical to Standard). In the current iter_004 data, Random = 0.175, which is at the LOW end of the trained range (0.064--0.359). The Random SAE exceeds only PAnneal (0.064) and is meaningfully lower than 5 of 7 trained architectures. Calling this "comparable" overstates the degeneracy.
- **Fix**: Recalibrate: "Our Random-SAE finding---that a control with no learned structure achieves semantic-hierarchy absorption of 0.175, exceeding only PAnneal (0.064) and falling near the lower bound of trained architectures---suggests the metric captures some geometric artifacts but is not entirely degenerate. The first-letter task, by contrast, correctly distinguishes trained from random SAEs (Random = 0.030 vs. trained range 0.008--0.576), confirming that task-specific degeneracy varies."

### Issue 2: Unsupported Claim About "Comparable" Random-SAE Scores

- **Location**: Section 2.4, paragraph 2
- **Quote**: "Our Random-SAE finding---that a control with no learned structure achieves comparable semantic-hierarchy absorption---suggests the metric may be measuring geometric properties"
- **Problem**: With Random = 0.175 and trained range 0.064--0.359, the Random SAE is at the 34th percentile. The difference between Random (0.175) and the trained mean (0.235) is 0.06, which is not negligible. The claim that scores are "comparable" is unsupported by the actual data.
- **Fix**: Soften the claim and add nuance: "The Random SAE's score of 0.175 falls within the lower portion of the trained range, indicating partial overlap between randomized and trained structures. This suggests the metric captures both learned structure and geometric artifacts, with the latter contributing more than expected for a validated benchmark."

---

## Major Issues

### Issue 3: Missing Goodhart's Law Connection

- **Location**: Entire section
- **Problem**: The proposal frames the paper around Goodhart's Law: "the SAE community is optimizing a benchmark metric that may not measure what it claims." The Related Work section never mentions Goodhart's Law, despite having a subsection (2.4) on construct validity in ML evaluation that would be the natural place for it. The section discusses construct validity theory (Cronbach & Meehl, 1955) and proxy metric critiques (BLEU, ImageNet accuracy) but misses the opportunity to connect these to the specific Goodhart's Law argument about absorption optimization.
- **Fix**: Add a paragraph in Section 2.4: "Goodhart's Law---'when a measure becomes a target, it ceases to be a good measure'---is particularly relevant to SAE benchmark design. As absorption scores have become a primary criterion for architecture comparison (SAEBench leaderboard, paper submissions), researchers have developed architectures that minimize first-letter absorption without validating that this improvement reflects genuine feature quality. This pattern mirrors well-documented cases in NLP (BLEU optimization producing fluent but unfaithful translations) and computer vision (ImageNet accuracy failing to predict robustness to distribution shift)."

### Issue 4: Missing Citation to Wang et al. (2025)

- **Location**: Section 2.3, paragraph 3
- **Quote**: "Kantamneni et al. (2025) showed that SAEs do not consistently outperform strong non-SAE baselines on downstream sparse probing tasks... Lieberum et al. (2024) found that feature interpretability ratings from automated methods correlate weakly with human judgments."
- **Problem**: The proposal explicitly cites Wang et al. (2025) as evidence of an "interpretability-utility disconnect" (Kendall's tau_b ~ 0.298 between SAEBench interpretability and steering utility). This is a key piece of the benchmark-gaming argument but is absent from the Related Work section.
- **Fix**: Add after the Kantamneni sentence: "Wang et al. (2025) found that SAEBench interpretability scores correlate weakly with steering utility (Kendall's tau_b ~ 0.298) and that this correlation vanishes after feature selection, suggesting that benchmark metrics do not predict practical utility."

### Issue 5: Inconsistent Terminology — "Pre-trained" vs. "Pretrained"

- **Location**: Section 2.1, paragraph 2
- **Quote**: "Templeton et al. (2024) extracted millions of interpretable features from Claude 3 Sonnet, demonstrating that SAEs can operate at frontier-model scale."
- **Problem**: The glossary specifies "pre-trained" as the preferred spelling (with hyphen), but the Related Work section uses "pretrained" (no hyphen) consistently. The Introduction uses "pre-trained" in Section 1.1. This inconsistency violates the glossary.
- **Fix**: Standardize to "pre-trained" throughout the Related Work section.

### Issue 6: Section 2.5 Overstates Positioning

- **Location**: Section 2.5, paragraph 1
- **Quote**: "Our study occupies the intersection of three literatures: SAE failure-mode characterization, benchmark validation, and construct-validity assessment."
- **Problem**: The claim is accurate but the paragraph that follows understates the paper's actual contribution. The paper does not "propose a new method" (true) and does not "derive new bounds or guarantees" (true), but it also does not fully deliver on the proposal's Goodhart's Law framing or the four-experiment design. The positioning should acknowledge the narrower scope.
- **Fix**: Add: "This study focuses on a single but critical question within this intersection: whether the dominant absorption metric generalizes from its original first-letter task to semantic hierarchies. We do not address the broader utility-disconnect question (whether absorption reduction predicts downstream performance) or the metric-decomposition question (what fraction of the score is attributable to learned structure vs. geometry), leaving these for future work."

### Issue 7: Missing arXiv Citation Format Issue

- **Location**: Section 2.2, paragraph 2
- **Quote**: "theoretical work that analyzes absorption as an optimization property (arXiv:2512.05534)"
- **Problem**: The citation "arXiv:2512.05534" is used without author names or a title. This is not a standard citation format and would not pass in a NeurIPS/ICML paper. The reader cannot identify the work from this reference alone.
- **Fix**: Replace with proper author-year citation: "theoretical work that analyzes absorption as an optimization property (Smith et al., 2025)" or remove if the work is not formally cited in the bibliography.

---

## Minor Issues

- **Section 2.1, paragraph 1**: "The theoretical foundation for SAEs rests on the superposition hypothesis articulated by Elhage et al. (2022)" — The Elhage et al. paper is about superposition in neural networks broadly, not specifically about SAEs. SAEs predate the superposition paper. → "The theoretical motivation for SAEs draws on the superposition hypothesis..."

- **Section 2.1, paragraph 2**: "The central tension in this line of work is between reconstruction fidelity and interpretability." — This is a good framing, but the sentence that follows ("SAEs optimize a trade-off...") is generic and could apply to any sparse coding method. → Add specificity: "SAEs optimize this trade-off via a dictionary-learning objective with sparsity regularization, but the Pareto frontier has become the default comparison criterion despite Karvonen et al.'s (2025) finding that frontier position does not predict downstream outcomes."

- **Section 2.2, paragraph 2**: "A critical counterpoint is Chanin's (2025) finding of feature hedging" — The citation format "Chanin's (2025)" is ambiguous. Chanin et al. (2024) is the absorption paper; the hedging paper is by a different set of authors (Chanin is likely not an author on the 2025 hedging paper). → Verify authorship and correct citation.

- **Section 2.3, paragraph 1**: "SAEBench has become the dominant community benchmark, with 200+ SAEs evaluated and an interactive leaderboard at neuronpedia.org/sae-bench." — The URL should not appear in body text; move to a footnote or the availability statement. → "SAEBench has become the dominant community benchmark, with 200+ SAEs evaluated and an interactive leaderboard.\footnote{neuronpedia.org/sae-bench}"

- **Section 2.3, paragraph 2**: "Chanin et al. (2024) explicitly noted this gap, listing 'finding examples of feature absorption unrelated to character identification' as future work." — This is a strong point, but the quote is from the paper's conclusion and should be verified against the original text. → Verify exact wording or paraphrase more conservatively.

- **Section 2.4, paragraph 1**: "Construct validity---the degree to which a measurement instrument captures the theoretical construct it claims to measure---is a cornerstone of psychometric theory (Cronbach & Meehl, 1955) but is rarely applied to ML benchmarks." — Good definition and framing. The examples (BLEU, ImageNet) are well-chosen. → Consider adding a specific ML benchmark-validation paper, e.g., Raji et al. (2020) on AI benchmarking or Blum & Hopcroft (2021) on metric design.

- **Section 2.5, paragraph 2**: "If the metric lacks construct validity, these comparisons may be misleading." — "May be" is weak. The paper's findings suggest they ARE misleading for semantic hierarchies. → "If the metric lacks construct validity, these comparisons are misleading for tasks beyond the benchmark's original scope."

- **Transition sentence**: "We now describe the measurement protocol that adapts the SAEBench absorption evaluator to semantic hierarchies, non-hierarchical controls, and a randomized baseline." — This is IDENTICAL to the Introduction's final sentence. The two sections should not end with the same transition. → Vary: "The following section formalizes the adapted measurement protocol."

---

## Visual Element Assessment

- [x] Figures/tables match outline plan: The outline does not plan any figures for Related Work, and none are present. Appropriate.
- [x] All visuals referenced before appearance: N/A.
- [x] Captions are self-explanatory: N/A.
- [ ] No text-heavy sections that need visual support: Section 2.2 lists multiple architectures and their absorption reductions in prose. A small comparison table (architecture, absorption rate, trade-off) would improve readability and cross-reference with the Method section's Table 1.

---

## What Works Well

1. **Effective three-literature positioning in Section 2.5**: The paragraph clearly states what the paper is NOT (not an architecture paper, not a theory paper) and what it IS (a methodological question about metric validity). This helps reviewers categorize the contribution correctly.

2. **Strong closing argument in Section 2.5**: "Our findings do not invalidate the first-letter task as a controlled experimental setting... but they caution against treating first-letter scores as a proxy for general absorption behavior." This is a well-calibrated claim that avoids overstatement.

3. **Good coverage of the absorption-mitigation literature in Section 2.2**: The section covers Matryoshka, OrtSAE, HSAE, JumpReLU, and Gated SAEs, plus the hedging counterpoint. A reader new to the field would get a clear picture of the state of the art.

4. **Appropriate breadth in Section 2.4**: The construct-validity subsection connects SAE benchmark validation to broader ML evaluation critiques (BLEU, ImageNet) without overextending. The psychometric grounding (Cronbach & Meehl, 1955) lends credibility.
