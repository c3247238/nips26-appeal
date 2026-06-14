# Critique: Related Work

## Summary Assessment

The related work is embedded in the Introduction (sections 1.1 and 1.2) rather than presented as a standalone section. The coverage of absorption theory, the SAEBench benchmark, and the first-letter task is accurate and well-sourced. However, the positioning relative to concurrent work has significant gaps: key architectural advances that directly address absorption (Matryoshka SAEs, OrtSAE, HSAEs) are omitted entirely, and the literature survey's identified research gaps are not reflected in the paper's framing. The transition from problem statement to the paper's specific contribution is clear, but the related work does not adequately establish why a construct-validity study is timely or necessary given existing mitigation efforts.

## Score: 5/10

**Justification**: The related work covers the core references (Chanin et al., SAEBench) accurately but misses critical concurrent work that directly targets absorption reduction. A NeurIPS reviewer would immediately ask: "Why study construct validity of a metric when Matryoshka SAEs and OrtSAE already claim to solve absorption?" The paper needs to address this tension. To reach 7+, the related work must engage with architectural mitigation papers and explain why metric validity matters even if architectures improve.

---

## Critical Issues

### Issue 1: Missing Engagement with Absorption-Mitigation Architectures

- **Location**: Section 1.1, paragraphs 1-3
- **Quote**: "Chanin et al. proved that this phenomenon is not an artifact of training dynamics but a structural consequence of the sparsity objective when features have hierarchical relationships."
- **Problem**: The related work mentions Chanin et al.'s analytical characterization of absorption but completely omits the wave of follow-up work that directly addresses it. Matryoshka SAEs (Bussmann et al., ICML 2025) report ~10x absorption reduction. OrtSAE (Korznikov et al., 2025) reduces absorption by 65% via orthogonality constraints. HSAEs introduce explicit hierarchical structure. A reviewer will ask why the paper studies metric validity without acknowledging that the field is actively trying to eliminate the problem the metric measures.
- **Fix**: Add a paragraph in 1.1 (after the Chanin et al. discussion) covering architectural mitigation efforts: Matryoshka SAEs, OrtSAE, and GBA. Frame the paper's contribution as: "While these architectures report reduced absorption scores, our study asks whether the metric itself measures what it claims---a question that must be answered before absorption-reduction claims can be interpreted."

### Issue 2: No Discussion of Feature Hedging as the Complementary Failure Mode

- **Location**: Section 1.1
- **Quote**: "Yet SAEs suffer from well-documented failure modes. Feature absorption... is among the most consequential."
- **Problem**: Chanin (2025) identified feature hedging as the opposite failure mode to absorption---where reconstruction loss drives correlated features to be represented together rather than split apart. Matryoshka SAEs trade absorption for hedging. This trade-off is directly relevant to the paper's finding that non-hierarchy pairs show higher "absorption" than hierarchies: the metric may be conflating absorption with hedging-like behavior.
- **Fix**: Add 2-3 sentences in 1.1 acknowledging feature hedging as the complementary failure mode, citing Chanin (2025). Connect to the paper's H2 finding: "Our hierarchy-specificity failure may reflect, in part, that the metric conflates absorption (sparsity-driven parent suppression) with hedging-like behavior (reconstruction-driven feature clustering)."

### Issue 3: Missing Citation for "Architecture Papers Routinely Report It as a Primary Metric"

- **Location**: Section 1.2, paragraph 2
- **Quote**: "These virtues have made first-letter absorption one of eight canonical evaluations in SAEBench, and architecture papers routinely report it as a primary metric (Bussmann et al., 2025; Rajamanoharan et al., 2024)."
- **Problem**: The Bussmann et al. (2025) citation refers to the Matryoshka SAE paper, which does report absorption---but Rajamanoharan et al. (2024) refers to the JumpReLU paper, which does NOT primarily focus on absorption reduction. The JumpReLU paper's main contribution is reconstruction fidelity and preventing dead features. Citing it as evidence that "architecture papers routinely report absorption as a primary metric" is misleading.
- **Fix**: Replace the Rajamanoharan et al. (2024) citation with a more appropriate reference. The Matryoshka SAE paper (Bussmann et al., 2025) and the OrtSAE paper (Korznikov et al., 2025) are better examples. Alternatively, cite SAEBench leaderboard papers or the SAEBench paper itself for the claim about routine reporting.

---

## Major Issues

### Issue 4: Gap Between Literature Survey and Paper Framing

- **Location**: Section 1.2, paragraph 3
- **Quote**: "Chanin et al. (2024) themselves noted that 'finding examples of feature absorption unrelated to character identification' remains open future work."
- **Problem**: The literature survey (context/literature.md) identified 10 research gaps, including Gap 2 (absorption metric construct validity) and Gap 7 (interaction between absorption and other failure modes). The paper cites only one of these gaps (Chanin et al.'s future work statement) and does not frame its contribution against the broader landscape of open problems. This makes the paper's motivation seem narrower than it is.
- **Fix**: Expand the framing in 1.2 to position the construct-validity question as one of several critical open problems. Add: "Beyond the specific question of semantic generalization, our study addresses a broader methodological concern: whether proxy metrics in SAE benchmarks faithfully measure the theoretical constructs they claim to capture (cf. Kantamneni et al., 2025, who showed SAEs often fail to outperform baselines on downstream tasks)."

### Issue 5: Missing Context on SAEBench's Broader Evaluation Framework

- **Location**: Section 1.2, paragraph 1
- **Quote**: "SAEBench (Karvonen et al., 2025) provided one, incorporating an absorption evaluator based on Chanin et al.'s first-letter classification task."
- **Problem**: The paper treats SAEBench as a container for the absorption metric without explaining why absorption was selected as one of eight canonical evaluations. The SAEBench paper explicitly organizes evaluations by capability (concept detection, interpretability, reconstruction, feature disentanglement). Absorption is categorized under "concept detection." This context matters because it shows absorption is meant to measure a specific capability, not serve as a general SAE quality score.
- **Fix**: Add one sentence: "In SAEBench's capability taxonomy, absorption falls under concept detection---the ability of SAE latents to preserve distinguishable concepts---making construct validity essential to its interpretability."

### Issue 6: No Mention of Concurrent Construct-Validity Concerns in the Broader SAE Literature

- **Location**: Section 1.2
- **Problem**: The paper frames construct validity as a novel concern for the absorption metric specifically, but broader concerns about SAE evaluation validity exist. Kantamneni et al. (2025) showed SAEs do not consistently outperform linear probes on downstream tasks. The "Are SAEs Useful?" paper (arXiv:2502.16681) questions whether SAE features are actually useful for interpretation. The "Feature Sensitivity" paper (arXiv:2509.23717) found many "interpretable" features have poor sensitivity. These papers establish that SAE evaluation metrics are under scrutiny broadly, not just absorption.
- **Fix**: Add a sentence in 1.2: "Our focus on absorption joins a growing body of work questioning whether SAE evaluation metrics capture meaningful interpretability (Kantamneni et al., 2025; arXiv:2509.23717)."

---

## Minor Issues

- **Section 1.1, paragraph 1**: "Sparse autoencoders (SAEs) have become the dominant approach" -- citation needed for "dominant approach." The claim is plausible but should be supported by a survey or benchmark adoption metric. Consider citing the SAE survey (arXiv:2503.05613) or SAEBench adoption statistics.

- **Section 1.1, paragraph 2**: "Chanin et al. proved that this phenomenon is not an artifact of training dynamics" -- the proof is for a specific toy model with L1 sparsity and hierarchical features. The generalization to all SAEs is overstated. Suggest: "Chanin et al. proved that absorption arises structurally from the sparsity objective for hierarchical features in their analytical model."

- **Section 1.2, paragraph 2**: "The first-letter benchmark has genuine strengths" -- the transition from critique to praise is abrupt. A brief bridge sentence would improve flow: "Despite its limitations, the benchmark has virtues that explain its adoption."

- **Section 1.2, paragraph 3**: "Without such a test, architecture comparisons that rank SAEs by first-letter absorption may be optimizing for a metric that does not reflect behavior on real conceptual hierarchies." -- This is the core motivation sentence. It is strong but could be sharpened: "Without validation, absorption-reduction claims---such as Matryoshka SAEs' reported 10x improvement---may reflect task-specific optimization rather than general hierarchical feature preservation."

- **Inconsistent citation format**: The outline references "Bussmann et al. (2025)" and "Rajamanoharan et al. (2024)" but the intro also uses "Chanin et al. (2024)" and "Karvonen et al. (2025)." Verify all citations have corresponding entries in the references section. The outline mentions "Korznikov et al. (2025)" and "Zhan et al. (2026)" but these are not cited in the intro's related work discussion.

---

## Visual Element Assessment

- [ ] Figures/tables match outline plan -- N/A for related work (integrated into intro)
- [x] All visuals referenced before appearance -- N/A
- [x] Captions are self-explanatory -- N/A
- [ ] No text-heavy sections that need visual support -- The related work discussion is entirely text. A small table summarizing the timeline of absorption research (Chanin 2024 -> SAEBench 2025 -> mitigation architectures 2025 -> this work) would help readers place the paper in context.

---

## What Works Well

1. **The Chanin et al. limitation quote is well-deployed.** Paragraph 3 of 1.2 directly quotes Chanin et al.'s acknowledgment that "finding examples of feature absorption unrelated to character identification" is future work. This is an excellent rhetorical move: it shows the original authors recognized the limitation, making the paper's contribution seem like natural next step rather than a critique of their work.

2. **The problem-solution structure is clear.** Section 1.2 moves cleanly from benchmark strengths (paragraph 2) to limitations (paragraph 3) to the paper's response (1.3). A reader can follow the logic: benchmark exists -> has a known gap -> this paper fills it.

3. **The theoretical-to-practical arc in 1.1 is effective.** The paragraph progresses from SAEs' promise (monosemanticity at scale) to absorption as a structural failure mode to the practical consequence (downstream interpretability tools miss high-level concepts). This gives readers both the technical and motivational context.
