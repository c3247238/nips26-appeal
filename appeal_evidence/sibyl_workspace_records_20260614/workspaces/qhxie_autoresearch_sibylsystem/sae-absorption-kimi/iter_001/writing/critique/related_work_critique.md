# Critique: Related Work

## Summary Assessment
The Related Work section provides a competent survey of the absorption literature, organized into three coherent thematic arcs: architectural mitigations, theoretical limits, and benchmarks. The writing is generally clear and the citations are well-placed. However, the section suffers from several issues that would draw reviewer criticism at a top venue: a vague framing of the "optimistic" vs. "skeptical" camps, missing citations for key claims, an unsupported characterization of prior work, and a "Relationship to This Paper" subsection that overstates novelty without acknowledging direct methodological predecessors.

## Score: 6/10
**Justification**: The section is readable and structurally sound, but it lacks the precision expected of a strong Related Work. Major issues include unsupported claims about what prior work "does not frame" as Pareto analysis, missing citations for JumpReLU and masked regularization, and a "Relationship to This Paper" subsection that reads more like an abstract than a genuine scholarly positioning. To reach 8/10, the section needs tighter claim-evidence alignment, explicit acknowledgment of prior multi-metric comparisons, and a more nuanced discussion of how the current work builds on SAEBench rather than merely "filling a gap."

## Critical Issues

### Issue 1: Unsupported Claim About Prior Framing
- **Location**: Paragraph 3, lines 12-13
- **Quote**: "Individual papers compare architectures on absorption *and* reconstruction, but they do not frame the comparison as a Pareto analysis in which no architecture dominates across all objectives."
- **Problem**: This is a strong claim about what prior work "does not frame," but no citations are provided to substantiate it. The proposal itself notes that OrtSAE and Matryoshka SAE papers "compare each other on multiple metrics (absorption, reconstruction, downstream tasks)"—so the claim that no one frames this as Pareto analysis may be true, but it requires explicit citations (or a citation to the proposal's novelty assessment) to avoid appearing as a strawman.
- **Fix**: Either cite the specific papers being characterized (OrtSAE, Matryoshka) and quote or paraphrase their framing, or soften the claim to: "To our knowledge, no prior work explicitly frames these multi-metric comparisons as a Pareto analysis..."

### Issue 2: Missing Citations for Key Architectural Proposals
- **Location**: Paragraph 2, line 7
- **Quote**: "Additional proposals include masked regularization, time-aware feature selection, and JumpReLU thresholding."
- **Problem**: The intro and method sections mention JumpReLU, masked regularization, and ATM as specific architectures, but the Related Work section fails to cite them. This is especially problematic because the paper's own methodology includes JumpReLU and GatedSAE families in Experiment 2.
- **Fix**: Add citations. For JumpReLU, cite Rajamanoharan et al. (2024) or the relevant SAEBench paper. For masked regularization, cite Narayanaswamy et al. (2026) as done elsewhere. For time-aware feature selection (ATM), cite Li & Ren (2025).

## Major Issues

### Issue 3: "Relationship to This Paper" Overstates Novelty
- **Location": "Relationship to This Paper" subsection, lines 21-23
- **Quote**: "Our contribution is distinct from the prior work in three ways. First, we conduct the first systematic, training-free, multi-objective Pareto evaluation of absorption-mitigation methods using existing pretrained checkpoints. Second, we quantify absorption's unique causal effect on downstream interpretability utility via the largest-scale controlled meta-analysis to date (314 SAEBench checkpoints). Third, we pilot and validate a task-agnostic absorption metric, testing whether the canonical first-letter benchmark generalizes beyond its original domain."
- **Problem**: This reads like a re-statement of the abstract rather than a genuine positioning within the literature. The phrase "distinct from the prior work in three ways" implies that no prior work shares *any* of these elements, which is misleading. SAEBench itself conducts systematic multi-objective evaluation; the current work's novelty is in the *Pareto framing* and the *specific focus on absorption-hedging tradeoffs*, not in being the first to evaluate multiple objectives. Similarly, the task-agnostic metric builds directly on Chanin et al.'s causal ablation framework.
- **Fix**: Rewrite to acknowledge prior contributions explicitly before stating the incremental advance. For example: "While SAEBench provides the foundational multi-objective evaluation framework and checkpoint corpus, our work is the first to apply a Pareto-dominance lens to the absorption-hedging-reconstruction tradeoff specifically. We also build on Chanin et al.'s ablation methodology to pilot a task-agnostic variant of their metric."

### Issue 4: Vague "Optimistic" vs. "Skeptical" Framing
- **Location**: Section headings and paragraph transitions, lines 1-13
- **Problem**: The section frames the literature as an "optimistic" camp (architectural mitigations) and a "skeptical" camp (theory/limits). This is catchy but reductive. Some of the same authors appear in both camps (Chanin et al. discovered absorption *and* proved hedging limits). The framing risks implying a false dichotomy that reviewers from these labs will find inaccurate.
- **Fix**: Soften the headings and transitions. Instead of "The Skeptical Turn," use "Theoretical Limits and Tradeoffs." Acknowledge that the hedging paper (Chanin et al., 2025) comes from the same research program as the absorption discovery paper (Chanin et al., 2024), showing intellectual continuity rather than opposition.

### Issue 5: Missing Discussion of SAEBench as Direct Predecessor
- **Location**: "Benchmarks and Evaluation Frameworks" subsection, lines 15-19
- **Problem**: The paper's largest experiment (E2) is entirely built on SAEBench data, yet the Related Work section treats SAEBench as one benchmark among others rather than as the direct methodological foundation for the current work. There is no discussion of how the current work extends SAEBench's analyses.
- **Fix**: Add a paragraph or sentence explicitly stating how the current work uses SAEBench. For example: "We build directly on SAEBench's comprehensive evaluation infrastructure: our meta-analysis uses their released metrics and checkpoint metadata, but focuses specifically on isolating absorption's partial correlation with downstream utility—a causal framing not present in the original benchmark."

## Minor Issues

- **Line 7**: "Matryoshka SAEs claim roughly a 10x reduction in absorption" — the "10x" figure should be verified against the actual Matryoshka paper. If it is an approximate claim, say "report roughly an order-of-magnitude reduction."
- **Line 7**: "Li & Ren (2025) proposed adaptive temporal masking (ATM)" — the abbreviation ATM is not used elsewhere in the paper. Either drop it or use it consistently.
- **Line 11**: "Chanin et al. (2025) proved that narrower SAEs reduce absorption but increase feature hedging" — "proved" is strong. If this is a theorem, say "proved theoretically"; if it is empirical, say "showed empirically."
- **Line 11**: "Roy et al. (2026) showed 'catastrophic interpretability collapse' under aggressive sparsification" — consider adding one clause explaining what this collapse entails (e.g., "...in which learned features lose all interpretability utility").
- **Line 17**: "CE-Bench (Gulko et al., 2025) offers a lightweight contrastive alternative" — CE-Bench is not mentioned elsewhere in the paper. If it is irrelevant to the current work, consider removing it to save space and focus.
- **Line 19**: "No prior work has proposed a fully task-agnostic absorption metric using automated hierarchy discovery." — This claim is made more forcefully here than in the proposal's novelty assessment. The proposal says "No prior work has proposed a fully task-agnostic absorption metric using automated hierarchy discovery" but qualifies it with "To our knowledge." Add that qualifier here.

## Visual Element Assessment
- [x] Figures/tables match outline plan (outline plans no figures for Related Work; section complies)
- [x] All visuals referenced before appearance (N/A)
- [x] Captions are self-explanatory (N/A)
- [x] No text-heavy sections that need visual support

## What Works Well
1. **Thematic organization**: The three-part structure (mitigations → limits → benchmarks) is logical and easy to follow. It avoids the dreaded "paper-by-paper" laundry list.
2. **Strong opening paragraph**: The first paragraph clearly defines feature absorption, explains why it matters, and introduces the canonical benchmark—all in three sentences. This is efficient and reader-friendly.
3. **Effective use of citations**: Key claims are generally anchored to specific papers (Chanin et al., Korznikov et al., Bussmann et al.), giving the section scholarly weight.
