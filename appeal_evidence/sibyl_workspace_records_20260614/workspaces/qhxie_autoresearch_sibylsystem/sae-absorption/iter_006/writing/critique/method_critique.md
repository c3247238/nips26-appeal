# Critique: Methodology (Section 3)

## Summary Assessment

The Methodology section is well-organized across six subsections that cover models, measurement protocol, controls, confound decomposition, CMI estimation, and cross-domain setup. The configurations table is clear and the control suite is a genuine strength. However, there are several internal inconsistencies between the methodology as described and how it is actually executed in Sections 4-6, the confound decomposition definitions are inverted relative to `notation.md` and `glossary.md`, and the section omits key procedural details (tokenization pipeline, probe training specifics, CMI sample sizes) that a reviewer would need to assess reproducibility.

## Score: 6/10
**Justification**: The section provides a functional experimental blueprint with a strong control suite and clear table, but the hedging/hierarchy-driven definition inversion is a critical internal inconsistency that propagates into the results sections. Missing procedural detail (how probes are trained, what optimizer, how vocabulary is tokenized) and multiple numerical discrepancies with the experiments section keep this below a 7. Fixing the definition inversion, adding the missing procedural detail, and reconciling all numbers with Section 4-5 would bring this to an 8.

## Critical Issues

### Issue 1: Confound decomposition definitions are inverted relative to notation.md and glossary.md

- **Location**: Section 3.4, paragraph 2 (definitions of hedging and hierarchy-driven)
- **Quote (method section)**: "**Hedging.** The token is a false negative at the current L0 but recovers at a higher L0... **Hierarchy-driven absorption.** The token is a false negative at all four tested L0 values..."
- **Quote (notation.md, Section 4)**: "**Hedging**: False-negative mechanism where the parent feature's information is spread across many latents, none clearing the activation threshold; **persists across all L0 values**" and "**Hierarchy-driven absorption**: False-negative mechanism where the child latent actively suppresses the parent; **appears only at low L0 and recovers at high L0**"
- **Quote (glossary.md)**: "**hedging**: ...Distinguished from absorption by **persistence across all L0 values**." and "**hierarchy-driven absorption / competitive exclusion**: ...Distinguished from hedging by **appearing only at low L0 values and recovering at higher L0**."
- **Problem**: The method section's operational definitions assign the labels in the opposite direction from both `notation.md` and `glossary.md`. In the method section, "hedging" = recovers at higher L0 (i.e., is transient), and "hierarchy-driven" = persists across all L0 values. In `notation.md` and `glossary.md`, "hedging" = persists across all L0 values, and "hierarchy-driven" = recovers at higher L0. The experiments section (4.2) follows the method section's convention (hedging = 98.6% at L0=22, described as "tokens whose parent information is spread across many latents... which resolve at higher L0"). This is a fundamental contradiction in the paper's core terminology that a reviewer will flag as confusion about the paper's central claim.
- **Fix**: Align the method section, notation.md, and glossary.md to a single consistent definition. The method section's operational convention (hedging = resolves at higher L0, hierarchy-driven = persists) appears to be the one actually used in the experiments. Update `notation.md` and `glossary.md` to match. Alternatively, if the notation/glossary definitions are intended, the method and experiments sections must be rewritten.

### Issue 2: Vocabulary count inconsistencies between method and experiments sections

- **Location**: Section 3.2 (vocabulary) and Section 3.4 (confound decomposition)
- **Quote (3.2)**: "we construct a vocabulary of 1,204 single-token alphabetic words... The confound decomposition (Section 3.4) uses a separately tokenized vocabulary of 1,196 words (1,195 tested after excluding X)."
- **Problem**: The method section says 1,195 tested words for confound decomposition, but Section 4.2 (experiments) reports "657 false negatives" out of an unspecified total at L0=22, and Section 5.1 reports "14.39%, measured on 1,195 words with F1=1.0 probes" at L0=82. Meanwhile, the method says "1,203 tested words at L0=82" for the main first-letter experiment. The reason for having two different vocabulary sizes (1,204 vs. 1,196) with two different tested counts (1,203 vs. 1,195) is not explained. A reviewer will ask: why does the confound decomposition use a different vocabulary? Why "separately tokenized"?
- **Fix**: Add one sentence explaining why the confound decomposition vocabulary differs (e.g., different tokenization pipeline, different date, model-specific tokenizer differences). Also explicitly state that the "1,195 tested" count at L0=82 in Section 5.1 corresponds to the confound decomposition vocabulary, not the 1,203-word improved first-letter vocabulary.

## Major Issues

### Issue 3: Missing probe training details critical for reproducibility

- **Location**: Section 3.2, "Probe training" paragraph
- **Quote**: "k-sparse logistic regression probes (k = 5) are trained per parent class on SAE latent activations, following the Chanin et al. protocol."
- **Problem**: This single sentence defers entirely to Chanin et al. for the training procedure. A reviewer needs to know: What optimizer? What regularization strength? How is k-sparsity enforced (L0 constraint, iterative pruning, or feature selection)? What is the train/test split? How many training tokens per class? Is the probe trained per-layer or transferred? The phrase "k-sparse logistic regression" is not standard nomenclature -- does it mean the top-k features by weight magnitude are retained, or that the probe uses only k SAE latents selected a priori?
- **Fix**: Add 2-3 sentences specifying: (a) what "k-sparse" means operationally (e.g., "the probe selects the k = 5 SAE latents with highest decoder cosine similarity to the probe direction"), (b) optimizer and hyperparameters, (c) train/test partition. This is essential for a methodology section.

### Issue 4: Absorption criterion description is ambiguous

- **Location**: Section 3.2, "Absorption criterion" paragraph
- **Quote**: "A token is classified as absorbed when (1) all k probe-associated latents fail to activate (z_j = 0) and (2) the probe correctly classifies the token. Among these false negatives, absorption is confirmed when the highest-activation latent has magnitude ratio >= 1.0 relative to the second-highest."
- **Problem**: Condition (2) says "the probe correctly classifies the token," but this seems to mean the probe classifies the token as belonging to the parent class. This should be stated explicitly. The final sentence introduces an additional filter ("absorption is confirmed when the highest-activation latent has magnitude ratio >= 1.0") that creates a two-stage classification: first "classified as absorbed," then "absorption is confirmed." It is unclear whether the absorption rates reported throughout the paper (e.g., 15.96%) use only conditions (1)+(2) or also require the magnitude ratio condition. The outline in Section 3.3 lists C1's expected outcome as "<2%" but the method reports C1 observed = 11.8%, so the thresholds are clearly important.
- **Fix**: Clarify whether reported absorption rates require both the false-negative condition AND the magnitude ratio filter. If the magnitude ratio is a secondary filter, state explicitly which results use which definition.

### Issue 5: CMI estimation sample sizes are undisclosed in the method section

- **Location**: Section 3.5
- **Quote**: "we collect activations from the word vocabulary plus 10,000 corpus tokens"
- **Problem**: Section 6.1 (experiments) reports "1,092 word activations plus 2,599 corpus token activations (3,691 total samples)" -- substantially fewer than the "10,000 corpus tokens" stated in the method. The method section either describes an intended design that was not implemented, or the experiments section reports a subset. Either way, the discrepancy undermines reproducibility. A 3,691-sample k-NN CMI estimate in d'=10 dimensions is feasible but a reviewer may question whether 10,000 corpus tokens (as described) vs. 2,599 (as executed) would change the results.
- **Fix**: Reconcile the method and experiments sections. If 2,599 corpus tokens were actually used, update the method to state "approximately 2,500-3,000 corpus tokens" and briefly explain the sampling procedure. If 10,000 were intended but a subset was used, explain why.

### Issue 6: C4 control described inconsistently

- **Location**: Section 3.3 (C4 description) vs. Section 4.1 (C4 results)
- **Quote (3.3)**: "C4 (Untrained SAE): Random encoder/decoder of the same dimensions yields 0.0% absorption"
- **Quote (4.1)**: "The untrained SAE control (C4) produces 0.0% absorption with mean probe F1 = 0.943"
- **Problem**: The method section says "random encoder/decoder," but Section 4.1 reports probe F1 = 0.943 on this untrained SAE. A random encoder/decoder should not yield F1 = 0.943 on any meaningful classification task. This suggests C4 is not truly random but rather an SAE with randomized weights that still preserves some structure (e.g., random orthogonal initialization). The high probe F1 needs explanation -- if the probes can achieve F1 = 0.943 on random SAE latents, this raises questions about what the probes are learning.
- **Fix**: Clarify what "untrained SAE" means operationally (random Gaussian weights? Xavier initialization? Untrained but structured?). Explain why probe F1 is so high (0.943) on random SAE activations -- this is itself a noteworthy result that deserves a sentence of explanation.

## Minor Issues

- **Section 3.1, paragraph 2**: "Lieberum et al., 2024" is cited for Gemma Scope, but the related work section cites "Rajamanoharan et al., 2024" for JumpReLU SAEs. The method section should cite Rajamanoharan et al. for the JumpReLU architecture description and Lieberum et al. for Gemma Scope specifically, as the related work does. Currently only Lieberum appears.
- **Section 3.2, vocabulary sentence**: "22 letters with >= 20 words" -- the inequality symbol should be rendered consistently as "$\geq$" in LaTeX. The raw ">=" looks like a formatting oversight.
- **Section 3.2, quality gate**: "10 of 25 letters pass this gate" at L0=82, but Section 4.3 also reports "10 of 25 letters passing the F1 > 0.85 quality gate." This is consistent but the intro (Section 1) says "18/25 letters pass F1 > 0.85 gate" -- this is from the outline's description of the improved first-letter results, not from the actual final text, but the outline number (18/25) differs from the method/experiments number (10/25). Verify the outline does not propagate the stale 18/25 figure into any section draft.
- **Section 3.4, last sentence**: "including 'eight,' 'lower,' 'liked,' 'offer,' and 'often'" lists 5 of 9 words. The experiments section (5.4) names only "often," "eight," and "lower" with specific feature IDs. Consider listing all 9 or none, for completeness.
- **Section 3.5**: "absorbed letters (alpha >= 0.10; n = 13) versus non-absorbed (alpha < 0.05; n = 9), with 3 letters in between excluded" -- this partition is defined using L0=82 results, but the CMI values are computed independently of L0. A reviewer may ask whether the partition would differ at L0=22 or L0=176. State this caveat explicitly.
- **Section 3.6, Table**: "City -> Country" lists "N_parents = 28" but the experiments section (Table 2) reports "N_parents = 28" for city-country. Section 3.2 says "189 single-token cities across 29 countries." The discrepancy between 28 and 29 parents should be reconciled (presumably one country was excluded -- state which one and why).
- **Section 3.6**: "Countries with fewer than 5 cities or probe F1 < 0.50 are excluded from absorption claims." The F1 threshold here is 0.50, but the quality gate in Section 3.2 is F1 > 0.85. The relationship between these two thresholds (0.50 for exclusion vs. 0.85 for primary claims) should be stated explicitly to avoid confusion.
- **Table 1**: "L12-65k" entry shows "L0 = varies" -- specify which L0 value was actually used, or state that multiple L0 values were tested and list them. "Varies" is not informative.
- **Table 1**: GPT-2 entries show "L0 = --" for L1-ReLU SAEs. A brief parenthetical "(L1-ReLU SAEs do not have configurable L0)" would help readers unfamiliar with the distinction, or simply state the measured L0 range.

## Visual Element Assessment

- [x] Figures/tables match outline plan -- Table 1 (SAE configurations) is present as planned
- [x] All visuals referenced before appearance -- Table 1 is referenced in the opening sentence of Section 3.1
- [x] Captions are self-explanatory -- Table 1 caption is adequate though it could note the Rajamanoharan et al. citation for JumpReLU
- [ ] No text-heavy sections that need visual support -- Section 3.4 (confound decomposition) would benefit from a small diagram or flowchart showing the classification logic (token -> check if FN at current L0 -> check recovery at higher L0 -> classify). The decision tree is described in prose but the branching logic is complex enough to warrant a visual.

## What Works Well

1. **The control suite (Section 3.3) is the methodological backbone of the paper.** Four controls with clear expected outcomes and explicit observed outcomes make the metric audit credible. Reporting C1 observed = 11.8% (vs. expected <2%) and C2 observed = 74.6% (vs. expected lower than measured) directly in the method section is unusually transparent and signals scientific rigor.

2. **The confound decomposition protocol (Section 3.4) is conceptually strong.** Using a multi-L0 sweep to classify false negatives by persistence is a clean operational test. The specific example of 9 persistent words (with names given) makes the abstract classification concrete. This protocol is the paper's primary methodological contribution.

3. **Table 1 efficiently communicates the experimental design space.** The table covers 8 configurations across 2 models, 4 L0 values, 2 widths, and 2 architectures in a compact format. A reader can immediately understand the scope of the study.
