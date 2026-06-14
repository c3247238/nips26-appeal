# Critique: Background and Related Work

## Summary Assessment
The section provides a competent three-part survey covering SAEs for mechanistic interpretability, the feature absorption literature, and the RAVEL dataset. It correctly positions EDA against prior work and identifies the gap motivating the paper's contributions. However, Section 2.1 consumes over half the ~600-word budget on SAE survey material not directly used by the method or experiments, the critical gap statement at the end of Section 2.2 covers only Gap 1 (detection) while omitting Gaps 2 (generalizability) and 3 (taxonomy), and the single most important positioning sentence -- explaining what Tang et al. proved versus what EDA adds -- is vague enough to obscure the novelty claim.

## Score: 6/10
**Justification**: Solid coverage with no factual errors, honest handling of the amortization gap confounder, and a clean RAVEL analogy. A 7 requires: (1) cutting Section 2.1 by ~40% and reallocating to the gap statement, (2) making the Tang et al. differentiation specific and quantitative, (3) restating all three gaps at the end of 2.2, and (4) grouping mitigations by which absorption subtype they address to foreshadow the taxonomy.

## Critical Issues

### Issue 1: The Tang et al. differentiation sentence is vague and undersells the novelty
- **Location**: Section 2.2, paragraph 2
- **Quote**: "At partial minima, the encoder direction $w_{e,j}$ for an absorbed parent latent shifts away from its decoder direction $d_j$ by incorporating child decoder components---a geometric signature that our Encoder-Decoder Alignment (EDA) metric formalizes."
- **Problem**: This is the single most important positioning sentence in the entire related work section, and it could mean anything from "we restate Tang et al.'s result" to "we derive new mathematics." The method section (Section 3.2) derives Theorem 1 -- a formal lower bound on EDA in terms of absorption degree $\delta$ and parent-child decoder angle $\theta_{jc}$. The related work should clearly preview the gap that Theorem 1 fills: Tang et al. characterize the phenomenon but do not yield a computable detection metric or a quantitative bound.
- **Fix**: Replace with: "Tang et al. prove that at partial minima of the biconvex SDL loss, absorbed latents' encoder directions incorporate child decoder components. Their result characterizes the phenomenon but does not yield a computable detection metric or a quantitative bound on the misalignment magnitude. Section 3 bridges this gap: we derive EDA as a scalar metric and Theorem 1 provides a formal lower bound on EDA in terms of absorption degree $\delta$ and the parent-child decoder angle $\theta_{jc}$."

### Issue 2: Gap statement at end of Section 2.2 covers only Gap 1
- **Location**: Section 2.2, final paragraph
- **Quote**: "For the hundreds of SAEs already deployed in interpretability pipelines, no post-hoc diagnostic exists that operates without activation data or known probe directions. Equally, all existing detection methods require running the model forward on evaluation data---precluding systematic auditing of the $d_{\text{SAE}} = 65{,}536$ latents in a typical Gemma Scope SAE. This gap motivates EDA: a weight-only screening metric computable from $W_e$ and $W_d$ alone."
- **Problem**: This paragraph is the payoff of the entire related work section, and it addresses only Gap 1 (detection without foreknowledge). The introduction defines three gaps; the related work should confirm that the literature review has validated all three. Gap 2 (every measurement uses first-letter) and Gap 3 (no taxonomy distinguishing structurally different absorption modes) are not restated. A reader who skipped the introduction would think the paper only addresses detection.
- **Fix**: Expand to explicitly map each gap to the reviewed literature: "Three gaps emerge from this review. First, no post-hoc diagnostic operates without activation data or probe directions, precluding systematic auditing of the $d_{\text{SAE}} = 65{,}536$ latents in a deployed SAE. Second, every published absorption measurement -- from Chanin et al. through SAEBench -- uses the first-letter spelling task; whether absorption occurs in entity-attribute or safety-relevant semantic hierarchies is untested. Third, all prior work treats absorbed latents as a single undifferentiated category, conflating structurally distinct failure modes that may require different remediation strategies."

## Major Issues

### Issue 3: Section 2.1 is disproportionately long and contains material not used by the paper
- **Location**: Section 2.1, paragraphs 1--2
- **Quote**: "Architectural variants followed: Gated SAEs decouple feature detection from magnitude estimation to reduce $L_1$-induced shrinkage \cite{rajamanoharan2024gated}; JumpReLU SAEs train $L_0$ directly via the straight-through estimator... TopK SAEs impose fixed sparsity with clean scaling laws up to 16M latents..."
- **Problem**: The outline allocates ~600 words for the entire Section 2. Section 2.1 alone runs ~350 words. The Gated/JumpReLU/TopK architectural survey is standard knowledge for NeurIPS MI reviewers, none of these variants are tested in the paper (beyond noting that Gemma Scope uses gated architecture), and the material does not connect back to the absorption argument. The reclaimed space is needed for the gap statement expansion (Critical Issue 2).
- **Fix**: Cut paragraphs 1--2 to ~120 words total. Keep: the SDL loss definition (needed for EDA derivation), the biconvex structure reference to Tang et al., and Gemma Scope/SAEBench as experimental infrastructure. Move or delete the Gated/JumpReLU/TopK details -- a footnote listing them with citations suffices.

### Issue 4: Chanin et al. metric description includes implementation thresholds that belong in Section 4
- **Location**: Section 2.2, paragraph 1
- **Quote**: "it identifies false-negative tokens where all split latents fail to activate, then detects absorption when the highest-ablation-effect latent has cosine similarity $> 0.025$ with the probe direction and exceeds the second-highest by $\geq 1.0$"
- **Problem**: The specific threshold values (cosine > 0.025, magnitude gap >= 1.0) are implementation details that the reader cannot evaluate without the experimental context of Section 4.1. In the related work, they interrupt the narrative flow and add ~25 words to an already over-budget section. The introduction already establishes that the metric requires probe directions.
- **Fix**: Replace with a characterization of what the metric requires and what it cannot do: "Their supervised metric identifies absorption by detecting false-negative tokens where parent-associated latents fail to activate, requiring pre-specified probe directions and activation data. This makes systematic auditing of all $d_{\text{SAE}}$ latents in a deployed SAE intractable." Move the threshold details to Section 4.1 where ground-truth labels are described.

### Issue 5: Architectural mitigations listed without differentiation relevant to the taxonomy
- **Location**: Section 2.2, paragraph 4
- **Quote**: "Several architectural responses aim to reduce absorption at training time. Matryoshka SAEs... OrtSAE... Narayanaswamy et al.... Costa et al.... Hierarchical SAE designs..."
- **Problem**: Six mitigation approaches are listed with roughly equal weight, but Section 6.3 makes a specific claim: most mitigations target late absorption (the ~25% minority), while the dominant early-absorption mode (~75%) requires dictionary-coverage solutions. The related work should foreshadow this distinction. Currently the list reads as a catalogue.
- **Fix**: Group mitigations into two categories: (1) approaches addressing encoder-decoder alignment or sparsity pressure (OrtSAE, MP-SAE, masked regularization); (2) approaches increasing dictionary capacity or hierarchical structure (Matryoshka, hierarchical SAEs). Add one sentence: "Section 6 reveals that these two strategy classes address fundamentally different absorption subtypes, with the dominant mode requiring the latter."

### Issue 6: The reliability-limits paragraph makes an unsupported ranking claim
- **Location**: Section 2.1, paragraph 3
- **Quote**: "Feature absorption---the focus of this paper---is arguably the most consequential of these reliability failures, because it creates systematic false negatives in feature activations without any visible degradation in reconstruction metrics."
- **Problem**: "Arguably the most consequential" is a ranking claim with no supporting evidence. The paper does not compare absorption's impact against the reproducibility failures (Song et al.) or feature-recovery failures (Cui et al., Korznikov et al.). The "because" clause provides a mechanism, not a comparative assessment.
- **Fix**: Replace the ranking with a characterization: "Feature absorption---the focus of this paper---is distinctive among these reliability failures because it creates systematic false negatives in feature activations without any visible degradation in reconstruction metrics." This states what makes absorption different without an unsupported "most consequential" claim.

### Issue 7: RAVEL section claims sparsity-pressure predicts frequency-imbalance modulation without derivation
- **Location**: Section 2.3, paragraph 2
- **Quote**: "providing variation in the frequency imbalance that the sparsity-pressure absorption mechanism predicts should modulate absorption rates"
- **Problem**: Neither Chanin et al. nor Tang et al. make this specific prediction. The claim presents the paper's own hypothesis as an established prediction from prior work. Section 5.2 reports the empirical pattern, but the causal link between hierarchy breadth and absorption rate is not established in the literature.
- **Fix**: Soften to: "If absorption severity depends on competition between parent and child latents for dictionary capacity, broader parent classes (with fewer entities per parent) should exhibit different absorption rates. Section 5 tests this variation."

## Minor Issues

- **Section 2.1, paragraph 1**: "representing features as overlapping linear directions in a phenomenon called *superposition*" -- "superposition" is not in the glossary. Either add it or remove the definitional framing since the NeurIPS MI audience knows the term.
- **Section 2.1, paragraph 1**: The biconvex SDL loss equation appears here (unnumbered) and again in Section 3.1 as Equation 1. Having it in two places creates ambiguity. Remove the display equation from Section 2.1 and reference forward to Equation 1, or assign a number here and reference it from Section 3.
- **Section 2.1, paragraph 2**: "$L_1$-induced shrinkage" is not defined in the glossary. Either define parenthetically or remove the mechanistic detail.
- **Section 2.1, paragraph 2**: "SAEs with millions of latents encoding safety-relevant concepts (deception, sycophancy, bias)" -- the parenthetical list is attributed via citation to scaling papers, but the safety-relevance claim needs its own citation or qualification.
- **Section 2.1, paragraph 3**: "TopK SAEs achieve only 0.80 PW-MCC reproducibility" -- "PW-MCC" is undefined in the glossary or notation table and is not used elsewhere in the paper. Replace with: "TopK SAEs show only 80% feature reproducibility across training runs."
- **Section 2.2, paragraph 2**: "a geometric signature that our Encoder-Decoder Alignment (EDA) metric formalizes" -- forward-references the paper's own contribution within related work. Consider: "a geometric signature that Section 3 formalizes as Encoder-Decoder Alignment (EDA)."
- **Section 2.2, paragraph 4**: "they reduce absorption on SAEBench while achieving the best scores on RAVEL and sparse probing" (re: Matryoshka) -- the RAVEL/sparse-probing claims are about general SAE quality, not absorption. This detail is tangential.
- **Section 2.2, paragraph 4**: "Hierarchical SAE designs with explicit structural constraints \cite{muchane2025hierarchical, luo2026hsae}" -- these two citations appear nowhere else in the paper. Verify they are real references (luo2026hsae has a 2026 date).
- **Section 2.3, paragraph 2**: "The structural parallel to the first-letter task is precise" -- "precise" overclaims. The parallel is structural but differs in co-occurrence statistics, number of parent classes, and semantic vs. syntactic nature. Use "direct" or "close."
- **Transition sentence**: "Armed with this context" is mildly informal for NeurIPS. Use: "Building on this foundation, Section 3 derives EDA from first principles."

## Visual Element Assessment
- [x] Figures/tables match outline plan (outline specifies no figures for Section 2; section has none)
- [x] All visuals referenced before appearance (N/A)
- [x] Captions are self-explanatory (N/A)
- [ ] No text-heavy sections that need visual support -- The mitigations paragraph lists 6 approaches in rapid succession. A small comparison table (method, mechanism, requires retraining?, absorption subtype targeted) would aid readability, though this may exceed space constraints.

## Cross-Section Consistency

**With Introduction:**
- The introduction lists "KronSAE" among architectural responses (intro line 7). Section 2.2 does NOT mention KronSAE. Either add KronSAE to the related work or remove it from the introduction.
- The introduction defines three gaps; the related work only confirms Gap 1 at its conclusion (see Critical Issue 2).

**With Method (Section 3):**
- Section 3.2 lists three EDA confounders: polysemanticity, amortization gap, training noise. Section 2.2 mentions only polysemanticity and the amortization gap. The related work should foreshadow all three or avoid implying the list is complete.
- The biconvex SDL loss equation appears unnumbered in Section 2.1 and as Equation 1 in Section 3.1. Pick one canonical location.

**Terminology:**
- "partial minima" (Section 2.1) is consistent with glossary's preferred "partial minimum."
- "sparse dictionary learning (SDL) loss" usage is consistent.
- Chanin et al. uses two citation keys (\cite{chanin2024absorption} and \cite{chanin2024toymodels}). Verify these are correctly differentiated in the bibliography.

## What Works Well
1. **The O'Neill et al. amortization gap paragraph** (Section 2.2, paragraph 3) is a model of honest positioning: it identifies a third confounder for the paper's own metric before the reader encounters it in the method section. Reviewers will credit this intellectual honesty.
2. **Section 2.3's structural analogy** between first-letter and RAVEL hierarchies ("both define a parent feature class whose activation should persist when any member child feature is active") is precise, concrete, and directly sets up Section 5's experimental design.
3. **The hierarchy-breadth variation argument** (Section 2.3: 6 parent classes for city-continent vs. ~100 for city-country) provides a specific, testable prediction that elevates the related work from survey to argument -- it explains why RAVEL was chosen and what the experiments will test.
