# Section Critique: Related Work (Section 2)

**File:** `writing/sections/related_work.md`
**Critic:** Sibyl Section Critic
**Date:** 2026-04-15

---

## Overall Score: 7.5 / 10

The Related Work section is well-organized, technically precise, and covers the relevant literature with appropriate depth. It successfully establishes the gap that the paper fills (no validation of the Chanin metric on JumpReLU SAEs) and threads the narrative arc from SAE basics through absorption to the paper's contribution. However, it has several structural, consistency, and coverage issues that weaken it relative to a top-tier venue standard.

---

## Dimension Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| Coverage & completeness | 7 | Good on absorption and mitigations; weak on broader MI evaluation and polysemanticity literature |
| Narrative arc & positioning | 8 | Clear logical flow from SAEs -> absorption -> mitigations -> theory; effective gap identification |
| Technical accuracy | 8 | Equations are correct; citations are well-placed; some minor imprecision noted below |
| Notation consistency | 7 | Mostly consistent with notation.md but several deviations |
| Glossary consistency | 7 | Several glossary violations |
| Cross-section consistency | 7 | Some numerical discrepancies vs. intro and experiments sections |
| Writing quality | 8 | Clear prose; appropriate density for an ML venue |
| Citation completeness | 6 | Missing key recent works; some references lack full detail |

---

## Strengths

1. **Clear taxonomy of SAE architectures.** The section cleanly distinguishes L1-ReLU and JumpReLU SAEs with precise mathematical definitions, establishing the architectural difference that motivates the entire study. The JumpReLU equation in Section 2.1 matches the notation table exactly.

2. **Effective gap identification in Section 2.3.** The final paragraph of Section 2.3 is the strongest part of the section. It explicitly names the gap (all mitigations benchmark against the Chanin metric on L1-ReLU SAEs; none validates on JumpReLU) and connects this gap to the paper's contribution. This paragraph earns its placement and sets up Sections 3-4 effectively.

3. **Balanced treatment of hedging vs. competitive exclusion (Section 2.2).** The distinction between feature hedging and competitive exclusion is well-articulated. The sentence "The false negatives from hedging resolve when L0 is increased [...] whereas absorption-driven false negatives persist across L0 values" gives the reader the key operational distinction needed to understand the confound decomposition in Section 3.4.

4. **Theoretical limitations of SAEs (Section 2.1, para 3).** Including Leask et al. (2025), Cui et al. (2025), and Engels et al. (2025) on the fundamental limitations of SAE feature recovery is a smart move. It pre-empts the reviewer objection "why question the metric at all?" by establishing that SAE features have known theoretical weaknesses.

5. **Rate-distortion framing (Section 2.4).** The connection between successive refinement and SAE feature absorption is novel and well-explained. The honest final sentence ("though as we report there, the diagnostic does not survive replication with perfect probes") is commendable scientific transparency.

---

## Weaknesses and Specific Improvement Suggestions

### Critical Issues

**C1. Numerical inconsistency between related_work.md and other sections.**

In Section 2.2 (line 17), the text states:
> "absorption rates of 15--35% are reported on the first-letter spelling task"

But the introduction (intro.md, line 5) states:
> "measured absorption rates of 15--35% across hundreds of SAEs on first-letter spelling"

And the experiments section (experiments.md, line 11) reports a measured rate of 13.4% for first-letter at L0=82. The paper.md version of Section 2.2 says "absorption rates range from 15% to 35% on first-letter spelling." These numbers are attributed to SAEBench/Chanin et al., but the related_work.md version adds "Across hundreds of SAEs evaluated in SAEBench (Karvonen et al., 2025; ICML 2025)" which the paper.md version does not. Verify that the 15-35% range is correctly attributed to the SAEBench literature (GPT-2 Small, L1-ReLU) and not to the current study's results. The attribution is technically correct but the phrase "on the first-letter spelling task" could mislead readers into thinking this range applies broadly rather than specifically to GPT-2 Small with L1-ReLU SAEs. Add a qualifier: "on the first-letter spelling task using GPT-2 Small with L1-ReLU SAEs."

**C2. The section title uses "Background and Related Work" but the paper.md version has a shortened Section 2 without the "Background" framing.**

In related_work.md (line 1), the section is titled "2 Background and Related Work." In paper.md (line 39), it is titled "2 Background and Related Work" as well, so this is actually consistent. However, the outline.md (line 42) labels it "2. Background and Related Work" which matches. No action needed here -- this is consistent.

**C3. The related_work.md version diverges from paper.md on multiple substantive points.**

The related_work.md (standalone section file) contains content that differs from the integrated paper.md version of Section 2. Several differences are substantive:

- related_work.md Section 2.1 mentions "Scaling to frontier models demonstrated that SAEs recover safety-relevant and semantically coherent features (Templeton et al., 2024; Gao et al., 2024)" as a separate paragraph point. paper.md condenses this into "SAEs have been scaled to frontier models (Templeton et al., 2024; Gao et al., 2024) and used for circuit analysis (Lindsey et al., 2025)." The paper.md version adds Lindsey et al. (2025) for circuit analysis, which the standalone section omits.

- related_work.md Section 2.2 provides more detail on the measurement protocol (k-sparse probes, cosine similarity threshold, false-negative rate definition) than paper.md. This duplication with Section 3 is unnecessary in the related_work.md version.

- related_work.md Section 2.2 includes "Tian et al. (2025) frame feature absorption as a special case of poor feature sensitivity" which is absent from paper.md. This reference should be retained -- it is a relevant perspective that broadens the framing.

**Recommendation:** Decide which version is canonical and synchronize. If related_work.md is the working draft, ensure paper.md is updated to include Tian et al. (2025) and any other additions. If paper.md is canonical, update related_work.md.

### Major Issues

**M1. Missing literature on SAE evaluation beyond absorption.**

The section covers absorption measurement and mitigation thoroughly but neglects the broader SAE evaluation literature. Specifically:

- **Polysemanticity analysis:** The paper's core argument is that absorption measurements are confounded. Prior work on polysemanticity evaluation (e.g., how SAE features are assessed for monosemanticity) is relevant context. The experiments section (line 1, experiments.md) mentions "across five hierarchy domains" but the related work does not discuss how hierarchy domains have been used in prior SAE evaluation beyond Chanin et al.

- **SAEBench specifics:** SAEBench (Karvonen et al., 2025) is cited but its 8-metric evaluation suite is not described. Since the paper positions itself as extending SAEBench, a brief description of what SAEBench measures would help readers understand the paper's broader evaluation context.

- **Activation patching as an interpretability method:** Section 3.5 and 4.5 use activation patching extensively, but the related work does not discuss prior uses of activation patching in mechanistic interpretability (e.g., Meng et al., 2022 on knowledge editing; Conmy et al., 2023 on automated circuit discovery). This gap makes the methodology section feel unsupported.

**Recommendation:** Add 2-3 sentences on SAEBench's scope and on activation patching as an established causal intervention method in mechanistic interpretability.

**M2. Section 2.3 (Architectural Mitigations) reads as a list, not a synthesis.**

The six mitigations are presented as bullet points with minimal comparative analysis. A stronger related work section would synthesize commonalities and differences:

- Which mitigations modify the encoder vs. the training objective vs. the dictionary structure?
- What is the shared assumption across all six (that the Chanin metric validly measures competitive exclusion)?
- Are any of these mitigations evaluated on JumpReLU SAEs? (The text says "None validates the metric on JumpReLU SAEs" at the end, but this could be stated more prominently.)

The ATM-SAE bullet (line 31) provides the most informative comparison ("absorption score of 0.007 versus 0.140 for TopK and 0.011 for JumpReLU on Gemma 2 2B"). This suggests ATM-SAE *has* been evaluated on JumpReLU architectures, which partially contradicts the concluding claim that "None validates the metric on JumpReLU SAEs." Clarify: is the issue that ATM-SAE reports a Chanin metric score on JumpReLU but does not validate the metric itself (i.e., does not run shuffled-label controls)?

**Recommendation:** Restructure Section 2.3 around the shared assumption (Chanin metric validity) rather than as a flat list. Clarify the ATM-SAE/JumpReLU point.

**M3. Section 2.4 technical detail is too deep for a related work section.**

Lines 43-46 describe CMI estimation details ($k$-nearest-neighbor estimator, $d' = 10$ subspace, probe quality confound, F1 > 0.85 gate). This level of methodological detail belongs in Section 3.7 (CMI Estimation), not in the related work. The related work should present the theoretical motivation and the gap (no prior work connects successive refinement to SAE absorption) without previewing the paper's specific estimation choices.

**Recommendation:** Trim Section 2.4 to the theoretical framework (Equitz and Cover, 1991; the Markov chain condition; CMI as a criterion) and move the estimation-specific details to Section 3.7. The final paragraph can state that "we test this prediction in Section 6" without previewing the result.

### Minor Issues

**m1. Glossary violation: "feature absorption" usage.**

The glossary (glossary.md, line 11) specifies that "feature absorption" should be expanded on first use in each section, not just "absorption." In related_work.md, the first use is in the section title of Section 2.2 ("Feature Absorption") and the first body text occurrence is "Chanin et al. (2024; NeurIPS 2025 Oral) define feature absorption" (line 17), which is correct. However, later occurrences in Section 2.3 use "absorption" without the "feature" qualifier on first use in that subsection (e.g., line 29: "SAEBench absorption score: 0.03 versus 0.29 for BatchTopK"). The glossary permits "absorption" after first use within a section, but Section 2.3 is a new subsection and arguably should re-introduce the full term.

**m2. Glossary violation: "SAE latent" vs. "SAE feature".**

The glossary (line 26) prefers "SAE latent" over "SAE feature." In Section 2.2 (line 17), the text uses both "parent latent" and "child latent" (correct), but also uses "parent-associated latents" (not in glossary) and "candidate SAE latents" (line 17, correct). In Section 2.3 (line 34), "MP-SAE (Costa et al., 2025) uses iterative encoding to refine latent representations, partially recovering missed features." The phrase "missed features" should be "missed latents" per the glossary preference.

**m3. Notation issue: $\tau_{\cos}$ and $\tau_{\text{mag}}$ first appearance.**

These threshold symbols first appear in Section 2.3 (line 36): "the metric's thresholds ($\tau_{\cos} \geq 0.025$, $\tau_{\text{mag}} \geq 1.0$)." The notation table defines $\tau_{\cos}$ as "Cosine similarity threshold for candidate feature identification" and $\tau_{\text{mag}}$ as "Magnitude gap threshold for absorption confirmation." However, in Section 2.2 (line 17), the same thresholds are described without the symbolic notation: "cosine similarity $\cos(d_j, v_p) \geq 0.025$." The symbols should be introduced consistently -- either use the symbolic notation in Section 2.2 where the thresholds first appear functionally, or defer to Section 3.

**m4. The closing sentence of the section (line 48) is a structural transition, not a content contribution.**

"Section 3 describes the experimental methodology for the three-pronged study: metric audit (Q1), L0 phase transition (Q2), and CMI diagnostic (Q3)" is pure road-mapping. While acceptable, it duplicates the roadmap already provided in the introduction (intro.md, lines 25-26). Consider removing this sentence or replacing it with a stronger summary of the gap.

**m5. Missing year/venue for KronSAE.**

Line 32: "KronSAE (2025) exploits Kronecker factorization..." provides only a year, no authors. This is the weakest citation in the section. If the paper has been identified, provide full authorship. If it has not been formally published, consider a footnote explaining the citation status.

**m6. Narayanaswamy et al. (2026) date.**

Line 33: This is dated 2026. Verify that this is correct (e.g., a preprint or upcoming publication). If so, note that it is a very recent work, which is fine but should be flagged for the camera-ready version.

**m7. The phrase "All published absorption measurements use GPT-2 Small with L1-ReLU SAEs" appears in both Section 2.2 and Section 2.3.**

Section 2.2 (line 17): "All published absorption measurements use GPT-2 Small with L1-ReLU SAEs; no study has validated the Chanin metric on JumpReLU SAEs."
Section 2.3 (line 36): "None validates the metric on JumpReLU SAEs, the architecture whose hard-threshold activation dynamics differ most from the L1-ReLU SAEs..."

The repetition is intentional (first for measurement, then for mitigations) but slightly redundant. Consider condensing the Section 2.2 version and making the Section 2.3 version the primary statement of the gap.

---

## Cross-Section Consistency Check

### vs. Introduction (intro.md)

| Item | Related Work | Introduction | Status |
|------|-------------|--------------|--------|
| Chanin et al. citation format | "Chanin et al. (2024; NeurIPS 2025 Oral)" | "Chanin et al. (2024; NeurIPS 2025 Oral)" | Consistent |
| Absorption rate range | "15--35%" | "15--35%" | Consistent |
| Shuffled control ratio | Not stated | "4.7x" | N/A (different scope) |
| Mitigations listed | Matryoshka, OrtSAE, ATM-SAE, KronSAE, MP-SAE, Masked reg | Matryoshka, OrtSAE, ATM-SAE, masked reg | Related work adds KronSAE and MP-SAE -- consistent (intro is selective) |
| Hedging definition | "incorrect L0 merges correlated features" | Not explicitly defined | Related work provides the definition; intro defers to Section 2 -- correct |

### vs. Method (method.md)

| Item | Related Work | Method | Status |
|------|-------------|--------|--------|
| Probe sparsity k=5 | Stated in Section 2.2 | Stated in Section 3.2 | Consistent but duplicated |
| Cosine threshold 0.025 | Stated in Section 2.2, 2.3 | Stated in Section 3.2 | Consistent but duplicated |
| CMI estimator | k-NN, d'=10 | k-NN, k_nn=5, d'=10 | Related work previews too much detail |
| Gemma Scope description | "400+ open JumpReLU SAEs on Gemma 2 2B/9B/27B" | Cited as Lieberum et al. (2024) | Consistent |

### vs. Experiments (experiments.md)

| Item | Related Work | Experiments | Status |
|------|-------------|-------------|--------|
| First-letter absorption at L0=82 | Not stated (defers to results) | 13.4% (Table 2) and 14.39% (Table 7) | Correct -- related work does not preview own results |
| Shuffled control rates | Not stated | 59.6% (Table 2) | Correct |
| ATM-SAE scores | "0.007 vs 0.140 for TopK and 0.011 for JumpReLU" | Not discussed | Consistent (ATM-SAE is cited work, not own result) |

### vs. Notation Table (notation.md)

| Symbol | Notation Table | Related Work | Status |
|--------|---------------|--------------|--------|
| $x$ | $x \in \mathbb{R}^{d_{\text{model}}}$ | $x \in \mathbb{R}^{d_{\text{model}}}$ (line 5) | Consistent |
| $z$ | $z \in \mathbb{R}^{d_{\text{SAE}}}$ | $z \in \mathbb{R}^{d_{\text{SAE}}}$ (line 5) | Consistent |
| $\theta_j$ | "JumpReLU per-latent activation threshold" | $\theta_j$ (line 9) | Consistent |
| $L_0$ | "Number of non-zero latents per forward pass" | "the number of non-zero latents per forward pass" (line 11) | Consistent |
| $\tau_{\cos}$ | "default 0.025" | $\geq 0.025$ (line 36) | Consistent |
| $v_p$ | "Probe direction" | $v_p$ (line 17) | Consistent |
| $k$ | "Scalar; k=5 throughout" | $k=5$ (line 17) | Consistent |

### vs. Glossary (glossary.md)

| Term | Glossary Preference | Related Work Usage | Status |
|------|--------------------|--------------------|--------|
| Feature absorption | "feature absorption" on first use per section | Correct in Section 2.2 | OK |
| SAE latent | Prefer "latent" over "feature" | Mixed: "missed features" in Section 2.3 (line 34) | VIOLATION |
| JumpReLU SAE | "JumpReLU SAE" (not "Jump-ReLU") | "JumpReLU SAEs" (correct) | OK |
| Gemma Scope | "Gemma Scope" (not "GemmaScope") | "Gemma Scope" (correct) | OK |
| SAEBench | "SAEBench" (not "SAE Bench") | "SAEBench" (correct) | OK |
| $L_0$ operating point | Always typeset as $L_0$ | $L_0$ (correct) | OK |
| Hedging | "hedging" preferred | "feature hedging" used in line 21 | Acceptable per glossary note |
| Competitive exclusion | "competitive exclusion" | "competitive suppression" (line 21), "competitive exclusion" (line 36) | Mixed -- Section 2.2 uses "competitive suppression" which the glossary says to avoid ("mutual suppression" is the listed avoid-term, but "competitive suppression" is close) |

---

## Actionable Improvement Plan (Priority-Ordered)

1. **[HIGH] Synchronize with paper.md.** Decide which version is canonical. Ensure Tian et al. (2025), Lindsey et al. (2025), and any other divergent citations are present in both or neither.

2. **[HIGH] Restructure Section 2.3 from list to synthesis.** Organize mitigations by mechanism type (encoder modification, training objective, dictionary structure). Make the shared assumption (Chanin metric validity on L1-ReLU only) the organizing principle. Clarify the ATM-SAE/JumpReLU evaluation point.

3. **[HIGH] Add activation patching and SAEBench context.** 2-3 sentences on activation patching as an established causal method and SAEBench's scope as an 8-metric suite.

4. **[MEDIUM] Trim Section 2.4 estimation details.** Move k-NN estimator specifics, d'=10 choice, and probe quality confound preview to Section 3.7. Keep the theoretical framework and the gap statement.

5. **[MEDIUM] Fix glossary violations.** Replace "missed features" with "missed latents" (Section 2.3). Ensure "competitive exclusion" is used consistently instead of "competitive suppression" (Section 2.2).

6. **[MEDIUM] Add qualifier to the 15-35% absorption range.** Specify "on the first-letter spelling task using GPT-2 Small with L1-ReLU SAEs" to prevent misinterpretation.

7. **[LOW] Provide full citation for KronSAE.** Add author names or explain the incomplete citation.

8. **[LOW] Remove or rework the closing roadmap sentence.** It duplicates the introduction's roadmap and adds no content.

9. **[LOW] Introduce $\tau_{\cos}$ and $\tau_{\text{mag}}$ notation consistently.** Either use the symbolic notation at first functional appearance (Section 2.2) or defer all symbolic notation to Section 3.

---

## Summary

The Related Work section is solid for a first draft: it covers the essential literature, establishes the gap clearly, and threads the narrative from SAE basics to the paper's three research questions. The main weaknesses are (1) structural -- Section 2.3 is a list rather than a synthesis, and Section 2.4 previews too much methodology; (2) coverage -- missing activation patching context and broader SAE evaluation literature; and (3) consistency -- minor divergences from paper.md and glossary violations. Addressing the high-priority items would raise this section to a 8.5-9.0.
