# Critique: Related Work Section

## Overall Score: 6/10

The Related Work section provides adequate coverage of the relevant literature but has significant weaknesses in structure, consistency with other sections, and depth of analysis. Below is a detailed breakdown.

---

## Strengths

1. **Logical organization.** The four subsections (Architectures, Absorption, Unsupervised Detection, Positioning) follow a sensible progression from background to the specific gap this paper addresses.
2. **Key citations present.** The section references all major relevant works: Chanin et al. [2024], Chen et al. [2025], Bussmann et al. [2025], Elhage et al. [2022], and Makhzani & Frey [2014].
3. **Clear positioning statement.** Section 2.4 effectively situates the paper between supervised detection and preventive architectures.
4. **Appropriate tone.** The writing is scholarly and avoids overstating contributions.

---

## Weaknesses and Improvement Suggestions

### 1. Inconsistency with the Introduction (Major)

The Related Work section and the Introduction describe different experimental results and paper structures. This is a critical inconsistency:

- **Intro/Abstract** describes a **pilot-scale vs. full-scale comparison** (F1 = 0.704 on 46 pairs vs. F1 = 0.007 on 3,702 pairs), with experiments E1-E7.
- **Related Work Section 2.3** only mentions the full-scale failure without acknowledging the pilot-scale success: "Our work is the first to empirically demonstrate that this conflation leads to near-random detection performance."
- **Related Work Section 2.4** does not mention scaling analysis at all.

**Fix:** Section 2.3 should acknowledge the pilot-scale result and frame the contribution as characterizing scaling limits, consistent with the Abstract and Introduction. Section 2.4 should explicitly mention the scaling analysis as a key contribution gap.

### 2. Terminology Inconsistency: "Collision" vs. "Absorption" (Major)

The Methodology section (3.1) formally defines both "Feature Collision" and "Absorption" as distinct concepts:
- Collision = multiple concepts activating the same SAE feature
- Absorption = parent suppressing child under co-occurrence

However, the Related Work section uses these terms interchangeably or inconsistently:
- Section 2.2: "Matryoshka SAEs reduce absorption rates from 0.49 to 0.05" -- this refers to collision rates, not absorption.
- Section 2.3: "flagged as potential collisions" in the context of absorption detection.
- Section 2.4: The positioning paragraph only mentions "absorption detection" but the paper actually tests both collision and absorption.

**Fix:** Clarify in Section 2.2 whether the 0.49 -> 0.05 reduction refers to collisions or absorptions (cite Bussmann et al. precisely). In Section 2.3, use "absorption" consistently since that is the paper's focus. Add a sentence noting that prior work sometimes conflates these terms.

### 3. Missing Citation for "Co-occurrence Clustering" Claim (Moderate)

Section 2.3 states: "Several works have proposed using feature co-occurrence patterns to identify related or absorbed features without supervision." However, only Chen et al. [2025] is cited as proposing co-occurrence clustering specifically for absorption. The claim "several works" is not substantiated.

**Fix:** Either (a) cite additional works that propose co-occurrence clustering for absorption detection, or (b) change "Several works" to "Chen et al. [2025] and related work" to accurately reflect the citation.

### 4. Inconsistent Citation Style (Minor)

- Section 2.1 uses "Makhzani & Frey, 2014" (ampersand).
- Other sections use "Chanin et al. [2024]" (brackets, no comma before year).
- Section 2.2 uses "Chanin et al. [2024]" in text but "Chen et al., 2025" in the HSAE sentence (comma before year).

**Fix:** Standardize citation style throughout. Pick one format (e.g., "Author et al. [Year]" without comma) and apply consistently.

### 5. Section 2.1 is Too Long Relative to Relevance (Moderate)

Section 2.1 discusses SAE architectures (JumpReLU, TopK, BatchTopK, Matryoshka, GemmaScope) at length, but only Matryoshka SAEs are relevant to the paper's focus on absorption. The other architectures are never mentioned again.

**Fix:** Condense Section 2.1 to 2-3 sentences. Keep the Matryoshka discussion (relevant to Section 2.2) and briefly mention that various SAE architectures exist, but absorption has been studied primarily in standard formulations.

### 6. Missing Discussion of Superposition Literature (Moderate)

Section 2.2 mentions superposition [Elhage et al., 2022] in one sentence but does not explain its relevance to absorption. The Introduction provides more context on superposition than the Related Work section.

**Fix:** Expand the superposition discussion by 1-2 sentences explaining why superposition creates the conditions for absorption (overlapping representations competing for activation).

### 7. DFDA Not Mentioned in Related Work (Minor)

DFDA is a key method in the paper (Section 3.3, Experiments E7/E4.7) but is not mentioned in Related Work. While DFDA is introduced as a novel contribution, if it builds on any prior inference-time compensation methods, they should be cited.

**Fix:** If DFDA is entirely novel, add a sentence in Section 2.4 stating that "To our knowledge, no prior work has explored inference-time compensation for absorbed features." If it builds on prior work, cite it.

### 8. Weak Connection Between Sections 2.3 and 2.4 (Minor)

Section 2.3 ends with "Our work is the first to empirically demonstrate that this conflation leads to near-random detection performance." Section 2.4 then starts with "Our work fills the gap..." The transition is abrupt.

**Fix:** Add a bridging sentence at the end of 2.3 or beginning of 2.4 that connects the empirical demonstration to the broader positioning.

### 9. Quantitative Claims Lack Precision (Minor)

- "reduce absorption rates from 0.49 to 0.05" -- specify what metric this is (collision rate? absorption rate? per-feature?)
- "near-random detection performance" -- the Introduction gives specific numbers (F1 = 0.007), but Related Work does not.

**Fix:** Add the specific F1 numbers to Section 2.3 to strengthen the empirical claim.

---

## Consistency Check Summary

| Check | Status | Notes |
|-------|--------|-------|
| Symbols/notation | Pass | No mathematical notation in Related Work; no conflicts |
| Terminology: "absorption" | **Fail** | Used inconsistently with "collision" |
| Terminology: "UAD" | Pass | Used consistently |
| Terminology: "co-occurrence clustering" | Pass | Used consistently |
| Citation style | **Fail** | Inconsistent formatting |
| Experimental results alignment | **Fail** | Pilot-scale result missing from Related Work |
| Contribution claims alignment | **Fail** | "scaling limits" framing absent from 2.3 and 2.4 |
| DFDA | **Fail** | Not mentioned; unclear if novel or based on prior work |

---

## Priority Fixes (Ranked)

1. **Align with Introduction:** Add pilot-scale vs. full-scale framing and "scaling limits" contribution to Sections 2.3 and 2.4.
2. **Fix collision/absorption terminology:** Distinguish these concepts consistently, matching the Methodology definitions.
3. **Condense Section 2.1:** Remove irrelevant architecture details; keep only Matryoshka discussion.
4. **Standardize citations:** Pick one format and apply throughout.
5. **Substantiate or narrow "Several works" claim in 2.3.**
6. **Add DFDA novelty claim or prior citations.**
7. **Expand superposition discussion slightly.**
8. **Add specific F1 numbers to strengthen empirical claims.**
9. **Improve 2.3 -> 2.4 transition.**
