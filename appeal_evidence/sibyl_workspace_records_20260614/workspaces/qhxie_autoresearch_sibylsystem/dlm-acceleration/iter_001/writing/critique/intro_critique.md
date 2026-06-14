# Critique: Introduction

*Revision round: 3 (current state of intro.md as of 2026-04-14)*

## Summary Assessment

The Introduction is substantively strong: it opens with a specific throughput number, motivates the composability gap concisely, and delivers contributions with quantified claims. The honest scoping paragraph is a significant asset at NeurIPS/ICML — it pre-empts reviewer skepticism. Two critical issues require immediate attention: a naming inconsistency (IGSD vs. CD-SSD that the glossary bans) and a data discrepancy in the Scope paragraph (35.1% AccRet figure not traceable to any verified experiment result). Several major issues include an unsupported AR throughput comparison, missing concurrent SSD/SSMD references in the speculative decoding enumeration that will expose the novelty claim as overstated, and section-number mismatches in the roadmap paragraph. The section is 6–7 paragraphs from submission-ready; none of the required changes are architectural.

## Score: 6/10

**Justification**: The opening paragraph, motivation structure, and honest positioning are all above average for the venue. The score is held down by: (1) systematic IGSD→CD-SSD rename miss throughout the contributions list (critical, signals incomplete proofreading), (2) the unexplained 5.13× vs. 4.69× arithmetic gap that looks like a data error to any reviewer who multiplies the stated individual speedups, (3) the 35.1% AccRet figure that conflicts with experiments data, and (4) a speculative decoding literature survey that omits SSD—the closest concurrent work. Fixing these four issues plus the minor table/roadmap inconsistencies would bring the score to 8/10. Reaching 9/10 requires an AR throughput citation.

---

## Critical Issues

### Issue 1: "IGSD" used throughout contributions list — method was renamed to CD-SSD

- **Location**: Contributions 2, 3, 4 (paragraphs starting "Exactly one pair..." through "IGSD: Information-Gain-Driven Self-Speculative Denoising") and the "Scope and Honest Positioning" paragraph.
- **Quote (contribution 4 title)**: "**IGSD: Information-Gain-Driven Self-Speculative Denoising.**"
- **Quote (contribution 2 body)**: "M1 (EntropyCache, $\eta=2.0$) + IGSD ($\tau=0.9$, $T_{\text{draft}}=16$)"
- **Quote (scope paragraph)**: "IGSD serves primarily as a composability study vehicle rather than a standalone acceleration proposal"
- **Problem**: The glossary.md and notation.md both mandate "CD-SSD (Coarse-Draft Self-Speculative Denoising)" as the canonical name, with IGSD explicitly listed under Banned Terms: "Do NOT use 'IGSD' in the paper body." The method is named "IGSD" in every contribution heading and in the scope paragraph. This is a systematic rename miss. A reviewer reading this intro will encounter "IGSD" here, then find "CD-SSD" in the method section — appearing either as a careless inconsistency or an undisclosed rename. The contribution 4 description also uses the banned phrase "training-free, no-auxiliary-model speculative approach for MDMs" which the glossary replaces with "training-free, reduced-step speculative denoising concurrent with SSD and SSMD."
- **Fix**: Replace every occurrence of "IGSD" in the Introduction with "CD-SSD." On first use, expand: "CD-SSD (Coarse-Draft Self-Speculative Denoising)." Specifically: contribution 4 title → "**CD-SSD: Coarse-Draft Self-Speculative Denoising.**"; contribution 2/3 bodies → "M1 + CD-SSD"; scope paragraph → "CD-SSD serves primarily as a composability study vehicle"; Table 1 rows "IGSD (this work)" and "M1+IGSD (this work)" → "CD-SSD (this work)" and "M1+CD-SSD (this work)". Replace banned phrase in contribution 4 description with the glossary-approved formulation.

### Issue 2: 5.13× combined speedup is stated without arithmetic explanation — looks like a calculation error

- **Location**: Contribution 2.
- **Quote**: "Exactly one pair, M1 (EntropyCache, $\eta=2.0$) + IGSD ($\tau=0.9$, $T_{\text{draft}}=16$), achieves super-multiplicative synergy ($\text{Ortho} = 1.385$, 5.13× combined speedup, QAS = 1.654)."
- **Problem**: A reader computing M1 standalone (1.38×, stated in Table 1 and contribution 4) × CD-SSD standalone (3.40×, contribution 4) gets 4.69×, not 5.13×. The Ortho metric definition — which explains why the combined speedup exceeds the product — is introduced two paragraphs later, after the reader has already encountered a number that does not add up. Cross-checking against the experiments outline (outline.md §5.2): combined_speedup = 5.13×, individual product = 4.69×, Ortho = 1.385. The arithmetic is internally consistent, but the intro does not provide the vocabulary to verify it at the point where the claim appears.
- **Fix**: Add a one-line parenthetical immediately after "5.13× combined speedup": "(product of individual speedups: 1.38 × 3.40 = 4.69×; Ortho = 5.13/4.69 = 1.385 > 1.0, reflecting super-multiplicative synergy defined in Section 4.1)." This both pre-empts reviewer confusion and serves as a natural forward-reference to the metric definition.

### Issue 3: "35.1% accuracy retention on GSM8K" in Scope paragraph conflicts with experiments data

- **Location**: Scope and Honest Positioning paragraph.
- **Quote**: "its 35.1% accuracy retention on GSM8K is not deployable independently"
- **Problem**: The Experiments section (experiments.md, Table 2) reports CD-SSD at operating point (τ=0.9, T_draft=16): GSM8K AccRet = 0.637 (63.7%), MATH500 AccRet = 0.885 (88.5%). 35.1% is not traceable to any single-benchmark AccRet at the operating point in any verified results file. Possible explanations: (a) it is a stale value from an earlier experiment run, (b) it averages across all four benchmarks including degenerate HumanEval/MBPP (which would pull the mean down), or (c) it refers to a different parameter configuration. The intro and experiments sections report materially different AccRet for the same method and operating point — a discrepancy that reviewers will flag on any quantitative pass.
- **Fix**: Verify the source of 35.1%. If it is a multi-benchmark average including degenerate coding benchmarks, state that: "CD-SSD's mean accuracy retention across all four benchmarks (GSM8K, MATH500, HumanEval, MBPP) is 35.1%, pulled down by near-zero coding performance; reasoning-task retention is 63.7% (GSM8K) and 88.5% (MATH500)." If it is a stale value, replace with the verified figure from igsd_pareto_full.json. Do not leave an inconsistency between the intro and the experiments section at this specific number.

---

## Major Issues

### Issue 4: "one-third the throughput of a comparable AR model" — no citation, no model named

- **Location**: Paragraph 1, closing sentence.
- **Quote**: "roughly one-third the throughput of a comparable AR model under similar conditions."
- **Problem**: No AR model is named, no measurement is cited, neither "comparable" nor "similar conditions" is defined. LLaDA-8B-Instruct at 31.0 TPS on a 97 GB VRAM GPU is in a specific hardware context that does not readily compare with published AR throughputs from different papers. A reviewer will request the specific AR model and measurement source. The claim is not falsifiable as stated.
- **Fix**: Either (a) measure Llama-3.1-8B-Instruct under identical conditions (same GPU, bf16, same benchmark) and cite: "versus ~X TPS for Llama-3.1-8B-Instruct [CITE:llama3] on the same hardware"; or (b) hedge without measurement: "substantially below the inference throughput of comparably-sized autoregressive models [CITE:llada, CITE:vllm]." Option (a) is preferred and would take one experiment to verify. Without evidence, this sentence should be softened to avoid being a credibility liability in the opening paragraph.

### Issue 5: SSD and SSMD absent from speculative decoding enumeration — directly weakens "missing method class" motivation

- **Location**: Paragraph 2, speculative decoding methods.
- **Quote**: "Two speculative decoding approaches exist: DualDiffusion [CITE:dualdiffusion], which uses an external draft model within the MDM regime, and DFlash [CITE:dflash]..."
- **Problem**: SSD (Gao et al., arXiv:2510.04147) achieves 3.46× lossless self-speculative speedup — using the same model, no training — and is explicitly identified in Section 4.2 as concurrent work directly comparable to CD-SSD. SSMD (Campbell et al., arXiv:2510.03929) is also concurrent. Both are omitted from the intro's enumeration. This creates two problems: (1) a reviewer familiar with SSD will see an obvious gap and question the literature coverage; (2) the "missing method class" motivation ("No training-free, no-auxiliary-model speculative approach for MDMs exists") is factually wrong — SSD fills exactly this gap. The contribution claim "IGSD fills the gap left by DualDiffusion and DFlash" collapses if SSD is acknowledged.
- **Fix**: Add SSD and SSMD to the speculative decoding enumeration: "Self-Speculative Decoding (SSD) [CITE:ssd] achieves 3.46× lossless speedup using the same model; SSMD [CITE:ssmd] uses an attention-mask switch; both are concurrent with this work." Then revise the "missing method class" gap to focus on composability: "No prior work tests whether self-speculative methods compose with KV-cache or AR-guidance methods — which is the gap CD-SSD is designed to fill as a composability study vehicle." This is a stronger and more accurate motivation than claiming a method class is missing when it is not.

### Issue 6: Contribution #2 uses "binary composability" as universal law — banned by glossary

- **Location**: Contribution 2 title and opening sentence.
- **Quote**: "**Binary composability finding.** MDM composability is not a gradient — it is binary."
- **Problem**: The glossary explicitly bans "'binary composability' (as universal law)" with required replacement: "'binary pattern observed across the three method pairs evaluated'." The claim "MDM composability is not a gradient — it is binary" asserts a general property of all MDM composability from a 3-pair, 2-seed experiment. A reviewer will challenge this as overgeneralization, correctly noting that the finding is confined to the three specific pairs studied under one model and hardware configuration.
- **Fix**: "**Binary composability pattern.** Among the three feasible method pairs evaluated, composability is binary: exactly one pair (M1+CD-SSD) achieves super-multiplicative synergy; both remaining pairs destructively interfere. Whether this pattern generalizes beyond these three pairs is an open question (Section 6.3)."

### Issue 7: Section roadmap mis-numbers sections relative to paper structure

- **Location**: Roadmap paragraph (final prose before Table 1 note).
- **Quote**: "Section 2 formalizes... Section 3 provides experimental results... Section 4 analyzes... Section 5 positions this work against related acceleration methods. Section 6 concludes."
- **Problem**: The outline places Methods at Section 4, Experiments at Section 5, Discussion at Section 6, Related Work at Section 7, and Conclusion at Section 8. The intro's roadmap compresses the structure to 6 sections, shifting Related Work to Section 5 and Conclusion to Section 6. A reader navigating from the intro's roadmap to the actual table of contents will find inconsistent numbering. The actual section headers in the existing section files use their own numbering (experiments.md opens as "# 5. Experiments") which currently aligns with the 8-section outline, not the 6-section roadmap.
- **Fix**: Update the roadmap to match the actual section numbering: "Section 2 surveys related work. Section 4 formalizes the composability framework... Section 5 reports all experimental results... Section 6 analyzes the frozen-token synergy mechanism and limitations. Section 8 concludes." Or, if the paper will use a compressed 6-section structure, update the outline and all section headers consistently before the roadmap is written. One must match the other before submission.

---

## Minor Issues

- **Table 1, Saber row, Published Speedup "251.4%"**: All other entries use ×-multipliers. "251.4%" will read as a percentage increase (= 3.51× speedup), not a ratio. Fix: "2.51×" or "~3.51×" depending on what "251.4%" means in the original paper. Add a footnote if the source uses different units.

- **Table 1, "IGSD (this work)" and "M1+IGSD (this work)" rows**: Use banned name. Fix to "CD-SSD (this work)" and "M1+CD-SSD (this work)."

- **Table 1 note cross-reference**: "See Section 6.3 for detailed discussion." After section renumbering, verify this cross-reference resolves correctly. Currently, the outline places Limitations at Section 6.3 (of Discussion), which would be correct if Discussion is Section 6 — but the roadmap maps Discussion to Section 4, creating a contradiction.

- **Contribution 1, "First systematic pairwise composability study"**: The word "First" approaches the banned pattern "To the best of our knowledge, this is the first work to..." The glossary does not explicitly ban this formulation but the Writing Quality Rules flag it as hollow self-praise when not backed by evidence. Add a parenthetical anchor: "no prior work measures pairwise Ortho across MDM acceleration families; see Section 5 for the closest related work (Kolbeinsson et al., 2024, on LLM intervention composability)."

- **Contribution 5, accuracy retention figures**: "accuracy retention collapses to 24–28% at 4×–8× step-jump." The Experiments section reports J=4 AccRet = 0.130 (13.0%), J=8 AccRet = 0.243 (24.3%). The intro's "24–28%" misses J=4's catastrophic 13% result and appears to be a benchmark-averaged or MATH500-inflated figure. Using the GSM8K-specific values (which are worse) would strengthen the claim: "collapses to 13% AccRet at J=4, reaching only 24% even at J=8."

- **Para 1, $O(N^2)$ forward pass**: N is not defined in the Introduction. Notation.md defines N as sequence length. Add "where N is the sequence length" on first use or replace with "a full quadratic-complexity forward pass over the entire sequence."

- **FreeCache vs. FlashDLM citation conflict**: The intro lists "FreeCache [CITE:flashdlm]" among KV-cache methods, but [CITE:flashdlm] is used in the Method section for M3 (AR-guided unmasking, FlashDLM's guided diffusion feature). If the same paper (FlashDLM) contributes both a KV-cache component and an AR-guidance component, credit it once in the appropriate category with a note, or use distinct cite keys. Otherwise a reviewer will flag the same citation appearing in two different method categories.

- **Figure 1 caption**: The inline alt text "Teaser: composability landscape and speed–quality scatter" does not describe the two-panel structure. Final caption should describe both panels: "(a) Ortho scores for all three pairs with threshold line at 1.0; (b) Speedup vs. AccRet scatter showing individual methods and combined pairs."

---

## Visual Element Assessment

- [x] Table 1 present and referenced — matches outline plan
- [x] Figure 1 referenced in body text ("As Figure 1 summarizes...")
- [ ] Figure 1 referenced BEFORE it appears in the file — FAIL: in the markdown draft, the figure `![...]` tag appears after the prose sentence that says "As Figure 1 summarizes." In LaTeX layout this will be resolved by float placement, but the markdown ordering is inverted. Move the figure tag to precede the referencing sentence.
- [ ] Figure 1 caption is self-explanatory — FAIL: "Teaser: composability landscape and speed–quality scatter" does not tell a reader what the two panels show without reading the body text. Update to describe both panels explicitly.
- [x] No text-heavy blocks lacking visual support (introduction correctly uses Table 1 + Figure 1)

---

## What Works Well

1. **Opening with hardware-specific throughput** (31.0 ± 4.0 tokens/s on GSM8K, NVIDIA RTX PRO 6000 Blackwell, 97 GB VRAM): Exactly the right concrete lead. This anchors the problem in real deployment cost rather than abstract motivation. The hardware specificity signals experimental rigor.

2. **Scope and Honest Positioning paragraph**: This paragraph is exceptional. Pre-empting "why is your 5.13× lower than published 15–26×?" with "this is an analysis paper; M1's 1.38× vs. published 15–26× reflects an implementation gap; the Ortho metric remains valid as a ratio" directly defuses the strongest adversarial reviewer objection. Very few ML papers include this level of self-assessment. Preserve the structure verbatim after fixing the IGSD→CD-SSD rename.

3. **Contribution 3, mechanistic explanation**: "tokens in $S_{\text{accept}}$ ($\alpha \approx 52\%$ at $\tau = 0.9$) are frozen — their identities do not change across all $T_{\text{full}} = 64$ refinement steps. EntropyCache assigns entropy $H_i = 0$ to these frozen positions, guaranteeing cache hits throughout the refine phase." This is specific enough that a reviewer can trace the reasoning through the entropy formula definition in Section 4. It is precisely the level of technical density that separates a 7-score intro from a 5-score one.
