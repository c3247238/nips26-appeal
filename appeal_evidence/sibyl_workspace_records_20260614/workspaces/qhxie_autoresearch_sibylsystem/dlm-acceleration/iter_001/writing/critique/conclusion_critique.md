# Critique: Conclusion

## Summary Assessment
The Conclusion (Section 8) contains strong mechanistic content in §8.1 and commendably honest limitations in §8.2. However, it remains unrevised relative to the previous critique round: all three critical issues from the prior review persist — the banned term IGSD appears 8+ times, the M2 J=4 accuracy figure is wrong (28% stated vs. 13.0% confirmed in Table 2), and the CD-SSD standalone AccRet is cited as 35% rather than the 63.7% primary benchmark result. Additionally, §8.3 presents as unresolved an experiment (tau=0.0 paradox) that §6.3 already resolves, and §8.4 is a near-verbatim restatement of §6.5. These are not style issues — they are factual contradictions between sections that will cause immediate reviewer rejection under a scrutiny pass.

## Score: 6/10
**Justification**: The mechanistic chain in §8.1 sub-bullet 2 and the calibrated limitations in §8.2 demonstrate the scientific depth needed for a high-score conclusion. The section loses 4 points for: (1) IGSD used throughout (banned term, glossary mandates CD-SSD), (2) two wrong numbers (M2 J=4 AccRet 28% vs. 13%, CD-SSD standalone AccRet 35% vs. 63.7%), (3) §8.3 tau=0.0 item framed as open when §6.3 resolves it, and (4) §8.4 that adds no content over §6.5. Reaching 8/10 requires fixing all four items; the mechanical fixes (1–2) are non-negotiable before submission.

---

## Critical Issues

### Issue 1: IGSD used throughout — banned term (8+ instances)
- **Location**: §8.1 all five sub-bullets; §8.3 paragraphs 1 and 4; sub-heading "IGSD fills a genuine gap"
- **Quote**: "M1 (EntropyCache, $\eta = 2.0$) combined with IGSD ($\tau = 0.9$, $T_{\text{draft}} = 16$) achieves..." and sub-heading "**IGSD fills a genuine gap.**" and "Information-Gain-Driven Self-Speculative Denoising is the first training-free, no-auxiliary-model speculative denoising method for MDMs."
- **Problem**: The glossary explicitly bans "IGSD" in the paper body and mandates "CD-SSD" (Coarse-Draft Self-Speculative Denoising). Additionally, the long-form name "Information-Gain-Driven Self-Speculative Denoising" has been replaced by "Coarse-Draft Self-Speculative Denoising" per the notation.md rename. The phrase "the first training-free, no-auxiliary-model speculative denoising method for MDMs" is explicitly banned; the glossary's Banned Terms table requires "training-free, reduced-step speculative denoising concurrent with SSD and SSMD." Every occurrence in the Conclusion is a cross-section inconsistency that reviewers will catch immediately when cross-checking against §4.
- **Fix**: (a) Replace every "IGSD" with "CD-SSD." (b) Replace "Information-Gain-Driven Self-Speculative Denoising" with "Coarse-Draft Self-Speculative Denoising." (c) Replace the banned first-of-its-kind claim with the glossary-compliant formulation. The sub-heading "IGSD fills a genuine gap" must become "CD-SSD fills a genuine gap" or be reworded to remove the banned term.

### Issue 2: CD-SSD standalone accuracy retention is wrong
- **Location**: §8.1, sub-heading "IGSD fills a genuine gap," sentence beginning "Although IGSD's standalone accuracy retention on reasoning (35% on GSM8K)..."
- **Quote**: "standalone accuracy retention on reasoning (35% on GSM8K) makes it unsuitable as a solo reasoning accelerator"
- **Problem**: Table 2 (Experiments §5.1) reports the 3-seed, full-benchmark primary result for CD-SSD at $\tau=0.9$, $T_{\text{draft}}=16$: GSM8K AccRet = **0.637** (63.7%, absolute accuracy 45.3%). The value 35% matches the ablation sub-experiment in Table 5 (2-seed, 200 GSM8K samples only). Citing the ablation figure as the primary operating-point result contradicts the primary table and creates a data discrepancy that reviewers who read §5.1 closely will flag. Note: the Introduction (§1) also states "35.1% accuracy retention on GSM8K" — this same error propagates there and requires a symmetric fix.
- **Fix**: Replace "35% on GSM8K" with "63.7% on GSM8K (absolute accuracy 45.3% vs. baseline 71.2%)." The argument that CD-SSD is "unsuitable as a solo reasoning accelerator" still holds at 45.3% absolute accuracy — state it in those terms. Audit all occurrences of "35%" and "35.1%" across all sections.

### Issue 3: M2 accuracy retention at J=4 is wrong by a factor of 2
- **Location**: §8.1, third sub-bullet: "at step-jump $J = 4$, accuracy retention collapses to 28%"
- **Quote**: "at step-jump $J = 4$, accuracy retention collapses to 28%; at $J = 8$, to 24%"
- **Problem**: Table 2 (Experiments §5.1) and Table 4 (failure mode atlas) both report M2 GSM8K AccRet at $J=4$ as **0.130** (13.0%). The outline §5.1 confirms "J=4x: GSM8K AccRet=13.0% (CATASTROPHIC)." The $J=8$ value (24%) is correct (Table 2: 0.243). Reporting 28% instead of 13% for $J=4$ halves the perceived severity of the step_starvation failure mode — 28% might suggest partial usefulness, while 13% is structurally catastrophic. This undermines the NO_GO verdict's credibility at exactly the moment the Conclusion is trying to crystallize it.
- **Fix**: Replace "at step-jump $J = 4$, accuracy retention collapses to 28%" with "at step-jump $J = 4$, accuracy retention collapses to 13.0%." Verify the source file (m2_pareto results) to confirm this is not a table transcription error elsewhere.

---

## Major Issues

### Issue 4: §8.4 "Broader Significance" restates §6.5 verbatim
- **Location**: §8.4, entire subsection
- **Quote**: "two trajectory-preserving methods should be tested for CHR synergy; any trajectory-modifying method should be treated as incompatible with all others by default until Ortho > 0.8 is demonstrated empirically. The detection signals (CHR < 40%, $\alpha > 0.75$, combined speedup < max individual speedup) are runtime-observable within the first 10–20 inference samples"
- **Problem**: This triple of detection signals (CHR < 40%, α > 0.75, combined speedup < max individual) appears in Discussion §6.5 with identical thresholds. The trajectory-preserving/modifying classification is developed in §6.1 and repeated here without new content. A conclusion section that re-summarizes the penultimate section adds zero information and costs the paper's final impression. The reader who reads §8.4 after §6.5 will perceive padding.
- **Fix**: Replace §8.4 with 3–4 sentences that establish the framework's *scope beyond this study* rather than restating what §6.5 already said. The key forward-looking contribution is that the composability framework (Ortho metric + failure mode atlas + trajectory taxonomy) constitutes a reusable evaluation protocol applicable to any future MDM acceleration method — a claim that §6.5 makes but does not emphasize as the paper's conceptual enduring contribution. Lead with that.

### Issue 5: §8.3 tau=0.0 paradox item is framed as unresolved when §6.3 already resolves it
- **Location**: §8.3, third bullet "Tau = 0.0 paradox resolution"
- **Quote**: "Resolving this paradox is necessary before claiming IGSD's partitioning step as a distinct contribution."
- **Problem**: Discussion §6.3 explicitly resolves this: "CD-SSD($\tau = 0.0$) and naive-T16 produce identical GSM8K accuracy... The confidence-partitioning machinery of CD-SSD adds zero measurable value over a plain 16-step denoising pass." Table 7 in §6.3 provides the decisive comparison. The Conclusion presenting this as open future work while §6.3 has already closed it is a logical inconsistency that will undermine reviewer confidence in the paper's internal coherence.
- **Fix**: Update this bullet to reflect the resolved status: "The tau=0.0 comparison (§6.3, Table 7) confirms CD-SSD's confidence-partitioning mechanism adds no standalone QAS benefit over plain step reduction (CD-SSD($\tau=0.0$) ≈ naive-T16, $-5.8\%$ QAS difference, within seed variance). Future work should investigate domain-specific $\tau$ calibration to recover the partition's theoretical benefit; the frozen-token synergy with M1 remains the partition's primary demonstrated value."

### Issue 6: Future work priorities are inverted relative to submission blockers
- **Location**: §8.3, item ordering
- **Problem**: The outline's pre-submission checklist marks "Full-scale M1+CD-SSD pairwise validation (3 seeds, GSM8K 1319 + MATH500 500)" as a **critical blocking item** — the paper's primary claim (Ortho=1.385) rests on 2-seed, 200-sample data. Yet §8.3 lists "SSD composability" first and the 3-seed validation second. A reviewer who has read §6.4 (where the 2-seed limitation is prominent) will be frustrated that the authors prioritize a follow-on experiment over validating the paper's own primary claim.
- **Fix**: Reorder §8.3 to lead with "Full 3-seed pairwise validation" as the highest-priority item needed to confirm the primary claim before final submission. State explicitly that this is a pre-submission requirement, not merely future work. Move SSD composability to second position.

---

## Minor Issues

- **§8.1, opening paragraph**: "composability is not a spectrum but a binary: exactly one method pair achieves super-multiplicative synergy" — nearly verbatim to Introduction contribution #2. Rephrase to convey what the binary *means* for deployment rather than restating that it was observed.
- **§8.1, fifth bullet parenthetical**: "(H4; M3 QAS = 1.582 for reasoning, IGSD QAS = 0.744 for coding, both consistent across seeds 42 and 123)" — uses banned term IGSD (should be CD-SSD) and the parenthetical disrupts paragraph flow. Integrate into a cleaner sentence or move to a footnote.
- **§8.1, "M3 must not be combined with any other method"**: Overstated. Table 6 shows M3+CD-SSD achieves QAS = 1.446 (interference, not zero). Revise to: "should not be combined with any other method; M3+CD-SSD achieves Ortho = 0.493, degrading QAS from 1.582 to 1.446 on reasoning benchmarks."
- **§8.2, missing benchmark scale context**: "200 GSM8K + 164 HumanEval samples" — Discussion §6.4 helpfully adds "15% of full benchmark scale." Including this fraction in §8.2 gives readers who read only the Conclusion the same calibration without requiring a cross-reference.
- **§8.3, item count**: "Four experiments remain open" — once the tau=0.0 paradox is updated to reflect its resolved status (Issue 5 above), the count drops to three. Update accordingly.
- **§8.3, SSD citation**: `[CITE:ssd]` placeholder — verify this resolves to the same citation key used in §3 and §6.4 (arXiv:2510.04147, Gao et al.). Inconsistent citation keys are a copy-editing failure reviewers notice at submission.
- **§8.2, CHR value inconsistency**: The Conclusion states "The cache hit rate climbs from 60% during standalone M1 to **96%** during the refine phase." Discussion §6.2 and notation.md report CHR_refine = 0.940 (94.0%). The 96% figure does not appear in any result table. Use 94.0% consistently.

---

## Visual Element Assessment

- [x] No figures planned — the outline's Figure & Table Plan marks "None" for the Conclusion. Correct for a conclusion.
- [x] No visuals are referenced in the section.
- [ ] Optional enhancement: a forward-reference to Figure 4 (Ortho bar chart) when summarizing the binary finding in §8.1 would anchor the quantitative claim to the paper's central visual, e.g., "As Figure 4 shows, Ortho = 1.385 for M1+CD-SSD vs. 0.493 and 0.301 for the two interfering pairs." Not required but would reinforce the visual's role.

---

## What Works Well

1. **§8.1 sub-bullet 2 (mechanistic explanation) is exceptionally rigorous for a conclusion.** The chain — tokens in $S_{\text{accept}}$ frozen at $\tau=0.9$, decoded entropy $H_i = 0$ for all frozen positions, guaranteed cache hits, CHR elevation 60% → 96%, resulting 9.4% Ortho premium above the multiplicative baseline — is fully traceable to Table 3 and §6.2. This level of mechanistic specificity in a conclusion section is rare and will distinguish the paper at review.

2. **§8.2 (Limitations) is scientifically honest and calibrated.** Reporting per-seed range [1.292, 1.478] for Ortho = 1.385 and explicitly flagging the 2-seed limitation in the same section that presents the primary finding demonstrates integrity. The three-point structure (statistical power, implementation gap, generalization) is well-ordered by severity. This will build reviewer trust.

3. **§8.1 deployment recipe (fifth bullet) is concise and decision-ready.** The three-rule structure (General: M1+IGSD/CD-SSD QAS=1.654; Reasoning: M3 w=0.3 QAS=1.675; Never: M2) with accompanying QAS values makes the section directly actionable without requiring table navigation. This is the Conclusion's most practitioner-useful element.
