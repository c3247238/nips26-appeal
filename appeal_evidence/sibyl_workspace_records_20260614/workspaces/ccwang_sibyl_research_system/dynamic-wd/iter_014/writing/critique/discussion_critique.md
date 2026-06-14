# Critique: Discussion

## Summary Assessment

The Discussion section is substantive and unusually honest for an ML paper: it documents negative results, flags ablation gaps, and avoids overclaiming. The taxonomy argument in 7.1 is clear and well-reasoned. The primary weaknesses are (1) a data inconsistency between the Discussion and Experiments on the gain ablation numbers, (2) the section currently merges content that the outline designated as a separate "Limitations and Future Work" section (Sections 7.7 and 7.8 in the draft vs. the outline's Section 8), creating structural confusion for the reader and an overlong section relative to the 1.5-page budget, and (3) several passages use banned phrasing or lack specific numerical anchoring.

## Score: 7/10

**Justification**: The analytical content is solid and the honest negative result framing is a genuine strength. The score is held back by the data inconsistency in 7.1 (fitting error numbers in Discussion do not match Conclusion), the merge of Limitations into Discussion when the outline keeps them separate, the absence of any figure references (the outline plans Figure 7 for CSI in this section), and several minor style violations. Fixing the data inconsistency and adding cross-references to specific figures/tables would push this to 8.

---

## Critical Issues

### Issue 1: Gain ablation numbers contradict Experiments section

- **Location**: Section 7.6, negative result #1 (line 41)
- **Quote**: "by 1.26 pp on CIFAR-100 ablation (FixedWD 70.53% vs. Kp_only 68.52%)"
- **Problem**: The difference is 70.53 - 68.52 = 2.01 pp, not 1.26 pp. The Experiments section Table 4 reports FixedWD at 70.53% and Kp_only at 68.52%, and the Discussion itself quotes those same numbers — yet states the gap as 1.26 pp. This arithmetic error will be caught by any reviewer checking the table.
- **Fix**: Change "1.26 pp" to "2.01 pp".

### Issue 2: Section 7.1 fitting error numbers are inconsistent with Conclusion

- **Location**: Section 7.1 (line 5) vs. conclusion.md (line 5)
- **Quote in Discussion**: "CWD (4.71% fitting error) ... SWD (45.81%) and DefazioCorrective (37.56%)"
- **Quote in Conclusion**: "The control law fits CWD's effective-WD trajectories to 4.71% error and CPR's to 9.57% ... Scheduling-based methods (SWD at 45.81%, DefazioCorrective at 37.56%)"
- **Problem**: The Discussion and Conclusion agree on these numbers. However, the Discussion omits the CPR fitting error (9.57%) in section 7.1 paragraph 1, while paragraph 2 mentions "CWD (4.71%) and CPR (9.57%)". Cross-checking with the outline (Section 7 key arguments) reveals the outline states fitting errors for CWD at "25.6%" and SWD at "40.7%" at 10 epochs, while the Discussion uses the 200-epoch numbers (4.71%, 45.81%) without clarifying this is the full-training result. The negative results section (7.6, item 2) then states "SWD (45.81%) and DefazioCorrective (37.56%) exceed the 20% error threshold" — this 20% threshold is never introduced in the section or cited from the Method. Readers cannot evaluate the threshold's significance without knowing its origin.
- **Fix**: In 7.6 item 2, add "(using the 20% fitting error threshold defined in Section 3.2)" or wherever the threshold is formally introduced. Verify the threshold is defined in the Method section before this forward reference.

---

## Major Issues

### Issue 3: Section 7.7 and 7.8 belong in a separate section per the outline

- **Location**: Lines 49–65
- **Problem**: The paper outline designates Section 8 as "Limitations and Future Work" (0.5 pages, separate from Discussion at 1.5 pages). The current draft folds Limitations and Future Work into Discussion as subsections 7.7 and 7.8. This creates two problems: (a) the compressed outline (Section "Discussion + Limitations + Conclusion" = 1.0 page) cannot accommodate this merged Discussion at its current length, and (b) readers expect Limitations to be a distinct section at NeurIPS/ICML venues. This is a structural decision that affects section numbering throughout the paper. The current discussion body up to 7.6 is the right length for a 1.5-page section; adding 7.7 and 7.8 pushes it to ~2.5 pages.
- **Fix**: Extract 7.7 and 7.8 into a standalone Section 8 "Limitations and Future Work" as the outline specifies. The Discussion should end after 7.6 with a forward reference: "Section 8 discusses limitations and future directions."

### Issue 4: Figure 7 (CSI comparison) is planned for Discussion in the outline but not referenced

- **Location**: Section 7.4 / 7.6 (CSI content) and outline Figure & Table Plan
- **Problem**: The outline's Figure & Table Plan designates Figure 7 ("CSI Comparison: UDWDC v1 vs v2") as appearing in "Experiments/Discussion". Section 7.4 (UDWDC Instability) is the natural home for this figure reference, yet the section contains no figure reference at all. The Experiments section already references this figure (Figure 7 in experiments.md line 132), but the Discussion's deep dive into CSI interpretation in 7.4 should either (a) re-reference Figure 7 or (b) include a new visualization. Without a visual anchor, the CSI=-2.41 and CSI>0.5 claims are abstract to readers who didn't absorb them in Section 6.
- **Fix**: Add "See Figure 7 for the CSI comparison across all methods." at the end of the first paragraph of Section 7.4.

### Issue 5: CWD confound (7.3) does not reference the corresponding data from Experiments

- **Location**: Section 7.3, second paragraph (line 19)
- **Quote**: "On CIFAR-10, CWD (90.32 ± 0.08%) underperforms FixedWD (90.68 ± 0.11%)"
- **Problem**: These numbers are drawn from Table 3 in the Experiments section, but the Discussion does not cite "Table 3". At NeurIPS, cross-references to tables are mandatory for reproduced numbers. Without the reference, a reviewer cannot quickly verify the claim.
- **Fix**: Change to "On CIFAR-10 (Table 3), CWD (90.32 ± 0.08%) underperforms FixedWD (90.68 ± 0.11%)".

### Issue 6: 7.5 Batch-Size section recommendation is imprecise and inconsistently stated vs. Experiments

- **Location**: Section 7.5, last sentence (line 37)
- **Quote**: "alignment-aware WD methods (CWD, any future continuous modulation) should be preferred at large batch sizes ($b \geq 512$) where the alignment signal is reliable. At small batch sizes ($b \leq 256$), the alignment noise may outweigh the information gain"
- **Problem**: The Experiments section (6.4) states the recommendation as "binary masking (CWD-style) is preferable at $b \leq 256$" — consistent. But the Discussion says alignment-aware methods should be *preferred* at $b \geq 512$, which is subtly the opposite claim: it says CWD is better at *large* batches, while the Experiments say CWD is preferable at *small* batches. These are inconsistent recommendations. Additionally, the 200-epoch CIFAR-10 data only has batch size 128 — the batch sweep is 10 epochs on CIFAR-100. Whether the recommendation holds for 200-epoch training is not established.
- **Fix**: Align the recommendation: "At small batch sizes ($b \leq 256$), binary masking (CWD-style) is preferred because the alignment signal is noisy and continuous modulation amplifies the noise. At large batch sizes ($b \geq 512$), the signal is reliable enough for continuous modulation (Figure 4), though the practical accuracy gain over CWD has not been validated in full-training experiments."

---

## Minor Issues

- **7.1, line 7**: "CPR achieves the highest accuracy on both CIFAR-10 (91.74%) and ImageNet (74.74%)" — cite "(Tables 3 and 5)" for traceability.
- **7.2, line 11**: "CIFAR-10 (10-epoch pilot), UDWDC achieves rank-1 BEM (55.87)" — the main CIFAR-10 results (Table 3, 200 epochs) show UDWDC v1 with BEM = -0.26. The Discussion cites pilot (10-epoch) BEM numbers, not the 200-epoch Table 3 numbers. This inconsistency is buried in a parenthetical. If the BEM ranking from pilot is the intended reference, clarify explicitly: "(10-epoch pilot, pre-Table 3 numbers)". If the 200-epoch numbers should be used, update to match Table 3.
- **7.3, line 21**: "We did not complete this ablation due to compute constraints, and flag it as essential future work." — The same point is repeated in 7.7 ("Budget-matched controls for ImageNet are limited to 2-epoch pilots"). Consider keeping only one location; the 7.7 limitations list is the more appropriate home.
- **7.4, line 29**: "UDWDC-v2's enormous WD budget (98599)" — lacks units and context. Add "(normalized units, 205,000x FixedWD's 0.48)" or whatever the actual ratio is. The number 98599 is hard for readers to contextualize without the comparison.
- **7.6, line 47**: "CWD has only 1 completed seed on ImageNet" — the outline states "5 seeds" for ImageNet methods, but Table 5 in Experiments shows CWD with only 1 seed. The Discussion acknowledges this limitation. However, "limiting the conclusions we can draw about CWD vs. UDWDC on ImageNet" understates the problem: with 1 vs. 3 seeds, no statistical test is valid. Be direct: "Statistical comparison of CWD and UDWDC on ImageNet is not valid with the current seed coverage and should be deferred to future work."
- **7.8, Future Work item 2**: "Testing $\delta_t^P$ vs. $\alpha_t$ as the CWD mask criterion" — $\delta_t^P$ is defined in notation.md as the geometry-corrected alignment for Adam, but $\alpha_t$ should be $\alpha_t^l$ (per-layer) per the notation table. Use the notation-consistent form.
- **Style**: Section 7.6 title "Honest Negative Results" uses an adjective that sounds self-congratulatory ("Honest"). All scientific results should be honest. Rename to "Negative Results" or "Limitations of the Framework".

---

## Visual Element Assessment

- [ ] Figures/tables match outline plan — **FAIL**: The outline plans Figure 7 (CSI comparison) for the Discussion, but the Discussion contains zero figure references.
- [ ] All visuals referenced before appearance — **N/A** (no figures referenced)
- [ ] Captions are self-explanatory — **N/A**
- [ ] No text-heavy sections that need visual support — **PARTIAL FAIL**: Section 7.4 is a dense technical argument about CSI values (-2.41, 0.5) across controller versions. Figure 7 from the Experiments is the direct visual support for this argument and should be cross-referenced here.

---

## What Works Well

1. **Section 7.3 (CWD Magnitude Confound)** is exemplary: it identifies a specific confound (50% WD reduction), quantifies it with exact numbers from both CIFAR (0.24 vs. 0.48) and ImageNet (0.26 vs. 0.52), proposes a concrete ablation (FixedWD at halved lambda), honestly acknowledges the ablation was not completed, and explains *why* (compute constraints). This is the kind of transparency reviewers at top venues reward.

2. **Section 7.4 (UDWDC Instability)** provides a clear mechanistic explanation of the failure mode (BN layers driving lambda toward minimum), not just the observation. The distinction between the v2 floor fix as an "engineering patch, not a principled solution" is the right framing and will be appreciated by reviewers who check whether the paper understands its own method's limitations.

3. **Section 7.1's three-way taxonomy** (open-loop / derivative / integral) is crisp and additive over what the Experiments section already covered. The one-sentence per-method summary at the end of paragraph 2 is well-constructed and would work as a memorable takeaway.
