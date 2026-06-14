# Critique: Conclusion

**Critic role**: Sibyl Section Critic
**Date**: 2026-03-18
**Revision round**: 4
**Section under review**: `writing/paper.md` §8 (Conclusion)
**Cross-references read**: paper.md (full), review.md (final review), prior conclusion_critique.md (Round 3)

---

## Summary Assessment

The Conclusion section (§8.1–§8.3) remains structurally sound and its epistemic calibration — conjecture framing, TOST power acknowledgment, scope qualifiers — continues to be among the paper's strongest elements. However, the final review has identified three blocking issues that the conclusion must address or explicitly acknowledge: (1) notation inconsistency with φ arguments, (2) potential inconsistency in the 18.3× effect-size ratio, and (3) the complete absence of all five planned figures. Rounds 2–3 identified additional issues (CIFAR-100 evidence gap, "established" overclaim, BN mechanistic caveat, Future Work ordering) that remain partially unresolved. This round prioritises the three critical issues from the final review and carries forward unresolved Round 3 issues.

## Score: 5/10

**Justification**: Downgraded from Round 3's 6/10 due to the final review surfacing a critical arithmetic inconsistency in the 18.3× ratio that propagates directly into §8.1's Finding #2, and the notation inconsistency (φ argument) that makes the conclusion's definition of the paper's core symbol inconsistent with its formal definition in §3.1. The section cannot be considered complete while the headline finding's arithmetic is unresolved or unacknowledged.

---

## Critical Issues (Final Review Priority)

### Issue FR-1: φ notation uses inconsistent argument signature

- **Location**: §8.1 opening paragraph and Finding #3
- **Quote**: "The shared `WDModulator` interface enables, for the first time, controlled comparison under identical optimizer internals with differences isolated to the modulation function $\varphi$."
- **Problem**: The conclusion never explicitly restates the φ argument signature, but the abstract and §1.4 — which the reader will have just read — define φ as φ(t, θ, g) using raw gradient g. The formal definition in §3.1 uses φ(t, θ, s_t) where s_t is optimizer state. The final review (Issue 4 in editor's issues) flags this as a [Major] inconsistency requiring global correction to φ(t, θ, s_t). The conclusion, which summarises the paper's contribution, must not reference φ in a way that reinforces the inconsistent g-based notation.
- **Status**: The conclusion does not worsen this inconsistency, but it also does not correct it. If the abstract and §1.4 are not fixed globally, any reader reaching §8.1 carrying the φ(t, θ, g) interpretation will continue in that misunderstanding.
- **Required action**: When the global fix is applied (standardising to φ(t, θ, s_t) per §3.1), verify that §8.1's reference to φ picks up the corrected notation. If the conclusion is revised for any reason, add the signature explicitly: "the modulation function φ(t, θ, s_t)."

### Issue FR-2: 18.3× ratio in §8.1 may be arithmetically inconsistent — must be resolved or acknowledged

- **Location**: §8.1, Finding #2
- **Quote**: "an effect-size ratio of approximately $18.3\times$ (Bootstrap BCa 95\% CI: [$12\times$, $29\times$]) relative to AdamW"
- **Problem**: The final review (editor's Critical Issue 1) identifies a blocking arithmetic inconsistency. Table 2 reports AdamW no_wd Cohen's d = 0.16; §6.2 reports SGD constant vs no_wd Cohen's d = 10.29; yet §6.2 derives the 18.3× ratio as 10.29 / d_AdamW. If d_AdamW = 0.16, then 10.29/0.16 = 64.3, not 18.3. The ratio 18.3× is arithmetically consistent with d_AdamW = 0.56 (from the pre-integration sections/experiments.md). The source of truth is not yet confirmed.

  The conclusion repeats the 18.3× figure without qualification. If the correct ratio is 64×, this is a factor-of-3.5 error in the paper's headline quantitative finding — visible in the abstract, §1.4, §4.3, §6.2, §7.1, and §8.1.

- **Three resolution paths**:
  1. **If d_AdamW = 0.56 is correct**: Update Table 2 to show 0.56 (not 0.16) for no_wd; the ratio 18.3× stands and the conclusion is correct as written.
  2. **If d_AdamW = 0.16 is correct**: The ratio must be updated to ~64× throughout, including the abstract, §1.4, §6.2, and §8.1. The bootstrap CI [$12×$, $29×$] would also need to be recomputed for the ~64× base.
  3. **If unresolved before final submission**: The conclusion should add an explicit caveat: "(Note: the AdamW Cohen's d denominator is being verified against the analysis JSON; the 18.3× figure will be confirmed or corrected in the camera-ready version.)" This is a last resort and is not recommended for submission.

- **Impact on §8.1**: The sentence beginning "SGD exhibits fundamentally different behavior, with... $18.3\times$" must be updated to reflect whichever ratio is arithmetically confirmed. The surrounding text (the BEM spectrum observation, the ρ regime boundary discussion) is unaffected.

### Issue FR-3: All five figures are [TODO] placeholders — the conclusion must be consistent with this state

- **Location**: §8.1, multiple findings
- **Problem**: The final review (editor's Critical Issue 2) confirms that no figures exist in writing/figures/ — all five are [TODO] placeholders. The conclusion in §8.1 references the visual evidence implicitly: "we established three central findings" relies on the reader accepting the statistical claims; the Discussion (§7.1, §7.2) references Figure 1 and Figure 2 for the visual summary of these claims. With no figures, the conclusion's repeated appeal to "87+ controlled experiments" is the sole visual anchor.

  The conclusion itself (§8.1–§8.3) does not directly reference figure numbers, so it is not technically internally inconsistent. However, the paper's visual communication weakness (3/10 in final review) means that the conclusion's confident summary tone — "we established three central findings" — is unsupported by visual evidence that reviewers expect. If figures are generated before submission, no change to the conclusion is needed. If figures remain absent, the conclusion's tone should be modestly hedged in one place to acknowledge that visual confirmation is pending.

- **Recommended action**: Generate Figures 1–5 using the existing `exp/results/` data. Figure 1 (violin/box plots showing the accuracy distribution across seven methods for both optimizers) is the single highest-priority visual — the conditional equivalence claim is far more credible visually than numerically alone. Figure 2 (AdamW vs SGD dual panel) is second priority for the 18.3× finding.

  If figures cannot be generated before the review deadline, add one sentence to §8.1's Finding #1: "Visual confirmation via distribution plots (Figure 1) is pending figure generation; the numerical evidence is presented in Table 2."

---

## Unresolved Issues from Round 3

### Issue C1 (Round 3): CIFAR-100 equivalence claim lacks supporting statistics (still unresolved)

- **Location**: §8.1, Finding #1
- **Quote**: "all seven weight decay strategies ... show no statistically detected difference on CIFAR-10 ... and CIFAR-100"
- **Status**: The parenthetical evidence is CIFAR-10 specific (< 0.3%, all p_adj > 0.05). CIFAR-100 accuracy range per Table 2 is ~0.76% (63.42 - 62.66), which is 2.5× the CIFAR-10 range. No separate CIFAR-100 p-values are given.
- **Required fix** (unchanged from Round 3): Either provide CIFAR-100 accuracy range and corrected p-values explicitly, or remove "and CIFAR-100" until substantiated.

### Issue C2 (Round 3): "Established" overclaims the evidence base (partially resolved)

- **Location**: §8.1, opening sentence
- **Current text**: "we **established** three central findings"
- **Status**: Per Round 3 critique, replace with "identified" or "observed, consistent with." The paper's current integrated draft still reads "established." This conflicts with §7.4 Limitation 1's honest acknowledgment that n=3 is insufficient for formal TOST equivalence.
- **Required fix**: Replace "established" with "identified" or "observed."

### Issue M1 (Round 3): Missing BN mechanistic caveat (still unresolved)

- **Location**: §8.1, Finding #2
- **Quote**: "This asymmetry is attributable to AdamW's implicit $\ell_\infty$ constraint, which absorbs weight decay perturbations in a mechanism absent from SGD."
- **Status**: The BN vs. ℓ∞ mechanistic ambiguity acknowledged in §7.2 is not carried into the conclusion. The conclusion makes a cleaner causal attribution than the evidence warrants.
- **Required fix**: Add one sentence: "We note that BN scale-invariance (D'Angelo et al., 2024) may contribute to or fully explain this invariance; disentangling the two mechanisms requires NoBN ablation experiments (Future Work bullet 1)."

### Issue M3 (Round 3): Future Work priority ordering corrected in paper but not in §8.2

- **Location**: §8.2 bullets 1–6
- **Status**: Round 3 recommended reordering to: (1) ρ regime sweep → (2) NoBN ablation → (3) n≥5 seeds → (4) large-scale validation → (5) optimizer generalisation → (6) formal proof. The current paper.md's §8.2 places "NoBN ablation" as bullet 1 and "ρ regime sweep" as bullet 2, which is an improvement over Round 3 but the formal proof bullet remains at position 6 (acceptable). Verify the current ordering matches the recommended sequence.

### Issue M4 (Round 3): 18.3× rho confound disclosure

- **Location**: §8.1, Finding #2
- **Status**: The ρ confound (SGD ρ = 0.05 vs AdamW ρ = 0.5, a 10× difference) is disclosed in the experiments §6.2 but not in the conclusion's summary. This is acceptable if the conclusion explicitly points to §6.2 for the qualified statement; however, if the conclusion is the only section a rushed reviewer reads closely, the confound will be missed.
- **Recommended addition**: Parenthetical "(though SGD and AdamW experiments use different λ values; see §6.2 for the ρ-confound discussion)."

---

## What Works Well

1. **Finding #3 (rho regime boundary) maintains correct epistemic status.** The conclusion uses "formalized" for the conjecture (Round 3 critique flagged this as an issue, but the current text is actually using "formalised these observations through the Phi Invariance Conjecture" which correctly names it a conjecture). The connection to Xie & Li (2024) and Defazio (2025) as "dual characterizations" via Theorem 1 is appropriately presented as a novel connection rather than a new proof.

2. **Future Work bullets are operationally specific and remain one of the paper's strongest elements.** The explicit parameter values for the ρ regime sweep (λ ∈ {5×10⁻⁵, 5×10⁻⁴, 5×10⁻³, 5×10⁻²} corresponding to ρ ∈ {0.05, 0.5, 5, 50}) signal a credible follow-up research agenda to reviewers.

3. **§8.3 Broader Impact is honest and principled.** "Knowing *when not to optimize* is as valuable as knowing *how to optimize*" remains the correct register for a paper with a conditional equivalence finding. The section avoids overclaiming.

4. **TOST power qualification is preserved.** Finding #1's "(n = 3; formal TOST equivalence requires n ≥ 5)" parenthetical is exemplary statistical honesty and should not be removed in any revision.

---

## Summary of Required Actions

| Priority | Issue | Action | Blocking? |
|----------|-------|--------|-----------|
| Critical (FR-2) | 18.3× arithmetic inconsistency | Verify correct AdamW d; update ratio or add caveat | Yes — cannot submit with unresolved arithmetic |
| Critical (FR-1) | φ(t, θ, s_t) notation | Apply global fix; verify conclusion picks up corrected notation | Yes — global fix required |
| Critical (FR-3) | All five figures are [TODO] | Generate Figures 1–5; if blocked, add pending-figure note to §8.1 | Yes — visual evidence required |
| High (C1) | CIFAR-100 evidence gap | Add CIFAR-100 range + p-values, or remove the claim | Yes — claim unsupported |
| High (C2) | "Established" overclaim | Replace with "identified" | No — but conflicts with limitations §7.4 |
| High (M1) | Missing BN caveat | Add one sentence on D'Angelo et al. confound | No — but mechanistic attribution is premature |
| Medium (M4) | 18.3× rho confound | Add parenthetical pointing to §6.2 | No |

---

*Reviewed by: Sibyl Section Critic*
*Review standard: NeurIPS/ICML main track*
*Revision round: 4 (incorporating Final Review critical issues)*
