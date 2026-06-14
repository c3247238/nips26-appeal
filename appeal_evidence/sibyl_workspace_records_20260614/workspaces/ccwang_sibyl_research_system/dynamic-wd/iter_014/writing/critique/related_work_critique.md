# Critique: Related Work (Section 2 — Background and Related Work)

## Summary Assessment

The related work section is well-organized and technically precise. It correctly identifies $\rho_t$ as the shared latent variable across four WD sub-traditions and provides concrete experimental numbers for most methods. The section is materially stronger than a typical background section because it makes a unification argument rather than simply summarizing prior work. However, it has a structural problem: it partially duplicates material that is more properly placed in the Method section (Sections 2.5–2.7 already appear verbatim or near-verbatim in Section 3), and several claims contain notation inconsistencies relative to `notation.md` and `glossary.md`. The transition from Section 2 to Section 3 is also undercut because Section 2 already announces the control-variable conclusion rather than leaving it as the setup for Section 3's formalization.

## Score: 6/10

**Justification**: The per-method descriptions are well-written with evidence and specific numbers. The major issues are (1) significant premature disclosure of the core argument (Sections 2.5–2.7 tell the reader the answer before Section 3 develops it), (2) an outline coverage mismatch (AdamS from the outline is never mentioned), (3) two notation deviations from `notation.md`, (4) one incorrect mapping in the CWD description that contradicts the Method section, and (5) the SWD description contains a claim inconsistent with the Method section's H1 fitting results. To reach 8/10, these must be resolved and the section must be trimmed so it motivates unification rather than announcing it.

---

## Critical Issues

### Issue 1: Premature disclosure undermines Section 3's contribution

- **Location**: Sections 2.5 ("Theoretical Foundations"), 2.6 ("Distinction from PIDAO"), and 2.7 ("The Shared Control Variable") — all three subsections
- **Quote**: "This is the key insight that identifies $\rho_t$ as the natural control variable for WD." (§2.5); "The four sub-traditions, despite their different formulations, all implicitly manipulate $\rho_t$" (§2.7); then each tradition's explicit mapping (§2.7 bullet list)
- **Problem**: The outline states the transition from Section 2 should be: "The fragmentation motivates a unified lens. Section 2 reviews each sub-tradition and identifies the shared control variable $\rho_t$." However, Section 2.7 already provides the full PID mapping that Section 3 is supposed to formalize. A reader who finishes Section 2 already knows: (a) the control variable is $\rho_t$, (b) SWD = P control, CWD = D control, CPR = I control, (c) the four traditions are unified. Section 3 then re-derives this, producing perceived repetition. The contribution of Section 3 — the unified control law with explicit $K_p$, $K_i$, $K_d$ mapping — is thus weakened before it appears.
- **Fix**: Section 2 should identify $\rho_t$ as a *pattern* across traditions (i.e., observe that each method happens to affect $\rho_t$) without claiming the full unification. Remove the PID-gain language from §2.7. Reframe §2.7 as: "Each tradition affects $\rho_t$ through a different lever. Section 3 formalizes this observation." Move the explicit gain mappings to Section 3 exclusively.

### Issue 2: SWD described as "proportional control" but empirically fails the unification test

- **Location**: Section 2.1, sentence 3: "In our framework, this corresponds to proportional control: $\lambda_t$ reacts to the current gradient magnitude --- a proxy for the control error $e_t = \rho_t - \rho^*(t)$."
- **Quote**: full sentence above
- **Problem**: The Method section (Table 2, H1 fitting) reports SWD has a 45.81% fitting error — explicitly classified as "FAIL" against the unified control law. The intro also reports 45.8% fitting error for SWD. The related-work section presents the SWD-as-P-control mapping as straightforward fact, but the actual empirical evidence contradicts this. A reviewer who reads both sections will flag the contradiction: Section 2 says SWD "corresponds to proportional control," Section 3 says the fit fails at 45.81%. This inconsistency damages the paper's credibility.
- **Fix**: In Section 2.1, qualify the claim: "SWD's global gradient-norm sensing is *broadly consistent* with proportional control, though Section 3 shows the per-layer fit has 45.81% error, indicating that SWD operates through mechanisms not fully captured by the per-layer $\rho_t^l$ feedback."

---

## Major Issues

### Issue 3: CWD alignment condition inverted relative to the Method section

- **Location**: Section 2.2, first sentence of CWD paragraph
- **Quote**: "CWD (Chen et al., ICLR 2026) applies a binary mask: WD is applied only when $\text{sign}(g_t) = \text{sign}(w_t)$ element-wise, meaning the gradient would move weights *away* from zero."
- **Problem**: The Method section (§3.2) states: "When $\alpha_t^l < 0$ (gradient opposes weight direction), the $-K_d \cdot \alpha_t^l \cdot e_t^l$ term *increases* $\lambda$, applying more decay precisely when the gradient already drives weights toward zero. This mirrors CWD's design rationale." The intro (§1, item 2) also states CWD applies "decay only when $\alpha_t < 0$." The related-work description ($\text{sign}(g_t) = \text{sign}(w_t)$, i.e., $\alpha_t > 0$) is the *opposite* condition. One of these is factually wrong about what CWD does. Additionally, the glossary defines CWD as "apply WD only when $\alpha_t < 0$," which contradicts the related-work sentence.
- **Fix**: Verify against the original CWD paper and correct to ensure consistent description across all sections. If CWD applies decay when $\alpha_t < 0$ (as stated in intro, method, and glossary), then the related-work description "when $\text{sign}(g_t) = \text{sign}(w_t)$" is incorrect and must be changed to "when $\text{sign}(g_t) \neq \text{sign}(w_t)$" (or equivalently, "when $\alpha_t < 0$").

### Issue 4: AdamS omitted despite being listed in the outline

- **Location**: Section 2.1 (WD Scheduling subsection)
- **Quote**: N/A — AdamS is absent
- **Problem**: The outline §2 "Key arguments" bullet explicitly lists: "WD scheduling (SWD, **AdamS**, ADANA)". The related work covers SWD and ADANA but never mentions AdamS. This is an outline coverage gap. AdamS is also referenced in the intro's fragmentation list by implication (ADANA is listed but AdamS is not). If AdamS is no longer relevant, the outline should be updated; if it remains relevant, it must be added.
- **Fix**: Either add a 1–2 sentence description of AdamS in §2.1, or explicitly update the outline to remove it and ensure the intro does not create an expectation of its coverage.

### Issue 5: Notation inconsistency — $\rho_t$ vs $\rho_t^l$ in Section 2

- **Location**: Throughout Sections 2.1–2.7 (multiple occurrences)
- **Examples**: "SWD senses $\|\nabla \mathcal{L}\|$ to modulate $\lambda_t$, reducing decay when gradients are large" (§2.1, no layer superscript); "Defazio's steady-state analysis implies WD drives $\rho_t$ toward a target" (§2.5, no superscript); "all implicitly manipulate $\rho_t$" (§2.7, no superscript); "the realized $\rho_t$ trajectory" (§2.7, no superscript).
- **Problem**: `notation.md` defines $\rho_t^l$ with superscript $l$ as the per-layer quantity: "$\rho_t^l = \|g_t^l\|_2 / (\|w_t^l\|_2 + \epsilon)$". The Method section uses $\rho_t^l$ consistently throughout. The related-work section alternates between $\rho_t$ (without superscript) and $\rho_t^l$ (with superscript), suggesting ambiguity about whether it is per-layer or global. A reviewer will ask whether Defazio's steady-state result applies per-layer or globally.
- **Fix**: Use $\rho_t^l$ whenever referring to the per-layer quantity; use $\rho_t$ only when explicitly referring to a global (averaged) quantity and state so explicitly. This is consistent with `notation.md`.

### Issue 6: BEM/CSI/AIS introduced in Section 2 without forward reference

- **Location**: Section 2.1, last sentence of SWD paragraph
- **Quote**: "SWD achieves 90.39 $\pm$ 0.19% on CIFAR-10/ResNet-20 in our experiments, close to FixedWD (90.68 $\pm$ 0.11%)."
- **Problem** (broader issue): The related-work section embeds experimental numbers from our own experiments (SWD: 90.39%, FixedWD: 90.68%, CWD: $\lambda_{\text{eff}} \approx 5 \times 10^{-5}$, CPR: 91.74%, CPR: $\lambda_{\text{eff}} \approx 10^{-3}$) throughout the per-method descriptions. These numbers belong in Section 6 (Experiments) or at most in Section 3 as setup. Presenting them in the related-work section before the experimental framework, metrics, or even the unified control law is established makes the section feel like it is cherry-picking numbers to build the narrative rather than reviewing the prior literature. More critically, the numbers use inconsistent metric context: the CPR number (91.74% ± 0.07%) is cited as if it is a summary statement about CPR the method, but it is actually our replication result under our experimental setup.
- **Fix**: In Section 2, remove our experimental numbers (move them to Section 6). Keep only numbers from the *original papers* (e.g., "CWD reports +0.61% on ImageNet ViT-S/16 in the original paper"). Add a forward reference: "We validate these characterizations in Section 6."

---

## Minor Issues

- **§2.1, ADANA citation**: "Ferbach et al., 2026" — year 2026 should be confirmed against the actual publication; if it is a preprint, cite as "2025" or "(preprint 2026)" consistently with how it appears in the intro.
- **§2.2, GWA description**: "While GWA does not directly modulate WD, it provides the theoretical basis for using $\alpha_t$ as a feedback signal" — the phrase "provides the theoretical basis" is unsupported in the text. Add a specific citation or result from GWA that substantiates this claim.
- **§2.3, first sentence**: "The decoupled WD tradition addresses the mathematical distinction between $L_2$ regularization and WD" — the glossary preferred phrase is "weight decay" not "WD" on first use in a section. This subsection uses both inconsistently.
- **§2.4, last sentence**: "AlphaDecay (He et al., 2025) assigns module-wise WD coefficients guided by spectral heavy-tailedness (HT-SR theory), scaling from 60M to 1B parameters." — No reference number is provided for HT-SR theory. Either cite the original Martin & Mahoney HT-SR paper or drop the theory acronym.
- **§2.7, bullet for "Alignment-aware methods"**: "CWD selectively disable WD for parameters where $\alpha_t > 0$" — This again states $\alpha_t > 0$, repeating the inversion from Issue 3. Must be corrected consistently.
- **§2.6, last sentence**: "PIDAO could be combined with our WD coefficient control for joint optimization." — "Our WD coefficient control" is the first informal self-reference. Use "the proposed framework" or "UDWDC" for consistency with the paper's terminology.
- **Page budget concern**: The outline notes that Background should be compressed to 1.5 pages in the final version. The current related-work section as written (including §2.5–2.7) is likely 3+ pages — nearly double the compressed budget. The plan to move §2.7 content to Section 3 and trim §2.5 will be essential for NeurIPS compliance.

---

## Visual Element Assessment

- [x] The outline's Figure & Table Plan lists no figures for Section 2 — this is correctly reflected (the section ends with `<!-- FIGURES - None -->`).
- [x] No tables are prescribed for Section 2 in the outline.
- [ ] Section 2.7 references the mapping that becomes Table 1 (in Section 3). If §2.7 survives in any form after the Issue 1 fix, a forward reference "see Table 1 in Section 3" should be added rather than restating the mapping inline.
- [ ] Consider whether a small inline table in §2.7 summarizing the four traditions (method, key mechanism, limitation) would aid reader orientation without pre-empting Section 3's control-law taxonomy. This would replace the current running prose in §2.7 with something more scannable.

---

## What Works Well

1. **Per-method descriptions are tight and evidence-backed** (§2.1–2.4): Each paragraph leads with the method's core mechanism, uses specific numbers where available, and connects to the paper's narrative without excessive padding. For example, the CPR description correctly characterizes the augmented Lagrangian as integral control with a specific effective-WD ratio ($10 \times$ baseline) — this is exactly the kind of concrete, load-bearing framing that reviewers appreciate.

2. **PIDAO distinction (§2.6) is unusually precise**: Rather than dismissing PIDAO with "our approach is different," the section explicitly states which control target differs (optimizer step vs. WD coefficient) and notes the methods are complementary. This is the right level of scholarly rigor and pre-empts an obvious reviewer question.

3. **The GWA and AdamO entries (§2.2) correctly frame indirect contributors**: GWA is acknowledged as theoretical underpinning rather than as a direct method competitor, and AdamO is distinguished as a structural (architectural) intervention versus a WD-coefficient modulation. These framings accurately position the related work without overclaiming.
