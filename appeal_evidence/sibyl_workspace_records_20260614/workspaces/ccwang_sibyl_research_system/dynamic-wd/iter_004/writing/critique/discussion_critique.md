# Critique: Discussion

**Section**: 7. Discussion
**Reviewer**: Section Critic (NeurIPS/ICML/ICLR standard)
**Date**: 2026-03-18
**Revision round**: 4 (updated to address three critical issues from final review)

---

## Summary Assessment

The Discussion is the paper's most intellectually candid section, and Section 7.4 (Limitations) is genuinely exemplary by conference standards. The four-subsection structure is clear and practitioner-friendly. However, a structural contradiction runs through the entire section: 7.1 and 7.2 make assertive, broad claims, while 7.4 honestly documents the evidentiary deficiencies that should have qualified those claims inline. The caveats are quarantined rather than integrated.

This revision round focuses on three critical issues flagged in the final review: (1) the 18.3× arithmetic consistency between Discussion and Experiments, (2) φ notation consistency in the Discussion section, and (3) the framing of Conjecture C1 (Phi Invariance) as "empirically supported only in Regime I."

## Score: 5.5/10

**Justification**: The section has strong bones: correct identification of boundary conditions, honest statistical disclosure, good prior-work connections. Score held from round 3 because the three critical issues from the final review remain unresolved in the current text, and represent blocking inconsistencies for submission. Detailed findings follow.

---

## Critical Issues (from Final Review)

### Issue FR-1: 18.3× Ratio Arithmetic — INCONSISTENCY CONFIRMED IN DISCUSSION

**Location**: Section 7.1, boundary conditions for SGD; also Section 6.2 Key Finding 3
**Status**: CRITICAL — arithmetic inconsistency confirmed after checking source data

**Finding from Analysis JSON** (`exp/results/analysis/sgd_baseline_analysis.json`):
- SGD constant vs. no_wd Cohen's d (CIFAR-10): **10.2867** (confirmed from effect_sizes array)
- AdamW no_wd Cohen's d (Table 2 in paper, line 295): **0.16**
- Arithmetic: 10.29 / 0.16 = **64.3×**, not 18.3×

**Discussion says (§7.1)**:
> "Our experiments establish that weight decay *presence* matters under SGD ($\Delta = 0.92\%$, $d = 10.29$, $p_{\text{adj}} = 0.002$ for constant vs.\ no\_wd; **18.3×** effect-size ratio vs.\ AdamW)."

**Experiments (§6.2) says**:
> "Comparing constant vs.\ no\_wd across optimizers: SGD $\Delta = 0.92\%$, $d = 10.29$; AdamW $\Delta = 0.05\%$, $d = 0.16$; ratio $\approx$ **18.3×**"

Both the Discussion (§7.1) and Experiments (§6.2) report 18.3×, so they are **internally consistent with each other** — but both are arithmetically wrong given the reported AdamW $d = 0.16$.

**The source of the inconsistency**: The pre-integration draft (`sections/experiments.md`) used AdamW $d = 0.56$ to arrive at 18.3× (10.29/0.56 = 18.37). This is arithmetically correct. When integrated into paper.md, Table 2 reports $d = 0.16$ for AdamW no_wd, but the 18.3× figure was carried over unchanged. The Discussion correctly copies 18.3× from §6.2, so it is not the origin of the error — but it perpetuates the same inconsistency.

**Fix required (in both §6.2 and §7.1)**:
- Option A: If AdamW $d = 0.56$ is correct, update Table 2's AdamW no_wd Cohen's d from 0.16 to 0.56. The 18.3× figure and Bootstrap CI would remain as-is.
- Option B: If AdamW $d = 0.16$ is correct (e.g., from paired vs. unpaired formula), the ratio should be updated from 18.3× to approximately 64×, with the Bootstrap CI regenerated.

**Verification note**: The accuracy delta for AdamW constant (90.13%) vs. no_wd (90.08%) is 0.05%, which at std ≈ 0.30–0.35% gives an unpaired Cohen's d ≈ 0.14–0.17. A paired Cohen's d of 0.56 would require std_diff ≈ 0.08–0.09%, meaning the three seeds all move in the same direction — plausible but needs to be confirmed from the raw seed data. Resolution requires running the paired t-test calculation explicitly on seed-level results.

**Impact on Discussion specifically**: The Discussion perpetuates the same error as §6.2. Until §6.2 is corrected, the Discussion number will need to be updated to match the corrected §6.2 value. The Discussion's mention of ">10×" amplification (Issue M3 from round 3, see below) also contradicts the claimed 18.3×.

---

### Issue FR-2: φ Notation in Discussion — MOSTLY CONSISTENT, ONE RESIDUAL ISSUE

**Location**: Section 7.2
**Status**: MINOR — Discussion uses φ consistently with the formal definition in most places, with one residual problem inherited from the final-review's broader notation audit

**Analysis of φ usage in the Discussion section**:

The formal definition (§3.1) establishes φ: ℤ≥0 × ℝ^d × S → ℝ^d≥0 where s_t denotes optimizer state that "may include raw gradients g_t, preconditioned updates u_t, or other statistics." The Discussion uses φ as follows:

- §7.2 "No-WD equivalence": "$\varphi = 0$" — correct shorthand for the constant zero function
- §7.2 "No-WD equivalence": "$\varphi$ values near 1.0" — appears in the CSI paragraph; actually this refers to CSI values, not φ values directly. No error.
- §7.2 elsewhere: no direct mentions of φ argument structure

**The residual inconsistency is in the abstract and §1.4, not in the Discussion**:
- Abstract: "$\varphi(t, \boldsymbol{\theta}, \mathbf{s}_t)$" — correct, matches formal definition
- §1.4 (Contribution 1): "$\varphi(t, \boldsymbol{\theta}, \mathbf{s}_t)$" — correct
- The final review flagged the abstract as using $\varphi(t, \theta, g)$ — this was **incorrect in the review**. The current abstract in paper.md does use $\mathbf{s}_t$, not $g$. The notation is already consistent between the abstract and formal definition.

**One residual issue found**: The BEM definition in §3.4 (line 154) uses $\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)$ with $\mathbf{g}_t$ instead of $\mathbf{s}_t$:
> $\sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]$

This is a notation inconsistency: the formal definition uses $\mathbf{s}_t$, but BEM uses $\mathbf{g}_t$. This is in §3.4, not the Discussion, but BEM is referenced in §6.4 and §7.1 (Decision heuristic step 2: "Use BEM to verify..."). If the inconsistency is noticed in the BEM definition, it reflects back on the Discussion's use of BEM as a tool.

**Fix**: The BEM integral in §3.4 should use $\mathbf{s}_t$ or add a parenthetical "where $\mathbf{s}_t$ may include $\mathbf{g}_t$, $\mathbf{u}_t$, or both" to clarify. The Discussion itself does not need φ notation changes.

**Conclusion on FR-2**: The Discussion section's own use of φ notation is consistent with the formal definition. The broader paper has one residual φ argument inconsistency (in §3.4 BEM formula), but this is not a Discussion section error.

---

### Issue FR-3: Conjecture C1 Framing as "Empirically Supported Only in Regime I" — CLEAR AND ACCURATE

**Location**: Section 8.1 (Conclusion), §7.4 Limitation 5, §4.2 Conjecture 1
**Status**: ACCURATE — the framing is appropriately calibrated

**Quote from Conclusion (§8.1, line 436)**:
> "The conjecture is currently **empirically supported only in Regime I** at $\rho \approx 0.5$; Regimes II and III await the $\lambda$ sweep."

**Verification of accuracy**:
- Regime I (ρ ≤ ρ₁ ≈ 1): Tested at ρ = 0.5 for AdamW. The paper provides empirical results at this point. ✓ Supported
- Regime II (ρ₁ < ρ < ρ₂): No experiments reported. The paper explicitly states "Regimes II and III are entirely untested" in §7.4 Limitation 5. ✓ Not supported
- Regime III (ρ ≥ ρ₂): No experiments reported. ✓ Not supported
- SGD results: SGD has ρ_SGD = 0.05, which falls in Regime I by the conjecture's own bounds (ρ₁ ≈ 1). SGD showing sensitivity to weight decay *presence* is explained as SGD lacking AdamW's ℓ∞ mechanism — this is about optimizer type, not ρ value. The conjecture's scope is explicitly AdamW, so SGD behavior is neither support nor counter-evidence for Regimes II/III.

**Is the framing clear?** Mostly yes. One subtle ambiguity exists: "empirically supported only in Regime I" correctly excludes Regimes II and III from support, but a casual reader might interpret "only in Regime I" as meaning the conjecture was *tested and failed* in other regimes. The more precise framing is "empirically tested only at ρ = 0.5 (Regime I); Regimes II and III are untested."

**§7.4 Limitation 5 language**:
> "The Phi Invariance Conjecture is supported by empirical evidence at $\rho = 0.5$ and motivated by theoretical arguments (Lemmas 1--3), but a formal proof is not provided. **Regimes II and III are entirely untested.**"

This is accurate and clear. The Conclusion's phrasing "empirically supported only in Regime I" is consistent with this.

**Fix recommended (minor clarity improvement)**: In §8.1 (Conclusion), change:
> "The conjecture is currently empirically supported only in Regime I at $\rho \approx 0.5$"

To:
> "The conjecture has been empirically tested only in Regime I at $\rho = 0.5$; Regimes II and III are entirely untested"

This eliminates the ambiguity about whether "only in Regime I" means tested-but-failed elsewhere vs. only tested there.

---

## Critical Issues (Carried Forward from Round 3)

### Issue C1: Opening boldface claim overstates the evidence

- **Location**: Section 7.1, first paragraph
- **Quote**: "under standard AdamW settings ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$, batch-normalized architectures), **no benefit to adopting dynamic weight decay strategies has been detected**"
- **Status**: Round 3 recommended change to "no detected benefit" with power caveat. Current text partially addresses this by adding "at CIFAR scale with ResNet-20" but still uses the assertive form without inline power caveat.
- **Fix**: Add after "statistically indistinguishable results": "(n=3; formal TOST at n≥5 pending; minimum detectable effect ≈0.77% at 80% power; see Section 7.4 Limitation 1)"

### Issue C2: CWD recommended for SGD users — PARTIALLY ADDRESSED

- **Location**: Section 7.1, decision heuristic step 3
- **Current text (revised)**: "If $\rho > 1$ or using SGD: ensure weight decay is non-zero (especially for SGD). Consider dynamic strategies, but note that specific strategy benefits in this regime are unconfirmed in the current paper."
- **Status**: Round 3 critique has been partially implemented. The current text no longer explicitly recommends CWD as "principled starting point." The fix is complete.

### Issue C3: Limitations quarantined — PARTIALLY ADDRESSED

- **Status**: §7.1 now includes more hedging than the previous round ("no benefit... has been detected"). The structural problem persists: §7.4 uses "consistent with rather than proven by" language, while §7.1 still uses assertive active-voice claims. Full integration of hedges requires adding the inline power caveat to §7.1 (see C1 fix).

---

## Major Issues (Carried Forward from Round 3)

### Issue M1: "Theorem 1" in Discussion — UNRESOLVED

- **Location**: Section 7.2, "Dual characterization" paragraph
- **Quote**: "Xie \& Li's (2024) constraint radius $\tau^* = \eta/\lambda = 1/\rho$ and Defazio's (2025) steady-state gradient-to-weight ratio $R_* = \lambda/\eta = \rho$ are dual characterizations of the same quantity (Theorem 1)."
- **Status**: The text-level fix was applied in §4.1, where "Theorem 1" was downgraded to "Observation 1 (Dual Characterization)" in the methods section. However, §7.2 still refers to "Theorem 1" rather than "Observation 1." This creates a naming inconsistency between the two sections.
- **Fix**: Change "Theorem 1" in §7.2 to "Observation 1" to match §4.1.

### Issue M2: BN confound framed as "open question" — PARTIALLY ADDRESSED

- **Location**: Section 7.3, "Mechanistically unresolved" paragraph
- **Current text**: "Whether AdamW's $\ell_\infty$ constraint alone is sufficient for invariance, or whether BN scale-invariance is a necessary co-factor, is the **single highest-priority open question**... Critically, establishing the mechanism is required before the explanatory claims in Section 4 can be considered confirmed."
- **Status**: Improved from round 3. The phrase "single highest-priority open question" and the statement that NoBN ablation is "required before... explanatory claims... can be considered confirmed" are stronger framings. However, "open question" still understates the issue as a *blocking ambiguity* for the theoretical interpretation.
- **Fix**: Replace "is the single highest-priority open question" with "is a fundamental ambiguity in the current paper's mechanistic interpretation — one that must be resolved before the ℓ∞ mechanism explanation in Section 4 can be confirmed."

### Issue M3: ">10×" inconsistent with "18.3×" — UNRESOLVED

- **Location**: Section 7.1, boundary conditions for SGD
- **Quote**: "Our experiments show a $> 10\times$ effect-size amplification under SGD compared to AdamW for weight decay presence."
- **Status**: This still reads ">10×" while the paper's central claim is "18.3×." These are simultaneously inconsistent (since 18.3 > 10, the claim is technically true but deliberately imprecise) and confusing to readers who expect the specific value.
- **Fix**: Replace ">10×" with the exact figure: "approximately $18.3\times$ (Bootstrap BCa 95\% CI: [$12.1\times$, $28.7\times$])". Then add the FR-1 caveat about the arithmetic inconsistency once that is resolved.

### Issue M4: Decision heuristic treats ρ=1 boundary as validated — PARTIALLY ADDRESSED

- **Location**: Section 7.1, decision heuristic steps 2–3
- **Current text**: Step 3 now says "Consider dynamic strategies, but note that specific strategy benefits in this regime are unconfirmed in the current paper."
- **Status**: The explicit caveat about Regime II/III unconfirmed status is better. Still missing: explicit labeling of the ρ=1 boundary as "predicted, not empirically located."
- **Fix**: Add to step 2: "The ρ<1 threshold is the conjectured Regime I/II boundary (ρ₁ ≈ 1), tested at ρ=0.5 only."

### Issue M5: CSI absent from Discussion — ADDRESSED

- **Location**: Section 7.2, "CSI and coupling stability" paragraph (new in current draft)
- **Status**: Resolved. The current §7.2 includes an explicit CSI paragraph: "CSI values near 1.0 for all methods in Regime I reflect the uniform stability of weight decay coupling under AdamW's implicit constraint. The small variation in CSI (0.98--1.03) confirms that no strategy destabilizes optimizer coupling at ρ=0.5..."
- **Verdict**: Issue M5 from round 3 has been addressed.

---

## Summary of Three Critical Issues for Final Review

| Issue | Status | Action Required |
|-------|--------|-----------------|
| FR-1: 18.3× arithmetic | **INCONSISTENCY CONFIRMED** | Discussion correctly mirrors §6.2, but both are arithmetically wrong (10.29/0.16 = 64.3, not 18.3). Must resolve whether AdamW d = 0.16 or 0.56; update Table 2 and both §6.2 and §7.1 consistently. |
| FR-2: φ notation in Discussion | **CONSISTENT** (minor issue in §3.4 BEM formula, not Discussion) | No Discussion-section change required. Fix BEM integral in §3.4 to use s_t notation. |
| FR-3: "Empirically supported only in Regime I" framing | **ACCURATE** (minor clarity improvement possible) | Optional: replace "supported only in Regime I" with "tested only at ρ=0.5 (Regime I)" in §8.1 for precision. |

---

## Cross-Section Consistency Check (Updated)

| Item | Discussion | Other Section | Status |
|------|-----------|---------------|--------|
| 18.3× ratio | "18.3×" (§7.1) | "18.3×" (§6.2) | Internally consistent — but both are arithmetically wrong vs. Table 2 d=0.16 |
| AdamW no_wd Cohen's d | Not stated in Discussion | d=0.16 (Table 2); d=0.56 (experiments.md pre-integration) | **CRITICAL: Table 2 and 18.3× are inconsistent** |
| ">10×" amplification | "≥10×" (§7.1) | "18.3×" (§6.2) | **INCONSISTENT** — use exact figure |
| φ arguments | φ=0, φ mentions (no argument structure) | φ(t,θ,s_t) formal def; φ(t,θ,g_t) in BEM §3.4 | Minor issue in §3.4, not Discussion |
| "Theorem 1" | §7.2 refers to "Theorem 1" | §4.1 renamed to "Observation 1" | **INCONSISTENT** — use "Observation 1" in §7.2 |
| Conjecture C1 scope | "empirically supported only in Regime I" (§8.1) | "Regimes II and III are entirely untested" (§7.4 Limitation 5) | Consistent — minor clarity improvement optional |
| CSI metric | Present in §7.2 | Present in §8.1 | Consistent ✓ (resolved from round 3) |
| CWD recommendation | Removed from step 3 | Table 3: CWD p_adj=0.349 | Consistent ✓ (resolved from round 3) |

---

## Priority Revision Checklist (Final Review Focus)

| Priority | Action | Location | Addresses |
|----------|--------|----------|-----------|
| **P0** | Resolve 18.3× arithmetic: determine correct AdamW Cohen's d (0.16 or 0.56), update Table 2 and §6.2 Key Finding 3 | §6.2, Table 2 | FR-1 |
| **P0** | Once §6.2 is fixed, propagate corrected ratio to §7.1 (Discussion) | §7.1 | FR-1 (Discussion) |
| **P0** | Replace ">10×" with exact "18.3×" (or corrected value) in §7.1 | §7.1 | M3 |
| **P1** | Change "Theorem 1" to "Observation 1" in §7.2 dual characterization paragraph | §7.2 | M1 |
| **P1** | Change "open question" to "fundamental mechanistic ambiguity" in §7.3 | §7.3 | M2 |
| **P1** | Add inline power caveat to §7.1 opening claim | §7.1 | C1, C3 |
| **P2** | Fix BEM integral notation in §3.4: use s_t not g_t | §3.4 | FR-2 |
| **P2** | Optional clarity fix in §8.1: "tested only at ρ=0.5" instead of "supported only in Regime I" | §8.1 | FR-3 |
| **P3** | Add "conjectured, tested at ρ=0.5 only" label to decision heuristic step 2 | §7.1 | M4 |

---

## What Works Well

1. **Section 7.4 (Limitations) is genuinely exemplary.** Seven numbered limitations covering statistical power (with MDE), architecture, dataset, regime boundary precision, conjecture status, SGD confound, and BEM bug disclosure. The acknowledgment that TOST requires n≈10 for adequate power given observed σ≈0.3% is unusually rigorous for an ML paper.

2. **The Verified / Predicted / Unknown structure in 7.3** is an excellent organizational device that precisely communicates the evidential hierarchy.

3. **CSI and coupling stability paragraph in 7.2 is new and good.** Provides mechanistic complement to accuracy equivalence.

4. **The ρ dual characterization (7.2)** connects two independently derived quantities (Xie & Li, Defazio) as dual representations. Even if algebraically straightforward, the synthesis is genuinely novel.

5. **The Conjecture C1 framing (FR-3)** is accurate and appropriately hedged. The paper is exemplary in never overclaiming its conjecture's evidential status.

---

*Critique authored by: Sibyl Section Critic*
*Revision round: 4*
*Date: 2026-03-18*
