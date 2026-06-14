# Critique: Analysis and Discussion (Section 6)

## Summary Assessment

The Discussion section is technically the strongest in the paper. Sections 6.1–6.2 deliver mechanistically precise arguments tied directly to experimental data, and Section 6.5 translates findings into actionable deployment rules with specific parameters. The section mostly avoids banned writing patterns and makes appropriate use of quantitative evidence. However, three problems prevent it from reaching publication quality: (1) a CHR inconsistency between this section and the Experiments section (94% vs. ~96%); (2) an undefined function in the core synergy equation that signals analytical sloppiness; and (3) a deep structural tension in Section 6.3 that the text understates — M1+CD-SSD is dominated by M1+naive-T16 in absolute QAS, which directly challenges CD-SSD's framing as a "composability vehicle." Additionally, the Conclusion section still uses "IGSD" throughout, while this section uses "CD-SSD" — a cross-section terminology inconsistency that a reviewer will flag.

## Score: 8/10

**Justification**: The frozen-token entropy-collapse argument (§6.2) and the tau=0.0 paradox resolution (§6.3) are publication-quality passages. To reach 9/10, the section needs to: (1) resolve the 94% vs. 96% CHR inconsistency with the Experiments section; (2) define $f(\text{CHR})$ in the synergy equation or remove the equation; (3) directly confront the CD-SSD vs. naive-T16 implication rather than relegating it to an "open question"; (4) ensure all "IGSD" references are renamed to "CD-SSD" per the pre-submission checklist.

---

## Critical Issues

### Issue 1: CHR value inconsistency — 94.0% (Discussion 6.2) vs. ~96% (Experiments 5.2)

- **Location**: Section 6.2, paragraph 1; Section 5.2, paragraph "M1 + IGSD — SYNERGY"
- **Quote (Discussion 6.2)**: "the KV-cache hit rate (CHR) jumps from 60% during M1 standalone operation to 94.0% during CD-SSD's refine phase"
- **Quote (Experiments 5.2)**: "Measured KV cache hit rate during the refine phase: ~96% (vs. ~60% during the draft phase)"
- **Problem**: The same measurement is reported as 94.0% in §6.2 and ~96% in §5.2. The notation.md authoritative record shows CHR_refine ≈ 0.940, and the per-seed breakdown in §6.2 (seed 123: GSM8K = 0.940, MATH500 = 0.864, HumanEval = 0.907) confirms 94.0% as the correct value. The ~96% in §5.2 appears to be an error. A reviewer spotting this discrepancy will treat it as a data integrity issue and question all other numbers.
- **Fix**: Canonicalize on 94.0% everywhere. Update Experiments §5.2 to "~94%" and add "(mean across seeds and benchmarks)" for clarity.

### Issue 2: QAS values in Table 7 are inconsistent with both Table 2 and Table 5

- **Location**: Section 6.3, Table 7
- **Quote**: "CD-SSD($\tau = 0.9$, $T_d = 16$) | 4.518x | 0.653 | 2.950"
- **Problem**: The same operating point (CD-SSD, $\tau = 0.9$, $T_d = 16$) now has three different QAS values in the paper: QAS = 1.194 in Table 2 (§5.1 Pareto table), QAS = 2.879 in the outline's §5.4 summary, and QAS = 2.950 in Table 7. The Table 7 caption does not identify the evaluation scope (200-sample GSM8K, 2-seed subset). Without that context, readers cannot reconcile the three values. The speedup in Table 7 (4.518×) also differs from Table 2's GSM8K speedup (4.57×), again without explanation.
- **Fix**: Add an explicit footnote to Table 7: "All configurations evaluated on 200 GSM8K samples, seeds [42, 123]. QAS = Speedup × GSM8K AccRet (standard formula, this subset only). Values differ from Table 2's 3-seed full-benchmark results due to subset and seed differences." Cross-reference Table 5 for the corresponding ablation numbers.

---

## Major Issues

### Issue 3: Section 6.3 understates the CD-SSD vs. naive-T16 dominance problem

- **Location**: Section 6.3, second numbered point and "Open question" paragraph
- **Quote**: "the absolute QAS difference (3.914 for M1+CD-SSD vs. 4.232 for M1+naive-T16, delta = -7.5%) shows that the partition overhead still dominates in the current implementation."
- **Problem**: The data in Table 7 establishes a harder result than the text acknowledges: naive-T16 dominates CD-SSD in both standalone (4.458 vs. 4.198 QAS) and combined (4.232 vs. 3.914 QAS) configurations. The section's current framing — treating CD-SSD as a "composability vehicle" that provides value through the frozen-token synergy — is undermined by the fact that the simpler alternative (M1+naive-T16) achieves higher absolute QAS while also achieving substantial Ortho = 0.949. The Ortho premium of 0.436 (1.385 vs. 0.949) is real, but since CD-SSD imposes partition overhead that lowers QAS in both configurations, a reader can reasonably ask: what is the deployment case for CD-SSD over naive T=16? The "open question" framing is too soft for a critical finding that affects the paper's contribution framing.
- **Fix**: Confront this directly with one of two framings: (a) "CD-SSD's contribution is the mechanistic demonstration that frozen-token structure amplifies KV synergy to Ortho = 1.385 — a qualitatively different synergy pattern from naive step reduction (Ortho = 0.949) — even if the current partition overhead prevents this from translating into an absolute QAS advantage. The finding has implications for MDM system design: methods that create structured frozen-token sets during inference can unlock super-multiplicative composability with KV-cache methods." OR (b) "The current CD-SSD implementation provides no QAS benefit over naive T=16 step reduction in either standalone or combined configuration. Its contribution is the mechanistic insight that frozen-token partition creates a qualitatively different KV synergy mode. Future work should explore whether reducing partition overhead (e.g., lighter refine phases, or task-adaptive $\tau$) can recover the QAS advantage that the synergy mechanism implies." Either framing is defensible; the current hedged framing is not.

### Issue 4: Section 6.2's synergy equation contains an undefined function

- **Location**: Section 6.2, equation block near end of paragraph 4
- **Quote**: "$$\text{Speedup}(M1 + \text{CD-SSD}) = \text{Speedup}(\text{CD-SSD}) \times f(\text{CHR}_{\text{refine}})$$"
- **Problem**: $f(\text{CHR})$ is described verbally as "M1's speedup function evaluated at the elevated hit rate during the refine phase" but is never defined mathematically. The equation cannot be verified or reproduced by a reader. The preceding verbal argument (9.4% premium from CHR 60% → 94%) is actually more rigorous than the equation itself, which adds formal appearance without content. A reviewer who tries to derive $f(0.94) = 5.13 / 3.40 = 1.509$ and $f(0.60) = 1.38$ from the equation will have no way to check whether this is self-consistent.
- **Fix**: Either (a) define $f$ explicitly — for example, under a simple linear KV-savings model, $\text{M1 speedup} \approx 1/(1 - \delta \cdot \text{CHR})$ where $\delta$ is the per-position attention cost fraction, giving $f(0.94)/f(0.60) \approx (1-0.60\delta)/(1-0.94\delta)$, which for $\delta \approx 0.28$ yields the observed ratio — and include the derivation in an appendix; or (b) remove the equation and state: "The CHR elevation from 60% to 94% amplifies M1's per-step savings within the refine phase, producing a combined speedup of 5.13× that exceeds the multiplicative expectation (1.38× × 3.40× = 4.69×) by 9.4%."

### Issue 5: Section 6.1's "testable prediction" is retrospective, not prospective

- **Location**: Section 6.1, final paragraph
- **Quote**: "The trajectory-preserving vs. trajectory-modifying classification predicts that any MDM acceleration method that modifies the unmasking order or step schedule will interfere with KV-caching."
- **Problem**: The trajectory-preserving vs. trajectory-modifying classification was derived from the observed results of the three pairs studied — it is a post-hoc rationalization fitted to the data, not a prior prediction. Presenting it as a "testable prediction" implies prospective validity that it does not have for the current dataset. A reviewer trained in scientific methodology will call this circular reasoning. The classification may well generalize to future methods, but that must be framed carefully.
- **Fix**: Reframe as a hypothesis generated from the current findings: "This post-hoc classification — derived from the observed binary composability pattern across three method pairs — generates a testable hypothesis for methods not yet evaluated: trajectory-modifying acceleration methods should exhibit Ortho < 0.8 when combined with KV-cache methods, while trajectory-preserving methods should not. The classification should be treated as a hypothesis awaiting validation on a broader set of method pairs, not a confirmed predictive law."

---

## Minor Issues

- **Stray "IGSD" in §6.3**: The phrase "5.13x at the 2-seed scale; 6.68x on the dedicated tau=0.0 comparison subset" and its surrounding context still uses implied "IGSD" identifiers in references to prior sections. The pre-submission checklist explicitly flags "IGSD renamed consistently to CD-SSD throughout all files" as a Major requirement. Scan for all remaining "IGSD" occurrences and replace with "CD-SSD."

- **Cross-section inconsistency — Conclusion still uses "IGSD"**: Sections 8.1 and 8.2 of the Conclusion use "IGSD" throughout (e.g., "M1 combined with IGSD ($\tau = 0.9$...)", "M3 + IGSD gives Ortho = 0.493", "SSD + M1... vs. IGSD's lossy variant"). This is a systematic naming inconsistency between the Discussion (correctly uses CD-SSD) and the Conclusion. The Conclusion section must be updated to use "CD-SSD" consistently.

- **Section 6.2, per-seed CHR breakdown understated**: "(from per-seed data: seed 123 GSM8K = 0.940, MATH500 = 0.864, HumanEval = 0.907)" is buried in a parenthetical but the 7.6 percentage-point range (86.4% to 94.0%) is substantive. Lower CHR on MATH500 suggests harder reasoning tasks reduce the frozen-token fraction. Promote this to a sentence: "CHR_refine varies by benchmark: 94.0% (GSM8K), 86.4% (MATH500), 90.7% (HumanEval) at seed 123, suggesting that more difficult reasoning tasks with longer uncertain chains reduce the fraction of frozen tokens."

- **Section 6.4, Limitation 3 — absolute accuracy caveat**: "The absolute GSM8K accuracy of 45.3% (vs. 71.2% baseline) is not deployable as a standalone reasoning accelerator." This is correct but note that 45.3% appears to reference the 3-seed full-scale GSM8K AccRet = 63.7%, which means absolute accuracy = 0.637 × 71.2% ≈ 45.3%. The derivation should be explicit rather than implicit: "GSM8K accuracy = 63.7% retention × 71.2% baseline = 45.3%."

- **Section 6.4, Limitation 6 — SSD speedup attribution**: "Self-Speculative Decoding (SSD) [CITE:ssd] achieves 3.46× lossless speedup via intermediate-layer exit drafting." The 3.46× was measured in [CITE:ssd] on a different setup. Add: "3.46× as reported in [CITE:ssd]; not reproduced under our evaluation protocol."

- **Section 6.5, Rule 2 — missing combined QAS caveat**: "QAS = 1.675 (GSM8K)" is correct but potentially misleading — M3's combined QAS across all benchmarks including HumanEval is substantially lower (~0.7) due to coding incompatibility. Add: "QAS = 1.675 (GSM8K-specific; combined across all benchmarks ≈ 0.7 due to M3's near-zero coding performance)."

- **Section 6.5, runtime detection signals — missing failure mode cross-references**: The three detection signals map directly to failure modes in Table 4 but don't cite them. Add: "CHR < 40% (failure mode F2); $\alpha > 0.75$ (F3); combined speedup < max(individual) (F4; see Table 4)."

- **Section 6.4, Limitation 4 — cross-reference to §4.3**: Dream-7B unavailability is first disclosed in §4.3. Add "(first disclosed in §4.3)" to avoid repeating setup details.

- **Notation: $K^{(\ell)}_t, V^{(\ell)}_t$ in §6.2**: The discussion writes these as $K^{(\ell)}_t, V^{(\ell)}_t$ with subscript $t$ and superscript $(\ell)$. The notation.md defines them as $K_t^{(\ell)}, V_t^{(\ell)}$ (subscript $t$, superscript $(\ell)$). These are equivalent but the formatting differs. Standardize to notation.md's convention throughout.

- **Section 6.3, "IGSD" in paragraph on standalone value gap**: The sentence "5.13x at the 2-seed scale; 6.68x on the dedicated tau=0.0 comparison subset" references IGSD-related numbers but the surrounding context uses "CD-SSD." Verify all references in this paragraph have been renamed.

---

## Visual Element Assessment

- [x] Figure 7 is planned in the outline's Figure & Table Plan (Section: Discussion 6.1)
- [x] Figure 8 is planned in the outline's Figure & Table Plan (Section: Discussion 6.2)
- [x] Figure 7 is cited before it appears: "Figure 7 illustrates the root cause" (§6.1 opening line)
- [x] Figure 8 is cited before it appears: "Figure 8 provides direct evidence" (§6.2 opening line)
- [x] Both figure captions are self-explanatory
- [ ] **Figure 7 caption omits M1+M3 from incompatible set**: Caption reads "incompatible (M2, M3+CD-SSD) composition" but M1+M3 (Ortho = 0.301) is also incompatible and is central to Section 6.1's argument. Fix: "incompatible (M2, M3+CD-SSD, M1+M3)."
- [ ] **Table 7 caption lacks scope specification**: Unlike Tables 2–6, Table 7 does not state the evaluation scope (200 samples, GSM8K only, 2 seeds). A reviewer will ask why QAS values differ from the main results tables. Fix caption to add "(200 GSM8K samples, seeds [42, 123], standard QAS formula)."
- [ ] **Section 6.5 has no visual aid**: Three deployment rules with runtime detection signals are presented as dense text. A decision table or rule-flowchart (3 rows: Rule 1/2/3 × deployment scenario/recommended config/detection signal) would substantially improve scanability. Not in the current figure plan; recommend adding as Table 8 in Section 6.5.

---

## What Works Well

1. **The frozen-token entropy collapse argument in §6.2** is the best analytical passage in the paper. The chain — partition ($\alpha = 0.52$) → frozen tokens → deterministic distribution → $H_i = 0$ → guaranteed cache hit ($0 < \eta = 2.0$) → measured $\text{CHR}_\text{refine} = 94.0\%$ → 9.4% speedup premium calculation — is mechanistically rigorous, independently verifiable, and directly grounded in the experimental data. This would survive peer review scrutiny.

2. **The tau=0.0 paradox resolution in §6.3** demonstrates scientific integrity. Rather than smoothing over the counter-intuitive +42.3% QAS improvement at $\tau = 0.0$, the section presents a decisive five-row comparison table, quantifies the -5.8% gap between CD-SSD($\tau = 0.0$) and naive-T16 as within-noise, and separates CD-SSD's two contributions (step reduction vs. composability vehicle). This level of transparency builds reviewer trust even when it partially undermines a contribution claim.

3. **The deployment guidance in §6.5 is operationally specific.** "Auto-reject $J > 3$ before execution," exact parameter values ($\eta = 2.0$, $\tau = 0.9$, $T_\text{draft} = 16$), and the 10–20 sample detection window are concrete enough for practitioners to implement. The failure mode cross-links (F1, F2, F3 in the detection signals) connect analysis to actionable engineering practice — a differentiator from papers that only report aggregate benchmarks.
