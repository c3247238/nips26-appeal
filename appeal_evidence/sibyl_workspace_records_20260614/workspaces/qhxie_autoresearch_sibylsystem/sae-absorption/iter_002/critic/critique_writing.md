# Writing Critique — SAE Absorption Paper

## Overall Assessment

The paper is well-structured, evidence-forward, and exemplary in reporting negative results. The Discussion section (Sections 5.1–5.7) is among the strongest in the pipeline — each failure mode is presented with exact numbers and multiple competing explanations. However, the paper contains **four critical numerical inconsistencies** that will immediately fail a reviewer's cross-check and a missing abstract that makes the document structurally incomplete.

---

## Critical Issues (Must Fix Before Review)

### 1. Phantom AUROC = 0.206

**Problem:** The Introduction (line 59) and Conclusion (line 993) both state: "the decoder-decoder cosine predictor fails entirely (AUROC = 0.206)."

**Reality:** Table 1 contains no entry for AUROC = 0.206. The closest value is the raw cosine cos(ê_c, d_c) at AUROC = 0.350. The value 0.206 appears in `B1_decoder_geometry.json` as the AUROC of the cos² threshold classifier between absorbed and non-absorbed inter-feature decoder angles — an entirely different metric, different label set (proxy), and different analysis context.

**Impact:** A reviewer comparing the Introduction claim to Table 1 will immediately identify this as a fabricated or misattributed number. The paper cannot contain a number in two prominent locations that does not appear in the results section.

**Fix:** Replace AUROC = 0.206 with AUROC = 0.350 (Table 1 entry for raw cosine) everywhere it appears (Introduction, Contribution 2 block, Conclusion). If the inter-feature decoder-decoder pairwise AUROC (0.206 or similar) is also relevant, add it explicitly to Table 1 or a supplementary table with proper attribution.

---

### 2. Three-Way Absorption Rate Range Inconsistency

**Problem:**
- Introduction: "92–97%" (0.919–0.968)
- Conclusion: "span only 0.919–0.968"
- Section 4.4: "0.876–0.978"
- Section 5.4: "0.876–0.978"

**Reality:** E1_phase_transition.json and C2_child_suppression_absorption.json confirm the broadest range (0.876–0.978) is correct. The narrower range (0.919–0.968) appears to silently exclude AJT configurations without stating this.

**Impact:** The same paper citing three different ranges for the same quantity is a fatal consistency error.

**Fix:** Use "0.876–0.978" uniformly in all four locations. If the primary jb-suite-only range is 0.938–0.967, present that as a scoped secondary observation: "Within the primary jb suite, rates span 0.938–0.967; including AJT configurations, the range is 0.876–0.978."

---

### 3. Three Different Spearman ρ Values

**Problem:**
- Introduction (line 67): ρ = 0.191, p = 0.574
- Section 4.4 (all 11 configs): ρ = −0.482, p = 0.133
- Section 4.4 (jb suite only): ρ = −0.100, p = 0.873

These three values differ in sign and magnitude. A positive 0.191, a negative −0.482, and a near-zero −0.100 cannot all describe the same variable pair.

**Reality:** The Introduction appears to report EDA_delta vs. 1/L0 from the B2 analysis (B2_scaling_curve.json: rho = 0.191). Sections 4.4 reports absorption_rate vs. 1/L0 from E1_phase_transition.json. These are different dependent variables and should not be presented as a single "absence of sparsity dependence."

**Fix:** Decide which analysis is canonical for the phase stability claim. If it is the absorption rate vs. 1/L0 (E1), use ρ = −0.482 (all 11) and ρ = −0.100 (jb suite) in both the Introduction and Section 4.4. Remove the EDA_delta vs. 1/L0 correlation from the Introduction unless it is presented as a separate, explicitly labeled analysis.

---

### 4. Missing Abstract

The paper_draft.md begins at "# Introduction" with no abstract. The outline.md contains a full draft abstract. A submitted paper without an abstract will be desk-rejected by any venue.

**Fix:** Add the abstract from outline.md before the Introduction. Then verify all numerical claims in the abstract against the corrected values for issues 1–3 above. In particular, the abstract uses "0.919–0.968" which must be corrected to "0.876–0.978."

---

## Major Issues

### 5. Roadmap Sentence Misdescribes Paper Flow

Line 77: "We first characterize the theoretical conditions under which absorption is preferred before comparing the results with our geometric predictions (Section~\ref{sec:related})"

This implies Related Work comes after the theory section, but in the paper it comes before Method. This reads as if the author forgot to update the paper structure description.

**Fix:** "Section~\ref{sec:related} surveys related work. Section~\ref{sec:method} formalizes the rate-distortion framework and experimental setup. Section~\ref{sec:experiments} reports detection, decomposition, and phase-stability results. Sections~\ref{sec:discussion} and~\ref{sec:conclusion} synthesize findings and scope open questions."

### 6. Section Numbering Inconsistency

- Introduction: unnumbered (`# Introduction`)
- Related Work: unnumbered (`# Related Work`)
- Method: numbered (`## 3.1`, `## 3.2`)
- Experiments: numbered with prefix (`# 4 Experiments`)

LaTeX will double-number section 4 if the `# 4` prefix is retained after auto-numbering.

**Fix:** Remove inline section numbers from headers. Let LaTeX auto-number. Verify the `\section{}` structure produces 1 Introduction, 2 Related Work, 3 Method, 4 Experiments, 5 Discussion, 6 Conclusion.

### 7. Duplicate Figure 1 Reference

Figure 1 (fig1_eda_method.pdf) is embedded at:
- Line 82: Between Introduction and Related Work (orphan — no figure plan here)
- Line 404: Within Section 3.3 (correct placement)

The figure will appear twice in the compiled output.

**Fix:** Remove the image embed at line 82. Replace with a forward reference: "Figure~\ref{fig:method} in Section~\ref{sec:method} illustrates this geometry."

### 8. EDA Magnitude Tension Repeated Unnecessarily

Sections 4.2 and 5.2 both discuss the observation that mean EDA = 0.671 (implying ~48° encoder-decoder angle) while Proposition 1 predicts small decoder angle at absorption onset. Section 4.2 ends with "We report this as an open question." Section 5.2 provides the fuller temporal-drift hypothesis.

**Fix:** Remove the brief acknowledgment in Section 4.2. Retain only the Discussion treatment in Section 5.2.

---

## Minor Issues

### 9. EDA Acronym Semantic Ambiguity

The acronym "EDA" stands for "Encoder-Decoder Alignment" in the glossary's expansion but the glossary parenthetical says "(dissociation metric)." The metric measures misalignment (high EDA = bad). The paper alternates between "encoder-decoder dissociation (EDA)" (correct framing) and "EDA" alone.

**Suggestion:** Define "EDA" as "Encoder-Decoder (mis)Alignment" or simply "Encoder-Decoder Angle" in Section 3.3 and use the definition consistently. The definition should make clear that EDA = 0 means aligned and EDA = 1 means orthogonal.

### 10. Table 1 Mixed Label Sets Without Clear Separation

Table 1 presents EDA at exact labels (n_pos=18) and cross-directional metrics at proxy labels (n_pos=50) in the same table. The footnote explains this but does not create a visual separator. Jaccard overlap = 0.115 means these two sets measure largely different features.

**Suggestion:** Add a horizontal rule or explicit row annotation (e.g., dagger notation) to distinguish exact-label from proxy-label entries. The footnote should state the Jaccard overlap to make explicit that these are different label sets, not just different sizes.

---

## What Works Well

1. **Proposition 1 proof and corollaries** (Section 3.2): The derivation is clean and the co-occurrence cancellation corollary is highlighted correctly. The Limitation paragraph after the proof is exemplary.

2. **Failure mode section** (Section 5.3): Each of three failure modes (L10 reversal, AJT reversal, encoder norm dominance) is presented with exact numbers and competing interpretations. No overclaiming. This section builds reviewer trust.

3. **Negative result handling**: Cross-domain null (animate_inanimate, noun_proper), phase-transition failure (LRT p=0.456), and hysteresis saturation are all reported prominently and not buried. The paper's honesty about hypothesis falsification is commendable.

4. **Claim-evidence density**: Nearly every quantitative claim in the paper has a traceable source file (D1, D2, E1, E2, etc.). The audit_report.json reveals no discrepancies between source JSON and paper text (excluding the four inconsistencies above).
