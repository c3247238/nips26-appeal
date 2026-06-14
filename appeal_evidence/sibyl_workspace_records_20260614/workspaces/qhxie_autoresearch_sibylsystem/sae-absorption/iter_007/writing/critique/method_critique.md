# Method Section Critique

**Section:** 3 Methodology (method.md)
**Iteration:** 007
**Overall Score:** 8 / 10

---

## Summary Assessment

The Method section is well-structured, technically precise, and provides a reproducible description of seven distinct methodological components. The writing is clear, the notation is consistent with notation.md and glossary.md, and the section correctly scopes itself to methodology without bleeding into results. However, several issues reduce the score from a potential 9--10: (1) a numerical discrepancy between the method section and the experiments section regarding first-letter absorption rates, (2) incomplete specification of certain experimental parameters, (3) a missing forward reference to the cross-domain experiments described in the introduction, and (4) the magnitude gap threshold $\tau_{\text{mag}}$ is used in Section 3.6 but never formally defined in Section 3.2 where the absorption measurement protocol is described.

---

## Detailed Critique

### 1. Numerical Inconsistency: Absorption Rates (Critical)

**Issue:** Section 3.2 states that at $L_0 = 82$, the absorption rate for first-letter is computed after applying the quality gate (10/25 letters pass F1 > 0.85). The introduction (Section 1) reports the measured absorption rate as 15.96%, while Table 2 in experiments.md reports 13.4% for the first-letter domain at $L_0 = 82$. Section 4.7 (Table 7) reports 14.39%. These three numbers (15.96%, 13.4%, 14.39%) should be reconciled. The method section itself does not report a specific aggregate rate, but the inconsistency across other sections that depend on this methodology undermines reproducibility.

**Recommendation:** Clarify in the method section whether the reported aggregate absorption rate includes or excludes letters failing the quality gate. Add a sentence specifying the aggregation rule (e.g., "Aggregate absorption rates are computed only over letters passing the quality gate" or "over all 25 letters"). Coordinate with the experiments section to ensure a single consistent number.

### 2. Magnitude Gap Threshold $\tau_{\text{mag}}$ Not Defined (Major)

**Issue:** Section 3.6 (Threshold Sensitivity Analysis) introduces $\tau_{\text{mag}}$ as "the magnitude gap for absorption confirmation" and varies it across $\{0.5, 1.0, 1.5, 2.0\}$. However, Section 3.2 (Absorption Measurement Protocol), which is the definitive description of how absorption is measured, never mentions $\tau_{\text{mag}}$ or any magnitude gap criterion. The absorption criterion in Section 3.2 only requires that "all $k$ probe-associated latents have zero activation." If $\tau_{\text{mag}}$ modulates what counts as "zero" (e.g., activation below a threshold), this needs to be stated explicitly in 3.2. Without this, the reader cannot understand what Section 3.6 is testing.

**Recommendation:** Add the magnitude gap definition to Section 3.2, either as part of the absorption criterion or as a separate paragraph. Specify its default value (1.0) and its operational meaning (e.g., "A latent is considered inactive if its activation $z_j < \tau_{\text{mag}}$" or however the threshold is applied). This also needs to be added to notation.md, which currently lists $\tau_{\text{mag}}$ but only in the "Rates and Statistics" section, not in the "Absorption Measurement" section.

### 3. Cross-Domain Methodology Absent (Major)

**Issue:** The introduction states: "We audit the Chanin metric on Gemma 2 2B with Gemma Scope JumpReLU SAEs across five hierarchy domains (first-letter spelling, city-country, city-continent, city-language, animal-class)." Table 2 in the experiments section reports results for all five domains. However, the method section only describes the vocabulary and protocol for first-letter spelling. There is no description of how the four additional domains (city-country, city-continent, city-language, animal-class) were constructed: what datasets were used (RAVEL? WordNet?), how many tokens per domain, what the parent/child structure is, or whether the same probe training protocol applies.

The glossary.md mentions RAVEL for city-based hierarchies and WordNet for animal-class, and the outline mentions these data sources, but the method section omits this entirely.

**Recommendation:** Add a paragraph or subsection (e.g., Section 3.2.1 "Cross-Domain Hierarchies") describing the vocabulary, token counts, parent-child structure, and data sources for each of the five domains. At minimum, provide a table analogous to the "Cross-Domain Hierarchies" table in notation.md.

### 4. Activation Patching Protocol: Incomplete Specification (Moderate)

**Issue:** Section 3.5 describes three patching methods clearly but has two gaps:
- **Decode-reencode details:** "Zero the child feature's activation in the SAE encoding $z$, decode through $W_d$ to obtain a modified reconstruction, then re-encode through $W_e$." This is clear, but does not specify whether the re-encoding uses the full encoder pipeline (including thresholds $\theta_j$) or just the linear encoder. For JumpReLU SAEs, this matters because the threshold could prevent parent latent recovery even if the linear pre-activation exceeds zero.
- **Recovery criterion:** "Check whether any of the 5 parent latents recover (activation goes from 0 to $> 0$)." This is a hard binary criterion. Given the JumpReLU threshold, "activation $> 0$" means the pre-activation exceeds $\theta_j$. This should be stated explicitly to distinguish from the case where pre-activation increases but does not cross the threshold.

**Recommendation:** Specify that re-encoding uses the full JumpReLU pipeline (including thresholds) and that "recovery" means post-threshold activation $> 0$.

### 5. Probe Training Details (Moderate)

**Issue:** Section 3.2 describes the $k$-sparse logistic regression probe but does not specify:
- Training data: How many tokens are used for probe training? Is this the full 1,204-word vocabulary? Are there train/test splits? The experiments section mentions "5,000 corpus token positions" for CMI estimation but the probe training data is not described.
- Regularization: Is there any regularization beyond the $k$-sparsity constraint?
- Feature selection mechanism: "The probe selects the $k$ latents whose decoder columns $d_j$ have the highest cosine similarity $\cos(d_j, v_p)$ with the probe direction $v_p$." But $v_p$ is the probe direction, which is learned by the probe. This is circular: the probe selects features based on cosine to its own direction. Is this an iterative procedure, or is $v_p$ initialized differently?

**Recommendation:** Clarify the probe training data, any train/test split, and whether the feature selection is a two-stage process (first fit a dense probe to learn $v_p$, then select top-$k$ by cosine to $v_p$, then refit the sparse probe). This is important for reproducibility and follows from the Chanin et al. (2024) protocol, which should be cited more specifically here.

### 6. Statistical Inference: Bootstrap Details (Minor)

**Issue:** Section 3.2 states "All confidence intervals are 95% bootstrap CIs with 10,000 resamples (seed = 42)." However, it does not specify the bootstrap method (percentile, BCa, basic). For small samples (e.g., 8 persistent core words, 10 letters passing quality gate), the choice of bootstrap method matters. The glossary says "bootstrap CI" but does not specify the variant.

**Recommendation:** Specify the bootstrap method (e.g., "percentile bootstrap" or "bias-corrected and accelerated (BCa) bootstrap").

### 7. Section Numbering Inconsistency with Experiments (Minor)

**Issue:** The method section is numbered as Section 3, and the experiments section is numbered as Section 4. However, the introduction states "Sections 2--3 provide background and methodology. Section 4 presents the metric audit..." and the outline places the L0 phase transition in Section 5 and CMI in Section 6. But the experiments section (experiments.md) is entirely numbered as Section 4 with subsections 4.1--4.10, covering all three topics. This means the outline's Sections 5 and 6 are absent as separate sections; they are folded into Section 4. The method section's forward references (e.g., "Section 3.6" for threshold sensitivity) are internally consistent, but the paper structure does not match the outline or the introduction's roadmap.

**Recommendation:** Either restructure the experiments section to match the outline (separate Sections 4, 5, 6) or update the introduction's roadmap to match the actual structure. The method section itself is fine, but the cross-section mismatch reduces coherence.

### 8. Notation Consistency (Minor)

**Issue (a):** The method section uses $w_{e,j}$ (lowercase) for encoder weights in the probe training description ("$W_{e,j} x + b_{e,j} > \theta_j$"), but notation.md uses $W_e$ (uppercase) for the encoder matrix. The notation table defines $w_{e,j}$ implicitly through $W_e \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ but does not explicitly list $w_{e,j}$ as the $j$-th row. This is fine mathematically but could be made explicit.

**Issue (b):** Section 3.5 uses $d_c$ for the child decoder column direction and $z_c$ for the child activation. Notation.md defines $d_j$ for the $j$-th decoder column and $z_c$ for the child activation. Using $d_c$ as shorthand for $d_{c}$ (the decoder column of the child feature) is natural but not in the notation table.

**Recommendation:** Add $w_{e,j}$ and $d_c$ to notation.md for completeness.

### 9. Confound Decomposition: Justification for $L_0$ Set (Minor)

**Issue:** Section 3.4 uses $L_0 \in \{22, 41, 82, 176\}$ without explaining why these specific values were chosen. Were they the only available pretrained SAEs in Gemma Scope at this layer and width? Or were they selected to span a range? The reader should know whether this is a constraint of the available pretrained models or a design choice.

**Recommendation:** Add a brief note explaining the selection (e.g., "These are the four $L_0$ configurations available in Gemma Scope for layer 12 with $d_{\text{SAE}} = 16{,}384$").

### 10. Glossary Compliance (Minor)

The method section generally follows glossary.md conventions well. Specific compliance notes:
- "Sparse Autoencoder (SAE)" is expanded on first use (line 3 via Table 1 caption). Good.
- "$k$-sparse logistic regression probe" is used correctly. Good.
- "feature absorption" is used as preferred. Good.
- "$L_0$" is typeset correctly throughout. Good.
- "cosine similarity" is used rather than "cosine distance." Good.
- Minor deviation: The glossary prefers "SAE latent" over "SAE feature," and the method section uses "latent" consistently. Good.

---

## Strengths

1. **Clear structure.** The seven subsections each describe a self-contained methodological component with enough detail for reproduction.
2. **Table 1 is well-designed.** The configuration table provides a clean summary of all SAE configurations with the right level of detail.
3. **Control suite is well-motivated.** Section 3.3 explains the expected behavior of each control, making the experimental logic transparent before results are shown.
4. **Confound decomposition (Section 3.4) is the methodological highlight.** The permissive/strict hedging distinction and the persistent core word analysis are novel contributions that are clearly described.
5. **Threshold sensitivity analysis (Section 3.6) is well-designed.** The parameter grid is comprehensive and the reported metrics (CV, Kendall $\tau$) are appropriate.
6. **Activation patching (Section 3.5) uses three complementary methods** plus controls, providing triangulation.

---

## Weaknesses Summary (Prioritized)

| Priority | Issue | Subsection | Effort |
|----------|-------|------------|--------|
| Critical | Numerical discrepancy (15.96% vs. 13.4% vs. 14.39%) | 3.2 + cross-section | Low (coordination) |
| Major | $\tau_{\text{mag}}$ undefined in absorption protocol | 3.2 | Low (add definition) |
| Major | Cross-domain methodology absent | 3.2 | Medium (add subsection) |
| Moderate | Activation patching: re-encoding and recovery criterion | 3.5 | Low (add 2 sentences) |
| Moderate | Probe training: data, splits, feature selection process | 3.2 | Medium (add paragraph) |
| Minor | Bootstrap method unspecified | 3.2 | Low (add 3 words) |
| Minor | Section numbering mismatch with outline/intro | Cross-section | Medium (restructure) |
| Minor | Notation table additions ($w_{e,j}$, $d_c$) | notation.md | Low |
| Minor | $L_0$ set justification | 3.4 | Low (add 1 sentence) |

---

## Recommendations for Revision

1. **Reconcile absorption rate numbers** across intro, experiments Table 2, and experiments Table 7. Define the aggregation rule in Section 3.2.
2. **Define $\tau_{\text{mag}}$** in Section 3.2 with its default value and operational meaning.
3. **Add cross-domain hierarchy descriptions** (data sources, vocabulary sizes, parent-child structure) either in Section 3.2 or a new 3.2.1.
4. **Specify the full JumpReLU re-encoding pipeline** in Section 3.5 and clarify the recovery criterion.
5. **Expand probe training description** with data sizes, train/test splits, and the feature selection procedure.
6. **Specify bootstrap method variant** (percentile, BCa).
7. **Align section numbering** between the outline, introduction roadmap, and actual paper structure.
