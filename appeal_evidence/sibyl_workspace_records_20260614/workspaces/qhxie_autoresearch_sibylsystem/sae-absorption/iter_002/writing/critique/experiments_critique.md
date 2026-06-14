# Critique: Experiments (Section 4)

## Summary Assessment

The Experiments section is unusually honest and data-dense, with specific numbers throughout and
explicit reporting of negative results and unresolved tensions. The structure flows logically from
detection validation through decomposition to scaling and stability analyses. The primary weakness
is a critical numerical inconsistency between the Introduction and this section regarding the
raw cosine baseline AUROC. Several secondary issues concern missing encoder norm z_null statistics,
missing full statistical columns in Table 1, and ambiguous label set scoping across subsections.

## Score: 7/10

**Justification**: The section earns its score through specific quantified claims, well-structured
subsections, and honest treatment of null results. It falls short of 8 due to one critical
cross-section numerical inconsistency, incomplete statistical reporting in Table 1, and a
structural issue where the strongest detector (encoder norm) is introduced and partially dismissed
without being listed in the summary or given consistent statistical treatment. A revised version
that fixes the inconsistency and fills Table 1's missing z_null and p-value columns would likely
score 8.

---

## Critical Issues

### Issue 1: Cross-section numerical inconsistency — raw cosine AUROC
- **Location**: Section 4.1, Table 1 footer and inline text (line 80, line 54); also Introduction
  Contribution 2
- **Quote (Experiments)**: "$\cos(\hat{e}_c, d_c)$ achieves only $\text{AUROC} = 0.350$"
  (line 54); Table 1 also shows 0.350 (line 80)
- **Quote (Introduction)**: "the decoder-decoder cosine predictor fails entirely (AUROC = 0.206)"
  — Introduction Contribution 2
- **Problem**: The Introduction and Experiments section report different AUROC values for the same
  raw cosine baseline (0.206 vs. 0.350). The experimental data file (D1_eda_validation.json)
  confirms AUROC = 0.3497 for `cos_enc_dec`, consistent with 0.350. The Introduction's "0.206"
  also mislabels this metric as "decoder-decoder cosine" when it is actually the encoder-decoder
  cosine of the same feature. This is a critical inconsistency: a reviewer will notice the
  discrepancy immediately and question data reliability. The Introduction must be corrected to
  match the confirmed experimental value of 0.350.
- **Fix**: In the Introduction, change "AUROC = 0.206" to "AUROC = 0.350" and correct the metric
  label from "decoder-decoder cosine predictor" to "the raw intra-feature cosine $\cos(\hat{e}_c,
  d_c)$." Also, the framing "fails entirely" is accurate for 0.350 (strongly anti-correlated with
  labels), so the rhetorical force is preserved.

### Issue 2: Table 1 is missing z_null and p-values for encoder norm, freq, decoder norm
- **Location**: Table 1 (lines 72–83)
- **Quote**: Encoder norm row has AUROC = 0.757 but all statistical columns (Cohen's $d$,
  $p$-value, $z_\text{null}$) are "—"
- **Problem**: Encoder norm is the strongest individual detector in the table. A reviewer will
  ask: is encoder norm's AUROC statistically significant above the permutation null? Without
  $z_\text{null}$ for encoder norm, frequency, and decoder norm, the reader cannot determine
  whether these baselines beat the null. The table caption explains that cross-directional
  metrics use proxy labels while EDA uses exact labels — but this label-set difference is not
  reflected in the table rows themselves (no "label set" column), creating ambiguity about which
  AUROC values are directly comparable.
- **Fix**: (1) Add a z_null column entry for encoder norm by running 100-permutation null for
  encoder norm score (the permutation infrastructure already exists per line 17-18). If the
  value is already in D1_eda_validation.json it should be reported. (2) Add a "Label set" column
  with values "exact (n=18)" or "proxy (n=50)" to clarify comparability. (3) At minimum, add a
  table footnote explicitly stating which rows use exact vs. proxy labels.

---

## Major Issues

### Issue 3: Proxy label set for cross-directional metrics — framing invites confusion
- **Location**: Section 4.1, lines 28–40 and Table 1 caption (line 83)
- **Quote**: "Proxy labels (letter features with high decoder-probe alignment; $n_\text{pos} = 50$,
  Jaccard overlap with exact labels $= 0.115$) yield $\text{AUROC} = 0.659$ ... confirming the
  signal is stable across label definitions." (lines 28–31); then immediately: "The strongest
  per-feature signal comes from the cross-directional metric ... [AUROC = 0.730]" (lines 33–38),
  with cross-directional using proxy labels per Table 1 caption.
- **Problem**: The section presents EDA AUROC (exact labels) and cross-directional AUROC (proxy
  labels) as directly comparable in the "strongest detector" framing. However, Jaccard = 0.115
  means these two label sets measure largely different things. The cross-directional metric's
  AUROC = 0.730 on proxy labels is not directly comparable to EDA's 0.650 on exact labels. The
  text does not state which label set is used for cross-directional until the table caption, so
  readers will draw incorrect comparisons.
- **Fix**: In lines 33–38 (cross-directional result), explicitly state the label set: "Using proxy
  labels ($n_\text{pos} = 50$), $\cos(\hat{e}_p, d_c)$ achieves AUROC = 0.730." Add one sentence
  noting that direct comparison with EDA's exact-label AUROC requires caution given Jaccard = 0.115.
  Consider adding a supplementary column showing cross-directional AUROC on exact labels (n=18)
  as well, even if noisier.

### Issue 4: L10 AUROC comes from pairwise analysis — but uses exact Chanin labels that are L6-only
- **Location**: Section 4.1, lines 58–64 (Layer 10 reversal paragraph)
- **Quote**: "At layer 10, EDA computed on the same gpt2-small-res-jb suite gives AUROC = 0.256
  (Cohen's $d = -0.890$, $p = 6.8 \times 10^{-10}$)"
- **Problem**: The D1_eda_validation.json data confirms that Chanin et al.'s exact labels are for
  layer 6 only (the JSON states: "Cannot compute AUROC against Chanin labels (labels are for L6
  only)"). The AUROC = 0.256 at L10 therefore must use proxy labels or a different label
  construction method. The section does not state which label set was used for L10, making this
  result unverifiable without inspecting raw data. This ambiguity undermines the "polarity
  reversal" conclusion, which is one of the paper's key findings.
- **Fix**: State explicitly which label set was used for L10 AUROC (e.g., "Using proxy labels
  transferred from L6 letter-feature identification on the L10 SAE..."). If proxy labels were
  used, note that the Jaccard overlap between L6 and L10 proxy labels may differ, and the
  reversal finding rests on the assumption that proxy labels identify the same conceptual features
  across layers.

### Issue 5: EDA_Delta number inconsistency between Section 4.3 and the Outline
- **Location**: Section 4.3, line 132 (Standard/L1 suite) vs. Outline Section 4.3
- **Quote (Section 4.3)**: "EDA$_\Delta$ is positive at all five layers and peaks at L4 ($+0.045$,
  AUROC $= 0.716$) and L6 ($+0.050$, AUROC $= 0.702$)"
- **Quote (Outline)**: "Standard/ReLU (L1 penalty, 24576): EDA delta = +0.045; AUROC = 0.702"
  (outline assigns 0.702 to the standard suite, not L4-specific); "TopK-32 (32768): EDA delta =
  +0.046; AUROC measurable"
- **Problem**: The outline records EDA_delta = +0.045 for the standard suite broadly, while the
  section specifies +0.045 at L4 and +0.050 at L6. These are not contradictory but the outline's
  AUROC 0.702 appears to correspond to the L6 primary value, not L4 (which is 0.716 per the
  text). Minor, but risks creating confusion during paper revision.
- **Fix**: Verify against B2_scaling_curve.json which AUROC value the outline intended (likely L6
  = 0.702 is the primary reference point). Ensure the outline and section agree on the L4 value
  (0.716) as the peak.

### Issue 6: Phase stability absorption rate range inconsistency across sections
- **Location**: Section 4.4 line 168 vs. Introduction Contribution 3 vs. Discussion 5.4
- **Quote (Section 4.4)**: "All 11 configurations produce uniformly high absorption rates:
  the range is $0.876$–$0.978$" (line 168)
- **Quote (Introduction Contribution 3)**: "Absorption rates across all 11 tested SAE
  configurations ... lie in the narrow range 0.919–0.968."
- **Quote (Discussion 5.4)**: "Absorption rates are 0.876–0.978 across all 11 tested
  configurations"
- **Problem**: The Introduction says 0.919–0.968 but Section 4.4 and the Discussion say
  0.876–0.978. This is a direct numerical inconsistency that will be immediately noticed by
  reviewers. The broader range (0.876–0.978) appears in the data file (E1_phase_transition.json
  shows AJT absorption rates of 0.876 and 0.978). The Introduction's narrower range either refers
  to the primary jb suite only or is simply wrong.
- **Fix**: Standardize to 0.876–0.978 across all sections (or clarify that 0.919–0.968 refers to
  the primary jb suite only and 0.876–0.978 is the full range across all 11 configs). The
  Introduction should match Experiments and Discussion.

---

## Minor Issues

- **Line 3, "We evaluate EDA and cross-directional metrics on the first-letter task"**: The
  section opens without a forward pointer to how the section is organized. Add one sentence:
  "We structure the evaluation into five analyses: (4.1) detection validation, (4.2) mechanistic
  decomposition, (4.3) architecture and scale sweep, (4.4) phase stability, and (4.5) cross-domain
  scope." This costs three lines and eliminates reader disorientation.

- **Line 12, base rate**: "base rate $= 7.3 \times 10^{-4}$" but Table 1 reports
  AUPRC/base $= 2.09\times$, implying AUPRC $= 2.09 \times 7.3 \times 10^{-4} = 0.00153$.
  The JSON confirms AUPRC = 0.001528. This is internally consistent, but rounding the base rate
  to $7.3 \times 10^{-4}$ hides that it is $0.000732421875 \approx 7.32 \times 10^{-4}$.
  Not wrong, but verify the rounding.

- **Line 43, encoder norm**: "yields the highest AUROC among individual weight-only features
  ($\text{AUROC} = 0.757$)". The actual confirmed value from D1_eda_validation.json is 0.7566.
  This rounds correctly to 0.757, but the text should match to 3 decimal places consistently
  with other reported values (e.g., 0.650, 0.730, 0.681, 0.595, 0.515, 0.350). Use 0.757 or
  0.756 consistently; the current section uses 0.757 and Table 1 also uses 0.757, which is
  fine.

- **Lines 89–98 (Section 4.2), AUROC = 1.000 for decoder**: "the decoder achieves
  AUROC $= 1.000$". AUROC of exactly 1.000 on proxy labels deserves a caveat. Is this on the
  n=50 proxy set where the labels were defined by decoder-probe alignment? If proxy labels are
  defined by thresholding decoder-probe cosine, then AUROC = 1.000 for the decoder is
  circular — the detector and the label criterion are the same quantity. The text should
  acknowledge this circularity explicitly, even in a footnote.

- **Line 161, Figure 3 caption**: "Figure 3 plots EDA$_\Delta$ (letter minus non-letter mean EDA)
  across 11 SAE configurations." Figure 3 appears in the text before Section 4.2's Figure 4
  reference (line 89), which is correct ordering. But the caption within the figure comment
  (line 241) describes the figure but the in-text reference on line 128 says "Figure 3 plots
  EDA$_\Delta$ ...across 11 SAE configurations." This is in Section 4.3, which is correct.
  No issue.

- **Line 174, Hypothesis H4**: The text references "Hypothesis H4" but the Method section does
  not define or number hypotheses. This forward reference to an undefined label will confuse
  readers. Either define H4 in the Method/Introduction or remove the label and just write
  "testing whether absorption rate follows a sigmoid-shaped transition."

- **Line 188, fine-tuning described as SAE from Section 4.4**: "we fine-tuned a high-sparsity
  SAE (gpt2-small-res-jb, layer 2, baseline $L_0 = 33.7$, absorption rate $= 0.959$)". The L0
  of 33.7 is inconsistent with the scaling curve's Layer 2 value (E1 JSON shows L2 L0 =
  18.535). Either the baseline is a different SAE config or the L0 value in the text is wrong.
  Verify against E2_hysteresis.json.

- **Line 212, absorption rate range for 4.5**: "The first-letter hierarchy yields absorption rate
  $= 0.0083$". The ratio-to-null = 10.0 implies null absorption = 0.00083. This is a very
  different meaning of "absorption rate" from the 0.876–0.978 range in Section 4.4 (which
  measures the fraction of child-feature activations suppressed per input). Section 4.5 uses
  the IG-ablation absorption event count divided by total test events. The two metrics measure
  different things and this terminological overlap will confuse readers. Add a clarifying
  sentence: "Note: this absorption rate measures the fraction of IG-ablation events showing
  child suppression across tested word-letter pairs, not the per-input child-inactive rate
  from Section 4.4."

---

## Visual Element Assessment

- [x] Figures/tables match outline plan: Figures 2, 3, 4, 5, Table 1 are all present and
  match the outline's Figure/Table Plan. Figure 6 (cross-domain bar chart) is listed as optional
  in the outline and is absent; this is acceptable but Section 4.5's null result would benefit
  visually from even a simple bar chart showing ratio-to-null = 10.0 vs. 1.0.
- [x] All visuals referenced before appearance: Figure 2 is referenced at line 22 (before line 66);
  Table 1 is referenced at line 23 (before line 70); Figure 4 at line 89 (before line 124);
  Figure 3 at line 128 (before line 161); Figure 5 at line 165 (before line 165 — same
  paragraph). Minor: Figure 5 is referenced and then immediately shown in the same paragraph;
  the reference should precede the interpretation paragraph, not be embedded in the first sentence.
- [x] Captions are self-explanatory: Table 1's caption includes key comparison notes and label
  set clarification. Figure captions in the comment block are descriptive. The visible figure
  references (lines 66, 124, 161) are one-line alt-text descriptors — adequate.
- [ ] Section 4.5 (cross-domain) has no figure. The null result for semantic hierarchies is
  important enough to deserve a visual. At minimum a two-bar chart (first-letter: 10x null,
  semantic: 1x null) would make the contrast immediately legible. The outline lists Figure 6
  as optional, but given that this is a scoped null result with only three data points, a bar
  chart takes negligible space and significantly aids reader comprehension.

---

## What Works Well

1. **Honest treatment of failure modes**: The paper reports three distinct failure modes (L10
   reversal, AJT reversal, encoder norm as confound) with quantified effect sizes. This is
   exactly what a top-venue reviewer wants to see: the section does not cherry-pick the positive
   results.

2. **Hysteresis experiment narrative**: Lines 184–197 document the checkpoint trajectory
   ($0.959 \to 0.959 \to 0.960 \to 0.960 \to 0.960$ at steps 100–500) rather than just
   reporting the endpoint. This level of procedural specificity is publication-ready and
   makes the metastability claim convincing.

3. **Summary paragraph** (lines 229–237): The section ends with a crisp two-paragraph summary
   that covers all five subsections in six sentences. Every sentence contains a specific number.
   This is good scientific writing.
