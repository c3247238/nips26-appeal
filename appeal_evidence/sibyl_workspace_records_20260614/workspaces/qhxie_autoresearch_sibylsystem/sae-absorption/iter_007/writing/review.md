# Writing Quality Review

## Summary

The paper audits the Chanin feature absorption metric on JumpReLU SAEs (Gemma 2 2B / Gemma Scope), reporting three findings: (1) the metric does not transfer -- shuffled-label controls exceed measured absorption in all five hierarchy domains, driven by candidate explosion at the default cosine threshold in high-dimensional space; (2) absorption declines monotonically with the $L_0$ operating point, exhibiting a phase transition at $L_0 \approx 40$--80 that is stable across layers; (3) the marginal CMI-absorption correlation at $L_0 = 82$ vanishes at $L_0 = 22$ where all probes achieve perfect F1, indicating probe quality confounds rather than rate-distortion theory drove the signal. The argument structure is clear and the claims are well-scoped.

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a clean problem-approach-evidence-conclusion arc across Sections 1--8. Each section transitions logically: the introduction poses three questions, the methodology describes the tools to answer them, Sections 4--6 answer Q1--Q3 in order, and the discussion connects the findings.

Two structural issues:

1. **Table numbering is out of reading order.** Table 7 ($L_0$ phase transition, Section 5.1) appears *after* Table 6 (CMI analysis, Section 6.2) in the numbering scheme, but *before* it in the text. A reviewer encountering "Table 7" in Section 5 after having just read Tables 1--5 in Sections 3--4 will notice the skip. The visual audit already flags this for LaTeX renumbering; it should be fixed before any external review.

2. **Table 8 appears in Section 3.7 (Methodology) but is numbered after tables that appear in Sections 4--6.** Table 8 defines the cross-domain hierarchy suite and logically belongs early. Its current number suggests it was appended after the main tables were laid out. Renumbering to follow Table 1 in Section 3 would be natural.

3. **Section 7.4 (Negative Results)** references "four of seven pre-registered hypotheses" (H2, H5, H6, H7) but the paper body never enumerates H1--H7 in a single place. The reader has to infer from the proposal what these hypotheses are. Either enumerate them in the methodology (Section 3) or in Section 7.4 itself.

### Notation & Terminology Consistency: 9/10

Cross-checking against `notation.md` and `glossary.md`:

- All core symbols ($x$, $\hat{x}$, $d_{\text{model}}$, $d_{\text{SAE}}$, $W_e$, $W_d$, $d_j$, $z$, $z_j$, $\theta_j$, $L_0$, $v_p$, $k$, $\tau_{\cos}$, $\tau_{\text{mag}}$, CMI, $d'$, $k_{\text{nn}}$) are used consistently throughout the paper and match `notation.md`.
- "$L_0$" is typeset correctly everywhere (no bare "L0" in the manuscript body).
- "SAE" is expanded on first use in each section. Confirmed.
- "Feature absorption" is used properly on first mention in each section. Confirmed.
- "JumpReLU SAE" and "L1-ReLU SAE" are consistently hyphenated per `glossary.md`.
- "Gemma Scope", "SAEBench", "SAELens", "TransformerLens" all match the glossary's preferred forms.

One deviation: `notation.md` lists C1 (Random probe) expected absorption as "~9.2%" but the paper reports "11.8%" in Section 3.3 and Table 2. The `notation.md` file appears to have a stale value and should be updated to match the paper's data-validated 11.8%.

### Claim-Evidence Integrity: 9/10

The automated data validation report (`data_validation_report.json`) confirms 84 of 85 numerical claims match their source data (integrity score = 0.988). The one missing-data item is the leave-one-out maximum-influence letter (V), which has source data unavailable in the JSON but is supported by the jackknife analysis.

Specific checks:

- **Table 2 numbers** (15.96%, 74.6%, 6.49%, 45.2%, etc.) all match source JSONs exactly.
- **Table 4** (strict hedging 41 tokens, 6.2%, CI [4.4%, 8.2%]) matches `tightened_hedging.json`.
- **Table 5** (0/8 recovery) matches `activation_patching_core_words.json`.
- **Table 6** (partial $\rho_s = -0.328$, restricted $\rho_s = -0.113$) matches `partial_correlation_cmi.json`.
- **Table 7** ($L_0$ absorption rates) matches `confound_decomposition_multi_l0.json`.

One internal inconsistency: The paper uses **656 FN tokens** consistently in Section 4.4 and Table 4, but the `data_validation_report.json` notes that the confound decomposition source has 657 FN tokens (from a 1,196-word vocabulary) while the tightened hedging analysis uses 656 (from a 1,195-word subset). The data validation report flags this as MEDIUM severity and recommends footnoting it. The paper currently uses 656 throughout without acknowledging the 657 figure at all. This is a minor numerical discrepancy (one token) but a reviewer who checks the source data will notice it.

The Section 5.1 reconciliation note ("14.39% in the $L_0$ sweep ... 15.96% in the first-letter replication ... difference of 1.57 percentage points") is well-handled and transparent.

### Visual Communication: 8/10

The paper has 5 figures and 8 tables -- exceeding the minimum requirement of 1 method diagram, 1 results table, and 1 analysis figure. All figures are referenced in text before they appear. Captions are self-explanatory and include key statistics.

Strengths:
- Figure 1 (control failure grouped bars) immediately communicates the central finding.
- Figure 3 ($L_0$ phase transition) with cross-layer overlay and CI bands is effective.
- Figure 4 (two-panel CMI scatter, colored by probe F1) clearly shows the confound.

Issues:
1. **No method diagram.** The paper lacks a visual overview of the absorption measurement pipeline, the four-control suite, or the confound decomposition procedure. Section 3 is entirely text-based. A schematic diagram of the pipeline (SAE encoding -> probe training -> candidate identification -> FN classification -> controls) would help readers who are unfamiliar with the Chanin protocol. The visual audit report suggests a 3-panel summary figure for the introduction; a method figure in Section 3 would serve a complementary purpose.

2. **Section 7 (Discussion) is text-heavy** with no visual elements across ~2 pages. The visual audit report notes this. A summary figure consolidating the three findings would improve accessibility.

3. **Figure 1 description in the outline** specifies a "multi-panel figure (histogram + schematic)" showing candidate count distributions, but the paper's Figure 1 is a "grouped bar chart" of absorption rates. The outline's Figure 1 concept (candidate explosion histogram) appears to have been merged into the text of Section 4.2 rather than rendered visually. The candidate explosion explanation -- arguably the paper's most important structural insight -- would benefit from its own figure.

### Writing Quality: 8/10

The prose is direct and precise. Sentences are generally short. Claims are backed by specific numbers. The paper avoids most banned patterns.

Specific issues:

1. **One surviving banned pattern (minor).** Section 2.1, paragraph 1: "SAEs have been scaled to frontier models (Templeton et al., 2024; Gao et al., 2024) and used for circuit analysis (Lindsey et al., 2025)." This sentence is a generic literature survey transition that could be cut or folded into the preceding sentence without loss.

2. **Abstract is 237 words.** Acceptable (target was 200--250), but dense. The abstract compresses three findings with many specific numbers ($\rho_s$, CI, CV, ratios). The first sentence of the abstract is 98 words long -- a reviewer may lose the thread. Consider splitting.

3. **Passive voice in a few places.**
   - Section 3.2: "Letters failing the gate are excluded from aggregate statistics but reported individually." (Passive, but acceptable for methods.)
   - Section 4.1: "A metric that assigns higher absorption scores to randomized labels than to true hierarchical labels is not measuring hierarchy-driven competitive exclusion." (This is active and effective.)

4. **Section 4.4, paragraph 3:** "The discrepancy between permissive (98.6%) and strict (6.2%) hedging is stark." The word "stark" is subjective. Replace with the specific magnitude: "The discrepancy between permissive (98.6%) and strict (6.2%) hedging spans 92.4 percentage points."

5. **Introduction paragraph structure.** The introduction is well-structured but long (~40 lines of markdown). The three findings paragraphs (Q1--Q3) each begin with a bold label, which is effective. The roadmap paragraph at the end is appropriately brief.

6. **Section 8 (Conclusion)** effectively mirrors the Q1/Q2/Q3 structure. The three recommendations are concrete and actionable. The final sentence about code release includes a placeholder "[URL]" that must be filled before submission.

## Issues for the Editor

1. **Major** -- **Table numbering out of sequence.** Section 5.1 references Table 7 while Section 6.2 references Table 6, breaking sequential order. **Fix**: Renumber all tables sequentially as they appear in the text. Table 8 (currently in Section 3.7) should become Table 2; current Tables 2--5 shift to 3--6; current Table 7 becomes 7; current Table 6 becomes 8. Alternatively, simply swap the numbers of Tables 6 and 7 as a minimal fix.

2. **Major** -- **No method diagram.** Section 3 is entirely text and tables. A reviewer unfamiliar with the Chanin protocol will struggle to follow the measurement pipeline, four-control suite, and confound decomposition without a visual guide. **Fix**: Add a pipeline/schematic figure (Figure 0 or Figure 1, renumbering existing figures) in Section 3 showing: input tokens -> SAE encoding -> probe training -> candidate identification -> FN classification -> four controls. This was partially planned in the outline ("Figure 1: Candidate Explosion, histogram + schematic") but the schematic was not realized.

3. **Major** -- **Hypotheses H1--H7 never enumerated in the paper body.** Section 7.4 and Section 8 reference "four of seven pre-registered hypotheses" but the reader cannot verify which four without consulting the proposal. **Fix**: Add a numbered hypothesis list in Section 1 or Section 3, or expand Section 7.4 to fully specify each hypothesis with its pre-registered target and observed value.

4. **Minor** -- **FN count discrepancy (656 vs. 657).** The confound decomposition and tightened hedging analyses use slightly different vocabulary sizes (1,196 vs. 1,195 words), producing 657 vs. 656 FN counts. The paper uses 656 throughout. The data validation report flags this. **Fix**: Add a brief footnote in Section 4.4 acknowledging the one-token difference and specifying which vocabulary size yields which count, similar to the existing vocabulary footnote in Section 3.2.

5. **Minor** -- **Placeholder "[URL]" in Section 8.** The code release URL at the end of the conclusion is a placeholder. **Fix**: Replace with actual URL before submission, or change to "Code and data will be released at [URL upon acceptance]" if the URL is not yet available.

## What Works Well

1. **Section 4.2 (Structural Explanation: Candidate Explosion in High Dimensions)** is the paper's best paragraph-level writing. The argument is crisp: a random unit vector in $\mathbb{R}^{2304}$ identifies 23.0% of decoder columns as candidates at $\tau_{\cos} = 0.025$; true and shuffled probes produce indistinguishable candidate counts (21.2% vs. 21.4%); the candidate step is therefore vacuous. This is exactly the kind of concrete, numbers-first explanation that makes the control failure immediately understandable.

2. **Table 6 (CMI-absorption correlation summary)** is a model of progressive statistical control. Each row adds one robustness check (raw -> partial -> restricted -> replicated), and the signal weakens monotonically. The "Interpretation" column anchors each row to a qualitative conclusion. This table alone tells the Section 6 story.

3. **Section 7.2 (Two Interpretations)** honestly presents both readings of the data without forcing premature resolution. The paragraph acknowledging that "Both interpretations agree on the empirical facts and the practical consequence" is the kind of calibrated language that builds reviewer trust.

SCORE: 8
