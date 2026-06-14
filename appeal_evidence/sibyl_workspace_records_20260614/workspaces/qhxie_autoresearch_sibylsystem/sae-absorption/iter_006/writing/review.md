# Writing Quality Review

## Summary

The paper audits the Chanin absorption metric on Gemma 2 2B JumpReLU SAEs across five hierarchy domains and reports three findings: (1) the metric fails basic controls -- shuffled labels produce 4.7x higher absorption than true labels in all five domains, with confound decomposition revealing that 98.6% of false negatives at L0=22 are hedging rather than competitive exclusion; (2) absorption declines monotonically from 42.85% to 0.84% as L0 increases from 22 to 176, exhibiting a phase transition around L0=40-80 stable across three layers; (3) conditional mutual information at subspace dimension d'=10 correlates negatively with absorption susceptibility (Spearman rho = -0.383, Cohen's d = -0.924), consistent with rate-distortion theory. The core argument -- validate metrics before building mitigations -- is clear and well-supported.

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a logical progression: background (Section 2), methodology (Section 3), metric audit (Section 4), L0 characterization (Section 5), rate-distortion diagnostic (Section 6), discussion and conclusion (Sections 7-8). Each section motivates the next explicitly (e.g., the metric audit constrains interpretation of subsequent results; the L0 sweep reveals a robust pattern despite metric limitations; CMI connects to theory). The abstract accurately represents all three findings and is upfront about key limitations (CMI d'=10 only, Bonferroni correction).

Two structural issues remain:

1. **Section 5.1 rate discrepancy note.** The paper reports L12-16k absorption at L0=82 as 14.39% (confound decomposition pipeline, 1,195 tested words) and 15.96% (improved first-letter protocol, 1,203 tested words). The explanatory note in Section 5.1 now clarifies the vocabulary size difference and probe set difference, which is an improvement. However, a reviewer will still find two numbers for the same configuration disorienting. The paper would be cleaner with one canonical number and the other in a footnote.

2. **Section 5.3 JumpReLU vs. L1-ReLU comparison earns limited space.** The paper acknowledges the cross-model confound (Gemma 2 2B vs. GPT-2 Small) and the comparison adds only the directional observation that L1-ReLU shows uniformly high absorption without an L0 phase transition. The same point could be made in 3 sentences within the discussion, freeing half a page for a method diagram or per-letter results table.

### Notation & Terminology Consistency: 8/10

Cross-checking against notation.md and glossary.md:

1. **Notation.md definitions are now correct.** Hedging is defined as "resolves at higher L0 values as sparsity relaxes" and hierarchy-driven as "persists across all tested L0 values regardless of sparsity pressure" -- matching the paper's operational definitions. The previous review's flagged inversion has been corrected.

2. **Outline.md line 118 still has inverted definitions.** The classification rule reads: "if token's parent FN persists across ALL L0 values, classify as hedging; if FN appears only at low L0 and recovers at high L0, classify as hierarchy-driven." This is backwards. The paper and notation.md use the correct definitions, but a contributor consulting the outline will be confused. Not reader-facing, but a maintenance hazard.

3. **JumpReLU author name.** The paper body, notation.md, and glossary.md now all use "Rajamanoharan et al., 2024" consistently. The visual audit confirmed this was standardized in the integration round.

4. **Spearman rho notation.** The paper uses "$\rho_s$" throughout (subscript s for Spearman). The glossary recommends "Spearman $\rho$" on first use and "$\rho$" thereafter. The paper's convention is arguably clearer and internally consistent. Minor discrepancy -- not worth changing.

5. **The variable $\alpha$ for absorption rate** appears in Section 3.5 ("$\alpha \geq 0.10$; $n = 13$") without being defined in the paper's running text. It is defined in notation.md but a reader of only the paper will not know what $\alpha$ refers to. Either define it on first use or replace with the phrase "absorption rate."

6. **Control C4.** Notation.md lists C4 as "Untrained SAE" -- matching the paper. The previous review's flag about a "Null-domain [planned]" mismatch no longer applies; notation.md has been updated.

### Claim-Evidence Integrity: 8/10

Substantial improvement from the previous review round. Key verifications:

**CMI group statistics.** The paper (paper.md) now reports absorbed mean CMI = 0.649 +/- 0.187, non-absorbed mean = 0.861 +/- 0.258, Mann-Whitney U = 28.0, p = 0.045 (two-sided). Source data (`exp/results/full/cmi_estimation.json`) confirms: absorbed_mean = 0.6492, absorbed_std = 0.1870 (population), non_absorbed_mean = 0.8612, U = 28.0, p = 0.04514. The paper rounds correctly. One precision note: the paper reports absorbed_std as 0.187, matching the population standard deviation (0.1870); the sample standard deviation is 0.1947. Most journals expect sample standard deviation. This is a minor point but worth deciding.

**CRITICAL: LaTeX file (main.tex) still contains old incorrect values.** The abstract (line 31) reports "0.687 vs. 0.861; Mann-Whitney $p = 0.042$". The introduction (line 56) reports "mean 0.687 $\pm$ 0.187 vs. 0.861 $\pm$ 0.258; Mann-Whitney $p = 0.042$; Cohen's $d = -0.924$". Section 6.2 (line 302) reports "Absorbed mean CMI: $0.687 \pm 0.187$" and line 304 reports "Mann-Whitney $U = 41.0$, two-sided $p = 0.042$." Figure 4 caption (line 313) also uses $p = 0.042$. These must be corrected to 0.649, U = 28.0, p = 0.045 to match both the source data and paper.md.

**L0 sweep absorption rates.** Source data confirms: L0=22: 42.85% (CI [0.4008, 0.4561]), L0=41: 37.49% (CI [0.3481, 0.4017]), L0=82: 14.39% (CI [0.1238, 0.164]), L0=176: 0.84% (CI [0.0033, 0.0142]). Paper matches exactly.

**Cross-L0 decomposition.** Source data confirms: L0=22: 657 FNs, 648 hedging (98.6%), 9 hierarchy-driven (1.4%). L0=41: 489 FNs, 480 hedging (98.2%), 9 hierarchy-driven (1.8%). L0=82: 185 FNs, 176 hedging (95.1%), 9 hierarchy-driven (4.9%). L0=176: 10 FNs, 1 hedging (10.0%), 9 hierarchy-driven (90.0%). Paper matches exactly.

**First-letter improved results.** Source data confirms: aggregate absorption rate = 15.96% (192/1203), mean probe F1 = 0.817, 10/25 letters pass F1 > 0.85 gate. Cross-layer: L10 = 13.88%, L20 = 13.55%. Paper matches exactly.

**Vocabulary size discrepancy.** The confound decomposition uses 1,196 words (1,195 tested, excluding letter X with n=1). The improved first-letter experiment uses 1,204 words (1,203 tested). The paper explains this in Section 3.2: "the minor size difference reflects separate tokenization runs on the same raw word list." This is adequate but could be clearer -- the first_letter_improved vocabulary has different per-letter counts (e.g., D=61 vs. D=60, G=64 vs. G=62) suggesting genuinely different tokenization, not just X exclusion.

**One discrepancy in the confound_decomposition.json (single-L0 version).** The earlier single-L0 confound decomposition (not multi-L0) reported 96.9% hierarchy-driven at L0=22 on only 574 words -- the opposite of the multi-L0 result (1.4% hierarchy-driven at L0=22 on 1,195 words). The paper correctly uses the multi-L0 result and explains in Section 7.4 that "the pilot's 96.9% hierarchy-driven figure was a methodological artifact---it classified based on within-L0 analysis rather than cross-L0 persistence." This is honest and well-handled.

### Visual Communication: 9/10

All five figures are present, referenced before they appear, and follow a logical visual narrative: control failure (Fig 1) -> decomposition (Fig 2) -> phase transition (Fig 3) -> theoretical predictor (Fig 4) -> dimension sensitivity (Fig 5). Captions are self-explanatory. Color scheme is consistent (verified against style_config.py). Figure 5 was promoted from appendix to main text at Section 6.5, which improves transparency about the CMI dimension limitation.

Two items:

1. **No method diagram.** A pipeline schematic (probe training -> feature identification -> absorption classification -> confound decomposition) would help readers unfamiliar with the Chanin protocol. For NeurIPS, a method figure is nearly expected. The paper relies entirely on prose in Section 3.

2. **Tables 3 and 4 from the outline were not included.** The per-letter breakdown (Table 3) would strengthen Section 4.3 by making the probe F1 vs. absorption rate relationship visible at a glance. The width-L0 grid (Table 4) would ground the regression statistics in Section 5.2. The prose descriptions are adequate but less scannable than tables.

### Writing Quality: 8/10

The writing is direct and evidence-first throughout. Numbers are specific and appropriately precise. Claims are hedged when warranted ("directionally correct," "pending validation," "if replicated"). The paper avoids all banned patterns in its current form.

Issues:

1. **Abstract length.** At approximately 290 words, the abstract exceeds the outline target of 200 words. For NeurIPS (no hard limit) this is acceptable, but the density is high. The abstract packs in five domain results, confound decomposition numbers, L0 sweep numbers, CMI statistics, and the geometric constant. The geometric constant detail and the Bonferroni correction could be deferred to the body without losing the abstract's message.

2. **Section 4.1 post-table redundancy.** The paragraph after Table 2 ("A metric that assigns higher absorption scores to randomized labels than to true hierarchical labels is not measuring hierarchy-driven competitive exclusion") is a near-verbatim repetition of the same point made in the introduction. The reader has encountered this claim three times by this point: in the introduction, in the prose before Table 2, and again after Table 2. Cut or merge.

3. **Section 7.1 hedging-the-hedging.** "Two caveats limit the strength of this conclusion" is followed by two defensive qualifications. The first caveat ("mitigations may genuinely improve feature quality through mechanisms that benefit features independently of competitive exclusion") is important and should stay. The second ("on L1-ReLU SAEs, where the metric was developed, competitive exclusion may dominate") is already stated in Section 5.3 and Section 7.2. Remove the duplication.

4. **Section 4.2 mentions "plus 4 additional words identified by the cross-L0 persistence criterion."** The source data shows only 5 hierarchy-driven words by name (eight, lower, liked, offer, often). The remaining 4 are not named. A reviewer will want to know what they are. Either name all 9 or explain why only 5 are named (e.g., the remaining 4 lack identifiable child latents with high magnitude ratio).

## Issues for the Editor

1. **Critical** -- **LaTeX file (main.tex) uses old incorrect CMI statistics.** The abstract (line 31), introduction (line 56), Section 6.2 (lines 302, 304), and Figure 4 caption (line 313) all report absorbed mean CMI = 0.687, Mann-Whitney U = 41.0, and p = 0.042. The correct values (matching source data and paper.md) are: absorbed mean CMI = 0.649, Mann-Whitney U = 28.0, p = 0.045. **Fix**: Search-and-replace all instances of 0.687 -> 0.649, U = 41.0 -> U = 28.0, and p = 0.042 -> p = 0.045 throughout main.tex. Also update the one-sided p reference if present.

2. **Major** -- **Outline.md line 118 has inverted confound decomposition definitions.** The classification rule reads: "if token's parent FN persists across ALL L0 values, classify as hedging; if FN appears only at low L0 and recovers at high L0, classify as hierarchy-driven." This is the opposite of the paper's correct definitions. **Fix**: Swap the labels in outline.md line 118 so hedging = "appears only at low L0 and recovers at high L0" and hierarchy-driven = "persists across ALL L0 values."

3. **Major** -- **The variable $\alpha$ is used without definition in the paper body.** Section 3.5 uses "$\alpha \geq 0.10$" and "$\alpha < 0.05$" to denote absorption rate thresholds, but $\alpha$ is never defined in the running text. **Fix**: Add a parenthetical definition on first use: "absorbed letters (absorption rate $\alpha \geq 0.10$; $n = 13$)."

4. **Minor** -- **Four of nine persistent core words are unnamed.** Section 4.2 and Section 5.4 name "eight," "lower," "liked," "offer," and "often" but refer to "plus 4 additional words" without identifying them. The source data (`cross_l0_classification` in `confound_decomposition_multi_l0.json`) lists only these 5 in the hierarchy_details at L0=22. Either all 9 words should be listed (by examining which words are FN at all four L0 values) or the text should explain that only 5 had identifiable absorbing features and the other 4 persisted as false negatives without clear absorption signal. **Fix**: Run the cross-L0 persistence check on the full data to identify and name all 9 words.

5. **Minor** -- **Standard deviation convention.** The paper reports absorbed CMI std = 0.187, which is the population standard deviation (N denominator). The sample standard deviation (N-1 denominator) is 0.195. Most ML venues expect sample standard deviation. **Fix**: Decide on one convention and apply consistently. If using population std, note this choice.

## What Works Well

1. **Section 4.2 (confound decomposition)** remains the paper's strongest contribution. The cross-L0 persistence criterion is a clean, operationally defined classification. The 98.6% hedging result at L0=22 is striking and directly supported by source data (648/657 false negatives resolve at higher L0). The monotonic shift from 98.6% hedging at L0=22 to 90% hierarchy-driven at L0=176 tells a compelling story about what the metric captures at different sparsity levels.

2. **The control suite (Section 3.3) and its universal failure (Section 4.1)** is the paper's most devastating argument. Shuffled-label absorption exceeding true-label absorption in all five domains (verified against source data: ratios range from 2.7x to infinity) is immediately convincing evidence that the metric does not measure what it claims to measure on JumpReLU SAEs. The presentation is clean and lets the numbers speak.

3. **Section 7.4 (negative results) and Section 7.5 (limitations)** demonstrate rigorous self-criticism. Four falsified hypotheses are reported with specific expected-vs-observed outcomes. The CMI dimension instability is discussed with both favorable and unfavorable interpretations. The Bonferroni-corrected p-value (0.236) is given equal prominence to the uncorrected p-value (0.059). This transparency will earn reviewer trust.

SCORE: 8
