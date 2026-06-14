# Critique: Introduction

## Summary Assessment

The introduction is ambitious and well-structured, presenting three headline findings with specific numbers and statistical tests. The narrative arc -- from the SAE reliability promise, through the absorption threat, to the narrow empirical base, and then the cross-domain contribution -- is logically sound. However, the section suffers from a critical data consistency problem: several numbers in the intro contradict the consolidation summary JSON (the project's authoritative results record), creating a risk that reviewers will find discrepancies between the intro and later sections. The section also contains a banned pattern ("The remainder of the paper is organized as follows..."), overloads the reader with statistical details better suited to the results section, and embeds a figure inline that is not yet generated.

## Score: 6/10
**Justification**: The narrative is clear and the contribution framing is strong, but critical numerical inconsistencies would trigger a desk reject at a careful venue. Fixing the data alignment, trimming the statistical overload, and tightening prose to avoid banned patterns would bring this to a 7-8. A fully consistent, concise intro with a working Figure 1 would reach 8-9.

## Critical Issues

### Issue 1: Numerical discrepancies between intro and consolidation summary

- **Location**: Paragraphs 4-6 (the three headline findings)
- **Quote (intro)**: "absorption rates range from 11.6% (city-language) to 45.1% (city-country), with first-letter at 27.1% and city-continent at 31.4%"
- **Quote (consolidation_summary.json)**: "first-letter 34.5%, city-continent 35.8%, city-country 18.5%, city-language 13.6%"
- **Problem**: The intro's numbers match the full-mode phase1_absorption_crossdomain.json (31.4% for city-continent confirmed by raw JSON: 0.3142857) and the experiments section. But the consolidation_summary.json -- labeled as the "writing-ready summary" -- reports substantially different values (e.g., city-country: 45.1% vs 18.5%; first-letter: 27.1% vs 34.5%). The consolidation summary's `absorption_rate_table` (which shows L24_16k first-letter at 0.345 and city-continent at 0.3584) also differs from its own `hypothesis_verdicts` section (which reports 34.5% for first-letter). This creates a situation where different parts of the paper could cite different numbers for the same measurement. The 4x variation claim ("11.6% to 45.1%") needs to be verified against the single authoritative source.
- **Fix**: Designate one file as the authoritative source (likely the full-mode `phase1_absorption_crossdomain.json` plus `phase1_absorption_firstletter.json`). Reconcile all numbers in the intro, experiments section, and consolidation summary to match. Explicitly document which run produced each number. If the intro uses L24_16k from the full run (city-continent = 31.4%, first-letter = 27.1%), update the consolidation summary. If the consolidation summary's numbers are correct, update the intro and experiments section.

### Issue 2: Kruskal-Wallis test statistic reported without source verification

- **Location**: Paragraph 4
- **Quote**: "Kruskal-Wallis $H$ = 299.95, $p$ = 7.4 $\times 10^{-66}$, $N$ = 3566"
- **Problem**: This specific test statistic (H=299.95, p=7.4e-66) does not appear in the consolidation summary, which reports "Kruskal-Wallis p=0.005" -- six orders of magnitude different. The experiments section (4.1) repeats the same H=299.95 and p=7.4e-66 values. One set of numbers is wrong. With N=3566 entities across 4 hierarchies, the extremely small p-value (7.4e-66) suggests this is a per-entity test (treating each entity's absorption status as a data point), while p=0.005 from the consolidation suggests a per-hierarchy aggregate test. The intro does not specify which statistical test level is being reported.
- **Fix**: Trace the H=299.95 statistic to its source experiment output file. If it is a valid per-entity-level test, note this explicitly ("Kruskal-Wallis test across N=3,566 entity-level absorption indicators"). If it is an error, replace with the correct values. Reconcile with the consolidation summary's p=0.005.

### Issue 3: Benign/pathological numbers inconsistent across sources

- **Location**: Paragraph 6
- **Quote**: "mean absolute logit change of 3.98 nats ($n$ = 1471 instances from 50 entities)"
- **Problem**: The consolidation summary reports n=1464 instances and mean=3.992, while the phase2 benign_pathological section in the consolidation reports n_fn_instances=1464. The method section (3.5) says "1,471 false negative instances." The intro says n=1471 and mean=3.98. These small discrepancies (1471 vs 1464, 3.98 vs 3.992) suggest rounding or different filtering criteria. While individually minor, a reviewer who checks the appendix against the intro will flag this.
- **Fix**: Use the exact values from the primary experimental output file (`phase2/benign_pathological.json`). Ensure all sections cite the same n and mean. If 1471 vs 1464 reflects different filtering criteria, explain the difference or standardize.

## Major Issues

### Issue 4: Statistical overload in the three findings paragraphs

- **Location**: Paragraphs 4-6 (the three headline findings)
- **Quote**: "Kruskal-Wallis $H$ = 299.95, $p$ = 7.4 $\times 10^{-66}$, $N$ = 3566... Wilcoxon signed-rank $p$ = 0.000218, Cohen's $d$ = 1.33, $n$ = 25 words... control: 1.5%... (control: 14.5%, $n$ = 93 entities)... mean absolute logit change of 3.98 nats ($n$ = 1471 instances from 50 entities), approximately 1000$\times$ the control perturbation (0.004 nats)"
- **Problem**: The intro reads like a compressed results section. Each headline finding contains 4-6 statistical quantities. An intro should convey the key insight and one anchor number per finding; the full statistical apparatus belongs in Sections 4-5. This density makes the intro harder to read and reduces the impact of the actually striking numbers (the 4x variation, the 32.5% vs 1.5% recovery, the 1000x pathological ratio).
- **Fix**: For each finding, lead with the insight and one or two numbers. Move detailed test statistics (H values, effect sizes, sample sizes, control conditions) to the results sections. Example for Finding 1: "Absorption rates range from 11.6% to 45.1% across hierarchies at layer 24, with no simple semantic-versus-syntactic ordering (details in Section 4)."

### Issue 5: Implications paragraph is dense and introduces new statistical claims

- **Location**: Paragraph 7 (implications)
- **Quote**: "ANOVA $p$ = 0.50--0.75 for architecture effect vs. $p$ = 0.005--0.063 for hierarchy effect... $\rho$ = 0.116... $\rho$ = 0.044... $\rho$ = $-$0.20... $R^2$ = 0.088"
- **Problem**: This paragraph introduces the architecture invariance finding AND five correlational failures in a single dense block. The GAS, CMI, Absorption Tax, and rate-distortion results are being mentioned here for the first time without context. A reader encountering "$\rho = 0.116$" has no idea what GAS is yet. This paragraph tries to do the work of Sections 6 and 7 simultaneously.
- **Fix**: Split into two shorter paragraphs. First: the architecture invariance implication (one sentence with one p-value). Second: the failure of correlational methods (frame as "five correlational approaches fail to predict absorption, motivating a shift to causal methods" -- without listing each rho). Detail the individual failures in Section 7 / appendices.

### Issue 6: Figure 1 is not self-contained and references a non-existent file

- **Location**: Between paragraphs 4 and 5
- **Quote**: `![Cross-domain absorption rates at layer 24...](figures/fig1_teaser.pdf)`
- **Problem**: (a) The outline specifies Figure 1 as a panel figure with a left schematic AND right bar chart, but the intro only references the right panel (bar chart of rates). The left panel (absorption measurement schematic) is missing from the caption. (b) The figure file `fig1_teaser.pdf` is referenced but the generation script `gen_fig1_teaser.py` is listed only in the HTML comment at the bottom. There is no indication this figure actually exists yet. (c) The caption lacks sufficient detail to be self-explanatory -- it does not define what "absorption rate" means for a reader seeing only the figure.
- **Fix**: (a) Update the caption to describe both panels as specified in the outline. (b) Verify the figure exists or mark it as TODO. (c) Expand the caption: "Left: schematic of absorption measurement -- a parent probe predicts correctly on raw activations but fails on SAE-reconstructed activations when a child feature is active. Right: absorption rates across four hierarchies at layer 24 with 16k SAE, showing 4x variation from 11.6% (city-language) to 45.1% (city-country)."

### Issue 7: The intro's method preview does not match the method section

- **Location**: Paragraph 3
- **Quote**: "training linear probes across four transformer layers, we measure absorption rates on four distinct feature hierarchies spanning syntactic (first-letter) and semantic (entity-attribute) domains"
- **Problem**: The method section (3.2) reveals that RAVEL probes are measured only at L24 (not "across four transformer layers") because probe quality at earlier layers is too low (F1 = 0.37-0.72 at L6). Only first-letter probes are measured across all four layers. The intro's phrasing "training linear probes across four transformer layers" implies all four hierarchies are measured at all four layers, which is misleading.
- **Fix**: Revise to: "training linear probes at four transformer layers for first-letter and at the best-performing layer (layer 24) for entity-attribute hierarchies."

## Minor Issues

- **Paragraph 1, sentence 2**: "if each SAE feature responds to a single, coherent concept (monosemanticity), then the sparse decomposition provides a human-readable vocabulary" -- on first use of "monosemanticity," the glossary requires introducing the full term. This is done correctly here. No change needed.

- **Paragraph 2**: "Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10--40% (Karvonen et al., 2025)" -- verify this attribution is accurate. The glossary entry for SAEBench cites Karvonen et al. (2025) as a benchmark paper, not a statement about DeepMind's research priorities. If this claim comes from a different source (blog post, internal communication), it needs a different citation or qualification ("as noted in...").

- **Paragraph 3**: "100% parent-child co-occurrence by construction" -- the phrase "by construction" is technically correct but could be clearer. Consider: "every word that starts with S co-occurs with the 'starts with S' parent by definition."

- **Paragraph 4**: "first-letter at 27.1%" -- per the notation guide, this should be "first-letter at 27.1\%" or use the absorption rate notation $\alpha$ if being formal.

- **Paragraph 5**: "the first interventional evidence for competitive exclusion in SAEs" -- the proposal says "first metric-independent causal evidence," while the intro says "first interventional evidence." Use consistent phrasing. The glossary defines "competitive exclusion" and says to define it on first use. The intro uses it in paragraph 2 without a formal definition; add "(the mechanism by which a child feature suppresses the parent feature during SAE encoding)" after first use.

- **Paragraph 5**: "Zero of 1471 instances qualify as benign" -- awkward phrasing. Consider "None of the 1,471 instances qualifies as benign."

- **Paragraph 7**: The roadmap sentence ("The remainder of the paper is organized as follows. Section 2 reviews... Section 3 describes... Section 4 presents...") is a banned pattern (filler transition). At a minimum, cut this to one sentence: "We describe the methodology in Section 3 and present results in Sections 4-6." Better: remove entirely, since the section structure is standard and self-evident.

- **Paragraph 4**: "Absorption is negligible at early layers (0.7--1.0% at layer 6)" -- this implies all four hierarchies were measured at layer 6, but the experiments section (4.2) clarifies that only first-letter was measured at L6 (RAVEL probes too poor). This is misleading.

- **Terminology**: The intro uses "activation budget" (paragraph 2: "the SAE's encoder allocates the activation budget to the more specific child") which does not appear in the glossary or notation table. Either add it to the glossary or replace with standard terminology ("the SAE allocates active features to the more specific child").

## Visual Element Assessment

- [ ] Figures/tables match outline plan -- **PARTIAL**: Outline specifies a two-panel Figure 1 (left: schematic, right: bar chart). The intro only references the bar chart.
- [ ] All visuals referenced before appearance -- **YES**: Figure 1 is referenced in paragraph 4 before the image tag.
- [ ] Captions are self-explanatory -- **NO**: The caption does not define "absorption rate" or explain the measurement procedure for a reader seeing only the figure.
- [ ] No text-heavy sections that need visual support -- **PASS**: The three findings are appropriately summarized in text with Figure 1 providing visual support.

## What Works Well

1. **Paragraph 2 is the strongest paragraph in the section.** It concisely defines absorption, gives a concrete example (Saturday/starts-with-S), cites the 10-40% safety degradation, and establishes urgency -- all in four sentences. This paragraph should be the template for the rest of the intro.

2. **The three-finding structure is effective.** Bold headers for each finding make the contributions scannable. The progression from descriptive (finding 1: rates vary) to mechanistic (finding 2: causal confirmation) to consequential (finding 3: always pathological) builds a compelling narrative arc.

3. **The "100% pathological" finding is presented with appropriate force.** The 1000x ratio between parent ablation and control is the paper's most striking number, and the intro correctly highlights it. The framing as falsifying a specific hypothesis (benign absorption) is strong scientific writing.
