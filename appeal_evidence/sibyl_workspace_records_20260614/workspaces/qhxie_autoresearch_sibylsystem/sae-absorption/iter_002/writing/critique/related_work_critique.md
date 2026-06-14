# Critique: Related Work

## Summary Assessment

The Related Work section is technically strong: every cited work is connected back to what this paper does differently, all numbers are sourced, and the summary table at the end is a useful positioning device. The section covers the six distinct threads the outline requires. The main weaknesses are a structural one (the Chanin-plus-SAEBench opening paragraph buries the paper's departure from prior work under a lot of setup) and several inter-section consistency issues that a reviewer will notice immediately when they read Method alongside Related Work.

## Score: 7/10

**Justification**: The section earns a 7 because it has real content density, specific numbers, and honest confound acknowledgment. It falls short of an 8 because (a) the lead of the first paragraph is not the paper's gap — it is Chanin et al.'s results — which slows the reader's understanding of the contribution; (b) three technical claims in this section are mildly inconsistent with what Method and Experiments actually report; and (c) the summary table contains one factual labeling error that a reviewer will flag.

---

## Critical Issues

### Issue 1: Summary table labels Tang et al. as "N/A" for probe-free and weight-only — incorrect

- **Location**: Summary table, row "Tang et al. \citeyear{tang2024unified}"
- **Quote**: `| Tang et al. \citeyear{tang2024unified} | N/A | N/A | No | No (training) | Theoretical |`
- **Problem**: "Probe-free" and "weight-only" are labeled N/A but Tang et al.'s method is a *training*-time intervention — it requires neither probes nor activation data for detection. It is therefore *not* evaluated on the detection dimensions (probe-free / weight-only), so the table should mark those two cells differently from "N/A" (which reads as "the dimension does not apply"). The Tang et al. row has a "No (training)" in the Post-hoc column, which is the right characterization — but columns 1 and 2 should say "Not applicable (training-time)" or be annotated with a footnote, not N/A, which the reader will conflate with "they did not need probes/weights" when the truth is "they operate at a different point in the pipeline."
- **Fix**: Change "N/A | N/A" to "— | —" (standard LaTeX em-dash for "dimension not evaluated") and add a table footnote: "$^\dagger$Tang et al. is a training-time intervention; probe-free/weight-only dimensions do not apply."

---

## Major Issues

### Issue 2: The section's first claim about absorption rates (15--35%) conflicts with Intro's framing of 92--97%

- **Location**: Paragraph 1, sentence 2: "report absorption rates of 15--35\% across Gemma Scope, Llama 3.2, and Qwen2 SAEs"
- **Quote**: "they report absorption rates of 15--35\% across Gemma Scope, Llama 3.2, and Qwen2 SAEs spanning multiple widths and layers."
- **Problem**: The Intro (Section 1) opens with "An SAE audit that trusts activation presence alone will conclude that the child concept is absent when it is not" and quotes the paper's own Phase Stability result: "the child latent appears to encode a concept, but that encoding is systematically suppressed on 92–97\% of inputs where the concept is present." The 15–35% figure is Chanin et al.'s *inter-model* absorption rate (fraction of child features absorbed across SAEs), whereas 92–97% is this paper's *per-feature* absorption rate (fraction of inputs where an absorbed child fails to fire). The two figures measure different things, but they appear in consecutive sections without explanation, and a reviewer reading front-to-back will flag the apparent contradiction. The Related Work section does not clarify the distinction.
- **Fix**: After "15--35\% across..." add a parenthetical: "(fraction of child features that are absorbed; our paper reports per-feature suppression rates of 0.919--0.968, a distinct quantity — see Section~\ref{sec:experiments})." Alternatively, add a sentence: "Note that this measures the fraction of features that absorb, not the suppression rate per absorbed feature; the latter is reported in Section~\ref{sec:experiments}."

### Issue 3: The OrtSAE claim about "reduces absorption by 65%" has no source

- **Location**: "Architectural Mitigations" paragraph, sentence about OrtSAE: "OrtSAE reduces absorption by 65\% on the first-letter task."
- **Quote**: "OrtSAE reduces absorption by 65\% on the first-letter task."
- **Problem**: The citation is \cite{korznikov2025ortsae} but the "65\%" figure has no in-text source for where this number comes from (the OrtSAE paper itself, or SAEBench, or an independent replication). No other figure in the architectural mitigation paragraph carries a specific percentage claim without a citation tag pointing to the specific table or result. If this comes from OrtSAE's paper, cite it explicitly. If from SAEBench, say so. Without attribution, a reviewer cannot verify it.
- **Fix**: Add the source inline: "OrtSAE reduces absorption by 65\% on the first-letter task \cite[Table 3]{korznikov2025ortsae}" (or whichever specific source and location applies).

### Issue 4: The EDA amortization confounder paragraph cites AUROC = 0.650, but this is the EDA result, not a cross-directional metric — the sentence is ambiguous

- **Location**: "Encoder-Decoder Alignment and the Amortization Gap" section, final sentence: "EDA achieves AUROC = 0.650 above permutation null despite this confounder"
- **Quote**: "We acknowledge this tension explicitly and note that EDA achieves AUROC = 0.650 above permutation null despite this confounder, suggesting that absorption-driven EDA is the dominant signal at GPT-2 Small layer 6."
- **Problem**: The statement "absorption-driven EDA is the dominant signal" is doing logical work that the AUROC alone cannot support. AUROC = 0.650 means EDA is a *statistically significant* signal (z = 2.49 above null), but "dominant" implies it exceeds the amortization-gap contribution — which requires showing the amortization gap EDA is small by comparison. No such quantification is given. The claim oversells the evidence.
- **Fix**: Replace "suggesting that absorption-driven EDA is the dominant signal" with "suggesting that absorption-driven encoder drift contributes meaningfully to EDA at this layer, though the amortization gap contribution cannot be independently quantified in this work."

### Issue 5: "Feature hedging" paragraph references Section~\ref{sec:experiments} but the claim is not straightforwardly in that section

- **Location**: "Encoder-Decoder Alignment and the Amortization Gap" section, last paragraph: "Our phase-stability results (Section~\ref{sec:experiments}) are consistent with this distinction: absorption rates remain uniformly high at 0.919--0.968 across widths from 12k to 98k, showing no mitigation from width alone."
- **Problem**: The hedging-vs-absorption distinction is about capacity regime: hedging at narrow widths, absorption regardless of width. The Related Work paragraph implies the paper tests hedging vs. absorption at narrow SAE widths, but the smallest width tested is 12k (which is not narrow by standard hedging-research standards — hedging typically occurs at widths in the 1k–4k range). The claim "no mitigation from width alone" is valid, but the implicit comparison to hedging may mislead readers who expect the paper also tested sub-12k widths.
- **Fix**: Add a qualifier: "across widths from 12k to 98k (widths at which feature hedging is not the dominant failure mode), showing no mitigation from width alone in this range."

---

## Minor Issues

- **Paragraph 1, sentence on SynthSAEBench**: "SynthSAEBench \cite{synthsaebench2026} constructs large-scale synthetic data with known ground-truth feature hierarchies..." — The year `2026` is notable because this is 2026-04-13. If the citation is genuinely a 2026 preprint, it is fine; but if this is a placeholder cite that does not yet exist, it needs to be removed or flagged as "in preparation." Verify that the bib entry is real.

- **Tian et al. sensitivity metric**: "EDA is strictly more parsimonious: it requires only the weight matrices, with no activation data or curation." The word "strictly" is strong; EDA requires a brief forward pass to compute activation frequencies for some baselines (as noted in Method Section 3.4). EDA itself requires only weights, but the reader needs to know that "strictly weight-only" refers to EDA specifically, not the whole detection suite described in the Related Work. This is clarified in the Method but not here — add "(EDA itself)" before "requires only the weight matrices."

- **Song et al. consistency paragraph**: "TopK SAEs achieve 0.80 pairwise match, the best among tested architectures" — the paper reports TopK SAEs achieve best consistency, but this contradicts the narrative goal of positioning TopK SAEs as a comparison architecture (Method 3.4). The sentence sits fine technically, but a reviewer might ask: if TopK SAEs are most consistent, why not test EDA primarily on TopK? A one-sentence acknowledgment would preempt this question: "This consistency advantage motivates including the TopK architecture in our scaling suite (Section~\ref{subsec:scaling})."

- **KronSAE claim**: "KronSAE \cite{kronsae2025} applies Kronecker product factorization to SAE latents and reduces mean absorption fraction across all sparsity levels." — No percentage or magnitude is given, making this the weakest claim in the architectural mitigations paragraph. All other architectures have at least one quantitative result (65% for OrtSAE, "ranks first" for Matryoshka, "0.0068 vs. 0.1402" for ATM). Add a number or say "reduces mean absorption fraction by an unreported margin \cite{kronsae2025}."

- **Masked regularization citation**: `\cite{narayanaswamy2026masked}` — like SynthSAEBench, this is a 2026 citation. Verify it exists and is publicly available.

- **Bricken et al. cross-directional cosine claim**: "No prior work uses $\cos(\hat{e}_p, d_c)$...as a pairwise absorption signal." — The Related Work section says this, but the intro does not. This is the correct place to claim novelty, but the claim is not falsifiable as written because it depends on the reader having searched all prior literature. Add "To the best of our literature survey, no prior work..." — wait, the glossary bans "to the best of our knowledge." Instead: "A systematic literature search found no prior work using..." or simply "No prior work in the SAE literature uses...". The current phrasing is fine per the glossary but should be flagged: the Bricken et al. observation described (decoder-decoder cosine clusters related features) is not formally cited as "not using encoder directions" — it is inferred. Make clear this is the authors' reading: "Bricken et al. \citeyear{bricken2023monosemanticity} observe that decoder-decoder cosine clusters related features; that work does not examine encoder directions or parent-child pairs specifically."

- **Narrative flow of the "Cross-Directional Cosine Measures" subsection**: The subsection correctly claims novelty for $\cos(\hat{e}_p, d_c)$, then gives the AUROC = 0.730 and Cohen's d = 0.552 — both of which also appear in the Intro. This triple-reporting (Intro abstract → intro contributions → related work) is redundant. In related work, cite the result as evidence but point the reader to Section~\ref{sec:experiments}: "We report full detection results in Section~\ref{sec:experiments}." The current repetition uses up space without adding positioning value.

---

## Visual Element Assessment

- [x] Figures/tables match outline plan: The outline assigns only Table 1 (summary table, inline) to this section. The summary table is present and inline.
- [x] All visuals referenced before appearance: The table is referenced in the paragraph immediately preceding it ("Table~\ref{tab:related} positions the contributions...").
- [x] Captions are self-explanatory: The table has no caption line in the current draft — only a bolded lead sentence before it. A LaTeX `\caption{}` should be added so the table is self-contained when referenced from other sections.
- [ ] No text-heavy sections that need visual support: The "Architectural Mitigations" paragraph presents five architectural families purely in prose. A compact comparison table (architecture / mitigation mechanism / training required / absorption reduction / source) would improve scannability and is warranted given the density. The outline's Figure and Table Plan does not include this, but it would strengthen the section.

---

## What Works Well

1. **The amortization gap confounder is acknowledged explicitly and early** ("The amortization gap is a confounder for EDA") — this preempts the most predictable reviewer attack on the paper's detection claims. The paragraph is honest and specific.

2. **The opening paragraph on Chanin et al. is technically precise** — the distinction between "absorption rates of 15--35\%" and "every tested SAE architecture" is correctly stated, and the limitation of the FeatureAbsorptionCalculator (requires probe directions) is correctly summarized in a single sentence.

3. **The cross-directional cosine novelty claim is properly scoped** — the section correctly distinguishes this work's $\cos(\hat{e}_p, d_c)$ from Bricken et al.'s decoder-decoder cosine by noting the unit of analysis (individual parent-child pairs vs. feature families) and the direction used (encoder vs. decoder). This is the kind of precise differentiation that reviewers reward.
