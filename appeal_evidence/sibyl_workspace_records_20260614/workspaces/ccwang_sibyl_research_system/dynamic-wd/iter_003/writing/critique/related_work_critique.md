# Critique: Related Work Section
**Paper:** When Does Dynamic Weight Decay Help? A Unified Framework Analysis Under AdamW
**Section reviewed:** Section 2 (Related Work)
**Reviewer:** Section Critic (iter_003)
**Date:** 2026-03-18

---

## Scores by Criterion

| Criterion | Score (1–10) | Rationale |
|-----------|-------------|-----------|
| 1. Coverage of relevant literature | 7 | All explicitly planned references are present; the four families are correctly identified. However, several adjacent literatures (optimizer-level regularization, learning rate–WD coupling, modern ResNet training recipes) that would sharpen the gap analysis are absent. |
| 2. Organization and taxonomy | 9 | The four-family taxonomy (temporal, alignment-aware, norm-matched, structural) is crisp, hierarchical, and directly anticipates the Phi framework's modulation axes. Flow within subsections is logical and coherent. |
| 3. Gap identification leading to this work | 7 | Section 2.3 ("Evaluation Fragmentation") articulates the gap clearly. However, the transition from the structural-effects family (2.2) to the fragmentation problem (2.3) is abrupt, and the fourth gap from the Introduction ("no theory for when dynamic WD matters") is mentioned in 2.3 only implicitly. |
| 4. Fair and accurate characterization of prior work | 6 | Most characterizations are accurate, but several claims about 2026 papers (Ferbach et al., Chen et al. CWD/AdamO, Truong & Truong) cite works that appear to be preprints or anticipated publications. There is no acknowledgment of this provenance uncertainty, which risks appearing misleading or speculative to reviewers. One characterization of ADANA ("up to 40% compute efficiency gains") is suspiciously strong and should be qualified or cited precisely. |
| 5. Connection to the paper's contributions | 7 | The closing sentences of each subsection successfully tie back to the paper's framework and metrics. However, the connection between the structural-effects family (Section 2.2 last paragraph) and any of the four paper contributions is never made explicit—this family appears as a disconnected appendage. |
| 6. Writing quality | 8 | Prose is clear, technical vocabulary is precise, and sentence structure is varied. Paragraph length is appropriate for a 0.75-page budget. Minor issues: one run-on sentence in §2.2 alignment-aware paragraph; "Radial Tug-of-War" appears in both §2.2 and the Introduction without cross-reference. |

**Overall Score: 7 / 10**

The Related Work section is competently structured and correctly covers all the methods that appear in the paper's experimental comparison. The taxonomy is its strongest feature and will benefit readers significantly. The main weaknesses are (a) uncertain citation provenance for several 2026 papers presented as established fact, (b) an isolated structural-effects paragraph whose connection to the paper's contributions is unclear, (c) a handful of missing references from adjacent relevant areas, and (d) a gap narrative that does not fully align with all four gaps claimed in Section 1.2.

---

## Top 3 Strengths

**Strength 1: Taxonomic clarity that directly scaffolds the Phi framework.**
The four-family taxonomy in §2.2 (temporal / alignment-aware / norm-matched / structural) is well-chosen and reader-friendly. Because the Phi framework in Section 3 organizes methods along the same four modulation axes, the taxonomy here functions as a conceptual primer that makes Section 3's Table 1 immediately intuitive. This tight structural coupling between Related Work and Methodology is exactly the kind of narrative architecture that strong NeurIPS papers exhibit.

**Strength 2: Precise and differentiated characterization of recent dynamic WD methods.**
Rather than grouping all dynamic WD methods into a single "related work" paragraph, the section correctly distinguishes the motivation and mechanism of each method. For example, the distinction between SWD's gradient-norm scheduling rationale and CWD's Pareto-optimal bilevel framework and AdamO's radial-tangential decoupling insight are all captured accurately and succinctly. This level of precision signals genuine engagement with the prior art and will strengthen the paper's credibility with reviewers who specialize in these methods.

**Strength 3: Explicit evaluation-fragmentation diagnosis that motivates the paper's metric contributions.**
Section 2.3 is the most strategically effective subsection. By naming the fragmentation problem concretely—different architectures, datasets, optimizers, hyperparameter protocols—and calling out specific examples (CWD on Lion/Muon, SWD on SGD-CIFAR, AlphaDecay on billion-parameter LLMs), the section makes the need for BEM/CSI/AIS undeniable. The OUI reference (Fernandez-Hernandez et al., 2025) is particularly well-placed: it acknowledges the existence of a diagnostic effort while correctly distinguishing it as per-method rather than comparative.

---

## Top 5 Issues with Suggestions

### Issue 1: Speculative or unverified citations for 2026 papers (Severity: HIGH)

**Problem:** Four citations are dated 2026: Ferbach et al. (ADANA), Chen et al. CWD (ICLR 2026), Chen et al. AdamO, and Truong & Truong. The current date is 2026-03-18 and ICLR 2026 decisions are recent. Presenting these as firmly established results—"reporting up to 40% compute efficiency gains" (ADANA), "proved ... Pareto-optimal" (CWD), "traverses a norm hierarchy" (Truong)—without any qualification creates risks:
1. If any of these papers are still under review or have not been published, the claim of citation is incorrect and will flag immediately for area chairs who know the community.
2. The ADANA "40% compute efficiency" figure is a specific quantitative claim that demands a precise citation (table number, section, dataset). If this number is wrong or context-dependent, reviewers from that community will notice.

**Suggestion:**
- For any 2026 paper not yet formally published, add "(preprint)" or "(under review)" to the citation, or move the reference to a footnote that flags the provenance.
- Replace "up to 40% compute efficiency gains" with the exact citation: "(Ferbach et al., 2026, Table X, [dataset/task])" and narrow the claim to the specific experimental condition where this gain was reported.
- Cross-check Chen et al. (ICLR 2026): confirm the acceptance date and use the venue name only if the paper has been officially accepted and is publicly available before submission.

---

### Issue 2: The structural-effects paragraph (§2.2 last paragraph) is disconnected from the paper's contributions (Severity: MEDIUM-HIGH)

**Problem:** The final paragraph of §2.2 cites Galanti et al. (2022), Kobayashi et al. (2024), and Truong & Truong (2026) on structural properties induced by weight decay (low-rank bias, nuclear norm regularization, norm hierarchy). These are interesting findings, but:
1. None of these structural results are evaluated, measured, or referenced in the paper's experimental sections (Sections 4–6).
2. None of the three diagnostic metrics (BEM, CSI, AIS) directly measures low-rank structure or nuclear norms.
3. The Phi framework does not explicitly model structural effects as a modulation axis—Table 1 contains no structural-effect modulator.
4. The Discussion section does not invoke these results when characterizing the Phi Invariance Conjecture's limitations.

The result is that this paragraph reads as a literature dump that increases word count without advancing the paper's argument. In a 0.75-page budget, this is costly.

**Suggestion:** Either (a) delete this paragraph and add one sentence noting that WD also induces structural properties (low-rank bias, spectral effects) that are orthogonal to the modulation focus of this paper, with a forward pointer to potential future work; or (b) connect it explicitly to the CSI metric (which uses spectral condition number) or to the Discussion section's boundary conditions for the Phi Invariance Conjecture. If the spectral condition number in CSI was inspired by the low-rank induction literature, say so here—that would make the paragraph earn its place.

---

### Issue 3: Fourth research gap from Section 1.2 is not adequately set up in Related Work (Severity: MEDIUM)

**Problem:** Section 1.2 claims four research gaps, the fourth being "No theory for when dynamic weight decay matters." The Related Work section sets up gaps 1 (no unified framework), 2 (no standardized metrics), and 3 (no controlled comparison) through §2.3's fragmentation argument. However, the theoretical gap—which is directly addressed by the Phi Invariance Conjecture in §6—receives no setup in Related Work.

Specifically, Related Work does not mention:
- Any theoretical results about AdamW's per-parameter adaptive scaling and its relationship to explicit regularization strength (e.g., the Xie & Li (2024) connection to L-infinity constrained optimization is introduced in §2.1 but not connected to the theoretical gap).
- The question of whether AdamW's adaptive updates already implicitly subsume WD modulation—which is the core theoretical claim of the paper.

**Suggestion:** Add 1–2 sentences at the end of §2.1 or §2.3 that note: "Despite these modern interpretations of WD dynamics, no theoretical framework predicts when the functional form of the Phi modulator becomes irrelevant. In particular, Xie & Li (2024)'s L-infinity interpretation suggests a potential absorption mechanism, but this has not been connected to the question of WD scheduling." This primes the reader for the conjecture in §6 and creates a clean four-gap narrative.

---

### Issue 4: Wang & Aitchison (2024) is cited without adequate context (Severity: MEDIUM)

**Problem:** The claim "Wang & Aitchison (2024) showed that optimal weight decay scales as an EMA timescale constant in epochs across model and dataset sizes" is placed in the norm-matched control group, which is a debatable placement. The EMA-timescale result is primarily about how to *set* the weight decay value (its magnitude as a function of training length), not about norm-matching per se. More importantly:
1. This is one of the paper's most relevant theoretical results—it directly informs the question of whether a "correct" constant weight decay exists, which is the baseline the paper defends in its main finding.
2. The characterization is too brief: it does not state that this work provides a principled prescription for λ selection, which is what makes it relevant to the context.
3. If Wang & Aitchison's prescription is followed and the constant baseline λ is correctly set, the invariance result becomes even stronger; this connection is never drawn.

**Suggestion:** Move the Wang & Aitchison citation to §2.1 or §2.3, and expand it slightly: "Wang & Aitchison (2024) showed that, under certain assumptions, optimal weight decay scales as an EMA timescale constant in training epochs and is invariant to model and dataset size—suggesting that a well-calibrated constant WD may already capture the available benefit, leaving little room for dynamic schedules to improve." This framing directly supports the paper's main finding.

---

### Issue 5: Kosson et al. (2023) rotational equilibrium result is not integrated into the gap narrative (Severity: LOW-MEDIUM)

**Problem:** The rotational equilibrium result from Kosson et al. (2023) is cited in §2.1 as context for why AdamW outperforms Adam+L2. This is accurate and well-placed. However, the paper's main finding—that dynamic WD does not help under AdamW—is arguably a direct consequence of AdamW already reaching a satisfactory rotational equilibrium: if WD's primary function is equilibrating the rotation of weight vectors (Kosson's view), and AdamW achieves this robustly regardless of the specific Phi modulator, then the invariance follows. This mechanistic link is never drawn.

**Suggestion:** In §2.1, after citing Kosson et al., add: "If rotational equilibrium is indeed the operative mechanism, then it may already be achieved robustly under standard AdamW, regardless of the specific form of WD modulation—a hypothesis our experiments will test." This one sentence creates a forward-looking thread that the Discussion section (§6.1) can pick up when formulating the mechanistic hypothesis for the Phi Invariance Conjecture.

---

## Missing References or Miscited Works

### Missing References

1. **Hanson & Pratt (1988) "Comparing biases for minimal network construction with back-propagation"** — Often cited alongside Krogh & Hertz (1991) as the original proposal of weight decay in neural networks. Including one original source is sufficient, but acknowledging both would show broader historical coverage.

2. **Zhang et al. (2024) "Transformers without Normalization" / related work on weight decay in normalization-free architectures** — Given that the paper's limitations note different behavior possible in Vision Transformers with layer normalization, at least one citation on WD behavior in transformer architectures would strengthen the claims about boundary conditions.

3. **Loshchilov & Hutter (2017) "SGDR: Stochastic Gradient Descent with Warm Restarts"** — Cosine learning rate scheduling (referenced as "cosine annealing") is cited in the experimental setup but its original source is absent from Related Work, even though the paper uses it and the cosine WD schedule is one of the evaluated methods.

4. **Zhuang et al. (2022) "Surrogate Gap Minimization Improves Sharpness-Aware Training" or other SAM-related work** — SAM and its variants (ASAM, mSAM) also modulate effective regularization per-parameter adaptively, and share conceptual overlap with the directional modulation family. The absence of any mention of sharpness-aware optimization as an adjacent approach is a gap that careful reviewers may notice.

5. **Outmezguine & Levi (2024)** — This paper is mentioned in the outline's reference list (Discussion section) but does not appear anywhere in the Related Work section. If it is relevant to the WD-as-dynamics-modifier interpretation or to the Phi Invariance Conjecture's theoretical underpinning, it should appear here with a brief characterization.

### Potentially Miscited Works

1. **Ferbach et al. (2026) ADANA "up to 40% compute efficiency gains"** — This specific quantitative claim appears nowhere verifiable from public preprints or publications as of 2026-03-18. The claim should be sourced to a specific table or result, or softened to "reported significant compute efficiency gains" until the publication is confirmed.

2. **Truong & Truong (2026)** — This citation appears in §2.2 (structural effects) and in the outline's reference list under the Discussion section. The characterization ("analyzes how weight decay traverses a norm hierarchy from shortcut to structured representations") is vague enough that it is unclear what claim of the paper it supports. This reference should either be expanded with a precise characterization or removed until the paper is verifiable.

3. **Xie & Li (2024) "AdamW implicitly performs l-infinity-norm constrained optimization"** — This result is attributed only to Xie & Li (2024), but the L-infinity connection was partially discussed in the original Loshchilov & Hutter (2019) paper as well. The citation is not wrong, but noting that this interpretation was developed across multiple works would be more accurate.

---

## Summary for Authors

The Related Work section performs its core function well: it establishes the four method families, introduces the fragmentation problem, and sets up the need for BEM/CSI/AIS. The taxonomy is a genuine strength that will help readers navigate the paper.

The most important revision is addressing the citation provenance of 2026 papers: reviewers and area chairs who work in this area will scrutinize whether these papers exist and whether the specific claims (especially the "40% compute efficiency" figure for ADANA) are accurately characterized. A small number of "(preprint)" qualifiers or softened claims will significantly increase the section's credibility.

The structural-effects paragraph should be either connected to the paper's contributions or condensed to a single sentence—in a 0.75-page section, every paragraph must earn its place. The missing connection between the Kosson rotational equilibrium result and the Phi Invariance Conjecture is a low-cost, high-impact addition that would substantially strengthen the paper's theoretical narrative.

---

## Addendum: Cross-Section Consistency Check

**Critical Terminology Inconsistency (requires fix before submission):**

Section 2.2 labels the fourth family as **"Structural effects"**, but:
- Section 1.2 (Research Gap) calls it **"spatial modulation"**
- Section 3.2 (Method, Table 1) uses **"Spatial"** as the axis label

This three-way mismatch will confuse readers trying to map the Related Work taxonomy onto the Phi framework's four-axis structure. The fix is straightforward: rename Section 2.2's fourth subsection heading from "Structural effects" to "Spatial modulation" (and note that WD induces structural effects as a consequence of this spatial modulation).

**AdamO Integration Gap:**
Related Work discusses AdamO (Chen et al. 2026b) under Section 2.2 alignment-aware methods. However, Method Section 3.2 Table 1 does not include AdamO as a Phi special case. If AdamO changes the optimizer architecture (decoupling radial/tangential dynamics) rather than just the WD modulator, it technically falls outside the Phi framework. The Related Work should acknowledge this boundary explicitly: "Note that AdamO modifies the optimizer's parameter update rule rather than solely the WD modulation function, placing it partially outside the scope of the Phi framework defined in Section 3."

**SWD Axis Classification Ambiguity:**
Related Work places SWD under "Temporal scheduling" but Method Section 3.2 classifies it as "Temporal-gradient" axis. Add a parenthetical: "(SWD also conditions on gradient norms, bridging temporal and directional modulation axes, as formalized in Section 3.2)."
