# Critique: Method (Section 3)

## Summary Assessment
The method section delivers a clean derivation of EDA from geometric first principles, a formally stated lower-bound theorem with proof sketch, the D-EDA directional extension, synthetic validation, and baseline comparisons. The theoretical core (Sections 3.1--3.2) is the strongest part: the derivation is complete, the theorem conditions are stated, and the necessary-but-not-sufficient caveat is explicit and well argued. However, the section structurally overreaches by including full quantitative validation results (Sections 3.4--3.5) that duplicate Section 4 content, the proof sketch omits a key intermediate step, and several technical quantities ($\mu$, $\delta$-absorption, the top-3 threshold) are introduced without formal definition or justification.

## Score: 7/10
**Justification**: The theoretical core (3.1--3.2) earns an 8 by itself. Sections 3.4 and 3.5 lower the score by bleeding empirical results into the method section, creating redundancy with Section 4 and violating the standard Method/Experiments separation expected at NeurIPS/ICML. To reach 8: (a) move synthetic validation numbers and baseline AUROC comparisons to Section 4, (b) formally define $\delta$-absorption before Theorem 1, (c) specify the D-EDA regularizer $\mu$, (d) resolve the global-minimum vs. partial-minimum inconsistency between 3.1 and 3.2.

---

## Critical Issues

### Issue 1: Sections 3.4 and 3.5 contain empirical validation that belongs in Section 4
- **Location**: Section 3.4 (SynthSAEBench) and Section 3.5 (EDA vs. Baselines)
- **Quote**: "EDA achieves perfect discrimination on this synthetic data. Across all 5 trials, AUROC = 1.000 and best-threshold F1 = 0.974." / "EDA outperforms all baselines. The strongest comparison is at L5-16k: EDA achieves AUROC = 0.698 vs. decoder cosine similarity at 0.302"
- **Problem**: The outline allocates Section 3 to theory/derivation (~900 words) and places baseline comparisons as a "brief preview of Figure 3." The current draft includes full AUROC values, F1 scores, median EDA values, separation ratios, DeLong p-values, and shuffled null AUROC --- all of which duplicate Section 4.2's content (which reports the identical +0.396 AUROC improvement and the same DeLong p-value). This creates three problems: (1) a reader encountering "L5-16k" in Section 3.5 has not yet been told what this notation means (defined in Section 4.1); (2) redundancy with Section 4.2 dilutes the validation section's impact; (3) the section exceeds its ~900 word budget. Reviewers at NeurIPS/ICML expect the Method section to establish "what the metric is and why it should work," reserving "how well it performs" for Experiments.
- **Fix**: Reduce 3.4 to the SynthSAEBench setup description and a one-sentence forward reference: "Section 4 reports AUROC = 1.000 on this benchmark, validating the lower bound's predictions." Reduce 3.5 to a single paragraph defining the three baselines conceptually with a forward reference: "Full detection results across 8 SAE configurations, including statistical comparisons, appear in Section 4." Remove all specific AUROC numbers and test statistics from Section 3.

### Issue 2: Equation numbering does not match the outline contract
- **Location**: Equations (1) through (5) throughout Section 3
- **Quote**: SDL loss = Equation (1), EDA = Equation (2), bound = Equation (3), residual = Equation (4), sparse regression = Equation (5)
- **Problem**: The outline states "EDA formula (Equation 1)." The current draft assigns Equation (1) to the SDL loss, pushing EDA to Equation (2). Any section writer or cross-reference using the outline's equation numbers will produce incorrect references. Cross-checking with the experiments section: Section 4 references the correct equation numbers from this draft (not the outline), so the sections are internally consistent but diverge from the outline.
- **Fix**: Update the outline to match the draft's numbering scheme (loss = Eq. 1, EDA = Eq. 2, etc.), since the experiments section already uses these numbers. The current numbering is logically sound --- it is the outline that needs updating. Add a note in the outline acknowledging the renumbering.

---

## Major Issues

### Issue 3: $\delta$-absorption is never formally defined
- **Location**: Section 3.2, Theorem 1 statement
- **Quote**: "if latent $j$ exhibits $\delta$-absorption of child $c$, then: $\text{EDA}(j) \geq \delta^2 \sin^2(\theta_{jc}) / (2 + \delta^2)$"
- **Problem**: Theorem 1 says "if latent $j$ exhibits $\delta$-absorption of child $c$" and the notation table defines $\delta$ as "absorption degree: magnitude of suppression of parent latent." Neither gives a formal mathematical definition. Is $\delta$ the activation suppression ($\mathbb{E}[z_j | z_c = 0] - \mathbb{E}[z_j | z_c > 0]$)? A ratio? An angle? An optimization-theoretic quantity from Tang et al.? Without this definition, Theorem 1 is not self-contained and cannot be verified by a reviewer.
- **Fix**: Add a formal definition before Theorem 1. For example: "**Definition.** Latent $j$ exhibits $\delta$-absorption of child $c$ if the encoder direction perturbation induced by child-active inputs has magnitude $\delta$: $\|w_{e,j}^{\text{child-active}} - w_{e,j}^{\text{child-inactive}}\| = \delta$ (Tang et al., 2025, Definition X)." Reference the specific Tang et al. definition.

### Issue 4: Proof sketch omits the key intermediate step producing the denominator
- **Location**: Section 3.2, proof sketch paragraph
- **Quote**: "Projecting this perturbation onto the component perpendicular to $d_j$ yields a residual with magnitude $\geq \delta \sin(\theta_{jc})$. Converting to cosine distance gives the quadratic form in Equation (3)."
- **Problem**: The jump from $\|\text{residual}\| \geq \delta \sin(\theta_{jc})$ to $\delta^2 \sin^2(\theta_{jc}) / (2 + \delta^2)$ is not explained. The denominator $(2 + \delta^2)$ is the non-trivial part of the bound and arises from the encoder's norm at the partial minimum. A reader cannot reconstruct the bound from the proof sketch alone. For a primary contribution theorem, this gap is significant.
- **Fix**: Add one sentence explaining the denominator: e.g., "At the partial minimum, the perturbed encoder satisfies $\|w_{e,j}\|^2 \leq 1 + \delta^2/2$ (Tang et al., 2025, Lemma X). Dividing the squared residual by $\|w_{e,j}\|^2$ and simplifying yields the quadratic form in Equation (3)." Alternatively, state "The full derivation appears in Appendix B" and provide the complete proof there.

### Issue 5: D-EDA regularizer $\mu$ is unspecified
- **Location**: Section 3.3, Equation (5)
- **Quote**: "minimizing $\|r_j - W_d \beta\|^2 + \mu \|\beta\|_1$"
- **Problem**: The LASSO regularization parameter $\mu$ is introduced but never given a value, a selection procedure, or a sensitivity analysis --- anywhere in the paper (verified against Sections 4--7). The notation.md file does not include $\mu$. A reviewer will flag this as an unspecified hyperparameter that could drive D-EDA's performance or its failure.
- **Fix**: Specify the value and selection method: e.g., "We set $\mu = 0.01$ via cross-validation on the pilot configuration (Appendix B reports sensitivity to $\mu \in \{0.001, 0.01, 0.1\}$)." Add $\mu$ to notation.md.

### Issue 6: "global minimum" in 3.1 vs. "partial minimum" in Theorem 1 --- inconsistent
- **Location**: Section 3.1 paragraph 2 vs. Section 3.2 Theorem 1
- **Quote**: 3.1: "At a global minimum of this biconvex loss, latent $j$'s encoder direction $w_{e,j}$ aligns with its decoder direction $d_j$" vs. Theorem 1: "For a Sparse Autoencoder at a partial minimum of the biconvex SDL loss"
- **Problem**: Section 3.1 states alignment holds at a "global minimum," then Theorem 1 operates at a "partial minimum." The glossary defines these as distinct concepts (partial minimum = encoder-decoder stationary but not jointly optimal). The transition is not explained. A reader will ask: does the alignment in 3.1 also hold at partial minima? If it does, the global-minimum qualification is misleading. If it does not, the derivation's setup does not connect to the theorem's conditions.
- **Fix**: Restate Section 3.1 to use "partial minimum" consistently: "At a partial minimum of this biconvex loss, absent absorption or other confounders, the first-order stationarity condition implies that $w_{e,j}$ aligns with $d_j$. Global minima are a special case." This directly connects the derivation to the theorem.

### Issue 7: The "absorbing source set" $S_j$ is defined but never used downstream
- **Location**: Section 3.3, final sentence of Step 3
- **Quote**: "The absorbing source set is $S_j = \{k : |\beta_k| \text{ significant} \wedge \cos(d_k, d_j) > 0.1\}$."
- **Problem**: $S_j$ does not appear in any subsequent section (4, 5, 6, or 7). The D-EDA evaluation in Section 4.2 uses the AUROC of the "D-EDA absorption indicator" (the variance-ratio quantity), not $S_j$. The variance-ratio indicator is never given a symbol. This creates a dead definition that adds notational overhead without payoff.
- **Fix**: Either (a) remove $S_j$ and give the variance-ratio indicator a proper symbol (e.g., $\text{D-EDA}_{\text{abs}}(j) = \sum_{k \in \text{top-3}} \beta_k^2 / \|r_j\|^2$), or (b) show explicitly how $S_j$ connects to the evaluation metric in Section 4.

### Issue 8: D-EDA limitation paragraph leaks Section 4 results prematurely
- **Location**: Section 3.3, "Limitation" paragraph
- **Quote**: "D-EDA does not outperform scalar EDA on Gemma Scope SAEs (Section 4). One exception: on GPT-2 Small layer 10, D-EDA achieves AUROC = 0.762 (95% CI: [0.686, 0.830]) where scalar EDA achieves only 0.336."
- **Problem**: This paragraph reports specific AUROC values from Section 4 before the reader has seen any D-EDA validation data. It undermines reader confidence in D-EDA at the point where the derivation section should be building momentum toward validation. Section 4.2 reports the same numbers in context with all other configurations.
- **Fix**: Replace the empirical results with a theoretical limitation statement: "The sparse projection in Equation (5) becomes ill-conditioned when $d_{\text{SAE}} \gg d_{\text{model}}$, because the decoder dictionary is highly overcomplete. Section 4 evaluates D-EDA empirically; Appendix B provides a conditioning analysis." This keeps the method section focused on *what* D-EDA is, not how well it works.

---

## Minor Issues

- **Section 3.1, paragraph 2**: "drifting toward directions that no longer serve latent $j$'s original detection role" --- "detection role" is informal. The glossary defines "encoder direction" and "decoder direction" as preferred terms. Rewrite as: "rotating away from $d_j$ toward the subspace spanned by child decoder directions."
- **Section 3.1, line 17**: "Computation takes under one second for a 65k-width SAE." --- On what hardware? CPU? GPU? Without platform specification, the timing claim is not reproducible.
- **Section 3.2, line 27**: "vanishes when $d_j = d_c$ (degenerate case where parent and child represent the same direction)" --- "represent the same direction" is ambiguous between concept and vector. Rewrite: "degenerate case where the parent and child decoder columns are identical ($d_j = d_c$)."
- **Section 3.2, line 29**: "O'Neill et al., 2024" cited without parenthetical format on first use in this section. Should be "(O'Neill et al., 2024)."
- **Section 3.3, Step 3**: The cosine threshold 0.1 in $S_j = \{k : |\beta_k| \text{ significant} \wedge \cos(d_k, d_j) > 0.1\}$ is arbitrary and unjustified. Why 0.1? Add a brief rationale: "chosen to exclude near-orthogonal components unlikely to represent parent-child relationships."
- **Section 3.3, Step 3**: The "top-3 decoder components" threshold for the D-EDA absorption indicator is stated without justification. Why 3? This is a hyperparameter whose sensitivity should be addressed. Add: "We set $k = 3$ based on pilot analysis; Appendix B reports sensitivity over $k \in \{1, 3, 5, 10\}$."
- **Section 3.4**: SynthSAEBench is described as "a synthetic benchmark with known ground-truth absorption labels" and the setup specifies $d_{\text{model}} = 64$, $d_{\text{SAE}} = 500$. The vast scale gap from real SAEs ($d_{\text{model}} = 2304$, $d_{\text{SAE}} = 16384$--$65536$) is not discussed. A reviewer will question whether perfect synthetic discrimination ($\text{AUROC} = 1.000$) at $d_{\text{model}} = 64$ implies anything about real SAE performance. Add a sentence acknowledging the scale gap.
- **Section 3.4**: "absorption injected by rotating encoder directions away from their decoder directions toward a randomly selected child decoder direction" --- the rotation method (Givens? geodesic? additive perturbation?) is unspecified. A reproducibility concern.
- **Section 3.4, random direction baseline**: "On a real Gemma Scope SAE (L12-16k, $d_{\text{SAE}} = 16384$), mean EDA for actual decoder directions is 0.214" --- this shifts from synthetic to real data mid-subsection without a transition. The subsection is titled "Synthetic Validation (SynthSAEBench)." Add a bolded sub-heading: "**Random direction control on a real SAE.**"
- **Section 3.4**: "1.000 $\pm$ 0.002" --- is this standard deviation, 95% CI, or range? Specify.
- **Section 3.4**: "the pre-registered 30% acceptability criterion" --- where was this pre-registered? Add a parenthetical or forward reference.
- **Section 3.5**: "DeLong $p \approx 0$" --- imprecise. Report the actual value or a bound: e.g., "$p < 10^{-10}$."
- **Section 3.5**: The dead feature indicator baseline is described but never quantitatively evaluated in this section. Either report its AUROC or defer its definition to Section 4.
- **Figure 2 caption**: Reports medians as 0.84 and 0.07, while the body text says 0.837 and 0.069. Per the glossary's writing conventions (3 decimal places for AUROC-scale values), use 0.837 and 0.069 in the caption.
- **Equation (4)**: $\|d_j\|^2$ appears in the denominator, but the notation conventions state decoder columns are unit-normalized ($\|d_j\| = 1$), making the denominator trivially 1. Either simplify to $r_j = w_{e,j} - (w_{e,j} \cdot d_j) d_j$ or note the general form is shown for completeness.
- **Figure comment block (lines 77--80)**: Lists "Figure 1: fig1_absorption_mechanism_desc.md" but Figure 1 is in the Introduction. This section only uses Figure 2. Clean up the comment block.

---

## Visual Element Assessment

- [x] Figure 2 (SynthSAEBench) matches the outline plan (two-panel: ROC curve + EDA distribution)
- [x] Figure 2 is referenced before appearance ("As shown in Figure 2")
- [ ] Figure 2 caption rounds median values (0.84 vs. 0.837) --- inconsistent with body text and glossary precision rules
- [ ] No back-reference to Figure 1b for the D-EDA geometry. Section 3.3's residual decomposition would benefit from "Recall Figure 1b" to aid comprehension
- [ ] No algorithm box for EDA or D-EDA computation. For a method section presenting a novel metric, Algorithm 1 (EDA: input $W_e$, $W_d$; output $\text{EDA}(j)$ for all $j$) and Algorithm 2 (D-EDA: input $W_e$, $W_d$, $\mu$; output $\text{D-EDA}_{\text{abs}}(j)$, $S_j$) would improve reproducibility
- [ ] The outline recommends an architecture/pipeline diagram for Method sections. None exists. Figure 1b partially serves this role but appears in the Introduction

---

## What Works Well

1. **Section 3.2's "Interpretation as a necessary condition" paragraph** is exemplary honest science. It enumerates three distinct confounders (polysemanticity, amortization gap, training noise) that produce nonzero EDA in non-absorbed latents, explicitly states EDA is a screening signal rather than a definitive diagnosis, and preempts the most obvious reviewer criticism. This calibrated framing strengthens rather than weakens the contribution.

2. **The D-EDA three-step decomposition** (compute residual, sparse projection, classify) is clearly structured with bolded step headers and a clean input-output chain. The absorption-vs-polysemanticity signature distinction (sparse $\beta$ with geometrically related components vs. dense $\beta$ with unrelated components) is a theoretically motivated and conceptually elegant idea, even if the empirical realization is limited by numerical conditioning.

3. **Theorem 1's monotonicity property and the $\sin^2(\theta_{jc})$ factor** are well motivated. The bound is strongest when parent and child are orthogonal and vanishes in the degenerate case ($d_j = d_c$) --- the geometric interpretation is immediate and correct. The proof sketch, while incomplete (Issue 4), conveys the right intuition.
