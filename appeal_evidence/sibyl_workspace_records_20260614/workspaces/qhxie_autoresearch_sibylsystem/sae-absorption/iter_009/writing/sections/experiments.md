# Section 4: Cross-Domain Absorption Results

Absorption rates at layer 24 with 16k JumpReLU SAEs range from 11.6% (city-language) to 45.1% (city-country), a 4$\times$ variation across hierarchies (Kruskal-Wallis $H$ = 299.95, $p$ = 7.4 $\times 10^{-66}$, $N$ = 3,566). Table 2 presents the complete cross-domain results; Figure 2 shows the layer-dependent absorption profile and cross-domain comparison.

## 4.1 Cross-Domain Absorption Rates

Table 2 reports absorption rates for all four hierarchies at layer 24 across two SAE widths. At 16k width: city-country shows the highest absorption (45.1%, 95% CI [42.2, 47.9]), followed by city-continent (31.4%, CI [28.9, 33.9]), first-letter (27.1%, CI [24.5, 29.5]), and city-language (11.6%, CI [9.7, 13.5]). Wider SAEs (65k) reduce absorption for all hierarchies, with the largest absolute reduction for city-country ($-$12.2 percentage points, from 45.1% to 32.9%) and smallest for city-continent ($-$0.2 pp, from 31.4% to 31.3%).

![Table 2: Cross-domain absorption rates at layer 24 on Gemma 2 2B with JumpReLU SAEs. Asterisk denotes probe F1 below the 0.80 relaxed gate. Absorption rates for city-country should be treated as upper bounds.](figures/table2_crossdomain.pdf)

The Kruskal-Wallis test confirms that absorption rates differ significantly across hierarchies at both SAE widths: $H$ = 299.95 ($p$ = 7.4 $\times 10^{-66}$) for 16k and $H$ = 238.56 ($p$ = 1.6 $\times 10^{-52}$) for 65k. Pairwise permutation tests with Bonferroni correction identify two significant comparisons against first-letter: city-language absorbs significantly less than first-letter at both 16k ($p_{\text{Bonf}}$ = 0.003, Cohen's $h$ = $-$0.73) and 65k ($p_{\text{Bonf}}$ = 0.043, Cohen's $h$ = $-$0.54). City-continent and city-country do not differ significantly from first-letter after correction ($p_{\text{Bonf}}$ = 1.0 for both at 16k).

**H2' (semantic $>$ syntactic ordering) is refuted.** No simple category-level ordering holds. City-country (a semantic hierarchy) exceeds first-letter (a syntactic hierarchy) at 16k, while city-language (also semantic) falls far below it. The pattern is hierarchy-specific: absorption severity depends on the particular hierarchy's class count, balance, and the model's internal representation of that attribute -- not on whether the hierarchy is syntactic or semantic.

**Probe quality caveat.** First-letter probes achieve $F_1$ = 1.0 at all layers (trained at token position $-6$ with the `sae-spelling` protocol), providing a gold-standard anchor. RAVEL probes at layer 24 range from $F_1$ = 0.87 (city-continent, relaxed gate) to $F_1$ = 0.73 (city-country, below gate). Probe errors inflate false negative counts, meaning RAVEL absorption rates are upper bounds on true absorption. The city-country rate of 45.1% is particularly susceptible to this inflation given the 80-class prediction task's inherent difficulty.

## 4.2 Layer Dependence

Figure 2(a) reveals that feature absorption concentrates at the final prediction layer. First-letter absorption at 16k SAEs rises from 1.0% at layer 6 through 4.7% at layer 12 and 2.0% at layer 18, then jumps to 27.1% at layer 24 -- a 27$\times$ increase from layer 6 to layer 24. The 65k pattern is qualitatively similar: 0.7% (L6), 5.0% (L12), 1.0% (L18), 17.7% (L24).

![Figure 2: (a) Layer-dependent absorption profile for first-letter (lines) and cross-domain hierarchies (L24 data points) using 16k and 65k JumpReLU SAEs. Absorption concentrates at layer 24 for first-letter (27$\times$ increase from L6). RAVEL hierarchies are measured only at L24 because RAVEL probes achieve their best F1 at this layer. (b) Cross-domain absorption rates at L24 with 95% CI error bars.](figures/fig2_layer_absorption.pdf)

The non-monotonic pattern -- a dip at layer 18 relative to layer 12 for first-letter -- suggests that absorption is not a simple function of layer depth. Instead, it reflects the model's computational demands at each layer: layer 24 is the final residual stream position before the unembedding matrix, where the model concentrates its predictions. RAVEL hierarchies are measured only at L24 because RAVEL probes achieve their best $F_1$ at this layer ($F_1$ = 0.73--0.87 at L24 vs. 0.37--0.72 at L6); probe quality at earlier layers would render absorption measurements unreliable.

The layer-dependence finding has methodological implications: absorption benchmarks conducted at intermediate layers (e.g., layer 12, where all four SAE architectures are available) may substantially underestimate the absorption that matters most for model output.

## 4.3 Per-Class Variation

Figure 3 shows absorption rates for the six continent classes at L24, revealing extreme within-hierarchy variance. Europe absorbs at 90.2% (16k) and 92.0% (65k) -- nearly all probe-correct instances become false negatives after SAE encoding. In contrast, Africa (3.9%) and South America (3.9%) show minimal absorption. Asia (24.4%), North America (19.1%), and Oceania (52.9%) fall between these extremes.

![Figure 3: Per-class absorption heatmap for city-continent at layer 24. Europe shows 90% absorption while Africa and South America show less than 4%. Cell annotations show absorption rate and entity count $n$. The 23$\times$ within-hierarchy variance (3.9% to 90.2%) exceeds the 4$\times$ between-hierarchy variance.](figures/fig3_perclass_heatmap.pdf)

This 23$\times$ within-hierarchy range (3.9% to 90.2%) exceeds the 4$\times$ between-hierarchy range (11.6% to 45.1%). Wider SAEs barely affect the per-class pattern: Europe remains above 90% at both widths, and Africa and South America remain below 6%. The SAE's failure is concentrated in specific subclasses -- those for which the model apparently relies most heavily on fine-grained entity features that compete with the coarser continent feature.

Similar within-hierarchy patterns appear in city-country: the United States shows 0% absorption (176 entities, zero false negatives) while 15 countries with small entity counts (Albania, Algeria, Argentina, Ecuador, etc.) show 100% absorption. India (0.021, $n$ = 49) and Indonesia (0.044, $n$ = 25) also resist absorption, suggesting that entity count in the training data affects whether the SAE allocates dedicated parent features. For city-language, Turkish (90.0%), Kazakh (88.9%), and Aymara (88.9%) show the highest absorption, while Chinese (2.0%) and English (3.7%) show the lowest.

## 4.4 Width Effect

Wider SAEs ($m$ = 65,536 vs. 16,384) reduce absorption for all hierarchies at L24, but the magnitude of improvement varies. City-country benefits most ($-$12.2 pp, from 45.1% to 32.9%), followed by first-letter ($-$9.4 pp, from 27.1% to 17.7%) and city-language ($-$3.8 pp, from 11.6% to 7.7%). City-continent shows negligible improvement ($-$0.2 pp, from 31.4% to 31.3%).

The asymmetric response to width suggests that absorption in city-continent is driven by a structural property of the hierarchy (perhaps the small number of classes, $K$ = 6, combined with the extreme Europe concentration) that adding more features does not resolve. In contrast, city-country's 80-class hierarchy offers more opportunities for wider SAEs to allocate additional parent features. The width effect is consistent with the intuition that absorption arises from competition for a finite number of active features ($L_0$): more dictionary entries provide more slots for parent features to survive alongside their children.

## 4.5 Statistical Robustness

Three controls bound the measurement:

**Threshold sensitivity.** A separate analysis (Appendix A) varies the cosine similarity threshold and magnitude gap threshold across a 5 $\times$ 4 grid (20 configurations). The false negative count remains constant at 87/576 across all 20 cells (CV = 0.077), confirming that absorption is a structural phenomenon of the SAE encoding, not an artifact of threshold selection.

**Shuffled hierarchy control.** Randomly permuting parent labels across entities and re-measuring absorption produces near-zero rates, confirming that the measured absorption is hierarchy-specific and not attributable to generic SAE reconstruction error.

**First-letter as gold standard.** Because first-letter probes achieve $F_1$ = 1.0, the 27.1% absorption rate at L24 with 16k SAEs is uncontaminated by probe error. This serves as the reference point against which RAVEL absorption rates (inflated by probe imperfection) should be interpreted. The finding that city-language (11.6%) falls significantly below first-letter ($p_{\text{Bonf}}$ = 0.003) holds despite the conservative direction of probe-quality bias: imperfect RAVEL probes would inflate, not deflate, the city-language rate.

<!-- FIGURES
- Figure 2: gen_fig2_layer_absorption.py, fig2_layer_absorption.pdf — Layer-dependent absorption profile (first-letter across 4 layers) and cross-domain comparison at L24 (4 hierarchies x 2 widths)
- Figure 3: gen_fig3_perclass_heatmap.py, fig3_perclass_heatmap.pdf — Per-class absorption heatmap for city-continent at L24 (6 classes x 2 SAE widths)
- Table 2: gen_table2_crossdomain.py, table2_crossdomain.pdf — Cross-domain absorption rates at L24 (4 hierarchies x 2 widths with 95% CI, probe F1, N, N_FN)
-->
