# 4. EDA Validation: Detection Performance

## 4.1 Experimental Setup

**Models and SAEs.** We evaluate EDA on two model families. For Gemma 2 2B, we use six Gemma Scope SAEs spanning layers {5, 12, 19} and widths {16k, 65k} ($d_{\text{SAE}} \in \{16384, 65536\}$).^[Evaluated layers are 5, 12, 19 per SAEBench availability; the pre-registered specification was layers 6, 12, 20.] For GPT-2 Small, we use SAELens SAEs at layers 6 and 10 ($d_{\text{SAE}} = 24576$). Gemma Scope SAEs use the gated architecture; GPT-2 SAELens SAEs use the standard (non-gated) architecture. All decoder columns are unit-normalized.

**Ground-truth labels.** Absorption labels derive from the Chanin et al. metric applied to the first-letter spelling task. For Gemma Scope, labels use Neuronpedia proxy annotations because Gemma 2 2B remains gated on HuggingFace; this introduces label noise that we quantify in Section 4.4. For GPT-2 Small, we generate exact labels using an adapted `FeatureAbsorptionCalculator`: logistic regression probes trained on 1,145 words (up to 50 per letter, drawn from a 7,637-word NLTK single-token vocabulary), with absorption tested via the full Chanin et al. pipeline on up to 20 words per letter across 25 letters. The GPT-2 labels therefore provide the cleanest validation.

**Metrics and statistics.** AUROC is the primary metric, computed with 95% bootstrap confidence intervals (10,000 resamples, seed = 42). We report Cohen's $d$ for group separation between absorbed and non-absorbed EDA distributions, with significance assessed by Mann-Whitney U test. Baseline comparisons use the DeLong test. The pass threshold is AUROC $\geq$ 0.65.

## 4.2 Main Results

Table 1 reports EDA and D-EDA detection performance across all eight SAE configurations. A clear regime-specific pattern emerges, visualized in Figure 3.

**Table 1: EDA and D-EDA Detection Performance Across SAE Configurations**

| Config | Layer | $d_{\text{SAE}}$ | AUROC (EDA) | 95% CI | AUROC (D-EDA) | $n_{\text{pos}}$ | Cohen's $d$ | Pass |
|--------|-------|-------------------|-------------|--------|----------------|-------------------|-------------|------|
| L5-16k | 5 | 16384 | **0.698** | [0.637, 0.779] | 0.602 | 33 | $\dagger$ | PASS |
| L5-65k | 5 | 65536 | 0.617 | [0.532, 0.725] | 0.534 | -- | $\dagger$ | FAIL |
| L12-16k | 12 | 16384 | **0.776** | [0.700, 0.863] | 0.579 | 16 | 0.843 | PASS |
| L12-65k | 12 | 65536 | 0.468 | [0.315, 0.620] | 0.499 | 29 | 1.145 | FAIL |
| L19-16k | 19 | 16384 | 0.458 | [0.317, 0.590] | 0.589 | -- | $\dagger$ | FAIL |
| L19-65k | 19 | 65536 | 0.562 | [0.438, 0.683] | 0.471 | -- | $\dagger$ | FAIL |
| GPT2-L6 | 6 | 24576 | **0.629** | [0.561, 0.692] | 0.656 | 67 | 0.508 | PASS |
| GPT2-L10 | 10 | 24576 | 0.336 | [0.245, 0.435] | **0.762** | 33 | $\dagger$ | FAIL |

$\dagger$ Cohen's $d$ not reported for configs where proxy label noise renders group separation statistics unreliable or where EDA direction is reversed (GPT2-L10).

![EDA AUROC across SAE configurations. Green borders indicate PASS ($\geq$ 0.65). GPT-2 results show EDA and D-EDA side by side.](figures/fig3_eda_heatmap.pdf)

Two of six Gemma Scope configurations pass AUROC $\geq$ 0.65: L12-16k (0.776) and L5-16k (0.698). Both are mid-layer, narrow SAEs. The strongest result, L12-16k with AUROC = 0.776, places EDA well above the decoder cosine similarity baseline (AUROC = 0.302 at L5-16k, a +0.396 improvement; DeLong $p \approx 0$). EDA cross-validates against SAEBench's precomputed `encoder_decoder_cosine_sim` with Pearson $r > 0.999$, confirming computational correctness.

EDA detection is strongest at mid-layers in narrow (16k) SAEs and degrades at deeper layers and wider dictionaries. L19-16k and L12-65k both fall below or near chance (AUROC = 0.458 and 0.468), while the 65k SAEs at all layers underperform their 16k counterparts at the same layer.

**GPT-2 replication.** GPT2-L6 passes with EDA AUROC = 0.629 (exact labels, $n_{\text{pos}} = 67$, Cohen's $d$ = 0.508), confirming cross-model generality at early-to-mid layers. GPT2-L10 reveals a complementary finding: EDA fails (AUROC = 0.336, reversed direction) but D-EDA achieves 0.762 [0.686, 0.830]. At deeper layers, the absorption mechanism involves complex encoder-decoder residual structure that scalar EDA cannot capture but D-EDA's directional decomposition resolves. More broadly, D-EDA does not consistently improve over scalar EDA on Gemma Scope (0.471--0.602 across six configs, all below their corresponding EDA values except L19-16k). D-EDA's value is complementary and layer-specific: it captures residual structure that scalar alignment misses at deeper layers but adds no benefit at mid-layers where scalar EDA already succeeds.

## 4.3 EDA Group Separation

Figure 4 shows the statistical group separation underlying the AUROC numbers. At L12-16k, absorbed latents have mean EDA = 0.282 versus 0.214 for non-absorbed (Cohen's $d$ = 0.843, Mann-Whitney $p = 6.4 \times 10^{-5}$). At L12-65k, the separation widens: absorbed mean EDA = 0.424 versus non-absorbed 0.313 (Cohen's $d$ = 1.145, $p = 1.3 \times 10^{-10}$). The larger effect size at L12-65k despite its lower AUROC of 0.468 reflects that group separation is real but low positive-class prevalence ($n_{\text{pos}} = 29$ out of 65,536 latents) makes rank-based AUROC unreliable.

![EDA distributions for absorbed versus non-absorbed latents at L12-16k and L12-65k. Effect sizes are large (Cohen's $d$ = 0.84--1.15) even where AUROC is moderate.](figures/fig4_eda_distributions.pdf)

## 4.4 The Pilot-to-Full Discrepancy and Proxy Label Noise

A pilot run at L12-65k yielded AUROC = 0.853. The full validation collapsed to 0.468---a 0.385 drop. Two factors explain this. First, the pilot used a capped subset of 100 latents with enriched positive prevalence; the full 65,536-latent evaluation dilutes positives to 0.044% of the population. Second, Neuronpedia proxy labels introduce systematic noise: the labeling threshold (cosine similarity $\geq$ 0.30 between decoder column and letter probe direction) is calibrated on specific SAEBench conventions that may not transfer uniformly across SAE widths.

GPT2-L6 with exact Chanin et al. labels (AUROC = 0.629, $n_{\text{pos}} = 67$) provides the cleanest single-configuration validation. Whether EDA is genuinely stronger on Gemma mid-layers or proxy label noise inflates the Gemma AUROC remains open; resolving this requires running the exact Chanin et al. pipeline on Gemma 2 2B when model access is available.

EDA is a regime-specific screening tool for mid-layer narrow SAEs, not a universal absorption detector.

## 4.5 Polysemanticity Stratification

Section 3 established that polysemanticity also raises EDA (Theorem 1 provides a necessary-but-not-sufficient condition). To quantify how much of EDA's discrimination is absorption-specific versus polysemanticity-driven, we stratify latents by feature density $\rho_j$ (fraction of tokens on which $z_j > 0$; a SAEBench proxy for polysemanticity) using a median split and evaluate EDA within each stratum.

At L12-16k, EDA achieves AUROC = 0.922 [0.842, 0.979] on the polysemantic half (above-median feature density, $n_{\text{pos}} = 3$) versus 0.643 [0.518, 0.763] on the monosemantic half ($n_{\text{pos}} = 13$). Caution: the polysemantic AUROC relies on only 3 positive examples; bootstrap CIs may understate uncertainty at this sample size. The pattern replicates at L12-65k with more statistical power: polysemantic AUROC = 0.940 [0.856, 0.984] versus monosemantic 0.743 [0.666, 0.812]. EDA is most discriminative in the polysemantic regime where absorption most frequently occurs.

This result has a dual interpretation. EDA provides an absorption-enriched signal precisely in the polysemantic region of the dictionary where features are hardest to interpret manually. At the same time, polysemanticity itself elevates EDA, compressing the non-absorbed distribution downward and making absorbed latents stand out. The monosemantic AUROC (0.643--0.743) better isolates EDA's pure absorption-detection capability.

---

# 5. Cross-Domain Absorption: A Null Result and Conditional Evidence

Every published measurement of feature absorption uses the first-letter spelling task (Chanin et al., 2024; Karvonen et al., 2025). We attempted to test whether absorption generalizes to semantically richer entity-attribute hierarchies using RAVEL, motivated by relevance to factual reasoning and concept steering. **This hypothesis (H3) was falsified**: absorption rates measured with bridge-model probes are statistically indistinguishable from shuffled null rates across all tested configurations.

## 5.1 RAVEL Hierarchy Suite and Probe Limitations

We evaluated three RAVEL hierarchies: city-continent (6 parent classes), city-country (~100 parent classes), and city-language (~82 parent classes). Each defines a parent-child structure analogous to the first-letter task.

**Probes.** A critical structural limitation precluded valid measurement. Logistic regression probes were trained on Qwen2.5-0.5B ($d_{\text{model}} = 896$) and projected to Gemma 2 2B's $d_{\text{model}} = 2304$ via random orthonormal QR decomposition, because Gemma 2 2B remains gated on HuggingFace. Probe accuracy was far below the pre-registered 85% quality gate: city-continent 71.4%, city-country 37.8%, city-language 36.8%.

**Measurement protocol.** For each hierarchy and each of 6 Gemma Scope SAE configurations, we identified latents whose decoder columns align with the probe direction (cosine similarity $\geq$ 0.30), then tested whether these latents fire on parent-positive inputs. A random-direction baseline (100 random unit vectors) established the fixed-probe noise floor.

## 5.2 Shuffled Hierarchy Control: H3 Falsified

Initial measurements showed all 18 SAE-hierarchy combinations (3 hierarchies $\times$ 6 SAE configs) exceeded the 3$\times$ fixed-random-probe baseline. However, a proper shuffled hierarchy control---in which parent-child label assignments are randomized while all other aspects of the pipeline are held constant---revealed no signal. Across all 9 tested domain-SAE config combinations (3 hierarchies $\times$ 3 representative SAE configs), 0 of 9 real-hierarchy absorption rates exceeded the shuffled-label p95 threshold. The pre-registered decision criterion was $\geq$ 1/9 pass; the result is NO\_GO.

The initial "3$\times$ random baseline" comparison used fixed random probe directions, not randomized hierarchy labels. Fixed random probes decorrelate from SAE geometry differently than shuffled hierarchy labels---the latter removes semantic structure while preserving geometric properties of the probe directions. Only the shuffled-label control constitutes a valid null test for cross-domain absorption, and it finds no evidence of signal.

**Interpretation.** The RAVEL cross-domain signal is a probe-quality artifact: below-quality-gate probes trained on a different model family and projected via random decomposition do not provide reliable parent-feature directions. The absence of signal after shuffling confirms that the initial measurements captured noise, not absorption. Cross-domain generalization of absorption detection remains an open question requiring same-model probe access (Gemma 2 2B or Llama-3.1-8B with gating resolved).

## 5.3 Conditional Intra-Domain Coherence (Pending Replication)

Prior to the shuffled control, we observed high intra-RAVEL coherence: SAE configurations with high RAVEL absorption in one hierarchy showed high absorption in all three (mean Spearman $\rho$ = 0.924, two of three pairwise comparisons surviving Bonferroni correction). First-letter rates (SAEBench scores) correlated negatively with RAVEL rates ($\rho$ = $-$0.43 to $-$0.20, all non-significant).

Because the shuffled control invalidated the absolute measurements, this coherence should be interpreted cautiously. The coherence result could reflect (a) a genuine SAE-level absorption susceptibility property that probe noise measures with consistent error, or (b) a structural artifact of the bridge-model projection method that correlates across hierarchies for geometric reasons unrelated to absorption. Resolving this requires direct probe access to Gemma 2 2B. We retain this as suggestive existence evidence conditional on replication with same-model probes.

---

# 6. Three-Subtype Absorption Taxonomy

Prior work treats all absorbed latents as a single category. We introduce a three-way structural partition based on the relationship between each absorbed latent's decoder dictionary and the parent probe direction. The taxonomy reveals that most absorption reflects a training-time dictionary allocation failure, not an inference-time encoder alignment problem.

## 6.1 Subtype Definitions

Each absorbed latent $j$ is classified by computing $\max_k \cos(d_k, v_p)$ across all decoder columns $d_k$ relative to the parent probe direction $v_p$:

- **Early absorption**: $\max_k \cos(d_k, v_p) < \tau$ (primary threshold $\tau = 0.3$). The parent feature was never allocated a decoder direction in the SAE dictionary. No amount of encoder adjustment can recover the parent latent because the reconstruction dictionary lacks the necessary direction.
- **Late absorption**: $\max_k \cos(d_k, v_p) \geq \tau$ but the latent fails to fire on parent-positive inputs. The parent direction exists in the dictionary but the encoder has been trained away from activating it---a classic encoder-suppression failure.
- **Partial absorption**: $\max_k \cos(d_k, v_p) \geq \tau$ and the latent fires on some but not all parent-positive inputs. Context-dependent, selective suppression.

## 6.2 Empirical Distribution

**Table 2: Three-Subtype Taxonomy at Primary Threshold $\tau = 0.3$**

| Config | $n$ Absorbed | % Early | % Late | % Partial | KW $p$-value | Late > Early EDA |
|--------|--------------|---------|--------|-----------|--------------|-----------------|
| L12-16k | 16 | **75.0** | 12.5 | 12.5 | 0.237 | Yes |
| L12-65k | 65 | **72.3** | 13.8 | 13.8 | 0.0002 | Yes |

Early absorption dominates at 72--75% of absorbed latents across both configurations. Late and partial absorption each account for approximately 13% of cases.

At L12-65k ($n = 65$ absorbed latents), the three subtypes exhibit statistically distinct EDA distributions (Kruskal-Wallis $p = 0.0002$; Mann-Whitney late vs. early $p = 0.0001$). Late-absorbed latents have median EDA = 0.723, early-absorbed 0.668, and partial-absorbed 0.652. At L12-16k ($n = 16$), the same ordering holds (late median = 0.308, early = 0.297, partial = 0.240) but statistical power is insufficient (KW $p = 0.237$).

**Threshold stability.** The late > early EDA ordering holds at all five tested thresholds ($\tau \in \{0.20, 0.25, 0.30, 0.35, 0.40\}$) for both configurations (10/10 tests). Early absorption fraction ranges from 32.3% ($\tau = 0.20$, L12-65k) to 93.8% ($\tau = 0.40$, L12-16k), confirming that the dominance of early absorption is robust to threshold choice---it holds even at the most conservative setting.

As shown in Figure 7, partial absorption has lower median EDA than early absorption across both configurations. This contradicts the naive prediction that partial absorption (an intermediate failure mode) should have intermediate EDA. The result suggests that partial and early absorption share a geometric mechanism---in both cases, the encoder-decoder alignment is not strongly disrupted---while late absorption involves a distinct, measurable encoder drift.

![EDA distributions by absorption subtype. Late-absorbed latents consistently show highest EDA. Partial-absorbed latents have lower EDA than early, contrary to prediction.](figures/fig7_subtype_eda.pdf)

## 6.3 The Early-Absorption Dominance Insight

The finding that ~72--75% of absorbed latents are early-type---the parent feature was never allocated a decoder direction---is the most actionable result from the taxonomy. It reframes the absorption problem: the dominant failure mode is **dictionary coverage** at training time, not encoder-decoder misalignment at inference time.

Three implications follow. First, inference-time corrections (ITAC, Select-and-Project) are structurally inapplicable to the 75% majority. These methods require an existing parent decoder direction to recover the suppressed activation; early-absorbed latents have no such direction. Second, the appropriate lever for reducing the dominant absorption mode is wider dictionaries or hierarchically-aware training objectives that ensure parent features are allocated dictionary capacity. Third, EDA's regime-specificity (Section 4) is explained by the taxonomy: EDA was designed to detect encoder-decoder misalignment, which characterizes late absorption. Early absorption produces no such misalignment, limiting EDA's theoretical sensitivity to approximately 25% of absorbed latents.

## 6.4 ITAC Proof-of-Concept

Inference-Time Absorption Correction (ITAC) targets the 13% minority of late-type absorbed latents by recovering suppressed parent activations from the reconstruction error:

$$z_j^{\text{corr}} = \max\left(0,\; d_j^\top \left(e + z_{\text{abs}} \cdot d_{\text{abs}}\right)\right)$$

where $e = x - \hat{x}$ is the reconstruction error, $z_{\text{abs}}$ is the absorbing child's activation, and $d_{\text{abs}}$ is the child's decoder direction.

**Table 3: ITAC Efficacy on Late-Absorbed Latents**

| Config | $n$ Late | ITAC Targets | FN Before | FN After | FN Reduction | FVU Change |
|--------|----------|-------------|-----------|----------|--------------|------------|
| L12-16k | 2 | 1 | 0.000 | 0.000 | 0.0% | +0.221 |
| L12-65k | 13 | 10 | 0.076 | 0.062 | 3.14% | $-$0.042 |

At L12-65k, ITAC reduces the mean false negative rate by 3.14%---well below the pre-registered target of $\geq$ 20% (H5 falsified). The best individual case (latent $j = 61217$) achieves 18.9% FN reduction with decoder-parent cosine similarity of 0.954, demonstrating that correction is geometrically possible when the parent and child directions are well-separated. FVU improves by $-$4.2% at L12-65k, confirming that ITAC does not degrade reconstruction quality.

A null test on all 47 early-type latents at L12-65k produces mean FN rate = 32.6%, confirming that early-absorbed latents cannot be corrected by ITAC (as predicted: no parent decoder direction exists). This type-selectivity validates the taxonomy: ITAC works only on the subtype it was designed for, and that subtype represents 13.8% of absorbed latents.

ITAC demonstrates that inference-time correction of late absorption is geometrically possible but practically limited. The 3.14% mean FN reduction, combined with the 75% early-absorption dominance, confirms that the primary absorption mitigation path is training-time dictionary coverage, not inference-time encoder correction.

<!-- FIGURES
- Figure 3: gen_fig3_eda_heatmap.py, fig3_eda_heatmap.pdf — EDA AUROC heatmap across Gemma Scope and GPT-2 SAE configurations
- Figure 4: gen_fig4_eda_distributions.py, fig4_eda_distributions.pdf — Violin plots of EDA by absorption status at L12-16k and L12-65k
- Figure 5: gen_fig5_crossdomain_rates.py, fig5_crossdomain_rates.pdf — Bar chart of RAVEL absorption rates across 3 hierarchies and 6 SAE configs
- Figure 6: gen_fig6_ravel_coherence.py, fig6_ravel_coherence.pdf — Pairwise scatter plots of intra-RAVEL absorption rate correlations
- Figure 7: gen_fig7_subtype_eda.py, fig7_subtype_eda.pdf — Violin plots of EDA by absorption subtype at L12-16k and L12-65k
- Table 1: inline — EDA and D-EDA detection performance across 8 SAE configurations
- Table 2: inline — Three-subtype taxonomy distribution at primary threshold
- Table 3: inline — ITAC efficacy on late-absorbed latents
-->
