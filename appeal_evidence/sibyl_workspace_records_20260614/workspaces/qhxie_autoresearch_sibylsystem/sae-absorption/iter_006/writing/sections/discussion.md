# 7 Discussion

## 7.1 Implications for the Absorption Mitigation Literature

The architectural mitigation wave---Matryoshka SAE (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), ATM-SAE (Li et al., 2025), masked regularization (Narayanaswamy et al., 2026)---targets competitive exclusion, the mechanism where a child latent actively suppresses a parent latent under sparsity pressure. Our confound decomposition classifies 648 of 657 false negatives at $L_0$=22 as hedging (98.6%) and only 9 as hierarchy-driven (1.4%), with 0 attributable to reconstruction error (Section 4.2). If mitigations are evaluated by the Chanin metric on JumpReLU SAEs, they may reduce the metric's output without addressing the dominant failure mode. The $L_0$ phase transition (42.85% at $L_0$=22 to 0.84% at $L_0$=176; Spearman $\rho_s = -1.0$) suggests that increasing the $L_0$ operating point---a training-time hyperparameter requiring no architectural change---may be more effective than encoder modifications for reducing measured absorption on JumpReLU SAEs.

This does not invalidate the mitigations themselves. Matryoshka SAE and OrtSAE may improve feature quality through mechanisms (hierarchical organization, orthogonality) that benefit features independently of competitive exclusion. The finding is narrower: the Chanin metric, as currently calibrated, does not provide the right evaluation signal for these methods on JumpReLU architectures. On L1-ReLU SAEs, where the metric was developed and validated, competitive exclusion may well dominate---our GPT-2 Small results show uniformly high absorption (61--67% across layers 8, 10, 11) with no $L_0$ phase transition, consistent with a different mechanistic regime. The appropriate recommendation is not to abandon mitigations but to validate evaluation metrics architecture-by-architecture.

## 7.2 The Metric Validity Question

The Chanin metric was developed on GPT-2 Small with L1-ReLU SAEs, where the soft L1 penalty produces gradual competition between features. JumpReLU SAEs have fundamentally different activation dynamics: the hard threshold $\theta_j$ creates a discrete boundary between active ($z_j > \theta_j$) and inactive ($z_j = 0$) states. A latent can have pre-activation just below $\theta_j$ without any competitive suppression occurring, yet the metric counts it as a false negative because $z_j = 0$.

The universal control failure quantifies this miscalibration. Shuffled controls exceed measured rates by 4.7$\times$ (first-letter), 6.9$\times$ (city-continent), 2.7$\times$ (city-language), 27.5$\times$ (animal-class), and $\infty$ (city-country: 10.3% shuffled vs. 0.0% measured). The net signal (measured minus shuffled) is negative in every domain (Table 2). The untrained SAE control produces 0.0% absorption with probe F1 = 0.943, confirming the metric's thresholds---not the probes or SAE representations---are miscalibrated for JumpReLU feature geometry.

Two explanations are compatible with the data. First, the cosine similarity threshold ($\tau_{\cos} = 0.025$) and magnitude gap threshold ($\tau_{\text{mag}} = 1.0$) may be too permissive for JumpReLU SAEs, where the hard threshold concentrates activations differently. Second, the metric's false-negative attribution step---counting all tokens where probe-associated latents do not fire as potential absorption---may systematically overcount on JumpReLU SAEs, where the discrete activation boundary produces more near-miss non-activations than the continuous L1-ReLU gradient.

We recommend a concrete validation protocol: before measuring absorption on any new SAE architecture, researchers should verify that shuffled-label absorption is lower than measured absorption in at least three hierarchy domains.

## 7.3 Rate-Distortion Theory as a Principled Diagnostic

The CMI-absorption correlation ($\rho_s = -0.383$, $p = 0.059$; Cohen's $d = -0.924$) at $d' = 10$ is directionally consistent with rate-distortion theory. Letters with low CMI---those whose parent features carry little unique information beyond the child ($I(X; f_{\text{parent}} \mid f_{\text{child}}) \approx 0.4$--$0.6$)---are preferentially absorbed. Letters J (CMI = 1.21), Z (CMI = 1.18), and Y (CMI = 1.11) have both high CMI and zero absorption at $L_0$=82. Letters T (CMI = 0.52), L (CMI = 0.59), and P (CMI = 0.71) have both low CMI and high absorption (31--37%). The one-sided Mann-Whitney test is significant ($p = 0.023$), and the large effect size ($d = -0.924$) indicates meaningful group-level separation.

The geometric constant degeneration ($c \approx 0.960$, CV = 2.16%) for unit-normalized Gemma Scope decoders simplifies the theoretical threshold to $L_{0,\text{crit}} \approx \lambda / \text{CMI}$, eliminating decoder geometry as a modulating factor. CMI/$c$ performs no better than raw CMI ($\rho_s = -0.374$ vs. $-0.383$; bootstrap 95% CI of the difference: $[-0.113, +0.085]$). This simplification is specific to normalized SAEs and may not hold for unnormalized architectures where $\|w_P\|$ varies across features.

As shown in Figure 5, the CMI-absorption correlation is restricted to $d' = 10$ and reverses sign at $d' \geq 20$ ($\rho_s = +0.048$ at $d' = 20$, $+0.299$ at $d' = 30$, $+0.197$ at $d' = 50$). The Bonferroni-corrected $p$-value for the $d' = 10$ result is 0.236 (4 tests), not significant at the 0.05 level. This dimension instability is the most significant limitation of the rate-distortion analysis. The top-10 decoder directions may capture the most absorption-relevant subspace---the directions along which parent and child features compete---while additional dimensions introduce noise from orthogonal feature interactions. An alternative, less favorable explanation is that $d' = 10$ is a statistical coincidence in a 4-test sweep. Dimension-agnostic MI estimators (MINE, KNIFE) or a theoretical derivation of the optimal subspace dimension from the SAE architecture are needed to resolve this question.

![Spearman rho between CMI and absorption rate as a function of subspace dimension d'. The negative correlation (theory-consistent) holds only at d'=10; at higher dimensions the sign reverses.](figures/cmi_dimension_sensitivity.pdf)

## 7.4 Negative Results

Four pre-registered hypotheses are falsified or unsupported:

- **H2 falsified.** Hierarchy-driven absorption at $L_0$=22 is 1.4% (9 of 657 false negatives), not the predicted $> 80\%$. The pilot's 96.9% hierarchy-driven figure was a methodological artifact of single-$L_0$ classification that did not use cross-$L_0$ persistence as the classification criterion. The corrected decomposition inverts the original finding: hedging dominates at low $L_0$, not competitive exclusion.
- **H4 falsified.** The unsupervised detection pipeline (conditional cosine + firing rate + ITAC) achieves $\rho_s = -0.125$ and AUROC = 0.47 against the probe-based gold standard on first-letter. No combination of geometric SAE weight signals discriminates absorbed from non-absorbed features. ITAC candidate pair median (1.35) does not separate from the random pair median (1.14; Mann-Whitney not significant).
- **H6 underpowered.** With $n = 5$ domains, no hierarchy predictor achieves corrected significance: co-occurrence ratio ($\rho_s = 0.40$, Bonferroni $p = 1.0$), fan-out ($\rho_s = 0.20$), depth ($\rho_s = -0.58$), parent frequency ($\rho_s = 0.40$). Bootstrap 95% CIs span $[-1.0, +1.0]$ for all predictors. Resolving H6 requires at minimum 15--20 independently measured domains.
- **H7 partially falsified.** Both JumpReLU and L1-ReLU SAEs show bimodal per-letter absorption distributions (Hartigan's dip test: JumpReLU at $L_0$=82, dip = 0.188, $p = 0.001$; all L1-ReLU layers BC $> 0.555$). The predicted JumpReLU-specific binary pattern was not observed. The actual architecture-conditional finding is distinct: JumpReLU shows a dramatic $L_0$-dependent phase transition (42.9% to 0.8%) while L1-ReLU shows uniformly high absorption (61--67%) across layers.

## 7.5 Limitations

**Metric dependence.** All absorption measurements depend on the Chanin metric, whose control failure on JumpReLU SAEs is a primary finding of this paper. The confound decomposition provides relative information (hedging vs. hierarchy-driven fractions) but absolute rates remain uncertain until the metric's thresholds are recalibrated for JumpReLU architectures. Activation patching---zeroing child features and measuring parent recovery---would provide metric-independent validation but has not been performed.

**CMI dimension instability.** The negative CMI-absorption correlation holds only at $d' = 10$ and reverses at $d' \geq 20$ (Figure 5). Bonferroni-corrected $p = 0.236$. Whether $d' = 10$ captures the absorption-relevant signal or is a statistical artifact in a 4-test sweep remains open. Future work should apply dimension-agnostic MI estimators and derive $d'$ from the SAE architecture rather than selecting it post hoc.

**Cross-architecture confound.** The JumpReLU vs. L1-ReLU comparison is cross-model (Gemma 2 2B, $d_{\text{model}} = 2304$, $d_{\text{SAE}} = 16384$ vs. GPT-2 Small, $d_{\text{model}} = 768$, $d_{\text{SAE}} = 24576$). Model capacity, tokenizer, and training data differences confound the architecture comparison. A controlled comparison requires training L1-ReLU SAEs on Gemma 2 2B activations.

**Persistent core validation.** The 9 persistent core words identified by cross-$L_0$ analysis are the strongest candidates for genuine competitive exclusion, but causal validation via activation patching (zeroing child features and measuring parent recovery) has not been executed. The "hierarchy-driven" classification rests on the cross-$L_0$ persistence criterion, not on direct mechanistic evidence.

**Cross-domain probe quality.** Mean probe F1 ranges from 0.602 (city-country) to 0.817 (first-letter improved). Domains with lower probe quality may underestimate true absorption. The city-continent absorption signal (6.49%) is driven by a single continent (Asia, 21.62%), raising the possibility of overfitting to a small number of per-parent examples.

**Single task family.** All five domains test hierarchical parent-child classification. Absorption in non-hierarchical feature relationships, safety-relevant features (where DeepMind's 2025 safety team reported SAE probe failures), or multi-dimensional features (Engels et al., 2025) remains unstudied. The generalizability of hedging dominance to non-hierarchical absorption is unknown.

<!-- FIGURES
- Figure 5: gen_cmi_dimension_sensitivity.py, cmi_dimension_sensitivity.pdf --- Spearman rho between CMI and absorption rate across subspace dimensions d'
-->
