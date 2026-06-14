# 6 Conclusion

Feature absorption — the suppression of child SAE latents by parent features — reduces the
reliability of sparse autoencoders for mechanistic interpretability by creating features that
appear to encode a concept yet fail to fire on 92–97\% of relevant inputs. Three results in
this paper address the detection, mechanism, and phase behavior of this failure mode.

**Probe-free absorption detection.** EDA, $\text{EDA}(j) = 1 - \cos(\hat{e}_j, d_j)$, achieves
AUROC $= 0.650$ against exact FeatureAbsorptionCalculator labels ($n_+ = 18$, $z_\text{null}
= 2.49$) on GPT-2 Small layer 6 — statistically significant and computed from SAE weights
alone, without probes or activation data. The cross-directional metric $\cos(\hat{e}_p, d_c)$
is stronger: AUROC $= 0.730$ (Cohen's $d = 0.552$, $p = 2.8 \times 10^{-9}$), a result not
anticipated by the original EDA theory and identified empirically from pairwise analysis. Both
detectors yield AUPRC above $2 \times$ base rate, providing a tractable candidate set for
downstream IG-based verification. Decoder cosine similarity between feature pairs fails as a
detector (AUROC $= 0.206$), confirming that absorption geometry is carried by encoder directions,
not between decoders.

**Geometric mechanism.** Proposition 1 proves that the rate-distortion training objective prefers
the absorbed solution when $\lambda > \sin^2(\theta_{p,c})$, with the co-occurrence frequency
$p_\text{co}$ canceling from the threshold. This provides the first closed-form, falsifiable
condition for absorption onset, and its key corollary — that rare and common feature pairs with
identical decoder angles are equally at risk — contradicts the intuition that infrequent co-occurrence
protects against absorption. The encoder-decoder decomposition confirms the mechanistic conjecture
(Proposition 2): at layer 6, letter-feature decoder-probe alignment ($0.383$) exceeds
encoder-probe alignment ($0.139$) by a margin of $0.244$ ($p = 3.5 \times 10^{-38}$), consistent
with decoders anchored to the child concept and encoders pulled toward the parent. This asymmetry
is absent in non-letter features (gap $= 0.043$), ruling out a universal SAE geometry effect.
One tension remains unresolved: observed EDA values near $0.67$ imply an encoder-decoder angle
of roughly $48°$, larger than the small angle predicted by Proposition 1 at absorption onset;
we hypothesize post-convergence encoder drift accounts for this gap, but the claim is unverified.

**Phase stability and architecture dependence.** Absorption rates span only $0.919$–$0.968$
across all 11 tested SAE configurations (GPT-2 Small, layers 2–10, widths 12k–98k, $L_0$
range $18$–$81$). A likelihood-ratio test rejects a sparsity-dependent sigmoid (LRT $p = 0.456$;
BIC difference $= -3.22$), and Spearman $\rho = 0.191$ ($p = 0.574$) confirms the absence of
a monotonic sparsity trend. Fine-tuning a high-absorption SAE ($\alpha = 0.959$) for 500 steps
at substantially reduced sparsity does not escape the absorbed state ($\alpha = 0.960$). The
practical implication is direct: absorption cannot be tuned away by adjusting $L_0$ or
choosing a different layer; only architectural interventions that increase $\sin^2(\theta_{p,c})$
— OrtSAE, Matryoshka SAE, ATM SAE — act on the training objective in the right way. AJT-trained
SAEs exhibit reversed EDA polarity (EDA$_\Delta < 0$, AUROC $= 0.154$–$0.354$), demonstrating
that training regime, not hyperparameter tuning, is the critical variable in absorption geometry.

**Scope and open questions.** All positive detection results are scoped to L1-penalized SAEs at
mid-layers of GPT-2 Small on first-letter orthographic hierarchies. Semantic hierarchy absorption
is absent in GPT-2 Small (animate-inanimate, noun-proper: ratio-to-null $= 1.0$), though three
explanations remain viable — insufficient semantic co-occurrence at this scale, parent feature
misidentification, or methodology artifacts — and Gemma Scope experiments are required to resolve
them. Three open questions merit direct follow-up: (1) whether EDA's magnitude tension is resolved
by measuring encoder drift trajectories during training; (2) whether $\cos(\hat{e}_p, d_c)$
remains the strongest cross-directional detector on Gemma-scale SAEs, where EDA's cross-model
generalization is unconfirmed; and (3) whether AJT training prevents encoder drift outright or
produces a qualitatively different geometry that requires an inverted detector.

The weight matrices of a trained SAE record the history of the training objective's geometry.
EDA reads that record.

<!-- FIGURES
- None
-->
