# Testable Hypotheses

## Primary Hypotheses

### H1: Lotka-Volterra Competition Coefficient as Unsupervised Absorption Detector

**Claim:** The competition coefficient α_ij = σ_ij · (f_j / f_i), where σ_ij is the co-activation rate normalized by the rarer feature's frequency, predicts SAE feature absorption without requiring pre-specified probe directions.

**Operationalization:**
- σ_ij = P(a_i > 0, a_j > 0) / min(f_i, f_j)
- α_ij = σ_ij · (f_j / f_i)
- Absorption predicted when α_ij > τ (threshold fit empirically, expected ≈ 1.0 based on LV theory)

**Expected outcomes:**
- Precision ≥ 0.70, Recall ≥ 0.65, F1 ≥ 0.65 against sae-spelling ground-truth labels on first-letter task (13-letter test set)
- Sharp (step-function-like) transition in absorption rate near α_ij ≈ 1 (sigmoid AIC lower than linear fit AIC)
- F1 degrades < 10% when transferring to JumpReLU and Matryoshka SAE architectures without re-fitting threshold

**Falsification criterion:** F1 < 0.50 on the 13-letter test set, OR the transition is smooth (linear fit has lower AIC than sigmoid fit) suggesting the LV mechanism is not load-bearing. In this case, α_ij may still have correlational value but the LV theory is not the correct mechanistic account.

---

### H2: Corpus PMI as Primary Driver of Absorption Patterns

**Claim:** Token-pair PMI from the SAE training corpus (OpenWebText) predicts which specific letter-token feature pairs are absorbed, independently of and incrementally beyond feature hierarchy depth, SAE width, and L0.

**Operationalization:**
- PMI(letter_category, specific_child_token) computed on 1M-token OpenWebText sample
- Regression: absorption_rate = β₀ + β₁ log(L0) + β₂ log(width) + β₃ layer + β₄ log(PMI) + ε
- Key test: significance of β₄ and partial R² of PMI term after controlling for SAE config variables

**Expected outcomes:**
- β₄ is positive (higher PMI → higher absorption rate) with p < 0.05 after Bonferroni correction
- Partial R² of PMI term ≥ 0.10 (explains at least 10% of residual variance beyond SAE config)
- Cross-model validation: PMI prediction direction replicates on GPT-2 SAEs (different training data, so PMI values will differ but should still be predictive)

**Falsification criterion:** β₄ is non-significant (p > 0.05) after Bonferroni correction, or partial R² of PMI < 0.05 — consistent with conventional view that absorption is primarily objective-driven.

---

### H3: Absorption Metric Decoupled from Downstream Task Performance

**Claim:** SAE absorption rate (Chanin metric) is weakly or not correlated with downstream interpretability and safety-relevant task performance across the 200+ SAEs in SAEBench.

**Operationalization:**
- Pearson and Spearman correlation: absorption_score vs. RAVEL, SCR, sparse probing F1, unlearning performance
- Partial correlation controlling for model family, layer, and SAE architecture class
- 1-sparse SAE probe vs. dense linear probe AUC on harmful intent (AdvBench-style) for 3 matched SAEs

**Expected outcomes:**
- |r| < 0.20 for all downstream SAEBench tasks after controlling for model/layer/architecture (H3 supported)
- 1-sparse SAE probe performance gap (vs. dense probe) does not decrease as absorption rate decreases across the 3 matched SAEs (H3 supported on safety task)

**Falsification criterion (and alternative hypothesis):** |r| < -0.30 for at least one downstream task — absorption rate is meaningfully correlated with downstream performance. This falsification is scientifically valuable: it would confirm the absorption research motivation and provide the first empirical proof of the causal chain.

---

### H4: Distributed Absorption Increases Monotonically with SAE Width (Width Paradox Resolution)

**Claim:** The distributed absorption score DAS(P, k=3) — measuring information loss when conditioning on the 3 highest-α children — increases monotonically with SAE width, even as single-child absorption rate DAS(k=1) shows a non-monotone relationship with width.

**Operationalization:**
- DAS(P, k=3) = 1 - I(X; f_P | f_{C1}, f_{C2}, f_{C3}) / H(f_P), estimated via multi-label logistic regression on activation statistics
- Computed for first-letter features at Gemma Scope widths {16k, 65k, 131k}, layer 12
- Width-DAS slope estimated via linear regression across the 3 width points (test: positive slope for k=3)

**Expected outcomes:**
- DAS(k=3) slope > 0 (increases with log-width) for ≥ 80% of tested letter features
- DAS(k=1) slope ≤ 0 or near-zero for ≥ 60% of letter features (non-monotone or decreasing)
- The increase in DAS(k=3) aligns with the increase in the fraction of features with max_j α_ij > 1 (more latents → more competitive exclusion pairs)

**Falsification criterion:** DAS(k=3) slope ≤ 0 (distributed absorption does not increase with width). This would mean the width paradox has a different explanation, most likely the L0/D confound — which would itself be an important finding.

---

## Secondary / Exploratory Hypotheses

### H5: Absorption Taxonomy — True Absorption Rate Significantly Exceeds Reported 15–35%

**Claim:** Comprehensive measurement using the three-type taxonomy (Type I: full single-latent, Type II: partial single-latent, Type III: distributed multi-latent) reveals that ≥ 40% of hierarchically related feature pairs exhibit some form of absorption — substantially more than the 15–35% Type I rate reported by Chanin et al.

**Operationalization:**
- Type I: Parent latent suppressed, Chanin metric positive
- Type II: Parent fires at < 50% of expected magnitude on expected-parent-token set (measured via normalized activation ratio compared to similar non-hierarchy tokens)
- Type III: DAS(k=3) > 0.60 and Type I not triggered
- Comprehensive rate = fraction of letter features with Type I OR II OR III absorption

**Expected outcomes:**
- Comprehensive absorption rate ≥ 40% (vs. Chanin et al.'s 15–35%)
- Type III (distributed) is the dominant type in wide SAEs (65k, 131k)
- Type I is the dominant type in narrow SAEs (16k) — consistent with concentrated competitive exclusion

**Falsification criterion:** Comprehensive absorption rate ≤ 20% — the 15–35% reported rate is not a substantial underestimate.

---

### H6: LV Competition Coefficient Explains OrtSAE's Success

**Claim:** OrtSAE's orthogonality penalty reduces absorption primarily by reducing σ_ij (niche overlap) rather than by changing α_ij through frequency ratios — i.e., it is mechanistically a niche-partitioning intervention in LV terms.

**Operationalization:**
- If OrtSAE checkpoints are available: compare σ_ij and f_j/f_i distributions between OrtSAE and standard SAE at matched L0
- Predict: σ_ij is substantially lower for OrtSAE (decoder orthogonalization reduces co-activation), but f_j/f_i ratio is similar

**Status:** Exploratory — depends on OrtSAE checkpoint availability. If unavailable, this becomes a theoretical prediction to motivate future work.

---

## Falsification Summary Table

| Hypothesis | Falsification Condition | Interpretation if Falsified |
|---|---|---|
| H1 (LV detector) | F1 < 0.50 on test set | LV theory not mechanistically correct; α_ij still descriptively useful |
| H2 (corpus PMI) | β₄ non-significant, partial R² < 0.05 | Absorption is objective-driven, not data-driven; conventional view upheld |
| H3 (downstream disconnection) | |r| > 0.30 for some downstream task | Absorption does matter for downstream performance; research motivation validated |
| H4 (distributed absorption) | DAS(k=3) slope ≤ 0 with width | Width paradox explained by L0/D confound, not distributed competitive exclusion |
| H5 (true absorption rate > 40%) | Comprehensive rate ≤ 20% | Chanin metric is not substantially underestimating; taxonomy adds little |

---

## Ground Truth and Measurement Notes

**Primary ground truth source:** sae-spelling first-letter task (Chanin et al., 2024), GitHub: https://github.com/lasr-spelling/sae-spelling.
- Ground truth labels: 26 × (absorbed latent, absorbing latent) pairs per SAE, validated via k-sparse probing + integrated gradients ablation
- Coverage: letters A–Z on Gemma 2 2B Gemma Scope SAEs (multiple widths and layers)

**Primary SAE source:** Gemma Scope (https://huggingface.co/google/gemma-scope), non-commercial research license.
- Widths tested: 16k, 65k, 131k (Gemma 2 2B)
- Layers tested: 6, 12, 20, 25
- Multiple L0 settings available at each configuration

**Corpus:** OpenWebText subset (~1M tokens for PMI estimation; ~10k tokens for activation statistics).

**Downstream task benchmarks:** SAEBench (arXiv:2503.09532) data available at neuronpedia.org/sae-bench. RAVEL implementation in the SAEBench GitHub repository.
