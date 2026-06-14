# Hypotheses: Absorption Quantification and Mitigation Benchmarking

## Hypothesis H1: Absorption Peaks in Middle Layers

**Research Question**: RQ1 (layer-wise absorption variation)
**Status**: UNRESOLVED (two pilot runs produced contradictory results)

**Statement**: Absorption severity peaks in middle layers (layer 6-10 in GPT-2; layer 8-14 in Gemma-2B) where hierarchical semantic features are most concentrated. Early layers (feature extraction) and late layers (task execution) have lower absorption.

**Mechanism**: Middle layers encode abstract semantic concepts that naturally form hierarchies (parent "math" → child "algebra"). Early layers capture surface features; late layers encode task-specific outputs where features are less hierarchical.

**Evidence (two independent runs)**:
- Run 1 (2026-04-26T02:19, jb SAE): Layer 4=0.0363, Layer 8=0.0402 (+10.6%) -- *consistent with H1*
- Run 2 (2026-04-26T18:01, different SAE): Layer 4=0.0684, Layer 8=0.0527 (-22.9%) -- *opposite to H1*

**Test**:
- Train/load SAEs at layers 2, 4, 6, 8, 10, 12 for GPT-2 Small
- Train/load SAEs at layers 4, 8, 12, 16, 20 for Gemma-2B
- Compute Chanin absorption scores per layer using consistent feature selection
- Use multiple random seeds to assess stability
- Plot absorption score vs. layer; expect non-monotonic peak in middle

**Expected Outcome**: U-shaped or unimodal curve peaking at layer 6-8 for GPT-2, layer 10-14 for Gemma-2B. If layer ordering is inconsistent across seeds, H1 is unresolvable in its current form.

**Falsification**: If absorption is monotonic across all layers across multiple seeds, H1 is falsified.

---

## Hypothesis H2: Mitigation Effectiveness Hierarchy

**Research Question**: RQ2 (mitigation effectiveness)
**Status**: PARTIALLY CONFIRMED (TopK confirms; ATM/OrtSAE pending; JumpReLU failed)

**Statement**: TopK SAE achieves the largest absorption reduction but at severe reconstruction cost. JumpReLU fails to converge. ATM preserves reconstruction quality while reducing absorption.

**Evidence (pilot, GPT-2 layer 8)**:
- Vanilla SAE: absorption=0.2253, MSE=13.53
- TopK SAE: absorption=0.066, MSE=110.23 (8x worse), absorption reduction=70.9%
- JumpReLU SAE: absorption=0.625, MSE=3419.61 -- **FAILED to converge**
- ATM full results: pending
- OrtSAE full results: pending

**Mechanism**: TopK's hard sparsity constraint forces the SAE to use fewer features, reducing absorption opportunity. JumpReLU's gradient-based sparsity may conflict with the absorption dynamics. ATM's dynamic masking may preserve important features while reducing absorption.

**Test**:
- Train SAE variants (vanilla, TopK, JumpReLU, OrtSAE, ATM, Matryoshka) on GPT-2 layer 8
- Measure absorption score and reconstruction CE loss for each
- Compare: absorption reduction vs. reconstruction quality tradeoff
- Plot Pareto frontier of absorption vs. reconstruction

**Expected Outcome**:
- TopK: highest absorption reduction, severe reconstruction penalty (confirmed: 70.9% reduction, 8x MSE)
- JumpReLU: failed to converge under tested configuration
- ATM: pending
- OrtSAE: pending
- Matryoshka: pending

**Falsification**: If no method achieves >40% absorption reduction without >2x MSE increase, H2 is strongly falsified for all methods.

---

## Hypothesis H3: Absorption is a Steering Signature, Not a Silencing Signal

**Research Question**: RQ3 (causal intervention reliability)
**Status**: REVERSED from original prediction (full experiment, N=100, contradicts pilot, N=20)

**Statement**: High-absorption features exhibit **higher** steering sensitivity than low-absorption features. Absorption does not reduce causal influence -- it redistributes it. Absorbed features may function as "hub" features with high residual stream leverage.

**Evidence**:
- Pilot (N=20 features, alpha=5): Low-absorption mean effect=0.791, High-absorption mean effect=0.438, ratio=1.81 -- *consistent with original H3*
- Full experiment (N=100 features, alpha=5): High-absorption mean effect=0.1035, Low-absorption mean effect=0.0874, ratio=0.84 -- *OPPOSITE to original H3*
- Full experiment Spearman r (UAS vs sensitivity): **+0.3548** (p=2.92e-04) -- *positive, not negative as originally predicted*

**Mechanism (revised)**: Absorbed features may represent "hub" directions in the residual stream -- geometrically close to many downstream computations. Steering hub directions produces outsized output changes because they are entangled with many concept representations simultaneously. This explains the positive correlation: absorbed features are more, not less, causally manipulable.

**Test**:
- Identify 100 high-absorption and 100 low-absorption features from GPT-2 layer 8 SAE (using UAS)
- Perform steering intervention at multiple alpha values [1, 3, 5, 10, 20]
- Add null controls: shuffled feature directions, random directions
- Measure logit lens change, output probability shift, and per-token steering effect
- Investigate whether absorbed feature steering effect correlates with feature activation frequency (hub features fire more frequently)
- Replicate with Gemma-2B layer 8

**Expected Outcome**: High-absorption features show larger steering effects (ratio < 1.0). After applying ATM/OrtSAE, the absorption distribution shifts and the steering effect distribution shifts accordingly.

**Falsification**: If absorbed and non-absorbed features show equal steering sensitivity in a well-powered replication with null controls, H3 is falsified.

---

## Hypothesis H4: UAS Correlates with Supervised Absorption

**Research Question**: RQ4 (unsupervised detection)
**Status**: CONFIRMED (strong correlation across all pilot runs)

**Statement**: The Unsupervised Absorption Score (UAS), computed from feature cosine similarity variance and activation frequency skewness, correlates significantly (Spearman r > 0.6) with the supervised absorption metric from Chanin et al.

**UAS Formula**:
```
UAS(f) = α * cos_sim_variance(f) + β * freq_skewness(f)
```
Where:
- `cos_sim_variance(f)` = variance of cosine similarities between feature f and all other features (high variance → feature overlaps with many others → absorbed)
- `freq_skewness(f)` = skewness of activation frequency distribution for feature f across contexts (high skew → feature fires in narrow contexts → absorbed child)

**Evidence**:
- Run 1 (N=100 features per layer): Layer 4 r=0.8147 (p=6.3e-25), Layer 8 r=0.7603 (p=4.5e-20), combined r=0.7875
- Run 2 (N=100 features per layer, different SAE): combined r=0.6466
- Both runs: r > 0.3 threshold by wide margin

**Mechanism**: Absorption is a geometric phenomenon -- absorbed features have directions close to other features, and their activation patterns are narrow subsets of their parent's pattern. Both signatures are detectable without probes.

**Test**:
- Compute UAS for all features in GPT-2 layer 8 SAE
- Compute supervised absorption scores for the same features (using Chanin first-letter probe)
- Compute Spearman correlation between UAS and supervised absorption
- Validate UAS on Gemma-2B SAE (different model family)
- Tune UAS hyperparameters (alpha, beta) to maximize correlation

**Expected Outcome**: UAS captures at least 36% of variance in supervised absorption (r > 0.6). Confirmed with r=0.65-0.79.

**Falsification**: If r < 0.4, UAS is not useful and we fall back to supervised metrics. Currently not falsified.

---

## Hypothesis H5: Absorption Degrades Downstream Discriminability

**Research Question**: RQ5 (contrarian hypothesis)
**Status**: DIRECTIONAL CONFIRMATION (marginal failure at 4.95% vs. 5% threshold)

**Statement**: High-absorption features consistently degrade downstream discriminability compared to low-absorption features, across both simple classification and causal reasoning tasks.

**Evidence (pilot, N=48 features, 3 UAS bins)**:
| Absorption Bin | UAS Range | Simple AUC | Causal AUC | Delta |
|---------------|-----------|------------|------------|-------|
| Low | 0.001-0.002 | 0.710 ± 0.147 | 0.547 ± 0.041 | -0.163 |
| Mid | 0.008-0.009 | 0.735 ± 0.176 | 0.555 ± 0.067 | -0.180 |
| High | 0.025-0.041 | 0.636 ± 0.166 | 0.522 ± 0.027 | -0.113 |

- Simple task: high-absorption 7.45% worse than low-absorption (AUC 0.636 vs 0.710)
- Causal task: high-absorption 2.51% worse than low-absorption (AUC 0.522 vs 0.547)
- Task-dependence delta: **4.95%** (threshold: 5%) -- marginal fail

**Note**: Causal task has low overall discriminability (AUC near 0.5), suggesting synthetic counterfactual pairs do not reliably engage GPT-2's causal reasoning.

**Mechanism**: Absorbed features fire as proxies for their children, producing noisy concept representations. Both simple and causal tasks suffer from this noise, but the effect is more visible on simple tasks (which require clean concept signals).

**Test**:
- Expand to 100+ features stratified by UAS
- Replace synthetic counterfactual pairs with real causal QA (e.g., CounterFact, TruthfulQA)
- Evaluate on both BIAS_IN_BIOS (simple) and causal QA (counterfactual)
- Compute per-feature discriminability AUC across both task types

**Expected Outcome**: High-absorption features show consistent degradation across both task types (simple delta > 5%, causal delta > 5%, total task-dependence delta > 5%).

**Falsification**: If absorbed and non-absorbed features perform equally on both task types, absorption has no practical consequence for interpretability -- a critical finding that would reframe the entire research area. Currently not falsified.

---

## Summary Table

| ID | Hypothesis | Status | Key Metric | Evidence |
|----|-----------|--------|------------|----------|
| H1 | Absorption peaks in mid-layers | UNRESOLVED | Absorption score vs. layer | Two pilot runs contradict (L4>L8 vs L8>L4) |
| H2 | Mitigation effectiveness hierarchy | PARTIALLY CONFIRMED | Absorption reduction + reconstruction tradeoff | TopK: 70.9% reduction, 8x MSE; JumpReLU: failed |
| H3 | Absorption is a steering signature | REVERSED | Steering effect size ratio | Full (N=100): r=+0.35, high-abs more steerable |
| H4 | UAS correlates with supervised absorption | CONFIRMED | Spearman r | r=0.65-0.79 across all runs |
| H5 | Absorption degrades downstream discriminability | DIRECTIONAL | Task accuracy delta | 4.95% vs 5% threshold (marginal fail) |
