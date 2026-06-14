# Testable Hypotheses with Expected Outcomes

## Evidence-Driven Revision Notice

This document has been revised based on experimental evidence from the full study (GPT-2 Small, 26 first-letter features, layers 0/4/8/10; Pythia-70M cross-validation), the 6-perspective ideation round, and the result debate synthesis. The original hypotheses (H1-H3) have been tested and their status updated. New hypotheses (H1b-H5) were added based on unexpected findings. H4 was subsequently tested with EC50 analysis and found NOT SUPPORTED. The current revision incorporates the Local Inhibition Graph framework as the primary research direction.

---

## Prior Hypotheses (From Previous Rounds -- For Reference)

### H1: Absorption Degrades Steering Effectiveness (Raw Metric)
**Result: FALSIFIED** --- r = +0.008 (layer 4), r = -0.301 (layer 8), p > 0.05. No significant correlation.

### H1b: Absorption Degrades Delta-Corrected Steering Effectiveness
**Result: PARTIALLY SUPPORTED** --- r = -0.431, p = 0.028 (layer 8, uncorrected). Does NOT survive Bonferroni (p = 0.334) or BH-FDR (q = 0.107).

### H2: Absorption Degrades Sparse Probing Accuracy
**Result: FALSIFIED** --- r = -0.003 (layer 4), r = -0.107 (layer 8), p > 0.05.

### H3: Consistency Across Configurations
**Result: FALSIFIED** --- Opposite-sign slopes; CV computation bug.

### H4: Absorption Affects Steering Efficiency, Not Capability (EC50)
**Result: NOT SUPPORTED** --- No significant EC50 difference (layer 4: p = 0.232; layer 8: p = 0.435).

### H5: Absorption Affects Recall (Coverage), Not Precision (Selectivity)
**Result: SUPPORTED** --- Precision = 1.0 universally at k >= 5; recall varies.

---

## New Hypotheses: Local Inhibition Graph Framework

### H6 (Primary): Graph Edges Predict Known Absorption Pairs

**Statement:** For a pretrained SAE, edges in the local inhibition graph (top-k correlated neighbors per latent) correspond to known absorption pairs with precision significantly above chance.

**Formalization:**
- Construct local inhibition graph: for each latent i, keep top-k neighbors with highest decoder correlation |G_ij| where G = W_dec^T W_dec.
- Use Chanin et al.'s absorption detection on first-letter features (A-Z) as ground truth.
- For each absorption pair (parent i, absorbing j), check if j is in N(i).

**Expected outcome:**
- Precision@20 >= 0.10 (vs. ~0.004 expected by chance = 20/24000 for 24K latents)
- This represents a 25x enrichment over chance
- Fisher exact test: p < 0.001 for enrichment

**Falsification criterion:** If precision@20 <= 0.05, the structural correspondence between decoder correlations and absorption pairs fails.

**Why this matters:** Validates the core theoretical claim that W_dec^T W_dec = G_LCA captures competitive suppression.

---

### H7 (Secondary): Inhibition Explains Precision-Recall Asymmetry

**Statement:** The competitive suppression mechanism explains why absorption affects recall (coverage) but not precision (selectivity).

**Formalization:**
- Precision invariance: Inhibition from child to parent does not cause the parent to fire for incorrect inputs. When a latent fires, it fires for the correct concept (selectivity preserved).
- Recall loss: Inhibition suppresses parent activation when the child fires, reducing the number of true positives detected (coverage reduced).

**Expected outcome:**
- Correlation between total incoming inhibition and recall: r < -0.3, p < 0.05
- Correlation between total incoming inhibition and precision: |r| < 0.1, p > 0.05 (no correlation)

**Falsification criterion:** If inhibition correlates with precision (suggesting suppression causes false positives), the mechanism explanation fails.

**Why this matters:** Provides a mechanistic explanation for the project's strongest finding (H5: precision invariant, recall variable).

---

### H8 (Secondary): Graph Predicts At-Risk Features

**Statement:** Latents with high total incoming inhibition (sum of edge weights from neighbors) are more likely to be absorbed, enabling pre-emptive identification without running the Chanin metric.

**Formalization:**
- For each first-letter feature latent, compute total_inhibition_i = sum_{j in N(i)} |G_ij|
- Test correlation: total_inhibition vs. absorption_rate

**Expected outcome:**
- Pearson r > 0.3, p < 0.05
- Features in the top quartile of total inhibition have >2x absorption rate vs. bottom quartile

**Falsification criterion:** If r < 0.2 or p > 0.05, the graph cannot predict at-risk features.

**Why this matters:** Provides a practical diagnostic tool --- practitioners can identify at-risk features from decoder correlations alone.

---

### H9 (Exploratory): Layer-Dependent Graph Structure

**Statement:** The inhibition graph structure varies with layer depth, with deeper layers showing stronger competitive dynamics, explaining the layer-dependent effects in the project's data.

**Formalization:**
- Construct graphs for layers 0, 4, 8, 10 of GPT-2 Small
- Compute mean edge weight, graph density, and clustering coefficient per layer
- Test correlation with layer index

**Expected outcome:**
- Mean edge weight increases with layer depth: r > 0.3
- Layer 8 (where H1b was significant) has the strongest inhibition structure
- Graph density increases with layer depth

**Falsification criterion:** If no systematic trend with layer depth, the layer-dependence of absorption effects is not explained by inhibition structure.

**Why this matters:** Explains why delta-corrected steering correlation was significant only at layer 8.

---

### H10 (Exploratory): Homeostatic Rebalancing Restores Parent Firing

**Statement:** A single-pass rebalancing of activations along graph edges restores parent feature firing without degrading reconstruction quality.

**Formalization:**
- For input activation a, compute original latents: z = f(W_enc * a + b_pre)
- Compute inhibition per latent: inh_i = sum_{j in N(i)} G_ij * z_j
- Apply boost: z'_i = z_i + alpha * inh_i
- Clip negative values; constrain: ||a - W_dec * z'||_2 <= (1 + epsilon) * ||a - W_dec * z||_2

**Expected outcome:**
- Parent feature firing rate increases by >20% after rebalancing
- Reconstruction error increase < 5%
- Optimal alpha in range [0.5, 2.0]

**Falsification criterion:** If reconstruction error increases > 5% or parent firing does not improve, the repair mechanism fails.

**Why this matters:** Provides the first training-free post-hoc repair for absorption, complementing the diagnostic tool.

---

## Hypothesis Testing Summary

| Hypothesis | Type | Test | Expected Outcome | Falsification |
|------------|------|------|------------------|---------------|
| H6 | Primary | Graph precision@20 vs. chance | >= 0.10 (25x enrichment) | <= 0.05 |
| H7 | Secondary | Inhibition vs. recall/precision | r(recall) < -0.3; r(precision) ~ 0 | r(precision) significant |
| H8 | Secondary | Total inhibition vs. absorption rate | r > 0.3, p < 0.05 | r < 0.2 |
| H9 | Exploratory | Graph stats vs. layer depth | r(mean_weight, layer) > 0.3 | No trend |
| H10 | Exploratory | Rebalancing efficacy | Parent firing +20%, recon error < 5% | Error > 5% or no improvement |

## Integration with Prior Findings

| Prior Finding | New Explanation |
|---|---|
| Precision = 1.0 universally (H5) | Inhibition preserves selectivity; no false positives introduced |
| Recall varies widely | Inhibition suppresses parent activation when child fires |
| Layer 8 effect stronger than layer 4 | Deeper layers have stronger hierarchical structure = stronger inhibition |
| Feature U (24.2% abs) still steers 100% | Decoder direction preserved; only encoder activation suppressed |
| Delta-corrected correlation at layer 8 | Baseline subtraction isolates unique information lost to inhibition |
| EC50 shows no efficiency degradation | Inhibition affects activation probability, not decoder alignment |

## Risk Assessment

| Hypothesis | Risk | Mitigation |
|------------|------|------------|
| H6 | Graph edges may not predict absorption | Structural correspondence is exact; if it fails, report as finding about decoder correlation limitations |
| H7 | Inhibition may correlate with precision | Test explicitly; if it does, the mechanism is more complex than competitive suppression |
| H8 | At-risk prediction may be weak | Use multiple graph statistics (degree, centrality, clustering) |
| H9 | Layer trend may be confounded | Control for feature frequency and dictionary size |
| H10 | Repair may degrade reconstruction | Alpha sweep with reconstruction constraint; report Pareto frontier if trade-off exists |
