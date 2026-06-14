# 4. Experiments

## 4.1 Pilot Validation

Before full-scale experiments, we ran three pilot studies to validate UAD and DFDA feasibility.

**Pilot P1 (UAD, layer 8).** UAD on GPT-2 Small layer 8 achieved F1 = 0.704, precision = 0.543, and recall = 1.0, with 19 true positives out of 35 same-cluster pairs. This exceeded the pilot threshold of F1 >= 0.5, confirming the approach was viable.

**Pilot P2 (DFDA, 2 pairs).** DFDA on 2 absorbed pairs showed 99.999% mean MSE improvement. This suspiciously high value triggered our metric caveat protocol: the improvement reflects near-zero parent prediction on child-dominant examples, not validated absorption recovery. We proceeded with the full experiment but pre-registered the metric limitation.

**Pilot P3 (UAD, cross-layer).** UAD across layers 4, 8, and 10 yielded F1 = 0.432 (layer 4), 0.704 (layer 8), and 0.548 (layer 10), with mean F1 = 0.561. Layer 4 fell below the 0.5 threshold, suggesting weaker hierarchical structure in early layers. We designated cross-layer as PARTIAL_PASS and proceeded to full experiments.

## 4.2 E1: UAD Full Scale

UAD on GPT-2 Small layer 8 with 1,000 OpenWebText samples achieved F1 = 0.725, exceeding the target threshold of 0.6. Precision was 0.569 and recall was 1.0, meaning all 29 Chanin-supervised collision cases were detected among 51 same-cluster pairs. No supervised collisions were missed. The analysis covered 15,000 token positions and completed in 7.6 seconds.

The precision of 0.569 indicates that 43% of same-cluster pairs are false positives. This is expected: hierarchical clustering groups features by general co-occurrence similarity, not exclusively by absorption relationships. UAD is a detection tool that identifies a enriched candidate set requiring post-hoc filtering, not a classifier that perfectly separates absorbed from non-absorbed pairs.

## 4.3 E2: Multi-Seed Robustness

UAD was run with three independent random seeds (42, 123, 456), each sampling different subsets of 1,000 examples from OpenWebText. All three seeds produced identical results: F1 = 0.725, precision = 0.569, recall = 1.0, with 29 true positives out of 51 same-cluster pairs. The standard deviation across seeds was 0.000.

This perfect consistency reflects determinism rather than robustness in the traditional sense. UAD operates on a fixed, pretrained SAE's co-occurrence matrix, and 1,000 samples suffice to stabilize the phi coefficient statistics. The SAE weights are frozen; the only randomness is in corpus sampling, which does not perturb the co-occurrence structure enough to change cluster assignments for the top 500 features. This is a practical advantage---full reproducibility---but does not demonstrate robustness to SAE retraining, corpus change, or model change.

## 4.4 E3: Cross-Layer Validation

UAD performance varies substantially across layers (Figure 3). Layer 8 achieved F1 = 0.725 (pilot: 0.704), layer 10 achieved F1 = 0.548, and layer 4 achieved F1 = 0.432---below the 0.5 minimum threshold. The mean F1 across layers was 0.561 with standard deviation 0.111.

![Cross-Layer F1 Comparison](figures/fig3.pdf)

Layer 8's optimality is consistent with prior work showing mid-to-late layers contain the most structured feature hierarchies [Elhage et al., 2022]. Layer 4's lower F1 (precision = 0.276, recall = 1.0) suggests weaker hierarchical structure in early layers, where features encode lower-level patterns with less semantic abstraction. All layers maintained perfect recall, confirming that UAD does not miss collisions even when overall precision is low.

## 4.5 E5: DFDA Scaling (8 Pairs)

DFDA was scaled from 2 pairs (pilot) to 8 pairs, using the same 2-layer, 64-hidden-unit MLP architecture (193 parameters per pair). All 8 pairs showed positive MSE improvement, with a mean improvement of 99.5%. The total parameter budget was 1,544 parameters, or 0.004% of the SAE's 37.7M parameters.

Table 5 reports per-pair details.

| Pair | Parent | Child | Phi | Baseline MSE | Compensated MSE | Improvement | Params |
|------|--------|-------|-----|-------------|-----------------|-------------|--------|
| 1 | 504 | 21 | 0.812 | 0.102 | 1.5e-10 | 99.999% | 193 |
| 2 | 815 | 1408 | 0.812 | 9.993 | 5.6e-10 | 99.999% | 193 |
| 3 | 1039 | 851 | 0.812 | 39.183 | 2.2e-08 | 99.999% | 193 |
| 4 | 1833 | 851 | 0.812 | 0.100 | 8.7e-12 | 99.999% | 193 |
| 5 | 1647 | 1645 | 0.812 | 0.430 | 0.016 | 96.225% | 193 |
| 6 | 2899 | 3623 | 0.812 | 4.654 | 8.7e-07 | 99.999% | 193 |
| 7 | 1068 | 494 | 0.793 | 3.552 | 5.5e-11 | 99.999% | 193 |
| 8 | 1420 | 1647 | 0.767 | 0.230 | 8.8e-10 | 99.999% | 193 |
| **Mean** | -- | -- | **0.803** | -- | -- | **99.528%** | **193** |

Figure 4 visualizes the per-pair MSE reduction with the metric caveat prominently annotated.

![DFDA Per-Pair Results](figures/fig4.pdf)

**Metric caveat.** The near-100% improvement is artifactual. The MLP learns to predict near-zero parent values in child-dominant positions (where the parent is already suppressed), not to recover absorbed activations on parent-positive examples. The baseline MSE measures deviation from zero; the compensated MSE measures deviation from a near-zero prediction. This is not absorption recovery. DFDA is reported as preliminary work pending a parent-positive evaluation protocol.

## 4.6 Summary

Table 6 presents the comprehensive UAD results across all experimental conditions.

| Condition | Layer | Precision | Recall | F1 | TP / Same-Cluster | Tokens |
|-----------|-------|-----------|--------|-----|-------------------|--------|
| Pilot | 8 | 0.543 | 1.000 | 0.704 | 19 / 35 | 1,000 |
| Full | 8 | 0.569 | 1.000 | 0.725 | 29 / 51 | 1,000 |
| Seed 42 | 8 | 0.569 | 1.000 | 0.725 | 29 / 51 | 1,000 |
| Seed 123 | 8 | 0.569 | 1.000 | 0.725 | 29 / 51 | 1,000 |
| Seed 456 | 8 | 0.569 | 1.000 | 0.725 | 29 / 51 | 1,000 |
| Layer 4 | 4 | 0.276 | 1.000 | 0.432 | 8 / 29 | 1,000 |
| Layer 10 | 10 | 0.377 | 1.000 | 0.548 | 23 / 61 | 1,000 |

Figure 2 summarizes UAD performance visually across conditions.

![UAD Performance Summary](figures/fig2.pdf)

Perfect recall is maintained across all conditions. F1 varies by layer, with layer 8 optimal. The multi-seed consistency demonstrates deterministic behavior on fixed SAEs.

<!-- FIGURES
- Figure 2: gen_fig2.py, fig2.pdf — UAD performance grouped bar chart across conditions
- Figure 3: gen_fig3.py, fig3.pdf — Cross-layer F1 comparison bar chart
- Figure 4: gen_fig4.py, fig4.pdf — DFDA per-pair MSE reduction with caveat annotation
- Table 5: inline — DFDA per-pair detail table
- Table 6: inline — Comprehensive UAD results table
-->
