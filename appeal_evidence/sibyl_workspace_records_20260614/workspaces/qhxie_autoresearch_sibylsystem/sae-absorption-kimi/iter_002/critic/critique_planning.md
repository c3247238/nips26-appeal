# Planning Critique: Construct Validity of SAEBench Absorption Metric

## Overall Assessment

The methodology is generally well-designed and follows the proposal closely. The SAE selection is appropriate, the statistical protocol is sound, and the ablation schedule covers key robustness checks. However, several planning decisions introduced or failed to prevent critical issues that undermine the study's validity.

**Methodology Quality: 6/10**

---

## Strengths

### 1. Appropriate SAE Selection

The 8 SAE configurations span a broad absorption-rate spectrum (from GatedSAE at 0.008 to TopK at 0.576 on first-letter tasks). The inclusion of PAnneal as an OrtSAE substitute is reasonable given checkpoint availability.

### 2. Sound Statistical Protocol

- Bootstrap CIs (B=10,000) are appropriate for small-n correlation
- Paired t-test for hierarchy specificity is the correct test
- Pre-registered thresholds (r > 0.6 for H1, p < 0.05 for H2) provide clear criteria

### 3. Good Ablation Coverage

The ablation schedule (tau_fs sweep, GPT-2 replication, Random-SAE control) covers the key robustness dimensions. The tau_fs robustness analysis is particularly well-executed.

### 4. Frequency Matching Control

The synthetic balanced dataset (N=100 sentences per concept, equal parent/child frequency) correctly addresses the frequency confound identified in the proposal.

---

## Critical Planning Flaws

### 1. Random-SAE Control Implementation Was Not Verified (CRITICAL)

The methodology describes the Random-SAE as "permuted decoder directions from Standard SAE" (outline) or "permutes the encoder matrix W_enc" (paper Section 3.1). The planning phase should have:
- Specified exactly which matrix is permuted
- Verified the implementation before running experiments
- Documented the expected behavior (decoder permutation -> identical encoder outputs -> identical absorption scores)

The planning failure here is that the Random-SAE control was not sufficiently specified, leading to contradictory descriptions and a potentially misinterpreted result.

### 2. Perfect Probe Accuracy Was Not Anticipated (MAJOR)

The methodology specifies a minimum probe AUROC threshold of 0.7, with all hierarchies achieving AUROC = 1.0. While this validates probe quality, it also means:
- resid_acc = sae_acc = 1.0 for all hierarchies
- The absorption formula collapses to k-sparse loss only

The planning phase should have anticipated that perfect accuracy would collapse the metric and designed a more challenging probe task (e.g., harder hierarchies, noisier data, or a different metric formulation).

### 3. Control Condition Structural Mismatch (MAJOR)

The planning phase did not recognize that:
- Hierarchy condition: multi-class (parent vs. 2-3 children)
- Non-hierarchy condition: binary (word A vs. word B)

This structural mismatch confounds the hierarchy specificity test. The planning phase should have designed structurally equivalent controls.

### 4. GPT-2 Replication Was Under-Specified (MAJOR)

The GPT-2 replication plan tested only 2 SAEs (Standard and TopK) with no correlation computation possible. The planning phase should have either:
- Tested enough SAEs on GPT-2 to compute a correlation (minimum 5-6)
- Framed GPT-2 as a qualitative check rather than a "replication"
- Recognized that near-zero scores on GPT-2 might indicate a ceiling effect

### 5. Fixed k=10 Was Not Justified (MAJOR)

The methodology specifies k=10 for k-sparse probing "following the SAEBench default." However:
- SAEBench uses tau_fs to determine k adaptively per sample, not a fixed k
- Different hierarchies have different numbers of children (2-3), requiring different representational capacity
- Fixed k systematically penalizes more complex hierarchies

The planning phase should have justified the fixed k choice or used an adaptive k.

---

## Minor Planning Issues

1. **Probe training variance:** The methodology specifies 200 Adam epochs with lr=0.05 but does not specify random seeds or variance estimation across seeds.
2. **Sentence templates not documented:** The exact templates used for synthetic data generation are not specified in the methodology.
3. **No power analysis:** The planning phase did not compute the required sample size for the correlation test.

---

## Resource Planning

The resource estimate (~45 minutes for full experiment) was accurate. The pilot design (2 SAEs x 3 hierarchies x 1 control pair, 10-15 minutes) was appropriate for rapid validation.

However, the pilot's "success criterion" ("show the expected ordering: Matryoshka < BatchTopK") was too weak. The pilot should have also checked:
- Whether sae_acc < resid_acc (to verify the metric captures SAE encoding loss)
- Whether the Random-SAE behaves as expected
- Whether probe AUROCs are in a reasonable range (not all 1.0)

---

## Recommendations for Future Planning

1. **Specify controls precisely:** Document exactly what is permuted, why, and what the expected behavior is.
2. **Anticipate metric collapse:** Design probe tasks that are challenging enough to produce sae_acc < resid_acc.
3. **Match control structures:** Use binary hierarchy pairs or multi-class non-hierarchy controls.
4. **Power analysis:** Compute required sample size before running the experiment.
5. **Pilot more thoroughly:** Check metric behavior, not just ranking order.
6. **Justify hyperparameters:** Explain why k=10 is appropriate for all hierarchies.
