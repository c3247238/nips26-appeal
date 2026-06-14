# Result Debate: Skeptic Perspective

## Overview

The results contain genuine signals but multiple confounds that could prevent publication at a top venue. The paper must address these honestly or reviewers will flag them as fatal weaknesses.

## Critical Weaknesses

### 1. Sample Size Remains Inadequate

n_pos = 18 absorbed features at L6. All AUROC estimates are derived from 18 positive examples, with 95% bootstrap CIs spanning ~0.15. A DeLong test significant at p=0.0012 with n=18 positives will concern reviewers: the result depends entirely on a handful of examples. If even 3-4 labels are incorrect (Neuronpedia proxy quality), the AUROC shifts substantially.

The TopK result (n_pos = 77) looks better on this dimension but uses proxy labels (decoder alignment, not Chanin IG), making it an apples-to-oranges comparison with the Standard result. The "cross-architecture" comparison is confounded by both (a) different label sources and (b) different hook points (resid_pre vs. resid_post).

### 2. Hook Confound in Cross-Architecture Comparison

A3 explicitly notes: "Standard SAE uses hook_resid_pre (before attention layer 6); TopK SAE uses hook_resid_post (after attention layer 6)." This means the two SAEs are trained on *different* representations. The AUROC difference (0.757 vs. 0.837) may reflect the different hook rather than the architecture. This confound is acknowledged in results but must be prominently stated in limitations.

### 3. H2 Oracle Test Methodology Is Questionable

The OMP oracle test checks whether OMP reconstruction reduces the false-negative rate for absorbed features. But the test assumes that (1) the OMP dictionary is the same as the SAE dictionary, and (2) "main features" from Chanin IG correctly identify the absorbed features. Neither is guaranteed. An OMP oracle that reduces FN rate to 0% would require perfect knowledge of the target signal — the 0% result might mean the oracle is poorly calibrated, not that amortization gap is irrelevant.

A more convincing H2 falsification would use a learned sparse code (LISTA or similar) and show that it also fails, ruling out implementation-specific failures.

### 4. H3 Probe Quality Problem Remains Unresolved

D2 entity-type AR = 0.000 (city-continent/country/language all fail). The hypothesis is that this is a probe artifact, not a true signal absence. But we don't have a within-domain control using proper probes to verify. Without this, we cannot say whether absorption generalizes across feature types — we can only say "we couldn't measure it." This is a weaker claim than "it does generalize" or "it doesn't generalize."

### 5. Theoretical Account of Encoder Norm Is Post Hoc

The theoretical motivation for encoder_norm (high norm due to competition pressure from absorbing children) is developed after the empirical finding. The mechanistic note in A2 is plausible but not formally derived. Theorem 1 in the iter_001 framing was for EDA, not encoder_norm. A reviewer will ask: what is the formal prediction for encoder_norm under the absorption model?

## Questions That Must Be Answered

1. What does the encoder_norm predict beyond AUROC? Does it correlate with absorption severity?
2. Why does encoder_norm > EDA? If absorption theory predicts EDA, why is encoder_norm better?
3. Can you falsify encoder_norm by showing it does NOT increase for non-absorbed polysemantic features?

## Verdict

The paper is not ready at current confidence level. The cross-architecture confound (hook point) must be resolved before claiming cross-architecture generalization. The n_pos=18 limitation must be addressed with power analysis or additional data. **Recommend REFINE to address these confounds before writing.**
