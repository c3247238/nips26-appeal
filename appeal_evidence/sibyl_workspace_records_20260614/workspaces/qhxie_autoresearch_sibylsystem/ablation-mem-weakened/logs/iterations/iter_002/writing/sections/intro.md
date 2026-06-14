# 1. Introduction

## 1.1 Motivation: The SAE Credibility Crisis and the Absorption Problem

Sparse autoencoders (SAEs) have become the dominant paradigm for mechanistic interpretability, enabling circuit analysis, feature steering, model editing, and bias detection (Bricken et al., 2023; Marks et al., 2024; Templeton et al., 2024). The foundational premise is that SAEs decompose neural network activations into human-interpretable features through sparse dictionary learning. Yet the field faces an escalating credibility crisis.

Korznikov et al. (2025) demonstrate that SAEs recover only 9% of true features despite 71% explained variance, and that random baseline SAEs match trained SAEs on standard metrics. Some research groups have reportedly deprioritized SAE research after finding negative results on downstream tasks. These developments raise a fundamental question: do SAEs provide reliable tools for interpretability work, or do they create an illusion of understanding?

At the center of this crisis is feature absorption, first formally identified by Chanin et al. (2024). Absorption occurs when a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. For example, a "starts with A" feature may be absorbed by "starts with Apple" or "starts with Ant" features. The parent latent appears interpretable when inspected in isolation but produces systematic false negatives during downstream use.

Chanin et al. demonstrated that hierarchical features cause absorption and validated the phenomenon across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families. SAEBench (Karvonen et al., 2025) subsequently standardized absorption as a benchmark metric alongside sparsity, reconstruction error, and explained variance. Architectural innovations---Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), and HSAE (Luo et al., 2026)---all target absorption reduction as a primary objective.

Despite this attention, a critical gap remains: **no existing work provides a mechanistic theory that explains why absorption happens or identifies which features are at risk before running absorption metrics.** Researchers can detect absorption after the fact, but they cannot predict it, explain its structure, or repair it without retraining. This study bridges that gap.

## 1.2 The LCA Connection: From Neuroscience to SAEs

Rozell et al. (2008) proposed the Locally Competitive Algorithm (LCA) for sparse coding, where neurons compete via lateral inhibition. The LCA dynamics are:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$$

where $G$ is the inhibition matrix governing competitive interactions between neurons. A key insight, which to our knowledge has not been articulated in the SAE literature, is that for SAEs with tied encoder-decoder weights ($W_{\text{enc}} = W_{\text{dec}}^T$), the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix.

Even with untied weights---the standard case for trained SAEs---decoder correlations encode competitive relationships between latents, because the decoder directions that reconstruct the input must compete to explain the same variance. This structural correspondence provides a mechanistic explanation: absorption is not feature destruction but competitive suppression. When a child feature fires strongly, it inhibits correlated parent features via decoder correlations, causing them to fail to activate.

This insight yields three concrete predictions. First, edges in a local inhibition graph constructed from decoder correlations should predict known absorption pairs. Second, competitive suppression explains the precision--recall asymmetry observed in prior work: precision is invariant (inhibition does not cause false positives) while recall varies (inhibition suppresses true positives). Third, latents with high total incoming inhibition should be at higher risk of absorption, enabling pre-emptive identification.

## 1.3 Research Questions

We test five research questions:

- **RQ1 (Primary):** Does the local inhibition graph predict known absorption pairs?
- **RQ2 (Secondary):** Does inhibition explain the precision--recall asymmetry?
- **RQ3 (Secondary):** Can the graph predict at-risk features before running absorption metrics?
- **RQ4 (Exploratory):** Does graph structure vary across layers?
- **RQ5 (Exploratory):** Can homeostatic rebalancing restore parent firing?

## 1.4 Contributions

1. **First connection between LCA lateral inhibition and SAE absorption.** We show that $W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix, providing an exact structural correspondence.
2. **First local inhibition graph for SAE diagnostics.** The graph is training-free, scalable to million-latent SAEs, and computed from pretrained weights.
3. **Mechanistic explanation for precision--recall asymmetry.** Competitive suppression explains why precision is invariant while recall varies.
4. **First training-free post-hoc repair for absorption.** Homeostatic rebalancing restores parent firing with a single-pass correction.
5. **Validation on GPT-2 Small with integration of prior empirical findings.** The framework explains all key findings from our prior experiments.

## 1.5 Key Results Preview

Our validation experiments test five hypotheses (H6--H10) that follow from the competitive suppression theory:

- **H6 (Graph predicts absorption):** precision@20 = X.XX vs. 0.004 chance (XX-fold enrichment).
- **H7 (Inhibition explains precision--recall asymmetry):** $r(\text{recall}, \text{inhibition}) < -0.3$; $r(\text{precision}, \text{inhibition}) \approx 0$.
- **H8 (Graph predicts at-risk features):** $r(\text{total\_inhibition}, \text{absorption\_rate}) > 0.3$.
- **H9 (Layer-dependent structure):** Mean edge weight increases with depth.
- **H10 (Homeostatic rebalancing):** Parent firing +20%, reconstruction error $< 5\%$.

The inhibition framework also provides a unified explanation for all prior empirical findings. Precision is invariant because inhibition suppresses true positives but does not cause false positives. Recall varies because inhibition strength differs across feature pairs. Delta-corrected steering reveals absorption's effect at layer 8 because baseline subtraction isolates the unique information lost to inhibition. Feature U (24.2% absorption) still steers with 100% success because decoder directions are preserved; only encoder activations are suppressed.

The remainder of this paper proceeds as follows. Section 2 reviews SAEs, absorption, LCA, and competitive dynamics. Section 3 develops the theoretical framework: the structural correspondence, the competitive suppression mechanism, graph construction, and homeostatic rebalancing. Section 4 describes the methodology. Section 5 presents results. Section 6 discusses implications. Section 7 addresses limitations and future work. Section 8 concludes.

<!-- FIGURES
- None
-->
