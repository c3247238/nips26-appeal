# Introduction

Sparse Autoencoders (SAEs) have emerged as a foundational tool for mechanistic interpretability, decomposing neural network activations into sparse, monosemantic feature representations. However, a critical reliability threat—feature absorption—undermines SAE-based analysis when applied to safety-critical decisions. When a parent feature activates strongly, its child features may substitute for it in sparse representations, breaking the intended hierarchical structure and producing representations that no longer correspond to their intended semantic meanings.

Feature absorption was documented as a widespread phenomenon by Chanin et al. (2024), who hypothesized that decoder geometry drove the effect. Korznikov et al. (2026) later proposed sparsity optimization as an alternative explanation. Neither study, however, conducted a factorial decomposition isolating the individual contributions of encoder and decoder components.

## The Absorption Problem

Consider a three-level hierarchical feature structure where "animal" (Level 0) is the parent of "dog" and "cat" (Level 1), which are themselves parents of "poodle" and "labrador" (Level 2). When the "animal" feature activates strongly, child features "dog" and "cat" may partially substitute for "animal" in the SAE's sparse representation. This substitution—absorption—means the internal representation no longer faithfully encodes which higher-level concept is actually present.

The consequence is severe for interpretability: a feature annotated as detecting "animal" concepts may fire strongly even when no such concept is present, because its child features are active. SAE-based safety analysis that relies on feature activations to identify harmful content becomes unreliable if absorption systematically distorts those activations.

## Our Contribution: Conditional Encoder-Driven Mechanism

We present the first factorial decomposition of the absorption mechanism, independently varying encoder and decoder initialization and training to isolate each component's contribution. Our key tool is a 2×2 factorial experimental design on synthetic stochastic hierarchies:

| Condition | Encoder | Decoder | Expected |
|-----------|---------|---------|----------|
| A | Random | Random | Baseline geometry only |
| B | Trained | Random | Encoder alignment only |
| C | Random | Trained | Decoder geometry only |
| D | Trained | Trained | Full training |

If absorption is encoder-driven, Conditions B and D should produce similar absorption rates (B ≈ D). If decoder geometry contributes, Condition C should exceed Condition A (C > A).

Our full experiment (5 seeds × 4 conditions × 100 samples) reveals a nuanced picture. **B ≈ D holds with delta = 0.037**, confirming the encoder is sufficient. However, **C ≈ A fails with delta = 12.10**—Condition C shows extreme variance (std = 17.13, range 0–43.84), indicating the decoder's role is configuration-dependent, not uniformly zero as pilot experiments suggested. The prevailing decoder-centric narrative is incomplete but not entirely wrong.

This conditional encoder-driven mechanism has two implications. First, it identifies the encoder's learned alignment with hierarchical co-activation patterns as the primary driver of absorption—during training, the encoder learns local minima where children substitute for parents. Second, it reveals that decoder contribution varies across random seed configurations, suggesting that decoder geometry amplifies absorption in some training regimes but has no effect in others.

## Additional Findings

Beyond the factorial decomposition, we characterize three additional phenomena with implications for SAE reliability:

**Hierarchy Strength Dependence (H_Comp)**. We tested whether absorption increases monotonically with parent-child cosine similarity across six levels (0.50–0.95). Contrary to the phase-transition hypothesis, **no monotonic relationship exists** (R² = 0.04; regression slope = −0.296, p = 0.703). Absorption ranges from 0.51 to 1.20 across levels with no clear trend.

**Sensitivity-Absorption Pareto Frontier (H_Pareto)**. The theoretical prediction of an irreducible trade-off between feature sensitivity (Hu et al., 2025) and absorption rate could not be confirmed. Across four L0 targets (16, 32, 64, 128), absorption collapsed to zero (mean = 0.0, std = 0.08) while sensitivity remained stable (mean = 0.1054, std = 0.0008). The frontier fit degenerates, indicating either the trade-off does not exist in synthetic SAEs or the measurement approach saturates.

**Safety-Critical Feature Vulnerability (H_Safe)**. Using Gemma Scope SAEs (gemma-2b, layer 12) and Neuronpedia annotations, we tested whether safety-critical features (deception, jailbreak, harm) show elevated absorption compared to non-safety controls. **The hypothesis was not confirmed**: Gemma Scope pilot shows p = 1.0 (both groups at 0.0 absorption); GPT-2 Small held-out validation shows p = 0.345 (safety mean = 233.13, non-safety mean = 221.70). Safety-critical features do not appear disproportionately absorbed—this is a positive result for SAE-based safety analysis, though larger-scale validation is required.

![Conceptual illustration of absorption phenomenon](figures/figure1_encoder_decoder_contribution.pdf)

**Figure 1**: (a) In an SAE with a hierarchical feature structure, the parent feature ("animal") activates when the input contains animal-related content. (b) When child features ("dog", "cat") activate strongly, they substitute for the parent in sparse representations, reducing parent activation through absorption. The decoder reconstruction remains accurate, but the latent code no longer corresponds to the intended semantic hierarchy.

## Implications for Safety-Critical Interpretability

If SAEs are to underpin safety analysis—detecting deceptive patterns, jailbreak attempts, or harmful content—absorption must be understood and accounted for. Our results inform this agenda in two ways. The encoder-driven mechanism identifies the encoder as the primary mitigation target; absorption does not arise inevitably from decoder geometry. The null result for safety-critical features suggests current SAE-based safety tools are not systematically undermined by absorption, but this requires validation at larger scale.

## Paper Structure

Section 2 provides background on SAEs and prior work on absorption. Section 3 details our 2×2 factorial experimental methodology, hierarchy strength sweep, Pareto frontier measurement, and safety-critical feature analysis. Section 4 reports results for all four hypotheses. Section 5 discusses implications, the configuration-dependent decoder role, and limitations. Section 6 concludes with contributions and future directions.

<!-- FIGURES
- Figure 1: figure1_encoder_decoder_contribution.pdf — Conceptual illustration of absorption phenomenon in SAE hierarchical feature structures
- None
-->
