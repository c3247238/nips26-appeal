# 1 Introduction

Absorption rates on the same sparse autoencoder (SAE) vary 15-fold depending on which model layer is measured.
On Gemma 2 2B with Gemma Scope JumpReLU SAEs, first-letter absorption ranges from 2.2% at layer 18 to 34.5% at layer 24 (Figure 1).
When measurement extends from the canonical first-letter spelling task to entity-attribute knowledge hierarchies -- city-country, city-continent, city-language -- rates differ significantly across hierarchy types (Kruskal-Wallis $p = 0.005$, 4 of 6 pairwise comparisons significant at $p < 0.05$).
These findings demonstrate that no single absorption rate characterizes an SAE: both layer position and hierarchy structure govern the severity of feature absorption.

Feature absorption is a failure mode where a general (parent) SAE latent fails to fire when a specific (child) latent co-occurs, because the SAE achieves lower $L_0$ by encoding the parent's information into the child's decoder direction (Chanin et al., 2024).
A concrete example: a latent labeled "starts with s" may correctly activate for inputs like "sun" and "stone" but fail to fire for "snake," because a dedicated "snake" latent absorbs the letter-level information into its own decoder direction.
The parent latent appears monosemantic on casual inspection -- it responds cleanly to s-initial tokens -- yet it silently misses a systematic subset of its semantic domain.
This false sense of monosemanticity threatens any downstream application that assumes SAE latents provide complete feature coverage.

The practical stakes are substantial.
Google DeepMind reported that feature absorption degraded safety-relevant feature detection by 10--40%, contributing to their decision to deprioritize SAE-based interpretability research (Lieberum et al., 2024).
Conversely, Anthropic's circuit-tracing work on Claude 3.5 Haiku demonstrates that when features are reliable, they enable powerful mechanistic understanding of frontier model behavior (Lindsey et al., 2025).
Absorption sits at the boundary between these outcomes: if SAE features cannot be trusted to fire reliably on their full semantic domain, circuit analyses built on those features inherit systematic blind spots.

Every published absorption measurement rests on a single proxy task: first-letter spelling (Chanin et al., 2024; Karvonen et al., 2025).
This task maps each token to its initial letter, producing 26 classes with near-uniform distribution and 100% parent-child co-occurrence by construction.
The sae-spelling benchmark and the absorption component of SAEBench both use this task exclusively.
Architectural mitigations -- Matryoshka SAE (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), ATM (Li et al., 2025), masked regularization (Narayanaswamy et al., 2026) -- report absorption improvements on first-letter alone.
Real knowledge hierarchies differ from first-letter spelling in three structural ways: class distributions are heavily imbalanced (France has more cities than Liechtenstein), hierarchies have multiple levels of depth (city $\to$ country $\to$ continent), and parent-child co-occurrence depends on context rather than being guaranteed by construction.
Whether absorption rates measured on first-letter spelling generalize to these richer hierarchies has not been tested.

This paper presents four contributions:

- **Cross-layer absorption characterization.** First-letter absorption on Gemma 2 2B spans 2.2% to 34.5% across four model layers (6, 12, 18, 24) with eight SAE configurations.
  This measurement is unconfounded: first-letter probes achieve $\text{F1} = 0.97$ at all layers.
  Layer 24 rates (25--35%) align with the 15--35% range reported by Chanin et al. (2024), suggesting prior work likely measured at later layers; layers 6 and 18 show absorption below 5%.

- **Cross-domain absorption measurement.** Extending measurement to three entity-attribute hierarchies from the RAVEL dataset (Huang et al., 2024), we find significant variation in measured absorption rates (Kruskal-Wallis $p = 0.005$).
  At layer 24, city-continent absorption (35.8%, 16k) is comparable to first-letter (34.5%), while city-country (18.5%) and city-language (13.6%) are significantly lower ($p = 0.004$ and $p = 0.0001$, respectively).
  This comparison is confounded by differential probe quality (first-letter $\text{F1} = 0.97$ vs. RAVEL $\text{F1} = 0.79$--$0.84$), which we report transparently as a limitation.

- **Interventional causal evidence for feature suppression.** Activation patching -- zeroing the highest-attribution child latent in SAE output -- recovers correct parent-class probe predictions at 32.5% (mean across 19 words with detected absorption), compared to 1.5% for magnitude-matched control features (Wilcoxon $p = 0.0002$, Cohen's $d = 1.33$, 95\% CI on difference: $[0.21, 0.42]$).
  This provides the first interventional (not merely correlational) evidence that child latents causally suppress parent-class information in SAE encodings.

- **Tightened hedging decomposition and honest negative results.** Decomposing 304 first-letter false negatives into three categories reveals that only 7.9% exhibit strict hedging (the parent latent itself recovers at higher $L_0$), while 86.2% show compensatory resolution (unrelated latents add sufficient information) and 5.9% persist.
  The widely-cited ${\sim}98\%$ hedging rate (Chanin et al., 2025) thus reflects a near-tautological upper bound.
  We also report definitive negative results for three proposed unsupervised absorption detectors: the Geometric Absorption Score (GAS, $\rho = 0.116$), Conditional Mutual Information (CMI, $\rho = 0.044$), and the Absorption Tax ranking ($\rho = -0.20$).

Figure 1 presents the central finding visually: a heatmap of absorption rates across layers and hierarchy types, showing both the 15-fold layer variation and the significant cross-domain differences.

![Feature absorption varies by layer and hierarchy type. First-letter absorption ranges from 2.2% (L18, 16k) to 34.5% (L24, 16k). Cross-domain rates at L24 differ significantly across hierarchy types (Kruskal-Wallis p=0.005). RAVEL hierarchies measured at L24 only (best probe layer). Probe F1 annotated per hierarchy.](figures/fig1_heatmap.pdf)

Before presenting our measurements, we formalize the absorption phenomenon and review existing evaluation methodology.

<!-- FIGURES
- Figure 1: gen_fig1_heatmap.py, fig1_heatmap.pdf — Layer x hierarchy absorption heatmap showing 15x layer variation and significant cross-domain differences
-->
