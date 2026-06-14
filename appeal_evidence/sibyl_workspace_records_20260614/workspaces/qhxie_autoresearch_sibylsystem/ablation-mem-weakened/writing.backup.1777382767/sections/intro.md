# 1. Introduction

## 1.1 The SAE Credibility Crisis

Sparse autoencoders (SAEs) have become the dominant paradigm for mechanistic interpretability, enabling circuit analysis, feature steering, model editing, and bias detection (Marks et al., 2024; Templeton et al., 2024). The foundational premise is that SAEs decompose neural network activations into human-interpretable features through sparse dictionary learning. Yet the field faces an escalating credibility crisis.

Korznikov et al. (2026) demonstrate that SAEs recover only 9% of true features despite 71% explained variance, and that random baseline SAEs match trained SAEs on standard metrics. Some research groups have reportedly deprioritized SAE research after finding negative results on downstream tasks. These developments raise a fundamental question: do SAEs provide reliable tools for interpretability work, or do they create an illusion of understanding?

## 1.2 Feature Absorption: The Gap Between Detection and Impact

At the center of this crisis is feature absorption, first formally identified by Chanin et al. (2024). Absorption occurs when a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. For example, a "starts with A" feature may be absorbed by "starts with Apple" or "starts with Ant" features. The parent latent appears interpretable when inspected in isolation but produces systematic false negatives during downstream use.

Chanin et al. demonstrated that hierarchical features cause absorption and validated the phenomenon across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families. SAEBench (Karvonen et al., 2025) subsequently standardized absorption as a benchmark metric alongside sparsity, reconstruction error, and explained variance. Architectural innovations---Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2026), and HSAE (Luo et al., 2026)---all target absorption reduction as a primary objective.

Despite this attention, a critical gap remains: **no existing work quantifies whether absorption degrades the interpretability tasks that motivate SAE research.** Researchers use SAEs for steering, circuit finding, and model editing. If absorbed features are systematically unreliable, these applications may produce false negatives or misleading results. Yet the field has optimized absorption metrics without establishing their relationship to task performance. This study bridges that gap.

## 1.3 Our Contribution

We provide the first systematic study connecting feature absorption detection to downstream task performance, specifically steering effectiveness and sparse probing accuracy. Using pre-trained SAEs from GPT-2 Small (124M parameters, 85M non-embedding; gpt2-small-res-jb, 24,576 latents), we measure absorption rates across layers 0, 4, 8, and 10 for 26 first-letter features (A--Z), then test steering effectiveness and sparse probing accuracy.

Our methodology is entirely training-free, making it accessible to any researcher with a GPU and open-source tools. We test four hypotheses:

- **H1 (Raw steering)**: Higher absorption rate leads to lower raw steering success rate (directional prediction: negative Pearson correlation, $p < 0.05$).
- **H1b (Delta steering)**: Higher absorption rate leads to lower delta steering effectiveness, where delta is feature-specific success minus random baseline success (directional prediction: negative Pearson correlation, $p < 0.05$).
- **H2 (Probing)**: Higher absorption rate leads to lower sparse probing F1 (directional prediction: negative Pearson correlation, $p < 0.05$).
- **H3 (Consistency)**: The degradation relationship is consistent across layers (regression slopes have the same sign and similar magnitude, $\text{CV} < 0.5$ where $\text{CV} = \sigma / |\mu|$).

The random baseline control in H1b is critical. Random feature steering achieves 34--38% success on our task, indicating that arbitrary decoder directions produce non-negligible effects. Without subtracting this baseline, raw steering success conflates feature-specific contribution with generic directional bias.

Our results are mixed. Raw steering success shows no significant correlation with absorption (Pearson $r = +0.008$, $p = 0.970$ at layer 4; $r = -0.301$, $p = 0.136$ at layer 8). Sparse probing shows no correlation at either layer ($r = -0.003$, $p = 0.987$ at layer 4; $r = -0.107$, $p = 0.604$ at layer 8). However, delta-corrected steering reveals a significant negative correlation at layer 8 (Pearson $r = -0.431$, $p = 0.028$; Spearman $\rho = -0.502$, $p = 0.009$). The relationship is inconsistent across layers (H3 not supported). These findings suggest that absorption, as measured by the Chanin et al. differential correlation method, does degrade steering effectiveness when properly measured---but the effect is layer-dependent and not detectable in raw steering metrics or in probing.

## 1.4 Paper Structure

We first establish that absorption detection and downstream task evaluation have proceeded in isolation (Section 2), then formalize the hypotheses that would confirm absorption as a critical failure mode (Section 3). Our training-free methodology (Section 4) and mixed results (Sections 5--6) challenge the assumption that absorption is uniformly harmful, leading to actionable guidance for the field (Section 8).

<!-- FIGURES
- None
-->
