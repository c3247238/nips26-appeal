# 6. Discussion

## 6.1 The Inhibition Graph as a Mechanistic Explanation

The LCA-SAE structural correspondence is exact, not metaphorical. For tied-weight SAEs, $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from Rozell et al.'s framework. Even with untied weights---the norm in modern SAE training---decoder correlations encode the same competitive relationships: when latent $j$ fires, its contribution to the reconstruction is $z_j \cdot W_{\text{dec}}[j]$, and the overlap $\langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ measures how much the reconstruction contributed by latent $j$ projects onto the encoder direction of latent $i$, reducing $i$'s net input.

This correspondence transforms absorption from a mysterious pathology into a predictable consequence of competitive suppression. The mechanism is causal, not merely correlational: child $j$ fires strongly, inhibits parent $i$ via $G_{ij}$, and parent $i$ fails to reach threshold. The parent's decoder direction remains unchanged, which is why steering still works and precision remains invariant.

The central empirical result supports this mechanistic claim. Precision standard deviation at layer 8 is 0.016 at $k_{\text{probe}} = 20$, while recall standard deviation is 0.167---more than 10x larger. Twenty-five of 26 first-letter features achieve precision = 1.0, while recall ranges from 0.21 to 0.49. This asymmetry is exactly what competitive suppression predicts: inhibition suppresses true positives (reducing recall) but does not introduce false positives (preserving precision).

## 6.2 Why Prior Work Found Null Results

Our prior experiments (iterations 1--8) found predominantly null correlations between absorption and downstream task performance. The inhibition framework explains why.

Raw steering metrics confound absorption-specific effects with generic directional bias. When steering adds $s \cdot W_{\text{dec}}[i]$ to the residual stream, it bypasses the encoder entirely. The decoder direction is intact regardless of absorption; what varies is the encoder activation, which steering does not depend on. This explains why raw steering success shows no correlation with absorption rate ($r = +0.008$ at layer 4, $r = -0.301$ at layer 8, both $p > 0.05$).

Delta-corrected metrics isolate the signal-specific component of steering that is lost to competitive suppression. By subtracting a random feature baseline, $\Delta S(f) = S_{\text{raw}}(f) - S_{\text{rand}}(f)$ removes the generic directional bias and reveals the absorption-specific effect. At layer 8, this yields $r = -0.431$ ($p = 0.028$ uncorrected), the strongest signal in the dataset. The inhibition framework explains why delta correction is essential: baseline subtraction isolates the encoder activation that competitive suppression removes, while the decoder direction---which steering directly uses---remains intact.

Low absorption variance in GPT-2 Small constrains statistical power. With only 4 high-absorption features at layer 8 (H: 19.0%, S: 16.0%, U: 24.2%, V: 14.7%) and $n = 26$ total features, the sample is underpowered for small-to-medium effects. The inhibition framework predicts that models with stronger hierarchical structure---deeper layers, larger models, or semantic hierarchy features---will show larger absorption variance and stronger correlations.

## 6.3 Practical Implications

The inhibition framework yields three practical recommendations for SAE practitioners.

**Training-free repair.** Homeostatic rebalancing was deferred pending improved graph construction, but the theoretical framework is sound. When the correct parent-child relationships are identified, the correction $z'_i = z_i + \alpha \sum_{j \in N(i)} G_{ij} z_j$ counteracts competitive suppression by adding back the inhibition that active neighbors exert. The single-pass nature makes it feasible for online deployment.

**Layer selection.** Deeper layers show stronger inhibition structure (mean edge weight 0.384 at layer 8 vs. 0.312 at layer 0), which correlates with stronger absorption effects. Practitioners should expect more absorption in deeper layers and apply delta-corrected metrics accordingly.

**Metric design.** The inhibition framework clarifies why delta-corrected metrics are essential for absorption research. Raw metrics confound absorption with generic steering capability; delta correction isolates the specific information loss caused by competitive suppression.

## 6.4 Relationship to Existing Solutions

The inhibition framework provides a theoretical lens for understanding existing architectural solutions.

Matryoshka SAEs (Bussmann et al., 2025) retrain with hierarchical dictionary structure. The inhibition framework explains why hierarchy helps: by organizing latents into nested subspaces, Matryoshka SAEs reduce the decoder correlations between parent and child features, thereby weakening $G_{ij}$ and reducing competitive suppression. Our approach complements Matryoshka SAEs by providing a diagnostic that works on pretrained SAEs without retraining.

OrtSAE (Korznikov et al., 2025) enforces orthogonality constraints on decoder directions. The inhibition framework explains why orthogonality helps: orthogonal directions have $G_{ij} = 0$, eliminating competitive suppression entirely. The trade-off is that strict orthogonality may limit representational capacity. Our framework quantifies this trade-off by measuring the correlation structure that orthogonality would remove.

HSAE and other hybrid architectures similarly modify the dictionary structure. The inhibition framework provides a unified language for comparing these approaches: each reduces absorption by modifying the decoder correlation structure, and the inhibition graph measures the remaining competitive dynamics.

## 6.5 Limitations of the Inhibition Framework

Five limitations constrain the current work.

**Local graph approximation.** The top-$k$ neighbor construction ($k = 20$) may miss long-range absorption relationships. A child feature that absorbs a parent may not be among the top-20 most correlated latents---the absorption relationship could involve context-dependent or non-local correlations. Testing larger $k$ values (50, 100, 1000) or hierarchical clustering would address this limitation.

**Narrow feature set.** The first-letter features (A--Z) are a specific, shallow hierarchy. Semantic hierarchies (e.g., animal $\rightarrow$ dog $\rightarrow$ poodle) may exhibit different absorption dynamics with different correlation structures. The inhibition framework predicts that deeper hierarchies will show stronger competitive suppression, but this prediction remains untested.

**Single model family.** All experiments use GPT-2 Small (124M parameters). Cross-model validation on Gemma-2-2B or Llama-3.1-8B would test whether the inhibition structure generalizes to larger models with different architectures. The pilot experiment on Gemma-2-2B (layer 8, 16K dictionary) was not completed due to resource constraints.

**Tied-weight approximation.** The exact structural correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$ assumes tied encoder-decoder weights. Modern SAEs typically use untied weights, so the correspondence holds as a first-order approximation. The gpt2-small-res-jb SAEs analyzed here use untied weights; the approximation error is small but non-zero.

**Homeostatic rebalancing deferred.** The repair experiment was not executed because the graph does not identify correct parent-child relationships with fixed $k = 20$. Future work with improved graph construction should test rebalancing and measure reconstruction error increase.

<!-- FIGURES
- None
-->
