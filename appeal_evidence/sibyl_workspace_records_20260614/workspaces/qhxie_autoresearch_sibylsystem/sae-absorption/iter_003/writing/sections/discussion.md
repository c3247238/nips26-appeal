# 5. Discussion

## 5.1 Implications of the H2 Falsification

The OMP oracle result (Section 4.3) has a single, clean implication: the feedforward encoder is not the bottleneck for absorption. OMP with $K = 53$ is the best possible $K$-sparse encoder given the fixed SAE dictionary — if it cannot reduce absorption, no inference-time encoder improvement will. This includes all variants proposed under the amortization gap framework: learned iterative thresholding (O'Neill et al.), recurrent encoders, or attention-based encoders.

The mechanism is consistent with the sparsity landscape account \citep{tang2025partial}: absorbed features exist at partial minima of the SDL training loss. At a partial minimum, the encoder direction $\mathbf{w}_{e,j}$ has been displaced from the decoder direction $\mathbf{d}_j$ by persistent gradient competition during training. This displacement is a property of the trained weight matrices, not of the inference procedure. The only interventions that can escape this attractor are ones that alter the loss landscape during training: masked regularization (suppressing the competing gradient), hierarchically-aware objectives (encouraging parent-child co-allocation), or structural changes to the dictionary (wider dictionaries, Matryoshka nesting).

**Practical guidance.** For SAE practitioners seeking to reduce absorption: (1) Do not expect iterative solvers or improved inference-time encoders to help. (2) Focus evaluation budget on training-time interventions. (3) Use EncNorm as a fast first-pass screen — latents with elevated EncNorm are candidates for targeted regularization during training.

## 5.2 Limitations

**Small positive-class sample.** Gold IG labels provide $n_\text{pos} = 18$ absorbed features at GPT-2 L6. Bootstrap CIs span approximately $0.13$ in AUROC; the DeLong test result ($p = 0.0012$) should be interpreted cautiously at this sample size. Three-context replication (Standard/L1, TopK-32k with proxy labels, layer monotonicity analysis) provides partial mitigation, but a larger labeled dataset from the full GPT-2 or Gemma Scope vocabulary would substantially strengthen the detection claims.

**Hook confound in cross-architecture comparison.** The Standard SAE hooks into \texttt{blocks.6.hook\_resid\_pre} (before attention) while the TopK-32k SAE hooks into \texttt{blocks.6.hook\_resid\_post} (after attention). These are different activation spaces with different representational geometry. The observed AUROC difference ($0.757$ vs.\ $0.837$) cannot be attributed to architecture (L1 vs.\ TopK) without first controlling for hook point. A matched experiment using two SAEs at the same hook would resolve this confound.

**H3 entity-type probe failure.** The cross-hierarchy absorption experiment (D2) reports near-zero absorption rates for all entity-type hierarchies (city-continent, city-country, city-language). This is almost certainly a probe quality artifact: probes were trained on Qwen2.5-0.5B and projected to GPT-2 activation space via random QR decomposition. Such projections do not preserve probe direction quality. Whether absorption generalizes beyond the first-letter spelling task remains an open question requiring probes trained directly on the target model.

**OMP fixed-dictionary assumption.** The OMP oracle fixes the pre-trained SAE decoder and only varies the encoder. A stronger version of the amortization gap hypothesis might posit that both encoder and decoder are jointly sub-optimal, and that re-training both with an iterative encoder would reduce absorption. Our experiment cannot rule out this joint hypothesis. We view this as an important target for future work.

## 5.3 Implications for SAE Design

The F1 result (67\% recovery, 33\% non-recovery in wider SAE) provides the clearest practical guidance: expanding dictionary width helps for features whose parent was simply absent from the narrower dictionary (the "early absorption" case identified by \citeauthor{eda2025}), but does not help for features whose parent exists in the dictionary but has been suppressed by training dynamics ("late absorption"). A diagnostic tool that distinguishes these two cases — for instance, the taxonomy in \citeauthor{eda2025} combined with EncNorm — would enable practitioners to apply targeted interventions.

Masked regularization \citep{narayanaswamy2026masked} directly addresses the gradient competition mechanism: by masking the sparsity gradient from high-frequency child features on their frequent co-occurring partners, it prevents the training-time attractor formation. Our H2 result provides independent theoretical support for this approach and motivates its extension to arbitrary feature co-occurrence patterns.
