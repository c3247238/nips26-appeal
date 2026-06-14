# Method

## Preliminaries

We consider the standard decoupled weight decay (SGDW) update rule \cite{loshchilov2019adamw}:
$$w_{t+1} = (1 - \lambda \gamma)\, w_t - \gamma\, g_t$$
where $w_t$ denotes the parameters at step $t$, $g_t$ is the (mini-batch) gradient, $\lambda$ is the weight decay coefficient, and $\gamma$ is the learning rate. We define the **gradient-to-weight ratio** for layer $l$ as:
$$r_t^l = \frac{\|g_t^l\|}{\|w_t^l\| + \varepsilon}$$
where $\varepsilon = 10^{-8}$ ensures numerical stability. This ratio measures the relative magnitude of the gradient signal to the current weight scale.

Defazio \cite{defazio2025} established that, for normalized layers under SGDW with constant learning rate and stationary gradient statistics, the ratio $r_t$ converges to a universal steady-state:
$$r^* = \frac{\lambda}{\gamma}$$
This equilibrium arises because weight decay shrinks $\|w_t\|$ while gradient updates grow it; when the two forces balance, the ratio stabilizes. The key insight is that $r^*$ characterizes the *norm-balanced operating regime* of the optimizer---the point where regularization and learning forces are in equilibrium. Deviations from this equilibrium indicate transitional phases where the optimization trajectory is far from its stable operating point.

**Scope of the equilibrium analysis.** The convergence $r_t \to \lambda / \gamma$ holds rigorously under three conditions: (a) normalized layers, (b) constant learning rate, and (c) approximately stationary gradient statistics. In practice, all three assumptions are violated to varying degrees: cosine learning rate schedules make $\gamma$ time-varying, batch normalization creates scale invariance that alters ratio dynamics, and data augmentation induces non-stationary gradient distributions. Rather than using the theoretical closed-form $\lambda/\gamma$, we therefore track the empirical quasi-static equilibrium via an exponential moving average, which adapts to these non-ideal conditions.

## EqWD: Equilibrium-Driven Weight Decay

We propose to modulate weight decay based on how far the current ratio deviates from its equilibrium. The EqWD algorithm operates per layer and per training step:

\begin{algorithm}[H]
\caption{EqWD: Equilibrium-Driven Weight Decay}
\begin{algorithmic}[1]
\REQUIRE Weights $w_t^l$, gradients $g_t^l$, base WD $\lambda_{\text{base}}$, EMA decay $\alpha$, sensitivity $\beta$, learning rate $\gamma$
\STATE \textbf{Initialize} $r^{*,l} \leftarrow r_0^l$ from first batch \hfill \COMMENT{EMA initialization}
\STATE $r_t^l \leftarrow \|g_t^l\| \,/\, (\|w_t^l\| + \varepsilon)$ \hfill \COMMENT{Compute ratio}
\STATE $r^{*,l} \leftarrow \alpha \cdot r^{*,l} + (1 - \alpha) \cdot r_t^l$ \hfill \COMMENT{Update EMA equilibrium}
\STATE $\text{dev}_t^l \leftarrow \min\bigl(|r_t^l - r^{*,l}| \,/\, (r^{*,l} + \varepsilon),\; \delta_{\max}\bigr)$ \hfill \COMMENT{Clamped normalized deviation}
\STATE $\lambda_t^l \leftarrow \lambda_{\text{base}} \cdot (1 + \beta \cdot \text{dev}_t^l)$ \hfill \COMMENT{Modulate WD}
\STATE $w_{t+1}^l \leftarrow (1 - \lambda_t^l \cdot \gamma) \cdot w_t^l - \gamma \cdot g_t^l$ \hfill \COMMENT{SGDW update}
\end{algorithmic}
\end{algorithm}

The algorithm introduces a single modulation factor:
$$\varphi_l(t) = 1 + \beta \cdot \frac{|r_t^l - r^{*,l}|}{r^{*,l} + \varepsilon}$$
which multiplies the base weight decay coefficient. The effective decay at each step and layer is $\lambda_t^l = \lambda_{\text{base}} \cdot \varphi_l(t)$. The subscript $l$ and argument $t$ emphasize that modulation is both per-layer and per-step; when the context is clear, we suppress layer indices for readability.

**Design rationale.** Each component of EqWD is motivated by a specific consideration:

*EMA equilibrium tracking ($r^{*,l}$).* Rather than using the theoretical steady-state $\lambda / \gamma$, we track the equilibrium empirically via an exponential moving average with decay $\alpha = 0.9$. This has two advantages: (i) it accounts for the fact that real training dynamics include batch normalization, data augmentation, and other factors not captured by the idealized analysis; and (ii) it automatically adapts to learning rate schedules, since $r^{*,l}$ tracks the quasi-static equilibrium as $\gamma$ changes. With $\alpha = 0.9$, the effective time constant is $1/(1-\alpha) = 10$ steps, meaning the EMA requires approximately 30 steps to converge to its local equilibrium. During these initial steps, modulation is driven by the EMA's transient behavior rather than a stable reference, but we find this has negligible effect on final performance.

*Normalized deviation.* We normalize the deviation by $r^{*,l}$ to ensure scale invariance across layers. Without normalization, layers with large absolute $r^*$ (typically later layers) would dominate the modulation signal, while early layers with small $r^*$ would be ignored.

*Additive form $(1 + \beta \cdot \text{dev})$.* The additive structure ensures that EqWD recovers fixed weight decay when $\beta = 0$, providing backward compatibility. The modulation is always $\geq 1$, meaning EqWD only *increases* weight decay relative to the baseline---it amplifies regularization during instability but never reduces it below the base level. This means the effective average weight decay over training is systematically higher for EqWD than for FixedWD with the same $\lambda_{\text{base}}$. We address this effective strength confound in Section 5.

*Per-layer granularity.* Different layers exhibit markedly different ratio dynamics. Early convolutional layers in ResNet-50 typically have small, stable $r_t^l$ values, while later layers show higher variance. Per-layer modulation allows EqWD to respond to these heterogeneous dynamics independently.

**Connections to existing methods.** EqWD relates to several prior methods through conceptual analogies:
- When $\beta = 0$, EqWD reduces to standard fixed weight decay (SGDW).
- A threshold variant using a binary indicator $\mathbb{1}[r_t^l > r^{*,l}]$ instead of continuous deviation would yield an approach conceptually analogous to CWD's masking strategy, though CWD operates per-parameter with sign alignment rather than per-layer with ratio deviation.
- Using $\|g_t\|$ alone as the modulation signal, rather than the ratio deviation, bears a conceptual resemblance to SWD's gradient-norm schedule, though SWD operates globally and inverts the modulation direction (reducing WD for large gradients). These are structural analogies, not formal reductions.
- Unlike CPR, which enforces a fixed norm target via a smooth augmented Lagrangian, EqWD uses the ratio as a dynamic signal without imposing norm constraints.

## Theoretical Analysis

We provide a theoretical justification for why ratio deviation is a principled modulation signal, connecting EqWD to the alignment-based generalization framework of Sun et al. \cite{sun2025cvpr}.

**Proposition 1** (Equilibrium recovery). *In the equilibrium phase where $r_t^l \approx r^{*,l}$, EqWD recovers fixed weight decay behavior: $\lambda_t^l \approx \lambda_{\text{base}}$. In transitional phases where $r_t^l \gg r^{*,l}$ or $r_t^l \ll r^{*,l}$, EqWD increases weight decay proportionally to the normalized deviation, providing stronger regularization when the optimization trajectory is furthest from the norm-balanced regime.*

*Proof sketch.* This follows directly from the definition of $\varphi_l(t)$. When $|r_t^l - r^{*,l}| / r^{*,l} \approx 0$, we have $\varphi_l(t) \approx 1$ and $\lambda_t^l \approx \lambda_{\text{base}}$. For a deviation of magnitude $\delta = |r_t^l - r^{*,l}| / r^{*,l}$, the effective weight decay is $\lambda_{\text{base}} \cdot (1 + \beta \delta)$, which is monotonically increasing in $\delta$. The deviation is clamped at $\delta_{\max} = 10$, bounding the maximum effective weight decay at $\lambda_{\text{base}} \cdot (1 + \beta \cdot \delta_{\max})$. For the default $\beta = 1.0$, this yields $11\lambda_{\text{base}}$. $\square$

**Proposition 2** (Ratio sufficiency). *Suppose the training dynamics satisfy the conditions under which ratio deviation $|r_t^l - r^{*,l}| / r^{*,l}$ and alignment deviation $|\alpha_t^l - \bar{\alpha}^l|$ (where $\alpha_t^l = \langle g_t^l, w_t^l \rangle / (\|g_t^l\| \cdot \|w_t^l\|)$) are both functions of the gradient and weight norms $(\|g_t^l\|, \|w_t^l\|)$. Then the ratio $r_t^l = \|g_t^l\| / \|w_t^l\|$ is a sufficient statistic for the alignment-relevant information, and modulating weight decay based on ratio deviation captures the same generalization-relevant signal as alignment-based modulation.*

*Proof sketch.* By definition, $r_t^l = \|g_t^l\| / \|w_t^l\|$ is a deterministic function of the two norms. If alignment deviation is also a function of these norms (i.e., $|\alpha_t^l - \bar{\alpha}^l| = f(\|g_t^l\|, \|w_t^l\|)$ for some function $f$), then $r_t^l$ and $\alpha_t^l$ carry the same information about the optimization state, and the ratio is sufficient for the alignment-relevant dynamics. The key question is whether this norm-sufficiency condition holds in practice. $\square$

**Empirical verification of Proposition 2.** The norm-sufficiency condition is an empirical question. Our Alignment Informativeness Score (AIS) diagnostic (Appendix F) directly tests this by measuring $\text{MI}(\hat{\delta}_t; \text{test\_acc} \mid \|g_t\|, \|w_t\|)$---the mutual information between cosine alignment deviation and test accuracy, conditioned on gradient and weight norms. Across all convolutional layers for CIFAR-100/ResNet-20 and VGG-16-BN, AIS is near zero (residual variance ratio $> 0.95$ everywhere). This means that, in the settings we evaluate, the cosine alignment signal carries no incremental information beyond what the norms already capture. The ratio $r_t^l = \|g_t^l\| / \|w_t^l\|$ is therefore empirically sufficient for the alignment-relevant modulation signal.

**Implications.** The combination of Proposition 2 and the AIS diagnostic provides a coherent justification for EqWD: the ratio deviation is not merely a proxy for alignment deviation (which would be a weaker claim), but rather a *sufficient* modulation signal that subsumes the alignment information. This makes EqWD's ratio-based modulation both computationally cheaper than explicit alignment computation and, in the settings tested, informationally complete.

**Scope and caveats.** The norm-sufficiency result is empirical and may not hold universally. In architectures with LayerNorm (e.g., Transformers), where normalization alters the relationship between norms and alignment, the ratio may no longer capture all alignment-relevant information. Extending the AIS diagnostic to Transformer architectures is an important direction for future work.

## Implementation Notes

EqWD is straightforward to implement and integrate with existing optimizers. The core logic requires computing two norms ($\|g_t^l\|$, $\|w_t^l\|$) per parameter group per step---operations that are already efficiently supported in modern deep learning frameworks and add approximately 2\% wall-clock overhead in our measurements.

**Initialization.** The EMA equilibrium $r^{*,l}$ is initialized to $r_0^l$ from the first training batch, as shown in Algorithm 1 (line 1). The first $\sim$30 steps serve as an implicit warm-up period during which the EMA converges to its local equilibrium.

**Numerical stability.** We use $\varepsilon = 10^{-8}$ in all divisions and clamp the normalized deviation to $[0, \delta_{\text{max}}]$ with $\delta_{\text{max}} = 10$ to prevent extreme modulation from very large transient deviations.

**Optimizer compatibility.** EqWD modifies only the weight decay coefficient and is compatible with any optimizer that supports decoupled weight decay, including SGD and AdamW. It can be implemented as a wrapper or callback that adjusts $\lambda$ before each parameter update. Our experiments in this work use SGDW; extension to AdamW, where second-moment scaling alters the effective gradient and may change the ratio dynamics, requires separate validation and is left for future work (see Section 5.6).

**Hyperparameters.** EqWD introduces two hyperparameters beyond the base weight decay $\lambda_{\text{base}}$: the sensitivity $\beta$ (default 1.0) and the EMA decay $\alpha$ (default 0.9). Our ablation studies (Section 4.3) show that the defaults are robust across datasets and architectures, and that $\beta \in [0.5, 2.0]$ consistently outperforms fixed weight decay.
