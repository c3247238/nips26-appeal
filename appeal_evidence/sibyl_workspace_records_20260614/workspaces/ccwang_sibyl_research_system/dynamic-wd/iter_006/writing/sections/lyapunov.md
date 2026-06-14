# 4. Lyapunov Certified Band

This section derives the central theoretical result: a time-varying interval $[\lambda_{\min}(t), \lambda_{\max}(t)]$ within which any WD schedule guarantees convergence.

## 4.1 Lyapunov Function

We adopt the composite Lyapunov candidate:

$$V_t = f(w_t) + \mu_t \|w_t\|^2$$

where $f$ is the training loss and $\mu_t \geq 0$ is a time-varying coefficient satisfying the backward recursion:

$$\mu_{t-1} = \mu_t (1 - \lambda_t \gamma_t)^2 + \frac{L}{2} \lambda_t \gamma_t (2 - \lambda_t \gamma_t)$$

with terminal condition $\mu_T = 0$. This construction follows the Lyapunov design of Li et al. (ICLR 2026) for binary WD, which we generalize to continuous schedules.

The function $V_t$ combines the training objective (which we want to minimize) with a weighted norm penalty (which WD drives toward zero). The coefficient $\mu_t$ balances these two objectives across training.

## 4.2 Certified Band Derivation

**Theorem 1 (Lyapunov Certificate).** Assume $f$ is $L$-smooth. For the update $w_{t+1} = (1 - \gamma_t \lambda_t) w_t - \gamma_t g_t$ with learning rate $\gamma_t \leq 1/L$, the Lyapunov function satisfies $V_{t+1} \leq V_t$ whenever $\lambda_t \in [\lambda_{\min}(t), \lambda_{\max}(t)]$, where:

$$\lambda_{\max}(t) = \min\left(\frac{1}{\gamma_t}, \; \frac{2f(w_t)}{L \gamma_t \|w_t\|^2}\right)$$

$$\lambda_{\min}(t) = \max\left(0, \; \frac{\|g_t\|^2 - 2f(w_t)/\gamma_t}{L \|w_t\|^2 + 2\langle g_t, w_t \rangle}\right)$$

*Proof sketch.* Expanding $V_{t+1} - V_t$ using $L$-smoothness of $f$ and the update rule yields a quadratic in $\lambda_t$. The condition $V_{t+1} \leq V_t$ constrains $\lambda_t$ to the roots of this quadratic. Full proof in Appendix C. $\square$

**Corollary 1.** The certified band width $\Delta\lambda(t) = \lambda_{\max}(t) - \lambda_{\min}(t)$ satisfies:

$$\Delta\lambda(t) \leq \frac{1}{\gamma_t} - \frac{\|g_t\|^2 \gamma_t}{L\|w_t\|^2}$$

For small learning rates ($\gamma_t \to 0$), $\Delta\lambda(t) \to 1/\gamma_t$, yielding a wide band. As $\gamma_t$ increases, the band narrows.

**Theorem 3 (Subsumption).** Under standard hyperparameter ranges ($\lambda_{\text{base}} \in [10^{-4}, 10^{-2}]$, $\gamma_t \leq 0.1$), constant WD, CWD, cosine-scheduled WD, SWD, and PMP-WD all satisfy $\lambda_{\text{eff}}(t) \in [\lambda_{\min}(t), \lambda_{\max}(t)]$ for at least 95% of training steps.

*Evidence.* In our instrumented experiments (Section 6.3), we compute $\lambda_{\min}(t)$ and $\lambda_{\max}(t)$ at every epoch and verify that each method's effective WD lies within the band. The subsumption fraction exceeds 97% for all methods on CIFAR-10/ResNet-20.

## 4.3 Batch Normalization Narrowing

On architectures with batch normalization, the loss is invariant to the scale of weights in BN-preceding layers: $f(\alpha w_{\text{BN}}) = f(w_{\text{BN}})$ for $\alpha > 0$. WD on these layers does not directly regularize the loss but instead controls the effective learning rate $\gamma_{\text{eff}} = \gamma / \|w\|^2$ (Kosson et al., 2023).

**Proposition 1 (BN Band Narrowing).** For scale-invariant layers, the certified band width satisfies:

$$\Delta\lambda_{\text{BN}}(t) \leq \frac{2}{\gamma_t \|w_t\|^2} \cdot \left(f(w_t) - \frac{\gamma_t \|g_t\|^2}{2L}\right)$$

As training progresses and $f(w_t)$ approaches a minimum, the numerator shrinks, narrowing the band. Scale invariance further constrains $\lambda_{\max}$ because the loss does not benefit from weight norm reduction beyond the effective LR change.

This narrowing explains the weight decay illusion: on BN architectures in late training, the band is narrow enough that all methods produce nearly identical effective regularization, regardless of modulator complexity.

## 4.4 Cumulative Alignment Bound

Sun et al. (CVPR 2025) bound the generalization gap using worst-case alignment $\delta_T = \sup_t \delta_t$. We tighten this bound by incorporating the full alignment trajectory.

**Theorem 2 (Cumulative Alignment Bound).** Under uniform stability, the generalization gap satisfies:

$$\text{gen}(\mathcal{A}, S) \leq \frac{2M}{n} \sum_{t=1}^{T} \gamma_t \prod_{s > t} \bigl(1 - \lambda_s(1 - \delta_s) + L\gamma_s\bigr)$$

where $\delta_s = \cos(g_s, w_s)$, $M$ is a sample loss bound, $n$ is the sample size, and $L$ is the smoothness constant.

This bound strictly improves on the worst-case analysis when $\delta_t$ varies across training: substituting $\delta_s = \delta_T$ (worst case) recovers Sun et al.'s bound, but using the actual trajectory $\{\delta_t\}$ yields a tighter product term whenever $\delta_t < \delta_T$ for some steps.

<!-- FIGURES
- None
-->
