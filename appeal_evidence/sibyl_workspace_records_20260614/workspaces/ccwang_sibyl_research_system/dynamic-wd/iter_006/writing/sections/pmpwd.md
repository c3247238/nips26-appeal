# 5. PMP-WD: Optimal Weight Decay from Pontryagin's Maximum Principle

## 5.1 Optimal Control Formulation

We cast WD scheduling as a continuous-time optimal control problem. The state is the parameter vector $w(t) \in \mathbb{R}^d$, the control is the WD coefficient $\lambda(t) \in [0, \Lambda_{\max}]$, and the dynamics follow:

$$\dot{w}(t) = -\nabla f(w(t)) - \lambda(t) w(t)$$

The objective is to minimize the terminal Lyapunov value:

$$J[\lambda(\cdot)] = V_T = f(w(T)) + \mu_T \|w(T)\|^2$$

subject to the dynamics and the certified band constraint $\lambda(t) \in [\lambda_{\min}(t), \lambda_{\max}(t)]$.

## 5.2 Bang-Bang Controller

**Theorem 4 (PMP-WD Optimality).** The Hamiltonian of the above system is:

$$H(w, p, \lambda) = \langle p, -\nabla f(w) - \lambda w \rangle$$

where $p(t)$ is the costate satisfying $\dot{p} = -\partial H / \partial w$. Because $H$ is linear in $\lambda$, the Pontryagin Maximum Principle yields a bang-bang optimal control:

$$\lambda^*(t) = \begin{cases} \Lambda_{\max} & \text{if } \langle p(t), w(t) \rangle > 0 \\ 0 & \text{if } \langle p(t), w(t) \rangle \leq 0 \end{cases}$$

The switching function $\sigma(t) = \langle p(t), w(t) \rangle$ determines whether weight decay is applied: when the costate aligns with the weights ($\sigma > 0$), decay reduces the Hamiltonian; when they oppose ($\sigma \leq 0$), zero decay is optimal.

**Connection to CWD.** CWD uses a per-parameter switching criterion $\text{sign}(w_i) \cdot \text{sign}(\Delta w_i)$. PMP-WD uses the global switching function $\sigma(t) = \langle p(t), w(t) \rangle$. Both implement bang-bang control; the difference is that PMP-WD integrates information across parameters via the inner product, while CWD operates parameter-wise. This structural similarity -- predicted by the theory -- explains CWD's empirical competitiveness as an approximation to the optimal controller.

## 5.3 Practical Implementation

Computing the costate $p(t)$ exactly requires solving the adjoint equation backward in time, which is impractical. We approximate $p(t)$ using the SGD momentum buffer $m_t$:

$$m_t = \beta \, m_{t-1} + g_t$$

The momentum buffer is an exponential moving average of past gradients, which approximates the costate under standard assumptions (the costate integrates future gradient information, and the momentum integrates past gradient information; for smooth losses, these are correlated). This approximation adds zero computational overhead, since $m_t$ is already maintained by SGD with momentum.

**Algorithm 1: PMP-WD**

```
Input: parameters w, learning rate gamma, base WD Lambda_max, 
       momentum coefficient beta
Initialize: m_0 = 0
For t = 1, ..., T:
    g_t = gradient of f at w_t
    m_t = beta * m_{t-1} + g_t
    sigma_t = <m_t, w_t>              # switching function
    lambda_t = Lambda_max * 1[sigma_t > 0]  # bang-bang control
    w_{t+1} = (1 - gamma_t * lambda_t) * w_t - gamma_t * g_t
```

The per-parameter switch rate (fraction of parameters where $\sigma > 0$) provides a diagnostic: a rate near 0.5 indicates balanced switching (observed in our experiments: $\approx$ 0.55 across training), while extreme rates near 0 or 1 reduce PMP-WD to no WD or constant WD respectively.

In our CIFAR-10 experiments, PMP-WD exhibits a switch count that grows linearly with training steps, and the switching function $\sigma(t)$ oscillates around zero with decreasing amplitude as the optimizer converges (see Section 6.3). The mean effective WD across training is $\approx 2.5 \times 10^{-4}$, roughly half of $\Lambda_{\max} = 5 \times 10^{-4}$, consistent with the $\approx 0.5$ BEM.

<!-- FIGURES
- None
-->
