# Theoretical Framework and AADWD Design

## Preliminaries and Problem Setup
\label{sec:prelim}

We consider stochastic gradient descent with decoupled weight decay (SGDW) for minimizing the empirical risk $f_S(w) = (1/n)\sum_{i=1}^{n} \ell(w; z_i)$ over a training set $S$. With time-varying weight decay, the update rule is:
\begin{equation}
w_{t+1} = (1 - \lambda_t)\, w_t - \gamma_t\, g_t,
\label{eq:sgdw}
\end{equation}
where $\gamma_t > 0$ is the learning rate, $\lambda_t \geq 0$ is the weight decay coefficient at step $t$, and $g_t = \nabla f_S(w_t; \xi_t)$ is a stochastic gradient computed on minibatch $\xi_t$.

We operate under standard assumptions:
\begin{assumption}[$L$-smoothness]
$f_S$ is $L$-smooth: $\|\nabla f_S(w) - \nabla f_S(w')\| \leq L\|w - w'\|$ for all $w, w'$.
\end{assumption}
\begin{assumption}[Bounded variance]
The stochastic gradient has bounded variance: $\mathbb{E}[\|g_t - \nabla f_S(w_t)\|^2] \leq \sigma^2$.
\end{assumption}
\begin{assumption}[Bounded gradient second moment]
$\mathbb{E}[\|g_t\|^2] \leq G^2$.
\end{assumption}

Following \citet{xie2024investigating}, we define the alignment quantity at step $t$:
\begin{equation}
\delta_t = \frac{|\langle \nabla f_S(w_t), w_t \rangle|}{\|\nabla f_S(w_t)\| \cdot \|w_t\|} \in [0, 1].
\label{eq:alignment}
\end{equation}
The quantity $\delta_t$ measures the cosine of the angle between the gradient direction and the parameter direction. When $\delta_t$ is small, the gradient is nearly orthogonal to the parameter vector, and weight decay (which shrinks $w_t$ toward zero) has minimal interference with the gradient descent direction. When $\delta_t$ is large, weight decay directly opposes the gradient, potentially impeding optimization.

For constant weight decay $\lambda_t = \lambda$, \citet{xie2024investigating} show that the convergence bound improves over the no-weight-decay case if and only if $\delta_T = \sup_{t \leq T} \delta_t < 1$, with the improvement proportional to $1 - \delta_T$.

## Convergence with Time-Varying Weight Decay

We extend the fixed-weight-decay analysis to the time-varying case. The key observation is that allowing $\lambda_t$ to vary enables the convergence bound to depend on a \emph{weighted average} of alignment rather than the supremum.

\begin{theorem}[Convergence with time-varying WD]
\label{thm:convergence}
Under Assumptions 1--3, suppose $\gamma_t = \gamma / \sqrt{T}$ and $\lambda_t \leq C\gamma_t^2$ for some constant $C > 0$. Then SGDW with time-varying weight decay (\ref{eq:sgdw}) satisfies:
\begin{equation}
\frac{1}{T}\sum_{t=1}^{T} \mathbb{E}\bigl[\|\nabla f_S(w_t)\|^2\bigr] \leq O\!\left(\frac{f_S(w_1) - f_S^*}{\gamma\sqrt{T}}\right) + O\!\left(\frac{L\gamma\sigma^2}{\sqrt{T}}\right) + O\!\left(\frac{\bar{\Lambda}_T G^2}{\sqrt{T}}\right),
\end{equation}
where $\bar{\Lambda}_T = (1/T)\sum_{t=1}^{T} \lambda_t / \gamma_t$ captures the average weight-decay-to-learning-rate ratio.
\end{theorem}

\emph{Proof sketch.} Define the augmented potential $\Phi_t = f_S(w_t) + \beta_t \|w_t\|^2$ with time-varying $\beta_t$ chosen to track $\lambda_t$. Expanding $\Phi_{t+1} - \Phi_t$ using $L$-smoothness and telescoping yields the result. The cross-term $\langle \nabla f_S(w_t), \lambda_t w_t \rangle$ is bounded using $|\langle \nabla f_S(w_t), w_t \rangle| \leq \delta_t \|\nabla f_S(w_t)\| \|w_t\|$ and absorbed into the potential via the condition $\lambda_t \leq C\gamma_t^2$. The full proof is in Appendix~E. \qed

\begin{theorem}[Alignment-weighted bound]
\label{thm:alignment_weighted}
Under the same conditions as Theorem~\ref{thm:convergence}, the generalization-relevant component of the convergence bound depends on the cumulative alignment:
\begin{equation}
\bar{\delta}_T = \frac{\sum_{t=1}^{T} \lambda_t \delta_t}{\sum_{t=1}^{T} \lambda_t}
\end{equation}
rather than $\delta_T = \sup_t \delta_t$. When $\delta_t$ varies significantly across training, time-varying $\lambda_t$ can improve the bound by assigning larger $\lambda_t$ to steps with smaller $\delta_t$.
\end{theorem}

\emph{Implication.} Theorem~\ref{thm:alignment_weighted} provides the theoretical motivation for AADWD: if $\delta_t$ varies substantially during training, an alignment-aware schedule can tighten the convergence bound from $\delta_T$ to $\bar{\delta}_T \leq \delta_T$. However, this improvement is only realized when the variation in $\delta_t$ is large enough to be exploited---a condition our experiments will show is \emph{not} met under standard training.

## The Three AADWD Variants
\label{sec:variants}

We construct the stochastic alignment proxy $\hat{\delta}_t$ using an exponential moving average of per-minibatch alignment measurements:
\begin{equation}
\hat{\delta}_t = \beta \cdot \hat{\delta}_{t-1} + (1 - \beta) \cdot \frac{|\langle g_t, w_t \rangle|}{(\|g_t\| \cdot \|w_t\| + \epsilon)},
\end{equation}
where $\beta \in \{0.9, 0.99, 0.999, 0.9999\}$ controls the smoothing window and $\epsilon = 10^{-8}$ prevents division by zero. We validate that $\hat{\delta}_t$ is a faithful proxy: the Pearson correlation between the minibatch EMA and large-batch alignment measurements is $r = 0.849$ on ResNet-20/CIFAR-10.

The three AADWD variants are:

\paragraph{Conservative.}
\begin{equation}
\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t \cdot (1 - \hat{\delta}_t)^p,\;\lambda_{\min},\;\lambda_{\max}\bigr), \quad p = 1.
\label{eq:conservative}
\end{equation}
\emph{Rationale:} Increase weight decay when alignment is low (gradients point away from parameters). When $\hat{\delta}_t \approx 0$, this degenerates to $\lambda_t \approx c \cdot \gamma_t$, a purely learning-rate-dependent schedule with no alignment contribution. As we will see, this is essentially what happens in practice.

\paragraph{Aggressive.}
\begin{equation}
\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t / (\hat{\delta}_t + \epsilon),\;\lambda_{\min},\;\lambda_{\max}\bigr).
\label{eq:aggressive}
\end{equation}
\emph{Rationale:} Apply weight decay inversely proportional to alignment. When $\hat{\delta}_t$ is small (misalignment), $\lambda_t$ is large; when $\hat{\delta}_t$ is large (alignment), $\lambda_t$ is small. This produces the most dynamic behavior among the three variants.

\paragraph{Square.}
\begin{equation}
\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t^2 \cdot (1 - \hat{\delta}_t),\;\lambda_{\min},\;\lambda_{\max}\bigr).
\label{eq:square}
\end{equation}
\emph{Rationale:} The $\gamma_t^2$ coupling satisfies the condition $\lambda_t = O(\gamma_t^2)$ from Theorem~\ref{thm:convergence}, providing the strongest convergence guarantee. However, under milestone schedules where $\gamma_t$ drops by $10\times$ at epochs 80 and 120, this causes $\lambda_t$ to drop by $100\times$ and $10{,}000\times$ respectively, effectively eliminating regularization in late training.

All variants share hyperparameters: scaling constant $c$, EMA coefficient $\beta$, and clipping bounds $\lambda_{\min} = 10^{-6}$, $\lambda_{\max} \in \{0.01, 0.05\}$.

## Stability Analysis of LR--WD Coupling
\label{sec:coupling}

A critical design element in all three AADWD formulas is the learning rate multiplier $\gamma_t$. We now analyze why this coupling is not merely convenient but structurally necessary.

\begin{proposition}[Decoupling instability]
\label{prop:instability}
Consider the decoupled aggressive variant: $\lambda_t = c / (\hat{\delta}_t + \epsilon)$ (removing $\gamma_t$). Under a milestone learning rate schedule where $\gamma_t$ drops by a factor of $10$ at step $t^*$, the ratio $\lambda_t / \gamma_t$ increases by $10\times$ at the milestone. If $\lambda_t / \gamma_t$ exceeds a critical threshold $\lambda^* / \gamma^*$, the weight decay term dominates the gradient term in (\ref{eq:sgdw}), triggering a positive feedback loop:
\begin{equation}
\|w_t\| \downarrow \;\to\; \hat{\delta}_t \text{ changes} \;\to\; \lambda_t \uparrow \;\to\; \|w_t\| \downarrow\downarrow \;\to\; \cdots
\end{equation}
leading to weight norm collapse ($\|w_T\| \to 0$) and training failure.
\end{proposition}

\emph{Informal argument.} In standard $\ell_2$ regularization, the effective weight decay is $\gamma_t \cdot \lambda$, which automatically scales with the learning rate. Decoupled weight decay \cite{loshchilov2019decoupled} deliberately removes this scaling for Adam---but compensates by treating $\lambda$ as a \emph{constant} that is tuned jointly with the learning rate. In AADWD, $\lambda_t$ is not constant but depends on the alignment state $\hat{\delta}_t$. Without the $\gamma_t$ damper, $\lambda_t$ can become arbitrarily large relative to the gradient signal after a learning rate drop. The learning rate multiplier serves as a natural ``speed limit'' that ensures weight decay remains commensurate with the optimization step size.

\begin{proposition}[Budget equivalence, informal]
\label{prop:budget}
Under mild conditions on the loss landscape (approximate convexity near convergence), if two weight decay schedules $\{\lambda_t^{(1)}\}_{t=1}^T$ and $\{\lambda_t^{(2)}\}_{t=1}^T$ satisfy $\sum_{t=1}^T \lambda_t^{(1)} = \sum_{t=1}^T \lambda_t^{(2)}$, their regularization effects on the final weight norm---and hence on generalization---are asymptotically equivalent.
\end{proposition}

\emph{Argument.} Near convergence, the weight norm evolves approximately as $\|w_T\|^2 \approx \|w_0\|^2 \prod_{t=1}^T (1 - 2\lambda_t) + R_T$, where $R_T$ collects gradient-dependent terms. For small $\lambda_t$, $\prod_t (1 - 2\lambda_t) \approx \exp(-2\sum_t \lambda_t)$, which depends only on the cumulative sum $\sum_t \lambda_t$. When the gradient-dependent residual $R_T$ is similar for both schedules (as it is when both schedules produce similar loss trajectories), the final weight norms---and hence the regularization effects---are determined by the weight decay budget alone.
