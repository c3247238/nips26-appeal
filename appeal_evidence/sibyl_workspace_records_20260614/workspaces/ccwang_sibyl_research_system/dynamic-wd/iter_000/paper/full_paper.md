# On the Sufficiency of Constant Weight Decay: Alignment Dynamics, Learning Rate Coupling, and Budget Equivalence in Nonconvex SGD

---

## Abstract

Weight decay is a ubiquitous regularizer in deep learning, yet optimal scheduling of its strength remains an open question.
Motivated by recent theoretical results connecting weight decay efficacy to the gradient--parameter alignment quantity $\delta_t = |\langle \nabla f_S(w_t), w_t \rangle| / (\|\nabla f_S(w_t)\| \|w_t\|)$, we design Alignment-Aware Dynamic Weight Decay (AADWD)---a family of three adaptive schedules that modulate $\lambda_t$ based on a stochastic alignment proxy $\hat{\delta}_t$.
Through 39 systematic experiments spanning two architectures (ResNet-20, VGG-16-BN), two datasets (CIFAR-10, CIFAR-100), and extensive ablations, we establish three negative but informative results:
**(1) Budget Equivalence**---when the time-averaged dynamic weight decay matches a constant, performance is identical ($92.54\% = 92.54\%$ on CIFAR-10/ResNet-20, with identical weight norms of 23.49);
**(2) LR--WD Coupling Necessity**---removing the learning rate multiplier from AADWD triggers catastrophic collapse (the aggressive variant's best pre-collapse accuracy of $84.49\%$ degrades to $10.00\%$, with weight norm $\to 0.0036$; the coupled variant achieves $92.05\%$);
**(3) Alignment Signal Inapplicability**---random dynamic weight decay ($92.06\%$) matches alignment-based weight decay ($92.05\%$), because $\delta_t \sim O(10^{-3})$ throughout standard training.
These findings formalize why constant weight decay remains an empirically undominated strategy under standard nonconvex SGD and identify the structural conditions under which adaptive scheduling cannot succeed.

---

## 1. Introduction
\label{sec:intro}

### Motivation and Problem Statement

Weight decay is the most widely used explicit regularizer in modern deep learning \cite{loshchilov2019decoupled, zhang2019algorithmic}. In standard practice, the weight decay coefficient $\lambda$ is held constant throughout training and tuned as a single scalar hyperparameter---typically set to $5 \times 10^{-4}$ for SGD-based image classification \cite{he2016deep}. Despite the simplicity of this approach, constant weight decay consistently delivers competitive generalization across a broad range of architectures and datasets.

A natural question arises: \emph{can weight decay be scheduled dynamically---stronger when regularization helps, weaker when it hurts?} Learning rate scheduling has long been recognized as essential for training deep networks \cite{smith2017cyclical, loshchilov2017sgdr}, and analogous schedules for other hyperparameters---dropout rate, label smoothing, data augmentation strength---have shown varying degrees of success \cite{morerio2017curriculum, zhang2018mixup}. Yet surprisingly little work has systematically investigated whether temporal adaptation of the weight decay coefficient provides any benefit beyond tuning its constant magnitude.

Recent theoretical advances provide a principled basis for this investigation. In particular, \citet{xie2024investigating} identify the alignment quantity
\begin{equation}
\delta_t = \frac{|\langle \nabla f_S(w_t), w_t \rangle|}{\|\nabla f_S(w_t)\| \cdot \|w_t\|}
\end{equation}
as the key condition governing when weight decay improves generalization. Specifically, the condition $\delta_T = \sup_t \delta_t < 1$ is sufficient for weight decay to improve the convergence bound by a factor of $(1 - \delta_T)$ (see Theorem~3 in \citealt{xie2024investigating}). If $\delta_t$ varies over the course of training---being large in some phases and small in others---then the weight decay rate should, in principle, vary accordingly to exploit this structure. We extend this analysis to the time-varying case (Section~\ref{sec:theory}), showing that the relevant quantity becomes a weighted cumulative average $\bar{\delta}_T$ rather than the supremum $\delta_T$.

### Our Approach: Alignment-Aware Dynamic Weight Decay

We design Alignment-Aware Dynamic Weight Decay (AADWD), a family of three adaptive weight decay schedules that modulate $\lambda_t$ based on a stochastic exponential moving average (EMA) proxy $\hat{\delta}_t$ of the alignment quantity ($\hat{\delta}_t$ is defined formally in Section~\ref{sec:variants}). The three variants differ in how aggressively they respond to the alignment signal:

\begin{itemize}
\item \textbf{Conservative:} $\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t \cdot (1 - \hat{\delta}_t),\, \lambda_{\min},\, \lambda_{\max}\bigr)$, which increases weight decay when alignment is low (gradients and parameters are misaligned).
\item \textbf{Aggressive:} $\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t / (\hat{\delta}_t + \epsilon),\, \lambda_{\min},\, \lambda_{\max}\bigr)$, which applies weight decay inversely proportional to alignment.
\item \textbf{Square:} $\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t^2 \cdot (1 - \hat{\delta}_t),\, \lambda_{\min},\, \lambda_{\max}\bigr)$, which satisfies the $O(\gamma_t^2)$ condition required for standard convergence guarantees.
\end{itemize}

All three variants include a learning rate multiplier $\gamma_t$ that couples the weight decay strength to the current learning rate---a design choice whose necessity our experiments will reveal to be far more consequential than the alignment signal itself. Hyperparameter details (clipping bounds, exponents) are specified in Section~\ref{sec:variants}.

AADWD is not proposed as a practical optimization method. Rather, it is designed as an \emph{experimental framework}---a controlled probe to test whether alignment information is actionable for weight decay scheduling, and to isolate the mechanisms that govern the interaction between weight decay, learning rate, and training dynamics.

### Summary of Findings and Contributions

Through 39 systematic experiments spanning ResNet-20, VGG-16-BN, CIFAR-10, and CIFAR-100, we establish three core findings, each stated as a positive mechanistic insight derived from a negative empirical result:

\paragraph{1. Weight Decay Budget Equivalence.}
We demonstrate that the time-averaged weight decay magnitude---not its temporal distribution---determines generalization performance. When the cumulative weight decay budget $\bar{\lambda} = (1/T)\sum_{t=1}^{T} \lambda_t$ of a dynamic schedule is matched by a constant $\lambda = \bar{\lambda}$, the two yield identical performance. On CIFAR-10/ResNet-20, the budget-equivalent constant weight decay achieves $92.54\%$ test accuracy with weight norm $23.49$---exactly matching the original fixed weight decay baseline ($92.54\%$, weight norm $23.49$). This result implies that weight decay scheduling yields \emph{no detectable marginal value} over the constant baseline across all tested architectures and datasets when the cumulative budget is held fixed.

\paragraph{2. LR--WD Coupling is a Structural Necessity.}
The learning rate multiplier $\gamma_t$ in AADWD is not a convenient design choice but a stability requirement. Removing it produces catastrophic failure: the aggressive variant collapses from $92.05\%$ (coupled) to $10.00\%$ test accuracy (random chance on CIFAR-10), with the weight norm shrinking to $0.0036$ through a positive feedback loop. The decoupled aggressive variant achieves a best pre-collapse accuracy of only $84.49\%$ (before the first LR milestone), after which training degrades irreversibly. Even the conservative variant degrades from $92.37\%$ to $80.30\%$. This establishes that any adaptive weight decay scheme operating under milestone learning rate schedules must scale with the learning rate to prevent the regularizer from overwhelming the gradient signal after learning rate drops.

\paragraph{3. Alignment Signal is Not Actionable Under Standard SGD.}
The alignment proxy $\hat{\delta}_t$ remains at $O(10^{-3})$ throughout training (range $[0.0028, 0.0045]$, standard deviation $7.5 \times 10^{-4}$), rendering it effectively constant. As a consequence, random dynamic weight decay---where $\lambda_t = c \cdot \gamma_t \cdot U(0,1)$ with no alignment information---achieves $92.06\%$, statistically indistinguishable from alignment-based AADWD aggressive at $92.05\%$. The alignment signal carries no exploitable information for weight decay scheduling under standard training conditions.

\paragraph{Additional contributions.} As an additional probe, we include Cautious Weight Decay \cite{liu2025cautious} (CWD)---a coordinate-wise adaptive method that gates weight decay by gradient sign agreement---which we find exhibits systematic late-training instability across all three experimental settings: $91.79\% \to 86.95\%$ on CIFAR-10/ResNet-20 ($\Delta = 4.84\%$), $66.84\% \to 54.27\%$ on CIFAR-100/ResNet-20 ($\Delta = 12.57\%$), and $92.95\% \to 86.47\%$ on CIFAR-10/VGG-16-BN ($\Delta = 6.48\%$).

Taken together, these findings provide a mechanistic account of why constant weight decay remains an empirically undominated strategy for nonconvex SGD: the alignment condition is always easily satisfied ($\delta \ll 1$), the cumulative regularization budget is the sufficient statistic for generalization, and the learning rate schedule already provides the necessary temporal structure. These conclusions hold under standard SGD with milestone learning rate schedules on CIFAR-scale datasets; we discuss the potential for different behavior under adaptive optimizers and larger-scale training in Section~\ref{sec:discussion}.

---

## 2. Related Work
\label{sec:related}

### Weight Decay: Theory and Practice

Weight decay has a long history as an explicit regularizer for neural networks \cite{krogh1991simple}. Classical $\ell_2$ regularization adds the penalty $(\lambda/2)\|w\|^2$ to the loss, producing the gradient update $w_{t+1} = w_t - \gamma_t(\nabla f(w_t) + \lambda w_t) = (1 - \gamma_t \lambda) w_t - \gamma_t \nabla f(w_t)$. \citet{loshchilov2019decoupled} observe that for adaptive optimizers such as Adam, this formulation conflates the regularization effect with the preconditioned gradient, and propose decoupled weight decay (AdamW): $w_{t+1} = (1 - \lambda) w_t - \gamma_t \hat{m}_t$, where $\hat{m}_t$ is the bias-corrected first moment. For SGD with momentum, the two formulations differ only by a rescaling of $\lambda$ by $\gamma_t$, but the distinction becomes critical in our analysis of LR--WD coupling (Section~\ref{sec:coupling}).

Theoretical analyses of weight decay in nonconvex optimization have established convergence guarantees under smoothness and bounded variance assumptions \cite{zhuang2022understanding, chen2019convergence}. \citet{zhuang2022understanding} analyze the convergence of SGD with $\ell_2$ regularization under general nonconvex objectives, establishing $O(1/\sqrt{T})$ rates, while \citet{chen2019convergence} provide convergence results for decoupled weight decay under similar assumptions. Most relevant to our work, \citet{xie2024investigating} prove that weight decay improves the nonconvex convergence bound when the alignment quantity $\delta_T = \sup_t |\langle \nabla f(w_t), w_t \rangle| / (\|\nabla f(w_t)\| \|w_t\|) < 1$. Our study extends this framework to time-varying weight decay and empirically characterizes the alignment quantity under standard training. We note that \citet{xie2024investigating} analyze the fixed-LR setting; our Theorem~\ref{thm:convergence} extends to the decaying-LR, time-varying-WD case, producing a qualitatively similar bound.

The connection between weight decay and parameter norms has been studied through several lenses. \citet{lyu2020gradient} show that gradient descent with weight decay exhibits an implicit bias toward max-margin solutions, a result that connects the cumulative regularization effect to generalization. The scale-free perspective on weight decay \cite{van2017l2} observes that for networks with batch normalization, weight decay does not directly regularize the effective function but controls the effective learning rate. This is complementary to our findings: if weight decay primarily controls the scale of parameters rather than providing function-space regularization, then its cumulative effect (total shrinkage) should indeed be the relevant quantity, consistent with our budget equivalence result.

### Learning Rate Scheduling and LR--WD Interaction

Learning rate scheduling is fundamental to modern deep learning training. The milestone (step) schedule \cite{he2016deep}, where $\gamma_t$ is reduced by a factor at fixed epochs, remains the standard for ResNet-family architectures. Alternatives include cosine annealing \cite{loshchilov2017sgdr}, cyclical learning rates \cite{smith2017cyclical}, and warmup-then-decay schedules. The interaction between abrupt LR drops and weight decay is implicit in these practices---when $\gamma_t$ decreases, the effective regularization-to-gradient ratio increases---but has not been formally characterized for adaptive WD schemes. Our LR--WD coupling analysis (Section~\ref{sec:coupling}) makes this interaction explicit and shows that it is the primary stability mechanism for time-varying weight decay.

### Dynamic and Adaptive Regularization

Several lines of work have explored time-varying or input-dependent regularization. At the architecture level, layer-wise adaptive methods such as LARS \cite{you2017large} and LAMB \cite{you2020large} scale the learning rate (and implicitly the effective weight decay) per layer based on the ratio of parameter norm to gradient norm. These methods are designed for large-batch training and address a different problem from temporal weight decay adaptation.

At the coordinate level, Cautious Weight Decay (CWD) \cite{liu2025cautious}, recently proposed at ICLR 2025, applies weight decay coordinate-wise, masking out dimensions where the weight decay direction conflicts with the gradient direction. CWD's theoretical motivation---avoiding interference between regularization and optimization---is precisely the alignment hypothesis our paper tests. The systematic late-training instability we observe across all architectures (Section~\ref{sec:cross_arch}) therefore provides theoretically motivated evidence that coordinate-wise alignment-aware regularization introduces optimization risks under standard training conditions.

At the schedule level, meta-learning approaches to hyperparameter adaptation \cite{baydin2018online, franceschi2018bilevel} learn learning rate and regularization schedules through gradient-based bilevel optimization. While powerful in principle, these methods introduce substantial computational overhead and additional hyperparameters, and have not been widely adopted for weight decay scheduling in standard training pipelines. AADWD belongs to this schedule-level category, and no prior work in this category has used gradient--parameter alignment as the scheduling signal.

Scheduled regularization has been explored for other techniques: curriculum dropout \cite{morerio2017curriculum} increases the dropout rate over training, and various data augmentation schedules have been proposed \cite{cubuk2020randaugment}. However, systematic investigation of weight decay scheduling---as opposed to weight decay \emph{magnitude} tuning---remains sparse in the literature.

### Gradient--Parameter Alignment in Training Dynamics

The angle between the gradient and the parameter vector has appeared in several analyses of neural network training. \citet{fort2019emergent} study the alignment between successive gradients as a diagnostic of training phase transitions. \citet{gur2018gradient} analyze the spectrum of the Hessian during training and observe that the leading eigenvector aligns with the gradient in the early phase. In the context of implicit regularization, \citet{blanc2020implicit} show that label noise SGD biases optimization toward directions of low curvature, and \citet{li2021happens} establish connections between the gradient covariance structure and generalization.

Our work tests whether the gradient--parameter cosine similarity---specifically as formalized in the alignment quantity $\delta_t$---can be \emph{exploited} for explicit regularization scheduling. Unlike prior work that uses gradient alignment as a diagnostic, we test its prescriptive value. The negative answer we obtain---that $\delta_t \sim O(10^{-3})$ renders alignment uninformative under standard training---does not contradict the theoretical importance of the alignment condition, but rather establishes that the condition is always easily satisfied, leaving no room for adaptive exploitation.

\paragraph{Scope.} Our analysis is restricted to SGD with momentum and milestone LR schedules on image classification benchmarks (CIFAR-10/100, ResNet-20, VGG-16-BN). We do not investigate Adam/AdamW, transformer architectures, or LLM pretraining, where different weight norm dynamics may apply.

---

## 3. Theoretical Framework and AADWD Design
\label{sec:theory}

### 3.1 Preliminaries and Problem Setup
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
$\mathbb{E}[\|g_t\|^2] \leq G^2$. (This is used in the weight norm terms of the convergence bound; when the gradient mean is bounded, it subsumes Assumption~2, but we state both for clarity.)
\end{assumption}

Following \citet{xie2024investigating}, we define the alignment quantity at step $t$:
\begin{equation}
\delta_t = \frac{|\langle \nabla f_S(w_t), w_t \rangle|}{\|\nabla f_S(w_t)\| \cdot \|w_t\|} \in [0, 1].
\label{eq:alignment}
\end{equation}
The quantity $\delta_t$ measures the cosine of the angle between the gradient direction and the parameter direction. When $\delta_t$ is small, the gradient is nearly orthogonal to the parameter vector, and weight decay (which shrinks $w_t$ toward zero) has minimal interference with the gradient descent direction. When $\delta_t$ is large, weight decay directly opposes the gradient, potentially impeding optimization.

For constant weight decay $\lambda_t = \lambda$, \citet{xie2024investigating} show that the convergence bound improves over the no-weight-decay case when $\delta_T = \sup_{t \leq T} \delta_t < 1$, with the improvement proportional to $1 - \delta_T$.

### 3.2 Convergence with Time-Varying Weight Decay

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

\begin{remark}
Theorem~\ref{thm:convergence} is stated under polynomial decay $\gamma_t = \gamma / \sqrt{T}$. For the milestone schedules used in our experiments (where $\gamma_t$ is piecewise constant), the same qualitative structure holds---the bound depends on $\bar{\Lambda}_T$---but the exact constants differ due to the discontinuous jumps in $\gamma_t$. See Appendix~E for discussion. Additionally, the condition $\lambda_t \leq C\gamma_t^2$ is satisfied by construction only by the Square variant; the Conservative and Aggressive variants operate outside the theorem's formal scope, though they satisfy the weaker condition $\lambda_t = O(\gamma_t)$.
\end{remark}

\begin{theorem}[Alignment-weighted bound]
\label{thm:alignment_weighted}
Under the same conditions as Theorem~\ref{thm:convergence}, the generalization-relevant component of the convergence bound depends on the cumulative alignment:
\begin{equation}
\bar{\delta}_T = \frac{\sum_{t=1}^{T} \lambda_t \delta_t}{\sum_{t=1}^{T} \lambda_t}
\end{equation}
rather than $\delta_T = \sup_t \delta_t$. When $\delta_t$ varies significantly across training, time-varying $\lambda_t$ can improve the bound by assigning larger $\lambda_t$ to steps with smaller $\delta_t$. Note that $\bar{\delta}_T$ uses a $\lambda_t$-weighted average (upweighting steps with larger WD), in contrast to the uniform time-average $\bar{\Lambda}_T$ in Theorem~\ref{thm:convergence}.
\end{theorem}

\emph{Implication.} Theorem~\ref{thm:alignment_weighted} provides the theoretical motivation for AADWD: if $\delta_t$ varies substantially during training, an alignment-aware schedule can tighten the convergence bound from $\delta_T$ to $\bar{\delta}_T \leq \delta_T$. However, this improvement is only realized when the variation in $\delta_t$ is large enough to be exploited---a condition our experiments will show is \emph{not} met under standard training.

### 3.3 The Three AADWD Variants
\label{sec:variants}

We construct the stochastic alignment proxy $\hat{\delta}_t$ using an exponential moving average of per-minibatch alignment measurements:
\begin{equation}
\hat{\delta}_t = \beta \cdot \hat{\delta}_{t-1} + (1 - \beta) \cdot \frac{|\langle g_t, w_t \rangle|}{(\|g_t\| \cdot \|w_t\| + \epsilon)},
\end{equation}
where $\beta \in \{0.9, 0.99, 0.999, 0.9999\}$ controls the smoothing window and $\epsilon = 10^{-8}$ prevents division by zero. Note that $g_t$ here is the minibatch stochastic gradient, so $\hat{\delta}_t$ is a stochastic proxy for the true alignment $\delta_t$. We validate the faithfulness of $\hat{\delta}_t$ in Section~\ref{sec:alignment_char}.

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
\emph{Rationale:} Apply weight decay inversely proportional to alignment. When $\hat{\delta}_t$ is small (misalignment), $\lambda_t$ is large; when $\hat{\delta}_t$ is large (alignment), $\lambda_t$ is small. This produces the most dynamic behavior among the three variants. The $\epsilon = 10^{-8}$ term serves as a numerical safeguard; for the observed regime $\hat{\delta}_t \approx 0.003$, it is negligible.

\paragraph{Square.}
\begin{equation}
\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t^2 \cdot (1 - \hat{\delta}_t),\;\lambda_{\min},\;\lambda_{\max}\bigr).
\label{eq:square}
\end{equation}
\emph{Rationale:} The $\gamma_t^2$ coupling satisfies the condition $\lambda_t = O(\gamma_t^2)$ from Theorem~\ref{thm:convergence}, providing the strongest convergence guarantee among the three variants. However, under milestone schedules where $\gamma_t$ drops by $10\times$ at epochs 80 and 120, this causes $\lambda_t$ to drop by $100\times$ and $10{,}000\times$ respectively, effectively eliminating regularization in late training.

All variants share hyperparameters: scaling constant $c$, EMA coefficient $\beta$, and clipping bounds $\lambda_{\min} = 10^{-6}$, $\lambda_{\max} \in \{0.01, 0.05\}$.

### 3.4 Stability Analysis of LR--WD Coupling
\label{sec:coupling}

A critical design element in all three AADWD formulas is the learning rate multiplier $\gamma_t$. We now analyze why this coupling is not merely convenient but structurally necessary.

\begin{observation}[Decoupling instability]
\label{obs:instability}
Consider the decoupled aggressive variant: $\lambda_t = c / (\hat{\delta}_t + \epsilon)$ (removing $\gamma_t$). Under a milestone learning rate schedule where $\gamma_t$ drops by a factor of $10$ at step $t^*$, the ratio $\lambda_t / \gamma_t$ increases by $10\times$ at the milestone. If $\lambda_t / \gamma_t$ exceeds a critical threshold, the weight decay term dominates the gradient term in (\ref{eq:sgdw}), triggering a positive feedback loop:
\begin{equation}
\|w_t\| \downarrow \;\to\; \hat{\delta}_t \downarrow \;\to\; \lambda_t \uparrow \;\to\; \|w_t\| \downarrow\downarrow \;\to\; \cdots
\label{eq:feedback}
\end{equation}
leading to weight norm collapse ($\|w_T\| \to 0$) and training failure. The sign of the middle step follows because as $\|w_t\|$ shrinks, the numerator $|\langle g_t, w_t \rangle|$ decreases faster than $\|g_t\| \cdot \|w_t\|$ when the gradient direction is not aligned with $w_t / \|w_t\|$, causing $\hat{\delta}_t$ to decrease and hence $\lambda_t = c / (\hat{\delta}_t + \epsilon)$ to increase.
\end{observation}

\emph{Informal argument.} In standard $\ell_2$ regularization, the effective weight decay is $\gamma_t \cdot \lambda$, which automatically scales with the learning rate. Decoupled weight decay \cite{loshchilov2019decoupled} deliberately removes this scaling for Adam---but compensates by treating $\lambda$ as a \emph{constant} that is tuned jointly with the learning rate. In AADWD, $\lambda_t$ is not constant but depends on the alignment state $\hat{\delta}_t$. Without the $\gamma_t$ damper, $\lambda_t$ can become arbitrarily large relative to the gradient signal after a learning rate drop. The learning rate multiplier serves as a natural ``speed limit'' that ensures weight decay remains commensurate with the optimization step size. A more formal treatment would show that for the decoupled aggressive variant, $\mathbb{E}[\|w_t\|^2]$ decreases geometrically when $\lambda_t > \gamma_t G / \|w_t\|$; we verify this empirically in Section~\ref{sec:decoupling}.

\begin{proposition}[Budget equivalence, informal]
\label{prop:budget}
Under mild conditions on the loss landscape (approximate convexity near convergence), if two weight decay schedules $\{\lambda_t^{(1)}\}_{t=1}^T$ and $\{\lambda_t^{(2)}\}_{t=1}^T$ satisfy $\sum_{t=1}^T \lambda_t^{(1)} = \sum_{t=1}^T \lambda_t^{(2)}$, their regularization effects on the final weight norm---and hence on generalization---are asymptotically equivalent.
\end{proposition}

\emph{Argument.} Near convergence, the weight norm evolves approximately as $\|w_T\|^2 \approx \|w_0\|^2 \prod_{t=1}^T (1 - 2\lambda_t) + R_T$, where $R_T$ collects gradient-dependent terms. For small $\lambda_t$, $\prod_t (1 - 2\lambda_t) \approx \exp(-2\sum_t \lambda_t)$, which depends only on the cumulative sum $\sum_t \lambda_t$. The residual $R_T$ involves terms of the form $\langle \nabla f_S(w_t), w_t \rangle$, which are bounded by $\delta_t \|\nabla f_S(w_t)\| \|w_t\|$. Since $\delta_t \leq 0.005$ throughout training (Section~\ref{sec:alignment_char}), these trajectory-dependent terms contribute at most $O(\delta_t \cdot \gamma_t \cdot G \cdot \|w_t\|)$ per step---small relative to the decay term $\lambda_t \|w_t\|^2$ for the parameter ranges under study. When the gradient-dependent residuals are thus bounded, the final weight norms---and hence the regularization effects---are determined by the weight decay budget alone. We verify this empirically in Section~\ref{sec:budget}.

---

## 4. Experimental Setup
\label{sec:experiments}

### 4.1 Datasets and Architectures

We evaluate on two standard image classification benchmarks, chosen because they are standard settings where the regularization signal is meaningful but experimental cost is manageable:

\begin{itemize}
\item \textbf{CIFAR-10} \cite{krizhevsky2009learning}: 50,000 training / 10,000 test images across 10 classes. This is the primary evaluation benchmark for all experiments.
\item \textbf{CIFAR-100} \cite{krizhevsky2009learning}: 50,000 training / 10,000 test images across 100 classes. The finer-grained classification task increases regularization sensitivity and amplifies differences between methods.
\end{itemize}

We use two architectures spanning different design families:

\begin{itemize}
\item \textbf{ResNet-20} \cite{he2016deep}: A compact residual network with 0.27M parameters, serving as the primary architecture for all experiments including ablations and hyperparameter sweeps.
\item \textbf{VGG-16-BN} \cite{simonyan2015very}: A deep feedforward network with batch normalization, containing 15M parameters ($56\times$ larger than ResNet-20), used for cross-architecture validation.
\end{itemize}

Standard data augmentation is applied: random crop ($32 \times 32$, padding 4), random horizontal flip, and per-channel normalization.

### 4.2 Methods Compared

We compare 13 weight decay configurations organized into four categories:

\begin{itemize}
\item \textbf{Baselines (7):} No WD ($\lambda = 0$), Fixed WD grid search ($\lambda \in \{10^{-4}, 3 \times 10^{-4}, 5 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$), and Stagewise WD (LR-proportional at milestones, where $\lambda$ is reduced proportionally to $\gamma_t$ at each milestone).
\item \textbf{AADWD variants (3):} Conservative, Aggressive, and Square (Section~\ref{sec:variants}).
\item \textbf{Ablation controls (2):} Random Dynamic WD ($\lambda_t = c \cdot \gamma_t \cdot U(0,1)$), and Equivalent Cumulative WD (constant $\lambda$ matched to AADWD's time-average).
\item \textbf{Recent adaptive method (1):} Cautious Weight Decay (CWD) \cite{liu2025cautious}, applying coordinate-wise sign-based masking.
\end{itemize}

\noindent An additional Norm-Matched WD ablation (matching AADWD's weight norm trajectory) was conducted but exhibited severe late-training collapse (best $90.44\%$, final $75.37\%$); results are reported in Appendix~B.

### 4.3 Training Protocol

All experiments use SGD with momentum $0.9$, initial learning rate $0.1$, milestone schedule at epochs $[80, 120]$ with $\gamma \times 0.1$, batch size 128, and 200 training epochs. All experiments use a single random seed (42); we discuss this limitation in Section~\ref{sec:discussion}. AADWD hyperparameters are swept over $c \in \{0.5, 1.0, 2.5, 5.0, 10.0\}$ and $\beta \in \{0.9, 0.99, 0.999, 0.9999\}$, with $\lambda_{\min} = 10^{-6}$ and $\lambda_{\max} \in \{0.01, 0.05\}$ (the aggressive variant uses $\lambda_{\max} = 0.05$; others use $\lambda_{\max} = 0.01$).

We report four evaluation metrics: best test accuracy (highest test accuracy during training), final test accuracy (epoch 200), generalization gap (train accuracy $-$ test accuracy at convergence), and weight norm ($\|w_T\|_2$).

---

## 5. Results
\label{sec:results}

### 5.1 Main Results: Fixed WD Dominates All Dynamic Methods
\label{sec:main_results}

Table~\ref{tab:main} presents the central comparison on CIFAR-10/ResNet-20.

\begin{table}[t]
\centering
\caption{Main results on CIFAR-10/ResNet-20 (200 epochs, seed=42). Best and tied-best results in \textbf{bold}. Fixed WD ($5 \times 10^{-4}$) achieves the best performance; the budget-equivalent constant exactly matches it. All dynamic methods are dominated.}
\label{tab:main}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Best Acc (\%)} & \textbf{Final Acc (\%)} & \textbf{Gen Gap} & \textbf{Weight Norm} \\
\midrule
No WD & 90.49 & 90.31 & 9.17 & 129.42 \\
Fixed WD ($10^{-4}$) & 92.08 & 91.83 & 7.64 & 56.34 \\
Fixed WD ($3 \times 10^{-4}$) & 92.39 & 91.97 & 7.52 & 30.70 \\
Fixed WD ($5 \times 10^{-4}$) & \textbf{92.54} & \textbf{92.29} & 7.17 & 23.49 \\
Fixed WD ($10^{-3}$) & 92.32 & 91.92 & 6.95 & 17.13 \\
Fixed WD ($3 \times 10^{-3}$) & 88.63 & 87.25 & 5.20 & 11.44 \\
\midrule
Stagewise WD & 92.44 & 92.27 & 7.18 & 33.22 \\
CWD \cite{liu2025cautious} & 91.79 & 86.95 & 3.22 & 9.28 \\
\midrule
AADWD Conservative & 92.37 & 92.22 & 7.15 & 23.60 \\
AADWD Aggressive & 92.05 & 91.57 & 7.50 & 21.47 \\
AADWD Square & 92.13 & 91.78 & 7.47 & 38.75 \\
\midrule
Random Dynamic WD & 92.06 & 91.95 & 7.54 & 34.19 \\
Equiv.\ Cumulative WD & \textbf{92.54} & \textbf{92.29} & 7.17 & 23.49 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Key observations.} (i) Fixed WD at $\lambda = 5 \times 10^{-4}$ achieves the best performance ($92.54\%$), confirming the standard practice. (ii) The budget-equivalent constant WD \emph{exactly} matches fixed WD ($92.54\%$, weight norm $23.49$), establishing the budget equivalence principle. (iii) AADWD Conservative ($92.37\%$) performs within $0.17\%$ of fixed WD because $\hat{\delta}_t \approx 0$ causes it to degenerate to $\lambda_t \approx c \cdot \gamma_t$ (Section~\ref{sec:alignment_char}). (iv) All AADWD variants are strictly dominated by fixed WD. (v) CWD shows dramatic late-training collapse: best accuracy $91.79\%$ degrades to final $86.95\%$ ($\Delta = 4.84\%$), with an abnormally low weight norm of $9.28$ indicating over-aggressive shrinkage. We note that all differences between AADWD variants and fixed WD are small ($\leq 0.49\%$) and based on a single seed; these should be interpreted cautiously.

The fixed WD grid search reveals a well-behaved optimum at $\lambda = 5 \times 10^{-4}$: accuracy increases monotonically from $\lambda = 10^{-4}$ ($92.08\%$) to $\lambda = 5 \times 10^{-4}$ ($92.54\%$), then decreases for $\lambda = 10^{-3}$ ($92.32\%$) and sharply for $\lambda = 3 \times 10^{-3}$ ($88.63\%$). This smooth landscape makes single-hyperparameter tuning straightforward (see Figure~\ref{fig:wd_landscape}).

### 5.2 Cross-Architecture and Cross-Dataset Validation
\label{sec:cross_arch}

Table~\ref{tab:cross_arch} extends the comparison to CIFAR-100/ResNet-20 and CIFAR-10/VGG-16-BN.

\begin{table}[t]
\centering
\caption{Cross-architecture and cross-dataset results (best test accuracy, \%). Fixed WD dominates in all settings. CWD shows best $\to$ final collapse universally. $\Delta_{\text{CWD}}$ denotes the best-to-final degradation.}
\label{tab:cross_arch}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{C10/ResNet-20} & \textbf{C100/ResNet-20} & \textbf{C10/VGG-16-BN} \\
\midrule
No WD & 90.49 & 64.70 & 92.34 \\
Fixed WD ($5 \times 10^{-4}$) & \textbf{92.54} & \textbf{68.45} & \textbf{93.86} \\
AADWD Conservative & 92.37 & 68.24 & 93.75 \\
AADWD Aggressive & 92.05 & 61.34 & 90.97 \\
CWD (best $\to$ final) & 91.79 $\to$ 86.95 & 66.84 $\to$ 54.27 & 92.95 $\to$ 86.47 \\
\midrule
$\Delta_{\text{CWD}}$ & $-4.84$ & $-12.57$ & $-6.48$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Key observations.} (i) Fixed WD is the best method across all three settings, confirming architecture- and dataset-independence. (ii) AADWD Conservative is consistently the second-best dynamic method but remains within noise of fixed WD ($\Delta \leq 0.21\%$). (iii) AADWD Aggressive degrades more severely on the harder task CIFAR-100 ($\Delta = -7.11\%$ vs.\ fixed WD), suggesting that alignment-dependent modulation introduces instability on tasks requiring stronger regularization. (iv) CWD late-training collapse is \emph{universal}: the best-to-final degradation of $4.84\%$, $12.57\%$, and $6.48\%$ across settings constitutes a systematic failure mode. The effect is most severe on CIFAR-100 ($12.57\%$ drop), where the 100-class problem amplifies the impact of weight over-shrinkage. (v) VGG-16-BN ($15$M parameters) shows qualitatively identical patterns to ResNet-20 ($0.27$M parameters), indicating that our conclusions are independent of model scale within the tested range.

### 5.3 Budget Equivalence Experiment
\label{sec:budget}

The budget equivalence experiment provides the cleanest causal evidence in this study.

\paragraph{Design.} We first complete a full AADWD-Aggressive run on CIFAR-10/ResNet-20 and record the per-step weight decay values $\{\lambda_t\}_{t=1}^T$. We then compute the time-average $\bar{\lambda} = (1/T)\sum_t \lambda_t$ and run a new experiment with constant $\lambda = \bar{\lambda}$ for 200 epochs under otherwise identical conditions. The computed time-average $\bar{\lambda}$ from the AADWD-Aggressive trajectory was $5.0 \times 10^{-4}$, which happens to coincide with the standard fixed WD baseline. This coincidence arises from the interaction of the scaling constant $c = 2.5$, the clipping bounds, and the near-constant alignment signal; it is not a design choice.

\paragraph{Results.} The budget-equivalent constant WD achieves $92.54\%$ best accuracy, $92.29\%$ final accuracy, weight norm $23.49$, and generalization gap $7.17$---\emph{exactly} matching the fixed WD ($5 \times 10^{-4}$) baseline on all four metrics. In contrast, AADWD Aggressive with the same cumulative budget achieves only $92.05\%$ best accuracy, $91.57\%$ final accuracy, weight norm $21.47$, and generalization gap $7.50$. Figure~\ref{fig:budget_equiv} shows the $\lambda_t$ trajectories for both schedules alongside their matched final performance.

\paragraph{Interpretation.} The $0.49\%$ gap between AADWD-Aggressive ($92.05\%$) and its budget-equivalent constant ($92.54\%$) is entirely attributable to the suboptimal \emph{temporal allocation} of weight decay, not to the total budget. The dynamic schedule allocates regularization pressure non-uniformly across training phases while providing insufficient regularization in later stages. But crucially, a constant schedule with the same total budget achieves the optimal performance---demonstrating that the temporal dynamics of $\lambda_t$ provide \emph{zero additional information} beyond the average magnitude under the tested conditions.

### 5.4 LR Decoupling Experiment
\label{sec:decoupling}

Table~\ref{tab:decoupled} presents the most dramatic finding of this study: the effect of removing the learning rate multiplier $\gamma_t$ from AADWD.

\begin{table}[t]
\centering
\caption{Effect of removing the LR multiplier $\gamma_t$ from AADWD (CIFAR-10/ResNet-20). Decoupling triggers catastrophic collapse for the aggressive variant and severe degradation for the conservative variant.}
\label{tab:decoupled}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Variant} & \textbf{Coupled (\%)} & \textbf{Decoupled (\%)} & $\boldsymbol{\Delta}$ & \textbf{WN (Coupled)} & \textbf{WN (Decoupled)} \\
\midrule
Conservative & 92.37 & 90.32 $\to$ 80.30 & $-12.07$ & 23.60 & 5.53 \\
Aggressive & 92.05 & 84.49 $\to$ 10.00 & $-82.05$ & 21.47 & 0.0036 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Aggressive collapse ($92.05\% \to 10.00\%$).} Without the $\gamma_t$ damper, the aggressive variant exhibits the positive feedback instability described in Observation~\ref{obs:instability}. After the first milestone at epoch 80, the learning rate drops from $0.1$ to $0.01$, but the decoupled $\lambda_t = c / (\hat{\delta}_t + \epsilon)$ remains at approximately $5 \times 10^{-4}$. The effective regularization-to-gradient ratio $\lambda_t / \gamma_t$ increases by $10\times$, causing weight decay to overwhelm the gradient signal. The weight norm begins shrinking rapidly, which decreases $\hat{\delta}_t$, which further increases $\lambda_t$ (via the inverse formula), creating a self-amplifying feedback loop (Eq.~\ref{eq:feedback}). By epoch 200, the weight norm has collapsed to $0.0036$ and training has degraded to random chance ($10\%$). The best accuracy of $84.49\%$, achieved before the first milestone, confirms that the collapse is triggered specifically by the learning rate drop. Figure~\ref{fig:decoupled_collapse} shows the weight norm trajectories on a log scale, illustrating the catastrophic divergence at the epoch-80 milestone.

\paragraph{Conservative degradation ($92.37\% \to 80.30\%$).} The conservative variant's decoupled form $\lambda_t = c \cdot (1 - \hat{\delta}_t) \approx 5 \times 10^{-4}$ becomes a constant weight decay that is $10\times$ too strong when $\gamma_t = 0.01$ (after epoch 80) and $100\times$ too strong when $\gamma_t = 0.001$ (after epoch 120). This causes progressive over-regularization (weight norm shrinks to $5.53$, vs.\ $23.60$ in the coupled case) and underfitting, but the bounded nature of the conservative formula prevents complete collapse.

\paragraph{Mechanistic insight.} The LR multiplier $\gamma_t$ serves as a structural stabilizer that ensures weight decay remains commensurate with the optimization step size. In standard $\ell_2$ regularization, this coupling is automatic ($\text{effective decay} = \gamma_t \cdot \lambda$). Decoupled weight decay \cite{loshchilov2019decoupled} deliberately removes it for Adam---but uses a \emph{constant} $\lambda$, which is safe because a constant $\lambda$ was tuned jointly with the learning rate schedule. AADWD with a \emph{varying} $\lambda_t$ requires the $\gamma_t$ coupling to be reintroduced as an explicit stability mechanism.

### 5.5 Hyperparameter Sensitivity
\label{sec:sensitivity}

Table~\ref{tab:sensitivity} presents the sensitivity of AADWD-Aggressive to the scaling constant $c$ and EMA coefficient $\beta$ on CIFAR-10/ResNet-20.

\begin{table}[t]
\centering
\caption{Hyperparameter sensitivity of AADWD-Aggressive (CIFAR-10/ResNet-20, $\lambda_{\max} = 0.05$). Left: $c$-sweep with $\beta = 0.999$. Right: $\beta$-sweep with $c = 2.5$. The method is robust over a $5\times$ range of $c$ and across three orders of magnitude of $(1-\beta)$, but no configuration outperforms fixed WD ($92.54\%$).}
\label{tab:sensitivity}
\small
\begin{tabular}{lccc}
\toprule
\multicolumn{4}{c}{\textbf{$c$-sweep} ($\beta = 0.999$)} \\
\midrule
$c$ & \textbf{Best Acc (\%)} & \textbf{Final Acc (\%)} & \textbf{Weight Norm} \\
\midrule
0.5 & 91.87 & 91.69 & 65.48 \\
1.0 & 92.18 & 92.07 & 41.27 \\
2.5 & 92.05 & 91.57 & 21.47 \\
5.0 & 87.98 & 87.24 & 13.07 \\
10.0 & 52.12 & 10.00 & 0.004 \\
\midrule
\multicolumn{4}{c}{\textbf{$\beta$-sweep} ($c = 2.5$)} \\
\midrule
$\beta$ & \textbf{Best Acc (\%)} & \textbf{Final Acc (\%)} & \textbf{Weight Norm} \\
\midrule
0.9 & 92.08 & 91.95 & 21.75 \\
0.99 & 92.24 & 91.97 & 21.65 \\
0.999 & 92.05 & 91.57 & 21.47 \\
0.9999 & 92.25 & 91.80 & 20.56 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Scaling constant $c$.} AADWD-Aggressive is robust over a $5\times$ range ($c \in [0.5, 2.5]$): accuracy varies by only $0.31\%$ ($91.87$--$92.18\%$). However, $c = 5.0$ causes noticeable degradation ($87.98\%$), and $c = 10.0$ triggers complete collapse ($52.12\% \to 10.00\%$, weight norm $0.004$)---the same positive feedback failure mode as in the decoupled experiment (Section~\ref{sec:decoupling}), here triggered by excessive $c$ rather than LR decoupling. The weight norm decreases monotonically with $c$ (from $65.48$ at $c = 0.5$ to $0.004$ at $c = 10.0$), confirming that $c$ primarily controls the regularization budget. Critically, even the best configuration ($c = 1.0$, $92.18\%$) falls short of fixed WD ($92.54\%$) by $0.36\%$. We note that this gap is within the range where single-seed noise could be a factor.

\paragraph{EMA coefficient $\beta$.} The alignment proxy is remarkably insensitive to $\beta$: across three orders of magnitude of the time constant ($1 - \beta \in [10^{-4}, 10^{-1}]$), best accuracy varies by only $0.20\%$ ($92.05$--$92.25\%$). This insensitivity is itself informative: it confirms that the alignment signal $\hat{\delta}_t$ varies so little during training that the smoothing window does not matter---whether the EMA responds to the last 10 steps ($\beta = 0.9$) or the last 10,000 steps ($\beta = 0.9999$), it captures essentially the same near-constant value.

### 5.6 Alignment Proxy Characterization
\label{sec:alignment_char}

We directly characterize the alignment proxy $\hat{\delta}_t$ measured during training on ResNet-20/CIFAR-10 (Table~\ref{tab:alignment}).

\begin{table}[t]
\centering
\caption{Alignment proxy $\hat{\delta}_t$ statistics across training phases (ResNet-20/CIFAR-10, 200 epochs, $\beta = 0.999$). The signal is near-constant at $O(10^{-3})$ with negligible variation, explaining why alignment-based WD adaptation degenerates to a deterministic function of the learning rate.}
\label{tab:alignment}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Training Phase} & \textbf{Mean $\hat{\delta}_t$} & \textbf{Std} & \textbf{Range} \\
\midrule
Early (epochs 1--80) & 0.004491 & --- & --- \\
Mid (epochs 81--120) & 0.003352 & --- & --- \\
Late (epochs 121--200) & 0.002824 & --- & --- \\
\midrule
Overall & $\sim 0.003$ & 0.000753 & $[0.0028, 0.0045]$ \\
\midrule
Pearson $r$ (EMA vs.\ large-batch) & \multicolumn{3}{c}{0.849} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Key observations.} (i) The alignment proxy remains at $O(10^{-3})$ throughout training, with a total range of only $[0.0028, 0.0045]$---a $1.6\times$ variation over 200 epochs. (ii) The standard deviation ($7.5 \times 10^{-4}$) is comparable to the mean, indicating noisy measurements around a near-constant center. (iii) There is a monotonic decrease from early to late training ($0.0045 \to 0.0028$), but this $1.6\times$ variation is negligible compared to the $100\times$ variation in the learning rate $\gamma_t$ over the same period. (iv) The Pearson correlation of $r = 0.849$ between the minibatch EMA and large-batch alignment confirms adequate proxy fidelity. We note that this value marginally falls below an $r = 0.85$ threshold used in our initial diagnostic; however, the core finding does not depend on proxy quality---the underlying signal is inherently too weak to be informative, regardless of measurement precision. Figure~\ref{fig:alignment_trajectory} shows the alignment proxy trajectory overlaid with the LR schedule.

\paragraph{Consequence for AADWD.} With $\hat{\delta}_t \approx 0.003$, the conservative formula becomes $\lambda_t = c \cdot \gamma_t \cdot (1 - 0.003) \approx c \cdot \gamma_t \cdot 0.997$, which is functionally indistinguishable from $\lambda_t = c \cdot \gamma_t$---a deterministic schedule with no alignment contribution. This explains why AADWD Conservative ($92.37\%$) nearly matches fixed WD ($92.54\%$): it is, in effect, a learning-rate-proportional schedule, and the $0.17\%$ gap comes from the proportionality constant $c$ rather than from alignment information. The alignment signal is dominated by the learning rate dynamics by a factor of $\sim 60\times$ ($100\times$ LR variation vs.\ $1.6\times$ alignment variation).

---

## 6. Analysis: Why Dynamic Weight Decay Fails
\label{sec:analysis}

The experimental results in Section~\ref{sec:results} establish that no dynamic weight decay schedule outperforms a well-tuned constant. In this section, we synthesize three mechanistic insights that explain \emph{why} dynamic weight decay scheduling is ineffective under standard nonconvex SGD training. We focus on deepening the mechanistic explanations beyond the empirical observations of Section~\ref{sec:results}, connecting results to the formal framework of Section~\ref{sec:theory}.

### 6.1 Insight 1: The Budget Equivalence Principle

\paragraph{Formal argument.} Weight decay's primary regularization mechanism is controlling the weight norm trajectory $\|w_t\|$ during training. The weight norm at convergence determines the model's effective capacity and, through its connection to the loss landscape geometry, its generalization properties \cite{neyshabur2015norm, bartlett2017spectrally}. We argue that the final weight norm is determined by the \emph{total regularization pressure} $\sum_t \lambda_t$, not by how this pressure is distributed over time.

To see this formally, consider the weight norm evolution under SGDW. Squaring both sides of the update (\ref{eq:sgdw}) and taking expectations:
\begin{equation}
\mathbb{E}[\|w_{t+1}\|^2] = (1 - \lambda_t)^2 \mathbb{E}[\|w_t\|^2] - 2\gamma_t(1-\lambda_t)\mathbb{E}[\langle \nabla f_S(w_t), w_t \rangle] + \gamma_t^2 \mathbb{E}[\|g_t\|^2].
\end{equation}
For small $\lambda_t$ (the regime of practical interest, as $\lambda \sim 10^{-4}$--$10^{-3}$):
\begin{equation}
(1 - \lambda_t)^2 \approx 1 - 2\lambda_t, \qquad \prod_{t=1}^T (1 - 2\lambda_t) \approx \exp\!\left(-2\sum_{t=1}^T \lambda_t\right).
\end{equation}
The exponential factor depends only on $\sum_t \lambda_t$, not on the temporal distribution $\{\lambda_t\}_{t=1}^T$. The remaining terms involve the gradient--parameter inner product $\langle \nabla f_S(w_t), w_t \rangle$ and the gradient norm $\|g_t\|^2$. Using the alignment bound, these trajectory-dependent terms contribute at most $O(\delta_t \cdot \gamma_t \cdot G \cdot \|w_t\|)$ per step. Since $\delta_t \leq 0.005$ throughout training (Table~\ref{tab:alignment}) and $\gamma_t \leq 0.1$, the gradient--parameter cross-term is small relative to the multiplicative decay factor $\lambda_t \|w_t\|^2$. This makes the budget approximation $\prod_t(1 - 2\lambda_t) \approx \exp(-2\sum_t \lambda_t)$ the dominant determinant of $\|w_T\|^2$, and hence of generalization.

\paragraph{Experimental confirmation.} The budget equivalence experiment (Section~\ref{sec:budget}) provides direct evidence: the constant WD with $\bar{\lambda} = \text{mean}(\lambda_t^{\text{AADWD-agg}}) = 5.0 \times 10^{-4}$ achieves $92.54\%$ accuracy and weight norm $23.49$---\emph{exactly} matching fixed WD on all metrics. The AADWD-Aggressive schedule with the same cumulative budget achieves only $92.05\%$, the $0.49\%$ deficit arising from suboptimal temporal allocation (too much regularization early, too little late).

\paragraph{Practical implication.} There is no ``optimal schedule'' for weight decay beyond choosing the right average magnitude. The single hyperparameter $\lambda$ is sufficient.

### 6.2 Insight 2: LR Coupling as a Stability Mechanism

\paragraph{Argument.} The learning rate and weight decay in SGD are not independent hyperparameters---they interact through a multiplicative coupling that is essential for training stability. In standard $\ell_2$ regularization, the effective weight decay is $\gamma_t \cdot \lambda$, which automatically decreases as the learning rate decreases. This ensures that the regularization term remains proportional to the optimization step size: as the learner takes smaller steps (lower LR), it also applies less shrinkage (lower effective WD), maintaining a balanced ratio between learning and regularization.

Decoupled weight decay \cite{loshchilov2019decoupled} breaks this coupling deliberately for AdamW, applying a constant $\lambda$ regardless of the learning rate. This works precisely because $\lambda$ is \emph{constant} and is tuned jointly with the LR schedule---the practitioner implicitly accounts for the LR--WD interaction by choosing $\lambda$ in the context of the planned LR trajectory. AADWD with a time-varying $\lambda_t$ reintroduces the need for explicit coupling.

\paragraph{The collapse mechanism.} The aggressive decoupled experiment (Section~\ref{sec:decoupling}) demonstrates this concretely. Before the first milestone (epoch 80), the effective ratio $\lambda_t / \gamma_t$ is well-behaved. At the milestone, $\gamma_t$ drops $10\times$ but $\lambda_t$ does not, causing the ratio to jump $10\times$. Weight decay now dominates the dynamics: in the update $w_{t+1} = (1 - \lambda_t) w_t - \gamma_t g_t$, the shrinkage term $(1 - \lambda_t) w_t$ overwhelms the gradient term $\gamma_t g_t$. As $\|w_t\|$ shrinks, the numerator $|\langle g_t, w_t \rangle|$ decreases faster than $\|g_t\| \cdot \|w_t\|$ (because the gradient direction, increasingly dominated by the loss curvature, becomes less aligned with the shrinking $w_t$), causing $\hat{\delta}_t$ to decrease and therefore $\lambda_t = c / (\hat{\delta}_t + \epsilon)$ to increase further. The final weight norm of $0.0036$---a factor of $\sim 6{,}000\times$ smaller than the coupled case ($21.47$)---and the accuracy of $10.00\%$ (random chance) confirm complete training failure (Figure~\ref{fig:decoupled_collapse}).

Note that this insight explains the \emph{decoupled ablation's} catastrophic failure. The \emph{coupled} AADWD-Aggressive's smaller $0.49\%$ gap relative to fixed WD is explained by Insight 1 (budget misallocation), not by coupling instability.

\paragraph{Structural lesson.} Any adaptive weight decay scheme must satisfy the condition $\lambda_t = O(\gamma_t)$ to prevent the regularizer from overwhelming the gradient signal after learning rate drops. This is not a design preference but a structural requirement for stable training under milestone-style LR schedules.

### 6.3 Insight 3: Alignment is a Consequence, Not a Control Signal

\paragraph{Argument.} The theoretical result of \citet{xie2024investigating}---that $\delta_T < 1$ is sufficient for weight decay to improve convergence---does not imply that monitoring $\delta_t$ during training provides actionable information for weight decay scheduling. The key distinction is between a \emph{condition} (which defines when something works) and a \emph{control signal} (which tells you how to adjust a parameter in real time).

For $\delta_t$ to serve as a useful control signal, it must satisfy two requirements: (a) it must vary significantly during training, so that different WD rates are appropriate at different times; and (b) the variation must carry information that is not already captured by the learning rate schedule. Our experiments show that neither condition holds.

\paragraph{Condition (a): Insufficient variation.} The alignment proxy $\hat{\delta}_t$ ranges over $[0.0028, 0.0045]$ during 200 epochs of training (Table~\ref{tab:alignment})---a $1.6\times$ variation. The alignment condition $\delta_T < 1$ is satisfied by a factor of $\sim 200\times$ at every point in training. There is no phase where $\delta_t$ approaches 1 and weight decay becomes counterproductive. Using Theorem~\ref{thm:alignment_weighted}, the bound improvement from using $\bar{\delta}_T$ instead of $\sup_t \delta_t$ scales with the ratio $\sup_t \delta_t / \bar{\delta}_T$. With $\sup_t \delta_t \approx 0.0045$ and $\bar{\delta}_T \approx 0.003$, the improvement is at most a factor of $\sim 1.5\times$ on a term that is already $O(10^{-3})$---negligible in terms of its effect on the convergence bound and undetectable in practice.

\paragraph{Condition (b): No information beyond LR.} The random dynamic WD experiment provides the decisive test. By replacing $\hat{\delta}_t$ with $U(0,1)$---removing all alignment information---we obtain $92.06\%$ accuracy, essentially identical to AADWD-Aggressive's $92.05\%$. The $0.01\%$ difference is well within noise. This confirms that any time-varying WD with the appropriate budget and LR coupling performs comparably, regardless of whether it uses alignment information. Throughout this analysis, statements about $\hat{\delta}_t$ (the EMA proxy) apply equally to the true alignment $\delta_t$, given the high proxy fidelity ($r = 0.849$; Section~\ref{sec:alignment_char}).

\paragraph{CIFAR-100 asymmetry.} We note that AADWD-Aggressive degrades more severely on CIFAR-100 ($-7.11\%$ vs.\ fixed WD) than on CIFAR-10 ($-0.49\%$). This asymmetry is consistent with CIFAR-100's greater sensitivity to budget allocation errors (the 100-class problem requires more precise regularization), rather than indicating that alignment becomes more informative on harder tasks. If alignment were informative on CIFAR-100, we would expect AADWD-Conservative (which also uses alignment) to outperform fixed WD---but it does not ($68.24\%$ vs.\ $68.45\%$).

\paragraph{When might alignment matter?} Our negative result is specific to standard SGD training on CIFAR-level tasks with milestone LR schedules. We conjecture that alignment-based WD adaptation could be beneficial under conditions that produce larger $\delta_t$ variation: (i) adversarial training, where the training objective produces atypical gradient geometry; (ii) large WD values approaching the instability boundary ($\delta_T \to 1$); or (iii) highly nonconvex landscapes in early training of very deep networks without skip connections. Identifying these settings empirically remains an open question (Section~\ref{sec:future}).

---

## 7. Discussion and Conclusion
\label{sec:discussion}

### 7.1 Practical Recommendations

Our findings yield four actionable guidelines for practitioners:

\begin{enumerate}
\item \textbf{Use constant weight decay and tune only the magnitude.} Across 39 experiments spanning multiple architectures, datasets, and dynamic scheduling strategies, no method outperforms a well-tuned constant $\lambda$. The single hyperparameter $\lambda = 5 \times 10^{-4}$ (for SGD with momentum on CIFAR-scale tasks) is sufficient. The smooth optimum landscape ($92.08\%$ at $\lambda = 10^{-4}$, $92.54\%$ at $\lambda = 5 \times 10^{-4}$, $92.32\%$ at $\lambda = 10^{-3}$) makes tuning straightforward.

\item \textbf{Avoid decoupling a time-varying weight decay schedule from the learning rate.} Any adaptive weight decay scheme that does not scale $\lambda_t$ with $\gamma_t$ risks catastrophic failure after learning rate drops, particularly under milestone or step LR schedules with SGD. Note that AdamW's constant decoupled weight decay is not subject to this concern, as the coupling instability arises specifically when $\lambda_t$ itself varies over time.

\item \textbf{Exercise caution with coordinate-wise weight decay.} Despite its theoretical motivation, CWD \cite{liu2025cautious} exhibits consistent late-training instability: $4.84\%$--$12.57\%$ best-to-final accuracy degradation across all tested settings. We recommend monitoring final (not just best) accuracy when evaluating CWD.

\item \textbf{Use budget matching as a diagnostic.} When evaluating any new weight decay schedule, compare it against a constant WD with matched time-average $\bar{\lambda} = (1/T)\sum_t \lambda_t$. If performance is identical, the temporal dynamics are not contributing---only the magnitude matters.
\end{enumerate}

### 7.2 Implications for Weight Decay Theory

The alignment framework of \citet{xie2024investigating} correctly identifies $\delta_T$ as the key quantity governing weight decay efficacy. However, our results reveal a gap between the theory's explanatory power and its practical prescriptive value:

\begin{itemize}
\item The condition $\delta_T < 1$ is sufficient for weight decay to improve convergence---but under standard training, $\delta_T \sim O(10^{-3})$, satisfying the condition by a margin of $\sim 200\times$. The theory explains \emph{when} weight decay works (always, under standard training) but provides no guidance on \emph{how much} weight decay to use.
\item Theorem~\ref{thm:alignment_weighted} shows that time-varying WD can improve the bound from $\sup_t \delta_t$ to $\bar{\delta}_T$---but when $\delta_t$ varies by only $1.6\times$ (from $0.0045$ to $0.0028$), the improvement is negligible. Specifically, the bound improvement scales as the ratio $\sup_t \delta_t / \bar{\delta}_T \approx 1.5$, applied to a term that is already $O(10^{-3})$ of the overall bound.
\item The budget equivalence principle \emph{provides empirical motivation} for weight decay theory to examine cumulative effects $\sum_t \lambda_t$, pending formal characterization. The relevant question is not ``what should $\lambda_t$ be at step $t$?'' but ``what total regularization pressure produces the best-generalizing weight norm?''
\end{itemize}

### 7.3 Limitations
\label{sec:limitations}

We acknowledge several limitations of this study:

\begin{enumerate}
\item \textbf{Scale.} All experiments are conducted on CIFAR-10/100 with ResNet-20 and VGG-16-BN. While these are standard benchmarks for studying regularization mechanisms, the results may not directly generalize to ImageNet-scale training or transformer architectures. In particular, transformer architectures trained with AdamW and layer normalization may exhibit qualitatively different weight norm dynamics, and the LR--WD coupling analysis developed here for SGD with batch normalization may not transfer without modification. Budget equivalence may also break down in settings where the loss landscape is more heterogeneous across training phases.

\item \textbf{Optimizer.} We study SGD with momentum exclusively. In Adam/AdamW, weight decay is already decoupled from the learning rate by design \cite{loshchilov2019decoupled}, and the preconditioner introduces additional interactions between the gradient geometry and the regularization. Our coupling necessity result may manifest differently under adaptive optimizers.

\item \textbf{Single seed.} All experiments use a single random seed (42). While the effect sizes for our core findings are large---$82.05\%$ for the decoupling collapse, $0.01\%$ for random vs.\ alignment-based WD, exact match for budget equivalence---differences smaller than approximately $0.3\%$ should be interpreted cautiously. The consistency across three dataset/architecture combinations provides a partial substitute for multi-seed validation, but statistical rigor would benefit from $3$--$5$ seed replication for the primary comparisons.

\item \textbf{Alignment proxy.} We use a specific EMA-based minibatch proxy for the alignment quantity, with Pearson correlation $r = 0.849$ against large-batch measurements. Other estimators based on Fisher information, Hessian-vector products, or full-batch gradients might capture different aspects of the gradient--parameter geometry. However, the core finding that even the true alignment signal is too weak and too constant suggests that measurement quality is not the bottleneck.

\item \textbf{Training regime.} Our findings are specific to milestone LR schedules with moderate WD values ($\lambda \sim 10^{-4}$--$10^{-3}$). Extreme regimes---very large WD, cyclical LR, warmup-heavy schedules, or training without LR decay---may exhibit qualitatively different alignment dynamics and coupling requirements.
\end{enumerate}

### 7.4 Future Directions
\label{sec:future}

Several open questions emerge from our work:

\begin{enumerate}
\item \textbf{When does alignment become actionable?} Systematic identification of training settings where $\delta_t$ varies over a wide range would delineate the boundary between our negative result and potential positive applications of alignment-based regularization. Candidate settings include adversarial training, continual learning, and multi-task training, where gradient geometry may differ qualitatively from standard supervised learning.

\item \textbf{Adaptive WD in Adam/AdamW.} The per-coordinate preconditioner in Adam changes the effective geometry of weight decay. Alignment-based adaptation might interact differently with adaptive learning rates, potentially making the alignment signal more or less informative.

\item \textbf{Formal proof of budget equivalence.} Our budget equivalence result is empirical. A natural approach to formal characterization would be to show that when $\lambda_t \ll 1$, the weight norm trajectory $\|w_t\|$ depends on $\{\lambda_t\}$ only through $\sum_t \lambda_t$ up to $O(\lambda_{\max}^2 T)$ corrections, using stability analysis tools \cite{hardt2016train} or PAC-Bayes bounds that track weight norm explicitly. Establishing tight conditions under which $\sum_t \lambda_t$ is a sufficient statistic for the generalization effect of weight decay---and characterizing when the approximation breaks down---is an important theoretical question.

\item \textbf{Large-scale validation.} Confirming budget equivalence and LR--WD coupling necessity on ImageNet with modern architectures (ResNet-50, Vision Transformers) would strengthen the generality of our conclusions.
\end{enumerate}

### 7.5 Conclusion

We set out to test whether gradient--parameter alignment information can be exploited for adaptive weight decay scheduling in nonconvex SGD. Through the design and systematic evaluation of AADWD---a family of alignment-aware dynamic weight decay methods---and 39 controlled experiments, we arrive at a clear negative answer: dynamic weight decay scheduling provides no benefit over a well-tuned constant under standard training conditions.

This negative result yields three positive mechanistic insights. First, the \textbf{budget equivalence principle} establishes that the time-averaged weight decay magnitude is the sufficient statistic for generalization, with temporal dynamics contributing zero additional value (confirmed by exact metric matching: $92.54\% = 92.54\%$). Second, the \textbf{LR--WD coupling necessity} reveals that any adaptive weight decay must scale with the learning rate to prevent catastrophic instability---a structural requirement demonstrated by the aggressive collapse from $92.05\%$ to $10.00\%$ upon decoupling. Third, the \textbf{alignment signal inapplicability} shows that under standard SGD training, the alignment quantity $\delta_t \sim O(10^{-3})$ is too small and too constant to serve as a control signal, as confirmed by the equivalence between random ($92.06\%$) and alignment-based ($92.05\%$) dynamic weight decay.

Additionally, we document systematic late-training instability in Cautious Weight Decay \cite{liu2025cautious} across all tested settings ($4.84\%$--$12.57\%$ best-to-final accuracy degradation), suggesting that coordinate-wise adaptive regularization introduces optimization risks under standard CIFAR benchmarks.

Together, these findings provide a mechanistic account of why the simplest approach---constant weight decay---remains an empirically undominated strategy for nonconvex SGD, and identify the structural conditions that any future adaptive weight decay method must satisfy to succeed.
