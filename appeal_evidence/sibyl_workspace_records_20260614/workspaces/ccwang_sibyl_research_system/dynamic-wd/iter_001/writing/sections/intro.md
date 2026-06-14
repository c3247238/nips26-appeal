# Introduction

## Motivation and Problem Statement

Weight decay is the most widely used explicit regularizer in modern deep learning \cite{loshchilov2019decoupled, zhang2019algorithmic}. In standard practice, the weight decay coefficient $\lambda$ is held constant throughout training and tuned as a single scalar hyperparameter---typically set to $5 \times 10^{-4}$ for SGD-based image classification \cite{he2016deep}. Despite the simplicity of this approach, constant weight decay consistently delivers competitive generalization across a broad range of architectures and datasets.

A natural question arises: \emph{can weight decay be scheduled dynamically---stronger when regularization helps, weaker when it hurts?} Learning rate scheduling has long been recognized as essential for training deep networks \cite{smith2017cyclical, loshchilov2017sgdr}, and analogous schedules for other hyperparameters---dropout rate, label smoothing, data augmentation strength---have shown varying degrees of success \cite{morerio2017curriculum, zhang2018mixup}. Yet surprisingly little work has systematically investigated whether temporal adaptation of the weight decay coefficient provides any benefit beyond tuning its constant magnitude.

Recent theoretical advances provide a principled basis for this investigation. In particular, work on the role of weight decay in enhancing nonconvex SGD \cite{xie2024investigating} identifies the alignment quantity
\begin{equation}
\delta_t = \frac{|\langle \nabla f_S(w_t), w_t \rangle|}{\|\nabla f_S(w_t)\| \cdot \|w_t\|}
\end{equation}
as the key condition governing when weight decay improves generalization. Specifically, the condition $\delta_T = \sup_t \delta_t < 1$ is necessary and sufficient for weight decay to reduce the convergence bound. If $\delta_t$ varies over the course of training---being large in some phases and small in others---then the weight decay rate should, in principle, vary accordingly to exploit this structure.

## Our Approach: Alignment-Aware Dynamic Weight Decay

We design Alignment-Aware Dynamic Weight Decay (AADWD), a family of three adaptive weight decay schedules that modulate $\lambda_t$ based on a stochastic exponential moving average (EMA) proxy $\hat{\delta}_t$ of the alignment quantity. The three variants differ in how aggressively they respond to the alignment signal:

\begin{itemize}
\item \textbf{Conservative:} $\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t \cdot (1 - \hat{\delta}_t),\, \lambda_{\min},\, \lambda_{\max}\bigr)$, which increases weight decay when alignment is low (gradients and parameters are misaligned).
\item \textbf{Aggressive:} $\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t / (\hat{\delta}_t + \epsilon),\, \lambda_{\min},\, \lambda_{\max}\bigr)$, which applies weight decay inversely proportional to alignment.
\item \textbf{Square:} $\lambda_t = \mathrm{clip}\bigl(c \cdot \gamma_t^2 \cdot (1 - \hat{\delta}_t),\, \lambda_{\min},\, \lambda_{\max}\bigr)$, which satisfies the $O(\gamma_t^2)$ condition required for standard convergence guarantees.
\end{itemize}

All three variants include a learning rate multiplier $\gamma_t$ that couples the weight decay strength to the current learning rate---a design choice whose necessity our experiments will reveal to be far more consequential than the alignment signal itself.

AADWD is not proposed as a practical optimization method. Rather, it is designed as an \emph{experimental framework}---a controlled probe to test whether alignment information is actionable for weight decay scheduling, and to isolate the mechanisms that govern the interaction between weight decay, learning rate, and training dynamics.

## Summary of Findings and Contributions

Through 39 systematic experiments spanning ResNet-20, VGG-16-BN, CIFAR-10, and CIFAR-100, we establish three core findings, each stated as a positive mechanistic insight derived from a negative empirical result:

\paragraph{1. Weight Decay Budget Equivalence.}
We demonstrate that the time-averaged weight decay magnitude---not its temporal distribution---determines generalization performance. When the cumulative weight decay budget $\bar{\lambda} = (1/T)\sum_{t=1}^{T} \lambda_t$ of a dynamic schedule is matched by a constant $\lambda = \bar{\lambda}$, the two yield identical performance. On CIFAR-10/ResNet-20, the budget-equivalent constant weight decay achieves $92.54\%$ test accuracy with weight norm $23.49$---exactly matching the original fixed weight decay baseline ($92.54\%$, weight norm $23.49$). This result implies that weight decay scheduling has \emph{zero marginal value} when the average magnitude is held constant.

\paragraph{2. LR--WD Coupling is a Structural Necessity.}
The learning rate multiplier $\gamma_t$ in AADWD is not a convenient design choice but a stability requirement. Removing it produces catastrophic failure: the aggressive variant collapses from $92.05\%$ to $10.00\%$ test accuracy (random chance on CIFAR-10), with the weight norm shrinking to $0.0036$ through a positive feedback loop. Even the conservative variant degrades from $92.37\%$ to $80.30\%$. This establishes that any adaptive weight decay scheme operating under milestone learning rate schedules must scale with the learning rate to prevent the regularizer from overwhelming the gradient signal after learning rate drops.

\paragraph{3. Alignment Signal is Not Actionable Under Standard SGD.}
The alignment proxy $\hat{\delta}_t$ remains at $O(10^{-3})$ throughout training (range $[0.0028, 0.0045]$, standard deviation $7.5 \times 10^{-4}$), rendering it effectively constant. As a consequence, random dynamic weight decay---where $\lambda_t = c \cdot \gamma_t \cdot U(0,1)$ with no alignment information---achieves $92.06\%$, statistically indistinguishable from alignment-based AADWD aggressive at $92.05\%$. The alignment signal carries no exploitable information for weight decay scheduling under standard training conditions.

\paragraph{Additional contributions.} We provide systematic cross-architecture evidence of late-training instability in Cautious Weight Decay \cite{liu2025cautious} (CWD), a recently proposed coordinate-wise adaptive method. CWD exhibits consistent best-to-final accuracy degradation across all three experimental settings: $91.79\% \to 86.95\%$ on CIFAR-10/ResNet-20 ($\Delta = 4.84\%$), $66.84\% \to 54.27\%$ on CIFAR-100/ResNet-20 ($\Delta = 12.57\%$), and $92.95\% \to 86.47\%$ on CIFAR-10/VGG-16-BN ($\Delta = 6.48\%$).

Taken together, these findings provide a mechanistic account of why constant weight decay remains the optimal strategy for nonconvex SGD: the alignment condition is always easily satisfied ($\delta \ll 1$), the cumulative regularization budget is the sufficient statistic for generalization, and the learning rate schedule already provides the necessary temporal structure. Dynamic weight decay scheduling adds complexity without benefit.
