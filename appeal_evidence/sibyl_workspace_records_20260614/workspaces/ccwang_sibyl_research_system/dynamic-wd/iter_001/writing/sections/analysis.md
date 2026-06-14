# Analysis: Why Dynamic Weight Decay Fails

The experimental results in Section~5 establish that no dynamic weight decay schedule outperforms a well-tuned constant. In this section, we synthesize three mechanistic insights that explain \emph{why} dynamic weight decay scheduling is ineffective under standard nonconvex SGD training.

## Insight 1: The Budget Equivalence Principle

\paragraph{Argument.} Weight decay's primary regularization mechanism is controlling the weight norm trajectory $\|w_t\|$ during training. The weight norm at convergence determines the model's effective capacity and, through its connection to the loss landscape geometry, its generalization properties \cite{neyshabur2015norm, bartlett2017spectrally}. We argue that the final weight norm is determined by the \emph{total regularization pressure} $\sum_t \lambda_t$, not by how this pressure is distributed over time.

To see this formally, consider the weight norm evolution under SGDW. Squaring both sides of the update (\ref{eq:sgdw}) and taking expectations:
\begin{equation}
\mathbb{E}[\|w_{t+1}\|^2] = (1 - \lambda_t)^2 \mathbb{E}[\|w_t\|^2] - 2\gamma_t(1-\lambda_t)\mathbb{E}[\langle \nabla f_S(w_t), w_t \rangle] + \gamma_t^2 \mathbb{E}[\|g_t\|^2].
\end{equation}
For small $\lambda_t$ (which is the regime of practical interest, as $\lambda \sim 10^{-4}$--$10^{-3}$):
\begin{equation}
(1 - \lambda_t)^2 \approx 1 - 2\lambda_t, \qquad \prod_{t=1}^T (1 - 2\lambda_t) \approx \exp\!\left(-2\sum_{t=1}^T \lambda_t\right).
\end{equation}
The exponential factor depends only on $\sum_t \lambda_t$, not on the temporal distribution $\{\lambda_t\}_{t=1}^T$. The remaining terms involve the gradient--parameter inner product $\langle \nabla f_S(w_t), w_t \rangle$ and the gradient norm $\|g_t\|^2$, which depend on the optimization trajectory. However, when two weight decay schedules with the same cumulative budget produce similar loss landscapes---as they do when the budget is the dominant factor---the trajectory-dependent terms are also approximately matched.

\paragraph{Experimental confirmation.} The budget equivalence experiment (Section~\ref{sec:budget}) provides direct evidence: the constant WD with $\bar{\lambda} = \text{mean}(\lambda_t^{\text{AADWD-agg}})$ achieves $92.54\%$ accuracy and weight norm $23.49$---\emph{exactly} matching fixed WD on all metrics. The AADWD-Aggressive schedule with the same cumulative budget achieves only $92.05\%$, the $0.49\%$ deficit arising from suboptimal temporal allocation (too much regularization early, too little late).

\paragraph{Practical implication.} There is no ``optimal schedule'' for weight decay beyond choosing the right average magnitude. The single hyperparameter $\lambda$ is sufficient. This is analogous to the observation that, for $\ell_2$ regularization in linear regression, only the regularization strength (not its temporal application) matters---but extends this intuition to the nonconvex deep learning setting.

## Insight 2: LR Coupling as a Stability Mechanism

\paragraph{Argument.} The learning rate and weight decay in SGD are not independent hyperparameters---they interact through a multiplicative coupling that is essential for training stability. In standard $\ell_2$ regularization, the effective weight decay is $\gamma_t \cdot \lambda$, which automatically decreases as the learning rate decreases. This ensures that the regularization term remains proportional to the optimization step size: as the learner takes smaller steps (lower LR), it also applies less shrinkage (lower effective WD), maintaining a balanced ratio between learning and regularization.

Decoupled weight decay \cite{loshchilov2019decoupled} breaks this coupling deliberately for AdamW, applying a constant $\lambda$ regardless of the learning rate. This works precisely because $\lambda$ is \emph{constant} and is tuned jointly with the LR schedule---the practitioner implicitly accounts for the LR--WD interaction by choosing $\lambda$ in the context of the planned LR trajectory.

AADWD introduces a \emph{time-varying} $\lambda_t$ that depends on the alignment state $\hat{\delta}_t$. If this $\lambda_t$ is not scaled by $\gamma_t$, it can become arbitrarily large relative to the gradient signal after a learning rate drop:

\begin{equation}
\text{Effective ratio} = \frac{\lambda_t}{\gamma_t} \xrightarrow[\gamma_t \downarrow]{} \infty \quad \text{(decoupled, $\lambda_t$ stays constant)}.
\end{equation}

\paragraph{The collapse mechanism.} The aggressive decoupled experiment (Section~\ref{sec:decoupling}) demonstrates this concretely. Before the first milestone (epoch 80), the effective ratio $\lambda_t / \gamma_t$ is well-behaved. At the milestone, $\gamma_t$ drops $10\times$ but $\lambda_t$ does not, causing the ratio to jump $10\times$. Weight decay now dominates the dynamics: in the update $w_{t+1} = (1 - \lambda_t) w_t - \gamma_t g_t$, the shrinkage term $(1 - \lambda_t) w_t$ overwhelms the gradient term $\gamma_t g_t$. The weight norm begins to collapse, which alters $\hat{\delta}_t$, which further increases $\lambda_t$ (in the aggressive formula), creating a runaway feedback loop. The final weight norm of $0.0036$---a factor of $\sim 6{,}000\times$ smaller than the coupled case ($21.47$)---and the accuracy of $10.00\%$ (random chance) confirm complete training failure.

The conservative decoupled variant avoids catastrophic collapse because $(1 - \hat{\delta}_t)$ is bounded in $[0, 1]$, limiting $\lambda_t$. But it still suffers severe degradation ($80.30\%$ vs.\ $92.37\%$) because the constant $\lambda \approx 5 \times 10^{-4}$ is $10\times$ too strong when $\gamma_t = 0.01$ and $100\times$ too strong when $\gamma_t = 0.001$.

\paragraph{Structural lesson.} Any adaptive weight decay scheme must satisfy the condition $\lambda_t = O(\gamma_t)$ to prevent the regularizer from overwhelming the gradient signal after learning rate drops. This is not a design preference but a structural requirement for stable training under milestone-style LR schedules.

## Insight 3: Alignment is a Consequence, Not a Control Signal

\paragraph{Argument.} The theoretical result of \citet{xie2024investigating}---that $\delta_T < 1$ is necessary and sufficient for weight decay to improve convergence---does not imply that monitoring $\delta_t$ during training provides actionable information for weight decay scheduling. The key distinction is between a \emph{condition} (which defines when something works) and a \emph{control signal} (which tells you how to adjust a parameter in real time).

For $\delta_t$ to serve as a useful control signal, it must satisfy two requirements: (a) it must vary significantly during training, so that different WD rates are appropriate at different times; and (b) the variation must carry information that is not already captured by the learning rate schedule. Our experiments show that neither condition holds.

\paragraph{Condition (a): Insufficient variation.} The alignment proxy $\hat{\delta}_t$ ranges over $[0.0028, 0.0045]$ during 200 epochs of training (Table~\ref{tab:alignment})---a $1.6\times$ variation. The alignment condition $\delta_T < 1$ is satisfied by a factor of $\sim 200\times$ at every point in training. There is no phase where $\delta_t$ approaches 1 and weight decay becomes counterproductive. The theoretical framework correctly identifies $\delta_T$ as the key quantity, but under standard training, this quantity is so far from the critical threshold that it carries no scheduling information.

\paragraph{Condition (b): No information beyond LR.} The random dynamic WD experiment provides the decisive test. By replacing $\hat{\delta}_t$ with $U(0,1)$---removing all alignment information---we obtain $92.06\%$ accuracy, essentially identical to AADWD-Aggressive's $92.05\%$. The $0.01\%$ difference is well within noise. This confirms that any time-varying WD with the appropriate budget and LR coupling performs comparably, regardless of whether it uses alignment information.

\paragraph{When might alignment matter?} Our negative result is specific to standard SGD training on CIFAR-level tasks with milestone LR schedules. We conjecture that alignment-based WD adaptation could be beneficial under conditions that produce larger $\delta_t$ variation:
\begin{enumerate}
\item \textbf{Non-standard architectures or training regimes} where the optimization trajectory explores regions of parameter space with high gradient--parameter alignment (e.g., adversarial training, where gradients are manipulated to point in specific directions).
\item \textbf{Much larger weight decay values} approaching the instability boundary ($\delta_T \to 1$), where the margin between beneficial and harmful regularization is thin.
\item \textbf{Highly nonconvex landscapes} (e.g., early training of very deep networks without skip connections) where $\delta_t$ may fluctuate significantly before stabilizing.
\end{enumerate}
Identifying these settings empirically remains an open question for future work.
