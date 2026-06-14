# Discussion and Conclusion

## Practical Recommendations

Our findings yield four actionable guidelines for practitioners:

\begin{enumerate}
\item \textbf{Use constant weight decay and tune only the magnitude.} Across 39 experiments spanning multiple architectures, datasets, and dynamic scheduling strategies, no method outperforms a well-tuned constant $\lambda$. The single hyperparameter $\lambda = 5 \times 10^{-4}$ (for SGD with momentum on CIFAR-scale tasks) is sufficient. The smooth optimum landscape ($92.08\%$ at $\lambda = 10^{-4}$, $92.54\%$ at $\lambda = 5 \times 10^{-4}$, $92.32\%$ at $\lambda = 10^{-3}$) makes tuning straightforward.

\item \textbf{Avoid decoupling weight decay from the learning rate in custom schedules.} Any adaptive weight decay scheme that does not scale $\lambda_t$ with $\gamma_t$ risks catastrophic failure after learning rate drops. This applies to any hand-designed or learned WD schedule used with milestone, step, or cosine LR schedules.

\item \textbf{Exercise caution with coordinate-wise weight decay.} Despite its theoretical motivation, CWD \cite{liu2025cautious} exhibits consistent late-training instability: $4.84\%$--$12.57\%$ best-to-final accuracy degradation across all tested settings. We recommend monitoring final (not just best) accuracy when evaluating CWD.

\item \textbf{Use budget matching as a diagnostic.} When evaluating any new weight decay schedule, compare it against a constant WD with matched time-average $\bar{\lambda} = (1/T)\sum_t \lambda_t$. If performance is identical, the temporal dynamics are not contributing---only the magnitude matters.
\end{enumerate}

## Implications for Weight Decay Theory

The alignment framework of \citet{xie2024investigating} correctly identifies $\delta_T$ as the key quantity governing weight decay efficacy. However, our results reveal a gap between the theory's explanatory power and its practical prescriptive value:

\begin{itemize}
\item The condition $\delta_T < 1$ is necessary and sufficient for weight decay to improve convergence---but under standard training, $\delta_T \sim O(10^{-3})$, satisfying the condition by a margin of $\sim 200\times$. The theory explains \emph{when} weight decay works (always, under standard training) but provides no guidance on \emph{how much} weight decay to use.
\item Theorem~\ref{thm:alignment_weighted} shows that time-varying WD can improve the bound from $\sup_t \delta_t$ to $\bar{\delta}_T$---but when $\delta_t$ varies by only $1.6\times$, the improvement is negligible.
\item The budget equivalence principle suggests that weight decay theory should shift focus from instantaneous rates $\lambda_t$ to cumulative effects $\sum_t \lambda_t$. The relevant question is not ``what should $\lambda_t$ be at step $t$?'' but ``what total regularization pressure produces the best-generalizing weight norm?''
\end{itemize}

## Limitations

We acknowledge several limitations of this study:

\begin{enumerate}
\item \textbf{Scale.} All experiments are conducted on CIFAR-10/100 with ResNet-20 and VGG-16-BN. While these are standard benchmarks for studying regularization mechanisms, the results may not directly generalize to ImageNet-scale training or transformer architectures. Budget equivalence, in particular, may break down in settings where the loss landscape is more heterogeneous across training phases.

\item \textbf{Optimizer.} We study SGD with momentum exclusively. In Adam/AdamW, weight decay is already decoupled from the learning rate by design \cite{loshchilov2019decoupled}, and the preconditioner introduces additional interactions between the gradient geometry and the regularization. Our coupling necessity result may manifest differently under adaptive optimizers.

\item \textbf{Single seed.} Most experiments use a single random seed (42). While the effect sizes for our core findings are large---$82.05\%$ for the decoupling collapse, $0.01\%$ for random vs.\ alignment-based WD, exact match for budget equivalence---statistical rigor would benefit from $3$--$5$ seed replication for the primary comparisons. We note that the consistency across three dataset/architecture combinations provides a partial substitute for multi-seed validation.

\item \textbf{Alignment proxy.} We use a specific EMA-based minibatch proxy for the alignment quantity. Other estimators based on Fisher information, Hessian-vector products, or full-batch gradients might capture different aspects of the gradient--parameter geometry. However, the high Pearson correlation ($r = 0.849$) between our proxy and large-batch measurements, combined with the negative result that even the true signal is too weak, suggests that measurement quality is not the bottleneck.

\item \textbf{Training regime.} Our findings are specific to milestone LR schedules with moderate WD values ($\lambda \sim 10^{-4}$--$10^{-3}$). Extreme regimes---very large WD, cyclical LR, warmup-heavy schedules, or training without LR decay---may exhibit qualitatively different alignment dynamics and coupling requirements.
\end{enumerate}

## Future Directions

Several open questions emerge from our work:

\begin{enumerate}
\item \textbf{When does alignment become actionable?} Systematic identification of training settings where $\delta_t$ varies over a wide range would delineate the boundary between our negative result and potential positive applications of alignment-based regularization.

\item \textbf{Adaptive WD in Adam/AdamW.} The per-coordinate preconditioner in Adam changes the effective geometry of weight decay. Alignment-based adaptation might interact differently with adaptive learning rates, potentially making the alignment signal more or less informative.

\item \textbf{Formal proof of budget equivalence.} Our budget equivalence result is empirical. Establishing tight theoretical conditions under which $\sum_t \lambda_t$ is a sufficient statistic for the generalization effect of weight decay---and characterizing when the approximation breaks down---is an important theoretical question.

\item \textbf{Large-scale validation.} Confirming budget equivalence and LR--WD coupling necessity on ImageNet with modern architectures (ResNet-50, Vision Transformers) would strengthen the generality of our conclusions.
\end{enumerate}

## Conclusion

We set out to test whether gradient--parameter alignment information can be exploited for adaptive weight decay scheduling in nonconvex SGD. Through the design and systematic evaluation of AADWD---a family of alignment-aware dynamic weight decay methods---and 39 controlled experiments, we arrive at a clear negative answer: dynamic weight decay scheduling provides no benefit over a well-tuned constant under standard training conditions.

This negative result yields three positive mechanistic insights. First, the \textbf{budget equivalence principle} establishes that the time-averaged weight decay magnitude is the sufficient statistic for generalization, with temporal dynamics contributing zero additional value (confirmed by exact metric matching: $92.54\% = 92.54\%$). Second, the \textbf{LR--WD coupling necessity} reveals that any adaptive weight decay must scale with the learning rate to prevent catastrophic instability---a structural requirement demonstrated by the aggressive collapse from $92.05\%$ to $10.00\%$ upon decoupling. Third, the \textbf{alignment signal inapplicability} shows that under standard SGD training, the alignment quantity $\delta_t \sim O(10^{-3})$ is too small and too constant to serve as a control signal, as confirmed by the equivalence between random ($92.06\%$) and alignment-based ($92.05\%$) dynamic weight decay.

Together, these findings provide a mechanistic account of why the simplest approach---constant weight decay---remains the best approach for nonconvex SGD, and identify the structural conditions that any future adaptive weight decay method must satisfy to succeed.
