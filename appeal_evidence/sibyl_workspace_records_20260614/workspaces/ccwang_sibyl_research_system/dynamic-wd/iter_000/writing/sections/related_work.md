# Related Work

## Weight Decay: Theory and Practice

Weight decay has a long history as an explicit regularizer for neural networks \cite{krogh1991simple}. Classical $\ell_2$ regularization adds the penalty $(\lambda/2)\|w\|^2$ to the loss, producing the gradient update $w_{t+1} = w_t - \gamma_t(\nabla f(w_t) + \lambda w_t) = (1 - \gamma_t \lambda) w_t - \gamma_t \nabla f(w_t)$. \citet{loshchilov2019decoupled} observe that for adaptive optimizers such as Adam, this formulation conflates the regularization effect with the preconditioned gradient, and propose decoupled weight decay (AdamW): $w_{t+1} = (1 - \lambda) w_t - \gamma_t \hat{m}_t$, where $\hat{m}_t$ is the bias-corrected first moment. For SGD with momentum, the two formulations differ only by a rescaling of $\lambda$ by $\gamma_t$, but the distinction becomes critical in our analysis of LR--WD coupling (Section~\ref{sec:coupling}).

Theoretical analyses of weight decay in nonconvex optimization have established convergence guarantees under smoothness and bounded variance assumptions \cite{zhuang2022understanding, chen2019convergence}. Most relevant to our work, \citet{xie2024investigating} prove that weight decay improves the nonconvex convergence bound if and only if the alignment quantity $\delta_T = \sup_t |\langle \nabla f(w_t), w_t \rangle| / (\|\nabla f(w_t)\| \|w_t\|) < 1$. Our study extends this framework to time-varying weight decay and empirically characterizes the alignment quantity under standard training.

The scale-free perspective on weight decay \cite{van2017l2} observes that for networks with batch normalization, weight decay does not directly regularize the effective function but controls the effective learning rate. This is complementary to our findings: if weight decay primarily controls the scale of parameters rather than providing function-space regularization, then its cumulative effect (total shrinkage) should indeed be the relevant quantity, consistent with our budget equivalence result.

## Dynamic and Adaptive Regularization

Several lines of work have explored time-varying or input-dependent regularization. Layer-wise adaptive methods such as LARS \cite{you2017large} and LAMB \cite{you2020large} scale the learning rate (and implicitly the effective weight decay) per layer based on the ratio of parameter norm to gradient norm. These methods are designed for large-batch training and address a different problem from temporal weight decay adaptation.

Cautious Weight Decay (CWD) \cite{liu2025cautious}, proposed at ICLR 2025, applies weight decay coordinate-wise, masking out dimensions where the weight decay direction conflicts with the gradient direction. While theoretically motivated by avoiding interference between regularization and optimization, our experiments reveal that CWD exhibits systematic late-training instability across multiple architectures and datasets (Section~\ref{sec:cross_arch}).

Meta-learning approaches to hyperparameter adaptation \cite{baydin2018online, franceschi2018bilevel} learn learning rate and regularization schedules through gradient-based bilevel optimization. While powerful in principle, these methods introduce substantial computational overhead and additional hyperparameters, and have not been widely adopted for weight decay scheduling in standard training pipelines.

Scheduled regularization has been explored for other techniques: curriculum dropout \cite{morerio2017curriculum} increases the dropout rate over training, and various data augmentation schedules have been proposed \cite{cubuk2020randaugment}. However, systematic investigation of weight decay scheduling---as opposed to weight decay \emph{magnitude} tuning---remains sparse in the literature.

## Gradient--Parameter Alignment in Training Dynamics

The angle between the gradient and the parameter vector has appeared in several analyses of neural network training. \citet{fort2019emergent} study the alignment between successive gradients as a diagnostic of training phase transitions. \citet{gur2018gradient} analyze the spectrum of the Hessian during training and observe that the leading eigenvector aligns with the gradient in the early phase.

In the context of implicit regularization, \citet{blanc2020implicit} show that label noise SGD biases optimization toward directions of low curvature, and \citet{li2021happens} establish connections between the gradient covariance structure and generalization. These works suggest that the geometry of the optimization trajectory---including gradient--parameter alignment---carries information about the learning dynamics.

Our work tests whether this geometric information can be \emph{exploited} for explicit regularization scheduling. The negative answer we obtain---that $\delta_t \sim O(10^{-3})$ renders alignment uninformative under standard training---does not contradict the theoretical importance of the alignment condition, but rather establishes that the condition is always easily satisfied, leaving no room for adaptive exploitation.
