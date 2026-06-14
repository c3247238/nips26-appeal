# Abstract

Weight decay is a ubiquitous regularizer in deep learning, yet optimal scheduling of its strength remains an open question.
Motivated by recent theoretical results connecting weight decay efficacy to the gradient--parameter alignment quantity $\delta_t = |\langle \nabla f(w_t), w_t \rangle| / (\|\nabla f(w_t)\| \|w_t\|)$, we design Alignment-Aware Dynamic Weight Decay (AADWD)---a family of three adaptive schedules that modulate $\lambda_t$ based on a stochastic alignment proxy $\hat{\delta}_t$.
Through 39 systematic experiments spanning two architectures (ResNet-20, VGG-16-BN), two datasets (CIFAR-10, CIFAR-100), and extensive ablations, we establish three negative but informative results:
**(1) Budget Equivalence**---when the time-averaged dynamic weight decay matches a constant, performance is identical ($92.54\% = 92.54\%$ on CIFAR-10/ResNet-20, with identical weight norms of 23.49);
**(2) LR--WD Coupling Necessity**---removing the learning rate multiplier from AADWD triggers catastrophic collapse ($84.49\% \to 10.00\%$, weight norm $\to 0.0036$);
**(3) Alignment Signal Inapplicability**---random dynamic weight decay ($92.06\%$) matches alignment-based weight decay ($92.05\%$), because $\delta \sim O(10^{-3})$ throughout standard training.
These findings formalize why constant weight decay remains optimal under standard nonconvex SGD and identify the structural conditions under which adaptive scheduling cannot succeed.
