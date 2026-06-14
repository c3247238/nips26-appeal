"""Defazio Corrective Weight Decay (arXiv:2506.02285, 2025).

LR-proportional corrective WD: adjusts WD to maintain constant EMA timescale
tau = 1 / (lambda * LR) as LR changes during training.

In unified framework: K_p > 0, K_i = 0, K_d = 0 with rho*(t) = eta_t / tau.
"""

import torch
from .base import WDOptimizer


class DefazioCorrective(WDOptimizer):
    """Defazio's corrective weight decay.

    Key insight: under a constant LR, WD drives per-layer gradient-to-weight
    ratios to a well-defined steady state. When LR changes (schedule), this
    steady state shifts. The corrective term adjusts WD to maintain the
    original steady state.

    lambda_t = lambda_base * (eta_t / eta_0)
    This ensures the product lambda * LR stays constant, maintaining the EMA
    timescale tau = 1 / (lambda * eta).
    """

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=1e-4, **kwargs):
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        self._initial_lr = lr

    def _compute_effective_wd(self):
        # lambda_t = lambda_base * (eta_t / eta_0)
        # This keeps lambda * eta = lambda_base * eta_0 = constant
        lr_ratio = self.current_lr / (self._initial_lr + self.eps)

        for group in self.param_groups:
            if group['apply_wd']:
                group['effective_wd'] = self.base_wd * lr_ratio
            else:
                group['effective_wd'] = 0.0

    def get_method_name(self):
        return "DefazioCorrective"
