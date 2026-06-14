"""SWD: Scheduled Weight Decay (gradient-norm-aware, Xie et al., NeurIPS 2023).

Dynamically adjusts WD based on gradient norm relative to a running average.

In unified framework: K_p > 0, K_i > 0, K_d = 0 with gradient-norm-based rho*.
"""

import torch
from .base import WDOptimizer


class SWDOptimizer(WDOptimizer):
    """Scheduled Weight Decay.

    lambda_t = lambda_base * (||g_t|| / EMA(||g||))

    Scales WD proportionally to the ratio of current gradient norm to its
    running average. When gradients spike, WD increases; when gradients are
    small, WD decreases.
    """

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=1e-4,
                 ema_beta=0.999, **kwargs):
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        self.ema_beta = ema_beta
        self._ema_gnorm = {}  # per-group EMA of gradient norm

    def _compute_effective_wd(self):
        for group in self.param_groups:
            if not group['apply_wd']:
                group['effective_wd'] = 0.0
                continue

            name = group['layer_name']
            param = group['params'][0]

            if param.grad is None:
                group['effective_wd'] = self.base_wd
                continue

            g_norm = torch.norm(param.grad.data).item()

            # Update EMA of gradient norm
            if name not in self._ema_gnorm:
                self._ema_gnorm[name] = g_norm
            else:
                self._ema_gnorm[name] = (self.ema_beta * self._ema_gnorm[name]
                                         + (1 - self.ema_beta) * g_norm)

            ema_g = self._ema_gnorm[name]

            # Scale WD by ratio of current to average gradient norm
            ratio = g_norm / (ema_g + self.eps)
            lambda_t = self.base_wd * ratio

            # Clip to reasonable range
            lambda_t = max(0.0, min(10.0 * self.base_wd, lambda_t))

            group['effective_wd'] = lambda_t

    def get_method_name(self):
        return "SWD"
