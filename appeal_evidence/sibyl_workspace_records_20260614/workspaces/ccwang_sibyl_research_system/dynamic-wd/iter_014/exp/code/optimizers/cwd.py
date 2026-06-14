"""CWD: Cautious Weight Decay (Chen et al., ICLR 2026).

Binary sign-alignment mask: apply WD only when gradient and weight
point in different directions (negative alignment).

In unified framework: K_p=0, K_i=0, K_d>0 with binary alpha.
"""

import torch
from .base import WDOptimizer


class CWDOptimizer(WDOptimizer):
    """Cautious Weight Decay.

    lambda_t^l = lambda_base * mask_t^l
    mask_t^l = 1 if sign(g) == sign(w) else 0 (element-wise, then averaged)

    More precisely: CWD applies WD only to parameters where gradient and weight
    have opposite signs (negative cosine alignment at element level).
    """

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=1e-4, **kwargs):
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        # Track per-group alignment mask ratio for diagnostics
        self._mask_ratios = {}

    def _apply_weight_decay(self):
        """Override to apply element-wise masked WD."""
        for group in self.param_groups:
            if not group['apply_wd']:
                continue
            wd = self.base_wd
            if wd == 0:
                continue
            for p in group['params']:
                if p.data is None or p.grad is None:
                    continue
                # CWD mask: decay when gradient opposes weight direction
                # sign(g) != sign(w) means they point in opposite directions
                # This means the gradient is pushing the weight toward zero
                # CWD: apply WD only when gradient aligns with weight (same sign)
                # i.e., both gradient and weight want to move in the same direction
                mask = (p.grad.data * p.data > 0).float()
                self._mask_ratios[group['layer_name']] = mask.mean().item()
                p.data.mul_(1 - self.current_lr * wd * mask)

    def _compute_effective_wd(self):
        """Effective WD is base_wd modulated by mask ratio."""
        for group in self.param_groups:
            if group['apply_wd']:
                name = group['layer_name']
                ratio = self._mask_ratios.get(name, 0.5)
                group['effective_wd'] = self.base_wd * ratio

    def get_method_name(self):
        return "CWD"
