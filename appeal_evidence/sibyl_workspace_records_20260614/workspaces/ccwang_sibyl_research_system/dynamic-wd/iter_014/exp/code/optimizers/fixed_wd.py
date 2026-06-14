"""Fixed (constant) weight decay baseline."""

from .base import WDOptimizer


class FixedWDOptimizer(WDOptimizer):
    """Standard SGDW/AdamW with constant lambda.

    K_p=0, K_i=0, K_d=0 in the unified framework.
    """

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=1e-4, **kwargs):
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)

    def _compute_effective_wd(self):
        for group in self.param_groups:
            if group['apply_wd']:
                group['effective_wd'] = self.base_wd

    def get_method_name(self):
        return "FixedWD"
