"""No weight decay baseline (null baseline)."""

from .base import WDOptimizer


class NoWDOptimizer(WDOptimizer):
    """No weight decay at all. Null baseline."""

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=0.0, **kwargs):
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=0.0, **kwargs)

    def _compute_effective_wd(self):
        for group in self.param_groups:
            group['effective_wd'] = 0.0

    def get_method_name(self):
        return "NoWD"
