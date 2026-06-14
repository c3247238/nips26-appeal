"""CPR: Constrained Parameter Regularization (Franke et al., NeurIPS 2024).

Augmented Lagrangian approach: drives per-matrix weight norm toward a target
(initialized from the weight norm at a reference point, typically initialization).

In unified framework: K_p > 0, K_i > 0, K_d = 0 with per-matrix rho*.
"""

import torch
from .base import WDOptimizer


class CPROptimizer(WDOptimizer):
    """Constrained Parameter Regularization.

    Uses an augmented Lagrangian to constrain each parameter group's norm:
      lambda_t^l = mu * max(0, ||w_t^l|| - kappa * ||w_0^l||)
    where kappa is the norm budget factor and mu is the penalty coefficient.

    The Lagrangian multiplier (dual variable) accumulates constraint violations,
    effectively providing integral control.
    """

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=1e-4,
                 kappa=1.0, mu=1.0, mu_lr=0.01, **kwargs):
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        self.kappa = kappa  # norm budget factor
        self.mu = mu  # initial penalty coefficient
        self.mu_lr = mu_lr  # dual variable learning rate

        # Store initial norms and dual variables
        self._init_norms = {}
        self._dual_vars = {}

        for group in self.param_groups:
            if group['apply_wd']:
                name = group['layer_name']
                param = group['params'][0]
                self._init_norms[name] = torch.norm(param.data).item()
                self._dual_vars[name] = 0.0  # Lagrangian multiplier

    def _compute_effective_wd(self):
        for group in self.param_groups:
            if not group['apply_wd']:
                group['effective_wd'] = 0.0
                continue

            name = group['layer_name']
            param = group['params'][0]

            w_norm = torch.norm(param.data).item()
            target_norm = self.kappa * self._init_norms[name]

            # Constraint violation
            violation = max(0.0, w_norm - target_norm)

            # Update dual variable (integral control)
            self._dual_vars[name] += self.mu_lr * violation

            # Effective WD from augmented Lagrangian
            if violation > 0:
                # Gradient of penalty: mu * violation * (w / ||w||) -> effective WD
                lambda_t = (self._dual_vars[name] + self.mu * violation) / (w_norm + self.eps)
            else:
                lambda_t = 0.0

            # Clip
            lambda_t = max(0.0, min(10.0 * self.base_wd, lambda_t))

            group['effective_wd'] = lambda_t

    def get_method_name(self):
        return "CPR"
