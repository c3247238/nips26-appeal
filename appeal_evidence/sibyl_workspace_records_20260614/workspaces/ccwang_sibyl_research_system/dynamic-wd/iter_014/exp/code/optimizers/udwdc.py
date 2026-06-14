"""UDWDC: Unified Dynamic Weight Decay Control optimizer.

Implements the PID-style controller:
  lambda_t^l = lambda_base + K_p * e_t^l + K_i * EMA(e_t^l) - K_d * alpha_t^l * e_t^l
  e_t^l = rho_t^l - rho*(t)
  rho*(t) = eta_t / tau
  tau = rho_0 / eta_0
"""

import torch
from .base import WDOptimizer


class UDWDCOptimizer(WDOptimizer):
    """Unified Dynamic Weight Decay Control.

    Full PID controller with alignment modulation in the unified framework.
    K_p > 0, K_i > 0, K_d > 0 with continuous cosine alignment.
    """

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=1e-4,
                 K_p=0.5, K_i=0.1, K_d=0.3,
                 ema_beta=0.999, lambda_min=0.0, lambda_max_factor=10.0,
                 **kwargs):
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)

        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.ema_beta = ema_beta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max_factor * weight_decay

        # Per-group EMA of control error (for integral term)
        self._ema_error = {}
        # Per-group EMA of alignment (for smoothing)
        self._ema_alpha = {}

        # rho_0 and tau will be computed from first step's gradient statistics
        self._tau = None
        self._rho_0 = None
        self._initialized = False

    def _initialize_from_gradients(self):
        """Compute tau from initial gradient statistics: tau = rho_0 / eta_0."""
        rho_values = []
        for group in self.param_groups:
            if not group['apply_wd']:
                continue
            param = group['params'][0]
            if param.grad is None:
                continue
            w_norm = torch.norm(param.data).item()
            g_norm = torch.norm(param.grad.data).item()
            if w_norm > self.eps:
                rho_values.append(g_norm / w_norm)

        if rho_values:
            self._rho_0 = sum(rho_values) / len(rho_values)
            self._tau = self._rho_0 / (self.base_lr + self.eps)
            self._initialized = True

    def _compute_rho_target(self) -> float:
        """Compute target ratio rho*(t) = eta_t / tau."""
        if self._tau is None or self._tau < self.eps:
            return self.base_wd  # fallback
        return self.current_lr / self._tau

    def _compute_effective_wd(self):
        """PID control law for per-layer WD."""
        if not self._initialized:
            self._initialize_from_gradients()
            if not self._initialized:
                # No gradients yet, use base WD
                for group in self.param_groups:
                    if group['apply_wd']:
                        group['effective_wd'] = self.base_wd
                return

        rho_target = self._compute_rho_target()

        for group in self.param_groups:
            if not group['apply_wd']:
                group['effective_wd'] = 0.0
                continue

            name = group['layer_name']
            param = group['params'][0]

            if param.grad is None:
                group['effective_wd'] = self.base_wd
                continue

            w = param.data
            g = param.grad.data

            w_norm = torch.norm(w).item()
            g_norm = torch.norm(g).item()

            # rho_t = ||g|| / ||w||
            rho_t = g_norm / (w_norm + self.eps)

            # Control error
            e_t = rho_t - rho_target

            # Alignment cosine: alpha_t = cos(g, w)
            dot = torch.dot(w.flatten(), g.flatten()).item()
            alpha_t = dot / (w_norm * g_norm + self.eps)

            # EMA-smoothed alignment
            if name not in self._ema_alpha:
                self._ema_alpha[name] = alpha_t
            else:
                self._ema_alpha[name] = (self.ema_beta * self._ema_alpha[name]
                                         + (1 - self.ema_beta) * alpha_t)
            smooth_alpha = self._ema_alpha[name]

            # EMA of control error (integral term)
            if name not in self._ema_error:
                self._ema_error[name] = e_t
            else:
                self._ema_error[name] = (self.ema_beta * self._ema_error[name]
                                         + (1 - self.ema_beta) * e_t)
            ema_e = self._ema_error[name]

            # PID control law
            lambda_t = (self.base_wd
                        + self.K_p * e_t
                        + self.K_i * ema_e
                        - self.K_d * smooth_alpha * e_t)

            # Clipping
            lambda_t = max(self.lambda_min, min(self.lambda_max, lambda_t))

            group['effective_wd'] = lambda_t

    def get_method_name(self):
        return "UDWDC"
