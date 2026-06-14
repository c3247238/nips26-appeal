"""UDWDC-v2: Stability-fixed Unified Dynamic Weight Decay Control optimizer.

Key fixes over UDWDC v1:
  1. EMA-smooth rho_t (beta=0.99) to reduce step-to-step noise
  2. Floor clipping: lambda_t >= 0.1 * lambda_base to prevent WD collapse
  3. Diagnostic assertion: total WD budget must remain > 0

Root cause of v1 instability:
  When rho_t < rho*(t), the proportional controller drives lambda toward zero,
  effectively disabling weight decay. In ablation, Kp_only and PD_control
  variants produced ZERO total WD budget.

Algorithm:
  rho_t^l = EMA(||g_t^l|| / ||w_t^l||, beta=0.99)  # smoothed rho
  e_t^l = rho_t^l - rho*(t)
  lambda_t^l = lambda_base + K_p * e_t + K_i * EMA(e_t) - K_d * alpha_t * e_t
  lambda_t^l = clamp(lambda_t^l, 0.1 * lambda_base, lambda_max_factor * lambda_base)
"""

import torch
from .base import WDOptimizer


class UDWDCv2Optimizer(WDOptimizer):
    """Unified Dynamic Weight Decay Control v2 (stability-fixed).

    Compared to UDWDC v1:
    - EMA-smoothed rho_t reduces noisy gradient ratio fluctuations
    - Floor clipping at 0.1 * lambda_base prevents WD from collapsing to zero
    - Tracks cumulative WD budget for diagnostic verification
    """

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=1e-4,
                 K_p=0.5, K_i=0.1, K_d=0.3,
                 ema_beta=0.999, rho_ema_beta=0.99,
                 lambda_min_factor=0.1, lambda_max_factor=10.0,
                 **kwargs):
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)

        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.ema_beta = ema_beta  # for alignment and error EMA
        self.rho_ema_beta = rho_ema_beta  # for rho_t smoothing (v2 fix #1)

        # v2 fix #2: Floor clipping at lambda_min_factor * lambda_base
        self.lambda_min_factor = lambda_min_factor
        self.lambda_min = lambda_min_factor * weight_decay
        self.lambda_max = lambda_max_factor * weight_decay

        # Per-group EMA states
        self._ema_rho = {}    # v2: EMA-smoothed rho_t
        self._ema_error = {}  # EMA of control error (integral term)
        self._ema_alpha = {}  # EMA-smoothed alignment

        # rho_0 and tau computed from first step
        self._tau = None
        self._rho_0 = None
        self._initialized = False

        # v2 fix #3: Cumulative WD budget tracking for diagnostic assertion
        self._cumulative_wd_budget = 0.0
        self._epoch_wd_budget = 0.0

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
                rho_val = g_norm / w_norm
                rho_values.append(rho_val)
                # Initialize rho EMA with first observation
                name = group['layer_name']
                self._ema_rho[name] = rho_val

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
        """PID control law for per-layer WD with v2 stability fixes."""
        if not self._initialized:
            self._initialize_from_gradients()
            if not self._initialized:
                # No gradients yet, use base WD (with floor guarantee)
                for group in self.param_groups:
                    if group['apply_wd']:
                        group['effective_wd'] = max(self.base_wd, self.lambda_min)
                return

        rho_target = self._compute_rho_target()

        for group in self.param_groups:
            if not group['apply_wd']:
                group['effective_wd'] = 0.0
                continue

            name = group['layer_name']
            param = group['params'][0]

            if param.grad is None:
                group['effective_wd'] = max(self.base_wd, self.lambda_min)
                continue

            w = param.data
            g = param.grad.data

            w_norm = torch.norm(w).item()
            g_norm = torch.norm(g).item()

            # Raw rho_t = ||g|| / ||w||
            raw_rho_t = g_norm / (w_norm + self.eps)

            # v2 fix #1: EMA-smooth rho_t to reduce step-to-step noise
            if name not in self._ema_rho:
                self._ema_rho[name] = raw_rho_t
            else:
                self._ema_rho[name] = (self.rho_ema_beta * self._ema_rho[name]
                                       + (1 - self.rho_ema_beta) * raw_rho_t)
            rho_t = self._ema_rho[name]

            # Control error using smoothed rho_t
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

            # v2 fix #2: Floor clipping at lambda_min to prevent WD collapse
            lambda_t = max(self.lambda_min, min(self.lambda_max, lambda_t))

            group['effective_wd'] = lambda_t

            # Track per-step WD budget contribution
            self._epoch_wd_budget += lambda_t * (w_norm ** 2)

    def end_epoch_check(self):
        """v2 diagnostic: Assert WD budget > 0 after every epoch.

        Returns the epoch WD budget and resets the accumulator.
        Raises a warning (not exception) if budget is zero to avoid
        crashing training, but logs the issue for system monitor.
        """
        budget = self._epoch_wd_budget
        self._cumulative_wd_budget += budget
        self._epoch_wd_budget = 0.0

        if budget <= 0:
            import warnings
            warnings.warn(
                f"[UDWDC-v2] Epoch WD budget is {budget:.6e} (should be > 0). "
                f"Cumulative budget: {self._cumulative_wd_budget:.6e}. "
                f"Check lambda_min={self.lambda_min}, K_p={self.K_p}, "
                f"K_i={self.K_i}, K_d={self.K_d}."
            )

        return budget

    def get_cumulative_wd_budget(self):
        """Return total cumulative WD budget so far."""
        return self._cumulative_wd_budget + self._epoch_wd_budget

    def get_method_name(self):
        return "UDWDC-v2"
