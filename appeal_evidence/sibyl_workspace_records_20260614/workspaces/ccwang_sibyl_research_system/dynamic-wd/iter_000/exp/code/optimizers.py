"""
Alignment-Aware Dynamic Weight Decay (AADWD) optimizer and baselines.

Implements:
- AADWD (Conservative, Aggressive, Square variants)
- CWD (Cautious Weight Decay)
- Standard SGD with weight decay (fixed, stagewise, no-WD)
- Ablation controls (Random-Dynamic-WD, Norm-Matched-WD, Equivalent-Cumulative-WD)
"""

import math
import torch
from torch.optim import SGD


class AADWDOptimizer:
    """Alignment-Aware Dynamic Weight Decay wrapper around SGD.

    Computes delta_hat_t (cosine alignment between gradient and parameters),
    applies EMA smoothing, and dynamically adjusts weight decay per step.

    Variants (coupled, gamma_t = current LR):
        conservative: lambda_t = clip(c * gamma_t * (1 - ema_delta), min, max)
        aggressive:   lambda_t = clip(c * gamma_t * ema_delta, min, max)
        square:       lambda_t = clip(c * gamma_t^2 * (1 - ema_delta), min, max)

    Decoupled variants (decouple_lr=True, removes gamma_t dependency):
        conservative: lambda_t = clip(c * (1 - ema_delta), min, max)
        aggressive:   lambda_t = clip(c * ema_delta, min, max)
    """

    def __init__(self, params, lr=0.1, momentum=0.9, c=1.0, beta=0.99,
                 lambda_min=1e-6, lambda_max=0.01, epsilon=1e-8,
                 variant="conservative", decouple_lr=False):
        assert variant in ("conservative", "aggressive", "square")
        self.c = c
        self.beta = beta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.epsilon = epsilon
        self.variant = variant
        self.decouple_lr = decouple_lr
        self.ema_delta = 0.0  # start from zero; EMA warms up naturally
        self.current_lambda = 0.0
        self.current_delta_hat = 0.0
        self.step_count = 0

        # Use SGD without built-in weight decay; we apply our own
        self.optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=0.0)
        # Cache device for vectorized alignment computation
        self._device = next(iter(self.optimizer.param_groups[0]['params'])).device

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _compute_alignment(self):
        """Compute delta_hat_t = |<g, w>| / (||g|| * ||w|| + eps).

        Vectorized: single GPU sync instead of per-parameter .item() calls.
        """
        dot_gw = torch.tensor(0.0, device=self._device)
        norm_g_sq = torch.tensor(0.0, device=self._device)
        norm_w_sq = torch.tensor(0.0, device=self._device)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data.flatten()
                w = p.data.flatten()
                dot_gw += torch.dot(g, w)
                norm_g_sq += torch.dot(g, g)
                norm_w_sq += torch.dot(w, w)
        # Single sync point
        delta_hat = (dot_gw.abs() / (norm_g_sq.sqrt() * norm_w_sq.sqrt() + self.epsilon)).item()
        return delta_hat

    def _compute_lambda(self, gamma_t, ema_delta):
        """Compute lambda_t based on variant and decoupling mode."""
        g = 1.0 if self.decouple_lr else gamma_t
        if self.variant == "conservative":
            raw = self.c * g * (1.0 - ema_delta)
        elif self.variant == "aggressive":
            raw = self.c * g * ema_delta
        elif self.variant == "square":
            g2 = 1.0 if self.decouple_lr else (gamma_t ** 2)
            raw = self.c * g2 * (1.0 - ema_delta)
        return max(self.lambda_min, min(self.lambda_max, raw))

    def _get_weight_norm(self):
        """Compute total weight norm."""
        total = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                total += p.data.norm().item() ** 2
        return math.sqrt(total)

    def step(self):
        """Perform one optimization step with dynamic weight decay."""
        gamma_t = self._get_lr()

        # Compute alignment
        delta_hat = self._compute_alignment()
        self.current_delta_hat = delta_hat

        # EMA update
        self.ema_delta = self.beta * self.ema_delta + (1.0 - self.beta) * delta_hat

        # Compute dynamic lambda
        lambda_t = self._compute_lambda(gamma_t, self.ema_delta)
        self.current_lambda = lambda_t

        # Apply weight decay manually: w = (1 - lambda_t) * w
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(1.0 - lambda_t)

        # SGD step (gradient descent with momentum)
        self.optimizer.step()
        self.step_count += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'ema_delta': self.ema_delta,
            'step_count': self.step_count,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.ema_delta = state['ema_delta']
        self.step_count = state['step_count']

    def get_metrics(self):
        """Return current step metrics for logging."""
        return {
            'lambda_t': self.current_lambda,
            'delta_hat_t': self.current_delta_hat,
            'ema_delta': self.ema_delta,
            'weight_norm': self._get_weight_norm(),
            'lr': self._get_lr(),
        }


class CWDOptimizer:
    """Cautious Weight Decay (CWD) - ICLR 2026 baseline.

    Apply weight decay only when sign(update) == sign(param) coordinate-wise.
    update = -lr * grad (with momentum in SGD).
    """

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=5e-4):
        self.weight_decay = weight_decay
        self.current_lambda = weight_decay
        self.current_delta_hat = 0.0
        self.ema_delta = 0.0
        self.step_count = 0

        # SGD without built-in WD; we handle decay ourselves
        self.optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=0.0)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _get_weight_norm(self):
        total = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                total += p.data.norm().item() ** 2
        return math.sqrt(total)

    def step(self):
        """Apply CWD: decay only where sign(momentum_update) == sign(param).

        Vectorized implementation for speed (avoids per-element Python indexing).
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.optimizer.state.get(p, {})
                if 'momentum_buffer' in state:
                    update_dir = state['momentum_buffer']
                else:
                    update_dir = p.grad.data

                # Vectorized mask: decay where sign(update) == sign(param)
                mask = (update_dir.sign() == p.data.sign())
                # Apply: p = p * (1 - wd * mask)  (mask is 0/1 float)
                p.data.mul_(1.0 - self.weight_decay * mask.float())

        # SGD step
        self.optimizer.step()
        self.step_count += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(), 'step_count': self.step_count}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.step_count = state['step_count']

    def get_metrics(self):
        return {
            'lambda_t': self.weight_decay,
            'delta_hat_t': self.current_delta_hat,
            'ema_delta': self.ema_delta,
            'weight_norm': self._get_weight_norm(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }


class RandomDynamicWDOptimizer:
    """Ablation: Random dynamic WD replacing alignment with U[0,1]."""

    def __init__(self, params, lr=0.1, momentum=0.9, c=1.0,
                 lambda_min=1e-6, lambda_max=0.01):
        self.c = c
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.current_lambda = 0.0
        self.step_count = 0

        self.optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=0.0)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _get_weight_norm(self):
        total = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                total += p.data.norm().item() ** 2
        return math.sqrt(total)

    def step(self):
        gamma_t = self.optimizer.param_groups[0]['lr']
        random_delta = torch.rand(1).item()
        raw = self.c * gamma_t * (1.0 - random_delta)
        lambda_t = max(self.lambda_min, min(self.lambda_max, raw))
        self.current_lambda = lambda_t

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(1.0 - lambda_t)

        self.optimizer.step()
        self.step_count += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_metrics(self):
        return {
            'lambda_t': self.current_lambda,
            'delta_hat_t': 0.0,
            'ema_delta': 0.0,
            'weight_norm': self._get_weight_norm(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }


class NormMatchedWDOptimizer:
    """Ablation: Fixed WD adjusted per epoch to match a target weight norm trajectory."""

    def __init__(self, params, lr=0.1, momentum=0.9, base_wd=5e-4):
        self.base_wd = base_wd
        self.current_wd = base_wd
        self.target_norms = None  # Set externally from AADWD run
        self.current_epoch = 0
        self.step_count = 0

        self.optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=0.0)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def set_target_norms(self, norms):
        """Set target weight norms from an AADWD run."""
        self.target_norms = norms

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _get_weight_norm(self):
        total = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                total += p.data.norm().item() ** 2
        return math.sqrt(total)

    def step(self):
        # If we have target norms, adjust WD to track them
        if self.target_norms and self.current_epoch < len(self.target_norms):
            current_norm = self._get_weight_norm()
            target_norm = self.target_norms[self.current_epoch]
            if current_norm > target_norm:
                self.current_wd = min(self.base_wd * 2.0,
                                      self.current_wd * 1.1)
            else:
                self.current_wd = max(self.base_wd * 0.1,
                                      self.current_wd * 0.9)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(1.0 - self.current_wd)

        self.optimizer.step()
        self.step_count += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_metrics(self):
        return {
            'lambda_t': self.current_wd,
            'delta_hat_t': 0.0,
            'ema_delta': 0.0,
            'weight_norm': self._get_weight_norm(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }


class EquivalentCumulativeWDOptimizer:
    """Ablation: Fixed WD = mean(lambda_t) from a prior AADWD run."""

    def __init__(self, params, lr=0.1, momentum=0.9, fixed_wd=5e-4):
        self.fixed_wd = fixed_wd
        self.step_count = 0
        self.optimizer = SGD(params, lr=lr, momentum=momentum,
                             weight_decay=fixed_wd)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _get_weight_norm(self):
        total = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                total += p.data.norm().item() ** 2
        return math.sqrt(total)

    def step(self):
        self.optimizer.step()
        self.step_count += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_metrics(self):
        return {
            'lambda_t': self.fixed_wd,
            'delta_hat_t': 0.0,
            'ema_delta': 0.0,
            'weight_norm': self._get_weight_norm(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }


def create_optimizer(model, method, lr=0.1, momentum=0.9, weight_decay=5e-4,
                     c=1.0, beta=0.99, lambda_min=1e-6, lambda_max=0.01,
                     variant="conservative", decouple_lr=False):
    """Factory function to create optimizers based on method name.

    Args:
        method: One of "no_wd", "fixed_wd", "stagewise_wd", "cwd",
                "aadwd_conservative", "aadwd_aggressive", "aadwd_square",
                "random_dynamic_wd", "norm_matched_wd", "equiv_cumulative_wd"
        decouple_lr: If True, AADWD lambda does not depend on current LR.
    """
    params = model.parameters()

    if method == "no_wd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=0.0)

    elif method == "fixed_wd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif method == "stagewise_wd":
        # Use fixed_wd initially; scheduler adjusts WD at milestones
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif method == "cwd":
        return CWDOptimizer(list(model.parameters()), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)

    elif method.startswith("aadwd_"):
        var = method.split("_", 1)[1]
        return AADWDOptimizer(list(model.parameters()), lr=lr, momentum=momentum,
                              c=c, beta=beta, lambda_min=lambda_min,
                              lambda_max=lambda_max, variant=var,
                              decouple_lr=decouple_lr)

    elif method == "random_dynamic_wd":
        return RandomDynamicWDOptimizer(list(model.parameters()), lr=lr,
                                        momentum=momentum, c=c,
                                        lambda_min=lambda_min,
                                        lambda_max=lambda_max)

    elif method == "norm_matched_wd":
        return NormMatchedWDOptimizer(list(model.parameters()), lr=lr,
                                      momentum=momentum, base_wd=weight_decay)

    elif method == "equiv_cumulative_wd":
        return EquivalentCumulativeWDOptimizer(list(model.parameters()), lr=lr,
                                               momentum=momentum,
                                               fixed_wd=weight_decay)

    else:
        raise ValueError(f"Unknown method: {method}")
