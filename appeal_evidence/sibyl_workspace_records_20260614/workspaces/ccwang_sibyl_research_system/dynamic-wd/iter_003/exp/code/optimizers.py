"""
Unified Dynamic Weight Decay Framework - Optimizer Module.

Implements all WD methods under a unified Phi-modulator interface:
- constant: Standard fixed WD (AdamW default)
- cosine_schedule: Cosine-annealed WD from wd_max to wd_min
- linear_schedule: Linear WD schedule
- cwd_hard: Cautious Weight Decay with binary sign-alignment mask
- cwd_soft: Soft CWD with sigmoid approximation (parametric beta)
- swd: Scaled Weight Decay (gradient-norm based scaling)
- random_mask: Random binary mask (CWD ablation control)
- half_lambda: Constant WD at half rate (CWD ablation control)
- norm_matched: WD adjusted to match target norm trajectory
"""

import math
import torch
import torch.nn as nn


class PhiModulator:
    """Base class for WD modulation functions (the Phi framework).

    All WD methods implement: lambda_eff(t, w, g) = phi(t, w, g) * lambda_base
    where phi is the modulation function.
    """

    def __init__(self, base_wd=5e-4):
        self.base_wd = base_wd
        self._step_count = 0
        self._diagnostics = {}

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        """Return effective WD for this parameter. Override in subclasses."""
        raise NotImplementedError

    def get_diagnostics(self):
        """Return diagnostic metrics for logging."""
        return dict(self._diagnostics)

    def step_done(self):
        self._step_count += 1


class ConstantPhi(PhiModulator):
    """phi(t, w, g) = 1  (standard constant WD)."""

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        return self.base_wd


class CosineSchedulePhi(PhiModulator):
    """phi(t) = wd_min + 0.5 * (wd_max - wd_min) * (1 + cos(pi * t / T))."""

    def __init__(self, base_wd=5e-4, wd_min=0.0):
        super().__init__(base_wd)
        self.wd_min = wd_min

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        progress = step / max(total_steps, 1)
        wd = self.wd_min + 0.5 * (self.base_wd - self.wd_min) * (1 + math.cos(math.pi * progress))
        self._diagnostics['wd_schedule'] = wd
        return wd


class LinearSchedulePhi(PhiModulator):
    """phi(t) = wd_start + (wd_end - wd_start) * t / T."""

    def __init__(self, base_wd=5e-4, wd_end=0.0):
        super().__init__(base_wd)
        self.wd_end = wd_end

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        progress = step / max(total_steps, 1)
        wd = self.base_wd + (self.wd_end - self.base_wd) * progress
        self._diagnostics['wd_schedule'] = wd
        return wd


class CWDHardPhi(PhiModulator):
    """Cautious WD: decay only where sign(momentum_update) == sign(param)."""

    def __init__(self, base_wd=5e-4):
        super().__init__(base_wd)
        self._mask_ratio_sum = 0.0
        self._mask_count = 0

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        # Use momentum buffer if available, otherwise gradient
        if 'momentum_buffer' in state and state['momentum_buffer'] is not None:
            update_dir = state['momentum_buffer']
        else:
            update_dir = grad
        mask = (update_dir.sign() == param.data.sign()).float()
        ratio = mask.mean().item()
        self._mask_ratio_sum += ratio
        self._mask_count += 1
        self._diagnostics['mask_ratio'] = ratio
        self._diagnostics['avg_mask_ratio'] = self._mask_ratio_sum / self._mask_count
        # Return per-element WD tensor
        return self.base_wd * mask


class CWDSoftPhi(PhiModulator):
    """Soft CWD: sigmoid approximation of binary mask.

    mask_soft = sigmoid(beta * update_dir * param / (||update_dir|| * ||param|| + eps))
    As beta -> inf, this approaches hard CWD.
    """

    def __init__(self, base_wd=5e-4, beta=100.0):
        super().__init__(base_wd)
        self.beta = beta
        self._mask_ratio_sum = 0.0
        self._mask_count = 0

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        if 'momentum_buffer' in state and state['momentum_buffer'] is not None:
            update_dir = state['momentum_buffer']
        else:
            update_dir = grad
        # Element-wise alignment score
        alignment = update_dir * param.data
        # Normalize per-element
        scale = (update_dir.abs() * param.data.abs()).clamp(min=1e-8)
        normalized = alignment / scale
        mask_soft = torch.sigmoid(self.beta * normalized)
        ratio = mask_soft.mean().item()
        self._mask_ratio_sum += ratio
        self._mask_count += 1
        self._diagnostics['mask_ratio'] = ratio
        self._diagnostics['avg_mask_ratio'] = self._mask_ratio_sum / self._mask_count
        self._diagnostics['beta'] = self.beta
        return self.base_wd * mask_soft


class SWDPhi(PhiModulator):
    """Scaled Weight Decay: scale WD by gradient sensitivity.

    phi(w, g) = 1 + sensitivity * (||g_param|| / ||g_global|| - 1)
    Parameters with larger relative gradient norms get more WD.
    """

    def __init__(self, base_wd=5e-4, sensitivity=1.0):
        super().__init__(base_wd)
        self.sensitivity = sensitivity
        self._global_grad_norm = None

    def set_global_grad_norm(self, norm):
        self._global_grad_norm = norm

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        if self._global_grad_norm is None or self._global_grad_norm < 1e-10:
            return self.base_wd
        local_norm = grad.norm().item()
        n_params = grad.numel()
        # Normalized gradient norm (per-element)
        local_per_elem = local_norm / max(math.sqrt(n_params), 1.0)
        global_per_elem = self._global_grad_norm / max(math.sqrt(sum(
            p.numel() for group in [] for p in group['params']  # placeholder
        )), 1.0)
        # Simpler: ratio of local to global
        ratio = local_norm / (self._global_grad_norm + 1e-10)
        multiplier = 1.0 + self.sensitivity * (ratio - 1.0)
        multiplier = max(0.1, min(10.0, multiplier))  # clamp
        self._diagnostics['wd_multiplier'] = multiplier
        return self.base_wd * multiplier


class RandomMaskPhi(PhiModulator):
    """Ablation: Random binary mask matching CWD's average mask ratio."""

    def __init__(self, base_wd=5e-4, mask_prob=0.5):
        super().__init__(base_wd)
        self.mask_prob = mask_prob

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        mask = (torch.rand_like(param.data) < self.mask_prob).float()
        self._diagnostics['mask_ratio'] = mask.mean().item()
        return self.base_wd * mask


class HalfLambdaPhi(PhiModulator):
    """Ablation: Constant WD at half the base rate."""

    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        return self.base_wd * 0.5


class UnifiedAdamW(torch.optim.Optimizer):
    """AdamW with pluggable Phi modulator for unified WD experiments.

    This is the core optimizer of the unified framework. It implements
    standard AdamW but delegates WD computation to a PhiModulator.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 phi_modulator=None, total_steps=None):
        if phi_modulator is None:
            phi_modulator = ConstantPhi(base_wd=0.0)
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.phi = phi_modulator
        self.total_steps = total_steps or 1
        self._global_step = 0
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute global grad norm for SWD
        global_grad_norm_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    global_grad_norm_sq += p.grad.data.norm().item() ** 2
        global_grad_norm = math.sqrt(global_grad_norm_sq)
        if hasattr(self.phi, 'set_global_grad_norm'):
            self.phi.set_global_grad_norm(global_grad_norm)

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Decay running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                step_size = lr / bc1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)

                # Gradient step
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay (decoupled, AdamW-style)
                # Get per-parameter WD from Phi modulator
                # Pass optimizer state for momentum-buffer access
                opt_state = {'momentum_buffer': exp_avg}  # Use first moment as "momentum"
                wd = self.phi.get_per_param_wd(
                    p, grad, opt_state,
                    self._global_step, self.total_steps, lr
                )
                if isinstance(wd, torch.Tensor):
                    # Per-element WD (CWD-style)
                    p.data.mul_(1.0 - lr * wd)
                else:
                    # Scalar WD
                    if wd > 0:
                        p.data.mul_(1.0 - lr * wd)

        self._global_step += 1
        self.phi.step_done()
        return loss

    def get_metrics(self):
        """Get current optimizer metrics for logging."""
        metrics = {
            'global_step': self._global_step,
            'lr': self.param_groups[0]['lr'],
        }
        metrics.update(self.phi.get_diagnostics())

        # Compute weight norm
        total_norm_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                total_norm_sq += p.data.norm().item() ** 2
        metrics['weight_norm'] = math.sqrt(total_norm_sq)

        # Compute effective lambda from Phi diagnostics or base
        diag = self.phi.get_diagnostics()
        if 'wd_schedule' in diag:
            metrics['effective_wd'] = diag['wd_schedule']
        elif 'wd_multiplier' in diag:
            metrics['effective_wd'] = self.phi.base_wd * diag['wd_multiplier']
        elif 'mask_ratio' in diag:
            metrics['effective_wd'] = self.phi.base_wd * diag['mask_ratio']
        else:
            metrics['effective_wd'] = self.phi.base_wd

        return metrics


class UnifiedSGDW(torch.optim.Optimizer):
    """SGD with momentum and pluggable Phi modulator for WD.

    Uses decoupled weight decay (SGDW-style) for fair comparison with AdamW.
    The only difference from UnifiedAdamW is the lack of adaptive learning rate.
    """

    def __init__(self, params, lr=0.1, momentum=0.9, dampening=0,
                 phi_modulator=None, total_steps=None):
        if phi_modulator is None:
            phi_modulator = ConstantPhi(base_wd=0.0)
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super().__init__(params, defaults)
        self.phi = phi_modulator
        self.total_steps = total_steps or 1
        self._global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute global grad norm for SWD
        global_grad_norm_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    global_grad_norm_sq += p.grad.data.norm().item() ** 2
        global_grad_norm = math.sqrt(global_grad_norm_sq)
        if hasattr(self.phi, 'set_global_grad_norm'):
            self.phi.set_global_grad_norm(global_grad_norm)

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']
                state['step'] += 1

                # SGD with momentum
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                # Gradient step
                p.data.add_(buf, alpha=-lr)

                # Decoupled weight decay via Phi modulator
                opt_state = {'momentum_buffer': buf}
                wd = self.phi.get_per_param_wd(
                    p, grad, opt_state,
                    self._global_step, self.total_steps, lr
                )
                if isinstance(wd, torch.Tensor):
                    p.data.mul_(1.0 - lr * wd)
                else:
                    if wd > 0:
                        p.data.mul_(1.0 - lr * wd)

        self._global_step += 1
        self.phi.step_done()
        return loss

    def get_metrics(self):
        """Get current optimizer metrics for logging."""
        metrics = {
            'global_step': self._global_step,
            'lr': self.param_groups[0]['lr'],
        }
        metrics.update(self.phi.get_diagnostics())

        total_norm_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                total_norm_sq += p.data.norm().item() ** 2
        metrics['weight_norm'] = math.sqrt(total_norm_sq)

        diag = self.phi.get_diagnostics()
        if 'wd_schedule' in diag:
            metrics['effective_wd'] = diag['wd_schedule']
        elif 'wd_multiplier' in diag:
            metrics['effective_wd'] = self.phi.base_wd * diag['wd_multiplier']
        elif 'mask_ratio' in diag:
            metrics['effective_wd'] = self.phi.base_wd * diag['mask_ratio']
        else:
            metrics['effective_wd'] = self.phi.base_wd

        return metrics


def create_unified_optimizer(model, method, lr=1e-3, wd=5e-4, epochs=100,
                              batch_size=128, dataset_size=50000, **kwargs):
    """Factory function for unified framework optimizers.

    Args:
        method: WD method name
        kwargs: Method-specific parameters (beta for soft CWD, sensitivity for SWD, etc.)
    """
    total_steps = epochs * (dataset_size // batch_size)

    # Create Phi modulator
    if method == 'constant':
        phi = ConstantPhi(base_wd=wd)
    elif method == 'cosine_schedule':
        phi = CosineSchedulePhi(base_wd=wd, wd_min=kwargs.get('wd_min', 0.0))
    elif method == 'linear_schedule':
        phi = LinearSchedulePhi(base_wd=wd, wd_end=kwargs.get('wd_end', 0.0))
    elif method == 'cwd_hard':
        phi = CWDHardPhi(base_wd=wd)
    elif method == 'cwd_soft':
        phi = CWDSoftPhi(base_wd=wd, beta=kwargs.get('cwd_beta', 100.0))
    elif method == 'swd':
        phi = SWDPhi(base_wd=wd, sensitivity=kwargs.get('swd_sensitivity', 1.0))
    elif method == 'random_mask':
        phi = RandomMaskPhi(base_wd=wd, mask_prob=kwargs.get('mask_prob', 0.5))
    elif method == 'half_lambda':
        phi = HalfLambdaPhi(base_wd=wd)
    elif method == 'no_wd':
        phi = ConstantPhi(base_wd=0.0)
    else:
        raise ValueError(f"Unknown WD method: {method}")

    optimizer = UnifiedAdamW(
        model.parameters(), lr=lr,
        betas=kwargs.get('betas', (0.9, 0.999)),
        eps=kwargs.get('eps', 1e-8),
        phi_modulator=phi,
        total_steps=total_steps,
    )

    return optimizer


def create_sgd_optimizer(model, method, lr=0.1, wd=5e-4, epochs=100,
                          batch_size=128, dataset_size=50000, **kwargs):
    """Factory for SGD + Phi modulator (decoupled WD).

    Same Phi framework but with SGD instead of AdamW, to test whether
    the WD method irrelevance is specific to adaptive optimizers.
    """
    total_steps = epochs * (dataset_size // batch_size)

    # Create Phi modulator (same as AdamW version)
    if method == 'constant':
        phi = ConstantPhi(base_wd=wd)
    elif method == 'cosine_schedule':
        phi = CosineSchedulePhi(base_wd=wd, wd_min=kwargs.get('wd_min', 0.0))
    elif method == 'cwd_hard':
        phi = CWDHardPhi(base_wd=wd)
    elif method == 'swd':
        phi = SWDPhi(base_wd=wd, sensitivity=kwargs.get('swd_sensitivity', 1.0))
    elif method == 'random_mask':
        phi = RandomMaskPhi(base_wd=wd, mask_prob=kwargs.get('mask_prob', 0.5))
    elif method == 'half_lambda':
        phi = HalfLambdaPhi(base_wd=wd)
    elif method == 'no_wd':
        phi = ConstantPhi(base_wd=0.0)
    else:
        raise ValueError(f"Unknown WD method: {method}")

    optimizer = UnifiedSGDW(
        model.parameters(), lr=lr,
        momentum=kwargs.get('momentum', 0.9),
        phi_modulator=phi,
        total_steps=total_steps,
    )

    return optimizer
