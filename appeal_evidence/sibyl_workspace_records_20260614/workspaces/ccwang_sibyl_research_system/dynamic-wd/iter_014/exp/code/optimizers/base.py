"""Base class for all WD optimizer wrappers with diagnostic logging interface."""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Any
import math


class WDOptimizer:
    """Base wrapper providing a unified interface for WD methods with diagnostic logging.

    All subclasses must implement `_compute_effective_wd()` which returns per-group
    effective lambda values. The base class handles the actual parameter update via SGDW.
    """

    def __init__(self, model, lr: float = 0.1, momentum: float = 0.9,
                 weight_decay: float = 1e-4, use_adamw: bool = False, **kwargs):
        self.model = model
        self.base_lr = lr
        self.current_lr = lr
        self.momentum = momentum
        self.base_wd = weight_decay
        self.use_adamw = use_adamw
        self.step_count = 0
        self.eps = 1e-8

        # Build parameter groups with per-layer tracking
        self.param_groups = self._build_param_groups(model)

        # Create underlying PyTorch optimizer (weight_decay=0, we handle WD manually)
        if use_adamw:
            self.optimizer = optim.Adam(
                self.param_groups, lr=lr, weight_decay=0,
                betas=kwargs.get('betas', (0.9, 0.999))
            )
        else:
            self.optimizer = optim.SGD(
                self.param_groups, lr=lr, momentum=momentum, weight_decay=0
            )

        # Diagnostic storage (per-group per-step)
        self._diagnostics: Dict[str, List[float]] = {}

    def _build_param_groups(self, model) -> List[Dict]:
        """Build per-layer parameter groups for per-layer WD control."""
        param_groups = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Skip bias and norm parameters from WD
                apply_wd = not (name.endswith('.bias') or 'bn' in name or
                               'norm' in name or 'layernorm' in name)
                param_groups.append({
                    'params': [param],
                    'layer_name': name,
                    'apply_wd': apply_wd,
                    'effective_wd': self.base_wd if apply_wd else 0.0,
                })
        return param_groups

    def _compute_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Compute per-layer diagnostic metrics."""
        diagnostics = {}
        for group in self.param_groups:
            name = group['layer_name']
            param = group['params'][0]

            if param.grad is None:
                continue

            w = param.data
            g = param.grad.data

            w_norm = torch.norm(w).item()
            g_norm = torch.norm(g).item()

            # rho_t = ||g|| / ||w||
            rho_t = g_norm / (w_norm + self.eps)

            # alpha_t = cos(g, w)
            dot = torch.dot(w.flatten(), g.flatten()).item()
            alpha_t = dot / (w_norm * g_norm + self.eps)

            diagnostics[name] = {
                'w_norm': w_norm,
                'g_norm': g_norm,
                'rho_t': rho_t,
                'alpha_t': alpha_t,
                'effective_wd': group.get('effective_wd', self.base_wd),
            }

        return diagnostics

    def _compute_effective_wd(self) -> None:
        """Compute effective WD for each parameter group. Override in subclasses."""
        # Default: use base_wd for all groups
        for group in self.param_groups:
            if group['apply_wd']:
                group['effective_wd'] = self.base_wd

    def _apply_weight_decay(self):
        """Apply manual weight decay to parameters."""
        for group in self.param_groups:
            if not group['apply_wd']:
                continue
            wd = group['effective_wd']
            if wd == 0:
                continue
            for p in group['params']:
                if p.data is not None:
                    # Decoupled WD: w = w - lr * wd * w
                    p.data.mul_(1 - self.current_lr * wd)

    def step(self, closure=None):
        """Perform a single optimization step with dynamic WD."""
        # 1. Compute effective WD per group
        self._compute_effective_wd()

        # 2. Apply weight decay (before optimizer step for decoupled WD)
        self._apply_weight_decay()

        # 3. Optimizer step (gradient descent only, no WD)
        self.optimizer.step(closure)

        self.step_count += 1

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Get current per-layer diagnostics."""
        return self._compute_diagnostics()

    def set_lr(self, lr: float):
        """Update learning rate for all parameter groups."""
        self.current_lr = lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups_optimizer(self):
        return self.optimizer.param_groups

    def get_method_name(self) -> str:
        return self.__class__.__name__.replace('Optimizer', '')
