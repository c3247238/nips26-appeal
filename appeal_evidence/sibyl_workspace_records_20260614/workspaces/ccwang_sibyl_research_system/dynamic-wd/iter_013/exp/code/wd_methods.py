"""
Weight Decay Methods: Unified implementations under the Phi framework.

All methods are implemented as optimizer wrappers that modify the effective
weight decay per layer per step. The Phi formulation:
    lambda_t(l) = lambda_base * Phi(cos_l, ||w_l||, t, tau_l, type_l)

Methods:
  1. NoWD       - Phi = 0 (no weight decay)
  2. FixedWD    - Phi = 1 (standard SGDW / AdamW)
  3. SWD        - Phi = f(||g||, t) (gradient-norm scheduled, NeurIPS 2023)
  4. CWD        - Phi = I[sign(w)*sign(g) > 0] (binary alignment mask, ICLR 2026)
  5. CPR        - Phi = max(0, ||w|| - tau) / ||w|| (per-matrix constraint, NeurIPS 2024)
  6. CAWD       - Phi = (1 + delta_hat) / 2 (continuous alignment modulation)
  7. EqWD       - Phi = 1 + beta * |r - r*| / r* (ratio-deviation driven)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List


class WDMethod:
    """Base class for weight decay methods."""

    name: str = "base"

    def __init__(self, lambda_base: float = 5e-4, **kwargs):
        self.lambda_base = lambda_base
        # Per-layer state (keyed by parameter name)
        self._state: Dict[str, dict] = {}

    def get_effective_wd(
        self,
        param_name: str,
        param: torch.Tensor,
        grad: torch.Tensor,
        step: int,
        lr: float,
        is_normalized: bool = False,
    ) -> float:
        """Return effective weight decay for this parameter at this step."""
        raise NotImplementedError

    def get_diagnostics(self, param_name: str) -> dict:
        """Return per-layer diagnostic values for logging."""
        return self._state.get(param_name, {})

    def reset(self):
        """Reset internal state (for new training runs)."""
        self._state = {}


class NoWD(WDMethod):
    """No weight decay baseline."""
    name = "NoWD"

    def get_effective_wd(self, param_name, param, grad, step, lr, is_normalized=False):
        return 0.0


class FixedWD(WDMethod):
    """Standard fixed weight decay (SGDW / AdamW style)."""
    name = "FixedWD"

    def get_effective_wd(self, param_name, param, grad, step, lr, is_normalized=False):
        return self.lambda_base


class SWD(WDMethod):
    """Scheduled Weight Decay (NeurIPS 2023 style).

    Modulates WD based on gradient norm relative to a running average.
    Phi = f(||g||, t) = ||g_t|| / EMA(||g_t||)
    """
    name = "SWD"

    def __init__(self, lambda_base: float = 5e-4, ema_alpha: float = 0.99, **kwargs):
        super().__init__(lambda_base, **kwargs)
        self.ema_alpha = ema_alpha

    def get_effective_wd(self, param_name, param, grad, step, lr, is_normalized=False):
        import math
        grad_norm = grad.norm().item()

        # Guard against inf/NaN from AMP GradScaler
        if not math.isfinite(grad_norm):
            return self.lambda_base  # fall back to fixed WD

        if param_name not in self._state:
            self._state[param_name] = {"ema_grad_norm": grad_norm}

        state = self._state[param_name]
        ema = state["ema_grad_norm"]
        ema = self.ema_alpha * ema + (1 - self.ema_alpha) * grad_norm
        state["ema_grad_norm"] = ema

        # Phi = grad_norm / ema (clamped to avoid explosion)
        phi = min(grad_norm / (ema + 1e-8), 3.0)
        state["phi"] = phi
        state["grad_norm"] = grad_norm

        return self.lambda_base * phi


class CWD(WDMethod):
    """Conflict-aware Weight Decay (ICLR 2026).

    Binary alignment mask: apply WD only when gradient and weight are aligned.
    Phi = I[sign(w) * sign(g) > 0] applied element-wise, but we compute the
    fraction of aligned elements as a scalar approximation for per-layer WD.
    """
    name = "CWD"

    def get_effective_wd(self, param_name, param, grad, step, lr, is_normalized=False):
        with torch.no_grad():
            # Element-wise: decay only where sign(w) == sign(g)
            aligned = (param.sign() * grad.sign() > 0).float()
            alignment_frac = aligned.mean().item()

        if param_name not in self._state:
            self._state[param_name] = {}
        self._state[param_name]["alignment_frac"] = alignment_frac
        self._state[param_name]["phi"] = alignment_frac

        return self.lambda_base * alignment_frac

    def apply_elementwise(self, param: torch.Tensor, grad: torch.Tensor,
                          lr: float) -> torch.Tensor:
        """Return element-wise WD update (for true CWD implementation)."""
        with torch.no_grad():
            mask = (param.sign() * grad.sign() > 0).float()
            return param * mask * self.lambda_base * lr


class CPR(WDMethod):
    """Constrained Parameter Regularization (NeurIPS 2024).

    Per-matrix norm constraint: Phi = max(0, ||w|| - tau) / ||w||.
    tau is the target norm (set as initial norm * scale_factor).
    """
    name = "CPR"

    def __init__(self, lambda_base: float = 5e-4, scale_factor: float = 1.0, **kwargs):
        super().__init__(lambda_base, **kwargs)
        self.scale_factor = scale_factor

    def get_effective_wd(self, param_name, param, grad, step, lr, is_normalized=False):
        with torch.no_grad():
            w_norm = param.norm().item()

        if param_name not in self._state:
            # Set target norm as initial norm * scale_factor
            self._state[param_name] = {"tau": w_norm * self.scale_factor}

        state = self._state[param_name]
        tau = state["tau"]

        phi = max(0.0, w_norm - tau) / (w_norm + 1e-8)
        state["phi"] = phi
        state["w_norm"] = w_norm
        state["tau"] = tau

        return self.lambda_base * phi


class CAWD(WDMethod):
    """Continuous Alignment-Weighted Decay.

    Uses the continuous cosine similarity (not binary) for modulation.
    Phi = (1 + delta_hat) / 2, where delta_hat = <g, w> / (||g|| ||w||).
    """
    name = "CAWD"

    def __init__(self, lambda_base: float = 5e-4, noise_sigma: float = 0.0, **kwargs):
        super().__init__(lambda_base, **kwargs)
        self.noise_sigma = noise_sigma  # For noise injection control experiment

    def get_effective_wd(self, param_name, param, grad, step, lr, is_normalized=False):
        import math
        with torch.no_grad():
            g_flat = grad.reshape(-1)
            w_flat = param.reshape(-1)
            g_norm = g_flat.norm().item()
            w_norm = w_flat.norm().item()

            # Guard against inf/NaN from AMP GradScaler
            if not math.isfinite(g_norm) or g_norm < 1e-12 or w_norm < 1e-12:
                delta_hat = 0.0
            else:
                delta_hat = (torch.dot(g_flat, w_flat) / (g_norm * w_norm + 1e-8)).item()

        # Optional noise injection (for control experiment)
        if self.noise_sigma > 0:
            delta_hat += torch.randn(1).item() * self.noise_sigma

        phi = (1.0 + delta_hat) / 2.0
        phi = max(0.0, min(phi, 1.0))  # Clamp to [0, 1]

        if param_name not in self._state:
            self._state[param_name] = {}
        self._state[param_name].update({
            "delta_hat": delta_hat,
            "phi": phi,
            "g_norm": g_norm,
            "w_norm": w_norm,
        })

        return self.lambda_base * phi


class EqWD(WDMethod):
    """Equilibrium-Driven Weight Decay.

    Uses gradient-to-weight ratio deviation from layer-specific equilibrium.
    r_t^l = ||g_t^l|| / (||w_t^l|| + eps)
    r*^l = EMA(r_t^l, alpha)
    dev_t^l = |r_t^l - r*^l| / (r*^l + eps)
    lambda_t^l = lambda_base * (1 + beta * dev_t^l)

    Layer-type-aware extension:
    - Normalized layers: ratio-deviation only (no alignment)
    - Non-normalized layers: ratio + alignment signal
    """
    name = "EqWD"

    def __init__(
        self,
        lambda_base: float = 5e-4,
        beta: float = 1.0,
        ema_alpha: float = 0.9,
        layer_type_aware: bool = False,
        alignment_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(lambda_base, **kwargs)
        self.beta = beta
        self.ema_alpha = ema_alpha
        self.layer_type_aware = layer_type_aware
        self.alignment_weight = alignment_weight

    def get_effective_wd(self, param_name, param, grad, step, lr, is_normalized=False):
        import math
        with torch.no_grad():
            g_norm = grad.norm().item()
            w_norm = param.norm().item()
            eps = 1e-8

            # Guard against inf/NaN from AMP GradScaler
            if not math.isfinite(g_norm):
                return self.lambda_base  # fall back to fixed WD

            # Per-layer ratio
            r_t = g_norm / (w_norm + eps)

            # Alignment (cosine similarity)
            g_flat = grad.reshape(-1)
            w_flat = param.reshape(-1)
            if g_norm < 1e-12 or w_norm < 1e-12:
                delta_hat = 0.0
            else:
                delta_hat = (torch.dot(g_flat, w_flat) / (g_norm * w_norm + eps)).item()

        if param_name not in self._state:
            self._state[param_name] = {"r_star": r_t, "step_count": 0}

        state = self._state[param_name]
        state["step_count"] += 1

        # Update EMA equilibrium estimate
        r_star = state["r_star"]
        r_star = self.ema_alpha * r_star + (1 - self.ema_alpha) * r_t
        state["r_star"] = r_star

        # Normalized deviation
        dev = abs(r_t - r_star) / (r_star + 1e-8)

        # Compute Phi based on layer type
        if self.layer_type_aware and is_normalized:
            # Normalized layers: ratio-deviation only, no alignment
            phi = 1.0 + self.beta * dev
        elif self.layer_type_aware and not is_normalized:
            # Non-normalized layers: ratio + alignment
            alignment_mod = abs(delta_hat) * self.alignment_weight
            phi = 1.0 + self.beta * (dev + alignment_mod)
        else:
            # Uniform mode: ratio-deviation only
            phi = 1.0 + self.beta * dev

        # Store diagnostics
        state.update({
            "r_t": r_t,
            "r_star": r_star,
            "dev": dev,
            "delta_hat": delta_hat,
            "phi": phi,
            "g_norm": g_norm,
            "w_norm": w_norm,
        })

        return self.lambda_base * phi


# ---------- Registry ----------

WD_METHODS = {
    "NoWD": NoWD,
    "FixedWD": FixedWD,
    "SWD": SWD,
    "CWD": CWD,
    "CPR": CPR,
    "CAWD": CAWD,
    "EqWD": EqWD,
}


def create_wd_method(name: str, **kwargs) -> WDMethod:
    """Factory function to create a WD method by name."""
    if name not in WD_METHODS:
        raise ValueError(f"Unknown WD method: {name}. Available: {list(WD_METHODS.keys())}")
    return WD_METHODS[name](**kwargs)


def get_method_names() -> List[str]:
    """Return list of all available WD method names."""
    return list(WD_METHODS.keys())


# ---------- Optimizer integration ----------

def apply_wd_step(
    model: nn.Module,
    wd_method: WDMethod,
    step: int,
    lr: float,
    normalized_layer_names: Optional[set] = None,
) -> Dict[str, float]:
    """Apply per-layer WD to model parameters.

    This should be called BEFORE the optimizer step.
    Returns dict of per-layer effective WD values.

    For CWD, use element-wise application.
    For all others, apply scalar WD uniformly to the layer.
    """
    if normalized_layer_names is None:
        normalized_layer_names = set()

    effective_wds = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        # Skip bias and normalization parameters (1D tensors)
        if param.dim() <= 1:
            continue

        is_normalized = name in normalized_layer_names

        if isinstance(wd_method, CWD):
            # Element-wise WD for CWD
            wd_update = wd_method.apply_elementwise(param, param.grad, lr)
            param.data.sub_(wd_update)
            # Also compute scalar phi for diagnostics
            wd_method.get_effective_wd(name, param, param.grad, step, lr, is_normalized)
            effective_wds[name] = wd_method._state.get(name, {}).get("phi", 0.0) * wd_method.lambda_base
        elif isinstance(wd_method, NoWD):
            effective_wds[name] = 0.0
        else:
            eff_wd = wd_method.get_effective_wd(name, param, param.grad, step, lr, is_normalized)
            # Apply: w = w - lr * wd * w (SGDW-style decoupled WD)
            param.data.mul_(1.0 - lr * eff_wd)
            effective_wds[name] = eff_wd

    return effective_wds


def detect_normalized_layers(model: nn.Module) -> set:
    """Detect which layers are preceded by normalization layers (BN/LN/GN).

    Returns set of parameter names that are in layers immediately after a
    normalization layer.
    """
    normalized = set()
    modules = list(model.named_modules())

    for i, (name, module) in enumerate(modules):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.LayerNorm, nn.GroupNorm)):
            # The next conv/linear layer after this BN is "normalized"
            for j in range(i + 1, len(modules)):
                next_name, next_module = modules[j]
                if isinstance(next_module, (nn.Conv2d, nn.Linear)):
                    for pname, _ in next_module.named_parameters():
                        full_name = f"{next_name}.{pname}" if next_name else pname
                        normalized.add(full_name)
                    break
    return normalized
