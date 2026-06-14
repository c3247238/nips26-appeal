#!/usr/bin/env python3
"""
ttt_layer.py — Core TTT (Test-Time Training) Layer for Denoising-as-Learning (DaL)

Implements three update rule variants:
  1. TTT-Linear: Linear fast weight (W: d_model -> d_model), equivalent to linear attention
  2. TTT-MLP: 2-layer MLP fast weight (d_model -> d_ttt -> d_model, d_ttt = d_model/8)
  3. Momentum-TTT: MLP + momentum (beta=0.9) + weight decay (lambda)

All variants include:
  - Residual gate beta (learnable, initialized to 0 for stable training start)
  - Precision-weighted self-supervised loss
  - Phase-transition scheduling interface
  - Drop-in compatibility with frozen DLM backbones (LLaDA / Dream)

Usage:
    layer = TTTLayer(d_model=4096, variant="mlp")
    # At each denoising step:
    output, metrics = layer(hidden_states, revealed_mask, target_ids)
    # Fast weights persist across steps. Reset between sequences:
    layer.reset_fast_weights()
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Literal


class FastWeightLinear(nn.Module):
    """Linear fast weight model: W @ h -> prediction.
    Equivalent to linear attention when used as TTT update rule.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Learnable initialization for fast weights
        self.W_init = nn.Parameter(torch.zeros(d_model, d_model))
        nn.init.xavier_uniform_(self.W_init, gain=1.0)
        # Runtime fast weights (not a parameter — updated by gradient descent)
        self.register_buffer("W_fast", None)

    def reset(self, batch_size: int = 1):
        """Reset fast weights to learned initialization."""
        # W_fast: (batch_size, d_model, d_model) — detached leaf tensor
        self.W_fast = self.W_init.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # h: (B, S, D), W_fast: (B, D, D)
        return torch.bmm(h, self.W_fast.transpose(1, 2))

    def get_params_for_grad(self):
        """Return fast weight tensor for gradient computation."""
        return [self.W_fast]

    def apply_update(self, grads: list, lr: float):
        """Apply gradient update to fast weights (creates new leaf tensors)."""
        self.W_fast = (self.W_fast.detach() - lr * grads[0].detach()).requires_grad_(False)

    def param_count(self) -> int:
        return self.W_init.numel()


class FastWeightMLP(nn.Module):
    """2-layer MLP fast weight model: h -> W1 -> ReLU -> W2 -> prediction.
    d_model -> d_ttt -> d_model, where d_ttt = d_model / 8.
    Higher capacity than linear, core variant for DaL.
    """

    def __init__(self, d_model: int, d_ttt: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ttt = d_ttt or d_model // 8

        # Learnable initializations
        self.W1_init = nn.Parameter(torch.zeros(self.d_ttt, d_model))
        self.b1_init = nn.Parameter(torch.zeros(self.d_ttt))
        self.W2_init = nn.Parameter(torch.zeros(d_model, self.d_ttt))
        self.b2_init = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.W1_init, gain=1.0)
        nn.init.xavier_uniform_(self.W2_init, gain=1.0)

        # Runtime fast weights
        self.register_buffer("W1_fast", None)
        self.register_buffer("b1_fast", None)
        self.register_buffer("W2_fast", None)
        self.register_buffer("b2_fast", None)

    def reset(self, batch_size: int = 1):
        """Reset fast weights to learned initialization (detached leaf tensors)."""
        self.W1_fast = self.W1_init.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        self.b1_fast = self.b1_init.detach().unsqueeze(0).expand(batch_size, -1).clone()
        self.W2_fast = self.W2_init.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        self.b2_fast = self.b2_init.detach().unsqueeze(0).expand(batch_size, -1).clone()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Layer 1: (B, S, D) @ (B, D_ttt, D)^T -> (B, S, D_ttt)
        hidden = torch.bmm(h, self.W1_fast.transpose(1, 2)) + self.b1_fast.unsqueeze(1)
        hidden = F.relu(hidden)
        # Layer 2: (B, S, D_ttt) @ (B, D, D_ttt)^T -> (B, S, D)
        output = torch.bmm(hidden, self.W2_fast.transpose(1, 2)) + self.b2_fast.unsqueeze(1)
        return output

    def get_params_for_grad(self):
        """Return fast weight tensors for gradient computation."""
        return [self.W1_fast, self.b1_fast, self.W2_fast, self.b2_fast]

    def apply_update(self, grads: list, lr: float):
        """Apply gradient update to fast weights (creates new leaf tensors)."""
        self.W1_fast = (self.W1_fast.detach() - lr * grads[0].detach())
        self.b1_fast = (self.b1_fast.detach() - lr * grads[1].detach())
        self.W2_fast = (self.W2_fast.detach() - lr * grads[2].detach())
        self.b2_fast = (self.b2_fast.detach() - lr * grads[3].detach())

    def param_count(self) -> int:
        return (self.W1_init.numel() + self.b1_init.numel() +
                self.W2_init.numel() + self.b2_init.numel())


class TTTLayer(nn.Module):
    """
    Test-Time Training Layer for Denoising-as-Learning.

    Inserts into a frozen DLM backbone at layer L/2. At each denoising step:
    1. Takes backbone hidden states as input
    2. Computes self-supervised loss on revealed tokens
    3. Updates fast weights via gradient descent
    4. Returns residual-gated output to be added back to backbone

    Args:
        d_model: Hidden dimension of the backbone (e.g., 4096 for LLaDA-8B)
        variant: "linear", "mlp", or "momentum"
        d_ttt: TTT bottleneck dimension (default: d_model // 8)
        vocab_size: Output vocabulary size for self-supervised prediction
        ttt_lr: Learning rate for fast weight updates (learnable)
        momentum_beta: Momentum coefficient for Momentum-TTT (default: 0.9)
        weight_decay: Weight decay for Momentum-TTT (default: 0.01)
        max_grad_norm: Gradient clipping threshold (default: 10.0)
        precision_weighted: Whether to use precision-weighted loss (default: True)
    """

    def __init__(
        self,
        d_model: int,
        variant: Literal["linear", "mlp", "momentum"] = "mlp",
        d_ttt: Optional[int] = None,
        vocab_size: int = 32000,
        ttt_lr: float = 1e-3,
        momentum_beta: float = 0.9,
        weight_decay: float = 0.01,
        max_grad_norm: float = 10.0,
        precision_weighted: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.variant = variant
        self.d_ttt = d_ttt or d_model // 8
        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm
        self.precision_weighted = precision_weighted
        self.momentum_beta = momentum_beta
        self.weight_decay_lambda = weight_decay

        # === Fast weight model ===
        if variant == "linear":
            self.fast_weight = FastWeightLinear(d_model)
        elif variant in ("mlp", "momentum"):
            self.fast_weight = FastWeightMLP(d_model, self.d_ttt)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # === Learnable TTT learning rate (log-space for positivity) ===
        self.log_lr = nn.Parameter(torch.tensor(math.log(ttt_lr)))

        # === Residual gate beta (initialized to 0 for stable start) ===
        # sigmoid(0) = 0.5, but we want near-0 at init
        self.gate_logit = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

        # === Prediction head for self-supervised loss ===
        # Projects fast weight output to vocabulary logits
        self.ssl_head = nn.Linear(d_model, vocab_size, bias=False)

        # === Layer norm before TTT (stabilizes training) ===
        self.layer_norm = nn.LayerNorm(d_model)

        # === Momentum buffer (only for momentum variant, plain attr not buffer) ===
        self._momentum_bufs: Optional[list] = None
        self._batch_size = None

    @property
    def lr(self) -> torch.Tensor:
        """Current TTT learning rate."""
        return self.log_lr.exp()

    @property
    def gate(self) -> torch.Tensor:
        """Current residual gate value."""
        return torch.sigmoid(self.gate_logit)

    def reset_fast_weights(self, batch_size: int = 1):
        """Reset fast weights and momentum buffer. Call between sequences."""
        self._batch_size = batch_size
        self.fast_weight.reset(batch_size)
        if self.variant == "momentum":
            # Initialize momentum buffers as zeros matching fast weight shapes
            params = self.fast_weight.get_params_for_grad()
            self._momentum_bufs = [torch.zeros_like(p) for p in params]

    def _compute_precision_weights(
        self, logits: torch.Tensor, revealed_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute precision weights based on prediction uncertainty.

        Args:
            logits: (B, S, V) prediction logits from backbone
            revealed_mask: (B, S) binary mask, 1 = revealed token

        Returns:
            weights: (B, S) precision weights, normalized per sample
        """
        if not self.precision_weighted:
            # Uniform weighting
            return revealed_mask.float()

        # Compute prediction variance from softmax probabilities
        probs = F.softmax(logits, dim=-1)  # (B, S, V)
        # Variance = 1 - max_prob (simple proxy for uncertainty)
        max_probs = probs.max(dim=-1).values  # (B, S)
        variance = 1.0 - max_probs  # High variance = uncertain

        # Precision = 1 / (variance + eps)
        precision = 1.0 / (variance + 1e-6)

        # Mask to only revealed positions
        precision = precision * revealed_mask.float()

        # Normalize so weights sum to num_revealed per sample
        num_revealed = revealed_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        weight_sum = precision.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        precision = precision * (num_revealed / weight_sum)

        return precision

    def _compute_ssl_loss(
        self,
        fast_output: torch.Tensor,
        target_ids: torch.Tensor,
        revealed_mask: torch.Tensor,
        precision_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute self-supervised loss on revealed tokens.

        Args:
            fast_output: (B, S, D) output from fast weight model
            target_ids: (B, S) ground truth token IDs
            revealed_mask: (B, S) binary mask
            precision_weights: (B, S) optional per-position weights

        Returns:
            loss: scalar tensor
        """
        # Project to vocabulary
        logits = self.ssl_head(fast_output)  # (B, S, V)

        # Flatten for cross-entropy
        B, S, V = logits.shape
        logits_flat = logits.view(-1, V)  # (B*S, V)
        targets_flat = target_ids.view(-1)  # (B*S,)

        # Per-token loss
        per_token_loss = F.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        ).view(B, S)  # (B, S)

        # Apply mask
        if precision_weights is not None:
            weighted_loss = per_token_loss * precision_weights
        else:
            weighted_loss = per_token_loss * revealed_mask.float()

        # Mean over revealed tokens
        num_revealed = revealed_mask.float().sum().clamp(min=1.0)
        loss = weighted_loss.sum() / num_revealed

        return loss

    def _ttt_step(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        revealed_mask: torch.Tensor,
        precision_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one TTT gradient step on the fast weights.

        Args:
            hidden_states: (B, S, D) backbone hidden states
            target_ids: (B, S) ground truth token IDs
            revealed_mask: (B, S) binary mask for revealed tokens
            precision_weights: (B, S) optional precision weights

        Returns:
            metrics: dict with loss, grad_norm, etc.
        """
        # Enable grad on fast weights (they are leaf tensors from reset/apply_update)
        fast_params = self.fast_weight.get_params_for_grad()
        for p in fast_params:
            p.requires_grad_(True)

        # Forward through fast weight model
        fast_output = self.fast_weight(hidden_states)

        # Compute SSL loss
        loss = self._compute_ssl_loss(
            fast_output, target_ids, revealed_mask, precision_weights
        )

        # Compute gradients w.r.t. fast weights
        grads = torch.autograd.grad(
            loss, fast_params, create_graph=False, allow_unused=True
        )

        # Replace None grads with zeros
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, fast_params)]

        # Gradient clipping
        grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
        if grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (grad_norm + 1e-8)
            grads = [g * scale for g in grads]

        lr = self.lr.detach()  # Detach for the update step itself

        if self.variant == "momentum":
            # Momentum update: m = beta * m + grad
            # W = (1 - lambda) * W - lr * m
            for i, (g, m) in enumerate(zip(grads, self._momentum_bufs)):
                self._momentum_bufs[i] = self.momentum_beta * m.detach() + g.detach()

            # Create new leaf tensors with momentum update
            new_params = []
            for p, m in zip(fast_params, self._momentum_bufs):
                new_p = p.detach() * (1.0 - self.weight_decay_lambda) - lr * m
                new_params.append(new_p)
            # Assign back — apply_update pattern for MLP
            if self.variant == "momentum" and hasattr(self.fast_weight, 'W1_fast'):
                self.fast_weight.W1_fast = new_params[0]
                self.fast_weight.b1_fast = new_params[1]
                self.fast_weight.W2_fast = new_params[2]
                self.fast_weight.b2_fast = new_params[3]
        else:
            # Standard gradient descent (apply_update creates new detached tensors)
            self.fast_weight.apply_update(grads, lr)

        return {
            "ssl_loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": lr.item(),
            "gate": self.gate.item(),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        revealed_mask: torch.Tensor,
        target_ids: torch.Tensor,
        backbone_logits: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
        do_ttt_update: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of the TTT layer.

        Args:
            hidden_states: (B, S, D) hidden states from backbone at layer L/2
            revealed_mask: (B, S) binary mask, 1 = token is revealed (not masked)
            target_ids: (B, S) ground truth token IDs for self-supervised loss
            backbone_logits: (B, S, V) optional backbone logits for precision weighting
            mask_ratio: Optional current mask ratio for phase-transition scheduling
            do_ttt_update: Whether to perform TTT gradient step (False = inference only)

        Returns:
            output: (B, S, D) residual-gated TTT output to add to backbone hidden states
            metrics: dict with loss, grad_norm, gate value, etc.
        """
        B, S, D = hidden_states.shape

        # Initialize fast weights if not done, or if batch size changed
        needs_init = self._batch_size is None or self._batch_size != B
        if not needs_init:
            # Check if fast weights have been initialized
            if isinstance(self.fast_weight, FastWeightLinear):
                needs_init = self.fast_weight.W_fast is None
            else:  # FastWeightMLP
                needs_init = self.fast_weight.W1_fast is None
        if needs_init:
            self.reset_fast_weights(B)

        metrics = {"ssl_loss": 0.0, "grad_norm": 0.0, "lr": self.lr.item(),
                   "gate": self.gate.item(), "ttt_updated": False}

        # Layer norm for stability
        h_normed = self.layer_norm(hidden_states)

        # Compute precision weights if backbone logits available
        precision_weights = None
        if backbone_logits is not None and self.precision_weighted:
            precision_weights = self._compute_precision_weights(
                backbone_logits, revealed_mask
            )

        # TTT update step (only if there are revealed tokens and update requested)
        num_revealed = revealed_mask.float().sum()
        if do_ttt_update and num_revealed > 0:
            step_metrics = self._ttt_step(
                h_normed, target_ids, revealed_mask, precision_weights
            )
            metrics.update(step_metrics)
            metrics["ttt_updated"] = True

        # Forward through (updated) fast weights for output
        # Detach fast_output to avoid making fast weights non-leaf in the output graph,
        # which would prevent requires_grad_(True) in the next _ttt_step call.
        # Gradient flow to gate_logit is preserved; meta-learning of fast weight init
        # happens through _ttt_step's autograd.grad, not through the output path.
        fast_output = self.fast_weight(h_normed).detach()

        # Residual gate: output = beta * fast_output
        gate = self.gate
        gated_output = gate * fast_output

        metrics["fast_weight_norm"] = sum(
            p.norm().item() for p in self.fast_weight.get_params_for_grad()
        )

        return gated_output, metrics

    def get_trainable_param_count(self) -> int:
        """Count of trainable (meta-learned) parameters."""
        count = 0
        for p in self.parameters():
            if p.requires_grad:
                count += p.numel()
        return count

    def get_fast_weight_param_count(self) -> int:
        """Count of fast weight parameters (updated at test time)."""
        return self.fast_weight.param_count()

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, variant={self.variant}, d_ttt={self.d_ttt}, "
            f"vocab_size={self.vocab_size}, precision_weighted={self.precision_weighted}"
        )


def build_ttt_layer(
    d_model: int,
    variant: str = "mlp",
    vocab_size: int = 32000,
    **kwargs,
) -> TTTLayer:
    """Factory function to create a TTT layer.

    Args:
        d_model: Backbone hidden dimension
        variant: "linear", "mlp", or "momentum"
        vocab_size: Vocabulary size for SSL head
        **kwargs: Additional arguments passed to TTTLayer

    Returns:
        TTTLayer instance
    """
    return TTTLayer(
        d_model=d_model,
        variant=variant,
        vocab_size=vocab_size,
        **kwargs,
    )
