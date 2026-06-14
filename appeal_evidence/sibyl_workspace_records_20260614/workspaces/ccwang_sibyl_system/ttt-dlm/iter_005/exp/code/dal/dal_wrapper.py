#!/usr/bin/env python3
"""
dal_wrapper.py — DaL (Denoising-as-Learning) Backbone Wrapper

Inserts TTT layer(s) into frozen LLaDA-8B / Dream-7B at layer L/2.
Handles:
  1. Extracting hidden states at insertion point via forward hooks
  2. Running TTT forward + self-supervised loss on revealed tokens R_t
  3. Gradient update of fast weights
  4. Residual injection back into backbone
  5. Fast weight persistence across denoising steps

Also implements MetaState-GRU baseline (simplified: GRU updater with
cross-attention mixer/injector) for controlled comparison.

Usage:
    wrapper = DaLWrapper(backbone_model, backbone_type="dream", variant="mlp")
    # For each sequence:
    wrapper.reset_state(batch_size=B)
    # For each denoising step:
    logits, metrics = wrapper.forward_step(input_ids, revealed_mask, target_ids, mask_ratio)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal, Tuple, List

# Import TTT layer from same directory
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttt_layer import TTTLayer, build_ttt_layer


# ==============================================================================
# Phase-Transition Scheduler
# ==============================================================================

class PhaseTransitionScheduler:
    """
    Concentrates TTT updates in the critical mask-ratio zone.

    Based on statistical physics analysis: DLM denoising undergoes phase
    transitions at r_crit ~ 0.4-0.5. TTT updates are most valuable here.

    Gaussian weighting centered at r_crit, with configurable sigma.
    Skips updates when mask ratio > high_cutoff or < low_cutoff.
    """

    def __init__(
        self,
        r_crit: float = 0.45,
        sigma: float = 0.15,
        high_cutoff: float = 0.80,
        low_cutoff: float = 0.15,
    ):
        self.r_crit = r_crit
        self.sigma = sigma
        self.high_cutoff = high_cutoff
        self.low_cutoff = low_cutoff

    def should_update(self, mask_ratio: float) -> bool:
        """Whether to perform TTT update at this mask ratio."""
        if mask_ratio > self.high_cutoff or mask_ratio < self.low_cutoff:
            return False
        return True

    def get_weight(self, mask_ratio: float) -> float:
        """Gaussian weight for TTT learning rate scaling."""
        if not self.should_update(mask_ratio):
            return 0.0
        z = (mask_ratio - self.r_crit) / self.sigma
        return math.exp(-0.5 * z * z)


# ==============================================================================
# MetaState-GRU Baseline
# ==============================================================================

class MetaStateGRU(nn.Module):
    """
    Simplified MetaState baseline: GRU updater with cross-attention mixer/injector.

    Architecture:
      - Mixer: Cross-attention from memory state to backbone hidden states
      - Updater: GRU cell that updates memory state
      - Injector: Cross-attention from backbone to (updated) memory state

    The memory state s_t persists across denoising steps.
    Parameter budget is matched to DaL variants (±10%).
    """

    def __init__(
        self,
        d_model: int,
        d_state: Optional[int] = None,
        num_state_tokens: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state or d_model
        self.num_state_tokens = num_state_tokens

        # Learnable initial state: (1, num_state_tokens, d_state)
        self.state_init = nn.Parameter(
            torch.randn(1, num_state_tokens, self.d_state) * 0.02
        )

        # Mixer: Cross-attention (state queries backbone)
        self.mixer_q = nn.Linear(self.d_state, self.d_state, bias=False)
        self.mixer_k = nn.Linear(d_model, self.d_state, bias=False)
        self.mixer_v = nn.Linear(d_model, self.d_state, bias=False)
        self.mixer_scale = 1.0 / math.sqrt(self.d_state)

        # GRU Updater
        self.gru = nn.GRUCell(self.d_state, self.d_state)

        # Injector: Cross-attention (backbone queries state)
        self.injector_q = nn.Linear(d_model, self.d_state, bias=False)
        self.injector_k = nn.Linear(self.d_state, self.d_state, bias=False)
        self.injector_v = nn.Linear(self.d_state, d_model, bias=False)

        # Residual gate (same as DaL for fair comparison)
        self.gate_logit = nn.Parameter(torch.tensor(-5.0))

        # Layer norms
        self.ln_state = nn.LayerNorm(self.d_state)
        self.ln_hidden = nn.LayerNorm(d_model)

        # Runtime state buffer
        self._state: Optional[torch.Tensor] = None

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def reset_state(self, batch_size: int = 1):
        """Reset memory state to learned initialization."""
        self._state = self.state_init.expand(batch_size, -1, -1).clone()

    def forward(
        self,
        hidden_states: torch.Tensor,
        revealed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hidden_states: (B, S, D) backbone hidden states
            revealed_mask: (B, S) optional, not used by GRU but kept for API compat

        Returns:
            output: (B, S, D) residual-gated output
            metrics: dict with state_norm, gate value
        """
        B, S, D = hidden_states.shape

        if self._state is None:
            self.reset_state(B)

        h_normed = self.ln_hidden(hidden_states)
        s = self.ln_state(self._state)

        # === Mixer: state queries backbone ===
        # q: (B, N, d_state), k: (B, S, d_state), v: (B, S, d_state)
        q = self.mixer_q(s)
        k = self.mixer_k(h_normed)
        v = self.mixer_v(h_normed)

        # Attention: (B, N, S)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.mixer_scale
        attn = F.softmax(attn, dim=-1)
        mixed = torch.bmm(attn, v)  # (B, N, d_state)

        # === GRU Update: per state token ===
        # Flatten state tokens for GRU
        s_flat = self._state.reshape(B * self.num_state_tokens, self.d_state)
        mixed_flat = mixed.reshape(B * self.num_state_tokens, self.d_state)
        s_new = self.gru(mixed_flat, s_flat)
        self._state = s_new.reshape(B, self.num_state_tokens, self.d_state)

        # === Injector: backbone queries updated state ===
        q_inj = self.injector_q(h_normed)     # (B, S, d_state)
        k_inj = self.injector_k(self._state)   # (B, N, d_state)
        v_inj = self.injector_v(self._state)   # (B, N, D)

        attn_inj = torch.bmm(q_inj, k_inj.transpose(1, 2)) * self.mixer_scale
        attn_inj = F.softmax(attn_inj, dim=-1)
        injected = torch.bmm(attn_inj, v_inj)  # (B, S, D)

        # Residual gate
        output = self.gate * injected

        metrics = {
            "state_norm": self._state.norm().item(),
            "gate": self.gate.item(),
        }
        return output, metrics

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# DaL Backbone Wrapper
# ==============================================================================

class DaLWrapper(nn.Module):
    """
    Wraps a frozen DLM backbone (LLaDA-8B or Dream-7B) with DaL TTT layer(s)
    or MetaState-GRU baseline.

    The wrapper:
      1. Hooks into the backbone at layer L/2 to extract hidden states
      2. Passes hidden states through the TTT layer (or GRU)
      3. Adds the residual output back into the backbone's hidden state stream
      4. Returns final logits with cross-step memory accumulated

    Args:
        backbone: The frozen DLM model
        backbone_type: "dream" or "llada"
        variant: "linear", "mlp", "momentum", or "metastate_gru"
        insertion_layer: Which layer to insert at (default: L//2)
        phase_scheduling: Whether to use phase-transition scheduling
        precision_weighted: Whether to use precision-weighted loss
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_type: Literal["dream", "llada"],
        variant: Literal["linear", "mlp", "momentum", "metastate_gru"] = "mlp",
        insertion_layer: Optional[int] = None,
        phase_scheduling: bool = False,
        precision_weighted: bool = True,
        ttt_lr: float = 1e-3,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.variant = variant
        self.phase_scheduling = phase_scheduling

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get backbone config
        if backbone_type == "dream":
            config = backbone.config
            self.d_model = config.hidden_size
            self.vocab_size = config.vocab_size
            self.n_layers = config.num_hidden_layers
            self.mask_token_id = config.mask_token_id
        elif backbone_type == "llada":
            # LLaDA has a nested structure
            if hasattr(backbone, 'config'):
                config = backbone.config
            else:
                raise ValueError("LLaDA backbone must have .config attribute")
            self.d_model = config.d_model if hasattr(config, 'd_model') else config.hidden_size
            self.vocab_size = config.vocab_size
            self.n_layers = config.n_layers if hasattr(config, 'n_layers') else config.num_hidden_layers
            self.mask_token_id = config.mask_token_id
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # Cache reference to inner model for LLaDA (LLaDAModelLM -> .model -> LLaDAModel)
        self._llada_inner = None
        if backbone_type == "llada":
            if hasattr(backbone, 'model') and hasattr(backbone.model, 'transformer'):
                self._llada_inner = backbone.model
            elif hasattr(backbone, 'transformer'):
                self._llada_inner = backbone
            else:
                raise ValueError("Cannot find LLaDA inner model with .transformer attribute")

        # Insertion point
        self.insertion_layer = insertion_layer if insertion_layer is not None else self.n_layers // 2

        # Build the adaptation module
        if variant == "metastate_gru":
            self.adapter = MetaStateGRU(
                d_model=self.d_model,
                d_state=self.d_model,
                num_state_tokens=8,
            )
        else:
            self.adapter = build_ttt_layer(
                d_model=self.d_model,
                variant=variant,
                vocab_size=self.vocab_size,
                ttt_lr=ttt_lr,
                precision_weighted=precision_weighted,
            )

        # Phase-transition scheduler
        self.scheduler = PhaseTransitionScheduler() if phase_scheduling else None

        # Hook storage
        self._hook_handle = None
        self._captured_hidden = None
        self._injection_delta = None

    def _get_backbone_layers(self) -> nn.ModuleList:
        """Get the list of transformer layers from backbone."""
        if self.backbone_type == "dream":
            # DreamModel -> self.model (DreamBaseModel) -> self.layers
            if hasattr(self.backbone, 'model'):
                return self.backbone.model.layers
            return self.backbone.layers
        elif self.backbone_type == "llada":
            tf = self._llada_inner.transformer
            if hasattr(tf, 'blocks'):
                return tf.blocks
            elif hasattr(tf, 'block_groups'):
                return tf.block_groups
            else:
                raise ValueError("Cannot find blocks in LLaDA transformer")
        raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

    def _register_hook(self):
        """Register a forward hook on the insertion layer to capture + inject."""
        layers = self._get_backbone_layers()
        target_layer = layers[self.insertion_layer]

        def hook_fn(module, input, output):
            # For both Dream and LLaDA, layer output[0] is hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            self._captured_hidden = hidden.clone()

            # Inject delta if available from previous computation
            if self._injection_delta is not None:
                if isinstance(output, tuple):
                    output = (hidden + self._injection_delta,) + output[1:]
                else:
                    output = hidden + self._injection_delta
                self._injection_delta = None

            return output

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def _remove_hook(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset_state(self, batch_size: int = 1):
        """
        Reset all cross-step state. Call at the start of each new sequence.

        Args:
            batch_size: Number of sequences in the batch
        """
        if self.variant == "metastate_gru":
            self.adapter.reset_state(batch_size)
        else:
            self.adapter.reset_fast_weights(batch_size)
        self._captured_hidden = None
        self._injection_delta = None

    def forward_step(
        self,
        input_ids: torch.Tensor,
        revealed_mask: torch.Tensor,
        target_ids: torch.Tensor,
        mask_ratio: Optional[float] = None,
        backbone_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single denoising step with TTT/GRU adaptation.

        This performs a TWO-PASS approach for clean integration:
          Pass 1: Forward through backbone to get hidden states at layer L/2
          Pass 2: Compute TTT/GRU delta, inject, and re-forward from L/2

        For efficiency in actual deployment, we use hooks to do this in a single pass.

        Args:
            input_ids: (B, S) current token IDs (with masked positions)
            revealed_mask: (B, S) binary mask, 1 = token is revealed (not masked)
            target_ids: (B, S) ground truth token IDs
            mask_ratio: Current mask ratio for phase-transition scheduling
            backbone_logits: Optional previous-step logits for precision weighting

        Returns:
            logits: (B, S, V) output logits
            metrics: dict with adaptation metrics
        """
        metrics = {"adapter_type": self.variant, "ttt_updated": False}

        # Check phase-transition scheduling
        do_update = True
        if self.scheduler is not None and mask_ratio is not None:
            do_update = self.scheduler.should_update(mask_ratio)
            if do_update:
                lr_scale = self.scheduler.get_weight(mask_ratio)
                metrics["phase_lr_scale"] = lr_scale
            else:
                metrics["phase_skipped"] = True

        # === Strategy: single-pass with hook ===
        # 1. Register hook on insertion layer
        # 2. Run backbone forward - hook captures hidden states
        # 3. Compute adapter output using captured hidden states
        # 4. For next step, the delta will be injected

        # But for cleaner TTT gradient flow, we use a two-phase approach:
        # Phase A: backbone forward (with any pending injection from previous step)
        # Phase B: compute adapter delta for this step

        self._register_hook()
        try:
            with torch.no_grad():
                if self.backbone_type == "dream":
                    outputs = self.backbone(
                        input_ids=input_ids,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    logits = outputs.logits
                elif self.backbone_type == "llada":
                    outputs = self.backbone(
                        input_ids=input_ids,
                        output_hidden_states=False,
                    )
                    logits = outputs.logits
                else:
                    raise ValueError(f"Unknown backbone_type: {self.backbone_type}")
        finally:
            self._remove_hook()

        # Now compute adapter output on captured hidden states
        if self._captured_hidden is not None and do_update:
            hidden = self._captured_hidden.detach()

            if self.variant == "metastate_gru":
                delta, adapter_metrics = self.adapter(hidden, revealed_mask)
                metrics.update(adapter_metrics)
                metrics["ttt_updated"] = True
            else:
                # TTT adapter
                delta, adapter_metrics = self.adapter(
                    hidden_states=hidden,
                    revealed_mask=revealed_mask,
                    target_ids=target_ids,
                    backbone_logits=backbone_logits if backbone_logits is not None else logits.detach(),
                    mask_ratio=mask_ratio,
                    do_ttt_update=True,
                )
                metrics.update(adapter_metrics)

            # Store delta for injection in next backbone forward pass
            # (This means the adaptation for step t affects the output of step t+1,
            #  which models cross-step memory accumulation)
            self._injection_delta = delta.detach()

            # For this step, also directly modify logits using the adapter output
            # by re-projecting through the backbone's final layers
            # (Simplified: use a learned projection from delta to logit space)
            # Actually, for the current step we modify logits directly via
            # the lm_head on the adapted hidden state
            if self.backbone_type == "dream":
                adapted_hidden = self._captured_hidden + delta
                # Apply final norm + lm_head
                with torch.no_grad():
                    normed = self.backbone.model.norm(adapted_hidden)
                    adapted_logits = self.backbone.lm_head(normed)
                # Blend: use adapted logits at revealed positions for the next step's benefit
                # but return backbone logits for this step's token prediction
                # (the adaptation primarily benefits future steps)
                metrics["adapted_logits_diff"] = (adapted_logits - logits).abs().mean().item()
            elif self.backbone_type == "llada":
                adapted_hidden = self._captured_hidden + delta
                tf = self._llada_inner.transformer
                with torch.no_grad():
                    normed = tf.ln_f(adapted_hidden)
                    if self._llada_inner.config.weight_tying:
                        adapted_logits = F.linear(normed, tf.wte.weight, None)
                    else:
                        adapted_logits = tf.ff_out(normed)
                metrics["adapted_logits_diff"] = (adapted_logits - logits).abs().mean().item()

        metrics["mask_ratio"] = mask_ratio

        return logits, metrics

    def forward_step_with_injection(
        self,
        input_ids: torch.Tensor,
        revealed_mask: torch.Tensor,
        target_ids: torch.Tensor,
        mask_ratio: Optional[float] = None,
        backbone_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Alternative forward that injects the adaptation into the CURRENT step's
        computation. This requires two backbone passes per step but gives more
        direct impact.

        Step 1: Forward backbone to capture hidden states at L/2
        Step 2: Compute adapter delta
        Step 3: Forward backbone again with delta injected at L/2

        More expensive (2x backbone forward) but more effective.
        """
        metrics = {"adapter_type": self.variant, "ttt_updated": False}

        do_update = True
        if self.scheduler is not None and mask_ratio is not None:
            do_update = self.scheduler.should_update(mask_ratio)
            if not do_update:
                metrics["phase_skipped"] = True

        # Pass 1: capture hidden states
        self._injection_delta = None  # No injection for pass 1
        self._register_hook()
        try:
            with torch.no_grad():
                if self.backbone_type == "dream":
                    outputs = self.backbone(
                        input_ids=input_ids,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    pass1_logits = outputs.logits
                elif self.backbone_type == "llada":
                    outputs = self.backbone(
                        input_ids=input_ids,
                        output_hidden_states=False,
                    )
                    pass1_logits = outputs.logits
        finally:
            self._remove_hook()

        if self._captured_hidden is None or not do_update:
            return pass1_logits, metrics

        # Compute adapter delta
        hidden = self._captured_hidden.detach()

        if self.variant == "metastate_gru":
            delta, adapter_metrics = self.adapter(hidden, revealed_mask)
            metrics.update(adapter_metrics)
            metrics["ttt_updated"] = True
        else:
            delta, adapter_metrics = self.adapter(
                hidden_states=hidden,
                revealed_mask=revealed_mask,
                target_ids=target_ids,
                backbone_logits=pass1_logits.detach(),
                mask_ratio=mask_ratio,
                do_ttt_update=True,
            )
            metrics.update(adapter_metrics)

        # Pass 2: forward with injection
        self._injection_delta = delta.detach()
        self._register_hook()
        try:
            with torch.no_grad():
                if self.backbone_type == "dream":
                    outputs2 = self.backbone(
                        input_ids=input_ids,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    logits = outputs2.logits
                elif self.backbone_type == "llada":
                    outputs2 = self.backbone(
                        input_ids=input_ids,
                        output_hidden_states=False,
                    )
                    logits = outputs2.logits
        finally:
            self._remove_hook()

        metrics["mask_ratio"] = mask_ratio
        metrics["logits_diff_from_injection"] = (logits - pass1_logits).abs().mean().item()

        return logits, metrics

    def get_trainable_param_count(self) -> int:
        """Count trainable parameters (adapter only, backbone is frozen)."""
        return sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)

    def get_adaptation_param_count(self) -> int:
        """Count fast weight / state parameters updated at test time."""
        if self.variant == "metastate_gru":
            return self.adapter.get_trainable_param_count()
        else:
            return self.adapter.get_fast_weight_param_count()


# ==============================================================================
# Utility: Simulate Denoising
# ==============================================================================

def simulate_denoising_step(
    input_ids: torch.Tensor,
    mask_token_id: int,
    current_mask_ratio: float,
    target_mask_ratio: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate one denoising step: unmask some tokens from current to target mask ratio.

    Args:
        input_ids: (B, S) current input with masked tokens
        mask_token_id: The ID used for masked positions
        current_mask_ratio: Current proportion of masked tokens
        target_mask_ratio: Target proportion after this step
        seed: Random seed for reproducibility

    Returns:
        new_input_ids: (B, S) with some previously masked tokens revealed
        newly_revealed: (B, S) binary mask of positions revealed in this step
    """
    if seed is not None:
        torch.manual_seed(seed)

    B, S = input_ids.shape
    is_masked = (input_ids == mask_token_id)
    n_masked = is_masked.float().sum(dim=-1, keepdim=True)
    n_to_reveal = ((current_mask_ratio - target_mask_ratio) * S)
    n_to_reveal = max(1, int(n_to_reveal))

    # Randomly select masked positions to reveal
    newly_revealed = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(B):
        masked_positions = is_masked[b].nonzero(as_tuple=True)[0]
        if len(masked_positions) > 0:
            n = min(n_to_reveal, len(masked_positions))
            perm = torch.randperm(len(masked_positions), device=input_ids.device)[:n]
            newly_revealed[b, masked_positions[perm]] = True

    return newly_revealed


def create_masked_input(
    target_ids: torch.Tensor,
    mask_token_id: int,
    mask_ratio: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a masked version of the input.

    Args:
        target_ids: (B, S) ground truth token IDs
        mask_token_id: ID to use for masked positions
        mask_ratio: Proportion of tokens to mask
        seed: Random seed

    Returns:
        input_ids: (B, S) with mask_ratio tokens replaced by mask_token_id
        revealed_mask: (B, S) binary mask, 1 = NOT masked (revealed)
    """
    if seed is not None:
        torch.manual_seed(seed)

    B, S = target_ids.shape
    mask_probs = torch.rand(B, S, device=target_ids.device)
    is_masked = mask_probs < mask_ratio

    input_ids = target_ids.clone()
    input_ids[is_masked] = mask_token_id
    revealed_mask = (~is_masked).float()

    return input_ids, revealed_mask
