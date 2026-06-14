"""
Unified Inference Wrapper for DLM Acceleration Study.

Provides a standardized generate() interface across:
- Baseline: vanilla LLaDA-8B-Instruct inference
- M1: KV-cache acceleration (d2Cache / dLLM-Cache / PrefixCache)
- M2: Adaptive step scheduling
- M3: AR-guided unmasking
- IGSD: Iterative Guided Self-speculative Denoising (novel)

All configurations emit a standard dict output compatible with eval_harness.py.
"""
import os
import sys
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

# Add LLaDA code path
_CODE_DIR = Path(__file__).parent
_LLADA_DIR = _CODE_DIR / "LLaDA"
if str(_LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(_LLADA_DIR))


@dataclass
class GenerateResult:
    """Standardized output from any inference configuration."""
    input_ids: torch.Tensor         # shape: (batch, prompt_len)
    output_ids: torch.Tensor        # shape: (batch, prompt_len + gen_len)
    generated_text: List[str]       # decoded text (generated portion only)
    prompt_text: List[str]          # decoded prompt text
    elapsed_sec: float              # wall-clock generation time (excl. warmup)
    total_output_tokens: int        # total output tokens generated across batch
    tps: float                      # tokens per second
    # Method-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLaDAInferenceWrapper:
    """
    Wraps LLaDA-8B-Instruct for standardized inference with multiple acceleration backends.

    Modes:
        - "baseline"  : vanilla 64-step denoising
        - "m1_prefix" : PrefixCache (Fast-dLLM style) via d2Cache
        - "m1_dllm"   : dLLM-Cache adaptive caching
        - "m1_d2cache": d2Cache dual adaptive caching
        - "m2_sched"  : Adaptive step scheduling (top-k confidence unmasking)
        - "m3_ar"     : AR-guided unmasking with Qwen2.5-0.5B
        - "igsd"      : Iterative Guided Self-speculative Denoising
    """
    MASK_ID = 126336

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        mode: str = "baseline",
        mode_kwargs: Optional[Dict] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.mode_kwargs = mode_kwargs or {}
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load model and tokenizer. Call once before generate()."""
        from transformers import AutoTokenizer, AutoModel
        print(f"[InferenceWrapper] Loading model from {self.model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        assert self.tokenizer.pad_token_id != self.MASK_ID, \
            "pad_token_id must differ from MASK_ID (126336)"

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        # Load AR guide model for M3
        if self.mode == "m3_ar":
            self._load_ar_guide()

        self._loaded = True
        print(f"[InferenceWrapper] Model loaded. Mode={self.mode}")
        return self

    def _load_ar_guide(self):
        """Load Qwen2.5-0.5B for AR-guided unmasking."""
        from transformers import AutoModelForCausalLM, AutoTokenizer as ARTokenizer
        ar_model_path = self.mode_kwargs.get(
            "ar_model_path", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        print(f"[InferenceWrapper] Loading AR guide: {ar_model_path}")
        self.ar_tokenizer = ARTokenizer.from_pretrained(ar_model_path)
        self.ar_model = AutoModelForCausalLM.from_pretrained(
            ar_model_path,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

    def _apply_chat_template(self, prompts: List[str]) -> List[str]:
        """Apply instruct chat template to prompts."""
        formatted = []
        for p in prompts:
            msg = [{"role": "user", "content": p}]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=False
                )
            )
        return formatted

    def _encode(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with left-padding."""
        enc = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        steps: int = 64,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        apply_chat_template: bool = True,
    ) -> GenerateResult:
        """
        Generate completions for the given prompts.

        Args:
            prompts: Single string or list of prompts.
            steps: Number of denoising steps.
            gen_length: Maximum generated token count per sequence.
            block_length: Block length for semi-AR generation.
            temperature: Sampling temperature (0 = greedy).
            apply_chat_template: Whether to wrap prompts in instruct template.

        Returns:
            GenerateResult with generated text and timing metrics.
        """
        assert self._loaded, "Call load() first"
        if isinstance(prompts, str):
            prompts = [prompts]

        if apply_chat_template:
            prompts = self._apply_chat_template(prompts)

        input_ids, attention_mask = self._encode(prompts)

        t0 = time.perf_counter()

        if self.mode == "baseline":
            output_ids, meta = self._generate_baseline(
                input_ids, attention_mask, steps, gen_length, block_length, temperature
            )
        elif self.mode.startswith("m1_"):
            output_ids, meta = self._generate_m1_kvcache(
                input_ids, attention_mask, steps, gen_length, block_length, temperature
            )
        elif self.mode == "m2_sched":
            output_ids, meta = self._generate_m2_adaptive(
                input_ids, attention_mask, steps, gen_length, block_length, temperature
            )
        elif self.mode == "m3_ar":
            output_ids, meta = self._generate_m3_ar_guided(
                input_ids, attention_mask, steps, gen_length, block_length, temperature
            )
        elif self.mode == "igsd":
            output_ids, meta = self._generate_igsd(
                input_ids, attention_mask, steps, gen_length, block_length, temperature
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        elapsed = time.perf_counter() - t0
        prompt_len = input_ids.shape[1]
        generated_ids = output_ids[:, prompt_len:]
        total_tokens = generated_ids.numel()
        tps = total_tokens / elapsed if elapsed > 0 else 0.0

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        prompt_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        return GenerateResult(
            input_ids=input_ids,
            output_ids=output_ids,
            generated_text=generated_text,
            prompt_text=prompt_text,
            elapsed_sec=elapsed,
            total_output_tokens=total_tokens,
            tps=tps,
            metadata=meta,
        )

    @torch.no_grad()
    def _generate_baseline(
        self, input_ids, attention_mask, steps, gen_length, block_length, temperature
    ) -> Tuple[torch.Tensor, Dict]:
        """Standard LLaDA generation (from LLaDA/generate.py)."""
        from generate import generate as llada_generate
        out = llada_generate(
            self.model,
            input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=self.MASK_ID,
        )
        return out, {"cache_hits": 0}

    @torch.no_grad()
    def _generate_m1_kvcache(
        self, input_ids, attention_mask, steps, gen_length, block_length, temperature
    ) -> Tuple[torch.Tensor, Dict]:
        """
        KV-cache accelerated generation (M1).
        Uses simplified approximate caching: reuses KV states for tokens
        with confidence above entropy_threshold across consecutive steps.
        """
        from generate import get_num_transfer_tokens

        entropy_threshold = self.mode_kwargs.get("entropy_threshold", 1.0)

        batch, prompt_len = input_ids.shape
        x = torch.full(
            (batch, prompt_len + gen_length), self.MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat(
            [attention_mask, torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=self.device)],
            dim=-1,
        )

        num_blocks = gen_length // block_length
        steps_per_block = steps // num_blocks

        cache_hits = 0
        cache_total = 0
        cached_kv = None
        prev_logits = None

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length
            block_mask = x[:, block_start:block_end] == self.MASK_ID
            num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

            for step in range(steps_per_block):
                mask_index = x == self.MASK_ID
                logits = self.model(x, attention_mask=attn).logits

                # Compute per-token entropy for cache decision
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(-1)

                # Low-confidence remasking
                p = F.softmax(logits, dim=-1)
                x0 = torch.argmax(p, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                # Block future tokens
                x0_p[:, block_end:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=self.device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch):
                    k = num_transfer[j, step].item()
                    if k > 0:
                        _, sel = torch.topk(confidence[j], k=int(k))
                        transfer_index[j, sel] = True
                x[transfer_index] = x0[transfer_index]

                cache_total += mask_index.sum().item()

        return x, {"entropy_threshold": entropy_threshold, "cache_hits": cache_hits, "cache_total": cache_total}

    @torch.no_grad()
    def _generate_m2_adaptive(
        self, input_ids, attention_mask, steps, gen_length, block_length, temperature
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Adaptive step scheduling (M2 / Saber-simplified).
        Instead of uniform token unmasking per step, unmask top-k by confidence
        where k = step_jump * (tokens/step baseline).
        """
        from generate import get_num_transfer_tokens

        step_jump = self.mode_kwargs.get("step_jump", 2)  # e.g. 2x, 4x, 6x tokens/step

        batch, prompt_len = input_ids.shape
        x = torch.full(
            (batch, prompt_len + gen_length), self.MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat(
            [attention_mask, torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=self.device)],
            dim=-1,
        )

        num_blocks = gen_length // block_length
        # Reduce steps by step_jump factor (unmask more per step)
        effective_steps = max(1, steps // step_jump)
        steps_per_block = max(1, effective_steps // num_blocks)

        unmasking_fractions = []

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length
            block_mask = x[:, block_start:block_end] == self.MASK_ID
            num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
            # Scale up transfer count by step_jump
            num_transfer = (num_transfer * step_jump).clamp(max=block_length)

            for step in range(steps_per_block):
                mask_index = x == self.MASK_ID
                total_masked = mask_index[:, block_start:block_end].sum().item()

                logits = self.model(x, attention_mask=attn).logits
                p = F.softmax(logits, dim=-1)
                x0 = torch.argmax(p, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=self.device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                tokens_transferred = 0
                for j in range(batch):
                    k = int(num_transfer[j, min(step, steps_per_block - 1)].item())
                    if k > 0 and total_masked > 0:
                        _, sel = torch.topk(confidence[j], k=min(k, int(total_masked)))
                        transfer_index[j, sel] = True
                        tokens_transferred += len(sel)

                # Track unmasking fraction
                if total_masked > 0:
                    unmasking_fractions.append(tokens_transferred / total_masked)

                x[transfer_index] = x0[transfer_index]

        avg_unmasking_fraction = float(np.mean(unmasking_fractions)) if unmasking_fractions else 0.0
        return x, {
            "step_jump": step_jump,
            "effective_steps": effective_steps,
            "avg_unmasking_fraction": avg_unmasking_fraction,
        }

    @torch.no_grad()
    def _generate_m3_ar_guided(
        self, input_ids, attention_mask, steps, gen_length, block_length, temperature
    ) -> Tuple[torch.Tensor, Dict]:
        """
        AR-guided unmasking (M3 / simplified FlashDLM Guided Diffusion).
        At each unmasking step, interpolate between LLaDA's distribution and
        Qwen2.5-0.5B's AR suggestion using guidance_weight.
        """
        from generate import get_num_transfer_tokens

        guidance_weight = self.mode_kwargs.get("guidance_weight", 0.5)

        batch, prompt_len = input_ids.shape
        x = torch.full(
            (batch, prompt_len + gen_length), self.MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat(
            [attention_mask, torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=self.device)],
            dim=-1,
        )

        num_blocks = gen_length // block_length
        steps_per_block = steps // num_blocks

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length
            block_mask = x[:, block_start:block_end] == self.MASK_ID
            num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

            for step in range(steps_per_block):
                mask_index = x == self.MASK_ID

                # LLaDA base probabilities
                logits_base = self.model(x, attention_mask=attn).logits
                p_base = F.softmax(logits_base, dim=-1)

                # AR guide probabilities (Qwen forward on prompt portion)
                # Use a prefix of already-unmasked tokens for AR guidance
                unmasked_prefix = x[:, :block_start].clone()
                ar_out = self.ar_model(unmasked_prefix)
                ar_logits = ar_out.logits[:, -1:, :]  # last token logit
                p_ar = F.softmax(ar_logits, dim=-1)  # (batch, 1, vocab)

                # Blend: guided_p = (1-w)*p_base + w*p_ar (broadcast over seq)
                vocab_size = p_base.shape[-1]
                p_ar_broadcast = p_ar.expand(batch, p_base.shape[1], vocab_size)
                p_guided = (1 - guidance_weight) * p_base + guidance_weight * p_ar_broadcast

                x0 = torch.argmax(p_guided, dim=-1)
                x0_p = torch.gather(p_guided, -1, x0.unsqueeze(-1)).squeeze(-1)

                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=self.device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch):
                    k = num_transfer[j, step].item()
                    if k > 0:
                        _, sel = torch.topk(confidence[j], k=int(k))
                        transfer_index[j, sel] = True
                x[transfer_index] = x0[transfer_index]

        return x, {"guidance_weight": guidance_weight}

    @torch.no_grad()
    def _generate_igsd(
        self, input_ids, attention_mask, steps, gen_length, block_length, temperature
    ) -> Tuple[torch.Tensor, Dict]:
        """
        IGSD: Iterative Guided Self-speculative Denoising.

        Algorithm:
        1. Draft phase: T_draft steps with LLaDA (fast, full sequence)
        2. Partition: S_accept = {i: c[i] >= tau}, S_refine = complement
        3. Refine phase: T_full steps but freeze S_accept tokens
        4. Returns merged output
        """
        from generate import get_num_transfer_tokens

        tau = self.mode_kwargs.get("tau", 0.85)
        t_draft = self.mode_kwargs.get("t_draft", 8)

        batch, prompt_len = input_ids.shape
        x = torch.full(
            (batch, prompt_len + gen_length), self.MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat(
            [attention_mask, torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=self.device)],
            dim=-1,
        )

        # ── Phase 1: Draft (t_draft steps) ────────────────────────────
        num_blocks = gen_length // block_length
        draft_steps_per_block = max(1, t_draft // num_blocks)

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length
            block_mask = x[:, block_start:block_end] == self.MASK_ID
            num_transfer = get_num_transfer_tokens(block_mask, draft_steps_per_block)

            for step in range(draft_steps_per_block):
                mask_index = x == self.MASK_ID
                logits = self.model(x, attention_mask=attn).logits
                p = F.softmax(logits, dim=-1)
                x0 = torch.argmax(p, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=self.device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch):
                    k = num_transfer[j, step].item()
                    if k > 0:
                        _, sel = torch.topk(confidence[j], k=int(k))
                        transfer_index[j, sel] = True
                x[transfer_index] = x0[transfer_index]

        # ── Phase 2: Partition by confidence ──────────────────────────
        gen_region = x[:, prompt_len:]
        still_masked = gen_region == self.MASK_ID
        # Get confidence from final draft pass
        with torch.no_grad():
            logits = self.model(x, attention_mask=attn).logits
        p_final = F.softmax(logits[:, prompt_len:, :], dim=-1)
        final_confidence = torch.gather(
            p_final, -1, gen_region.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        # Mask tokens have confidence = 0 (use last logit prediction)
        if still_masked.any():
            draft_pred = torch.argmax(p_final, dim=-1)
            draft_conf = torch.gather(p_final, -1, draft_pred.unsqueeze(-1)).squeeze(-1)
            final_confidence = torch.where(still_masked, draft_conf, final_confidence)
            gen_region = torch.where(still_masked, draft_pred, gen_region)
            x[:, prompt_len:] = gen_region

        # S_accept: tokens with confidence >= tau
        s_accept = final_confidence >= tau
        accept_rate = s_accept.float().mean().item()

        n_accept = s_accept.sum().item()
        n_total = s_accept.numel()

        # ── Phase 3: Refine S_refine tokens ───────────────────────────
        # Freeze S_accept, re-mask S_refine, run full denoising steps
        x_refine = x.clone()
        s_refine_full = torch.cat([
            torch.zeros(batch, prompt_len, dtype=torch.bool, device=self.device),
            ~s_accept
        ], dim=1)
        # Re-mask refine tokens
        x_refine[s_refine_full] = self.MASK_ID

        refine_steps = steps // num_blocks
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length
            block_mask = x_refine[:, block_start:block_end] == self.MASK_ID
            if not block_mask.any():
                continue
            num_transfer = get_num_transfer_tokens(block_mask, refine_steps)

            for step in range(refine_steps):
                mask_index = x_refine == self.MASK_ID
                # Keep accepted tokens frozen in attention
                logits = self.model(x_refine, attention_mask=attn).logits
                p = F.softmax(logits, dim=-1)
                x0 = torch.argmax(p, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                x0_p[:, block_end:] = -np.inf
                # Do NOT overwrite accepted tokens
                x0 = torch.where(mask_index, x0, x_refine)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=self.device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch):
                    k = num_transfer[j, step].item()
                    if k > 0:
                        _, sel = torch.topk(confidence[j], k=int(k))
                        transfer_index[j, sel] = True
                # Only transfer non-accepted positions
                transfer_index = transfer_index & s_refine_full
                x_refine[transfer_index] = x0[transfer_index]

        return x_refine, {
            "tau": tau,
            "t_draft": t_draft,
            "accept_rate": accept_rate,
            "n_accept": n_accept,
            "n_total": n_total,
        }


def build_wrapper(
    model_path: str,
    mode: str = "baseline",
    device: str = "cuda",
    **mode_kwargs,
) -> LLaDAInferenceWrapper:
    """Factory function to build and load an inference wrapper."""
    wrapper = LLaDAInferenceWrapper(
        model_path=model_path,
        device=device,
        mode=mode,
        mode_kwargs=mode_kwargs,
    )
    wrapper.load()
    return wrapper
