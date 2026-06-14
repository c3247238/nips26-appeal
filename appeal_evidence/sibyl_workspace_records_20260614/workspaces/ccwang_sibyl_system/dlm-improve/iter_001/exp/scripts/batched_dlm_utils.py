#!/usr/bin/env python3
"""
Batched DLM Inference Utilities for LLaDA-8B.

Provides batch-parallel versions of the standard denoising, entropy-revise,
and other DLM methods.  Designed for high-VRAM GPUs (e.g. RTX PRO 6000, 97 GB)
where batch_size=1 wastes >80 GB.

Key features:
  - Left-pad prompts to uniform length with correct attention_mask
  - All denoising steps run in batch-parallel
  - Per-sample early-finish tracking (mask fully revealed)
  - Auto batch-size probing via binary search (find_max_batch_size)
  - Flash Attention 2 with fallback
  - Optional torch.compile

Usage:
    from batched_dlm_utils import BatchedDLMInference

    engine = BatchedDLMInference(
        model_path="/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct",
        device="cuda",
    )
    # auto-detect optimal batch size
    bs = engine.find_max_batch_size(gen_length=256, prompt_max_len=512)

    # batch standard denoising
    results = engine.batched_standard_denoising(
        prompt_ids_list=[[101, 202, ...], [101, 303, ...]],
        gen_length=256,
        num_steps=64,
        batch_size=bs,
    )
    # results: list[dict] with keys "generated_ids", "gen_start", "gen_end", "nfe"
"""

from __future__ import annotations

import gc
import math
import shutil
import warnings
from typing import Any

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

MASK_TOKEN_ID = 126336


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_schedule(t: int, T: int) -> float:
    """Fraction unmasked after step *t* of *T*.  t=0 -> 0, t=T -> 1."""
    return 1.0 - math.cos(math.pi * t / (2 * T))


def _left_pad_batch(
    prompt_ids_list: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad a list of variable-length prompt id lists.

    Returns:
        padded_ids:  (B, max_prompt_len)  long tensor
        attn_mask:   (B, max_prompt_len)  bool tensor (True = real token)
        prompt_lens: list[int]            original lengths
    """
    prompt_lens = [len(p) for p in prompt_ids_list]
    max_len = max(prompt_lens)
    B = len(prompt_ids_list)

    padded = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)

    for i, (ids, plen) in enumerate(zip(prompt_ids_list, prompt_lens)):
        padded[i, max_len - plen:] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[i, max_len - plen:] = True

    return padded, mask, prompt_lens


def _expand_positions(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Expand a 1-D boolean mask by a local radius."""
    if radius <= 0 or mask.numel() == 0:
        return mask.clone()
    expanded = mask.clone()
    for shift in range(1, radius + 1):
        expanded[:-shift] |= mask[shift:]
        expanded[shift:] |= mask[:-shift]
    return expanded


def _contiguous_islands(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return inclusive contiguous spans where ``mask`` is True."""
    islands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            islands.append((start, idx - 1))
            start = None
    if start is not None:
        islands.append((start, mask.numel() - 1))
    return islands


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BatchedDLMInference:
    """High-throughput batched DLM inference engine.

    Parameters
    ----------
    model_path : str
        HuggingFace model directory.
    device : str
        CUDA device (default ``"cuda"``).
    use_flash_attn : bool
        Try Flash Attention 2; falls back to default if unavailable.
    use_compile : bool
        Wrap model forward with ``torch.compile`` for extra speed.
    """

    def __init__(
        self,
        model_path: str = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct",
        device: str = "cuda",
        use_flash_attn: bool = True,
        use_compile: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.model_path = model_path

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        # Use eos as pad if no pad token defined
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # Model -- try flash_attention_2, otherwise force eager.
        # LLaDA currently does not support HuggingFace SDPA on this stack.
        attn_impl = "eager"
        if use_flash_attn:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                print("[BatchedDLM] Using Flash Attention 2")
            except ImportError:
                print("[BatchedDLM] flash-attn not installed, forcing eager attention")

        load_kwargs: dict[str, Any] = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        load_kwargs["attn_implementation"] = attn_impl

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs,
            ).to(self.device).eval()
        except ValueError as exc:
            if attn_impl != "flash_attention_2" or "does not support Flash Attention" not in str(exc):
                raise
            print("[BatchedDLM] Model rejected Flash Attention 2, retrying with eager attention")
            load_kwargs["attn_implementation"] = "eager"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs,
            ).to(self.device).eval()

        if use_compile and shutil.which("ptxas") is None:
            print("[BatchedDLM] ptxas not found, skipping torch.compile")
            use_compile = False

        if use_compile:
            print("[BatchedDLM] Compiling model forward with torch.compile")
            self.model = torch.compile(self.model)

        vram_mb = torch.cuda.memory_allocated(self.device) / 1e6
        print(f"[BatchedDLM] Model loaded. VRAM used: {vram_mb:.0f} MB")

    # ------------------------------------------------------------------
    # Batch size probing
    # ------------------------------------------------------------------

    def find_max_batch_size(
        self,
        gen_length: int = 256,
        prompt_max_len: int = 512,
        lo: int = 1,
        hi: int = 64,
        safety_margin: float = 0.90,
    ) -> int:
        """Binary-search for the largest batch size that does not OOM.

        Runs a dummy forward pass at each candidate size.  Returns
        ``floor(found_max * safety_margin)`` to leave headroom.

        Parameters
        ----------
        gen_length : int
            Number of generation tokens appended to prompts.
        prompt_max_len : int
            Maximum prompt length (used for the dummy input).
        lo, hi : int
            Search bounds.
        safety_margin : float
            Fraction of the found maximum to actually use (default 0.90).

        Returns
        -------
        int
            Safe maximum batch size (>= 1).
        """
        total_len = prompt_max_len + gen_length
        best = lo

        print(f"[BatchedDLM] Probing batch size (seq_len={total_len}, "
              f"range=[{lo}, {hi}], margin={safety_margin:.0%}) ...")

        while lo <= hi:
            mid = (lo + hi) // 2
            success = self._try_forward(mid, total_len)
            if success:
                best = mid
                lo = mid + 1
                print(f"  bs={mid} OK")
            else:
                hi = mid - 1
                print(f"  bs={mid} OOM")

        safe = max(1, int(best * safety_margin))
        print(f"[BatchedDLM] Max batch size found: {best}, using safe={safe}")
        return safe

    def _try_forward(self, batch_size: int, seq_len: int) -> bool:
        """Attempt a dummy forward pass; return True if it fits in memory."""
        try:
            dummy = torch.full(
                (batch_size, seq_len), self.pad_token_id,
                dtype=torch.long, device=self.device,
            )
            attn = torch.ones_like(dummy, dtype=torch.long)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                self.model(input_ids=dummy, attention_mask=attn)
            del dummy, attn
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return False

    # ------------------------------------------------------------------
    # Batched standard denoising
    # ------------------------------------------------------------------

    def batched_standard_denoising(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_steps: int,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """Batch-parallel standard cosine-schedule denoising.

        Parameters
        ----------
        prompt_ids_list : list[list[int]]
            Token ids for each prompt (variable length OK).
        gen_length : int
            Number of mask tokens to append and denoise.
        num_steps : int
            Total denoising steps (cosine schedule).
        batch_size : int | None
            If None, processes everything in one batch (may OOM).
        seed : int
            Base random seed; sample ``i`` gets ``seed + i``.

        Returns
        -------
        list[dict]
            Per-sample results with keys:
            ``input_ids`` (1-D tensor), ``gen_start``, ``gen_end``, ``nfe``.
        """
        if batch_size is None:
            batch_size = len(prompt_ids_list)

        all_results: list[dict] = [None] * len(prompt_ids_list)  # type: ignore[list-item]

        for chunk_start in range(0, len(prompt_ids_list), batch_size):
            chunk_end = min(chunk_start + batch_size, len(prompt_ids_list))
            chunk_prompts = prompt_ids_list[chunk_start:chunk_end]
            chunk_results = self._standard_denoising_batch(
                chunk_prompts, gen_length, num_steps, seed, chunk_start,
            )
            for i, r in enumerate(chunk_results):
                all_results[chunk_start + i] = r

        return all_results

    def _standard_denoising_batch(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_steps: int,
        seed: int,
        global_offset: int,
    ) -> list[dict]:
        """Run standard denoising on a single batch."""
        B = len(prompt_ids_list)
        device = self.device

        # Left-pad prompts
        padded_prompts, prompt_mask, prompt_lens = _left_pad_batch(
            prompt_ids_list, self.pad_token_id, device,
        )
        max_prompt_len = padded_prompts.shape[1]

        # Append gen_length mask tokens
        gen_masks = torch.full(
            (B, gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device,
        )
        input_ids = torch.cat([padded_prompts, gen_masks], dim=1)  # (B, max_prompt+gen)

        # Build attention mask: prompt real tokens + all gen positions
        gen_attn = torch.ones((B, gen_length), dtype=torch.bool, device=device)
        attn_mask = torch.cat([prompt_mask, gen_attn], dim=1)  # (B, total_len)

        # Per-sample gen boundaries (in the padded coordinate system)
        gen_start = max_prompt_len  # same for all samples after left-padding
        gen_end = max_prompt_len + gen_length

        # Track per-sample completion and NFE
        sample_done = torch.zeros(B, dtype=torch.bool, device=device)
        nfe_per_sample = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(num_steps):
            if sample_done.all():
                break

            frac_prev = cosine_schedule(step, num_steps) if step > 0 else 0.0
            frac_curr = cosine_schedule(step + 1, num_steps)

            # Identify masked positions in gen region for each sample
            gen_region = input_ids[:, gen_start:gen_end]  # (B, gen_length)
            is_masked = (gen_region == MASK_TOKEN_ID)      # (B, gen_length)
            num_masked = is_masked.sum(dim=1)              # (B,)

            # Mark newly-done samples
            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done

            # Only forward active samples
            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = input_ids[active_indices]
            active_attn = attn_mask[active_indices].long()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=active_input,
                    attention_mask=active_attn,
                )
                logits = outputs.logits  # (A, seq_len, vocab)

            nfe_per_sample[active_indices] += 1

            # Process each active sample
            active_gen_logits = logits[:, gen_start:gen_end, :]  # (A, gen_length, vocab)
            active_gen_region = gen_region[active_indices]        # (A, gen_length)
            active_is_masked = (active_gen_region == MASK_TOKEN_ID)

            for ai, gi in enumerate(active_indices):
                gi = gi.item()
                sample_masked = active_is_masked[ai]       # (gen_length,)
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]  # positions within gen region
                masked_logits = active_gen_logits[ai, masked_positions]  # (n_masked, vocab)

                probs = F.softmax(masked_logits, dim=-1)
                top1_conf, top1_token = probs.max(dim=-1)

                num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
                num_to_unmask = min(num_to_unmask, n_masked)

                _, topk_idx = top1_conf.topk(num_to_unmask)
                pos_to_unmask = masked_positions[topk_idx]
                tokens_to_place = top1_token[topk_idx]

                input_ids[gi, gen_start + pos_to_unmask] = tokens_to_place

        # Build per-sample results
        results = []
        for i in range(B):
            results.append({
                "input_ids": input_ids[i],               # full padded sequence
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": nfe_per_sample[i].item(),
                "prompt_pad_offset": max_prompt_len - prompt_lens[i],
            })

        return results

    def _draft_batch(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_steps: int,
        selection_mode: str,
    ) -> list[dict]:
        """Run a deterministic draft pass with a configurable unmask rule."""
        B = len(prompt_ids_list)
        device = self.device

        padded_prompts, prompt_mask, prompt_lens = _left_pad_batch(
            prompt_ids_list, self.pad_token_id, device,
        )
        max_prompt_len = padded_prompts.shape[1]

        gen_masks = torch.full(
            (B, gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device,
        )
        input_ids = torch.cat([padded_prompts, gen_masks], dim=1)
        gen_attn = torch.ones((B, gen_length), dtype=torch.bool, device=device)
        attn_mask = torch.cat([prompt_mask, gen_attn], dim=1)

        gen_start = max_prompt_len
        gen_end = max_prompt_len + gen_length
        sample_done = torch.zeros(B, dtype=torch.bool, device=device)
        nfe_per_sample = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(num_steps):
            if sample_done.all():
                break

            frac_prev = cosine_schedule(step, num_steps) if step > 0 else 0.0
            frac_curr = cosine_schedule(step + 1, num_steps)

            gen_region = input_ids[:, gen_start:gen_end]
            is_masked = (gen_region == MASK_TOKEN_ID)
            num_masked = is_masked.sum(dim=1)

            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done

            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = input_ids[active_indices]
            active_attn = attn_mask[active_indices].long()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=active_input,
                    attention_mask=active_attn,
                )
                logits = outputs.logits

            nfe_per_sample[active_indices] += 1

            active_gen_logits = logits[:, gen_start:gen_end, :]
            active_gen_region = gen_region[active_indices]
            active_is_masked = (active_gen_region == MASK_TOKEN_ID)

            for ai, gi_t in enumerate(active_indices):
                gi = gi_t.item()
                sample_masked = active_is_masked[ai]
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = active_gen_logits[ai, masked_positions]
                probs = F.softmax(masked_logits, dim=-1)
                top2_vals, top2_idx = probs.topk(k=min(2, probs.shape[-1]), dim=-1)
                top1_conf = top2_vals[:, 0]
                top1_token = top2_idx[:, 0]
                margin = top2_vals[:, 0] - top2_vals[:, 1] if top2_vals.shape[1] > 1 else top1_conf

                num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
                num_to_unmask = min(num_to_unmask, n_masked)

                if selection_mode == "confidence":
                    score = top1_conf
                elif selection_mode == "margin":
                    score = margin
                else:
                    raise ValueError(f"Unsupported draft selection mode: {selection_mode}")

                _, topk_idx = score.topk(num_to_unmask)
                pos_to_unmask = masked_positions[topk_idx]
                tokens_to_place = top1_token[topk_idx]
                input_ids[gi, gen_start + pos_to_unmask] = tokens_to_place

        results = []
        for i in range(B):
            results.append({
                "input_ids": input_ids[i].clone(),
                "attn_mask": attn_mask[i].clone(),
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": int(nfe_per_sample[i].item()),
                "prompt_pad_offset": max_prompt_len - prompt_lens[i],
            })
        return results

    # ------------------------------------------------------------------
    # Batched entropy-revise denoising
    # ------------------------------------------------------------------

    def batched_entropy_revise(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_draft_steps: int = 64,
        revision_fraction: float = 0.15,
        revision_steps: int = 3,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """Batch-parallel entropy-based revision (draft + entropy remask + revise).

        Parameters
        ----------
        prompt_ids_list : list[list[int]]
            Token ids for each prompt.
        gen_length : int
            Generation region length.
        num_draft_steps : int
            Draft phase denoising steps.
        revision_fraction : float
            Fraction of gen tokens to remask based on entropy.
        revision_steps : int
            Number of revision denoising steps.
        batch_size : int | None
            Chunk size; None = all at once.
        seed : int
            Base seed.

        Returns
        -------
        list[dict]
            Per-sample results with keys: ``input_ids``, ``gen_start``,
            ``gen_end``, ``nfe``, ``tokens_changed``, ``entropy_stats``.
        """
        if batch_size is None:
            batch_size = len(prompt_ids_list)

        all_results: list[dict] = [None] * len(prompt_ids_list)  # type: ignore[list-item]

        for chunk_start in range(0, len(prompt_ids_list), batch_size):
            chunk_end = min(chunk_start + batch_size, len(prompt_ids_list))
            chunk_prompts = prompt_ids_list[chunk_start:chunk_end]
            chunk_results = self._entropy_revise_batch(
                chunk_prompts, gen_length, num_draft_steps,
                revision_fraction, revision_steps, seed, chunk_start,
            )
            for i, r in enumerate(chunk_results):
                all_results[chunk_start + i] = r

        return all_results

    def _entropy_revise_batch(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_draft_steps: int,
        revision_fraction: float,
        revision_steps: int,
        seed: int,
        global_offset: int,
    ) -> list[dict]:
        """Entropy-revise on a single batch."""
        B = len(prompt_ids_list)
        device = self.device

        # Left-pad prompts
        padded_prompts, prompt_mask, prompt_lens = _left_pad_batch(
            prompt_ids_list, self.pad_token_id, device,
        )
        max_prompt_len = padded_prompts.shape[1]

        # Append gen_length mask tokens
        gen_masks = torch.full(
            (B, gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device,
        )
        input_ids = torch.cat([padded_prompts, gen_masks], dim=1)

        gen_attn = torch.ones((B, gen_length), dtype=torch.bool, device=device)
        attn_mask = torch.cat([prompt_mask, gen_attn], dim=1)

        gen_start = max_prompt_len
        gen_end = max_prompt_len + gen_length

        sample_done = torch.zeros(B, dtype=torch.bool, device=device)
        nfe_per_sample = torch.zeros(B, dtype=torch.long, device=device)

        # ── Phase 1: Standard cosine draft ──
        for step in range(num_draft_steps):
            if sample_done.all():
                break

            frac_prev = cosine_schedule(step, num_draft_steps) if step > 0 else 0.0
            frac_curr = cosine_schedule(step + 1, num_draft_steps)

            gen_region = input_ids[:, gen_start:gen_end]
            is_masked = (gen_region == MASK_TOKEN_ID)
            num_masked = is_masked.sum(dim=1)

            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done

            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = input_ids[active_indices]
            active_attn = attn_mask[active_indices].long()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=active_input,
                    attention_mask=active_attn,
                )
                logits = outputs.logits

            nfe_per_sample[active_indices] += 1

            active_gen_logits = logits[:, gen_start:gen_end, :]
            active_gen_region = gen_region[active_indices]
            active_is_masked = (active_gen_region == MASK_TOKEN_ID)

            for ai, gi_t in enumerate(active_indices):
                gi = gi_t.item()
                sample_masked = active_is_masked[ai]
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = active_gen_logits[ai, masked_positions]

                probs = F.softmax(masked_logits, dim=-1)
                top1_conf, top1_token = probs.max(dim=-1)

                num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
                num_to_unmask = min(num_to_unmask, n_masked)

                _, topk_idx = top1_conf.topk(num_to_unmask)
                pos_to_unmask = masked_positions[topk_idx]
                tokens_to_place = top1_token[topk_idx]

                input_ids[gi, gen_start + pos_to_unmask] = tokens_to_place

        # ── Entropy scoring pass (all samples) ──
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask.long(),
            )
            logits = outputs.logits  # (B, total_len, vocab)

        nfe_per_sample += 1

        gen_logits = logits[:, gen_start:gen_end, :]         # (B, gen_length, vocab)
        probs_all = F.softmax(gen_logits, dim=-1)
        log_probs_all = F.log_softmax(gen_logits, dim=-1)
        entropy = -(probs_all * log_probs_all).sum(dim=-1)   # (B, gen_length)

        # Per-sample entropy stats (before revision)
        entropy_stats_list = []
        for i in range(B):
            entropy_stats_list.append({
                "mean_entropy": float(entropy[i].mean().item()),
                "max_entropy": float(entropy[i].max().item()),
            })

        # ── Phase 2: Entropy-based remasking + revision ──
        num_to_revise = max(1, int(round(revision_fraction * gen_length)))

        # Per-sample: pick top-entropy positions to remask
        _, revision_targets = entropy.topk(num_to_revise, dim=1)  # (B, num_to_revise)

        # Save original tokens before remasking
        original_tokens = torch.gather(
            input_ids[:, gen_start:gen_end], 1, revision_targets,
        ).clone()  # (B, num_to_revise)

        # Remask selected positions
        for i in range(B):
            input_ids[i, gen_start + revision_targets[i]] = MASK_TOKEN_ID

        # Reset done tracking for revision phase
        sample_done.fill_(False)

        for rev_step in range(revision_steps):
            if sample_done.all():
                break

            gen_region = input_ids[:, gen_start:gen_end]
            is_masked = (gen_region == MASK_TOKEN_ID)
            num_masked = is_masked.sum(dim=1)

            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done

            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = input_ids[active_indices]
            active_attn = attn_mask[active_indices].long()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=active_input,
                    attention_mask=active_attn,
                )
                logits = outputs.logits

            nfe_per_sample[active_indices] += 1

            active_gen_logits = logits[:, gen_start:gen_end, :]
            active_gen_region = gen_region[active_indices]
            active_is_masked = (active_gen_region == MASK_TOKEN_ID)

            for ai, gi_t in enumerate(active_indices):
                gi = gi_t.item()
                sample_masked = active_is_masked[ai]
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = active_gen_logits[ai, masked_positions]

                probs = F.softmax(masked_logits, dim=-1)
                top1_conf, top1_token = probs.max(dim=-1)

                target_unmasked = int(round((rev_step + 1) / revision_steps * num_to_revise))
                already_unmasked = num_to_revise - n_masked
                num_to_unmask = max(1, target_unmasked - already_unmasked)
                num_to_unmask = min(num_to_unmask, n_masked)

                if num_to_unmask > 0 and len(top1_conf) > 0:
                    k = min(num_to_unmask, len(top1_conf))
                    _, topk_idx = top1_conf.topk(k)
                    pos_to_unmask = masked_positions[topk_idx]
                    tokens_to_place = top1_token[topk_idx]
                    input_ids[gi, gen_start + pos_to_unmask] = tokens_to_place

        # ── Force-unmask any remaining masked tokens ──
        gen_region = input_ids[:, gen_start:gen_end]
        remaining_masked = (gen_region == MASK_TOKEN_ID)
        if remaining_masked.any():
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask.long(),
                )
                logits = outputs.logits

            nfe_per_sample += 1

            for i in range(B):
                sample_remaining = remaining_masked[i]
                if not sample_remaining.any():
                    continue
                masked_pos = torch.where(sample_remaining)[0]
                pos_logits = logits[i, gen_start + masked_pos]
                probs = F.softmax(pos_logits, dim=-1)
                _, tokens = probs.max(dim=-1)
                input_ids[i, gen_start + masked_pos] = tokens

        # ── Compute per-sample tokens_changed ──
        new_tokens = torch.gather(
            input_ids[:, gen_start:gen_end], 1, revision_targets,
        )  # (B, num_to_revise)
        tokens_changed = (new_tokens != original_tokens).sum(dim=1)  # (B,)

        # Build results
        results = []
        for i in range(B):
            rev_entropy = entropy[i, revision_targets[i]]
            es = entropy_stats_list[i]
            es.update({
                "revision_mean_entropy": float(rev_entropy.mean().item()),
                "num_revised": num_to_revise,
                "tokens_changed": int(tokens_changed[i].item()),
            })
            results.append({
                "input_ids": input_ids[i],
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": nfe_per_sample[i].item(),
                "tokens_changed": int(tokens_changed[i].item()),
                "entropy_stats": es,
                "prompt_pad_offset": max_prompt_len - prompt_lens[i],
            })

        return results

    # ------------------------------------------------------------------
    # Batched random-revise denoising
    # ------------------------------------------------------------------

    def batched_random_revise(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_draft_steps: int = 64,
        revision_fraction: float = 0.15,
        revision_steps: int = 3,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """Batch-parallel random-revise (draft + random remask + revise).

        Same interface as ``batched_entropy_revise`` but remasking positions
        are chosen randomly instead of by entropy.
        """
        if batch_size is None:
            batch_size = len(prompt_ids_list)

        all_results: list[dict] = [None] * len(prompt_ids_list)  # type: ignore[list-item]

        for chunk_start in range(0, len(prompt_ids_list), batch_size):
            chunk_end = min(chunk_start + batch_size, len(prompt_ids_list))
            chunk_prompts = prompt_ids_list[chunk_start:chunk_end]
            chunk_results = self._random_revise_batch(
                chunk_prompts, gen_length, num_draft_steps,
                revision_fraction, revision_steps, seed, chunk_start,
            )
            for i, r in enumerate(chunk_results):
                all_results[chunk_start + i] = r

        return all_results

    def _random_revise_batch(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_draft_steps: int,
        revision_fraction: float,
        revision_steps: int,
        seed: int,
        global_offset: int,
    ) -> list[dict]:
        """Random-revise on a single batch."""
        B = len(prompt_ids_list)
        device = self.device

        padded_prompts, prompt_mask, prompt_lens = _left_pad_batch(
            prompt_ids_list, self.pad_token_id, device,
        )
        max_prompt_len = padded_prompts.shape[1]

        gen_masks = torch.full(
            (B, gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device,
        )
        input_ids = torch.cat([padded_prompts, gen_masks], dim=1)

        gen_attn = torch.ones((B, gen_length), dtype=torch.bool, device=device)
        attn_mask = torch.cat([prompt_mask, gen_attn], dim=1)

        gen_start = max_prompt_len
        gen_end = max_prompt_len + gen_length

        sample_done = torch.zeros(B, dtype=torch.bool, device=device)
        nfe_per_sample = torch.zeros(B, dtype=torch.long, device=device)

        # ── Phase 1: Standard cosine draft ──
        for step in range(num_draft_steps):
            if sample_done.all():
                break

            frac_prev = cosine_schedule(step, num_draft_steps) if step > 0 else 0.0
            frac_curr = cosine_schedule(step + 1, num_draft_steps)

            gen_region = input_ids[:, gen_start:gen_end]
            is_masked = (gen_region == MASK_TOKEN_ID)
            num_masked = is_masked.sum(dim=1)

            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done

            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = input_ids[active_indices]
            active_attn = attn_mask[active_indices].long()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=active_input,
                    attention_mask=active_attn,
                )
                logits = outputs.logits

            nfe_per_sample[active_indices] += 1

            active_gen_logits = logits[:, gen_start:gen_end, :]
            active_gen_region = gen_region[active_indices]
            active_is_masked = (active_gen_region == MASK_TOKEN_ID)

            for ai, gi_t in enumerate(active_indices):
                gi = gi_t.item()
                sample_masked = active_is_masked[ai]
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = active_gen_logits[ai, masked_positions]

                probs = F.softmax(masked_logits, dim=-1)
                top1_conf, top1_token = probs.max(dim=-1)

                num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
                num_to_unmask = min(num_to_unmask, n_masked)

                _, topk_idx = top1_conf.topk(num_to_unmask)
                pos_to_unmask = masked_positions[topk_idx]
                tokens_to_place = top1_token[topk_idx]

                input_ids[gi, gen_start + pos_to_unmask] = tokens_to_place

        # ── Phase 2: Random remasking + revision ──
        num_to_revise = max(1, int(round(revision_fraction * gen_length)))

        # Random positions per sample
        revision_targets = torch.stack([
            torch.randperm(gen_length, device=device)[:num_to_revise]
            for _ in range(B)
        ])  # (B, num_to_revise)

        original_tokens = torch.gather(
            input_ids[:, gen_start:gen_end], 1, revision_targets,
        ).clone()

        for i in range(B):
            input_ids[i, gen_start + revision_targets[i]] = MASK_TOKEN_ID

        sample_done.fill_(False)

        for rev_step in range(revision_steps):
            if sample_done.all():
                break

            gen_region = input_ids[:, gen_start:gen_end]
            is_masked = (gen_region == MASK_TOKEN_ID)
            num_masked = is_masked.sum(dim=1)

            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done

            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = input_ids[active_indices]
            active_attn = attn_mask[active_indices].long()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=active_input,
                    attention_mask=active_attn,
                )
                logits = outputs.logits

            nfe_per_sample[active_indices] += 1

            active_gen_logits = logits[:, gen_start:gen_end, :]
            active_gen_region = gen_region[active_indices]
            active_is_masked = (active_gen_region == MASK_TOKEN_ID)

            for ai, gi_t in enumerate(active_indices):
                gi = gi_t.item()
                sample_masked = active_is_masked[ai]
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = active_gen_logits[ai, masked_positions]

                probs = F.softmax(masked_logits, dim=-1)
                top1_conf, top1_token = probs.max(dim=-1)

                target_unmasked = int(round((rev_step + 1) / revision_steps * num_to_revise))
                already_unmasked = num_to_revise - n_masked
                num_to_unmask = max(1, target_unmasked - already_unmasked)
                num_to_unmask = min(num_to_unmask, n_masked)

                if num_to_unmask > 0 and len(top1_conf) > 0:
                    k = min(num_to_unmask, len(top1_conf))
                    _, topk_idx = top1_conf.topk(k)
                    pos_to_unmask = masked_positions[topk_idx]
                    tokens_to_place = top1_token[topk_idx]
                    input_ids[gi, gen_start + pos_to_unmask] = tokens_to_place

        # Force-unmask remaining
        gen_region = input_ids[:, gen_start:gen_end]
        remaining_masked = (gen_region == MASK_TOKEN_ID)
        if remaining_masked.any():
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask.long(),
                )
                logits = outputs.logits

            nfe_per_sample += 1

            for i in range(B):
                sample_remaining = remaining_masked[i]
                if not sample_remaining.any():
                    continue
                masked_pos = torch.where(sample_remaining)[0]
                pos_logits = logits[i, gen_start + masked_pos]
                probs = F.softmax(pos_logits, dim=-1)
                _, tokens = probs.max(dim=-1)
                input_ids[i, gen_start + masked_pos] = tokens

        new_tokens = torch.gather(
            input_ids[:, gen_start:gen_end], 1, revision_targets,
        )
        tokens_changed = (new_tokens != original_tokens).sum(dim=1)

        results = []
        for i in range(B):
            results.append({
                "input_ids": input_ids[i],
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": nfe_per_sample[i].item(),
                "tokens_changed": int(tokens_changed[i].item()),
                "entropy_stats": {},
                "prompt_pad_offset": max_prompt_len - prompt_lens[i],
            })

        return results

    # ------------------------------------------------------------------
    # Batched MGCD (Memory-Gated Consensus Diffusion)
    # ------------------------------------------------------------------

    def batched_mgcd(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_draft_steps: int = 12,
        main_draft_steps: int | None = None,
        scout_steps: int | None = None,
        revision_steps: int = 58,
        revision_fraction: float = 0.08,
        bridge_radius: int = 1,
        disagreement_weight: float = 0.7,
        entropy_weight: float = 0.3,
        support_threshold: float = 0.6,
        entropy_margin: float = 0.03,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """Batch-parallel MGCD pilot path."""
        if batch_size is None:
            batch_size = len(prompt_ids_list)

        anchor_steps = main_draft_steps or num_draft_steps
        scout_steps = scout_steps or num_draft_steps

        all_results: list[dict] = [None] * len(prompt_ids_list)  # type: ignore[list-item]

        for chunk_start in range(0, len(prompt_ids_list), batch_size):
            chunk_end = min(chunk_start + batch_size, len(prompt_ids_list))
            chunk_prompts = prompt_ids_list[chunk_start:chunk_end]
            chunk_results = self._mgcd_batch(
                chunk_prompts,
                gen_length,
                anchor_steps,
                scout_steps,
                revision_steps,
                revision_fraction,
                bridge_radius,
                disagreement_weight,
                entropy_weight,
                support_threshold,
                entropy_margin,
                seed,
                chunk_start,
            )
            for i, r in enumerate(chunk_results):
                all_results[chunk_start + i] = r

        return all_results

    def _mgcd_batch(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        anchor_steps: int,
        scout_steps: int,
        revision_steps: int,
        revision_fraction: float,
        bridge_radius: int,
        disagreement_weight: float,
        entropy_weight: float,
        support_threshold: float,
        entropy_margin: float,
        seed: int,
        global_offset: int,
    ) -> list[dict]:
        """Approximate MGCD with dual draft consensus and island-gated revision."""
        del seed, global_offset  # deterministic pilot path
        B = len(prompt_ids_list)
        device = self.device

        draft_a = self._draft_batch(
            prompt_ids_list=prompt_ids_list,
            gen_length=gen_length,
            num_steps=anchor_steps,
            selection_mode="confidence",
        )
        draft_b = self._draft_batch(
            prompt_ids_list=prompt_ids_list,
            gen_length=gen_length,
            num_steps=scout_steps,
            selection_mode="margin",
        )

        gen_start = draft_a[0]["gen_start"]
        gen_end = draft_a[0]["gen_end"]

        draft_a_input = torch.stack([r["input_ids"] for r in draft_a], dim=0)
        draft_a_attn = torch.stack([r["attn_mask"] for r in draft_a], dim=0).long()
        draft_b_input = torch.stack([r["input_ids"] for r in draft_b], dim=0)
        draft_b_attn = torch.stack([r["attn_mask"] for r in draft_b], dim=0).long()

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits_a = self.model(input_ids=draft_a_input, attention_mask=draft_a_attn).logits
            logits_b = self.model(input_ids=draft_b_input, attention_mask=draft_b_attn).logits

        draft_a_entropy = -(
            F.softmax(logits_a[:, gen_start:gen_end, :], dim=-1)
            * F.log_softmax(logits_a[:, gen_start:gen_end, :], dim=-1)
        ).sum(dim=-1)
        draft_b_entropy = -(
            F.softmax(logits_b[:, gen_start:gen_end, :], dim=-1)
            * F.log_softmax(logits_b[:, gen_start:gen_end, :], dim=-1)
        ).sum(dim=-1)

        revised_input = draft_a_input.clone()
        revised_attn = draft_a_attn.clone()
        revision_masks: list[torch.Tensor] = []
        commit_masks: list[torch.Tensor] = []
        contested_masks: list[torch.Tensor] = []
        bridge_masks: list[torch.Tensor] = []
        mgcd_stats: list[dict[str, Any]] = []
        nfe_per_sample = torch.tensor(
            [r["nfe"] + draft_b[i]["nfe"] + 2 for i, r in enumerate(draft_a)],
            dtype=torch.long,
            device=device,
        )

        for i in range(B):
            tokens_a = draft_a_input[i, gen_start:gen_end]
            tokens_b = draft_b_input[i, gen_start:gen_end]
            disagreement = tokens_a != tokens_b
            entropy_mix = 0.5 * (draft_a_entropy[i] + draft_b_entropy[i])
            entropy_min = entropy_mix.min()
            entropy_max = entropy_mix.max()
            entropy_range = float((entropy_max - entropy_min).item())
            if entropy_range > 1e-6:
                entropy_norm = (entropy_mix - entropy_min) / (entropy_max - entropy_min)
            else:
                entropy_norm = torch.zeros_like(entropy_mix)

            contested = torch.zeros(gen_length, dtype=torch.bool, device=device)
            num_to_revise = max(1, int(round(revision_fraction * gen_length)))
            disagreement_budget = min(
                int(disagreement.sum().item()),
                int(round(num_to_revise * disagreement_weight)),
            )
            if disagreement_budget > 0:
                disagreement_scores = entropy_norm.masked_fill(~disagreement, -1.0)
                _, top_disagreement_idx = disagreement_scores.topk(disagreement_budget)
                contested[top_disagreement_idx] = True

            remaining_budget = num_to_revise - int(contested.sum().item())
            if remaining_budget > 0:
                entropy_scores = entropy_norm * entropy_weight
                entropy_scores = entropy_scores.masked_fill(contested, -1.0)
                _, top_entropy_idx = entropy_scores.topk(remaining_budget)
                contested[top_entropy_idx] = True
            bridge = _expand_positions(contested, bridge_radius) & (~contested)
            revision_mask = contested | bridge

            revision_masks.append(revision_mask)
            commit_masks.append(contested)
            contested_masks.append(contested)
            bridge_masks.append(bridge)

            revised_input[i, gen_start:gen_end][revision_mask] = MASK_TOKEN_ID
            islands = _contiguous_islands(revision_mask.detach().cpu())
            mean_span = sum((end - start + 1) for start, end in islands) / max(1, len(islands))
            mgcd_stats.append({
                "locked_token_ratio": float((~revision_mask).float().mean().item()),
                "contested_token_ratio": float(contested.float().mean().item()),
                "bridge_token_ratio": float(bridge.float().mean().item()),
                "contested_island_count": len(islands),
                "mean_contested_span_length": float(mean_span),
                "accepted_updates": 0,
                "rejected_updates": 0,
                "candidate_support_ratio": 0.0,
            })

        sample_done = torch.zeros(B, dtype=torch.bool, device=device)
        total_revision_targets = torch.tensor(
            [int(mask.sum().item()) for mask in revision_masks],
            dtype=torch.long,
            device=device,
        )

        for rev_step in range(revision_steps):
            if sample_done.all():
                break

            gen_region = revised_input[:, gen_start:gen_end]
            is_masked = gen_region == MASK_TOKEN_ID
            num_masked = is_masked.sum(dim=1)
            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done
            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = revised_input[active_indices]
            active_attn = revised_attn[active_indices]

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids=active_input, attention_mask=active_attn)
                logits = outputs.logits

            nfe_per_sample[active_indices] += 1
            active_gen_logits = logits[:, gen_start:gen_end, :]
            active_gen_region = gen_region[active_indices]
            active_is_masked = active_gen_region == MASK_TOKEN_ID

            for ai, gi_t in enumerate(active_indices):
                gi = gi_t.item()
                sample_masked = active_is_masked[ai]
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = active_gen_logits[ai, masked_positions]
                probs = F.softmax(masked_logits, dim=-1)
                top1_conf, top1_token = probs.max(dim=-1)

                target_unmasked = int(round((rev_step + 1) / max(1, revision_steps) * total_revision_targets[gi].item()))
                already_unmasked = total_revision_targets[gi].item() - n_masked
                num_to_unmask = max(1, target_unmasked - already_unmasked)
                num_to_unmask = min(num_to_unmask, n_masked)
                _, topk_idx = top1_conf.topk(num_to_unmask)
                pos_to_unmask = masked_positions[topk_idx]
                tokens_to_place = top1_token[topk_idx]
                revised_input[gi, gen_start + pos_to_unmask] = tokens_to_place

        remaining_masked = revised_input[:, gen_start:gen_end] == MASK_TOKEN_ID
        if remaining_masked.any():
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids=revised_input, attention_mask=revised_attn)
                logits = outputs.logits

            nfe_per_sample += 1
            for i in range(B):
                sample_remaining = remaining_masked[i]
                if not sample_remaining.any():
                    continue
                masked_pos = torch.where(sample_remaining)[0]
                pos_logits = logits[i, gen_start + masked_pos]
                probs = F.softmax(pos_logits, dim=-1)
                _, tokens = probs.max(dim=-1)
                revised_input[i, gen_start + masked_pos] = tokens

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            revised_logits = self.model(input_ids=revised_input, attention_mask=revised_attn).logits

        nfe_per_sample += 1
        revised_entropy = -(
            F.softmax(revised_logits[:, gen_start:gen_end, :], dim=-1)
            * F.log_softmax(revised_logits[:, gen_start:gen_end, :], dim=-1)
        ).sum(dim=-1)

        results = []
        for i in range(B):
            baseline_tokens = draft_a_input[i, gen_start:gen_end].clone()
            candidate_tokens = revised_input[i, gen_start:gen_end]
            final_tokens = baseline_tokens.clone()
            revision_mask = revision_masks[i]
            commit_mask = commit_masks[i]
            islands = _contiguous_islands(revision_mask.detach().cpu())
            accepted = 0
            rejected = 0
            support_ratios: list[float] = []
            draft_b_tokens = draft_b_input[i, gen_start:gen_end]

            for start, end in islands:
                island_commit_mask = commit_mask[start:end + 1]
                if not island_commit_mask.any():
                    continue

                baseline_entropy_slice = draft_a_entropy[i, start:end + 1][island_commit_mask]
                candidate_entropy_slice = revised_entropy[i, start:end + 1][island_commit_mask]
                baseline_score = float(baseline_entropy_slice.mean().item())
                candidate_score = float(candidate_entropy_slice.mean().item())

                baseline_slice = baseline_tokens[start:end + 1][island_commit_mask]
                draft_b_slice = draft_b_tokens[start:end + 1][island_commit_mask]
                candidate_slice = candidate_tokens[start:end + 1][island_commit_mask]
                support_ratio = float(
                    ((candidate_slice == baseline_slice) | (candidate_slice == draft_b_slice))
                    .float()
                    .mean()
                    .item()
                )
                support_ratios.append(support_ratio)

                if (
                    candidate_score + entropy_margin <= baseline_score
                    and support_ratio >= support_threshold
                ):
                    island_positions = torch.arange(start, end + 1, device=device)[island_commit_mask]
                    final_tokens[island_positions] = candidate_tokens[island_positions]
                    accepted += 1
                else:
                    rejected += 1

            final_input = draft_a_input[i].clone()
            final_input[gen_start:gen_end] = final_tokens
            tokens_changed = int((final_tokens[commit_mask] != baseline_tokens[commit_mask]).sum().item())
            stats = mgcd_stats[i]
            stats["accepted_updates"] = accepted
            stats["rejected_updates"] = rejected
            stats["revision_target_count"] = int(revision_mask.sum().item())
            stats["commit_target_count"] = int(commit_mask.sum().item())
            stats["candidate_support_ratio"] = float(sum(support_ratios) / max(1, len(support_ratios)))

            entropy_stats = {
                "draft_a_mean_entropy": float(draft_a_entropy[i].mean().item()),
                "draft_b_mean_entropy": float(draft_b_entropy[i].mean().item()),
                "revised_mean_entropy": float(revised_entropy[i].mean().item()),
                "tokens_changed": tokens_changed,
            }
            results.append({
                "input_ids": final_input,
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": int(nfe_per_sample[i].item()),
                "tokens_changed": tokens_changed,
                "entropy_stats": entropy_stats,
                "mgcd_stats": stats,
                "prompt_pad_offset": draft_a[i]["prompt_pad_offset"],
            })

        return results

    # ------------------------------------------------------------------
    # Batched DCD-lite (deferred commitment with confidence-aware windows)
    # ------------------------------------------------------------------

    def batched_dcd(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_steps: int = 64,
        base_confidence_threshold: float = 0.90,
        final_confidence_threshold: float = 0.55,
        window_radius: int = 2,
        late_commit_start: float = 0.75,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """Approximate DCD with confidence-aware deferred commitment windows.

        This is a low-cost pilot implementation rather than a paper-faithful
        reproduction. It defers locally uncertain regions by keeping tokens
        around low-confidence positions masked until later denoising steps.
        """
        if batch_size is None:
            batch_size = len(prompt_ids_list)

        all_results: list[dict] = [None] * len(prompt_ids_list)  # type: ignore[list-item]

        for chunk_start in range(0, len(prompt_ids_list), batch_size):
            chunk_end = min(chunk_start + batch_size, len(prompt_ids_list))
            chunk_prompts = prompt_ids_list[chunk_start:chunk_end]
            chunk_results = self._dcd_batch(
                chunk_prompts,
                gen_length,
                num_steps,
                base_confidence_threshold,
                final_confidence_threshold,
                window_radius,
                late_commit_start,
                seed,
                chunk_start,
            )
            for i, r in enumerate(chunk_results):
                all_results[chunk_start + i] = r

        return all_results

    def _dcd_batch(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_steps: int,
        base_confidence_threshold: float,
        final_confidence_threshold: float,
        window_radius: int,
        late_commit_start: float,
        seed: int,
        global_offset: int,
    ) -> list[dict]:
        """Run a DCD-style deferred-commitment batch."""
        del seed, global_offset  # Pilot path is deterministic once logits are fixed.
        B = len(prompt_ids_list)
        device = self.device

        padded_prompts, prompt_mask, prompt_lens = _left_pad_batch(
            prompt_ids_list, self.pad_token_id, device,
        )
        max_prompt_len = padded_prompts.shape[1]

        gen_masks = torch.full(
            (B, gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device,
        )
        input_ids = torch.cat([padded_prompts, gen_masks], dim=1)

        gen_attn = torch.ones((B, gen_length), dtype=torch.bool, device=device)
        attn_mask = torch.cat([prompt_mask, gen_attn], dim=1)

        gen_start = max_prompt_len
        gen_end = max_prompt_len + gen_length

        sample_done = torch.zeros(B, dtype=torch.bool, device=device)
        nfe_per_sample = torch.zeros(B, dtype=torch.long, device=device)
        deferred_counts = torch.zeros(B, dtype=torch.long, device=device)
        confident_commit_counts = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(num_steps):
            if sample_done.all():
                break

            frac_prev = cosine_schedule(step, num_steps) if step > 0 else 0.0
            frac_curr = cosine_schedule(step + 1, num_steps)
            progress = (step + 1) / max(1, num_steps)
            threshold = (
                base_confidence_threshold
                + (final_confidence_threshold - base_confidence_threshold) * progress
            )
            late_phase = progress >= late_commit_start

            gen_region = input_ids[:, gen_start:gen_end]
            is_masked = (gen_region == MASK_TOKEN_ID)
            num_masked = is_masked.sum(dim=1)

            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done

            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = input_ids[active_indices]
            active_attn = attn_mask[active_indices].long()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=active_input,
                    attention_mask=active_attn,
                )
                logits = outputs.logits

            nfe_per_sample[active_indices] += 1

            active_gen_logits = logits[:, gen_start:gen_end, :]
            active_gen_region = gen_region[active_indices]
            active_is_masked = (active_gen_region == MASK_TOKEN_ID)

            for ai, gi_t in enumerate(active_indices):
                gi = gi_t.item()
                sample_masked = active_is_masked[ai]
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = active_gen_logits[ai, masked_positions]
                probs = F.softmax(masked_logits, dim=-1)
                top1_conf, top1_token = probs.max(dim=-1)

                num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
                num_to_unmask = min(num_to_unmask, n_masked)

                confident_mask = top1_conf >= threshold
                blocked_mask = torch.zeros(n_masked, dtype=torch.bool, device=device)

                if not late_phase:
                    uncertain_positions = masked_positions[~confident_mask]
                    for pos in uncertain_positions:
                        center = int(pos.item())
                        local_block = (
                            (masked_positions >= max(0, center - window_radius))
                            & (masked_positions <= min(gen_length - 1, center + window_radius))
                        )
                        blocked_mask |= local_block
                    deferred_counts[gi] += int((~confident_mask).sum().item())

                candidate_mask = confident_mask & (~blocked_mask)
                candidate_positions = masked_positions[candidate_mask]
                candidate_conf = top1_conf[candidate_mask]
                candidate_tokens = top1_token[candidate_mask]

                if candidate_positions.numel() == 0:
                    # If everything is deferred, allow the single most confident position
                    # to commit so the schedule keeps making progress.
                    top_idx = int(torch.argmax(top1_conf).item())
                    candidate_positions = masked_positions[top_idx:top_idx + 1]
                    candidate_conf = top1_conf[top_idx:top_idx + 1]
                    candidate_tokens = top1_token[top_idx:top_idx + 1]

                k = min(num_to_unmask, candidate_positions.numel())
                _, topk_idx = candidate_conf.topk(k)
                pos_to_unmask = candidate_positions[topk_idx]
                tokens_to_place = candidate_tokens[topk_idx]
                confident_commit_counts[gi] += k
                input_ids[gi, gen_start + pos_to_unmask] = tokens_to_place

        gen_region = input_ids[:, gen_start:gen_end]
        remaining_masked = (gen_region == MASK_TOKEN_ID)
        if remaining_masked.any():
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask.long(),
                )
                logits = outputs.logits

            nfe_per_sample += 1

            for i in range(B):
                sample_remaining = remaining_masked[i]
                if not sample_remaining.any():
                    continue
                masked_pos = torch.where(sample_remaining)[0]
                pos_logits = logits[i, gen_start + masked_pos]
                probs = F.softmax(pos_logits, dim=-1)
                _, tokens = probs.max(dim=-1)
                input_ids[i, gen_start + masked_pos] = tokens

        results = []
        for i in range(B):
            results.append({
                "input_ids": input_ids[i],
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": nfe_per_sample[i].item(),
                "prompt_pad_offset": max_prompt_len - prompt_lens[i],
                "deferred_count": int(deferred_counts[i].item()),
                "committed_count": int(confident_commit_counts[i].item()),
                "window_radius": window_radius,
                "base_confidence_threshold": base_confidence_threshold,
                "final_confidence_threshold": final_confidence_threshold,
                "late_commit_start": late_commit_start,
            })

        return results

    # ------------------------------------------------------------------
    # Batched prophet (confidence-based early stopping)
    # ------------------------------------------------------------------

    def batched_prophet(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_steps: int = 64,
        confidence_threshold: float = 0.95,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """Batch-parallel prophet with per-sample early stopping."""
        if batch_size is None:
            batch_size = len(prompt_ids_list)

        all_results: list[dict] = [None] * len(prompt_ids_list)  # type: ignore[list-item]

        for chunk_start in range(0, len(prompt_ids_list), batch_size):
            chunk_end = min(chunk_start + batch_size, len(prompt_ids_list))
            chunk_prompts = prompt_ids_list[chunk_start:chunk_end]
            chunk_results = self._prophet_batch(
                chunk_prompts, gen_length, num_steps,
                confidence_threshold, seed, chunk_start,
            )
            for i, r in enumerate(chunk_results):
                all_results[chunk_start + i] = r

        return all_results

    def _prophet_batch(
        self,
        prompt_ids_list: list[list[int]],
        gen_length: int,
        num_steps: int,
        confidence_threshold: float,
        seed: int,
        global_offset: int,
    ) -> list[dict]:
        """Prophet on a single batch."""
        B = len(prompt_ids_list)
        device = self.device

        padded_prompts, prompt_mask, prompt_lens = _left_pad_batch(
            prompt_ids_list, self.pad_token_id, device,
        )
        max_prompt_len = padded_prompts.shape[1]

        gen_masks = torch.full(
            (B, gen_length), MASK_TOKEN_ID, dtype=torch.long, device=device,
        )
        input_ids = torch.cat([padded_prompts, gen_masks], dim=1)

        gen_attn = torch.ones((B, gen_length), dtype=torch.bool, device=device)
        attn_mask = torch.cat([prompt_mask, gen_attn], dim=1)

        gen_start = max_prompt_len
        gen_end = max_prompt_len + gen_length

        sample_done = torch.zeros(B, dtype=torch.bool, device=device)
        nfe_per_sample = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(num_steps):
            if sample_done.all():
                break

            frac_prev = cosine_schedule(step, num_steps) if step > 0 else 0.0
            frac_curr = cosine_schedule(step + 1, num_steps)

            gen_region = input_ids[:, gen_start:gen_end]
            is_masked = (gen_region == MASK_TOKEN_ID)
            num_masked = is_masked.sum(dim=1)

            newly_done = (num_masked == 0) & (~sample_done)
            sample_done |= newly_done

            active = ~sample_done
            if not active.any():
                break

            active_indices = torch.where(active)[0]
            active_input = input_ids[active_indices]
            active_attn = attn_mask[active_indices].long()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=active_input,
                    attention_mask=active_attn,
                )
                logits = outputs.logits

            nfe_per_sample[active_indices] += 1

            active_gen_logits = logits[:, gen_start:gen_end, :]
            active_gen_region = gen_region[active_indices]
            active_is_masked = (active_gen_region == MASK_TOKEN_ID)

            for ai, gi_t in enumerate(active_indices):
                gi = gi_t.item()
                sample_masked = active_is_masked[ai]
                n_masked = sample_masked.sum().item()
                if n_masked == 0:
                    sample_done[gi] = True
                    continue

                masked_positions = torch.where(sample_masked)[0]
                masked_logits = active_gen_logits[ai, masked_positions]

                probs = F.softmax(masked_logits, dim=-1)
                top1_conf, top1_token = probs.max(dim=-1)

                # Prophet early stopping: if past 1/3 of steps and all
                # remaining tokens are confident, fill everything at once
                if (step > num_steps // 3
                        and top1_conf.min().item() >= confidence_threshold):
                    input_ids[gi, gen_start + masked_positions] = top1_token
                    sample_done[gi] = True
                    continue

                num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
                num_to_unmask = min(num_to_unmask, n_masked)

                _, topk_idx = top1_conf.topk(num_to_unmask)
                pos_to_unmask = masked_positions[topk_idx]
                tokens_to_place = top1_token[topk_idx]

                input_ids[gi, gen_start + pos_to_unmask] = tokens_to_place

        results = []
        for i in range(B):
            results.append({
                "input_ids": input_ids[i],
                "gen_start": gen_start,
                "gen_end": gen_end,
                "nfe": nfe_per_sample[i].item(),
                "prompt_pad_offset": max_prompt_len - prompt_lens[i],
            })

        return results

    # ------------------------------------------------------------------
    # Convenience: decode batch results
    # ------------------------------------------------------------------

    def decode_results(
        self,
        results: list[dict],
        skip_mask: bool = True,
    ) -> list[str]:
        """Decode generation regions from batch results to text.

        Parameters
        ----------
        results : list[dict]
            Output from any batched method.
        skip_mask : bool
            If True, filter out any remaining MASK tokens before decoding.

        Returns
        -------
        list[str]
            Decoded text for each sample.
        """
        texts = []
        for r in results:
            gen_tokens = r["input_ids"][r["gen_start"]:r["gen_end"]].cpu().tolist()
            if skip_mask:
                gen_tokens = [t for t in gen_tokens if t != MASK_TOKEN_ID]
            texts.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
        return texts


# ---------------------------------------------------------------------------
# Standalone usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== BatchedDLMInference Demo ===\n")

    engine = BatchedDLMInference(
        model_path="/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct",
        device="cuda",
    )

    # Probe max batch size
    bs = engine.find_max_batch_size(gen_length=256, prompt_max_len=512)
    print(f"\nOptimal batch size: {bs}\n")

    # Example prompts
    prompts = [
        "Solve the following math problem step by step.\n\nQuestion: What is 2+3?\n\nAnswer:",
        "Solve the following math problem step by step.\n\nQuestion: What is 7*8?\n\nAnswer:",
    ]
    prompt_ids_list = [
        engine.tokenizer.encode(p, add_special_tokens=False) for p in prompts
    ]

    # Standard denoising
    print("Running batched standard denoising (64 steps) ...")
    results = engine.batched_standard_denoising(
        prompt_ids_list, gen_length=256, num_steps=64, batch_size=bs,
    )
    texts = engine.decode_results(results)
    for i, t in enumerate(texts):
        print(f"  Sample {i}: NFE={results[i]['nfe']}")
        print(f"    {t[:200]}\n")

    # Entropy revise
    print("Running batched entropy-revise (64 draft + 3 rev) ...")
    results = engine.batched_entropy_revise(
        prompt_ids_list, gen_length=256, num_draft_steps=64,
        revision_fraction=0.15, revision_steps=3, batch_size=bs,
    )
    texts = engine.decode_results(results)
    for i, t in enumerate(texts):
        es = results[i]["entropy_stats"]
        print(f"  Sample {i}: NFE={results[i]['nfe']}, "
              f"tokens_changed={es['tokens_changed']}")
        print(f"    {t[:200]}\n")

    print("Done.")
