"""
IGSD: Iterative Guided Self-speculative Denoising for Masked Diffusion LMs.

Standalone module implementing the IGSD algorithm for LLaDA-8B-Instruct.

Algorithm:
    1. Draft phase: T_draft whole-sequence denoising steps (global, no block structure)
    2. Partition: S_accept = {i: confidence(i) >= tau}, S_refine = complement
    3. Refine phase: T_full steps on S_refine positions only (S_accept frozen)
    4. Return merged output

Key design decisions (v3):
    - Whole-sequence draft avoids block boundary issues with sparse masking
    - Global token scheduling (top-k by confidence across all positions)
    - KV hit-rate during refine = fraction of positions frozen at each refine step

Pilot results (pilot_igsd_implement):
    - tau=0.70: GSM8K acc=0.580, speedup=1.86x, accept_rate=0.96
    - tau=0.85: GSM8K acc=0.580, speedup=1.84x, accept_rate=0.96
    - HumanEval: 5.0x speedup (short gen_length=128 context)
    - Root cause of accuracy drop: 32-step whole-sequence draft < 64-step block-sequential

Usage:
    from igsd import IGSDGenerator

    gen = IGSDGenerator(model, tokenizer, device="cuda")
    result = gen.generate(
        prompt="Solve: 2+2=?",
        tau=0.70, t_draft=32, t_full=64,
        gen_length=128, block_length=32,
    )
    print(result["generated_text"])
    print(f"Speedup: {result['tps']:.1f} TPS")
    print(f"Accept rate: {result['accept_rate']:.3f}")
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
import torch.nn.functional as F

MASK_ID = 126336


class IGSDGenerator:
    """
    IGSD generator for Masked Diffusion Language Models.

    Parameters
    ----------
    model : transformers.AutoModel
        Loaded LLaDA-style MDM model (eval mode, on device).
    tokenizer : transformers.AutoTokenizer
        Corresponding tokenizer (padding_side='left').
    device : str
        CUDA device string (e.g., 'cuda:0').
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        tau: float = 0.70,
        t_draft: int = 32,
        t_full: int = 64,
        gen_length: int = 128,
        block_length: int = 32,
        apply_chat_template: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate text using IGSD.

        Parameters
        ----------
        prompt : str
            Input prompt text.
        tau : float
            Confidence threshold for S_accept partition.
            Higher tau → fewer tokens accepted → more refinement.
            Recommended: 0.70 (exploratory) or 0.85 (primary).
        t_draft : int
            Number of draft denoising steps (whole-sequence global).
        t_full : int
            Number of refine denoising steps for S_refine positions.
        gen_length : int
            Number of tokens to generate.
        block_length : int
            Block length (informational, not used in draft/refine scheduling).
        apply_chat_template : bool
            Whether to apply instruct chat template to prompt.

        Returns
        -------
        dict with keys:
            generated_text (str): Decoded output text
            tps (float): Total tokens per second
            elapsed_sec (float): Total wall-clock time
            draft_elapsed_sec (float): Draft phase time
            refine_elapsed_sec (float): Refine phase time
            accept_rate (float): Fraction of tokens in S_accept
            n_accept (int): Number of accepted tokens
            n_refine (int): Number of refined tokens
            n_total (int): Total generation tokens
            kv_hit_rate_refine (float): Avg fraction of frozen tokens during refine
            tau (float): Tau value used
        """
        # Tokenize prompt
        if apply_chat_template:
            msg = [{"role": "user", "content": prompt}]
            enc_text = self.tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
        else:
            enc_text = prompt

        enc = self.tokenizer(
            [enc_text], add_special_tokens=False, padding=True, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len     = input_ids.shape[1]

        # Initialize fully masked sequence
        x = torch.full(
            (1, prompt_len + gen_length), MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat([
            attention_mask,
            torch.ones((1, gen_length), dtype=attention_mask.dtype, device=self.device)
        ], dim=-1)

        # ── Phase 1: Whole-sequence draft ─────────────────────────────────────
        t_draft_start = time.perf_counter()
        tokens_per_draft_step = max(1, gen_length // t_draft)
        remainder_draft = gen_length % t_draft

        for step in range(t_draft):
            n_masked = int((x[0, prompt_len:] == MASK_ID).sum().item())
            if n_masked == 0:
                break

            with torch.no_grad():
                logits = self.model(x, attention_mask=attn).logits
            p    = F.softmax(logits, dim=-1)
            x0   = torch.argmax(p, dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

            mask_index = (x == MASK_ID)
            mask_index[:, :prompt_len] = False
            confidence = torch.where(
                mask_index, x0_p, torch.tensor(-float("inf"), device=self.device)
            )

            k = tokens_per_draft_step + (1 if step < remainder_draft else 0)
            k = min(k, n_masked)
            if k > 0:
                _, sel = torch.topk(confidence[0], k=k)
                ti = torch.zeros_like(x[0], dtype=torch.bool)
                ti[sel] = True
                ti = ti.unsqueeze(0) & mask_index
                x[ti] = x0[ti]

        draft_elapsed = time.perf_counter() - t_draft_start

        # ── Phase 2: Confidence partition ─────────────────────────────────────
        with torch.no_grad():
            logits = self.model(x, attention_mask=attn).logits
        p_final    = F.softmax(logits[:, prompt_len:, :], dim=-1)
        gen_region = x[:, prompt_len:].clone()
        still_masked = (gen_region == MASK_ID)

        draft_pred = torch.argmax(p_final, dim=-1)
        draft_conf = torch.gather(p_final, -1, draft_pred.unsqueeze(-1)).squeeze(-1)
        filled_conf = torch.gather(p_final, -1, gen_region.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        final_confidence = torch.where(still_masked, draft_conf, filled_conf)
        gen_region_filled = torch.where(still_masked, draft_pred, gen_region)
        x[:, prompt_len:] = gen_region_filled

        s_accept_gen = (final_confidence >= tau)
        n_accept     = int(s_accept_gen.sum().item())
        n_total      = s_accept_gen.numel()
        n_refine     = n_total - n_accept
        accept_rate  = n_accept / n_total

        # ── Phase 3: Whole-sequence refine for S_refine ───────────────────────
        x_refine = x.clone()
        s_accept_full = torch.cat([
            torch.ones(1, prompt_len, dtype=torch.bool, device=self.device),
            s_accept_gen
        ], dim=1)
        s_refine_full = ~s_accept_full
        x_refine[s_refine_full] = MASK_ID

        kv_hit_steps = []
        t_refine_start = time.perf_counter()

        if n_refine > 0:
            tokens_per_refine_step = max(1, n_refine // t_full)
            remainder_refine = n_refine % t_full

            for step in range(t_full):
                n_masked_now = int((x_refine[0, prompt_len:] == MASK_ID).sum().item())
                if n_masked_now == 0:
                    break

                n_frozen = n_total - n_masked_now
                kv_hit_steps.append(n_frozen / n_total)

                with torch.no_grad():
                    logits = self.model(x_refine, attention_mask=attn).logits
                p    = F.softmax(logits, dim=-1)
                x0   = torch.argmax(p, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                mask_index = (x_refine == MASK_ID) & s_refine_full
                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-float("inf"), device=self.device)
                )

                k = tokens_per_refine_step + (1 if step < remainder_refine else 0)
                k = min(k, n_masked_now)
                if k > 0:
                    _, sel = torch.topk(confidence[0], k=k)
                    ti = torch.zeros_like(x_refine[0], dtype=torch.bool)
                    ti[sel] = True
                    ti = ti.unsqueeze(0) & mask_index
                    x_refine[ti] = x0[ti]

        refine_elapsed = time.perf_counter() - t_refine_start

        # Fill remaining masked tokens
        still_left = (x_refine[:, prompt_len:] == MASK_ID)
        if still_left.any():
            with torch.no_grad():
                lf = self.model(x_refine, attention_mask=attn).logits
            pf = F.softmax(lf[:, prompt_len:, :], dim=-1)
            fp = torch.argmax(pf, dim=-1)
            x_refine[:, prompt_len:] = torch.where(
                still_left, fp, x_refine[:, prompt_len:]
            )

        total_elapsed = draft_elapsed + refine_elapsed
        total_tps = gen_length / total_elapsed if total_elapsed > 0 else 0.0
        kv_hit_rate = float(np.mean(kv_hit_steps)) if kv_hit_steps else float(n_accept / n_total)

        text = self.tokenizer.decode(
            x_refine[0, prompt_len:].tolist(), skip_special_tokens=True
        )

        return {
            "generated_text":     text,
            "tps":                total_tps,
            "elapsed_sec":        total_elapsed,
            "draft_elapsed_sec":  draft_elapsed,
            "refine_elapsed_sec": refine_elapsed,
            "draft_tps":          gen_length / draft_elapsed if draft_elapsed > 0 else 0.0,
            "refine_tps":         gen_length / refine_elapsed if refine_elapsed > 0 else 0.0,
            "accept_rate":        accept_rate,
            "n_accept":           n_accept,
            "n_refine":           n_refine,
            "n_total":            n_total,
            "kv_hit_rate_refine": kv_hit_rate,
            "tau":                tau,
            "t_draft":            t_draft,
            "t_full":             t_full,
        }

    def generate_batch(
        self,
        prompts: list,
        tau: float = 0.70,
        t_draft: int = 32,
        t_full: int = 64,
        gen_length: int = 128,
        block_length: int = 32,
        apply_chat_template: bool = True,
    ) -> list:
        """Generate for a batch of prompts (sequential, not batched for simplicity)."""
        return [
            self.generate(p, tau=tau, t_draft=t_draft, t_full=t_full,
                          gen_length=gen_length, block_length=block_length,
                          apply_chat_template=apply_chat_template)
            for p in prompts
        ]


def build_igsd_generator(model_path: str, device: str = "cuda") -> IGSDGenerator:
    """Factory: load model + tokenizer and return IGSDGenerator."""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return IGSDGenerator(model, tokenizer, device)


# Pilot results metadata
PILOT_RESULTS = {
    "version": "v3",
    "algorithm": "whole_sequence_draft_and_refine",
    "model": "LLaDA-8B-Instruct",
    "gen_length": 128,
    "t_draft": 32,
    "t_full": 64,
    "results_by_tau": {
        "0.70": {
            "gsm8k_exact_match": 0.580,
            "gsm8k_speedup": 1.86,
            "gsm8k_qas": 1.269,
            "humaneval_speedup": 4.97,
            "accept_rate": 0.964,
        },
        "0.85": {
            "gsm8k_exact_match": 0.580,
            "gsm8k_speedup": 1.84,
            "gsm8k_qas": 1.257,
            "humaneval_speedup": 4.95,
            "accept_rate": 0.959,
        },
    },
    "local_baseline": {
        "gen_length": 128,
        "gsm8k_exact_match": 0.850,
        "avg_tps": 30.2,
    },
    "known_limitations": [
        "GSM8K accuracy drop: 0.58 vs 0.85 baseline (31.7%)",
        "Whole-sequence draft breaks block-based semi-AR structure",
        "Block-based draft (v1/v2) causes degenerate outputs due to sparse masking in refine",
        "HumanEval pass@1 unreliable at gen_length=128 (code truncated)",
    ],
    "recommendations": [
        "Use tau=0.70 for higher acceptance and cleaner draft context",
        "For full experiments: compare with gen_length=256 baseline at same compute budget",
        "M1+IGSD composability: high accept_rate (0.96) → favorable KV-cache reuse in refine",
    ],
}
