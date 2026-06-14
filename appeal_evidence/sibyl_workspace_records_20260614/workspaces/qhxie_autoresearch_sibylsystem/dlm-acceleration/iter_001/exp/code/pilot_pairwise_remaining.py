"""
Pairwise Orthogonality Pilot: Remaining Meaningful Pairs.

Task: pilot_pairwise_remaining
Tests 3 pairwise combinations on 100 GSM8K samples (seed=42):
  1. M1+IGSD: IGSD with entropy-enhanced acceptance threshold
  2. M3+IGSD: IGSD draft guided by Qwen2.5-0.5B (guidance_weight=0.3)
  3. M1+M3:   M3 with entropy-based Qwen-call skip for confident positions

Orthogonality: Ortho = QAS_combined / max(QAS_A, QAS_B)
  - Ortho > 1.0: synergy
  - Ortho = 1.0: neutral
  - Ortho < 1.0: interference

Reference individual QAS from pilot/full runs:
  - M1: QAS=0.836 (from m1_pareto_full)
  - IGSD: QAS=1.194 (from igsd_pareto_full, best tau=0.9, T=16)
  - M3: QAS=1.675 (from m3_pareto_full, best gw=0.3)
"""
import os, sys, json, time, random, re, subprocess, gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

WORKSPACE  = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED     = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH = str(SHARED / "checkpoints" / "llada-8b-instruct")
QWEN_PATH  = str(SHARED / "checkpoints" / "qwen2.5-0.5b")
CODE_DIR   = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilot_pairwise"
TASK_ID    = "pilot_pairwise_remaining"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reference individual QAS scores for Ortho computation
REFERENCE_QAS = {
    "M1":   0.836,   # entropy cache, threshold=2.0
    "IGSD": 1.194,   # tau=0.9, T_draft=16
    "M3":   1.675,   # guidance_weight=0.3
}
BASELINE_GSM8K_TPS = 31.013
BASELINE_GSM8K_ACC = 0.7122
MASK_ID = 126336
N_SAMPLES = 100
SEED = 42
_LLADA_TOKENIZER = None

def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total, metric=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "step": step, "total_steps": total,
        "updated_at": datetime.now().isoformat(), "metric": metric or {},
    }))

def mark_done(status="success", summary=""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists(): pid_f.unlink()
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))

def load_gsm8k(n=100, seed=42):
    with open(SHARED / "datasets" / "gsm8k" / "test.json") as f:
        data = json.load(f)
    return random.Random(seed).sample(data, min(n, len(data)))

GSM8K_8SHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So they planted 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is 5.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 + 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is 9.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had $23. She bought 5 bagels for $3 each. So she spent 5 * 3 = $15. Now she has 23 - 15 = $8. The answer is 8.

"""

def gsm8k_prompt(q): return GSM8K_8SHOT + f"Question: {q}\nAnswer:"

def extract_answer(text):
    m = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None

def gsm8k_match(pred, gold):
    p, g = extract_answer(pred), extract_answer(gold)
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-6
    except: return p.strip() == g.strip()

def get_qwen_scores_simple(qwen_model, qwen_tokenizer, x, x0, device, max_ctx=512):
    """Simple Qwen scoring: returns score tensor same shape as x."""
    global _LLADA_TOKENIZER
    batch, seq_len = x.shape
    scores = torch.zeros(batch, seq_len, device=device)
    for b in range(batch):
        ids = x[b]
        mask_pos = (ids == MASK_ID).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0: continue
        pad_tok = qwen_tokenizer.pad_token_id or qwen_tokenizer.eos_token_id
        unmasked = ids.clone(); unmasked[ids == MASK_ID] = pad_tok
        seq = unmasked.cpu().tolist()[-max_ctx:]
        offset = max(0, len(unmasked) - max_ctx)
        try:
            text = _LLADA_TOKENIZER.decode(seq, skip_special_tokens=True)
            qenc = qwen_tokenizer(text, return_tensors="pt", max_length=max_ctx,
                                  truncation=True, add_special_tokens=False)
            qout = qwen_model(qenc["input_ids"].to(device))
            ql = qout.logits
            for pos in mask_pos:
                pi, qp = pos.item(), pos.item() - offset
                if qp < 0 or qp >= ql.shape[1]: continue
                ltok = x0[b, pi].item()
                if ltok == MASK_ID: continue
                tok_txt = _LLADA_TOKENIZER.decode([ltok], skip_special_tokens=True)
                if not tok_txt.strip(): continue
                qtids = qwen_tokenizer.encode(tok_txt, add_special_tokens=False)
                if not qtids: continue
                qtok = qtids[0]
                if qtok >= ql.shape[-1]: continue
                scores[b, pi] = float(F.softmax(ql[0, qp, :], dim=-1)[qtok].item())
        except Exception: pass
    return scores


# ── Combined Generation Functions ─────────────────────────────────────────────

@torch.no_grad()
def generate_m1_igsd(model, tokenizer, prompt, device,
                     tau=0.9, t_draft=16, t_full=64,
                     m1_entropy_threshold=2.0, gen_length=256):
    """M1+IGSD: IGSD with M1-enhanced acceptance in draft phase.

    M1 contribution: during draft, if token's entropy < m1_threshold
    (low entropy = high confidence), it counts as an 'accepted' candidate
    even if its max prob < tau. This extends S_accept.
    """
    msg = [{"role": "user", "content": prompt}]
    pt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([pt], add_special_tokens=False, padding=True, return_tensors="pt")
    inp = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    batch, plen = inp.shape[0], inp.shape[1]

    x = torch.full((batch, plen + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :plen] = inp
    full_attn = torch.cat([attn, torch.ones((batch, gen_length), device=device, dtype=attn.dtype)], dim=-1)

    t0 = time.perf_counter()

    # DRAFT PHASE
    for step in range(t_draft):
        mask_idx = x == MASK_ID
        if not mask_idx.any(): break
        logits = model(x, attention_mask=full_attn).logits
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        remaining = mask_idx[:, plen:].sum(dim=-1).float()
        n_unmask = torch.ceil(remaining / (t_draft - step)).long().clamp(min=1)

        gen_conf = torch.where(mask_idx[:, plen:], x0_p[:, plen:],
                               torch.tensor(-float('inf'), device=device))
        for b in range(batch):
            k = min(n_unmask[b].item(), (x[b, plen:] == MASK_ID).sum().item())
            if k <= 0: continue
            _, sel = torch.topk(gen_conf[b], k=int(k))
            for s in sel:
                x[b, plen + s.item()] = x0[b, plen + s.item()]

    # PARTITION: S_accept = {high tau} UNION {M1: low entropy}
    final_logits = model(x, attention_mask=full_attn).logits
    final_probs = F.softmax(final_logits, dim=-1)
    gen_ids = x[:, plen:]
    confidence = torch.gather(final_probs[:, plen:, :], -1, gen_ids.unsqueeze(-1)).squeeze(-1)

    # Entropy for M1 component
    entropy = -(final_probs[:, plen:, :] * torch.log(final_probs[:, plen:, :] + 1e-9)).sum(dim=-1)

    # S_accept: either high confidence (IGSD) or low entropy (M1)
    s_accept = (confidence >= tau) | (entropy <= m1_entropy_threshold)
    accept_rate = s_accept.float().mean().item()

    # REFINE PHASE: on S_refine only
    x_refine = x.clone()
    for b in range(batch):
        refine_pos = (~s_accept[b]).nonzero(as_tuple=True)[0] + plen
        x_refine[b, refine_pos] = MASK_ID

    for step in range(t_full):
        mask_idx = x_refine == MASK_ID
        if not mask_idx[:, plen:].any(): break
        logits = model(x_refine, attention_mask=full_attn).logits
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        remaining = mask_idx[:, plen:].sum(dim=-1).float()
        n_unmask = torch.ceil(remaining / (t_full - step)).long().clamp(min=1)
        gen_conf = torch.where(mask_idx[:, plen:], x0_p[:, plen:],
                               torch.tensor(-float('inf'), device=device))
        for b in range(batch):
            k = min(n_unmask[b].item(), (x_refine[b, plen:] == MASK_ID).sum().item())
            if k <= 0: continue
            _, sel = torch.topk(gen_conf[b], k=int(k))
            for s in sel:
                ap = plen + s.item()
                if x_refine[b, ap] == MASK_ID:
                    x_refine[b, ap] = x0[b, ap]

    elapsed = time.perf_counter() - t0
    text = tokenizer.batch_decode(x_refine[:, plen:], skip_special_tokens=True)[0]
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    return text, tps, {"accept_rate": accept_rate}


@torch.no_grad()
def generate_m3_igsd(llada_model, tokenizer, qwen_model, qwen_tok, prompt, device,
                     tau=0.9, t_draft=16, t_full=64, guidance_weight=0.3, gen_length=256):
    """M3+IGSD: IGSD with Qwen guidance in DRAFT phase token selection."""
    global _LLADA_TOKENIZER
    msg = [{"role": "user", "content": prompt}]
    pt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([pt], add_special_tokens=False, padding=True, return_tensors="pt")
    inp = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    batch, plen = inp.shape[0], inp.shape[1]

    x = torch.full((batch, plen + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :plen] = inp
    full_attn = torch.cat([attn_mask, torch.ones((batch, gen_length), device=device, dtype=attn_mask.dtype)], dim=-1)

    t0 = time.perf_counter()

    # DRAFT PHASE with Qwen guidance
    for step in range(t_draft):
        mask_idx = x == MASK_ID
        if not mask_idx.any(): break
        logits = llada_model(x, attention_mask=full_attn).logits
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        # Add Qwen guidance
        if guidance_weight > 0:
            qscores = get_qwen_scores_simple(qwen_model, qwen_tok, x, x0, device)
            blended = (1 - guidance_weight) * x0_p + guidance_weight * qscores
        else:
            blended = x0_p

        remaining = mask_idx[:, plen:].sum(dim=-1).float()
        n_unmask = torch.ceil(remaining / (t_draft - step)).long().clamp(min=1)
        gen_conf = torch.where(mask_idx[:, plen:], blended[:, plen:],
                               torch.tensor(-float('inf'), device=device))
        for b in range(batch):
            k = min(n_unmask[b].item(), (x[b, plen:] == MASK_ID).sum().item())
            if k <= 0: continue
            _, sel = torch.topk(gen_conf[b], k=int(k))
            for s in sel:
                x[b, plen + s.item()] = x0[b, plen + s.item()]

    # PARTITION (standard IGSD)
    final_logits = llada_model(x, attention_mask=full_attn).logits
    final_probs = F.softmax(final_logits, dim=-1)
    confidence = torch.gather(final_probs[:, plen:, :], -1, x[:, plen:].unsqueeze(-1)).squeeze(-1)
    s_accept = confidence >= tau
    accept_rate = s_accept.float().mean().item()

    # REFINE PHASE (standard IGSD, no guidance)
    x_r = x.clone()
    for b in range(batch):
        refine_pos = (~s_accept[b]).nonzero(as_tuple=True)[0] + plen
        x_r[b, refine_pos] = MASK_ID

    for step in range(t_full):
        mask_idx = x_r == MASK_ID
        if not mask_idx[:, plen:].any(): break
        logits = llada_model(x_r, attention_mask=full_attn).logits
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        remaining = mask_idx[:, plen:].sum(dim=-1).float()
        n_unmask = torch.ceil(remaining / (t_full - step)).long().clamp(min=1)
        gen_conf = torch.where(mask_idx[:, plen:], x0_p[:, plen:],
                               torch.tensor(-float('inf'), device=device))
        for b in range(batch):
            k = min(n_unmask[b].item(), (x_r[b, plen:] == MASK_ID).sum().item())
            if k <= 0: continue
            _, sel = torch.topk(gen_conf[b], k=int(k))
            for s in sel:
                ap = plen + s.item()
                if x_r[b, ap] == MASK_ID: x_r[b, ap] = x0[b, ap]

    elapsed = time.perf_counter() - t0
    text = tokenizer.batch_decode(x_r[:, plen:], skip_special_tokens=True)[0]
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    return text, tps, {"accept_rate": accept_rate}


@torch.no_grad()
def generate_m1_m3(llada_model, tokenizer, qwen_model, qwen_tok, prompt, device,
                   guidance_weight=0.3, m1_threshold=2.0, gen_length=256, block_length=32, steps=64):
    """M1+M3: Block-sequential generation with Qwen guidance + entropy skip."""
    from generate import get_num_transfer_tokens
    global _LLADA_TOKENIZER

    msg = [{"role": "user", "content": prompt}]
    pt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([pt], add_special_tokens=False, padding=True, return_tensors="pt")
    inp = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    batch, plen = inp.shape[0], inp.shape[1]

    x = torch.full((batch, plen + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :plen] = inp
    full_attn = torch.cat([attn_mask, torch.ones((batch, gen_length), device=device, dtype=attn_mask.dtype)], dim=-1)

    num_blocks = gen_length // block_length
    spb = steps // num_blocks
    t0 = time.perf_counter()

    qwen_calls_skipped = 0

    for bi in range(num_blocks):
        bs = plen + bi * block_length
        be = plen + (bi + 1) * block_length
        bm = x[:, bs:be] == MASK_ID
        nt = get_num_transfer_tokens(bm, spb)

        for step in range(spb):
            mask_idx = x == MASK_ID
            if not mask_idx[:, bs:be].any(): continue

            logits = llada_model(x, attention_mask=full_attn).logits
            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

            # M1: compute entropy; skip Qwen for low-entropy (high-confidence) positions
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            needs_guidance = entropy > m1_threshold  # high entropy = uncertain = use Qwen

            if guidance_weight > 0 and needs_guidance[:, bs:be].any():
                qscores = get_qwen_scores_simple(qwen_model, qwen_tok, x, x0, device)
                blended = torch.where(needs_guidance,
                                      (1 - guidance_weight) * x0_p + guidance_weight * qscores,
                                      x0_p)
            else:
                blended = x0_p
                qwen_calls_skipped += 1

            blended[:, be:] = -float('inf')
            x0 = torch.where(mask_idx, x0, x)
            conf = torch.where(mask_idx, blended, torch.tensor(-float('inf'), device=device))

            ti = torch.zeros_like(x0, dtype=torch.bool)
            for b in range(batch):
                k = nt[b, step].item()
                if k > 0:
                    _, sel = torch.topk(conf[b], k=int(k))
                    ti[b, sel] = True
            x[ti] = x0[ti]

    elapsed = time.perf_counter() - t0
    text = tokenizer.batch_decode(x[:, plen:], skip_special_tokens=True)[0]
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    return text, tps, {"qwen_calls_skipped": qwen_calls_skipped}


def eval_pair(gen_fn, data, label):
    """Evaluate a combined generation function on GSM8K data."""
    tps_list, correct = [], 0
    for i, item in enumerate(data):
        try:
            text, tps, stats = gen_fn(gsm8k_prompt(item["question"]))
            if gsm8k_match(text, item["answer"]): correct += 1
            if i >= 3: tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] {label}[{i}]: {e}")
        if (i+1) % 25 == 0:
            print(f"  {label} [{i+1}/{len(data)}] acc={correct/(i+1):.3f}")

    acc = correct / len(data) if data else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    speedup = avg_tps / max(BASELINE_GSM8K_TPS, 1e-6)
    acc_ret = acc / max(BASELINE_GSM8K_ACC, 1e-6)
    qas = float(speedup * acc_ret)
    print(f"  {label}: acc={acc:.3f}, tps={avg_tps:.1f}, speedup={speedup:.3f}x, QAS={qas:.3f}")
    return {"label": label, "acc": acc, "avg_tps": avg_tps, "speedup": speedup, "acc_ret": acc_ret, "qas": qas}


def main():
    global _LLADA_TOKENIZER
    write_pid()
    start_time = datetime.now()
    device = os.environ.get("CUDA_DEVICE", "cuda:0")
    print(f"[pilot_pairwise] Starting at {start_time.isoformat()} on {device}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("[pilot_pairwise] Loading LLaDA-8B-Instruct...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    _LLADA_TOKENIZER = tokenizer
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    print("[pilot_pairwise] Loading Qwen2.5-0.5B...")
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    qwen_model = AutoModel.from_pretrained(
        QWEN_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"[pilot_pairwise] Models loaded. VRAM: {torch.cuda.memory_allocated(device)//1024**2} MB")

    data = load_gsm8k(N_SAMPLES, SEED)
    print(f"[pilot_pairwise] {len(data)} GSM8K samples (seed={SEED})")

    results = {}
    pairs = [
        ("M1+IGSD",  lambda p: generate_m1_igsd(model, tokenizer, p, device)),
        ("M3+IGSD",  lambda p: generate_m3_igsd(model, tokenizer, qwen_model, qwen_tok, p, device)),
        ("M1+M3",    lambda p: generate_m1_m3(model, tokenizer, qwen_model, qwen_tok, p, device)),
    ]

    for i, (label, gen_fn) in enumerate(pairs):
        print(f"\n[pilot_pairwise] === Pair {i+1}/{len(pairs)}: {label} ===")
        r = eval_pair(gen_fn, data, label)
        results[label] = r
        report_progress(i+1, len(pairs), {"pair": label, "qas": r["qas"]})
        torch.cuda.empty_cache(); gc.collect()

    # Compute Ortho scores
    ortho_results = []
    for label, r in results.items():
        methods = label.split("+")
        ref_qas = [REFERENCE_QAS.get(m, 1.0) for m in methods]
        max_ref = max(ref_qas)
        ortho = r["qas"] / max(max_ref, 1e-6)
        print(f"  {label}: QAS={r['qas']:.3f}, max_individual={max_ref:.3f}, Ortho={ortho:.3f}")
        ortho_results.append({
            "pair": label, "methods": methods,
            "combined_qas": r["qas"],
            "individual_qas": dict(zip(methods, ref_qas)),
            "max_individual_qas": max_ref,
            "ortho": ortho,
            "speedup": r["speedup"],
            "acc_ret": r["acc_ret"],
        })

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    output = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": round(elapsed_min, 2),
        "ortho_results": ortho_results,
        "raw_results": results,
        "reference_qas": REFERENCE_QAS,
    }

    out_path = RESULTS_DIR / "all_pairs_ortho.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[pilot_pairwise] Results → {out_path}")

    # Update gpu_progress
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        if TASK_ID not in gp["completed"]: gp["completed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}): del gp["running"][TASK_ID]
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 55, "actual_min": round(elapsed_min),
        }
        gp_path.write_text(json.dumps(gp, indent=2))
    except Exception as e:
        print(f"WARNING: gpu_progress: {e}")

    best_ortho = max(ortho_results, key=lambda x: x["ortho"])
    summary = f"Pairwise pilot done. Best pair: {best_ortho['pair']} (Ortho={best_ortho['ortho']:.3f})"
    mark_done(status="success", summary=summary)
    report_progress(len(pairs), len(pairs), {"status": "done"})
    print(f"[pilot_pairwise] Done in {elapsed_min:.1f} min. {summary}")
    return output


if __name__ == "__main__":
    main()
