"""
Full Pairwise Orthogonality Study.

Task: full_pairwise_ortho
Tests 3 meaningful pairwise combinations across 2 benchmarks, 2 seeds:
  1. M1+IGSD: IGSD with entropy-enhanced acceptance threshold
  2. M3+IGSD: IGSD draft guided by Qwen2.5-0.5B (guidance_weight=0.3)
  3. M1+M3:   M3 with entropy-based Qwen-call skip for confident positions

Benchmarks: GSM8K (200 samples) + HumanEval (164 samples)
Seeds: 42, 123

Orthogonality: Ortho = QAS_combined / max(QAS_A, QAS_B)
"""
import os, sys, json, time, random, re, subprocess, gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED      = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH  = str(SHARED / "checkpoints" / "llada-8b-instruct")
QWEN_PATH   = str(SHARED / "checkpoints" / "qwen2.5-0.5b")
CODE_DIR    = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full_pairwise"
TASK_ID     = "full_pairwise_ortho"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reference individual QAS from full runs
REFERENCE_QAS = {
    "M1":   0.836,
    "IGSD": 1.194,
    "M3":   1.675,
}
BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
}

SEEDS    = [42, 123]
N_GSM8K  = 200
N_HE     = 164
MASK_ID  = 126336

# Combination hyper-params (best from individual pilots/full runs)
M1_THRESHOLD  = 2.0   # entropy threshold for M1
IGSD_TAU      = 0.9   # confidence threshold for IGSD partition
IGSD_T_DRAFT  = 16    # draft steps for IGSD
IGSD_T_FULL   = 64    # refine steps for IGSD
M3_GW         = 0.3   # guidance weight for M3


# ── Helpers ────────────────────────────────────────────────────────────────────
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


# ── Data ───────────────────────────────────────────────────────────────────────
def load_gsm8k(n, seed):
    with open(SHARED / "datasets" / "gsm8k" / "test.json") as f:
        data = json.load(f)
    return random.Random(seed).sample(data, min(n, len(data)))

def load_humaneval(n, seed):
    with open(SHARED / "datasets" / "humaneval" / "test.json") as f:
        data = json.load(f)
    return random.Random(seed).sample(data, min(n, len(data)))

GSM8K_8SHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So they planted 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason had 20 lollipops originally. Then he gave Denny some lollipops. Now Jason has 12 lollipops. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 + 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. For each of 4 days (monday to thursday) 5 more computers were added. So 5 * 4 = 20 computers were added. Now 9 + 20 = 29 computers are in the server room. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did Michael have at the end of wednesday?
Answer: Michael started with 58 golf balls. He lost 23 on Tuesday, leaving 58 - 23 = 35. Then he lost 2 more on Wednesday, leaving 35 - 2 = 33 golf balls. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had $23. She bought 5 bagels for $3 each. So she spent 5 * 3 = $15. Now she has 23 - 15 = $8. The answer is 8.

"""

def gsm8k_prompt(q):
    return GSM8K_8SHOT + f"Question: {q}\nAnswer:"

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

def he_pass_at_1(code, problem):
    full = problem["prompt"] + code + "\n" + problem["test"]
    full += f"\ncheck({problem['entry_point']})\n"
    try:
        r = subprocess.run(["python", "-c", full], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except: return False


# ── Qwen guidance ──────────────────────────────────────────────────────────────
@torch.no_grad()
def get_qwen_scores(qwen_model, qwen_tokenizer, llada_tokenizer,
                    x_ids, prompt_len, x0_ids, device):
    """Compute Qwen probability for each LLaDA-predicted token."""
    seq_str = llada_tokenizer.decode(x0_ids[prompt_len:], skip_special_tokens=True)
    q_enc = qwen_tokenizer(seq_str, return_tensors="pt", truncation=True, max_length=512).to(device)
    q_ids = q_enc["input_ids"]
    if q_ids.shape[1] < 2:
        gen_len = x0_ids.shape[0] - prompt_len
        return torch.ones(gen_len, device=device) * 0.5

    with torch.no_grad():
        q_logits = qwen_model(**q_enc).logits  # (1, L, V_qwen)

    q_probs = F.softmax(q_logits[0], dim=-1)  # (L, V_qwen)
    gen_len = x0_ids.shape[0] - prompt_len
    scores = torch.zeros(gen_len, device=device)

    q_tokens = q_ids[0]
    q_len = q_tokens.shape[0]
    for i in range(min(gen_len, q_len - 1)):
        tok = q_tokens[i + 1].item()
        if 0 <= tok < q_probs.shape[1]:
            scores[i] = q_probs[i, tok]
        else:
            scores[i] = 0.5
    return scores


# ── Generation: M1+IGSD ────────────────────────────────────────────────────────
@torch.no_grad()
def generate_m1_igsd(model, tokenizer, prompt, device,
                     tau=IGSD_TAU, t_draft=IGSD_T_DRAFT, t_full=IGSD_T_FULL,
                     m1_thresh=M1_THRESHOLD, gen_length=256):
    """IGSD with entropy-enhanced S_accept: accept if EITHER high-conf OR low-entropy."""
    msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attn = torch.cat([attention_mask,
                      torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    t0 = time.perf_counter()

    # === DRAFT PHASE ===
    for step in range(t_draft):
        mask_index = x == MASK_ID
        if not mask_index.any(): break
        logits = model(x, attention_mask=attn).logits
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        remaining = mask_index[:, prompt_len:].sum(dim=-1).float()
        steps_left = t_draft - step
        n_unmask = torch.ceil(remaining / steps_left).long().clamp(min=1)

        gen_conf = torch.where(mask_index[:, prompt_len:], x0_p[:, prompt_len:],
                               torch.tensor(-float('inf'), device=device))
        for b in range(batch):
            k = min(n_unmask[b].item(), (x[b, prompt_len:] == MASK_ID).sum().item())
            if k <= 0: continue
            _, sel = torch.topk(gen_conf[b], k=int(k))
            for s in sel:
                abs_pos = prompt_len + s.item()
                if x[b, abs_pos] == MASK_ID:
                    x[b, abs_pos] = x0[b, abs_pos]

    # === PARTITION (M1-enhanced) ===
    final_logits = model(x, attention_mask=attn).logits
    final_probs = F.softmax(final_logits, dim=-1)
    gen_ids = x[:, prompt_len:]
    confidence = torch.gather(final_probs[:, prompt_len:, :], -1,
                              gen_ids.unsqueeze(-1)).squeeze(-1)
    entropy = -(final_probs[:, prompt_len:, :] * torch.log(final_probs[:, prompt_len:, :] + 1e-10)).sum(-1)

    # M1+IGSD: accept if high-confidence OR low-entropy
    s_accept = (confidence >= tau) | (entropy <= m1_thresh)

    # === REFINE PHASE ===
    x_refine = x.clone()
    for b in range(batch):
        refine_positions = (~s_accept[b]).nonzero(as_tuple=True)[0] + prompt_len
        x_refine[b, refine_positions] = MASK_ID

    for step in range(t_full):
        mask_index = x_refine == MASK_ID
        if not mask_index[:, prompt_len:].any(): break
        logits = model(x_refine, attention_mask=attn).logits
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        remaining = mask_index[:, prompt_len:].sum(dim=-1).float()
        steps_left = t_full - step
        n_unmask = torch.ceil(remaining / steps_left).long().clamp(min=1)

        gen_conf = torch.where(mask_index[:, prompt_len:], x0_p[:, prompt_len:],
                               torch.tensor(-float('inf'), device=device))
        for b in range(batch):
            k = min(n_unmask[b].item(), (x_refine[b, prompt_len:] == MASK_ID).sum().item())
            if k <= 0: continue
            _, sel = torch.topk(gen_conf[b], k=int(k))
            for s in sel:
                abs_pos = prompt_len + s.item()
                if x_refine[b, abs_pos] == MASK_ID:
                    x_refine[b, abs_pos] = x0[b, abs_pos]

    elapsed = time.perf_counter() - t0
    gen_text = tokenizer.batch_decode(x_refine[:, prompt_len:], skip_special_tokens=True)[0]
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    return gen_text, tps


# ── Generation: M3+IGSD ────────────────────────────────────────────────────────
@torch.no_grad()
def generate_m3_igsd(model, tokenizer, qwen_model, qwen_tokenizer, prompt, device,
                     tau=IGSD_TAU, t_draft=IGSD_T_DRAFT, t_full=IGSD_T_FULL,
                     gw=M3_GW, gen_length=256):
    """IGSD with Qwen-guided draft phase."""
    msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attn = torch.cat([attention_mask,
                      torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    t0 = time.perf_counter()

    # === DRAFT PHASE (Qwen-guided) ===
    for step in range(t_draft):
        mask_index = x == MASK_ID
        if not mask_index.any(): break
        logits = model(x, attention_mask=attn).logits
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        # Get Qwen guidance
        try:
            qscores = get_qwen_scores(qwen_model, qwen_tokenizer, tokenizer,
                                      x[0], prompt_len, x0[0], device)
            blended = (1 - gw) * x0_p[0, prompt_len:] + gw * qscores
        except Exception:
            blended = x0_p[0, prompt_len:]

        remaining = mask_index[:, prompt_len:].sum(dim=-1).float()
        steps_left = t_draft - step
        n_unmask = torch.ceil(remaining / steps_left).long().clamp(min=1)

        gen_conf_b = torch.where(mask_index[0, prompt_len:], blended,
                                 torch.tensor(-float('inf'), device=device))
        k = min(n_unmask[0].item(), (x[0, prompt_len:] == MASK_ID).sum().item())
        if k > 0:
            _, sel = torch.topk(gen_conf_b, k=int(k))
            for s in sel:
                abs_pos = prompt_len + s.item()
                if x[0, abs_pos] == MASK_ID:
                    x[0, abs_pos] = x0[0, abs_pos]

    # === PARTITION (standard IGSD) ===
    final_logits = model(x, attention_mask=attn).logits
    final_probs = F.softmax(final_logits, dim=-1)
    gen_ids = x[:, prompt_len:]
    confidence = torch.gather(final_probs[:, prompt_len:, :], -1,
                              gen_ids.unsqueeze(-1)).squeeze(-1)
    s_accept = confidence >= tau

    # === REFINE PHASE (standard) ===
    x_refine = x.clone()
    for b in range(batch):
        refine_positions = (~s_accept[b]).nonzero(as_tuple=True)[0] + prompt_len
        x_refine[b, refine_positions] = MASK_ID

    for step in range(t_full):
        mask_index = x_refine == MASK_ID
        if not mask_index[:, prompt_len:].any(): break
        logits = model(x_refine, attention_mask=attn).logits
        probs = F.softmax(logits, dim=-1)
        x0 = torch.argmax(probs, dim=-1)
        x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        remaining = mask_index[:, prompt_len:].sum(dim=-1).float()
        steps_left = t_full - step
        n_unmask = torch.ceil(remaining / steps_left).long().clamp(min=1)

        gen_conf = torch.where(mask_index[:, prompt_len:], x0_p[:, prompt_len:],
                               torch.tensor(-float('inf'), device=device))
        for b in range(batch):
            k = min(n_unmask[b].item(), (x_refine[b, prompt_len:] == MASK_ID).sum().item())
            if k <= 0: continue
            _, sel = torch.topk(gen_conf[b], k=int(k))
            for s in sel:
                abs_pos = prompt_len + s.item()
                if x_refine[b, abs_pos] == MASK_ID:
                    x_refine[b, abs_pos] = x0[b, abs_pos]

    elapsed = time.perf_counter() - t0
    gen_text = tokenizer.batch_decode(x_refine[:, prompt_len:], skip_special_tokens=True)[0]
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    return gen_text, tps


# ── Generation: M1+M3 ──────────────────────────────────────────────────────────
@torch.no_grad()
def generate_m1_m3(model, tokenizer, qwen_model, qwen_tokenizer, prompt, device,
                   gw=M3_GW, m1_thresh=M1_THRESHOLD,
                   gen_length=256, block_length=32, steps=64):
    """M3 block-sequential generation with entropy-based Qwen-skip."""
    msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attn = torch.cat([attention_mask,
                      torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    num_blocks = (gen_length + block_length - 1) // block_length
    steps_per_block = max(1, steps // num_blocks)

    t0 = time.perf_counter()

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end   = min(prompt_len + gen_length, block_start + block_length)

        for step in range(steps_per_block):
            mask_index = (x[:, block_start:block_end] == MASK_ID)
            if not mask_index.any(): break

            logits = model(x, attention_mask=attn).logits[:, block_start:block_end, :]
            probs  = F.softmax(logits, dim=-1)
            x0     = torch.argmax(probs, dim=-1)
            x0_p   = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

            # Entropy for Qwen skip decision
            entropy = -(probs * torch.log(probs + 1e-10)).sum(-1)  # (B, block_len)
            needs_guidance = (entropy > m1_thresh).any(dim=-1)  # (B,)

            # Get Qwen scores if needed
            if needs_guidance[0].item():
                try:
                    full_x0 = torch.argmax(
                        F.softmax(model(x, attention_mask=attn).logits, dim=-1), dim=-1)
                    qscores_full = get_qwen_scores(qwen_model, qwen_tokenizer, tokenizer,
                                                   x[0], prompt_len, full_x0[0], device)
                    # Only block slice
                    bs = block_start - prompt_len
                    be = block_end - prompt_len
                    qscores = qscores_full[bs:be]
                    blended = (1 - gw) * x0_p[0] + gw * qscores
                except Exception:
                    blended = x0_p[0]
            else:
                blended = x0_p[0]  # skip Qwen

            # How many to unmask
            remaining = mask_index[0].sum().float()
            steps_left = steps_per_block - step
            k = int(torch.ceil(remaining / steps_left).clamp(min=1).item())
            k = min(k, mask_index[0].sum().item())
            if k <= 0: continue

            block_conf = torch.where(mask_index[0], blended,
                                     torch.tensor(-float('inf'), device=device))
            _, sel = torch.topk(block_conf, k=k)
            for s in sel:
                abs_pos = block_start + s.item()
                if x[0, abs_pos] == MASK_ID:
                    x[0, abs_pos] = x0[0, s.item()]

    elapsed = time.perf_counter() - t0
    gen_text = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    return gen_text, tps


# ── Evaluation ─────────────────────────────────────────────────────────────────
def eval_gsm8k(gen_fn, gsm8k_data, tag):
    tps_list, correct = [], 0
    for i, item in enumerate(gsm8k_data):
        try:
            text, tps = gen_fn(gsm8k_prompt(item["question"]))
            if gsm8k_match(text, item["answer"]): correct += 1
            if i >= 3: tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] {tag} gsm8k[{i}]: {e}")
        if (i+1) % 50 == 0:
            print(f"  {tag} gsm8k [{i+1}/{len(gsm8k_data)}] acc={correct/(i+1):.3f}")
    acc = correct / len(gsm8k_data) if gsm8k_data else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return acc, avg_tps

def eval_humaneval(gen_fn, he_data, tag):
    tps_list, passed = [], 0
    for i, item in enumerate(he_data):
        try:
            code, tps = gen_fn(f"Complete the following Python function:\n\n{item['prompt']}")
            if he_pass_at_1(code, item): passed += 1
            tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] {tag} he[{i}]: {e}")
    p1 = passed / len(he_data) if he_data else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return p1, avg_tps

def compute_metrics(gsm_acc, gsm_tps, he_p1, he_tps):
    gsm_speedup = gsm_tps / max(BASELINE["gsm8k"]["avg_tps"], 1e-6)
    gsm_ret     = gsm_acc / max(BASELINE["gsm8k"]["exact_match"], 1e-6)
    he_speedup  = he_tps / max(BASELINE["humaneval"]["avg_tps"], 1e-6)
    bl_he = BASELINE["humaneval"]["pass_at_1"]
    he_ret = he_p1 / max(bl_he, 1e-6) if bl_he > 0.01 else (1.0 if he_p1 >= bl_he else 0.0)

    combined_speedup = (gsm_speedup * N_GSM8K + he_speedup * N_HE) / (N_GSM8K + N_HE)
    combined_ret     = (gsm_ret * N_GSM8K + he_ret * N_HE) / (N_GSM8K + N_HE)
    qas = combined_speedup * combined_ret
    return {"gsm_speedup": gsm_speedup, "gsm_ret": gsm_ret,
            "he_speedup": he_speedup, "he_ret": he_ret,
            "combined_speedup": combined_speedup, "combined_ret": combined_ret,
            "qas": qas}


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now()
    device = os.environ.get("CUDA_DEVICE", "cuda:0")
    print(f"[full_pairwise_ortho] Starting at {start_time.isoformat()} on {device}")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    print("[full_pairwise_ortho] Loading LLaDA-8B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    print("[full_pairwise_ortho] Loading Qwen2.5-0.5B...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"[full_pairwise_ortho] Models loaded. VRAM: {torch.cuda.memory_allocated(device)//1024**2} MB")

    # Pre-load datasets
    datasets = {}
    for seed in SEEDS:
        datasets[seed] = {
            "gsm8k":     load_gsm8k(N_GSM8K, seed),
            "humaneval": load_humaneval(N_HE, seed),
        }

    # Combinations to test
    combos = [
        ("M1+IGSD", "M1", "IGSD"),
        ("M3+IGSD", "M3", "IGSD"),
        ("M1+M3",   "M1", "M3"),
    ]

    all_results = {}
    total_steps = len(combos) * len(SEEDS)
    prog_step = 0

    for combo_name, mA, mB in combos:
        all_results[combo_name] = {}
        for seed in SEEDS:
            print(f"\n[full_pairwise_ortho] {combo_name} | seed={seed}")
            gsm_data = datasets[seed]["gsm8k"]
            he_data  = datasets[seed]["humaneval"]

            # Define generation functions for this combo
            if combo_name == "M1+IGSD":
                def gsm_fn(prompt): return generate_m1_igsd(model, tokenizer, prompt, device, gen_length=256)
                def he_fn(prompt):  return generate_m1_igsd(model, tokenizer, prompt, device, gen_length=256)
            elif combo_name == "M3+IGSD":
                def gsm_fn(prompt): return generate_m3_igsd(model, tokenizer, qwen_model, qwen_tokenizer, prompt, device, gen_length=256)
                def he_fn(prompt):  return generate_m3_igsd(model, tokenizer, qwen_model, qwen_tokenizer, prompt, device, gen_length=256)
            else:  # M1+M3
                def gsm_fn(prompt): return generate_m1_m3(model, tokenizer, qwen_model, qwen_tokenizer, prompt, device, gen_length=256)
                def he_fn(prompt):  return generate_m1_m3(model, tokenizer, qwen_model, qwen_tokenizer, prompt, device, gen_length=256)

            gsm_acc, gsm_tps = eval_gsm8k(gsm_fn, gsm_data, combo_name)
            he_p1, he_tps    = eval_humaneval(he_fn, he_data, combo_name)
            m = compute_metrics(gsm_acc, gsm_tps, he_p1, he_tps)

            ref_qas_a = REFERENCE_QAS[mA]
            ref_qas_b = REFERENCE_QAS[mB]
            ortho = m["qas"] / max(ref_qas_a, ref_qas_b, 1e-6)

            print(f"  {combo_name}: speedup={m['combined_speedup']:.3f}x, ret={m['combined_ret']:.3f}, QAS={m['qas']:.3f}, Ortho={ortho:.3f}")
            all_results[combo_name][str(seed)] = {
                "gsm8k": {"exact_match": gsm_acc, "avg_tps": gsm_tps},
                "humaneval": {"pass_at_1": he_p1, "avg_tps": he_tps},
                **m,
                "ref_qas_A": ref_qas_a, "ref_qas_B": ref_qas_b,
                "ortho": ortho,
            }
            prog_step += 1
            report_progress(prog_step, total_steps, {"combo": combo_name, "seed": seed, "qas": m["qas"]})
            torch.cuda.empty_cache(); gc.collect()

    # ── Aggregate over seeds ───────────────────────────────────────────────────
    summary = []
    for combo_name, mA, mB in combos:
        seed_data = all_results[combo_name]
        avg_qas    = float(np.mean([v["qas"]   for v in seed_data.values()]))
        avg_speedup= float(np.mean([v["combined_speedup"] for v in seed_data.values()]))
        avg_ret    = float(np.mean([v["combined_ret"]     for v in seed_data.values()]))
        avg_ortho  = float(np.mean([v["ortho"]  for v in seed_data.values()]))
        ref_max    = max(REFERENCE_QAS[mA], REFERENCE_QAS[mB])
        summary.append({
            "combo": combo_name,
            "methods": [mA, mB],
            "avg_speedup": avg_speedup,
            "avg_acc_ret": avg_ret,
            "avg_qas": avg_qas,
            "ortho": avg_ortho,
            "ref_qas_max": ref_max,
            "verdict": "SYNERGY" if avg_ortho > 1.05 else ("NEUTRAL" if avg_ortho > 0.95 else "INTERFERENCE"),
        })

    print("\n[full_pairwise_ortho] === PAIRWISE SUMMARY ===")
    print(f"{'Combo':<15} {'Speedup':>9} {'AccRet':>7} {'QAS':>6} {'Ortho':>7} {'Verdict':<15}")
    for s in summary:
        print(f"  {s['combo']:<13} {s['avg_speedup']:>9.3f}x {s['avg_acc_ret']:>7.3f} "
              f"{s['avg_qas']:>6.3f} {s['ortho']:>7.3f} {s['verdict']:<15}")

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "seeds": SEEDS,
        "n_gsm8k": N_GSM8K,
        "n_humaneval": N_HE,
        "reference_qas": REFERENCE_QAS,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": round(elapsed_min, 2),
        "pairwise_summary": summary,
        "all_results": all_results,
    }

    out_path = RESULTS_DIR / "full_pairwise_ortho.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[full_pairwise_ortho] Results → {out_path}")

    # Update gpu_progress
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {}
        gp.setdefault("completed", [])
        if TASK_ID not in gp["completed"]: gp["completed"].append(TASK_ID)
        gp.get("running", {}).pop(TASK_ID, None)
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 180, "actual_min": round(elapsed_min),
            "start_time": start_time.isoformat(), "end_time": end_time.isoformat(),
        }
        gp_path.write_text(json.dumps(gp, indent=2))
    except Exception as e:
        print(f"WARNING: gpu_progress update failed: {e}")

    top_combo = max(summary, key=lambda x: x["avg_qas"])
    summ_str = (f"Pairwise ortho done in {elapsed_min:.1f}min. "
                f"Best: {top_combo['combo']} QAS={top_combo['avg_qas']:.3f} Ortho={top_combo['ortho']:.3f}")
    mark_done(status="success", summary=summ_str)
    print("[full_pairwise_ortho] Done.")
    return output


if __name__ == "__main__":
    main()
