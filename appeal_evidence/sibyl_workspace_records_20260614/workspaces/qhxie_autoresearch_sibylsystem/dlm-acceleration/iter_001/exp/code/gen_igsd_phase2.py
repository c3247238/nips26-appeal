"""Generate 4 parallel Phase 2 finisher scripts for IGSD."""
import re
from pathlib import Path

CODE_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current/exp/code")
src = (CODE_DIR / "full_igsd_pareto.py").read_text()

# Extract header (everything up to def main)
main_start = src.index('\ndef main():')
header = src[:main_start]

configs = [
    (0.9, 16, 123, 0),
    (0.9, 16, 456, 1),
    (0.9, 32, 123, 4),
    (0.9, 32, 456, 5),
]

for tau, t_draft, seed, gpu in configs:
    tag = f"tau{str(tau).replace('.','')}_td{t_draft}_s{seed}"
    task_id = f"igsd_p2_{tag}"
    out_file = f"igsd_p2_{tag}.json"
    
    hdr = header.replace('TASK_ID       = "full_igsd_pareto"', f'TASK_ID       = "{task_id}"')
    hdr = re.sub(r'device\s*=\s*"cuda:0"', f'device = "cuda:{gpu}"', hdr)

    new_main = f'''
def main():
    write_pid()
    start_time = datetime.now()
    print(f"[{task_id}] tau={tau}, T_draft={t_draft}, seed={seed}, cuda:{gpu}", flush=True)

    device = "cuda:{gpu}"
    report_progress(0, 4, {{"status": "loading_model"}})

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left": tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    vram = profile_vram(device)
    print(f"[{task_id}] Model loaded. VRAM={{vram.get('vram_used_mb',0)}}MB", flush=True)

    gsm8k_data   = load_gsm8k()
    math500_data = load_math500()
    he_data      = load_humaneval()
    mbpp_data    = load_mbpp()

    partial_path = RESULTS_DIR / "{out_file}"
    all_results = {{}}
    completed_steps = [0]
    total_steps = 4

    config_key = "tau_{tau}_tdraft_{t_draft}"
    seed_key = "{seed}"

    # Check if already done
    if partial_path.exists():
        try:
            existing = json.loads(partial_path.read_text())
            if config_key in existing and seed_key in existing.get(config_key, {{}}):
                print(f"[{task_id}] Already done, skipping.", flush=True)
                mark_done(status="success", summary="already done")
                return
        except: pass

    random.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed})
    run_one_seed(model, tokenizer, gsm8k_data, math500_data, he_data, mbpp_data,
                 {seed}, {tau}, {t_draft}, device, all_results, partial_path,
                 completed_steps, total_steps)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    mark_done(status="success", summary=f"tau={tau},td={t_draft},seed={seed} done in {{elapsed:.1f}}min")
    report_progress(4, 4, {{"status": "done"}})
    print(f"[{task_id}] Done in {{elapsed:.1f}} minutes.", flush=True)


if __name__ == "__main__":
    main()
'''
    
    script = hdr + new_main
    out_path = CODE_DIR / f"igsd_p2_{tag}.py"
    out_path.write_text(script)
    print(f"Written: igsd_p2_{tag}.py (GPU {gpu})")

print("All 4 finisher scripts generated.")
