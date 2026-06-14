"""Generate the IGSD preseed script for top-2 configs on GPU 7."""
import re
from pathlib import Path

CODE_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current/exp/code")
src = (CODE_DIR / "full_igsd_pareto.py").read_text()

# Find the main() function and replace it
main_start = src.index('\ndef main():')
header = src[:main_start]

# Apply substitutions to header
header = header.replace('TASK_ID       = "full_igsd_pareto"', 'TASK_ID       = "full_igsd_preseed"')
header = re.sub(r'device\s*=\s*"cuda:0"', 'device = "cuda:7"', header)
# Keep all other functions intact

new_main = '''
def main():
    write_pid()
    start_time = datetime.now()
    print(f"[full_igsd_preseed] Starting preseed of top-2 configs on cuda:7", flush=True)
    print(f"[full_igsd_preseed] Configs: (tau=0.7, T_draft=16), (tau=0.8, T_draft=16)", flush=True)
    print(f"[full_igsd_preseed] Seeds: [123, 456]", flush=True)

    # Top-2 configs based on QAS ranking from seed=42 sweep
    TOP_CONFIGS = [(0.7, 16), (0.8, 16)]
    PRESEED_SEEDS = [123, 456]

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = "cuda:7"
    report_progress(0, 16, {"status": "loading_model"})

    print(f"[full_igsd_preseed] Loading LLaDA-8B-Instruct on cuda:7...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    vram_after_load = profile_vram(device)
    print(f"[full_igsd_preseed] Model loaded. VRAM: {vram_after_load.get(\'vram_used_mb\',0)} MB", flush=True)

    print("[full_igsd_preseed] Loading datasets...", flush=True)
    gsm8k_data   = load_gsm8k()
    math500_data = load_math500()
    he_data      = load_humaneval()
    mbpp_data    = load_mbpp()

    partial_path = RESULTS_DIR / "igsd_pareto_preseed.json"
    all_results = {}
    if partial_path.exists():
        try:
            all_results = json.loads(partial_path.read_text())
            print(f"[full_igsd_preseed] Resuming: {len(all_results)} configs in preseed partial", flush=True)
        except:
            all_results = {}

    total_steps = len(TOP_CONFIGS) * len(PRESEED_SEEDS) * 4  # 2 configs × 2 seeds × 4 benchmarks = 16
    completed_steps = [0]

    for tau, t_draft in TOP_CONFIGS:
        config_key = f"tau_{tau}_tdraft_{t_draft}"
        print(f"\\n[full_igsd_preseed] Config: tau={tau}, T_draft={t_draft}", flush=True)
        for seed in PRESEED_SEEDS:
            seed_key = str(seed)
            if seed_key in all_results.get(config_key, {}):
                print(f"  [skip] seed={seed} already done", flush=True)
                completed_steps[0] += 4
                continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            run_one_seed(model, tokenizer, gsm8k_data, math500_data, he_data, mbpp_data,
                         seed, tau, t_draft, device, all_results, partial_path,
                         completed_steps, total_steps)

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    out_path = RESULTS_DIR / "igsd_pareto_preseed_done.json"
    out_path.write_text(json.dumps({
        "task_id": "full_igsd_preseed",
        "top_configs": TOP_CONFIGS,
        "preseed_seeds": PRESEED_SEEDS,
        "elapsed_minutes": elapsed_min,
        "results": all_results,
    }, indent=2))
    print(f"[full_igsd_preseed] Done in {elapsed_min:.1f} min. Results in {out_path}", flush=True)

    mark_done(status="success", summary=f"Preseed done in {elapsed_min:.1f} min")
    report_progress(total_steps, total_steps, {"status": "done"})


if __name__ == "__main__":
    main()
'''

result = header + new_main
out_path = CODE_DIR / "full_igsd_preseed.py"
out_path.write_text(result)
print(f"Written: full_igsd_preseed.py")
