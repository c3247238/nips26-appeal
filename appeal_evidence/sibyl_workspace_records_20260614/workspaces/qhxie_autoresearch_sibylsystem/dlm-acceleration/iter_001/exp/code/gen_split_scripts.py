"""
Generate split scripts for full_m1_pareto and full_igsd_preseed.
"""
import re
from pathlib import Path

CODE_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current/exp/code")

# ── M1 split: t2 (threshold=2.0) and t3 (threshold=3.0) ─────────────────────
src = (CODE_DIR / "full_m1_pareto.py").read_text()

for thresh, gpu in [(2.0, 4), (3.0, 6)]:
    tag = "t2" if thresh == 2.0 else "t3"
    out_name = f"full_m1_pareto_{tag}.py"
    s = src

    # Override ENTROPY_THRESHOLDS
    s = re.sub(
        r'^ENTROPY_THRESHOLDS\s*=\s*\[.*?\]',
        f'ENTROPY_THRESHOLDS = [{thresh}]',
        s, flags=re.MULTILINE
    )
    # Override TASK_ID
    s = re.sub(
        r'^TASK_ID\s*=\s*"full_m1_pareto"',
        f'TASK_ID = "full_m1_pareto_{tag}"',
        s, flags=re.MULTILINE
    )
    # Override RESULTS_DIR to same dir (so we share the dir but separate files)
    # Override device
    s = re.sub(
        r'device\s*=\s*"cuda:0"',
        f'device = "cuda:{gpu}"',
        s
    )
    # Override partial_path  filename
    s = s.replace(
        '"m1_pareto_partial.json"',
        f'"m1_pareto_partial_{tag}.json"'
    )
    # Also keep reading the BASE partial to skip already-done thresholds
    # Insert code at the start of main() after partial_path definition:
    # load from base partial first, then from our own partial
    old_partial_block = '''    partial_path = RESULTS_DIR / "m1_pareto_partial_{tag}.json"
    all_threshold_results = {{}}
    if partial_path.exists():
        try:
            all_threshold_results = json.loads(partial_path.read_text())
            print(f"[full_m1_pareto] Resuming: found {{len(all_threshold_results)}} partial threshold results", flush=True)
        except:
            all_threshold_results = {{}}'''.format(tag=tag)

    new_partial_block = f'''    partial_path = RESULTS_DIR / "m1_pareto_partial_{tag}.json"
    all_threshold_results = {{}}
    # Load base partial (thresholds 0.5, 1.0 already done) to skip them
    base_partial_path = RESULTS_DIR / "m1_pareto_partial.json"
    if base_partial_path.exists():
        try:
            all_threshold_results = json.loads(base_partial_path.read_text())
            print(f"[full_m1_pareto_{tag}] Loaded base partial: {{len(all_threshold_results)}} thresholds done", flush=True)
        except:
            all_threshold_results = {{}}
    if partial_path.exists():
        try:
            own = json.loads(partial_path.read_text())
            all_threshold_results.update(own)
            print(f"[full_m1_pareto_{tag}] Resuming own partial: {{len(own)}} own threshold results", flush=True)
        except:
            pass'''

    s = s.replace(old_partial_block, new_partial_block)

    # Override output file
    s = s.replace(
        '"m1_pareto_full.json"',
        f'"m1_pareto_full_{tag}.json"'
    )
    # Fix print tags
    s = s.replace('[full_m1_pareto]', f'[full_m1_pareto_{tag}]')

    (CODE_DIR / out_name).write_text(s)
    print(f"Written: {out_name} (threshold={thresh}, GPU={gpu})")

print("Done generating M1 split scripts.")
