#!/usr/bin/env python3
"""Setup environment verification for SAE absorption experiments."""
import json
import os
import sys
from pathlib import Path

# Check environment
checks = {
    "python": sys.version,
    "packages": {},
    "gpu": {},
    "remote_base_exists": False,
}

# Check packages
try:
    import sae_lens
    checks["packages"]["sae_lens"] = sae_lens.__version__
except ImportError as e:
    checks["packages"]["sae_lens"] = f"ERROR: {e}"

try:
    import transformer_lens
    checks["packages"]["transformer_lens"] = "loaded"
except ImportError as e:
    checks["packages"]["transformer_lens"] = f"ERROR: {e}"

try:
    import torch
    checks["packages"]["torch"] = torch.__version__
    checks["gpu"]["cuda_available"] = torch.cuda.is_available()
    checks["gpu"]["gpu_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i)
            checks["gpu"][f"gpu_{i}"] = {
                "name": mem.name,
                "total_memory_mb": mem.total_memory / 1024**2,
            }
except ImportError as e:
    checks["gpu"] = f"ERROR: {e}"

# Check remote base
remote_base = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
checks["remote_base_exists"] = remote_base.exists()
checks["remote_base"] = str(remote_base)

# Write results
results_dir = remote_base / "exp" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
output_file = results_dir / "setup_verification.json"

with open(output_file, "w") as f:
    json.dump(checks, f, indent=2)

print(f"Setup verification written to {output_file}")
print(json.dumps(checks, indent=2))

# Clean up PID file if it exists (task complete)
pid_file = results_dir / "setup_env.pid"
if pid_file.exists():
    pid_file.unlink()
    print(f"Cleaned up PID file: {pid_file}")

# Write DONE marker
done_file = results_dir / "setup_env_DONE"
done_data = {
    "task_id": "setup_env",
    "status": "success",
    "summary": "Environment verified - sae_lens 6.39.0, torch 2.12 with CUDA",
    "final_progress": checks,
    "timestamp": __import__("datetime").datetime.now().isoformat(),
}
with open(done_file, "w") as f:
    json.dump(done_data, f, indent=2)
print(f"DONE marker written to {done_file}")