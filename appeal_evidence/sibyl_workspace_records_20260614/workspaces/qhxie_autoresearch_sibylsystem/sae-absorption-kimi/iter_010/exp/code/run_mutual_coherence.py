import json, torch, numpy as np
from pathlib import Path
from datetime import datetime
from sae_lens.synthetic.synthetic_model import SyntheticModel
from sae_lens.synthetic.synthetic_sae_runner import SyntheticModelConfig
from sae_lens.synthetic.hierarchy import HierarchyConfig
from sae_lens import StandardTrainingSAEConfig, SAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FEATURES, HIDDEN_DIM, EXPANSION = 1024, 256, 8

hierarchy_cfg = HierarchyConfig(total_root_nodes=32, branching_factor=4, max_depth=3)
model_cfg = SyntheticModelConfig(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, hierarchy=hierarchy_cfg, seed=42)
synthetic_model = SyntheticModel(cfg=model_cfg)
synthetic_model.to(DEVICE)

d_sae = HIDDEN_DIM * EXPANSION
sae_cfg = StandardTrainingSAEConfig(d_in=HIDDEN_DIM, d_sae=d_sae, l1_coefficient=5e-3, device=DEVICE, apply_b_dec_to_input=False)
sae = SAE.from_dict(sae_cfg.to_dict())
sae.to(DEVICE)

decoder = sae.W_dec
decoder_norm = decoder / (decoder.norm(dim=1, keepdim=True) + 1e-6)
gram = decoder_norm @ decoder_norm.T
diag = torch.diag(gram)
off_diag = gram - torch.diag(diag)

mutual_coherence = off_diag.abs().max().item()
mean_off_diag = off_diag.abs().mean().item()

results = {
    "task_id": "pilot_rq3_mutual_coherence",
    "mutual_coherence_max": mutual_coherence,
    "mutual_coherence_mean": mean_off_diag,
    "d_sae": d_sae,
    "timestamp": datetime.now().isoformat(),
}

results_dir = Path("exp/results/pilots")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "pilot_rq3_mutual_coherence_results.json", "w") as f:
    json.dump(results, f, indent=2)

marker = results_dir / "pilot_rq3_mutual_coherence_DONE"
marker.write_text(json.dumps({"status":"success","timestamp":datetime.now().isoformat()}))
print(f"Mutual coherence: max={mutual_coherence:.4f}, mean={mean_off_diag:.4f}")
