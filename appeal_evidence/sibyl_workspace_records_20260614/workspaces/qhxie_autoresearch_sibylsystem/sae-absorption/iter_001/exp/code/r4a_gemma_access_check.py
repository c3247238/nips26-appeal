"""
r4a_gemma_access_check.py
Round 4-A Setup: Verify Gemma 2B Access or Confirm Llama Fallback

This script:
1. Tests HuggingFace access to Gemma 2B (google/gemma-2-2b)
2. Tests HuggingFace access to Llama fallback models
3. Checks SAELens catalog for available SAE releases for whichever model works
4. Actually attempts to load a small SAE weight matrix to confirm end-to-end functionality
5. Writes access_status.json documenting the primary model and available SAE configs

Runs on GPU 5 (PILOT mode, estimated ~10 min)
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "r4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "r4a_gemma_access_check"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"

# Write PID file immediately
PID_FILE.write_text(str(os.getpid()))

print(f"[{datetime.now().isoformat()}] Starting {TASK_ID}")
print(f"PID: {os.getpid()}")

def write_progress(step: str, details: dict):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "step": step,
        "details": details,
        "updated_at": datetime.now().isoformat(),
    }))
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {step}: {details}")

def mark_done(status: str, summary: str, result: dict):
    if PID_FILE.exists():
        PID_FILE.unlink()
    marker = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker, indent=2))

# ── Step 1: SAELens catalog ────────────────────────────────────────────────────
write_progress("catalog_lookup", {"msg": "Loading SAELens pretrained SAE catalog"})

from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

releases = get_pretrained_saes_directory()
all_releases = list(releases.keys())

# Gemma releases of interest
GEMMA_TARGET_RELEASES = [
    "gemma-scope-2b-pt-res",
    "gemma-scope-2b-pt-res-canonical",
]
# Llama releases of interest
LLAMA_TARGET_RELEASES = [
    "llama_scope_lxr_8x",
    "llama_scope_lxr_32x",
    "llama-3-8b-it-res-jh",
]

gemma_available = [r for r in GEMMA_TARGET_RELEASES if r in releases]
llama_available = [r for r in LLAMA_TARGET_RELEASES if r in releases]

write_progress("catalog_loaded", {
    "total_releases": len(all_releases),
    "gemma_sae_releases_found": gemma_available,
    "llama_sae_releases_found": llama_available,
})

# ── Step 2: Test Gemma 2B model access ────────────────────────────────────────
write_progress("gemma_access_test", {"msg": "Testing google/gemma-2-2b HuggingFace access"})

gemma_model_accessible = False
gemma_error = None
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("google/gemma-2-2b", token=os.environ.get("HF_TOKEN"))
    gemma_model_accessible = True
    gemma_d_model = cfg.hidden_size
    gemma_n_layers = cfg.num_hidden_layers
    write_progress("gemma_access_granted", {
        "model": "google/gemma-2-2b",
        "d_model": gemma_d_model,
        "n_layers": gemma_n_layers,
    })
except Exception as e:
    gemma_error = str(e)[:300]
    write_progress("gemma_access_denied", {"error": gemma_error[:100]})
    print(f"  Gemma 2B access DENIED: {gemma_error[:100]}")

# ── Step 3: Test Llama model access ───────────────────────────────────────────
llama_model_accessible = False
llama_model_name = None
llama_d_model = None
llama_error = None

llama_candidates = [
    ("meta-llama/Llama-3.1-8B", "llama-3.1-8b"),
    ("meta-llama/Llama-3.1-3B", "llama-3.1-3b"),
    ("meta-llama/Llama-3.2-3B", "llama-3.2-3b"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct"),
]

write_progress("llama_access_test", {"msg": "Testing Llama model access"})
for hf_name, short_name in llama_candidates:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(hf_name, token=os.environ.get("HF_TOKEN"))
        llama_model_accessible = True
        llama_model_name = hf_name
        llama_d_model = cfg.hidden_size
        llama_n_layers = cfg.num_hidden_layers
        write_progress("llama_access_granted", {
            "model": hf_name,
            "d_model": llama_d_model,
            "n_layers": llama_n_layers,
        })
        break
    except Exception as e:
        llama_error = str(e)[:200]
        write_progress("llama_access_denied", {"model": hf_name, "error": str(e)[:80]})

# ── Step 4: Test SAE loading ───────────────────────────────────────────────────
# Try to load a small SAE weight matrix to confirm SAELens works end-to-end
write_progress("sae_load_test", {"msg": "Testing SAE loading from Gemma Scope catalog"})

sae_test_results = []

# Test Gemma Scope SAE (SAE weights don't require Gemma model access)
# The SAE weights are in google/gemma-scope-2b-pt-res which is NOT gated
gemma_sae_configs_to_test = [
    ("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_82",  "L12-16k"),
    ("gemma-scope-2b-pt-res", "layer_12/width_65k/average_l0_72",  "L12-65k"),
    ("gemma-scope-2b-pt-res", "layer_5/width_16k/average_l0_68",   "L5-16k"),
    ("gemma-scope-2b-pt-res", "layer_5/width_65k/average_l0_73",   "L5-65k"),
    ("gemma-scope-2b-pt-res", "layer_19/width_16k/average_l0_73",  "L19-16k"),
    ("gemma-scope-2b-pt-res", "layer_19/width_65k/average_l0_73",  "L19-65k"),
]

# Check which Gemma Scope IDs actually exist in catalog
if "gemma-scope-2b-pt-res" in releases:
    rel = releases["gemma-scope-2b-pt-res"]
    valid_ids = set(rel.saes_map.keys()) if hasattr(rel, 'saes_map') else set()

    # Map target configs to nearest valid ID
    def find_nearest_id(layer: int, width_k: int, valid_ids: set) -> str:
        """Find the nearest valid SAE ID by layer and width."""
        layer_str = f"layer_{layer}/width_{width_k}k"
        matches = [id for id in valid_ids if id.startswith(layer_str)]
        if not matches:
            return None
        # Prefer moderate L0 values (not too sparse, not too dense)
        # Sort by l0 value and pick the middle one
        def get_l0(id_str):
            try:
                return int(id_str.split("average_l0_")[1])
            except:
                return 999
        matches_sorted = sorted(matches, key=get_l0)
        return matches_sorted[len(matches_sorted)//2]  # median L0

    # Rebuild test configs with validated IDs
    target_configs = [
        (5, 16, "L5-16k"),
        (5, 65, "L5-65k"),
        (12, 16, "L12-16k"),
        (12, 65, "L12-65k"),
        (19, 16, "L19-16k"),
        (19, 65, "L19-65k"),
    ]

    gemma_sae_configs_to_test = []
    gemma_available_sae_configs = []

    for layer, width_k, name in target_configs:
        sae_id = find_nearest_id(layer, width_k, valid_ids)
        if sae_id:
            gemma_sae_configs_to_test.append(("gemma-scope-2b-pt-res", sae_id, name, layer, width_k))
            gemma_available_sae_configs.append({
                "name": name,
                "release": "gemma-scope-2b-pt-res",
                "sae_id": sae_id,
                "layer": layer,
                "width_k": width_k,
            })
        else:
            print(f"  WARNING: No valid ID found for {name} (L{layer}-{width_k}k)")

    write_progress("gemma_sae_ids_resolved", {
        "n_configs": len(gemma_sae_configs_to_test),
        "configs": [(name, sae_id) for _, sae_id, name, _, _ in gemma_sae_configs_to_test],
    })
else:
    gemma_available_sae_configs = []
    write_progress("gemma_sae_release_not_found", {"error": "gemma-scope-2b-pt-res not in catalog"})

# Actually try loading one Gemma SAE to confirm weights are accessible
from sae_lens import SAE

gemma_sae_loadable = False
gemma_sae_test_result = {}

if gemma_sae_configs_to_test:
    release, sae_id, name, layer, width_k = gemma_sae_configs_to_test[2]  # Try L12-16k first
    write_progress("sae_load_attempt", {"release": release, "sae_id": sae_id, "name": name})
    try:
        t0 = time.time()
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device="cpu")
        t1 = time.time()
        W_enc = sae.W_enc
        W_dec = sae.W_dec
        gemma_sae_loadable = True
        gemma_sae_test_result = {
            "release": release,
            "sae_id": sae_id,
            "name": name,
            "W_enc_shape": list(W_enc.shape),
            "W_dec_shape": list(W_dec.shape),
            "d_in": W_enc.shape[0],
            "d_sae": W_enc.shape[1],
            "load_time_sec": round(t1 - t0, 1),
        }
        write_progress("sae_load_success", gemma_sae_test_result)
        del sae  # Free memory
    except Exception as e:
        gemma_sae_test_result = {"error": str(e)[:300]}
        write_progress("sae_load_failed", {"error": str(e)[:200]})

# ── Step 5: Test Llama SAE loading (if Llama model is accessible) ─────────────
llama_sae_configs = []
llama_sae_loadable = False
llama_sae_test_result = {}

# llama_scope_lxr_8x maps to meta-llama/Llama-3.1-8B
# We can try to load the SAE weights regardless of model access
if "llama_scope_lxr_8x" in releases:
    llama_rel = releases["llama_scope_lxr_8x"]
    llama_ids = list(llama_rel.saes_map.keys()) if hasattr(llama_rel, 'saes_map') else []

    # Target layer 12 (l12r_8x) and layer 6 (l6r_8x)
    target_llama_ids = [(f"l12r_8x", "L12-8x"), (f"l6r_8x", "L6-8x")]

    for llama_id, llama_name in target_llama_ids:
        if llama_id in llama_ids:
            llama_sae_configs.append({
                "name": llama_name,
                "release": "llama_scope_lxr_8x",
                "sae_id": llama_id,
                "model": "meta-llama/Llama-3.1-8B",
            })

    # Try loading one Llama SAE
    if llama_sae_configs:
        test_cfg = llama_sae_configs[0]
        write_progress("llama_sae_load_attempt", {"release": test_cfg["release"], "sae_id": test_cfg["sae_id"]})
        try:
            t0 = time.time()
            sae = SAE.from_pretrained(release=test_cfg["release"], sae_id=test_cfg["sae_id"], device="cpu")
            t1 = time.time()
            llama_sae_loadable = True
            llama_sae_test_result = {
                "release": test_cfg["release"],
                "sae_id": test_cfg["sae_id"],
                "W_enc_shape": list(sae.W_enc.shape),
                "W_dec_shape": list(sae.W_dec.shape),
                "d_in": sae.W_enc.shape[0],
                "d_sae": sae.W_enc.shape[1],
                "load_time_sec": round(t1 - t0, 1),
            }
            write_progress("llama_sae_load_success", llama_sae_test_result)
            del sae
        except Exception as e:
            llama_sae_test_result = {"error": str(e)[:300]}
            write_progress("llama_sae_load_failed", {"error": str(e)[:200]})

# ── Step 6: Determine primary model and strategy ───────────────────────────────
# Decision logic:
# - If Gemma 2B model is accessible AND Gemma SAEs load: primary = Gemma 2B
# - If Gemma 2B model is NOT accessible (gated) but Gemma SAEs load:
#   For EDA validation (weight-only, no activations): Gemma SAEs are usable
#   For label generation (needs model): must use a fallback model
# - For cross-domain analysis that requires running the model: use available model

write_progress("deciding_strategy", {"msg": "Determining primary model and strategy"})

# Key insight: EDA = 1 - cos(W_enc_j, W_dec_j) is WEIGHT ONLY, no model needed
# RAVEL probes and first-letter probes DO need model activations
# FeatureAbsorptionCalculator also needs model

# If Gemma 2B model is gated but SAE weights load:
#   - EDA weights-only analysis: CAN proceed with Gemma SAEs
#   - Label generation for validation: must use fallback (note: Neuronpedia proxy labels still usable)
#   - Probe training: need fallback model

# Check for GPT-2 SAEs (confirmed working from Phase 5)
gpt2_sae_available = "gpt2-small-res-jb" in releases or any("gpt2" in r for r in releases)
gpt2_releases = [r for r in releases.keys() if "gpt2" in r.lower() and "small" in r.lower()]

strategy_notes = []
if gemma_model_accessible:
    primary_model = "google/gemma-2-2b"
    primary_sae_release = "gemma-scope-2b-pt-res"
    fallback_used = False
    strategy_notes.append("Gemma 2B model accessible, using as primary")
elif gemma_sae_loadable:
    # Gemma SAE weights load but model is gated
    # For R4-A: use Gemma SAE weights for EDA (no model needed)
    # For probe training and label generation: use GPT-2 or other available model
    primary_model = "google/gemma-2-2b"  # SAE is for this model
    primary_sae_release = "gemma-scope-2b-pt-res"
    fallback_used = False  # SAE weights ARE accessible
    strategy_notes.append("Gemma 2B model GATED but SAE WEIGHTS load (gemma-scope-2b-pt-res is public)")
    strategy_notes.append("EDA weight-only analysis: CAN proceed with Gemma Scope SAE weights")
    strategy_notes.append("Probe training / label generation: need alternative model for activations")
    strategy_notes.append("R4-A direct labels: will use Gemma Scope canonical approach (same as R3) OR GPT-2 as bridge")
else:
    primary_model = "meta-llama/Llama-3.1-8B" if llama_model_accessible else "gpt2"
    primary_sae_release = "llama_scope_lxr_8x" if llama_sae_loadable else "gpt2-small-res-jb"
    fallback_used = True
    strategy_notes.append("Both Gemma 2B model AND SAE weights inaccessible, using full fallback")

# ── Step 7: Check previously used SAE configs from Phase 1 ────────────────────
write_progress("checking_previous_results", {"msg": "Loading Phase 1/3/5 SAE configs for continuity"})

phase1_sae_configs = []
try:
    with open(WORKSPACE / "exp" / "results" / "full" / "phase1_eda_deda_validation.json") as f:
        phase1 = json.load(f)
    for item in phase1.get("per_sae_results", []):
        cfg = item.get("config", {})
        if cfg:
            phase1_sae_configs.append({
                "name": cfg.get("name"),
                "release": cfg.get("release"),
                "sae_id": cfg.get("sae_id"),
                "layer": cfg.get("layer"),
                "width_k": cfg.get("width_k"),
                "d_in": cfg.get("d_in"),
                "d_sae": cfg.get("d_sae"),
            })
except Exception as e:
    write_progress("phase1_load_error", {"error": str(e)[:100]})

# ── Step 8: Build access_status.json ──────────────────────────────────────────
write_progress("writing_results", {"msg": "Writing access_status.json"})

access_status = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",

    # Model access status
    "gemma_2b_model_accessible": gemma_model_accessible,
    "gemma_2b_model_error": gemma_error if not gemma_model_accessible else None,
    "gemma_sae_weights_loadable": gemma_sae_loadable,
    "gemma_sae_test_result": gemma_sae_test_result,

    "llama_model_accessible": llama_model_accessible,
    "llama_model_name": llama_model_name if llama_model_accessible else None,
    "llama_d_model": llama_d_model if llama_model_accessible else None,
    "llama_sae_loadable": llama_sae_loadable,
    "llama_sae_test_result": llama_sae_test_result,

    # Primary model decision
    "primary_model": primary_model,
    "primary_sae_release": primary_sae_release,
    "fallback_used": fallback_used,
    "strategy_notes": strategy_notes,

    # Available SAE configs
    "available_gemma_sae_configs": gemma_available_sae_configs,
    "available_llama_sae_configs": llama_sae_configs,
    "gpt2_releases_available": gpt2_releases[:5],

    # Continuity with previous phases
    "phase1_sae_configs": phase1_sae_configs,

    # Round 4 pipeline gating
    "r4a_eda_can_proceed": gemma_sae_loadable,  # EDA is weight-only
    "r4a_direct_labels_approach": (
        "weight_only_eda" if not gemma_model_accessible else "direct_absorption_labels"
    ),
    "r4b_ravel_probes_approach": (
        "skip_or_gpt2_bridge" if not gemma_model_accessible and not llama_model_accessible
        else "proper_on_target_model"
    ),
    "r4c_llama_replication_approach": (
        "llama_scope_weights_only" if llama_sae_loadable and not llama_model_accessible
        else ("full_llama_pipeline" if llama_model_accessible else "additional_gpt2_configs")
    ),

    # Go/no-go per pass criteria
    "pass_criteria_met": gemma_sae_loadable,  # At minimum, Gemma Scope SAE weights must load
    "pass_criteria_notes": [
        f"Gemma SAE weights loadable: {gemma_sae_loadable}",
        f"n_available_gemma_sae_configs: {len(gemma_available_sae_configs)}",
        f"Gemma 2B model accessible: {gemma_model_accessible}",
        f"Llama model accessible: {llama_model_accessible}",
    ],
}

# Save to r4 results directory
output_path = RESULTS_DIR / "r4a_access_status.json"
output_path.write_text(json.dumps(access_status, indent=2))
print(f"\n  Results saved to: {output_path}")

# ── Step 9: Write pilot summary ───────────────────────────────────────────────
pilot_summary = {
    "overall_recommendation": "GO" if gemma_sae_loadable else "NO_GO",
    "candidates": [
        {
            "candidate_id": "shared",
            "go_no_go": "GO" if gemma_sae_loadable else "NO_GO",
            "confidence": 0.90 if gemma_sae_loadable else 0.10,
            "key_findings": {
                "gemma_model_accessible": gemma_model_accessible,
                "gemma_sae_weights_accessible": gemma_sae_loadable,
                "n_gemma_sae_configs": len(gemma_available_sae_configs),
                "llama_model_accessible": llama_model_accessible,
                "llama_sae_accessible": llama_sae_loadable,
                "primary_model": primary_model,
                "primary_sae_release": primary_sae_release,
            },
            "notes": "; ".join(strategy_notes),
        }
    ],
}

pilots_dir = WORKSPACE / "exp" / "results" / "pilots"
pilots_dir.mkdir(exist_ok=True)
(pilots_dir / f"{TASK_ID}_pilot_summary.json").write_text(json.dumps(pilot_summary, indent=2))

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ACCESS CHECK SUMMARY")
print("="*60)
print(f"Gemma 2B model (google/gemma-2-2b):  {'GRANTED' if gemma_model_accessible else 'GATED (still requires ToS)'}")
print(f"Gemma Scope SAE weights:              {'LOADABLE' if gemma_sae_loadable else 'FAILED'}")
print(f"  → n_available_configs: {len(gemma_available_sae_configs)}")
if gemma_sae_test_result and "W_enc_shape" in gemma_sae_test_result:
    print(f"  → Test SAE shape: W_enc={gemma_sae_test_result['W_enc_shape']}, W_dec={gemma_sae_test_result['W_dec_shape']}")
print(f"Llama-3.1-8B model:                   {'GRANTED' if llama_model_accessible else 'GATED'}")
print(f"Llama Scope SAE weights:              {'LOADABLE' if llama_sae_loadable else 'FAILED'}")
print()
print(f"PRIMARY DECISION:")
print(f"  primary_model:       {primary_model}")
print(f"  primary_sae_release: {primary_sae_release}")
print(f"  fallback_used:       {fallback_used}")
for note in strategy_notes:
    print(f"  → {note}")
print()
print(f"R4 PIPELINE GATING:")
print(f"  r4a_eda (weight-only): {'GO' if access_status['r4a_eda_can_proceed'] else 'BLOCKED'}")
print(f"  r4a direct labels:     {access_status['r4a_direct_labels_approach']}")
print(f"  r4b RAVEL probes:      {access_status['r4b_ravel_probes_approach']}")
print(f"  r4c Llama replication: {access_status['r4c_llama_replication_approach']}")
print()
print(f"PASS CRITERIA: {'MET' if access_status['pass_criteria_met'] else 'NOT MET'}")
print("="*60)

# Mark done
mark_done(
    status="success",
    summary=f"Access check complete. Gemma SAE weights: {'loadable' if gemma_sae_loadable else 'FAILED'}. Primary: {primary_model}.",
    result=access_status,
)

print(f"\n[{datetime.now().isoformat()}] Task {TASK_ID} completed successfully")
