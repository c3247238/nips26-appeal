"""
Pilot: Tier 3 — Magnitude Interaction on CIFAR-100 (ResNet-18)
Using best ordering (Flip→Crop→CJ, order_2) and worst ordering (CJ→Flip→Crop, order_5)
3 magnitude levels: M=5 (low), M=9 (standard), M=14 (high)
1 seed, 10 epochs (pilot)
Primary metric: accuracy spread (best - worst ordering) at each magnitude level
Pass criteria: Both orderings run at all 3 magnitude levels for 10 epochs;
               no NaN at M=14; pilot spread at M=14 exceeds spread at M=5
"""
import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ---- Config ----
TASK_ID = "tier3_magnitude"
RESULTS_DIR = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/results/full")
PILOTS_DIR = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/results/pilots")
GPU_PROGRESS_PATH = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/gpu_progress.json")
CIFAR100_PATH = "/home/qhxie/sibyl_system/shared/datasets/cifar100"

# Pilot mode settings
NUM_EPOCHS = 10
BATCH_SIZE = 128
SEEDS = [42]
DEVICE = "cuda:0"  # When CUDA_VISIBLE_DEVICES=0 is set, GPU 0 is mapped to cuda:0

# CIFAR-100 normalization stats
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Best and worst orderings from tier1_analysis
TARGET_ORDERINGS = {
    "order_2": {
        "label": "Flip→Crop→CJ",
        "ops": ["flip", "crop", "cj"],
        "role": "best",
    },
    "order_5": {
        "label": "CJ→Flip→Crop",
        "ops": ["cj", "flip", "crop"],
        "role": "worst",
    },
}

# Magnitude configurations
# M=5: low magnitude, M=9: standard (matches tier1), M=14: high magnitude
MAGNITUDE_CONFIGS = {
    "M5": {
        "level": 5,
        "label": "Low",
        "crop_padding": 1,
        "flip_prob": 0.5,
        "cj_brightness": 0.15,
        "cj_contrast": 0.15,
        "cj_saturation": 0.15,
        "cj_hue": 0.0,
        "rotation": 5,
    },
    "M9": {
        "level": 9,
        "label": "Standard",
        "crop_padding": 4,
        "flip_prob": 0.5,
        "cj_brightness": 0.4,
        "cj_contrast": 0.4,
        "cj_saturation": 0.4,
        "cj_hue": 0.0,
        "rotation": 0,
    },
    "M14": {
        "level": 14,
        "label": "High",
        "crop_padding": 8,
        "flip_prob": 0.5,
        "cj_brightness": 0.8,
        "cj_contrast": 0.8,
        "cj_saturation": 0.8,
        "cj_hue": 0.1,
        "rotation": 30,
    },
}


def build_transform_with_magnitude(ordering_ops, mag_config):
    """Build a transform pipeline from an ordering list with given magnitude."""
    ops_map = {
        "crop": transforms.RandomCrop(32, padding=mag_config["crop_padding"]),
        "flip": transforms.RandomHorizontalFlip(p=mag_config["flip_prob"]),
        "cj": transforms.ColorJitter(
            brightness=mag_config["cj_brightness"],
            contrast=mag_config["cj_contrast"],
            saturation=mag_config["cj_saturation"],
            hue=mag_config["cj_hue"],
        ),
    }
    aug_transforms = [ops_map[op] for op in ordering_ops]
    # Add rotation for high magnitude if specified
    if mag_config.get("rotation", 0) > 0:
        aug_transforms.append(transforms.RandomRotation(mag_config["rotation"]))
    return transforms.Compose(
        aug_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(ordering_ops, mag_config):
    """Get CIFAR-100 train/val datasets with given transform ordering and magnitude."""
    transform_train = build_transform_with_magnitude(ordering_ops, mag_config)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root=CIFAR100_PATH, train=True, download=False, transform=transform_train
    )
    valset = torchvision.datasets.CIFAR100(
        root=CIFAR100_PATH, train=False, download=False, transform=transform_val
    )
    return trainset, valset


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
    return total_loss / total, correct / total


def run_single(ordering_key, ordering_ops, mag_config, seed, device):
    set_seed(seed)
    trainset, valset = get_datasets(ordering_ops, mag_config)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=256, shuffle=False,
                           num_workers=4, pin_memory=True)

    model = models.resnet18(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    per_epoch = []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, valloader, criterion, device)
        scheduler.step()
        # Check for NaN
        if np.isnan(val_acc) or np.isnan(train_loss):
            print(f"    WARNING: NaN detected at epoch {epoch}!")
            return float('nan'), per_epoch
        per_epoch.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 4),
        })
        print(f"    epoch {epoch}/{NUM_EPOCHS}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    final_val_acc = per_epoch[-1]["val_acc"]
    return final_val_acc, per_epoch


def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def write_progress(current_run, total_runs, ordering_key=None, mag_key=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    progress = {
        "task_id": TASK_ID,
        "epoch": current_run,
        "total_epochs": total_runs,
        "ordering": ordering_key,
        "magnitude": mag_key,
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress))


def mark_done(status="success", summary=""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def update_gpu_progress(status, actual_min, start_time, end_time):
    try:
        if GPU_PROGRESS_PATH.exists():
            data = json.loads(GPU_PROGRESS_PATH.read_text())
        else:
            data = {"completed": [], "failed": [], "running": {}, "timings": {}}
        if status == "success":
            if TASK_ID not in data["completed"]:
                data["completed"].append(TASK_ID)
        else:
            if TASK_ID not in data["failed"]:
                data["failed"].append(TASK_ID)
        data["running"].pop(TASK_ID, None)
        data["timings"][TASK_ID] = {
            "planned_min": 50,
            "actual_min": actual_min,
            "start_time": start_time,
            "end_time": end_time,
            "config_snapshot": {
                "model": "resnet18",
                "dataset": "cifar100",
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "mode": "pilot",
                "seeds": SEEDS,
                "orderings": 2,
                "magnitude_levels": 3,
            }
        }
        GPU_PROGRESS_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning: could not update gpu_progress.json: {e}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PILOTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()

    device = torch.device(DEVICE)
    print(f"[{TASK_ID}] PILOT MODE: Tier 3 Magnitude Interaction — ResNet-18 x CIFAR-100")
    print(f"[{TASK_ID}] Device: {device}")
    print(f"[{TASK_ID}] Orderings: best (Flip→Crop→CJ) and worst (CJ→Flip→Crop)")
    print(f"[{TASK_ID}] Magnitude levels: M5 (low), M9 (standard), M14 (high)")
    print(f"[{TASK_ID}] Seeds: {SEEDS}, Epochs: {NUM_EPOCHS}")
    print(f"[{TASK_ID}] Pass criteria: no NaN at M14; spread[M14] > spread[M5]")

    start_time = datetime.now().isoformat()
    start_ts = time.time()

    # Results structure: results[mag_key][ordering_key] = {...}
    results = {}
    total_runs = len(MAGNITUDE_CONFIGS) * len(TARGET_ORDERINGS) * len(SEEDS)
    run_count = 0
    any_nan = False

    for mag_key, mag_config in MAGNITUDE_CONFIGS.items():
        results[mag_key] = {
            "level": mag_config["level"],
            "label": mag_config["label"],
            "config": mag_config,
            "orderings": {},
        }
        for ordering_key, ordering_info in TARGET_ORDERINGS.items():
            results[mag_key]["orderings"][ordering_key] = {
                "label": ordering_info["label"],
                "role": ordering_info["role"],
                "ops": ordering_info["ops"],
                "per_seed": {},
            }
            for seed in SEEDS:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] mag={mag_key}({mag_config['label']}) "
                      f"ordering={ordering_key}({ordering_info['label']}) seed={seed}")
                write_progress(run_count - 1, total_runs, ordering_key, mag_key)
                t0 = time.time()
                final_acc, per_epoch = run_single(
                    ordering_key, ordering_info["ops"], mag_config, seed, device
                )
                elapsed = time.time() - t0
                is_nan = bool(np.isnan(final_acc))
                if is_nan:
                    any_nan = True
                    print(f"  -> NaN detected! (elapsed {elapsed:.1f}s)")
                else:
                    print(f"  -> final val_acc={final_acc:.4f} (elapsed {elapsed:.1f}s)")
                results[mag_key]["orderings"][ordering_key]["per_seed"][str(seed)] = {
                    "final_val_acc": None if is_nan else float(final_acc),
                    "is_nan": is_nan,
                    "per_epoch": per_epoch,
                    "elapsed_sec": round(elapsed, 1),
                }
                write_progress(run_count, total_runs, ordering_key, mag_key)

    # Aggregate per magnitude
    spread_by_mag = {}
    for mag_key in results:
        ordering_means = {}
        for ordering_key in results[mag_key]["orderings"]:
            per_seed = results[mag_key]["orderings"][ordering_key]["per_seed"]
            accs = [v["final_val_acc"] for v in per_seed.values() if v["final_val_acc"] is not None]
            if accs:
                mean_acc = round(float(np.mean(accs)), 4)
                std_acc = round(float(np.std(accs)), 4)
            else:
                mean_acc = None
                std_acc = None
            results[mag_key]["orderings"][ordering_key]["mean_val_acc"] = mean_acc
            results[mag_key]["orderings"][ordering_key]["std_val_acc"] = std_acc
            ordering_means[ordering_key] = mean_acc

        valid_means = [v for v in ordering_means.values() if v is not None]
        if len(valid_means) >= 2:
            spread = round(max(valid_means) - min(valid_means), 4)
        else:
            spread = None
        results[mag_key]["spread"] = spread
        spread_by_mag[mag_key] = spread
        print(f"\n[{mag_key}] Orderings:")
        for ok, ov in results[mag_key]["orderings"].items():
            print(f"  {ok} ({ov['label']}): mean={ov['mean_val_acc']}")
        print(f"[{mag_key}] Spread: {spread}")

    # Pass criteria check
    nan_at_m14 = bool(any(
        results["M14"]["orderings"][ok]["per_seed"][str(SEEDS[0])].get("is_nan", False)
        for ok in results["M14"]["orderings"]
    ))
    spread_m14 = spread_by_mag.get("M14")
    spread_m5 = spread_by_mag.get("M5")
    spread_increases = bool(
        spread_m14 is not None and spread_m5 is not None and spread_m14 > spread_m5
    )
    pass_criteria_met = bool((not nan_at_m14) and spread_increases)

    print(f"\n[{TASK_ID}] Summary:")
    print(f"  NaN at M14: {nan_at_m14}")
    print(f"  Spread M5={spread_m5}, M9={spread_by_mag.get('M9')}, M14={spread_m14}")
    print(f"  Spread increases M5→M14: {spread_increases}")
    print(f"  Pass criteria met: {pass_criteria_met}")

    summary = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "target_orderings": {k: v["label"] for k, v in TARGET_ORDERINGS.items()},
        "magnitude_levels": {k: v["level"] for k, v in MAGNITUDE_CONFIGS.items()},
        "spread_by_magnitude": spread_by_mag,
        "any_nan": any_nan,
        "nan_at_m14": nan_at_m14,
        "spread_increases_with_magnitude": spread_increases,
        "pass_criteria_met": pass_criteria_met,
        "results": results,
    }

    # Write to full results dir (this is where tier4_correlation_analysis will read from)
    out_path = RESULTS_DIR / "tier3_results.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path.write_text(json.dumps(summary, indent=2, cls=NumpyEncoder))
    print(f"\n[{TASK_ID}] Results written to {out_path}")

    end_time = datetime.now().isoformat()
    actual_min = round((time.time() - start_ts) / 60, 1)
    update_gpu_progress(
        "success" if pass_criteria_met else "failed",
        actual_min, start_time, end_time
    )
    mark_done(
        "success" if pass_criteria_met else "failed",
        f"spread_m5={spread_m5} spread_m9={spread_by_mag.get('M9')} spread_m14={spread_m14} "
        f"nan_at_m14={nan_at_m14} spread_increases={spread_increases}"
    )
    print(f"[{TASK_ID}] Done in {actual_min} min.")


if __name__ == "__main__":
    main()
