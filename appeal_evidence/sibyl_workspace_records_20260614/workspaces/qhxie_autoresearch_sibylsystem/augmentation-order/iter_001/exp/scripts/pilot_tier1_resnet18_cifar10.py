"""
Pilot for Tier 1: Full Factorial — ResNet-18 x CIFAR-10
PILOT MODE: 100-sample subset, 10 epochs, seed 42 only.
Pass criteria:
  - All 6 orderings run 10 epochs without error
  - val_accuracy variance across orderings > 0.0% (any spread)
  - per-epoch logging works
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
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ---- Config ----
TASK_ID = "tier1_resnet18_cifar10"
RESULTS_DIR = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/results")
PILOTS_DIR = RESULTS_DIR / "pilots"
GPU_PROGRESS_PATH = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/gpu_progress.json")
CIFAR10_PATH = "/home/qhxie/sibyl_system/shared/datasets/cifar10"

# PILOT settings
NUM_EPOCHS = 10
BATCH_SIZE = 128
SEEDS = [42]
SUBSET_SIZE = 100
DEVICE = "cuda:0"  # Set CUDA_VISIBLE_DEVICES=0 externally

# Full run settings (for reference, not used in pilot)
FULL_EPOCHS = 200
FULL_SEEDS = [42, 43, 44, 45, 46]
PLANNED_MIN = 45

# ---- Orderings ----
ORDERINGS = {
    "order_0": ["crop", "flip", "cj"],   # Crop -> Flip -> ColorJitter (conventional)
    "order_1": ["crop", "cj", "flip"],
    "order_2": ["flip", "crop", "cj"],
    "order_3": ["flip", "cj", "crop"],
    "order_4": ["cj", "crop", "flip"],
    "order_5": ["cj", "flip", "crop"],   # ColorJitter-first (reversibility-first)
}

ORDERING_LABELS = {
    "order_0": "Crop→Flip→CJ",
    "order_1": "Crop→CJ→Flip",
    "order_2": "Flip→Crop→CJ",
    "order_3": "Flip→CJ→Crop",
    "order_4": "CJ→Crop→Flip",
    "order_5": "CJ→Flip→Crop",
}


def build_transform(ordering_ops):
    """Build a transform pipeline from an ordering list."""
    ops = {
        "crop": transforms.RandomCrop(32, padding=4),
        "flip": transforms.RandomHorizontalFlip(p=0.5),
        "cj": transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    }
    aug_transforms = [ops[op] for op in ordering_ops]
    return transforms.Compose(
        aug_transforms + [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(ordering_ops, seed):
    """Get CIFAR-10 subsets with given transform ordering."""
    transform_train = build_transform(ordering_ops)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=False,
                                             transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=False,
                                           transform=transform_val)
    # Fixed subset: same indices across all orderings (reproducible, fair comparison)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(trainset), SUBSET_SIZE, replace=False).tolist()
    subset = Subset(trainset, indices)
    return subset, valset


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


def run_single(ordering_key, ordering_ops, seed, device):
    set_seed(seed)
    trainset, valset = get_datasets(ordering_ops, seed)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valloader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    per_epoch = []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, valloader, criterion, device)
        scheduler.step()
        per_epoch.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        print(f"    epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    final_val_acc = per_epoch[-1]["val_acc"]
    return final_val_acc, per_epoch


def write_pid():
    PILOTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def write_progress(run_count, total_runs, ordering_key=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": run_count,
        "total_epochs": total_runs,
        "ordering": ordering_key,
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress))


def mark_done(status="success", summary=""):
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
            "planned_min": PLANNED_MIN,
            "actual_min": actual_min,
            "start_time": start_time,
            "end_time": end_time,
            "config_snapshot": {
                "model": "resnet18",
                "dataset": "cifar10",
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "subset_size": SUBSET_SIZE,
                "seeds": SEEDS,
                "orderings": 6,
                "mode": "pilot",
                "gpu_model": "RTX PRO 6000",
            }
        }
        GPU_PROGRESS_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning: could not update gpu_progress.json: {e}")


def main():
    PILOTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()

    device = torch.device(DEVICE)
    print(f"[{TASK_ID}] PILOT MODE: {SUBSET_SIZE} samples, {NUM_EPOCHS} epochs, seeds={SEEDS}")
    print(f"[{TASK_ID}] Running on device: {device}")
    print(f"[{TASK_ID}] PyTorch version: {torch.__version__}")
    print(f"[{TASK_ID}] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[{TASK_ID}] GPU: {torch.cuda.get_device_name(0)}")

    start_time = datetime.now().isoformat()
    start_ts = time.time()

    results = {}
    total_runs = len(ORDERINGS) * len(SEEDS)
    run_count = 0
    errors = []

    for ordering_key, ordering_ops in ORDERINGS.items():
        results[ordering_key] = {
            "label": ORDERING_LABELS[ordering_key],
            "ops": ordering_ops,
            "per_seed": {}
        }
        for seed in SEEDS:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] ordering={ordering_key} ({ORDERING_LABELS[ordering_key]}) seed={seed}")
            write_progress(run_count - 1, total_runs, ordering_key)
            t0 = time.time()
            try:
                final_acc, per_epoch = run_single(ordering_key, ordering_ops, seed, device)
                elapsed = time.time() - t0
                print(f"  -> final val_acc={final_acc:.4f} (elapsed {elapsed:.1f}s)")
                results[ordering_key]["per_seed"][str(seed)] = {
                    "final_val_acc": final_acc,
                    "per_epoch": per_epoch,
                    "elapsed_sec": round(elapsed, 1),
                }
            except Exception as e:
                elapsed = time.time() - t0
                error_msg = str(e)
                print(f"  ERROR: {error_msg}")
                errors.append({"ordering": ordering_key, "seed": seed, "error": error_msg})
                results[ordering_key]["per_seed"][str(seed)] = {
                    "final_val_acc": 0.0,
                    "per_epoch": [],
                    "error": error_msg,
                    "elapsed_sec": round(elapsed, 1),
                }
            write_progress(run_count, total_runs, ordering_key)

    # Aggregate
    for ordering_key in results:
        seed_results = results[ordering_key]["per_seed"]
        accs = [seed_results[str(s)]["final_val_acc"] for s in SEEDS if str(s) in seed_results]
        if accs:
            results[ordering_key]["mean_val_acc"] = round(float(np.mean(accs)), 4)
            results[ordering_key]["std_val_acc"] = round(float(np.std(accs)), 4)
        else:
            results[ordering_key]["mean_val_acc"] = 0.0
            results[ordering_key]["std_val_acc"] = 0.0
        print(f"  {ordering_key} ({ORDERING_LABELS[ordering_key]}): "
              f"mean={results[ordering_key]['mean_val_acc']:.4f} "
              f"std={results[ordering_key]['std_val_acc']:.4f}")

    # Compute spread and pass criteria
    all_means = [results[k]["mean_val_acc"] for k in results]
    valid_means = [v for v in all_means if v > 0.0]
    spread = round(max(valid_means) - min(valid_means), 4) if len(valid_means) >= 2 else 0.0
    best_ordering = max(results, key=lambda k: results[k]["mean_val_acc"])
    worst_ordering = min(results, key=lambda k: results[k].get("mean_val_acc", 0.0))
    max_val_acc = max(valid_means) if valid_means else 0.0

    # Pass criteria:
    # 1. All 6 orderings ran without error
    # 2. Any spread in val_accuracy across orderings (spread > 0.0%)
    # 3. Per-epoch logging works (per_epoch list is non-empty)
    orderings_without_error = sum(
        1 for ok in ORDERINGS
        if not results[ok]["per_seed"].get(str(SEEDS[0]), {}).get("error")
    )
    any_epoch_data = any(
        len(results[ok]["per_seed"].get(str(SEEDS[0]), {}).get("per_epoch", [])) > 0
        for ok in ORDERINGS
    )
    pass_criteria_met = (
        orderings_without_error == 6
        and spread > 0.0
        and any_epoch_data
        and len(errors) == 0
    )

    summary = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "timestamp": datetime.now().isoformat(),
        "spread": spread,
        "max_val_acc": max_val_acc,
        "best_ordering": best_ordering,
        "worst_ordering": worst_ordering,
        "best_label": ORDERING_LABELS[best_ordering],
        "worst_label": ORDERING_LABELS[worst_ordering],
        "pass_criteria_met": pass_criteria_met,
        "orderings_without_error": orderings_without_error,
        "errors": errors,
        "orderings": results,
        "seeds": SEEDS,
        "subset_size": SUBSET_SIZE,
        "epochs": NUM_EPOCHS,
    }

    out_path = PILOTS_DIR / f"{TASK_ID}_pilot.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{TASK_ID}] Pilot results written to {out_path}")
    print(f"[{TASK_ID}] Spread: {spread:.4f} ({spread*100:.2f}%)")
    print(f"[{TASK_ID}] Best: {best_ordering} ({ORDERING_LABELS[best_ordering]}) = {results[best_ordering]['mean_val_acc']:.4f}")
    print(f"[{TASK_ID}] Worst: {worst_ordering} ({ORDERING_LABELS[worst_ordering]}) = {results[worst_ordering]['mean_val_acc']:.4f}")
    print(f"[{TASK_ID}] Pass criteria met: {pass_criteria_met}")
    print(f"[{TASK_ID}] Errors: {len(errors)}")

    end_time = datetime.now().isoformat()
    actual_min = round((time.time() - start_ts) / 60, 1)

    final_status = "success" if pass_criteria_met else "failed"
    update_gpu_progress(final_status, actual_min, start_time, end_time)
    mark_done(final_status, f"pilot: spread={spread:.4f} best={best_ordering} errors={len(errors)} pass={pass_criteria_met}")
    print(f"[{TASK_ID}] Done in {actual_min} min. Status: {final_status}")


if __name__ == "__main__":
    main()
