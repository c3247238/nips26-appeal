"""
Pilot Tier 0: 6 Orderings of {RandomCrop, RandomHorizontalFlip, ColorJitter} on CIFAR-10 5k subset
ResNet-18, 10 epochs, seeds [42, 43]
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
TASK_ID = "pilot_tier0"
RESULTS_DIR = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/results/pilots")
GPU_PROGRESS_PATH = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/gpu_progress.json")
CIFAR10_PATH = "/home/qhxie/sibyl_system/shared/datasets/cifar10"
NUM_EPOCHS = 10
BATCH_SIZE = 128
SEEDS = [42, 43]
SUBSET_SIZE = 5000
# When CUDA_VISIBLE_DEVICES=3 is set, GPU 3 is mapped to cuda:0 inside the process
DEVICE = "cuda:0"

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


def build_transform(ordering):
    """Build a transform pipeline from an ordering list."""
    ops = {
        "crop": transforms.RandomCrop(32, padding=4),
        "flip": transforms.RandomHorizontalFlip(p=0.5),
        "cj": transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    }
    aug_transforms = [ops[op] for op in ordering]
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


def get_datasets(ordering, seed):
    """Get CIFAR-10 subsets with given transform ordering."""
    transform_train = build_transform(ordering)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=False,
                                             transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=False,
                                           transform=transform_val)
    # Fixed subset: same indices for all orderings & seeds (reproducible)
    rng = np.random.RandomState(42)  # Always use seed 42 for subset selection
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
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

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
    final_val_acc = per_epoch[-1]["val_acc"]
    return final_val_acc, per_epoch


def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def write_progress(epoch, total_epochs, ordering_key=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total_epochs,
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
            "planned_min": 15,
            "actual_min": actual_min,
            "start_time": start_time,
            "end_time": end_time,
            "config_snapshot": {
                "model": "resnet18",
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "subset_size": SUBSET_SIZE,
                "seeds": SEEDS,
                "orderings": 6,
                "gpu": "RTX PRO 6000",
            }
        }
        GPU_PROGRESS_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning: could not update gpu_progress.json: {e}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()

    device = torch.device(DEVICE)
    print(f"[pilot_tier0] Running on device: {device}")
    print(f"[pilot_tier0] Orderings: {list(ORDERINGS.keys())}")
    print(f"[pilot_tier0] Seeds: {SEEDS}")
    print(f"[pilot_tier0] Subset size: {SUBSET_SIZE}, Epochs: {NUM_EPOCHS}")

    start_time = datetime.now().isoformat()
    start_ts = time.time()

    results = {}
    total_runs = len(ORDERINGS) * len(SEEDS)
    run_count = 0

    for ordering_key, ordering_ops in ORDERINGS.items():
        results[ordering_key] = {
            "label": ORDERING_LABELS[ordering_key],
            "ops": ordering_ops,
            "per_seed": {}
        }
        for seed in SEEDS:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] ordering={ordering_key} ({ORDERING_LABELS[ordering_key]}) seed={seed}")
            write_progress(0, NUM_EPOCHS * total_runs, ordering_key)
            t0 = time.time()
            final_acc, per_epoch = run_single(ordering_key, ordering_ops, seed, device)
            elapsed = time.time() - t0
            print(f"  -> final val_acc={final_acc:.4f} (elapsed {elapsed:.1f}s)")
            results[ordering_key]["per_seed"][str(seed)] = {
                "final_val_acc": final_acc,
                "per_epoch": per_epoch,
                "elapsed_sec": round(elapsed, 1),
            }
            write_progress(run_count, total_runs, ordering_key)

    # Aggregate
    for ordering_key in results:
        accs = [results[ordering_key]["per_seed"][str(s)]["final_val_acc"] for s in SEEDS]
        results[ordering_key]["mean_val_acc"] = round(float(np.mean(accs)), 4)
        results[ordering_key]["std_val_acc"] = round(float(np.std(accs)), 4)
        print(f"  {ordering_key} ({ORDERING_LABELS[ordering_key]}): mean={results[ordering_key]['mean_val_acc']:.4f} std={results[ordering_key]['std_val_acc']:.4f}")

    all_means = [results[k]["mean_val_acc"] for k in results]
    spread = round(max(all_means) - min(all_means), 4)
    best_ordering = max(results, key=lambda k: results[k]["mean_val_acc"])
    worst_ordering = min(results, key=lambda k: results[k]["mean_val_acc"])
    max_val_acc = max(all_means)
    pass_criteria_met = max_val_acc > 0.55

    if spread > 0.003:
        recommendation = "HIGH_CONFIDENCE"
    elif spread > 0.001:
        recommendation = "CAUTIOUS"
    else:
        recommendation = "FLAG_UNDERPOWERED"

    summary = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "spread": spread,
        "max_val_acc": max_val_acc,
        "best_ordering": best_ordering,
        "worst_ordering": worst_ordering,
        "best_label": ORDERING_LABELS[best_ordering],
        "worst_label": ORDERING_LABELS[worst_ordering],
        "recommendation": recommendation,
        "pass_criteria_met": pass_criteria_met,
        "orderings": results,
        "seeds": SEEDS,
        "subset_size": SUBSET_SIZE,
        "epochs": NUM_EPOCHS,
    }

    out_path = RESULTS_DIR / "tier0_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[pilot_tier0] Results written to {out_path}")
    print(f"[pilot_tier0] Spread: {spread:.4f} ({spread*100:.2f}%)")
    print(f"[pilot_tier0] Best: {best_ordering} ({ORDERING_LABELS[best_ordering]}) = {results[best_ordering]['mean_val_acc']:.4f}")
    print(f"[pilot_tier0] Worst: {worst_ordering} ({ORDERING_LABELS[worst_ordering]}) = {results[worst_ordering]['mean_val_acc']:.4f}")
    print(f"[pilot_tier0] Recommendation: {recommendation}")
    print(f"[pilot_tier0] Pass criteria met: {pass_criteria_met}")

    end_time = datetime.now().isoformat()
    actual_min = round((time.time() - start_ts) / 60, 1)
    update_gpu_progress("success", actual_min, start_time, end_time)
    mark_done("success", f"spread={spread:.4f} best={best_ordering} worst={worst_ordering} rec={recommendation}")
    print(f"[pilot_tier0] Done in {actual_min} min.")


if __name__ == "__main__":
    main()
