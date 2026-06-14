"""
Pilot: Tier 1 Full — ResNet-18 x CIFAR-100
Task ID: tier1_resnet18_cifar100_full
Mode: PILOT (10 epochs, full train set, seed=42)

Key addition over previous pilot (tier1_resnet18_cifar100):
  - per-class accuracy logging (required for per_class_analysis downstream task)
  - uses task ID tier1_resnet18_cifar100_full to match task_plan.json

Pass criteria:
  - All 6 orderings run 10 epochs without error
  - val_accuracy > 30% for at least 3 orderings
  - per-class accuracy logging enabled
"""
import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

TASK_ID = "tier1_resnet18_cifar100_full"
GPU_ID = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ---- Paths ----
REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT = "augmentation-order"
PROJECT_DIR = Path(f"{REMOTE_BASE}/projects/{PROJECT}")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR = RESULTS_DIR / "pilots"
GPU_PROGRESS_PATH = PROJECT_DIR / "exp" / "gpu_progress.json"
CIFAR100_PATH = f"{REMOTE_BASE}/shared/datasets/cifar100"

FULL_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config ----
NUM_EPOCHS = 10
BATCH_SIZE = 128
SEEDS = [42]  # Pilot: single seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100

# CIFAR-100 normalization stats
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ---- CIFAR-100 class names ----
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# CIFAR-100 superclass taxonomy for per-class analysis
CIFAR100_SUPERCLASSES = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_sized_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}

# ---- Orderings ----
ORDERINGS = {
    "order_0": {"label": "Crop→Flip→CJ", "ops": ["crop", "flip", "cj"]},
    "order_1": {"label": "Crop→CJ→Flip", "ops": ["crop", "cj", "flip"]},
    "order_2": {"label": "Flip→Crop→CJ", "ops": ["flip", "crop", "cj"]},
    "order_3": {"label": "Flip→CJ→Crop", "ops": ["flip", "cj", "crop"]},
    "order_4": {"label": "CJ→Crop→Flip", "ops": ["cj", "crop", "flip"]},
    "order_5": {"label": "CJ→Flip→Crop", "ops": ["cj", "flip", "crop"]},
}


def build_transform(ordering_ops):
    op_map = {
        "crop": transforms.RandomCrop(32, padding=4),
        "flip": transforms.RandomHorizontalFlip(p=0.5),
        "cj": transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    }
    aug_ops = [op_map[op] for op in ordering_ops]
    return transforms.Compose(aug_ops + [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(ordering_ops):
    transform_train = build_transform(ordering_ops)
    transform_val = get_val_transform()
    trainset = torchvision.datasets.CIFAR100(
        root=CIFAR100_PATH, train=True, download=False, transform=transform_train
    )
    valset = torchvision.datasets.CIFAR100(
        root=CIFAR100_PATH, train=False, download=False, transform=transform_val
    )
    return trainset, valset


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
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


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
    return total_loss / total, correct / total


def compute_per_class_accuracy(model, val_loader):
    """Compute per-class and per-superclass accuracy on validation set."""
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

    # Per-class accuracy (by name)
    per_class_acc = {}
    for i in range(NUM_CLASSES):
        name = CIFAR100_CLASSES[i]
        per_class_acc[name] = round(class_correct[i] / max(class_total[i], 1), 4)

    # Per-superclass accuracy
    per_superclass_acc = {}
    for superclass, class_names in CIFAR100_SUPERCLASSES.items():
        sc_correct = sum(class_correct[CIFAR100_CLASSES.index(cn)] for cn in class_names if cn in CIFAR100_CLASSES)
        sc_total = sum(class_total[CIFAR100_CLASSES.index(cn)] for cn in class_names if cn in CIFAR100_CLASSES)
        per_superclass_acc[superclass] = round(sc_correct / max(sc_total, 1), 4)

    return per_class_acc, per_superclass_acc


def run_single(ordering_key, ordering_ops, seed):
    set_seed(seed)
    trainset, valset = get_datasets(ordering_ops)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=256, shuffle=False,
                           num_workers=4, pin_memory=True)

    model = models.resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    per_epoch = []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, valloader, criterion)
        scheduler.step()
        per_epoch.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        print(f"    epoch {epoch}/{NUM_EPOCHS}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # Compute per-class accuracy on final model
    per_class_acc, per_superclass_acc = compute_per_class_accuracy(model, valloader)

    final_val_acc = per_epoch[-1]["val_acc"]
    return {
        "final_val_acc": final_val_acc,
        "per_epoch": per_epoch,
        "per_class_accuracy": per_class_acc,
        "per_superclass_accuracy": per_superclass_acc,
        "elapsed_sec": None,  # filled by caller
    }


def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID={os.getpid()} written to {pid_file}")


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
    print(f"[{TASK_ID}] DONE marker written: status={status}")


def update_gpu_progress(status, actual_min, start_time_str, end_time_str):
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
            "planned_min": 45,
            "actual_min": actual_min,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "config_snapshot": {
                "model": "resnet18",
                "dataset": "cifar100",
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "mode": "pilot",
                "seeds": SEEDS,
                "orderings": 6,
                "gpu": GPU_ID,
                "full_train_set": True,
            }
        }
        GPU_PROGRESS_PATH.write_text(json.dumps(data, indent=2))
        print(f"[{TASK_ID}] gpu_progress.json updated.")
    except Exception as e:
        print(f"Warning: could not update gpu_progress.json: {e}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()

    print(f"[{TASK_ID}] PILOT MODE: ResNet-18 x CIFAR-100 with per-class accuracy")
    print(f"[{TASK_ID}] Device: {DEVICE}")
    print(f"[{TASK_ID}] GPU: {GPU_ID}")
    print(f"[{TASK_ID}] Orderings: {list(ORDERINGS.keys())}")
    print(f"[{TASK_ID}] Seeds (pilot): {SEEDS}")
    print(f"[{TASK_ID}] Epochs: {NUM_EPOCHS}")
    print(f"[{TASK_ID}] Pass criteria: val_accuracy > 30% at epoch 10 for >= 3 orderings + per-class accuracy logged")

    start_time_str = datetime.now().isoformat()
    start_ts = time.time()

    ordering_results = {}
    total_runs = len(ORDERINGS) * len(SEEDS)
    run_count = 0

    for ordering_key, odata in ORDERINGS.items():
        ordering_ops = odata["ops"]
        ordering_label = odata["label"]
        ordering_results[ordering_key] = {
            "label": ordering_label,
            "ops": ordering_ops,
            "per_seed": {}
        }
        for seed in SEEDS:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] ordering={ordering_key} ({ordering_label}) seed={seed}")
            write_progress(run_count - 1, NUM_EPOCHS * total_runs, ordering_key)
            t0 = time.time()
            try:
                result = run_single(ordering_key, ordering_ops, seed)
                result["elapsed_sec"] = round(time.time() - t0, 1)
                ordering_results[ordering_key]["per_seed"][str(seed)] = result
                print(f"  -> final val_acc={result['final_val_acc']:.4f} (elapsed {result['elapsed_sec']:.1f}s)")
                print(f"  -> per_class logged: {len(result['per_class_accuracy'])} classes")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                ordering_results[ordering_key]["per_seed"][str(seed)] = {
                    "error": str(e),
                    "final_val_acc": None,
                }
            write_progress(run_count, total_runs, ordering_key)

    # Aggregate statistics
    all_means = []
    for ordering_key in ordering_results:
        seed_accs = [
            ordering_results[ordering_key]["per_seed"][str(s)].get("final_val_acc")
            for s in SEEDS
        ]
        valid_accs = [a for a in seed_accs if a is not None]
        mean_acc = round(float(np.mean(valid_accs)), 4) if valid_accs else None
        ordering_results[ordering_key]["mean_val_acc"] = mean_acc
        ordering_results[ordering_key]["std_val_acc"] = round(float(np.std(valid_accs)), 4) if valid_accs else None
        if mean_acc is not None:
            all_means.append((ordering_key, mean_acc))
            print(f"  {ordering_key} ({ordering_results[ordering_key]['label']}): mean={mean_acc:.4f}")

    if all_means:
        sorted_means = sorted(all_means, key=lambda x: x[1], reverse=True)
        best_ordering, best_acc = sorted_means[0]
        worst_ordering, worst_acc = sorted_means[-1]
        spread = round(best_acc - worst_acc, 4)
        orderings_above_30 = sum(1 for _, acc in all_means if acc > 0.30)
    else:
        best_ordering = worst_ordering = None
        best_acc = worst_acc = spread = 0.0
        orderings_above_30 = 0

    # Check per-class accuracy is present
    has_per_class = all(
        "per_class_accuracy" in ordering_results[oid]["per_seed"].get(str(SEEDS[0]), {})
        for oid in ordering_results
    )

    # Pass criteria
    pass_criteria_met = (orderings_above_30 >= 3) and has_per_class

    if spread > 0.003:
        recommendation = "GO"
    elif spread > 0.001:
        recommendation = "GO_CAUTIOUS"
    else:
        recommendation = "FLAG_UNDERPOWERED"

    summary_data = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "timestamp": datetime.now().isoformat(),
        "spread": spread,
        "max_val_acc": best_acc,
        "best_ordering": best_ordering,
        "worst_ordering": worst_ordering,
        "best_label": ordering_results[best_ordering]["label"] if best_ordering else None,
        "worst_label": ordering_results[worst_ordering]["label"] if worst_ordering else None,
        "recommendation": recommendation,
        "pass_criteria_met": pass_criteria_met,
        "has_per_class_accuracy": has_per_class,
        "orderings_above_30pct": orderings_above_30,
        "orderings": ordering_results,
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "config": {
            "model": "resnet18",
            "dataset": "cifar100",
            "batch_size": BATCH_SIZE,
            "full_train_set": True,
            "note": "Full train set used (not 100 samples) — 100 samples gives <1% val_acc on CIFAR-100",
        }
    }

    out_path = FULL_DIR / "tier1_resnet18_cifar100_full_pilot.json"
    out_path.write_text(json.dumps(summary_data, indent=2))
    print(f"\n[{TASK_ID}] Results written to {out_path}")

    pilots_path = PILOTS_DIR / "tier1_resnet18_cifar100_full_pilot.json"
    pilots_path.write_text(json.dumps(summary_data, indent=2))

    print(f"[{TASK_ID}] === PILOT SUMMARY ===")
    print(f"  Spread: {spread:.4f} ({spread*100:.2f}%)")
    if best_ordering:
        print(f"  Best: {best_ordering} ({ordering_results[best_ordering]['label']}) = {best_acc:.4f}")
        print(f"  Worst: {worst_ordering} ({ordering_results[worst_ordering]['label']}) = {worst_acc:.4f}")
    print(f"  Orderings above 30%: {orderings_above_30}/6")
    print(f"  Has per-class accuracy: {has_per_class}")
    print(f"  Pass criteria met: {pass_criteria_met}")
    print(f"  Recommendation: {recommendation}")

    end_time_str = datetime.now().isoformat()
    actual_min = round((time.time() - start_ts) / 60, 1)
    update_gpu_progress(
        "success" if pass_criteria_met else "failed",
        actual_min,
        start_time_str,
        end_time_str,
    )
    summary_str = (
        f"spread={spread:.4f} best={best_ordering} worst={worst_ordering} "
        f"above30pct={orderings_above_30}/6 per_class={'YES' if has_per_class else 'NO'} rec={recommendation}"
    )
    mark_done("success" if pass_criteria_met else "failed", summary_str)
    print(f"[{TASK_ID}] Pilot completed in {actual_min} min.")


if __name__ == "__main__":
    main()
