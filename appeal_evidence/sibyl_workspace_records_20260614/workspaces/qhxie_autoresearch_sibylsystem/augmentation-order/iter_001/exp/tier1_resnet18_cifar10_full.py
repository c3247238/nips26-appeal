"""
Tier 1 Full: ResNet-18 x CIFAR-10
FULL SCALE: 200 epochs, 5 seeds [42..46], full 50k training set.
6 permutations of {RandomCrop, RandomHorizontalFlip, ColorJitter}.
SGD + cosine annealing, lr=0.1, momentum=0.9, wd=5e-4.
Saves per-epoch val_accuracy, final accuracy, and model checkpoints.

Orderings:
  order_0: Crop->Flip->CJ  (conventional torchvision default)
  order_1: Crop->CJ->Flip
  order_2: Flip->Crop->CJ
  order_3: Flip->CJ->Crop
  order_4: CJ->Crop->Flip
  order_5: CJ->Flip->Crop  (reversibility-sorted)

CRITICAL: Model checkpoints saved for Tier 4 feature extraction.
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
TASK_ID = "tier1_resnet18_cifar10_full"
REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT = "augmentation-order"
RESULTS_DIR = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/results")
FULL_DIR = RESULTS_DIR / "full"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints" / TASK_ID
GPU_PROGRESS_PATH = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/gpu_progress.json")
EXPERIMENT_STATE_PATH = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/experiment_state.json")
CIFAR10_PATH = f"{REMOTE_BASE}/shared/datasets/cifar10"

# FULL settings
NUM_EPOCHS = 200
BATCH_SIZE = 256  # Will be probed; start high for RTX PRO 6000 (97GB VRAM)
SEEDS = [42, 43, 44, 45, 46]
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=3 set externally, maps to cuda:0
PLANNED_MIN = 45

# ---- Orderings ----
ORDERINGS = {
    "order_0": ["crop", "flip", "cj"],   # Crop -> Flip -> ColorJitter (conventional)
    "order_1": ["crop", "cj", "flip"],
    "order_2": ["flip", "crop", "cj"],
    "order_3": ["flip", "cj", "crop"],
    "order_4": ["cj", "crop", "flip"],
    "order_5": ["cj", "flip", "crop"],   # reversibility-sorted
}

ORDERING_LABELS = {
    "order_0": "Crop->Flip->CJ",
    "order_1": "Crop->CJ->Flip",
    "order_2": "Flip->Crop->CJ",
    "order_3": "Flip->CJ->Crop",
    "order_4": "CJ->Crop->Flip",
    "order_5": "CJ->Flip->Crop",
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_transform(ordering_ops):
    """Build augmentation pipeline from an ordering list."""
    ops_map = {
        "crop": transforms.RandomCrop(32, padding=4),
        "flip": transforms.RandomHorizontalFlip(p=0.5),
        "cj": transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    }
    aug_transforms = [ops_map[op] for op in ordering_ops]
    return transforms.Compose(
        aug_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(ordering_ops):
    """Get full CIFAR-10 train/val datasets with given transform ordering."""
    transform_train = build_transform(ordering_ops)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=True,
                                             download=False, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=False,
                                           download=False, transform=transform_val)
    return trainset, valset


def probe_batch_size(device, start=512, min_bs=64):
    """Binary search for max stable batch size on this GPU."""
    import gc
    # Create a small dummy model for probing
    model = models.resnet18(num_classes=10).to(device)
    model.eval()
    high, best = start, min_bs
    lo = min_bs
    while lo <= high:
        mid = (lo + high) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            dummy = torch.randn(mid, 3, 32, 32).to(device)
            with torch.no_grad():
                _ = model(dummy)
            best = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache()
            gc.collect()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return best


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
    return total_loss / total, correct / total


def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID {os.getpid()} written to {pid_file}")


def write_progress(run_idx, total_runs, epoch=0, total_epochs=NUM_EPOCHS, ordering_key=None, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "run_idx": run_idx,
        "total_runs": total_runs,
        "ordering": ordering_key,
        "loss": loss,
        "metric": metric or {},
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
    print(f"[{TASK_ID}] DONE marker written: {marker}")


def update_experiment_state(status):
    """Update experiment_state.json."""
    try:
        if EXPERIMENT_STATE_PATH.exists():
            data = json.loads(EXPERIMENT_STATE_PATH.read_text())
        else:
            data = {"schema_version": 1, "tasks": {}}
        if TASK_ID not in data["tasks"]:
            data["tasks"][TASK_ID] = {}
        data["tasks"][TASK_ID]["status"] = status
        if status == "completed":
            data["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
        EXPERIMENT_STATE_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning: could not update experiment_state.json: {e}")


def update_gpu_progress(status, actual_min, start_time, end_time, config_snapshot):
    """Update gpu_progress.json with timing and config."""
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
        data.get("running", {}).pop(TASK_ID, None)
        data["timings"][TASK_ID] = {
            "planned_min": PLANNED_MIN,
            "actual_min": round(actual_min, 1),
            "start_time": start_time,
            "end_time": end_time,
            "config_snapshot": config_snapshot,
        }
        GPU_PROGRESS_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning: could not update gpu_progress.json: {e}")


def load_checkpoint_state(ordering_key):
    """Check if a partial checkpoint exists for resumption."""
    checkpoint_dir = CHECKPOINTS_DIR / ordering_key
    state_file = checkpoint_dir / "training_state.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            pass
    return None


def run_single(ordering_key, ordering_ops, seed, device, batch_size):
    """Run a single (ordering, seed) experiment and save checkpoint."""
    set_seed(seed)
    trainset, valset = get_datasets(ordering_ops)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valloader = DataLoader(valset, batch_size=512, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = models.resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Checkpoint directory per ordering
    ckpt_dir = CHECKPOINTS_DIR / ordering_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    per_epoch = []
    start_epoch = 1

    # Check for partial resume
    resume_ckpt = ckpt_dir / f"seed_{seed}_latest.pt"
    if resume_ckpt.exists():
        try:
            ckpt = torch.load(resume_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            per_epoch = ckpt.get("per_epoch", [])
            print(f"    [Resume] Loaded checkpoint at epoch {ckpt['epoch']} for {ordering_key}/seed={seed}")
        except Exception as e:
            print(f"    [Resume] Warning: failed to load checkpoint: {e}. Starting fresh.")
            start_epoch = 1
            per_epoch = []

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
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

        # Save latest checkpoint every 10 epochs for resume support
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "per_epoch": per_epoch,
                "ordering_key": ordering_key,
                "seed": seed,
            }, resume_ckpt)

        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            print(f"    epoch {epoch:3d}/{NUM_EPOCHS}: train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    final_val_acc = per_epoch[-1]["val_acc"]

    # Save final checkpoint (named by epoch for Tier 4 feature extraction)
    final_ckpt = ckpt_dir / f"seed_{seed}_final.pt"
    torch.save({
        "epoch": NUM_EPOCHS,
        "model": model.state_dict(),
        "ordering_key": ordering_key,
        "ordering_ops": ordering_ops,
        "seed": seed,
        "final_val_acc": final_val_acc,
    }, final_ckpt)

    # Cleanup latest checkpoint to save space
    if resume_ckpt.exists():
        resume_ckpt.unlink()

    return final_val_acc, per_epoch


def main():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"[{TASK_ID}] FULL SCALE: {NUM_EPOCHS} epochs, seeds={SEEDS}")
    print(f"[{TASK_ID}] Device: {device}")
    print(f"[{TASK_ID}] PyTorch: {torch.__version__}")
    print(f"[{TASK_ID}] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[{TASK_ID}] GPU: {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[{TASK_ID}] VRAM total: {vram_total:.1f} GB")

    write_pid()

    # Probe batch size
    print(f"[{TASK_ID}] Probing max batch size...")
    batch_size = probe_batch_size(device, start=1024, min_bs=64)
    # For training, use a slightly conservative batch to ensure stability
    batch_size = min(batch_size, 512)
    print(f"[{TASK_ID}] Using batch_size={batch_size}")

    # Save GPU profile
    gpu_profile = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6) if torch.cuda.is_available() else 0,
        "max_batch_size": batch_size,
        "utilization_pct": round(batch_size / 1024 * 100, 1),
    }
    (RESULTS_DIR / f"{TASK_ID}_gpu_profile.json").write_text(json.dumps(gpu_profile, indent=2))

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
            write_progress(run_count - 1, total_runs, epoch=0, ordering_key=ordering_key)
            t0 = time.time()
            try:
                final_acc, per_epoch = run_single(ordering_key, ordering_ops, seed, device, batch_size)
                elapsed = time.time() - t0
                print(f"  -> final val_acc={final_acc:.4f} (elapsed {elapsed/60:.1f} min)")
                results[ordering_key]["per_seed"][str(seed)] = {
                    "final_val_acc": final_acc,
                    "per_epoch": per_epoch,
                    "elapsed_sec": round(elapsed, 1),
                }
            except Exception as e:
                import traceback
                elapsed = time.time() - t0
                error_msg = str(e)
                tb = traceback.format_exc()
                print(f"  ERROR: {error_msg}")
                print(tb)
                errors.append({"ordering": ordering_key, "seed": seed, "error": error_msg})
                results[ordering_key]["per_seed"][str(seed)] = {
                    "final_val_acc": 0.0,
                    "per_epoch": [],
                    "error": error_msg,
                    "elapsed_sec": round(elapsed, 1),
                }
            write_progress(run_count, total_runs,
                           epoch=NUM_EPOCHS, ordering_key=ordering_key,
                           metric={"val_acc": results[ordering_key]["per_seed"][str(seed)]["final_val_acc"]})

    # Aggregate per-ordering mean/std
    for ordering_key in results:
        seed_data = results[ordering_key]["per_seed"]
        accs = [seed_data[str(s)]["final_val_acc"]
                for s in SEEDS if str(s) in seed_data and "error" not in seed_data[str(s)]]
        if accs:
            results[ordering_key]["mean_val_acc"] = round(float(np.mean(accs)), 4)
            results[ordering_key]["std_val_acc"] = round(float(np.std(accs)), 4)
            results[ordering_key]["n_seeds_ok"] = len(accs)
        else:
            results[ordering_key]["mean_val_acc"] = 0.0
            results[ordering_key]["std_val_acc"] = 0.0
            results[ordering_key]["n_seeds_ok"] = 0
        print(f"  {ordering_key} ({ORDERING_LABELS[ordering_key]}): "
              f"mean={results[ordering_key]['mean_val_acc']:.4f} "
              f"std={results[ordering_key]['std_val_acc']:.4f} "
              f"(n={results[ordering_key]['n_seeds_ok']})")

    # Compute spread across orderings
    all_means = [results[k]["mean_val_acc"] for k in results]
    valid_means = [v for v in all_means if v > 0.0]
    spread = round(max(valid_means) - min(valid_means), 4) if len(valid_means) >= 2 else 0.0
    best_ordering = max(results, key=lambda k: results[k]["mean_val_acc"])
    worst_ordering = min(results, key=lambda k: results[k]["mean_val_acc"])
    best_acc = results[best_ordering]["mean_val_acc"]
    worst_acc = results[worst_ordering]["mean_val_acc"]

    # Flip-first vs Crop-first comparison (H4)
    flip_first_orders = ["order_2", "order_3"]  # Flip->X->Y
    crop_first_orders = ["order_0", "order_1"]  # Crop->X->Y
    flip_first_mean = np.mean([results[k]["mean_val_acc"] for k in flip_first_orders])
    crop_first_mean = np.mean([results[k]["mean_val_acc"] for k in crop_first_orders])
    h4_flip_wins = bool(flip_first_mean > crop_first_mean)

    # Pass criteria for full scale
    pass_criteria_met = (
        spread > 0.003  # >0.3% spread
        and best_acc > 0.85  # Reasonable accuracy on full CIFAR-10
        and len(errors) < total_runs // 2  # Fewer than half failed
    )

    summary = {
        "task_id": TASK_ID,
        "mode": "full",
        "timestamp": datetime.now().isoformat(),
        "spread": spread,
        "spread_pct": round(spread * 100, 2),
        "best_ordering": best_ordering,
        "worst_ordering": worst_ordering,
        "best_label": ORDERING_LABELS[best_ordering],
        "worst_label": ORDERING_LABELS[worst_ordering],
        "best_acc": best_acc,
        "worst_acc": worst_acc,
        "flip_first_mean": round(float(flip_first_mean), 4),
        "crop_first_mean": round(float(crop_first_mean), 4),
        "h4_flip_wins": h4_flip_wins,
        "pass_criteria_met": pass_criteria_met,
        "total_runs": total_runs,
        "errors": errors,
        "n_errors": len(errors),
        "orderings": results,
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        "checkpoint_dir": str(CHECKPOINTS_DIR),
    }

    out_path = FULL_DIR / "tier1_resnet18_cifar10.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{TASK_ID}] Full results written to {out_path}")
    print(f"[{TASK_ID}] Spread: {spread:.4f} ({spread*100:.2f}%)")
    print(f"[{TASK_ID}] Best: {best_ordering} ({ORDERING_LABELS[best_ordering]}) = {best_acc:.4f}")
    print(f"[{TASK_ID}] Worst: {worst_ordering} ({ORDERING_LABELS[worst_ordering]}) = {worst_acc:.4f}")
    print(f"[{TASK_ID}] H4 Flip-first wins: {h4_flip_wins} ({flip_first_mean:.4f} vs {crop_first_mean:.4f})")
    print(f"[{TASK_ID}] Pass criteria met: {pass_criteria_met}")
    print(f"[{TASK_ID}] Errors: {len(errors)}")

    end_time = datetime.now().isoformat()
    actual_min = round((time.time() - start_ts) / 60, 1)

    config_snapshot = {
        "model": "resnet18",
        "dataset": "cifar10",
        "batch_size": batch_size,
        "epochs": NUM_EPOCHS,
        "seeds": SEEDS,
        "orderings": 6,
        "mode": "full",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_count": 1,
    }

    final_status = "success" if pass_criteria_met else "partial"
    update_gpu_progress(final_status, actual_min, start_time, end_time, config_snapshot)
    update_experiment_state("completed")
    mark_done(final_status, f"full: spread={spread:.4f} ({spread*100:.2f}%) best={best_ordering} errors={len(errors)}")
    print(f"[{TASK_ID}] Done in {actual_min:.1f} min. Status: {final_status}")


if __name__ == "__main__":
    main()
