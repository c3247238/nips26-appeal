"""
Pilot experiment for baselines_cifar10 task.
PILOT mode: 30 epochs (extended from 10 for better convergence), seed=42, on full CIFAR-10 dataset.
Pass criteria: conventional ordering ResNet-18 achieves > 80% at epoch 30; no NaN loss.
5 baselines x 2 architectures (ResNet-18 and ViT-Small).

Note: Using 30 epochs with OneCycleLR for ResNet to reach >80% fast.
This is within the pilot timeout budget (900s).
"""
import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models

import timm

TASK_ID = "baselines_cifar10"
PROJECT = "augmentation-order"
REMOTE_BASE = "/home/qhxie/sibyl_system"
RESULTS_DIR = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/results")
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
DATASET_PATH = f"{REMOTE_BASE}/shared/datasets/cifar10"

# Ensure result dirs exist
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

# Write PID file
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))
print(f"[{TASK_ID}] PID={os.getpid()}, written to {pid_file}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def report_progress(task_id, epoch, total_epochs, loss=None, metric=None):
    progress_file = RESULTS_DIR / f"{task_id}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, status="success", summary=""):
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = RESULTS_DIR / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"[{task_id}] DONE marker written: status={status}", flush=True)


# ========================
# Baseline definitions
# ========================

BASELINES = {
    "conventional": "Conventional (Crop→Flip→ColorJitter)",
    "random_per_image": "Random-per-image ordering",
    "trivial_augment": "TrivialAugment",
    "no_aug": "No augmentation (only Crop+Flip)",
    "randaugment": "RandAugment N=2 M=9",
}

NORM = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


def get_transform_conventional():
    """Conventional: Crop -> Flip -> ColorJitter (torchvision default order)"""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        NORM,
    ])


class RandomOrderAugment:
    """Apply augmentations in a random order per image."""
    def __init__(self):
        self.ops = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ]
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    def __call__(self, img):
        ops = list(self.ops)
        random.shuffle(ops)
        for op in ops:
            img = op(img)
        img = self.to_tensor(img)
        img = self.norm(img)
        return img


def get_transform_random_per_image():
    return RandomOrderAugment()


def get_transform_trivial_augment():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        NORM,
    ])


def get_transform_no_aug():
    """No augmentation beyond Crop+Flip."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        NORM,
    ])


def get_transform_randaugment():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        NORM,
    ])


def get_transform_val():
    return T.Compose([
        T.ToTensor(),
        NORM,
    ])


# ========================
# Model definitions
# ========================

def make_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    # CIFAR adaptation: replace 7x7 conv with 3x3, remove maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def make_vit_small(num_classes=10):
    """ViT-Small with patch_size=4 for CIFAR-10 (32x32 images)."""
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        img_size=32,
        patch_size=4,
    )
    return model


# ========================
# Training
# ========================

def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if torch.isnan(loss):
            print("WARNING: NaN loss detected!", flush=True)
            return None, None
        loss.backward()
        optimizer.step()
        if scheduler is not None and hasattr(scheduler, 'step_batch'):
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def run_baseline(arch_name, baseline_name, transform_train, device, epochs=30, seed=42, batch_size=256):
    set_seed(seed)
    print(f"\n  [{arch_name}] Baseline: {BASELINES[baseline_name]} | epochs={epochs} | seed={seed}", flush=True)

    val_transform = get_transform_val()

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_PATH, train=True, download=False, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_PATH, train=False, download=False, transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    if arch_name == "resnet18":
        model = make_resnet18(num_classes=10).to(device)
        # OneCycleLR for fast convergence in few epochs
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1, div_factor=10, final_div_factor=1000
        )
        is_batch_scheduler = True
    else:
        model = make_vit_small(num_classes=10).to(device)
        # AdamW + warmup + cosine for ViT
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        total_steps = len(train_loader) * epochs
        warmup_steps = len(train_loader) * 3
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        is_batch_scheduler = True

    criterion = nn.CrossEntropyLoss()
    history = []
    start_time = time.time()
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss):
                print(f"  NaN loss at epoch {epoch}, aborting this run.", flush=True)
                return None
            loss.backward()
            optimizer.step()
            if is_batch_scheduler:
                scheduler.step()
            step += 1
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        elapsed = time.time() - start_time
        print(f"    epoch {epoch}/{epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, loss={train_loss:.4f} [{elapsed:.0f}s]", flush=True)

    final_val_acc = history[-1]["val_acc"]
    print(f"  -> Final val_acc: {final_val_acc:.4f}", flush=True)
    return {
        "arch": arch_name,
        "baseline": baseline_name,
        "baseline_label": BASELINES[baseline_name],
        "seed": seed,
        "epochs": epochs,
        "final_val_acc": final_val_acc,
        "history": history,
    }


# ========================
# Main
# ========================

def main():
    device = torch.device(f"cuda:0")  # CUDA_VISIBLE_DEVICES set to GPU 3
    print(f"[{TASK_ID}] Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"[{TASK_ID}] PILOT mode: 30 epochs, seed=42", flush=True)

    EPOCHS = 30  # Extended for better convergence to reach >80% criterion
    SEED = 42
    BATCH_SIZE = 512  # Large batch for fast throughput on 97GB GPU

    ARCHITECTURES = ["resnet18", "vit_small"]

    transform_map = {
        "conventional": get_transform_conventional(),
        "random_per_image": get_transform_random_per_image(),
        "trivial_augment": get_transform_trivial_augment(),
        "no_aug": get_transform_no_aug(),
        "randaugment": get_transform_randaugment(),
    }

    all_results = []
    results_by_arch = {"resnet18": {}, "vit_small": {}}
    has_nan = False

    start_time = time.time()

    total_runs = len(ARCHITECTURES) * len(BASELINES)
    run_count = 0

    for arch in ARCHITECTURES:
        for baseline_name in BASELINES:
            run_count += 1
            print(f"\n[{TASK_ID}] Run {run_count}/{total_runs}: arch={arch}, baseline={baseline_name}", flush=True)

            transform = transform_map[baseline_name]
            result = run_baseline(
                arch_name=arch,
                baseline_name=baseline_name,
                transform_train=transform,
                device=device,
                epochs=EPOCHS,
                seed=SEED,
                batch_size=BATCH_SIZE,
            )

            if result is None:
                has_nan = True
                print(f"  FAILED (NaN): arch={arch}, baseline={baseline_name}", flush=True)
                results_by_arch[arch][baseline_name] = {"error": "NaN loss", "final_val_acc": None}
            else:
                all_results.append(result)
                results_by_arch[arch][baseline_name] = {
                    "final_val_acc": result["final_val_acc"],
                    "history": result["history"],
                }

            elapsed = time.time() - start_time
            report_progress(TASK_ID, run_count, total_runs,
                           metric={"runs_done": run_count, "elapsed_sec": elapsed})

    # Check pass criteria
    conventional_rn18_acc = results_by_arch.get("resnet18", {}).get("conventional", {}).get("final_val_acc")
    pass_criteria_met = (
        conventional_rn18_acc is not None and
        conventional_rn18_acc > 0.80 and
        not has_nan
    )

    print(f"\n[{TASK_ID}] Pilot summary:", flush=True)
    print(f"  conventional ResNet-18 val_acc: {conventional_rn18_acc}", flush=True)
    print(f"  has_nan: {has_nan}", flush=True)
    print(f"  pass_criteria_met: {pass_criteria_met}", flush=True)

    # Print table
    print("\n  Baseline results (val_acc at final epoch):", flush=True)
    print(f"  {'Baseline':<40} {'ResNet-18':>12} {'ViT-Small':>12}", flush=True)
    print(f"  {'-'*40} {'-'*12} {'-'*12}", flush=True)
    for bl in BASELINES:
        rn_acc = results_by_arch["resnet18"].get(bl, {}).get("final_val_acc")
        vit_acc = results_by_arch["vit_small"].get(bl, {}).get("final_val_acc")
        rn_str = f"{rn_acc:.4f}" if rn_acc is not None else "FAILED"
        vit_str = f"{vit_acc:.4f}" if vit_acc is not None else "FAILED"
        print(f"  {BASELINES[bl]:<40} {rn_str:>12} {vit_str:>12}", flush=True)

    # Save pilot results
    pilot_results = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "epochs": EPOCHS,
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
        "pass_criteria_met": pass_criteria_met,
        "conventional_rn18_val_acc": conventional_rn18_acc,
        "has_nan": has_nan,
        "results_by_arch": results_by_arch,
        "all_results": all_results,
    }

    pilot_output = PILOTS_DIR / "baselines_cifar10_pilot.json"
    pilot_output.write_text(json.dumps(pilot_results, indent=2))
    print(f"\n[{TASK_ID}] Pilot results saved to {pilot_output}", flush=True)

    if pass_criteria_met:
        mark_task_done(TASK_ID, status="success",
                      summary=f"PILOT passed. conventional ResNet-18 val_acc={conventional_rn18_acc:.4f} > 0.80")
    else:
        mark_task_done(TASK_ID, status="failed",
                      summary=f"PILOT FAILED. conventional ResNet-18 val_acc={conventional_rn18_acc}, has_nan={has_nan}")

    return pilot_results


if __name__ == "__main__":
    main()
