"""
Full run for Tier 1: Full Factorial — ViT-Small x CIFAR-10
FULL MODE: Full 50k training set, 200 epochs, 5 seeds [42,43,44,45,46].
All 6 orderings of {RandomCrop, RandomHorizontalFlip, ColorJitter}.
AdamW lr=1e-3, wd=0.05, 5-epoch linear warmup + cosine decay.
Saves per-ordering checkpoints for Tier 4 feature extraction.
Output: exp/results/full/tier1_vit_cifar10.json
"""
import os
import sys
import json
import time
import random
import math
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ---- Config ----
TASK_ID = "tier1_vit_cifar10_full"
RESULTS_DIR = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/results")
FULL_DIR = RESULTS_DIR / "full"
CKPT_DIR = RESULTS_DIR / "checkpoints" / "tier1_vit_cifar10_full"
GPU_PROGRESS_PATH = Path("/home/qhxie/sibyl_system/projects/augmentation-order/exp/gpu_progress.json")
CIFAR10_PATH = "/home/qhxie/sibyl_system/shared/datasets/cifar10"

# Full run settings
NUM_EPOCHS = 200
WARMUP_EPOCHS = 5
BATCH_SIZE = 256  # ViT benefits from larger batch
LR = 1e-3
WEIGHT_DECAY = 0.05
SEEDS = [42, 43, 44, 45, 46]
DEVICE = "cuda:0"
PLANNED_MIN = 660  # ~11 hours estimated for 30 runs

# CIFAR-10 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# ---- Orderings ----
ORDERINGS = {
    "order_0": ["crop", "flip", "cj"],
    "order_1": ["crop", "cj", "flip"],
    "order_2": ["flip", "crop", "cj"],
    "order_3": ["flip", "cj", "crop"],
    "order_4": ["cj", "crop", "flip"],
    "order_5": ["cj", "flip", "crop"],
}

ORDERING_LABELS = {
    "order_0": "Crop->Flip->CJ",
    "order_1": "Crop->CJ->Flip",
    "order_2": "Flip->Crop->CJ",
    "order_3": "Flip->CJ->Crop",
    "order_4": "CJ->Crop->Flip",
    "order_5": "CJ->Flip->Crop",
}


def build_transform(ordering_ops):
    ops = {
        "crop": transforms.RandomCrop(32, padding=4),
        "flip": transforms.RandomHorizontalFlip(p=0.5),
        "cj": transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    }
    aug_transforms = [ops[op] for op in ordering_ops]
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


# ---- ViT-Small for CIFAR-10 (patch=4) ----

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmallCIFAR(nn.Module):
    """ViT-Small adapted for 32x32 CIFAR images with patch_size=4.
    embed_dim=384, depth=12, num_heads=6 (standard ViT-S config).
    Returns penultimate features for Tier 4 extraction."""

    def __init__(self, num_classes=10, img_size=32, patch_size=4,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # 64

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Return penultimate (CLS) features for Tier 4 extraction."""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token: (B, embed_dim)

    def forward(self, x):
        return self.head(self.forward_features(x))


def get_lr_with_warmup(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    """Linear warmup then cosine decay."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_decay


def get_datasets(ordering_ops):
    transform_train = build_transform(ordering_ops)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=False,
                                             transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=False,
                                           transform=transform_val)
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
        # Gradient clipping for ViT stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def run_single(ordering_key, ordering_ops, seed, device, save_checkpoint=False):
    set_seed(seed)
    trainset, valset = get_datasets(ordering_ops)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=512, shuffle=False,
                           num_workers=4, pin_memory=True)

    model = ViTSmallCIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    per_epoch = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        # Manual LR scheduling with warmup
        current_lr = get_lr_with_warmup(epoch, WARMUP_EPOCHS, NUM_EPOCHS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, valloader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # Save best checkpoint if requested (only seed 42 for Tier 4)
            if save_checkpoint:
                ckpt_path = CKPT_DIR / f"{ordering_key}_seed{seed}_best.pt"
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch + 1,
                    "ordering_key": ordering_key,
                    "seed": seed,
                    "val_acc": val_acc,
                    "model_state_dict": model.state_dict(),
                    "embed_dim": 384,
                    "patch_size": 4,
                    "depth": 12,
                    "num_heads": 6,
                }, ckpt_path)

        per_epoch.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(current_lr, 7),
        })
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == WARMUP_EPOCHS - 1:
            print(f"    epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f} lr={current_lr:.6f}")
            sys.stdout.flush()

    final_val_acc = per_epoch[-1]["val_acc"]
    return final_val_acc, best_val_acc, best_epoch, per_epoch


def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID {os.getpid()} written to {pid_file}")
    sys.stdout.flush()


def write_progress(run_count, total_runs, ordering_key=None, seed=None):
    progress = {
        "task_id": TASK_ID,
        "run": run_count,
        "total_runs": total_runs,
        "ordering": ordering_key,
        "seed": seed,
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress, indent=2))


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
    }, indent=2))
    print(f"[{TASK_ID}] DONE marker written: {marker}")
    sys.stdout.flush()


def update_gpu_progress(status, actual_min, start_time, end_time):
    try:
        if GPU_PROGRESS_PATH.exists():
            data = json.loads(GPU_PROGRESS_PATH.read_text())
        else:
            data = {"completed": [], "failed": [], "running": {}, "timings": {}}
        if status in ("success", "partial"):
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
                "model": "vit_small_patch4",
                "dataset": "cifar10",
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "warmup_epochs": WARMUP_EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "seeds": SEEDS,
                "orderings": 6,
                "mode": "full",
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "patch_size": 4,
            }
        }
        GPU_PROGRESS_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning: could not update gpu_progress.json: {e}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    write_pid()

    device = torch.device(DEVICE)
    print(f"[{TASK_ID}] FULL MODE: full 50k dataset, {NUM_EPOCHS} epochs, seeds={SEEDS}")
    print(f"[{TASK_ID}] Model: ViT-Small (patch4, embed=384, depth=12, heads=6)")
    print(f"[{TASK_ID}] Optimizer: AdamW lr={LR}, wd={WEIGHT_DECAY}, warmup={WARMUP_EPOCHS}ep + cosine")
    print(f"[{TASK_ID}] Running on device: {device}")
    print(f"[{TASK_ID}] PyTorch version: {torch.__version__}")
    print(f"[{TASK_ID}] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[{TASK_ID}] GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[{TASK_ID}] GPU Memory: {total_mem:.1f} GB")
    sys.stdout.flush()

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
            # Save checkpoint only for seed 42 (used in Tier 4 feature extraction)
            save_ckpt = (seed == 42)
            print(f"\n[{run_count}/{total_runs}] ordering={ordering_key} ({ORDERING_LABELS[ordering_key]}) seed={seed}"
                  f"{' [saving checkpoint]' if save_ckpt else ''}")
            sys.stdout.flush()
            write_progress(run_count - 1, total_runs, ordering_key, seed)
            t0 = time.time()
            try:
                final_acc, best_acc, best_epoch, per_epoch = run_single(
                    ordering_key, ordering_ops, seed, device, save_checkpoint=save_ckpt
                )
                elapsed = time.time() - t0
                print(f"  -> final val_acc={final_acc:.4f} best val_acc={best_acc:.4f} "
                      f"(best epoch {best_epoch}, elapsed {elapsed:.1f}s)")
                sys.stdout.flush()
                results[ordering_key]["per_seed"][str(seed)] = {
                    "final_val_acc": final_acc,
                    "best_val_acc": best_acc,
                    "best_epoch": best_epoch,
                    "per_epoch": per_epoch,
                    "elapsed_sec": round(elapsed, 1),
                }
            except Exception as e:
                elapsed = time.time() - t0
                error_msg = str(e)
                print(f"  ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                errors.append({"ordering": ordering_key, "seed": seed, "error": error_msg})
                results[ordering_key]["per_seed"][str(seed)] = {
                    "final_val_acc": 0.0,
                    "best_val_acc": 0.0,
                    "best_epoch": 0,
                    "per_epoch": [],
                    "error": error_msg,
                    "elapsed_sec": round(elapsed, 1),
                }
            write_progress(run_count, total_runs, ordering_key, seed)

            # Save intermediate results after each (ordering, seed) run
            intermediate = {
                "task_id": TASK_ID,
                "mode": "full",
                "timestamp": datetime.now().isoformat(),
                "progress": f"{run_count}/{total_runs}",
                "errors_so_far": errors,
                "orderings": results,
                "seeds": SEEDS,
                "epochs": NUM_EPOCHS,
                "status": "in_progress",
            }
            # Save to full/ directory
            (FULL_DIR / "tier1_vit_cifar10.json").write_text(json.dumps(intermediate, indent=2))

    # Aggregate per ordering
    print("\n=== Final Results ===")
    for ordering_key in results:
        seed_results = results[ordering_key]["per_seed"]
        accs = [seed_results[str(s)]["final_val_acc"] for s in SEEDS if str(s) in seed_results and "error" not in seed_results[str(s)]]
        best_accs = [seed_results[str(s)].get("best_val_acc", seed_results[str(s)]["final_val_acc"])
                     for s in SEEDS if str(s) in seed_results and "error" not in seed_results[str(s)]]
        if accs:
            results[ordering_key]["mean_val_acc"] = round(float(np.mean(accs)), 4)
            results[ordering_key]["std_val_acc"] = round(float(np.std(accs)), 4)
            results[ordering_key]["mean_best_val_acc"] = round(float(np.mean(best_accs)), 4)
            results[ordering_key]["std_best_val_acc"] = round(float(np.std(best_accs)), 4)
        else:
            results[ordering_key]["mean_val_acc"] = 0.0
            results[ordering_key]["std_val_acc"] = 0.0
            results[ordering_key]["mean_best_val_acc"] = 0.0
            results[ordering_key]["std_best_val_acc"] = 0.0
        print(f"  {ordering_key} ({ORDERING_LABELS[ordering_key]}): "
              f"mean={results[ordering_key]['mean_val_acc']:.4f} "
              f"std={results[ordering_key]['std_val_acc']:.4f} "
              f"best_mean={results[ordering_key]['mean_best_val_acc']:.4f}")

    all_means = [results[k]["mean_val_acc"] for k in results]
    valid_means = [v for v in all_means if v > 0.0]
    spread = round(max(valid_means) - min(valid_means), 4) if len(valid_means) >= 2 else 0.0
    best_ordering = max(results, key=lambda k: results[k]["mean_val_acc"])
    worst_ordering = min(results, key=lambda k: results[k].get("mean_val_acc", 0.0))
    max_val_acc = max(valid_means) if valid_means else 0.0

    print(f"\n[{TASK_ID}] Spread: {spread:.4f} ({spread*100:.2f}%)")
    print(f"[{TASK_ID}] Best: {best_ordering} ({ORDERING_LABELS[best_ordering]}) = {results[best_ordering]['mean_val_acc']:.4f}")
    print(f"[{TASK_ID}] Worst: {worst_ordering} ({ORDERING_LABELS[worst_ordering]}) = {results[worst_ordering]['mean_val_acc']:.4f}")
    print(f"[{TASK_ID}] Errors: {len(errors)}")

    # Final summary
    summary = {
        "task_id": TASK_ID,
        "mode": "full",
        "model": "vit_small_patch4",
        "dataset": "cifar10",
        "timestamp": datetime.now().isoformat(),
        "spread": spread,
        "max_val_acc": max_val_acc,
        "best_ordering": best_ordering,
        "worst_ordering": worst_ordering,
        "best_label": ORDERING_LABELS[best_ordering],
        "worst_label": ORDERING_LABELS[worst_ordering],
        "errors": errors,
        "orderings": results,
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "optimizer": {"type": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY},
        "checkpoint_dir": str(CKPT_DIR),
        "status": "completed",
    }

    out_path = FULL_DIR / "tier1_vit_cifar10.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{TASK_ID}] Final results written to {out_path}")

    end_time = datetime.now().isoformat()
    actual_min = round((time.time() - start_ts) / 60, 1)

    final_status = "success" if len(errors) == 0 else "partial"
    update_gpu_progress(final_status, actual_min, start_time, end_time)
    mark_done(final_status, f"full: spread={spread:.4f} best={best_ordering} errors={len(errors)}")
    print(f"[{TASK_ID}] Done in {actual_min} min. Status: {final_status}")


if __name__ == "__main__":
    main()
