"""
Tier 1 Full: ViT-Small/4 x CIFAR-100 — GPU 6 parallel runner
Handles orderings: order_2, order_3, order_4, order_5 (all 5 seeds, 200 epochs)
order_0 and order_1 are handled by the GPU 3 process (PID 81261)

Results are saved per-ordering to allow merging with the GPU 3 results.
Final merge produces exp/results/full/tier1_vit_cifar100_full.json
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

# ---- Config ----
TASK_ID = "tier1_vit_cifar100_full_gpu6"  # Different ID to avoid conflicts
REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT = "augmentation-order"
RESULTS_DIR = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/results")
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints" / "tier1_vit_cifar100_full"  # Shared checkpoint dir
GPU_PROGRESS_PATH = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/gpu_progress.json")
EXPERIMENT_STATE_PATH = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/experiment_state.json")
CIFAR100_PATH = f"{REMOTE_BASE}/shared/datasets/cifar100"

# FULL settings - this runner handles orderings 2-5
NUM_EPOCHS = 200
BATCH_SIZE = 256  # will be probed
SEEDS = [42, 43, 44, 45, 46]
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=6 set externally, maps to cuda:0
PLANNED_MIN = 270  # ~4.5 hours for 20 runs x 200 epochs (orderings 2-5)

# CIFAR-100 normalization
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# This runner handles orderings 2-5 only
ORDERINGS = {
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


# ---- ViT-Small Architecture ----
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
    """ViT-Small adapted for 32x32 CIFAR images with patch_size=4."""
    def __init__(self, num_classes=100, img_size=32, patch_size=4,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # 64

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

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

    def forward(self, x):
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
        return self.head(x[:, 0])


# ---- Helpers ----
def build_transform(ordering_ops):
    ops_map = {
        "crop": transforms.RandomCrop(32, padding=4),
        "flip": transforms.RandomHorizontalFlip(p=0.5),
        "cj": transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    }
    aug_transforms = [ops_map[op] for op in ordering_ops]
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


def get_datasets(ordering_ops):
    transform_train = build_transform(ordering_ops)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    trainset = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=True,
                                              download=False, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=False,
                                            download=False, transform=transform_val)
    return trainset, valset


def probe_batch_size(device, start=512, min_bs=32):
    import gc
    model = ViTSmallCIFAR(num_classes=100).to(device)
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
    print(f"[{TASK_ID}] PID {os.getpid()} written to {pid_file}", flush=True)


def write_progress(run_idx, total_runs, epoch=0, total_epochs=NUM_EPOCHS,
                   ordering_key=None, loss=None, metric=None):
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
    print(f"[{TASK_ID}] DONE marker written: {marker}", flush=True)


def run_single(ordering_key, ordering_ops, seed, device, batch_size):
    set_seed(seed)
    trainset, valset = get_datasets(ordering_ops)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valloader = DataLoader(valset, batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = ViTSmallCIFAR(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ckpt_dir = CHECKPOINTS_DIR / ordering_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    per_epoch = []
    start_epoch = 1

    resume_ckpt = ckpt_dir / f"seed_{seed}_latest.pt"
    if resume_ckpt.exists():
        try:
            ckpt = torch.load(resume_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            per_epoch = ckpt.get("per_epoch", [])
            print(f"    [Resume] Loaded checkpoint at epoch {ckpt['epoch']} "
                  f"for {ordering_key}/seed={seed}", flush=True)
        except Exception as e:
            print(f"    [Resume] Warning: failed to load checkpoint: {e}. Starting fresh.", flush=True)
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
                  f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    final_val_acc = per_epoch[-1]["val_acc"]

    final_ckpt = ckpt_dir / f"seed_{seed}_final.pt"
    torch.save({
        "epoch": NUM_EPOCHS,
        "model": model.state_dict(),
        "ordering_key": ordering_key,
        "ordering_ops": ordering_ops,
        "seed": seed,
        "final_val_acc": final_val_acc,
    }, final_ckpt)

    if resume_ckpt.exists():
        resume_ckpt.unlink()

    return final_val_acc, per_epoch


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"[{TASK_ID}] Parallel runner for orderings: {list(ORDERINGS.keys())}", flush=True)
    print(f"[{TASK_ID}] FULL SCALE: {NUM_EPOCHS} epochs, seeds={SEEDS}", flush=True)
    print(f"[{TASK_ID}] Model: ViT-Small (patch4, img32, embed=384, depth=12, heads=6, classes=100)", flush=True)
    print(f"[{TASK_ID}] Optimizer: AdamW (lr=1e-3, weight_decay=0.05) + CosineAnnealingLR", flush=True)
    print(f"[{TASK_ID}] Device: {device}", flush=True)
    print(f"[{TASK_ID}] PyTorch: {torch.__version__}", flush=True)
    print(f"[{TASK_ID}] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[{TASK_ID}] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[{TASK_ID}] VRAM total: {vram_total:.1f} GB", flush=True)

    write_pid()

    print(f"[{TASK_ID}] Probing max batch size for ViT-Small...", flush=True)
    batch_size = probe_batch_size(device, start=512, min_bs=32)
    batch_size = min(batch_size, 256)
    print(f"[{TASK_ID}] Using batch_size={batch_size}", flush=True)

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
            print(f"\n[{run_count}/{total_runs}] ordering={ordering_key} "
                  f"({ORDERING_LABELS[ordering_key]}) seed={seed}", flush=True)
            write_progress(run_count - 1, total_runs, epoch=0, ordering_key=ordering_key)
            t0 = time.time()
            try:
                final_acc, per_epoch = run_single(ordering_key, ordering_ops, seed, device, batch_size)
                elapsed = time.time() - t0
                print(f"  -> final val_acc={final_acc:.4f} (elapsed {elapsed/60:.1f} min)", flush=True)
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
                print(f"  ERROR: {error_msg}", flush=True)
                print(tb, flush=True)
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

        # Save partial results for this ordering immediately
        ordering_result_path = RESULTS_DIR / f"tier1_vit_cifar100_full_{ordering_key}.json"
        ordering_result_path.write_text(json.dumps({
            "ordering_key": ordering_key,
            "label": ORDERING_LABELS[ordering_key],
            "per_seed": results[ordering_key]["per_seed"],
        }, indent=2))
        print(f"[{TASK_ID}] Saved {ordering_key} results to {ordering_result_path}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"[{TASK_ID}] RESULTS SUMMARY (orderings 2-5)", flush=True)
    print(f"{'='*60}", flush=True)
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
              f"(n={results[ordering_key]['n_seeds_ok']})", flush=True)

    # Save partial results for orderings 2-5
    partial_out = RESULTS_DIR / "tier1_vit_cifar100_full_partial_gpu6.json"
    partial_out.write_text(json.dumps({
        "task_id": TASK_ID,
        "mode": "full",
        "orderings_covered": list(ORDERINGS.keys()),
        "timestamp": datetime.now().isoformat(),
        "orderings": results,
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        "errors": errors,
        "n_errors": len(errors),
    }, indent=2))
    print(f"\n[{TASK_ID}] Partial results written to {partial_out}", flush=True)

    end_time = datetime.now().isoformat()
    actual_min = round((time.time() - start_ts) / 60, 1)

    mark_done("success" if len(errors) == 0 else "partial",
              f"orderings 2-5 full: actual_min={actual_min} errors={len(errors)}")
    print(f"[{TASK_ID}] Done in {actual_min:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
