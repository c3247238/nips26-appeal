"""
Tier 4b: InfoNCE Mutual Information Estimation - PILOT mode
Pilot budget: 100 samples, seed 42, timeout 900s

For each of the 6 Tier 1 orderings on CIFAR-10 (and CIFAR-100 subset):
1. Sample 100 (image, label) pairs
2. Apply the ordering augmentation
3. Train a small ResNet-18 as feature extractor (10 epochs, pilot)
4. Estimate I(y; augmented_x) via InfoNCE lower bound using a linear classifier
   trained for 10 epochs on frozen features
5. Compute Spearman rho between per-ordering MI ranking and per-ordering accuracy ranking

Also: compare reversibility-sorted ordering (CJ->Flip->Crop) vs. conventional (Crop->Flip->CJ)
MI in paired test.

Output: exp/results/full/tier4b_mi.json
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as T
import torchvision.models as models

# ─── Paths ──────────────────────────────────────────────────────────────────
REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT = "augmentation-order"
PROJECT_DIR = Path(REMOTE_BASE) / "projects" / PROJECT
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
FULL_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "tier4b_infonce_mi"
PID_FILE = FULL_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = FULL_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = FULL_DIR / f"{TASK_ID}_DONE"

# Write PID immediately
PID_FILE.write_text(str(os.getpid()))
print(f"[{TASK_ID}] PID={os.getpid()}, GPU=3, mode=PILOT")

# ─── Seed ────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{TASK_ID}] Using device: {DEVICE}")

# ─── Pilot config ─────────────────────────────────────────────────────────
PILOT_SAMPLES = 100    # subset size per dataset
PILOT_EPOCHS = 10      # epochs for feature extractor training
LR_EPOCHS = 10         # epochs for linear classifier
BATCH_SIZE = 32

# ─── Augmentation orderings ──────────────────────────────────────────────────
ORDERINGS = {
    "order_0": {"label": "Crop→Flip→CJ", "ops": ["crop", "flip", "cj"]},
    "order_1": {"label": "Crop→CJ→Flip", "ops": ["crop", "cj", "flip"]},
    "order_2": {"label": "Flip→Crop→CJ", "ops": ["flip", "crop", "cj"]},
    "order_3": {"label": "Flip→CJ→Crop", "ops": ["flip", "cj", "crop"]},
    "order_4": {"label": "CJ→Crop→Flip", "ops": ["cj", "crop", "flip"]},
    "order_5": {"label": "CJ→Flip→Crop", "ops": ["cj", "flip", "crop"]},
}

def build_transform(ops, img_size=32, pad=4):
    """Build torchvision transform from op list."""
    op_map = {
        "crop": T.RandomCrop(img_size, padding=pad),
        "flip": T.RandomHorizontalFlip(p=0.5),
        "cj": T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    }
    transform_list = [op_map[op] for op in ops]
    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010]))
    return T.Compose(transform_list)

def get_base_transform():
    """No augmentation - just to tensor and normalize."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010])
    ])


# ─── Dataset helpers ─────────────────────────────────────────────────────────
def load_cifar_subset(dataset_name, n_samples, seed, transform):
    """Load CIFAR-10 or CIFAR-100 subset with given transform."""
    shared_dir = Path(REMOTE_BASE) / "shared" / "datasets"
    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=str(shared_dir / "cifar10"), train=True,
            download=False, transform=transform
        )
        n_classes = 10
    else:
        ds = torchvision.datasets.CIFAR100(
            root=str(shared_dir / "cifar100"), train=True,
            download=False, transform=transform
        )
        n_classes = 100

    # Stratified subset
    rng = np.random.RandomState(seed)
    targets = np.array(ds.targets)
    per_class = max(1, n_samples // n_classes)
    indices = []
    for c in range(n_classes):
        c_idx = np.where(targets == c)[0]
        chosen = rng.choice(c_idx, min(per_class, len(c_idx)), replace=False)
        indices.extend(chosen.tolist())
    indices = indices[:n_samples]
    return Subset(ds, indices), n_classes


# ─── ResNet-18 feature extractor ─────────────────────────────────────────────
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        base = models.resnet18(weights=None)
        # Adapt for 32x32 CIFAR
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(base.children())[:-1])  # remove fc
        self.feat_dim = 512
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return feat, self.classifier(feat)


def train_feature_extractor(dataset_name, ordering_ops, n_samples, epochs, seed, device):
    """Train ResNet-18 on augmented data; return frozen feature extractor."""
    transform = build_transform(ordering_ops)
    subset, n_classes = load_cifar_subset(dataset_name, n_samples, seed, transform)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = ResNetFeatureExtractor(n_classes=n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_acc = 0.0
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            _, logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        final_acc = correct / total

    return model, final_acc, n_classes


# ─── InfoNCE MI estimation ────────────────────────────────────────────────────
def infonce_mi_estimate(features, labels, n_classes, device, epochs=10):
    """
    Estimate I(y; z) via InfoNCE lower bound using a linear classifier.

    The InfoNCE lower bound on MI:
      I(y; z) >= E[log p(y|z)] - log(1/n_classes)  [when using CE loss]

    More precisely, the MI lower bound = log(n_classes) - H(y|z)
    where H(y|z) is approximated by cross-entropy of linear classifier.

    Returns: MI estimate (nats), linear classifier accuracy
    """
    # Normalize features
    features = F.normalize(features, dim=1)

    # One-hot encode labels
    n = len(labels)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, n), shuffle=True)

    feat_dim = features.shape[1]
    classifier = nn.Linear(feat_dim, n_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for feat, lbl in loader:
            feat, lbl = feat.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = classifier(feat)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * lbl.size(0)
            correct += (logits.argmax(1) == lbl).sum().item()
            total += lbl.size(0)
        best_loss = min(best_loss, total_loss / total)

    linear_acc = correct / total

    # MI lower bound: I(y;z) >= log(n_classes) - H(y|z)
    # H(y|z) approximated by best cross-entropy loss
    h_y_given_z = best_loss  # nats (cross-entropy)
    h_y = math.log(n_classes)  # entropy of uniform prior
    mi_estimate = max(0.0, h_y - h_y_given_z)

    return mi_estimate, linear_acc


def extract_features(model, dataset_name, n_samples, seed, device):
    """Extract features from frozen feature extractor using base (no aug) transform."""
    base_transform = get_base_transform()
    subset, n_classes = load_cifar_subset(dataset_name, n_samples, seed, base_transform)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feat, _ = model(imgs)
            all_features.append(feat.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels, n_classes


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ─── Main experiment ──────────────────────────────────────────────────────────
def run_tier4b_pilot():
    start_time = time.time()
    results = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "pilot_samples": PILOT_SAMPLES,
        "pilot_epochs": PILOT_EPOCHS,
        "seed": SEED,
        "datasets": {},
        "spearman_rho": {},
        "paired_comparison": {},
        "pass_criteria_met": False,
        "errors": []
    }

    total_orderings = len(ORDERINGS)
    datasets_to_run = ["cifar10", "cifar100"]
    step = 0
    total_steps = len(datasets_to_run) * total_orderings

    for ds_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        ds_results = {
            "orderings": {},
            "mi_ranking": [],
            "acc_ranking": [],
            "spearman_rho": None,
        }

        mi_values = {}
        acc_values = {}
        ordering_errors = []

        for ord_id, ord_info in ORDERINGS.items():
            step += 1
            label = ord_info["label"]
            ops = ord_info["ops"]
            print(f"\n[{step}/{total_steps}] {ds_name} | {ord_id}: {label}")

            try:
                # 1. Train feature extractor
                print(f"  Training ResNet-18 feature extractor ({PILOT_EPOCHS} epochs)...")
                t0 = time.time()
                model, train_acc, n_classes = train_feature_extractor(
                    ds_name, ops, PILOT_SAMPLES, PILOT_EPOCHS, SEED, DEVICE
                )
                train_time = time.time() - t0
                print(f"  Train acc: {train_acc:.4f}, time: {train_time:.1f}s")

                # 2. Extract features (frozen encoder, no augmentation)
                print(f"  Extracting features...")
                features, labels, _ = extract_features(model, ds_name, PILOT_SAMPLES, SEED, DEVICE)

                # 3. InfoNCE MI estimation
                print(f"  Estimating InfoNCE MI...")
                mi_est, linear_acc = infonce_mi_estimate(
                    features.to(DEVICE), labels.to(DEVICE), n_classes, DEVICE, epochs=LR_EPOCHS
                )
                print(f"  MI estimate: {mi_est:.4f} nats, Linear acc: {linear_acc:.4f}")

                # Record accuracy from training (as proxy for ordering effect)
                mi_values[ord_id] = mi_est
                acc_values[ord_id] = train_acc

                ds_results["orderings"][ord_id] = {
                    "label": label,
                    "ops": ops,
                    "train_acc": round(float(train_acc), 4),
                    "linear_acc": round(float(linear_acc), 4),
                    "mi_estimate": round(float(mi_est), 4),
                    "mi_finite": math.isfinite(mi_est),
                }

                report_progress(TASK_ID, FULL_DIR, step, total_steps,
                                metric={"mi": mi_est, "linear_acc": linear_acc})

            except Exception as e:
                import traceback
                err_msg = f"{ds_name}/{ord_id}: {str(e)}"
                print(f"  ERROR: {err_msg}")
                traceback.print_exc()
                ordering_errors.append(err_msg)
                results["errors"].append(err_msg)
                ds_results["orderings"][ord_id] = {
                    "label": label,
                    "ops": ops,
                    "error": str(e),
                    "mi_estimate": None,
                    "mi_finite": False,
                }

        # ─── Spearman rho: MI ranking vs accuracy ranking ─────────────────
        valid_orders = [
            oid for oid, v in ds_results["orderings"].items()
            if v.get("mi_estimate") is not None and v.get("mi_finite", False)
        ]

        if len(valid_orders) >= 3:
            # Rank by MI
            mi_sorted = sorted(valid_orders, key=lambda k: mi_values[k], reverse=True)
            acc_sorted = sorted(valid_orders, key=lambda k: acc_values[k], reverse=True)

            mi_rank = {oid: i+1 for i, oid in enumerate(mi_sorted)}
            acc_rank = {oid: i+1 for i, oid in enumerate(acc_sorted)}

            # Spearman rho
            n = len(valid_orders)
            d_sq_sum = sum((mi_rank[oid] - acc_rank[oid])**2 for oid in valid_orders)
            spearman_rho = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))

            ds_results["mi_ranking"] = mi_sorted
            ds_results["acc_ranking"] = acc_sorted
            ds_results["spearman_rho"] = round(float(spearman_rho), 4)
            ds_results["n_valid_orderings"] = n

            print(f"\n[{ds_name}] Spearman rho (MI vs acc): {spearman_rho:.4f}")
            print(f"  MI ranking:  {[ORDERINGS[o]['label'] for o in mi_sorted]}")
            print(f"  Acc ranking: {[ORDERINGS[o]['label'] for o in acc_sorted]}")
        else:
            ds_results["spearman_rho"] = None
            ds_results["error"] = f"Not enough valid orderings: {len(valid_orders)}"

        # ─── Paired comparison: CJ→Flip→Crop vs Crop→Flip→CJ ────────────
        conv_id = "order_0"   # Crop→Flip→CJ (conventional)
        rev_id = "order_5"    # CJ→Flip→Crop (reversibility-sorted)

        conv_mi = ds_results["orderings"].get(conv_id, {}).get("mi_estimate")
        rev_mi = ds_results["orderings"].get(rev_id, {}).get("mi_estimate")
        conv_acc = ds_results["orderings"].get(conv_id, {}).get("train_acc")
        rev_acc = ds_results["orderings"].get(rev_id, {}).get("train_acc")

        if conv_mi is not None and rev_mi is not None:
            ds_results["paired_comparison"] = {
                "conventional_order_mi": conv_mi,
                "reversibility_sorted_mi": rev_mi,
                "mi_diff": round(float(rev_mi - conv_mi), 4),
                "reversibility_sorted_wins_mi": rev_mi > conv_mi,
                "conventional_acc": conv_acc,
                "reversibility_sorted_acc": rev_acc,
                "acc_diff": round(float(rev_acc - conv_acc), 4) if conv_acc and rev_acc else None,
                "reversibility_sorted_wins_acc": rev_acc > conv_acc if conv_acc and rev_acc else None,
            }
            print(f"\n[{ds_name}] Paired comparison:")
            print(f"  Conventional (Crop→Flip→CJ): MI={conv_mi:.4f}, Acc={conv_acc:.4f}")
            print(f"  Reversibility (CJ→Flip→Crop): MI={rev_mi:.4f}, Acc={rev_acc:.4f}")
        else:
            ds_results["paired_comparison"] = {"error": "Missing MI values for comparison"}

        results["datasets"][ds_name] = ds_results
        results["spearman_rho"][ds_name] = ds_results.get("spearman_rho")

    # ─── Overall pass criteria check ──────────────────────────────────────────
    # Pass: InfoNCE MI estimates computed for all 6 orderings; MI values finite; rho runs without error
    all_mi_finite = True
    mi_count = 0
    for ds_name in datasets_to_run:
        for ord_id, ord_data in results["datasets"].get(ds_name, {}).get("orderings", {}).items():
            mi_val = ord_data.get("mi_estimate")
            if mi_val is not None:
                mi_count += 1
                if not math.isfinite(float(mi_val)):
                    all_mi_finite = False

    rho_computed = any(
        results["datasets"].get(ds, {}).get("spearman_rho") is not None
        for ds in datasets_to_run
    )

    results["pass_criteria_met"] = (mi_count >= 6) and all_mi_finite and rho_computed
    results["mi_values_finite"] = all_mi_finite
    results["mi_count"] = mi_count
    results["rho_computed"] = rho_computed

    # ─── Save results ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_sec"] = round(elapsed, 1)
    results["timestamp_end"] = datetime.now().isoformat()

    output_path = FULL_DIR / "tier4b_mi.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{TASK_ID}] Results saved to: {output_path}")
    print(f"[{TASK_ID}] Pass criteria met: {results['pass_criteria_met']}")
    print(f"[{TASK_ID}] Total MI count: {mi_count}, All finite: {all_mi_finite}")
    print(f"[{TASK_ID}] Rho computed: {rho_computed}")
    print(f"[{TASK_ID}] Elapsed: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    try:
        results = run_tier4b_pilot()
        status = "success" if results["pass_criteria_met"] else "partial"
        summary = (
            f"MI estimated for {results['mi_count']} orderings, "
            f"finite={results['mi_values_finite']}, "
            f"rho_computed={results['rho_computed']}, "
            f"elapsed={results['elapsed_sec']}s"
        )
        mark_task_done(TASK_ID, FULL_DIR, status=status, summary=summary)
        print(f"\n[{TASK_ID}] DONE marker written (status={status})")
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_task_done(TASK_ID, FULL_DIR, status="failed", summary=str(e))
        sys.exit(1)
