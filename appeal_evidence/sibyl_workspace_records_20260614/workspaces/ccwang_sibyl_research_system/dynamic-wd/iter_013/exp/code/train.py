#!/usr/bin/env python3
"""
Unified training script for all WD methods.

Supports:
- CIFAR-10/100 (single GPU) and ImageNet (DDP multi-GPU)
- All 7 WD methods via --wd_method flag
- Per-layer diagnostic logging
- Checkpoint saving and metric export to JSON
- Mixed precision training (--amp)
- Configurable via command-line arguments or JSON config

Usage:
    # CIFAR-100 / ResNet-20 / FixedWD
    python train.py --dataset cifar100 --arch resnet20 --wd_method FixedWD --epochs 200 --seed 42

    # ImageNet / ResNet-50 / EqWD with DDP
    torchrun --nproc_per_node=2 train.py --dataset imagenet --arch resnet50 \
        --wd_method EqWD --epochs 90 --batch_size 256 --amp --imagenet_dir /data/imagenet

    # From JSON config
    python train.py --config config.json
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from wd_methods import create_wd_method, apply_wd_step, detect_normalized_layers
from diagnostics import DiagnosticsLogger, ProgressReporter
from data_utils import (
    get_cifar_loaders, get_imagenet_loaders, get_model, count_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified WD training script")

    # Dataset and model
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--arch", type=str, default="resnet20",
                        choices=["resnet20", "vgg16bn", "vgg16", "resnet50", "resnet101"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--imagenet_dir", type=str, default="/data/imagenet")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # LR schedule
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "step", "none"])
    parser.add_argument("--step_milestones", type=str, default="30,60,80",
                        help="Comma-separated epoch milestones for step LR")
    parser.add_argument("--step_gamma", type=float, default=0.1)

    # Weight decay method
    parser.add_argument("--wd_method", type=str, default="FixedWD",
                        choices=["NoWD", "FixedWD", "SWD", "CWD", "CPR", "CAWD", "EqWD"])
    parser.add_argument("--wd_lambda", type=float, default=5e-4,
                        help="Base weight decay lambda")
    parser.add_argument("--eqwd_beta", type=float, default=1.0,
                        help="EqWD adaptation strength")
    parser.add_argument("--eqwd_ema_alpha", type=float, default=0.9,
                        help="EqWD EMA decay rate")
    parser.add_argument("--eqwd_layer_aware", action="store_true",
                        help="Enable layer-type-aware EqWD")
    parser.add_argument("--cawd_noise_sigma", type=float, default=0.0,
                        help="Noise injection sigma for CAWD control experiment")

    # Mixed precision
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")

    # Diagnostics
    parser.add_argument("--diag_interval", type=int, default=10,
                        help="Log per-layer diagnostics every N steps")

    # Output
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--task_id", type=str, default="default")
    parser.add_argument("--save_checkpoints", type=str, default="",
                        help="Comma-separated epochs to save checkpoints (e.g., '100,200')")

    # Config file (overrides CLI args)
    parser.add_argument("--config", type=str, default="",
                        help="JSON config file (overrides all CLI args)")

    # DDP
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Initialize DDP if applicable. Returns (rank, world_size, is_distributed)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, True, local_rank
    return 0, 1, False, 0


def get_lr_scheduler(optimizer, args, steps_per_epoch: int):
    """Create learning rate scheduler."""
    if args.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * steps_per_epoch
        )
    elif args.lr_schedule == "step":
        milestones = [int(x) for x in args.step_milestones.split(",")]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.step_gamma
        )
    return None


def train_one_epoch(
    model, train_loader, criterion, optimizer, scheduler,
    wd_method, normalized_layers, diagnostics, scaler,
    epoch, global_step, device, args, rank,
):
    """Train for one epoch. Returns (avg_loss, avg_acc, global_step)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if args.amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Apply custom WD BEFORE optimizer step
        # For the underlying model (unwrap DDP if needed)
        raw_model = model.module if hasattr(model, "module") else model
        effective_wds = apply_wd_step(
            raw_model, wd_method, global_step, current_lr, normalized_layers
        )

        # Log diagnostics
        if rank == 0:
            diagnostics.log_step(raw_model, global_step, epoch, effective_wds, wd_method)

        # Optimizer step
        if args.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Step-level LR scheduler (for cosine)
        if scheduler and args.lr_schedule == "cosine":
            scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        global_step += 1

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * correct / total

    # Epoch-level LR scheduler (for step)
    if scheduler and args.lr_schedule == "step":
        scheduler.step()

    return avg_loss, avg_acc, global_step


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, topk=(1,)):
    """Evaluate model on test/val set. Returns (avg_loss, top1_acc, [top5_acc])."""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        _, pred = outputs.topk(max(topk), 1, True, True)
        pred = pred.t()
        correct_mask = pred.eq(targets.view(1, -1).expand_as(pred))

        correct_top1 += correct_mask[:1].reshape(-1).float().sum().item()
        if 5 in topk:
            correct_top5 += correct_mask[:5].reshape(-1).float().sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total
    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total if 5 in topk else None

    return avg_loss, top1, top5


def main():
    args = parse_args()

    # Load config if specified
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    # Setup distributed
    rank, world_size, is_distributed, local_rank = setup_distributed()

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    is_imagenet = args.dataset == "imagenet"
    if is_imagenet:
        num_classes = 1000
        train_loader, test_loader = get_imagenet_loaders(
            data_dir=args.imagenet_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=is_distributed,
            seed=args.seed,
        )
        topk = (1, 5)
        # Default: step LR for ImageNet
        if args.lr_schedule == "cosine":
            args.lr_schedule = "step"
            args.step_milestones = "30,60,80"
    else:
        num_classes = 100 if args.dataset == "cifar100" else 10
        train_loader, test_loader = get_cifar_loaders(
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            distributed=is_distributed,
            seed=args.seed,
        )
        topk = (1,)

    # ---- Model ----
    model = get_model(args.arch, num_classes=num_classes)
    n_params = count_parameters(model)
    model = model.to(device)

    if rank == 0:
        print(f"[INFO] Model: {args.arch}, Params: {n_params:,}, Classes: {num_classes}")
        print(f"[INFO] Dataset: {args.dataset}, Epochs: {args.epochs}, BS: {args.batch_size}")
        print(f"[INFO] WD Method: {args.wd_method}, Lambda: {args.wd_lambda}")
        if is_distributed:
            print(f"[INFO] DDP: rank={rank}, world_size={world_size}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if hasattr(model, "module") else model

    # ---- WD Method ----
    wd_kwargs = {"lambda_base": args.wd_lambda}
    if args.wd_method == "EqWD":
        wd_kwargs.update({
            "beta": args.eqwd_beta,
            "ema_alpha": args.eqwd_ema_alpha,
            "layer_type_aware": args.eqwd_layer_aware,
        })
    elif args.wd_method == "CAWD":
        wd_kwargs["noise_sigma"] = args.cawd_noise_sigma

    wd_method = create_wd_method(args.wd_method, **wd_kwargs)

    # Detect normalized layers
    normalized_layers = detect_normalized_layers(raw_model)
    if rank == 0:
        print(f"[INFO] Normalized layers detected: {len(normalized_layers)}")

    # ---- Optimizer (NO built-in WD -- we apply it manually) ----
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.0,  # WD applied manually via wd_method
    )

    scheduler = get_lr_scheduler(optimizer, args, len(train_loader))
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if args.amp else None

    # ---- Diagnostics ----
    diagnostics = DiagnosticsLogger(
        log_interval=args.diag_interval,
        output_dir=str(output_dir),
        task_id=args.task_id,
    )

    # Progress reporter (for system monitor)
    progress_reporter = ProgressReporter(args.task_id, str(output_dir))
    if rank == 0:
        progress_reporter.write_pid()

    # Checkpoint epochs
    ckpt_epochs = set()
    if args.save_checkpoints:
        ckpt_epochs = {int(x) for x in args.save_checkpoints.split(",")}

    # ---- Training Loop ----
    global_step = 0
    best_acc = 0.0
    start_time = time.time()
    results_history = []

    try:
        for epoch in range(1, args.epochs + 1):
            if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_loss, train_acc, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                wd_method, normalized_layers, diagnostics, scaler,
                epoch, global_step, device, args, rank,
            )

            test_loss, test_top1, test_top5 = evaluate(
                model, test_loader, criterion, device, topk=topk
            )

            current_lr = optimizer.param_groups[0]["lr"]

            if rank == 0:
                # Log epoch metrics
                extra = {}
                if test_top5 is not None:
                    extra["test_top5"] = test_top5
                diagnostics.log_epoch(
                    epoch, train_loss, test_loss, test_top1,
                    train_acc=train_acc, lr=current_lr, extra=extra,
                )

                # Print progress
                msg = (f"Epoch {epoch}/{args.epochs} | "
                       f"LR: {current_lr:.6f} | "
                       f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                       f"Test Loss: {test_loss:.4f} Top1: {test_top1:.2f}%")
                if test_top5 is not None:
                    msg += f" Top5: {test_top5:.2f}%"
                print(msg)

                # Track best
                if test_top1 > best_acc:
                    best_acc = test_top1

                # Epoch result
                result = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_top1": test_top1,
                    "lr": current_lr,
                }
                if test_top5 is not None:
                    result["test_top5"] = test_top5
                results_history.append(result)

                # Report progress
                metric = {"test_top1": test_top1, "best_top1": best_acc}
                if test_top5 is not None:
                    metric["test_top5"] = test_top5
                progress_reporter.report_progress(
                    epoch, args.epochs, global_step, 0, train_loss, metric
                )

                # Save checkpoint
                if epoch in ckpt_epochs or epoch == args.epochs:
                    ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "args": vars(args),
                    }, ckpt_path)

        # ---- Finalize ----
        elapsed = time.time() - start_time

        if rank == 0:
            # Save final results
            final_results = {
                "task_id": args.task_id,
                "dataset": args.dataset,
                "arch": args.arch,
                "wd_method": args.wd_method,
                "wd_lambda": args.wd_lambda,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_params": n_params,
                "best_test_top1": best_acc,
                "final_test_top1": results_history[-1]["test_top1"],
                "final_train_loss": results_history[-1]["train_loss"],
                "final_train_acc": results_history[-1]["train_acc"],
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60,
                "history": results_history,
            }
            if is_imagenet:
                final_results["final_test_top5"] = results_history[-1].get("test_top5")

            # Add EqWD-specific params
            if args.wd_method == "EqWD":
                final_results["eqwd_beta"] = args.eqwd_beta
                final_results["eqwd_ema_alpha"] = args.eqwd_ema_alpha
                final_results["eqwd_layer_aware"] = args.eqwd_layer_aware

            results_path = output_dir / f"{args.task_id}_results.json"
            with open(results_path, "w") as f:
                json.dump(final_results, f, indent=2)

            # Save diagnostics
            diagnostics.save(str(output_dir / "diagnostics"))

            # Mark done
            progress_reporter.mark_done(
                status="success",
                summary=f"Best Top1: {best_acc:.2f}%, Elapsed: {elapsed/60:.1f}min",
                results=final_results,
            )

            print(f"\n[DONE] Best Top1: {best_acc:.2f}%, Time: {elapsed/60:.1f} min")
            print(f"[DONE] Results saved to {results_path}")

    except Exception as e:
        if rank == 0:
            progress_reporter.mark_done(
                status="failed",
                summary=f"Error at epoch {epoch if 'epoch' in dir() else '?'}: {str(e)}",
            )
        raise

    finally:
        if is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
