"""
Tier 1 Full: ResNet-18 x CIFAR-100
FULL SCALE: 200 epochs, 5 seeds [42..46], full 50k training set.
6 permutations of {RandomCrop, RandomHorizontalFlip, ColorJitter}.
SGD + cosine annealing, lr=0.1, momentum=0.9, wd=5e-4.
"""
import os, sys, json, time, random
import numpy as np
from datetime import datetime
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as transforms, torchvision.models as models

TASK_ID = "tier1_resnet18_cifar100_full"
REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT = "augmentation-order"
RESULTS_DIR = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/results")
FULL_DIR = RESULTS_DIR / "full"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints" / TASK_ID
GPU_PROGRESS_PATH = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/gpu_progress.json")
EXPERIMENT_STATE_PATH = Path(f"{REMOTE_BASE}/projects/{PROJECT}/exp/experiment_state.json")
CIFAR100_PATH = f"{REMOTE_BASE}/shared/datasets/cifar100"

NUM_EPOCHS = 200
BATCH_SIZE = 256
SEEDS = [42, 43, 44, 45, 46]
DEVICE = "cuda:0"
PLANNED_MIN = 55

ORDERINGS = {
    "order_0": ["crop", "flip", "cj"],
    "order_1": ["crop", "cj", "flip"],
    "order_2": ["flip", "crop", "cj"],
    "order_3": ["flip", "cj", "crop"],
    "order_4": ["cj", "crop", "flip"],
    "order_5": ["cj", "flip", "crop"],
}
ORDERING_LABELS = {
    "order_0": "Crop->Flip->CJ", "order_1": "Crop->CJ->Flip",
    "order_2": "Flip->Crop->CJ", "order_3": "Flip->CJ->Crop",
    "order_4": "CJ->Crop->Flip", "order_5": "CJ->Flip->Crop",
}

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def build_transform(ordering_ops):
    ops_map = {
        "crop": transforms.RandomCrop(32, padding=4),
        "flip": transforms.RandomHorizontalFlip(p=0.5),
        "cj":   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    }
    return transforms.Compose([ops_map[op] for op in ordering_ops] + [
        transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def get_datasets(ordering_ops):
    tv = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    tr = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=True,  download=False, transform=build_transform(ordering_ops))
    va = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=False, download=False, transform=tv)
    return tr, va

def probe_batch_size(device, start=512, min_bs=64):
    import gc
    model = models.resnet18(num_classes=100).to(device); model.eval()
    hi, best, lo = start, min_bs, min_bs
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache(); gc.collect()
            with torch.no_grad(): _ = model(torch.randn(mid,3,32,32).to(device))
            best = mid; lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            hi = mid - 1; torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()
    return best

def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss=correct=total=0
    for x,y in loader:
        x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
        optimizer.zero_grad(); out = model(x); loss = criterion(out,y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()*x.size(0)
        correct += out.max(1)[1].eq(y).sum().item(); total += x.size(0)
    return total_loss/total, correct/total

def eval_epoch(model, loader, criterion, device):
    model.eval(); total_loss=correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
            out = model(x); loss = criterion(out,y)
            total_loss += loss.item()*x.size(0)
            correct += out.max(1)[1].eq(y).sum().item(); total += x.size(0)
    return total_loss/total, correct/total

def write_pid():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR/f"{TASK_ID}.pid").write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID {os.getpid()}")

def write_progress(run_idx, total_runs, epoch=0, ordering_key=None, metric=None):
    (RESULTS_DIR/f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id":TASK_ID,"epoch":epoch,"total_epochs":NUM_EPOCHS,
        "run_idx":run_idx,"total_runs":total_runs,"ordering":ordering_key,
        "metric":metric or {},"updated_at":datetime.now().isoformat()}))

def mark_done(status="success", summary=""):
    p = RESULTS_DIR/f"{TASK_ID}.pid"
    if p.exists(): p.unlink()
    (RESULTS_DIR/f"{TASK_ID}_DONE").write_text(json.dumps(
        {"task_id":TASK_ID,"status":status,"summary":summary,"timestamp":datetime.now().isoformat()}))
    print(f"[{TASK_ID}] DONE: {status}")

def update_state(status):
    try:
        data = json.loads(EXPERIMENT_STATE_PATH.read_text()) if EXPERIMENT_STATE_PATH.exists() else {"schema_version":1,"tasks":{}}
        data["tasks"].setdefault(TASK_ID,{})["status"] = status
        if status=="completed": data["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
        EXPERIMENT_STATE_PATH.write_text(json.dumps(data,indent=2))
    except Exception as e: print(f"Warning: {e}")

def update_gpu_progress(status, actual_min, start_time, end_time, cfg):
    try:
        data = json.loads(GPU_PROGRESS_PATH.read_text()) if GPU_PROGRESS_PATH.exists() else {"completed":[],"failed":[],"running":{},"timings":{}}
        (data["completed"] if status=="success" else data["failed"]).append(TASK_ID) if TASK_ID not in data.get("completed",[]) else None
        data.get("running",{}).pop(TASK_ID,None)
        data["timings"][TASK_ID] = {"planned_min":PLANNED_MIN,"actual_min":round(actual_min,1),"start_time":start_time,"end_time":end_time,"config_snapshot":cfg}
        GPU_PROGRESS_PATH.write_text(json.dumps(data,indent=2))
    except Exception as e: print(f"Warning: {e}")

def run_single(ordering_key, ordering_ops, seed, device, batch_size):
    set_seed(seed)
    tr, va = get_datasets(ordering_ops)
    trl = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, persistent_workers=True)
    val = DataLoader(va, batch_size=512,        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model = models.resnet18(num_classes=100).to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    ckpt_dir = CHECKPOINTS_DIR/ordering_key; ckpt_dir.mkdir(parents=True,exist_ok=True)
    resume_ckpt = ckpt_dir/f"seed_{seed}_latest.pt"
    per_epoch = []; start_epoch = 1
    if resume_ckpt.exists():
        try:
            ck = torch.load(resume_ckpt, map_location=device)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sch.load_state_dict(ck["scheduler"])
            start_epoch = ck["epoch"]+1; per_epoch = ck.get("per_epoch",[])
            print(f"    [Resume] epoch {ck['epoch']} for {ordering_key}/seed={seed}")
        except Exception as e: print(f"    [Resume] failed: {e}"); start_epoch=1; per_epoch=[]
    for epoch in range(start_epoch, NUM_EPOCHS+1):
        tl,ta = train_epoch(model,trl,opt,crit,device)
        vl,va_ = eval_epoch(model,val,crit,device)
        sch.step()
        per_epoch.append({"epoch":epoch,"train_loss":round(tl,4),"train_acc":round(ta,4),"val_loss":round(vl,4),"val_acc":round(va_,4)})
        if epoch%10==0 or epoch==NUM_EPOCHS:
            torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sch.state_dict(),"per_epoch":per_epoch,"ordering_key":ordering_key,"seed":seed},resume_ckpt)
            print(f"    epoch {epoch:3d}/{NUM_EPOCHS}: val_acc={va_:.4f}")
    final_acc = per_epoch[-1]["val_acc"]
    torch.save({"epoch":NUM_EPOCHS,"model":model.state_dict(),"ordering_key":ordering_key,"ordering_ops":ordering_ops,"seed":seed,"final_val_acc":final_acc}, ckpt_dir/f"seed_{seed}_final.pt")
    if resume_ckpt.exists(): resume_ckpt.unlink()
    return final_acc, per_epoch

def main():
    FULL_DIR.mkdir(parents=True,exist_ok=True); CHECKPOINTS_DIR.mkdir(parents=True,exist_ok=True)
    device = torch.device(DEVICE)
    print(f"[{TASK_ID}] FULL SCALE: {NUM_EPOCHS} epochs, seeds={SEEDS}, CIFAR-100")
    if torch.cuda.is_available():
        print(f"[{TASK_ID}] GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    write_pid()
    batch_size = min(probe_batch_size(device, start=1024, min_bs=64), 512)
    print(f"[{TASK_ID}] batch_size={batch_size}")
    start_time = datetime.now().isoformat(); start_ts = time.time()
    results = {}; total_runs = len(ORDERINGS)*len(SEEDS); run_count = 0; errors = []
    for ordering_key, ordering_ops in ORDERINGS.items():
        results[ordering_key] = {"label":ORDERING_LABELS[ordering_key],"ops":ordering_ops,"per_seed":{}}
        for seed in SEEDS:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {ordering_key} ({ORDERING_LABELS[ordering_key]}) seed={seed}")
            write_progress(run_count-1, total_runs, epoch=0, ordering_key=ordering_key)
            t0 = time.time()
            try:
                final_acc, per_epoch = run_single(ordering_key, ordering_ops, seed, device, batch_size)
                elapsed = time.time()-t0
                print(f"  -> val_acc={final_acc:.4f} ({elapsed/60:.1f} min)")
                results[ordering_key]["per_seed"][str(seed)] = {"final_val_acc":final_acc,"per_epoch":per_epoch,"elapsed_sec":round(elapsed,1)}
            except Exception as e:
                import traceback; elapsed=time.time()-t0
                print(f"  ERROR: {e}\n{traceback.format_exc()}")
                errors.append({"ordering":ordering_key,"seed":seed,"error":str(e)})
                results[ordering_key]["per_seed"][str(seed)] = {"final_val_acc":0.0,"per_epoch":[],"error":str(e),"elapsed_sec":round(elapsed,1)}
            write_progress(run_count, total_runs, epoch=NUM_EPOCHS, ordering_key=ordering_key,
                           metric={"val_acc":results[ordering_key]["per_seed"][str(seed)]["final_val_acc"]})
    for k in results:
        accs = [results[k]["per_seed"][str(s)]["final_val_acc"] for s in SEEDS if str(s) in results[k]["per_seed"] and "error" not in results[k]["per_seed"][str(s)]]
        results[k]["mean_val_acc"] = round(float(np.mean(accs)),4) if accs else 0.0
        results[k]["std_val_acc"]  = round(float(np.std(accs)),4)  if accs else 0.0
        results[k]["n_seeds_ok"]   = len(accs)
        print(f"  {k}: mean={results[k]['mean_val_acc']:.4f} std={results[k]['std_val_acc']:.4f}")
    all_means = [results[k]["mean_val_acc"] for k in results]
    valid_means = [v for v in all_means if v>0]
    spread = round(max(valid_means)-min(valid_means),4) if len(valid_means)>=2 else 0.0
    best_k  = max(results, key=lambda k: results[k]["mean_val_acc"])
    worst_k = min(results, key=lambda k: results[k]["mean_val_acc"])
    flip_first_mean = np.mean([results[k]["mean_val_acc"] for k in ["order_2","order_3"]])
    crop_first_mean = np.mean([results[k]["mean_val_acc"] for k in ["order_0","order_1"]])
    pass_ok = spread>0.003 and results[best_k]["mean_val_acc"]>0.70 and len(errors)<total_runs//2
    summary = {
        "task_id":TASK_ID,"mode":"full","timestamp":datetime.now().isoformat(),
        "spread":spread,"spread_pct":round(spread*100,2),
        "best_ordering":best_k,"worst_ordering":worst_k,
        "best_label":ORDERING_LABELS[best_k],"worst_label":ORDERING_LABELS[worst_k],
        "best_acc":results[best_k]["mean_val_acc"],"worst_acc":results[worst_k]["mean_val_acc"],
        "flip_first_mean":round(float(flip_first_mean),4),"crop_first_mean":round(float(crop_first_mean),4),
        "h4_flip_wins":bool(flip_first_mean>crop_first_mean),
        "pass_criteria_met":pass_ok,"total_runs":total_runs,"errors":errors,"n_errors":len(errors),
        "orderings":results,"seeds":SEEDS,"epochs":NUM_EPOCHS,"batch_size":batch_size,
        "checkpoint_dir":str(CHECKPOINTS_DIR),
    }
    out = FULL_DIR/"tier1_resnet18_cifar100.json"
    out.write_text(json.dumps(summary,indent=2))
    print(f"\n[{TASK_ID}] Spread: {spread:.4f} ({spread*100:.2f}%), Best: {best_k}={results[best_k]['mean_val_acc']:.4f}, Errors: {len(errors)}")
    end_time = datetime.now().isoformat(); actual_min = round((time.time()-start_ts)/60,1)
    cfg = {"model":"resnet18","dataset":"cifar100","batch_size":batch_size,"epochs":NUM_EPOCHS,"seeds":SEEDS,"orderings":6,"mode":"full"}
    final_status = "success" if pass_ok else "partial"
    update_gpu_progress(final_status, actual_min, start_time, end_time, cfg)
    update_state("completed")
    mark_done(final_status, f"spread={spread:.4f} best={best_k} errors={len(errors)}")
    print(f"[{TASK_ID}] Done in {actual_min:.1f} min.")

if __name__ == "__main__":
    main()
