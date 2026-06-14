"""Sanity check training run for setup task."""
import os
import json
from datetime import datetime
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.models import resnet18
from torch import nn, optim

REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/augmentation-order"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
SHARED_DATASETS = f"{REMOTE_BASE}/shared/datasets"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/exp", exist_ok=True)

# Write PID file
pid_file = Path(RESULTS_DIR) / "setup.pid"
pid_file.write_text(str(os.getpid()))
print(f"PID file written: {pid_file} (PID={os.getpid()})")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Define transforms: conventional ordering Crop->Flip->ColorJitter
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(
    root=f"{SHARED_DATASETS}/cifar10", train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root=f"{SHARED_DATASETS}/cifar10", train=False, download=False, transform=transform_test)

# Use 1k subset for sanity check
torch.manual_seed(42)
train_sub = data.Subset(trainset, list(range(1000)))
test_sub  = data.Subset(testset,  list(range(500)))

trainloader = data.DataLoader(train_sub, batch_size=128, shuffle=True, num_workers=2)
testloader  = data.DataLoader(test_sub,  batch_size=128, shuffle=False, num_workers=2)

# ResNet-18
model = resnet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 1 epoch training
print("Training for 1 epoch on 1k subset...")
model.train()
train_loss = 0.0
for inputs, labels in trainloader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
print(f"Train loss: {train_loss/len(trainloader):.4f}")

# Validate
model.eval()
correct = total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

val_acc = correct / total
print(f"Sanity val_accuracy (1 epoch, 1k subset): {val_acc:.4f}")

# Verify all 6 orderings can be instantiated
print("Verifying 6 ordering pipelines...")
orderings = [
    ["crop", "flip", "cj"],  # order_0: conventional
    ["crop", "cj", "flip"],  # order_1
    ["flip", "crop", "cj"],  # order_2
    ["flip", "cj", "crop"],  # order_3
    ["cj", "crop", "flip"],  # order_4
    ["cj", "flip", "crop"],  # order_5: reversibility-first
]
op_map = {
    "crop": transforms.RandomCrop(32, padding=4),
    "flip": transforms.RandomHorizontalFlip(),
    "cj":   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
}
import PIL.Image
import numpy as np

for i, ordering in enumerate(orderings):
    ops = [op_map[o] for o in ordering]
    pipeline = transforms.Compose(ops + [transforms.ToTensor()])
    dummy = PIL.Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    out = pipeline(dummy)
    assert out.shape == (3, 32, 32), f"Unexpected output shape: {out.shape}"

print(f"All 6 orderings instantiated and verified successfully.")

pass_criteria_met = val_acc > 0.10
print(f"Pass criteria (val_acc > 10%): {'PASS' if pass_criteria_met else 'FAIL'} ({val_acc:.4f})")

# Write setup_done.json
result = {
    "task_id": "setup",
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "cifar10_path": f"{SHARED_DATASETS}/cifar10",
    "cifar100_path": f"{SHARED_DATASETS}/cifar100",
    "sanity_val_accuracy": float(val_acc),
    "pass_criteria_met": bool(pass_criteria_met),
    "orderings_verified": 6,
    "device": str(device),
    "cuda_available": torch.cuda.is_available(),
    "torch_version": torch.__version__,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
}
setup_done_path = f"{PROJECT_DIR}/exp/setup_done.json"
with open(setup_done_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"setup_done.json written to: {setup_done_path}")

# Write DONE marker
done_marker = Path(RESULTS_DIR) / "setup_DONE"
done_marker.write_text(json.dumps({
    "task_id": "setup",
    "status": "success",
    "summary": f"Setup complete. Val_accuracy={val_acc:.4f}. 6 orderings verified. CUDA={torch.cuda.is_available()}.",
    "timestamp": datetime.now().isoformat(),
}))
print(f"DONE marker written: {done_marker}")

# Remove PID file
if pid_file.exists():
    pid_file.unlink()
    print("PID file removed.")

print("\n=== SETUP TASK COMPLETE ===")
print(f"  CIFAR-10 path: {SHARED_DATASETS}/cifar10")
print(f"  CIFAR-100 path: {SHARED_DATASETS}/cifar100")
print(f"  Sanity accuracy: {val_acc:.4f} (> 10% required)")
print(f"  All 6 orderings: verified")
print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
