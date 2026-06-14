#!/bin/bash
# Setup script for augmentation-order project
set -e

REMOTE_BASE="/home/qhxie/sibyl_system"
PROJECT="augmentation-order"
PROJECT_DIR="${REMOTE_BASE}/projects/${PROJECT}"
CONDA_BASE="/home/qhxie/miniforge3"
ENV_NAME="sibyl_${PROJECT}"
RESULTS_DIR="${PROJECT_DIR}/exp/results"
SHARED_DATASETS="${REMOTE_BASE}/shared/datasets"

echo "=== Step 1: Create conda environment ==="
if ! ${CONDA_BASE}/bin/conda env list | grep -q "^${ENV_NAME}"; then
    ${CONDA_BASE}/bin/conda create -n ${ENV_NAME} python=3.10 -y
    echo "Environment created: ${ENV_NAME}"
else
    echo "Environment already exists: ${ENV_NAME}"
fi

echo "=== Step 2: Install dependencies ==="
${CONDA_BASE}/bin/conda run -n ${ENV_NAME} pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
${CONDA_BASE}/bin/conda run -n ${ENV_NAME} pip install timm scipy numpy matplotlib -q
echo "Dependencies installed."

echo "=== Step 3: Verify install ==="
${CONDA_BASE}/bin/conda run -n ${ENV_NAME} python -c "
import torch, torchvision, timm, scipy, numpy
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('timm:', timm.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "=== Step 4: Download CIFAR-10 and CIFAR-100 ==="
mkdir -p ${SHARED_DATASETS}/cifar10 ${SHARED_DATASETS}/cifar100
${CONDA_BASE}/bin/conda run -n ${ENV_NAME} python -c "
import torchvision
import os

cifar10_path = '${SHARED_DATASETS}/cifar10'
cifar100_path = '${SHARED_DATASETS}/cifar100'

print('Downloading CIFAR-10...')
torchvision.datasets.CIFAR10(root=cifar10_path, train=True, download=True)
torchvision.datasets.CIFAR10(root=cifar10_path, train=False, download=True)
print('CIFAR-10 done.')

print('Downloading CIFAR-100...')
torchvision.datasets.CIFAR100(root=cifar100_path, train=True, download=True)
torchvision.datasets.CIFAR100(root=cifar100_path, train=False, download=True)
print('CIFAR-100 done.')
"

echo "=== Step 5: Sanity training run ==="
${CONDA_BASE}/bin/conda run -n ${ENV_NAME} python - << 'PYEOF'
import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from pathlib import Path

REMOTE_BASE = "/home/qhxie/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/augmentation-order"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
SHARED_DATASETS = f"{REMOTE_BASE}/shared/datasets"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Write PID file
pid_file = Path(RESULTS_DIR) / "setup.pid"
pid_file.write_text(str(os.getpid()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
trainset = torchvision.datasets.CIFAR10(root=f"{SHARED_DATASETS}/cifar10", train=True,
                                         download=False, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root=f"{SHARED_DATASETS}/cifar10", train=False,
                                         download=False, transform=transform_test)

# Use 1k subset for sanity check
import torch.utils.data as data
train_sub = data.Subset(trainset, list(range(1000)))
test_sub  = data.Subset(testset,  list(range(500)))

trainloader = data.DataLoader(train_sub, batch_size=128, shuffle=True,  num_workers=2)
testloader  = data.DataLoader(test_sub,  batch_size=128, shuffle=False, num_workers=2)

# ResNet-18
from torchvision.models import resnet18
model = resnet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

# 1 epoch
model.train()
for inputs, labels in trainloader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

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
orderings = [
    ["crop", "flip", "cj"],
    ["crop", "cj", "flip"],
    ["flip", "crop", "cj"],
    ["flip", "cj", "crop"],
    ["cj", "crop", "flip"],
    ["cj", "flip", "crop"],
]
op_map = {
    "crop": transforms.RandomCrop(32, padding=4),
    "flip": transforms.RandomHorizontalFlip(),
    "cj":   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
}
for ordering in orderings:
    ops = [op_map[o] for o in ordering]
    pipeline = transforms.Compose(ops + [transforms.ToTensor()])
    # Test on a dummy image
    import PIL.Image
    import numpy as np
    dummy = PIL.Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    _ = pipeline(dummy)

print(f"All 6 orderings instantiated successfully.")
assert val_acc > 0.10, f"val_accuracy {val_acc} < 0.10 random baseline!"

# Write setup_done.json
result = {
    "task_id": "setup",
    "status": "success",
    "cifar10_path": f"{SHARED_DATASETS}/cifar10",
    "cifar100_path": f"{SHARED_DATASETS}/cifar100",
    "sanity_val_accuracy": val_acc,
    "pass_criteria_met": val_acc > 0.10,
    "orderings_verified": 6,
    "device": str(device),
    "cuda_available": torch.cuda.is_available(),
}
import json
with open(f"{PROJECT_DIR}/exp/setup_done.json", "w") as f:
    json.dump(result, f, indent=2)
print("setup_done.json written.")

# Write DONE marker
from datetime import datetime
done_marker = Path(RESULTS_DIR) / "setup_DONE"
done_marker.write_text(json.dumps({
    "task_id": "setup",
    "status": "success",
    "summary": f"Setup complete. Val_accuracy={val_acc:.4f}. 6 orderings verified.",
    "timestamp": datetime.now().isoformat(),
}))
# Remove PID file
if pid_file.exists():
    pid_file.unlink()

print("DONE marker written. Setup complete!")
PYEOF

echo "=== Setup script finished ==="
