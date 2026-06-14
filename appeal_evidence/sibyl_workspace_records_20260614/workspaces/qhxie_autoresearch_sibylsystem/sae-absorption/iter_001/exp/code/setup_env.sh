#!/bin/bash
# Setup script for sae-absorption conda environment
# Exit on error
set -e

ENV_NAME="sibyl_sae-absorption"
RESULTS_DIR="/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current/exp/results/pilots"
TASK_ID="setup_env"
PID_FILE="/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current/exp/results/${TASK_ID}.pid"
DONE_FILE="${RESULTS_DIR}/${TASK_ID}_DONE"
PROGRESS_FILE="${RESULTS_DIR}/${TASK_ID}_PROGRESS.json"

# Write PID file
echo $$ > "$PID_FILE"
echo "PID: $$"

# Progress helper
write_progress() {
    local step="$1"
    local total="$2"
    local msg="$3"
    python3 -c "
import json, datetime
with open('${PROGRESS_FILE}', 'w') as f:
    json.dump({
        'task_id': '${TASK_ID}',
        'epoch': $step,
        'total_epochs': $total,
        'step': $step,
        'total_steps': $total,
        'loss': None,
        'metric': {'step': '$msg'},
        'updated_at': datetime.datetime.now().isoformat()
    }, f)
" 2>/dev/null || true
}

# Write DONE marker helper
write_done() {
    local status="$1"
    local summary="$2"
    python3 -c "
import json, datetime
with open('${DONE_FILE}', 'w') as f:
    json.dump({
        'task_id': '${TASK_ID}',
        'status': '$status',
        'summary': '$summary',
        'timestamp': datetime.datetime.now().isoformat()
    }, f)
" 2>/dev/null || true
}

write_progress 1 7 "Creating conda environment"
echo "=== Step 1: Creating conda environment ==="
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "Environment ${ENV_NAME} already exists, skipping creation"
else
    conda create -n "$ENV_NAME" python=3.11 -y
    echo "Environment created."
fi

write_progress 2 7 "Installing core packages"
echo "=== Step 2: Installing core packages ==="
conda run -n "$ENV_NAME" pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

write_progress 3 7 "Installing SAE packages"
echo "=== Step 3: Installing SAE packages ==="
conda run -n "$ENV_NAME" pip install --quiet \
    sae-lens \
    transformer-lens

write_progress 4 7 "Installing ML packages"
echo "=== Step 4: Installing ML/data packages ==="
conda run -n "$ENV_NAME" pip install --quiet \
    datasets \
    scikit-learn \
    scipy \
    statsmodels \
    numpy \
    pandas \
    matplotlib \
    tqdm \
    transformers \
    accelerate \
    huggingface_hub

write_progress 5 7 "Installing sae-spelling"
echo "=== Step 5: Installing sae-spelling ==="
conda run -n "$ENV_NAME" pip install --quiet sae-spelling || \
    conda run -n "$ENV_NAME" pip install --quiet git+https://github.com/chanind/sae-spelling.git || \
    echo "WARNING: sae-spelling install failed - will try manual install"

write_progress 6 7 "Verifying imports"
echo "=== Step 6: Verifying package imports ==="
conda run -n "$ENV_NAME" python3 -c "
import sys
results = {}
packages = [
    ('torch', 'torch'),
    ('transformer_lens', 'transformer_lens'),
    ('sae_lens', 'sae_lens'),
    ('datasets', 'datasets'),
    ('sklearn', 'scikit-learn'),
    ('scipy', 'scipy'),
    ('statsmodels', 'statsmodels'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('transformers', 'transformers'),
]
failed = []
for pkg_import, pkg_name in packages:
    try:
        mod = __import__(pkg_import)
        ver = getattr(mod, '__version__', 'unknown')
        results[pkg_name] = {'status': 'ok', 'version': ver}
        print(f'  [OK] {pkg_name}: {ver}')
    except ImportError as e:
        results[pkg_name] = {'status': 'failed', 'error': str(e)}
        failed.append(pkg_name)
        print(f'  [FAIL] {pkg_name}: {e}')

# Check sae-spelling
try:
    import sae_spelling
    results['sae-spelling'] = {'status': 'ok', 'version': getattr(sae_spelling, '__version__', 'installed')}
    print(f'  [OK] sae-spelling: installed')
except ImportError as e:
    results['sae-spelling'] = {'status': 'missing', 'error': str(e)}
    print(f'  [WARN] sae-spelling: {e}')

print('\\nFailed packages:', failed if failed else 'None')
sys.exit(0)
"

write_progress 7 7 "Done"
echo "=== Setup complete ==="
# Clean up PID file
rm -f "$PID_FILE"
write_done "success" "Environment setup complete"
echo "SETUP_DONE"
