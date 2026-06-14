#!/usr/bin/env python3
"""
Wrapper to run train_dal_momentum via import (avoids Blackwell CUBLAS issue
that only manifests when running the script directly).
"""
import sys
sys.path.insert(0, '/home/ccwang/sibyl_system/projects/ttt-dlm/exp/code/dal')
import train_dal_momentum as tdm

# Use 200 samples (sufficient for 5000 steps with batch=2, wraps around)
tdm.NUM_TRAIN_SAMPLES = 200

try:
    success = tdm.main()
    sys.exit(0 if success else 1)
except RuntimeError as e:
    if "CUDA" in str(e) or "CUBLAS" in str(e):
        import traceback
        print(f"\nCUDA ERROR (recoverable): {e}")
        traceback.print_exc()
        tdm.mark_done("cuda_error", f"CUDA error: {str(e)[:200]}. Restart with checkpoint.")
        sys.exit(42)
    else:
        import traceback
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        tdm.mark_done("error", f"Fatal: {str(e)[:200]}")
        sys.exit(1)
except Exception as e:
    import traceback
    print(f"\nFATAL ERROR: {e}")
    traceback.print_exc()
    tdm.mark_done("error", f"Fatal: {str(e)[:200]}")
    sys.exit(1)
