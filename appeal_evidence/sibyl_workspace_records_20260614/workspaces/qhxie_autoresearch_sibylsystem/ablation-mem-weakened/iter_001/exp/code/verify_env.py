"""Environment verification script for Feature Absorption experiments."""
import sys
import json
from pathlib import Path

def check_import(module_name, attr=None):
    """Check if a module can be imported."""
    try:
        mod = __import__(module_name)
        if attr:
            return hasattr(mod, attr), getattr(mod, attr, None)
        return True, mod
    except ImportError as e:
        return False, str(e)

def main():
    results = {
        "python_version": sys.version,
        "checks": {},
        "gpu": {},
        "all_pass": True
    }

    # Core imports
    checks = [
        ("torch", None),
        ("transformers", None),
        ("datasets", None),
        ("accelerate", None),
        ("sae_lens", None),
        ("transformer_lens", None),
        ("sae_bench", None),
        ("numpy", None),
        ("scipy", None),
        ("sklearn", None),
        ("pandas", None),
        ("matplotlib", None),
        ("seaborn", None),
        ("tqdm", None),
    ]

    for mod, attr in checks:
        ok, val = check_import(mod, attr)
        results["checks"][mod] = {"ok": ok}
        if ok and attr is None:
            try:
                ver = getattr(val, "__version__", "unknown")
                results["checks"][mod]["version"] = ver
            except:
                pass
        if not ok:
            results["checks"][mod]["error"] = val
            results["all_pass"] = False

    # GPU check
    try:
        import torch
        results["gpu"]["cuda_available"] = torch.cuda.is_available()
        results["gpu"]["cuda_version"] = torch.version.cuda
        results["gpu"]["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            results["gpu"][f"device_{i}"] = torch.cuda.get_device_name(i)
            results["gpu"][f"device_{i}_memory_mb"] = torch.cuda.get_device_properties(i).total_memory // (1024**2)
    except Exception as e:
        results["gpu"]["error"] = str(e)
        results["all_pass"] = False

    # SAE-specific check: can we load a simple SAE config
    try:
        from sae_lens import SAE
        results["checks"]["sae_lens_load"] = {"ok": True}
    except Exception as e:
        results["checks"]["sae_lens_load"] = {"ok": False, "error": str(e)}
        results["all_pass"] = False

    # Write results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "env_verification.json"
    output_path.write_text(json.dumps(results, indent=2))

    # Print summary
    print("=" * 60)
    print("Environment Verification Summary")
    print("=" * 60)
    for mod, info in results["checks"].items():
        status = "PASS" if info["ok"] else "FAIL"
        ver = info.get("version", "")
        print(f"  {mod:20s} [{status}] {ver}")
    print("-" * 60)
    print(f"GPU: {results['gpu'].get('device_count', 0)} device(s)")
    for i in range(results['gpu'].get('device_count', 0)):
        name = results['gpu'].get(f'device_{i}', 'unknown')
        mem = results['gpu'].get(f'device_{i}_memory_mb', 0)
        print(f"  GPU {i}: {name} ({mem} MiB)")
    print("-" * 60)
    print(f"Overall: {'ALL PASS' if results['all_pass'] else 'SOME CHECKS FAILED'}")
    print(f"Results saved to: {output_path}")

    return 0 if results["all_pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
