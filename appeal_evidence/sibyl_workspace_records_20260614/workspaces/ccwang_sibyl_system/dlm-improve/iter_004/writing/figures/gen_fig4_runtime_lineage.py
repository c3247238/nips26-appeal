from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style_config import COLORS, FIG_WIDTH_FULL, FONT_SIZE


ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "exp" / "results" / "espd_fullscale_bundle_v1.json"
CONTROLS_PATH = ROOT / "exp" / "results" / "gsm8k_controls_full_v1.json"
OUT_PATH = ROOT / "writing" / "figures" / "fig4_runtime_lineage.pdf"


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    controls = json.loads(CONTROLS_PATH.read_text())

    labels = ["cand_espd", "ESPD-FixedFrontier", "CARD-84", "RAND-84"]
    batch = [
        bundle["runtime"]["candidate_effective_batch_size"],
        bundle["runtime"]["fixed_frontier_effective_batch_size"],
        bundle["runtime"]["shared_controls_effective_batch_size"],
        bundle["runtime"]["shared_controls_effective_batch_size"],
    ]
    vram = [
        bundle["runtime"]["candidate_peak_vram_mb"],
        bundle["runtime"]["fixed_frontier_peak_vram_mb"],
        bundle["runtime"]["shared_controls_peak_vram_mb"],
        bundle["runtime"]["shared_controls_peak_vram_mb"],
    ]
    latency = [
        next(x for x in bundle["decision_matrix"] if x["method"] == "cand_espd")["wall_clock_sec"],
        next(x for x in bundle["decision_matrix"] if x["method"] == "ESPD-FixedFrontier")["wall_clock_sec"],
        controls["arms"]["card84"]["latency_sec"],
        controls["arms"]["rand84"]["latency_sec"],
    ]

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_FULL, 3.8))
    palettes = [COLORS["ours"], COLORS["ablation"], COLORS["baseline"], COLORS["baseline"]]
    x = np.arange(len(labels))

    axes[0].bar(x, batch, color=palettes)
    axes[0].set_title("Effective batch size")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    axes[1].bar(x, vram, color=palettes)
    axes[1].set_title("Peak VRAM (MB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    axes[2].bar(x, latency, color=palettes)
    axes[2].set_title("Wall-clock latency (s)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_PATH)


if __name__ == "__main__":
    main()
