from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from style_config import COLORS, FIG_WIDTH, FONT_SIZE


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "exp" / "results" / "espd_fullscale_bundle_v1.json"
OUT_PATH = ROOT / "writing" / "figures" / "fig2_quality_speed.pdf"


def main() -> None:
    data = json.loads(RESULT_PATH.read_text())
    points = data["decision_matrix"]
    labels = [p["method"] for p in points]
    x = [p["speed_at_equal_quality_band"] for p in points]
    y = [p["quality_at_equal_compute"] for p in points]
    colors = []
    for label in labels:
        if label == "cand_espd":
            colors.append(COLORS["ours"])
        elif label == "ESPD-FixedFrontier":
            colors.append(COLORS["ablation"])
        else:
            colors.append(COLORS["baseline"])

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.5))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.scatter(x, y, s=70, c=colors)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Speed at equal-quality band (tok/s)")
    ax.set_ylabel("Quality at equal compute")
    ax.set_title("Quality-Speed Positioning Under a Unified Runtime Contract")
    ax.margins(0.1)
    fig.tight_layout()
    fig.savefig(OUT_PATH)


if __name__ == "__main__":
    main()
