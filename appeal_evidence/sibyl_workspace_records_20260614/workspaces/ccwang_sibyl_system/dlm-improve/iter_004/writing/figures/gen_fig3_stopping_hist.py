from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style_config import COLORS, FIG_WIDTH, FONT_SIZE


ROOT = Path(__file__).resolve().parents[2]
CAND_PATH = ROOT / "exp" / "results" / "espd_gsm8k_full_v1.json"
SHAM_PATH = ROOT / "exp" / "results" / "espd_fixed_frontier_gsm8k_full_v1.json"
OUT_PATH = ROOT / "writing" / "figures" / "fig3_stopping_hist.pdf"


def main() -> None:
    cand = json.loads(CAND_PATH.read_text())["metrics"]["stopped_step_histogram"]
    sham = json.loads(SHAM_PATH.read_text())["metrics"]["stopped_step_histogram"]
    steps = [1, 2, 3]
    cand_vals = [cand.get(str(s), 0) for s in steps]
    sham_vals = [sham.get(str(s), 0) for s in steps]

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.5))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    x = np.arange(len(steps))
    width = 0.34
    ax.bar(x - width / 2, cand_vals, width, label="cand_espd", color=COLORS["ours"])
    ax.bar(x + width / 2, sham_vals, width, label="ESPD-FixedFrontier", color=COLORS["ablation"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("Stopped step")
    ax.set_ylabel("Number of samples")
    ax.set_title("Stopping Behavior of the Candidate and the Sham")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH)


if __name__ == "__main__":
    main()
