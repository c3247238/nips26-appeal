from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent
WORKSPACE = FIG_DIR.parent.parent
sys.path.insert(0, str(FIG_DIR))

from style_config import COLORS, FIG_WIDTH_FULL, apply_global_style  # noqa: E402


def main() -> None:
    apply_global_style(plt)

    csv_path = WORKSPACE / "exp/pilot_evidence_closure_v1/analysis/packaging/harm_profile_table.csv"
    out_path = FIG_DIR / "mbpp_harm_profile.pdf"

    failure_order = [
        "NameError",
        "SyntaxError",
        "IndentationError",
        "AssertionError",
        "TypeError",
        "other",
    ]
    arm_order = ["dnb64", "dnb84", "card84", "rand84"]
    arm_labels = {
        "dnb64": "DNB-64",
        "dnb84": "DNB-84",
        "card84": "CARD-84",
        "rand84": "RAND-84",
    }
    arm_colors = {
        "dnb64": COLORS["baseline"],
        "dnb84": COLORS["neutral_dark"],
        "card84": COLORS["card"],
        "rand84": COLORS["sham"],
    }

    pivot = {failure: {arm: 0 for arm in arm_order} for failure in failure_order}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            failure = row["failure_mode"]
            arm = row["arm"]
            if failure in pivot and arm in pivot[failure]:
                pivot[failure][arm] = int(row["count"])

    x = np.arange(len(failure_order))
    width = 0.2

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, 4.6))

    for offset, arm in enumerate(arm_order):
        ax.bar(
            x + (offset - 1.5) * width,
            [pivot[failure][arm] for failure in failure_order],
            width=width,
            color=arm_colors[arm],
            edgecolor="white",
            linewidth=0.8,
            label=arm_labels[arm],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(failure_order, rotation=20, ha="right")
    ax.set_ylabel("Failure count on MBPP")
    ax.set_title("MBPP failure modes localize harm without separating CARD-84 from RAND-84")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
