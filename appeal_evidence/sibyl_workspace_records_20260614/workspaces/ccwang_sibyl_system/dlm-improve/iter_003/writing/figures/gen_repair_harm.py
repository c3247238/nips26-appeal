from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent
WORKSPACE = FIG_DIR.parent.parent
sys.path.insert(0, str(FIG_DIR))

from style_config import COLORS, FIG_WIDTH_FULL, apply_global_style  # noqa: E402


def main() -> None:
    apply_global_style(plt)

    csv_path = WORKSPACE / "exp/pilot_evidence_closure_v1/analysis/packaging/repair_harm_table.csv"
    out_path = FIG_DIR / "repair_harm.pdf"

    order = [
        "dnb84_vs_dnb64",
        "card84_vs_dnb64",
        "rand84_vs_dnb64",
        "card84_vs_dnb84",
        "card84_vs_rand84",
    ]
    label_map = {
        "dnb84_vs_dnb64": "DNB-84 vs DNB-64",
        "card84_vs_dnb64": "CARD-84 vs DNB-64",
        "rand84_vs_dnb64": "RAND-84 vs DNB-64",
        "card84_vs_dnb84": "CARD-84 vs DNB-84",
        "card84_vs_rand84": "CARD-84 vs RAND-84",
    }
    rows_by_comparison: dict[str, dict[str, int | str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["dataset"] != "gsm8k":
                continue
            rows_by_comparison[row["comparison"]] = {
                "comparison": row["comparison"],
                "fixed": int(row["fixed"]),
                "harmed": int(row["harmed"]),
                "unchanged_correct": int(row["unchanged_correct"]),
                "unchanged_wrong": int(row["unchanged_wrong"]),
                "net_repaired": int(row["net_repaired"]),
            }
    rows = [rows_by_comparison[key] for key in order]

    stacks = [
        ("unchanged_wrong", COLORS["neutral_light"], "Unchanged wrong"),
        ("unchanged_correct", COLORS["neutral_dark"], "Unchanged correct"),
        ("fixed", COLORS["success"], "Fixed"),
        ("harmed", COLORS["harm"], "Harmed"),
    ]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, 4.8))
    bottom = [0] * len(rows)

    for key, color, label in stacks:
        values = [int(row[key]) for row in rows]
        ax.bar(
            range(len(rows)),
            values,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([label_map[str(row["comparison"])] for row in rows], rotation=15, ha="right")
    ax.set_ylabel("Audited sample count")
    ax.set_title("GSM8K repair and harm counts under stronger controls")
    ax.set_ylim(0, 50)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx, row in enumerate(rows):
        ax.text(
            idx,
            51.2,
            f"net {int(row['net_repaired']):+d}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
