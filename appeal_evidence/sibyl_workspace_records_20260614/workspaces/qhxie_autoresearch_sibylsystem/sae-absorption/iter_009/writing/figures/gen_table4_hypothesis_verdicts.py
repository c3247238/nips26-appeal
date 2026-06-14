"""Generate Table 4: Hypothesis Verdict Summary as a PDF figure.

Produces a publication-quality table summarizing all 9 hypotheses with
verdict, key metric, confidence, and paper section reference.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Load consolidation summary ──────────────────────────────────────
workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
consolidation_path = os.path.join(workspace, "exp", "results", "consolidation_summary.json")

with open(consolidation_path, "r") as f:
    data = json.load(f)

verdicts = data["hypothesis_verdicts"]

# ── Build table data ────────────────────────────────────────────────
rows = []
for v in verdicts:
    hyp = v["hypothesis"]
    name = v["name"]
    verdict = v["verdict"]
    confidence = v.get("confidence", "")
    section = v.get("paper_section", "")

    # Extract a concise key metric string
    stats = v.get("statistics", {})
    key_evidence = v.get("key_evidence", "")

    # Build concise metric summaries
    if hyp == "H1":
        metric = "Kruskal-Wallis p=7.4e-66"
    elif hyp == "H2'":
        metric = "No simple ordering; 4x range"
    elif hyp == "H3":
        metric = r"$\chi^2$=91.5, p=1.0e-19"
    elif hyp == "H4":
        metric = r"$\rho$=0.116, AUROC=0.571"
    elif hyp == "H5":
        metric = r"T(G) $\rho$=-0.20, 50% conc."
    elif hyp == "H6":
        metric = "Arch p=0.50; Hier p=0.005"
    elif hyp == "H7":
        metric = "d=1.33, p=0.000218 (FL only)"
    elif hyp == "H8":
        metric = r"0% benign; $|\Delta|$=3.98"
    elif hyp == "H9":
        metric = r"$\rho$=0.250, R$^2$=0.088"
    else:
        metric = ""

    rows.append([hyp, name, verdict, metric, confidence, section])

# ── Define verdict colors ───────────────────────────────────────────
verdict_colors = {
    "STRONGLY_SUPPORTED": "#2ecc71",
    "SUPPORTED": "#2ecc71",
    "SUPPORTED_FIRSTLETTER_ONLY": "#f39c12",
    "PARTIALLY_SUPPORTED": "#f39c12",
    "REFUTED": "#e74c3c",
    "FALSIFIED": "#e74c3c",
    "NOT_SUPPORTED": "#95a5a6",
    "DEFINITIVE_NEGATIVE": "#e74c3c",
}

# ── Render table ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.axis("off")

col_labels = ["Hyp.", "Name", "Verdict", "Key Metric", "Conf.", "Section"]
col_widths = [0.06, 0.22, 0.16, 0.24, 0.10, 0.12]

cell_text = [[r[0], r[1], r[2].replace("_", " "), r[3], r[4], r[5]] for r in rows]

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    colWidths=col_widths,
    loc="center",
    cellLoc="left",
)

table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1.0, 1.55)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold", fontsize=8)

# Style verdict cells with color coding
for i, row in enumerate(rows):
    verdict = row[2]
    color = verdict_colors.get(verdict, "#ecf0f1")
    # Color the verdict cell
    cell = table[i + 1, 2]
    cell.set_facecolor(color)
    cell.set_text_props(color="white" if verdict in ["STRONGLY_SUPPORTED", "SUPPORTED", "REFUTED", "FALSIFIED", "DEFINITIVE_NEGATIVE"] else "#2c3e50",
                        fontweight="bold", fontsize=7)

    # Alternate row shading for other cells
    bg = "#f8f9fa" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        if j != 2:  # skip verdict column
            table[i + 1, j].set_facecolor(bg)

# Tighten
plt.tight_layout(pad=0.5)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table4_hypothesis_verdicts.pdf")
fig.savefig(out_path, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Table 4 saved to {out_path}")
print(f"  Rows: {len(rows)}")
print(f"  Verdicts: {[r[2] for r in rows]}")
