#!/usr/bin/env python3
"""QC figure for the MD hand-off: shows that the matched nonbinders overlap the
binders in Boltz (ligand pLDDT + geometry) while the full NB pool separates."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
LONG = f"{ROOT}/results/md_handoff/md_handoff_figure_data_long.csv"
OUT = f"{ROOT}/results/md_handoff/md_handoff_qc.png"

rows = list(csv.DictReader(open(LONG)))
LIGS = ["LCAM", "LCA3S"]
TITLE = {"LCAM": "LCA (LCAM)", "LCA3S": "LCA-3-S (LCA3S)"}


def sub(lig, cls, insel):
    xs, ys = [], []
    for r in rows:
        if r["ligand"] != lig:
            continue
        if r["class"] != cls:
            continue
        if insel is not None and int(r["in_selection"]) != insel:
            continue
        xs.append(float(r["plddt_ligand"]))
        ys.append(float(r["geometry_score"]))
    return np.array(xs), np.array(ys)


fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, lig in zip(axes, LIGS):
    # unselected nonbinders (the pool we did NOT send)
    xn, yn = sub(lig, "nonbinder", 0)
    ax.scatter(xn, yn, s=10, c="#cfcfcf", alpha=0.5, label=f"NB pool, not sent (n={len(xn)})", zorder=1)
    # selected (matched) nonbinders
    xs, ys = sub(lig, "nonbinder", 1)
    ax.scatter(xs, ys, s=26, c="#e08214", alpha=0.85, edgecolor="none",
               label=f"NB, matched & sent (n={len(xs)})", zorder=2)
    # binders
    xb, yb = sub(lig, "binder", 1)
    ax.scatter(xb, yb, s=46, c="#2166ac", edgecolor="white", linewidth=0.5,
               label=f"binder (n={len(xb)})", zorder=3)
    ax.set_xlabel("ligand pLDDT")
    ax.set_ylabel("water-network geometry score")
    ax.set_title(TITLE[lig])
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)

fig.suptitle("MD hand-off: matched nonbinders overlap binders in Boltz feature space",
             fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
