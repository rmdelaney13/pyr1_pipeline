#!/usr/bin/env python3
"""
Plot Boltz2 binary metrics for CDCA non-binder designs.

Usage:
    python ml_modelling/analysis/plot_cdca_nonbinders_boltz.py \
        --csv /scratch/alpine/ryde3462/boltz_cdca_nonbinders/boltz_cdca_nonbinders_results.csv

    # Or locally after scp:
    python ml_modelling/analysis/plot_cdca_nonbinders_boltz.py \
        --csv ml_modelling/data/boltz_cdca_nonbinders_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
})

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures" / "cdca_nonbinders"

# Variant group colors
GROUP_COLORS = {"59a": "#E74C3C", "59l": "#3498DB", "59v": "#2ECC71", "wt_mandi": "#9B59B6"}
GROUP_LABELS = {"59a": "59A", "59l": "59L", "59v": "59V", "wt_mandi": "59R (WT)"}


def _savefig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def assign_group(name):
    for prefix in ("59a", "59l", "59v", "wt_mandi"):
        if name.startswith(prefix):
            return prefix
    return "other"


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["group"] = df["name"].apply(assign_group)
    df["group_label"] = df["group"].map(GROUP_LABELS)
    return df


# ── Fig 1: H-bond distance vs Ligand pLDDT scatter ──────────────────────
def fig1_hbond_vs_ligand_plddt(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    for grp in ["59a", "59l", "59v", "wt_mandi"]:
        sub = df[df["group"] == grp]
        ax.scatter(sub["binary_hbond_distance"], sub["binary_plddt_ligand"],
                   c=GROUP_COLORS[grp], label=GROUP_LABELS[grp],
                   s=60, alpha=0.8, edgecolors="black", linewidth=0.5)

    # Annotate points
    for _, row in df.iterrows():
        ax.annotate(row["name"].split("_")[-1], (row["binary_hbond_distance"], row["binary_plddt_ligand"]),
                    fontsize=6, alpha=0.6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    ax.axvline(3.5, color="grey", linestyle="--", alpha=0.4, label="H-bond cutoff (3.5 A)")
    ax.set_xlabel("H-bond Distance to Water (A)")
    ax.set_ylabel("Ligand pLDDT")
    ax.legend(fontsize=9)
    ax.set_title("CDCA Non-binders: H-bond Distance vs Ligand pLDDT")
    fig.tight_layout()
    _savefig(fig, "fig1_hbond_vs_ligand_plddt.png")


# ── Fig 2: Pocket pLDDT + key metrics strip/box ─────────────────────────
def fig2_pocket_metrics(df):
    metrics = [
        ("binary_plddt_pocket", "Pocket pLDDT"),
        ("binary_plddt_ligand", "Ligand pLDDT"),
        ("binary_iptm", "ipTM"),
        ("binary_confidence_score", "Confidence"),
        ("binary_hbond_distance", "H-bond Distance (A)"),
        ("binary_affinity_pred_value", "Affinity Pred"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    order = ["59a", "59l", "59v", "wt_mandi"]
    present = [g for g in order if g in df["group"].values]

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        plot_df = df[df["group"].isin(present)].copy()
        plot_df["group"] = pd.Categorical(plot_df["group"], categories=present, ordered=True)

        sns.stripplot(data=plot_df, x="group", y=col, hue="group",
                      palette=GROUP_COLORS, alpha=0.7, size=7, jitter=0.2,
                      ax=ax, legend=False)
        sns.boxplot(data=plot_df, x="group", y=col, color="white",
                    fliersize=0, width=0.4,
                    boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color="black"), capprops=dict(color="black"),
                    ax=ax)

        xlabels = [GROUP_LABELS.get(g, g) for g in present]
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(label)

        # Add n and median
        for i, g in enumerate(present):
            vals = plot_df[plot_df["group"] == g][col].dropna()
            if len(vals) > 0:
                ax.text(i, ax.get_ylim()[1], f"n={len(vals)}\nmed={vals.median():.2f}",
                        ha="center", va="bottom", fontsize=7, color="grey")

    fig.suptitle("CDCA Non-binders: Key Boltz2 Metrics by Pos-59 Group",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "fig2_pocket_metrics_by_group.png")


# ── Fig 3: OH-to-water distance (core hydroxyl contact) ─────────────────
def fig3_oh_contact(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: OH-to-water scatter vs confidence
    ax = axes[0]
    for grp in ["59a", "59l", "59v", "wt_mandi"]:
        sub = df[df["group"] == grp]
        ax.scatter(sub["binary_oh_to_water_dist"], sub["binary_confidence_score"],
                   c=GROUP_COLORS[grp], label=GROUP_LABELS[grp],
                   s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
    ax.axvline(3.5, color="red", linestyle="--", alpha=0.4, label="3.5 A cutoff")
    ax.set_xlabel("OH-to-Water Distance (A)")
    ax.set_ylabel("Confidence Score")
    ax.legend(fontsize=8)
    ax.set_title("Core OH near Water?")

    # Panel B: COO-to-R116 distance (artifact check)
    ax = axes[1]
    for grp in ["59a", "59l", "59v", "wt_mandi"]:
        sub = df[df["group"] == grp]
        ax.scatter(sub["binary_coo_to_r116_dist"], sub["binary_confidence_score"],
                   c=GROUP_COLORS[grp], label=GROUP_LABELS[grp],
                   s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
    ax.axvline(4.0, color="orange", linestyle="--", alpha=0.5, label="Salt bridge cutoff (4 A)")
    ax.set_xlabel("COO-to-R116 Distance (A)")
    ax.set_ylabel("Confidence Score")
    ax.legend(fontsize=8)
    ax.set_title("COO-R116 Salt Bridge Artifact?")

    # Panel C: OH vs COO water proximity
    ax = axes[2]
    for grp in ["59a", "59l", "59v", "wt_mandi"]:
        sub = df[df["group"] == grp]
        ax.scatter(sub["binary_oh_to_water_dist"], sub["binary_coo_to_water_dist"],
                   c=GROUP_COLORS[grp], label=GROUP_LABELS[grp],
                   s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
    # Diagonal
    lim = max(df["binary_oh_to_water_dist"].max(), df["binary_coo_to_water_dist"].max()) + 1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="OH = COO")
    ax.set_xlabel("OH-to-Water Distance (A)")
    ax.set_ylabel("COO-to-Water Distance (A)")
    ax.legend(fontsize=8)
    ax.set_title("Ligand Orientation")

    fig.suptitle("CDCA Non-binders: Core OH Contact & Orientation Analysis",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "fig3_oh_contact_and_orientation.png")


# ── Fig 4: Summary scatter — ipTM vs pocket pLDDT colored by OH contact ─
def fig4_summary_scatter(df):
    fig, ax = plt.subplots(figsize=(9, 7))

    has_oh = df["binary_oh_to_water_dist"].notna()
    oh_close = has_oh & (df["binary_oh_to_water_dist"] < 3.5)
    oh_far = has_oh & (df["binary_oh_to_water_dist"] >= 3.5)

    ax.scatter(df.loc[oh_close, "binary_iptm"], df.loc[oh_close, "binary_plddt_pocket"],
               c="#2ECC71", s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
               label=f"OH near water (<3.5 A, n={oh_close.sum()})", marker="o")
    ax.scatter(df.loc[oh_far, "binary_iptm"], df.loc[oh_far, "binary_plddt_pocket"],
               c="#E74C3C", s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
               label=f"OH far from water (>=3.5 A, n={oh_far.sum()})", marker="X")

    # Annotate
    for _, row in df.iterrows():
        short = row["name"].replace("wt_mandi_", "wt_").replace("59a_", "a").replace("59l_", "l").replace("59v_", "v")
        ax.annotate(short, (row["binary_iptm"], row["binary_plddt_pocket"]),
                    fontsize=6, alpha=0.6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("ipTM")
    ax.set_ylabel("Pocket pLDDT")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("CDCA Non-binders: ipTM vs Pocket pLDDT\n(colored by core OH water contact)")
    fig.tight_layout()
    _savefig(fig, "fig4_iptm_vs_pocket_plddt_oh.png")


def main():
    parser = argparse.ArgumentParser(description="Plot CDCA non-binder Boltz2 results")
    parser.add_argument("--csv", required=True, help="Boltz results CSV from analyze_boltz_output.py")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.csv)
    print(f"  {len(df)} variants, groups: {df['group'].value_counts().to_dict()}")

    print("\nFig 1: H-bond distance vs ligand pLDDT")
    fig1_hbond_vs_ligand_plddt(df)

    print("\nFig 2: Pocket metrics by group")
    fig2_pocket_metrics(df)

    print("\nFig 3: OH contact & orientation")
    fig3_oh_contact(df)

    print("\nFig 4: Summary scatter (ipTM vs pocket pLDDT, OH contact)")
    fig4_summary_scatter(df)

    # Print quick summary table
    print("\n=== Quick Summary ===")
    summary_cols = ["binary_iptm", "binary_plddt_pocket", "binary_plddt_ligand",
                    "binary_hbond_distance", "binary_oh_to_water_dist", "binary_coo_to_r116_dist",
                    "binary_confidence_score"]
    print(df[["name", "group_label"] + summary_cols].to_string(index=False, float_format="%.3f"))

    print("\nDone!")


if __name__ == "__main__":
    main()
