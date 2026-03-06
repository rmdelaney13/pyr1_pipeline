#!/usr/bin/env python3
"""
Nitazene Boltz2 Retrospective Analysis: SSM + Multi-Mutant Validation.

Generates figures showing how Boltz2 structural metrics vary across
single-site (SSM), double, triple, and quadruple mutants of a nitazene
biosensor. Tests whether increased pocket disruption enables Boltz2
discrimination between binders and non-binders.

Usage:
    # SSM-only (before multi-mutant results):
    python plot_nitazene_boltz.py \
        --ssm-boltz /scratch/.../boltz_nitazene_results.csv \
        --ssm-labels ml_modelling/data/nitazene_ssm_labeled.csv

    # With multi-mutant results:
    python plot_nitazene_boltz.py \
        --ssm-boltz /scratch/.../boltz_nitazene_results.csv \
        --ssm-labels ml_modelling/data/nitazene_ssm_labeled.csv \
        --multi-boltz /scratch/.../boltz_multimutant_results.csv \
        --multi-labels ml_modelling/data/nitazene_multimutant_boltz.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# Colors
BINDER_COLOR = "#2CA02C"      # green for binders
NONBINDER_COLOR = "#D62728"   # red for non-binders
AMBIGUOUS_COLOR = "#999999"   # grey for ambiguous

# Mutant-type colors (gradient from light to dark for increasing disruption)
MUTANT_COLORS = {
    "base":             "#2CA02C",  # green
    "single_binder":    "#66BB6A",  # light green
    "single_nonbinder": "#90CAF9",  # light blue
    "double_negative":  "#42A5F5",  # blue
    "triple_negative":  "#1565C0",  # dark blue
    "quad_negative":    "#0D47A1",  # navy
}

MUTANT_LABELS = {
    "base":             "Base (binder)",
    "single_binder":    "SSM binder",
    "single_nonbinder": "SSM non-binder",
    "double_negative":  "Double mutant",
    "triple_negative":  "Triple mutant",
    "quad_negative":    "Quad mutant",
}

MUTANT_ORDER = ["base", "single_binder", "single_nonbinder",
                "double_negative", "triple_negative", "quad_negative"]

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures" / "nitazene"


def _savefig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def safe_auc(y_true, y_score):
    """AUC with error handling for constant labels."""
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def load_ssm_data(boltz_csv, labels_csv):
    """Load SSM Boltz2 results and merge with labels."""
    boltz = pd.read_csv(boltz_csv)
    labels = pd.read_csv(labels_csv)

    # Build lookup: nitazene_{pos}_{aa} -> label, enrichment
    label_map = {}
    for _, row in labels.iterrows():
        key = f"nitazene_{int(row['position'])}_{row['amino_acid']}"
        label_map[key] = {
            "label": row["label_nitazene"],
            "enrichment": row["enrichment_nitazene"],
            "is_base": row["is_base"],
            "position": int(row["position"]),
            "amino_acid": row["amino_acid"],
        }

    # Merge
    records = []
    for _, row in boltz.iterrows():
        name = row["name"]
        info = label_map.get(name, {})
        label = info.get("label", "missing")

        if label == "binder":
            is_base = info.get("is_base", False)
            mutant_type = "base" if is_base else "single_binder"
            binary_label = 1
        elif label == "non-binder":
            mutant_type = "single_nonbinder"
            binary_label = 0
        else:
            mutant_type = "ambiguous"
            binary_label = None

        records.append({
            "name": name,
            "mutant_type": mutant_type,
            "binary_label": binary_label,
            "enrichment": info.get("enrichment", None),
            "position": info.get("position", None),
            **{k: row[k] for k in row.index if k != "name"},
        })

    return pd.DataFrame(records)


def load_multi_data(boltz_csv, labels_csv):
    """Load multi-mutant Boltz2 results and merge with labels."""
    boltz = pd.read_csv(boltz_csv)
    labels = pd.read_csv(labels_csv)

    label_map = {}
    for _, row in labels.iterrows():
        label_map[row["name"]] = {
            "label": int(row["label"]),
            "mutant_type": row["mutant_type"],
        }

    records = []
    for _, row in boltz.iterrows():
        name = row["name"]
        info = label_map.get(name, {})
        mt = info.get("mutant_type", "unknown")
        bl = info.get("label", None)

        records.append({
            "name": name,
            "mutant_type": mt,
            "binary_label": bl,
            "enrichment": None,
            "position": None,
            **{k: row[k] for k in row.index if k != "name"},
        })

    return pd.DataFrame(records)


# ── Figure 1: Strip plots by mutant type ─────────────────────────
def fig1_strip_by_mutant_type(df):
    """Strip/swarm plots showing score distributions by mutant type."""
    metrics = [
        ("binary_iptm", "ipTM", True),
        ("binary_plddt_pocket", "Pocket pLDDT", True),
        ("binary_plddt_ligand", "Ligand pLDDT", True),
        ("binary_hbond_distance", "H-bond Distance (\u00c5)", False),
        ("binary_confidence_score", "Confidence Score", True),
        ("binary_complex_iplddt", "Interface pLDDT", True),
    ]

    present_types = [t for t in MUTANT_ORDER if t in df["mutant_type"].values]
    palette = {t: MUTANT_COLORS[t] for t in present_types}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (col, label, higher_better) in enumerate(metrics):
        ax = axes[idx]
        plot_df = df[df["mutant_type"].isin(present_types)].copy()
        plot_df["mutant_type"] = pd.Categorical(
            plot_df["mutant_type"], categories=present_types, ordered=True
        )

        sns.stripplot(
            data=plot_df, x="mutant_type", y=col,
            hue="mutant_type", palette=palette,
            alpha=0.6, size=4, jitter=0.3, ax=ax, legend=False,
        )

        # Add boxplot overlay
        sns.boxplot(
            data=plot_df, x="mutant_type", y=col,
            color="white", fliersize=0, width=0.5,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=1),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black"), capprops=dict(color="black"),
            ax=ax,
        )

        ax.set_ylabel(label)
        ax.set_xlabel("")
        xlabels = [MUTANT_LABELS.get(t, t).replace(" ", "\n") for t in present_types]
        ax.set_xticklabels(xlabels, fontsize=9, rotation=0)

        # Compute per-group medians
        for i, t in enumerate(present_types):
            vals = plot_df[plot_df["mutant_type"] == t][col].dropna()
            if len(vals) > 0:
                ax.text(i, ax.get_ylim()[1], f"n={len(vals)}",
                        ha="center", va="bottom", fontsize=8, color="grey")

    fig.suptitle("Nitazene Boltz2 Metrics by Mutant Class", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "fig1_metrics_by_mutant_type.png")


# ── Figure 2: Binder vs non-binder violins (SSM only) ────────────
def fig2_ssm_binder_violins(df):
    """Violin plots comparing binders vs non-binders in SSM data."""
    ssm = df[df["mutant_type"].isin(["base", "single_binder", "single_nonbinder"])].copy()
    ssm["group"] = ssm["binary_label"].map({1: "Binder", 0: "Non-binder"})

    metrics = [
        ("binary_iptm", "ipTM"),
        ("binary_plddt_pocket", "Pocket pLDDT"),
        ("binary_plddt_ligand", "Ligand pLDDT"),
        ("binary_hbond_distance", "H-bond Distance (\u00c5)"),
        ("binary_confidence_score", "Confidence"),
        ("binary_affinity_pred_value", "Affinity Pred"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        plot_data = ssm[ssm["binary_label"].notna()].copy()

        sns.violinplot(
            data=plot_data, x="group", y=col,
            palette={"Binder": BINDER_COLOR, "Non-binder": NONBINDER_COLOR},
            order=["Non-binder", "Binder"],
            inner="box", cut=0, ax=ax,
        )

        # AUC annotation
        valid = plot_data[[col, "binary_label"]].dropna()
        if len(valid) > 0:
            auc = safe_auc(valid["binary_label"], valid[col])
            ax.text(0.95, 0.95, f"AUC={auc:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("")
        ax.set_ylabel(label)

        # Add counts
        nb = len(plot_data[plot_data["group"] == "Non-binder"])
        b = len(plot_data[plot_data["group"] == "Binder"])
        ax.set_title(f"{label}\n(B={b}, NB={nb})", fontsize=11)

    fig.suptitle("Nitazene SSM: Binder vs Non-Binder (Single Mutants Only)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "fig2_ssm_binder_violins.png")


# ── Figure 3: AUC by mutant depth ────────────────────────────────
def fig3_auc_by_depth(df):
    """Bar chart showing AUC improvement as mutant complexity increases."""
    metrics = [
        ("binary_iptm", "ipTM"),
        ("binary_plddt_pocket", "Pocket pLDDT"),
        ("binary_confidence_score", "Confidence"),
        ("binary_complex_iplddt", "Interface pLDDT"),
    ]

    # Define comparison groups: binders vs each negative depth
    depths = {
        "SSM (1-mut)": ["single_nonbinder"],
        "Double (2-mut)": ["double_negative"],
        "Triple (3-mut)": ["triple_negative"],
        "Quad (4-mut)": ["quad_negative"],
    }

    binder_types = ["base", "single_binder"]
    binders = df[df["mutant_type"].isin(binder_types)]

    # Only include depths that exist in data
    available_depths = {k: v for k, v in depths.items()
                        if any(t in df["mutant_type"].values for t in v)}

    if not available_depths:
        print("  Skipping fig3: no negative groups found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(available_depths)
    depth_colors = ["#90CAF9", "#42A5F5", "#1565C0", "#0D47A1"]

    for di, (depth_name, neg_types) in enumerate(available_depths.items()):
        negatives = df[df["mutant_type"].isin(neg_types)]
        aucs = []
        for col, _ in metrics:
            combined = pd.concat([
                binders[["name", col, "binary_label"]].assign(binary_label=1),
                negatives[["name", col]].assign(binary_label=0),
            ])
            valid = combined[[col, "binary_label"]].dropna()
            auc = safe_auc(valid["binary_label"], valid[col])
            aucs.append(auc)

        bars = ax.bar(x + di * width - 0.4 + width / 2, aucs, width,
                      label=f"{depth_name} (n={len(negatives)})",
                      color=depth_colors[di % len(depth_colors)], edgecolor="black",
                      linewidth=0.5)
        # Value labels
        for bar, auc_val in zip(bars, aucs):
            if not np.isnan(auc_val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{auc_val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("AUC (binders vs negatives)")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Discrimination Improves with Pocket Disruption\n"
                 f"(Positives: {len(binders)} base + SSM binders)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    _savefig(fig, "fig3_auc_by_mutant_depth.png")


# ── Figure 4: Scatter — ipTM vs pocket pLDDT colored by type ─────
def fig4_scatter_iptm_pocket(df):
    """Scatter plot of ipTM vs pocket pLDDT, colored by mutant type."""
    present_types = [t for t in MUTANT_ORDER if t in df["mutant_type"].values]
    palette = {t: MUTANT_COLORS[t] for t in present_types}

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot in reverse order so binders are on top
    for t in reversed(present_types):
        subset = df[df["mutant_type"] == t]
        marker = "D" if t == "base" else "o"
        size = 80 if t == "base" else 30
        ax.scatter(subset["binary_iptm"], subset["binary_plddt_pocket"],
                   c=palette[t], label=MUTANT_LABELS[t],
                   marker=marker, s=size, alpha=0.7, edgecolors="black",
                   linewidth=0.3 if t != "base" else 1.0)

    ax.set_xlabel("ipTM")
    ax.set_ylabel("Pocket pLDDT")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Nitazene: ipTM vs Pocket pLDDT by Mutant Class",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    _savefig(fig, "fig4_scatter_iptm_pocket.png")


# ── Figure 5: Enrichment vs Boltz2 score (SSM only) ──────────────
def fig5_enrichment_correlation(df):
    """Scatter plot of SSM enrichment vs Boltz2 metrics."""
    ssm = df[df["mutant_type"].isin(["base", "single_binder", "single_nonbinder"])].copy()
    ssm = ssm[ssm["enrichment"].notna()].copy()
    ssm["enrichment"] = ssm["enrichment"].astype(float)

    metrics = [
        ("binary_iptm", "ipTM"),
        ("binary_plddt_pocket", "Pocket pLDDT"),
        ("binary_confidence_score", "Confidence"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        valid = ssm[[col, "enrichment", "binary_label"]].dropna()

        # Color by label
        colors = valid["binary_label"].map({1: BINDER_COLOR, 0: NONBINDER_COLOR})
        ax.scatter(valid["enrichment"], valid[col], c=colors, alpha=0.6, s=25,
                   edgecolors="black", linewidth=0.3)

        # Spearman correlation
        if len(valid) > 5:
            rho, pval = __import__("scipy").stats.spearmanr(valid["enrichment"], valid[col])
            ax.text(0.05, 0.95, f"Spearman r={rho:.3f}\np={pval:.1e}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.axvline(-1, color="grey", linestyle="--", alpha=0.4, label="Binder cutoff")
        ax.axvline(-3, color="grey", linestyle=":", alpha=0.4, label="Non-binder cutoff")
        ax.set_xlabel("SSM Enrichment (log2)")
        ax.set_ylabel(label)
        ax.set_title(label)

    # Legend
    handles = [
        mpatches.Patch(color=BINDER_COLOR, label="Binder (>-1)"),
        mpatches.Patch(color=NONBINDER_COLOR, label="Non-binder (<-3)"),
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=9)

    fig.suptitle("SSM Enrichment vs Boltz2 Metrics (Nitazene)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "fig5_enrichment_vs_boltz.png")


# ── Figure 6: Summary table ──────────────────────────────────────
def fig6_summary_table(df):
    """Summary statistics table as a figure."""
    metrics = [
        ("binary_iptm", "ipTM", True),
        ("binary_plddt_pocket", "Pocket pLDDT", True),
        ("binary_plddt_ligand", "Ligand pLDDT", True),
        ("binary_hbond_distance", "H-bond Dist", False),
        ("binary_confidence_score", "Confidence", True),
    ]

    present_types = [t for t in MUTANT_ORDER if t in df["mutant_type"].values]
    binder_types = ["base", "single_binder"]

    rows = []
    for mt in present_types:
        subset = df[df["mutant_type"] == mt]
        row = {"Group": MUTANT_LABELS[mt], "n": len(subset)}
        for col, label, higher in metrics:
            vals = subset[col].dropna()
            row[f"{label} med"] = f"{vals.median():.3f}" if len(vals) > 0 else "-"
            # AUC vs binders (only for non-binder types)
            if mt not in binder_types:
                binders_df = df[df["mutant_type"].isin(binder_types)]
                combined = pd.concat([
                    binders_df[[col]].assign(label=1),
                    subset[[col]].assign(label=0),
                ])
                valid = combined[[col, "label"]].dropna()
                auc = safe_auc(valid["label"], valid[col] if higher else -valid[col])
                row[f"{label} AUC"] = f"{auc:.3f}" if not np.isnan(auc) else "-"
            else:
                row[f"{label} AUC"] = "-"
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 2 + 0.4 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color header
    for j in range(len(summary_df.columns)):
        table[0, j].set_facecolor("#E8E8E8")
        table[0, j].set_text_props(fontweight="bold")

    # Color rows by group
    for i, mt in enumerate(present_types):
        color = MUTANT_COLORS.get(mt, "#FFFFFF")
        for j in range(len(summary_df.columns)):
            table[i + 1, j].set_facecolor(color + "30")  # 30 = ~19% opacity

    ax.set_title("Nitazene Boltz2 Summary by Mutant Class",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    _savefig(fig, "fig6_summary_table.png")


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Nitazene Boltz2 analysis figures")
    parser.add_argument("--ssm-boltz", required=True,
                        help="Boltz2 results CSV for SSM predictions")
    parser.add_argument("--ssm-labels", required=True,
                        help="Labeled SSM CSV (from parse_nitazene_ssm.py)")
    parser.add_argument("--multi-boltz", default=None,
                        help="Boltz2 results CSV for multi-mutant predictions")
    parser.add_argument("--multi-labels", default=None,
                        help="Multi-mutant labels CSV (from generate_nitazene_multimutants.py)")
    args = parser.parse_args()

    print("Loading SSM data...")
    df_ssm = load_ssm_data(args.ssm_boltz, args.ssm_labels)
    print(f"  SSM: {len(df_ssm)} rows")
    print(f"  Types: {df_ssm['mutant_type'].value_counts().to_dict()}")

    df = df_ssm.copy()

    if args.multi_boltz and args.multi_labels:
        print("\nLoading multi-mutant data...")
        df_multi = load_multi_data(args.multi_boltz, args.multi_labels)
        # Exclude single binders/base from multi CSV (already in SSM)
        df_multi = df_multi[df_multi["mutant_type"].isin(
            ["double_negative", "triple_negative", "quad_negative"]
        )]
        print(f"  Multi: {len(df_multi)} rows")
        print(f"  Types: {df_multi['mutant_type'].value_counts().to_dict()}")
        df = pd.concat([df_ssm, df_multi], ignore_index=True)

    print(f"\nTotal: {len(df)} rows")
    print(f"All types: {df['mutant_type'].value_counts().to_dict()}")

    print("\n=== Generating Figures ===")
    print("\nFig 1: Metrics by mutant type (strip plots)")
    fig1_strip_by_mutant_type(df)

    print("\nFig 2: SSM binder vs non-binder violins")
    fig2_ssm_binder_violins(df)

    print("\nFig 3: AUC by mutant depth")
    fig3_auc_by_depth(df)

    print("\nFig 4: ipTM vs pocket pLDDT scatter")
    fig4_scatter_iptm_pocket(df)

    print("\nFig 5: Enrichment vs Boltz2 (SSM)")
    fig5_enrichment_correlation(df)

    print("\nFig 6: Summary table")
    fig6_summary_table(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
