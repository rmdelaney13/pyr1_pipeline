#!/usr/bin/env python
"""
Comprehensive PYR1 ML Dataset Analysis.

Generates figures and statistics organized into 6 parts that map 1:1
to a writeup outline.  Each section can be run independently.

Usage:
    python comprehensive_pyr1_analysis.py --csv all_features.csv
    python comprehensive_pyr1_analysis.py --csv all_features.csv --section 2.3
    python comprehensive_pyr1_analysis.py --csv all_features.csv --section 2

Output: figures/ subdirectory + stdout statistics (pipe to file for writeup).
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIG_DIR = SCRIPT_DIR / "figures" / "comprehensive"

# ── Style ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
LABEL_COLORS = {0.0: "#3b82f6", 0.25: "#a78bfa", 0.75: "#f59e0b", 1.0: "#ef4444"}
LABEL_NAMES = {0.0: "Negative", 0.25: "Weak", 0.75: "Moderate", 1.0: "Strong"}
BINARY_COLORS = {"Non-binder": "#3b82f6", "Binder": "#ef4444"}
SOURCE_COLORS = {
    "experimental": "#22c55e", "win_ssm": "#3b82f6",
    "pnas_cutler": "#f59e0b", "LCA_screen": "#a855f7",
    "artificial_swap": "#ef4444", "artificial_ala_scan": "#f97316",
}
TIER_COLORS = {
    "Tier 1": "#22c55e", "Tier 2": "#3b82f6", "Tier 3": "#f59e0b",
    "Tier 4": "#a855f7", "Tier 5": "#ef4444",
}

# ── Helpers ──────────────────────────────────────────────────────

def _savefig(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


def _feature_cols(df, stages=None, exclude_sparse=False):
    """Get numeric feature columns, optionally filtered by stage prefix."""
    skip = {"docking_clash_flag", "docking_total_attempts",
            "rosetta_n_structures_relaxed"}
    cols = [c for c in df.columns
            if (c.startswith("docking_") or c.startswith("rosetta_") or
                c.startswith("af3_") or c.startswith("conformer_"))
            and not c.endswith("_status") and c not in skip]
    if stages:
        cols = [c for c in cols
                if any(c.startswith(s + "_") for s in stages)]
    if exclude_sparse:
        n = len(df)
        cols = [c for c in cols if df[c].notna().sum() / n >= 0.5]
    cols = [c for c in cols if df[c].notna().any()]
    # Exclude constant columns (cause NaN correlations)
    cols = [c for c in cols if df[c].dropna().nunique() > 1]
    return cols


def _tier_label(source):
    mapping = {
        "experimental": "Tier 1", "win_ssm": "Tier 2",
        "pnas_cutler": "Tier 3", "LCA_screen": "Tier 4",
        "artificial_swap": "Tier 5", "artificial_ala_scan": "Tier 5",
    }
    return mapping.get(source, source)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path.name}")

    # Drop all-NaN columns
    always_nan = [c for c in df.columns if df[c].isna().all()]
    if always_nan:
        df.drop(columns=always_nan, inplace=True)

    # Derived columns
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["binder"] = (df["label"] >= 0.75).astype(int)
    df["binder_name"] = df["binder"].map({0: "Non-binder", 1: "Binder"})
    if "label_source" in df.columns:
        df["tier"] = df["label_source"].map(_tier_label)

    return df


# ═════════════════════════════════════════════════════════════════
# PART 1: DATASET CHARACTERIZATION
# ═════════════════════════════════════════════════════════════════

def section_1_1(df):
    """Tier-by-tier data inventory."""
    print("\n" + "=" * 70)
    print("1.1  TIER-BY-TIER DATA INVENTORY")
    print("=" * 70)

    # Summary table
    rows = []
    for src in ["experimental", "win_ssm", "pnas_cutler", "LCA_screen",
                "artificial_swap", "artificial_ala_scan"]:
        sub = df[df["label_source"] == src]
        if len(sub) == 0:
            continue
        rows.append({
            "Source": src,
            "Tier": _tier_label(src),
            "Pairs": len(sub),
            "Ligands": sub["ligand_name"].nunique(),
            "Variants": sub["variant_name"].nunique(),
            "Binders": int(sub["binder"].sum()),
            "Binder %": f"{100 * sub['binder'].mean():.1f}%",
            "Confidence": f"{pd.to_numeric(sub['label_confidence'], errors='coerce').mean():.2f}",
        })
    summary = pd.DataFrame(rows)
    print(f"\n{summary.to_string(index=False)}")

    # Label distribution per tier
    print("\n  Label distribution per source:")
    for src in ["experimental", "win_ssm", "pnas_cutler", "LCA_screen",
                "artificial_swap", "artificial_ala_scan"]:
        sub = df[df["label_source"] == src]
        if len(sub) == 0:
            continue
        dist = sub["label"].value_counts().sort_index()
        parts = [f"{LABEL_NAMES.get(k, k)}={v}" for k, v in dist.items()]
        print(f"    {src:25s}: {', '.join(parts)}")

    # Feature completeness by tier
    stages = {"conformer": "conformer_", "docking": "docking_",
              "rosetta": "rosetta_", "af3_binary": "af3_binary_",
              "af3_ternary": "af3_ternary_"}
    print("\n  Feature completeness by source:")
    header = f"  {'Source':25s}" + "".join(f"{s:>12s}" for s in stages)
    print(header)
    print("  " + "-" * len(header))
    for src in ["experimental", "win_ssm", "pnas_cutler", "LCA_screen",
                "artificial_swap", "artificial_ala_scan"]:
        sub = df[df["label_source"] == src]
        if len(sub) == 0:
            continue
        parts = [f"{src:25s}"]
        for stage, prefix in stages.items():
            stage_cols = [c for c in df.columns if c.startswith(prefix)
                          and not c.endswith("_status")]
            if stage_cols:
                completeness = sub[stage_cols].notna().any(axis=1).mean()
                parts.append(f"{completeness:>11.0%}")
            else:
                parts.append(f"{'N/A':>12s}")
        print("  " + "".join(parts))

    # Figure: tier composition
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: pair counts by tier
    tier_counts = df.groupby("tier").size().reindex(
        ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5"])
    colors = [TIER_COLORS.get(t, "#94a3b8") for t in tier_counts.index]
    axes[0].barh(tier_counts.index, tier_counts.values, color=colors)
    axes[0].set_xlabel("Number of pairs")
    axes[0].set_title("Dataset size by tier")
    for i, v in enumerate(tier_counts.values):
        axes[0].text(v + 20, i, str(v), va="center", fontsize=9)

    # Panel 2: binder rate by tier
    binder_rates = df.groupby("tier")["binder"].mean().reindex(
        ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5"])
    axes[1].barh(binder_rates.index, binder_rates.values, color=colors)
    axes[1].set_xlabel("Binder fraction")
    axes[1].set_title("Binder rate by tier")
    axes[1].set_xlim(0, 1.05)
    for i, v in enumerate(binder_rates.values):
        axes[1].text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)

    # Panel 3: label distribution stacked bar
    label_counts = df.groupby(["tier", "label"]).size().unstack(fill_value=0)
    label_counts = label_counts.reindex(
        ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5"])
    label_counts_norm = label_counts.div(label_counts.sum(axis=1), axis=0)
    left = np.zeros(len(label_counts_norm))
    for label_val in [0.0, 0.25, 0.75, 1.0]:
        if label_val in label_counts_norm.columns:
            vals = label_counts_norm[label_val].values
            axes[2].barh(label_counts_norm.index, vals, left=left,
                         color=LABEL_COLORS[label_val],
                         label=LABEL_NAMES[label_val])
            left += vals
    axes[2].set_xlabel("Fraction")
    axes[2].set_title("Label distribution by tier")
    axes[2].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    _savefig(fig, "1_1_tier_overview.png")


def section_1_2(df):
    """Label quality audit."""
    print("\n" + "=" * 70)
    print("1.2  LABEL QUALITY AUDIT")
    print("=" * 70)

    # Tier 2: WIN SSM Kd distribution
    if "affinity_uM" in df.columns:
        t2 = df[(df["label_source"] == "win_ssm") & df["affinity_uM"].notna()]
        if len(t2) > 0:
            print(f"\n  Tier 2 (WIN SSM) affinity distribution:")
            print(f"    n={len(t2)}, range={t2['affinity_uM'].min():.3f}-"
                  f"{t2['affinity_uM'].max():.1f} uM")
            for label in sorted(t2["label"].unique()):
                sub = t2[t2["label"] == label]
                print(f"    label={label} ({LABEL_NAMES.get(label, '?')}): "
                      f"n={len(sub)}, Kd range={sub['affinity_uM'].min():.3f}-"
                      f"{sub['affinity_uM'].max():.1f} uM")

    # Tier 3: PNAS min_conc from source
    pnas_src = DATA_DIR / "pnas_data.csv"
    if pnas_src.exists():
        src = pd.read_csv(pnas_src)
        min_conc_counts = src["min_conc"].value_counts()
        print(f"\n  Tier 3 (PNAS) source concentration data:")
        print(f"    Total source rows: {len(src)}")
        for conc, n in min_conc_counts.items():
            print(f"    min_conc={conc}: {n} rows")

    # Tier 4: LCA binder split
    t4 = df[df["label_source"] == "LCA_screen"]
    t1_lca = df[(df["label_source"] == "experimental") &
                df["ligand_name"].str.contains("Lithocholic", case=False, na=False)]
    print(f"\n  Tier 4 (LCA screen): {len(t4)} pairs, "
          f"{int(t4['binder'].sum())} binders")
    print(f"  Tier 1 LCA binders: {int(t1_lca['binder'].sum())} pairs")
    print(f"  Total LCA binders across tiers: "
          f"{int(t4['binder'].sum()) + int(t1_lca['binder'].sum())}")

    # Tier 5: audit for potential false negatives
    t5 = df[df["label_source"].isin(["artificial_swap", "artificial_ala_scan"])]
    t5_sigs = set(t5["variant_signature"].dropna())
    known_binder_sigs = set(
        df[(df["binder"] == 1) &
           ~df["label_source"].isin(["artificial_swap", "artificial_ala_scan"])]
        ["variant_signature"].dropna()
    )
    overlap = t5_sigs & known_binder_sigs
    print(f"\n  Tier 5 (artificial negatives): {len(t5)} pairs")
    print(f"    Variant signatures shared with known binders: {len(overlap)}")
    if overlap:
        print(f"    (these are swap negatives — same variant, different ligand)")

    # Figure: label quality
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: confidence weight distribution
    if "label_confidence" in df.columns:
        conf = pd.to_numeric(df["label_confidence"], errors="coerce")
        for src, color in SOURCE_COLORS.items():
            vals = conf[df["label_source"] == src].dropna()
            if len(vals) > 0:
                axes[0].hist(vals, bins=20, alpha=0.6, color=color,
                             label=f"{src} (n={len(vals)})", density=True)
        axes[0].set_xlabel("Label confidence weight")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Confidence weight distribution by source")
        axes[0].legend(fontsize=7)

    # Panel 2: Kd distribution for tier 2
    if "affinity_uM" in df.columns:
        t2 = df[(df["label_source"] == "win_ssm") & df["affinity_uM"].notna()]
        if len(t2) > 0:
            for label, color in LABEL_COLORS.items():
                vals = t2.loc[t2["label"] == label, "affinity_uM"]
                if len(vals) > 0:
                    axes[1].hist(np.log10(vals.clip(lower=0.01)), bins=20,
                                 alpha=0.6, color=color,
                                 label=f"{LABEL_NAMES[label]} (n={len(vals)})")
            axes[1].set_xlabel("log10(Kd, uM)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("WIN SSM Kd distribution by label")
            axes[1].axvline(np.log10(0.5), color="red", ls="--", alpha=0.5,
                            label="500 nM")
            axes[1].axvline(np.log10(2.0), color="orange", ls="--", alpha=0.5,
                            label="2 uM")
            axes[1].legend(fontsize=7)

    plt.tight_layout()
    _savefig(fig, "1_2_label_quality.png")


# ═════════════════════════════════════════════════════════════════
# PART 2: AF3 AS AN ALLOSTERIC INTERFACE PREDICTOR
# ═════════════════════════════════════════════════════════════════

def section_2_1(df):
    """AF3 confidence metrics by binding strength."""
    print("\n" + "=" * 70)
    print("2.1  AF3 CONFIDENCE METRICS BY BINDING STRENGTH")
    print("=" * 70)

    af3_metrics = {
        "af3_binary_ipTM": "Binary ipTM",
        "af3_ternary_ipTM": "Ternary ipTM",
        "af3_binary_pLDDT_ligand": "Binary pLDDT (ligand)",
        "af3_ternary_pLDDT_ligand": "Ternary pLDDT (ligand)",
        "af3_binary_pLDDT_protein": "Binary pLDDT (protein)",
        "af3_ternary_pLDDT_protein": "Ternary pLDDT (protein)",
        "af3_binary_interface_PAE": "Binary interface PAE",
        "af3_ternary_interface_PAE": "Ternary interface PAE",
    }
    af3_metrics = {k: v for k, v in af3_metrics.items() if k in df.columns}

    # Print discriminability stats
    print("\n  Single-feature discriminability (Spearman r with label):")
    metric_stats = []
    for col, name in af3_metrics.items():
        valid = df[["label", col]].dropna()
        if len(valid) > 30:
            r, p = stats.spearmanr(valid["label"], valid[col])
            metric_stats.append((name, col, r, p, len(valid)))
    metric_stats.sort(key=lambda x: abs(x[2]), reverse=True)
    for name, col, r, p, n in metric_stats:
        print(f"    {name:30s}: r={r:+.3f} (p={p:.2e}, n={n})")

    # Per-tier breakdown for top metric
    if metric_stats:
        top_col = metric_stats[0][1]
        top_name = metric_stats[0][0]
        print(f"\n  {top_name} by tier and label:")
        for src in ["experimental", "win_ssm", "pnas_cutler", "LCA_screen"]:
            sub = df[df["label_source"] == src]
            for label in sorted(sub["label"].unique()):
                vals = sub.loc[sub["label"] == label, top_col].dropna()
                if len(vals) > 3:
                    print(f"    {src:20s} label={label}: "
                          f"mean={vals.mean():.3f}, median={vals.median():.3f} "
                          f"(n={len(vals)})")

    # Figure: violin plots
    n_metrics = len(af3_metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (col, name) in enumerate(af3_metrics.items()):
        ax = axes[idx]
        plot_df = df[["label_name", col]].dropna()
        if len(plot_df) < 20:
            ax.set_visible(False)
            continue
        order = ["Negative", "Weak", "Moderate", "Strong"]
        order = [o for o in order if o in plot_df["label_name"].values]
        palette = [LABEL_COLORS[{v: k for k, v in LABEL_NAMES.items()}[o]]
                   for o in order]
        sns.violinplot(data=plot_df, x="label_name", y=col, order=order,
                       palette=palette, ax=ax, cut=0, inner="quartile")
        ax.set_xlabel("")
        ax.set_title(name, fontsize=10)

    for idx in range(len(af3_metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("AF3 Confidence Metrics by Binding Strength", fontsize=14, y=1.01)
    plt.tight_layout()
    _savefig(fig, "2_1_af3_confidence_by_label.png")

    # Figure: single-feature ROC curves
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        fig, ax = plt.subplots(figsize=(8, 7))
        for name, col, r, p, n in metric_stats:
            valid = df[["binder", col]].dropna()
            y = valid["binder"].values
            # Flip sign for PAE (lower=better)
            scores = valid[col].values
            if "PAE" in col:
                scores = -scores
            auc = roc_auc_score(y, scores)
            fpr, tpr, _ = roc_curve(y, scores)
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Single-Feature ROC: AF3 Metrics")
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        _savefig(fig, "2_1_af3_single_feature_roc.png")
    except ImportError:
        print("  (scikit-learn not installed — skipping ROC curves)")


def section_2_2(df):
    """Binary vs ternary predictions."""
    print("\n" + "=" * 70)
    print("2.2  BINARY VS TERNARY AF3 PREDICTIONS")
    print("=" * 70)

    metric_pairs = [
        ("af3_binary_ipTM", "af3_ternary_ipTM", "ipTM"),
        ("af3_binary_pLDDT_ligand", "af3_ternary_pLDDT_ligand", "pLDDT (ligand)"),
        ("af3_binary_pLDDT_protein", "af3_ternary_pLDDT_protein", "pLDDT (protein)"),
        ("af3_binary_interface_PAE", "af3_ternary_interface_PAE", "Interface PAE"),
    ]
    metric_pairs = [(b, t, n) for b, t, n in metric_pairs
                    if b in df.columns and t in df.columns]

    # Delta analysis
    print("\n  Delta (ternary - binary) by label:")
    for bcol, tcol, name in metric_pairs:
        valid = df[[bcol, tcol, "label"]].dropna()
        valid["delta"] = valid[tcol] - valid[bcol]
        print(f"\n  {name}:")
        for label in sorted(valid["label"].unique()):
            sub = valid[valid["label"] == label]
            print(f"    label={label} ({LABEL_NAMES.get(label, '?'):8s}): "
                  f"delta mean={sub['delta'].mean():+.4f}, "
                  f"std={sub['delta'].std():.4f} (n={len(sub)})")

    # Binary-ternary RMSD
    bt_col = "af3_binary_ligand_RMSD_bt"
    if bt_col in df.columns:
        print(f"\n  Binary-to-ternary ligand RMSD by label:")
        for label in sorted(df["label"].unique()):
            vals = df.loc[df["label"] == label, bt_col].dropna()
            if len(vals) > 3:
                print(f"    label={label}: mean={vals.mean():.2f}, "
                      f"median={vals.median():.2f} (n={len(vals)})")

    # Figure: scatter + delta histograms
    n_pairs = len(metric_pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(13, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for idx, (bcol, tcol, name) in enumerate(metric_pairs):
        valid = df[[bcol, tcol, "label_name", "binder_name"]].dropna()

        # Scatter
        ax = axes[idx, 0]
        for label, color in LABEL_COLORS.items():
            lname = LABEL_NAMES[label]
            mask = valid["label_name"] == lname
            if mask.sum() > 0:
                ax.scatter(valid.loc[mask, bcol], valid.loc[mask, tcol],
                           c=color, alpha=0.3, s=10, label=lname)
        lim = [min(valid[bcol].min(), valid[tcol].min()),
               max(valid[bcol].max(), valid[tcol].max())]
        ax.plot(lim, lim, "k--", alpha=0.3)
        ax.set_xlabel(f"Binary {name}")
        ax.set_ylabel(f"Ternary {name}")
        ax.set_title(f"{name}: Binary vs Ternary")
        if idx == 0:
            ax.legend(fontsize=7, markerscale=2)

        # Delta histogram
        ax = axes[idx, 1]
        valid["delta"] = valid[tcol] - valid[bcol]
        for bname, color in BINARY_COLORS.items():
            vals = valid.loc[valid["binder_name"] == bname, "delta"]
            if len(vals) > 0:
                ax.hist(vals, bins=40, alpha=0.6, color=color,
                        label=bname, density=True)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(f"Delta {name} (ternary - binary)")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}: Water effect")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _savefig(fig, "2_2_binary_vs_ternary.png")

    # --- Second figure: binary-to-ternary ligand RMSD landscape ---
    bt_col = "af3_binary_ligand_RMSD_bt"
    water_col = "af3_binary_min_dist_to_ligand_O"
    if bt_col not in df.columns:
        return

    valid_bt = df[df[bt_col].notna()].copy()
    print(f"\n  Binary-to-ternary RMSD landscape: {len(valid_bt)} pairs")

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: bt RMSD box plots by data source (binder vs non-binder)
    ax = axes2[0, 0]
    sources_order = ["experimental", "win_ssm", "pnas_cutler", "LCA_screen", "artificial"]
    plot_data = valid_bt.copy()
    plot_data["source_grp"] = plot_data["label_source"].apply(
        lambda x: "artificial" if x.startswith("artificial") else x)
    source_positions = []
    source_labels = []
    pos = 0
    for src in sources_order:
        sub = plot_data[plot_data["source_grp"] == src]
        if len(sub) == 0:
            continue
        binders = sub[sub["binder"] == 1][bt_col].dropna()
        non_binders = sub[sub["binder"] == 0][bt_col].dropna()
        if len(binders) > 2:
            bp = ax.boxplot([binders], positions=[pos], widths=0.35,
                            patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor("#ef4444")
            bp["boxes"][0].set_alpha(0.6)
        if len(non_binders) > 2:
            bp = ax.boxplot([non_binders], positions=[pos + 0.4], widths=0.35,
                            patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor("#3b82f6")
            bp["boxes"][0].set_alpha(0.6)
        source_labels.append(src.replace("_", "\n"))
        source_positions.append(pos + 0.2)
        pos += 1.2
    ax.set_xticks(source_positions)
    ax.set_xticklabels(source_labels, fontsize=8)
    ax.set_ylabel("Binary-to-ternary ligand RMSD (A)")
    ax.set_title("A. BT RMSD by source (red=binder, blue=non-binder)")

    # Panel B: bt RMSD distribution by label
    ax = axes2[0, 1]
    for label in [0.0, 0.25, 0.75, 1.0]:
        vals = valid_bt.loc[valid_bt["label"] == label, bt_col].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=40, alpha=0.5, color=LABEL_COLORS[label],
                    label=f"{LABEL_NAMES[label]} (n={len(vals)})", density=True)
    ax.set_xlabel("Binary-to-ternary ligand RMSD (A)")
    ax.set_ylabel("Density")
    ax.set_title("B. BT RMSD by binding strength")
    ax.legend(fontsize=7)

    # Panels C & D: experimental binders — bt RMSD vs water distance and vs ternary ipTM
    exp_sources = ["experimental", "LCA_screen"]
    exp_mask = (valid_bt["label_source"].isin(exp_sources)) & (valid_bt["binder"] == 1)
    exp = valid_bt[exp_mask].copy()

    def _ligand_group(name):
        n = str(name).lower()
        if "win" in n:
            return "WIN"
        elif "lithocholic" in n or "glycolithocholic" in n:
            return "LCA"
        elif "jwh" in n or "4f-mdmb" in n:
            return "Cannabinoid"
        elif "nitazene" in n or "menitazene" in n:
            return "Nitazene"
        elif "diazinon" in n or "azinphos" in n or "pirimiphos" in n:
            return "Organophosphate"
        else:
            return "Other"

    group_colors = {"WIN": "#3b82f6", "LCA": "#22c55e", "Cannabinoid": "#8b5cf6",
                    "Nitazene": "#ef4444", "Organophosphate": "#f59e0b", "Other": "#6b7280"}

    if len(exp) > 5:
        exp["lig_group"] = exp["ligand_name"].apply(_ligand_group)

        # Panel C: bt RMSD vs water distance
        if water_col in exp.columns:
            ax = axes2[1, 0]
            for grp, color in group_colors.items():
                mask = exp["lig_group"] == grp
                sub = exp.loc[mask].dropna(subset=[water_col, bt_col])
                if len(sub) > 0:
                    ax.scatter(sub[water_col], sub[bt_col],
                               c=color, alpha=0.6, s=30, label=f"{grp} (n={len(sub)})",
                               edgecolors="white", linewidths=0.3)
            ax.axvline(3.0, color="green", ls="--", alpha=0.5)
            ax.axvline(4.0, color="red", ls="--", alpha=0.3)
            ax.set_xlabel("Binary water distance (A)")
            ax.set_ylabel("Binary-to-ternary RMSD (A)")
            ax.set_title("C. Experimental binders: BT RMSD vs water dist")
            ax.legend(fontsize=7, markerscale=1.5)

        # Panel D: bt RMSD vs ternary ipTM
        iptm_col = "af3_ternary_ipTM"
        if iptm_col in exp.columns:
            ax = axes2[1, 1]
            for grp, color in group_colors.items():
                mask = exp["lig_group"] == grp
                sub = exp.loc[mask].dropna(subset=[iptm_col, bt_col])
                if len(sub) > 0:
                    ax.scatter(sub[iptm_col], sub[bt_col],
                               c=color, alpha=0.6, s=30, label=f"{grp} (n={len(sub)})",
                               edgecolors="white", linewidths=0.3)
            ax.axvline(0.9, color="orange", ls="--", alpha=0.5, label="ipTM 0.9")
            ax.set_xlabel("Ternary ipTM")
            ax.set_ylabel("Binary-to-ternary RMSD (A)")
            ax.set_title("D. Experimental binders: BT RMSD vs ternary ipTM")
            ax.legend(fontsize=7, markerscale=1.5)

    plt.tight_layout()
    _savefig(fig2, "2_2b_bt_rmsd_landscape.png")


def section_2_3(df):
    """Water-mediated H-bond geometry from AF3."""
    print("\n" + "=" * 70)
    print("2.3  WATER-MEDIATED H-BOND GEOMETRY")
    print("=" * 70)

    dist_cols = {"binary": "af3_binary_min_dist_to_ligand_O",
                 "ternary": "af3_ternary_min_dist_to_ligand_O"}
    angle_cols = {"binary": "af3_binary_hbond_water_angle",
                  "ternary": "af3_ternary_hbond_water_angle"}
    dist_cols = {k: v for k, v in dist_cols.items() if v in df.columns}
    angle_cols = {k: v for k, v in angle_cols.items() if v in df.columns}

    if not dist_cols:
        print("  (No H-bond geometry columns found — skipping)")
        return

    # Stats for each mode
    for mode, col in dist_cols.items():
        valid = df[[col, "binder", "label"]].dropna()
        binders = valid[valid["binder"] == 1][col]
        nonbinders = valid[valid["binder"] == 0][col]
        print(f"\n  {mode.upper()} distance to conserved water:")
        print(f"    Binders    (n={len(binders):4d}): mean={binders.mean():.2f}, "
              f"median={binders.median():.2f}")
        print(f"    Non-binders(n={len(nonbinders):4d}): mean={nonbinders.mean():.2f}, "
              f"median={nonbinders.median():.2f}")
        if len(binders) > 5 and len(nonbinders) > 5:
            u, p = stats.mannwhitneyu(binders, nonbinders)
            r, rp = stats.spearmanr(valid[col], valid["binder"])
            print(f"    Mann-Whitney p={p:.2e}, Spearman r={r:.3f}")

    # Angle analysis (water-centered: Pro88:O — water:O — ligand_O)
    # Ideal tetrahedral water geometry ~ 104.5 degrees
    for mode, col in angle_cols.items():
        valid = df[[col, "binder", "label"]].dropna()
        binders = valid[valid["binder"] == 1][col]
        nonbinders = valid[valid["binder"] == 0][col]
        print(f"\n  {mode.upper()} water-centered H-bond angle:")
        print(f"    Binders    (n={len(binders):4d}): mean={binders.mean():.1f}, "
              f"median={binders.median():.1f}")
        print(f"    Non-binders(n={len(nonbinders):4d}): mean={nonbinders.mean():.1f}, "
              f"median={nonbinders.median():.1f}")
        if len(binders) > 5 and len(nonbinders) > 5:
            r, rp = stats.spearmanr(valid[col], valid["binder"])
            print(f"    Spearman r={r:.3f} (p={rp:.2e})")

    # Combined distance + angle analysis
    bcol_dist = dist_cols.get("binary")
    bcol_angle = angle_cols.get("binary")
    if bcol_dist and bcol_angle:
        valid = df[[bcol_dist, bcol_angle, "binder"]].dropna()
        # "Good geometry" = close distance AND near-tetrahedral angle
        good_geom = valid[(valid[bcol_dist] < 4.0) &
                          (valid[bcol_angle] > 70) &
                          (valid[bcol_angle] < 140)]
        close_only = valid[(valid[bcol_dist] < 4.0)]
        print(f"\n  Combined distance + angle filter (binary):")
        print(f"    Distance <4A only     : {len(close_only):4d} pairs, "
              f"{int(close_only['binder'].sum()):3d} binders "
              f"({100*close_only['binder'].mean():.1f}% precision)")
        print(f"    Distance <4A + angle 70-140: {len(good_geom):4d} pairs, "
              f"{int(good_geom['binder'].sum()):3d} binders "
              f"({100*good_geom['binder'].mean():.1f}% precision)")

    # Binding mode classification
    bcol = dist_cols.get("binary")
    if bcol:
        valid = df[df[bcol].notna()].copy()
        valid["water_mode"] = pd.cut(valid[bcol],
                                     bins=[0, 4, 6, 100],
                                     labels=["Water-engaged (<4A)",
                                             "Intermediate (4-6A)",
                                             "Non-water (>6A)"])
        print(f"\n  Binding mode classification (binary distance):")
        for mode in ["Water-engaged (<4A)", "Intermediate (4-6A)", "Non-water (>6A)"]:
            sub = valid[valid["water_mode"] == mode]
            nb = int(sub["binder"].sum())
            print(f"    {mode:25s}: {len(sub):4d} pairs, {nb:3d} binders "
                  f"({100*nb/max(len(sub),1):.1f}%)")

        # By tier
        print(f"\n  Water-engaged (<4A) binders by source:")
        close_binders = valid[(valid[bcol] < 4.0) & (valid["binder"] == 1)]
        for src, grp in close_binders.groupby("label_source"):
            print(f"    {src:25s}: {len(grp)} binders")

        # By ligand family
        families = {
            "LCA": df["ligand_name"].str.match(r"^Lithocholic", case=False, na=False),
            "WIN": df["ligand_name"].str.contains("WIN", case=False, na=False),
        }
        print(f"\n  Water distance by ligand family (binary):")
        for fam, mask in families.items():
            sub = df.loc[mask & df[bcol].notna()]
            b = sub[sub["binder"] == 1][bcol]
            nb = sub[sub["binder"] == 0][bcol]
            if len(b) > 0:
                print(f"    {fam} binders     (n={len(b):3d}): "
                      f"mean={b.mean():.2f}, median={b.median():.2f}")
            if len(nb) > 0:
                print(f"    {fam} non-binders (n={len(nb):3d}): "
                      f"mean={nb.mean():.2f}, median={nb.median():.2f}")

    # Figure: distance distributions (3 rows x 2 cols)
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    xlim_max = 15  # consistent x-axis for both binary and ternary

    # Panel 1: binary distance by label
    if bcol:
        ax = axes[0, 0]
        for label in [0.0, 0.25, 0.75, 1.0]:
            vals = df.loc[df["label"] == label, bcol].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=40, range=(0, xlim_max), alpha=0.5,
                        color=LABEL_COLORS[label],
                        label=f"{LABEL_NAMES[label]} (n={len(vals)})", density=True)
        ax.axvline(4.0, color="red", ls="--", alpha=0.5, label="4 A threshold")
        ax.set_xlim(0, xlim_max)
        ax.set_xlabel("Distance to conserved water (A)")
        ax.set_ylabel("Density")
        ax.set_title("Binary: Water distance by label")
        ax.legend(fontsize=7)

    # Panel 2: ternary distance by label (same x-axis as binary)
    tcol = dist_cols.get("ternary")
    if tcol:
        ax = axes[0, 1]
        for label in [0.0, 0.25, 0.75, 1.0]:
            vals = df.loc[df["label"] == label, tcol].dropna()
            if len(vals) > 0:
                # Clip to xlim_max for histogram binning
                vals_clipped = vals.clip(upper=xlim_max)
                ax.hist(vals_clipped, bins=40, range=(0, xlim_max), alpha=0.5,
                        color=LABEL_COLORS[label],
                        label=f"{LABEL_NAMES[label]} (n={len(vals)})", density=True)
        ax.axvline(4.0, color="red", ls="--", alpha=0.5)
        ax.set_xlim(0, xlim_max)
        ax.set_xlabel("Distance to conserved water (A)")
        ax.set_ylabel("Density")
        ax.set_title("Ternary: Water distance by label")
        ax.legend(fontsize=7)

    # Panel 3: water distance vs BINARY ipTM scatter
    if bcol and "af3_binary_ipTM" in df.columns:
        ax = axes[1, 0]
        valid = df[[bcol, "af3_binary_ipTM", "binder_name"]].dropna()
        for bname, color in BINARY_COLORS.items():
            mask = valid["binder_name"] == bname
            ax.scatter(valid.loc[mask, "af3_binary_ipTM"],
                       valid.loc[mask, bcol],
                       c=color, alpha=0.3, s=10, label=bname)
        ax.axhline(4.0, color="red", ls="--", alpha=0.5)
        ax.set_ylim(0, xlim_max)
        ax.set_xlabel("AF3 Binary ipTM")
        ax.set_ylabel("Water distance (A)")
        ax.set_title("Water distance vs Binary ipTM")
        ax.legend(fontsize=8, markerscale=2)

    # Panel 4: water distance vs TERNARY ipTM scatter
    if bcol and "af3_ternary_ipTM" in df.columns:
        ax = axes[1, 1]
        valid = df[[bcol, "af3_ternary_ipTM", "binder_name"]].dropna()
        for bname, color in BINARY_COLORS.items():
            mask = valid["binder_name"] == bname
            ax.scatter(valid.loc[mask, "af3_ternary_ipTM"],
                       valid.loc[mask, bcol],
                       c=color, alpha=0.3, s=10, label=bname)
        ax.axhline(4.0, color="red", ls="--", alpha=0.5)
        ax.axhline(3.0, color="green", ls="--", alpha=0.5, label="3 A threshold")
        ax.axvline(0.9, color="orange", ls="--", alpha=0.5, label="ipTM 0.9")
        ax.set_ylim(0, xlim_max)
        ax.set_xlabel("AF3 Ternary ipTM")
        ax.set_ylabel("Water distance (A)")
        ax.set_title("Water distance vs Ternary ipTM")
        ax.legend(fontsize=7, markerscale=2)

    # Panel 5: by ligand family
    if bcol:
        ax = axes[2, 0]
        families_ext = {
            "LCA binder": (df["ligand_name"].str.match(r"^Lithocholic", case=False, na=False) &
                           (df["binder"] == 1)),
            "LCA non-bind": (df["ligand_name"].str.match(r"^Lithocholic", case=False, na=False) &
                             (df["binder"] == 0)),
            "WIN binder": (df["ligand_name"].str.contains("WIN", case=False, na=False) &
                           (df["binder"] == 1)),
            "WIN non-bind": (df["ligand_name"].str.contains("WIN", case=False, na=False) &
                             (df["binder"] == 0)),
            "PNAS binder": ((df["label_source"] == "pnas_cutler") &
                            (df["binder"] == 1)),
        }
        fam_colors = ["#22c55e", "#86efac", "#3b82f6", "#93c5fd", "#f59e0b"]
        for (fname, mask), color in zip(families_ext.items(), fam_colors):
            vals = df.loc[mask, bcol].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=30, range=(0, xlim_max), alpha=0.5, color=color,
                        label=f"{fname} (n={len(vals)})", density=True)
        ax.axvline(4.0, color="red", ls="--", alpha=0.5)
        ax.set_xlim(0, xlim_max)
        ax.set_xlabel("Binary water distance (A)")
        ax.set_ylabel("Density")
        ax.set_title("Water distance by ligand family")
        ax.legend(fontsize=7)

    # Panel 6: hide unused panel
    axes[2, 1].axis("off")

    plt.tight_layout()
    _savefig(fig, "2_3_water_geometry.png")

    # --- Second figure: experimental binders landscape ---
    exp_sources = ["experimental", "LCA_screen"]
    exp_mask = (df["label_source"].isin(exp_sources)) & (df["binder"] == 1)
    exp = df[exp_mask].copy()
    if len(exp) < 5 or bcol not in exp.columns:
        return

    print(f"\n  Experimental binder landscape: {len(exp)} binders")

    # Assign colors by ligand family
    def _ligand_group(name):
        n = str(name).lower()
        if "win" in n:
            return "WIN"
        elif "lithocholic" in n or "glycolithocholic" in n:
            return "LCA"
        elif "jwh" in n or "4f-mdmb" in n:
            return "Cannabinoid"
        elif "nitazene" in n or "menitazene" in n:
            return "Nitazene"
        elif "diazinon" in n or "azinphos" in n or "pirimiphos" in n:
            return "Organophosphate"
        else:
            return "Other"

    exp["lig_group"] = exp["ligand_name"].apply(_ligand_group)
    group_colors = {"WIN": "#3b82f6", "LCA": "#22c55e", "Cannabinoid": "#8b5cf6",
                    "Nitazene": "#ef4444", "Organophosphate": "#f59e0b", "Other": "#6b7280"}

    plot_specs = []
    if "af3_binary_ipTM" in exp.columns:
        plot_specs.append(("af3_binary_ipTM", "Binary ipTM"))
    if "af3_binary_pLDDT_ligand" in exp.columns:
        plot_specs.append(("af3_binary_pLDDT_ligand", "Binary pLDDT (ligand)"))
    if "af3_ternary_ipTM" in exp.columns:
        plot_specs.append(("af3_ternary_ipTM", "Ternary ipTM"))
    if "af3_ternary_pLDDT_ligand" in exp.columns:
        plot_specs.append(("af3_ternary_pLDDT_ligand", "Ternary pLDDT (ligand)"))

    if not plot_specs:
        return

    n_panels = len(plot_specs)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows))
    if nrows == 1:
        axes2 = axes2.reshape(1, -1)

    for idx, (col, title) in enumerate(plot_specs):
        ax = axes2[idx // ncols, idx % ncols]
        valid = exp[[bcol, col, "lig_group", "ligand_name"]].dropna()
        for grp, color in group_colors.items():
            mask = valid["lig_group"] == grp
            if mask.sum() > 0:
                ax.scatter(valid.loc[mask, bcol], valid.loc[mask, col],
                           c=color, alpha=0.6, s=30, label=f"{grp} (n={mask.sum()})",
                           edgecolors="white", linewidths=0.3)
        ax.axvline(3.0, color="green", ls="--", alpha=0.5, label="3 A")
        ax.axvline(4.0, color="red", ls="--", alpha=0.3)
        ax.set_xlim(0, xlim_max)
        ax.set_xlabel("Binary water distance (A)")
        ax.set_ylabel(title)
        ax.set_title(f"Experimental binders: {title} vs water distance")
        ax.legend(fontsize=7, markerscale=1.5)

    # Hide unused panels
    for idx in range(n_panels, nrows * ncols):
        axes2[idx // ncols, idx % ncols].axis("off")

    plt.tight_layout()
    _savefig(fig2, "2_3b_experimental_binder_landscape.png")


def section_2_4(df):
    """AF3 ligand RMSD (agreement with Rosetta docking)."""
    print("\n" + "=" * 70)
    print("2.4  AF3 LIGAND RMSD (AF3-ROSETTA AGREEMENT)")
    print("=" * 70)

    warnings.filterwarnings("ignore", message="An input array is constant")

    rmsd_cols = {
        "af3_binary_ligand_RMSD_min": "Binary RMSD (min)",
        "af3_binary_ligand_RMSD_bestdG": "Binary RMSD (best dG)",
        "af3_ternary_ligand_RMSD_min": "Ternary RMSD (min)",
        "af3_ternary_ligand_RMSD_bestdG": "Ternary RMSD (best dG)",
    }
    rmsd_cols = {k: v for k, v in rmsd_cols.items() if k in df.columns}

    # Stats
    print("\n  Ligand RMSD (AF3 vs Rosetta) by label:")
    for col, name in rmsd_cols.items():
        print(f"\n  {name}:")
        for label in sorted(df["label"].unique()):
            vals = df.loc[df["label"] == label, col].dropna()
            if len(vals) > 3:
                print(f"    label={label}: mean={vals.mean():.2f}, "
                      f"median={vals.median():.2f} (n={len(vals)})")
        valid = df[["binder", col]].dropna()
        if len(valid) > 30:
            r, p = stats.spearmanr(valid["binder"], valid[col])
            print(f"    Spearman r with binder: {r:.3f} (p={p:.2e})")

    # Figure: RMSD distributions + scatter
    n_cols = min(len(rmsd_cols), 2)
    fig, axes = plt.subplots(1, n_cols + 1, figsize=(6 * (n_cols + 1), 5))

    for idx, (col, name) in enumerate(list(rmsd_cols.items())[:n_cols]):
        ax = axes[idx]
        for bname, color in BINARY_COLORS.items():
            vals = df.loc[df["binder_name"] == bname, col].dropna()
            if len(vals) > 0:
                ax.hist(vals.clip(upper=15), bins=40, alpha=0.6, color=color,
                        label=bname, density=True)
        ax.set_xlabel(f"{name} (A)")
        ax.set_ylabel("Density")
        ax.set_title(name)
        ax.legend(fontsize=8)

    # Scatter: RMSD vs ipTM
    b_rmsd = "af3_binary_ligand_RMSD_min"
    b_iptm = "af3_binary_ipTM"
    if b_rmsd in df.columns and b_iptm in df.columns:
        ax = axes[-1]
        valid = df[[b_rmsd, b_iptm, "binder_name"]].dropna()
        for bname, color in BINARY_COLORS.items():
            mask = valid["binder_name"] == bname
            ax.scatter(valid.loc[mask, b_iptm],
                       valid.loc[mask, b_rmsd].clip(upper=15),
                       c=color, alpha=0.3, s=10, label=bname)
        ax.set_xlabel("AF3 Binary ipTM")
        ax.set_ylabel("Ligand RMSD to Rosetta (A)")
        ax.set_title("Structural consensus: RMSD vs ipTM")
        ax.legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    _savefig(fig, "2_4_af3_rosetta_rmsd.png")


# ═════════════════════════════════════════════════════════════════
# PART 3: ROSETTA SCORING ANALYSIS
# ═════════════════════════════════════════════════════════════════

def section_3_1(df):
    """Rosetta features by binding strength."""
    print("\n" + "=" * 70)
    print("3.1  ROSETTA FEATURES BY BINDING STRENGTH")
    print("=" * 70)

    rosetta_feats = _feature_cols(df, stages=["rosetta"])
    if not rosetta_feats:
        print("  No Rosetta features found")
        return

    # Global correlations
    print("\n  Rosetta feature correlations with label (global):")
    corrs = []
    for f in rosetta_feats:
        valid = df[["label", f]].dropna()
        if len(valid) > 30:
            r, p = stats.spearmanr(valid["label"], valid[f])
            if not np.isnan(r):
                corrs.append((f, r, p, len(valid)))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, r, p, n in corrs:
        print(f"    {feat:40s}: r={r:+.3f} (p={p:.2e}, n={n})")

    # Per-family correlations
    families = {
        "LCA": df["ligand_name"].str.match(r"^Lithocholic", case=False, na=False),
        "WIN": df["ligand_name"].str.contains("WIN", case=False, na=False),
        "All": pd.Series(True, index=df.index),
    }
    print("\n  Rosetta dG_sep_best correlation by ligand family:")
    dg_col = "rosetta_dG_sep_best"
    if dg_col in df.columns:
        for fam, mask in families.items():
            valid = df.loc[mask, ["label", dg_col]].dropna()
            if len(valid) > 30:
                r, p = stats.spearmanr(valid["label"], valid[dg_col])
                print(f"    {fam:10s}: r={r:+.3f} (p={p:.2e}, n={len(valid)})")

    # Figure: Rosetta distributions by label
    top_feats = [f for f, _, _, _ in corrs[:6]]
    n_plots = len(top_feats)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, feat in enumerate(top_feats):
        ax = axes[idx]
        for bname, color in BINARY_COLORS.items():
            vals = df.loc[df["binder_name"] == bname, feat].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=30, alpha=0.6, color=color,
                        label=bname, density=True)
        ax.set_xlabel(feat.replace("rosetta_", ""), fontsize=9)
        ax.set_title(f"r={corrs[idx][1]:.3f}", fontsize=10)
        ax.legend(fontsize=7)
    for idx in range(len(top_feats), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Top Rosetta Features: Binder vs Non-binder", fontsize=13, y=1.01)
    plt.tight_layout()
    _savefig(fig, "3_1_rosetta_by_label.png")


def section_3_2(df):
    """Rosetta vs AF3 agreement."""
    print("\n" + "=" * 70)
    print("3.2  ROSETTA vs AF3 AGREEMENT")
    print("=" * 70)

    dg_col = "rosetta_dG_sep_best"
    iptm_col = "af3_binary_ipTM"

    if dg_col not in df.columns or iptm_col not in df.columns:
        print("  Missing required columns")
        return

    valid = df[[dg_col, iptm_col, "binder", "binder_name", "label"]].dropna()
    r, p = stats.spearmanr(valid[dg_col], valid[iptm_col])
    print(f"\n  Rosetta dG_sep_best vs AF3 binary ipTM:")
    print(f"    Spearman r={r:.3f} (p={p:.2e}, n={len(valid)})")

    # Disagreement analysis
    # High ipTM but poor dG = AF3 says binder, Rosetta says no
    q_iptm = valid[iptm_col].quantile(0.75)
    q_dg = valid[dg_col].quantile(0.25)  # more negative = better
    agree_good = valid[(valid[iptm_col] > q_iptm) & (valid[dg_col] < q_dg)]
    disagree_af3 = valid[(valid[iptm_col] > q_iptm) & (valid[dg_col] >= q_dg)]
    disagree_ros = valid[(valid[iptm_col] <= q_iptm) & (valid[dg_col] < q_dg)]
    agree_bad = valid[(valid[iptm_col] <= q_iptm) & (valid[dg_col] >= q_dg)]

    print(f"\n  Agreement analysis (ipTM q75={q_iptm:.3f}, dG q25={q_dg:.1f}):")
    for name, sub in [("Both good (agree)", agree_good),
                      ("AF3 good, Rosetta bad", disagree_af3),
                      ("Rosetta good, AF3 bad", disagree_ros),
                      ("Both bad (agree)", agree_bad)]:
        if len(sub) > 0:
            br = sub["binder"].mean()
            print(f"    {name:30s}: n={len(sub):4d}, binder rate={br:.1%}")

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    for bname, color in BINARY_COLORS.items():
        mask = valid["binder_name"] == bname
        ax1.scatter(valid.loc[mask, iptm_col], valid.loc[mask, dg_col],
                    c=color, alpha=0.3, s=10, label=bname)
    ax1.set_xlabel("AF3 Binary ipTM")
    ax1.set_ylabel("Rosetta dG_sep_best (REU)")
    ax1.set_title(f"AF3 vs Rosetta (Spearman r={r:.3f})")
    ax1.legend(fontsize=8, markerscale=2)

    # Correlation matrix of key features
    key_feats = [c for c in ["af3_binary_ipTM", "af3_ternary_ipTM",
                             "af3_binary_pLDDT_ligand", "af3_binary_interface_PAE",
                             "rosetta_dG_sep_best", "rosetta_dG_sep_mean",
                             "rosetta_hbonds_to_ligand_mean",
                             "rosetta_dsasa_int_mean",
                             "docking_convergence_ratio", "docking_best_score"]
                 if c in df.columns]
    corr_mat = df[key_feats].corr(method="spearman")
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax2,
                xticklabels=[f.replace("_", "\n") for f in key_feats],
                yticklabels=[f.replace("_", "\n") for f in key_feats])
    ax2.set_title("Feature cross-correlation (Spearman)")
    ax2.tick_params(labelsize=7)

    plt.tight_layout()
    _savefig(fig, "3_2_rosetta_vs_af3.png")


def section_3_3(df):
    """Rosetta water placement potential."""
    print("\n" + "=" * 70)
    print("3.3  ROSETTA FEATURES vs WATER ENGAGEMENT")
    print("=" * 70)

    bcol = "af3_binary_min_dist_to_ligand_O"
    if bcol not in df.columns:
        print("  No water distance column — skipping")
        return

    rosetta_feats = ["rosetta_hbonds_to_ligand_mean", "rosetta_buried_unsats_best",
                     "rosetta_buried_unsats_mean", "rosetta_dG_sep_best",
                     "rosetta_dsasa_int_mean", "rosetta_nres_int_mean"]
    rosetta_feats = [f for f in rosetta_feats if f in df.columns]

    # Classify water mode
    valid = df[df[bcol].notna()].copy()
    valid["water_mode"] = "Intermediate"
    valid.loc[valid[bcol] < 4.0, "water_mode"] = "Water-engaged"
    valid.loc[valid[bcol] >= 6.0, "water_mode"] = "Non-water"

    print("\n  Rosetta features by water engagement mode (binders only):")
    binders = valid[valid["binder"] == 1]
    for feat in rosetta_feats:
        print(f"\n  {feat}:")
        for mode in ["Water-engaged", "Intermediate", "Non-water"]:
            vals = binders.loc[binders["water_mode"] == mode, feat].dropna()
            if len(vals) > 3:
                print(f"    {mode:18s}: mean={vals.mean():.3f}, "
                      f"median={vals.median():.3f} (n={len(vals)})")

    # Can Rosetta predict water engagement?
    print("\n  Rosetta features predicting water engagement (all pairs):")
    valid_feats = valid[rosetta_feats + [bcol]].dropna()
    for feat in rosetta_feats:
        r, p = stats.spearmanr(valid_feats[feat], valid_feats[bcol])
        print(f"    {feat:40s}: r={r:+.3f} (p={p:.2e})")

    # Figure
    n_feats = min(len(rosetta_feats), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    mode_colors = {"Water-engaged": "#22c55e", "Intermediate": "#f59e0b",
                   "Non-water": "#ef4444"}
    for idx, feat in enumerate(rosetta_feats[:n_feats]):
        ax = axes[idx]
        for mode, color in mode_colors.items():
            vals = binders.loc[binders["water_mode"] == mode, feat].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=20, alpha=0.5, color=color,
                        label=f"{mode} (n={len(vals)})", density=True)
        ax.set_xlabel(feat.replace("rosetta_", ""), fontsize=9)
        ax.set_title(feat.replace("rosetta_", ""), fontsize=10)
        ax.legend(fontsize=7)
    for idx in range(n_feats, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Rosetta Features by Water Engagement Mode (Binders Only)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _savefig(fig, "3_3_rosetta_water_engagement.png")


# ═════════════════════════════════════════════════════════════════
# PART 4: WITHIN-FAMILY DEEP DIVES
# ═════════════════════════════════════════════════════════════════

def section_4_1(df):
    """LCA family deep dive."""
    print("\n" + "=" * 70)
    print("4.1  LCA FAMILY DEEP DIVE")
    print("=" * 70)

    lca_mask = df["ligand_name"].str.match(r"^Lithocholic", case=False, na=False)
    lca = df[lca_mask].copy()

    print(f"\n  LCA pairs: {len(lca)}")
    print(f"  Binders: {int(lca['binder'].sum())}")
    print(f"  Ligands: {lca['ligand_name'].value_counts().to_dict()}")

    # Per-source
    print(f"\n  Per source:")
    for src, grp in lca.groupby("label_source"):
        nb = int(grp["binder"].sum())
        print(f"    {src:25s}: {len(grp)} pairs ({nb} binders)")

    # Variant overlap: bind both LCA and LCA-3S?
    lca_plain = lca[lca["ligand_name"] == "Lithocholic Acid"]
    lca_3s = lca[lca["ligand_name"] == "Lithocholic Acid 3 -S"]
    binder_sigs_plain = set(
        lca_plain[lca_plain["binder"] == 1]["variant_signature"].dropna())
    binder_sigs_3s = set(
        lca_3s[lca_3s["binder"] == 1]["variant_signature"].dropna())
    both = binder_sigs_plain & binder_sigs_3s
    only_plain = binder_sigs_plain - binder_sigs_3s
    only_3s = binder_sigs_3s - binder_sigs_plain
    print(f"\n  Variant specificity (LCA vs LCA-3S):")
    print(f"    Bind both: {len(both)}")
    print(f"    Only LCA: {len(only_plain)}")
    print(f"    Only LCA-3S: {len(only_3s)}")

    # Top features for LCA (filter to rows with features first)
    lca_with_feats = lca.dropna(subset=_feature_cols(lca)[:1]) if _feature_cols(lca) else lca
    all_feats = _feature_cols(lca_with_feats, exclude_sparse=True)
    if not all_feats:
        all_feats = _feature_cols(lca)
    print(f"\n  Top 10 features for LCA binding (Spearman, n={len(lca_with_feats)} with features):")
    corrs = []
    for f in all_feats:
        valid = lca[["label", f]].dropna()
        if len(valid) > 30:
            r, p = stats.spearmanr(valid["label"], valid[f])
            if not np.isnan(r):
                corrs.append((f, r, p))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, r, p in corrs[:10]:
        d = "+" if r > 0 else "-"
        print(f"    {d} {feat}: r={r:.3f} (p={p:.2e})")


def section_4_2(df):
    """WIN family deep dive."""
    print("\n" + "=" * 70)
    print("4.2  WIN FAMILY DEEP DIVE")
    print("=" * 70)

    win_mask = df["ligand_name"].str.contains("WIN", case=False, na=False)
    win = df[win_mask].copy()

    print(f"\n  WIN pairs: {len(win)}")
    print(f"  Binders: {int(win['binder'].sum())}")

    # Per-source
    for src, grp in win.groupby("label_source"):
        nb = int(grp["binder"].sum())
        print(f"    {src:25s}: {len(grp)} pairs ({nb} binders)")

    # Kd regression analysis
    if "affinity_uM" in win.columns:
        win_kd = win[win["affinity_uM"].notna()].copy()
        if len(win_kd) > 20:
            print(f"\n  WIN Kd regression analysis (n={len(win_kd)}):")
            win_kd["log_kd"] = np.log10(win_kd["affinity_uM"].clip(lower=0.01))

            all_feats = _feature_cols(win_kd, exclude_sparse=True)
            if not all_feats:
                all_feats = _feature_cols(win_kd)
            corrs = []
            for f in all_feats:
                valid = win_kd[["log_kd", f]].dropna()
                if len(valid) > 20:
                    r, p = stats.spearmanr(valid["log_kd"], valid[f])
                    if not np.isnan(r):
                        corrs.append((f, r, p, len(valid)))
            corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n  Top 10 features predicting WIN Kd (Spearman):")
            for feat, r, p, n in corrs[:10]:
                d = "+" if r > 0 else "-"
                print(f"    {d} {feat}: r={r:.3f} (p={p:.2e}, n={n})")
        else:
            print(f"\n  WIN Kd regression: only {len(win_kd)} rows with affinity_uM — skipping")
    else:
        print("\n  WIN Kd regression: affinity_uM column not found in CSV")

    # Top features for WIN binding
    all_feats = _feature_cols(win, exclude_sparse=True)
    corrs = []
    for f in all_feats:
        valid = win[["label", f]].dropna()
        if len(valid) > 30:
            r, p = stats.spearmanr(valid["label"], valid[f])
            if not np.isnan(r):
                corrs.append((f, r, p))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top 10 features for WIN binding (Spearman):")
    for feat, r, p in corrs[:10]:
        d = "+" if r > 0 else "-"
        print(f"    {d} {feat}: r={r:.3f} (p={p:.2e})")


def section_4_3(df):
    """PNAS diverse ligands deep dive."""
    print("\n" + "=" * 70)
    print("4.3  PNAS DIVERSE LIGANDS DEEP DIVE")
    print("=" * 70)

    pnas = df[df["label_source"] == "pnas_cutler"].copy()
    bcol = "af3_binary_min_dist_to_ligand_O"

    print(f"\n  PNAS pairs: {len(pnas)}")
    print(f"  Binders: {int(pnas['binder'].sum())}")
    print(f"  Unique ligands: {pnas['ligand_name'].nunique()}")

    if bcol not in df.columns:
        return

    pnas_binders = pnas[pnas["binder"] == 1].copy()
    pnas_valid = pnas_binders[pnas_binders[bcol].notna()]

    # Water mode classification
    water_engaged = pnas_valid[pnas_valid[bcol] < 4.0]
    non_water = pnas_valid[pnas_valid[bcol] >= 6.0]

    print(f"\n  PNAS binder binding modes:")
    print(f"    Water-engaged (<4A): {len(water_engaged)} binders")
    print(f"    Non-water (>6A): {len(non_water)} binders")

    if len(water_engaged) > 0:
        print(f"\n  Water-engaged PNAS binders:")
        for _, row in water_engaged.sort_values(bcol).iterrows():
            print(f"    {row['ligand_name']:30s} dist={row[bcol]:.2f} "
                  f"ipTM={row.get('af3_binary_ipTM', float('nan')):.3f}")

    # Compare water-engaged PNAS binders to tier 1
    if len(water_engaged) > 0:
        t1_binders = df[(df["label_source"] == "experimental") &
                        (df["binder"] == 1)]
        compare_feats = ["af3_binary_ipTM", "af3_binary_pLDDT_ligand",
                         "rosetta_dG_sep_best", "docking_convergence_ratio",
                         bcol]
        compare_feats = [f for f in compare_feats if f in df.columns]

        print(f"\n  Feature comparison: water-engaged PNAS vs Tier 1 binders:")
        print(f"  {'Feature':40s} {'PNAS (n=' + str(len(water_engaged)) + ')':>20s} "
              f"{'Tier 1 (n=' + str(len(t1_binders)) + ')':>20s}")
        print("  " + "-" * 82)
        for feat in compare_feats:
            pnas_vals = water_engaged[feat].dropna()
            t1_vals = t1_binders[feat].dropna()
            if len(pnas_vals) > 0 and len(t1_vals) > 0:
                print(f"  {feat:40s} {pnas_vals.mean():>19.3f} "
                      f"{t1_vals.mean():>19.3f}")

    # ── PNAS potency analysis by min_conc (affinity_uM) ──
    if "affinity_uM" in pnas.columns:
        pnas_aff = pnas[pnas["affinity_uM"].notna()].copy()
        pnas_aff["affinity_uM"] = pd.to_numeric(pnas_aff["affinity_uM"],
                                                  errors="coerce")
        pnas_aff = pnas_aff[pnas_aff["affinity_uM"].notna()]

        if len(pnas_aff) > 0:
            print(f"\n  PNAS potency analysis (min_conc from affinity_uM):")
            print(f"    Pairs with affinity_uM: {len(pnas_aff)}")

            # Group by concentration bucket
            conc_bins = {
                "1 uM (strong)": pnas_aff[pnas_aff["affinity_uM"] <= 1.0],
                "10 uM (moderate)": pnas_aff[(pnas_aff["affinity_uM"] > 1.0) &
                                              (pnas_aff["affinity_uM"] <= 10.0)],
                "100 uM (weak)": pnas_aff[pnas_aff["affinity_uM"] > 10.0],
            }

            print(f"\n    Concentration distribution:")
            for label, grp in conc_bins.items():
                print(f"      {label:25s}: n={len(grp)}")

            # Feature comparison across concentration groups
            compare_feats_potency = [
                "af3_binary_ipTM", "af3_ternary_ipTM",
                "af3_binary_pLDDT_ligand", "af3_ternary_pLDDT_ligand",
                bcol, "af3_ternary_min_dist_to_ligand_O",
                "rosetta_dG_sep_best", "rosetta_dG_sep_mean",
                "rosetta_hbonds_to_ligand_mean",
                "docking_convergence_ratio", "docking_best_score",
            ]
            compare_feats_potency = [f for f in compare_feats_potency
                                     if f in pnas_aff.columns]

            print(f"\n    Feature means by PNAS concentration group:")
            header = f"    {'Feature':40s}"
            for label in conc_bins:
                short = label.split("(")[0].strip()
                header += f" {short:>12s}"
            print(header)
            print("    " + "-" * (40 + 12 * len(conc_bins)))

            for feat in compare_feats_potency:
                line = f"    {feat:40s}"
                for label, grp in conc_bins.items():
                    vals = grp[feat].dropna()
                    if len(vals) > 0:
                        line += f" {vals.mean():>11.3f}"
                    else:
                        line += f" {'n/a':>11s}"
                print(line)

            # Spearman correlation: features vs log(concentration)
            pnas_aff["log_conc"] = np.log10(pnas_aff["affinity_uM"].clip(
                lower=0.01))
            print(f"\n    Feature correlation with log10(min_conc) "
                  f"(n={len(pnas_aff)}):")
            print(f"    (negative r = lower conc = stronger binding)")
            corrs = []
            for feat in compare_feats_potency:
                valid = pnas_aff[["log_conc", feat]].dropna()
                if len(valid) > 10:
                    r, p = stats.spearmanr(valid["log_conc"], valid[feat])
                    if not np.isnan(r):
                        corrs.append((feat, r, p, len(valid)))
            corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            for feat, r, p, n in corrs:
                d = "+" if r > 0 else "-"
                print(f"      {d} {feat:38s}: r={r:+.3f} (p={p:.2e}, n={n})")

            # Water engagement by concentration
            if bcol in pnas_aff.columns:
                print(f"\n    Water engagement by concentration:")
                for label, grp in conc_bins.items():
                    valid = grp[grp[bcol].notna()]
                    if len(valid) > 0:
                        we = (valid[bcol] < 4.0).sum()
                        nw = (valid[bcol] >= 6.0).sum()
                        mid = len(valid) - we - nw
                        print(f"      {label:25s}: water-engaged={we}, "
                              f"intermediate={mid}, non-water={nw}")
        else:
            print("\n  PNAS potency: no valid affinity_uM values found")
    else:
        print("\n  PNAS potency: affinity_uM column not in CSV")


# ═════════════════════════════════════════════════════════════════
# PART 5: ML MODELS & FILTER SELECTION
# ═════════════════════════════════════════════════════════════════

def section_5_1(df):
    """Global model analysis."""
    print("\n" + "=" * 70)
    print("5.1  GLOBAL MODEL ANALYSIS")
    print("=" * 70)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, average_precision_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("  scikit-learn not installed — skipping")
        return

    # Build feature sets
    docking_feats = _feature_cols(df, stages=["docking"])
    rosetta_feats = _feature_cols(df, stages=["rosetta"])
    hbond_geom_cols = {"af3_binary_min_dist_to_ligand_O",
                       "af3_ternary_min_dist_to_ligand_O",
                       "af3_binary_hbond_water_angle",
                       "af3_ternary_hbond_water_angle"}
    af3_confidence = [f for f in _feature_cols(df, stages=["af3"])
                      if f not in hbond_geom_cols
                      and df[f].notna().sum() / len(df) >= 0.5]
    af3_with_hbond = af3_confidence + [f for f in _feature_cols(df, stages=["af3"])
                                        if f in hbond_geom_cols
                                        and f in df.columns]
    af3_all = _feature_cols(df, stages=["af3"])
    conformer_feats = _feature_cols(df, stages=["conformer"])

    bcol = "af3_binary_min_dist_to_ligand_O"

    models = [
        ("A: Docking only", docking_feats),
        ("B: Docking + Rosetta", docking_feats + rosetta_feats),
        ("C: AF3 confidence (no geom)", conformer_feats + docking_feats + rosetta_feats + af3_confidence),
        ("D: + H-bond geometry", conformer_feats + docking_feats + rosetta_feats + af3_with_hbond),
    ]

    # Model E: water-mediated only
    if bcol in df.columns:
        water_mask = (df[bcol] < 5.0) | (df["binder"] == 0)
        water_df = df[water_mask | df[bcol].isna()].copy()
        # Relabel: only water-engaged binders count
        water_df.loc[water_df[bcol].notna() & (water_df[bcol] >= 5.0), "binder"] = 0
    else:
        water_df = None

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        use_group = True
    except ImportError:
        use_group = False

    results = []
    for name, feats in models:
        if not feats:
            continue
        sub = df[feats + ["binder", "ligand_name"]].dropna(subset=feats + ["binder"])
        n_binders = int(sub["binder"].sum())
        if len(sub) < 50 or n_binders < 10:
            print(f"\n  [{name}] Insufficient data ({len(sub)} rows, "
                  f"{n_binders} binders) — skipping")
            continue

        X = sub[feats].values
        y = sub["binder"].values
        groups = sub["ligand_name"].values

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1))
        ])

        if use_group and len(set(groups)) >= 5:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            cv_type = "GroupKFold"
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_type = "StratifiedKFold"
            groups = None

        y_prob = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y, groups):
            pipe.fit(X[train_idx], y[train_idx])
            y_prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

        auc_roc = roc_auc_score(y, y_prob)
        auc_pr = average_precision_score(y, y_prob)

        # Feature importance
        pipe.fit(X, y)
        importances = pipe.named_steps["clf"].feature_importances_
        imp_df = pd.DataFrame({"feature": feats, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)

        print(f"\n  [{name}] {len(sub)} rows, {n_binders} binders, "
              f"{len(feats)} features ({cv_type})")
        print(f"    ROC-AUC: {auc_roc:.3f}  PR-AUC: {auc_pr:.3f} "
              f"(baseline: {y.mean():.3f})")
        print(f"    Top 5 features: "
              f"{', '.join(imp_df.head(5)['feature'].tolist())}")

        results.append({
            "name": name, "auc_roc": auc_roc, "auc_pr": auc_pr,
            "n": len(sub), "n_binders": n_binders,
            "imp_df": imp_df, "y": y, "y_prob": y_prob,
        })

    # Model E: water-mediated only (uses all features incl. H-bond geometry)
    if water_df is not None:
        all_core = conformer_feats + docking_feats + rosetta_feats + af3_with_hbond
        sub = water_df[all_core + ["binder", "ligand_name"]].dropna(
            subset=all_core + ["binder"])
        n_binders = int(sub["binder"].sum())
        if len(sub) >= 50 and n_binders >= 10:
            X = sub[all_core].values
            y = sub["binder"].values
            groups = sub["ligand_name"].values

            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1))
            ])

            if use_group and len(set(groups)) >= 5:
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                groups = None

            y_prob = np.zeros(len(y))
            for train_idx, test_idx in cv.split(X, y, groups):
                pipe.fit(X[train_idx], y[train_idx])
                y_prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

            auc_roc = roc_auc_score(y, y_prob)
            auc_pr = average_precision_score(y, y_prob)

            pipe.fit(X, y)
            importances = pipe.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({"feature": all_core, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False)

            print(f"\n  [E: Water-mediated only] {len(sub)} rows, "
                  f"{n_binders} binders (non-water binders relabeled as 0)")
            print(f"    ROC-AUC: {auc_roc:.3f}  PR-AUC: {auc_pr:.3f}")
            print(f"    Top 5 features: "
                  f"{', '.join(imp_df.head(5)['feature'].tolist())}")

            results.append({
                "name": "E: Water-mediated only", "auc_roc": auc_roc,
                "auc_pr": auc_pr, "n": len(sub), "n_binders": n_binders,
                "imp_df": imp_df, "y": y, "y_prob": y_prob,
            })

    # Figure: model comparison
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart of AUCs
        names = [r["name"] for r in results]
        roc_aucs = [r["auc_roc"] for r in results]
        pr_aucs = [r["auc_pr"] for r in results]
        x = np.arange(len(names))

        ax1.bar(x - 0.15, roc_aucs, 0.3, label="ROC-AUC", color="#3b82f6")
        ax1.bar(x + 0.15, pr_aucs, 0.3, label="PR-AUC", color="#ef4444")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
        ax1.set_ylabel("AUC")
        ax1.set_title("Model Comparison")
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Feature importance for best model
        best = max(results, key=lambda r: r["auc_roc"])
        imp = best["imp_df"].head(15)
        ax2.barh(imp["feature"][::-1], imp["importance"][::-1], color="#64748b")
        ax2.set_xlabel("Importance (Gini)")
        ax2.set_title(f"Top Features — {best['name']}")
        ax2.tick_params(labelsize=8)

        plt.tight_layout()
        _savefig(fig, "5_1_model_comparison.png")


def section_5_3(df):
    """Optimal filter combinations for design."""
    print("\n" + "=" * 70)
    print("5.3  OPTIMAL FILTER COMBINATIONS FOR DESIGN")
    print("=" * 70)

    # Single-feature thresholds
    candidates = {
        "af3_binary_min_dist_to_ligand_O": ("<=", [2.5, 3.0, 3.5, 4.0, 5.0]),
        "af3_binary_ipTM": (">=", [0.7, 0.75, 0.8, 0.85, 0.9]),
        "af3_ternary_ipTM": (">=", [0.7, 0.75, 0.8, 0.85, 0.9]),
        "af3_binary_pLDDT_ligand": (">=", [50, 60, 70, 80]),
        "docking_convergence_ratio": ("<=", [0.1, 0.15, 0.2, 0.3]),
        "rosetta_dG_sep_best": ("<=", [-5, -10, -15, -20]),
    }

    # Focus on strong binders (label >= 0.75) vs all others
    print("\n  Single-feature threshold analysis:")
    print(f"  Total: {len(df)} pairs, {int(df['binder'].sum())} binders")
    print(f"\n  {'Feature':45s} {'Thresh':>8s} {'Pass':>6s} {'TP':>5s} "
          f"{'Prec':>8s} {'Recall':>8s} {'Enrich':>8s}")
    print("  " + "-" * 90)

    best_filters = []
    for feat, (op, thresholds) in candidates.items():
        if feat not in df.columns:
            continue
        valid = df[df[feat].notna()].copy()
        n_binders = int(valid["binder"].sum())
        if n_binders < 10:
            continue

        for thresh in thresholds:
            if op == "<=":
                passing = valid[valid[feat] <= thresh]
            else:
                passing = valid[valid[feat] >= thresh]

            n_pass = len(passing)
            tp = int(passing["binder"].sum())
            if n_pass == 0:
                continue
            precision = tp / n_pass
            recall = tp / n_binders if n_binders > 0 else 0
            enrichment = precision / (n_binders / len(valid)) if n_binders > 0 else 0

            print(f"  {feat:45s} {op}{thresh:<7} {n_pass:>5d} {tp:>5d} "
                  f"{precision:>7.1%} {recall:>7.1%} {enrichment:>7.1f}x")

            best_filters.append({
                "feature": feat, "op": op, "threshold": thresh,
                "n_pass": n_pass, "tp": tp, "precision": precision,
                "recall": recall, "enrichment": enrichment,
            })

    # Best 2-feature combinations
    if best_filters:
        print("\n\n  Top 2-feature filter combinations:")
        print(f"  (testing Pareto-optimal single filters combined)")

        # Pick top single filters by enrichment (>1.5x, recall >5%)
        good_singles = [f for f in best_filters
                        if f["enrichment"] > 1.5 and f["recall"] > 0.05]
        if not good_singles:
            # Fallback: just take the top 8 by enrichment
            good_singles = sorted(best_filters, key=lambda x: x["enrichment"],
                                  reverse=True)[:8]
        good_singles.sort(key=lambda x: x["enrichment"], reverse=True)

        # Test pairwise combinations of top singles from different features
        seen_pairs = set()
        combo_results = []
        for i, f1 in enumerate(good_singles[:8]):
            for f2 in good_singles[i + 1:8]:
                if f1["feature"] == f2["feature"]:
                    continue
                pair_key = tuple(sorted([f1["feature"], f2["feature"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                valid = df[df[f1["feature"]].notna() &
                           df[f2["feature"]].notna()].copy()
                n_binders = int(valid["binder"].sum())
                if n_binders < 10:
                    continue

                if f1["op"] == "<=":
                    m1 = valid[f1["feature"]] <= f1["threshold"]
                else:
                    m1 = valid[f1["feature"]] >= f1["threshold"]
                if f2["op"] == "<=":
                    m2 = valid[f2["feature"]] <= f2["threshold"]
                else:
                    m2 = valid[f2["feature"]] >= f2["threshold"]

                passing = valid[m1 & m2]
                n_pass = len(passing)
                tp = int(passing["binder"].sum())
                if n_pass == 0:
                    continue
                precision = tp / n_pass
                recall = tp / n_binders
                enrichment = precision / (n_binders / len(valid))

                combo_results.append({
                    "filters": (f"{f1['feature']} {f1['op']}{f1['threshold']}",
                                f"{f2['feature']} {f2['op']}{f2['threshold']}"),
                    "n_pass": n_pass, "tp": tp, "precision": precision,
                    "recall": recall, "enrichment": enrichment,
                })

        combo_results.sort(key=lambda x: x["enrichment"], reverse=True)
        print(f"\n  {'Filter 1':45s} {'Filter 2':45s} {'Pass':>5s} {'TP':>4s} "
              f"{'Prec':>7s} {'Recall':>7s} {'Enrich':>7s}")
        print("  " + "-" * 120)
        for c in combo_results[:15]:
            print(f"  {c['filters'][0]:45s} {c['filters'][1]:45s} "
                  f"{c['n_pass']:>5d} {c['tp']:>4d} "
                  f"{c['precision']:>6.1%} {c['recall']:>6.1%} "
                  f"{c['enrichment']:>6.1f}x")


def section_5_4(df):
    """Design filter landscape — scatter plot of key filters."""
    print("\n" + "=" * 70)
    print("5.4  DESIGN FILTER LANDSCAPE")
    print("=" * 70)

    water_col = "af3_binary_min_dist_to_ligand_O"
    iptm_col = "af3_ternary_ipTM"
    conv_col = "docking_convergence_ratio"

    needed = [water_col, iptm_col, "binder", "binder_name", "label_source"]
    if not all(c in df.columns for c in needed):
        print("  Missing required columns")
        return

    valid = df[df[water_col].notna() & df[iptm_col].notna()].copy()
    print(f"  Pairs with both water distance and ternary ipTM: {len(valid)}")

    # ── Panel layout: 2x2 ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ── Panel A: water dist vs ternary ipTM, colored by binder ──
    ax = axes[0, 0]
    for bname, color in [("Non-binder", "#3b82f6"), ("Binder", "#ef4444")]:
        mask = valid["binder_name"] == bname
        alpha = 0.15 if bname == "Non-binder" else 0.6
        zorder = 1 if bname == "Non-binder" else 2
        ax.scatter(valid.loc[mask, water_col], valid.loc[mask, iptm_col],
                   c=color, alpha=alpha, s=12, label=bname, zorder=zorder,
                   edgecolors="none")
    # Threshold lines
    ax.axvline(x=3.0, color="#22c55e", linestyle="--", linewidth=1.5,
               label="Water dist = 3 \u00c5", zorder=3)
    ax.axhline(y=0.9, color="#f59e0b", linestyle="--", linewidth=1.5,
               label="Ternary ipTM = 0.9", zorder=3)
    # High-confidence quadrant
    ax.fill_betweenx([0.9, 1.0], 0, 3.0, alpha=0.08, color="#22c55e",
                     zorder=0)
    # Stats in quadrant
    in_quad = valid[(valid[water_col] <= 3.0) & (valid[iptm_col] >= 0.9)]
    n_q = len(in_quad)
    tp_q = int(in_quad["binder"].sum())
    prec_q = tp_q / n_q if n_q > 0 else 0
    ax.text(1.5, 0.95, f"n={n_q}, {tp_q} binders\n{prec_q:.0%} precision",
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#22c55e", alpha=0.9), zorder=4)
    ax.set_xlabel("Binary water distance (\u00c5)")
    ax.set_ylabel("Ternary ipTM")
    ax.set_xlim(0, 15)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("A. Design filter landscape")
    ax.legend(fontsize=8, loc="lower right", markerscale=2)

    # ── Panel B: same scatter, colored by source ──
    ax = axes[0, 1]
    source_order = ["experimental", "win_ssm", "pnas_cutler", "LCA_screen",
                    "artificial_swap", "artificial_ala_scan"]
    for src in source_order:
        mask = valid["label_source"] == src
        if mask.sum() == 0:
            continue
        color = SOURCE_COLORS.get(src, "#94a3b8")
        ax.scatter(valid.loc[mask, water_col], valid.loc[mask, iptm_col],
                   c=color, alpha=0.4, s=12, label=src, edgecolors="none")
    ax.axvline(x=3.0, color="#22c55e", linestyle="--", linewidth=1.5)
    ax.axhline(y=0.9, color="#f59e0b", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Binary water distance (\u00c5)")
    ax.set_ylabel("Ternary ipTM")
    ax.set_xlim(0, 15)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("B. By data source")
    ax.legend(fontsize=7, loc="lower right", markerscale=2)

    # ── Panel C: water dist vs ternary ipTM, marker size = convergence ──
    ax = axes[1, 0]
    if conv_col in df.columns:
        plot_df = valid[valid[conv_col].notna()].copy()
        # Invert convergence: lower = better = bigger marker
        max_conv = plot_df[conv_col].quantile(0.95)
        sizes = 5 + 80 * (1 - plot_df[conv_col].clip(0, max_conv) / max_conv)
        for bname, color in [("Non-binder", "#3b82f6"), ("Binder", "#ef4444")]:
            mask = plot_df["binder_name"] == bname
            alpha = 0.15 if bname == "Non-binder" else 0.5
            zorder = 1 if bname == "Non-binder" else 2
            ax.scatter(plot_df.loc[mask, water_col],
                       plot_df.loc[mask, iptm_col],
                       c=color, alpha=alpha, s=sizes[mask],
                       label=bname, zorder=zorder, edgecolors="none")
        ax.axvline(x=3.0, color="#22c55e", linestyle="--", linewidth=1.5)
        ax.axhline(y=0.9, color="#f59e0b", linestyle="--", linewidth=1.5)
        # Triple-filter quadrant
        triple = plot_df[(plot_df[water_col] <= 3.0) &
                         (plot_df[iptm_col] >= 0.9) &
                         (plot_df[conv_col] <= 0.1)]
        n_t = len(triple)
        tp_t = int(triple["binder"].sum())
        prec_t = tp_t / n_t if n_t > 0 else 0
        ax.text(1.5, 0.95,
                f"+ conv \u22640.1\nn={n_t}, {tp_t} binders\n{prec_t:.0%} precision",
                ha="center", va="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#22c55e", alpha=0.9), zorder=4)
        ax.set_xlabel("Binary water distance (\u00c5)")
        ax.set_ylabel("Ternary ipTM")
        ax.set_xlim(0, 15)
        ax.set_ylim(0.5, 1.0)
        ax.set_title("C. Size = docking convergence (larger = better)")
        ax.legend(fontsize=8, loc="lower right", markerscale=1.5)
    else:
        ax.text(0.5, 0.5, "Docking convergence data unavailable",
                transform=ax.transAxes, ha="center")

    # ── Panel D: precision-recall at different filter combos ──
    ax = axes[1, 1]
    combos = []
    for wt in [2.5, 3.0, 3.5, 4.0, 5.0]:
        for it in [0.85, 0.9, 0.95]:
            sub = valid[(valid[water_col] <= wt) & (valid[iptm_col] >= it)]
            n_s = len(sub)
            tp_s = int(sub["binder"].sum())
            total_b = int(valid["binder"].sum())
            if n_s > 0 and total_b > 0:
                combos.append({
                    "label": f"wd\u2264{wt}, ipTM\u2265{it}",
                    "precision": tp_s / n_s,
                    "recall": tp_s / total_b,
                    "n": n_s,
                    "water_thresh": wt,
                    "iptm_thresh": it,
                })
    if combos:
        combo_df = pd.DataFrame(combos)
        colors_pr = {0.85: "#a78bfa", 0.9: "#f59e0b", 0.95: "#ef4444"}
        for it, grp in combo_df.groupby("iptm_thresh"):
            ax.plot(grp["recall"], grp["precision"], "o-",
                    color=colors_pr.get(it, "#94a3b8"),
                    label=f"ipTM \u2265 {it}", markersize=6)
            for _, row in grp.iterrows():
                ax.annotate(f"wd\u2264{row['water_thresh']:.0f}",
                            (row["recall"], row["precision"]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7)
    baseline = valid["binder"].mean()
    ax.axhline(y=baseline, color="gray", linestyle=":", linewidth=1,
               label=f"Baseline ({baseline:.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("D. Filter precision-recall tradeoff")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    _savefig(fig, "5_4_design_filter_landscape.png")

    # Print summary stats
    print(f"\n  Filter performance summary:")
    print(f"  {'Filters':<40s} {'Pass':>5s} {'TP':>4s} {'Prec':>7s} {'Recall':>7s}")
    print("  " + "-" * 65)
    total_b = int(valid["binder"].sum())
    for wt in [3.0, 4.0]:
        for it in [0.85, 0.9]:
            sub = valid[(valid[water_col] <= wt) & (valid[iptm_col] >= it)]
            n_s = len(sub)
            tp_s = int(sub["binder"].sum())
            if n_s > 0:
                print(f"  water \u2264{wt}\u00c5 + ipTM \u2265{it:<23} "
                      f"{n_s:>5d} {tp_s:>4d} {tp_s/n_s:>6.1%} "
                      f"{tp_s/total_b:>6.1%}")
    if conv_col in valid.columns:
        sub = valid[(valid[water_col] <= 3.0) & (valid[iptm_col] >= 0.9) &
                    (valid[conv_col] <= 0.1)]
        n_s = len(sub)
        tp_s = int(sub["binder"].sum())
        if n_s > 0:
            label = 'water \u22643\u00c5 + ipTM \u22650.9 + conv \u22640.1'
            print(f"  {label:<40s} "
                  f"{n_s:>5d} {tp_s:>4d} {tp_s/n_s:>6.1%} "
                  f"{tp_s/total_b:>6.1%}")


# ═════════════════════════════════════════════════════════════════
# PART 6: DATASET GAPS & ACTION PLAN
# ═════════════════════════════════════════════════════════════════

def section_6_1(df):
    """Where the model fails."""
    print("\n" + "=" * 70)
    print("6.1  WHERE THE MODEL FAILS")
    print("=" * 70)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("  scikit-learn not installed — skipping")
        return

    feats = _feature_cols(df, exclude_sparse=True)
    if not feats:
        # Fallback: use any non-constant features without sparse filtering
        feats = _feature_cols(df)
    if not feats:
        print("  No usable features found — skipping")
        return

    meta_cols = ["binder", "ligand_name", "label_source", "variant_name"]
    sub = df[feats + meta_cols].dropna(
        subset=feats + ["binder"]).copy()

    n_binders = int(sub["binder"].sum())
    if len(sub) < 50 or n_binders < 10:
        print(f"  Insufficient data ({len(sub)} rows, {n_binders} binders)")
        return

    X = sub[feats].values
    y = sub["binder"].values
    groups = sub["ligand_name"].values

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1))
    ])

    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y, groups):
            pipe.fit(X[train_idx], y[train_idx])
            y_prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    except Exception:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            pipe.fit(X[train_idx], y[train_idx])
            y_prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    sub["y_prob"] = y_prob

    # False positives: non-binders ranked highest
    fp = sub[(sub["binder"] == 0)].nlargest(20, "y_prob")
    print(f"\n  Top 20 false positives (non-binders ranked as binders):")
    print(f"  {'Ligand':30s} {'Source':20s} {'P(binder)':>10s}")
    for _, row in fp.iterrows():
        print(f"  {str(row['ligand_name'])[:30]:30s} "
              f"{str(row['label_source']):20s} {row['y_prob']:.3f}")

    # False negatives: binders ranked lowest
    fn = sub[(sub["binder"] == 1)].nsmallest(20, "y_prob")
    print(f"\n  Top 20 false negatives (binders ranked as non-binders):")
    print(f"  {'Ligand':30s} {'Source':20s} {'P(binder)':>10s}")
    for _, row in fn.iterrows():
        print(f"  {str(row['ligand_name'])[:30]:30s} "
              f"{str(row['label_source']):20s} {row['y_prob']:.3f}")

    # Per-source error analysis
    print(f"\n  Per-source error analysis:")
    for src, grp in sub.groupby("label_source"):
        n = len(grp)
        n_b = int(grp["binder"].sum())
        # Mean predicted prob for binders vs non-binders
        if n_b > 0:
            mean_b = grp.loc[grp["binder"] == 1, "y_prob"].mean()
        else:
            mean_b = float("nan")
        mean_nb = grp.loc[grp["binder"] == 0, "y_prob"].mean()
        separation = mean_b - mean_nb if n_b > 0 else float("nan")
        print(f"    {src:25s}: n={n:4d}, binders={n_b:3d}, "
              f"P(b|binder)={mean_b:.3f}, P(b|non-b)={mean_nb:.3f}, "
              f"separation={separation:+.3f}")


def section_6_2(df):
    """Data improvement priorities."""
    print("\n" + "=" * 70)
    print("6.2  DATA IMPROVEMENT PRIORITIES")
    print("=" * 70)

    print("""
  RANKED PRIORITIES:

  1. BACKFILL TIER 3 AFFINITY DATA
     - Map min_conc from pnas_data.csv into affinity_uM column
     - Enables potency-aware modeling (1 uM vs 100 uM distinction)
     - Purely computational — no new experiments needed

  2. EXPERIMENTAL VALIDATION (20-50 pairs)
     - Test top model predictions for pairs NOT in training data
     - Gives real precision estimate + new training data at decision boundary
     - Highest-value experiment for model improvement

  3. BILE ACID PILOT PANELS (10 variants each)
     - CDCA, CA, DCA, UDCA: 10 variants each = 40 new data points
     - Run pipeline first, test top predictions + random controls
     - Bootstraps bile-acid-specific models

  4. KYNURENINE PILOT SCREEN (20-30 variants)
     - Novel scaffold — model can't predict without data
     - Run pipeline for AF3 features, test a small panel
     - Even 20 data points enables a kynurenine-specific model

  5. RE-EVALUATE MANDIPROPAMID & ABA DATA
     - Mandi: double water bond geometry (different from standard PYR1)
     - ABA: poor quality reported — worth auditing
     - These are SSM datasets = high-quality if labels are reliable

  6. REBALANCE NEGATIVES
     - Tier 4 (1966 LCA negatives) = 47% of dataset
     - Consider downsampling to 500 representative LCA negatives
     - Or use stratified sampling to balance source representation
""")

    # Quantify current imbalances
    print("  Current dataset composition:")
    for src in ["experimental", "win_ssm", "pnas_cutler", "LCA_screen",
                "artificial_swap", "artificial_ala_scan"]:
        sub = df[df["label_source"] == src]
        pct = 100 * len(sub) / len(df)
        print(f"    {src:25s}: {len(sub):4d} ({pct:5.1f}%)")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

SECTIONS = {
    "1.1": ("Tier-by-tier data inventory", section_1_1),
    "1.2": ("Label quality audit", section_1_2),
    "2.1": ("AF3 confidence by binding strength", section_2_1),
    "2.2": ("Binary vs ternary predictions", section_2_2),
    "2.3": ("Water-mediated H-bond geometry", section_2_3),
    "2.4": ("AF3 ligand RMSD", section_2_4),
    "3.1": ("Rosetta features by binding strength", section_3_1),
    "3.2": ("Rosetta vs AF3 agreement", section_3_2),
    "3.3": ("Rosetta water placement potential", section_3_3),
    "4.1": ("LCA family deep dive", section_4_1),
    "4.2": ("WIN family deep dive", section_4_2),
    "4.3": ("PNAS diverse ligands", section_4_3),
    "5.1": ("Global model analysis", section_5_1),
    "5.3": ("Optimal filter combinations", section_5_3),
    "5.4": ("Design filter landscape", section_5_4),
    "6.1": ("Where the model fails", section_6_1),
    "6.2": ("Data improvement priorities", section_6_2),
}


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive PYR1 ML dataset analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sections: " + ", ".join(
            f"{k} ({v[0]})" for k, v in SECTIONS.items()))
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to all_features.csv")
    parser.add_argument("--section", type=str, default=None,
                        help="Run specific section (e.g., '2.3' or '2' for all Part 2)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PYR1 ML DATASET — COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    df = load_data(args.csv)
    print(f"\nDataset: {len(df)} pairs, {df['ligand_name'].nunique()} ligands, "
          f"{df['binder'].sum()} binders")

    # Determine which sections to run
    if args.section:
        if "." in args.section:
            # Specific section: "2.3"
            to_run = {args.section: SECTIONS[args.section]}
        else:
            # Whole part: "2"
            to_run = {k: v for k, v in SECTIONS.items()
                      if k.startswith(args.section + ".")}
    else:
        to_run = SECTIONS

    if not to_run:
        print(f"No sections match '{args.section}'")
        print(f"Available: {', '.join(SECTIONS.keys())}")
        return

    for key, (name, func) in to_run.items():
        try:
            func(df)
        except Exception as e:
            print(f"\n  ERROR in section {key}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
