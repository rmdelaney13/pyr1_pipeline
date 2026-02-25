#!/usr/bin/env python
"""
Exploratory analysis of PYR1 ML pipeline preliminary features.

Generates figures and prints summary statistics for the feature table
produced by aggregate_ml_features.py.

Usage:
    python explore_preliminary_features.py
    python explore_preliminary_features.py --csv path/to/features.csv

Output: figures saved to ml_modelling/analysis/figures/
"""

import argparse
import sys
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

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CSV = PROJECT_ROOT / "ml_modelling" / "data" / "preliminary_features.csv"
FIG_DIR = SCRIPT_DIR / "figures"

# ── Style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
LABEL_COLORS = {0.0: "#3b82f6", 0.25: "#a78bfa", 0.75: "#f59e0b", 1.0: "#ef4444"}
LABEL_NAMES = {0.0: "Negative", 0.25: "Weak", 0.75: "Moderate", 1.0: "Strong"}
TIER_ORDER = ["Negative", "Weak", "Moderate", "Strong"]
BINARY_COLORS = {"Non-binder": "#3b82f6", "Binder": "#ef4444"}


# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path.name}")

    # Map label tiers to readable names
    df["label_name"] = df["label"].map(LABEL_NAMES)

    # Binary label for classification: strong + moderate = binder
    df["binder"] = (df["label"] >= 0.75).astype(int)
    df["binder_name"] = df["binder"].map({0: "Non-binder", 1: "Binder"})

    # Drop columns that are 0% complete
    always_nan = [c for c in df.columns if df[c].isna().all()]
    if always_nan:
        print(f"Dropping {len(always_nan)} all-NaN columns: {always_nan}")
        df.drop(columns=always_nan, inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════
# 2. COMPLETENESS REPORT
# ═══════════════════════════════════════════════════════════════════

def plot_completeness(df: pd.DataFrame):
    """Bar chart: % complete for each numeric feature, colored by pipeline stage."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ("label", "label_confidence", "binder")]

    stage_colors = {
        "conformer": "#22c55e",
        "docking": "#3b82f6",
        "rosetta": "#f59e0b",
        "af3_binary": "#ef4444",
        "af3_ternary": "#a855f7",
    }

    completeness = []
    for col in numeric_cols:
        pct = (1 - df[col].isna().mean()) * 100
        stage = "other"
        for s in stage_colors:
            if col.startswith(s):
                stage = s
                break
        completeness.append({"feature": col, "pct_complete": pct, "stage": stage})

    comp_df = pd.DataFrame(completeness).sort_values("pct_complete", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(comp_df) * 0.28)))
    colors = [stage_colors.get(r["stage"], "#94a3b8") for _, r in comp_df.iterrows()]
    ax.barh(comp_df["feature"], comp_df["pct_complete"], color=colors, edgecolor="none")
    ax.set_xlabel("% Complete")
    ax.set_title("Feature Completeness by Pipeline Stage")
    ax.set_xlim(0, 105)
    ax.axvline(50, color="gray", ls="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s.replace("_", " ").title())
                       for s, c in stage_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_completeness.png", dpi=150)
    plt.close(fig)
    print("  -> 01_completeness.png")


def print_completeness_by_tier(df: pd.DataFrame):
    """Print table: completeness broken down by label tier."""
    stages = ["conformer_status", "docking_status", "rosetta_status",
              "af3_binary_status", "af3_ternary_status"]
    print("\n  Stage completeness by label tier:")
    print(f"  {'Stage':<22} {'Negative':>10} {'Weak':>10} {'Moderate':>10} {'Strong':>10} {'Overall':>10}")
    print("  " + "-" * 72)
    for stage_col in stages:
        if stage_col not in df.columns:
            continue
        row = [stage_col.replace("_status", "")]
        for tier in [0.0, 0.25, 0.75, 1.0]:
            subset = df[df["label"] == tier]
            pct = (subset[stage_col] == "COMPLETE").mean() * 100
            row.append(f"{pct:.0f}%")
        overall = (df[stage_col] == "COMPLETE").mean() * 100
        row.append(f"{overall:.0f}%")
        print(f"  {row[0]:<22} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10}")


# ═══════════════════════════════════════════════════════════════════
# 3. LABEL & DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════

def plot_label_overview(df: pd.DataFrame):
    """Label distribution and ligand composition."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 3a. Label tier distribution
    ax = axes[0]
    tier_counts = df["label_name"].value_counts().reindex(TIER_ORDER)
    bars = ax.bar(tier_counts.index, tier_counts.values,
                  color=[LABEL_COLORS[k] for k in [0.0, 0.25, 0.75, 1.0]])
    ax.set_title("Label Distribution")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, tier_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(val), ha="center", fontsize=10)

    # 3b. Top 10 ligands
    ax = axes[1]
    lig_counts = df["ligand_name"].value_counts().head(10)
    ax.barh(lig_counts.index[::-1], lig_counts.values[::-1], color="#64748b")
    ax.set_title("Top 10 Ligands (by pair count)")
    ax.set_xlabel("Count")

    # 3c. Binary label balance
    ax = axes[2]
    binder_counts = df["binder_name"].value_counts()
    ax.pie(binder_counts.values, labels=binder_counts.index, autopct="%1.1f%%",
           colors=[BINARY_COLORS[k] for k in binder_counts.index],
           startangle=90, textprops={"fontsize": 12})
    ax.set_title("Binary Classification Balance")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_label_overview.png", dpi=150)
    plt.close(fig)
    print("  -> 02_label_overview.png")


# ═══════════════════════════════════════════════════════════════════
# 4. FEATURE DISTRIBUTIONS BY LABEL
# ═══════════════════════════════════════════════════════════════════

# The key features to focus on, grouped by stage
KEY_FEATURES = {
    "Docking": [
        "docking_best_score",
        "docking_convergence_ratio",
        "docking_num_clusters",
        "docking_score_range",
        "docking_cluster_1_rmsd",
        "docking_pass_rate",
    ],
    "Rosetta": [
        "rosetta_dG_sep_best",
        "rosetta_dG_sep_mean",
        "rosetta_buried_unsats_best",
        "rosetta_dsasa_int_mean",
        "rosetta_hbonds_to_ligand_mean",
        "rosetta_shape_comp_best",
    ],
    "AF3 Binary": [
        "af3_binary_ipTM",
        "af3_binary_pLDDT_ligand",
        "af3_binary_interface_PAE",
        "af3_binary_ligand_RMSD_min",
        "af3_binary_min_dist_to_ligand_O",
        "af3_binary_hbond_water_angle",
    ],
    "AF3 Ternary": [
        "af3_ternary_ipTM",
        "af3_ternary_pLDDT_ligand",
        "af3_ternary_interface_PAE",
        "af3_ternary_ligand_RMSD_min",
        "af3_ternary_min_dist_to_ligand_O",
        "af3_ternary_hbond_water_angle",
    ],
}


def plot_feature_distributions(df: pd.DataFrame):
    """Violin plots for key features, split by graded label."""
    for stage_name, features in KEY_FEATURES.items():
        # Filter to features that exist in the dataframe
        features = [f for f in features if f in df.columns]
        if not features:
            continue

        n = len(features)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            plot_df = df[["label_name", feat]].dropna(subset=[feat])
            if len(plot_df) < 10:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(feat)
                continue

            order = [t for t in TIER_ORDER if t in plot_df["label_name"].values]
            palette = {LABEL_NAMES[k]: v for k, v in LABEL_COLORS.items()}
            sns.violinplot(data=plot_df, x="label_name", y=feat, order=order,
                           palette=palette, inner="box", cut=0, ax=ax, density_norm="width")
            ax.set_xlabel("")
            ax.set_title(feat.replace("_", " "), fontsize=11)

            # Add sample count
            for j, tier in enumerate(order):
                n_pts = (plot_df["label_name"] == tier).sum()
                ax.text(j, ax.get_ylim()[0], f"n={n_pts}", ha="center",
                        va="top", fontsize=8, color="gray")

        # Hide unused axes
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{stage_name} Features by Label Tier", fontsize=14, y=1.02)
        plt.tight_layout()
        fname = f"03_{stage_name.lower().replace(' ', '_')}_distributions.png"
        fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {fname}")


def plot_binary_boxplots(df: pd.DataFrame):
    """Side-by-side box plots for binder vs non-binder on the most discriminative features."""
    top_features = [
        "docking_best_score", "docking_convergence_ratio",
        "rosetta_dG_sep_best", "rosetta_dG_sep_mean",
        "rosetta_buried_unsats_best", "rosetta_hbonds_to_ligand_mean",
        "af3_binary_ipTM", "af3_binary_ligand_RMSD_min",
        "af3_ternary_ipTM", "af3_ternary_ligand_RMSD_min",
        "af3_binary_min_dist_to_ligand_O", "af3_binary_hbond_water_angle",
        "af3_ternary_min_dist_to_ligand_O", "af3_ternary_hbond_water_angle",
        "docking_pass_rate", "rosetta_dsasa_int_mean",
    ]
    top_features = [f for f in top_features if f in df.columns]

    n = len(top_features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        plot_df = df[["binder_name", feat]].dropna(subset=[feat])
        sns.boxplot(data=plot_df, x="binder_name", y=feat,
                    palette=BINARY_COLORS, ax=ax, fliersize=2)
        ax.set_xlabel("")
        ax.set_title(feat.replace("_", " "), fontsize=10)

        # Mann-Whitney U test
        grp0 = plot_df[plot_df["binder_name"] == "Non-binder"][feat].dropna()
        grp1 = plot_df[plot_df["binder_name"] == "Binder"][feat].dropna()
        if len(grp0) > 5 and len(grp1) > 5:
            u_stat, p_val = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.text(0.5, 0.97, f"p={p_val:.2e} {sig}", ha="center", va="top",
                    transform=ax.transAxes, fontsize=8, color="darkred")

    for j in range(len(top_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Binder vs Non-binder Feature Comparison (Mann-Whitney U)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_binary_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 04_binary_boxplots.png")


# ═══════════════════════════════════════════════════════════════════
# 5. CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_correlation_matrix(df: pd.DataFrame):
    """Correlation heatmap of numeric features (complete rows only)."""
    feature_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ("label", "label_confidence", "binder",
                                 "docking_clash_flag", "docking_total_attempts")]

    # Use pairwise-complete correlations
    corr = df[feature_cols].corr(min_periods=30)

    # Drop features that have no valid correlations
    valid = corr.notna().sum() > 1
    corr = corr.loc[valid, valid]

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})
    ax.set_title("Feature Correlation Matrix (pairwise complete)", fontsize=13)
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("  -> 05_correlation_matrix.png")


def plot_top_correlations_with_label(df: pd.DataFrame):
    """Bar chart: features most correlated with the binding label."""
    feature_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ("label", "label_confidence", "binder",
                                 "docking_clash_flag", "docking_total_attempts")]

    correlations = {}
    for col in feature_cols:
        valid = df[["label", col]].dropna()
        if len(valid) > 30:
            r, p = stats.spearmanr(valid["label"], valid[col])
            correlations[col] = {"spearman_r": r, "p_value": p, "n": len(valid)}

    corr_df = pd.DataFrame(correlations).T.sort_values("spearman_r", key=abs, ascending=False)
    corr_df = corr_df.head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#ef4444" if r > 0 else "#3b82f6" for r in corr_df["spearman_r"]]
    ax.barh(corr_df.index[::-1], corr_df["spearman_r"][::-1], color=colors[::-1])
    ax.set_xlabel("Spearman correlation with label")
    ax.set_title("Top 20 Features Correlated with Binding Label")
    ax.axvline(0, color="black", lw=0.8)

    # Annotate with significance
    for i, (feat, row) in enumerate(corr_df[::-1].iterrows()):
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        offset = 0.01 if row["spearman_r"] < 0 else -0.01
        ha = "left" if row["spearman_r"] < 0 else "right"
        ax.text(row["spearman_r"] + offset, i, f'{sig} (n={int(row["n"])})',
                va="center", ha=ha, fontsize=7, color="gray")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_label_correlations.png", dpi=150)
    plt.close(fig)
    print("  -> 06_label_correlations.png")

    return corr_df


# ═══════════════════════════════════════════════════════════════════
# 6. LIGAND-LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_ligand_effects(df: pd.DataFrame):
    """Check if certain features are ligand-specific vs variant-specific."""
    # For each feature, compute variance explained by ligand vs residual
    features_to_check = [
        "docking_best_score", "docking_convergence_ratio",
        "rosetta_dG_sep_best", "af3_binary_ipTM",
    ]
    features_to_check = [f for f in features_to_check if f in df.columns]

    if not features_to_check:
        return

    # Focus on ligands with enough data points in both classes
    ligand_counts = df.groupby("ligand_name").agg(
        n=("label", "count"),
        n_binders=("binder", "sum")
    )
    informative_ligands = ligand_counts[
        (ligand_counts["n"] >= 10) & (ligand_counts["n_binders"] >= 2)
    ].index.tolist()

    if len(informative_ligands) < 2:
        print("  (Skipping ligand effect plot — not enough ligands with both classes)")
        return

    sub = df[df["ligand_name"].isin(informative_ligands)].copy()

    n = len(features_to_check)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features_to_check):
        plot_df = sub[["ligand_name", "binder_name", feat]].dropna(subset=[feat])
        if len(plot_df) < 20:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        sns.boxplot(data=plot_df, x="ligand_name", y=feat, hue="binder_name",
                    palette=BINARY_COLORS, ax=ax, fliersize=1)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(feat.replace("_", " "), fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Feature Distributions by Ligand (binder vs non-binder)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_ligand_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 07_ligand_effects.png")


# ═══════════════════════════════════════════════════════════════════
# 7. PAIRWISE SCATTER OF TOP FEATURES
# ═══════════════════════════════════════════════════════════════════

def plot_pairwise_scatter(df: pd.DataFrame):
    """Pairwise scatter of the most informative features."""
    scatter_features = [
        "docking_best_score", "docking_convergence_ratio",
        "rosetta_dG_sep_best", "rosetta_hbonds_to_ligand_mean",
        "af3_binary_ipTM", "af3_binary_ligand_RMSD_min",
        "af3_binary_min_dist_to_ligand_O", "af3_binary_hbond_water_angle",
    ]
    scatter_features = [f for f in scatter_features if f in df.columns]

    if len(scatter_features) < 2:
        return

    sub = df[scatter_features + ["binder_name"]].dropna()
    if len(sub) < 20:
        print("  (Skipping pairwise scatter — <20 complete rows)")
        return

    print(f"  Pairwise scatter using {len(sub)} rows with complete data")

    g = sns.pairplot(sub, hue="binder_name", palette=BINARY_COLORS,
                     diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15},
                     corner=True, height=2.2)
    g.figure.suptitle(f"Pairwise Feature Scatter (n={len(sub)} complete rows)", y=1.02)
    g.savefig(FIG_DIR / "08_pairwise_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    print("  -> 08_pairwise_scatter.png")


# ═══════════════════════════════════════════════════════════════════
# 8. BASELINE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

def _run_single_model(df_sub, feature_cols, model_name, fig_suffix,
                      sample_weights=None, group_col=None):
    """Run a single RF model on complete-case data. No imputation.

    Args:
        df_sub: DataFrame with features + 'binder' column
        feature_cols: List of feature column names
        model_name: Display name for the model
        fig_suffix: Suffix for output filenames
        sample_weights: Optional array of per-sample weights (aligned to df_sub index).
            If provided, used for training and weighted evaluation metrics.
        group_col: Optional column name for group-aware CV (e.g., 'ligand_name').
            Prevents data leakage when the same ligand appears in train and test.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        classification_report, roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve, confusion_matrix
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Complete cases only — no imputation
    required_cols = feature_cols + ["binder"]
    extra_cols = []
    if sample_weights is not None:
        df_sub = df_sub.copy()
        df_sub["_sample_weight"] = sample_weights
        extra_cols.append("_sample_weight")
    if group_col and group_col in df_sub.columns:
        extra_cols.append(group_col)

    sub = df_sub[required_cols + extra_cols].dropna(subset=feature_cols + ["binder"]).copy()

    n_binders = int(sub["binder"].sum())
    n_total = len(sub)
    print(f"\n  [{model_name}] {n_total} complete rows, {n_binders} binders, "
          f"{len(feature_cols)} features")

    if n_total < 50 or n_binders < 10:
        print(f"  [{model_name}] Insufficient data — skipping")
        return None

    X = sub[feature_cols].values
    y = sub["binder"].values
    w = sub["_sample_weight"].values if sample_weights is not None else None

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])

    # Choose CV strategy: ligand-grouped if possible, else stratified
    groups = None
    if group_col and group_col in sub.columns:
        groups = sub[group_col].values
        n_unique = len(set(groups))
        if n_unique >= 5:
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                print(f"  [{model_name}] GroupKFold by {group_col} ({n_unique} groups)"
                      " — no ligand leakage between folds")
            except ImportError:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                groups = None
                print(f"  [{model_name}] StratifiedGroupKFold unavailable, using StratifiedKFold")
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            groups = None
            print(f"  [{model_name}] Too few groups ({n_unique}) for GroupKFold — using StratifiedKFold")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Manual CV loop to support sample_weight in fit + group-aware splits
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y, groups):
        fit_params = {}
        if w is not None:
            fit_params["clf__sample_weight"] = w[train_idx]
        pipe.fit(X[train_idx], y[train_idx], **fit_params)
        y_prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    # Compute metrics — weighted if sample weights provided
    auc_roc = roc_auc_score(y, y_prob, sample_weight=w)
    auc_pr = average_precision_score(y, y_prob, sample_weight=w)
    weighted_tag = " (confidence-weighted)" if w is not None else ""
    print(f"  [{model_name}] ROC-AUC: {auc_roc:.3f}  PR-AUC: {auc_pr:.3f} "
          f"(random baseline: {y.mean():.3f}){weighted_tag}")
    print(classification_report(y, y_pred, target_names=["Non-binder", "Binder"],
                                sample_weight=w))

    # Feature importance (fit on full data for ranking)
    fit_params = {"clf__sample_weight": w} if w is not None else {}
    pipe.fit(X, y, **fit_params)
    importances = pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    return {
        "name": model_name, "auc_roc": auc_roc, "auc_pr": auc_pr,
        "y": y, "y_prob": y_prob, "y_pred": y_pred,
        "n_total": n_total, "n_binders": n_binders,
        "imp_df": imp_df, "baseline": y.mean(),
        "sample_weights": w,
    }


def _plot_classifier_results(results, fig_prefix, title_suffix=""):
    """Plot ROC/PR curves, feature importance, and confusion matrices for a set of model results."""
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

    if not results:
        print("  No models could be trained")
        return

    model_colors = ["#22c55e", "#f59e0b", "#ef4444"]

    # ── Combined ROC + PR plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for r, color in zip(results, model_colors):
        w = r.get("sample_weights")
        fpr, tpr, _ = roc_curve(r["y"], r["y_prob"], sample_weight=w)
        ax1.plot(fpr, tpr, color=color, lw=2,
                 label=f'{r["name"]} (AUC={r["auc_roc"]:.3f}, n={r["n_total"]})')

        prec, rec, _ = precision_recall_curve(r["y"], r["y_prob"], sample_weight=w)
        ax2.plot(rec, prec, color=color, lw=2,
                 label=f'{r["name"]} (AUC={r["auc_pr"]:.3f})')

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curves{title_suffix}")
    ax1.legend(fontsize=8)

    baseline = results[0]["baseline"]
    ax2.axhline(baseline, color="gray", ls="--", alpha=0.5, label=f"Random = {baseline:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curves{title_suffix}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{fig_prefix}_classifier_curves.png", dpi=150)
    plt.close(fig)
    print(f"  -> {fig_prefix}_classifier_curves.png")

    # ── Feature importance from the richest model ──
    best = results[-1]
    imp_df = best["imp_df"].head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#64748b")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f'Top 20 Feature Importances — {best["name"]} (n={best["n_total"]}){title_suffix}')
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{fig_prefix}_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  -> {fig_prefix}_feature_importance.png")

    # ── Confusion matrices side by side ──
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y"], r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Non-binder", "Binder"],
                    yticklabels=["Non-binder", "Binder"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f'{r["name"]}\nn={r["n_total"]}, ROC={r["auc_roc"]:.3f}', fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{fig_prefix}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  -> {fig_prefix}_confusion_matrix.png")


def run_baseline_classifier(df: pd.DataFrame):
    """
    Train separate RF classifiers on complete-case feature tiers.
    No imputation — avoids missingness-as-signal leakage.

    Runs two evaluation modes:
      1) Unweighted — all samples treated equally
      2) Confidence-weighted — tier confidence weights training and metrics
         (tier 1: 1.0, tier 2: 0.95, tier 3: 0.7, tier 4: 0.9, tier 5: 0.5)

    Models per mode:
      A) Docking-only features (most rows available)
      B) Docking + Rosetta (fewer rows, richer features)
      C) Docking + Rosetta + AF3 (fewest rows, all features)
    """
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
    except ImportError:
        print("\n  scikit-learn not installed — skipping classifier. "
              "Install with: pip install scikit-learn")
        return

    # Define feature tiers — each is complete-case within its stage
    docking_feats = [c for c in df.columns if c.startswith("docking_")
                     and not c.endswith("_status")
                     and c not in ("docking_clash_flag", "docking_total_attempts")]
    rosetta_feats = [c for c in df.columns if c.startswith("rosetta_")
                     and not c.endswith("_status")
                     and c != "rosetta_n_structures_relaxed"]
    af3_feats = [c for c in df.columns if c.startswith("af3_")
                 and not c.endswith("_status")]
    conformer_feats = [c for c in df.columns if c.startswith("conformer_")
                       and not c.endswith("_status")]

    # Filter to features that actually have data
    docking_feats = [f for f in docking_feats if df[f].notna().any()]
    rosetta_feats = [f for f in rosetta_feats if df[f].notna().any()]
    af3_feats = [f for f in af3_feats if df[f].notna().any()]
    conformer_feats = [f for f in conformer_feats if df[f].notna().any()]

    models_config = [
        ("A: Docking only", docking_feats, "a"),
        ("B: Docking + Rosetta", docking_feats + rosetta_feats, "b"),
        ("C: All stages", conformer_feats + docking_feats + rosetta_feats + af3_feats, "c"),
    ]

    # ── Mode 1: Unweighted ──
    print("\n  --- Unweighted evaluation (GroupKFold by ligand) ---")
    results_unweighted = []
    for name, feats, suffix in models_config:
        if not feats:
            print(f"\n  [{name}] No features available — skipping")
            continue
        r = _run_single_model(df, feats, name, suffix, group_col="ligand_name")
        if r:
            results_unweighted.append(r)

    _plot_classifier_results(results_unweighted, "09", " — Unweighted (ligand-grouped CV)")

    # ── Mode 2: Confidence-weighted ──
    has_confidence = "label_confidence" in df.columns and df["label_confidence"].notna().any()
    results_weighted = []
    if has_confidence:
        print("\n  --- Confidence-weighted evaluation (GroupKFold by ligand) ---")
        weights = pd.to_numeric(df["label_confidence"], errors="coerce").fillna(0.5)

        # Print weight distribution by source
        if "label_source" in df.columns:
            print("  Confidence weights by source:")
            for src, grp in df.groupby("label_source"):
                w = pd.to_numeric(grp["label_confidence"], errors="coerce").fillna(0.5)
                print(f"    {src}: weight={w.mean():.2f} (n={len(grp)})")

        for name, feats, suffix in models_config:
            if not feats:
                continue
            r = _run_single_model(df, feats, f"{name} (weighted)", suffix,
                                  sample_weights=weights.values,
                                  group_col="ligand_name")
            if r:
                results_weighted.append(r)

        _plot_classifier_results(results_weighted, "13",
                                 " — Confidence-Weighted")
    else:
        print("\n  (No label_confidence column — skipping weighted evaluation)")

    # ── Comparison summary ──
    if results_unweighted and results_weighted:
        print("\n  --- Unweighted vs Weighted comparison (All stages model) ---")
        for label, res_list in [("Unweighted", results_unweighted),
                                ("Weighted", results_weighted)]:
            best = res_list[-1]  # "C: All stages" model
            print(f"    {label}: ROC-AUC={best['auc_roc']:.3f}  "
                  f"PR-AUC={best['auc_pr']:.3f}  n={best['n_total']}")

    return results_unweighted


# ═══════════════════════════════════════════════════════════════════
# 9. PER-SOURCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

SOURCE_COLORS = {
    "experimental": "#22c55e",
    "win_ssm": "#3b82f6",
    "pnas_cutler": "#f59e0b",
    "LCA_screen": "#a855f7",
    "artificial_swap": "#ef4444",
    "artificial_ala_scan": "#f97316",
}

SOURCE_ORDER = ["experimental", "win_ssm", "pnas_cutler", "LCA_screen",
                "artificial_swap", "artificial_ala_scan"]


def plot_per_source_overview(df: pd.DataFrame):
    """Dataset overview broken down by label_source: counts, label balance, confidence."""
    if "label_source" not in df.columns:
        print("  (No label_source column — skipping per-source overview)")
        return

    sources = [s for s in SOURCE_ORDER if s in df["label_source"].values]
    if not sources:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 9a. Sample count per source, stacked by label tier
    ax = axes[0]
    tier_data = {}
    for tier in TIER_ORDER:
        tier_data[tier] = []
        for src in sources:
            n = ((df["label_source"] == src) & (df["label_name"] == tier)).sum()
            tier_data[tier].append(n)

    bottoms = np.zeros(len(sources))
    for tier in TIER_ORDER:
        vals = tier_data[tier]
        label_key = [k for k, v in LABEL_NAMES.items() if v == tier][0]
        ax.bar(range(len(sources)), vals, bottom=bottoms,
               color=LABEL_COLORS[label_key], label=tier)
        bottoms += np.array(vals)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.replace("_", "\n") for s in sources], fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Samples per Source (by label tier)")
    ax.legend(fontsize=8)

    # 9b. Binder fraction per source
    ax = axes[1]
    binder_fracs = []
    for src in sources:
        sub = df[df["label_source"] == src]
        binder_fracs.append(sub["binder"].mean())
    colors = [SOURCE_COLORS.get(s, "#94a3b8") for s in sources]
    bars = ax.bar(range(len(sources)), binder_fracs, color=colors)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.replace("_", "\n") for s in sources], fontsize=8)
    ax.set_ylabel("Binder Fraction")
    ax.set_title("Binder Rate by Source")
    ax.axhline(df["binder"].mean(), color="gray", ls="--", alpha=0.5,
               label=f"Overall: {df['binder'].mean():.2f}")
    ax.legend(fontsize=8)
    for bar, frac in zip(bars, binder_fracs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{frac:.2f}", ha="center", fontsize=9)

    # 9c. Confidence distribution per source
    ax = axes[2]
    if "label_confidence" in df.columns:
        conf_data = []
        for src in sources:
            vals = pd.to_numeric(
                df.loc[df["label_source"] == src, "label_confidence"],
                errors="coerce"
            ).dropna()
            conf_data.append(vals.values)
        parts = ax.violinplot(conf_data, positions=range(len(sources)),
                              showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(SOURCE_COLORS.get(sources[i], "#94a3b8"))
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([s.replace("_", "\n") for s in sources], fontsize=8)
        ax.set_ylabel("Label Confidence")
        ax.set_title("Confidence Distribution by Source")
    else:
        ax.text(0.5, 0.5, "No confidence data", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "14_per_source_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 14_per_source_overview.png")


def plot_per_source_feature_dists(df: pd.DataFrame):
    """Key feature distributions per source, colored by binder/non-binder."""
    if "label_source" not in df.columns:
        return

    features = [
        "docking_best_score", "docking_convergence_ratio",
        "rosetta_dG_sep_best", "rosetta_hbonds_to_ligand_mean",
        "af3_binary_ipTM", "af3_binary_ligand_RMSD_min",
        "af3_binary_min_dist_to_ligand_O", "af3_binary_hbond_water_angle",
    ]
    features = [f for f in features if f in df.columns]
    if not features:
        return

    sources = [s for s in SOURCE_ORDER if s in df["label_source"].values]
    # Merge small sources for readability
    df = df.copy()
    df["source_short"] = df["label_source"].replace({
        "artificial_swap": "artificial",
        "artificial_ala_scan": "artificial",
    })
    source_short = [s for s in ["experimental", "win_ssm", "pnas_cutler",
                                "LCA_screen", "artificial"]
                    if s in df["source_short"].values]

    nrows = (len(features) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 4.5 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        plot_df = df[["source_short", "binder_name", feat]].dropna(subset=[feat])
        if len(plot_df) < 20:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(feat)
            continue

        sns.boxplot(data=plot_df, x="source_short", y=feat, hue="binder_name",
                    palette=BINARY_COLORS, ax=ax, fliersize=1,
                    order=source_short)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(feat.replace("_", " "), fontsize=10)
        if i > 0:
            ax.get_legend().remove()
        else:
            ax.legend(fontsize=8, loc="upper right")

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions per Data Source (binder vs non-binder)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "15_per_source_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 15_per_source_features.png")


# ═══════════════════════════════════════════════════════════════════
# 10. CROSS-SOURCE GENERALIZATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_cross_source_generalization(df: pd.DataFrame):
    """Leave-one-source-out: train on N-1 sources, test on held-out source.

    This tests whether the model generalizes across experimental assays,
    not just across random splits within the same assay.
    """
    if "label_source" not in df.columns:
        print("  (No label_source column — skipping cross-source evaluation)")
        return

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("  (scikit-learn not installed — skipping)")
        return

    # Use all available features
    feature_cols = [c for c in df.columns
                    if (c.startswith("docking_") or c.startswith("rosetta_") or
                        c.startswith("af3_") or c.startswith("conformer_"))
                    and not c.endswith("_status")
                    and c not in ("docking_clash_flag", "docking_total_attempts",
                                  "rosetta_n_structures_relaxed")]
    feature_cols = [f for f in feature_cols if df[f].notna().any()]

    sub = df[feature_cols + ["binder", "label_source"]].dropna(
        subset=feature_cols + ["binder"]).copy()

    if len(sub) < 100:
        print("  (Insufficient complete-case data for cross-source evaluation)")
        return

    sources = sub["label_source"].unique()
    # Only test on sources with both classes
    testable = []
    for src in sources:
        src_sub = sub[sub["label_source"] == src]
        if src_sub["binder"].sum() >= 3 and (1 - src_sub["binder"]).sum() >= 3:
            testable.append(src)

    if len(testable) < 2:
        print("  (Too few testable sources — skipping cross-source evaluation)")
        return

    print(f"\n  Leave-one-source-out evaluation ({len(testable)} testable sources, "
          f"{len(feature_cols)} features, {len(sub)} complete rows)")

    results = []
    for test_src in testable:
        train = sub[sub["label_source"] != test_src]
        test = sub[sub["label_source"] == test_src]

        X_train, y_train = train[feature_cols].values, train["binder"].values
        X_test, y_test = test[feature_cols].values, test["binder"].values

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y_test, y_prob)
        except ValueError:
            ap = float("nan")

        n_test = len(y_test)
        n_binders = int(y_test.sum())
        results.append({
            "source": test_src, "n_test": n_test, "n_binders": n_binders,
            "binder_frac": n_binders / n_test, "roc_auc": auc, "pr_auc": ap,
        })
        print(f"    Test={test_src}: ROC-AUC={auc:.3f}, PR-AUC={ap:.3f} "
              f"(n={n_test}, {n_binders} binders)")

    # Plot
    res_df = pd.DataFrame(results).dropna(subset=["roc_auc"])
    if res_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC-AUC per source
    colors = [SOURCE_COLORS.get(s, "#94a3b8") for s in res_df["source"]]
    bars = ax1.barh(res_df["source"], res_df["roc_auc"], color=colors)
    ax1.axvline(0.5, color="gray", ls="--", alpha=0.5, label="Random")
    ax1.set_xlabel("ROC-AUC")
    ax1.set_title("Leave-One-Source-Out: ROC-AUC")
    ax1.set_xlim(0, 1.05)
    for bar, row in zip(bars, res_df.itertuples()):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{row.roc_auc:.3f} (n={row.n_test})", va="center", fontsize=9)
    ax1.legend(fontsize=8)

    # PR-AUC per source with baseline
    bars = ax2.barh(res_df["source"], res_df["pr_auc"], color=colors)
    for _, row in res_df.iterrows():
        ax2.plot(row["binder_frac"], row["source"], "k|", markersize=12, mew=2)
    ax2.set_xlabel("PR-AUC")
    ax2.set_title("Leave-One-Source-Out: PR-AUC (| = random baseline)")
    ax2.set_xlim(0, 1.05)
    for bar, row in zip(bars, res_df.itertuples()):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{row.pr_auc:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "16_cross_source_generalization.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> 16_cross_source_generalization.png")

    return results


# ═══════════════════════════════════════════════════════════════════
# 11. PRECISION@K / THRESHOLD SWEEP
# ═══════════════════════════════════════════════════════════════════

def plot_precision_at_k(df: pd.DataFrame):
    """Threshold sweep showing: if you rank by model score and pick top N,
    how many are real binders?

    This is the practical question for design selection: 'I have 500 designs,
    I want to order 50 — which 50 should I pick?'
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        return

    # Use all features, complete cases
    feature_cols = [c for c in df.columns
                    if (c.startswith("docking_") or c.startswith("rosetta_") or
                        c.startswith("af3_") or c.startswith("conformer_"))
                    and not c.endswith("_status")
                    and c not in ("docking_clash_flag", "docking_total_attempts",
                                  "rosetta_n_structures_relaxed")]
    feature_cols = [f for f in feature_cols if df[f].notna().any()]

    extra = ["binder", "ligand_name"]
    extra = [c for c in extra if c in df.columns]
    sub = df[feature_cols + extra].dropna(subset=feature_cols + ["binder"]).copy()

    if len(sub) < 100 or sub["binder"].sum() < 10:
        print("  (Insufficient data for precision@k analysis)")
        return

    X = sub[feature_cols].values
    y = sub["binder"].values

    # Use ligand-grouped CV if possible
    groups = sub["ligand_name"].values if "ligand_name" in sub.columns else None
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    except ImportError:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        groups = None

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])

    y_prob = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        y_prob[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    # Sort by predicted probability (descending)
    order = np.argsort(-y_prob)
    y_sorted = y[order]
    cumulative_binders = np.cumsum(y_sorted)
    n_samples = len(y_sorted)
    k_values = np.arange(1, n_samples + 1)

    precision_at_k = cumulative_binders / k_values
    recall_at_k = cumulative_binders / y.sum()
    binder_rate = y.mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Precision@k curve
    ax = axes[0]
    ax.plot(k_values, precision_at_k, color="#ef4444", lw=2)
    ax.axhline(binder_rate, color="gray", ls="--", alpha=0.5,
               label=f"Random baseline: {binder_rate:.2f}")
    ax.set_xlabel("Top K selected")
    ax.set_ylabel("Precision (fraction that are binders)")
    ax.set_title("Precision @ K")
    ax.legend(fontsize=9)
    # Annotate practical points
    for top_k in [25, 50, 100, 200]:
        if top_k < n_samples:
            p = precision_at_k[top_k - 1]
            ax.annotate(f"top {top_k}: {p:.0%}",
                        xy=(top_k, p), fontsize=8, color="darkred",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                        xytext=(top_k + n_samples * 0.05, p + 0.05))

    # Recall@k curve
    ax = axes[1]
    ax.plot(k_values, recall_at_k, color="#3b82f6", lw=2)
    ax.set_xlabel("Top K selected")
    ax.set_ylabel("Recall (fraction of all binders found)")
    ax.set_title("Recall @ K")
    for top_k in [25, 50, 100, 200]:
        if top_k < n_samples:
            r = recall_at_k[top_k - 1]
            ax.annotate(f"top {top_k}: {r:.0%}",
                        xy=(top_k, r), fontsize=8, color="darkblue",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                        xytext=(top_k + n_samples * 0.05, r - 0.05))

    # Enrichment factor curve
    ax = axes[2]
    enrichment = precision_at_k / binder_rate
    ax.plot(k_values, enrichment, color="#22c55e", lw=2)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="Random (EF=1.0)")
    ax.set_xlabel("Top K selected")
    ax.set_ylabel("Enrichment Factor")
    ax.set_title("Enrichment Factor @ K")
    ax.legend(fontsize=9)
    for top_k in [25, 50, 100]:
        if top_k < n_samples:
            ef = enrichment[top_k - 1]
            ax.annotate(f"top {top_k}: {ef:.1f}x",
                        xy=(top_k, ef), fontsize=8, color="darkgreen",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                        xytext=(top_k + n_samples * 0.05, ef - 0.3))

    fig.suptitle(f"Design Selection Curves (n={n_samples}, {int(y.sum())} binders, "
                 f"ligand-grouped CV)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "17_precision_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 17_precision_at_k.png")

    # Print a practical selection table
    print("\n  Design selection table (model-ranked):")
    print(f"  {'Top K':>8} {'Binders':>10} {'Precision':>12} {'Recall':>10} {'Enrichment':>12}")
    print("  " + "-" * 56)
    for top_k in [10, 25, 50, 100, 200, 500]:
        if top_k <= n_samples:
            p = precision_at_k[top_k - 1]
            r = recall_at_k[top_k - 1]
            ef = enrichment[top_k - 1]
            nb = int(cumulative_binders[top_k - 1])
            print(f"  {top_k:>8} {nb:>10} {p:>12.1%} {r:>10.1%} {ef:>12.1f}x")


# ═══════════════════════════════════════════════════════════════════
# 12. HIGH-CONFIDENCE CORRELATION COMPARISON
# ═══════════════════════════════════════════════════════════════════

def plot_high_confidence_correlations(df: pd.DataFrame):
    """Compare feature-label correlations: all data vs high-confidence subset.

    High-confidence data (experimental, win_ssm) has the most reliable labels.
    If correlations shift substantially, the lower-confidence tiers may be
    introducing noise or different biology.
    """
    if "label_confidence" not in df.columns:
        print("  (No label_confidence column — skipping high-confidence comparison)")
        return

    feature_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ("label", "label_confidence", "binder",
                                 "docking_clash_flag", "docking_total_attempts")]

    confidence = pd.to_numeric(df["label_confidence"], errors="coerce")
    high_conf = df[confidence >= 0.85].copy()

    if len(high_conf) < 50:
        print(f"  (Only {len(high_conf)} high-confidence rows — skipping)")
        return

    print(f"\n  Correlation comparison: all data (n={len(df)}) vs "
          f"high-confidence >= 0.85 (n={len(high_conf)})")

    # Compute Spearman for both
    corrs_all = {}
    corrs_hc = {}
    for col in feature_cols:
        valid_all = df[["label", col]].dropna()
        valid_hc = high_conf[["label", col]].dropna()
        if len(valid_all) > 30:
            r, _ = stats.spearmanr(valid_all["label"], valid_all[col])
            corrs_all[col] = r
        if len(valid_hc) > 20:
            r, _ = stats.spearmanr(valid_hc["label"], valid_hc[col])
            corrs_hc[col] = r

    # Features present in both
    common = sorted(set(corrs_all) & set(corrs_hc),
                    key=lambda x: abs(corrs_all.get(x, 0)), reverse=True)
    if not common:
        return

    top_n = min(20, len(common))
    common = common[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.4)))

    # Side-by-side bar chart
    y_pos = np.arange(top_n)
    width = 0.35
    r_all = [corrs_all[f] for f in common]
    r_hc = [corrs_hc[f] for f in common]

    ax1.barh(y_pos + width, r_all, width, color="#3b82f6", alpha=0.8,
             label=f"All data (n={len(df)})")
    ax1.barh(y_pos, r_hc, width, color="#ef4444", alpha=0.8,
             label=f"High confidence (n={len(high_conf)})")
    ax1.set_yticks(y_pos + width/2)
    ax1.set_yticklabels(common, fontsize=8)
    ax1.set_xlabel("Spearman r with label")
    ax1.set_title("Feature-Label Correlations: All vs High-Confidence")
    ax1.axvline(0, color="black", lw=0.5)
    ax1.legend(fontsize=9)
    ax1.invert_yaxis()

    # Scatter: all r vs high-confidence r
    all_r = [corrs_all.get(f, 0) for f in common]
    hc_r = [corrs_hc.get(f, 0) for f in common]
    ax2.scatter(all_r, hc_r, c="#64748b", s=40, alpha=0.7)
    lim = max(abs(min(all_r + hc_r)), abs(max(all_r + hc_r))) * 1.1
    ax2.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
    ax2.set_xlabel("Spearman r (all data)")
    ax2.set_ylabel("Spearman r (high confidence)")
    ax2.set_title("Correlation Stability")
    # Label points that shifted the most
    for f, ra, rh in zip(common, all_r, hc_r):
        if abs(ra - rh) > 0.1:
            ax2.annotate(f.replace("_", "\n"), xy=(ra, rh), fontsize=6,
                         color="red", ha="center")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "18_high_confidence_correlations.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> 18_high_confidence_correlations.png")

    # Print features where correlation changes sign or shifts by >0.1
    shifted = [(f, corrs_all[f], corrs_hc[f])
               for f in common if abs(corrs_all[f] - corrs_hc[f]) > 0.1]
    if shifted:
        print("  Features with unstable correlations (|shift| > 0.1):")
        for f, ra, rh in shifted:
            print(f"    {f}: all={ra:.3f} → high_conf={rh:.3f} (shift={rh-ra:+.3f})")


# ═══════════════════════════════════════════════════════════════════
# 13. SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def plot_shap_summary(df: pd.DataFrame):
    """SHAP summary plot showing how each feature drives predictions.

    Requires: pip install shap
    """
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("  (shap not installed — skipping SHAP analysis. Install with: pip install shap)")
        return

    feature_cols = [c for c in df.columns
                    if (c.startswith("docking_") or c.startswith("rosetta_") or
                        c.startswith("af3_") or c.startswith("conformer_"))
                    and not c.endswith("_status")
                    and c not in ("docking_clash_flag", "docking_total_attempts",
                                  "rosetta_n_structures_relaxed")]
    feature_cols = [f for f in feature_cols if df[f].notna().any()]

    sub = df[feature_cols + ["binder"]].dropna().copy()
    if len(sub) < 100:
        print("  (Insufficient data for SHAP analysis)")
        return

    X = sub[feature_cols].values
    y = sub["binder"].values

    # Train model on all data (SHAP explains the model, not generalization)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    # Scale for interpretability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf.fit(X_scaled, y)

    # SHAP values (use TreeExplainer for speed)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled)

    # For binary classification, use the positive class
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # class 1 = binder
    else:
        shap_vals = shap_values

    # Summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.3)))
    shap.summary_plot(shap_vals, X_scaled, feature_names=feature_cols,
                      show=False, max_display=25)
    plt.title(f"SHAP Feature Impact on Binder Prediction (n={len(sub)})")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(FIG_DIR / "19_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 19_shap_summary.png")

    # Bar plot of mean |SHAP|
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.3)))
    shap.summary_plot(shap_vals, X_scaled, feature_names=feature_cols,
                      plot_type="bar", show=False, max_display=25)
    plt.title(f"Mean |SHAP| Feature Importance (n={len(sub)})")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(FIG_DIR / "20_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 20_shap_importance.png")


# ═══════════════════════════════════════════════════════════════════
# 14. AF3 BINARY vs TERNARY COMPARISON
# ═══════════════════════════════════════════════════════════════════

def plot_af3_binary_vs_ternary(df: pd.DataFrame):
    """Scatter: AF3 binary ipTM vs ternary ipTM, colored by label."""
    if "af3_binary_ipTM" not in df.columns or "af3_ternary_ipTM" not in df.columns:
        return

    sub = df[["af3_binary_ipTM", "af3_ternary_ipTM", "binder_name", "label_name"]].dropna()
    if len(sub) < 20:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ipTM comparison
    for label, color in LABEL_COLORS.items():
        mask = sub["label_name"] == LABEL_NAMES[label]
        if mask.sum() == 0:
            continue
        ax1.scatter(sub.loc[mask, "af3_binary_ipTM"],
                    sub.loc[mask, "af3_ternary_ipTM"],
                    c=color, label=LABEL_NAMES[label], alpha=0.5, s=20)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("AF3 Binary ipTM")
    ax1.set_ylabel("AF3 Ternary ipTM")
    ax1.set_title("AF3 Binary vs Ternary ipTM")
    ax1.legend(fontsize=9)

    # ipTM difference histogram
    sub["iptm_diff"] = sub["af3_ternary_ipTM"] - sub["af3_binary_ipTM"]
    for binder_name, color in BINARY_COLORS.items():
        vals = sub.loc[sub["binder_name"] == binder_name, "iptm_diff"]
        ax2.hist(vals, bins=30, alpha=0.6, color=color, label=binder_name, density=True)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("ipTM (ternary - binary)")
    ax2.set_ylabel("Density")
    ax2.set_title("Water Effect on ipTM")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "12_af3_binary_vs_ternary.png", dpi=150)
    plt.close(fig)
    print("  -> 12_af3_binary_vs_ternary.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Explore PYR1 ML preliminary features")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to preliminary_features.csv")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PYR1 PRELIMINARY FEATURES — EXPLORATORY ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_and_clean(args.csv)

    # Summary stats
    print(f"\nDataset: {len(df)} pairs, {df['ligand_name'].nunique()} ligands, "
          f"{df['variant_name'].nunique()} variants")
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    print(f"Binary: {df['binder'].sum()} binders, {(~df['binder'].astype(bool)).sum()} non-binders")

    print_completeness_by_tier(df)

    # Generate all figures
    print("\nGenerating figures...")

    # ── Dataset overview ──
    print("\n── Dataset Overview ──")
    plot_completeness(df)
    plot_label_overview(df)

    # ── Feature distributions (global) ──
    print("\n── Feature Distributions ──")
    plot_feature_distributions(df)
    plot_binary_boxplots(df)

    # ── Correlations ──
    print("\n── Correlation Analysis ──")
    plot_correlation_matrix(df)
    corr_df = plot_top_correlations_with_label(df)

    # Print top correlations
    if corr_df is not None:
        print("\nTop 10 features correlated with binding label (Spearman):")
        for feat, row in corr_df.head(10).iterrows():
            direction = "+" if row["spearman_r"] > 0 else "-"
            print(f"  {direction} {feat}: r={row['spearman_r']:.3f} "
                  f"(p={row['p_value']:.2e}, n={int(row['n'])})")

    # ── Ligand & pairwise analysis ──
    print("\n── Ligand & Pairwise Analysis ──")
    plot_ligand_effects(df)
    plot_pairwise_scatter(df)

    # ── AF3 binary vs ternary ──
    print("\n── AF3 Binary vs Ternary ──")
    plot_af3_binary_vs_ternary(df)

    # ── Per-source analysis (NEW) ──
    print("\n── Per-Source Analysis ──")
    plot_per_source_overview(df)
    plot_per_source_feature_dists(df)

    # ── High-confidence correlation comparison (NEW) ──
    print("\n── High-Confidence Correlation Comparison ──")
    plot_high_confidence_correlations(df)

    # ── Baseline classifier (with ligand-grouped CV) ──
    print("\n" + "=" * 60)
    print("BASELINE CLASSIFIER (ligand-grouped CV)")
    print("=" * 60)
    run_baseline_classifier(df)

    # ── Cross-source generalization (NEW) ──
    print("\n" + "=" * 60)
    print("CROSS-SOURCE GENERALIZATION")
    print("=" * 60)
    evaluate_cross_source_generalization(df)

    # ── Precision@k / design selection (NEW) ──
    print("\n" + "=" * 60)
    print("DESIGN SELECTION CURVES (Precision@K)")
    print("=" * 60)
    plot_precision_at_k(df)

    # ── SHAP analysis (NEW, optional) ──
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)
    plot_shap_summary(df)

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
