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
                      sample_weights=None):
    """Run a single RF model on complete-case data. No imputation.

    Args:
        df_sub: DataFrame with features + 'binder' column
        feature_cols: List of feature column names
        model_name: Display name for the model
        fig_suffix: Suffix for output filenames
        sample_weights: Optional array of per-sample weights (aligned to df_sub index).
            If provided, used for training and weighted evaluation metrics.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        classification_report, roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve, confusion_matrix
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Complete cases only — no imputation
    required_cols = feature_cols + ["binder"]
    if sample_weights is not None:
        df_sub = df_sub.copy()
        df_sub["_sample_weight"] = sample_weights
        required_cols = required_cols + ["_sample_weight"]

    sub = df_sub[required_cols].dropna(subset=feature_cols + ["binder"]).copy()

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Manual CV loop to support sample_weight in fit
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
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
    print("\n  --- Unweighted evaluation ---")
    results_unweighted = []
    for name, feats, suffix in models_config:
        if not feats:
            print(f"\n  [{name}] No features available — skipping")
            continue
        r = _run_single_model(df, feats, name, suffix)
        if r:
            results_unweighted.append(r)

    _plot_classifier_results(results_unweighted, "09", " — Unweighted")

    # ── Mode 2: Confidence-weighted ──
    has_confidence = "label_confidence" in df.columns and df["label_confidence"].notna().any()
    results_weighted = []
    if has_confidence:
        print("\n  --- Confidence-weighted evaluation ---")
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
                                  sample_weights=weights.values)
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
# 9. AF3 BINARY vs TERNARY COMPARISON
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
    plot_completeness(df)
    plot_label_overview(df)
    plot_feature_distributions(df)
    plot_binary_boxplots(df)
    plot_correlation_matrix(df)
    corr_df = plot_top_correlations_with_label(df)
    plot_ligand_effects(df)
    plot_pairwise_scatter(df)
    plot_af3_binary_vs_ternary(df)

    # Print top correlations
    if corr_df is not None:
        print("\nTop 10 features correlated with binding label (Spearman):")
        for feat, row in corr_df.head(10).iterrows():
            direction = "+" if row["spearman_r"] > 0 else "-"
            print(f"  {direction} {feat}: r={row['spearman_r']:.3f} (p={row['p_value']:.2e}, n={int(row['n'])})")

    # Baseline classifier
    print("\n" + "=" * 60)
    print("BASELINE CLASSIFIER")
    print("=" * 60)
    run_baseline_classifier(df)

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
