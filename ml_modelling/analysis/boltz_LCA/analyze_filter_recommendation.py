#!/usr/bin/env python3
"""
Head-to-head comparison of Boltz2 filter strategies for iterative MPNN pipeline.

Validates on LCA (52 binders), GLCA (18 binders), LCA-3-S (59 binders),
and pooled (129 binders / 1500 non-binders).

Tests 7 candidate strategies including the biologically suspect ternary_hbond_distance
and grounded alternatives (Trp211 lock distance, HAB1 pLDDT).

Produces 5 figures + console summary + CSV.

Usage:
    python analyze_filter_recommendation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent
LABELS_DIR = RESULTS_DIR.parents[1] / "data" / "boltz_lca_conjugates"
OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Colors ───────────────────────────────────────────────────────────────────

BINDER_COLOR = "#D62728"
NONBINDER_COLOR = "#1F77B4"

STRATEGIES = OrderedDict([
    ("A", {"name": "Pocket pLDDT only",
           "desc": "binary_plddt_pocket (raw)",
           "color": "#1F77B4", "gated": False}),
    ("B", {"name": "Z(pocket) + Z(tern ipLDDT)",
           "desc": "Z(binary_plddt_pocket) + Z(ternary_complex_iplddt)",
           "color": "#FF7F0E", "gated": False}),
    ("C", {"name": "Gate + Z(tern ipLDDT + hbond)",
           "desc": "Gate plddt_lig>=0.80, Z(tern_ipLDDT)+Z(tern_hbond_dist)",
           "color": "#2CA02C", "gated": True}),
    ("D", {"name": "Gate + Z(tern ipLDDT + Trp211)",
           "desc": "Gate plddt_lig>=0.80, Z(tern_ipLDDT)+Z(-tern_trp211_dist)",
           "color": "#D62728", "gated": True}),
    ("E", {"name": "Gate + tern ipLDDT (raw)",
           "desc": "Gate plddt_lig>=0.80, ternary_complex_iplddt (raw)",
           "color": "#9467BD", "gated": True}),
    ("F", {"name": "Gate + Z(tern ipLDDT + HAB1)",
           "desc": "Gate plddt_lig>=0.80, Z(tern_ipLDDT)+Z(tern_plddt_hab1)",
           "color": "#8C564B", "gated": True}),
    ("G", {"name": "Boltz score only",
           "desc": "binary_boltz_score (raw)",
           "color": "#E377C2", "gated": False}),
    ("H", {"name": "pLDDT+hbond gate, pocket rank",
           "desc": "Gate plddt_lig>=0.80 + hbond_dist<=4A, rank by pocket pLDDT",
           "color": "#17BECF", "gated": True}),
])


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(ligand_key: str) -> pd.DataFrame:
    """Load merged Boltz2 results and join with experimental labels."""
    merged = pd.read_csv(RESULTS_DIR / f"boltz_{ligand_key}_merged_results.csv")
    labels = pd.read_csv(LABELS_DIR / f"boltz_{ligand_key}_binary.csv")

    label_map = labels.set_index("pair_id")["label"].to_dict()
    merged["label"] = merged["name"].map(label_map)
    merged = merged.dropna(subset=["label"])
    merged["label"] = merged["label"].astype(int)

    for col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="ignore")

    merged["ligand_key"] = ligand_key
    return merged


print("Loading datasets...")
df_lca = load_dataset("lca")
df_glca = load_dataset("glca")
df_lca3s = load_dataset("lca3s")
df_pooled = pd.concat([df_lca, df_glca, df_lca3s], ignore_index=True)

DATASETS = OrderedDict([
    ("LCA", df_lca),
    ("GLCA", df_glca),
    ("LCA-3-S", df_lca3s),
    ("Pooled", df_pooled),
])

for name, df in DATASETS.items():
    n_pos = int(df["label"].sum())
    n_neg = int((df["label"] == 0).sum())
    print(f"  {name}: {len(df)} rows, {n_pos} binders, {n_neg} non-binders")


# ═══════════════════════════════════════════════════════════════════════════
# Step 2-3: Compute strategy scores
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_strategy_scores(df: pd.DataFrame, gate_threshold: float = 0.80) -> pd.DataFrame:
    """Add score columns for all 7 strategies. Higher = better for all."""
    df = df.copy()

    # Strategy A: binary_plddt_pocket (raw)
    df["score_A"] = pd.to_numeric(df["binary_plddt_pocket"], errors="coerce")

    # Strategy B: Z(binary_plddt_pocket) + Z(ternary_complex_iplddt)
    bp = pd.to_numeric(df["binary_plddt_pocket"], errors="coerce")
    ti = pd.to_numeric(df["ternary_complex_iplddt"], errors="coerce")
    valid_b = bp.notna() & ti.notna()
    score_b = pd.Series(np.nan, index=df.index)
    if valid_b.sum() > 2:
        score_b[valid_b] = zscore(bp[valid_b].astype(float)) + zscore(ti[valid_b].astype(float))
    df["score_B"] = score_b

    # Gate mask for strategies C-F
    gate_vals = pd.to_numeric(df["binary_plddt_ligand"], errors="coerce")
    gate_mask = gate_vals >= gate_threshold

    # Strategy C: gated, Z(ternary_ipLDDT) + Z(ternary_hbond_distance)
    df["score_C"] = _gated_composite(
        df, gate_mask,
        "ternary_complex_iplddt", True,
        "ternary_hbond_distance", True,
    )

    # Strategy D: gated, Z(ternary_ipLDDT) + Z(-ternary_trp211_ligand_distance)
    df["score_D"] = _gated_composite(
        df, gate_mask,
        "ternary_complex_iplddt", True,
        "ternary_trp211_ligand_distance", False,  # lower = better, so negate
    )

    # Strategy E: gated, ternary_complex_iplddt raw
    score_e = pd.Series(-np.inf, index=df.index)
    ti_gated = pd.to_numeric(df.loc[gate_mask, "ternary_complex_iplddt"], errors="coerce")
    valid_e = ti_gated.notna()
    score_e.loc[gate_mask.values & valid_e.reindex(df.index, fill_value=False).values] = \
        ti_gated[valid_e].values
    # Simpler approach:
    score_e = pd.Series(-np.inf, index=df.index)
    for idx in df.index[gate_mask]:
        v = pd.to_numeric(df.loc[idx, "ternary_complex_iplddt"], errors="coerce")
        if pd.notna(v):
            score_e.loc[idx] = v
    df["score_E"] = score_e

    # Strategy F: gated, Z(ternary_ipLDDT) + Z(ternary_plddt_hab1)
    df["score_F"] = _gated_composite(
        df, gate_mask,
        "ternary_complex_iplddt", True,
        "ternary_plddt_hab1", True,
    )

    # Strategy G: binary_boltz_score (raw)
    df["score_G"] = pd.to_numeric(df["binary_boltz_score"], errors="coerce")

    # Strategy H: pLDDT ligand gate + hbond distance gate, rank by pocket pLDDT
    hbond = pd.to_numeric(df["binary_hbond_distance"], errors="coerce")
    hbond_gate = gate_mask & (hbond <= 4.0)
    score_h = pd.Series(-np.inf, index=df.index)
    pocket_vals = pd.to_numeric(df.loc[hbond_gate, "binary_plddt_pocket"], errors="coerce")
    valid_h = pocket_vals.notna()
    score_h.loc[hbond_gate[hbond_gate].index[valid_h.values]] = pocket_vals[valid_h].values
    df["score_H"] = score_h

    return df


def _gated_composite(df, gate_mask, col1, hib1, col2, hib2):
    """Compute Z-score composite within gated subset. Non-gated get -inf."""
    score = pd.Series(-np.inf, index=df.index)
    gated_idx = df.index[gate_mask]

    v1 = pd.to_numeric(df.loc[gated_idx, col1], errors="coerce")
    v2 = pd.to_numeric(df.loc[gated_idx, col2], errors="coerce")
    valid = v1.notna() & v2.notna()

    if valid.sum() < 3:
        return score

    valid_idx = gated_idx[valid.values]
    z1 = zscore(v1[valid].astype(float))
    z2 = zscore(v2[valid].astype(float))

    if not hib1:
        z1 = -z1
    if not hib2:
        z2 = -z2

    score.loc[valid_idx] = np.asarray(z1) + np.asarray(z2)
    return score


# Compute scores for all datasets
print("\nComputing strategy scores...")
scored = {}
for name, df in DATASETS.items():
    if name == "Pooled":
        # For pooled: compute Z-scores per-ligand to avoid cross-ligand distortion
        parts = []
        for lig_key in df["ligand_key"].unique():
            sub = df[df["ligand_key"] == lig_key]
            parts.append(compute_all_strategy_scores(sub))
        scored[name] = pd.concat(parts, ignore_index=True)
    else:
        scored[name] = compute_all_strategy_scores(df)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_roc(df, score_col):
    """Compute ROC curve and AUC. Returns (fpr, tpr, auc) or None."""
    vals = df[score_col]
    mask = np.isfinite(vals) & vals.notna()
    if mask.sum() < 10:
        return None
    y = df.loc[mask, "label"].values
    s = vals[mask].values
    if len(np.unique(y)) < 2:
        return None
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, auc


def enrichment_curve(df, score_col, n_points=200):
    """Compute enrichment curve: (frac_pct, recall_pct)."""
    vals = df[score_col]
    mask = np.isfinite(vals) & vals.notna()
    sub = df[mask].copy()
    sub["_score"] = vals[mask].values
    sub = sub.sort_values("_score", ascending=False)
    n_total = len(sub)
    n_pos = int(sub["label"].sum())
    if n_pos == 0:
        return np.array([]), np.array([])
    fracs = np.linspace(0, 1, n_points)
    recall = []
    for f in fracs:
        n_take = max(1, int(np.ceil(f * n_total)))
        recall.append(sub.head(n_take)["label"].sum() / n_pos)
    return fracs * 100, np.array(recall) * 100


def binders_at_topN(df, score_col, N):
    """Count binders in top-N by score."""
    vals = df[score_col]
    mask = np.isfinite(vals) & vals.notna()
    sub = df[mask].sort_values(score_col, ascending=False)
    n_take = min(N, len(sub))
    n_captured = int(sub.head(n_take)["label"].sum())
    n_total_binders = int(df["label"].sum())
    return n_captured, n_total_binders


def compute_optimal_mcc(df, score_col, n_thresholds=200):
    """Compute optimal MCC by sweeping score thresholds.

    Returns (best_mcc, best_threshold, mcc_at_top100) or None.
    """
    vals = df[score_col]
    mask = np.isfinite(vals) & vals.notna()
    if mask.sum() < 10:
        return None
    y = df.loc[mask, "label"].values
    s = vals[mask].values
    if len(np.unique(y)) < 2:
        return None

    # Sweep thresholds from score percentiles
    thresholds = np.percentile(s, np.linspace(1, 99, n_thresholds))

    best_mcc = -1.0
    best_thr = thresholds[0]
    for thr in thresholds:
        pred = (s >= thr).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        mcc = matthews_corrcoef(y, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr

    # MCC at top-100 cutoff
    n_take = min(100, len(s))
    sorted_idx = np.argsort(-s)
    pred_top100 = np.zeros(len(s), dtype=int)
    pred_top100[sorted_idx[:n_take]] = 1
    mcc_top100 = matthews_corrcoef(y, pred_top100)

    return best_mcc, best_thr, mcc_top100


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Gate threshold optimization
# ═══════════════════════════════════════════════════════════════════════════

def sweep_gate_threshold(df, gate_col, thresholds):
    """Sweep gate thresholds and compute within-gate AUCs."""
    gate_vals = pd.to_numeric(df[gate_col], errors="coerce")
    n_pos_total = int(df["label"].sum())
    n_neg_total = int((df["label"] == 0).sum())

    rows = []
    for thr in thresholds:
        mask = gate_vals >= thr
        sub = df[mask]
        n_pos = int(sub["label"].sum())
        n_neg = int((sub["label"] == 0).sum())

        row = {
            "threshold": thr,
            "n_pass": int(mask.sum()),
            "pct_binders_retained": 100 * n_pos / max(1, n_pos_total),
            "pct_nonbinders_removed": 100 * (n_neg_total - n_neg) / max(1, n_neg_total),
        }

        # Within-gate AUC for key metrics
        for metric, hib in [
            ("ternary_complex_iplddt", True),
            ("ternary_trp211_ligand_distance", False),
            ("ternary_hbond_distance", True),
        ]:
            if n_pos < 5 or n_neg < 10 or metric not in sub.columns:
                row[f"auc_{metric}"] = np.nan
                continue
            v = pd.to_numeric(sub[metric], errors="coerce")
            m = v.notna()
            y = sub.loc[m, "label"].values
            if len(np.unique(y)) < 2 or m.sum() < 10:
                row[f"auc_{metric}"] = np.nan
                continue
            s = v[m].values if hib else -v[m].values
            try:
                row[f"auc_{metric}"] = roc_auc_score(y, s)
            except Exception:
                row[f"auc_{metric}"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def find_optimal_gate(df, gate_col, target_retention=0.90):
    """Find highest gate threshold retaining >= target_retention of binders."""
    thresholds = np.linspace(0.50, 0.95, 91)
    sweep = sweep_gate_threshold(df, gate_col, thresholds)
    passing = sweep[sweep["pct_binders_retained"] >= target_retention * 100]
    if len(passing) == 0:
        return thresholds[0]
    return passing["threshold"].max()


print("\nGate threshold optimization (binary_plddt_ligand)...")
for name in ["LCA", "GLCA", "LCA-3-S"]:
    opt = find_optimal_gate(DATASETS[name], "binary_plddt_ligand")
    print(f"  {name}: optimal gate = {opt:.2f} (retains >=90% binders)")

GATE_THRESHOLD = 0.80  # Use fixed threshold as validated
print(f"\nUsing gate threshold: {GATE_THRESHOLD}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: ROC comparison (2x2: LCA, GLCA, LCA-3-S, Pooled)
# ═══════════════════════════════════════════════════════════════════════════

print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, (ds_name, df) in zip(axes, scored.items()):
    for sid, info in STRATEGIES.items():
        result = compute_roc(df, f"score_{sid}")
        if result is None:
            continue
        fpr, tpr, auc = result
        lw = 2.5 if info["gated"] else 1.5
        ax.plot(fpr, tpr, color=info["color"], lw=lw,
                label=f"{sid}: {info['name']}  ({auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    n_pos = int(df["label"].sum())
    n_neg = int((df["label"] == 0).sum())
    ax.set_title(f"{ds_name}  (B={n_pos}, NB={n_neg})", fontsize=11)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(fontsize=7, loc="lower right")

fig.suptitle("ROC Curves: 7 Filter Strategies Across Ligand Datasets", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_roc_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved fig1_roc_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: Enrichment at top-100
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: enrichment curves on pooled data
ax = axes[0]
ax.plot([0, 100], [0, 100], "k--", lw=0.8, alpha=0.4, label="Random")

df_p = scored["Pooled"]
top100_pct = 100 * 100 / len(df_p)
ax.axvline(top100_pct, color="gray", lw=1.2, ls=":", alpha=0.7)
ax.text(top100_pct + 0.5, 5, f"Top 100\n({top100_pct:.0f}%)", fontsize=8, color="gray")

for sid, info in STRATEGIES.items():
    fracs, recall = enrichment_curve(df_p, f"score_{sid}")
    if len(fracs) == 0:
        continue
    lw = 2.5 if info["gated"] else 1.5
    ax.plot(fracs, recall, color=info["color"], lw=lw, label=f"{sid}: {info['name']}")

ax.set_xlabel("Top X% of designs scored", fontsize=11)
ax.set_ylabel("% binders captured", fontsize=11)
ax.set_title("Enrichment Curves (Pooled, n=1748)", fontsize=11)
ax.legend(fontsize=7.5, loc="lower right")
ax.set_xlim(0, 100)
ax.set_ylim(0, 101)

# Right: bar chart — binders at top-100 for each dataset
ax2 = axes[1]
ds_names_bar = ["LCA", "GLCA", "LCA-3-S", "Pooled"]
n_strategies = len(STRATEGIES)
n_datasets = len(ds_names_bar)
x = np.arange(n_strategies)
bar_width = 0.18
offsets = np.linspace(-(n_datasets - 1) / 2, (n_datasets - 1) / 2, n_datasets) * bar_width
ds_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

for i, ds_name in enumerate(ds_names_bar):
    df = scored[ds_name]
    bars = []
    for sid in STRATEGIES.keys():
        n_cap, n_tot = binders_at_topN(df, f"score_{sid}", 100)
        bars.append(100 * n_cap / max(1, n_tot))
    ax2.bar(x + offsets[i], bars, bar_width, color=ds_colors[i], alpha=0.80,
            label=ds_name, edgecolor="black", linewidth=0.5)

    # Annotate with counts
    for j, sid in enumerate(STRATEGIES.keys()):
        n_cap, n_tot = binders_at_topN(df, f"score_{sid}", 100)
        ax2.text(x[j] + offsets[i], bars[j] + 1, f"{n_cap}/{n_tot}",
                 ha="center", fontsize=5.5, rotation=90, va="bottom")

ax2.set_xticks(x)
ax2.set_xticklabels([f"{sid}" for sid in STRATEGIES.keys()], fontsize=10)
ax2.set_ylabel("% binders captured at top-100", fontsize=11)
ax2.set_title("Binder Recall at Top-100 Selection", fontsize=11)
ax2.legend(fontsize=8, loc="upper right")
ax2.set_ylim(0, 120)

# Strategy labels below
for j, (sid, info) in enumerate(STRATEGIES.items()):
    ax2.text(x[j], -8, info["name"], ha="center", fontsize=6, rotation=30, va="top")

fig.suptitle("Filter Strategy Enrichment: Top-100 Selection\n"
             f"(Gate threshold: binary_plddt_ligand >= {GATE_THRESHOLD})",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig(OUT_DIR / "fig2_enrichment_top100.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved fig2_enrichment_top100.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3: Gate threshold optimization
# ═══════════════════════════════════════════════════════════════════════════

thresholds = np.linspace(0.50, 0.92, 50)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, ds_name in zip(axes, ["LCA", "GLCA", "LCA-3-S"]):
    df = DATASETS[ds_name]
    sweep = sweep_gate_threshold(df, "binary_plddt_ligand", thresholds)

    ax2 = ax.twinx()

    # AUC lines
    for metric, label, color, ls in [
        ("auc_ternary_complex_iplddt", "tern ipLDDT AUC", "#1F77B4", "-"),
        ("auc_ternary_trp211_ligand_distance", "tern Trp211 AUC", "#D62728", "-"),
        ("auc_ternary_hbond_distance", "tern hbond AUC (suspect)", "#2CA02C", "--"),
    ]:
        ax.plot(sweep["threshold"], sweep[metric], color=color, lw=1.8, ls=ls, label=label)

    # Binder retention on secondary axis
    ax2.fill_between(sweep["threshold"], sweep["pct_binders_retained"],
                     alpha=0.08, color="gray")
    ax2.plot(sweep["threshold"], sweep["pct_binders_retained"],
             color="gray", lw=1, ls="--", label="% binders retained")
    ax2.set_ylabel("% binders retained", fontsize=9, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 110)

    ax.axvline(GATE_THRESHOLD, color="black", lw=1.2, ls=":", alpha=0.6)
    ax.axhline(0.70, color="black", lw=0.5, ls=":", alpha=0.3)
    ax.set_xlabel("binary_plddt_ligand threshold", fontsize=10)
    ax.set_ylabel("Within-gate AUC", fontsize=10)
    n_pos = int(df["label"].sum())
    ax.set_title(f"{ds_name} (B={n_pos})", fontsize=11)
    ax.set_ylim(0.40, 1.0)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle("Gate Threshold Optimization: binary_plddt_ligand\n"
             "(within-gate AUC for ranking metrics vs binder retention)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_gate_optimization.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved fig3_gate_optimization.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4: Ternary hbond distance concern
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

df_p = df_pooled.copy()

# Left: binary vs ternary hbond distance scatter
ax = axes[0]
for label_val, color, name, s, alpha, zorder in [
    (0, NONBINDER_COLOR, "Non-binder", 12, 0.25, 2),
    (1, BINDER_COLOR, "Binder", 30, 0.8, 3),
]:
    sub = df_p[df_p["label"] == label_val]
    x = pd.to_numeric(sub["binary_hbond_distance"], errors="coerce")
    y = pd.to_numeric(sub["ternary_hbond_distance"], errors="coerce")
    mask = x.notna() & y.notna()
    ax.scatter(x[mask], y[mask], c=color, s=s, alpha=alpha, zorder=zorder, label=name)

ax.axhline(3.5, color="black", lw=1, ls="--", alpha=0.5)
ax.text(0.5, 3.7, "Max H-bond distance (3.5 A)", fontsize=8, color="black", alpha=0.7)
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], "k:", lw=0.8, alpha=0.3)
ax.set_xlabel("Binary H-bond distance (A)", fontsize=10)
ax.set_ylabel("Ternary H-bond distance (A)", fontsize=10)
ax.set_title("H-bond distance: binary vs ternary\n"
             "Binders have ternary dist >> 3.5A (not real H-bonds)", fontsize=10)
ax.legend(fontsize=8)

# Right: violin of ternary_trp211_ligand_distance (grounded alternative)
ax = axes[1]
vals_pos = pd.to_numeric(df_p.loc[df_p["label"] == 1, "ternary_trp211_ligand_distance"],
                         errors="coerce").dropna().values
vals_neg = pd.to_numeric(df_p.loc[df_p["label"] == 0, "ternary_trp211_ligand_distance"],
                         errors="coerce").dropna().values

parts = ax.violinplot([vals_neg, vals_pos], positions=[0, 1],
                      showmedians=True, showextrema=True, widths=0.7)
parts["cmedians"].set_color("white")
parts["cmedians"].set_linewidth(2)
for pc, color in zip(parts["bodies"], [NONBINDER_COLOR, BINDER_COLOR]):
    pc.set_facecolor(color)
    pc.set_alpha(0.70)
for pname in ("cbars", "cmins", "cmaxes"):
    parts[pname].set_color("black")
    parts[pname].set_linewidth(0.6)

rng = np.random.default_rng(42)
for vals, px, color in [(vals_neg, 0, NONBINDER_COLOR), (vals_pos, 1, BINDER_COLOR)]:
    jx = rng.uniform(px - 0.12, px + 0.12, size=len(vals))
    ax.scatter(jx, vals, s=5, alpha=0.3, color=color, zorder=3)

# AUC
score_pos = -vals_pos  # lower = better
score_neg = -vals_neg
all_scores = np.concatenate([score_neg, score_pos])
all_labels = np.array([0] * len(vals_neg) + [1] * len(vals_pos))
auc = roc_auc_score(all_labels, all_scores)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Non-binder", "Binder"], fontsize=10)
ax.set_ylabel("Ternary Trp211-ligand distance (A)", fontsize=10)
ax.set_title(f"Trp211 lock distance (lower = lock engaged)\n"
             f"AUC = {auc:.3f} (biologically grounded alternative)", fontsize=10)

fig.suptitle("Ternary H-bond Distance Artifact vs Biologically Grounded Alternative\n"
             "(Pooled data: 129 binders, 1500 non-binders)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_hbond_distance_concern.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved fig4_hbond_distance_concern.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5: Head-to-head summary bar chart
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 6))

ds_names_bar = ["LCA", "GLCA", "LCA-3-S", "Pooled"]
n_s = len(STRATEGIES)
n_d = len(ds_names_bar)
x = np.arange(n_s)
bar_width = 0.18
offsets = np.linspace(-(n_d - 1) / 2, (n_d - 1) / 2, n_d) * bar_width
ds_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

for i, ds_name in enumerate(ds_names_bar):
    df = scored[ds_name]
    bars = []
    annotations = []
    for sid in STRATEGIES.keys():
        n_cap, n_tot = binders_at_topN(df, f"score_{sid}", 100)
        bars.append(100 * n_cap / max(1, n_tot))
        annotations.append(f"{n_cap}/{n_tot}")

    rects = ax.bar(x + offsets[i], bars, bar_width, color=ds_colors[i], alpha=0.80,
                   label=ds_name, edgecolor="black", linewidth=0.5)

    for j, (rect, ann) in enumerate(zip(rects, annotations)):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1,
                ann, ha="center", fontsize=6, rotation=90, va="bottom")

ax.set_xticks(x)
xlabels = [f"{sid}: {info['name']}" for sid, info in STRATEGIES.items()]
ax.set_xticklabels(xlabels, fontsize=7.5, rotation=25, ha="right")
ax.set_ylabel("% binders captured at top-100", fontsize=11)
ax.set_title("Strategy Comparison: Binder Recall at Top-100 Selection\n"
             f"(Gate threshold: binary_plddt_ligand >= {GATE_THRESHOLD})",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, 130)
ax.axhline(100, color="gray", lw=0.5, ls=":", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_strategy_summary.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved fig5_strategy_summary.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6: Matthews Correlation Coefficient
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ds_names_bar = ["LCA", "GLCA", "LCA-3-S", "Pooled"]
n_s = len(STRATEGIES)
n_d = len(ds_names_bar)
x = np.arange(n_s)
bar_width = 0.18
offsets = np.linspace(-(n_d - 1) / 2, (n_d - 1) / 2, n_d) * bar_width
ds_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# Left panel: optimal MCC (best threshold)
ax = axes[0]
for i, ds_name in enumerate(ds_names_bar):
    df = scored[ds_name]
    bars = []
    for sid in STRATEGIES.keys():
        result = compute_optimal_mcc(df, f"score_{sid}")
        bars.append(result[0] if result else 0.0)
    rects = ax.bar(x + offsets[i], bars, bar_width, color=ds_colors[i], alpha=0.80,
                   label=ds_name, edgecolor="black", linewidth=0.5)
    for j, rect in enumerate(rects):
        if bars[j] > 0.01:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005,
                    f"{bars[j]:.2f}", ha="center", fontsize=5.5, rotation=90, va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([sid for sid in STRATEGIES.keys()], fontsize=10)
ax.set_ylabel("Optimal MCC", fontsize=11)
ax.set_title("Optimal MCC (best threshold per strategy)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, 0.7)
ax.axhline(0, color="black", lw=0.5)

for j, (sid, info) in enumerate(STRATEGIES.items()):
    ax.text(x[j], -0.04, info["name"], ha="center", fontsize=5.5, rotation=30, va="top")

# Right panel: MCC at top-100 cutoff
ax = axes[1]
for i, ds_name in enumerate(ds_names_bar):
    df = scored[ds_name]
    bars = []
    for sid in STRATEGIES.keys():
        result = compute_optimal_mcc(df, f"score_{sid}")
        bars.append(result[2] if result else 0.0)
    rects = ax.bar(x + offsets[i], bars, bar_width, color=ds_colors[i], alpha=0.80,
                   label=ds_name, edgecolor="black", linewidth=0.5)
    for j, rect in enumerate(rects):
        if bars[j] > 0.01:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005,
                    f"{bars[j]:.2f}", ha="center", fontsize=5.5, rotation=90, va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([sid for sid in STRATEGIES.keys()], fontsize=10)
ax.set_ylabel("MCC at top-100 cutoff", fontsize=11)
ax.set_title("MCC when selecting top 100 as positives", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(-0.1, 0.7)
ax.axhline(0, color="black", lw=0.5)

for j, (sid, info) in enumerate(STRATEGIES.items()):
    ax.text(x[j], -0.12, info["name"], ha="center", fontsize=5.5, rotation=30, va="top")

fig.suptitle("Matthews Correlation Coefficient: Strategy Comparison\n"
             f"(Gate threshold: binary_plddt_ligand >= {GATE_THRESHOLD})",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(OUT_DIR / "fig6_mcc_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved fig6_mcc_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Console summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  FILTER STRATEGY COMPARISON (top 100 selection)")
print("=" * 90)

header = f"{'Strategy':<4} {'Description':<45}"
for ds_name in ["LCA", "GLCA", "LCA-3-S", "Pooled"]:
    header += f" {ds_name:>6}_AUC {ds_name:>4}_MCC {ds_name:>6}_B@100"
print(header)
print("-" * 90)

results_rows = []
for sid, info in STRATEGIES.items():
    row_str = f"{sid:<4} {info['name']:<45}"
    row_data = {"strategy": sid, "name": info["name"]}

    for ds_name in ["LCA", "GLCA", "LCA-3-S", "Pooled"]:
        df = scored[ds_name]
        result = compute_roc(df, f"score_{sid}")
        auc_val = result[2] if result else np.nan
        n_cap, n_tot = binders_at_topN(df, f"score_{sid}", 100)

        mcc_result = compute_optimal_mcc(df, f"score_{sid}")
        opt_mcc = mcc_result[0] if mcc_result else np.nan
        mcc_100 = mcc_result[2] if mcc_result else np.nan

        row_str += f"   {auc_val:5.3f}  {opt_mcc:5.3f}  {n_cap:2d}/{n_tot:<3d}"
        ds_key = ds_name.lower().replace("-", "")
        row_data[f"{ds_key}_auc"] = round(auc_val, 4) if pd.notna(auc_val) else None
        row_data[f"{ds_key}_mcc"] = round(opt_mcc, 4) if pd.notna(opt_mcc) else None
        row_data[f"{ds_key}_binders_at_100"] = f"{n_cap}/{n_tot}"

    print(row_str)
    results_rows.append(row_data)

print("-" * 90)

# Find best strategy by pooled enrichment
best_sid = None
best_capture = 0
for sid in STRATEGIES.keys():
    n_cap, n_tot = binders_at_topN(scored["Pooled"], f"score_{sid}", 100)
    if n_cap > best_capture:
        best_capture = n_cap
        best_sid = sid

print(f"\nRECOMMENDATION: Strategy {best_sid}")
print(f"  {STRATEGIES[best_sid]['desc']}")
print(f"  Gate: binary_plddt_ligand >= {GATE_THRESHOLD}")
print(f"  Pooled: captures {best_capture}/129 binders in top 100")
print(f"\n  NOTE: Gate threshold is absolute (cross-batch stable).")
print(f"        Z-score ranking must be recomputed each round on the full pool.")

# Also print runner-up
second_best_sid = None
second_best = 0
for sid in STRATEGIES.keys():
    if sid == best_sid:
        continue
    n_cap, _ = binders_at_topN(scored["Pooled"], f"score_{sid}", 100)
    if n_cap > second_best:
        second_best = n_cap
        second_best_sid = sid

if second_best_sid:
    print(f"\n  Runner-up: Strategy {second_best_sid} ({STRATEGIES[second_best_sid]['name']})")
    print(f"  Pooled: captures {second_best}/129 binders in top 100")


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Save CSV
# ═══════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(results_rows)
csv_path = RESULTS_DIR / "filter_strategy_comparison.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nSaved summary CSV: {csv_path}")
print(f"All figures saved to: {OUT_DIR}")
