#!/usr/bin/env python3
"""
Presentation-quality figures demonstrating Strategy H (Boltz2 filter).

Strategy H:
  1. Pre-filter:  binary_plddt_ligand >= 0.80
  2. Structural:  binary_hbond_distance <= 4.0 A
  3. Rank by:     binary_plddt_pocket (higher = better)
  4. Select:      top 100

Produces 4 figures + MD candidate CSV (LCA top 100).

Usage:
    python plot_strategy_h_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent
LABELS_DIR = RESULTS_DIR.parents[1] / "data" / "boltz_lca_conjugates"
OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────

BINDER_COLOR = "#D62728"
NONBINDER_COLOR = "#7F7F7F"
ACCENT_COLOR = "#17BECF"
PASS_FILL = "#17BECF"

DS_COLORS = OrderedDict([
    ("LCA", "#4C72B0"),
    ("GLCA", "#55A868"),
    ("LCA-3-S", "#C44E52"),
])

GATE_PLDDT = 0.80
GATE_HBOND = 4.0
TOP_N = 100

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(ligand_key: str) -> pd.DataFrame:
    """Load merged Boltz2 results and join with experimental labels."""
    merged = pd.read_csv(RESULTS_DIR / f"boltz_{ligand_key}_merged_results.csv")
    labels = pd.read_csv(LABELS_DIR / f"boltz_{ligand_key}_binary.csv")

    label_map = labels.set_index("pair_id")["label"].to_dict()
    merged["label"] = merged["name"].map(label_map)
    merged = merged.dropna(subset=["label"])
    merged["label"] = merged["label"].astype(int)

    # Join variant metadata for CSV output
    meta_cols = ["pair_id", "variant_name", "variant_signature", "label_tier"]
    meta = labels[meta_cols].copy()
    merged = merged.merge(meta, left_on="name", right_on="pair_id", how="left")

    for col in merged.columns:
        if col not in ("name", "pair_id", "variant_name", "variant_signature",
                       "label_tier", "ligand_key"):
            merged[col] = pd.to_numeric(merged[col], errors="ignore")

    merged["ligand_key"] = ligand_key
    return merged


def compute_strategy_h(df: pd.DataFrame) -> pd.DataFrame:
    """Add Strategy H gate columns and score."""
    df = df.copy()
    plddt_lig = pd.to_numeric(df["binary_plddt_ligand"], errors="coerce")
    hbond = pd.to_numeric(df["binary_hbond_distance"], errors="coerce")
    pocket = pd.to_numeric(df["binary_plddt_pocket"], errors="coerce")

    df["gate_plddt"] = plddt_lig >= GATE_PLDDT
    df["gate_hbond"] = hbond <= GATE_HBOND
    df["gate_both"] = df["gate_plddt"] & df["gate_hbond"]

    score = pd.Series(-np.inf, index=df.index)
    mask = df["gate_both"] & pocket.notna()
    score[mask] = pocket[mask].values
    df["score_H"] = score
    return df


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


# ── Load datasets ────────────────────────────────────────────────────────────

print("Loading datasets...")
datasets = OrderedDict()
for key, name in [("lca", "LCA"), ("glca", "GLCA"), ("lca3s", "LCA-3-S")]:
    df = load_dataset(key)
    df = compute_strategy_h(df)
    datasets[name] = df
    n_pos = int(df["label"].sum())
    n_pass = int(df["gate_both"].sum())
    n_pass_bind = int((df["gate_both"] & (df["label"] == 1)).sum())
    print(f"  {name}: {len(df)} designs, {n_pos} binders | "
          f"gate pass: {n_pass} ({n_pass_bind} binders)")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Gate Scatter Plot
# ═════════════════════════════════════════════════════════════════════════════

print("\nFigure 1: Gate scatter...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (name, df) in zip(axes, datasets.items()):
    hbond = pd.to_numeric(df["binary_hbond_distance"], errors="coerce")
    plddt = pd.to_numeric(df["binary_plddt_ligand"], errors="coerce")

    # Non-binders
    nb = df["label"] == 0
    ax.scatter(hbond[nb], plddt[nb], s=12, alpha=0.35, c=NONBINDER_COLOR,
               label="Non-binder", zorder=1, rasterized=True)
    # Binders
    bd = df["label"] == 1
    ax.scatter(hbond[bd], plddt[bd], s=35, alpha=0.85, c=BINDER_COLOR,
               edgecolors="k", linewidths=0.3, label="Binder", zorder=2)

    # Gate lines
    ax.axhline(GATE_PLDDT, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.axvline(GATE_HBOND, color="k", ls="--", lw=1.0, alpha=0.6)

    # Pass region rectangle (single shaded region for the intersection)
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, GATE_PLDDT), GATE_HBOND, 1.0 - GATE_PLDDT,
                      linewidth=1.5, edgecolor=ACCENT_COLOR,
                      facecolor=PASS_FILL, alpha=0.12, zorder=0)
    ax.add_patch(rect)

    # Annotation
    n_pass_bind = int((df["gate_both"] & bd).sum())
    n_total_bind = int(bd.sum())
    ax.text(0.03, 0.97, f"{n_pass_bind}/{n_total_bind} binders pass",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.set_xlabel("H-bond distance to water (\u00C5)")
    ax.set_ylabel("Ligand pLDDT")
    ax.set_ylim(0.3, 1.02)

    if name == "LCA":
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

# Gate label annotations on first panel
axes[0].text(GATE_HBOND + 0.15, 0.35, f"hbond > {GATE_HBOND}\u00C5\n(excluded)",
             fontsize=8, color="gray", style="italic")
axes[0].text(0.5, GATE_PLDDT - 0.04, f"pLDDT < {GATE_PLDDT}\n(excluded)",
             fontsize=8, color="gray", style="italic", ha="center")

fig.tight_layout()
fig.savefig(OUT_DIR / "strategy_h_gate_scatter.png")
plt.close(fig)
print("  Saved strategy_h_gate_scatter.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Filter Funnel
# ═════════════════════════════════════════════════════════════════════════════

print("Figure 2: Filter funnel...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, df) in zip(axes, datasets.items()):
    n_total = len(df)
    n_bind_total = int(df["label"].sum())

    # Stage counts: (total, binders) at each stage
    stages = []

    # Stage 1: All
    stages.append(("All designs", n_total, n_bind_total))

    # Stage 2: Pass pLDDT gate
    pass1 = df[df["gate_plddt"]]
    stages.append(("pLDDT \u2265 0.80", len(pass1), int(pass1["label"].sum())))

    # Stage 3: Pass both gates
    pass2 = df[df["gate_both"]]
    stages.append((f"+ hbond \u2264 {GATE_HBOND}\u00C5", len(pass2), int(pass2["label"].sum())))

    # Stage 4: Top 100
    top = df.nlargest(min(TOP_N, len(pass2)), "score_H")
    top = top[np.isfinite(top["score_H"])]
    n_top = len(top)
    n_top_bind = int(top["label"].sum())
    stages.append((f"Top {n_top}", n_top, n_top_bind))

    y_positions = np.arange(len(stages))[::-1]
    max_width = n_total

    for i, (label, n, n_b) in enumerate(stages):
        y = y_positions[i]
        n_nb = n - n_b
        frac = n_b / n * 100 if n > 0 else 0

        # Non-binder bar
        ax.barh(y, n_nb, height=0.6, color=NONBINDER_COLOR, alpha=0.4,
                left=0, edgecolor="none")
        # Binder bar
        ax.barh(y, n_b, height=0.6, color=BINDER_COLOR, alpha=0.8,
                left=n_nb, edgecolor="none")

        # Annotation
        ax.text(max_width * 1.02, y,
                f"{n_b} binders / {n} total ({frac:.0f}%)",
                va="center", ha="left", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in stages], fontsize=10)
    ax.set_xlabel("Number of designs")
    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max_width * 1.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if name == "LCA":
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=BINDER_COLOR, alpha=0.8, label="Binder"),
            Patch(facecolor=NONBINDER_COLOR, alpha=0.4, label="Non-binder"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
                  framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_DIR / "strategy_h_funnel.png")
plt.close(fig)
print("  Saved strategy_h_funnel.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Enrichment Curve
# ═════════════════════════════════════════════════════════════════════════════

print("Figure 3: Enrichment curve...")
fig, ax = plt.subplots(figsize=(6, 5))

# Random baseline
ax.plot([0, 100], [0, 100], ls="--", color="gray", alpha=0.5, lw=1,
        label="Random")

for name, df in datasets.items():
    color = DS_COLORS[name]
    fracs, recall = enrichment_curve(df, "score_H")
    ax.plot(fracs, recall, color=color, lw=2.0, label=name)

    # Top-100 marker
    frac_100 = TOP_N / len(df) * 100
    n_cap, n_tot = binders_at_topN(df, "score_H", TOP_N)
    recall_100 = n_cap / n_tot * 100

    ax.axvline(frac_100, color=color, ls=":", lw=0.8, alpha=0.5)
    ax.plot(frac_100, recall_100, "o", color=color, ms=7, zorder=5)
    ax.annotate(f"{recall_100:.0f}%",
                xy=(frac_100, recall_100),
                xytext=(frac_100 + 3, recall_100 - 5),
                fontsize=10, fontweight="bold", color=color)

ax.set_xlabel("Fraction of pool selected (%)")
ax.set_ylabel("Binders captured (%)")
ax.set_title("Strategy H Enrichment", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", framealpha=0.9)
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / "strategy_h_enrichment.png")
plt.close(fig)
print("  Saved strategy_h_enrichment.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: Pocket pLDDT Distribution (gated only)
# ═════════════════════════════════════════════════════════════════════════════

print("Figure 4: Pocket pLDDT distribution...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (name, df) in zip(axes, datasets.items()):
    gated = df[df["gate_both"]].copy()
    pocket = pd.to_numeric(gated["binary_plddt_pocket"], errors="coerce")
    gated = gated[pocket.notna()]
    pocket = pocket[pocket.notna()]

    binders = gated["label"] == 1
    nonbinders = gated["label"] == 0

    # KDE plots
    bins = np.linspace(pocket.min() - 0.01, pocket.max() + 0.01, 40)
    ax.hist(pocket[nonbinders], bins=bins, density=True, alpha=0.35,
            color=NONBINDER_COLOR, label="Non-binder", edgecolor="none")
    ax.hist(pocket[binders], bins=bins, density=True, alpha=0.5,
            color=BINDER_COLOR, label="Binder", edgecolor="none")

    # Top-100 cutoff line
    sorted_pocket = pocket.sort_values(ascending=False)
    n_take = min(TOP_N, len(sorted_pocket))
    if n_take > 0 and n_take < len(sorted_pocket):
        cutoff = sorted_pocket.iloc[n_take - 1]
        ax.axvline(cutoff, color=ACCENT_COLOR, ls="--", lw=2.0, zorder=5)

        # Shade selected region
        ylims = ax.get_ylim()
        ax.axvspan(cutoff, pocket.max() + 0.01, alpha=0.08,
                   color=ACCENT_COLOR, zorder=0)

        # Annotation
        n_cap, n_tot = binders_at_topN(df, "score_H", TOP_N)
        ax.text(0.97, 0.95,
                f"Top {n_take}: {n_cap}/{n_tot} binders",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85))

    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.set_xlabel("Pocket pLDDT (ranking metric)")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if name == "LCA":
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / "strategy_h_pocket_dist.png")
plt.close(fig)
print("  Saved strategy_h_pocket_dist.png")


# ═════════════════════════════════════════════════════════════════════════════
# CSV Output: LCA Top 100 MD Candidates
# ═════════════════════════════════════════════════════════════════════════════

print("\nGenerating MD candidate CSV...")
df_lca = datasets["LCA"]

# Top 100 by Strategy H score
valid = np.isfinite(df_lca["score_H"])
top100 = df_lca[valid].nlargest(TOP_N, "score_H").copy()
top100["rank"] = range(1, len(top100) + 1)

# Build PDB path column
BOLTZ_BASE = "output_lca_binary"
top100["pdb_path"] = top100["name"].apply(
    lambda x: f"{BOLTZ_BASE}/boltz_results_{x}/predictions/{x}/{x}_model_0.pdb"
)

# Select output columns
out_cols = [
    "name", "rank", "variant_name", "variant_signature",
    "label", "label_tier",
    "binary_plddt_ligand", "binary_hbond_distance", "binary_plddt_pocket",
    "binary_boltz_score", "binary_complex_iplddt", "binary_iptm",
    "pdb_path",
]
out = top100[out_cols].rename(columns={"name": "pair_id"})
out = out.sort_values("rank")

csv_path = RESULTS_DIR / "md_candidates_lca_top100.csv"
out.to_csv(csv_path, index=False, float_format="%.4f")

n_bind = int(out["label"].sum())
n_nonbind = len(out) - n_bind
print(f"  Saved {csv_path.name}: {len(out)} designs ({n_bind} binders, {n_nonbind} non-binders)")

print("\nDone.")
