#!/usr/bin/env python3
"""
Generate figures for Boltz2 binary scoring analysis.

Produces:
  1. ROC curves (per-ligand + pooled) for top metrics
  2. AUC heatmap (metrics × ligands)
  3. Violin plots of binder vs non-binder distributions
  4. 2D scatter of best metric pair
  5. Enrichment curves

Usage:
    python scripts/plot_boltz_scoring.py \
        --data LCA results_lca.csv labels_lca.csv \
        --data GLCA results_glca.csv labels_glca.csv \
        --data LCA-3-S results_lca3s.csv labels_lca3s.csv \
        --out-dir ml_modelling/analysis/boltz_LCA/scoring

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# ── Style ──
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

LIGAND_COLORS = {
    'LCA': '#2196F3',
    'GLCA': '#4CAF50',
    'LCA3S': '#FF9800',
    'LCA-3-S': '#FF9800',
    'POOLED': '#9C27B0',
}
BINDER_COLORS = {'Binder': '#E53935', 'Non-binder': '#90A4AE'}


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING (same as analyze_boltz_scoring.py)
# ═══════════════════════════════════════════════════════════════════

def load_results_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k == 'name':
                    parsed[k] = v
                    continue
                try:
                    parsed[k] = float(v) if v else None
                except ValueError:
                    parsed[k] = None
            rows.append(parsed)
    return rows


def load_labels_csv(path):
    labels = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row.get('pair_id', row.get('name', '')).strip()
            label = float(row.get('label', 0))
            labels[pair_id] = label >= 0.5
    return labels


def merge_results_labels(results, labels):
    merged = []
    for row in results:
        name = row['name']
        if name not in labels:
            continue
        row['is_binder'] = labels[name]
        merged.append(row)
    return merged


# ═══════════════════════════════════════════════════════════════════
# ROC COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_roc_curve(labels, scores):
    """Compute full ROC curve (FPR, TPR arrays) + AUC.
    Orients so AUC >= 0.5 (flips if needed)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None, None, None, False
    pairs.sort(key=lambda x: -x[0])

    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, None, False

    fprs, tprs = [0.0], [0.0]
    tp, fp = 0, 0
    auc = 0.0

    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)

    auc = auc / (n_pos * n_neg)

    # Flip if AUC < 0.5
    flipped = False
    if auc < 0.5:
        fprs = [1 - f for f in fprs]
        tprs = [1 - t for t in tprs]
        auc = 1 - auc
        flipped = True

    return np.array(fprs), np.array(tprs), auc, flipped


def compute_enrichment_curve(labels, scores):
    """Compute enrichment curve: (fraction_screened, fraction_binders_found)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None, None

    # Orient so AUC >= 0.5
    pairs.sort(key=lambda x: -x[0])
    n_total = len(pairs)
    n_pos = sum(1 for _, l in pairs if l)
    if n_pos == 0:
        return None, None

    # Check if we need to flip
    # Quick AUC calc
    tp, fp, auc = 0, 0, 0.0
    n_neg = n_total - n_pos
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
    if n_neg > 0 and auc / (n_pos * n_neg) < 0.5:
        pairs.reverse()

    xs, ys = [0.0], [0.0]
    found = 0
    for i, (_, label) in enumerate(pairs, 1):
        if label:
            found += 1
        xs.append(i / n_total)
        ys.append(found / n_pos)

    return np.array(xs), np.array(ys)


# ═══════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

# Top metrics to highlight (from analysis results)
TOP_METRICS = [
    ('binary_hbond_distance', 'H-bond distance (\u00C5)', -1),
    ('binary_hbond_angle', 'H-bond angle (\u00B0)', -1),
    ('binary_plddt_protein', 'Protein pLDDT', -1),
    ('binary_complex_pde', 'Complex PDE', -1),
    ('binary_affinity_probability_binary', 'P(binder)', +1),
    ('binary_affinity_pred_value', 'Affinity pIC50', +1),
]

ALL_METRICS = [
    ('binary_iptm', 'ipTM'),
    ('binary_complex_plddt', 'Complex pLDDT'),
    ('binary_complex_iplddt', 'Interface pLDDT'),
    ('binary_plddt_protein', 'Protein pLDDT'),
    ('binary_plddt_ligand', 'Ligand pLDDT'),
    ('binary_complex_pde', 'Complex PDE'),
    ('binary_complex_ipde', 'Interface PDE'),
    ('binary_hbond_distance', 'H-bond distance'),
    ('binary_hbond_angle', 'H-bond angle'),
    ('binary_affinity_probability_binary', 'P(binder)'),
    ('binary_affinity_pred_value', 'Affinity pIC50'),
    ('binary_boltz_score', 'Boltz score'),
    ('binary_geometry_dist_score', 'Geom dist'),
    ('binary_geometry_ang_score', 'Geom angle'),
    ('binary_geometry_score', 'Geometry'),
    ('binary_total_score', 'Total score'),
]

METRIC_COLORS = [
    '#E53935', '#1E88E5', '#43A047', '#FB8C00',
    '#8E24AA', '#00ACC1', '#D81B60', '#FDD835',
]


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: ROC CURVES
# ═══════════════════════════════════════════════════════════════════

def plot_roc_curves(all_datasets, out_dir):
    """Multi-panel ROC curves: one panel per ligand + pooled."""
    ligands = list(all_datasets.keys())
    n_panels = len(ligands) + 1  # +1 for pooled
    ncols = min(n_panels, 2)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Pool all data
    pooled = []
    for rows in all_datasets.values():
        pooled.extend(rows)

    datasets_with_pooled = list(all_datasets.items()) + [('POOLED', pooled)]

    for idx, (ligand, rows) in enumerate(datasets_with_pooled):
        ax = axes[idx]
        n_b = sum(1 for r in rows if r['is_binder'])
        n_nb = len(rows) - n_b

        labels_arr = [r['is_binder'] for r in rows]

        for i, (key, label, _) in enumerate(TOP_METRICS):
            scores = [r.get(key) for r in rows]
            fprs, tprs, auc, flipped = compute_roc_curve(labels_arr, scores)
            if fprs is None:
                continue
            color = METRIC_COLORS[i % len(METRIC_COLORS)]
            flip_note = " (inv)" if flipped else ""
            ax.plot(fprs, tprs, color=color, linewidth=1.5,
                    label=f'{label}: {auc:.3f}{flip_note}')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{ligand} ({n_b}B / {n_nb}NB)')
        ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

    # Hide unused axes
    for idx in range(len(datasets_with_pooled), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('ROC Curves: Template-Based Boltz2 Binary Predictions', fontsize=13, y=1.02)
    fig.tight_layout()
    path = out_dir / 'fig1_roc_curves.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: AUC HEATMAP
# ═══════════════════════════════════════════════════════════════════

def plot_auc_heatmap(all_datasets, out_dir):
    """Heatmap of AUC values: metrics (rows) × ligands (columns)."""
    ligands = list(all_datasets.keys())

    # Pool all data
    pooled = []
    for rows in all_datasets.values():
        pooled.extend(rows)
    cols = ligands + ['POOLED']
    all_data = dict(all_datasets)
    all_data['POOLED'] = pooled

    metric_labels = []
    auc_matrix = []

    for key, label in ALL_METRICS:
        row_aucs = []
        for col in cols:
            rows = all_data[col]
            labels_arr = [r['is_binder'] for r in rows]
            scores = [r.get(key) for r in rows]
            fprs, tprs, auc, _ = compute_roc_curve(labels_arr, scores)
            row_aucs.append(auc if auc is not None else 0.5)
        metric_labels.append(label)
        auc_matrix.append(row_aucs)

    auc_arr = np.array(auc_matrix)

    # Sort by pooled AUC
    pooled_idx = cols.index('POOLED')
    sort_idx = np.argsort(-auc_arr[:, pooled_idx])
    auc_arr = auc_arr[sort_idx]
    metric_labels = [metric_labels[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(4 + 0.8 * len(cols), 0.45 * len(metric_labels) + 1.5))

    im = ax.imshow(auc_arr, cmap='RdYlGn', vmin=0.4, vmax=0.85, aspect='auto')

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha='right')
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)

    # Annotate cells
    for i in range(len(metric_labels)):
        for j in range(len(cols)):
            val = auc_arr[i, j]
            color = 'white' if val < 0.5 or val > 0.75 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='ROC AUC')
    ax.set_title('AUC Heatmap: Template-Based Binary Predictions')
    fig.tight_layout()
    path = out_dir / 'fig2_auc_heatmap.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: VIOLIN / STRIP PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_violin_distributions(all_datasets, out_dir):
    """Violin plots of binder vs non-binder for top universal metrics."""
    # Use the recommended universal metrics
    plot_metrics = [
        ('binary_hbond_distance', 'H-bond distance (\u00C5)'),
        ('binary_hbond_angle', 'H-bond angle (\u00B0)'),
        ('binary_plddt_protein', 'Protein pLDDT'),
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_affinity_pred_value', 'Affinity pIC50'),
        ('binary_complex_pde', 'Complex PDE'),
    ]

    ligands = list(all_datasets.keys())
    n_metrics = len(plot_metrics)
    n_ligands = len(ligands)

    fig, axes = plt.subplots(n_metrics, n_ligands, figsize=(3.5 * n_ligands, 2.5 * n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    if n_ligands == 1:
        axes = axes.reshape(-1, 1)

    for row, (key, label) in enumerate(plot_metrics):
        for col, ligand in enumerate(ligands):
            ax = axes[row, col]
            rows = all_datasets[ligand]

            b_vals = [r[key] for r in rows if r['is_binder'] and r.get(key) is not None]
            nb_vals = [r[key] for r in rows if not r['is_binder'] and r.get(key) is not None]

            if not b_vals or not nb_vals:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            # Violin plot
            parts = ax.violinplot([nb_vals, b_vals], positions=[0, 1],
                                  showmedians=True, showextrema=False)
            for i, pc in enumerate(parts['bodies']):
                color = BINDER_COLORS['Non-binder'] if i == 0 else BINDER_COLORS['Binder']
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts['cmedians'].set_color('black')

            # Strip (jitter) overlay
            for i, (vals, color) in enumerate([(nb_vals, BINDER_COLORS['Non-binder']),
                                                (b_vals, BINDER_COLORS['Binder'])]):
                jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
                ax.scatter(i + jitter, vals, s=6, alpha=0.4, color=color,
                          edgecolors='none', zorder=3)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['NB', 'B'], fontsize=8)

            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 0:
                n_b = len(b_vals)
                n_nb = len(nb_vals)
                ax.set_title(f'{ligand} ({n_b}B/{n_nb}NB)', fontsize=10)

    fig.suptitle('Binder vs Non-binder Distributions', fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / 'fig3_violin_distributions.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: 2D SCATTER (BEST PAIR)
# ═══════════════════════════════════════════════════════════════════

def plot_2d_scatter(all_datasets, out_dir):
    """2D scatter of best metric pair, colored by binder status."""
    # Best universal pair: H-bond distance + Protein pLDDT
    x_key, x_label = 'binary_hbond_distance', 'H-bond distance (\u00C5)'
    y_key, y_label = 'binary_plddt_protein', 'Protein pLDDT'

    ligands = list(all_datasets.keys())
    ncols = min(len(ligands), 3)
    nrows = math.ceil(len(ligands) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if len(ligands) == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx, ligand in enumerate(ligands):
        ax = axes[idx]
        rows = all_datasets[ligand]

        for is_binder, label, color, alpha, zorder, size in [
            (False, 'Non-binder', BINDER_COLORS['Non-binder'], 0.3, 1, 12),
            (True, 'Binder', BINDER_COLORS['Binder'], 0.8, 2, 25),
        ]:
            subset = [r for r in rows if r['is_binder'] == is_binder
                      and r.get(x_key) is not None and r.get(y_key) is not None]
            xs = [r[x_key] for r in subset]
            ys = [r[y_key] for r in subset]
            ax.scatter(xs, ys, s=size, alpha=alpha, color=color,
                      edgecolors='none', zorder=zorder, label=label)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        n_b = sum(1 for r in rows if r['is_binder'])
        ax.set_title(f'{ligand} ({n_b} binders)')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    for idx in range(len(ligands), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Best Universal Metric Pair', fontsize=13, y=1.02)
    fig.tight_layout()
    path = out_dir / 'fig4_2d_scatter.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: ENRICHMENT CURVES
# ═══════════════════════════════════════════════════════════════════

def plot_enrichment_curves(all_datasets, out_dir):
    """Enrichment curves for top metrics, per-ligand + pooled."""
    ligands = list(all_datasets.keys())

    pooled = []
    for rows in all_datasets.values():
        pooled.extend(rows)

    n_panels = len(ligands) + 1
    ncols = min(n_panels, 2)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    datasets_with_pooled = list(all_datasets.items()) + [('POOLED', pooled)]

    for idx, (ligand, rows) in enumerate(datasets_with_pooled):
        ax = axes[idx]
        n_b = sum(1 for r in rows if r['is_binder'])
        labels_arr = [r['is_binder'] for r in rows]

        for i, (key, label, _) in enumerate(TOP_METRICS[:4]):
            scores = [r.get(key) for r in rows]
            xs, ys = compute_enrichment_curve(labels_arr, scores)
            if xs is None:
                continue
            color = METRIC_COLORS[i % len(METRIC_COLORS)]
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label)

        # Random baseline
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8, label='Random')

        ax.set_xlabel('Fraction screened')
        ax.set_ylabel('Fraction binders found')
        ax.set_title(f'{ligand} ({n_b} binders)')
        ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    for idx in range(len(datasets_with_pooled), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Enrichment Curves: Template-Based Binary Predictions', fontsize=13, y=1.02)
    fig.tight_layout()
    path = out_dir / 'fig5_enrichment_curves.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: METRIC CONSISTENCY BAR CHART
# ═══════════════════════════════════════════════════════════════════

def plot_consistency_bars(all_datasets, out_dir):
    """Grouped bar chart: AUC per metric per ligand."""
    ligands = list(all_datasets.keys())

    # Pool
    pooled = []
    for rows in all_datasets.values():
        pooled.extend(rows)

    cols = ligands + ['POOLED']
    all_data = dict(all_datasets)
    all_data['POOLED'] = pooled

    metric_keys = [k for k, _ in ALL_METRICS]
    metric_labels = [l for _, l in ALL_METRICS]

    aucs = {}  # {metric_label: {ligand: auc}}
    for key, label in ALL_METRICS:
        aucs[label] = {}
        for col in cols:
            rows = all_data[col]
            labels_arr = [r['is_binder'] for r in rows]
            scores = [r.get(key) for r in rows]
            _, _, auc, _ = compute_roc_curve(labels_arr, scores)
            aucs[label][col] = auc if auc is not None else 0.5

    # Sort by pooled AUC
    sorted_labels = sorted(metric_labels, key=lambda l: aucs[l].get('POOLED', 0.5), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(sorted_labels))
    width = 0.8 / len(cols)

    for i, col in enumerate(cols):
        color = LIGAND_COLORS.get(col, '#666666')
        vals = [aucs[label][col] for label in sorted_labels]
        offset = (i - len(cols) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=col, color=color, alpha=0.8)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('ROC AUC')
    ax.set_ylim(0.4, 0.9)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Metric AUC Across Ligands')
    fig.tight_layout()
    path = out_dir / 'fig6_consistency_bars.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate scoring analysis figures for Boltz2 binary predictions")
    parser.add_argument("--data", nargs=3, action="append", required=True,
                        metavar=("LIGAND", "RESULTS_CSV", "LABELS_CSV"),
                        help="Ligand name, Boltz results CSV, and input labels CSV")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for figures")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets
    all_datasets = {}
    for ligand, results_path, labels_path in args.data:
        print(f"Loading {ligand}...")
        results = load_results_csv(results_path)
        labels = load_labels_csv(labels_path)
        merged = merge_results_labels(results, labels)
        n_b = sum(1 for r in merged if r['is_binder'])
        n_nb = len(merged) - n_b
        print(f"  {len(merged)} rows ({n_b} binders, {n_nb} non-binders)")
        all_datasets[ligand] = merged

    print(f"\nGenerating figures in {out_dir}/...\n")

    plot_roc_curves(all_datasets, out_dir)
    plot_auc_heatmap(all_datasets, out_dir)
    plot_violin_distributions(all_datasets, out_dir)
    plot_2d_scatter(all_datasets, out_dir)
    plot_enrichment_curves(all_datasets, out_dir)
    plot_consistency_bars(all_datasets, out_dir)

    print(f"\nDone! {6} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
