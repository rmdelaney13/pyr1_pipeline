#!/usr/bin/env python3
"""Plot Boltz prediction results: binder vs non-binder comparison figures.

Usage:
    python scripts/plot_boltz_results.py results_binary_affinity.csv --out-dir figures/
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_results(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k == 'name':
                    continue
                try:
                    row[k] = float(v) if v else None
                except ValueError:
                    row[k] = None
            rows.append(row)
    return rows


def classify_binder(name):
    num = int(name.split('_')[1])
    return num >= 3059


def compute_roc(labels, scores, higher_is_binder=True):
    """Compute ROC curve points. Returns (fpr_list, tpr_list, auc)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return [], [], None
    if not higher_is_binder:
        pairs = [(-s, l) for s, l in pairs]
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [], [], None

    fpr_list = [0.0]
    tpr_list = [0.0]
    tp, fp = 0, 0
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        fpr_list.append(fp / n_neg)
        tpr_list.append(tp / n_pos)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    return fpr_list, tpr_list, auc


def main():
    parser = argparse.ArgumentParser(description="Plot Boltz analysis results")
    parser.add_argument("csv", help="Results CSV from analyze_boltz_output.py")
    parser.add_argument("--out-dir", default="figures", help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(args.csv)
    labels = [classify_binder(r['name']) for r in rows]
    binders = [r for r, l in zip(rows, labels) if l]
    non_binders = [r for r, l in zip(rows, labels) if not l]

    print(f"Loaded {len(rows)} predictions ({len(binders)} binders, {len(non_binders)} non-binders)")

    COLORS = {'binder': '#2196F3', 'non_binder': '#9E9E9E'}

    # ─── 1. Box plots of key metrics ───
    box_metrics = [
        ('binary_iptm', 'ipTM', True),
        ('binary_complex_iplddt', 'Interface pLDDT', True),
        ('binary_complex_plddt', 'Complex pLDDT', True),
        ('binary_affinity_probability_binary', 'P(binder)', True),
        ('binary_hbond_distance', 'H-bond Distance (A)', False),
        ('binary_complex_ipde', 'Interface PDE', False),
    ]

    # Check for ternary
    has_ternary = any(r.get('ternary_iptm') is not None for r in rows)
    if has_ternary:
        box_metrics.extend([
            ('ternary_iptm', 'Ternary ipTM', True),
            ('ternary_protein_iptm', 'Ternary Protein ipTM', True),
            ('ternary_affinity_probability_binary', 'Ternary P(binder)', True),
        ])

    n_metrics = len(box_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, (key, label, higher_better) in enumerate(box_metrics):
        ax = axes[idx]
        b_vals = [r.get(key) for r in binders if r.get(key) is not None]
        nb_vals = [r.get(key) for r in non_binders if r.get(key) is not None]

        if not b_vals and not nb_vals:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            [nb_vals, b_vals],
            labels=['Non-binder', 'Binder'],
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color='black', linewidth=1.5),
        )
        bp['boxes'][0].set_facecolor(COLORS['non_binder'])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(COLORS['binder'])
        bp['boxes'][1].set_alpha(0.7)

        # Overlay individual points (jittered)
        for i, (vals, color) in enumerate([(nb_vals, COLORS['non_binder']), (b_vals, COLORS['binder'])]):
            x = np.random.normal(i + 1, 0.06, size=len(vals))
            ax.scatter(x, vals, alpha=0.3, s=8, color=color, zorder=2)

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_ylabel(label, fontsize=8)

        # Cohen's d annotation
        if b_vals and nb_vals:
            pooled_std = np.sqrt((np.std(b_vals)**2 + np.std(nb_vals)**2) / 2)
            if pooled_std > 0:
                d = (np.mean(b_vals) - np.mean(nb_vals)) / pooled_std
                ax.text(0.95, 0.95, f"d={d:+.2f}", transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Boltz Metrics: Binder vs Non-binder', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'boxplots.png', dpi=200, bbox_inches='tight')
    print(f"  Saved {out_dir / 'boxplots.png'}")
    plt.close(fig)

    # ─── 2. ROC curves ───
    roc_metrics = [
        ('binary_affinity_probability_binary', 'P(binder)', True),
        ('binary_complex_plddt', 'Complex pLDDT', True),
        ('binary_complex_iplddt', 'Interface pLDDT', True),
        ('binary_iptm', 'ipTM', True),
        ('binary_hbond_distance', 'H-bond Distance', False),
        ('binary_complex_ipde', 'Interface PDE', False),
    ]
    if has_ternary:
        roc_metrics.extend([
            ('ternary_iptm', 'Ternary ipTM', True),
            ('ternary_protein_iptm', 'Ternary Protein ipTM', True),
            ('ternary_affinity_probability_binary', 'Ternary P(binder)', True),
        ])

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.tab10

    for i, (key, label, higher_better) in enumerate(roc_metrics):
        scores = [r.get(key) for r in rows]
        fpr, tpr, auc = compute_roc(labels, scores, higher_is_binder=higher_better)
        if auc is not None:
            ax.plot(fpr, tpr, color=cmap(i), linewidth=1.5,
                    label=f'{label} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves: Binder vs Non-binder', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_dir / 'roc_curves.png', dpi=200, bbox_inches='tight')
    print(f"  Saved {out_dir / 'roc_curves.png'}")
    plt.close(fig)

    # ─── 3. P(binder) vs ipTM scatter ───
    fig, ax = plt.subplots(figsize=(7, 6))

    for group, color, label, marker in [
        (non_binders, COLORS['non_binder'], 'Non-binder', 'o'),
        (binders, COLORS['binder'], 'Binder', '^'),
    ]:
        x = [r.get('binary_iptm') for r in group if r.get('binary_iptm') is not None and r.get('binary_affinity_probability_binary') is not None]
        y = [r.get('binary_affinity_probability_binary') for r in group if r.get('binary_iptm') is not None and r.get('binary_affinity_probability_binary') is not None]
        ax.scatter(x, y, c=color, label=label, marker=marker, alpha=0.5, s=25, edgecolors='none')

    ax.set_xlabel('ipTM', fontsize=11)
    ax.set_ylabel('P(binder)', fontsize=11)
    ax.set_title('ipTM vs P(binder)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / 'iptm_vs_pbinder.png', dpi=200, bbox_inches='tight')
    print(f"  Saved {out_dir / 'iptm_vs_pbinder.png'}")
    plt.close(fig)

    # ─── 4. P(binder) distribution ───
    fig, ax = plt.subplots(figsize=(7, 4.5))

    b_pbind = [r.get('binary_affinity_probability_binary') for r in binders
               if r.get('binary_affinity_probability_binary') is not None]
    nb_pbind = [r.get('binary_affinity_probability_binary') for r in non_binders
                if r.get('binary_affinity_probability_binary') is not None]

    if b_pbind and nb_pbind:
        bins = np.linspace(
            min(min(b_pbind), min(nb_pbind)),
            max(max(b_pbind), max(nb_pbind)),
            30)
        ax.hist(nb_pbind, bins=bins, alpha=0.6, color=COLORS['non_binder'],
                label=f'Non-binder (n={len(nb_pbind)})', density=True)
        ax.hist(b_pbind, bins=bins, alpha=0.6, color=COLORS['binder'],
                label=f'Binder (n={len(b_pbind)})', density=True)
        ax.set_xlabel('P(binder)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Boltz P(binder) Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / 'pbinder_distribution.png', dpi=200, bbox_inches='tight')
    print(f"  Saved {out_dir / 'pbinder_distribution.png'}")
    plt.close(fig)

    # ─── 5. H-bond distance distribution ───
    fig, ax = plt.subplots(figsize=(7, 4.5))

    b_hbond = [r.get('binary_hbond_distance') for r in binders
               if r.get('binary_hbond_distance') is not None]
    nb_hbond = [r.get('binary_hbond_distance') for r in non_binders
                if r.get('binary_hbond_distance') is not None]

    if b_hbond and nb_hbond:
        bins = np.linspace(0, max(max(b_hbond), max(nb_hbond)), 30)
        ax.hist(nb_hbond, bins=bins, alpha=0.6, color=COLORS['non_binder'],
                label=f'Non-binder (n={len(nb_hbond)})', density=True)
        ax.hist(b_hbond, bins=bins, alpha=0.6, color=COLORS['binder'],
                label=f'Binder (n={len(b_hbond)})', density=True)
        ax.axvline(4.0, color='red', linestyle='--', alpha=0.7, label='4A threshold')
        ax.set_xlabel('H-bond Water Distance (A)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Conserved Water H-bond Distance', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / 'hbond_distance_distribution.png', dpi=200, bbox_inches='tight')
    print(f"  Saved {out_dir / 'hbond_distance_distribution.png'}")
    plt.close(fig)

    # ─── 6. Effect size bar chart ───
    fig, ax = plt.subplots(figsize=(8, 5))

    all_metrics = box_metrics + [m for m in roc_metrics if m not in box_metrics]
    effect_data = []
    for key, label, higher_better in all_metrics:
        b_vals = [r.get(key) for r in binders if r.get(key) is not None]
        nb_vals = [r.get(key) for r in non_binders if r.get(key) is not None]
        if b_vals and nb_vals:
            pooled_std = np.sqrt((np.std(b_vals)**2 + np.std(nb_vals)**2) / 2)
            if pooled_std > 0:
                d = (np.mean(b_vals) - np.mean(nb_vals)) / pooled_std
                effect_data.append((label, d))

    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_effects = []
    for label, d in effect_data:
        if label not in seen:
            seen.add(label)
            unique_effects.append((label, d))

    unique_effects.sort(key=lambda x: abs(x[1]))
    labels_plot = [e[0] for e in unique_effects]
    d_vals = [e[1] for e in unique_effects]
    colors = [COLORS['binder'] if d > 0 else '#E57373' for d in d_vals]

    ax.barh(range(len(labels_plot)), d_vals, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(range(len(labels_plot)))
    ax.set_yticklabels(labels_plot, fontsize=8)
    ax.set_xlabel("Cohen's d (positive = binder higher)", fontsize=10)
    ax.set_title("Effect Size: Binder vs Non-binder", fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(-0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    legend_elements = [
        Patch(facecolor=COLORS['binder'], alpha=0.8, label='Binder higher'),
        Patch(facecolor='#E57373', alpha=0.8, label='Non-binder higher'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(out_dir / 'effect_sizes.png', dpi=200, bbox_inches='tight')
    print(f"  Saved {out_dir / 'effect_sizes.png'}")
    plt.close(fig)

    # ─── 7. Ternary-specific plots if available ───
    if has_ternary:
        # Binary vs ternary ipTM
        fig, ax = plt.subplots(figsize=(7, 6))
        for group, color, label, marker in [
            (non_binders, COLORS['non_binder'], 'Non-binder', 'o'),
            (binders, COLORS['binder'], 'Binder', '^'),
        ]:
            x = [r.get('binary_iptm') for r in group
                 if r.get('binary_iptm') is not None and r.get('ternary_iptm') is not None]
            y = [r.get('ternary_iptm') for r in group
                 if r.get('binary_iptm') is not None and r.get('ternary_iptm') is not None]
            if x and y:
                ax.scatter(x, y, c=color, label=label, marker=marker, alpha=0.5, s=25, edgecolors='none')

        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', alpha=0.3)
        ax.set_xlabel('Binary ipTM', fontsize=11)
        ax.set_ylabel('Ternary ipTM', fontsize=11)
        ax.set_title('Binary vs Ternary ipTM', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / 'binary_vs_ternary_iptm.png', dpi=200, bbox_inches='tight')
        print(f"  Saved {out_dir / 'binary_vs_ternary_iptm.png'}")
        plt.close(fig)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
