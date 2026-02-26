#!/usr/bin/env python3
"""Publication figure: Boltz metric discrimination of binders vs non-binders.

4-panel figure:
  A) ROC curves: best combination vs top individual metrics
  B) Combined score distribution (binders vs non-binders)
  C) H-bond geometry scatter (distance vs angle, colored by binder)
  D) Enrichment curve (fraction of binders recovered in top-N ranked)

Usage:
    python scripts/plot_discrimination_figure.py results_binary_affinity.csv --out-dir figures/
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    """Returns (fpr_list, tpr_list, auc)."""
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

    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    return fpr_list, tpr_list, auc


def compute_combined_score(rows, metric_keys, signs):
    """Z-score combination with equal weights. Returns dict {name: score}."""
    stats = []
    for key in metric_keys:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if len(vals) < 2:
            return {}
        stats.append((np.mean(vals), np.std(vals)))

    scores = {}
    for row in rows:
        z_sum = 0.0
        complete = True
        for i, key in enumerate(metric_keys):
            v = row.get(key)
            if v is None:
                complete = False
                break
            mu, sigma = stats[i]
            if sigma < 1e-9:
                complete = False
                break
            z_sum += signs[i] * (v - mu) / sigma
        if complete:
            scores[row['name']] = z_sum / len(metric_keys)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Results CSV")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(args.csv)
    labels = [classify_binder(r['name']) for r in rows]
    binders = [r for r, l in zip(rows, labels) if l]
    non_binders = [r for r, l in zip(rows, labels) if not l]

    n_b = len(binders)
    n_nb = len(non_binders)
    print(f"Loaded {len(rows)} predictions ({n_b} binders, {n_nb} non-binders)")

    # Colors
    C_BIND = '#2196F3'
    C_NONB = '#9E9E9E'
    C_COMBO = '#E91E63'

    # ── Compute best combination score: ipLDDT + hb_dist + hb_ang + P(bind) ──
    combo_keys = [
        'binary_complex_iplddt',
        'binary_hbond_distance',
        'binary_hbond_angle',
        'binary_affinity_probability_binary',
    ]
    combo_signs = [+1, -1, -1, +1]
    combo_labels_short = ['ipLDDT', 'hb_dist', 'hb_ang', 'P(bind)']

    combo_scores = compute_combined_score(rows, combo_keys, combo_signs)

    # ── Create figure ──
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28)

    # ═══════════════════════════════════════════════
    # A) ROC curves
    # ═══════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])

    # Combined score ROC
    combo_scores_list = [combo_scores.get(r['name']) for r in rows]
    combo_labels_list = labels
    fpr, tpr, auc = compute_roc(combo_labels_list, combo_scores_list, higher_is_binder=True)
    if auc is not None:
        ax_a.plot(fpr, tpr, color=C_COMBO, linewidth=2.5,
                  label=f'Combined (AUC={auc:.3f})', zorder=5)

    # Individual metrics
    indiv_metrics = [
        ('binary_complex_iplddt', 'Interface pLDDT', True, '#4CAF50'),
        ('binary_hbond_distance', 'H-bond distance', False, '#FF9800'),
        ('binary_hbond_angle', 'H-bond angle', False, '#9C27B0'),
        ('binary_affinity_probability_binary', 'P(binder)', True, '#00BCD4'),
        ('binary_complex_plddt', 'Complex pLDDT', True, '#795548'),
        ('binary_iptm', 'ipTM', True, '#607D8B'),
    ]

    for key, label, higher, color in indiv_metrics:
        scores = [r.get(key) for r in rows]
        f, t, a = compute_roc(labels, scores, higher_is_binder=higher)
        if a is not None:
            ax_a.plot(f, t, color=color, linewidth=1.2, alpha=0.7,
                      label=f'{label} ({a:.3f})')

    ax_a.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax_a.set_xlabel('False Positive Rate')
    ax_a.set_ylabel('True Positive Rate')
    ax_a.set_title('A) ROC: Combined vs Individual Metrics', fontweight='bold')
    ax_a.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_aspect('equal')

    # ═══════════════════════════════════════════════
    # B) Combined score distribution
    # ═══════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])

    b_scores = [combo_scores[r['name']] for r in binders if r['name'] in combo_scores]
    nb_scores = [combo_scores[r['name']] for r in non_binders if r['name'] in combo_scores]

    if b_scores and nb_scores:
        all_s = b_scores + nb_scores
        bins = np.linspace(min(all_s) - 0.1, max(all_s) + 0.1, 35)

        ax_b.hist(nb_scores, bins=bins, alpha=0.6, color=C_NONB, density=True,
                  label=f'Non-binder (n={len(nb_scores)})', edgecolor='white', linewidth=0.3)
        ax_b.hist(b_scores, bins=bins, alpha=0.7, color=C_BIND, density=True,
                  label=f'Binder (n={len(b_scores)})', edgecolor='white', linewidth=0.3)

        # Cohen's d
        pooled_std = np.sqrt((np.std(b_scores)**2 + np.std(nb_scores)**2) / 2)
        if pooled_std > 0:
            d = (np.mean(b_scores) - np.mean(nb_scores)) / pooled_std
            ax_b.text(0.97, 0.95, f"d = {d:+.2f}\nAUC = {auc:.3f}",
                      transform=ax_b.transAxes, ha='right', va='top', fontsize=9,
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.9))

        # Median lines
        ax_b.axvline(np.median(b_scores), color=C_BIND, linestyle='--', linewidth=1.5, alpha=0.8)
        ax_b.axvline(np.median(nb_scores), color=C_NONB, linestyle='--', linewidth=1.5, alpha=0.8)

    ax_b.set_xlabel('Combined Score (z-score)')
    ax_b.set_ylabel('Density')
    ax_b.set_title('B) Combined Score Distribution', fontweight='bold')
    ax_b.legend(fontsize=8)

    # ═══════════════════════════════════════════════
    # C) H-bond geometry scatter
    # ═══════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])

    for group, color, label, marker, zorder in [
        (non_binders, C_NONB, 'Non-binder', 'o', 1),
        (binders, C_BIND, 'Binder', '^', 2),
    ]:
        x = [r.get('binary_hbond_distance') for r in group
             if r.get('binary_hbond_distance') is not None
             and r.get('binary_hbond_angle') is not None]
        y = [r.get('binary_hbond_angle') for r in group
             if r.get('binary_hbond_distance') is not None
             and r.get('binary_hbond_angle') is not None]
        ax_c.scatter(x, y, c=color, label=label, marker=marker,
                     alpha=0.5, s=20, edgecolors='none', zorder=zorder)

    # Ideal zone: distance ~2.7A, angle ~90.5° (from 3QN1 crystal structure)
    ideal_d, ideal_a = 2.7, 90.5
    ax_c.axvline(ideal_d, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax_c.axhline(ideal_a, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax_c.plot(ideal_d, ideal_a, 'r*', markersize=15, zorder=10, label='3QN1 reference (2.7A, 90.5°)')

    # Draw ideal zone ellipse
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((ideal_d, ideal_a), width=2*0.8, height=2*25,
                       fill=False, edgecolor='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax_c.add_patch(ellipse)

    ax_c.set_xlabel('H-bond Water Distance (A)')
    ax_c.set_ylabel('H-bond Water Angle (°)')
    ax_c.set_title('C) Water Network Geometry', fontweight='bold')
    ax_c.legend(fontsize=7, loc='upper right')
    ax_c.set_xlim(0, 15)
    ax_c.set_ylim(0, 180)

    # ═══════════════════════════════════════════════
    # D) Enrichment curve
    # ═══════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])

    # Rank by combined score
    scored_rows = [(combo_scores.get(r['name']), classify_binder(r['name']), r['name'])
                   for r in rows if r['name'] in combo_scores]
    scored_rows.sort(key=lambda x: -x[0])

    n_total = len(scored_rows)
    n_binders_total = sum(1 for _, l, _ in scored_rows if l)

    # Enrichment: cumulative fraction of binders found
    cum_binders = 0
    x_frac = [0.0]
    y_recall = [0.0]
    for i, (score, is_binder, name) in enumerate(scored_rows):
        if is_binder:
            cum_binders += 1
        x_frac.append((i + 1) / n_total)
        y_recall.append(cum_binders / n_binders_total)

    ax_d.plot(x_frac, y_recall, color=C_COMBO, linewidth=2, label='Combined score')

    # Also plot enrichment for individual metrics
    for key, label, higher, color in [
        ('binary_complex_iplddt', 'ipLDDT', True, '#4CAF50'),
        ('binary_affinity_probability_binary', 'P(binder)', True, '#00BCD4'),
        ('binary_complex_plddt', 'Complex pLDDT', True, '#795548'),
    ]:
        metric_rows = [(r.get(key), classify_binder(r['name']))
                       for r in rows if r.get(key) is not None]
        if not higher:
            metric_rows = [(-s, l) for s, l in metric_rows]
        metric_rows.sort(key=lambda x: -x[0])
        n_t = len(metric_rows)
        n_b_t = sum(1 for _, l in metric_rows if l)
        if n_b_t == 0:
            continue
        cum = 0
        xf = [0.0]
        yr = [0.0]
        for i, (s, l) in enumerate(metric_rows):
            if l:
                cum += 1
            xf.append((i + 1) / n_t)
            yr.append(cum / n_b_t)
        ax_d.plot(xf, yr, color=color, linewidth=1.2, alpha=0.7, label=label)

    # Random baseline
    ax_d.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8, label='Random')

    # Enrichment factor annotations
    for frac in [0.1, 0.2, 0.3]:
        idx = int(frac * n_total)
        if idx < len(scored_rows):
            recall_at_frac = sum(1 for s, l, n in scored_rows[:idx] if l) / n_binders_total
            ef = recall_at_frac / frac
            ax_d.annotate(f'EF={ef:.1f}x',
                          xy=(frac, recall_at_frac),
                          xytext=(frac + 0.05, recall_at_frac + 0.08),
                          fontsize=7, color=C_COMBO,
                          arrowprops=dict(arrowstyle='->', color=C_COMBO, lw=0.8))

    ax_d.set_xlabel('Fraction of Dataset Screened')
    ax_d.set_ylabel('Fraction of Binders Found (Recall)')
    ax_d.set_title('D) Enrichment Curve', fontweight='bold')
    ax_d.legend(fontsize=7, loc='lower right')
    ax_d.set_xlim(-0.02, 1.02)
    ax_d.set_ylim(-0.02, 1.02)

    # ── Save ──
    fig.suptitle(
        'Boltz2 Binder Discrimination: ipLDDT + H-bond Distance + H-bond Angle + P(binder)',
        fontsize=12, fontweight='bold', y=0.98)

    fig.savefig(out_dir / 'discrimination_figure.png', dpi=300, bbox_inches='tight')
    print(f"Saved {out_dir / 'discrimination_figure.png'}")

    fig.savefig(out_dir / 'discrimination_figure.pdf', bbox_inches='tight')
    print(f"Saved {out_dir / 'discrimination_figure.pdf'}")

    plt.close(fig)

    # Print enrichment stats
    print(f"\nEnrichment factors (combined score):")
    for pct in [5, 10, 15, 20, 25, 30]:
        idx = int(pct / 100 * n_total)
        if idx > 0 and idx <= len(scored_rows):
            found = sum(1 for s, l, n in scored_rows[:idx] if l)
            ef = (found / n_binders_total) / (pct / 100)
            print(f"  Top {pct}% ({idx} variants): {found}/{n_binders_total} binders found "
                  f"({100*found/n_binders_total:.0f}% recall), EF={ef:.2f}x")


if __name__ == "__main__":
    main()
