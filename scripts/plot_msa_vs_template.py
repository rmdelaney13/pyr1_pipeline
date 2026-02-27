#!/usr/bin/env python3
"""
Generate figures comparing MSA vs template Boltz2 binary predictions.

Reads the paired_comparison.csv produced by deep_compare_msa_vs_template.py
and generates publication-quality figures.

Usage:
    python scripts/plot_msa_vs_template.py \
        --csv ml_modelling/analysis/boltz_LCA/msa_vs_template/paired_comparison.csv \
        --out-dir ml_modelling/analysis/boltz_LCA/msa_vs_template

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
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

MSA_COLOR = '#2196F3'
TMPL_COLOR = '#FF9800'
BINDER_COLOR = '#E53935'
NONBINDER_COLOR = '#90A4AE'
ENSEMBLE_COLOR = '#9C27B0'


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_paired_csv(path):
    """Load paired_comparison.csv → list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k == 'name':
                    parsed[k] = v
                elif k == 'is_binder':
                    parsed[k] = int(v) == 1
                else:
                    try:
                        parsed[k] = float(v) if v else None
                    except ValueError:
                        parsed[k] = None
            rows.append(parsed)
    return rows


# ═══════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════

def roc_auc_oriented(labels, scores):
    """ROC AUC, auto-oriented so AUC >= 0.5. Returns (auc, flipped)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None, False
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, False
    tp, fp, auc = 0, 0, 0.0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
    auc /= (n_pos * n_neg)
    if auc < 0.5:
        return 1 - auc, True
    return auc, False


def compute_roc_curve(labels, scores):
    """Full ROC curve, auto-oriented. Returns (fprs, tprs, auc, flipped)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None, None, None, False
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, None, False
    fprs, tprs = [0.0], [0.0]
    tp, fp, auc = 0, 0, 0.0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)
    auc /= (n_pos * n_neg)
    flipped = False
    if auc < 0.5:
        fprs = [1 - f for f in fprs]
        tprs = [1 - t for t in tprs]
        auc = 1 - auc
        flipped = True
    return np.array(fprs), np.array(tprs), auc, flipped


def bootstrap_auc(labels, scores, n_boot=2000, seed=42):
    """Bootstrap 95% CI for oriented AUC."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if len(pairs) < 10:
        return None, None, None
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(pairs), size=len(pairs))
        boot_s = [pairs[i][0] for i in idx]
        boot_l = [pairs[i][1] for i in idx]
        a, _ = roc_auc_oriented(boot_l, boot_s)
        if a is not None:
            aucs.append(a)
    if len(aucs) < 100:
        return None, None, None
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ═══════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

METRICS = [
    ('binary_iptm', 'ipTM'),
    ('binary_complex_plddt', 'Complex pLDDT'),
    ('binary_complex_iplddt', 'Interface pLDDT'),
    ('binary_plddt_protein', 'Protein pLDDT'),
    ('binary_plddt_ligand', 'Ligand pLDDT'),
    ('binary_complex_pde', 'Complex PDE'),
    ('binary_complex_ipde', 'Interface PDE'),
    ('binary_hbond_distance', 'H-bond dist'),
    ('binary_hbond_angle', 'H-bond angle'),
    ('binary_affinity_probability_binary', 'P(binder)'),
    ('binary_affinity_pred_value', 'Affinity pIC50'),
]


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: BOOTSTRAP AUC BAR CHART WITH ERROR BARS
# ═══════════════════════════════════════════════════════════════════

def fig_bootstrap_auc(rows, out_dir):
    """Side-by-side AUC bars with bootstrap 95% CIs."""
    labels_arr = [r['is_binder'] for r in rows]

    metric_labels, msa_aucs, msa_los, msa_his = [], [], [], []
    tmpl_aucs, tmpl_los, tmpl_his = [], [], []

    for key, label in METRICS:
        msa_scores = [r.get(f'msa_{key}') for r in rows]
        tmpl_scores = [r.get(f'tmpl_{key}') for r in rows]

        m_mean, m_lo, m_hi = bootstrap_auc(labels_arr, msa_scores)
        t_mean, t_lo, t_hi = bootstrap_auc(labels_arr, tmpl_scores)

        if m_mean is None or t_mean is None:
            continue

        metric_labels.append(label)
        msa_aucs.append(m_mean)
        msa_los.append(m_mean - m_lo)
        msa_his.append(m_hi - m_mean)
        tmpl_aucs.append(t_mean)
        tmpl_los.append(t_mean - t_lo)
        tmpl_his.append(t_hi - t_mean)

    # Sort by MSA AUC descending
    idx = np.argsort([-a for a in msa_aucs])
    metric_labels = [metric_labels[i] for i in idx]
    msa_aucs = [msa_aucs[i] for i in idx]
    msa_los = [msa_los[i] for i in idx]
    msa_his = [msa_his[i] for i in idx]
    tmpl_aucs = [tmpl_aucs[i] for i in idx]
    tmpl_los = [tmpl_los[i] for i in idx]
    tmpl_his = [tmpl_his[i] for i in idx]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metric_labels))
    w = 0.35
    ax.bar(x - w/2, msa_aucs, w, label='MSA', color=MSA_COLOR, alpha=0.85,
           yerr=[msa_los, msa_his], capsize=3, error_kw={'linewidth': 0.8, 'color': '#333'})
    ax.bar(x + w/2, tmpl_aucs, w, label='Template', color=TMPL_COLOR, alpha=0.85,
           yerr=[tmpl_los, tmpl_his], capsize=3, error_kw={'linewidth': 0.8, 'color': '#333'})
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('ROC AUC')
    ax.set_ylim(0.42, 0.82)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('LCA: MSA vs Template — Per-Metric AUC (bootstrap 95% CI)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig1_bootstrap_auc.png')
    fig.savefig(out_dir / 'fig1_bootstrap_auc.pdf')
    plt.close(fig)
    print(f"  Saved fig1_bootstrap_auc")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: ROC CURVES (6-PANEL)
# ═══════════════════════════════════════════════════════════════════

def fig_roc_curves(rows, out_dir):
    """2x3 grid of ROC curves for key metrics, MSA vs Template."""
    key_metrics = [
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_complex_iplddt', 'Interface pLDDT'),
        ('binary_iptm', 'ipTM'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_hbond_angle', 'H-bond angle'),
    ]

    labels_arr = [r['is_binder'] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(key_metrics):
        ax = axes[idx]
        for prefix, source_label, color, lw in [
            ('msa_', 'MSA', MSA_COLOR, 2.0),
            ('tmpl_', 'Template', TMPL_COLOR, 2.0),
        ]:
            scores = [r.get(f'{prefix}{key}') for r in rows]
            fprs, tprs, auc, flipped = compute_roc_curve(labels_arr, scores)
            if fprs is not None:
                flip_note = " (inv)" if flipped else ""
                ax.plot(fprs, tprs, color=color, linewidth=lw,
                        label=f'{source_label}: {auc:.3f}{flip_note}')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(label, fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

    fig.suptitle('LCA: MSA vs Template — ROC Comparison', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig2_roc_curves.png')
    fig.savefig(out_dir / 'fig2_roc_curves.pdf')
    plt.close(fig)
    print(f"  Saved fig2_roc_curves")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: PAIRED SCATTER (MSA VALUE VS TEMPLATE VALUE)
# ═══════════════════════════════════════════════════════════════════

def fig_paired_scatter(rows, out_dir):
    """2x3 scatter: each point is a variant, x=MSA value, y=Template value."""
    scatter_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_plddt_ligand', 'Ligand pLDDT'),
        ('binary_hbond_angle', 'H-bond angle'),
    ]

    binders = [r for r in rows if r['is_binder']]
    nonbinders = [r for r in rows if not r['is_binder']]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(scatter_metrics):
        ax = axes[idx]

        for subset, color, alpha, size, zorder, cat_label in [
            (nonbinders, NONBINDER_COLOR, 0.25, 10, 1, f'Non-binder (n={len(nonbinders)})'),
            (binders, BINDER_COLOR, 0.85, 30, 2, f'Binder (n={len(binders)})'),
        ]:
            xs, ys = [], []
            for r in subset:
                mv = r.get(f'msa_{key}')
                tv = r.get(f'tmpl_{key}')
                if mv is not None and tv is not None:
                    xs.append(mv)
                    ys.append(tv)
            ax.scatter(xs, ys, s=size, alpha=alpha, color=color,
                       edgecolors='none', zorder=zorder, label=cat_label)

        # Diagonal
        lims = [ax.get_xlim(), ax.get_ylim()]
        lo = min(lims[0][0], lims[1][0])
        hi = max(lims[0][1], lims[1][1])
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=0.8)

        # Pearson r
        xs_all = [r.get(f'msa_{key}') for r in rows]
        ys_all = [r.get(f'tmpl_{key}') for r in rows]
        valid = [(x, y) for x, y in zip(xs_all, ys_all) if x is not None and y is not None]
        if len(valid) > 3:
            r_val = np.corrcoef([v[0] for v in valid], [v[1] for v in valid])[0, 1]
            ax.text(0.05, 0.95, f'r = {r_val:.3f}', transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel(f'MSA {label}')
        ax.set_ylabel(f'Template {label}')
        ax.set_title(label, fontsize=11)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9)

    fig.suptitle('LCA: Paired Predictions — Same Variant, MSA vs Template', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig3_paired_scatter.png')
    fig.savefig(out_dir / 'fig3_paired_scatter.pdf')
    plt.close(fig)
    print(f"  Saved fig3_paired_scatter")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: DISTRIBUTION COMPARISON (HISTOGRAMS)
# ═══════════════════════════════════════════════════════════════════

def fig_distributions(rows, out_dir):
    """Side-by-side histograms: MSA (left) vs Template (right), binder/non-binder."""
    dist_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_complex_iplddt', 'Interface pLDDT'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_hbond_angle', 'H-bond angle'),
        ('binary_affinity_probability_binary', 'P(binder)'),
    ]

    binders = [r for r in rows if r['is_binder']]
    nonbinders = [r for r in rows if not r['is_binder']]

    fig, axes = plt.subplots(len(dist_metrics), 2, figsize=(11, 3 * len(dist_metrics)))

    for row_idx, (key, label) in enumerate(dist_metrics):
        for col_idx, (prefix, source_label) in enumerate([('msa_', 'MSA'), ('tmpl_', 'Template')]):
            ax = axes[row_idx, col_idx]
            col_key = f'{prefix}{key}'

            b_vals = [r[col_key] for r in binders if r.get(col_key) is not None]
            nb_vals = [r[col_key] for r in nonbinders if r.get(col_key) is not None]

            if not b_vals or not nb_vals:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            all_vals = b_vals + nb_vals
            bins = np.linspace(min(all_vals), max(all_vals), 35)

            ax.hist(nb_vals, bins=bins, alpha=0.55, color=NONBINDER_COLOR,
                    label=f'NB (n={len(nb_vals)})', density=True)
            ax.hist(b_vals, bins=bins, alpha=0.7, color=BINDER_COLOR,
                    label=f'B (n={len(b_vals)})', density=True)

            auc, flipped = roc_auc_oriented(
                [True]*len(b_vals) + [False]*len(nb_vals),
                b_vals + nb_vals)
            flip_note = " inv" if flipped else ""
            auc_str = f"AUC={auc:.3f}{flip_note}" if auc else ""

            ax.set_title(f'{source_label} {label} ({auc_str})', fontsize=10)
            ax.legend(fontsize=7, loc='upper right')
            if col_idx == 0:
                ax.set_ylabel('Density')

    fig.suptitle('LCA: Binder vs Non-binder Distributions — MSA vs Template', fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig4_distributions.png')
    fig.savefig(out_dir / 'fig4_distributions.pdf')
    plt.close(fig)
    print(f"  Saved fig4_distributions")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: DELTA VIOLIN (TEMPLATE - MSA) SPLIT BY BINDER STATUS
# ═══════════════════════════════════════════════════════════════════

def fig_delta_violins(rows, out_dir):
    """Violin plots of (Template - MSA) deltas for binders vs non-binders."""
    delta_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_complex_iplddt', 'Interface pLDDT'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_hbond_angle', 'H-bond angle'),
        ('binary_affinity_probability_binary', 'P(binder)'),
    ]

    binders = [r for r in rows if r['is_binder']]
    nonbinders = [r for r in rows if not r['is_binder']]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(delta_metrics):
        ax = axes[idx]
        delta_key = f'delta_{key}'

        b_deltas = [r[delta_key] for r in binders if r.get(delta_key) is not None]
        nb_deltas = [r[delta_key] for r in nonbinders if r.get(delta_key) is not None]

        if not b_deltas or not nb_deltas:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        parts = ax.violinplot([nb_deltas, b_deltas], positions=[0, 1],
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(NONBINDER_COLOR if i == 0 else BINDER_COLOR)
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        # Jitter overlay
        rng = np.random.RandomState(42)
        for pos, vals, color, s in [(0, nb_deltas, NONBINDER_COLOR, 4),
                                     (1, b_deltas, BINDER_COLOR, 8)]:
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            alpha = 0.15 if pos == 0 else 0.6
            ax.scatter(pos + jitter, vals, s=s, alpha=alpha, color=color,
                       edgecolors='none', zorder=3)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-binder', 'Binder'])
        ax.set_ylabel(f'\u0394 {label}\n(Template \u2212 MSA)')
        ax.set_title(label, fontsize=11)

        # Annotate means
        b_mean = np.mean(b_deltas)
        nb_mean = np.mean(nb_deltas)
        ax.text(0.98, 0.95,
                f'B mean: {b_mean:+.4f}\nNB mean: {nb_mean:+.4f}',
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('LCA: Per-Variant Score Change (Template \u2212 MSA)', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig5_delta_violins.png')
    fig.savefig(out_dir / 'fig5_delta_violins.pdf')
    plt.close(fig)
    print(f"  Saved fig5_delta_violins")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: RANK AGREEMENT (SPEARMAN HORIZONTAL BAR)
# ═══════════════════════════════════════════════════════════════════

def fig_rank_agreement(rows, out_dir):
    """Horizontal bar chart of Spearman rank correlation between MSA and template."""
    labels_list, spearmans = [], []

    for key, label in METRICS:
        msa_vals, tmpl_vals = [], []
        for r in rows:
            mv = r.get(f'msa_{key}')
            tv = r.get(f'tmpl_{key}')
            if mv is not None and tv is not None:
                msa_vals.append(mv)
                tmpl_vals.append(tv)

        if len(msa_vals) < 10:
            continue

        # Spearman
        rx = np.argsort(np.argsort(msa_vals)).astype(float)
        ry = np.argsort(np.argsort(tmpl_vals)).astype(float)
        sp = np.corrcoef(rx, ry)[0, 1]

        labels_list.append(label)
        spearmans.append(sp)

    # Sort
    idx = np.argsort(spearmans)
    labels_list = [labels_list[i] for i in idx]
    spearmans = [spearmans[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(labels_list))
    colors = [MSA_COLOR if s >= 0.3 else (TMPL_COLOR if s >= 0.1 else BINDER_COLOR) for s in spearmans]
    ax.barh(y, spearmans, height=0.6, color=colors, alpha=0.8)

    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0.3, color='gray', linestyle=':', alpha=0.4)
    ax.text(0.32, len(labels_list) - 0.5, 'weak\nagreement', fontsize=7, color='gray', va='top')

    for i, sp in enumerate(spearmans):
        ax.text(sp + 0.01 if sp >= 0 else sp - 0.01, i, f'{sp:.3f}',
                va='center', ha='left' if sp >= 0 else 'right', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_list, fontsize=9)
    ax.set_xlabel('Spearman rank correlation (MSA vs Template)')
    ax.set_title('LCA: How Similarly Do MSA and Template Rank Variants?', fontsize=12)
    ax.set_xlim(-0.15, 0.55)

    ax.legend(handles=[
        Patch(facecolor=MSA_COLOR, label='r \u2265 0.3 (weak+)'),
        Patch(facecolor=TMPL_COLOR, label='0.1 \u2264 r < 0.3'),
        Patch(facecolor=BINDER_COLOR, label='r < 0.1 (no agreement)'),
    ], loc='lower right', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / 'fig6_rank_agreement.png')
    fig.savefig(out_dir / 'fig6_rank_agreement.pdf')
    plt.close(fig)
    print(f"  Saved fig6_rank_agreement")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7: SUMMARY PANEL (4-IN-1)
# ═══════════════════════════════════════════════════════════════════

def fig_summary_panel(rows, out_dir):
    """4-panel summary figure: AUC bars, best ROC, P(binder) scatter, delta violin."""
    labels_arr = [r['is_binder'] for r in rows]
    binders = [r for r in rows if r['is_binder']]
    nonbinders = [r for r in rows if not r['is_binder']]

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: AUC bar chart (simplified top 6)
    ax = fig.add_subplot(gs[0, 0])
    top_metrics = [
        ('binary_complex_plddt', 'Cpx pLDDT'),
        ('binary_complex_iplddt', 'Iface pLDDT'),
        ('binary_plddt_protein', 'Prot pLDDT'),
        ('binary_hbond_angle', 'H-bond angle'),
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_hbond_distance', 'H-bond dist'),
    ]
    msa_aucs, tmpl_aucs, labels_list = [], [], []
    for key, label in top_metrics:
        msa_a, _ = roc_auc_oriented(labels_arr, [r.get(f'msa_{key}') for r in rows])
        tmpl_a, _ = roc_auc_oriented(labels_arr, [r.get(f'tmpl_{key}') for r in rows])
        if msa_a and tmpl_a:
            labels_list.append(label)
            msa_aucs.append(msa_a)
            tmpl_aucs.append(tmpl_a)

    x = np.arange(len(labels_list))
    w = 0.35
    ax.bar(x - w/2, msa_aucs, w, label='MSA', color=MSA_COLOR, alpha=0.85)
    ax.bar(x + w/2, tmpl_aucs, w, label='Template', color=TMPL_COLOR, alpha=0.85)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('ROC AUC')
    ax.set_ylim(0.45, 0.78)
    ax.legend(fontsize=8)
    ax.set_title('A. Per-Metric AUC', fontsize=11, fontweight='bold')

    # Panel B: ROC for P(binder) and Interface pLDDT
    ax = fig.add_subplot(gs[0, 1])
    for key, label, ls in [
        ('binary_affinity_probability_binary', 'P(binder)', '-'),
        ('binary_complex_iplddt', 'Iface pLDDT', '--'),
    ]:
        for prefix, source, color in [('msa_', 'MSA', MSA_COLOR), ('tmpl_', 'Tmpl', TMPL_COLOR)]:
            scores = [r.get(f'{prefix}{key}') for r in rows]
            fprs, tprs, auc, flipped = compute_roc_curve(labels_arr, scores)
            if fprs is not None:
                flip_note = "*" if flipped else ""
                ax.plot(fprs, tprs, color=color, linestyle=ls, linewidth=1.8,
                        label=f'{source} {label}: {auc:.3f}{flip_note}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title('B. ROC: Key Metrics', fontsize=11, fontweight='bold')

    # Panel C: P(binder) paired scatter
    ax = fig.add_subplot(gs[1, 0])
    key = 'binary_affinity_probability_binary'
    for subset, color, alpha, size, zorder, cat_label in [
        (nonbinders, NONBINDER_COLOR, 0.25, 10, 1, 'Non-binder'),
        (binders, BINDER_COLOR, 0.85, 30, 2, 'Binder'),
    ]:
        xs = [r[f'msa_{key}'] for r in subset if r.get(f'msa_{key}') is not None and r.get(f'tmpl_{key}') is not None]
        ys = [r[f'tmpl_{key}'] for r in subset if r.get(f'msa_{key}') is not None and r.get(f'tmpl_{key}') is not None]
        ax.scatter(xs, ys, s=size, alpha=alpha, color=color, edgecolors='none',
                   zorder=zorder, label=cat_label)

    lims = [ax.get_xlim(), ax.get_ylim()]
    lo = min(lims[0][0], lims[1][0])
    hi = max(lims[0][1], lims[1][1])
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('MSA P(binder)')
    ax.set_ylabel('Template P(binder)')
    ax.legend(loc='upper left', fontsize=8)

    valid = [(r.get(f'msa_{key}'), r.get(f'tmpl_{key}')) for r in rows
             if r.get(f'msa_{key}') is not None and r.get(f'tmpl_{key}') is not None]
    if valid:
        r_val = np.corrcoef([v[0] for v in valid], [v[1] for v in valid])[0, 1]
        ax.text(0.95, 0.05, f'r = {r_val:.3f}', transform=ax.transAxes,
                fontsize=9, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_title('C. P(binder): MSA vs Template', fontsize=11, fontweight='bold')

    # Panel D: Delta violin for Interface pLDDT
    ax = fig.add_subplot(gs[1, 1])
    key = 'binary_complex_iplddt'
    delta_key = f'delta_{key}'
    b_deltas = [r[delta_key] for r in binders if r.get(delta_key) is not None]
    nb_deltas = [r[delta_key] for r in nonbinders if r.get(delta_key) is not None]

    if b_deltas and nb_deltas:
        parts = ax.violinplot([nb_deltas, b_deltas], positions=[0, 1],
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(NONBINDER_COLOR if i == 0 else BINDER_COLOR)
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        rng = np.random.RandomState(42)
        for pos, vals, color, s in [(0, nb_deltas, NONBINDER_COLOR, 4),
                                     (1, b_deltas, BINDER_COLOR, 10)]:
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            alpha = 0.15 if pos == 0 else 0.6
            ax.scatter(pos + jitter, vals, s=s, alpha=alpha, color=color,
                       edgecolors='none', zorder=3)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-binder', 'Binder'])
        ax.set_ylabel('\u0394 Interface pLDDT\n(Template \u2212 MSA)')

        b_mean = np.mean(b_deltas)
        nb_mean = np.mean(nb_deltas)
        ax.text(0.98, 0.95,
                f'B: {b_mean:+.4f}\nNB: {nb_mean:+.4f}',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title('D. Template Effect on Iface pLDDT', fontsize=11, fontweight='bold')

    fig.suptitle('LCA: MSA vs Template Boltz2 Binary Prediction Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(out_dir / 'fig7_summary_panel.png')
    fig.savefig(out_dir / 'fig7_summary_panel.pdf')
    plt.close(fig)
    print(f"  Saved fig7_summary_panel")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot MSA vs template comparison figures")
    parser.add_argument("--csv", required=True, help="paired_comparison.csv from deep_compare_msa_vs_template.py")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv}...")
    rows = load_paired_csv(args.csv)
    n_b = sum(1 for r in rows if r['is_binder'])
    n_nb = len(rows) - n_b
    print(f"  {len(rows)} variants ({n_b} binders, {n_nb} non-binders)")

    print(f"\nGenerating figures in {out_dir}/...\n")

    fig_bootstrap_auc(rows, out_dir)
    fig_roc_curves(rows, out_dir)
    fig_paired_scatter(rows, out_dir)
    fig_distributions(rows, out_dir)
    fig_delta_violins(rows, out_dir)
    fig_rank_agreement(rows, out_dir)
    fig_summary_panel(rows, out_dir)

    print(f"\nDone! 7 figures (PNG + PDF) saved to {out_dir}/")


if __name__ == "__main__":
    main()
