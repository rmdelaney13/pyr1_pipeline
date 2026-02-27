#!/usr/bin/env python3
"""
Compare MSA-based vs template-based Boltz2 predictions side-by-side.

Loads two result CSVs (MSA and template), merges with a shared labels CSV,
and compares per-metric AUC, Cohen's d, and P(binder) distributions.

Usage:
    python scripts/compare_msa_vs_template.py \
        --msa-results results_all_affinity.csv \
        --template-results boltz_lca_binary_template_results.csv \
        --labels boltz_lca_binary.csv \
        --ligand LCA \
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
except ImportError:
    plt = None


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
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


def merge(results, labels):
    merged = []
    for row in results:
        name = row['name']
        if name in labels:
            row['is_binder'] = labels[name]
            merged.append(row)
    return merged


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
    auc = auc / (n_pos * n_neg)

    if auc < 0.5:
        return 1 - auc, True
    return auc, False


def cohens_d(g1, g2):
    if len(g1) < 2 or len(g2) < 2:
        return None
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = math.sqrt((s1**2 + s2**2) / 2)
    if pooled < 1e-12:
        return None
    return (np.mean(g1) - np.mean(g2)) / pooled


def compute_roc_curve(labels, scores):
    """Full ROC curve, auto-oriented."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None, None, None
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, None

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

    auc = auc / (n_pos * n_neg)
    if auc < 0.5:
        fprs = [1 - f for f in fprs]
        tprs = [1 - t for t in tprs]
        auc = 1 - auc

    return np.array(fprs), np.array(tprs), auc


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

METRICS = [
    ('binary_iptm', 'ipTM'),
    ('binary_complex_plddt', 'Complex pLDDT'),
    ('binary_complex_iplddt', 'Interface pLDDT'),
    ('binary_plddt_protein', 'Protein pLDDT'),
    ('binary_plddt_pocket', 'Pocket pLDDT'),
    ('binary_plddt_ligand', 'Ligand pLDDT'),
    ('binary_complex_pde', 'Complex PDE'),
    ('binary_complex_ipde', 'Interface PDE'),
    ('binary_hbond_distance', 'H-bond dist'),
    ('binary_hbond_angle', 'H-bond angle'),
    ('binary_affinity_probability_binary', 'P(binder)'),
    ('binary_affinity_pred_value', 'Affinity pIC50'),
    ('binary_boltz_score', 'Boltz score'),
    ('binary_geometry_dist_score', 'Geom dist'),
    ('binary_geometry_ang_score', 'Geom angle'),
    ('binary_geometry_score', 'Geometry'),
    ('binary_total_score', 'Total score'),
]


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compare MSA vs template Boltz2 predictions")
    parser.add_argument("--msa-results", required=True, help="MSA-based results CSV")
    parser.add_argument("--template-results", required=True, help="Template-based results CSV")
    parser.add_argument("--labels", required=True, help="Labels CSV with pair_id + label")
    parser.add_argument("--ligand", default="LCA", help="Ligand name for titles")
    parser.add_argument("--out-dir", default=None, help="Output directory for figures")
    args = parser.parse_args()

    labels = load_labels_csv(args.labels)
    print(f"Labels: {sum(labels.values())} binders, {sum(1 for v in labels.values() if not v)} non-binders")

    msa_rows = merge(load_results_csv(args.msa_results), labels)
    tmpl_rows = merge(load_results_csv(args.template_results), labels)

    msa_b = sum(1 for r in msa_rows if r['is_binder'])
    msa_nb = len(msa_rows) - msa_b
    tmpl_b = sum(1 for r in tmpl_rows if r['is_binder'])
    tmpl_nb = len(tmpl_rows) - tmpl_b

    print(f"\nMSA results:      {len(msa_rows)} matched ({msa_b} binders, {msa_nb} non-binders)")
    print(f"Template results: {len(tmpl_rows)} matched ({tmpl_b} binders, {tmpl_nb} non-binders)")

    # Find shared predictions
    msa_names = {r['name'] for r in msa_rows}
    tmpl_names = {r['name'] for r in tmpl_rows}
    shared = msa_names & tmpl_names
    print(f"Shared predictions: {len(shared)}")

    # ── Per-metric comparison ──
    print(f"\n{'='*80}")
    print(f"  {args.ligand}: MSA vs Template — Per-Metric AUC Comparison")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<20} {'MSA AUC':>8} {'Tmpl AUC':>9} {'Delta':>7} {'MSA d':>7} {'Tmpl d':>7} {'Winner':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    comparison = []
    for key, label in METRICS:
        # MSA
        msa_labels = [r['is_binder'] for r in msa_rows if r.get(key) is not None]
        msa_scores = [r[key] for r in msa_rows if r.get(key) is not None]
        msa_auc, msa_flip = roc_auc_oriented(msa_labels, msa_scores)

        # Template
        tmpl_labels = [r['is_binder'] for r in tmpl_rows if r.get(key) is not None]
        tmpl_scores = [r[key] for r in tmpl_rows if r.get(key) is not None]
        tmpl_auc, tmpl_flip = roc_auc_oriented(tmpl_labels, tmpl_scores)

        if msa_auc is None or tmpl_auc is None:
            continue

        # Cohen's d
        msa_d = cohens_d(
            [r[key] for r in msa_rows if r['is_binder'] and r.get(key) is not None],
            [r[key] for r in msa_rows if not r['is_binder'] and r.get(key) is not None]
        )
        tmpl_d = cohens_d(
            [r[key] for r in tmpl_rows if r['is_binder'] and r.get(key) is not None],
            [r[key] for r in tmpl_rows if not r['is_binder'] and r.get(key) is not None]
        )

        delta = tmpl_auc - msa_auc
        winner = "MSA" if msa_auc > tmpl_auc + 0.01 else ("TMPL" if tmpl_auc > msa_auc + 0.01 else "~tie")

        msa_d_str = f"{msa_d:+.3f}" if msa_d is not None else "N/A"
        tmpl_d_str = f"{tmpl_d:+.3f}" if tmpl_d is not None else "N/A"

        print(f"  {label:<20} {msa_auc:>8.3f} {tmpl_auc:>9.3f} {delta:>+7.3f} {msa_d_str:>7} {tmpl_d_str:>7} {winner:>8}")

        comparison.append({
            'key': key, 'label': label,
            'msa_auc': msa_auc, 'tmpl_auc': tmpl_auc, 'delta': delta,
            'msa_d': msa_d, 'tmpl_d': tmpl_d, 'winner': winner,
        })

    # ── Summary ──
    msa_wins = sum(1 for c in comparison if c['winner'] == 'MSA')
    tmpl_wins = sum(1 for c in comparison if c['winner'] == 'TMPL')
    ties = sum(1 for c in comparison if c['winner'] == '~tie')

    print(f"\n  Summary: MSA wins {msa_wins}, Template wins {tmpl_wins}, ties {ties}")

    # Highlight key metrics
    print(f"\n  Key metrics:")
    for key_metric in ['P(binder)', 'Affinity pIC50', 'ipTM', 'H-bond dist', 'H-bond angle', 'Protein pLDDT']:
        for c in comparison:
            if c['label'] == key_metric:
                print(f"    {c['label']:<20} MSA={c['msa_auc']:.3f}  Tmpl={c['tmpl_auc']:.3f}  delta={c['delta']:+.3f}  {c['winner']}")

    # ── Paired comparison on shared predictions ──
    if shared:
        print(f"\n{'='*80}")
        print(f"  Paired comparison on {len(shared)} shared predictions")
        print(f"{'='*80}")

        msa_by_name = {r['name']: r for r in msa_rows}
        tmpl_by_name = {r['name']: r for r in tmpl_rows}

        for key, label in [('binary_affinity_probability_binary', 'P(binder)'),
                           ('binary_affinity_pred_value', 'Affinity pIC50'),
                           ('binary_iptm', 'ipTM'),
                           ('binary_hbond_distance', 'H-bond dist')]:
            msa_vals, tmpl_vals, is_binder_list = [], [], []
            for name in sorted(shared):
                m_val = msa_by_name[name].get(key)
                t_val = tmpl_by_name[name].get(key)
                if m_val is not None and t_val is not None:
                    msa_vals.append(m_val)
                    tmpl_vals.append(t_val)
                    is_binder_list.append(msa_by_name[name]['is_binder'])

            if not msa_vals:
                continue

            msa_arr = np.array(msa_vals)
            tmpl_arr = np.array(tmpl_vals)
            diff = tmpl_arr - msa_arr
            corr = np.corrcoef(msa_arr, tmpl_arr)[0, 1]

            print(f"\n  {label}:")
            print(f"    Correlation (MSA vs Tmpl): r={corr:.3f}")
            print(f"    Mean diff (Tmpl - MSA):    {diff.mean():+.4f} +/- {diff.std():.4f}")

            b_idx = [i for i, b in enumerate(is_binder_list) if b]
            nb_idx = [i for i, b in enumerate(is_binder_list) if not b]
            if b_idx:
                b_diff = diff[b_idx]
                print(f"    Binders diff:              {b_diff.mean():+.4f} +/- {b_diff.std():.4f}")
            if nb_idx:
                nb_diff = diff[nb_idx]
                print(f"    Non-binders diff:          {nb_diff.mean():+.4f} +/- {nb_diff.std():.4f}")

    # ── Figures ──
    if args.out_dir and plt is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: AUC comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        labels_list = [c['label'] for c in comparison]
        msa_aucs = [c['msa_auc'] for c in comparison]
        tmpl_aucs = [c['tmpl_auc'] for c in comparison]

        # Sort by MSA AUC
        sort_idx = np.argsort([-a for a in msa_aucs])
        labels_list = [labels_list[i] for i in sort_idx]
        msa_aucs = [msa_aucs[i] for i in sort_idx]
        tmpl_aucs = [tmpl_aucs[i] for i in sort_idx]

        x = np.arange(len(labels_list))
        w = 0.35
        ax.bar(x - w/2, msa_aucs, w, label='MSA', color='#2196F3', alpha=0.8)
        ax.bar(x + w/2, tmpl_aucs, w, label='Template', color='#FF9800', alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('ROC AUC')
        ax.set_ylim(0.4, 0.85)
        ax.legend()
        ax.set_title(f'{args.ligand}: MSA vs Template — Per-Metric AUC')
        fig.tight_layout()
        path = out_dir / f'fig_msa_vs_template_auc_{args.ligand.lower()}.png'
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"\n  Saved {path}")

        # Figure 2: ROC curves for key metrics (MSA vs Template)
        key_metrics = [
            ('binary_affinity_probability_binary', 'P(binder)'),
            ('binary_iptm', 'ipTM'),
            ('binary_hbond_distance', 'H-bond dist'),
            ('binary_plddt_pocket', 'Pocket pLDDT'),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        for idx, (key, label) in enumerate(key_metrics):
            ax = axes[idx]

            for source, rows, color, ls in [
                ('MSA', msa_rows, '#2196F3', '-'),
                ('Template', tmpl_rows, '#FF9800', '-'),
            ]:
                src_labels = [r['is_binder'] for r in rows if r.get(key) is not None]
                src_scores = [r[key] for r in rows if r.get(key) is not None]
                fprs, tprs, auc = compute_roc_curve(src_labels, src_scores)
                if fprs is not None:
                    ax.plot(fprs, tprs, color=color, linestyle=ls, linewidth=1.8,
                            label=f'{source}: {auc:.3f}')

            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title(label)
            ax.legend(loc='lower right', fontsize=9)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect('equal')

        fig.suptitle(f'{args.ligand}: MSA vs Template ROC Comparison', fontsize=13, y=1.02)
        fig.tight_layout()
        path = out_dir / f'fig_msa_vs_template_roc_{args.ligand.lower()}.png'
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"  Saved {path}")

        # Figure 3: P(binder) distribution comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (source, rows, title_suffix) in enumerate([
            ('MSA', msa_rows, 'MSA'),
            ('Template', tmpl_rows, 'Template'),
        ]):
            ax = axes[idx]
            b_vals = [r['binary_affinity_probability_binary'] for r in rows
                      if r['is_binder'] and r.get('binary_affinity_probability_binary') is not None]
            nb_vals = [r['binary_affinity_probability_binary'] for r in rows
                       if not r['is_binder'] and r.get('binary_affinity_probability_binary') is not None]

            if b_vals and nb_vals:
                bins = np.linspace(
                    min(min(b_vals), min(nb_vals)),
                    max(max(b_vals), max(nb_vals)),
                    30
                )
                ax.hist(nb_vals, bins=bins, alpha=0.6, color='#90A4AE',
                       label=f'Non-binder (n={len(nb_vals)})', density=True)
                ax.hist(b_vals, bins=bins, alpha=0.7, color='#E53935',
                       label=f'Binder (n={len(b_vals)})', density=True)

                auc, _ = roc_auc_oriented(
                    [r['is_binder'] for r in rows if r.get('binary_affinity_probability_binary') is not None],
                    [r['binary_affinity_probability_binary'] for r in rows if r.get('binary_affinity_probability_binary') is not None]
                )
                ax.set_title(f'{title_suffix} P(binder) — AUC={auc:.3f}' if auc else f'{title_suffix} P(binder)')
                ax.set_xlabel('P(binder)')
                ax.set_ylabel('Density')
                ax.legend(fontsize=9)

        fig.suptitle(f'{args.ligand}: P(binder) Distribution — MSA vs Template', fontsize=13)
        fig.tight_layout()
        path = out_dir / f'fig_msa_vs_template_pbinder_{args.ligand.lower()}.png'
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"  Saved {path}")

        # Figure 4: Paired scatter (MSA vs Template) for shared predictions
        if shared:
            msa_by_name = {r['name']: r for r in msa_rows}
            tmpl_by_name = {r['name']: r for r in tmpl_rows}

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()

            for idx, (key, label) in enumerate(key_metrics):
                ax = axes[idx]
                for name in sorted(shared):
                    m_val = msa_by_name[name].get(key)
                    t_val = tmpl_by_name[name].get(key)
                    if m_val is not None and t_val is not None:
                        is_b = msa_by_name[name]['is_binder']
                        color = '#E53935' if is_b else '#90A4AE'
                        alpha = 0.8 if is_b else 0.3
                        size = 25 if is_b else 8
                        ax.scatter(m_val, t_val, s=size, alpha=alpha, color=color, edgecolors='none')

                # Diagonal
                lims = [ax.get_xlim(), ax.get_ylim()]
                lo = min(lims[0][0], lims[1][0])
                hi = max(lims[0][1], lims[1][1])
                ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=0.8)
                ax.set_xlabel(f'MSA {label}')
                ax.set_ylabel(f'Template {label}')
                ax.set_title(label)

            # Add legend to first panel
            from matplotlib.patches import Patch
            axes[0].legend(handles=[
                Patch(facecolor='#E53935', label='Binder'),
                Patch(facecolor='#90A4AE', label='Non-binder'),
            ], loc='upper left', fontsize=8)

            fig.suptitle(f'{args.ligand}: Paired Predictions (MSA vs Template)', fontsize=13, y=1.02)
            fig.tight_layout()
            path = out_dir / f'fig_msa_vs_template_paired_{args.ligand.lower()}.png'
            fig.savefig(path, dpi=300)
            plt.close(fig)
            print(f"  Saved {path}")

    # ── Write comparison CSV ──
    if args.out_dir:
        out_path = Path(args.out_dir) / f'msa_vs_template_comparison_{args.ligand.lower()}.csv'
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'metric', 'label', 'msa_auc', 'tmpl_auc', 'delta',
                'msa_d', 'tmpl_d', 'winner'])
            w.writeheader()
            for c in comparison:
                c_out = dict(c)
                del c_out['key']
                c_out['msa_d'] = f"{c['msa_d']:+.4f}" if c['msa_d'] is not None else ""
                c_out['tmpl_d'] = f"{c['tmpl_d']:+.4f}" if c['tmpl_d'] is not None else ""
                c_out['msa_auc'] = f"{c['msa_auc']:.4f}"
                c_out['tmpl_auc'] = f"{c['tmpl_auc']:.4f}"
                c_out['delta'] = f"{c['delta']:+.4f}"
                w.writerow(c_out)
        print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
