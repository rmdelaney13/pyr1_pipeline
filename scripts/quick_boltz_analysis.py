#!/usr/bin/env python3
"""Quick analysis of Boltz prediction results: binder vs non-binder comparison.

Includes per-metric Cohen's d, a z-score combined score, and ROC AUC.

Usage:
    python scripts/quick_boltz_analysis.py /scratch/alpine/ryde3462/boltz_lca/results_all.csv
"""

import sys
import csv
import numpy as np
from pathlib import Path

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
    """pair_3059+ = binder, below = non-binder."""
    num = int(name.split('_')[1])
    return num >= 3059


def summarize(values, label):
    """Print summary stats for a list of values."""
    vals = [v for v in values if v is not None]
    if not vals:
        print(f"  {label}: no data")
        return
    arr = np.array(vals)
    print(f"  {label}: n={len(arr)}, mean={arr.mean():.3f}, "
          f"median={np.median(arr):.3f}, std={arr.std():.3f}, "
          f"range=[{arr.min():.3f}, {arr.max():.3f}]")


def compute_roc_auc(labels, scores):
    """Compute ROC AUC without sklearn. Higher score = predicted binder."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    tp, fp = 0, 0
    auc = 0.0
    prev_score = None
    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            pass
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
        prev_score = score
    return auc / (n_pos * n_neg)


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_boltz_analysis.py <results.csv>")
        sys.exit(1)

    rows = load_results(sys.argv[1])
    print(f"Loaded {len(rows)} predictions\n")

    binders = [r for r in rows if classify_binder(r['name'])]
    non_binders = [r for r in rows if not classify_binder(r['name'])]

    print(f"Binders (pair_3059+): {len(binders)}")
    print(f"Non-binders (<pair_3059): {len(non_binders)}")
    print()

    # Key metrics: (column_key, display_label, sign)
    # sign=+1 means higher = more binder-like, sign=-1 means lower = more binder-like
    # sign=0 means not directional (excluded from combined z-score)
    metrics = [
        ('binary_iptm', 'ipTM', +1),
        # NOTE: binary_ligand_iptm excluded — identical to binary_iptm for 2-chain complexes
        ('binary_complex_plddt', 'Complex pLDDT', +1),
        ('binary_complex_iplddt', 'Interface pLDDT', +1),
        ('binary_plddt_protein', 'Protein pLDDT', +1),
        ('binary_plddt_ligand', 'Ligand pLDDT', +1),
        ('binary_complex_pde', 'Complex PDE', -1),
        ('binary_complex_ipde', 'Interface PDE', -1),
        ('binary_hbond_distance', 'H-bond water distance (A)', -1),
        ('binary_hbond_angle', 'H-bond angle (deg)', 0),  # not directional
        ('binary_affinity_probability_binary', 'P(binder)', +1),
        ('binary_affinity_pred_value', 'Affinity pIC50', 0),  # empirically inverted, exclude from z-score
        ('binary_boltz_score', 'Boltz Score (lig_pLDDT+P(bind))', +1),
        ('binary_geometry_dist_score', 'Geometry Dist Score (Gauss@2.7A)', +1),
        ('binary_geometry_ang_score', 'Geometry Ang Score (Gauss@109.5)', +1),
        ('binary_geometry_score', 'Geometry Score (combined)', +1),
        ('binary_total_score', 'Total Score (Boltz+Geom)', +1),
    ]

    # Ternary metrics if present
    has_ternary = any(r.get('ternary_iptm') is not None for r in rows)
    if has_ternary:
        metrics.extend([
            ('ternary_iptm', 'Ternary ipTM', +1),
            ('ternary_ligand_iptm', 'Ternary Ligand ipTM', +1),
            ('ternary_protein_iptm', 'Ternary Protein ipTM', +1),
            ('ternary_complex_plddt', 'Ternary Complex pLDDT', +1),
            ('ternary_complex_iplddt', 'Ternary Interface pLDDT', +1),
            ('ternary_hbond_distance', 'Ternary H-bond distance (A)', -1),
            ('ternary_hbond_angle', 'Ternary H-bond angle (deg)', 0),
            ('ternary_affinity_probability_binary', 'Ternary P(binder)', +1),
            ('ternary_affinity_pred_value', 'Ternary Affinity pIC50', +1),
            ('ternary_boltz_score', 'Ternary Boltz Score', +1),
            ('ternary_geometry_dist_score', 'Ternary Geometry Dist Score', +1),
            ('ternary_geometry_ang_score', 'Ternary Geometry Ang Score', +1),
            ('ternary_geometry_score', 'Ternary Geometry Score (combined)', +1),
            ('ternary_total_score', 'Ternary Total Score', +1),
            ('ligand_rmsd_binary_vs_ternary', 'Ligand RMSD binary vs ternary', 0),
        ])

    # ── Per-metric analysis ──
    effect_sizes = {}
    for key, label, sign in metrics:
        print(f"── {label} ({key}) ──")
        summarize([r.get(key) for r in binders], "Binders    ")
        summarize([r.get(key) for r in non_binders], "Non-binders")

        b_vals = [r.get(key) for r in binders if r.get(key) is not None]
        nb_vals = [r.get(key) for r in non_binders if r.get(key) is not None]
        if b_vals and nb_vals:
            b_mean = np.mean(b_vals)
            nb_mean = np.mean(nb_vals)
            diff = b_mean - nb_mean
            pooled_std = np.sqrt((np.std(b_vals)**2 + np.std(nb_vals)**2) / 2)
            if pooled_std > 0:
                d = diff / pooled_std
                effect_sizes[key] = d
                # ROC AUC for this metric
                labels = [classify_binder(r['name']) for r in rows]
                scores = [r.get(key) for r in rows]
                if sign == -1:
                    scores = [-s if s is not None else None for s in scores]
                auc = compute_roc_auc(labels, scores)
                auc_str = f", AUC={auc:.3f}" if auc is not None else ""
                print(f"  Cohen's d={d:+.3f}{auc_str}")
        print()

    # ── Effect size ranking ──
    print("=" * 60)
    print("EFFECT SIZE RANKING (|Cohen's d|, descending):")
    ranked = sorted(effect_sizes.items(), key=lambda x: abs(x[1]), reverse=True)
    for key, d in ranked:
        label = next(l for k, l, _ in metrics if k == key)
        direction = "binder > non-binder" if d > 0 else "binder < non-binder"
        print(f"  |d|={abs(d):.3f}  d={d:+.3f}  {label} ({direction})")
    print()

    # ── Combined z-score ──
    # Use metrics with |d| > 0.1 and known directionality (sign != 0)
    print("=" * 60)
    print("COMBINED Z-SCORE:")
    combo_metrics = [(k, l, s) for k, l, s in metrics
                     if s != 0 and k in effect_sizes and abs(effect_sizes[k]) > 0.1]
    if combo_metrics:
        print(f"  Using {len(combo_metrics)} metrics with |d|>0.1:")
        for k, l, s in combo_metrics:
            print(f"    {l}: sign={'+' if s > 0 else '-'}, d={effect_sizes[k]:+.3f}")
        print()

        # Compute z-scores across all rows
        for row in rows:
            row['_combined_z'] = 0.0
            row['_combined_n'] = 0

        for key, label, sign in combo_metrics:
            all_vals = [r.get(key) for r in rows]
            valid = [v for v in all_vals if v is not None]
            if len(valid) < 2:
                continue
            mu = np.mean(valid)
            sigma = np.std(valid)
            if sigma < 1e-9:
                continue
            for row in rows:
                v = row.get(key)
                if v is not None:
                    z = sign * (v - mu) / sigma
                    row['_combined_z'] += z
                    row['_combined_n'] += 1

        # Normalize by number of contributing metrics
        for row in rows:
            if row['_combined_n'] > 0:
                row['combined_score'] = row['_combined_z'] / row['_combined_n']
            else:
                row['combined_score'] = None
            del row['_combined_z']
            del row['_combined_n']

        b_combo = [r['combined_score'] for r in binders if r.get('combined_score') is not None]
        nb_combo = [r['combined_score'] for r in non_binders if r.get('combined_score') is not None]

        summarize(b_combo, "Binders    ")
        summarize(nb_combo, "Non-binders")

        if b_combo and nb_combo:
            d = (np.mean(b_combo) - np.mean(nb_combo)) / np.sqrt(
                (np.std(b_combo)**2 + np.std(nb_combo)**2) / 2)
            labels = [classify_binder(r['name']) for r in rows]
            scores = [r.get('combined_score') for r in rows]
            auc = compute_roc_auc(labels, scores)
            auc_str = f", AUC={auc:.3f}" if auc is not None else ""
            print(f"  Combined Cohen's d={d:+.3f}{auc_str}")
        print()

        # Classification at various thresholds
        print("  CLASSIFICATION (binder if combined_score >= threshold):")
        for thr in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            tp = sum(1 for r in binders if (r.get('combined_score') or -999) >= thr)
            fp = sum(1 for r in non_binders if (r.get('combined_score') or -999) >= thr)
            fn = sum(1 for r in binders if (r.get('combined_score') or -999) < thr)
            tn = sum(1 for r in non_binders if (r.get('combined_score') or -999) < thr)
            acc = (tp + tn) / max(tp + fp + fn + tn, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            print(f"    z >= {thr:+.2f}: acc={acc:.2f}, prec={prec:.2f}, "
                  f"recall={rec:.2f}  (TP={tp}, FP={fp}, FN={fn}, TN={tn})")
    print()

    # ── Top/bottom predictions ──
    print("=" * 60)
    sort_key = 'combined_score' if any(r.get('combined_score') is not None for r in rows) else 'binary_iptm'
    print(f"TOP 15 by {sort_key}:")
    sorted_rows = sorted(rows, key=lambda r: r.get(sort_key) or -999, reverse=True)
    for r in sorted_rows[:15]:
        binder = "BINDER" if classify_binder(r['name']) else "non-binder"
        combo = f"{r.get('combined_score', 0):.3f}" if r.get('combined_score') is not None else "N/A"
        pbind = f"{r.get('binary_affinity_probability_binary', 0):.3f}" if r.get('binary_affinity_probability_binary') is not None else "N/A"
        iptm = f"{r.get('binary_iptm', 0):.3f}" if r.get('binary_iptm') is not None else "N/A"
        print(f"  {r['name']:12s}  combined={combo:>6}  P(bind)={pbind:>6}  "
              f"ipTM={iptm:>6}  {binder}")

    print()
    print(f"BOTTOM 15 by {sort_key}:")
    for r in sorted_rows[-15:]:
        binder = "BINDER" if classify_binder(r['name']) else "non-binder"
        combo = f"{r.get('combined_score', 0):.3f}" if r.get('combined_score') is not None else "N/A"
        pbind = f"{r.get('binary_affinity_probability_binary', 0):.3f}" if r.get('binary_affinity_probability_binary') is not None else "N/A"
        iptm = f"{r.get('binary_iptm', 0):.3f}" if r.get('binary_iptm') is not None else "N/A"
        print(f"  {r['name']:12s}  combined={combo:>6}  P(bind)={pbind:>6}  "
              f"ipTM={iptm:>6}  {binder}")

    # H-bond quality check
    print()
    print("=" * 60)
    print("H-BOND WATER NETWORK CHECK (distance < 4A = good coordination):")
    for group_name, group in [("Binders", binders), ("Non-binders", non_binders)]:
        dists = [r.get('binary_hbond_distance') for r in group if r.get('binary_hbond_distance') is not None]
        if dists:
            good = sum(1 for d in dists if d < 4.0)
            print(f"  {group_name}: {good}/{len(dists)} ({100*good/len(dists):.0f}%) with distance < 4A")

    # Classification accuracy at various ipTM thresholds
    print()
    print("=" * 60)
    print("CLASSIFICATION (binder if metric >= threshold):")
    for metric_key, metric_label in [('binary_iptm', 'ipTM'), ('binary_affinity_probability_binary', 'P(binder)')]:
        vals = [r.get(metric_key) for r in rows if r.get(metric_key) is not None]
        if not vals:
            continue
        print(f"\n  {metric_label}:")
        thresholds = sorted(set(np.percentile(vals, [10, 25, 50, 75, 90]).tolist()))
        for threshold in thresholds:
            tp = sum(1 for r in binders if (r.get(metric_key) or 0) >= threshold)
            fp = sum(1 for r in non_binders if (r.get(metric_key) or 0) >= threshold)
            fn = sum(1 for r in binders if (r.get(metric_key) or 0) < threshold)
            tn = sum(1 for r in non_binders if (r.get(metric_key) or 0) < threshold)
            acc = (tp + tn) / max(tp + fp + fn + tn, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            print(f"    >= {threshold:.3f}: acc={acc:.2f}, prec={prec:.2f}, "
                  f"recall={rec:.2f}  (TP={tp}, FP={fp}, FN={fn}, TN={tn})")


if __name__ == "__main__":
    main()
