#!/usr/bin/env python3
"""Quick analysis of Boltz prediction results: binder vs non-binder comparison.

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
            # Convert numeric fields
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

    # Key metrics to compare
    metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_ligand_iptm', 'Ligand ipTM'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_complex_iplddt', 'Interface pLDDT'),
        ('binary_plddt_protein', 'Protein pLDDT'),
        ('binary_plddt_ligand', 'Ligand pLDDT'),
        ('binary_complex_pde', 'Complex PDE (lower=better)'),
        ('binary_complex_ipde', 'Interface PDE (lower=better)'),
        ('binary_hbond_distance', 'H-bond water distance (A)'),
        ('binary_hbond_angle', 'H-bond angle (deg)'),
    ]

    # Also check ternary if present
    has_ternary = any(r.get('ternary_iptm') is not None for r in rows)
    if has_ternary:
        metrics.extend([
            ('ternary_iptm', 'Ternary ipTM'),
            ('ternary_ligand_iptm', 'Ternary Ligand ipTM'),
            ('ternary_protein_iptm', 'Ternary Protein ipTM'),
            ('ternary_complex_plddt', 'Ternary Complex pLDDT'),
            ('ternary_hbond_distance', 'Ternary H-bond distance (A)'),
            ('ternary_hbond_angle', 'Ternary H-bond angle (deg)'),
            ('ligand_rmsd_binary_vs_ternary', 'Ligand RMSD binary vs ternary'),
        ])

    for key, label in metrics:
        print(f"── {label} ({key}) ──")
        summarize([r.get(key) for r in binders], "Binders    ")
        summarize([r.get(key) for r in non_binders], "Non-binders")

        # Quick separation check
        b_vals = [r.get(key) for r in binders if r.get(key) is not None]
        nb_vals = [r.get(key) for r in non_binders if r.get(key) is not None]
        if b_vals and nb_vals:
            b_mean = np.mean(b_vals)
            nb_mean = np.mean(nb_vals)
            diff = b_mean - nb_mean
            pooled_std = np.sqrt((np.std(b_vals)**2 + np.std(nb_vals)**2) / 2)
            if pooled_std > 0:
                effect_size = diff / pooled_std
                print(f"  Delta(binder-nonbinder)={diff:+.3f}, effect_size(Cohen's d)={effect_size:+.3f}")
        print()

    # Top/bottom predictions
    print("=" * 60)
    print("TOP 10 by binary ipTM:")
    sorted_iptm = sorted(rows, key=lambda r: r.get('binary_iptm') or 0, reverse=True)
    for r in sorted_iptm[:10]:
        binder = "BINDER" if classify_binder(r['name']) else "non-binder"
        print(f"  {r['name']:12s}  ipTM={r.get('binary_iptm', 'N/A'):>6}  "
              f"pLDDT={r.get('binary_complex_plddt', 'N/A'):>6}  "
              f"hbond_dist={r.get('binary_hbond_distance', 'N/A'):>6}  "
              f"{binder}")

    print()
    print("BOTTOM 10 by binary ipTM:")
    for r in sorted_iptm[-10:]:
        binder = "BINDER" if classify_binder(r['name']) else "non-binder"
        print(f"  {r['name']:12s}  ipTM={r.get('binary_iptm', 'N/A'):>6}  "
              f"pLDDT={r.get('binary_complex_plddt', 'N/A'):>6}  "
              f"hbond_dist={r.get('binary_hbond_distance', 'N/A'):>6}  "
              f"{binder}")

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
    print("CLASSIFICATION (binder if ipTM >= threshold):")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        tp = sum(1 for r in binders if (r.get('binary_iptm') or 0) >= threshold)
        fp = sum(1 for r in non_binders if (r.get('binary_iptm') or 0) >= threshold)
        fn = sum(1 for r in binders if (r.get('binary_iptm') or 0) < threshold)
        tn = sum(1 for r in non_binders if (r.get('binary_iptm') or 0) < threshold)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"  ipTM >= {threshold:.2f}: acc={acc:.2f}, prec={prec:.2f}, "
              f"recall={rec:.2f}  (TP={tp}, FP={fp}, FN={fn}, TN={tn})")


if __name__ == "__main__":
    main()
