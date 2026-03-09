#!/usr/bin/env python3
"""Quick check: what geometry metrics did specific 'bad' designs get?"""
import csv
import sys
from pathlib import Path

bad_designs = {
    'ca': ['exp_r9_0012'],
    'cdca': ['exp_r10_0079', 'exp_r6_0036', 'exp_r7_0078', 'exp_r8_0100'],
}

sel_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/scratch/alpine/ryde3462/expansion/selection')

for lig, names in bad_designs.items():
    csv_path = sel_dir / f"selection_{lig}.csv"
    if not csv_path.exists():
        print(f"{lig}: CSV not found")
        continue

    print(f"\n{'='*70}")
    print(f"  {lig.upper()} - Known bad designs")
    print(f"{'='*70}")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Also collect stats for context
    all_lig_rmsd = []
    all_o_rmsd = []
    all_c_rmsd = []
    for row in all_rows:
        for col, lst in [('ligand_rmsd', all_lig_rmsd),
                          ('o_rmsd', all_o_rmsd),
                          ('c_rmsd', all_c_rmsd)]:
            v = row.get(col, '')
            if v:
                try:
                    lst.append(float(v))
                except ValueError:
                    pass

    print(f"  Pool stats (n={len(all_rows)}):")
    for label, lst in [('ligand_rmsd', all_lig_rmsd),
                        ('c_rmsd', all_c_rmsd),
                        ('o_rmsd', all_o_rmsd)]:
        if lst:
            s = sorted(lst)
            print(f"    {label:12s}: median={s[len(s)//2]:.3f}  "
                  f"p75={s[3*len(s)//4]:.3f}  p90={s[9*len(s)//10]:.3f}  "
                  f"max={s[-1]:.3f}")

    for name in names:
        matches = [r for r in all_rows if r.get('name', '') == name]
        if not matches:
            print(f"\n  {name}: NOT IN SELECTION (filtered out or not in top-300)")
            continue

        row = matches[0]
        print(f"\n  {name}:")
        for col in ['ligand_rmsd', 'c_rmsd', 'o_rmsd',
                     'max_dev', 'max_o_dev', 'max_dev_atom',
                     'planarity_ratio', 'oh_fingerprint', 'pose_cluster',
                     'binary_total_score', 'composite_zscore', 'tier']:
            val = row.get(col, 'N/A')
            print(f"    {col:25s}: {val}")
