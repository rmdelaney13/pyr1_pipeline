#!/usr/bin/env python3
"""Extract PDBs at different shape RMSD tiers for visual calibration.

Picks ~3 designs per tier (low/medium/high/worst) per ligand so you can
inspect them in PyMOL and decide on a geometry cutoff.

Usage:
    python scripts/geometry_diagnostic.py \
        --selection-dir /scratch/alpine/ryde3462/expansion/selection \
        --expansion-root /scratch/alpine/ryde3462/expansion/ligandmpnn \
        --ref-ligand-dir ligands \
        --ligands ca cdca dca
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_pocket_evolution import find_pdb_for_design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--expansion-root", required=True)
    parser.add_argument("--ref-ligand-dir", required=True)
    parser.add_argument("--ligands", nargs='+', default=['ca', 'cdca', 'dca'])
    args = parser.parse_args()

    sel_dir = Path(args.selection_dir)
    out_base = sel_dir / "geometry_diagnostic"
    out_base.mkdir(exist_ok=True)

    for lig in args.ligands:
        csv_path = sel_dir / f"selection_{lig}.csv"
        if not csv_path.exists():
            print(f"  {lig}: no selection CSV found, skipping")
            continue

        # Read all rows with ligand_rmsd (all-atom, ligand-to-ligand)
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                sr = row.get('ligand_rmsd', row.get('shape_rmsd', ''))
                if sr and sr != '':
                    try:
                        rows.append((row['name'], float(sr), row))
                    except ValueError:
                        pass

        if not rows:
            print(f"  {lig}: no shape_rmsd values in CSV")
            continue

        rows.sort(key=lambda x: x[1])
        n = len(rows)

        # Define tiers: best, 25th percentile, median, 75th, worst
        tier_indices = {
            'best': [0, 1, 2],
            'p25': [max(0, n//4 - 1), n//4, min(n-1, n//4 + 1)],
            'median': [max(0, n//2 - 1), n//2, min(n-1, n//2 + 1)],
            'p75': [max(0, 3*n//4 - 1), 3*n//4, min(n-1, 3*n//4 + 1)],
            'worst': [max(0, n-3), max(0, n-2), n-1],
        }

        lig_dir = out_base / lig
        lig_dir.mkdir(exist_ok=True)

        # Copy reference ligand
        ref_lig = Path(args.ref_ligand_dir)
        ref_pdbs = list(ref_lig.glob(f"{lig}_*.pdb"))
        ref_name = None
        if ref_pdbs:
            shutil.copy2(ref_pdbs[0], lig_dir / ref_pdbs[0].name)
            ref_name = ref_pdbs[0].name

        print(f"\n  {lig.upper()}: {n} designs with ligand_rmsd")
        print(f"    Range: {rows[0][1]:.3f} - {rows[-1][1]:.3f} A")

        pymol_lines = ["from pymol import cmd\nimport os\n"]
        pymol_lines.append(f"d = os.path.dirname(os.path.abspath(__file__))\n")

        if ref_name:
            pymol_lines.append(f'cmd.load(os.path.join(d, "{ref_name}"), "reference")')
            pymol_lines.append('cmd.color("white", "reference")')
            pymol_lines.append('cmd.show("sticks", "reference")\n')

        colors = {
            'best': 'green',
            'p25': 'cyan',
            'median': 'yellow',
            'p75': 'orange',
            'worst': 'red',
        }

        for tier, indices in tier_indices.items():
            # Deduplicate indices
            seen = set()
            unique_idx = []
            for idx in indices:
                if idx not in seen and idx < n:
                    seen.add(idx)
                    unique_idx.append(idx)

            print(f"\n    {tier}:")
            pymol_lines.append(f"\n# --- {tier} ---")

            for idx in unique_idx:
                name, shape_rmsd, row = rows[idx]
                o_rmsd = row.get('o_rmsd', '?')
                c_rmsd = row.get('c_rmsd', '?')
                score = row.get('composite_zscore', row.get('binary_total_score', '?'))
                fp = row.get('oh_fingerprint', '')

                pdb_path = find_pdb_for_design(name, lig, args.expansion_root)
                if pdb_path is None:
                    print(f"      {name}: rmsd={shape_rmsd:.3f} (PDB not found)")
                    continue

                dest_name = f"{tier}_{shape_rmsd:.2f}_{name}_model_0.pdb"
                shutil.copy2(pdb_path, lig_dir / dest_name)

                print(f"      {name}: lig_rmsd={shape_rmsd:.3f}  "
                      f"C={c_rmsd}  O={o_rmsd}  score={score}  {fp}")

                obj_name = f"{tier}_{name}"
                pymol_lines.append(
                    f'cmd.load(os.path.join(d, "{dest_name}"), "{obj_name}")')
                pymol_lines.append(
                    f'cmd.color("{colors[tier]}", "{obj_name}")')

        # Write PyMOL script
        pymol_lines.append('\n# Show ligand as sticks')
        pymol_lines.append('cmd.show("sticks", "chain B")')
        pymol_lines.append('cmd.show("cartoon", "chain A")')
        pymol_lines.append('cmd.set("cartoon_transparency", 0.7)')
        pymol_lines.append('cmd.zoom("chain B")')
        pymol_lines.append('print("Colors: green=best, cyan=p25, yellow=median, '
                           'orange=p75, red=worst")')

        pymol_path = lig_dir / f"load_diagnostic_{lig}.py"
        with open(pymol_path, 'w') as f:
            f.write('\n'.join(pymol_lines) + '\n')

        print(f"\n    PyMOL script: {pymol_path}")
        print(f"    PDBs in: {lig_dir}")

    print(f"\n  All diagnostic PDBs in: {out_base}")


if __name__ == "__main__":
    main()
