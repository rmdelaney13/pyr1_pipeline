#!/usr/bin/env python3
"""
Select top N designs by binary_total_score and copy their Boltz PDBs to a staging dir.

Reads the scores CSV from analyze_boltz_output.py, selects the top N rows by
binary_total_score, locates the corresponding PDB files in the Boltz output
directory tree, and copies them to a staging directory for LigandMPNN redesign.

Usage:
    python expansion_select.py \
        --scores /scratch/.../scores.csv \
        --boltz-dirs /scratch/.../output_ca_binary /scratch/.../round_1/boltz_output \
        --out-dir /scratch/.../round_1/selected_pdbs \
        --top-n 100

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import List, Optional


def find_pdb_for_name(name: str, boltz_dirs: List[str]) -> Optional[Path]:
    """Locate the Boltz output PDB for a given prediction name.

    Searches: boltz_dir/boltz_results_{name}/predictions/{name}/{name}_model_0.pdb
    """
    for d in boltz_dirs:
        d = Path(d)
        # Direct path
        pdb = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.pdb"
        if pdb.exists():
            return pdb
        # Try CIF
        cif = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.cif"
        if cif.exists():
            return cif
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Select top N designs and copy PDBs for MPNN redesign")
    parser.add_argument("--scores", required=True,
                        help="Scores CSV from analyze_boltz_output.py")
    parser.add_argument("--boltz-dirs", nargs='+', action='append', required=True,
                        help="Boltz output directories to search for PDBs "
                             "(can be repeated: --boltz-dirs A B --boltz-dirs C)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for selected PDBs")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top designs to select (default: 100)")
    parser.add_argument("--score-column", default="binary_total_score",
                        help="Column to rank by (default: binary_total_score)")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten boltz_dirs: action='append' + nargs='+' gives list of lists
    boltz_dirs = [d for group in args.boltz_dirs for d in group]

    print(f"Searching {len(boltz_dirs)} Boltz output directories:")
    for d in boltz_dirs:
        exists = Path(d).is_dir()
        print(f"  {'OK' if exists else 'MISSING'}: {d}")

    # Read and sort scores
    rows = []
    with open(args.scores) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_val = row.get(args.score_column)
            if score_val is None or score_val == '':
                continue
            try:
                row['_sort_score'] = float(score_val)
            except ValueError:
                continue
            rows.append(row)

    rows.sort(key=lambda r: r['_sort_score'], reverse=True)
    top_rows = rows[:args.top_n]

    print(f"\nLoaded {len(rows)} scored designs, selecting top {len(top_rows)}")
    if top_rows:
        print(f"  Score range: {top_rows[0]['_sort_score']:.4f} - {top_rows[-1]['_sort_score']:.4f}")

    # Copy PDBs
    copied = 0
    missing = 0
    missing_names = []
    manifest_lines = []

    for row in top_rows:
        name = row['name']
        pdb_path = find_pdb_for_name(name, boltz_dirs)
        if pdb_path is None:
            missing += 1
            missing_names.append(name)
            continue

        dest = out_dir / pdb_path.name
        shutil.copy2(pdb_path, dest)
        manifest_lines.append(str(dest))
        copied += 1

    # Write manifest
    manifest_path = out_dir.parent / "selected_manifest.txt"
    with open(manifest_path, 'w') as f:
        for line in manifest_lines:
            f.write(line + '\n')

    print(f"\nCopied {copied} PDBs to {out_dir}")
    if missing:
        print(f"  ({missing} PDBs not found)", file=sys.stderr)
        # Show first few missing names for debugging
        show = missing_names[:5]
        print(f"  First missing: {', '.join(show)}" +
              (f" (+ {missing - 5} more)" if missing > 5 else ""),
              file=sys.stderr)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
