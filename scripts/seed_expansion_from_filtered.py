#!/usr/bin/env python3
"""Seed expansion round selected_pdbs/ from Z-score filtered results.

Reads the filtered top100 CSV (from filter_expansion_designs.py),
locates each design's PDB file across expansion round dirs, and
copies them into the specified round's selected_pdbs/ directory.
Creates selected_manifest.txt for the MPNN submission script.

Usage:
    python scripts/seed_expansion_from_filtered.py \
        --filtered-csv /scratch/.../filtered/top100_ca.csv \
        --expansion-root /scratch/.../expansion/ligandmpnn \
        --ligand ca \
        --round 5 \
        --top-n 75
"""

import argparse
import csv
import shutil
from pathlib import Path


def find_pdb_path(expansion_root, ligand, name):
    """Find PDB file for a prediction name across round dirs."""
    lig_dir = Path(expansion_root) / ligand
    for round_dir in sorted(lig_dir.glob("round_*")):
        boltz_dir = round_dir / "boltz_output"
        if not boltz_dir.exists():
            continue
        pdb = (boltz_dir / f"boltz_results_{name}" / "predictions"
               / name / f"{name}_model_0.pdb")
        if pdb.exists():
            return pdb
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Seed expansion round from Z-score filtered results")
    parser.add_argument(
        "--filtered-csv", required=True,
        help="Path to filtered top100 CSV (from filter_expansion_designs.py)")
    parser.add_argument(
        "--expansion-root", required=True,
        help="Expansion root (e.g., /scratch/.../expansion/ligandmpnn)")
    parser.add_argument(
        "--ligand", required=True,
        help="Ligand name (ca, cdca, dca)")
    parser.add_argument(
        "--round", type=int, required=True,
        help="Round number to seed (e.g., 5)")
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Use top N designs from filtered CSV (default: all)")

    args = parser.parse_args()

    root = Path(args.expansion_root)
    lig = args.ligand.lower()
    round_dir = root / lig / f"round_{args.round}"
    selected_dir = round_dir / "selected_pdbs"
    manifest_path = round_dir / "selected_manifest.txt"

    # Check if already seeded
    if selected_dir.exists():
        n_existing = len(list(selected_dir.glob("*.pdb")))
        print(f"WARNING: {selected_dir} already exists ({n_existing} PDBs)")
        print(f"  Delete it to re-seed: rm -rf {selected_dir}")
        return

    # Read filtered CSV
    with open(args.filtered_csv) as f:
        rows = list(csv.DictReader(f))

    if args.top_n:
        rows = rows[:args.top_n]

    print(f"Seeding round {args.round} for {lig.upper()} from {len(rows)} filtered designs")

    # Create output directory
    selected_dir.mkdir(parents=True, exist_ok=True)

    # Copy PDBs and build manifest
    manifest_lines = []
    n_copied = 0
    n_missing = 0

    for row in rows:
        name = row.get('name', '')
        if not name:
            continue

        pdb_path = find_pdb_path(str(root), lig, name)
        if pdb_path is None:
            print(f"  WARNING: PDB not found for {name}")
            n_missing += 1
            continue

        dst = selected_dir / f"{name}.pdb"
        shutil.copy2(str(pdb_path), str(dst))
        manifest_lines.append(str(dst))
        n_copied += 1

    # Write manifest
    with open(manifest_path, 'w') as f:
        for line in manifest_lines:
            f.write(line + '\n')

    print(f"  Copied: {n_copied} PDBs to {selected_dir}")
    if n_missing:
        print(f"  Missing: {n_missing} PDBs")
    print(f"  Manifest: {manifest_path} ({len(manifest_lines)} entries)")
    print(f"  Ready for: bash slurm/run_expansion.sh {lig} {args.round}")


if __name__ == "__main__":
    main()
