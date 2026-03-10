#!/usr/bin/env python3
"""
Quick analysis: for each clustered docked PDB, measure distance from each
ligand oxygen (O1, O2, O3, O4) to the nearest water oxygen.
Reports which oxygen is closest and shows distributions.
"""

import os, sys, glob, argparse
import numpy as np
from collections import Counter

def parse_pdb_coords(pdb_path):
    """Extract ligand oxygen and water oxygen coordinates from PDB."""
    lig_oxygens = {}   # atom_name -> (x, y, z)
    water_oxygens = [] # list of (x, y, z)

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            # Water oxygens
            if res_name in ("HOH", "TP3", "WAT", "TIP", "TP3W"):
                if atom_name == "O":
                    water_oxygens.append(np.array([x, y, z]))
                continue

            # Ligand oxygens — any non-standard residue with O1/O2/O3/O4
            PROTEIN_RES = {
                "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
            }
            if res_name not in PROTEIN_RES and atom_name in ("O1", "O2", "O3", "O4"):
                lig_oxygens[atom_name] = np.array([x, y, z])

    return lig_oxygens, water_oxygens


def analyze_pdb(pdb_path):
    """Return dict of {atom_name: min_distance_to_water} and closest atom."""
    lig_oxygens, water_oxygens = parse_pdb_coords(pdb_path)

    if not lig_oxygens or not water_oxygens:
        return None

    distances = {}
    for name, lig_xyz in lig_oxygens.items():
        min_dist = min(np.linalg.norm(lig_xyz - w) for w in water_oxygens)
        distances[name] = min_dist

    closest = min(distances, key=distances.get)
    return {"distances": distances, "closest": closest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_dir", help="Directory containing docked PDBs (clustered output)")
    parser.add_argument("--pattern", default="*rep_*.pdb", help="Glob pattern for PDB files")
    args = parser.parse_args()

    pdbs = sorted(glob.glob(os.path.join(args.pdb_dir, args.pattern)))
    if not pdbs:
        # Try recursive
        pdbs = sorted(glob.glob(os.path.join(args.pdb_dir, "**", args.pattern), recursive=True))
    if not pdbs:
        print(f"No PDBs matching '{args.pattern}' in {args.pdb_dir}")
        sys.exit(1)

    print(f"Analyzing {len(pdbs)} PDBs...\n")

    closest_counts = Counter()
    all_distances = {name: [] for name in ["O1", "O2", "O3", "O4"]}
    failed = 0

    for pdb in pdbs:
        result = analyze_pdb(pdb)
        if result is None:
            failed += 1
            continue

        closest_counts[result["closest"]] += 1
        for name, dist in result["distances"].items():
            all_distances[name].append(dist)

    # --- Report ---
    print("=" * 60)
    print("CLOSEST OXYGEN TO WATER — DISTRIBUTION")
    print("=" * 60)
    total = sum(closest_counts.values())
    for name in sorted(closest_counts, key=closest_counts.get, reverse=True):
        count = closest_counts[name]
        pct = 100 * count / total
        print(f"  {name}: {count:4d} / {total} ({pct:5.1f}%)")

    print(f"\n  (failed to parse: {failed})")

    print("\n" + "=" * 60)
    print("DISTANCE TO NEAREST WATER (Angstroms)")
    print("=" * 60)
    print(f"  {'Atom':<6} {'N':>5} {'Mean':>6} {'Std':>6} {'Min':>6} {'Q25':>6} {'Med':>6} {'Q75':>6} {'Max':>6}")
    print(f"  {'-'*5:<6} {'-'*5:>5} {'-'*5:>6} {'-'*5:>6} {'-'*5:>6} {'-'*5:>6} {'-'*5:>6} {'-'*5:>6} {'-'*5:>6}")
    for name in ["O1", "O2", "O3", "O4"]:
        vals = all_distances[name]
        if not vals:
            continue
        a = np.array(vals)
        print(f"  {name:<6} {len(a):5d} {a.mean():6.2f} {a.std():6.2f} {a.min():6.2f} "
              f"{np.percentile(a, 25):6.2f} {np.median(a):6.2f} {np.percentile(a, 75):6.2f} {a.max():6.2f}")

    # Histogram-style view of closest atom distances
    print("\n" + "=" * 60)
    print("CLOSEST-ATOM DISTANCE DISTRIBUTION (water-O to winning atom)")
    print("=" * 60)
    winning_dists = []
    for pdb in pdbs:
        result = analyze_pdb(pdb)
        if result is None:
            continue
        closest = result["closest"]
        winning_dists.append(result["distances"][closest])

    if winning_dists:
        a = np.array(winning_dists)
        bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 999]
        labels = ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0-5.0", ">5.0"]
        for i, label in enumerate(labels):
            count = np.sum((a >= bins[i]) & (a < bins[i + 1]))
            bar = "#" * int(count / max(1, len(a)) * 50)
            print(f"  {label:>8}: {count:4d}  {bar}")


if __name__ == "__main__":
    main()
