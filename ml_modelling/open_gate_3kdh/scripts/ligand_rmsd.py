#!/usr/bin/env python3
"""Compute ligand RMSD between threaded-relaxed structures and Boltz predictions.

Superimposes each relaxed structure onto its Boltz prediction using pocket-floor
CA anchors (Kabsch), then computes heavy-atom RMSD of the ligand. This measures
how much the ligand moved relative to the pocket during relax.

Outputs a CSV with per-structure RMSD values and summary statistics.

Usage:
    python ligand_rmsd.py \
        --relaxed-dir outputs/threaded_relaxed/ \
        --boltz-dir inputs/boltz_predictions/ \
        --alignment alignment_map.json \
        --csv ../analysis/boltz_LCA/md_candidates_lca_top100.csv \
        --output-csv outputs/ligand_rmsd.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse functions from place_ligand.py
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))
from place_ligand import parse_pdb_atoms, get_ca_coords, get_ligand_atoms, kabsch_superpose


def get_ligand_coords_by_name(atoms, chain="B"):
    """Extract ligand heavy-atom coordinates keyed by atom name."""
    coords = {}
    for atom in atoms:
        if atom["record"] == "HETATM" and atom["chain"] == chain:
            if atom["element"] == "H" or atom["name"].startswith("H"):
                continue
            coords[atom["name"]] = np.array([atom["x"], atom["y"], atom["z"]])
    return coords


def compute_ligand_rmsd(relaxed_pdb, boltz_pdb, anchor_3kdh, anchor_boltz,
                        relaxed_chain="A", boltz_chain="A",
                        lig_chain_relaxed="B", lig_chain_boltz="B"):
    """Compute ligand RMSD after superposing on pocket-floor CAs.

    Returns:
        dict with rmsd, n_atoms_matched, sup_rmsd, n_anchors
    """
    relaxed_atoms = parse_pdb_atoms(str(relaxed_pdb))
    boltz_atoms = parse_pdb_atoms(str(boltz_pdb))

    # Get pocket-floor CAs for superposition
    relaxed_ca = get_ca_coords(relaxed_atoms, anchor_3kdh, chain=relaxed_chain)
    boltz_ca = get_ca_coords(boltz_atoms, anchor_boltz, chain=boltz_chain)

    common_anchors = []
    for ab, ak in zip(anchor_boltz, anchor_3kdh):
        if ab in boltz_ca and ak in relaxed_ca:
            common_anchors.append((ab, ak))

    if len(common_anchors) < 4:
        return {"ligand_rmsd": None, "n_atoms_matched": 0,
                "sup_rmsd": None, "n_anchors": len(common_anchors),
                "error": "too few anchors"}

    mobile_coords = np.array([relaxed_ca[ak] for _, ak in common_anchors])
    target_coords = np.array([boltz_ca[ab] for ab, _ in common_anchors])

    R, t, sup_rmsd = kabsch_superpose(mobile_coords, target_coords)

    # Get ligand atoms by name
    relaxed_lig = get_ligand_coords_by_name(relaxed_atoms, chain=lig_chain_relaxed)
    boltz_lig = get_ligand_coords_by_name(boltz_atoms, chain=lig_chain_boltz)

    # Match atoms by name
    common_names = sorted(set(relaxed_lig.keys()) & set(boltz_lig.keys()))
    if not common_names:
        return {"ligand_rmsd": None, "n_atoms_matched": 0,
                "sup_rmsd": round(sup_rmsd, 3), "n_anchors": len(common_anchors),
                "error": "no matching ligand atoms"}

    # Transform relaxed ligand coords into Boltz frame
    relaxed_xyz = np.array([relaxed_lig[n] for n in common_names])
    boltz_xyz = np.array([boltz_lig[n] for n in common_names])

    transformed = (R @ relaxed_xyz.T).T + t
    diff = transformed - boltz_xyz
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    # Per-atom distances for debugging
    per_atom = np.sqrt(np.sum(diff ** 2, axis=1))

    return {
        "ligand_rmsd": round(rmsd, 3),
        "n_atoms_matched": len(common_names),
        "sup_rmsd": round(sup_rmsd, 3),
        "n_anchors": len(common_anchors),
        "max_atom_deviation": round(float(per_atom.max()), 3),
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute ligand RMSD between relaxed and Boltz structures"
    )
    parser.add_argument("--relaxed-dir", required=True,
                        help="Directory with *_threaded_relaxed.pdb files")
    parser.add_argument("--boltz-dir", required=True,
                        help="Directory with Boltz prediction PDBs")
    parser.add_argument("--alignment", required=True,
                        help="Path to alignment_map.json")
    parser.add_argument("--csv", required=True,
                        help="CSV with pair_id column")
    parser.add_argument("--output-csv",
                        help="Output CSV path (default: stdout)")
    parser.add_argument("--single",
                        help="Process single pair_id")
    args = parser.parse_args()

    with open(args.alignment) as f:
        alignment_map = json.load(f)

    anchor_3kdh = alignment_map["anchor_positions_3kdh"]
    anchor_boltz = alignment_map["anchor_positions_boltz"]

    df = pd.read_csv(args.csv)
    if args.single:
        df = df[df["pair_id"] == args.single]

    relaxed_dir = Path(args.relaxed_dir)
    boltz_dir = Path(args.boltz_dir)

    results = []
    for _, row in df.iterrows():
        pair_id = row["pair_id"]
        relaxed_pdb = relaxed_dir / f"{pair_id}_threaded_relaxed.pdb"
        boltz_pdb = boltz_dir / f"{pair_id}.pdb"

        if not relaxed_pdb.exists():
            results.append({"pair_id": pair_id, "ligand_rmsd": None,
                           "error": "relaxed PDB not found"})
            continue
        if not boltz_pdb.exists():
            results.append({"pair_id": pair_id, "ligand_rmsd": None,
                           "error": "Boltz PDB not found"})
            continue

        result = compute_ligand_rmsd(relaxed_pdb, boltz_pdb,
                                     anchor_3kdh, anchor_boltz)
        result["pair_id"] = pair_id
        results.append(result)

    out_df = pd.DataFrame(results)
    cols = ["pair_id", "ligand_rmsd", "sup_rmsd", "n_atoms_matched",
            "n_anchors", "max_atom_deviation", "error"]
    cols = [c for c in cols if c in out_df.columns]
    out_df = out_df[cols]

    if args.output_csv:
        out_df.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(out_df)} rows to {args.output_csv}")
    else:
        print(out_df.to_string(index=False))

    # Summary stats
    valid = out_df["ligand_rmsd"].dropna()
    if len(valid) > 0:
        print(f"\n--- Summary ({len(valid)} structures) ---")
        print(f"  Mean ligand RMSD:   {valid.mean():.3f} A")
        print(f"  Median:             {valid.median():.3f} A")
        print(f"  Std:                {valid.std():.3f} A")
        print(f"  Min:                {valid.min():.3f} A")
        print(f"  Max:                {valid.max():.3f} A")
        print(f"  < 1.0 A:            {(valid < 1.0).sum()}")
        print(f"  1.0-2.0 A:          {((valid >= 1.0) & (valid < 2.0)).sum()}")
        print(f"  > 2.0 A:            {(valid >= 2.0).sum()}")


if __name__ == "__main__":
    main()
