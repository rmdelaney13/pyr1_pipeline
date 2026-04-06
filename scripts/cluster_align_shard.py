#!/usr/bin/env python3
"""
Stage 1: Parse and align a shard of PDBs, save aligned ligand coords.

Called by SLURM array. Each task processes designs[start:end].

Usage:
    python scripts/cluster_align_shard.py \
        --expansion-root /path/to/ligandmpnn \
        --ref-pdb /path/to/3QN1_H2O.pdb \
        --name-list /path/to/design_names.txt \
        --shard-index 0 --shard-size 1000 \
        --out-dir /path/to/shards/
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ── Same parsing/alignment code as cluster_ligand_poses.py ───────

# Default atom order (CDCA). Overridden at runtime by auto-detection from first PDB.
LIGAND_ATOM_ORDER = [
    'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54',
    'C55', 'C56', 'C57', 'C58', 'C59', 'C60', 'C61', 'C62', 'C63', 'C64',
    'C65', 'C66', 'C67', 'C68', 'O41', 'O42', 'O43', 'O44',
]


def detect_ligand_atoms(pdb_path, ligand_chain='B'):
    """Auto-detect ligand heavy atom names from a Boltz PDB."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if line[21] != ligand_chain:
                continue
            aname = line[12:16].strip()
            elem = line[76:78].strip() if len(line) > 76 else aname[0]
            if elem != 'H' and aname not in atoms:
                atoms.append(aname)
    return sorted(atoms)


def parse_pdb(pdb_path, protein_chain='A', ligand_chain='B'):
    ca_coords = {}
    ligand_coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            ch = line[21]
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if ch == protein_chain:
                if atom_name == 'CA':
                    resnum = int(line[22:26].strip())
                    if resnum not in ca_coords:
                        ca_coords[resnum] = np.array([x, y, z])
            elif ch == ligand_chain:
                elem = line[76:78].strip() if len(line) > 76 else atom_name[0]
                if elem != 'H':
                    ligand_coords[atom_name] = np.array([x, y, z])
    return ca_coords, ligand_coords


def kabsch_rotation(P, Q):
    C = P.T @ Q
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R


def align_ligand(ca_coords, ligand_coords, ref_ca, atom_order=None):
    if atom_order is None:
        atom_order = LIGAND_ATOM_ORDER
    common = sorted(set(ca_coords) & set(ref_ca))
    if len(common) < 10:
        return None
    mobile = np.array([ca_coords[r] for r in common])
    fixed = np.array([ref_ca[r] for r in common])
    mobile_center = mobile.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    R = kabsch_rotation(mobile - mobile_center, fixed - fixed_center)
    coords = np.empty((len(atom_order), 3))
    for i, aname in enumerate(atom_order):
        if aname not in ligand_coords:
            return None
        coords[i] = R @ (ligand_coords[aname] - mobile_center) + fixed_center
    return coords


def find_pdb(name, expansion_root):
    root = Path(expansion_root)
    for round_dir in sorted(root.glob("round_*")):
        pdb = (round_dir / "boltz_output" / f"boltz_results_{name}" /
               "predictions" / name / f"{name}_model_0.pdb")
        if pdb.exists():
            return pdb
    return None


def find_pdb_flat(name, boltz_dir):
    """Find PDB in a flat boltz output directory (no round_* nesting)."""
    pdb = Path(boltz_dir) / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.pdb"
    return pdb if pdb.exists() else None


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--expansion-root',
                       help='Expansion root with round_*/boltz_output/ layout')
    group.add_argument('--boltz-dir', nargs='+',
                       help='Flat boltz output directory (e.g. systematic). '
                            'Multiple dirs can be given for expansion rounds.')
    parser.add_argument('--ref-pdb', required=True)
    parser.add_argument('--name-list', required=True,
                        help='Text file with one design name per line')
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-size', type=int, default=1000)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    # Load reference
    ref_ca, _ = parse_pdb(args.ref_pdb)

    # Load name list and slice to this shard
    with open(args.name_list) as f:
        all_names = [line.strip() for line in f if line.strip()]

    start = args.shard_index * args.shard_size
    end = min(start + args.shard_size, len(all_names))
    names = all_names[start:end]
    print(f"Shard {args.shard_index}: processing {len(names)} designs [{start}:{end}]",
          flush=True)

    if not names:
        print("Empty shard, nothing to do.")
        return

    # Auto-detect ligand atom order from first available PDB
    atom_order = None
    for name in names:
        if args.boltz_dir:
            pdb_path = None
            for bd in args.boltz_dir:
                pdb_path = find_pdb_flat(name, bd)
                if pdb_path:
                    break
        else:
            pdb_path = find_pdb(name, args.expansion_root)
        if pdb_path is not None:
            atom_order = detect_ligand_atoms(str(pdb_path))
            if atom_order:
                print(f"  Auto-detected {len(atom_order)} ligand atoms: "
                      f"{atom_order[0]}..{atom_order[-1]}", flush=True)
                break
    if not atom_order:
        atom_order = LIGAND_ATOM_ORDER
        print(f"  Using default atom order ({len(atom_order)} atoms)", flush=True)

    # Parse and align
    aligned_names = []
    aligned_coords = []
    missing = 0
    failed = 0

    for i, name in enumerate(names):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(names)}...", flush=True)

        if args.boltz_dir:
            pdb_path = None
            for bd in args.boltz_dir:
                pdb_path = find_pdb_flat(name, bd)
                if pdb_path:
                    break
        else:
            pdb_path = find_pdb(name, args.expansion_root)
        if pdb_path is None:
            missing += 1
            continue

        ca, lig = parse_pdb(str(pdb_path))
        coords = align_ligand(ca, lig, ref_ca, atom_order=atom_order)
        if coords is None:
            failed += 1
            continue

        aligned_names.append(name)
        aligned_coords.append(coords)

    print(f"  Aligned: {len(aligned_coords)}, Missing: {missing}, Failed: {failed}",
          flush=True)

    # Save shard
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if aligned_coords:
        coords_array = np.stack(aligned_coords)
        np.save(out_dir / f"coords_{args.shard_index:04d}.npy", coords_array)
        with open(out_dir / f"names_{args.shard_index:04d}.json", 'w') as f:
            json.dump(aligned_names, f)

    print(f"  Saved shard {args.shard_index} to {out_dir}", flush=True)


if __name__ == '__main__':
    main()
