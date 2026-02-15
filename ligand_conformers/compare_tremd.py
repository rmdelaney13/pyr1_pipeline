#!/usr/bin/env python3
"""
compare_tremd.py — compare conformer-generation output against TREMD reference.

For each TREMD reference conformer, find the best-matching generated conformer
(lowest heavy-atom RMSD after optimal alignment) and report recovery statistics.

Usage:

    # After running conformer generation:
    python -m ligand_conformers --input <SMILES> --input-type smiles \
        --outdir win_mmff/ --num-confs 500 -k 10 --ligand-id win

    # Then compare:
    python ligand_conformers/compare_tremd.py \
        --ref  ligand_conformers/win/win_12_conf.sdf \
        --gen  win_mmff/conformers_final.sdf \
        --label "MMFF-only" \
        --outdir win_mmff/

    # Or compare two runs side by side (MMFF vs MMFF+OpenMM):
    python ligand_conformers/compare_tremd.py \
        --ref  ligand_conformers/win/win_12_conf.sdf \
        --gen  win_mmff/conformers_final.sdf  win_openmm/conformers_final.sdf \
        --label "MMFF-only"  "MMFF+OpenMM" \
        --outdir comparison/

    # Also compare against ALL generated conformers (not just final K):
    python ligand_conformers/compare_tremd.py \
        --ref  ligand_conformers/win/win_12_conf.sdf \
        --gen  win_mmff/conformers_raw.sdf  win_mmff/conformers_final.sdf \
        --label "all-500-minimised"  "final-K-selected" \
        --outdir comparison/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def load_conformers(sdf_path: Path) -> Tuple[List[Chem.Mol], List[str]]:
    """Load all molecules from an SDF.  Returns (mols, names)."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mols, names = [], []
    for i, mol in enumerate(suppl):
        if mol is None:
            print(f"  WARNING: Could not parse molecule {i} from {sdf_path}")
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"conf_{i}"
        mols.append(mol)
        names.append(name)
    return mols, names


def heavy_atom_indices(mol: Chem.Mol) -> List[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def build_atom_map(ref_mol: Chem.Mol, gen_mol: Chem.Mol) -> Optional[List[Tuple[int, int]]]:
    """Build a heavy-atom mapping between ref and gen molecules.

    Strategy:
        1. Try direct substructure match (fast, works when molecules are identical).
        2. Fall back to MCS-based mapping.

    Returns list of (gen_idx, ref_idx) pairs, or None on failure.
    """
    # Strip Hs for matching
    ref_noH = Chem.RemoveHs(ref_mol)
    gen_noH = Chem.RemoveHs(gen_mol)

    # Try direct substructure match (ref in gen)
    match = gen_noH.GetSubstructMatch(ref_noH)
    if match and len(match) == ref_noH.GetNumAtoms():
        # match[i] = index in gen_noH that corresponds to atom i in ref_noH
        # We need (gen_idx, ref_idx) pairs
        # But we need indices in the original H-containing molecules.
        # The RemoveHs preserves heavy atom order, so heavy atom i in noH
        # corresponds to heavy_atom_indices(mol)[i] in the original.
        ref_ha = heavy_atom_indices(ref_mol)
        gen_ha = heavy_atom_indices(gen_mol)
        atom_map = []
        for ref_noH_idx in range(ref_noH.GetNumAtoms()):
            gen_noH_idx = match[ref_noH_idx]
            if ref_noH_idx < len(ref_ha) and gen_noH_idx < len(gen_ha):
                atom_map.append((gen_ha[gen_noH_idx], ref_ha[ref_noH_idx]))
        return atom_map

    # Try reverse match (gen in ref)
    match = ref_noH.GetSubstructMatch(gen_noH)
    if match and len(match) == gen_noH.GetNumAtoms():
        ref_ha = heavy_atom_indices(ref_mol)
        gen_ha = heavy_atom_indices(gen_mol)
        atom_map = []
        for gen_noH_idx in range(gen_noH.GetNumAtoms()):
            ref_noH_idx = match[gen_noH_idx]
            if gen_noH_idx < len(gen_ha) and ref_noH_idx < len(ref_ha):
                atom_map.append((gen_ha[gen_noH_idx], ref_ha[ref_noH_idx]))
        return atom_map

    # Fall back to MCS
    print("  INFO: Direct substructure match failed, using MCS...")
    mcs = rdFMCS.FindMCS([ref_noH, gen_noH],
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrder,
                         timeout=30)
    if mcs.smartsString:
        pattern = Chem.MolFromSmarts(mcs.smartsString)
        ref_match = ref_noH.GetSubstructMatch(pattern)
        gen_match = gen_noH.GetSubstructMatch(pattern)
        if ref_match and gen_match:
            ref_ha = heavy_atom_indices(ref_mol)
            gen_ha = heavy_atom_indices(gen_mol)
            atom_map = []
            for ri, gi in zip(ref_match, gen_match):
                if ri < len(ref_ha) and gi < len(gen_ha):
                    atom_map.append((gen_ha[gi], ref_ha[ri]))
            return atom_map

    return None


def compute_rmsd(ref_mol: Chem.Mol, gen_mol: Chem.Mol,
                 atom_map: List[Tuple[int, int]]) -> float:
    """Aligned heavy-atom RMSD using the provided atom map.

    Works on single-conformer molecules (confId=0 for the first conformer).
    """
    # AlignMol modifies gen_mol in place — work on a copy
    gen_copy = Chem.RWMol(gen_mol)
    rmsd = rdMolAlign.AlignMol(gen_copy, ref_mol,
                                prbCid=0, refCid=0,
                                atomMap=atom_map)
    return rmsd


def compare_one_set(
    ref_mols: List[Chem.Mol],
    ref_names: List[str],
    gen_mols: List[Chem.Mol],
    gen_names: List[str],
    label: str,
) -> List[dict]:
    """Compare each reference conformer against all generated conformers.

    Returns one row per reference conformer with columns:
        ref_name, best_gen_name, best_rmsd, rank_in_gen, label, recovered
    """
    if not gen_mols:
        print(f"  WARNING: No generated conformers for label={label}")
        return []

    # Build atom map using the first pair (same molecule, map is reusable)
    atom_map = build_atom_map(ref_mols[0], gen_mols[0])
    if atom_map is None:
        print(f"  ERROR: Could not establish atom mapping for label={label}")
        return []
    print(f"  Atom map: {len(atom_map)} heavy atoms matched")

    rows = []
    for ri, (ref_mol, ref_name) in enumerate(zip(ref_mols, ref_names)):
        best_rmsd = float("inf")
        best_gi = -1

        for gi, gen_mol in enumerate(gen_mols):
            try:
                rmsd = compute_rmsd(ref_mol, gen_mol, atom_map)
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_gi = gi
            except Exception as exc:
                # Alignment can fail on edge cases
                pass

        rows.append({
            "ref_index": ri,
            "ref_name": ref_name,
            "best_gen_index": best_gi,
            "best_gen_name": gen_names[best_gi] if best_gi >= 0 else "",
            "best_rmsd_A": round(best_rmsd, 4),
            "recovered_1A": best_rmsd <= 1.0,
            "recovered_1_5A": best_rmsd <= 1.5,
            "recovered_2A": best_rmsd <= 2.0,
            "label": label,
        })

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(rows: List[dict], label: str):
    """Print a human-readable summary table."""
    n = len(rows)
    if n == 0:
        return

    rmsds = [r["best_rmsd_A"] for r in rows]
    rec_1 = sum(1 for r in rows if r["recovered_1A"])
    rec_15 = sum(1 for r in rows if r["recovered_1_5A"])
    rec_2 = sum(1 for r in rows if r["recovered_2A"])

    print(f"\n{'='*65}")
    print(f"  {label}:  {n} TREMD refs  vs  generated conformers")
    print(f"{'='*65}")
    print(f"  {'Ref':<25s}  {'Best match':<25s}  {'RMSD(Å)':>8s}  {'<1Å':>4s}")
    print(f"  {'-'*25}  {'-'*25}  {'-'*8}  {'-'*4}")
    for r in rows:
        tag = " *" if r["recovered_1A"] else ""
        print(f"  {r['ref_name']:<25s}  {r['best_gen_name']:<25s}  "
              f"{r['best_rmsd_A']:8.3f}  {tag}")

    print(f"\n  Recovery rates:")
    print(f"    RMSD < 1.0 Å:  {rec_1}/{n}  ({100*rec_1/n:.0f}%)")
    print(f"    RMSD < 1.5 Å:  {rec_15}/{n}  ({100*rec_15/n:.0f}%)")
    print(f"    RMSD < 2.0 Å:  {rec_2}/{n}  ({100*rec_2/n:.0f}%)")
    print(f"    Mean best RMSD:   {np.mean(rmsds):.3f} Å")
    print(f"    Median best RMSD: {np.median(rmsds):.3f} Å")
    print(f"    Max best RMSD:    {np.max(rmsds):.3f} Å")
    print()


def write_csv(all_rows: List[dict], outdir: Path):
    csv_path = outdir / "tremd_comparison.csv"
    if not all_rows:
        return
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[OK] CSV: {csv_path}")


def write_json_summary(all_rows: List[dict], outdir: Path):
    """Write a machine-readable summary grouped by label."""
    summary = {}
    for r in all_rows:
        lbl = r["label"]
        if lbl not in summary:
            summary[lbl] = {"n_ref": 0, "rmsds": [], "recovered_1A": 0,
                            "recovered_1_5A": 0, "recovered_2A": 0}
        summary[lbl]["n_ref"] += 1
        summary[lbl]["rmsds"].append(r["best_rmsd_A"])
        if r["recovered_1A"]:
            summary[lbl]["recovered_1A"] += 1
        if r["recovered_1_5A"]:
            summary[lbl]["recovered_1_5A"] += 1
        if r["recovered_2A"]:
            summary[lbl]["recovered_2A"] += 1

    for lbl, s in summary.items():
        n = s["n_ref"]
        s["mean_rmsd"] = round(float(np.mean(s["rmsds"])), 4)
        s["median_rmsd"] = round(float(np.median(s["rmsds"])), 4)
        s["max_rmsd"] = round(float(np.max(s["rmsds"])), 4)
        s["recovery_rate_1A"] = round(s["recovered_1A"] / n, 4) if n else 0
        s["recovery_rate_1_5A"] = round(s["recovered_1_5A"] / n, 4) if n else 0
        s["recovery_rate_2A"] = round(s["recovered_2A"] / n, 4) if n else 0
        del s["rmsds"]

    json_path = outdir / "tremd_comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] JSON: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compare generated conformers against TREMD references.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ref", required=True, type=Path,
                   help="SDF with TREMD reference conformers.")
    p.add_argument("--gen", required=True, nargs="+", type=Path,
                   help="One or more SDFs with generated conformers to compare.")
    p.add_argument("--label", nargs="+", default=None,
                   help="Labels for each --gen SDF (default: filenames).")
    p.add_argument("--outdir", type=Path, default=Path("."),
                   help="Output directory for CSV and JSON results.")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[1.0, 1.5, 2.0],
                   help="RMSD thresholds for recovery (default: 1.0 1.5 2.0).")
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Labels
    labels = args.label
    if labels is None:
        labels = [p.stem for p in args.gen]
    if len(labels) != len(args.gen):
        print("ERROR: Number of --label must match number of --gen files.",
              file=sys.stderr)
        sys.exit(1)

    # Load references
    print(f"Loading TREMD references: {args.ref}")
    ref_mols, ref_names = load_conformers(args.ref)
    print(f"  Loaded {len(ref_mols)} reference conformers")

    all_rows = []
    for gen_path, label in zip(args.gen, labels):
        print(f"\nLoading generated: {gen_path} [{label}]")
        gen_mols, gen_names = load_conformers(gen_path)
        print(f"  Loaded {len(gen_mols)} generated conformers")

        rows = compare_one_set(ref_mols, ref_names, gen_mols, gen_names, label)
        all_rows.extend(rows)
        print_summary(rows, label)

    write_csv(all_rows, args.outdir)
    write_json_summary(all_rows, args.outdir)


if __name__ == "__main__":
    main()
