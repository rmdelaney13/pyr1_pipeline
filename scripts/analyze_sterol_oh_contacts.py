#!/usr/bin/env python3
"""
Identify protein residues satisfying CDCA core sterol hydroxyl groups.

CDCA has two sterol OH groups (3α-OH and 7α-OH). LCA only has 3α-OH.
The 7α-OH is the key differentiator — it needs a pocket residue to
satisfy it via H-bond. This script finds those contacts.

For each Boltz prediction:
  1. Identifies ligand hydroxyl oxygens (non-carboxylate O atoms)
  2. Finds protein heavy atoms within H-bonding distance (<3.5 A)
  3. Reports which residues contact each OH

Usage:
    python scripts/analyze_sterol_oh_contacts.py \
        --boltz-dir /scratch/alpine/ryde3462/boltz_cdca_nonbinders/output_binary \
        --out /scratch/alpine/ryde3462/boltz_cdca_nonbinders/oh_contacts.csv
"""

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HBOND_CUTOFF = 3.5  # Angstroms


def _resolve_element(atom) -> str:
    elem = atom.element.strip().upper() if atom.element else ''
    if not elem:
        name = atom.get_name().strip()
        if name:
            elem = name[0].upper()
    return elem


def find_boltz_predictions(output_dir: str) -> List[Dict[str, Path]]:
    """Find all Boltz prediction PDBs."""
    out = Path(output_dir)
    predictions = []
    for results_dir in sorted(out.glob("boltz_results_*")):
        pred_dir = results_dir / "predictions"
        if not pred_dir.exists():
            continue
        for name_dir in sorted(pred_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            name = name_dir.name
            struct_path = name_dir / f"{name}_model_0.pdb"
            if not struct_path.exists():
                struct_path = name_dir / f"{name}_model_0.cif"
            if not struct_path.exists():
                continue
            predictions.append({'name': name, 'structure': struct_path})
    return predictions


def find_carboxylate_oxygens(ligand_atoms):
    """Return set of atom ids that are carboxylate oxygens."""
    oxygens = [a for a in ligand_atoms if _resolve_element(a) == 'O']
    carbons = [a for a in ligand_atoms if _resolve_element(a) == 'C']

    coo_ids = set()
    for c in carbons:
        c_coord = c.get_coord()
        bonded_o = [o for o in oxygens if np.linalg.norm(o.get_coord() - c_coord) < 1.65]
        if len(bonded_o) == 2:
            for o in bonded_o:
                coo_ids.add(id(o))
    return coo_ids


def analyze_oh_contacts(struct_path: Path, protein_chain='A', ligand_chain='B'):
    """Find protein contacts to each ligand hydroxyl oxygen.

    Returns list of dicts, one per hydroxyl O found:
        {oh_index, oh_atom_name, oh_coord,
         contacts: [{res_name, res_num, atom_name, distance}, ...]}
    """
    from Bio.PDB import PDBParser, MMCIFParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ext = struct_path.suffix.lower()
        if ext == '.cif':
            struct = MMCIFParser(QUIET=True).get_structure("s", str(struct_path))
        else:
            struct = PDBParser(QUIET=True).get_structure("s", str(struct_path))

    # Get ligand heavy atoms
    ligand_atoms = []
    for model in struct:
        for chain in model:
            if chain.id == ligand_chain:
                for res in chain:
                    for atom in res:
                        elem = _resolve_element(atom)
                        if elem not in ('H', ''):
                            ligand_atoms.append(atom)
        break

    if not ligand_atoms:
        return []

    # Identify carboxylate O vs hydroxyl O
    coo_ids = find_carboxylate_oxygens(ligand_atoms)
    hydroxyl_oxygens = [a for a in ligand_atoms
                        if _resolve_element(a) == 'O' and id(a) not in coo_ids]

    # Get all protein heavy atoms
    protein_atoms = []
    for model in struct:
        for chain in model:
            if chain.id == protein_chain:
                for res in chain:
                    for atom in res:
                        elem = _resolve_element(atom)
                        if elem not in ('H', ''):
                            protein_atoms.append(atom)
        break

    if not protein_atoms:
        return []

    # For each hydroxyl O, find protein contacts
    results = []
    for i, oh_atom in enumerate(hydroxyl_oxygens):
        oh_coord = oh_atom.get_coord()
        contacts = []

        for patom in protein_atoms:
            dist = float(np.linalg.norm(patom.get_coord() - oh_coord))
            if dist <= HBOND_CUTOFF:
                res = patom.get_parent()
                # Only count polar contacts (O, N, S) for H-bond partners
                p_elem = _resolve_element(patom)
                contacts.append({
                    'res_name': res.get_resname(),
                    'res_num': res.get_id()[1],
                    'atom_name': patom.get_name(),
                    'atom_element': p_elem,
                    'distance': round(dist, 2),
                    'is_polar': p_elem in ('O', 'N', 'S'),
                })

        # Sort by distance
        contacts.sort(key=lambda c: c['distance'])

        results.append({
            'oh_index': i,
            'oh_atom_name': oh_atom.get_name(),
            'oh_coord': oh_coord.tolist(),
            'contacts': contacts,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze sterol OH contacts in Boltz predictions")
    parser.add_argument("--boltz-dir", required=True, help="Boltz output directory")
    parser.add_argument("--out", required=True, help="Output CSV")
    parser.add_argument("--cutoff", type=float, default=3.5, help="H-bond cutoff in Angstroms (default: 3.5)")
    args = parser.parse_args()

    global HBOND_CUTOFF
    HBOND_CUTOFF = args.cutoff

    predictions = find_boltz_predictions(args.boltz_dir)
    print(f"Found {len(predictions)} predictions")

    rows = []
    for pred in predictions:
        name = pred['name']
        oh_results = analyze_oh_contacts(pred['structure'])

        if not oh_results:
            print(f"  {name}: no hydroxyl O found")
            continue

        # Sort OHs by position in structure (use coord to distinguish 3α vs 7α)
        # The two OHs have different positions; we label them OH_1 and OH_2
        for oh in oh_results:
            polar_contacts = [c for c in oh['contacts'] if c['is_polar']]
            all_contacts = oh['contacts']

            # Closest polar contact (H-bond partner)
            closest_polar = polar_contacts[0] if polar_contacts else None
            # Closest any contact
            closest_any = all_contacts[0] if all_contacts else None

            # All polar contacts as summary string
            polar_summary = "; ".join(
                f"{c['res_name']}{c['res_num']}:{c['atom_name']}({c['distance']}A)"
                for c in polar_contacts[:5]
            )

            row = {
                'name': name,
                'oh_index': oh['oh_index'],
                'oh_atom': oh['oh_atom_name'],
                'oh_x': round(oh['oh_coord'][0], 1),
                'oh_y': round(oh['oh_coord'][1], 1),
                'oh_z': round(oh['oh_coord'][2], 1),
                'n_contacts_total': len(all_contacts),
                'n_contacts_polar': len(polar_contacts),
                'closest_polar_res': f"{closest_polar['res_name']}{closest_polar['res_num']}" if closest_polar else "none",
                'closest_polar_atom': closest_polar['atom_name'] if closest_polar else "none",
                'closest_polar_dist': closest_polar['distance'] if closest_polar else None,
                'polar_contacts': polar_summary if polar_summary else "none",
            }
            rows.append(row)

        # Print summary for this variant
        print(f"  {name}:")
        for oh in oh_results:
            polar = [c for c in oh['contacts'] if c['is_polar']]
            if polar:
                top = polar[0]
                print(f"    OH_{oh['oh_index']} ({oh['oh_atom_name']}): "
                      f"closest polar = {top['res_name']}{top['res_num']}:{top['atom_name']} "
                      f"({top['distance']} A), {len(polar)} total polar contacts")
            else:
                print(f"    OH_{oh['oh_index']} ({oh['oh_atom_name']}): no polar contacts within {HBOND_CUTOFF} A")

    # Write CSV
    if rows:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")

    # Print aggregate: which residues appear most often as OH contacts
    print("\n=== Residue Frequency as OH H-bond Partner ===")
    from collections import Counter
    res_counts = Counter()
    for row in rows:
        if row['closest_polar_res'] != "none":
            res_counts[row['closest_polar_res']] += 1
    for res, count in res_counts.most_common(10):
        print(f"  {res}: {count}/{len(rows)} OH contacts")


if __name__ == "__main__":
    main()
