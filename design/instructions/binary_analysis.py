#!/usr/bin/env python3

"""
binary_analysis.py

Aggregates AlphaFold3 results by recursively searching for JSONs.
Outputs:
  1. Ligand Chain pLDDT
  2. Ligand Chain iPTM
  3. Minimum distance from Ref Point to ANY Ligand Oxygen

Distance:
  REF: chain A, residue 201, atom O1 (default)
  AF3: ligand_chain, residue 1 -> Searches ALL Oxygens, finds closest to Ref.
"""

import os
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from Bio.PDB import MMCIFParser, Superimposer

# ----------------------------
# 1. File Discovery
# ----------------------------
def find_af3_files(base_dir):
    targets = {}
    base = Path(base_dir)

    print(f"Scanning {base_dir} recursively...")

    for summary_path in base.rglob("*_summary_confidences.json"):
        target_name = summary_path.name.replace("_summary_confidences.json", "")
        parent_dir = summary_path.parent

        full_conf_path = parent_dir / f"{target_name}_confidences.json"
        cif_path = parent_dir / f"{target_name}_model.cif"

        targets[target_name] = {
            "summary": str(summary_path),
            "full": str(full_conf_path) if full_conf_path.exists() else None,
            "cif": str(cif_path) if cif_path.exists() else None,
        }

    return targets


# ----------------------------
# 2. JSON Parsing
# ----------------------------
def get_chain_order(full_conf_path):
    if not full_conf_path:
        return None

    try:
        data = json.loads(Path(full_conf_path).read_text())
        for key in ["token_chain_ids", "atom_chain_ids", "chain_ids"]:
            if key in data and isinstance(data[key], list):
                unique = []
                seen = set()
                for c in data[key]:
                    c = str(c)
                    if c not in seen:
                        unique.append(c)
                        seen.add(c)
                return unique
    except Exception:
        pass

    return ["A", "B"]


def extract_json_metrics(file_map, ligand_chain="B"):
    plddt = None
    iptm = None

    # pLDDT
    if file_map.get("full"):
        try:
            data = json.loads(Path(file_map["full"]).read_text())
            plddts = np.array(data.get("atom_plddts", []), dtype=float)
            chains = np.array(data.get("atom_chain_ids", []), dtype=str)

            mask = (chains == ligand_chain)
            if mask.any():
                plddt = float(plddts[mask].mean())
        except Exception as e:
            print(f"Warning reading pLDDT for {file_map['full']}: {e}")

    # iPTM
    if file_map.get("summary"):
        try:
            data = json.loads(Path(file_map["summary"]).read_text())
            chain_iptms = data.get("chain_iptm", [])

            if chain_iptms:
                chain_order = get_chain_order(file_map.get("full"))
                if chain_order and ligand_chain in chain_order:
                    idx = chain_order.index(ligand_chain)
                    if idx < len(chain_iptms):
                        iptm = float(chain_iptms[idx])
                elif len(chain_iptms) == 2 and ligand_chain == "B":
                    iptm = float(chain_iptms[1])
        except Exception as e:
            print(f"Warning reading iPTM for {file_map['summary']}: {e}")

    return plddt, iptm


# ----------------------------
# 3. Geometry
# ----------------------------
def get_full_structure_and_ca_atoms(file_path):
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("model", file_path)
    except Exception:
        return None, None

    ca_atoms = []
    for model in structure:
        all_ca = []
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "CA":
                        all_ca.append(atom)

        # Sort CA atoms to ensure alignment consistency
        ca_atoms = sorted(
            all_ca,
            key=lambda a: (a.get_parent().get_parent().id, a.get_parent().get_id()[1]),
        )
        break

    return structure, ca_atoms


def get_specific_atom(structure, chain_id, res_id, atom_name):
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[1] != res_id:
                    continue
                for atom in residue:
                    if atom.get_name() == atom_name:
                        return atom
    return None


def get_ligand_atoms(structure, chain_id, res_id):
    """Returns a list of all atoms in the specific ligand residue."""
    atoms = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[1] != res_id:
                    continue
                for atom in residue:
                    atoms.append(atom)
    return atoms


def get_ligand_heavy_atoms(structure, chain_id="B"):
    """Extract non-hydrogen ligand atoms, sorted by name for consistent pairing."""
    atoms = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    for atom in res:
                        if atom.element != "H":
                            atoms.append(atom)
        break
    atoms.sort(key=lambda a: a.get_name())
    return atoms


def calculate_binary_ternary_rmsd(
    binary_cif_path, ternary_cif_path, protein_chain="A", ligand_chain="B"
):
    """
    Align binary protein to ternary protein (CA atoms), then compute
    ligand heavy-atom RMSD between the two predictions.

    Returns RMSD in Angstroms, or None on failure.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        t_struct = parser.get_structure("ternary", ternary_cif_path)
        b_struct = parser.get_structure("binary", binary_cif_path)
    except Exception:
        return None

    # CA atoms for protein alignment
    def _ca(struct):
        ca = []
        for model in struct:
            for chain in model:
                if chain.id == protein_chain:
                    for res in chain:
                        for atom in res:
                            if atom.get_name() == "CA":
                                ca.append(atom)
            break
        return sorted(ca, key=lambda a: a.get_parent().get_id()[1])

    t_ca = _ca(t_struct)
    b_ca = _ca(b_struct)
    t_lig = get_ligand_heavy_atoms(t_struct, ligand_chain)
    b_lig = get_ligand_heavy_atoms(b_struct, ligand_chain)

    if not t_ca or not b_ca or not t_lig or not b_lig:
        return None
    if len(t_ca) != len(b_ca) or len(t_lig) != len(b_lig):
        return None

    si = Superimposer()
    si.set_atoms(t_ca, b_ca)
    si.apply(list(b_struct.get_atoms()))

    diffs = []
    for a_t, a_b in zip(t_lig, b_lig):
        d = a_t.get_coord() - a_b.get_coord()
        diffs.append(np.sum(d * d))

    return float(np.sqrt(np.average(diffs)))


def find_ternary_cif(ternary_dir, design_id):
    """Find the ternary CIF matching a binary design_id."""
    base = Path(ternary_dir)
    # Try flat: {design_id}_model.cif
    f = base / f"{design_id}_model.cif"
    if f.exists():
        return str(f)
    # Try subfolder: {design_id}/{design_id}_model.cif
    f = base / design_id / f"{design_id}_model.cif"
    if f.exists():
        return str(f)
    return None


def get_min_dist_to_ligand_oxygen(
    ref_model_path,
    af_cif_path,
    ligand_chain="B",
    ligand_res_id=1,
    ref_chain="A",
    ref_res_id=201,
    ref_atom="O1",
):
    """
    1. Loads Ref and AF structures.
    2. Aligns AF to Ref using CA atoms.
    3. Finds Ref atom.
    4. Iterates over ALL atoms in AF Ligand.
    5. If atom is Oxygen, calc distance.
    6. Returns Minimum Distance.
    """
    if not af_cif_path:
        return None

    ref_struct, ref_ca = get_full_structure_and_ca_atoms(ref_model_path)
    af_struct, af_ca = get_full_structure_and_ca_atoms(af_cif_path)

    if not ref_struct or not af_struct or not ref_ca or not af_ca:
        return None
    if len(ref_ca) != len(af_ca):
        return None

    # 1. Get Reference Point (e.g. A 201 O1)
    ref_atom_obj = get_specific_atom(ref_struct, ref_chain, ref_res_id, ref_atom)
    if not ref_atom_obj:
        return None

    # 2. Get All Ligand Atoms
    ligand_atoms = get_ligand_atoms(af_struct, ligand_chain, ligand_res_id)
    if not ligand_atoms:
        return None

    # 3. Superimpose (Align AF to Ref)
    superimposer = Superimposer()
    superimposer.set_atoms(ref_ca, af_ca)
    # Apply rotation/translation to the ligand atoms so they are in Ref frame
    superimposer.apply(ligand_atoms)

    # 4. Search for closest Oxygen
    min_dist = float('inf')
    found_any_oxygen = False

    ref_coord = ref_atom_obj.get_coord()

    for atom in ligand_atoms:
        # Check if element is Oxygen (using element property or name fallback)
        element = atom.element.upper() if atom.element else atom.get_name().strip()[0].upper()
        
        if element == "O":
            dist = np.linalg.norm(atom.get_coord() - ref_coord)
            if dist < min_dist:
                min_dist = dist
                found_any_oxygen = True

    return float(min_dist) if found_any_oxygen else None


# ----------------------------
# 4. Main
# ----------------------------
def main(args):
    target_map = find_af3_files(args.inference_dir)
    all_targets = sorted(target_map.keys())
    print(f"Found {len(all_targets)} targets.")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Updated column name
    dist_col = f"min_dist_to_ligand_O_aligned"

    has_ternary = bool(args.ternary_dir)
    headers = ["target", "ligand_plddt", "ligand_iptm", dist_col]
    if has_ternary:
        headers.append("ligand_rmsd")

    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for i, tgt in enumerate(all_targets, 1):
            files = target_map[tgt]

            plddt, iptm = extract_json_metrics(files, ligand_chain=args.ligand_chain)

            # Use new function
            dist = get_min_dist_to_ligand_oxygen(
                args.ref_model,
                files.get("cif"),
                ligand_chain=args.ligand_chain,
                ligand_res_id=args.ligand_res_id,
                ref_chain=args.ref_chain,
                ref_res_id=args.ref_res_id,
                ref_atom=args.ref_atom,
            )

            row = [
                tgt,
                f"{plddt:.2f}" if plddt is not None else "NA",
                f"{iptm:.2f}" if iptm is not None else "NA",
                f"{dist:.3f}" if dist is not None else "NA",
            ]

            # Binary-to-ternary ligand RMSD
            if has_ternary:
                rmsd = None
                binary_cif = files.get("cif")
                if binary_cif:
                    ternary_cif = find_ternary_cif(args.ternary_dir, tgt)
                    if ternary_cif:
                        rmsd = calculate_binary_ternary_rmsd(
                            binary_cif, ternary_cif,
                            protein_chain=args.ref_chain,
                            ligand_chain=args.ligand_chain,
                        )
                row.append(f"{rmsd:.3f}" if rmsd is not None else "NA")

            writer.writerow(row)

            if i % 100 == 0:
                print(f"Processed {i} targets...")

    print(f"\nDone. Results written to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate AF3 metrics using recursive search.")
    parser.add_argument("--inference_dir", required=True)
    parser.add_argument("--ref_model", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--ligand_chain", default="B")
    parser.add_argument("--ternary_dir", default=None,
                        help="Ternary inference dir for binary-to-ternary ligand RMSD")

    # Removed --ligand_oxygen_atom as we now search ALL oxygens
    parser.add_argument("--ligand_res_id", type=int, default=1)
    parser.add_argument("--ref_chain", default="A")
    parser.add_argument("--ref_res_id", type=int, default=201)
    parser.add_argument("--ref_atom", default="O1")

    args = parser.parse_args()
    main(args)
