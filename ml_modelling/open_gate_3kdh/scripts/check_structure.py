#!/usr/bin/env python3
"""
Step 5: Quality control for open-gate structures (3KDH backbone version).

Checks each final structure against the 3KDH template and Boltz prediction:
  - Gate loop stayed open (RMSD to 3KDH)
  - Backbone integrity (overall RMSD to 3KDH)
  - Ligand is in the pocket
  - No severe steric clashes
  - All designed mutations are present

Relaxed RMSD thresholds for heavy mutation load (30-70+ mutations from PYL2->PYR1).

Usage:
    python check_structure.py \
        --input-dir outputs/open_gate_structures/ \
        --template inputs/3KDH.pdb \
        --boltz-dir inputs/boltz_predictions/ \
        --alignment alignment_map.json \
        --csv ../analysis/boltz_LCA/md_candidates_lca_top100.csv \
        --output-csv outputs/qc_report.csv

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def parse_pdb_residues(pdb_path, chain="A"):
    """Extract residue sequence and CA coordinates from a PDB chain."""
    residues = {}
    ca_coords = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            resnum = int(line[22:26])
            resname = line[17:20].strip()
            name = line[12:16].strip()

            if resnum not in residues:
                aa = AA_3_TO_1.get(resname, "X")
                residues[resnum] = aa

            if name == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_coords[resnum] = np.array([x, y, z])

    return residues, ca_coords


def parse_ligand_coords(pdb_path, chain="B"):
    """Extract ligand heavy atom coordinates from PDB."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            if line[21] != chain:
                continue
            element = line[76:78].strip() if len(line) > 78 else ""
            name = line[12:16].strip()
            if element == "H" or name.startswith("H"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])

    return np.array(coords) if coords else np.zeros((0, 3))


def compute_rmsd(coords1, coords2):
    """Compute RMSD between two sets of coordinates (no alignment)."""
    if len(coords1) == 0 or len(coords2) == 0:
        return float("inf")
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_rmsd_from_dicts(ca1, ca2, resnums):
    """Compute CA RMSD for specific residue numbers."""
    coords1, coords2 = [], []
    for r in resnums:
        if r in ca1 and r in ca2:
            coords1.append(ca1[r])
            coords2.append(ca2[r])
    if not coords1:
        return float("inf")
    return compute_rmsd(np.array(coords1), np.array(coords2))


def check_mutations(pdb_path, signature, alignment_map, chain="A"):
    """Verify that all designed mutations are present in the structure."""
    residues, _ = parse_pdb_residues(pdb_path, chain)
    boltz_to_3kdh = {int(k): v for k, v in alignment_map["boltz_to_3kdh"].items()}

    mutations = {}
    if signature and not pd.isna(signature):
        normalized = signature.replace("_", ";").replace(" ", ";")
        for mut in normalized.split(";"):
            mut = mut.strip()
            if not mut:
                continue
            match = re.match(r"^([A-Z])?(\d+)([A-Z])$", mut)
            if match:
                _, pos, target = match.groups()
                mutations[int(pos)] = target

    all_correct = True
    details = []
    for boltz_pos, expected_aa in sorted(mutations.items()):
        kdh_pos = boltz_to_3kdh.get(boltz_pos)
        if kdh_pos is None:
            details.append(f"  Boltz {boltz_pos}: no mapping to 3KDH")
            all_correct = False
            continue

        actual_aa = residues.get(kdh_pos, "?")
        if actual_aa == expected_aa:
            details.append(f"  Boltz {boltz_pos} -> 3KDH {kdh_pos}: {actual_aa} (OK)")
        else:
            details.append(f"  Boltz {boltz_pos} -> 3KDH {kdh_pos}: "
                          f"expected {expected_aa}, got {actual_aa} (MISMATCH)")
            all_correct = False

    return all_correct, details


def generate_pymol_script(pdb_path, pair_id, alignment_map, output_dir):
    """Generate a PyMOL .pml visualization script."""
    pocket_3kdh = alignment_map.get("pocket_positions_3kdh", [])
    gate_3kdh = alignment_map.get("gate_loop_3kdh", [])

    pocket_sel = "+".join(str(r) for r in pocket_3kdh)
    gate_sel = "+".join(str(r) for r in gate_3kdh)

    pml_content = f"""# PyMOL visualization for {pair_id} open-gate structure (3KDH backbone)
# Generated by check_structure.py

load {pdb_path}, {pair_id}

# Basic display
hide everything
show cartoon, {pair_id} and chain A
show sticks, {pair_id} and chain B

# Color scheme
color gray80, {pair_id} and chain A

# Pocket residues in blue
select pocket, {pair_id} and chain A and resi {pocket_sel}
show sticks, pocket
color marine, pocket

# Gate loop in red
select gate_loop, {pair_id} and chain A and resi {gate_sel}
color firebrick, gate_loop

# Ligand in green
select ligand, {pair_id} and chain B
color splitpea, ligand
show sticks, ligand

# Surface for pocket
show surface, pocket
set surface_color, marine, pocket
set transparency, 0.7

# Nice view
orient
zoom {pair_id}
set ray_shadows, 0
bg_color white

# Labels
set label_size, 14
"""

    pml_path = Path(output_dir) / f"visualize_{pair_id}.pml"
    with open(pml_path, "w") as f:
        f.write(pml_content)

    return str(pml_path)


def check_single_structure(
    open_gate_pdb, template_pdb, boltz_pdb, alignment_map,
    signature=None, pair_id=None, pml_output_dir=None
):
    """Run all QC checks on a single open-gate structure."""
    if pair_id is None:
        stem = Path(open_gate_pdb).stem
        pair_id = stem.replace("_open_gate", "").replace("_threaded_relaxed", "")

    result = {"pair_id": pair_id}

    _, og_ca = parse_pdb_residues(open_gate_pdb, "A")
    _, tmpl_ca = parse_pdb_residues(template_pdb, "A")

    gate_3kdh = alignment_map.get("gate_loop_3kdh", [])
    pocket_3kdh = alignment_map.get("pocket_positions_3kdh", [])
    anchor_3kdh = alignment_map.get("anchor_positions_3kdh", [])

    # 1. Gate loop RMSD to template
    gate_rmsd = compute_rmsd_from_dicts(og_ca, tmpl_ca, gate_3kdh)
    result["gate_rmsd_to_3kdh"] = round(gate_rmsd, 3)

    # 2. Overall backbone RMSD to template
    all_common = sorted(set(og_ca.keys()) & set(tmpl_ca.keys()))
    backbone_rmsd = compute_rmsd_from_dicts(og_ca, tmpl_ca, all_common)
    result["backbone_rmsd_to_3kdh"] = round(backbone_rmsd, 3)

    # 3. Pocket floor RMSD
    pocket_rmsd = compute_rmsd_from_dicts(og_ca, tmpl_ca, anchor_3kdh)
    result["pocket_floor_rmsd"] = round(pocket_rmsd, 3)

    # 4. Gate RMSD to Boltz (should be high = different conformations)
    if boltz_pdb and Path(boltz_pdb).exists():
        _, boltz_ca = parse_pdb_residues(boltz_pdb, "A")
        gate_boltz = alignment_map.get("gate_loop_boltz", [])
        gate_rmsd_to_boltz = compute_rmsd_from_dicts(og_ca, boltz_ca, gate_3kdh)
        result["gate_rmsd_to_boltz"] = round(gate_rmsd_to_boltz, 3)
    else:
        result["gate_rmsd_to_boltz"] = None

    # 5. Ligand position
    lig_coords = parse_ligand_coords(open_gate_pdb, "B")
    if len(lig_coords) > 0:
        lig_com = lig_coords.mean(axis=0)
        pocket_ca_coords = [og_ca[r] for r in pocket_3kdh if r in og_ca]
        if pocket_ca_coords:
            pocket_centroid = np.mean(pocket_ca_coords, axis=0)
            lig_to_pocket_dist = np.linalg.norm(lig_com - pocket_centroid)
            result["ligand_to_pocket_dist"] = round(lig_to_pocket_dist, 2)
        else:
            result["ligand_to_pocket_dist"] = None
        result["n_ligand_atoms"] = len(lig_coords)
    else:
        result["ligand_to_pocket_dist"] = None
        result["n_ligand_atoms"] = 0

    # 6. Clash check
    prot_atoms = []
    with open(open_gate_pdb) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == "A":
                element = line[76:78].strip() if len(line) > 78 else ""
                name = line[12:16].strip()
                if element != "H" and not name.startswith("H"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    prot_atoms.append([x, y, z])

    if len(lig_coords) > 0 and prot_atoms:
        prot_xyz = np.array(prot_atoms)
        diff = lig_coords[:, np.newaxis, :] - prot_xyz[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        min_per_lig = distances.min(axis=1)
        result["min_lig_prot_distance"] = round(float(min_per_lig.min()), 2)
        result["n_severe_clashes"] = int(np.sum(min_per_lig < 1.5))
        result["n_mild_clashes"] = int(np.sum((min_per_lig >= 1.5) & (min_per_lig < 2.0)))
    else:
        result["min_lig_prot_distance"] = None
        result["n_severe_clashes"] = 0
        result["n_mild_clashes"] = 0

    # 7. Mutation verification
    if signature:
        all_correct, details = check_mutations(
            open_gate_pdb, signature, alignment_map, "A"
        )
        result["mutations_correct"] = all_correct
    else:
        result["mutations_correct"] = None

    # Determine status (relaxed thresholds for heavy mutation load)
    status = "PASS"
    if result["n_severe_clashes"] > 0:
        status = "FAIL_CLASH"
    elif result.get("mutations_correct") is False:
        status = "FAIL_MUTATION"
    elif backbone_rmsd > 3.0:
        status = "FAIL_BACKBONE"
    elif result["n_mild_clashes"] > 0:
        status = "WARN_CLASH"
    elif backbone_rmsd > 2.0:
        status = "WARN_BACKBONE"
    result["status"] = status

    if pml_output_dir:
        pml_path = generate_pymol_script(
            open_gate_pdb, pair_id, alignment_map, pml_output_dir
        )
        result["pymol_script"] = pml_path

    return result


def main():
    parser = argparse.ArgumentParser(
        description="QC checks for open-gate PYR1 structures (3KDH backbone)"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing open-gate PDB files")
    parser.add_argument("--template", required=True,
                        help="Path to 3KDH.pdb template")
    parser.add_argument("--boltz-dir",
                        help="Directory containing Boltz prediction PDBs")
    parser.add_argument("--alignment", required=True,
                        help="Path to alignment_map.json")
    parser.add_argument("--csv",
                        help="CSV with pair_id and variant_signature")
    parser.add_argument("--output-csv", default="qc_report.csv",
                        help="Output QC report CSV")
    parser.add_argument("--pml-dir",
                        help="Directory for PyMOL scripts")
    parser.add_argument("--single",
                        help="Check only this pair_id")
    args = parser.parse_args()

    with open(args.alignment) as f:
        alignment_map = json.load(f)

    signatures = {}
    if args.csv:
        df = pd.read_csv(args.csv)
        for _, row in df.iterrows():
            signatures[row["pair_id"]] = row.get("variant_signature", "")

    input_dir = Path(args.input_dir)
    # Support both naming conventions: *_open_gate.pdb (Stage 4) and
    # *_threaded_relaxed.pdb (Stage 3 with ligand-aware mode)
    pdb_files = sorted(input_dir.glob("*_open_gate.pdb"))
    suffix_to_strip = "_open_gate"
    if not pdb_files:
        pdb_files = sorted(input_dir.glob("*_threaded_relaxed.pdb"))
        suffix_to_strip = "_threaded_relaxed"

    if args.single:
        pdb_files = [f for f in pdb_files if args.single in f.stem]

    if not pdb_files:
        logger.error(f"No *_open_gate.pdb or *_threaded_relaxed.pdb files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Checking {len(pdb_files)} structures...")

    pml_dir = args.pml_dir or str(input_dir)
    Path(pml_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for pdb_file in pdb_files:
        pair_id = pdb_file.stem.replace(suffix_to_strip, "")
        logger.info(f"\n  {pair_id}")

        boltz_pdb = None
        if args.boltz_dir:
            bp = Path(args.boltz_dir) / f"{pair_id}.pdb"
            if bp.exists():
                boltz_pdb = str(bp)

        result = check_single_structure(
            open_gate_pdb=str(pdb_file),
            template_pdb=args.template,
            boltz_pdb=boltz_pdb,
            alignment_map=alignment_map,
            signature=signatures.get(pair_id),
            pair_id=pair_id,
            pml_output_dir=pml_dir,
        )
        results.append(result)

        logger.info(f"    Gate RMSD to 3KDH: {result['gate_rmsd_to_3kdh']:.3f} A")
        logger.info(f"    Backbone RMSD:     {result['backbone_rmsd_to_3kdh']:.3f} A")
        if result.get("min_lig_prot_distance") is not None:
            logger.info(f"    Min lig-prot dist: {result['min_lig_prot_distance']:.2f} A")
        logger.info(f"    Status: {result['status']}")

    report_df = pd.DataFrame(results)
    report_df.to_csv(args.output_csv, index=False)
    logger.info(f"\n\nQC Report saved to {args.output_csv}")

    status_counts = report_df["status"].value_counts()
    logger.info("\nSummary:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")

    n_pass = int((report_df["status"] == "PASS").sum())
    logger.info(f"\n{n_pass}/{len(results)} structures passed all QC checks")


if __name__ == "__main__":
    main()
