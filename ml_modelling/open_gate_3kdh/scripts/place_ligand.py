#!/usr/bin/env python3
"""
Step 4: Place the Boltz-predicted ligand into the open-gate threaded structure.

Uses pocket-floor CA superposition (Kabsch algorithm) to transform the ligand
coordinates from the Boltz closed-gate prediction into the open-gate frame.
Only pocket-floor anchor residues are used — the gate loop is excluded.

This version uses 3KDH (PYL2) backbone template instead of 3K3K (PYR1).

Usage:
    python place_ligand.py \
        --threaded-pdb outputs/threaded_relaxed/pair_3098_threaded_relaxed.pdb \
        --boltz-pdb inputs/boltz_predictions/pair_3098.pdb \
        --alignment alignment_map.json \
        --output outputs/open_gate_structures/pair_3098_open_gate.pdb

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_pdb_atoms(pdb_path, record_types=("ATOM", "HETATM")):
    """Parse PDB file into a list of atom dicts."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in record_types:
                continue
            try:
                atom = {
                    "record": rec,
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21],
                    "resnum": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "occupancy": float(line[54:60]) if len(line) > 60 else 1.0,
                    "bfactor": float(line[60:66]) if len(line) > 66 else 0.0,
                    "element": line[76:78].strip() if len(line) > 78 else "",
                    "line": line,
                }
                atoms.append(atom)
            except (ValueError, IndexError):
                continue
    return atoms


def get_ca_coords(atoms, resnums, chain="A"):
    """Extract CA coordinates for specific residues."""
    ca_coords = {}
    resnum_set = set(resnums)
    for atom in atoms:
        if (atom["chain"] == chain and
            atom["name"] == "CA" and
            atom["resnum"] in resnum_set):
            ca_coords[atom["resnum"]] = np.array([atom["x"], atom["y"], atom["z"]])
    return ca_coords


def get_ligand_atoms(atoms, chain="B"):
    """Extract ligand heavy atoms (non-H) from PDB."""
    lig_atoms = []
    for atom in atoms:
        if atom["record"] == "HETATM" and atom["chain"] == chain:
            if atom["element"] == "H" or atom["name"].startswith("H"):
                continue
            lig_atoms.append(atom)
    return lig_atoms


def get_conect_records(pdb_path):
    """Extract CONECT records from PDB file."""
    conect = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("CONECT"):
                conect.append(line)
    return conect


def kabsch_superpose(coords_mobile, coords_target):
    """Compute optimal rotation and translation using Kabsch (SVD) algorithm."""
    assert coords_mobile.shape == coords_target.shape
    n = coords_mobile.shape[0]

    centroid_mobile = coords_mobile.mean(axis=0)
    centroid_target = coords_target.mean(axis=0)

    mobile_centered = coords_mobile - centroid_mobile
    target_centered = coords_target - centroid_target

    H = mobile_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T
    t = centroid_target - R @ centroid_mobile

    transformed = (R @ coords_mobile.T).T + t
    diff = transformed - coords_target
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    return R, t, rmsd


def transform_coordinates(coords, R, t):
    """Apply rotation and translation to coordinates."""
    return (R @ coords.T).T + t


def clash_check(protein_atoms, ligand_atoms, severe_threshold=1.5, mild_threshold=2.0):
    """Check for steric clashes between protein and ligand."""
    prot_coords = []
    for atom in protein_atoms:
        if atom["element"] not in ("H", "") and not atom["name"].startswith("H"):
            prot_coords.append([atom["x"], atom["y"], atom["z"]])

    if not prot_coords or not ligand_atoms:
        return {"n_severe": 0, "n_mild": 0, "min_distance": float("inf"), "details": []}

    prot_xyz = np.array(prot_coords)
    lig_xyz = np.array([[a["x"], a["y"], a["z"]] for a in ligand_atoms])

    diff = lig_xyz[:, np.newaxis, :] - prot_xyz[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    min_per_lig_atom = distances.min(axis=1)
    min_distance = min_per_lig_atom.min()

    severe_count = int(np.sum(min_per_lig_atom < severe_threshold))
    mild_count = int(np.sum((min_per_lig_atom >= severe_threshold) &
                            (min_per_lig_atom < mild_threshold)))

    details = []
    for i, d in enumerate(min_per_lig_atom):
        if d < mild_threshold:
            severity = "SEVERE" if d < severe_threshold else "MILD"
            details.append(f"  {ligand_atoms[i]['name']}: min dist = {d:.2f} A ({severity})")

    return {
        "n_severe": severe_count,
        "n_mild": mild_count,
        "min_distance": float(min_distance),
        "details": details,
    }


def write_merged_pdb(protein_atoms, ligand_atoms, conect_records, output_path):
    """Write merged protein + ligand PDB file."""
    lines = []
    serial = 1
    old_to_new_serial = {}

    for atom in protein_atoms:
        if atom["record"] != "ATOM":
            continue
        old_to_new_serial[atom["serial"]] = serial
        line = (
            f"ATOM  {serial:>5d} {atom['name']:>4s} {atom['resname']:>3s} "
            f"{atom['chain']}{atom['resnum']:>4d}    "
            f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
            f"{atom['occupancy']:6.2f}{atom['bfactor']:6.2f}"
            f"           {atom['element']:>2s}\n"
        )
        lines.append(line)
        serial += 1

    lines.append(f"TER   {serial:>5d}      {protein_atoms[-1]['resname']:>3s} "
                 f"{protein_atoms[-1]['chain']}{protein_atoms[-1]['resnum']:>4d}\n")
    serial += 1

    lig_serial_start = serial
    for atom in ligand_atoms:
        old_to_new_serial[atom["serial"]] = serial
        name = atom["name"]
        if len(name) < 4:
            name = f" {name:<3s}"
        line = (
            f"HETATM{serial:>5d} {name} {atom['resname']:>3s} "
            f"{atom['chain']}{atom['resnum']:>4d}    "
            f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
            f"{atom['occupancy']:6.2f}{atom['bfactor']:6.2f}"
            f"           {atom['element']:>2s}\n"
        )
        lines.append(line)
        serial += 1

    for conect_line in conect_records:
        parts = conect_line.split()
        if len(parts) < 3:
            continue
        try:
            old_serials = [int(p) for p in parts[1:]]
            new_serials = []
            valid = True
            for s in old_serials:
                ns = old_to_new_serial.get(s)
                if ns is None:
                    valid = False
                    break
                new_serials.append(ns)
            if valid:
                new_line = "CONECT" + "".join(f"{s:>5d}" for s in new_serials) + "\n"
                lines.append(new_line)
        except ValueError:
            continue

    lines.append("END\n")

    with open(output_path, "w") as f:
        f.writelines(lines)


def minimize_complex(output_pdb, params_path, pocket_distance=6.0):
    """Optional: Rosetta minimization of the protein-ligand complex."""
    try:
        import pyrosetta
        from pyrosetta import pose_from_pdb
        from pyrosetta.rosetta.protocols.minimization_packing import MinMover
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            RestrictToRepacking, IncludeCurrent, PreventRepacking
        )
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    except ImportError:
        logger.warning("PyRosetta not available — skipping minimization")
        return

    pyrosetta.init(
        f"-extra_res_fa {params_path} "
        "-ignore_unrecognized_res true "
        "-ex1 -ex2 "
        "-use_input_sc "
        "-mute all"
    )

    pose = pose_from_pdb(str(output_pdb))
    sfxn = pyrosetta.create_score_function("ref2015")

    score_before = sfxn(pose)
    logger.info(f"  Score before minimization: {score_before:.1f}")

    lig_res = None
    for i in range(1, pose.total_residue() + 1):
        if not pose.residue(i).is_protein():
            lig_res = i
            break

    if lig_res is None:
        logger.warning("  No ligand residue found — skipping minimization")
        return

    from pyrosetta.rosetta.core.select.residue_selector import (
        ResidueIndexSelector, NeighborhoodResidueSelector
    )
    lig_sel = ResidueIndexSelector(str(lig_res))
    pocket_sel = NeighborhoodResidueSelector(lig_sel, pocket_distance, True)
    pocket_vec = pocket_sel.apply(pose)
    pocket_residues = {i for i in range(1, pose.total_residue() + 1) if pocket_vec[i]}

    tf = TaskFactory()
    tf.push_back(IncludeCurrent())
    tf.push_back(RestrictToRepacking())
    prevent = PreventRepacking()
    for i in range(1, pose.total_residue() + 1):
        if i not in pocket_residues:
            prevent.include_residue(i)
    tf.push_back(prevent)

    packer = PackRotamersMover(sfxn)
    packer.task_factory(tf)
    packer.apply(pose)

    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    for res_idx in pocket_residues:
        mm.set_chi(res_idx, True)
    mm.set_jump(True)

    min_mover = MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(sfxn)
    min_mover.min_type("dfpmin_armijo_nonmonotone")
    min_mover.tolerance(0.01)
    min_mover.max_iter(50)
    min_mover.apply(pose)

    score_after = sfxn(pose)
    logger.info(f"  Score after minimization: {score_after:.1f} "
               f"(delta: {score_after - score_before:+.1f})")

    pose.dump_pdb(str(output_pdb))
    logger.info(f"  Minimized structure saved to {output_pdb}")


def process_single(
    threaded_pdb, boltz_pdb, alignment_map, output_path,
    params_path=None, do_minimize=False, ligand_chain="B",
    threaded_chain="A", boltz_chain="A"
):
    """Place ligand from one Boltz PDB into one threaded open-gate structure."""
    pair_id = Path(output_path).stem.replace("_open_gate", "")

    if Path(output_path).exists():
        logger.info(f"  {pair_id}: already exists — skipping")
        return {"pair_id": pair_id, "status": "SKIPPED"}

    threaded_atoms = parse_pdb_atoms(threaded_pdb)
    boltz_atoms = parse_pdb_atoms(boltz_pdb)

    # Get anchor positions (Boltz and 3KDH numbering)
    anchor_boltz = alignment_map["anchor_positions_boltz"]
    anchor_3kdh = alignment_map["anchor_positions_3kdh"]

    threaded_ca = get_ca_coords(threaded_atoms, anchor_3kdh, chain=threaded_chain)
    boltz_ca = get_ca_coords(boltz_atoms, anchor_boltz, chain=boltz_chain)

    common_anchors = []
    for ab, ak in zip(anchor_boltz, anchor_3kdh):
        if ab in boltz_ca and ak in threaded_ca:
            common_anchors.append((ab, ak))

    if len(common_anchors) < 4:
        logger.error(f"  {pair_id}: Only {len(common_anchors)} common anchors — need >= 4")
        return {"pair_id": pair_id, "status": "FAIL_ANCHORS"}

    mobile_coords = np.array([boltz_ca[ab] for ab, _ in common_anchors])
    target_coords = np.array([threaded_ca[ak] for _, ak in common_anchors])

    R, t, sup_rmsd = kabsch_superpose(mobile_coords, target_coords)
    logger.info(f"  Superposition RMSD ({len(common_anchors)} anchor CAs): {sup_rmsd:.3f} A")

    if sup_rmsd > 1.5:
        logger.warning(f"  WARNING: Pocket superposition RMSD {sup_rmsd:.3f} A > 1.5 A!")

    lig_atoms = get_ligand_atoms(boltz_atoms, chain=ligand_chain)
    if not lig_atoms:
        logger.error(f"  {pair_id}: No ligand atoms found in Boltz PDB")
        return {"pair_id": pair_id, "status": "FAIL_NO_LIGAND"}

    lig_coords = np.array([[a["x"], a["y"], a["z"]] for a in lig_atoms])
    transformed_coords = transform_coordinates(lig_coords, R, t)

    for i, atom in enumerate(lig_atoms):
        atom["x"] = transformed_coords[i, 0]
        atom["y"] = transformed_coords[i, 1]
        atom["z"] = transformed_coords[i, 2]

    protein_atoms = [a for a in threaded_atoms if a["record"] == "ATOM"]

    clash = clash_check(protein_atoms, lig_atoms)
    logger.info(f"  Clashes: {clash['n_severe']} severe, {clash['n_mild']} mild, "
               f"min dist = {clash['min_distance']:.2f} A")
    for detail in clash["details"]:
        logger.info(detail)

    conect = get_conect_records(boltz_pdb)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_merged_pdb(protein_atoms, lig_atoms, conect, output_path)
    logger.info(f"  Merged structure: {output_path}")

    if do_minimize and params_path:
        logger.info("  Running Rosetta minimization...")
        minimize_complex(output_path, params_path)
    elif do_minimize and not params_path:
        logger.warning("  --minimize requested but no --params provided — skipping")

    status = "PASS"
    if clash["n_severe"] > 0:
        status = "FAIL_CLASH"
    elif clash["n_mild"] > 0:
        status = "WARN_CLASH"

    return {
        "pair_id": pair_id,
        "n_ligand_atoms": len(lig_atoms),
        "n_anchors": len(common_anchors),
        "superposition_rmsd": sup_rmsd,
        "min_distance": clash["min_distance"],
        "n_severe_clashes": clash["n_severe"],
        "n_mild_clashes": clash["n_mild"],
        "status": status,
        "output_pdb": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Place ligand from Boltz prediction into open-gate structure (3KDH backbone)"
    )

    parser.add_argument("--threaded-pdb",
                        help="Path to threaded/relaxed open-gate PDB")
    parser.add_argument("--boltz-pdb",
                        help="Path to Boltz prediction PDB (source of ligand)")
    parser.add_argument("--output",
                        help="Output path for the merged PDB")

    parser.add_argument("--threaded-dir",
                        help="Directory containing threaded/relaxed PDBs")
    parser.add_argument("--boltz-dir",
                        help="Directory containing Boltz prediction PDBs")
    parser.add_argument("--output-dir",
                        help="Output directory for merged structures")
    parser.add_argument("--csv",
                        help="CSV with pair_id column for batch processing")
    parser.add_argument("--single",
                        help="Process only this pair_id from CSV")

    parser.add_argument("--alignment", required=True,
                        help="Path to alignment_map.json")
    parser.add_argument("--params",
                        help="Path to ligand .params file for Rosetta minimization")
    parser.add_argument("--minimize", action="store_true",
                        help="Run Rosetta minimization after ligand placement")
    parser.add_argument("--ligand-chain", default="B",
                        help="Ligand chain ID in Boltz PDB (default: B)")

    args = parser.parse_args()

    with open(args.alignment) as f:
        alignment_map = json.load(f)

    results = []

    if args.csv:
        if not args.threaded_dir or not args.boltz_dir or not args.output_dir:
            parser.error("Batch mode requires --threaded-dir, --boltz-dir, --output-dir")

        df = pd.read_csv(args.csv)
        if args.single:
            df = df[df["pair_id"] == args.single]

        threaded_dir = Path(args.threaded_dir)
        boltz_dir = Path(args.boltz_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, (_, row) in enumerate(df.iterrows()):
            pair_id = row["pair_id"]
            logger.info(f"\n[{i+1}/{len(df)}] {pair_id}")

            threaded_pdb = threaded_dir / f"{pair_id}_threaded_relaxed.pdb"
            boltz_pdb = boltz_dir / f"{pair_id}.pdb"
            output_pdb = output_dir / f"{pair_id}_open_gate.pdb"

            if not threaded_pdb.exists():
                logger.warning(f"  Threaded PDB not found: {threaded_pdb} — skipping")
                results.append({"pair_id": pair_id, "status": "MISSING_THREADED"})
                continue
            if not boltz_pdb.exists():
                logger.warning(f"  Boltz PDB not found: {boltz_pdb} — skipping")
                results.append({"pair_id": pair_id, "status": "MISSING_BOLTZ"})
                continue

            result = process_single(
                str(threaded_pdb), str(boltz_pdb), alignment_map, str(output_pdb),
                params_path=args.params, do_minimize=args.minimize,
                ligand_chain=args.ligand_chain
            )
            results.append(result)

        summary_df = pd.DataFrame(results)
        summary_path = output_dir / "placement_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary: {summary_path}")

        pass_n = sum(1 for r in results if r["status"] == "PASS")
        warn_n = sum(1 for r in results if r["status"] == "WARN_CLASH")
        fail_n = sum(1 for r in results if r["status"].startswith("FAIL"))
        skip_n = sum(1 for r in results if r["status"] in ("SKIPPED", "MISSING_THREADED", "MISSING_BOLTZ"))
        logger.info(f"Results: {pass_n} PASS, {warn_n} WARN, {fail_n} FAIL, {skip_n} SKIPPED")

    else:
        if not args.threaded_pdb or not args.boltz_pdb or not args.output:
            parser.error("Single mode requires --threaded-pdb, --boltz-pdb, --output")

        result = process_single(
            args.threaded_pdb, args.boltz_pdb, alignment_map, args.output,
            params_path=args.params, do_minimize=args.minimize,
            ligand_chain=args.ligand_chain
        )
        logger.info(f"\nResult: {result['status']}")


if __name__ == "__main__":
    main()
